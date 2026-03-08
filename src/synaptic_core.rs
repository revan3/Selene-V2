// =============================================================================
// src/synaptic_core.rs — Núcleo Sináptico da Selene V2
// =============================================================================
//
// Este arquivo é o coração de todo o processamento neural da Selene.
// Cada neurônio, cada spike, cada traço de plasticidade passa por aqui.
//
// Modelo base: Izhikevich (2003) — "Simple Model of Spiking Neurons"
//   dv/dt = 0.04v² + 5v + 140 − u + I   (dinâmica do potencial de membrana)
//   du/dt = a(bv − u)                     (variável de recuperação)
//   if v ≥ 30mV → v = c, u += d          (reset após spike)
//
// Onde dt deve estar SEMPRE em milissegundos.
// O código resolve internamente a conversão de dt_segundos → substeps de 1ms.
//
// Precisão mista: FP32 (5%) → FP16 (35%) → INT8 (50%) → INT4 (10%)
//   - FP32: neurônios críticos (decisão, emoção intensa)
//   - FP16: neurônios de reconhecimento e working memory
//   - INT8: neurônios de processamento em massa
//   - INT4: neurônios de background / reservatório
//
// STDP (Spike-Timing Dependent Plasticity):
//   - Traço pré-sináptico decai com tau=20ms
//   - LTP (potenciação): spike pós quando traço pré está alto → aumenta peso
//   - LTD (depressão anti-Hebbiana): spike pós sem traço pré → diminui peso
//
// Tipos neuronais (Izhikevich 2003):
//   RS  — Regular Spiking      — córtex piramidal, 80% dos neurônios
//   IB  — Intrinsic Bursting   — camada 5, resposta emocional em burst
//   CH  — Chattering           — visual V2/V3, reconhecimento rápido
//   FS  — Fast Spiking         — interneurônios GABAérgicos inibitórios
//   LT  — Low-Threshold Spiking— interneurônios de threshold baixo
//   TC  — Thalamo-Cortical     — tálamo, dois modos (tônico/burst)
//   RZ  — Resonator            — giro dentado, detecção rítmica
// =============================================================================

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use half::f16;
use crate::config::Config;
use crate::compressor::salient::{SalientPoint, SalientCompressor};

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 1: TIPOS DE NEURÔNIO BIOLÓGICO
// ─────────────────────────────────────────────────────────────────────────────

/// Tipos funcionais de neurônios baseados no modelo Izhikevich 2003.
///
/// Cada variante representa um padrão de disparo biologicamente distinto.
/// Os parâmetros a, b, c, d são constantes por tipo — não variam por instância,
/// o que economiza 16 bytes por neurônio comparado ao design anterior.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TipoNeuronal {
    /// Regular Spiking — neurônio piramidal excitatório padrão.
    /// Dispara regularmente sob corrente constante, frequência proporcional à intensidade.
    /// 80% dos neurônios corticais. Parâmetros: a=0.02, b=0.2, c=-65, d=8
    RS,

    /// Intrinsic Bursting — dispara em bursts iniciais, depois regular.
    /// Encontrado na camada 5 do córtex. Ideal para amígdala e respostas de medo
    /// intensas. Parâmetros: a=0.02, b=0.2, c=-55, d=4
    IB,

    /// Chattering — bursts rápidos repetitivos.
    /// Córtex visual camadas 2/3. Excelente para reconhecimento de padrões
    /// repetitivos (faces, bordas, ritmos). Parâmetros: a=0.02, b=0.2, c=-50, d=2
    CH,

    /// Fast Spiking — interneurônio inibitório GABAérgico.
    /// Dispara muito rápido sem adaptação. Essencial para inibição lateral,
    /// controle de ganho e prevenção de epilepsia cortical.
    /// Parâmetros: a=0.1, b=0.2, c=-65, d=2
    FS,

    /// Low-Threshold Spiking — interneurônio de limiar baixo.
    /// Ativa com correntes mínimas. Usado em circuitos de controle fino.
    /// Parâmetros: a=0.02, b=0.25, c=-65, d=2
    LT,

    /// Thalamo-Cortical — neurônio talâmico com dois modos operacionais.
    /// Modo tônico (acordado): dispara regularmente.
    /// Modo burst (sono/desatenção): silêncio seguido de burst.
    /// Parâmetros: a=0.02, b=0.25, c=-65, d=0.05
    TC,

    /// Resonator — responde preferencialmente a frequências específicas.
    /// Giro dentado do hipocampo. Detecta padrões rítmicos temporais.
    /// Parâmetros: a=0.1, b=0.26, c=-65, d=2
    RZ,
}

impl TipoNeuronal {
    /// Retorna os quatro parâmetros Izhikevich para este tipo.
    /// (a, b, c, d) — ver header do arquivo para significado biológico.
    #[inline]
    pub fn parametros(&self) -> (f32, f32, f32, f32) {
        match self {
            TipoNeuronal::RS => (0.02, 0.20, -65.0, 8.0),
            TipoNeuronal::IB => (0.02, 0.20, -55.0, 4.0),
            TipoNeuronal::CH => (0.02, 0.20, -50.0, 2.0),
            TipoNeuronal::FS => (0.10, 0.20, -65.0, 2.0),
            TipoNeuronal::LT => (0.02, 0.25, -65.0, 2.0),
            TipoNeuronal::TC => (0.02, 0.25, -65.0, 0.05),
            TipoNeuronal::RZ => (0.10, 0.26, -65.0, 2.0),
        }
    }

    /// Threshold de disparo padrão. Neurônios do cerebelo têm threshold menor.
    #[inline]
    pub fn threshold_padrao(&self) -> f32 {
        match self {
            TipoNeuronal::TC => 25.0, // talâmico dispara mais facilmente
            TipoNeuronal::FS => 25.0, // interneurônios respondem mais rápido
            _               => 30.0, // padrão Izhikevich
        }
    }

    /// Retorna se este tipo é inibitório (GABAérgico).
    /// Usado pelos lóbulos para aplicar inibição lateral corretamente.
    #[inline]
    pub fn e_inibitorico(&self) -> bool {
        matches!(self, TipoNeuronal::FS | TipoNeuronal::LT)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 2: PRECISÃO MISTA
// ─────────────────────────────────────────────────────────────────────────────

/// Nível de precisão numérica do peso sináptico.
///
/// A escolha de precisão afeta memória e fidelidade do peso:
///   FP32 → 4 bytes, range ±3.4×10³⁸, sem arredondamento
///   FP16 → 2 bytes, range ±65504, erro ~0.1%
///   INT8 → 1 byte, range depende da escala da camada
///   INT4 → 0.5 bytes efetivos (dois valores empacotados em 1 byte)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PrecisionType {
    FP32,
    FP16,
    INT8,
    INT4,
}

/// INT4 empacotado: dois valores de 4 bits em um único byte.
///
/// Layout do byte:
///   bits [7:4] → nibble alto (high)  — valores -8..+7
///   bits [3:0] → nibble baixo (low)  — valores -8..+7
///
/// Uso de memória real: 0.5 bytes por neurônio (dois INT4 por byte).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Int4Par(u8);

impl Int4Par {
    /// Cria um par empacotando dois valores de 4 bits.
    pub fn novo(alto: i8, baixo: i8) -> Self {
        let h = (alto.max(-8).min(7) as u8) & 0x0F;
        let l = (baixo.max(-8).min(7) as u8) & 0x0F;
        Self((h << 4) | l)
    }

    /// Extrai o nibble alto (bits 7:4), com extensão de sinal.
    #[inline]
    pub fn alto(&self) -> i8 {
        let v = (self.0 >> 4) as i8;
        if v & 0x08 != 0 { v | -16i8 } else { v }
    }

    /// Extrai o nibble baixo (bits 3:0), com extensão de sinal.
    #[inline]
    pub fn baixo(&self) -> i8 {
        let v = (self.0 & 0x0F) as i8;
        if v & 0x08 != 0 { v | -16i8 } else { v }
    }
}

/// Peso sináptico com suporte a precisão mista.
///
/// A escala para INT8 e INT4 é fornecida pela `CamadaHibrida` (escala compartilhada),
/// não armazenada por neurônio — economizando 4 bytes por instância.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PesoNeuronio {
    FP32(f32),
    FP16(f16),
    INT8(i8),   // escala vem da camada
    INT4(u8),   // Int4Par empacotado, escala vem da camada
}

impl PesoNeuronio {
    /// Converte para f32 usando a escala da camada.
    #[inline]
    pub fn valor_f32(&self, escala: f32) -> f32 {
        match self {
            PesoNeuronio::FP32(v)   => *v,
            PesoNeuronio::FP16(v)   => v.to_f32(),
            PesoNeuronio::INT8(v)   => (*v as f32) * escala,
            PesoNeuronio::INT4(raw) => {
                let par = Int4Par(*raw);
                par.alto() as f32 * escala // usa nibble alto por padrão
            }
        }
    }

    /// Bytes reais por neurônio (INT4 = 0.5, mas retorna 1 porque compartilha byte).
    pub fn bytes_reais(&self) -> usize {
        match self {
            PesoNeuronio::FP32(_) => 4,
            PesoNeuronio::FP16(_) => 2,
            PesoNeuronio::INT8(_) => 1,
            PesoNeuronio::INT4(_) => 1, // dois INT4 por byte quando empacotados
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 3: NEURÔNIO HÍBRIDO
// ─────────────────────────────────────────────────────────────────────────────

/// Neurônio Izhikevich com precisão mista e plasticidade STDP.
///
/// # Campos de estado (sempre FP32 para estabilidade numérica)
/// - `v`: potencial de membrana em mV (repouso: -65mV, spike: +30mV)
/// - `u`: variável de recuperação (corrente de K⁺, feedback negativo)
///
/// # Plasticidade STDP
/// - `trace_pre`: traço pré-sináptico — sobe a 1.0 no spike, decai com tau=20ms
/// - LTP: quando neurônio dispara com trace_pre alto → peso aumenta
/// - LTD: quando neurônio dispara sem trace_pre (anti-Hebbian) → peso diminui
///
/// # Período refratário
/// - `refr_count`: contador de steps em refratório absoluto
/// - Durante refratário: v clampeado em -70mV, sem processamento
///
/// # Threshold adaptivo
/// - `threshold`: sobe após spike (fadiga), retorna ao padrão em repouso
/// - Previne hiperativação com correntes de entrada muito altas
#[derive(Debug, Clone)]
pub struct NeuronioHibrido {
    // Identificação compacta: u32 (4 bytes) em vez de Uuid (16 bytes)
    pub id: u32,

    // Tipo neuronal — determina a,b,c,d (não armazenados por instância)
    pub tipo: TipoNeuronal,

    // Precisão do peso sináptico
    pub precisao: PrecisionType,
    pub peso: PesoNeuronio,

    // Estado Izhikevich (FP32 para estabilidade numérica)
    pub v: f32,  // potencial de membrana (mV)
    pub u: f32,  // variável de recuperação

    // Período refratário absoluto
    pub refr_count: u16,

    // Threshold adaptivo (sobe após spike, retorna ao padrão em repouso)
    pub threshold: f32,

    // STDP — dois traços para LTP e LTD separados
    pub trace_pre: f32,   // traço pré-sináptico (LTP)
    pub trace_pos: f32,   // traço pós-sináptico (LTD anti-Hebbiano)
    pub last_spike_ms: f32,
}

/// Constantes de plasticidade STDP.
/// Definidas como constantes do módulo para fácil ajuste e visibilidade.
const TAU_STDP_MS: f32 = 20.0;     // constante de tempo do traço (ms)
const LTP_RATE: f32   = 0.012;     // taxa de potenciação (Long-Term Potentiation)
const LTD_RATE: f32   = 0.006;     // taxa de depressão (Long-Term Depression)
const PESO_MAX: f32   = 2.5;       // peso máximo (previne runaway)
const PESO_MIN: f32   = 0.0;       // peso mínimo (sem pesos negativos no STDP)
const THRESHOLD_BASE_DELTA: f32 = 2.0;  // quanto o threshold sobe por spike
const THRESHOLD_DECAY: f32 = 0.995;     // quanto o threshold retorna por step

impl NeuronioHibrido {
    /// Cria um novo neurônio com tipo e precisão especificados.
    ///
    /// O ID é um u32 simples — único dentro da camada, não globalmente.
    /// Isso economiza 12 bytes por neurônio comparado ao UUID.
    pub fn new(id: u32, tipo: TipoNeuronal, precisao: PrecisionType) -> Self {
        let peso = match precisao {
            PrecisionType::FP32 => PesoNeuronio::FP32(1.0),
            PrecisionType::FP16 => PesoNeuronio::FP16(f16::from_f32(1.0)),
            PrecisionType::INT8 => PesoNeuronio::INT8(100), // 100 * escala_camada ≈ 1.0
            PrecisionType::INT4 => PesoNeuronio::INT4(Int4Par::novo(7, 7).0), // ~1.0 com escala dinâmica
        };

        Self {
            id,
            tipo,
            precisao,
            peso,
            v: -65.0,
            u: 0.0,
            refr_count: 0,
            threshold: tipo.threshold_padrao(),
            trace_pre: 0.0,
            trace_pos: 0.0,
            last_spike_ms: -1000.0,
        }
    }

    /// Integra um passo de simulação e retorna `true` se houve spike.
    ///
    /// # Correção do dt (Bug crítico resolvido)
    /// O modelo Izhikevich foi publicado com dt em **milissegundos**.
    /// Este método recebe `dt_segundos` (como vem da Config) e converte
    /// internamente para substeps de 1ms, garantindo estabilidade numérica
    /// em qualquer frequência de simulação (100Hz a 6400Hz).
    ///
    /// # Parâmetros
    /// - `input_current`: corrente de entrada em pA (picoampères)
    /// - `dt_segundos`: passo de tempo em SEGUNDOS (ex: 0.005 para 200Hz)
    /// - `current_time_ms`: tempo atual em milissegundos (para STDP)
    /// - `escala_camada`: fator de escala para quantização INT8/INT4
    pub fn update(
        &mut self,
        input_current: f32,
        dt_segundos: f32,
        current_time_ms: f32,
        escala_camada: f32,
    ) -> bool {
        // ── Período refratário absoluto ────────────────────────────────────
        // Durante o refratário (1-3ms após spike), o canal de sódio está inativo
        // e o neurônio não pode disparar independentemente da corrente recebida.
        if self.refr_count > 0 {
            self.refr_count -= 1;
            self.v = -70.0; // hiperpolarização pós-spike
            // Traços STDP continuam decaindo durante o refratário
            let dt_ms = dt_segundos * 1000.0;
            self.trace_pre *= (-dt_ms / TAU_STDP_MS).exp();
            self.trace_pos *= (-dt_ms / TAU_STDP_MS).exp();
            // Threshold adaptivo retorna ao padrão em repouso
            let threshold_base = self.tipo.threshold_padrao();
            self.threshold = threshold_base + (self.threshold - threshold_base) * THRESHOLD_DECAY;
            return false;
        }

        // ── Adaptação do input à precisão do neurônio ─────────────────────
        // Neurônios INT8/INT4 quantizam a corrente de entrada.
        // CORREÇÃO: a escala é dinâmica (baseada no range real das correntes),
        // não fixa em 0.125 como antes — o que causava overflow de 97.7%.
        let input_adaptado = match self.precisao {
            PrecisionType::INT8 => {
                // Quantiza para 8 bits usando a escala da camada
                let quantizado = (input_current / escala_camada)
                    .round()
                    .clamp(-128.0, 127.0) as i8;
                (quantizado as f32) * escala_camada
            }
            PrecisionType::INT4 => {
                // Quantiza para 4 bits usando escala adaptativa
                // Range INT4: -8 a +7, então escala = I_max / 7.0
                let quantizado = (input_current / escala_camada)
                    .round()
                    .clamp(-8.0, 7.0) as i8;
                (quantizado as f32) * escala_camada
            }
            _ => input_current, // FP32 e FP16: sem quantização do input
        };

        // ── Conversão dt_segundos → substeps de 1ms ───────────────────────
        // CORREÇÃO DO BUG CRÍTICO: o modelo Izhikevich exige dt em ms.
        // Convertemos o dt externo (segundos) para ms e subdividimos em
        // steps de ~1ms. Isso garante estabilidade numérica e frequência
        // de disparo correta em TODOS os modos de operação.
        //
        // Exemplos:
        //   Boost200: dt=0.005s → 5ms → 5 substeps de 1ms
        //   Humano:   dt=0.01s  → 10ms → 10 substeps de 1ms
        //   Ultra:    dt=0.0003s → 0.3ms → 1 substep de 0.3ms
        let dt_ms = dt_segundos * 1000.0;
        let n_substeps = (dt_ms.round() as usize).max(1);
        let dt_interno = dt_ms / n_substeps as f32; // ~1ms por substep

        // Recupera parâmetros do tipo neuronal (sem alocar — são constantes)
        let (a, b, c, d) = self.tipo.parametros();

        let mut spiked = false;

        for _ in 0..n_substeps {
            // ── Equação de Izhikevich (dt_interno em ms) ─────────────────
            // dv/dt = 0.04v² + 5v + 140 − u + I
            // du/dt = a(bv − u)
            //
            // O fator 0.04v² vem da condutância do canal de potássio
            // linearizado. Valores típicos: v ∈ [-70, +30] mV
            self.v += dt_interno * (
                0.04 * self.v.powi(2)
                + 5.0 * self.v
                + 140.0
                - self.u
                + input_adaptado
            );
            self.u += dt_interno * a * (b * self.v - self.u);

            // ── Detecção de spike ─────────────────────────────────────────
            // Threshold adaptivo: sobe após cada spike (fadiga de curto prazo)
            // e retorna ao padrão em repouso — previne hiperativação.
            if self.v >= self.threshold {
                // Reset de Izhikevich
                self.v = c;
                self.u += d;
                spiked = true;

                // Threshold adaptivo: sobe após spike
                self.threshold += THRESHOLD_BASE_DELTA;

                // Período refratário: calcula quantos substeps de 1ms ficam
                // bloqueados. Para 2ms de refratário e dt_interno=1ms: 2 steps.
                // CORREÇÃO: antes era (2.0/dt_segundos) = 400 steps = 2000ms!
                let refr_ms = 2.0f32; // 2ms de refratário absoluto biológico
                self.refr_count = (refr_ms / dt_interno).round() as u16;

                break; // um spike por tick (não pode disparar duas vezes no mesmo dt)
            }
        }

        // ── Decaimento dos traços STDP ────────────────────────────────────
        // CORREÇÃO: usa dt_ms (não dt_segundos) para decay correto.
        // exp(-1ms/20ms) = 0.951 → meia-vida de ~14ms (biológico: ~10-30ms)
        // Antes: exp(-0.005/20) = 0.99975 → meia-vida de 28 SEGUNDOS
        let decay = (-dt_ms / TAU_STDP_MS).exp();
        self.trace_pre *= decay;
        self.trace_pos *= decay;

        // ── Atualização de STDP no spike ──────────────────────────────────
        if spiked {
            // LTP (Long-Term Potentiation): spike quando traço pré estava alto
            // → o neurônio pré disparou pouco antes deste → correlação → potencia
            let delta_ltp = LTP_RATE * self.trace_pre;

            // LTD anti-Hebbiana: spike sem correlação prévia → deprime
            // Isso implementa o "anti-Hebb": se pré não disparou antes de pós,
            // a sinapse foi ativada de forma anticausal → enfraquece.
            let delta_ltd = if self.trace_pre < 0.1 {
                -LTD_RATE * (1.0 - self.trace_pre)
            } else {
                0.0
            };

            // Atualiza peso de acordo com a precisão
            self.atualizar_peso(delta_ltp + delta_ltd);

            // Traço pós sobe no spike (para LTD do próximo neurônio)
            self.trace_pos = 1.0;
            // Traço pré também sobe (auto-correlação)
            self.trace_pre = (self.trace_pre + 0.5).min(1.0);

            self.last_spike_ms = current_time_ms;
        }

        // Threshold adaptivo retorna ao padrão em repouso (a cada step)
        let threshold_base = self.tipo.threshold_padrao();
        self.threshold = threshold_base + (self.threshold - threshold_base) * THRESHOLD_DECAY;

        spiked
    }

    /// Atualiza o peso sináptico preservando a precisão do neurônio.
    fn atualizar_peso(&mut self, delta: f32) {
        match &mut self.peso {
            PesoNeuronio::FP32(v) => {
                *v = (*v + delta).clamp(PESO_MIN, PESO_MAX);
            }
            PesoNeuronio::FP16(v) => {
                let novo = (v.to_f32() + delta).clamp(PESO_MIN, PESO_MAX);
                *v = f16::from_f32(novo);
            }
            PesoNeuronio::INT8(v) => {
                // INT8: quantiza o delta e aplica diretamente ao valor inteiro
                let novo_f = (*v as f32) + delta * 10.0; // escala interna
                *v = novo_f.clamp(-127.0, 127.0) as i8;
            }
            PesoNeuronio::INT4(raw) => {
                // INT4: atualiza apenas o nibble alto (nibble baixo = reserva)
                let par = Int4Par(*raw);
                let novo = (par.alto() as f32 + delta * 4.0)
                    .clamp(-8.0, 7.0) as i8;
                *raw = Int4Par::novo(novo, par.baixo()).0;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 4: CAMADA HÍBRIDA
// ─────────────────────────────────────────────────────────────────────────────

/// Camada de neurônios híbridos com distribuição mista de precisão e tipo.
///
/// Encapsula um vetor de `NeuronioHibrido` e fornece:
/// - `update()`: processa todos os neurônios em batch
/// - `escala_camada`: escala compartilhada para INT8/INT4 (dinâmica)
/// - Estatísticas de distribuição de precisão e tipo
///
/// A escala dinâmica resolve o bug de overflow de INT4 (antes fixada em 0.125,
/// agora calculada como `I_max / 7.0` para cobrir o range real das correntes).
#[derive(Debug)]
pub struct CamadaHibrida {
    pub neuronios: Vec<NeuronioHibrido>,

    /// Escala compartilhada para quantização INT8/INT4.
    /// CORREÇÃO: antes declarada mas nunca usada. Agora passada ao update() de cada neurônio.
    /// Calculada como: I_max_esperado / 127.0 para INT8, I_max / 7.0 para INT4.
    pub escala_camada: f32,

    /// Nome da camada (para logging e checkpointing).
    pub nome: String,
}

impl CamadaHibrida {
    /// Cria uma nova camada com distribuição de tipos e precisão configurável.
    ///
    /// # Parâmetros
    /// - `n_neurons`: quantidade de neurônios
    /// - `nome`: identificador da camada
    /// - `tipo_principal`: tipo neuronal dominante (RS para maioria das camadas)
    /// - `tipo_secundario`: tipo minoritário e sua proporção (ex: FS para 20% de inibição)
    /// - `distribuicao_precisao`: proporções de FP32/FP16/INT8/INT4
    /// - `escala_camada`: escala para quantização (I_max / 127 para INT8)
    pub fn new(
        n_neurons: usize,
        nome: &str,
        tipo_principal: TipoNeuronal,
        tipo_secundario: Option<(TipoNeuronal, f32)>,
        distribuicao_precisao: Option<Vec<(PrecisionType, f32)>>,
        escala_camada: f32,
    ) -> Self {
        let mut neuronios = Vec::with_capacity(n_neurons);

        // Distribuição de precisão (normalizada para soma = 1.0)
        let dist = distribuicao_precisao.unwrap_or_else(|| vec![
            (PrecisionType::FP32, 0.05),
            (PrecisionType::FP16, 0.35),
            (PrecisionType::INT8, 0.50),
            (PrecisionType::INT4, 0.10),
        ]);
        let total: f32 = dist.iter().map(|(_, p)| p).sum();
        let dist: Vec<(PrecisionType, f32)> = dist.into_iter()
            .map(|(t, p)| (t, p / total))
            .collect();

        // Proporção do tipo secundário (ex: 20% FS para inibição lateral)
        let prop_secundario = tipo_secundario.map(|(_, p)| p).unwrap_or(0.0);
        let tipo_sec = tipo_secundario.map(|(t, _)| t).unwrap_or(tipo_principal);

        // Construção dos neurônios: distribuição de tipo e precisão por índice
        let mut acumulado_prec = 0.0f32;
        let mut prec_iter = dist.iter().peekable();
        let (mut precisao_atual, mut prob_prec) = *prec_iter.next().unwrap();

        for i in 0..n_neurons {
            let progresso = i as f32 / n_neurons as f32;

            // Avança a distribuição de precisão conforme o progresso
            while progresso > acumulado_prec + prob_prec {
                acumulado_prec += prob_prec;
                if let Some((t, p)) = prec_iter.next() {
                    precisao_atual = *t;
                    prob_prec = *p;
                } else {
                    break;
                }
            }

            // Tipo neuronal: últimos N% são do tipo secundário
            let tipo = if progresso >= (1.0 - prop_secundario) {
                tipo_sec
            } else {
                tipo_principal
            };

            neuronios.push(NeuronioHibrido::new(i as u32, tipo, precisao_atual));
        }

        Self {
            neuronios,
            escala_camada,
            nome: nome.to_string(),
        }
    }

    /// Processa um tick para todos os neurônios da camada.
    ///
    /// `input_currents`: correntes de entrada em pA para cada neurônio.
    /// Se o slice for menor que o número de neurônios, o restante recebe 0.0.
    /// A escala_camada é passada a cada neurônio para quantização correta de INT8/INT4.
    pub fn update(&mut self, input_currents: &[f32], dt: f32, current_time_ms: f32) -> Vec<bool> {
        let escala = self.escala_camada;
        self.neuronios.iter_mut().enumerate().map(|(i, n)| {
            let input = input_currents.get(i).copied().unwrap_or(0.0);
            n.update(input, dt, current_time_ms, escala)
        }).collect()
    }

    /// Versão com compressão de spikes (pontos salientes).
    pub fn update_compact(
        &mut self,
        input_points: &[Vec<SalientPoint>],
        dt: f32,
        current_time_ms: f32,
        compressor: &SalientCompressor,
    ) -> Vec<bool> {
        let escala = self.escala_camada;
        self.neuronios.iter_mut().enumerate().map(|(i, n)| {
            let pontos = input_points.get(i).cloned().unwrap_or_default();
            let reconstructed = compressor.decompress(&pontos);
            let input = reconstructed.iter().sum::<f32>() / reconstructed.len().max(1) as f32;
            n.update(input, dt, current_time_ms, escala)
        }).collect()
    }

    /// Adiciona um novo neurônio à camada (neurogênese).
    pub fn adicionar_neuronio(&mut self, tipo: TipoNeuronal, precisao: PrecisionType) -> u32 {
        let id = self.neuronios.len() as u32;
        self.neuronios.push(NeuronioHibrido::new(id, tipo, precisao));
        id
    }

    /// Estatísticas da distribuição de precisão e tipo neuronal.
    pub fn estatisticas(&self) -> CamadaStats {
        let mut stats = CamadaStats::default();
        stats.total = self.neuronios.len();

        for n in &self.neuronios {
            match n.precisao {
                PrecisionType::FP32 => stats.fp32 += 1,
                PrecisionType::FP16 => stats.fp16 += 1,
                PrecisionType::INT8 => stats.int8 += 1,
                PrecisionType::INT4 => stats.int4 += 1,
            }
            match n.tipo {
                TipoNeuronal::RS => stats.tipo_rs += 1,
                TipoNeuronal::IB => stats.tipo_ib += 1,
                TipoNeuronal::CH => stats.tipo_ch += 1,
                TipoNeuronal::FS => stats.tipo_fs += 1,
                TipoNeuronal::LT => stats.tipo_lt += 1,
                TipoNeuronal::TC => stats.tipo_tc += 1,
                TipoNeuronal::RZ => stats.tipo_rz += 1,
            }
            stats.bytes_total += n.peso.bytes_reais();
        }
        stats
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 5: ESTATÍSTICAS
// ─────────────────────────────────────────────────────────────────────────────

/// Estatísticas de uma camada: distribuição de precisão, tipo e memória.
#[derive(Debug, Default)]
pub struct CamadaStats {
    pub total:       usize,
    // Precisão
    pub fp32:        usize,
    pub fp16:        usize,
    pub int8:        usize,
    pub int4:        usize,
    pub bytes_total: usize,
    // Tipo neuronal
    pub tipo_rs:     usize,
    pub tipo_ib:     usize,
    pub tipo_ch:     usize,
    pub tipo_fs:     usize,
    pub tipo_lt:     usize,
    pub tipo_tc:     usize,
    pub tipo_rz:     usize,
}

impl CamadaStats {
    pub fn bytes_por_neuronio(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.bytes_total as f32 / self.total as f32 }
    }

    pub fn prop_inibitorios(&self) -> f32 {
        if self.total == 0 { 0.0 }
        else { (self.tipo_fs + self.tipo_lt) as f32 / self.total as f32 }
    }
}
