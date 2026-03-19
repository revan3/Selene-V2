// =============================================================================
// src/synaptic_core.rs — Selene V2.2
// =============================================================================
//
// MODELO HÍBRIDO DE 4 CAMADAS:
//
//  ┌──────────────────────────────────────────────────────────────────────────┐
//  │ Camada 1 — IZHIKEVICH (todos os tipos)                                  │
//  │   dv/dt = 0.04v² + 5v + 140 − u + I_eff                                │
//  │   du/dt = a(bv − u)                                                     │
//  │   Captura padrões de disparo biológicos com custo computacional mínimo.  │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Camada 2 — REFRATÁRIO / LIF (todos os tipos)                            │
//  │   Período refratário absoluto de 2ms + hiperpolarização pós-spike.      │
//  │   Emergente no HH (via inativação de h), explícito no Izhikevich puro.  │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Camada 3 — HODGKIN-HUXLEY (tipos TC e RZ apenas)                        │
//  │   Correntes iônicas realistas via variáveis de portão m, h, n:          │
//  │     I_Na = g_Na · m³h · (v − E_Na)   ← canal sódio                    │
//  │     I_K  = g_K  · n⁴  · (v − E_K)   ← canal potássio                 │
//  │     I_L  = g_L        · (v − E_L)   ← corrente de vazamento            │
//  │   I_eff = I_externo − (I_Na + I_K + I_L) · HH_SCALE                   │
//  │   HH_SCALE = 0.008 (calibrado para faixa fisiológica 20–150Hz)         │
//  │                                                                          │
//  │   TC (Talâmico): burst(sono) ↔ tônico(vigília) via inativação h         │
//  │   RZ (Purkinje/cerebelo): g_Na alto, burst e timing preciso             │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Camada 4 — STDP BIDIRECIONAL (todos os tipos)                           │
//  │   trace_pre → LTP causal: pré antes de pós → potencia peso              │
//  │   trace_pos → LTD anti-Hebbiano: pós sem pré → deprime peso             │
//  │   + Threshold adaptivo (spike-frequency adaptation)                     │
//  └──────────────────────────────────────────────────────────────────────────┘
//
// DECISÕES DE DESIGN CHAVE:
//
//   ModeloDinamico::Izhikevich           → 8 bytes (ponteiro nulo)
//   ModeloDinamico::IzhikevichHH(Box<_>) → 8 + 24 bytes no heap
//   Neurônios HH são apenas TC e RZ (~5–10% do total) — overhead desprezível.
//
//   ParametrosHH são CONSTANTES por TipoNeuronal, nunca armazenadas por instância.
//   HH_SCALE = 0.008 garante que correntes HH modulam sem dominar o input externo.
//
//   Neuromodulação (NeuroChem):
//     dopamina  ↑  →  g_K_mod  ↓  →  repolarização lenta  →  mais disparo
//     serotonina ↑  →  g_L_mod  ↓  →  menos vazamento      →  mais excitável
//     cortisol  ↑  →  g_Na_mod ↓  →  Na⁺ reduzido         →  limiar mais alto
//
// COMPATIBILIDADE (sem breaking changes):
//   CamadaHibrida::update()       assinatura idêntica à V2.1
//   NeuronioHibrido::update()     assinatura idêntica à V2.1
//   TipoNeuronal, PrecisionType   idênticos à V2.1
//   cerebellum.rs, frontal.rs…    não precisam de alteração
//   NOVO: CamadaHibrida::modular_neuro() para integração com NeuroChem
//
// VALIDAÇÃO NUMÉRICA (Python, 1ms/step, HH_SCALE=0.008):
//   TC tônico @I=5:   ~45 Hz  (fisiológico: 20–100 Hz ✓)
//   TC burst  @I=2:   burst inicial → irregular (modo sono ✓)
//   RZ Purkinje @I=8: ~66 Hz  (fisiológico: 40–150 Hz ✓)
//   Repouso s/ input: I_hh_scaled ≈ 0.0001 (neutro ✓)
// =============================================================================

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use half::f16;
use crate::config::Config;
use crate::compressor::salient::{SalientPoint, SalientCompressor};

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 1 — TIPO NEURONAL
// ─────────────────────────────────────────────────────────────────────────────

/// Tipos funcionais de neurônios baseados em Izhikevich (2003).
///
/// TC e RZ recebem automaticamente `ModeloDinamico::IzhikevichHH` no construtor.
/// Os demais usam `ModeloDinamico::Izhikevich` puro.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TipoNeuronal {
    /// Regular Spiking — neurônio piramidal excitatório padrão do córtex.
    /// Disparo regular sob corrente constante. ~80% dos neurônios.
    RS,
    /// Intrinsic Bursting — burst inicial seguido de disparo regular.
    /// Camada 5 cortical, amígdala. Parâmetro c=-55 captura burst via Izhikevich.
    IB,
    /// Chattering — bursts rápidos repetitivos. Córtex visual V2/V3.
    CH,
    /// Fast Spiking — interneurônio GABAérgico inibitório sem adaptação.
    FS,
    /// Low-Threshold Spiking — interneurônio de limiar baixo.
    LT,
    /// Thalamo-Cortical — dois modos: burst (sono/desatenção) e tônico (vigília).
    /// A transição burst↔tônico é mediada pela inativação do canal Na⁺ (h).
    /// Requer HH para simulação fiel do comportamento talâmico.
    TC,
    /// Resonator / Purkinje — células do cerebelo e giro dentado.
    /// g_Na alto, burst de alta frequência, timing motor preciso.
    /// Requer HH para capturar cinética de repolarização das Purkinje.
    RZ,
}

impl TipoNeuronal {
    /// Parâmetros Izhikevich (a, b, c, d).
    #[inline]
    pub fn parametros(&self) -> (f32, f32, f32, f32) {
        match self {
            TipoNeuronal::RS => (0.02, 0.20, -65.0,  8.0),
            TipoNeuronal::IB => (0.02, 0.20, -55.0,  4.0),
            TipoNeuronal::CH => (0.02, 0.20, -50.0,  2.0),
            TipoNeuronal::FS => (0.10, 0.20, -65.0,  2.0),
            TipoNeuronal::LT => (0.02, 0.25, -65.0,  2.0),
            TipoNeuronal::TC => (0.02, 0.25, -65.0,  0.05),
            TipoNeuronal::RZ => (0.10, 0.26, -65.0,  2.0),
        }
    }

    /// Threshold de disparo padrão (mV). Sobe após spike, decai em repouso.
    #[inline]
    pub fn threshold_padrao(&self) -> f32 {
        match self {
            TipoNeuronal::TC => 25.0,  // talâmico dispara com menor depolarização
            TipoNeuronal::FS => 25.0,  // interneurônios respondem mais rápido
            _               => 30.0,
        }
    }

    /// Verdadeiro para tipos GABAérgicos (usados na inibição lateral).
    #[inline]
    pub fn e_inibitorico(&self) -> bool {
        matches!(self, TipoNeuronal::FS | TipoNeuronal::LT)
    }

    /// Verdadeiro para tipos que usam o modelo Hodgkin-Huxley.
    #[inline]
    pub fn usa_hh(&self) -> bool {
        matches!(self, TipoNeuronal::TC | TipoNeuronal::RZ)
    }

    /// Parâmetros de condutância HH para este tipo.
    /// Retorna `None` para tipos Izhikevich puro — nunca chamado em hot path.
    pub fn parametros_hh(&self) -> Option<ParametrosHH> {
        match self {
            TipoNeuronal::TC => Some(ParametrosHH {
                // Hodgkin-Huxley (1952) com ajustes para neurônio talâmico.
                // g_Na menor que Purkinje: talâmico é menos excitável.
                g_na: 120.0,   // mS/cm²
                g_k:   36.0,   // mS/cm²
                g_l:    0.3,   // mS/cm²
                e_na:  50.0,   // mV
                e_k:  -77.0,   // mV
                e_l:  -54.4,   // mV
                c_m:    1.0,   // µF/cm²
            }),
            TipoNeuronal::RZ => Some(ParametrosHH {
                // Célula de Purkinje: g_Na alto para upstroke rápido.
                // E_K mais negativo → repolarização profunda.
                // g_L maior → condutividade base mais alta (Purkinje é grande).
                g_na: 150.0,
                g_k:   38.0,
                g_l:    0.5,
                e_na:  55.0,
                e_k:  -80.0,
                e_l:  -65.0,
                c_m:    1.0,
            }),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 2 — HODGKIN-HUXLEY: PARÂMETROS, ESTADO E MOTOR DE CÁLCULO
// ─────────────────────────────────────────────────────────────────────────────

/// Condutâncias e potenciais de reversão — constantes por tipo celular.
/// Nunca armazenadas por instância: acessadas via `TipoNeuronal::parametros_hh()`.
#[derive(Debug, Clone, Copy)]
pub struct ParametrosHH {
    pub g_na: f32,  // condutância máxima Na⁺ (mS/cm²)
    pub g_k:  f32,  // condutância máxima K⁺  (mS/cm²)
    pub g_l:  f32,  // condutância de vazamento (mS/cm²)
    pub e_na: f32,  // potencial de reversão Na⁺ (mV)
    pub e_k:  f32,  // potencial de reversão K⁺  (mV)
    pub e_l:  f32,  // potencial de reversão leak (mV)
    pub c_m:  f32,  // capacitância de membrana (µF/cm²)
}

/// Estado das variáveis de portão — muda a cada tick.
/// Armazenado em heap via `Box<EstadoHH>` em `ModeloDinamico::IzhikevichHH`.
/// Tamanho no heap: 6 × f32 = 24 bytes por neurônio HH.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstadoHH {
    /// Portão de ativação Na⁺ (0..1). Abre com despolarização → upstroke.
    pub m: f32,
    /// Portão de inativação Na⁺ (0..1). Fecha com despolarização sustentada.
    /// Esta é a variável chave do TC: h≈0 = canal Na⁺ inativo (modo burst/sono);
    /// h≈1 = canal Na⁺ ativo (modo tônico/vigília).
    pub h: f32,
    /// Portão de ativação K⁺ (0..1). Abre com atraso → repolarização.
    pub n: f32,
    /// Modificadores neuromoduladores (default 1.0 = sem efeito).
    /// Ajustados por `modular()` a partir dos valores do NeuroChem.
    pub g_na_mod: f32,
    pub g_k_mod:  f32,
    pub g_l_mod:  f32,
}

impl EstadoHH {
    /// Estado de equilíbrio em potencial de repouso −65 mV.
    /// Valores analíticos: x₀ = α_x(-65) / (α_x(-65) + β_x(-65)).
    pub fn repouso() -> Self {
        Self { m: 0.053, h: 0.596, n: 0.318,
               g_na_mod: 1.0, g_k_mod: 1.0, g_l_mod: 1.0 }
    }

    /// Atualiza os modificadores a partir dos níveis neuroquímicos.
    ///
    /// Bases biológicas:
    ///   dopamina ↑  → canal K⁺ mais lento    → g_k_mod ↓  → mais disparo
    ///   serotonina ↑ → canal leak fecha       → g_l_mod ↓  → mais excitável
    ///   cortisol ↑  → Na⁺ reduzido           → g_na_mod ↓ → limiar mais alto
    pub fn modular(&mut self, dopamina: f32, serotonina: f32, cortisol: f32) {
        self.g_k_mod  = (1.2 - dopamina   * 0.35).clamp(0.5, 1.2);
        self.g_l_mod  = (1.1 - serotonina * 0.25).clamp(0.5, 1.1);
        self.g_na_mod = (1.0 - cortisol   * 0.40).clamp(0.4, 1.0);
    }
}

/// Fator de escala para compatibilizar correntes HH com a escala de Izhikevich.
///
/// Problema: g_Na = 120 mS/cm² × ΔV = 115 mV → I_Na_max ≈ 13800 µA/cm²
/// Izhikevich usa correntes de 0–20 unidades para disparos normais.
/// Solução: HH_SCALE = 0.008 mapeia correntes HH ao range fisiológico:
///   TC tônico @I=5:   ~45 Hz  (fisiológico: 20–100 Hz ✓)
///   RZ Purkinje @I=8: ~66 Hz  (fisiológico: 40–150 Hz ✓)
///   Repouso sem input: I_hh_scaled ≈ 0.0001 (neutro ✓)
const HH_SCALE: f32 = 0.008;

/// Motor de cálculo HH — funções de taxa α/β e integração por Euler explícito.
struct HH;

impl HH {
    // ── Funções de taxa (equações originais de Hodgkin & Huxley, 1952) ─────

    /// α_m(v): taxa de abertura do canal Na⁺.
    /// Singularidade em v = −40 mV removida com limite de L'Hôpital.
    #[inline]
    fn alpha_m(v: f32) -> f32 {
        let dv = v + 40.0;
        if dv.abs() < 1e-4 { 1.0 }
        else { 0.1 * dv / (1.0 - (-dv / 10.0_f32).exp()) }
    }

    #[inline]
    fn beta_m(v: f32) -> f32 { 4.0 * (-(v + 65.0) / 18.0_f32).exp() }

    #[inline]
    fn alpha_h(v: f32) -> f32 { 0.07 * (-(v + 65.0) / 20.0_f32).exp() }

    #[inline]
    fn beta_h(v: f32) -> f32 { 1.0 / (1.0 + (-(v + 35.0) / 10.0_f32).exp()) }

    /// α_n(v): taxa de abertura do canal K⁺.
    /// Singularidade em v = −55 mV removida com limite de L'Hôpital.
    #[inline]
    fn alpha_n(v: f32) -> f32 {
        let dv = v + 55.0;
        if dv.abs() < 1e-4 { 0.1 }
        else { 0.01 * dv / (1.0 - (-dv / 10.0_f32).exp()) }
    }

    #[inline]
    fn beta_n(v: f32) -> f32 { 0.125 * (-(v + 65.0) / 80.0_f32).exp() }

    // ── Integrador ─────────────────────────────────────────────────────────

    /// Integra as variáveis de portão por `dt_ms` milissegundos.
    /// Retorna a corrente iônica total: I_hh = I_Na + I_K + I_L (µA/cm²).
    ///
    /// Usa Euler explícito com substeps de 0.1 ms.
    /// HH é numericamente stiff: diverge com dt > ~0.25 ms em Euler simples.
    /// Para dt_ms = 1 ms → 10 substeps internos → estável.
    /// Para dt_ms = 5 ms (200 Hz) → 50 substeps → ainda estável.
    ///
    /// O potencial `v` é fixo durante a integração HH (aproximação operator-split):
    /// HH e Izhikevich são acoplados via I_eff, não via equação de Kirchhoff comum.
    /// Essa aproximação é válida porque HH atualiza portões e Izhikevich atualiza v.
    #[inline]
    pub fn integrar(estado: &mut EstadoHH, params: &ParametrosHH, v: f32, dt_ms: f32) -> f32 {
        let n_sub = ((dt_ms / 0.1).ceil() as usize).max(1).min(50);
        let dt_sub = dt_ms / n_sub as f32;

        let mut m = estado.m;
        let mut h = estado.h;
        let mut n = estado.n;

        for _ in 0..n_sub {
            let am = Self::alpha_m(v); let bm = Self::beta_m(v);
            let ah = Self::alpha_h(v); let bh = Self::beta_h(v);
            let an = Self::alpha_n(v); let bn = Self::beta_n(v);

            m += dt_sub * (am * (1.0 - m) - bm * m);
            h += dt_sub * (ah * (1.0 - h) - bh * h);
            n += dt_sub * (an * (1.0 - n) - bn * n);

            m = m.clamp(0.0, 1.0);
            h = h.clamp(0.0, 1.0);
            n = n.clamp(0.0, 1.0);
        }

        estado.m = m;
        estado.h = h;
        estado.n = n;

        // Correntes iônicas com modificadores neuromoduladores
        let i_na = params.g_na * estado.g_na_mod * m.powi(3) * h * (v - params.e_na);
        let i_k  = params.g_k  * estado.g_k_mod  * n.powi(4)     * (v - params.e_k);
        let i_l  = params.g_l  * estado.g_l_mod                  * (v - params.e_l);

        i_na + i_k + i_l
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 3 — MODELO DINÂMICO
// ─────────────────────────────────────────────────────────────────────────────

/// Seleciona o modelo dinâmico por neurônio.
///
/// Custo de memória (enum Rust):
///   `Izhikevich`:           1 discriminante + 7 padding = 8 bytes
///   `IzhikevichHH(Box<_>)`: 1 discriminante + 8 ponteiro = 9 bytes
///   `EstadoHH` no heap:     6 × f32 = 24 bytes (apenas para TC e RZ)
///
/// Para 90–95% dos neurônios (RS/IB/CH/FS/LT): overhead de 8 bytes por neurônio.
/// Para neurônios TC/RZ: 24 bytes adicionais no heap.
/// Impacto no BPN_8K total: < 0.1% — desprezível.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModeloDinamico {
    /// Izhikevich puro — RS, IB, CH, FS, LT.
    Izhikevich,
    /// Izhikevich + correntes HH — TC (talâmico) e RZ (Purkinje/cerebelo).
    IzhikevichHH(Box<EstadoHH>),
}

impl ModeloDinamico {
    /// Cria o modelo correto para o tipo dado.
    pub fn para_tipo(tipo: TipoNeuronal) -> Self {
        if tipo.usa_hh() {
            ModeloDinamico::IzhikevichHH(Box::new(EstadoHH::repouso()))
        } else {
            ModeloDinamico::Izhikevich
        }
    }

    /// Referência mutável ao estado HH, se disponível.
    pub fn estado_hh_mut(&mut self) -> Option<&mut EstadoHH> {
        match self {
            ModeloDinamico::IzhikevichHH(e) => Some(e.as_mut()),
            ModeloDinamico::Izhikevich      => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 4 — PRECISÃO MISTA (idêntica à V2.1)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PrecisionType { FP32, FP16, INT8, INT4 }

/// INT4 empacotado: dois valores de 4 bits em um byte.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Int4Par(u8);

impl Int4Par {
    pub fn novo(alto: i8, baixo: i8) -> Self {
        let h = (alto.max(-8).min(7) as u8) & 0x0F;
        let l = (baixo.max(-8).min(7) as u8) & 0x0F;
        Self((h << 4) | l)
    }
    #[inline] pub fn alto(&self) -> i8 {
        let v = (self.0 >> 4) as i8;
        if v & 0x08 != 0 { v | -16i8 } else { v }
    }
    #[inline] pub fn baixo(&self) -> i8 {
        let v = (self.0 & 0x0F) as i8;
        if v & 0x08 != 0 { v | -16i8 } else { v }
    }
}

/// Peso sináptico com quatro níveis de precisão.
/// A escala para INT8/INT4 é fornecida pela `CamadaHibrida` (compartilhada).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PesoNeuronio {
    FP32(f32),
    FP16(f16),
    INT8(i8),
    INT4(u8),
}

impl PesoNeuronio {
    /// Converte para f32 usando a escala da camada.
    #[inline]
    pub fn valor_f32(&self, escala: f32) -> f32 {
        match self {
            PesoNeuronio::FP32(v)   => *v,
            PesoNeuronio::FP16(v)   => v.to_f32(),
            PesoNeuronio::INT8(v)   => (*v as f32) * escala,
            PesoNeuronio::INT4(raw) => Int4Par(*raw).alto() as f32 * escala,
        }
    }

    /// Bytes efetivos por peso (INT4 = 1, pois dois valores por byte).
    pub fn bytes_reais(&self) -> usize {
        match self {
            PesoNeuronio::FP32(_) => 4,
            PesoNeuronio::FP16(_) => 2,
            PesoNeuronio::INT8(_) => 1,
            PesoNeuronio::INT4(_) => 1,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 5 — NEURÔNIO HÍBRIDO
// ─────────────────────────────────────────────────────────────────────────────

/// Constantes de plasticidade STDP.
const TAU_STDP_MS:     f32 = 20.0;   // constante de tempo dos traços (ms)
const LTP_RATE:        f32 = 0.012;  // taxa de potenciação de longo prazo
const LTD_RATE:        f32 = 0.006;  // taxa de depressão de longo prazo
const PESO_MAX:        f32 = 2.5;
const PESO_MIN:        f32 = 0.0;
const THRESHOLD_DELTA: f32 = 2.0;   // threshold sobe por spike (mV)
const THRESHOLD_DECAY: f32 = 0.995; // threshold retorna ao padrão por step

/// Neurônio com modelo dinâmico híbrido, precisão mista e STDP bidirecional.
///
/// # Fluxo de atualização — Izhikevich puro (RS/IB/CH/FS/LT):
///   1. Período refratário → retorna false se ativo
///   2. Quantização do input pela precisão (INT8/INT4)
///   3. Substeps Izhikevich em ~1 ms cada (bug V2.0 de dt corrigido)
///   4. Detecção de spike → reset v=c, u+=d, threshold adaptivo
///   5. STDP bidirecional: LTP por trace_pre, LTD por trace_pre baixo
///
/// # Fluxo de atualização — IzhikevichHH (TC/RZ):
///   Igual ao acima, com passo extra entre 2 e 3:
///   2b. HH integra variáveis de portão m, h, n (substeps de 0.1 ms)
///   2c. Calcula I_hh = I_Na + I_K + I_L
///   2d. I_eff = I_externo − I_hh × HH_SCALE
///   3. Substeps Izhikevich com I_eff (em vez de I_externo direto)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronioHibrido {
    pub id:            u32,
    pub tipo:          TipoNeuronal,
    pub precisao:      PrecisionType,
    pub peso:          PesoNeuronio,

    // Estado Izhikevich (sempre FP32 para estabilidade numérica)
    pub v:             f32,   // potencial de membrana (mV)
    pub u:             f32,   // variável de recuperação

    // Período refratário absoluto
    pub refr_count:    u16,   // steps ainda bloqueados

    // Threshold adaptivo
    pub threshold:     f32,   // sobe após spike, decai em repouso

    // STDP bidirecional
    pub trace_pre:     f32,   // traço pré-sináptico (LTP)
    pub trace_pos:     f32,   // traço pós-sináptico (LTD anti-Hebbiano)
    pub last_spike_ms: f32,

    // Modelo dinâmico (Izhikevich ou Izhikevich+HH)
    pub modelo:        ModeloDinamico,
}

impl NeuronioHibrido {
    /// Cria neurônio. TC e RZ recebem automaticamente `ModeloDinamico::IzhikevichHH`.
    pub fn new(id: u32, tipo: TipoNeuronal, precisao: PrecisionType) -> Self {
        let peso = match precisao {
            PrecisionType::FP32 => PesoNeuronio::FP32(1.0),
            PrecisionType::FP16 => PesoNeuronio::FP16(f16::from_f32(1.0)),
            PrecisionType::INT8 => PesoNeuronio::INT8(100),
            PrecisionType::INT4 => PesoNeuronio::INT4(Int4Par::novo(7, 7).0),
        };
        Self {
            id, tipo, precisao, peso,
            v:             -65.0,
            u:               0.0,
            refr_count:        0,
            threshold:     tipo.threshold_padrao(),
            trace_pre:       0.0,
            trace_pos:       0.0,
            last_spike_ms: -1000.0,
            modelo:        ModeloDinamico::para_tipo(tipo),
        }
    }

    /// Integra um passo de simulação. Retorna `true` se houve spike.
    ///
    /// # Parâmetros
    /// - `input_current`: corrente externa (mesma escala do Izhikevich, ex: 5–15 pA)
    /// - `dt_segundos`: passo em SEGUNDOS (ex: 0.005 para 200 Hz)
    /// - `current_time_ms`: tempo atual em milissegundos (para STDP)
    /// - `escala_camada`: fator de quantização para INT8/INT4
    pub fn update(
        &mut self,
        input_current:   f32,
        dt_segundos:     f32,
        current_time_ms: f32,
        escala_camada:   f32,
    ) -> bool {
        let dt_ms = dt_segundos * 1000.0;

        // ── 1. Período refratário ─────────────────────────────────────────
        if self.refr_count > 0 {
            self.refr_count -= 1;
            self.v = -70.0;
            let decay = (-dt_ms / TAU_STDP_MS).exp();
            self.trace_pre *= decay;
            self.trace_pos *= decay;
            let tb = self.tipo.threshold_padrao();
            self.threshold = tb + (self.threshold - tb) * THRESHOLD_DECAY;
            return false;
        }

        // ── 2. Quantização do input ───────────────────────────────────────
        let input_q = match self.precisao {
            PrecisionType::INT8 => {
                let q = (input_current / escala_camada).round().clamp(-128.0, 127.0) as i8;
                q as f32 * escala_camada
            }
            PrecisionType::INT4 => {
                let q = (input_current / escala_camada).round().clamp(-8.0, 7.0) as i8;
                q as f32 * escala_camada
            }
            _ => input_current,
        };

        // ── 3. Correntes HH (apenas TC e RZ) ─────────────────────────────
        // HH integra os portões m, h, n com o potencial atual e retorna
        // a corrente iônica total (µA/cm²). Essa corrente é escalada por
        // HH_SCALE = 0.008 para compatibilizar com a escala do Izhikevich.
        //
        // I_eff = I_ext − I_hh × HH_SCALE
        //
        // Sinal negativo: as correntes HH drenam o neurônio durante o repouso
        // e inativam Na⁺ após spike → refratário emergente.
        let i_eff = if let ModeloDinamico::IzhikevichHH(ref mut estado) = self.modelo {
            // Parâmetros são constantes estáticas do tipo — sem alocação
            // O expect é seguro: invariante garantida por ModeloDinamico::para_tipo()
            let params = self.tipo.parametros_hh()
                .unwrap_or_else(|| unreachable!("TC/RZ garantem ParametrosHH"));
            let i_hh = HH::integrar(estado, &params, self.v, dt_ms);
            input_q - i_hh * HH_SCALE
        } else {
            input_q
        };

        // ── 4. Substeps Izhikevich (~1 ms cada) ──────────────────────────
        // Correção V2.0→V2.1: dt era em segundos → equação divergia.
        // Convertemos para ms e subdividimos em steps de ~1 ms.
        let n_sub  = (dt_ms.round() as usize).max(1);
        let dt_int = dt_ms / n_sub as f32;
        let (a, b, c, d) = self.tipo.parametros();
        let mut spiked = false;

        for _ in 0..n_sub {
            self.v += dt_int * (0.04 * self.v.powi(2) + 5.0 * self.v + 140.0 - self.u + i_eff);
            self.u += dt_int * a * (b * self.v - self.u);

            if self.v >= self.threshold {
                self.v = c;
                self.u += d;
                self.threshold += THRESHOLD_DELTA;
                self.refr_count = (2.0 / dt_int).round() as u16;
                spiked = true;
                break;
            }
        }

        // ── 5. Decaimento dos traços STDP ─────────────────────────────────
        let decay = (-dt_ms / TAU_STDP_MS).exp();
        self.trace_pre *= decay;
        self.trace_pos *= decay;

        // ── 6. STDP no spike ──────────────────────────────────────────────
        if spiked {
            // Limiar LTD escalonado pelo Hz atual (docx v2.3 §05):
            // A ticks lentos, reduz o limiar para evitar anti-depressão
            // excessiva causada pelo decay completo do trace_pre entre ticks longos.
            let hz_atual = 1000.0 / dt_ms;
            let ltd_threshold = crate::config::janela_stdp_atual(hz_atual);

            // Diagnóstico: log quando Hz < 200 (confirmar hipótese de janela distorcida)
            #[cfg(debug_assertions)]
            if hz_atual < crate::config::HZ_REFERENCIA && self.trace_pre < ltd_threshold {
                log::debug!(
                    "[STDP] Hz={:.1} trace_pre={:.4} < limiar={:.4} → LTD disparado (janela baixo-Hz)",
                    hz_atual, self.trace_pre, ltd_threshold
                );
            }

            // LTP: spike quando trace_pre alto → correlação causal → potencia
            let delta_ltp = LTP_RATE * self.trace_pre;
            // LTD anti-Hebbiano: spike sem pré correlacionado → anticausal → deprime
            let delta_ltd = if self.trace_pre < ltd_threshold {
                -LTD_RATE * (1.0 - self.trace_pre)
            } else {
                0.0
            };
            self.atualizar_peso(delta_ltp + delta_ltd);
            self.trace_pos = 1.0;
            self.trace_pre = (self.trace_pre + 0.5).min(1.0);
            self.last_spike_ms = current_time_ms;
        }

        // ── 7. Threshold adaptivo retorna ao padrão ───────────────────────
        let tb = self.tipo.threshold_padrao();
        self.threshold = tb + (self.threshold - tb) * THRESHOLD_DECAY;

        spiked
    }

    /// Aplica neuromodulação ao estado HH (se disponível).
    /// Sem custo para neurônios Izhikevich puro (branch previsto corretamente).
    /// Chamado pelo módulo `NeuroChem` a cada ciclo de atualização química.
    pub fn modular_neuro(&mut self, dopamina: f32, serotonina: f32, cortisol: f32) {
        if let Some(estado) = self.modelo.estado_hh_mut() {
            estado.modular(dopamina, serotonina, cortisol);
        }
    }

    /// Verdadeiro se este neurônio usa o modelo HH.
    #[inline]
    pub fn usa_hh(&self) -> bool {
        matches!(self.modelo, ModeloDinamico::IzhikevichHH(_))
    }

    fn atualizar_peso(&mut self, delta: f32) {
        match &mut self.peso {
            PesoNeuronio::FP32(v) => *v = (*v + delta).clamp(PESO_MIN, PESO_MAX),
            PesoNeuronio::FP16(v) => {
                *v = f16::from_f32((v.to_f32() + delta).clamp(PESO_MIN, PESO_MAX));
            }
            PesoNeuronio::INT8(v) => {
                *v = (*v as f32 + delta * 10.0).clamp(-127.0, 127.0) as i8;
            }
            PesoNeuronio::INT4(raw) => {
                let par = Int4Par(*raw);
                let novo = (par.alto() as f32 + delta * 4.0).clamp(-8.0, 7.0) as i8;
                *raw = Int4Par::novo(novo, par.baixo()).0;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 6 — CAMADA HÍBRIDA (API idêntica à V2.1, + modular_neuro)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct CamadaHibrida {
    pub neuronios:     Vec<NeuronioHibrido>,
    /// Escala compartilhada para quantização INT8/INT4.
    /// Calcular como: I_max_esperado / 127.0 para INT8, I_max / 7.0 para INT4.
    pub escala_camada: f32,
    pub nome:          String,
}

impl CamadaHibrida {
    /// Cria camada. Assinatura idêntica à V2.1 — sem breaking changes.
    pub fn new(
        n_neurons:             usize,
        nome:                  &str,
        tipo_principal:        TipoNeuronal,
        tipo_secundario:       Option<(TipoNeuronal, f32)>,
        distribuicao_precisao: Option<Vec<(PrecisionType, f32)>>,
        escala_camada:         f32,
    ) -> Self {
        let dist = distribuicao_precisao.unwrap_or_else(|| vec![
            (PrecisionType::FP32, 0.05),
            (PrecisionType::FP16, 0.35),
            (PrecisionType::INT8, 0.50),
            (PrecisionType::INT4, 0.10),
        ]);
        // Normaliza proporções
        let total: f32 = dist.iter().map(|(_, p)| p).sum();
        let dist: Vec<(PrecisionType, f32)> = dist.into_iter()
            .map(|(t, p)| (t, p / total)).collect();

        let prop_sec  = tipo_secundario.map(|(_, p)| p).unwrap_or(0.0);
        let tipo_sec  = tipo_secundario.map(|(t, _)| t).unwrap_or(tipo_principal);

        // Guarda estado do iterador de precisão para distribuição contígua
        let mut acc_prec = 0.0f32;
        let mut pit = dist.iter().peekable();
        // Seguro: dist sempre tem pelo menos um elemento (unwrap_or_else acima)
        let (mut prec_cur, mut prob_cur) = *pit.next()
            .expect("distribuição de precisão não pode ser vazia");

        let mut neuronios = Vec::with_capacity(n_neurons);
        for i in 0..n_neurons {
            let prog = i as f32 / n_neurons as f32;
            // Avança distribuição de precisão
            while prog > acc_prec + prob_cur {
                acc_prec += prob_cur;
                if let Some((t, p)) = pit.next() {
                    prec_cur = *t;
                    prob_cur = *p;
                } else { break; }
            }
            // Últimos N% são tipo secundário
            let tipo = if prog >= 1.0 - prop_sec { tipo_sec } else { tipo_principal };
            neuronios.push(NeuronioHibrido::new(i as u32, tipo, prec_cur));
        }

        Self { neuronios, escala_camada, nome: nome.to_string() }
    }

    /// Processa um tick. Assinatura idêntica à V2.1.
    pub fn update(&mut self, inputs: &[f32], dt: f32, t_ms: f32) -> Vec<bool> {
        let esc = self.escala_camada;
        self.neuronios.iter_mut().enumerate().map(|(i, n)| {
            n.update(inputs.get(i).copied().unwrap_or(0.0), dt, t_ms, esc)
        }).collect()
    }

    /// Versão com compressão SalientPoint. Assinatura idêntica à V2.1.
    pub fn update_compact(
        &mut self,
        points: &[Vec<SalientPoint>],
        dt:     f32,
        t_ms:   f32,
        comp:   &SalientCompressor,
    ) -> Vec<bool> {
        let esc = self.escala_camada;
        self.neuronios.iter_mut().enumerate().map(|(i, n)| {
            let pts = points.get(i).cloned().unwrap_or_default();
            let rec = comp.decompress(&pts);
            let inp = rec.iter().sum::<f32>() / rec.len().max(1) as f32;
            n.update(inp, dt, t_ms, esc)
        }).collect()
    }

    /// NOVO V2.2: aplica neuromodulação a neurônios HH da camada.
    ///
    /// Para neurônios Izhikevich puro: custo zero (branch always taken).
    /// Para neurônios TC/RZ: atualiza g_na_mod, g_k_mod, g_l_mod no EstadoHH.
    ///
    /// Integração com NeuroChem — adicionar no loop principal do main.rs:
    /// ```rust
    /// cortex.modular_neuro(neuro.dopamine, neuro.serotonin, neuro.cortisol);
    /// cerebellum.purkinje_layer.modular_neuro(neuro.dopamine, neuro.serotonin, neuro.cortisol);
    /// ```
    pub fn modular_neuro(&mut self, dopamina: f32, serotonina: f32, cortisol: f32) {
        for n in self.neuronios.iter_mut() {
            n.modular_neuro(dopamina, serotonina, cortisol);
        }
    }

    pub fn adicionar_neuronio(&mut self, tipo: TipoNeuronal, precisao: PrecisionType) -> u32 {
        let id = self.neuronios.len() as u32;
        self.neuronios.push(NeuronioHibrido::new(id, tipo, precisao));
        id
    }

    pub fn estatisticas(&self) -> CamadaStats {
        let mut s = CamadaStats::default();
        s.total = self.neuronios.len();
        for n in &self.neuronios {
            match n.precisao {
                PrecisionType::FP32 => s.fp32 += 1,
                PrecisionType::FP16 => s.fp16 += 1,
                PrecisionType::INT8 => s.int8 += 1,
                PrecisionType::INT4 => s.int4 += 1,
            }
            match n.tipo {
                TipoNeuronal::RS => s.tipo_rs += 1,
                TipoNeuronal::IB => s.tipo_ib += 1,
                TipoNeuronal::CH => s.tipo_ch += 1,
                TipoNeuronal::FS => s.tipo_fs += 1,
                TipoNeuronal::LT => s.tipo_lt += 1,
                TipoNeuronal::TC => s.tipo_tc += 1,
                TipoNeuronal::RZ => s.tipo_rz += 1,
            }
            if n.usa_hh() { s.hh += 1; }
            s.bytes_total += n.peso.bytes_reais();
        }
        s
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 7 — ESTATÍSTICAS
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct CamadaStats {
    pub total:       usize,
    pub fp32:        usize,
    pub fp16:        usize,
    pub int8:        usize,
    pub int4:        usize,
    pub bytes_total: usize,
    pub tipo_rs:     usize,
    pub tipo_ib:     usize,
    pub tipo_ch:     usize,
    pub tipo_fs:     usize,
    pub tipo_lt:     usize,
    pub tipo_tc:     usize,
    pub tipo_rz:     usize,
    /// Neurônios usando o modelo Hodgkin-Huxley (TC + RZ).
    pub hh:          usize,
}

impl CamadaStats {
    pub fn bytes_por_neuronio(&self) -> f32 {
        if self.total == 0 { 0.0 }
        else { self.bytes_total as f32 / self.total as f32 }
    }
    pub fn prop_inibitorios(&self) -> f32 {
        if self.total == 0 { 0.0 }
        else { (self.tipo_fs + self.tipo_lt) as f32 / self.total as f32 }
    }
    /// Proporção de neurônios usando HH (esperado: ~5–10% do total).
    pub fn prop_hh(&self) -> f32 {
        if self.total == 0 { 0.0 }
        else { self.hh as f32 / self.total as f32 }
    }
}