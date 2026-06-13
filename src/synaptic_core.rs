// =============================================================================
// src/synaptic_core.rs — Selene V3.1 — Modelo Unificado de Neurônio Biológico
// =============================================================================
//
// MODELO EVOLUTIVO DE 7 CAMADAS + EXTENSÕES V3.1:
//
//  ┌──────────────────────────────────────────────────────────────────────────┐
//  │ Camada 1 — IZHIKEVICH (todos os tipos)                                  │
//  │   dv/dt = 0.04v² + 5v + 140 − u + I_eff                                │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Camada 2 — HODGKIN-HUXLEY (TC e RZ) + I_T Ca²⁺ (TC e LT)              │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Camada 3 — NOVOS CANAIS IÔNICOS                                          │
//  │   I_NaP | I_M (KCNQ) | I_A (Kv4) | I_BK | I_T                         │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Camada 4 — SHORT-TERM PLASTICITY (Tsodyks-Markram 1997)                │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Camada 5 — Ca²⁺ DUAL: AHP (SK) + NMDA (LTP trigger)                   │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Camada 6 — STDP 3 FATORES bidirecional (RPE+ → LTP | RPE− → LTD)      │
//  │   Gate ChIN: dopamina só efetiva quando ChIN está em pausa              │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Camada 7 — ACh COMO 4º NEUROMODULADOR                                   │
//  └──────────────────────────────────────────────────────────────────────────┘
//
// EXTENSÕES V3.1:
//   RS  BAC Firing:         input_apical + spike → burst imediato + ca_nmda×2
//   SST → RS:               gating de plasticidade (reduz ca_nmda e elig_trace)
//   DA_N RPE−:              hiperpolarização → mod_dopa < 1.0 → LTD invertido
//   TC + ACh:               ACh > 1.2 → +5mV resting → inativa I_T (vigília)
//   NGF Neurogliaform:      Late-Spiking, normalização divisiva broadcast
//   LC_N Locus Coeruleus:   burst → zera I_M/AHP de RS → atenção hiperfocada
//   ChIN Cholinergic:       tônico → pausa libera gate dopaminérgico STDP
//   Astrócito:              glia — atividade alta >1000ms → ca_nmda_max × 2
//
// REFERÊNCIAS:
//   BAC Firing:  Larkum et al. (1999) — Ca²⁺ spike apical coincidence
//   SST gating:  Silberberg & Markram (2007) — Martinotti → apical dendrite
//   DA_N RPE−:   Schultz et al. (1997) — dopamine dip hypothesis
//   TC ACh:      McCormick & Prince (1987) — muscarinic → tonic mode
//   NGF:         Olah et al. (2009) — volume GABA, Late-Spiking
//   LC_N:        Sara (2009) — LC-NE arousal, attention reset
//   ChIN:        Goldberg et al. (2012) — pause → DA-STDP gate
//   Astrócito:   Henneberger et al. (2010) — D-serine co-agonist NMDA
// =============================================================================

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(clippy::excessive_precision)]

use serde::{Deserialize, Serialize};
use half::f16;
use rayon::prelude::*;
use std::sync::OnceLock;
use crate::config::Config;
use crate::compressor::salient::{SalientPoint, SalientCompressor};

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 1 — TIPO NEURONAL
// ─────────────────────────────────────────────────────────────────────────────

/// Tipos funcionais de neurônios baseados em Izhikevich (2003) + subtipos biológicos.
///
/// TC e RZ usam `ModeloDinamico::IzhikevichHH` (Hodgkin-Huxley completo).
/// Os demais usam `ModeloDinamico::Izhikevich` puro.
///
/// Tipos originais (7): RS, IB, CH, FS, LT, TC, RZ
/// Izhikevich adicionais (6): PS, PB, AC, BI, DAP, IIS
/// Subtipos biológicos (4): PV, SST, VIP, DA_N
/// V3.1 — novos tipos (3): NGF, LC_N, ChIN
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TipoNeuronal {
    // ── Tipos originais ────────────────────────────────────────────────────────
    /// Regular Spiking — neurônio piramidal excitatório padrão do córtex.
    RS,
    /// Intrinsic Bursting — burst inicial seguido de disparo regular.
    IB,
    /// Chattering — bursts rápidos repetitivos. Córtex visual V2/V3.
    CH,
    /// Fast Spiking — interneurônio GABAérgico inibitório sem adaptação.
    FS,
    /// Low-Threshold Spiking — interneurônio de limiar baixo.
    LT,
    /// Thalamo-Cortical — dois modos: burst (sono/desatenção) e tônico (vigília).
    TC,
    /// Resonator / Purkinje — células do cerebelo e giro dentado.
    RZ,

    // ── Tipos Izhikevich adicionais ────────────────────────────────────────────
    /// Phasic Spiking — dispara APENAS no onset do estímulo; silencia depois.
    PS,
    /// Phasic Bursting — burst único na borda de subida do estímulo.
    PB,
    /// Accommodating — adapta progressivamente até silêncio total.
    AC,
    /// Bistable — dois estados estáveis (ON/OFF) com histerese.
    BI,
    /// Depolarizing Afterpotential — rebound despolarizante após spike.
    DAP,
    /// Inhibition-Induced Spiking — dispara quando a inibição é REMOVIDA.
    IIS,

    // ── Subtipos biológicos ────────────────────────────────────────────────────
    /// Parvalbumin interneuron — FS de alta precisão, Ca²⁺-buffered.
    PV,
    /// Somatostatin interneuron (Martinotti) — adapting, inibição dendrítica.
    /// Inibe compartimento apical de pirâmides; controla janela de plasticidade.
    SST,
    /// VIP interneuron — disinhibitory; inibe SST e PV, desinibe pirâmides.
    VIP,
    /// Dopaminergic neuron — pacemaker lento ~4 Hz, AHP prolongado.
    /// VTA e SNc; fonte real do sinal dopaminérgico de recompensa.
    DA_N,

    // ── V3.1: novos tipos ────────────────────────────────────────────────────
    /// Neurogliaform — interneurônio Late-Spiking de inibição volumétrica (GABA-B).
    /// Sem pesos diretos `lateral_w`; ao disparar subtrai de TODOS os inputs
    /// da camada via normalização divisiva. Alta g_a (Kv1.x / D-current) → delay 1º spike.
    NGF,
    /// Locus Coeruleus — controlador global de noradrenalina.
    /// Burst: zera I_M (w_m) e AHP (ca_intra) de todos os RS → atenção hiperfocada.
    /// Reduz eficácia de sinapses fracas (melhora SNR).
    LC_N,
    /// Cholinergic Interneuron — pacemaker tônico ~5 Hz.
    /// Quando inibido (pausa), libera o gate dopaminérgico do STDP 3-fatores
    /// → consolidação sináptica por coincidência DA + elig_trace.
    ChIN,

    // ── V4.6: novos tipos biofísicos ──────────────────────────────────────────
    /// Grid Cell — neurônio de grade do córtex entorrinal (Hafting et al. 2005).
    /// Estabilidade de frequência → cria mapas hexagonais por interferência espacial.
    /// Alta g_nap (oscilação subliminar de membrana persistente) + g_m moderado.
    GridCell,
    /// Mirror Cell — subtipo piramidal pré-motor (Rizzolatti & Craighero 2004).
    /// Aprendizado vicariante: liga caminhos sensoriais a comandos motores via
    /// coincidência STDP de 3 fatores. Excitatório, similar a RS com STDP forte.
    MirrorCell,
    /// Medium Spiny Neuron — célula inibidora GABAérgica do corpo estriado.
    /// Forte adaptação pelo canal M (g_m alto) evita loops infinitos de feedback.
    /// Receptores D1/D2 sensíveis à dopamina (modulação via mod_dopa de DA_N).
    MSN,

    // ── V4.6: célula-tronco / hibridização autônoma ───────────────────────────
    /// Neurônio híbrido — fenótipo definido em runtime por um `DnaNeuronal`.
    /// Variante UNIT (preserva `Copy`); o genoma viaja no campo
    /// `NeuronioHibrido::dna`. Os parâmetros biofísicos efetivos são lidos do DNA
    /// pelos getters `*_efetivo()` do neurônio, não deste enum.
    Hybrid,
}

impl TipoNeuronal {
    /// Parâmetros Izhikevich (a, b, c, d).
    #[inline]
    pub fn parametros(&self) -> (f32, f32, f32, f32) {
        match self {
            // Originais
            TipoNeuronal::RS  => (0.02,  0.20, -65.0,  8.0),
            TipoNeuronal::IB  => (0.02,  0.20, -55.0,  4.0),
            TipoNeuronal::CH  => (0.02,  0.20, -50.0,  2.0),
            TipoNeuronal::FS  => (0.10,  0.20, -65.0,  2.0),
            TipoNeuronal::LT  => (0.02,  0.25, -65.0,  2.0),
            TipoNeuronal::TC  => (0.02,  0.25, -65.0,  0.05),
            TipoNeuronal::RZ  => (0.10,  0.26, -65.0,  2.0),
            // Izhikevich adicionais
            TipoNeuronal::PS  => (0.02,  0.25, -65.0,  6.0),
            TipoNeuronal::PB  => (0.02,  0.25, -55.0,  0.05),
            TipoNeuronal::AC  => (0.02,  1.00, -55.0,  4.0),
            TipoNeuronal::BI  => (0.10,  0.26, -60.0,  0.0),
            TipoNeuronal::DAP => (1.00,  0.20, -60.0, -21.0),
            TipoNeuronal::IIS => (0.02, -1.00, -60.0,  8.0),
            // Subtipos biológicos
            TipoNeuronal::PV  => (0.10,  0.20, -67.0,  2.0),
            TipoNeuronal::SST => (0.02,  0.27, -65.0,  2.0),
            TipoNeuronal::VIP => (0.05,  0.20, -65.0,  8.0),
            TipoNeuronal::DA_N=> (0.006, 0.14, -65.0,  8.0),
            // V3.1
            // NGF: Late-Spiking; alta I_A cria atraso no 1º spike (D-current)
            TipoNeuronal::NGF => (0.02,  0.25, -65.0,  2.0),
            // LC_N: burst-pacemaker similar a IB, limiar levemente baixo
            TipoNeuronal::LC_N=> (0.02,  0.20, -55.0,  6.0),
            // ChIN: tônico com adaptação suave — a baixo, c intermediário
            TipoNeuronal::ChIN=> (0.05,  0.20, -60.0,  5.0),
            // V4.6
            // GridCell: ressonador estável (a maior → oscilação rítmica robusta)
            TipoNeuronal::GridCell   => (0.05,  0.26, -60.0,  4.0),
            // MirrorCell: piramidal pré-motor — idêntico a RS
            TipoNeuronal::MirrorCell => (0.02,  0.20, -65.0,  8.0),
            // MSN: down-state profundo, dispara difícil, forte after-spike reset
            TipoNeuronal::MSN        => (0.01,  0.20, -55.0,  8.0),
            // Hybrid: default tipo-RS; valores reais vêm do DNA via getters do neurônio
            TipoNeuronal::Hybrid     => (0.02,  0.20, -65.0,  8.0),
        }
    }

    /// Threshold de disparo padrão (mV).
    #[inline]
    pub fn threshold_padrao(&self) -> f32 {
        match self {
            TipoNeuronal::TC  => 25.0,
            TipoNeuronal::FS  => 25.0,
            TipoNeuronal::LC_N=> 25.0,  // burst fácil
            TipoNeuronal::MSN => 35.0,  // limiar alto: down-state, difícil disparar
            _                 => 30.0,
        }
    }

    /// Verdadeiro para tipos GABAérgicos (usados na inibição lateral).
    #[inline]
    pub fn e_inibitorico(&self) -> bool {
        matches!(self,
            TipoNeuronal::FS  | TipoNeuronal::LT |
            TipoNeuronal::PV  | TipoNeuronal::SST | TipoNeuronal::VIP |
            TipoNeuronal::NGF |  // NGF: GABA volumétrico (inibição divisiva)
            TipoNeuronal::MSN)  // MSN: GABAérgico do corpo estriado
    }

    /// Verdadeiro para tipos que usam o modelo Hodgkin-Huxley.
    #[inline]
    pub fn usa_hh(&self) -> bool {
        matches!(self, TipoNeuronal::TC | TipoNeuronal::RZ)
    }

    /// Constante de tempo de remoção de Ca²⁺ intracelular (ms) — específica por tipo.
    #[inline]
    pub fn tau_ca_ms(&self) -> f32 {
        match self {
            TipoNeuronal::RS  => 80.0,
            TipoNeuronal::IB  => 90.0,
            TipoNeuronal::CH  => 120.0,
            TipoNeuronal::FS  => 20.0,
            TipoNeuronal::LT  => 50.0,
            TipoNeuronal::TC  => 60.0,
            TipoNeuronal::RZ  => 60.0,
            TipoNeuronal::PS  => 70.0,
            TipoNeuronal::PB  => 85.0,
            TipoNeuronal::AC  => 100.0,
            TipoNeuronal::BI  => 75.0,
            TipoNeuronal::DAP => 40.0,
            TipoNeuronal::IIS => 55.0,
            TipoNeuronal::PV  => 15.0,
            TipoNeuronal::SST => 60.0,
            TipoNeuronal::VIP => 70.0,
            TipoNeuronal::DA_N=> 150.0,
            // V3.1
            TipoNeuronal::NGF => 40.0,   // GABA rápido mas Ca²⁺ moderado
            TipoNeuronal::LC_N=> 80.0,   // NA burst — Ca²⁺ médio
            TipoNeuronal::ChIN=> 60.0,   // tônico — Ca²⁺ basal
            // V4.6
            TipoNeuronal::GridCell   => 70.0,  // ressonador — Ca²⁺ rítmico
            TipoNeuronal::MirrorCell => 80.0,  // igual a RS
            TipoNeuronal::MSN        => 100.0, // adaptação lenta forte
            TipoNeuronal::Hybrid     => 80.0,  // default; sobreposto pelo DNA
        }
    }

    /// Atividade média alvo para plasticidade homeostática BCM.
    #[inline]
    pub fn bcm_theta(&self) -> f32 {
        match self {
            TipoNeuronal::RS  => 0.10,
            TipoNeuronal::IB  => 0.08,
            TipoNeuronal::CH  => 0.15,
            TipoNeuronal::FS  => 0.25,
            TipoNeuronal::LT  => 0.07,
            TipoNeuronal::TC  => 0.05,
            TipoNeuronal::RZ  => 0.12,
            TipoNeuronal::PS  => 0.03,
            TipoNeuronal::PB  => 0.04,
            TipoNeuronal::AC  => 0.06,
            TipoNeuronal::BI  => 0.10,
            TipoNeuronal::DAP => 0.08,
            TipoNeuronal::IIS => 0.06,
            TipoNeuronal::PV  => 0.30,
            TipoNeuronal::SST => 0.08,
            TipoNeuronal::VIP => 0.12,
            TipoNeuronal::DA_N=> 0.04,
            // V3.1
            TipoNeuronal::NGF => 0.05,   // baixa taxa — inibe em rajadas
            TipoNeuronal::LC_N=> 0.04,   // burst esporádico
            TipoNeuronal::ChIN=> 0.10,   // tônico ~5 Hz
            // V4.6
            TipoNeuronal::GridCell   => 0.12,  // disparo rítmico moderado
            TipoNeuronal::MirrorCell => 0.10,  // igual a RS
            TipoNeuronal::MSN        => 0.06,  // esparso — gate de ação
            TipoNeuronal::Hybrid     => 0.10,  // default; sobreposto pelo DNA
        }
    }

    /// Parâmetros de condutância HH para este tipo.
    pub fn parametros_hh(&self) -> Option<ParametrosHH> {
        match self {
            TipoNeuronal::TC => Some(ParametrosHH {
                g_na: 120.0, g_k: 36.0, g_l: 0.3,
                e_na: 50.0, e_k: -77.0, e_l: -54.4, c_m: 1.0,
                g_h: 1.5,
            }),
            TipoNeuronal::RZ => Some(ParametrosHH {
                g_na: 150.0, g_k: 38.0, g_l: 0.5,
                e_na: 55.0, e_k: -80.0, e_l: -65.0, c_m: 1.0,
                g_h: 0.0,
            }),
            _ => None,
        }
    }

    /// Item 2 — Faixa de frequência de disparo plausível in vivo (min, max em Hz).
    ///
    /// O cérebro NÃO opera todo na frequência máxima: a maioria dos piramidais
    /// dispara 1–20 Hz, interneurônios FS/PV chegam a 200 Hz, pacemakers ~1–8 Hz.
    /// O clock de SIMULAÇÃO permanece 200 Hz (5 ms) por estabilidade numérica da
    /// integração; estas faixas são a banda biológica esperada que o teste A/B
    /// usa para validar que as taxas EMERGENTES são realistas.
    /// Hz de disparo ≠ Hz de integração.
    pub fn faixa_hz(&self) -> (f32, f32) {
        match self {
            TipoNeuronal::RS  => (1.0, 20.0),
            TipoNeuronal::IB  => (5.0, 40.0),
            TipoNeuronal::CH  => (20.0, 80.0),   // chattering — bursts rápidos
            TipoNeuronal::FS  => (20.0, 200.0),  // fast-spiking — sem adaptação
            TipoNeuronal::LT  => (5.0, 40.0),
            TipoNeuronal::TC  => (4.0, 100.0),   // burst (sono) ↔ tônico (vigília)
            TipoNeuronal::RZ  => (10.0, 100.0),
            TipoNeuronal::PS  => (1.0, 15.0),
            TipoNeuronal::PB  => (1.0, 15.0),
            TipoNeuronal::AC  => (1.0, 10.0),
            TipoNeuronal::BI  => (1.0, 30.0),
            TipoNeuronal::DAP => (5.0, 40.0),
            TipoNeuronal::IIS => (1.0, 20.0),
            TipoNeuronal::PV  => (30.0, 200.0),  // parvalbumin — alta precisão
            TipoNeuronal::SST => (5.0, 40.0),
            TipoNeuronal::VIP => (5.0, 50.0),
            TipoNeuronal::DA_N=> (1.0, 8.0),     // pacemaker dopaminérgico lento
            TipoNeuronal::NGF => (1.0, 30.0),
            TipoNeuronal::LC_N=> (1.0, 20.0),
            TipoNeuronal::ChIN=> (2.0, 10.0),    // tônico ~5 Hz
            TipoNeuronal::GridCell   => (5.0, 40.0),
            TipoNeuronal::MirrorCell => (1.0, 25.0),
            TipoNeuronal::MSN        => (0.1, 20.0), // down-state → disparo esparso
            TipoNeuronal::Hybrid     => (1.0, 200.0),// fenótipo livre
        }
    }

    /// V4.6.1 — Fator de adaptação por spike (multiplica `THRESHOLD_DELTA`).
    /// CORE-TUNING de fidelidade: a curva F-I mostrava piramidais a disparar
    /// muito acima do biológico sob corrente forte. Piramidais reais têm forte
    /// adaptação de frequência (I_M/AHP/acomodação de limiar) que limita a taxa
    /// sustentada a ~20–40 Hz; FS/PV quase não adaptam (mantêm-se rápidos).
    /// Este fator concentra a correção SEM mexer em threshold/responsividade
    /// (o neurônio continua a responder ao input — só satura mais cedo).
    #[inline]
    pub fn fator_adaptacao(&self) -> f32 {
        match self {
            // Piramidais excitatórios — adaptação forte (puxa a taxa para a faixa)
            TipoNeuronal::RS | TipoNeuronal::MirrorCell => 6.0,
            TipoNeuronal::IB  => 4.5,
            TipoNeuronal::CH  => 3.0,  // chattering: adapta menos que RS
            TipoNeuronal::AC  => 7.0,  // accommodating: adaptação máxima por definição
            TipoNeuronal::MSN => 4.0,  // forte canal-M
            // Fast-spiking / interneurônios rápidos — quase sem adaptação
            TipoNeuronal::FS | TipoNeuronal::PV => 0.25,
            // Intermediários
            TipoNeuronal::LT | TipoNeuronal::TC | TipoNeuronal::GridCell => 1.5,
            TipoNeuronal::RZ => 1.2,
            // Demais (moduladores, variantes, Hybrid) — padrão neutro
            _ => 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 2 — HODGKIN-HUXLEY: PARÂMETROS, ESTADO E MOTOR DE CÁLCULO
// ─────────────────────────────────────────────────────────────────────────────

/// Condutâncias e potenciais de reversão — constantes por tipo celular.
#[derive(Debug, Clone, Copy)]
pub struct ParametrosHH {
    pub g_na: f32,
    pub g_k:  f32,
    pub g_l:  f32,
    pub e_na: f32,
    pub e_k:  f32,
    pub e_l:  f32,
    pub c_m:  f32,
    pub g_h:  f32,
}

/// Estado das variáveis de portão — muda a cada tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstadoHH {
    pub m: f32,
    pub h: f32,
    pub n: f32,
    pub q_ih: f32,
    pub g_na_mod: f32,
    pub g_k_mod:  f32,
    pub g_l_mod:  f32,
}

impl EstadoHH {
    pub fn repouso() -> Self {
        Self { m: 0.053, h: 0.596, n: 0.318,
               q_ih: 0.01,
               g_na_mod: 1.0, g_k_mod: 1.0, g_l_mod: 1.0 }
    }

    pub fn modular(&mut self, dopamina: f32, serotonina: f32, cortisol: f32) {
        self.g_k_mod  = (1.2 - dopamina   * 0.35).clamp(0.5, 1.2);
        self.g_l_mod  = (1.1 - serotonina * 0.25).clamp(0.5, 1.1);
        self.g_na_mod = (1.0 - cortisol   * 0.40).clamp(0.4, 1.0);
    }
}

const HH_SCALE: f32 = 0.008;

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 3 — MODELO DINÂMICO
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModeloDinamico {
    Izhikevich,
    IzhikevichHH(Box<EstadoHH>),
}

impl ModeloDinamico {
    pub fn para_tipo(tipo: TipoNeuronal) -> Self {
        if tipo.usa_hh() {
            ModeloDinamico::IzhikevichHH(Box::new(EstadoHH::repouso()))
        } else {
            ModeloDinamico::Izhikevich
        }
    }

    pub fn estado_hh_mut(&mut self) -> Option<&mut EstadoHH> {
        match self {
            ModeloDinamico::IzhikevichHH(e) => Some(e.as_mut()),
            ModeloDinamico::Izhikevich      => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 4 — PRECISÃO MISTA
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PrecisionType { FP32, FP16, INT8, INT4 }

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 4b — CICLO DE VIDA NEURAL (V3.6)
// ─────────────────────────────────────────────────────────────────────────────

/// Número de ticks sem corrente/spike para um neurônio tornar-se Dormant.
/// 20.000 ticks @ 200Hz ≈ 100s de inatividade absoluta.
/// Reduzido de 100.000 (V3.6) para conter OOM de runtime: neurônios ociosos
/// paginam para NVMe muito mais cedo (restauráveis via restaurar_do_nvme).
pub const INACTIVITY_THRESHOLD: u64 = 20_000;

/// Estado do ciclo de vida de um NeuronioHibrido.
/// Active  → processando correntes ou disparando recentemente.
/// Dormant → inativo por INACTIVITY_THRESHOLD ticks; elegível para evicção SSD.
/// Swapped → serializado no NVMe (swap_manager.nvme_index); não está em RAM nem SSD.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum NeuronalStatus {
    #[default]
    Active,
    Dormant,
    Swapped,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PesoNeuronio {
    FP32(f32),
    FP16(f16),
    INT8(i8),
    INT4(u8),
}

impl PesoNeuronio {
    #[inline]
    pub fn valor_f32(&self, escala: f32) -> f32 {
        match self {
            PesoNeuronio::FP32(v)   => *v,
            PesoNeuronio::FP16(v)   => v.to_f32(),
            PesoNeuronio::INT8(v)   => (*v as f32) * escala,
            PesoNeuronio::INT4(raw) => Int4Par(*raw).alto() as f32 * escala,
        }
    }

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
// SEÇÃO 5 — CONSTANTES V3
// ─────────────────────────────────────────────────────────────────────────────

/// Potenciais de reversão (mV)
const E_NA: f32 = 50.0;
const E_K:  f32 = -77.0;
const E_CA: f32 = 120.0;
const E_H:  f32 = -30.0;

// STDP
const TAU_STDP_MS:    f32 = 20.0;
const LTP_RATE:       f32 = 0.012;
const LTD_RATE:       f32 = 0.006;
const PESO_MAX:       f32 = 2.5;
const PESO_MIN:       f32 = 0.0;
const THRESHOLD_DELTA: f32 = 0.5;
const THRESHOLD_DECAY: f32 = 0.985;

// Ca²⁺ AHP (SK channels)
const CA_POR_SPIKE: f32 = 2.0;
const CA_MAX:       f32 = 12.0;
const G_AHP:        f32 = 1.8;

// BCM homeostático
const TAU_BCM_MS:       f32 = 5000.0;
const BCM_RATE:         f32 = 0.002;
/// θ_M slide tau: 6× mais lento que activity_avg (BCM 1982, θ_M tracking)
const TAU_BCM_THETA_MS: f32 = 30_000.0;

// Plasticidade 3 fatores
const TAU_ELIG_MS:    f32 = 500.0;
const ELIG_RATE:      f32 = 0.02;
const DOPA_GATE:      f32 = 0.008;

// NMDA Ca²⁺
const NMDA_CA_RATE:    f32 = 0.4;
const TAU_NMDA_CA_MS:  f32 = 50.0;

// BK channels (fast AHP)
const BK_PER_SPIKE:  f32 = 0.6;
const TAU_BK_MS:     f32 = 5.0;

// ── V3.1: Novos mecanismos ───────────────────────────────────────────────────

// RS BAC Firing (Back-propagating AP → Ca²⁺ spike apical; Larkum et al. 1999)
/// Limiar de input apical para disparar BAC firing no RS (coincidência apical+somática).
const APICAL_THRESHOLD:  f32 = 2.0;
/// Corrente extra injetada durante os 5ms de burst BAC (pA equivalente).
const BURST_CURRENT:     f32 = 15.0;
/// Duração do burst BAC após coincidência apical+somática (ms).
const BURST_DURATION_MS: f32 = 5.0;

// SST Plasticity Gating (Silberberg & Markram 2007 — Martinotti → apical dendrite)
/// Fração de ca_nmda reduzida por ativação SST por conexão ativa.
const SST_CA_GATE:   f32 = 0.30;
/// Fração de elig_trace reduzida por ativação SST (fecha janela de consolidação).
const SST_ELIG_GATE: f32 = 0.50;

// NGF Divisive Normalization (Olah et al. 2009 — GABA volume transmission)
/// Contribuição ao pool divisivo por spike de NGF.
const NGF_DIVISIVE_STEP: f32 = 0.6;
/// Constante de tempo de decaimento do pool divisivo NGF (ms) — GABA-B lento.
const TAU_NGF_MS:        f32 = 20.0;

// LC_N (Locus Coeruleus; Sara 2009 — NA arousal reset)
/// Duração dos efeitos de burst LC_N em RS (ms) — janela de NA elevada.
const TAU_LC_BURST_MS: f32 = 300.0;

// DA_N RPE negativo (Schultz 1997 — dopamine dip hypothesis)
/// Hiperpolarização acumulada para sinalizar RPE negativo (≈ ciclo pacemaker 4Hz = 250ms).
const DAN_HYPERPOL_THRESHOLD_MS: f32 = 200.0;
/// Valor de mod_dopa para RPE negativo (< 1.0 → LTD no STDP).
const DAN_RPE_NEG_MOD_DOPA: f32 = 0.6;

// Astrócito (Henneberger et al. 2010 — D-serine co-agonist NMDA)
/// Limiar de atividade média da camada para trigger astrocítico [0.0–1.0].
const ASTRO_ACTIVITY_THRESHOLD: f32 = 0.4;
/// Duração de atividade alta para trigger de facilitação LTP (ms).
const ASTRO_HIGH_DURATION_MS:   f32 = 1000.0;
/// Multiplicador de ca_nmda_max quando astrócito ativo (D-serina → amplifica NMDA).
const ASTRO_CA_NMDA_SCALE: f32 = 2.0;

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 5a-V4 — NEURÔNIO HÍBRIDO MULTICOMPARTIMENTAL (V4)
//
// Extensão NÃO-QUEBRANTE do V3.1. Apenas RS/IB recebem compartimentos dendríticos;
// todos os outros tipos → `compartimentos: None`. O metabolismo (ATP + bomba
// Na⁺K⁺-ATPase + [K⁺]o dinâmico) é ativo em todos os tipos, mas com efeito
// mínimo em tipos não-piramidais (ATP alto, bomba fraca).
//
// REFERÊNCIAS:
//   Rall (1967)                — cable theory: acoplamento axial entre compartimentos
//   Larkum, Zhu & Sakmann 1999 — BAC firing: coincidência apical+somática
//   Schiller et al. (2000)     — NMDA spike dendrítico no tufo apical
//   Kole & Stuart (2012)       — AIS como ponto de iniciação do spike (g_Na 5×)
//   Anastassiou et al. (2011)  — acoplamento ephaptic bidirecional
//   PLOS CompBiol (2020)       — Na/K ATPase = 75% do gasto energético neuronal
//   Kager, Wadman & Somjen 2000— [K⁺]o dinâmico e kindling
//   Hodgkin & Katz (1949)      — E_K Nernst dinâmico
// ─────────────────────────────────────────────────────────────────────────────

/// Estado dos brain states — modula a condutância de acoplamento apical.
/// Vigília: amplificação apical máxima. NREM: isolamento. REM: drive independente.
/// Frontiers CompNeurosci (2025); arXiv:2311.06074
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EstadoBrainState {
    /// Vigília: amplificação apical máxima — g_c_apical × 1.0
    Vigilia,
    /// NREM profundo: isolamento apical — g_c_apical × 0.3
    NremProfundo,
    /// REM: drive apical independente — g_c_apical × 1.2
    Rem,
}

impl EstadoBrainState {
    /// Multiplicador efetivo de g_c_apical para o estado atual.
    #[inline]
    pub fn fator_apical(&self) -> f32 {
        match self {
            EstadoBrainState::Vigilia      => 1.0,
            EstadoBrainState::NremProfundo => 0.3,
            EstadoBrainState::Rem          => 1.2,
        }
    }
}

/// Estado dos 5 compartimentos acoplados (AIS, Soma/Basal, Tronco, Tufo Apical,
/// Extracelular). Apenas RS/IB instanciam este struct (via `Some(Box<...>)`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstadoCompartimentos {
    // ── AIS — Axon Initial Segment (Kole & Stuart 2012) ──────────────────────
    // Spike SEMPRE inicia aqui (g_Na = 200, 5× maior que HH padrão).
    pub v_ais:         f32,   // init: -65.0 mV
    pub m_ais:         f32,   // portão Na AIS, init: 0.053
    pub h_ais:         f32,   // inativação Na AIS, init: 0.596
    pub n_ais:         f32,   // portão K AIS, init: 0.318
    pub ais_spiked:    bool,  // AIS disparou antes do soma neste tick

    // ── Tronco apical — Ca²⁺ hotzone (Larkum 1999) ──────────────────────────
    pub v_trunk:       f32,   // init: -68.0 mV
    pub ca_trunk:      f32,   // Ca²⁺ no tronco, init: 0.0
    pub g_ca_trunk:    f32,   // condutância Ca²⁺ L-type, default: 2.0

    // ── Tufo apical — NMDA spike dendrítico (Schiller 2000) ─────────────────
    pub v_apical:      f32,   // init: -70.0 mV
    pub ca_apical:     f32,   // Ca²⁺ no tufo, init: 0.0
    pub nmda_gate:     f32,   // portão NMDA spike [0,1], init: 0.0
    pub nmda_spike_ms: f32,   // duração do NMDA spike ativo (ms), init: 0.0

    // ── BAP — Back-Propagating Action Potential ─────────────────────────────
    pub bap_active:    bool,  // AP propagando soma → apical, init: false
    pub bap_timer_ms:  f32,   // ms desde início do BAP, init: 0.0

    // ── Coincidência dendrítica (Larkum et al. 1999) ────────────────────────
    pub coincidencia_ativa: bool, // BAP + ca_trunk > 1.5 + nmda_gate > 0.4
}

impl EstadoCompartimentos {
    pub fn novo() -> Self {
        Self {
            v_ais: -65.0, m_ais: 0.053, h_ais: 0.596, n_ais: 0.318,
            ais_spiked: false,
            v_trunk: -68.0, ca_trunk: 0.0, g_ca_trunk: 2.0,
            v_apical: -70.0, ca_apical: 0.0, nmda_gate: 0.0, nmda_spike_ms: 0.0,
            bap_active: false, bap_timer_ms: 0.0,
            coincidencia_ativa: false,
        }
    }
}

/// Estado metabólico — ATP, bomba Na⁺K⁺-ATPase eletrogenica, [K⁺]o e [Na⁺]i
/// dinâmicos, E_K(t) via Nernst temporal, potencial extracelular ephaptic.
/// Ativo em TODOS os tipos (sem quebra); efeito mínimo em não-piramidais.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstadoMetabolico {
    pub atp:       f32,  // concentração [0.05, 2.5 mM], init: 2.5
    pub na_intra:  f32,  // [Na⁺] intracelular (mM), init: 10.0
    pub k_o:       f32,  // [K⁺] extracelular (mM), init: 3.0
    pub e_k_dyn:   f32,  // E_K dinâmico via Nernst (mV), init: -77.0
    pub i_pump:    f32,  // corrente da bomba Na/K (hiperpolarizante), init: 0.0
    pub v_e:       f32,  // potencial extracelular local (ephaptic), init: 0.0
}

impl Default for EstadoMetabolico {
    fn default() -> Self { Self::novo() }
}

impl Default for EstadoBrainState {
    fn default() -> Self { EstadoBrainState::Vigilia }
}

impl EstadoMetabolico {
    pub fn novo() -> Self {
        Self {
            atp: 2.5, na_intra: 10.0, k_o: 3.0,
            e_k_dyn: -77.0, i_pump: 0.0, v_e: 0.0,
        }
    }

    /// Nernst dinâmico: E_K(t) = RT/F × ln([K⁺]o / [K⁺]i)
    /// RT/F a 37°C ≈ 26.7 mV; [K⁺]i ≈ 140 mM (fixo). Hodgkin & Katz (1949).
    pub fn atualizar_ek(&mut self) {
        const RT_F: f32 = 26.7;
        const KI:   f32 = 140.0;
        self.e_k_dyn = RT_F * (self.k_o.max(0.1) / KI).ln();
    }
}

// ── Constantes V4: compartimentos (Rall 1967; Larkum 1999; Kole & Stuart 2012) ─
const G_C_TRUNK:      f32 = 0.15;  // condutância soma↔tronco (mS/cm²)
const G_C_APICAL:     f32 = 0.08;  // condutância tronco↔tufo apical
const G_C_AIS:        f32 = 0.25;  // condutância AIS↔soma
const G_NA_AIS:       f32 = 200.0; // g_Na no AIS (5× maior — Kole & Stuart 2012)
const G_K_AIS:        f32 = 60.0;
const NMDA_THRESHOLD: f32 = 0.35;  // limiar de i_apical para NMDA spike
const BAP_DECAY_TAU:  f32 = 3.0;   // tau de decaimento do BAP (ms)
const BAP_TOTAL_MS:   f32 = 6.0;   // duração total do BAP (ms)

// ── Constantes V4: metabolismo (PLOS CompBiol 2020; Kager et al. 2000) ────────
const ATP_PROD_RATE:        f32 = 0.8;   // produção mitocondrial (mM/ms, Michaelis)
const ATP_BASAL_COST:       f32 = 0.4;   // custo basal (mM/ms)
const ATP_MAX:              f32 = 2.5;   // concentração máxima (mM)
const K_ATP:                f32 = 0.5;   // constante Michaelis da bomba
const RHO_PUMP:             f32 = 8.0;   // condutância máxima da bomba (pA)
const ATP_COST_PER_SPIKE:   f32 = 0.12;  // custo por spike (mM)
const KO_REST:              f32 = 3.0;   // [K⁺]o em repouso (mM)
const KO_RELEASE_PER_SPIKE: f32 = 0.15;  // K⁺ liberado por spike (mM)
const KO_CLEARANCE:         f32 = 0.02;  // clearance glial (ms⁻¹)
const MET_KI:               f32 = 140.0; // [K⁺]i fixo (mM)
const MET_RT_F:             f32 = 26.7;  // RT/F a 37°C (mV)

// ── V4.6 Item 2: throughput 200 Hz ────────────────────────────────────────────
/// Subsampling metabólico: a física de voltagem corre a cada tick (5ms @ 200Hz),
/// mas ATP / bomba Na/K / [K⁺]o são lentos (τ ~ dezenas–centenas de ms) e só
/// precisam recalcular a cada N ticks, com dt multiplicado por N. Erro desprezável.
const METAB_SUBSAMPLE: u64 = 10;

/// Default serde do campo `otimizar` (true = otimização 200 Hz ligada em produção).
/// Função dedicada porque o default de `bool` no serde é `false` — sem isto, estado
/// V4.5 carregado teria a otimização desligada. Per-neurônio (não global) → o teste
/// A/B liga/desliga por camada sem race condition entre testes paralelos.
fn otimizar_padrao() -> bool { true }

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 5b — NTC: NEURAL TEXTURE COMPRESSION
//
// Analogia com compressão de texturas GPU: neurônios C0/C1 usam LUTs pré-
// computadas (INT4) ou Izhikevich direto sem canais iônicos (INT8).
// Reduz carga do hot path 200Hz; correctness exata é trocada por throughput.
//
// LUT 16×16 para dv_base = 0.04v² + 5v + 140 − u (parte estática Izhikevich)
//   v ∈ [−80, +32.5] mV em 16 passos de 7.5 mV
//   u ∈ [−20, +20]   em 16 passos de 2.667
//   Escala 3.0 → i8 cobre range [−384, +381] mV/ms (erro max ±1.5 mV/ms)
// ─────────────────────────────────────────────────────────────────────────────

const NTC_LUT_SCALE: f32 = 3.0;
const NTC_LUT_V_MIN:  f32 = -80.0;
const NTC_LUT_V_STEP: f32 = 7.5;
const NTC_LUT_U_MIN:  f32 = -20.0;
const NTC_LUT_U_STEP: f32 = 2.6667;

static NTC_LUT_FP4: OnceLock<[[i8; 16]; 16]> = OnceLock::new();

fn ntc_lut_fp4() -> &'static [[i8; 16]; 16] {
    NTC_LUT_FP4.get_or_init(|| {
        let mut lut = [[0i8; 16]; 16];
        for vi in 0..16usize {
            let v = NTC_LUT_V_MIN + vi as f32 * NTC_LUT_V_STEP;
            for ui in 0..16usize {
                let u = NTC_LUT_U_MIN + ui as f32 * NTC_LUT_U_STEP;
                let dv_base = 0.04 * v * v + 5.0 * v + 140.0 - u;
                lut[vi][ui] = (dv_base / NTC_LUT_SCALE)
                    .round()
                    .clamp(-128.0, 127.0) as i8;
            }
        }
        lut
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 6 — EXTENSÃO V3: CONDUTÂNCIAS DOS NOVOS CANAIS
// ─────────────────────────────────────────────────────────────────────────────

pub trait TipoNeuronalV3 {
    fn g_nap(&self) -> f32;
    fn g_m(&self) -> f32;
    fn g_a(&self) -> f32;
    fn g_t(&self) -> f32;
    fn g_bk(&self) -> f32;
}

impl TipoNeuronalV3 for TipoNeuronal {
    fn g_nap(&self) -> f32 {
        match self {
            TipoNeuronal::RS  => 1.5,
            TipoNeuronal::IB  => 2.0,
            TipoNeuronal::CH  => 1.0,
            TipoNeuronal::FS  => 0.3,
            TipoNeuronal::LT  => 1.2,
            TipoNeuronal::TC  => 0.8,
            TipoNeuronal::RZ  => 1.5,
            TipoNeuronal::PS  => 0.8,
            TipoNeuronal::PB  => 1.0,
            TipoNeuronal::AC  => 0.5,
            TipoNeuronal::BI  => 2.5,
            TipoNeuronal::DAP => 3.0,
            TipoNeuronal::IIS => 0.5,
            TipoNeuronal::PV  => 0.2,
            TipoNeuronal::SST => 1.0,
            TipoNeuronal::VIP => 0.8,
            TipoNeuronal::DA_N=> 1.2,
            // V3.1
            TipoNeuronal::NGF => 0.4,   // baixo — inibição é primária
            TipoNeuronal::LC_N=> 1.8,   // burst requer boa excitabilidade
            TipoNeuronal::ChIN=> 1.0,   // tônico moderado
            // V4.6
            TipoNeuronal::GridCell   => 1.2, // I_NaP alto → oscilação de membrana
            TipoNeuronal::MirrorCell => 1.5, // igual a RS
            TipoNeuronal::MSN        => 0.5, // baixo — down-state
            TipoNeuronal::Hybrid     => 1.5, // default; sobreposto pelo DNA
        }
    }

    fn g_m(&self) -> f32 {
        match self {
            TipoNeuronal::RS  => 3.0,
            TipoNeuronal::IB  => 4.0,
            TipoNeuronal::CH  => 2.0,
            TipoNeuronal::FS  => 0.3,
            TipoNeuronal::LT  => 3.0,
            TipoNeuronal::TC  => 1.5,
            TipoNeuronal::RZ  => 1.0,
            TipoNeuronal::PS  => 5.0,
            TipoNeuronal::PB  => 4.0,
            TipoNeuronal::AC  => 8.0,
            TipoNeuronal::BI  => 1.0,
            TipoNeuronal::DAP => 0.5,
            TipoNeuronal::IIS => 0.2,
            TipoNeuronal::PV  => 0.2,
            TipoNeuronal::SST => 4.0,
            TipoNeuronal::VIP => 2.0,
            TipoNeuronal::DA_N=> 1.5,
            // V3.1
            TipoNeuronal::NGF => 3.5,   // adaptação moderada — Late-Spiking
            TipoNeuronal::LC_N=> 1.0,   // LC precisa disparar facilmente
            TipoNeuronal::ChIN=> 2.0,   // adaptação suave para manter tônico
            // V4.6
            TipoNeuronal::GridCell   => 2.5, // g_m moderado (sugerido)
            TipoNeuronal::MirrorCell => 2.5, // adaptação tipo RS-pré-motor
            TipoNeuronal::MSN        => 8.0, // FORTE adaptação M → evita loops
            TipoNeuronal::Hybrid     => 3.0, // default; sobreposto pelo DNA
        }
    }

    fn g_a(&self) -> f32 {
        match self {
            TipoNeuronal::RS  => 8.0,
            TipoNeuronal::IB  => 8.0,
            TipoNeuronal::CH  => 6.0,
            TipoNeuronal::FS  => 0.5,
            TipoNeuronal::LT  => 20.0,
            TipoNeuronal::TC  => 8.0,
            TipoNeuronal::RZ  => 4.0,
            TipoNeuronal::PS  => 6.0,
            TipoNeuronal::PB  => 5.0,
            TipoNeuronal::AC  => 4.0,
            TipoNeuronal::BI  => 3.0,
            TipoNeuronal::DAP => 2.0,
            TipoNeuronal::IIS => 15.0,
            TipoNeuronal::PV  => 0.3,
            TipoNeuronal::SST => 12.0,
            TipoNeuronal::VIP => 8.0,
            TipoNeuronal::DA_N=> 6.0,
            // V3.1
            // NGF: Kv1.x/D-current muito alto → atraso característico Late-Spiking
            TipoNeuronal::NGF => 30.0,
            TipoNeuronal::LC_N=> 5.0,
            TipoNeuronal::ChIN=> 6.0,
            // V4.6
            TipoNeuronal::GridCell   => 6.0, // g_a sugerido
            TipoNeuronal::MirrorCell => 7.0, // similar a RS
            TipoNeuronal::MSN        => 6.0, // I_A relevante no down-state
            TipoNeuronal::Hybrid     => 8.0, // default; sobreposto pelo DNA
        }
    }

    fn g_t(&self) -> f32 {
        match self {
            TipoNeuronal::TC  => 8.0,
            TipoNeuronal::LT  => 10.0,
            TipoNeuronal::IIS => 6.0,
            TipoNeuronal::DA_N=> 3.0,
            _                 => 0.0,
        }
    }

    fn g_bk(&self) -> f32 {
        match self {
            TipoNeuronal::RS  => 2.0,
            TipoNeuronal::IB  => 2.0,
            TipoNeuronal::CH  => 2.0,
            TipoNeuronal::FS  => 1.0,
            TipoNeuronal::LT  => 2.0,
            TipoNeuronal::TC  => 2.0,
            TipoNeuronal::RZ  => 5.0,
            TipoNeuronal::PS  => 1.5,
            TipoNeuronal::PB  => 2.0,
            TipoNeuronal::AC  => 1.0,
            TipoNeuronal::BI  => 1.5,
            TipoNeuronal::DAP => 0.5,
            TipoNeuronal::IIS => 1.0,
            TipoNeuronal::PV  => 3.0,
            TipoNeuronal::SST => 1.5,
            TipoNeuronal::VIP => 1.0,
            TipoNeuronal::DA_N=> 4.0,
            // V3.1
            TipoNeuronal::NGF => 1.5,
            TipoNeuronal::LC_N=> 3.0,  // NA release precisa de AHP rápido pós-burst
            TipoNeuronal::ChIN=> 2.0,
            // V4.6
            TipoNeuronal::GridCell   => 2.0, // g_bk sugerido
            TipoNeuronal::MirrorCell => 2.0, // similar a RS
            TipoNeuronal::MSN        => 1.5, // AHP moderado
            TipoNeuronal::Hybrid     => 2.0, // default; sobreposto pelo DNA
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 6b — DNA NEURONAL (V4.6) — genoma digital para hibridização autônoma
//
// Unifica TODAS as variáveis biofísicas de um neurônio numa assinatura numérica
// serializável. É a unidade de hereditariedade da neuroevolução:
//   • `TipoNeuronal::extrair_dna()` lê qualquer tipo puro → DNA.
//   • `gerar_especie_hibrida()` cruza dois DNAs + mutação → DNA inédito.
//   • Neurônios `Hybrid` lêem os genes do DNA via getters `*_efetivo()`.
// ─────────────────────────────────────────────────────────────────────────────

/// Genoma digital de um neurônio — todos os parâmetros biofísicos num só struct.
/// Serializável → permite persistir, clonar e registrar espécies evoluídas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DnaNeuronal {
    // Izhikevich
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub threshold: f32,
    // Condutâncias dos canais iônicos
    pub g_nap: f32,
    pub g_m:   f32,
    pub g_a:   f32,
    pub g_t:   f32,
    pub g_bk:  f32,
    // Constantes de tempo / homeostase
    pub tau_ca_ms:  f32,
    pub bcm_theta:  f32,
    // Flags fenotípicas
    pub e_inibitorico: bool,
    pub usa_hh:        bool,

    // ── V4.6 (genoma expandido): genes ESTRUTURAIS ────────────────────────────
    // Permitem fenótipos QUALITATIVAMENTE novos, não só pontos novos no espaço
    // paramétrico: dinâmica HH, dendritos e plasticidade de curto prazo evoluíveis.
    /// Dendritos multicompartimentais (AIS+soma+tronco+tufo apical). Antes só RS/IB.
    #[serde(default)]
    pub tem_compartimentos: bool,
    /// Plasticidade de curto prazo (Tsodyks-Markram): depressão/facilitação/mista.
    #[serde(default)]
    pub tipo_stp: TipoSTP,
    /// Condutâncias Hodgkin-Huxley — usadas quando `usa_hh`. Permite spikes com
    /// forma realista, sag I_h, etc. (antes HH era exclusivo de TC/RZ puros).
    #[serde(default = "dna_g_na_padrao")] pub g_na: f32,
    #[serde(default = "dna_g_k_padrao")]  pub g_k:  f32,
    #[serde(default = "dna_g_l_padrao")]  pub g_l:  f32,
    #[serde(default)]                     pub g_h:  f32,
}

// Defaults serde dos genes HH (estado pré-expansão carrega com HH "neutro").
fn dna_g_na_padrao() -> f32 { 120.0 }
fn dna_g_k_padrao()  -> f32 { 36.0 }
fn dna_g_l_padrao()  -> f32 { 0.3 }

impl DnaNeuronal {
    /// Aplica limites biofísicos plausíveis a cada gene (evita espécies impossíveis
    /// que desestabilizariam a rede após mutação). Chamado após crossover/mutação.
    pub fn clampar(&mut self) {
        self.a         = self.a.clamp(0.002, 1.0);
        self.b         = self.b.clamp(-1.0, 1.0);
        self.c         = self.c.clamp(-70.0, -40.0);
        self.d         = self.d.clamp(-25.0, 12.0);
        self.threshold = self.threshold.clamp(20.0, 40.0);
        self.g_nap     = self.g_nap.clamp(0.0, 4.0);
        self.g_m       = self.g_m.clamp(0.0, 10.0);
        self.g_a       = self.g_a.clamp(0.0, 35.0);
        self.g_t       = self.g_t.clamp(0.0, 12.0);
        self.g_bk      = self.g_bk.clamp(0.0, 6.0);
        self.tau_ca_ms = self.tau_ca_ms.clamp(10.0, 200.0);
        self.bcm_theta = self.bcm_theta.clamp(0.001, 0.5);
        // Genes HH (só relevantes se usa_hh, mas mantém limites sãos sempre).
        self.g_na = self.g_na.clamp(50.0, 250.0);
        self.g_k  = self.g_k.clamp(10.0, 80.0);
        self.g_l  = self.g_l.clamp(0.05, 1.0);
        self.g_h  = self.g_h.clamp(0.0, 3.0);
    }
}

impl TipoNeuronal {
    /// Exporta a assinatura numérica exata de QUALQUER tipo puro como `DnaNeuronal`.
    /// Para `Hybrid` (sem genes próprios no enum) devolve o default tipo-RS.
    pub fn extrair_dna(&self) -> DnaNeuronal {
        let (a, b, c, d) = self.parametros();
        DnaNeuronal {
            a, b, c, d,
            threshold:     self.threshold_padrao(),
            g_nap:         self.g_nap(),
            g_m:           self.g_m(),
            g_a:           self.g_a(),
            g_t:           self.g_t(),
            g_bk:          self.g_bk(),
            tau_ca_ms:     self.tau_ca_ms(),
            bcm_theta:     self.bcm_theta(),
            e_inibitorico: self.e_inibitorico(),
            usa_hh:        self.usa_hh(),
            // Genes estruturais extraídos do tipo puro.
            tem_compartimentos: matches!(self, TipoNeuronal::RS | TipoNeuronal::IB),
            tipo_stp:           SinapseSTP::para_tipo(*self).tipo,
            g_na: self.parametros_hh().map(|p| p.g_na).unwrap_or(120.0),
            g_k:  self.parametros_hh().map(|p| p.g_k ).unwrap_or(36.0),
            g_l:  self.parametros_hh().map(|p| p.g_l ).unwrap_or(0.3),
            g_h:  self.parametros_hh().map(|p| p.g_h ).unwrap_or(0.0),
        }
    }
}

/// Gera uma espécie híbrida inédita a partir de dois pais.
///
/// Crossover por gene (mistura probabilística pai_a/pai_b) + mutação adaptativa
/// com ruído gaussiano controlado nos genes críticos (`g_m`, `g_nap`).
///
/// `taxa_mutacao` ∈ [0.0, 1.0] escala a amplitude do ruído (recomendado 0.10–0.15).
///
/// NOTA (design Copy-preserving): devolve o **genoma** (`DnaNeuronal`) — a "espécie".
/// Para instanciar um indivíduo, use `NeuronioHibrido::novo_hibrido(id, dna, prec)`.
/// Não devolve `TipoNeuronal` porque a variante `Hybrid` é unit (preserva `Copy`)
/// e o genoma precisa viajar à parte, no campo `dna` do neurônio.
pub fn gerar_especie_hibrida(
    pai_a: &TipoNeuronal,
    pai_b: &TipoNeuronal,
    taxa_mutacao: f32,
) -> DnaNeuronal {
    use rand::Rng;
    let da = pai_a.extrair_dna();
    let db = pai_b.extrair_dna();
    let mut rng = rand::thread_rng();

    // Crossover: mistura por gene. p ∈ [0,1] → herda fração de cada pai.
    let mix = |x: f32, y: f32, rng: &mut rand::rngs::ThreadRng| -> f32 {
        let p: f32 = rng.gen_range(0.0..1.0);
        x * p + y * (1.0 - p)
    };
    // Mutação: ruído relativo gaussiano-aproximado (soma de uniformes) ± taxa.
    let mutar = |valor: f32, escala: f32, rng: &mut rand::rngs::ThreadRng| -> f32 {
        // (u1+u2+u3)/3 ∈ [-1,1] aproxima gaussiana centrada (teorema central do limite).
        let r: f32 = (rng.gen_range(-1.0..1.0)
                    + rng.gen_range(-1.0..1.0)
                    + rng.gen_range(-1.0..1.0)) / 3.0;
        valor * (1.0 + r * escala)
    };

    let m = taxa_mutacao.clamp(0.0, 0.5);
    let mut filho = DnaNeuronal {
        a:         mix(da.a, db.a, &mut rng),
        b:         mix(da.b, db.b, &mut rng),
        c:         mix(da.c, db.c, &mut rng),
        d:         mix(da.d, db.d, &mut rng),
        threshold: mix(da.threshold, db.threshold, &mut rng),
        // Genes CRÍTICOS recebem mutação ativa (exploração de fenótipos novos):
        g_nap:     mutar(mix(da.g_nap, db.g_nap, &mut rng), m, &mut rng),
        g_m:       mutar(mix(da.g_m,   db.g_m,   &mut rng), m, &mut rng),
        // Genes estruturais: só crossover (mutação fraca para estabilidade):
        g_a:       mutar(mix(da.g_a,  db.g_a,  &mut rng), m * 0.3, &mut rng),
        g_t:       mix(da.g_t,  db.g_t,  &mut rng),
        g_bk:      mutar(mix(da.g_bk, db.g_bk, &mut rng), m * 0.3, &mut rng),
        tau_ca_ms: mix(da.tau_ca_ms, db.tau_ca_ms, &mut rng),
        bcm_theta: mix(da.bcm_theta, db.bcm_theta, &mut rng),
        // Fenótipo herdado do pai dominante no crossover de g_m (regra simples):
        e_inibitorico: if da.g_m >= db.g_m { da.e_inibitorico } else { db.e_inibitorico },
        // ── Genes estruturais: herança Mendeliana + mutação rara (exploração) ──
        // usa_hh / tem_compartimentos / tipo_stp herdam de um pai ao acaso;
        // com prob = taxa_mutacao, a flag estrutural inverte → fenótipo inédito
        // (ex.: um híbrido HH com dendritos que nenhum pai tinha).
        usa_hh: {
            let base = if rng.gen_bool(0.5) { da.usa_hh } else { db.usa_hh };
            if rng.gen::<f32>() < m { !base } else { base }
        },
        tem_compartimentos: {
            let base = if rng.gen_bool(0.5) { da.tem_compartimentos } else { db.tem_compartimentos };
            if rng.gen::<f32>() < m { !base } else { base }
        },
        tipo_stp: if rng.gen_bool(0.5) { da.tipo_stp } else { db.tipo_stp },
        // Condutâncias HH: crossover (mutação leve em g_na — molda a forma do spike).
        g_na: mutar(mix(da.g_na, db.g_na, &mut rng), m * 0.3, &mut rng),
        g_k:  mix(da.g_k, db.g_k, &mut rng),
        g_l:  mix(da.g_l, db.g_l, &mut rng),
        g_h:  mutar(mix(da.g_h, db.g_h, &mut rng), m * 0.3, &mut rng),
    };
    filho.clampar();
    filho
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 7 — MOTOR HH V3 (inclui I_T Ca²⁺ para TC/LT)
// ─────────────────────────────────────────────────────────────────────────────

struct HhV3;

impl HhV3 {
    #[inline] fn alpha_m(v: f32) -> f32 {
        let dv = v + 40.0;
        if dv.abs() < 1e-4 { 1.0 } else { 0.1 * dv / (1.0 - (-dv / 10.0).exp()) }
    }
    #[inline] fn beta_m(v: f32) -> f32 { 4.0 * (-(v + 65.0) / 18.0).exp() }
    #[inline] fn alpha_h(v: f32) -> f32 { 0.07 * (-(v + 65.0) / 20.0).exp() }
    #[inline] fn beta_h(v: f32) -> f32  { 1.0 / (1.0 + (-(v + 35.0) / 10.0).exp()) }
    #[inline] fn alpha_n(v: f32) -> f32 {
        let dv = v + 55.0;
        if dv.abs() < 1e-4 { 0.1 } else { 0.01 * dv / (1.0 - (-dv / 10.0).exp()) }
    }
    #[inline] fn beta_n(v: f32) -> f32 { 0.125 * (-(v + 65.0) / 80.0).exp() }

    pub fn integrar(estado: &mut EstadoHH, params: &ParametrosHH, v: f32, dt_ms: f32) -> f32 {
        let n_sub = ((dt_ms / 0.1).ceil() as usize).max(1).min(50);
        let dt_sub = dt_ms / n_sub as f32;
        let mut m = estado.m; let mut h = estado.h; let mut n = estado.n;

        for _ in 0..n_sub {
            let am = Self::alpha_m(v); let bm = Self::beta_m(v);
            let ah = Self::alpha_h(v); let bh = Self::beta_h(v);
            let an = Self::alpha_n(v); let bn = Self::beta_n(v);
            m += dt_sub * (am * (1.0 - m) - bm * m);
            h += dt_sub * (ah * (1.0 - h) - bh * h);
            n += dt_sub * (an * (1.0 - n) - bn * n);
            m = m.clamp(0.0, 1.0); h = h.clamp(0.0, 1.0); n = n.clamp(0.0, 1.0);
        }
        estado.m = m; estado.h = h; estado.n = n;

        let i_na = params.g_na * estado.g_na_mod * m.powi(3) * h * (v - E_NA);
        let i_k  = params.g_k  * estado.g_k_mod  * n.powi(4) * (v - E_K);
        let i_l  = params.g_l  * estado.g_l_mod  * (v - params.e_l);

        let i_h = if params.g_h > 0.0 {
            let alpha_q = (0.001 * (-0.1 * (v + 75.0)).exp()).min(10.0);
            let beta_q  = (0.001 * (0.1  * (v + 75.0)).exp()).min(10.0);
            let q_new = estado.q_ih + dt_sub * (alpha_q * (1.0 - estado.q_ih) - beta_q * estado.q_ih);
            estado.q_ih = q_new.clamp(0.0, 1.0);
            params.g_h * estado.q_ih * (v - E_H)
        } else { 0.0 };

        i_na + i_k + i_l + i_h
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 8 — SHORT-TERM PLASTICITY (Tsodyks-Markram 1997)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum TipoSTP {
    #[default]
    Depression,
    Facilitation,
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinapseSTP {
    pub x:       f32,
    pub u_stp:   f32,
    pub u0:      f32,
    tau_rec:     f32,
    tau_fac:     f32,
    pub tipo:    TipoSTP,
}

impl SinapseSTP {
    pub fn para_tipo(tipo: TipoNeuronal) -> Self {
        match tipo {
            TipoNeuronal::RS | TipoNeuronal::IB => Self {
                x: 1.0, u_stp: 0.45, u0: 0.45, tau_rec: 800.0, tau_fac: 0.0,
                tipo: TipoSTP::Depression,
            },
            TipoNeuronal::FS => Self {
                x: 1.0, u_stp: 0.25, u0: 0.25, tau_rec: 600.0, tau_fac: 0.0,
                tipo: TipoSTP::Depression,
            },
            TipoNeuronal::CH | TipoNeuronal::LT => Self {
                x: 1.0, u_stp: 0.15, u0: 0.15, tau_rec: 300.0, tau_fac: 150.0,
                tipo: TipoSTP::Facilitation,
            },
            TipoNeuronal::TC => Self {
                x: 1.0, u_stp: 0.30, u0: 0.30, tau_rec: 500.0, tau_fac: 50.0,
                tipo: TipoSTP::Mixed,
            },
            TipoNeuronal::RZ => Self {
                x: 1.0, u_stp: 0.50, u0: 0.50, tau_rec: 400.0, tau_fac: 0.0,
                tipo: TipoSTP::Depression,
            },
            TipoNeuronal::PS | TipoNeuronal::PB => Self {
                x: 1.0, u_stp: 0.60, u0: 0.60, tau_rec: 700.0, tau_fac: 0.0,
                tipo: TipoSTP::Depression,
            },
            TipoNeuronal::AC => Self {
                x: 1.0, u_stp: 0.40, u0: 0.40, tau_rec: 900.0, tau_fac: 0.0,
                tipo: TipoSTP::Depression,
            },
            TipoNeuronal::BI => Self {
                x: 1.0, u_stp: 0.20, u0: 0.20, tau_rec: 200.0, tau_fac: 300.0,
                tipo: TipoSTP::Facilitation,
            },
            TipoNeuronal::DAP => Self {
                x: 1.0, u_stp: 0.35, u0: 0.35, tau_rec: 400.0, tau_fac: 80.0,
                tipo: TipoSTP::Mixed,
            },
            TipoNeuronal::IIS => Self {
                x: 1.0, u_stp: 0.10, u0: 0.10, tau_rec: 1000.0, tau_fac: 0.0,
                tipo: TipoSTP::Depression,
            },
            TipoNeuronal::PV => Self {
                x: 1.0, u_stp: 0.30, u0: 0.30, tau_rec: 500.0, tau_fac: 0.0,
                tipo: TipoSTP::Depression,
            },
            TipoNeuronal::SST => Self {
                x: 1.0, u_stp: 0.10, u0: 0.10, tau_rec: 200.0, tau_fac: 500.0,
                tipo: TipoSTP::Facilitation,
            },
            TipoNeuronal::VIP => Self {
                x: 1.0, u_stp: 0.20, u0: 0.20, tau_rec: 300.0, tau_fac: 200.0,
                tipo: TipoSTP::Facilitation,
            },
            TipoNeuronal::DA_N => Self {
                x: 1.0, u_stp: 0.15, u0: 0.15, tau_rec: 1500.0, tau_fac: 0.0,
                tipo: TipoSTP::Depression,
            },
            // V3.1
            // NGF: GABA volumétrico lento (GABA-B like) — fortemente facilitante
            TipoNeuronal::NGF => Self {
                x: 1.0, u_stp: 0.10, u0: 0.10, tau_rec: 300.0, tau_fac: 800.0,
                tipo: TipoSTP::Facilitation,
            },
            // LC_N: burst NA depleta vesículas rapidamente
            TipoNeuronal::LC_N => Self {
                x: 1.0, u_stp: 0.40, u0: 0.40, tau_rec: 600.0, tau_fac: 0.0,
                tipo: TipoSTP::Depression,
            },
            // ChIN: tônico sustentado, recuperação intermediária
            TipoNeuronal::ChIN => Self {
                x: 1.0, u_stp: 0.25, u0: 0.25, tau_rec: 400.0, tau_fac: 100.0,
                tipo: TipoSTP::Mixed,
            },
            // V4.6
            // GridCell: depressão suave — mantém ritmo estável
            TipoNeuronal::GridCell => Self {
                x: 1.0, u_stp: 0.30, u0: 0.30, tau_rec: 500.0, tau_fac: 0.0,
                tipo: TipoSTP::Depression,
            },
            // MirrorCell: igual a RS (piramidal depressiva)
            TipoNeuronal::MirrorCell => Self {
                x: 1.0, u_stp: 0.45, u0: 0.45, tau_rec: 800.0, tau_fac: 0.0,
                tipo: TipoSTP::Depression,
            },
            // MSN: facilitante — integra drive cortical convergente
            TipoNeuronal::MSN => Self {
                x: 1.0, u_stp: 0.15, u0: 0.15, tau_rec: 300.0, tau_fac: 400.0,
                tipo: TipoSTP::Facilitation,
            },
            // Hybrid: default depressivo tipo-RS (STP não é gene do DNA)
            TipoNeuronal::Hybrid => Self {
                x: 1.0, u_stp: 0.45, u0: 0.45, tau_rec: 800.0, tau_fac: 0.0,
                tipo: TipoSTP::Depression,
            },
        }
    }

    /// V4.6 — Constrói a STP a partir do gene `TipoSTP` (para neurônios Hybrid).
    /// Parâmetros canônicos por classe (Tsodyks-Markram 1997).
    pub fn from_tipo_stp(tipo: TipoSTP) -> Self {
        match tipo {
            TipoSTP::Depression => Self {
                x: 1.0, u_stp: 0.45, u0: 0.45, tau_rec: 800.0, tau_fac: 0.0, tipo,
            },
            TipoSTP::Facilitation => Self {
                x: 1.0, u_stp: 0.15, u0: 0.15, tau_rec: 300.0, tau_fac: 500.0, tipo,
            },
            TipoSTP::Mixed => Self {
                x: 1.0, u_stp: 0.30, u0: 0.30, tau_rec: 500.0, tau_fac: 200.0, tipo,
            },
        }
    }

    pub fn tick(&mut self, spiked: bool, dt_ms: f32) -> f32 {
        self.x += dt_ms * (1.0 - self.x) / self.tau_rec;
        self.x = self.x.clamp(0.0, 1.0);

        if self.tau_fac > 0.0 {
            self.u_stp += dt_ms * (self.u0 - self.u_stp) / self.tau_fac;
            self.u_stp = self.u_stp.clamp(0.01, 1.0);
        }

        if spiked {
            let eficacia = self.u_stp * self.x;
            self.x -= self.u_stp * self.x;
            self.x = self.x.clamp(0.0, 1.0);
            if self.tau_fac > 0.0 {
                self.u_stp += self.u0 * (1.0 - self.u_stp);
                self.u_stp = self.u_stp.clamp(0.0, 1.0);
            }
            eficacia
        } else {
            self.u_stp * self.x
        }
    }

    pub fn fator(&self) -> f32 {
        let efic = self.u_stp * self.x;
        let inicial = self.u0;
        if inicial < 1e-6 { 1.0 } else { (efic / inicial).clamp(0.05, 3.0) }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 9 — ESTADO DOS CANAIS EXTRAS (por neurônio)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstadoCanaisExtras {
    pub w_m:          f32,
    pub a_ka:         f32,
    pub b_ka:         f32,
    pub m_t:          f32,
    pub h_t:          f32,
    pub q_bk:         f32,
    pub ca_nmda:      f32,
    pub elig_trace:   f32,
    pub mod_ach:      f32,
    pub stp_efficacy: f32,
    pub stp:          SinapseSTP,

    // ── V3.1: novos campos ──────────────────────────────────────────────────

    /// RS BAC firing: ms restantes de corrente extra injetada após coincidência apical+somática.
    /// Criado quando RS dispara E input_apical > APICAL_THRESHOLD.
    pub burst_remaining_ms: f32,

    /// DA_N RPE negativo: ms acumulados de hiperpolarização sem disparo esperado.
    /// Quando > DAN_HYPERPOL_THRESHOLD_MS → sinaliza RPE negativo → mod_dopa = 0.6.
    pub dan_hyperpol_ms: f32,

    /// ChIN gate: true = janela de plasticidade aberta (ChIN em pausa).
    /// Injetado pelo CamadaHibrida antes de cada tick.
    /// Quando true, dopamina no STDP 3-fatores é efetiva.
    pub chin_window_open: bool,

    /// Limite máximo de ca_nmda — escalonado pelo Astrócito quando ativo.
    /// Padrão 2.0; astrócito ativo → 4.0 (D-serina amplifica NMDA).
    pub ca_nmda_max: f32,
}

impl EstadoCanaisExtras {
    pub fn para_tipo(tipo: TipoNeuronal) -> Self {
        Self {
            w_m:                0.047,
            a_ka:               0.36,
            b_ka:               0.10,
            m_t:                0.01,
            h_t:                0.018,
            q_bk:               0.0,
            ca_nmda:            0.0,
            elig_trace:         0.0,
            mod_ach:            1.0,
            stp_efficacy:       1.0,
            stp:                SinapseSTP::para_tipo(tipo),
            burst_remaining_ms: 0.0,
            dan_hyperpol_ms:    0.0,
            // Default true: janela DA-STDP aberta quando nenhum ChIN existe/ativa.
            // CamadaHibrida sobrescreve via chin_paused antes de cada update paralelo
            // (Pre-tick B). Sem isso, neurônios isolados (testes, ou layers sem ChIN
            // no pool) nunca consolidam por dopamina — bug que zerava o 3º fator STDP.
            chin_window_open:   true,
            ca_nmda_max:        2.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 10 — NEURÔNIO HÍBRIDO
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronioHibrido {
    pub id:            u32,
    pub tipo:          TipoNeuronal,
    pub precisao:      PrecisionType,
    pub peso:          PesoNeuronio,
    pub v:             f32,
    pub u:             f32,
    pub refr_count:    u16,
    pub threshold:     f32,
    pub trace_pre:     f32,
    pub trace_pos:     f32,
    pub last_spike_ms: f32,
    pub ca_intra:      f32,
    pub mod_dopa:      f32,
    pub mod_sero:      f32,
    pub mod_cort:      f32,
    pub activity_avg:  f32,
    pub modelo:        ModeloDinamico,
    pub extras:        Box<EstadoCanaisExtras>,

    /// RS BAC: entrada do compartimento dendrítico apical.
    /// Injetado pelo CamadaHibrida ou chamador antes de cada tick.
    /// Consumido (resetado para 0.0) ao final de update().
    pub input_apical:  f32,

    /// BDNF concentration [0.0, 2.0] — mediador early→late LTP (Turrigiano 2022).
    /// Incrementado quando delta_ltp > 0. Amplifica eligibility trace (up 2x).
    /// Decai com τ=30s. Convertido para late-LTP em reconsolidacao.rs quando > threshold.
    pub bdnf:          f32,

    /// Limiar de modificação BCM por neurônio [0.001, 0.5].
    /// Inicializado com `tipo.bcm_theta()`; desliza com activity_avg² (τ=30s).
    /// LTP quando activity_avg > theta_m; LTD quando abaixo (BCM 1982).
    pub theta_m:       f32,

    // ── V3.6: Ciclo de Vida Neural ────────────────────────────────────────────
    /// Contador de inatividade em ticks. Incrementado a cada chamada a
    /// SwapManager::tick_atividade_neuronal(); resetado a 0 quando a corrente
    /// do neurônio é > 0.1 (neurônio recebendo input). Quando ≥ INACTIVITY_THRESHOLD
    /// o status passa para Dormant e o neurônio é elegível para evicção ao NVMe.
    pub activity_timer: u64,
    /// Estado do ciclo de vida: Active (processando), Dormant (inativo), Swapped (no NVMe).
    pub status: NeuronalStatus,

    // ── V4: Neurônio multicompartimental ──────────────────────────────────────
    // serde(default): estado V3.1 persistido (sem estes campos) ainda carrega.
    /// 5 compartimentos dendríticos (AIS, Soma, Tronco, Tufo, Extracelular).
    /// `Some` apenas para RS/IB; `None` para todos os outros tipos (não-quebrante).
    #[serde(default)]
    pub compartimentos: Option<Box<EstadoCompartimentos>>,
    /// Estado metabólico (ATP, bomba Na/K, [K⁺]o, E_K dinâmico). Ativo em todos.
    #[serde(default)]
    pub metabolismo:    Box<EstadoMetabolico>,
    /// Brain state corrente — modula condutância de acoplamento apical.
    #[serde(default)]
    pub brain_state:    EstadoBrainState,

    // ── V4.6: genoma híbrido / célula-tronco ──────────────────────────────────
    /// Genoma digital. `Some` apenas quando `tipo == Hybrid` (fenótipo em runtime).
    /// `Option<Box<_>>` usa niche optimization: `None` = ponteiro nulo, 0 bytes em heap.
    /// Os getters `*_efetivo()` lêem daqui quando presente; senão usam `self.tipo`.
    #[serde(default)]
    pub dna: Option<Box<DnaNeuronal>>,

    /// Contador de subsampling metabólico (V4.6 item 2). Avança a cada tick;
    /// o metabolismo só recalcula a cada `METAB_SUBSAMPLE` ticks (dt multiplicado).
    #[serde(default)]
    pub metab_tick: u64,

    /// Item 2 — otimização 200 Hz ativa neste neurônio (event-driven + subsampler).
    /// `true` em produção. O teste A/B desliga por camada para medir a referência.
    #[serde(default = "otimizar_padrao")]
    pub otimizar: bool,
}

impl NeuronioHibrido {
    pub fn new(id: u32, tipo: TipoNeuronal, precisao: PrecisionType) -> Self {
        let peso = match precisao {
            PrecisionType::FP32 => PesoNeuronio::FP32(1.0),
            PrecisionType::FP16 => PesoNeuronio::FP16(half::f16::from_f32(1.0)),
            PrecisionType::INT8 => PesoNeuronio::INT8(100),
            PrecisionType::INT4 => PesoNeuronio::INT4(0x77u8),
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
            ca_intra:        0.0,
            mod_dopa:        1.0,
            mod_sero:        1.0,
            mod_cort:        0.0,
            activity_avg:    0.0,
            modelo:        ModeloDinamico::para_tipo(tipo),
            extras:        Box::new(EstadoCanaisExtras::para_tipo(tipo)),
            input_apical:    0.0,
            bdnf:            0.0,
            theta_m:         tipo.bcm_theta(),
            activity_timer:  0,
            status:          NeuronalStatus::Active,
            compartimentos: match tipo {
                TipoNeuronal::RS | TipoNeuronal::IB => {
                    Some(Box::new(EstadoCompartimentos::novo()))
                }
                _ => None,
            },
            metabolismo:  Box::new(EstadoMetabolico::novo()),
            brain_state:  EstadoBrainState::Vigilia,
            dna:          None,
            metab_tick:   0,
            otimizar:     true,
        }
    }

    /// Constrói um neurônio `Hybrid` cujo fenótipo é definido pelo `dna`.
    /// Usado pela neuroevolução / célula-tronco para instanciar espécies inéditas.
    pub fn novo_hibrido(id: u32, dna: DnaNeuronal, precisao: PrecisionType) -> Self {
        let mut n = Self::new(id, TipoNeuronal::Hybrid, precisao);
        // Inicializa estado dependente dos genes a partir do DNA.
        n.threshold = dna.threshold;
        n.theta_m   = dna.bcm_theta;

        // ── Genes ESTRUTURAIS (genoma expandido V4.6) ─────────────────────────
        // Dinâmica HH evoluível: troca o modelo Izhikevich puro por HH se o gene pedir.
        if dna.usa_hh {
            n.modelo = ModeloDinamico::IzhikevichHH(Box::new(EstadoHH::repouso()));
        }
        // Dendritos multicompartimentais evoluíveis.
        if dna.tem_compartimentos && n.compartimentos.is_none() {
            n.compartimentos = Some(Box::new(EstadoCompartimentos::novo()));
        }
        // Plasticidade de curto prazo evoluível.
        n.extras.stp = SinapseSTP::from_tipo_stp(dna.tipo_stp);

        n.dna = Some(Box::new(dna));
        n
    }

    /// Parâmetros HH efetivos: do DNA (Hybrid) ou do tipo puro. Para tipos sem HH
    /// devolve um conjunto HH genérico (só usado se o modelo for IzhikevichHH).
    #[inline]
    pub fn parametros_hh_efetivo(&self) -> ParametrosHH {
        if let Some(d) = &self.dna {
            ParametrosHH {
                g_na: d.g_na, g_k: d.g_k, g_l: d.g_l,
                e_na: E_NA, e_k: E_K, e_l: -54.4, c_m: 1.0, g_h: d.g_h,
            }
        } else {
            self.tipo.parametros_hh().unwrap_or(ParametrosHH {
                g_na: 120.0, g_k: 36.0, g_l: 0.3,
                e_na: E_NA, e_k: E_K, e_l: -54.4, c_m: 1.0, g_h: 0.0,
            })
        }
    }

    // ── V4.6: getters de genes efetivos (DNA tem prioridade sobre o tipo) ──────
    /// Parâmetros Izhikevich efetivos: do DNA se `Hybrid`, senão do tipo puro.
    #[inline]
    pub fn parametros_efetivos(&self) -> (f32, f32, f32, f32) {
        match &self.dna {
            Some(d) => (d.a, d.b, d.c, d.d),
            None    => self.tipo.parametros(),
        }
    }
    #[inline]
    pub fn threshold_padrao_efetivo(&self) -> f32 {
        match &self.dna { Some(d) => d.threshold, None => self.tipo.threshold_padrao() }
    }
    #[inline]
    pub fn tau_ca_efetivo(&self) -> f32 {
        match &self.dna { Some(d) => d.tau_ca_ms, None => self.tipo.tau_ca_ms() }
    }
    #[inline]
    pub fn g_nap_efetivo(&self) -> f32 {
        match &self.dna { Some(d) => d.g_nap, None => self.tipo.g_nap() }
    }
    #[inline]
    pub fn g_m_efetivo(&self) -> f32 {
        match &self.dna { Some(d) => d.g_m, None => self.tipo.g_m() }
    }
    #[inline]
    pub fn g_a_efetivo(&self) -> f32 {
        match &self.dna { Some(d) => d.g_a, None => self.tipo.g_a() }
    }
    #[inline]
    pub fn g_t_efetivo(&self) -> f32 {
        match &self.dna { Some(d) => d.g_t, None => self.tipo.g_t() }
    }
    #[inline]
    pub fn g_bk_efetivo(&self) -> f32 {
        match &self.dna { Some(d) => d.g_bk, None => self.tipo.g_bk() }
    }
    /// GABAérgico efetivo: do DNA se `Hybrid`, senão do tipo puro.
    #[inline]
    pub fn e_inibitorico_efetivo(&self) -> bool {
        match &self.dna { Some(d) => d.e_inibitorico, None => self.tipo.e_inibitorico() }
    }

    /// Item 2 — Event-driven: `true` quando o neurônio está tão silencioso que as
    /// equações pesadas (canais iônicos HH/I_M/I_A/I_BK/I_T + compartimentos
    /// dendríticos) podem ser puladas neste tick SEM alterar a dinâmica observável.
    ///
    /// O core Izhikevich (barato) e TODOS os decaimentos continuam a correr — só
    /// se omite o cálculo de correntes de canais, que perto do repouso é ~0.
    ///
    /// Pacemakers e tipos autônomos (DA_N, ChIN, LC_N, TC, IIS, BI, GridCell)
    /// NUNCA pulam — geram atividade própria sem input. Híbridos também não pulam
    /// (fenótipo desconhecido → conservador).
    #[inline]
    fn quiescente(&self, input_efetivo: f32) -> bool {
        let autonomo = matches!(self.tipo,
            TipoNeuronal::DA_N | TipoNeuronal::ChIN | TipoNeuronal::LC_N |
            TipoNeuronal::TC   | TipoNeuronal::IIS  | TipoNeuronal::BI   |
            TipoNeuronal::GridCell);
        if !self.otimizar || autonomo || self.dna.is_some() {
            return false;
        }
        input_efetivo.abs()           < 0.05
            && self.refr_count            == 0
            && self.ca_intra              < 0.05
            && self.extras.ca_nmda        < 0.02
            && self.extras.elig_trace     < 0.02
            && self.extras.burst_remaining_ms <= 0.0
            && self.v                     < self.threshold - 15.0
            && self.activity_avg          < 0.005
    }

    /// Item 1 — Zera buffers transitórios de plasticidade/canais (limpeza pós-sono).
    /// Devolve o neurônio a baseline SEM destruir o que foi aprendido:
    /// `peso`, `threshold`, `activity_avg` e `theta_m` permanecem intactos.
    /// Alvo: neurônios que entram em `Dormant` ao encerrar o ciclo de sono.
    pub fn liberar_buffers_temporarios(&mut self) {
        self.extras.ca_nmda            = 0.0;
        self.extras.elig_trace         = 0.0;
        self.extras.q_bk               = 0.0;
        self.extras.burst_remaining_ms = 0.0;
        self.extras.dan_hyperpol_ms    = 0.0;
        self.trace_pre                 = 0.0;
        self.trace_pos                 = 0.0;
        self.ca_intra                  = 0.0;
        self.input_apical              = 0.0;
    }

    pub fn update(
        &mut self,
        input_current:   f32,
        dt_segundos:     f32,
        current_time_ms: f32,
        escala_camada:   f32,
    ) -> bool {
        let dt_ms = dt_segundos * 1000.0;

        // ── 1. Período refratário ────────────────────────────────────────
        if self.refr_count > 0 {
            self.refr_count -= 1;
            self.v = -70.0;
            let decay = (-dt_ms / TAU_STDP_MS).exp();
            self.trace_pre *= decay;
            self.trace_pos *= decay;
            let tb = self.threshold_padrao_efetivo();
            self.threshold = tb + (self.threshold - tb) * THRESHOLD_DECAY;
            self.extras.elig_trace *= (-dt_ms / TAU_ELIG_MS).exp();
            self.extras.ca_nmda   *= (-dt_ms / TAU_NMDA_CA_MS).exp();
            self.extras.q_bk      *= (-dt_ms / TAU_BK_MS).exp();
            // Decai contador BAC durante refratário
            self.extras.burst_remaining_ms = (self.extras.burst_remaining_ms - dt_ms).max(0.0);
            return false;
        }

        // ── NTC Fast Paths: INT4/INT8 pulam HH + canais iônicos ─────────
        match self.precisao {
            PrecisionType::INT4 => return self.ntc_update_int4(
                input_current, dt_ms, current_time_ms, escala_camada,
            ),
            PrecisionType::INT8 => return self.ntc_update_int8(
                input_current, dt_ms, current_time_ms, escala_camada,
            ),
            _ => {}
        }

        // ── 2. Quantização do input ──────────────────────────────────────
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

        // ── 3. Registra eficácia STP (para output sináptico) ─────────────
        self.extras.stp_efficacy = self.extras.stp.fator();
        let input_stp = input_q;

        // ── 4. Correntes HH (TC, RZ e híbridos com gene usa_hh) ──────────
        // Pré-computa os parâmetros HH antes do borrow mutável de self.modelo
        // (híbridos lêem do DNA; tipos puros, do tipo). Custo desprezável.
        let params_hh = self.parametros_hh_efetivo();
        let v_atual = self.v;
        let i_hh = if let ModeloDinamico::IzhikevichHH(ref mut estado) = self.modelo {
            HhV3::integrar(estado, &params_hh, v_atual, dt_ms)
        } else {
            0.0
        };

        // ── Item 2: Event-driven — neurônio silencioso pula a física pesada ──
        // Canais iônicos e compartimentos perto do repouso contribuem ~0; pular
        // poupa a maior parte do custo sem alterar a dinâmica observável.
        // (quiescente() já respeita self.otimizar.)
        let pular_pesado = self.quiescente(input_stp);

        // ── 5. Novos canais iônicos ──────────────────────────────────────
        let i_extra = if pular_pesado { 0.0 } else { self.calcular_canais_extras(dt_ms) };

        // ── 5b. Compartimentos dendríticos + AIS (RS/IB apenas) ───────────
        let i_comp = if !pular_pesado && self.compartimentos.is_some() {
            let api = self.input_apical;
            self.integrar_compartimentos(dt_ms, api)
        } else {
            0.0
        };

        // ── 5c. Metabolismo: ATP + bomba Na/K + [K⁺]o dinâmico ────────────
        // CHAMADA 1: antes dos substeps — retorna i_pump para I_eff.
        //
        // Item 2 (subsampler): metabolismo é caro mas lento. SÓ subsampla quando o
        // neurônio está OCIOSO (ATP perto do equilíbrio, sem interesse); durante
        // atividade (ca_intra alto / burst) corre a CADA tick para não distorcer a
        // depleção/recuperação de ATP. `metab_tick` acumula ticks desde a última
        // execução → dt × ticks dá a integral EXATA, independente da cadência.
        self.metab_tick = self.metab_tick.wrapping_add(1);
        let metab_ativo = self.ca_intra > 0.3 || self.extras.burst_remaining_ms > 0.0;
        let deve_rodar_metab = !self.otimizar
            || metab_ativo
            || self.metab_tick >= METAB_SUBSAMPLE;
        let i_pump = if deve_rodar_metab {
            let escala_dt = self.metab_tick as f32; // ticks acumulados desde a última
            self.metab_tick = 0;
            self.atualizar_metabolismo(false, dt_ms * escala_dt)
        } else {
            // Ocioso entre execuções: reutiliza a última corrente da bomba.
            self.metabolismo.i_pump * HH_SCALE * 0.5
        };

        // ── 5d. ATP penalty: ATP baixo → threshold sobe (modo economia) ───
        let atp_penalty = if self.metabolismo.atp < 0.8 {
            (0.8 - self.metabolismo.atp) * 8.0
        } else {
            0.0
        };

        // ── 6. I_eff base ─────────────────────────────────────────────────
        let mut i_eff = input_stp
            - (i_hh + i_extra) * HH_SCALE
            + i_comp   // corrente dos compartimentos dendríticos
            - i_pump;  // bomba Na/K eletrogenica (hiperpolarizante)

        // ── 6a. RS BAC burst: injeta corrente extra se burst ativo ────────
        // Biológico: AP retrógrado ativa canais de Ca²⁺ apicais →
        // burst de alta frequência sustentado por BURST_DURATION_MS.
        if self.tipo == TipoNeuronal::RS && self.extras.burst_remaining_ms > 0.0 {
            i_eff += BURST_CURRENT;
            self.extras.burst_remaining_ms = (self.extras.burst_remaining_ms - dt_ms).max(0.0);
        }

        // ── 6b. TC + ACh: depolarização muscarínica → modo tônico de vigília
        // M1/M4 receptors → PLC → IP3 → bloqueia I_Kleak → +5mV de bias.
        // Em calcular_canais_extras(), h_T é inativado mais rapidamente (tau reduzido).
        if self.tipo == TipoNeuronal::TC && self.extras.mod_ach > 1.2 {
            let ach_exc = (self.extras.mod_ach - 1.2).clamp(0.0, 0.8);
            // +5mV × excesso ACh: empurra v de repouso para cima (burst → tônico)
            i_eff += 5.0 * ach_exc;
        }

        // ── 7. Substeps Izhikevich (~1 ms cada) ──────────────────────────
        let n_sub  = (dt_ms.round() as usize).max(1);
        let dt_int = dt_ms / n_sub as f32;
        let (a, b, c, d) = self.parametros_efetivos();
        let mut spiked = false;

        let neuro_thresh_offset = -(self.mod_dopa - 1.0) * 2.0
                                  + self.mod_cort * 4.5
                                  - (self.mod_sero - 1.0) * 0.8
                                  - (self.extras.mod_ach - 1.0) * 1.5;
        // V4.6.1 core-tuning: o AHP (Ca²⁺) é o limitador dominante da taxa
        // sustentada. Escala por tipo → piramidais adaptam forte (~20–40 Hz),
        // FS/PV quase nada (mantêm-se rápidos). Corrige o achado F-I.
        let g_ahp_scale = match self.tipo {
            TipoNeuronal::FS | TipoNeuronal::PV => 0.1,
            TipoNeuronal::RS | TipoNeuronal::MirrorCell => 3.2,
            TipoNeuronal::IB  => 3.6,
            TipoNeuronal::CH  => 2.6,
            TipoNeuronal::AC  => 3.5,
            TipoNeuronal::LT | TipoNeuronal::TC | TipoNeuronal::RZ
                | TipoNeuronal::GridCell => 1.8,
            TipoNeuronal::MSN => 2.2,
            _ => 1.0,
        };
        let ahp_extra = G_AHP * self.ca_intra * g_ahp_scale;
        let threshold_efetivo = self.threshold
            + neuro_thresh_offset
            + ahp_extra
            + atp_penalty; // ATP baixo → threshold sobe (modo economia)

        for _ in 0..n_sub {
            self.v += dt_int * (0.04 * self.v.powi(2) + 5.0 * self.v + 140.0 - self.u + i_eff);
            self.u += dt_int * a * (b * self.v - self.u);
            self.v = self.v.clamp(-100.0, 100.0);

            if self.v >= threshold_efetivo {
                self.v = c;
                self.u += d;
                self.threshold += THRESHOLD_DELTA * self.tipo.fator_adaptacao();
                self.refr_count = (2.0 / dt_int).round() as u16;
                spiked = true;
                break;
            }
        }

        // ── 7a. RS BAC Firing: coincidência apical+somática ───────────────
        // Se RS disparou E recebeu input apical suficiente → trigger burst.
        // Biológico: Larkum 1999 — spike retrógrado + EPSP apical → Ca²⁺ spike
        // → ca_nmda × 2.0 (amplifica janela NMDA) + burst de 5ms.
        if spiked && self.tipo == TipoNeuronal::RS && self.input_apical > APICAL_THRESHOLD {
            self.extras.burst_remaining_ms = BURST_DURATION_MS;
            // Coincidência amplifica entrada de Ca²⁺ NMDA — potencial de LTP máximo
            let nmda_cap = self.extras.ca_nmda_max;
            self.extras.ca_nmda = (self.extras.ca_nmda * 2.0).min(nmda_cap);
        }

        // ── 7b. V4: pós-spike — ativa BAP + debita ATP do spike ───────────
        if spiked {
            // BAP: AP retrógrado propaga soma → tronco → tufo (RS/IB apenas)
            if let Some(comp) = self.compartimentos.as_mut() {
                comp.bap_active   = true;
                comp.bap_timer_ms = 0.0;
            }
            // CHAMADA 2: debita ATP_COST_PER_SPIKE + libera K⁺ (dt=0.0)
            self.atualizar_metabolismo(true, 0.0);
        }

        self.input_apical = 0.0; // one-shot: consumido a cada tick

        // ── 8. Ca²⁺ AHP (SK) + BK rápido pós-spike ──────────────────────
        let ca_decay = (-dt_ms / self.tau_ca_efetivo()).exp();
        self.ca_intra *= ca_decay;
        self.extras.q_bk *= (-dt_ms / TAU_BK_MS).exp();
        self.extras.q_bk = self.extras.q_bk.clamp(0.0, 1.0);

        if spiked {
            self.ca_intra = (self.ca_intra + CA_POR_SPIKE).min(CA_MAX);
            self.extras.q_bk = (self.extras.q_bk + BK_PER_SPIKE).min(1.0);
        }
        if self.mod_sero > 1.0 {
            self.ca_intra *= 1.0 - (self.mod_sero - 1.0) * 0.05;
        }

        // ── 9. BCM homeostático ───────────────────────────────────────────
        let bcm_decay = (-dt_ms / TAU_BCM_MS).exp();
        let spike_val = if spiked { 1.0 } else { 0.0 };
        self.activity_avg = self.activity_avg * bcm_decay + spike_val * (1.0 - bcm_decay);
        // θ_M desliza com activity² — regra BCM dinâmica (Bienenstock-Cooper-Munro 1982)
        self.theta_m += (self.activity_avg.powi(2) - self.theta_m) * dt_ms / TAU_BCM_THETA_MS;
        self.theta_m = self.theta_m.clamp(0.001, 0.5);

        // ── 10. Decaimento dos traços STDP ────────────────────────────────
        let decay = (-dt_ms / TAU_STDP_MS).exp();
        self.trace_pre *= decay;
        self.trace_pos *= decay;

        // ── 11. STDP V3.1: 3 fatores bidirecional + gate ChIN ─────────────
        if spiked {
            let mg_unblock = 1.0 / (1.0 + 0.28 * (-0.062 * self.v).exp());
            if self.trace_pre > 0.05 {
                let ach_ltp_boost = if self.extras.mod_ach > 1.0 { 1.2 } else { 1.0 };
                let nmda_in = NMDA_CA_RATE * self.trace_pre * mg_unblock * ach_ltp_boost;
                let nmda_cap = self.extras.ca_nmda_max;
                self.extras.ca_nmda = (self.extras.ca_nmda + nmda_in).min(nmda_cap);
            }

            // BCM canônica (Bienenstock-Cooper-Munro 1982): θ_M por neurônio, desliza com activity².
            // bcm_raw > 0 → zona LTP (activity acima de θ_M → fortalecer sinapses ativas).
            // bcm_raw < 0 → zona LTD (activity abaixo de θ_M → enfraquecer sinapses fracas).
            let bcm_raw = self.activity_avg * (self.activity_avg - self.theta_m);
            // Normalizar por θ_M² para escala adimensional; mapear para gate multiplicativo [0.1, 2.0]
            let bcm_scaled = (bcm_raw / self.theta_m.powi(2).max(1e-4)).clamp(-3.0, 5.0);
            let bcm_mod = (1.0 + bcm_scaled * BCM_RATE * 100.0).clamp(0.1, 2.0);

            // BDNF amplifies eligibility uptake (early→late LTP mediator)
            let bdnf_amp = (1.0 + self.bdnf * 0.5).min(2.0); // up to 2x boost
            let elig_bump = ELIG_RATE * self.extras.ca_nmda * bcm_mod.max(0.0) * bdnf_amp;
            self.extras.elig_trace = (self.extras.elig_trace + elig_bump).min(1.0);

            let hz_atual = 1000.0 / dt_ms;
            let ltd_threshold = crate::config::janela_stdp_atual(hz_atual);
            let delta_ltp = LTP_RATE * self.trace_pre * bcm_mod.max(0.1);

            // BDNF release: proportional to LTP induction (Turrigiano 2022)
            if delta_ltp > 0.0 {
                const BDNF_RELEASE_RATE: f32 = 0.15; // molar equivalents
                self.bdnf = (self.bdnf + BDNF_RELEASE_RATE * delta_ltp).min(2.0);
            }

            let delta_ltd = if self.trace_pre < ltd_threshold {
                -LTD_RATE * (1.0 - self.trace_pre) / bcm_mod.max(0.1)
            } else { 0.0 };

            // ── STDP fator 3: dopamina bidirecional + gate ChIN ─────────────
            // RPE⁺ (mod_dopa > 1.0) → LTP proporcional ao burst de dopamina.
            // RPE⁻ (mod_dopa < 1.0) → LTD invertido: penaliza elig_trace ativo.
            // Gate ChIN: dopamina só é efetiva quando ChIN está em PAUSA (chin_window_open = true).
            // Biológico: ChIN ativo → ACh → M1 → suprime DARPP-32 → bloqueia consolidação.
            let dopa_diff = self.mod_dopa - 1.0;
            let delta_dopa3 = if !self.extras.chin_window_open {
                // ChIN em disparo tônico: janela fechada → sem consolidação sináptica
                0.0
            } else if dopa_diff > 0.0 {
                // RPE positivo: LTP dopaminérgico (ΔW = DOPA_GATE × DA_burst × elig)
                let burst = dopa_diff.min(2.0);
                let d3 = DOPA_GATE * burst * self.extras.elig_trace;
                // Uso parcial do traço de elegibilidade
                self.extras.elig_trace *= 1.0 - burst * 0.1;
                d3
            } else if dopa_diff < -0.01 {
                // RPE negativo: LTD invertido (ΔW = −DOPA_GATE × |RPE⁻| × elig)
                // Biológico: DA dip → D2/D1 desbalanceados → LTD nas sinapses co-ativas
                let neg = (-dopa_diff).min(1.0);
                let d3 = -DOPA_GATE * neg * self.extras.elig_trace;
                self.extras.elig_trace *= 1.0 - neg * 0.05;
                d3
            } else {
                0.0
            };

            self.atualizar_peso(delta_ltp + delta_ltd + delta_dopa3);
            self.trace_pos = 1.0;
            self.trace_pre = (self.trace_pre + 0.5).min(1.0);
            self.last_spike_ms = current_time_ms;
        }

        // ── 12. Decaimentos finais ────────────────────────────────────────
        self.extras.elig_trace *= (-dt_ms / TAU_ELIG_MS).exp();
        self.extras.ca_nmda   *= (-dt_ms / TAU_NMDA_CA_MS).exp();

        // BDNF decay: τ = 30s (slow consolidation window)
        const TAU_BDNF_MS: f32 = 30_000.0;
        self.bdnf *= (-dt_ms / TAU_BDNF_MS).exp();

        // ── 13. Threshold retorna ao padrão ──────────────────────────────
        let tb = self.threshold_padrao_efetivo();
        self.threshold = tb + (self.threshold - tb) * THRESHOLD_DECAY;

        // ── 14. Atualiza estado STP para o próximo tick ───────────────────
        self.extras.stp.tick(spiked, dt_ms);

        // ── 15. DA_N: rastreia hiperpolarização para RPE negativo ─────────
        // Pacemaker natural DA_N ~4 Hz → 1 spike esperado a cada 250ms.
        // Se v < −70mV por > DAN_HYPERPOL_THRESHOLD_MS sem disparar →
        // pausa inesperada → sinal de RPE negativo → mod_dopa → 0.6.
        if self.tipo == TipoNeuronal::DA_N {
            if !spiked && self.v < -70.0 {
                self.extras.dan_hyperpol_ms += dt_ms;
                if self.extras.dan_hyperpol_ms > DAN_HYPERPOL_THRESHOLD_MS {
                    // Pausa de burst esperado → RPE negativo
                    self.mod_dopa = DAN_RPE_NEG_MOD_DOPA;
                }
            } else {
                // Disparando ou saindo da hiperpolarização → limpa acumulador
                self.extras.dan_hyperpol_ms = (self.extras.dan_hyperpol_ms - dt_ms * 2.0).max(0.0);
                if spiked {
                    // Spike real → RPE neutro (restaura mod_dopa para ≥ 1.0)
                    self.mod_dopa = self.mod_dopa.max(1.0);
                }
            }
        }

        spiked
    }

    fn calcular_canais_extras(&mut self, dt_ms: f32) -> f32 {
        let v = self.v;
        use TipoNeuronalV3;
        // E_K dinâmico (Nernst temporal) — todos os canais de K⁺ respondem ao
        // acúmulo de [K⁺]o em tempo real. Hodgkin & Katz (1949).
        let e_k = self.metabolismo.e_k_dyn;

        // ── I_NaP: Na⁺ persistente ───────────────────────────────────────
        let m_nap_inf = 1.0 / (1.0 + (-(v + 52.0) / 5.0).clamp(-30.0, 30.0).exp());
        let i_nap = self.g_nap_efetivo() * m_nap_inf * (v - E_NA);

        // ── I_M: M-current (KCNQ) ────────────────────────────────────────
        let w_inf_m = 1.0 / (1.0 + (-(v + 35.0) / 10.0).clamp(-30.0, 30.0).exp());
        let tau_w = {
            let ex = ((v + 35.0) / 40.0).clamp(-20.0, 20.0).exp();
            let ey = (-(v + 35.0) / 20.0).clamp(-20.0, 20.0).exp();
            (400.0 / (3.3 * (ex + ey).max(1e-8))).clamp(5.0, 1000.0)
        };
        let g_m_eff = self.g_m_efetivo() * (1.0 - (self.extras.mod_ach - 1.0) * 0.35).clamp(0.1, 1.0);
        let decay_w = (-dt_ms / tau_w).exp();
        self.extras.w_m = w_inf_m + (self.extras.w_m - w_inf_m) * decay_w;
        self.extras.w_m = self.extras.w_m.clamp(0.0, 1.0);
        let i_m = g_m_eff * self.extras.w_m * (v - e_k);

        // ── I_A: A-type K⁺ ───────────────────────────────────────────────
        let a_inf = 1.0 / (1.0 + (-(v + 60.0) / 8.5).clamp(-30.0, 30.0).exp());
        let tau_a = {
            let ea = (-0.025 * (v + 35.0)).clamp(-20.0, 20.0).exp();
            let eb = (0.025 * (v + 79.5)).clamp(-20.0, 20.0).exp();
            (1.0 / (ea + eb).max(1e-8)).clamp(1.0, 50.0)
        };
        let b_inf = 1.0 / (1.0 + ((v + 78.0) / 6.0).clamp(-30.0, 30.0).exp());
        let tau_b = {
            let bc = ((v + 63.0) / 30.0).clamp(-20.0, 20.0).exp();
            let bd = (-(v + 63.0) / 30.0).clamp(-20.0, 20.0).exp();
            (1.0 / (0.001 * (bc + bd)).max(1e-8)).clamp(10.0, 400.0)
        };
        let decay_a = (-dt_ms / tau_a).exp();
        let decay_b = (-dt_ms / tau_b).exp();
        self.extras.a_ka = a_inf + (self.extras.a_ka - a_inf) * decay_a;
        self.extras.b_ka = b_inf + (self.extras.b_ka - b_inf) * decay_b;
        self.extras.a_ka = self.extras.a_ka.clamp(0.0, 1.0);
        self.extras.b_ka = self.extras.b_ka.clamp(0.0, 1.0);
        let i_a = self.g_a_efetivo() * self.extras.a_ka.powi(3) * self.extras.b_ka * (v - e_k);

        // ── I_BK: BK channels ────────────────────────────────────────────
        let i_bk = self.g_bk_efetivo() * self.extras.q_bk * (v - e_k);

        // ── I_T: T-type Ca²⁺ (TC e LT) ──────────────────────────────────
        let i_t = if self.g_t_efetivo() > 0.0 {
            let m_t_inf = 1.0 / (1.0 + (-(v + 57.0) / 6.2).clamp(-30.0, 30.0).exp());
            let tau_mt = {
                let c1 = (-(v + 132.0) / 16.7).clamp(-20.0, 20.0).exp();
                let c2 = ((v + 16.8) / 18.2).clamp(-20.0, 20.0).exp();
                (0.612 + 1.0 / (c1 + c2).max(1e-8)).clamp(0.5, 50.0)
            };
            let h_t_inf = 1.0 / (1.0 + ((v + 81.0) / 4.0).clamp(-30.0, 30.0).exp());
            let tau_ht_base = if v < -80.0 {
                (((v + 467.0) / 66.6).clamp(-20.0, 20.0).exp()).clamp(5.0, 500.0)
            } else {
                (28.0 + (-(v + 22.0) / 10.5).clamp(-20.0, 20.0).exp()).clamp(5.0, 500.0)
            };

            // TC + ACh: muscarínico acelera inativação de h_T → modo tônico de vigília.
            // Biológico: ACh → PKA → fosforila Cav3.x → tau_ht reduzido até 10× com ACh forte.
            let tau_ht = if self.tipo == TipoNeuronal::TC && self.extras.mod_ach > 1.2 {
                let ach_exc = (self.extras.mod_ach - 1.2).clamp(0.0, 0.8);
                // Reduz tau_ht por até 90% — h_T vai a zero (I_T inativo) rapidamente
                (tau_ht_base * (1.0 - 0.9 * ach_exc)).max(1.0)
            } else {
                tau_ht_base
            };

            let decay_mt = (-dt_ms / tau_mt).exp();
            let decay_ht = (-dt_ms / tau_ht).exp();
            self.extras.m_t = m_t_inf + (self.extras.m_t - m_t_inf) * decay_mt;
            self.extras.h_t = h_t_inf + (self.extras.h_t - h_t_inf) * decay_ht;
            self.extras.m_t = self.extras.m_t.clamp(0.0, 1.0);
            self.extras.h_t = self.extras.h_t.clamp(0.0, 1.0);
            self.g_t_efetivo() * self.extras.m_t.powi(2) * self.extras.h_t * (v - E_CA)
        } else { 0.0 };

        i_nap + i_m + i_a + i_bk + i_t
    }

    /// V4 — Metabolismo: ATP + bomba Na⁺K⁺-ATPase eletrogenica + [K⁺]o dinâmico.
    ///
    /// Chamada DUPLA por tick:
    ///   1. `atualizar_metabolismo(false, dt_ms)` antes dos substeps Izhikevich
    ///      → produz/consome ATP basal, integra bomba, retorna `i_pump`.
    ///   2. `atualizar_metabolismo(true, 0.0)` após o spike
    ///      → debita `ATP_COST_PER_SPIKE` e libera K⁺ extracelular.
    ///
    /// Retorna a corrente hiperpolarizante eletrogenica da bomba (já × HH_SCALE).
    /// PLOS CompBiol (2020): Na/K ATPase = ~75% do gasto energético neuronal.
    fn atualizar_metabolismo(&mut self, spiked: bool, dt_ms: f32) -> f32 {
        let m = &mut self.metabolismo;

        // 1. Produção mitocondrial (Michaelis-Menten do substrato)
        let prod = ATP_PROD_RATE * dt_ms * (1.0 - m.atp / ATP_MAX);

        // 2. Consumo (basal + custo de spike). ATP é f32 → clamp, NUNCA negativo.
        let custo = ATP_BASAL_COST * dt_ms
            + if spiked { ATP_COST_PER_SPIKE } else { 0.0 };
        m.atp = (m.atp + prod - custo).clamp(0.05, ATP_MAX);

        // 3. Bomba Na⁺K⁺-ATPase eletrogenica (PLOS CompBiol 2020)
        let f_atp = m.atp / (K_ATP + m.atp);
        let f_na  = (m.na_intra / (m.na_intra + 10.0)).powi(3);
        let f_ko  = (m.k_o / (m.k_o + 1.5)).powi(2);
        m.i_pump  = RHO_PUMP * f_atp * f_na * f_ko;

        // 4. [Na⁺]i — entra com spike, bomba extrui
        if spiked { m.na_intra += 1.5; }
        m.na_intra = (m.na_intra - dt_ms * 0.008 * m.i_pump).clamp(7.0, 35.0);

        // 5. [K⁺]o — liberado por spike, clearance glial (Kager et al. 2000)
        if spiked { m.k_o += KO_RELEASE_PER_SPIKE; }
        m.k_o += dt_ms * (-KO_CLEARANCE * (m.k_o - KO_REST));
        m.k_o = m.k_o.clamp(1.5, 12.0);

        // 6. E_K dinâmico (Nernst): ~−77mV repouso → ~−55mV acúmulo intenso
        m.e_k_dyn = MET_RT_F * (m.k_o.max(0.1) / MET_KI).ln();

        // 7. Corrente hiperpolarizante eletrogenica no soma
        m.i_pump * HH_SCALE * 0.5
    }

    /// V4 — Integra os compartimentos dendríticos (AIS, Tronco, Tufo Apical) e
    /// retorna a corrente líquida que chega ao soma.
    ///
    /// Chamada apenas quando `self.compartimentos.is_some()` (RS/IB).
    /// `i_apical_input`: drive top-down no tufo apical (consumido de input_apical).
    ///
    /// Rall (1967) cable theory; Larkum 1999 BAC; Kole & Stuart 2012 AIS;
    /// Schiller 2000 NMDA spike.
    fn integrar_compartimentos(&mut self, dt_ms: f32, i_apical_input: f32) -> f32 {
        let v_soma     = self.v;
        let fator_api  = self.brain_state.fator_apical();
        let comp = match self.compartimentos.as_mut() {
            Some(c) => c,
            None    => return 0.0,
        };

        // ── 1. AIS — ponto de iniciação do spike (g_Na 5×, Kole & Stuart 2012) ──
        // Portões HH padrão integrados no v_ais; alta g_Na garante que o AIS
        // dispara antes (ou junto) do soma — nunca depois.
        {
            let va = comp.v_ais;
            let n_sub = ((dt_ms / 0.1).ceil() as usize).max(1).min(20);
            let dt_sub = dt_ms / n_sub as f32;
            let (mut m, mut h, mut n) = (comp.m_ais, comp.h_ais, comp.n_ais);
            for _ in 0..n_sub {
                let am = HhV3::alpha_m(va); let bm = HhV3::beta_m(va);
                let ah = HhV3::alpha_h(va); let bh = HhV3::beta_h(va);
                let an = HhV3::alpha_n(va); let bn = HhV3::beta_n(va);
                m += dt_sub * (am * (1.0 - m) - bm * m);
                h += dt_sub * (ah * (1.0 - h) - bh * h);
                n += dt_sub * (an * (1.0 - n) - bn * n);
                m = m.clamp(0.0, 1.0); h = h.clamp(0.0, 1.0); n = n.clamp(0.0, 1.0);
            }
            comp.m_ais = m; comp.h_ais = h; comp.n_ais = n;

            let i_na_ais = G_NA_AIS * m.powi(3) * h * (va - E_NA);
            let i_k_ais  = G_K_AIS  * n.powi(4) * (va - self.metabolismo.e_k_dyn);
            let i_l_ais  = 0.3 * (va - (-65.0));
            // Acoplamento axial AIS↔soma (Rall 1967)
            let i_ais_soma = G_C_AIS * (v_soma - va);
            let dv_ais = (-(i_na_ais + i_k_ais + i_l_ais) * HH_SCALE + i_ais_soma) * dt_ms;
            comp.v_ais = (va + dv_ais).clamp(-90.0, 60.0);
            comp.ais_spiked = comp.v_ais >= -55.0;
        }

        // ── 2. BAP — Back-Propagating AP (soma → dendrito) ──────────────────────
        let (bap_trunk, bap_tuft) = if comp.bap_active {
            let amp = 0.45 * (-comp.bap_timer_ms / BAP_DECAY_TAU).exp();
            comp.bap_timer_ms += dt_ms;
            if comp.bap_timer_ms >= BAP_TOTAL_MS {
                comp.bap_active = false;
            }
            (amp * 0.70, amp * 0.42)
        } else {
            (0.0, 0.0)
        };

        // ── 3. Tufo apical — NMDA spike dendrítico (Schiller et al. 2000) ───────
        if i_apical_input > NMDA_THRESHOLD {
            // Sobe rápido (τ=3ms) em direção a 1.0
            comp.nmda_gate += (1.0 - comp.nmda_gate) * (1.0 - (-dt_ms / 3.0).exp());
            comp.nmda_spike_ms += dt_ms;
        } else {
            // Decai lento (τ=20ms)
            comp.nmda_gate *= (-dt_ms / 20.0).exp();
            comp.nmda_spike_ms = 0.0;
        }
        comp.nmda_gate = comp.nmda_gate.clamp(0.0, 1.0);
        if comp.nmda_gate > 0.5 {
            comp.ca_apical += 0.05 * dt_ms * comp.nmda_gate;
        }
        comp.ca_apical *= (-dt_ms / 80.0).exp();
        comp.ca_apical  = comp.ca_apical.clamp(0.0, 10.0);

        // I_Ca_L apical (canal L-type, ativação instantânea m_inf)
        let m_ca_inf_ap = 1.0 / (1.0 + (-(comp.v_apical + 30.0) / 6.0)
            .clamp(-30.0, 30.0).exp());
        let i_ca_apical = 2.0 * m_ca_inf_ap * (comp.v_apical - E_CA);

        // g_c_apical efetivo modulado pelo brain state
        let g_c_apical_eff = G_C_APICAL * fator_api;

        // Atualiza v_apical: acoplamento tronco↔tufo + BAP + drive NMDA + leak
        let i_tuft_trunk = g_c_apical_eff * (comp.v_trunk - comp.v_apical);
        let i_nmda_drive = comp.nmda_gate * 18.0; // despolarização do NMDA spike
        let i_leak_ap    = 0.05 * (comp.v_apical - (-70.0));
        let dv_apical = (-i_ca_apical * HH_SCALE - i_leak_ap
            + i_tuft_trunk + bap_tuft + i_nmda_drive * HH_SCALE
            + i_apical_input.max(0.0)) * dt_ms;
        comp.v_apical = (comp.v_apical + dv_apical).clamp(-90.0, 40.0);

        // ── 4. Tronco apical — Ca²⁺ hotzone (Larkum 1999) ──────────────────────
        if comp.v_trunk > -40.0 {
            comp.ca_trunk += 0.04 * dt_ms * ((comp.v_trunk + 40.0) / 30.0).clamp(0.0, 3.0);
        }
        comp.ca_trunk *= (-dt_ms / 90.0).exp();
        comp.ca_trunk  = comp.ca_trunk.clamp(0.0, 10.0);

        let m_inf_trunk = 1.0 / (1.0 + (-(comp.v_trunk + 35.0) / 5.0)
            .clamp(-30.0, 30.0).exp());
        let i_ca_trunk = comp.g_ca_trunk * m_inf_trunk * (comp.v_trunk - E_CA);

        // ── 5. Acoplamento axial bidirecional (Rall 1967) ──────────────────────
        let i_trunk_soma  = G_C_TRUNK    * (v_soma - comp.v_trunk);
        let i_trunk_tuft  = g_c_apical_eff * (comp.v_apical - comp.v_trunk);
        let i_leak_trunk  = 0.05 * (comp.v_trunk - (-68.0));
        let dv_trunk = (-i_ca_trunk * HH_SCALE - i_leak_trunk
            + i_trunk_soma + i_trunk_tuft + bap_trunk) * dt_ms;
        comp.v_trunk = (comp.v_trunk + dv_trunk).clamp(-90.0, 40.0);

        // ── 6. Coincidência dendrítica (Larkum et al. 1999) ────────────────────
        // BAP retrógrado + NMDA spike apical → BAC firing.
        // Ca trunk é calculado mas o coupling G_C_TRUNK=0.15 limita trunk a ~-43mV.
        // BAC é suprimido em NREM (fator_apical=0.3): acetilcolina e modulação são
        // necessários para amplificação dendrítica (Larkum 2013, Softky & Koch 1993).
        comp.coincidencia_ativa = comp.nmda_gate > 0.4
            && comp.bap_active
            && fator_api >= 0.8; // ativo em Vigília (1.0) e REM (1.2), não NREM (0.3)
        // Boost escala com fator_apical: NREM (0.3) suprime BAC, REM (1.2) amplifica.
        let coincidencia_boost = if comp.coincidencia_ativa {
            12.0 * 0.3 * fator_api
        } else {
            0.0
        };

        // ── 8. Drive apical independente (base do REM drive) ───────────────────
        let apical_drive = comp.nmda_gate * 2.5 * 0.4;

        // ── 9. Corrente líquida ao soma ────────────────────────────────────────
        let i_soma_from_trunk = G_C_TRUNK * (comp.v_trunk - v_soma);
        i_soma_from_trunk + coincidencia_boost + apical_drive
    }

    /// Aplica neuromodulação V3 — dopamina, serotonina, cortisol, acetilcolina.
    pub fn modular_neuro_v3(
        &mut self,
        dopamina:     f32,
        serotonina:   f32,
        cortisol:     f32,
        acetilcolina: f32,
    ) {
        // DA_N auto-regula mod_dopa via RPE no passo 15 de update().
        // Sobrescrever externamente apagaria o sinal de RPE calculado.
        if self.tipo != TipoNeuronal::DA_N {
            self.mod_dopa = dopamina;
        }
        self.mod_sero = serotonina;
        self.mod_cort = cortisol;
        self.extras.mod_ach = acetilcolina.clamp(0.0, 3.0);
        if let ModeloDinamico::IzhikevichHH(ref mut estado) = self.modelo {
            estado.modular(dopamina, serotonina, cortisol);
        }
    }

    /// Aplica neuromodulação sem ACh (compat. V2).
    pub fn modular_neuro(&mut self, dopamina: f32, serotonina: f32, cortisol: f32) {
        self.modular_neuro_v3(dopamina, serotonina, cortisol, 1.0);
    }

    // ── NTC: caminho rápido INT4 — LUT Izhikevich, sem HH/iônico/STDP peso ──
    // Usado por neurônios C0/C1 FP4. Economiza ~65% das operações por tick.
    fn ntc_update_int4(
        &mut self,
        input_current: f32,
        dt_ms:         f32,
        current_time_ms: f32,
        escala:        f32,
    ) -> bool {
        let input_q = {
            let q = (input_current / escala.max(1e-8)).round().clamp(-8.0, 7.0) as i8;
            q as f32 * escala
        };

        let lut = ntc_lut_fp4();
        let vi = ((self.v - NTC_LUT_V_MIN) / NTC_LUT_V_STEP)
            .round().clamp(0.0, 15.0) as usize;
        let ui = ((self.u - NTC_LUT_U_MIN) / NTC_LUT_U_STEP)
            .round().clamp(0.0, 15.0) as usize;
        let dv_base = lut[vi][ui] as f32 * NTC_LUT_SCALE;

        let (a, b, c, d) = self.parametros_efetivos();
        self.v = (self.v + dt_ms * (dv_base + input_q)).clamp(-100.0, 100.0);
        self.u += dt_ms * a * (b * self.v - self.u);

        let spiked = if self.v >= self.threshold {
            self.v = c;
            self.u += d;
            self.threshold += THRESHOLD_DELTA * self.tipo.fator_adaptacao();
            self.refr_count = (2.0 / dt_ms.max(0.1)).round() as u16;
            true
        } else {
            false
        };

        // Ca²⁺ AHP (SK) simplificado
        self.ca_intra *= (-dt_ms / self.tau_ca_efetivo()).exp();
        if spiked { self.ca_intra = (self.ca_intra + CA_POR_SPIKE).min(CA_MAX); }

        // BCM homeostático
        let bcm_decay = (-dt_ms / TAU_BCM_MS).exp();
        let sv = if spiked { 1.0 } else { 0.0 };
        self.activity_avg = self.activity_avg * bcm_decay + sv * (1.0 - bcm_decay);
        self.theta_m += (self.activity_avg.powi(2) - self.theta_m) * dt_ms / TAU_BCM_THETA_MS;
        self.theta_m = self.theta_m.clamp(0.001, 0.5);

        // Decaimento de traços STDP (sem atualização de peso — peso fixo em FP4)
        let decay = (-dt_ms / TAU_STDP_MS).exp();
        self.trace_pre *= decay;
        self.trace_pos *= decay;
        if spiked {
            self.trace_pos = 1.0;
            self.trace_pre = (self.trace_pre + 0.5).min(1.0);
            self.last_spike_ms = current_time_ms;
        }

        // Threshold retorna ao padrão
        let tb = self.threshold_padrao_efetivo();
        self.threshold = tb + (self.threshold - tb) * THRESHOLD_DECAY;

        spiked
    }

    // ── NTC: caminho rápido INT8 — Izhikevich direto, sem HH/iônico ─────────
    // Usado por neurônios C1/C2 INT8. Mantém STDP simplificado e STP.
    fn ntc_update_int8(
        &mut self,
        input_current: f32,
        dt_ms:         f32,
        current_time_ms: f32,
        escala:        f32,
    ) -> bool {
        let input_q = {
            let q = (input_current / escala.max(1e-8)).round().clamp(-128.0, 127.0) as i8;
            q as f32 * escala
        };

        // STP (armazena fator; compatibilidade com monitores externos)
        self.extras.stp_efficacy = self.extras.stp.fator();
        let i_eff = input_q;

        let (a, b, c, d) = self.parametros_efetivos();
        let n_sub  = (dt_ms.round() as usize).max(1);
        let dt_int = dt_ms / n_sub as f32;
        let mut spiked = false;

        for _ in 0..n_sub {
            self.v += dt_int * (0.04 * self.v.powi(2) + 5.0 * self.v + 140.0 - self.u + i_eff);
            self.u += dt_int * a * (b * self.v - self.u);
            self.v = self.v.clamp(-100.0, 100.0);
            if self.v >= self.threshold {
                self.v = c;
                self.u += d;
                self.threshold += THRESHOLD_DELTA * self.tipo.fator_adaptacao();
                self.refr_count = (2.0 / dt_int).round() as u16;
                spiked = true;
                break;
            }
        }

        // Ca²⁺ AHP (SK) + BK rápido
        self.ca_intra *= (-dt_ms / self.tau_ca_efetivo()).exp();
        self.extras.q_bk *= (-dt_ms / TAU_BK_MS).exp();
        if spiked {
            self.ca_intra = (self.ca_intra + CA_POR_SPIKE).min(CA_MAX);
            self.extras.q_bk = (self.extras.q_bk + BK_PER_SPIKE).min(1.0);
        }

        // BCM homeostático
        let bcm_decay = (-dt_ms / TAU_BCM_MS).exp();
        let sv = if spiked { 1.0 } else { 0.0 };
        self.activity_avg = self.activity_avg * bcm_decay + sv * (1.0 - bcm_decay);
        self.theta_m += (self.activity_avg.powi(2) - self.theta_m) * dt_ms / TAU_BCM_THETA_MS;
        self.theta_m = self.theta_m.clamp(0.001, 0.5);

        // STDP simplificado: traços + atualização de peso sem Ca NMDA
        let decay = (-dt_ms / TAU_STDP_MS).exp();
        self.trace_pre *= decay;
        self.trace_pos *= decay;
        if spiked {
            if self.trace_pre > 0.05 {
                let delta_ltp = LTP_RATE * self.trace_pre;
                let delta_dopa = if self.mod_dopa > 1.0 {
                    DOPA_GATE * (self.mod_dopa - 1.0).min(2.0)
                } else {
                    0.0
                };
                self.atualizar_peso(delta_ltp + delta_dopa);
            }
            self.trace_pos = 1.0;
            self.trace_pre = (self.trace_pre + 0.5).min(1.0);
            self.last_spike_ms = current_time_ms;
        }

        // Elig trace decay
        self.extras.elig_trace *= (-dt_ms / TAU_ELIG_MS).exp();
        // Ca NMDA decay
        self.extras.ca_nmda *= (-dt_ms / TAU_NMDA_CA_MS).exp();

        // Threshold retorna ao padrão
        let tb = self.threshold_padrao_efetivo();
        self.threshold = tb + (self.threshold - tb) * THRESHOLD_DECAY;

        // STP tick
        self.extras.stp.tick(spiked, dt_ms);

        spiked
    }

    fn atualizar_peso(&mut self, delta: f32) {
        match &mut self.peso {
            PesoNeuronio::FP32(v) => *v = (*v + delta).clamp(PESO_MIN, PESO_MAX),
            PesoNeuronio::FP16(v) => {
                *v = half::f16::from_f32((v.to_f32() + delta).clamp(PESO_MIN, PESO_MAX));
            }
            PesoNeuronio::INT8(v) => {
                *v = (*v as f32 + delta * 10.0).clamp(-127.0, 127.0) as i8;
            }
            PesoNeuronio::INT4(raw) => {
                let alto_bits = ((*raw >> 4) & 0x0F) as i8;
                let alto: i8 = if alto_bits & 0x08 != 0 { alto_bits | -16i8 } else { alto_bits };
                let baixo_bits = (*raw & 0x0F) as i8;
                let baixo: i8 = if baixo_bits & 0x08 != 0 { baixo_bits | -16i8 } else { baixo_bits };
                let novo = (alto as f32 + delta * 4.0).clamp(-8.0, 7.0) as i8;
                let h = (novo.max(-8).min(7) as u8) & 0x0F;
                let l = (baixo.max(-8).min(7) as u8) & 0x0F;
                *raw = (h << 4) | l;
            }
        }
    }

    pub fn peso_f32(&self, escala: f32) -> f32 {
        self.peso.valor_f32(escala)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 11 — CAMADA HÍBRIDA
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct CamadaHibrida {
    pub neuronios:          Vec<NeuronioHibrido>,
    pub escala_camada:      f32,
    pub nome:               String,
    /// PV/FS lateral: inibição de voltagem (via corrente elétrica) → soma de RS.
    pub lateral_w:          Vec<Vec<(usize, f32)>>,
    pub prev_spikes:        Vec<bool>,

    // ── V3.1: novos campos ──────────────────────────────────────────────────

    /// SST→RS: gating de plasticidade dendrítica (reduz ca_nmda + elig_trace).
    /// Construído em init_lateral_inhibition() junto com lateral_w.
    pub sst_w:              Vec<Vec<(usize, f32)>>,

    /// Pool divisivo acumulado de spikes NGF do tick anterior.
    /// Aplicado a todos os inputs como: input_efetivo = input / (1 + ngf_divisive).
    pub ngf_divisive:       f32,

    /// true quando nenhum ChIN disparou no tick anterior (pausa = janela DA-STDP aberta).
    pub chin_paused:        bool,

    /// true enquanto LC_N está em burst — RS perdem adaptação (atenção hiperfocada).
    pub lc_burst_active:    bool,

    /// ms restantes de efeito de burst LC_N (decai para 0 → lc_burst_active = false).
    pub lc_burst_remaining: f32,

    /// Astrócito local — monitora atividade e escala ca_nmda_max quando alta por >1s.
    pub astrocito:          Astrocito,

    /// V4 — Potencial extracelular local acumulado (mV) — campo ephaptic.
    /// Bidirecional: modula timing de spike ±0.5–2ms (Anastassiou et al. 2011).
    pub ephaptic_pool:      f32,
}

impl CamadaHibrida {
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
        let total: f32 = dist.iter().map(|(_, p)| p).sum();
        let dist: Vec<(PrecisionType, f32)> = dist.into_iter()
            .map(|(t, p)| (t, p / total)).collect();

        let prop_sec = tipo_secundario.map(|(_, p)| p).unwrap_or(0.0);
        let tipo_sec = tipo_secundario.map(|(t, _)| t).unwrap_or(tipo_principal);

        let mut acc_prec = 0.0f32;
        let mut pit = dist.iter().peekable();
        let (mut prec_cur, mut prob_cur) = *pit.next().expect("dist não pode ser vazia");

        let mut neuronios = Vec::with_capacity(n_neurons);
        for i in 0..n_neurons {
            let prog = i as f32 / n_neurons as f32;
            while prog > acc_prec + prob_cur {
                acc_prec += prob_cur;
                if let Some((t, p)) = pit.next() { prec_cur = *t; prob_cur = *p; } else { break; }
            }
            let tipo = if prog >= 1.0 - prop_sec { tipo_sec } else { tipo_principal };
            neuronios.push(NeuronioHibrido::new(i as u32, tipo, prec_cur));
        }

        let n = neuronios.len();
        Self {
            neuronios,
            escala_camada,
            nome: nome.to_string(),
            lateral_w:          Vec::new(),
            sst_w:              Vec::new(),
            prev_spikes:        vec![false; n],
            ngf_divisive:       0.0,
            chin_paused:        false,
            lc_burst_active:    false,
            lc_burst_remaining: 0.0,
            astrocito:          Astrocito::new(),
            ephaptic_pool:      0.0,
        }
    }

    /// V4.6 Item 3b — Microcircuito cortical biologicamente COMPLETO.
    ///
    /// Acorda a maquinaria V3.1 que estava DORMENTE: as zonas só criavam RS+FS,
    /// então SST gating, VIP disinibição, DA_N/RPE, gate ChIN e normalização NGF
    /// (todos já implementados em `update()`/`init_lateral_inhibition`) nunca eram
    /// exercitados. Este construtor cria os interneurônios que faltavam.
    ///
    /// Proporções ~neocórtex (Markram et al. 2015, "Reconstruction of Neocortical
    /// Microcircuitry"): RS 70% · FS/PV 10% · SST 8% · VIP 4% · NGF 3% · DA_N 3% · ChIN 2%.
    /// Chama `init_lateral_inhibition` internamente (PV/FS, SST→apical, VIP→SST/PV).
    pub fn nova_cortical_rica(n_neurons: usize, nome: &str, escala: f32) -> Self {
        // Base RS com a distribuição de precisão padrão; tipos reatribuídos abaixo.
        let mut c = Self::new(n_neurons, nome, TipoNeuronal::RS, None, None, escala);

        // Faixas de índice [lo, hi) → tipo. RS ocupa [0.0, 0.70) (não listado).
        let faixas = [
            (TipoNeuronal::FS,   0.70, 0.80),
            (TipoNeuronal::SST,  0.80, 0.88),
            (TipoNeuronal::VIP,  0.88, 0.92),
            (TipoNeuronal::NGF,  0.92, 0.95),
            (TipoNeuronal::DA_N, 0.95, 0.98),
            (TipoNeuronal::ChIN, 0.98, 1.00),
        ];
        let n = c.neuronios.len().max(1);
        for (i, neur) in c.neuronios.iter_mut().enumerate() {
            let prog = i as f32 / n as f32;
            for &(tipo, lo, hi) in &faixas {
                if prog >= lo && prog < hi {
                    // Reconstrói preservando id e precisão (distribuição mantida).
                    *neur = NeuronioHibrido::new(neur.id, tipo, neur.precisao);
                    break;
                }
            }
        }
        c.init_lateral_inhibition(6, 3.0);
        c
    }

    /// V4.6.1 — Conecta os interneurônios/neuromoduladores órfãos a uma zona já
    /// construída: reatribui a CAUDA da população (preserva os primeiros índices →
    /// leitura posicional/rate intacta) e MANTÉM a contagem (recebem input → de fato
    /// disparam, ao contrário de simplesmente anexar). Acorda a maquinaria V3.1
    /// dormente: DA_N→RPE, SST gating, VIP disinibição, NGF normalização.
    ///
    /// Chame ANTES de `init_lateral_inhibition` (que então cabeia PV/SST/VIP).
    /// `fracao` ∈ [0, 0.5] — proporção da população convertida (ex.: 0.18).
    pub fn enriquecer_interneuronios(&mut self, fracao: f32) {
        let n = self.neuronios.len();
        if n < 12 { return; }
        let total = ((n as f32) * fracao.clamp(0.0, 0.5)).round() as usize;
        if total == 0 { return; }
        // Mix ~Markram (proporções relativas dentro da fração convertida).
        let mix = [
            (TipoNeuronal::PV,   0.34), (TipoNeuronal::SST,  0.27),
            (TipoNeuronal::VIP,  0.14), (TipoNeuronal::NGF,  0.10),
            (TipoNeuronal::DA_N, 0.10), (TipoNeuronal::ChIN, 0.05),
        ];
        let prec = self.neuronios[n - 1].precisao;
        let mut idx = n - total; // começa na cauda
        for (tipo, p) in mix {
            let q = (((total as f32) * p).round() as usize).max(1);
            for _ in 0..q {
                if idx >= n { break; }
                let id = self.neuronios[idx].id;
                self.neuronios[idx] = NeuronioHibrido::new(id, tipo, prec);
                idx += 1;
            }
        }
    }

    /// V4.6.1 — Reatribui a cauda da população a UM tipo específico (ex.: GridCell
    /// no hipocampo/entorrinal, MirrorCell em zona pré-motora/social). Preserva os
    /// primeiros índices e a contagem (recebem input).
    pub fn reatribuir_cauda(&mut self, tipo: TipoNeuronal, fracao: f32) {
        let n = self.neuronios.len();
        if n < 8 { return; }
        let total = ((n as f32) * fracao.clamp(0.0, 0.5)).round() as usize;
        let prec = self.neuronios[n - 1].precisao;
        for idx in (n - total)..n {
            let id = self.neuronios[idx].id;
            self.neuronios[idx] = NeuronioHibrido::new(id, tipo, prec);
        }
    }

    /// V4.6.1 — Popula a cauda com VÁRIOS tipos (divididos igualmente), permitindo
    /// uma zona hospedar múltiplos tipos especializados sem sobrescrever. Preserva
    /// os primeiros índices e a contagem (recebem input → funcionam).
    pub fn popular_cauda(&mut self, tipos: &[TipoNeuronal], fracao_total: f32) {
        let n = self.neuronios.len();
        if n < 8 || tipos.is_empty() { return; }
        let total = ((n as f32) * fracao_total.clamp(0.0, 0.5)).round() as usize;
        if total == 0 { return; }
        let prec = self.neuronios[n - 1].precisao;
        let por_tipo = (total / tipos.len()).max(1);
        let mut idx = n - total;
        for &tipo in tipos {
            for _ in 0..por_tipo {
                if idx >= n { break; }
                let id = self.neuronios[idx].id;
                self.neuronios[idx] = NeuronioHibrido::new(id, tipo, prec);
                idx += 1;
            }
        }
    }

    pub fn init_lateral_inhibition(&mut self, n_vizinhos: usize, peso_inhib: f32) {
        let n = self.neuronios.len();
        self.lateral_w = vec![Vec::new(); n];
        self.sst_w     = vec![Vec::new(); n];

        let fs_idx: Vec<usize> = self.neuronios.iter().enumerate()
            .filter(|(_, n)| n.tipo == TipoNeuronal::FS || n.tipo == TipoNeuronal::PV)
            .map(|(i, _)| i).collect();
        let sst_idx: Vec<usize> = self.neuronios.iter().enumerate()
            .filter(|(_, n)| n.tipo == TipoNeuronal::SST)
            .map(|(i, _)| i).collect();
        let vip_idx: Vec<usize> = self.neuronios.iter().enumerate()
            .filter(|(_, n)| n.tipo == TipoNeuronal::VIP)
            .map(|(i, _)| i).collect();
        let rs_idx: Vec<usize> = self.neuronios.iter().enumerate()
            .filter(|(_, n)| !n.e_inibitorico_efetivo())
            .map(|(i, _)| i).collect();

        if rs_idx.is_empty() { return; }

        // ── PV/FS → RS: inibição de voltagem (corrente perisomal) ──────────
        if !fs_idx.is_empty() {
            for &fs in &fs_idx {
                let mut vizinhos: Vec<(usize, usize)> = rs_idx.iter()
                    .filter(|&&rs| rs != fs)
                    .map(|&rs| {
                        let dist = (fs as isize - rs as isize).unsigned_abs();
                        (rs, dist.min(n.saturating_sub(dist)))
                    }).collect();
                vizinhos.sort_by_key(|&(_, d)| d);
                for (rs, _) in vizinhos.into_iter().take(n_vizinhos) {
                    self.lateral_w[fs].push((rs, peso_inhib));
                }
            }
        }
        for &rs in &rs_idx {
            let prox = (rs + 1) % n;
            if !self.neuronios[prox].e_inibitorico_efetivo() {
                self.lateral_w[rs].push((prox, 0.8));
            }
        }

        // ── VIP → SST: circuito desinibitório (VIP inibe SST → libera RS) ────
        // VIP dispara → SST recebe corrente inibitória (lateral_w) → SST não dispara
        // → ca_nmda e elig_trace de RS não são gatados → janela de plasticidade aberta.
        // Biológico: VIP (disinibitory interneuron) ativa via projeções de PFC top-down.
        if !vip_idx.is_empty() && !sst_idx.is_empty() {
            for &vip in &vip_idx {
                for &sst in &sst_idx {
                    // VIP → SST com peso moderado (inibe SST perisomal, GABA-A)
                    self.lateral_w[vip].push((sst, peso_inhib * 0.8));
                }
            }
        }

        // ── SST → RS: gating de plasticidade dendrítica (sst_w) ────────────
        // SST projeta para compartimento apical → inibe ca_nmda e elig_trace.
        // SST cobre mais neurônios RS do que PV (cobertura dendrítica ampla).
        if !sst_idx.is_empty() {
            for &sst in &sst_idx {
                let mut vizinhos: Vec<(usize, usize)> = rs_idx.iter()
                    .filter(|&&rs| rs != sst)
                    .map(|&rs| {
                        let dist = (sst as isize - rs as isize).unsigned_abs();
                        (rs, dist.min(n.saturating_sub(dist)))
                    }).collect();
                vizinhos.sort_by_key(|&(_, d)| d);
                for (rs, _) in vizinhos.into_iter().take(n_vizinhos * 2) {
                    self.sst_w[sst].push((rs, peso_inhib * 0.7));
                }
            }
        }
    }

    pub fn update(&mut self, inputs: &[f32], dt: f32, t_ms: f32) -> Vec<bool> {
        let esc   = self.escala_camada;
        let n     = self.neuronios.len();
        let dt_ms = dt * 1000.0;

        // ── Pre-tick A: NGF divisive normalization ──────────────────────────
        // Pool divisivo NGF: acumula NGF_DIVISIVE_STEP por spike NGF do tick anterior,
        // decai com TAU_NGF_MS. Aplicado como: input_efetivo = input / (1 + ngf_divisive).
        let ngf_spikes = self.neuronios.iter().enumerate()
            .filter(|(i, n_)| {
                n_.tipo == TipoNeuronal::NGF
                    && self.prev_spikes.get(*i).copied().unwrap_or(false)
            })
            .count() as f32;
        self.ngf_divisive = (self.ngf_divisive * (-dt_ms / TAU_NGF_MS).exp()
            + ngf_spikes * NGF_DIVISIVE_STEP).min(5.0);

        // ── Pre-tick B: ChIN state — detecta pausa ──────────────────────────
        // ChIN pausa = nenhum ChIN disparou no tick anterior → janela DA-STDP abre.
        let chin_ativo = self.neuronios.iter().enumerate().any(|(i, n_)| {
            n_.tipo == TipoNeuronal::ChIN
                && self.prev_spikes.get(i).copied().unwrap_or(false)
        });
        self.chin_paused = !chin_ativo;

        // ── Pre-tick C: LC_N burst — reset de adaptação em RS ──────────────
        // LC_N em burst → NA → zera I_M (w_m) e AHP (ca_intra) de todos os RS.
        // Biológico: NE → β1 → cAMP → PKA → modula canais de K⁺ e Ca²⁺.
        let lc_spike = self.neuronios.iter().enumerate().any(|(i, n_)| {
            n_.tipo == TipoNeuronal::LC_N
                && self.prev_spikes.get(i).copied().unwrap_or(false)
        });
        if lc_spike {
            self.lc_burst_active    = true;
            self.lc_burst_remaining = TAU_LC_BURST_MS;
        } else {
            self.lc_burst_remaining = (self.lc_burst_remaining - dt_ms).max(0.0);
            if self.lc_burst_remaining == 0.0 { self.lc_burst_active = false; }
        }
        if self.lc_burst_active {
            for n_ in &mut self.neuronios {
                if n_.tipo == TipoNeuronal::RS {
                    // Zera adaptação: w_m (I_M) e ca_intra (AHP-SK) → atenção hiperfocada.
                    // Biológico: NE → β1 → cAMP → PKA → reduz I_M e Ca²⁺ AHP.
                    n_.extras.w_m = 0.0;
                    n_.ca_intra   = 0.0;
                    // NOTA: SNR (silenciamento de sinapses fracas) é aplicado
                    // no loop paralelo via scaling de input_div — não aqui,
                    // pois stp_efficacy seria sobrescrito por update() step 3.
                }
            }
        }

        // ── Pre-tick D: SST gating de plasticidade dendrítica ───────────────
        // SST dispara → ca_nmda e elig_trace dos RS alvo são reduzidos.
        // Biológico: SST (Martinotti) → compartimento apical → inibe NMDA via shunting.
        if !self.sst_w.is_empty() {
            for (from, targets) in self.sst_w.iter().enumerate() {
                if self.prev_spikes.get(from).copied().unwrap_or(false) {
                    for &(to, strength) in targets {
                        if to < n {
                            let n_ = &mut self.neuronios[to];
                            if !n_.e_inibitorico_efetivo() {
                                // Fecha janela de plasticidade dendrítica
                                n_.extras.ca_nmda    = (n_.extras.ca_nmda
                                    * (1.0 - SST_CA_GATE * strength)).max(0.0);
                                n_.extras.elig_trace = (n_.extras.elig_trace
                                    * (1.0 - SST_ELIG_GATE * strength)).max(0.0);
                            }
                        }
                    }
                }
            }
        }

        // ── Pre-tick E: PV/FS lateral current (inibição perisomal) ─────────
        let mut lateral_current = vec![0.0f32; n];
        if !self.lateral_w.is_empty() {
            for (from, neighbors) in self.lateral_w.iter().enumerate() {
                if self.prev_spikes.get(from).copied().unwrap_or(false) {
                    for &(to, w) in neighbors {
                        if to < n { lateral_current[to] += w; }
                    }
                }
            }
        }

        // ── Pre-tick F: Campo ephaptic (Anastassiou et al. 2011) ────────────
        // V_e_local = κ × Σ I_transmembrana(vizinhos). Bidirecional: modula
        // timing de spike ±0.5–2ms — NÃO é excitação direta.
        const KAPPA_EPHAPTIC: f32 = 0.05;
        const TAU_EPH_MS:     f32 = 2.0;

        let n_spikes_prev  = self.prev_spikes.iter().filter(|&&s| s).count() as f32;
        let v_avg_prev     = self.neuronios.iter().map(|n| n.v).sum::<f32>()
                             / self.neuronios.len().max(1) as f32;
        let i_trans_approx = n_spikes_prev * 50.0 + (v_avg_prev + 70.0).max(0.0) * 0.3;

        self.ephaptic_pool  = self.ephaptic_pool * (-dt_ms / TAU_EPH_MS).exp()
                              + KAPPA_EPHAPTIC * i_trans_approx;
        self.ephaptic_pool  = self.ephaptic_pool.clamp(-5.0, 5.0);

        // ── Parallel update: injeta estado global, roda cada neurônio ───────
        let ngf_div   = self.ngf_divisive;
        let chin_p    = self.chin_paused;
        let astro_cap = self.astrocito.ca_nmda_max();
        let lc_active = self.lc_burst_active; // captura para SNR no closure
        let eph_pool  = self.ephaptic_pool;   // campo ephaptic (capturado por cópia)

        let spikes: Vec<bool> = self.neuronios.par_iter_mut().enumerate().map(|(i, n_)| {
            let ext = inputs.get(i).copied().unwrap_or(0.0);
            let lat = lateral_current.get(i).copied().unwrap_or(0.0);
            // Normalização divisiva NGF: input_efetivo = input / (1 + pool_divisivo)
            let raw_div = (ext + lat) / (1.0 + ngf_div);
            // LC_N SNR: sinapses fracas de RS são silenciadas durante burst de NA.
            // Biológico: NE → α1 → reduz condutância de fundo → melhora relação sinal/ruído.
            // Aplicado aqui (não em Pre-tick C) porque stp_efficacy é recalculado em update().
            let base_div = if lc_active
                && n_.tipo == TipoNeuronal::RS
                && n_.extras.stp.fator() < 0.5
            {
                raw_div * 0.3 // silencia sinapse fraca — só sinapses fortes passam
            } else {
                raw_div
            };
            // V4 — Ephaptic: campo extracelular modula timing (bidirecional)
            let i_eph     = 0.12 * (eph_pool - n_.v) * HH_SCALE;
            let input_div = base_div + i_eph;
            // Injeta estado global antes do update (capturados por cópia — sem contention)
            n_.extras.chin_window_open = chin_p;
            n_.extras.ca_nmda_max      = astro_cap;
            n_.update(input_div, dt, t_ms, esc)
        }).collect();

        // ── Post-tick A: DA_N RPE broadcast ────────────────────────────────
        // Propaga mod_dopa autocomputado dos neurônios DA_N para todos os neurônios
        // excitatórios da camada. RPE⁺ → LTP nos RS; RPE⁻ → LTD invertido.
        // Biológico: VTA/SNc libera DA de forma volumétrica → modula plasticidade cortical.
        {
            let mut dan_sum = 0.0f32;
            let mut dan_count = 0usize;
            for n_ in &self.neuronios {
                if n_.tipo == TipoNeuronal::DA_N {
                    dan_sum += n_.mod_dopa;
                    dan_count += 1;
                }
            }
            if dan_count > 0 {
                let rpe = dan_sum / dan_count as f32;
                for n_ in &mut self.neuronios {
                    if matches!(n_.tipo,
                        TipoNeuronal::RS | TipoNeuronal::IB | TipoNeuronal::CH |
                        TipoNeuronal::DAP | TipoNeuronal::VIP)
                    {
                        n_.mod_dopa = rpe;
                    }
                }
            }
        }

        // ── Post-tick B: atualiza astrócito com taxa de spike atual ────────
        let spike_rate = spikes.iter().filter(|&&s| s).count() as f32 / n.max(1) as f32;
        self.astrocito.update(spike_rate, dt_ms);

        if self.prev_spikes.len() != n { self.prev_spikes = vec![false; n]; }
        self.prev_spikes.copy_from_slice(&spikes);
        spikes
    }

    /// Injeta input apical nos neurônios RS/IB antes do próximo update().
    ///
    /// Representa projeções top-down de camadas corticais superiores para
    /// o compartimento dendrítico apical de neurônios piramidais (RS/IB).
    /// Se o RS disparar E `input_apical > APICAL_THRESHOLD` → BAC burst + ca_nmda×2.
    ///
    /// Chame antes de `update()`:
    ///   `camada.set_apical_inputs(&top_down); camada.update(&bottom_up, dt, t_ms);`
    pub fn set_apical_inputs(&mut self, apical: &[f32]) {
        for (i, n) in self.neuronios.iter_mut().enumerate() {
            if matches!(n.tipo, TipoNeuronal::RS | TipoNeuronal::IB) {
                n.input_apical = apical.get(i).copied().unwrap_or(0.0);
            }
        }
    }

    pub fn modular_neuro(&mut self, dopamina: f32, serotonina: f32, cortisol: f32) {
        self.modular_neuro_v3(dopamina, serotonina, cortisol, 1.0);
    }

    pub fn modular_neuro_v3(
        &mut self,
        dopamina: f32, serotonina: f32, cortisol: f32, acetilcolina: f32,
    ) {
        for n in &mut self.neuronios {
            n.modular_neuro_v3(dopamina, serotonina, cortisol, acetilcolina);
        }
    }

    /// Item 1 — Limpeza agressiva de memória ao encerrar o ciclo de sono.
    ///
    /// 1. Zera buffers transitórios dos neurônios `Dormant`.
    /// 2. `shrink_to_fit()` devolve a capacidade excedente dos `Vec` (neurônios,
    ///    listas de adjacência laterais/SST, prev_spikes) ao alocador → ao SO.
    ///
    /// NÃO destrói pesos aprendidos nem topologia ativa. Idempotente e seguro
    /// chamar a cada despertar.
    ///
    /// ⚠️ NOTA HONESTA: os `Vec` desta camada têm tamanho ~fixo após init; o ganho
    /// aqui é marginal. A retenção de GB pós-sono observada NÃO nasce neste módulo
    /// — investigar buffers de replay em `sleep_cycle.rs` e índices em
    /// `storage/swap_manager.rs` (evicção NVMe subdimensionada).
    pub fn compactar_memoria(&mut self) {
        for n in &mut self.neuronios {
            if n.status == NeuronalStatus::Dormant {
                n.liberar_buffers_temporarios();
            }
        }
        self.neuronios.shrink_to_fit();
        self.prev_spikes.shrink_to_fit();
        for v in &mut self.lateral_w { v.shrink_to_fit(); }
        self.lateral_w.shrink_to_fit();
        for v in &mut self.sst_w { v.shrink_to_fit(); }
        self.sst_w.shrink_to_fit();
    }

    /// Item 1 — Propaga o `EstadoBrainState` a toda a camada e dispara a limpeza
    /// na transição Sono→Vigília. Liga o `EstadoBrainState` (antes usado só em
    /// testes) ao ciclo de sono real — chame a partir do gestor de sono ao mudar
    /// de fase (ex.: ao sair de `NremProfundo`/`Rem` para `Vigilia`).
    pub fn set_brain_state(&mut self, novo: EstadoBrainState) {
        let anterior = self.neuronios.first()
            .map(|n| n.brain_state)
            .unwrap_or(EstadoBrainState::Vigilia);
        let acordando = anterior != EstadoBrainState::Vigilia
            && novo == EstadoBrainState::Vigilia;

        for n in &mut self.neuronios {
            n.brain_state = novo;
        }
        if acordando {
            self.compactar_memoria();
        }
    }

    /// Item 2 — Liga/desliga a otimização 200 Hz em todos os neurônios da camada.
    /// Usado pelo teste A/B para medir a referência (sem otimização) vs produção.
    pub fn set_otimizacao(&mut self, ativa: bool) {
        for n in &mut self.neuronios {
            n.otimizar = ativa;
        }
    }

    /// V4.6 Item 6 — Implanta um neurônio na camada mantendo `prev_spikes` e as
    /// listas de adjacência (`lateral_w`, `sst_w`) em sincronia de tamanho.
    /// Usado pela neurogênese (célula-tronco) durante o sono. Devolve o índice.
    /// O novo neurônio nasce SEM conexões laterais (linhas vazias) — não perturba
    /// a topologia existente; será cabeado só se sobreviver à prova.
    pub fn adicionar_neuronio(&mut self, neuronio: NeuronioHibrido) -> usize {
        let idx = self.neuronios.len();
        self.neuronios.push(neuronio);
        self.prev_spikes.push(false);
        if !self.lateral_w.is_empty() { self.lateral_w.push(Vec::new()); }
        if !self.sst_w.is_empty()     { self.sst_w.push(Vec::new()); }
        idx
    }

    pub fn estatisticas(&self) -> CamadaStats {
        let mut s = CamadaStats::default();
        for n in &self.neuronios {
            s.total += 1;
            match n.precisao {
                PrecisionType::FP32 => s.fp32 += 1,
                PrecisionType::FP16 => s.fp16 += 1,
                PrecisionType::INT8 => s.int8 += 1,
                PrecisionType::INT4 => s.int4 += 1,
            }
            match n.tipo {
                TipoNeuronal::RS   => s.tipo_rs   += 1,
                TipoNeuronal::IB   => s.tipo_ib   += 1,
                TipoNeuronal::CH   => s.tipo_ch   += 1,
                TipoNeuronal::FS   => s.tipo_fs   += 1,
                TipoNeuronal::LT   => s.tipo_lt   += 1,
                TipoNeuronal::TC   => s.tipo_tc   += 1,
                TipoNeuronal::RZ   => s.tipo_rz   += 1,
                TipoNeuronal::PS   => s.tipo_ps   += 1,
                TipoNeuronal::PB   => s.tipo_pb   += 1,
                TipoNeuronal::AC   => s.tipo_ac   += 1,
                TipoNeuronal::BI   => s.tipo_bi   += 1,
                TipoNeuronal::DAP  => s.tipo_dap  += 1,
                TipoNeuronal::IIS  => s.tipo_iis  += 1,
                TipoNeuronal::PV   => s.tipo_pv   += 1,
                TipoNeuronal::SST  => s.tipo_sst  += 1,
                TipoNeuronal::VIP  => s.tipo_vip  += 1,
                TipoNeuronal::DA_N => s.tipo_dan  += 1,
                TipoNeuronal::NGF  => s.tipo_ngf  += 1,
                TipoNeuronal::LC_N => s.tipo_lcn  += 1,
                TipoNeuronal::ChIN => s.tipo_chin += 1,
                TipoNeuronal::GridCell   => s.tipo_grid   += 1,
                TipoNeuronal::MirrorCell => s.tipo_mirror += 1,
                TipoNeuronal::MSN        => s.tipo_msn    += 1,
                TipoNeuronal::Hybrid     => s.tipo_hybrid += 1,
            }
            if n.tipo.usa_hh() { s.hh += 1; }
            s.bytes_total += std::mem::size_of::<NeuronioHibrido>();
        }
        s
    }

    pub fn estatisticas_v3(&self) -> CamadaStatsV3 {
        let n = self.neuronios.len() as f32;
        if n == 0.0 { return CamadaStatsV3::default(); }

        let total_spikes: usize = self.prev_spikes.iter().filter(|&&s| s).count();
        let media_v = self.neuronios.iter().map(|n| n.v).sum::<f32>() / n;
        let media_w_m = self.neuronios.iter().map(|n| n.extras.w_m).sum::<f32>() / n;
        let media_ca_nmda = self.neuronios.iter().map(|n| n.extras.ca_nmda).sum::<f32>() / n;
        let media_elig = self.neuronios.iter().map(|n| n.extras.elig_trace).sum::<f32>() / n;
        let media_stp = self.neuronios.iter().map(|n| n.extras.stp_efficacy).sum::<f32>() / n;
        let n_it = self.neuronios.iter().filter(|n| n.extras.m_t > 0.1).count();

        CamadaStatsV3 {
            n_neurons:     self.neuronios.len(),
            spike_rate:    total_spikes as f32 / n,
            media_v,
            media_w_m,
            media_ca_nmda,
            media_elig,
            media_stp_efficacy: media_stp,
            n_com_it_ativo: n_it,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 12 — ESTATÍSTICAS
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
    // Izhikevich adicionais
    pub tipo_ps:     usize,
    pub tipo_pb:     usize,
    pub tipo_ac:     usize,
    pub tipo_bi:     usize,
    pub tipo_dap:    usize,
    pub tipo_iis:    usize,
    // Subtipos biológicos
    pub tipo_pv:     usize,
    pub tipo_sst:    usize,
    pub tipo_vip:    usize,
    pub tipo_dan:    usize,
    pub hh:          usize,
    // V3.1
    pub tipo_ngf:    usize,
    pub tipo_lcn:    usize,
    pub tipo_chin:   usize,
    // V4.6
    pub tipo_grid:   usize,
    pub tipo_mirror: usize,
    pub tipo_msn:    usize,
    pub tipo_hybrid: usize,
}

impl CamadaStats {
    pub fn bytes_por_neuronio(&self) -> f32 {
        if self.total == 0 { 0.0 }
        else { self.bytes_total as f32 / self.total as f32 }
    }
    pub fn prop_inibitorios(&self) -> f32 {
        if self.total == 0 { 0.0 }
        else {
            (self.tipo_fs + self.tipo_lt + self.tipo_pv
                + self.tipo_sst + self.tipo_vip + self.tipo_ngf)
                as f32 / self.total as f32
        }
    }
    pub fn prop_hh(&self) -> f32 {
        if self.total == 0 { 0.0 }
        else { self.hh as f32 / self.total as f32 }
    }
}

#[derive(Debug, Default)]
pub struct CamadaStatsV3 {
    pub n_neurons:          usize,
    pub spike_rate:         f32,
    pub media_v:            f32,
    pub media_w_m:          f32,
    pub media_ca_nmda:      f32,
    pub media_elig:         f32,
    pub media_stp_efficacy: f32,
    pub n_com_it_ativo:     usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 13 — ASTRÓCITO (célula da glia — facilitação LTP tripartite)
// ─────────────────────────────────────────────────────────────────────────────

/// Astrócito — célula da glia que monitora atividade local e facilita consolidação.
///
/// Quando a taxa de spike da camada supera ASTRO_ACTIVITY_THRESHOLD por >1000ms,
/// o astrócito libera D-serina (co-agonista do receptor NMDA NR1), ampliando o
/// limite máximo de ca_nmda de 2.0 para 4.0 (ASTRO_CA_NMDA_SCALE × base).
///
/// Biológico: Henneberger et al. (2010) — D-serine de astrócito é necessária
/// para indução de LTP hipocampal. Feedback positivo moderado: atividade alta
/// → astrócito libera D-serina → NMDA mais eficiente → mais LTP → consolidação.
#[derive(Debug)]
pub struct Astrocito {
    /// EMA da taxa de spike local (tau ≈ 100ms). Atualizado a cada tick.
    pub activity_avg: f32,
    /// Acumulador de tempo em estado de alta atividade (ms).
    pub high_activity_ms: f32,
    /// Fator de escala para ca_nmda_max [1.0 = basal, ASTRO_CA_NMDA_SCALE = ativo].
    pub ca_nmda_scale: f32,
}

impl Astrocito {
    pub fn new() -> Self {
        Self { activity_avg: 0.0, high_activity_ms: 0.0, ca_nmda_scale: 1.0 }
    }

    /// Atualiza estado do astrócito com base na taxa de spike atual da camada.
    ///
    /// - `spike_rate`: fração de neurônios que dispararam neste tick [0.0–1.0].
    /// - `dt_ms`: passo de tempo em milissegundos.
    pub fn update(&mut self, spike_rate: f32, dt_ms: f32) {
        // EMA da atividade local: tau ≈ 100ms
        let alpha = (-dt_ms / 100.0).exp();
        self.activity_avg = self.activity_avg * alpha + spike_rate * (1.0 - alpha);

        if self.activity_avg >= ASTRO_ACTIVITY_THRESHOLD {
            // Atividade persistentemente alta: acumula tempo para trigger
            self.high_activity_ms += dt_ms;
            if self.high_activity_ms >= ASTRO_HIGH_DURATION_MS {
                // Sinalização tripartite ativa: D-serina → amplifica NMDA
                // ca_nmda_max = 2.0 × ASTRO_CA_NMDA_SCALE = 4.0
                self.ca_nmda_scale = ASTRO_CA_NMDA_SCALE;
            }
        } else {
            // Atividade voltou ao normal: drena o acumulador e decai escala
            self.high_activity_ms = (self.high_activity_ms - dt_ms * 2.0).max(0.0);
            if self.high_activity_ms == 0.0 {
                // Re-captura de D-serina: decai exponencialmente (tau ≈ 500ms)
                let decay = (-dt_ms / 500.0).exp();
                self.ca_nmda_scale = 1.0 + (self.ca_nmda_scale - 1.0) * decay;
            }
        }
    }

    /// Limite máximo de ca_nmda considerando o estado atual do astrócito.
    /// Padrão: 2.0 × 1.0 = 2.0. Ativo: 2.0 × 2.0 = 4.0.
    #[inline]
    pub fn ca_nmda_max(&self) -> f32 {
        2.0 * self.ca_nmda_scale
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 14 — TESTES V4 (Neurônio Híbrido Multicompartimental)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod testes_v4 {
    use super::*;

    /// dt fixo de 1 ms (resolução fina para dinâmica de compartimentos).
    const DT: f32 = 0.001;

    fn rodar(n: &mut NeuronioHibrido, input: f32, apical: f32, n_ticks: usize) -> usize {
        let mut spikes = 0;
        for i in 0..n_ticks {
            n.input_apical = apical;
            if n.update(input, DT, i as f32, 1.0) { spikes += 1; }
        }
        spikes
    }

    #[test]
    fn rs_dispara_com_compartimentos_ativos() {
        let mut n = NeuronioHibrido::new(0, TipoNeuronal::RS, PrecisionType::FP32);
        assert!(n.compartimentos.is_some(), "RS deve ter compartimentos = Some");
        let spikes = rodar(&mut n, 22.0, 0.0, 2000);
        assert!(spikes > 0, "RS com compartimentos deve disparar (got {spikes})");
    }

    #[test]
    fn fs_dispara_sem_compartimentos_sem_regressao() {
        let mut n = NeuronioHibrido::new(0, TipoNeuronal::FS, PrecisionType::FP32);
        assert!(n.compartimentos.is_none(), "FS deve ter compartimentos = None");
        let spikes = rodar(&mut n, 22.0, 0.0, 2000);
        assert!(spikes > 0, "FS sem compartimentos deve disparar (sem regressão; got {spikes})");
    }

    #[test]
    fn atp_cai_apos_burst_e_recupera_em_500ms() {
        let mut n = NeuronioHibrido::new(0, TipoNeuronal::RS, PrecisionType::FP32);
        // Burst intenso por 200ms (drive forte → >20 spikes/s)
        let s = rodar(&mut n, 35.0, 0.0, 200);
        assert!(s >= 4, "burst deve produzir spikes (got {s})");
        let atp_baixo = n.metabolismo.atp;
        // Recuperação: 500ms sem input
        rodar(&mut n, 0.0, 0.0, 500);
        let atp_rec = n.metabolismo.atp;
        assert!(atp_rec > atp_baixo,
            "ATP deve recuperar em 500ms: rec={atp_rec} > baixo={atp_baixo}");
        assert!(n.metabolismo.atp >= 0.05, "ATP nunca negativo");
    }

    #[test]
    fn ko_sobe_apos_burst_e_volta_ao_repouso() {
        let mut n = NeuronioHibrido::new(0, TipoNeuronal::RS, PrecisionType::FP32);
        let ko0 = n.metabolismo.k_o;
        rodar(&mut n, 35.0, 0.0, 200);
        let ko_alto = n.metabolismo.k_o;
        assert!(ko_alto > ko0, "[K+]o deve subir após burst: {ko_alto} > {ko0}");
        // Clearance glial: tau ≈ 50ms → 2000ms relaxa totalmente
        rodar(&mut n, 0.0, 0.0, 2000);
        let ko_rec = n.metabolismo.k_o;
        assert!(ko_rec < ko_alto, "[K+]o deve cair via clearance glial");
        assert!((ko_rec - KO_REST).abs() < 0.5,
            "[K+]o deve voltar perto do repouso ({KO_REST}): got {ko_rec}");
    }

    #[test]
    fn coincidencia_dendritica_produz_boost() {
        // Controle: sem input apical
        let mut n_ctrl = NeuronioHibrido::new(0, TipoNeuronal::RS, PrecisionType::FP32);
        let s_ctrl = rodar(&mut n_ctrl, 12.0, 0.0, 2000);

        // Coincidência: input apical forte (NMDA spike) + spike somático (BAP).
        // nmda_gate > 0.4 em ~3ms; RS com I=12 dispara em poucos ms → 500 ticks basta.
        let mut n_api = NeuronioHibrido::new(0, TipoNeuronal::RS, PrecisionType::FP32);
        let mut coincidiu = false;
        for i in 0..2000 {
            n_api.input_apical = 3.0;
            n_api.update(12.0, DT, i as f32, 1.0);
            if let Some(c) = &n_api.compartimentos {
                if c.coincidencia_ativa { coincidiu = true; break; }
            }
        }
        assert!(coincidiu,
            "coincidência (BAP ativo + nmda_gate>0.4) deve ativar com I=12 e apical=3");
        let s_api = {
            let mut n2 = NeuronioHibrido::new(0, TipoNeuronal::RS, PrecisionType::FP32);
            rodar(&mut n2, 12.0, 3.0, 2000)
        };
        assert!(s_api > s_ctrl,
            "coincidência dendrítica deve aumentar disparo: api={s_api} > ctrl={s_ctrl}");
    }

    #[test]
    fn ais_spike_precede_ou_coincide_nunca_depois() {
        let mut n = NeuronioHibrido::new(0, TipoNeuronal::RS, PrecisionType::FP32);
        let mut ais_recente: Option<usize> = None;
        let mut violacoes = 0;
        for i in 0..3000 {
            n.input_apical = 0.0;
            let somatico = n.update(22.0, DT, i as f32, 1.0);
            let ais = n.compartimentos.as_ref()
                .map(|c| c.ais_spiked).unwrap_or(false);
            if ais { ais_recente = Some(i); }
            if somatico {
                // integrar_compartimentos roda ANTES do substep somático no
                // mesmo tick → AIS já foi avaliado. Spike somático nunca pode
                // ocorrer sem AIS ter disparado neste tick ou recentemente.
                let ok = ais || ais_recente
                    .map(|t| i.saturating_sub(t) <= 5)
                    .unwrap_or(false);
                if !ok { violacoes += 1; }
            }
        }
        assert_eq!(violacoes, 0,
            "AIS deve preceder ou coincidir com o spike somático — nunca depois");
    }

    #[test]
    fn brain_state_modula_acoplamento_apical() {
        // NREM suprime a coincidência dendrítica (fator_api=0.3 < gate 0.8).
        // Vigília (fator_api=1.0) permite BAC → coincidencia_ativa enquanto NREM bloqueia.
        let mut n_vig  = NeuronioHibrido::new(0, TipoNeuronal::RS, PrecisionType::FP32);
        let mut n_nrem = NeuronioHibrido::new(0, TipoNeuronal::RS, PrecisionType::FP32);
        n_vig.brain_state  = EstadoBrainState::Vigilia;
        n_nrem.brain_state = EstadoBrainState::NremProfundo;

        let mut coinc_vig  = 0usize;
        let mut coinc_nrem = 0usize;
        for i in 0..1500 {
            n_vig.input_apical  = 2.0;
            n_nrem.input_apical = 2.0;
            n_vig.update(10.0, DT, i as f32, 1.0);
            n_nrem.update(10.0, DT, i as f32, 1.0);
            if let Some(c) = &n_vig.compartimentos  { if c.coincidencia_ativa { coinc_vig  += 1; } }
            if let Some(c) = &n_nrem.compartimentos { if c.coincidencia_ativa { coinc_nrem += 1; } }
        }
        assert_eq!(coinc_nrem, 0,
            "NREM (fator_apical=0.3) deve suprimir coincidência: coinc_nrem={coinc_nrem}");
        assert!(coinc_vig > coinc_nrem,
            "Vigília deve ter mais coincidências que NREM: vig={coinc_vig} nrem={coinc_nrem}");
    }

    #[test]
    fn ek_dinamico_responde_a_ko() {
        let mut m = EstadoMetabolico::novo();
        m.k_o = 3.0;  m.atualizar_ek();
        let ek_repouso = m.e_k_dyn;
        m.k_o = 10.0; m.atualizar_ek();
        let ek_alto = m.e_k_dyn;
        assert!(ek_alto > ek_repouso,
            "E_K deve despolarizar com [K+]o alto: {ek_alto} > {ek_repouso}");
        // Nernst puro: E_K = 26.7 * ln(3/140) ≈ -102.6 mV (biologicamente correto;
        // o valor de -77 mV dos modelos Izhikevich é Goldman, não Nernst puro).
        assert!(ek_repouso < -95.0 && ek_repouso > -115.0,
            "E_K Nernst (~-102mV com [K+]o=3 mM): got {ek_repouso}");
    }

    #[test]
    fn camada_ephaptic_acumula_e_decai() {
        let mut c = CamadaHibrida::new(
            16, "teste_eph", TipoNeuronal::RS, None, None, 1.0,
        );
        let inputs = vec![25.0f32; 16];
        for i in 0..50 { c.update(&inputs, DT, i as f32 * 1.0); }
        let eph_ativo = c.ephaptic_pool;
        // Sem input → decai
        let zeros = vec![0.0f32; 16];
        for i in 0..200 { c.update(&zeros, DT, (50 + i) as f32); }
        let eph_decaido = c.ephaptic_pool;
        assert!(eph_ativo.abs() > 0.0, "ephaptic_pool deve acumular com atividade");
        assert!(eph_decaido.abs() < eph_ativo.abs() + 1e-3,
            "ephaptic_pool deve decair sem atividade: {eph_decaido} vs {eph_ativo}");
        assert!(eph_ativo.abs() <= 5.0, "ephaptic_pool clamp [-5,5]");
    }

    // ───────────────────────── V4.6 — Item 2: 200 Hz ─────────────────────────

    /// Teste A/B: a otimização 200 Hz (event-driven + subsampler metabólico) NÃO
    /// pode alterar significativamente a taxa de spikes emergente da rede.
    #[test]
    fn ab_otimizacao_200hz_preserva_taxa_de_spikes() {
        fn medir(otim: bool) -> usize {
            let mut c = CamadaHibrida::new(
                64, "ab", TipoNeuronal::RS, Some((TipoNeuronal::FS, 0.2)), None, 1.0,
            );
            c.set_otimizacao(otim); // per-camada → sem estado global → sem race
            let mut total = 0usize;
            for t in 0..1000 {
                // Input determinístico: ~1/3 dos neurônios excitados a cada tick.
                let inputs: Vec<f32> = (0..64)
                    .map(|i| if (i + t) % 3 == 0 { 18.0 } else { 0.0 })
                    .collect();
                total += c.update(&inputs, DT, t as f32).iter().filter(|&&s| s).count();
            }
            total
        }
        let com = medir(true);
        let sem = medir(false);

        assert!(sem > 0 && com > 0, "ambos os modos devem disparar (com={com}, sem={sem})");
        let diff = (com as f32 - sem as f32).abs() / sem as f32;
        assert!(diff < 0.10,
            "otimização 200Hz deve preservar a taxa de spikes (±10%): \
             com_otim={com}, sem_otim={sem}, diff={:.1}%", diff * 100.0);
    }

    /// Faixas biológicas: FS dispara mais que RS sob o mesmo drive (sanity check
    /// de que a heterogeneidade de frequência é real, não todos no máximo).
    #[test]
    fn fs_dispara_mais_que_rs_mesmo_drive() {
        let mut rs = NeuronioHibrido::new(0, TipoNeuronal::RS, PrecisionType::FP32);
        let mut fs = NeuronioHibrido::new(1, TipoNeuronal::FS, PrecisionType::FP32);
        let s_rs = rodar(&mut rs, 22.0, 0.0, 2000);
        let s_fs = rodar(&mut fs, 22.0, 0.0, 2000);
        assert!(s_fs > s_rs, "FS deve disparar mais que RS (fs={s_fs}, rs={s_rs})");
    }

    // ───────────────────── V4.6 — Item 4: DNA / Hybrid ───────────────────────

    #[test]
    fn dna_roundtrip_preserva_genes_do_tipo() {
        for tipo in [TipoNeuronal::RS, TipoNeuronal::FS,
                     TipoNeuronal::MSN, TipoNeuronal::GridCell] {
            let dna = tipo.extrair_dna();
            let (a, _, _, _) = tipo.parametros();
            assert_eq!(dna.a, a, "{tipo:?}: gene a deve casar");
            assert_eq!(dna.g_m, tipo.g_m(), "{tipo:?}: gene g_m deve casar");
            assert_eq!(dna.e_inibitorico, tipo.e_inibitorico(),
                "{tipo:?}: flag e_inibitorico deve casar");
        }
    }

    #[test]
    fn hibrido_le_genes_do_dna_nao_do_tipo() {
        let mut dna = TipoNeuronal::RS.extrair_dna();
        dna.g_m = 9.5;
        dna.e_inibitorico = true;
        let n = NeuronioHibrido::novo_hibrido(0, dna.clone(), PrecisionType::FP32);
        assert_eq!(n.tipo, TipoNeuronal::Hybrid);
        assert_eq!(n.g_m_efetivo(), 9.5, "hybrid deve ler g_m do DNA");
        assert!(n.e_inibitorico_efetivo(), "hybrid deve herdar e_inibitorico do DNA");
        assert_eq!(n.parametros_efetivos().0, dna.a, "hybrid deve ler a do DNA");
    }

    #[test]
    fn especie_hibrida_fica_nos_limites_e_dispara() {
        let dna = gerar_especie_hibrida(&TipoNeuronal::RS, &TipoNeuronal::FS, 0.12);
        assert!(dna.g_m >= 0.0 && dna.g_m <= 10.0, "g_m clampado: {}", dna.g_m);
        assert!(dna.a >= 0.002 && dna.a <= 1.0, "a clampado: {}", dna.a);
        let mut n = NeuronioHibrido::novo_hibrido(0, dna, PrecisionType::FP32);
        let spikes = rodar(&mut n, 25.0, 0.0, 2000);
        assert!(spikes > 0, "espécie híbrida deve ser viável e disparar (got {spikes})");
    }

    // ─────────────────── V4.6 — Item 3: novos tipos puros ────────────────────

    #[test]
    fn novos_tipos_disparam_com_drive_forte() {
        for tipo in [TipoNeuronal::GridCell, TipoNeuronal::MirrorCell, TipoNeuronal::MSN] {
            let mut n = NeuronioHibrido::new(0, tipo, PrecisionType::FP32);
            let spikes = rodar(&mut n, 45.0, 0.0, 3000);
            assert!(spikes > 0, "{tipo:?} deve disparar com drive forte (got {spikes})");
        }
    }

    // ──────────────────── V4.6 — Item 1: limpeza pós-sono ────────────────────

    #[test]
    fn set_brain_state_compacta_ao_acordar() {
        let mut c = CamadaHibrida::new(32, "sono", TipoNeuronal::RS, None, None, 1.0);
        c.set_brain_state(EstadoBrainState::NremProfundo);
        assert!(c.neuronios.iter().all(|n| n.brain_state == EstadoBrainState::NremProfundo));

        // Neurônio Dormant com buffer "sujo" deve ser limpo ao acordar.
        c.neuronios[0].status = NeuronalStatus::Dormant;
        c.neuronios[0].extras.elig_trace = 0.9;
        c.set_brain_state(EstadoBrainState::Vigilia);

        assert_eq!(c.neuronios[0].extras.elig_trace, 0.0,
            "Dormant deve ter buffers temporários limpos ao acordar");
        assert!(c.neuronios.iter().all(|n| n.brain_state == EstadoBrainState::Vigilia));
    }

    // ──────────── V4.6 — Item 3b: acordar maquinaria V3.1 dormente ────────────

    #[test]
    fn cortical_rica_cria_interneuronios_e_acorda_maquinaria() {
        let c = CamadaHibrida::nova_cortical_rica(200, "rica", 1.0);
        let tem = |t: TipoNeuronal| c.neuronios.iter().any(|n| n.tipo == t);
        // Os tipos antes órfãos agora EXISTEM na rede.
        for t in [TipoNeuronal::FS, TipoNeuronal::SST, TipoNeuronal::VIP,
                  TipoNeuronal::NGF, TipoNeuronal::DA_N, TipoNeuronal::ChIN] {
            assert!(tem(t), "microcircuito deve conter {t:?}");
        }
        // SST gating cabeado (init_lateral_inhibition preencheu sst_w).
        assert!(c.sst_w.iter().any(|v| !v.is_empty()),
            "SST→RS deve estar cabeado (gating de plasticidade ativo)");
    }

    #[test]
    fn cortical_rica_dan_modula_dopamina_dos_rs() {
        let mut c = CamadaHibrida::nova_cortical_rica(200, "rica_da", 1.0);
        // Força RPE positivo nos DA_N (input estável os manteria em 1.0).
        for n in &mut c.neuronios {
            if n.tipo == TipoNeuronal::DA_N { n.mod_dopa = 1.5; }
        }
        let inputs = vec![20.0f32; 200];
        // Um tick basta: Post-tick A propaga rpe(=média DA_N) → RS/IB/CH/DAP/VIP.
        c.update(&inputs, DT, 0.0);
        let rs_modulado = c.neuronios.iter()
            .filter(|n| n.tipo == TipoNeuronal::RS)
            .any(|n| (n.mod_dopa - 1.5).abs() < 0.2);
        assert!(rs_modulado,
            "broadcast DA_N→RS (Post-tick A) deve copiar o RPE dos DA_N para os RS \
             — prova de que a maquinaria antes morta está viva");
    }

    // ───────────── V4.6 — Genoma EXPANDIDO: neurônios estruturalmente novos ─────

    /// Cria um fenótipo que NENHUM dos 23 tipos puros é: inibidor com dinâmica
    /// Hodgkin-Huxley + dendritos multicompartimentais + sinapse facilitante.
    #[test]
    fn genoma_expandido_cria_neuronio_estruturalmente_novo() {
        let mut dna = TipoNeuronal::RS.extrair_dna();
        dna.usa_hh             = true;             // dinâmica HH (RS puro é Izhikevich)
        dna.tem_compartimentos = true;             // dendritos
        dna.tipo_stp           = TipoSTP::Facilitation;
        dna.e_inibitorico      = true;             // GABAérgico
        dna.clampar();

        let mut n = NeuronioHibrido::novo_hibrido(0, dna, PrecisionType::FP32);
        // Estruturas qualitativamente novas presentes:
        assert!(matches!(n.modelo, ModeloDinamico::IzhikevichHH(_)),
            "deve adotar dinâmica Hodgkin-Huxley");
        assert!(n.compartimentos.is_some(), "deve possuir dendritos multicompartimentais");
        assert_eq!(n.extras.stp.tipo, TipoSTP::Facilitation, "STP facilitante");
        assert!(n.e_inibitorico_efetivo(), "fenótipo inibitório");

        // E continua VIÁVEL (dispara) — não é só uma combinação válida no papel.
        let mut spikes = 0;
        for t in 0..3000 {
            if n.update(40.0, DT, t as f32, 1.0) { spikes += 1; }
        }
        assert!(spikes > 0,
            "neurônio estruturalmente novo (HH+dendritos+facilitação) deve disparar (got {spikes})");
    }

    /// O crossover/mutação explora o espaço ESTRUTURAL (não só o paramétrico):
    /// cruzando RS (Izhikevich, c/ dendritos) × TC (HH, s/ dendritos) emergem
    /// combinações estruturais distintas.
    #[test]
    fn crossover_explora_genes_estruturais() {
        use std::collections::HashSet;
        let mut combos: HashSet<(bool, bool)> = HashSet::new();
        for _ in 0..120 {
            let d = gerar_especie_hibrida(&TipoNeuronal::RS, &TipoNeuronal::TC, 0.30);
            combos.insert((d.usa_hh, d.tem_compartimentos));
        }
        assert!(combos.len() >= 2,
            "deve explorar ≥2 combinações estruturais (usa_hh, tem_compartimentos); got {}",
            combos.len());
    }

    // ──────── V4.6.1 — Conexão de tipos órfãos a zonas (enriquecimento) ────────

    #[test]
    fn enriquecer_conecta_orfaos_e_zona_continua_viva() {
        let mut c = CamadaHibrida::new(
            120, "zona", TipoNeuronal::RS, Some((TipoNeuronal::FS, 0.2)), None, 1.0,
        );
        c.enriquecer_interneuronios(0.18);
        c.init_lateral_inhibition(4, 2.5);
        // Os tipos antes órfãos agora EXISTEM na zona.
        let tem = |t: TipoNeuronal| c.neuronios.iter().any(|n| n.tipo == t);
        for t in [TipoNeuronal::DA_N, TipoNeuronal::SST, TipoNeuronal::VIP,
                  TipoNeuronal::PV, TipoNeuronal::NGF, TipoNeuronal::ChIN] {
            assert!(tem(t), "zona enriquecida deve conter {t:?}");
        }
        // A contagem é preservada (reatribuição, não anexação) → input mapeado.
        assert_eq!(c.neuronios.len(), 120, "enriquecer preserva a contagem");
        // E a zona continua a disparar (não morre nem trava) sob drive.
        let mut spikes = 0usize;
        for t in 0..400 {
            let inputs = vec![18.0f32; c.neuronios.len()];
            spikes += c.update(&inputs, 0.005, t as f32).iter().filter(|&&s| s).count();
        }
        assert!(spikes > 0, "zona enriquecida deve continuar viva (got {spikes})");
    }

    #[test]
    fn reatribuir_cauda_coloca_tipo_especifico() {
        let mut c = CamadaHibrida::new(80, "hip", TipoNeuronal::RS, None, None, 1.0);
        c.reatribuir_cauda(TipoNeuronal::GridCell, 0.15);
        let grid = c.neuronios.iter().filter(|n| n.tipo == TipoNeuronal::GridCell).count();
        assert!(grid >= 10, "deve colocar ~15% de GridCell na cauda (got {grid})");
        assert_eq!(c.neuronios.len(), 80, "contagem preservada");
    }
}
