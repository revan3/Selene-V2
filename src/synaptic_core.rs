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
        }
    }

    /// Threshold de disparo padrão (mV).
    #[inline]
    pub fn threshold_padrao(&self) -> f32 {
        match self {
            TipoNeuronal::TC  => 25.0,
            TipoNeuronal::FS  => 25.0,
            TipoNeuronal::LC_N=> 25.0,  // burst fácil
            _                 => 30.0,
        }
    }

    /// Verdadeiro para tipos GABAérgicos (usados na inibição lateral).
    #[inline]
    pub fn e_inibitorico(&self) -> bool {
        matches!(self,
            TipoNeuronal::FS  | TipoNeuronal::LT |
            TipoNeuronal::PV  | TipoNeuronal::SST | TipoNeuronal::VIP |
            TipoNeuronal::NGF)  // NGF: GABA volumétrico (inibição divisiva)
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
const TAU_BCM_MS: f32 = 5000.0;
const BCM_RATE:   f32 = 0.002;

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
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TipoSTP {
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
        }
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
            let tb = self.tipo.threshold_padrao();
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

        // ── 4. Correntes HH (TC e RZ) ────────────────────────────────────
        let i_hh = if let ModeloDinamico::IzhikevichHH(ref mut estado) = self.modelo {
            let params = self.tipo.parametros_hh()
                .unwrap_or_else(|| unreachable!("TC/RZ garantem ParametrosHH"));
            HhV3::integrar(estado, &params, self.v, dt_ms)
        } else {
            0.0
        };

        // ── 5. Novos canais iônicos ──────────────────────────────────────
        let i_extra = self.calcular_canais_extras(dt_ms);

        // ── 6. I_eff base ─────────────────────────────────────────────────
        let mut i_eff = input_stp - (i_hh + i_extra) * HH_SCALE;

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
        let (a, b, c, d) = self.tipo.parametros();
        let mut spiked = false;

        let neuro_thresh_offset = -(self.mod_dopa - 1.0) * 2.0
                                  + self.mod_cort * 4.5
                                  - (self.mod_sero - 1.0) * 0.8
                                  - (self.extras.mod_ach - 1.0) * 1.5;
        let g_ahp_scale = if self.tipo == TipoNeuronal::FS { 0.1 } else { 1.0 };
        let ahp_extra = G_AHP * self.ca_intra * g_ahp_scale;
        let threshold_efetivo = self.threshold + neuro_thresh_offset + ahp_extra;

        for _ in 0..n_sub {
            self.v += dt_int * (0.04 * self.v.powi(2) + 5.0 * self.v + 140.0 - self.u + i_eff);
            self.u += dt_int * a * (b * self.v - self.u);
            self.v = self.v.clamp(-100.0, 100.0);

            if self.v >= threshold_efetivo {
                self.v = c;
                self.u += d;
                self.threshold += THRESHOLD_DELTA;
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
        self.input_apical = 0.0; // one-shot: consumido a cada tick

        // ── 8. Ca²⁺ AHP (SK) + BK rápido pós-spike ──────────────────────
        let ca_decay = (-dt_ms / self.tipo.tau_ca_ms()).exp();
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

            let bcm_theta = self.tipo.bcm_theta();
            let bcm_mod = if self.activity_avg > bcm_theta {
                let excess = (self.activity_avg - bcm_theta) / bcm_theta.max(0.01);
                1.0 - BCM_RATE * excess.min(5.0)
            } else {
                let deficit = (bcm_theta - self.activity_avg) / bcm_theta.max(0.01);
                1.0 + BCM_RATE * deficit.min(5.0)
            };

            let elig_bump = ELIG_RATE * self.extras.ca_nmda * bcm_mod.max(0.0);
            self.extras.elig_trace = (self.extras.elig_trace + elig_bump).min(1.0);

            let hz_atual = 1000.0 / dt_ms;
            let ltd_threshold = crate::config::janela_stdp_atual(hz_atual);
            let delta_ltp = LTP_RATE * self.trace_pre * bcm_mod.max(0.1);
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

        // ── 13. Threshold retorna ao padrão ──────────────────────────────
        let tb = self.tipo.threshold_padrao();
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

        // ── I_NaP: Na⁺ persistente ───────────────────────────────────────
        let m_nap_inf = 1.0 / (1.0 + (-(v + 52.0) / 5.0).clamp(-30.0, 30.0).exp());
        let i_nap = self.tipo.g_nap() * m_nap_inf * (v - E_NA);

        // ── I_M: M-current (KCNQ) ────────────────────────────────────────
        let w_inf_m = 1.0 / (1.0 + (-(v + 35.0) / 10.0).clamp(-30.0, 30.0).exp());
        let tau_w = {
            let ex = ((v + 35.0) / 40.0).clamp(-20.0, 20.0).exp();
            let ey = (-(v + 35.0) / 20.0).clamp(-20.0, 20.0).exp();
            (400.0 / (3.3 * (ex + ey).max(1e-8))).clamp(5.0, 1000.0)
        };
        let g_m_eff = self.tipo.g_m() * (1.0 - (self.extras.mod_ach - 1.0) * 0.35).clamp(0.1, 1.0);
        let decay_w = (-dt_ms / tau_w).exp();
        self.extras.w_m = w_inf_m + (self.extras.w_m - w_inf_m) * decay_w;
        self.extras.w_m = self.extras.w_m.clamp(0.0, 1.0);
        let i_m = g_m_eff * self.extras.w_m * (v - E_K);

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
        let i_a = self.tipo.g_a() * self.extras.a_ka.powi(3) * self.extras.b_ka * (v - E_K);

        // ── I_BK: BK channels ────────────────────────────────────────────
        let i_bk = self.tipo.g_bk() * self.extras.q_bk * (v - E_K);

        // ── I_T: T-type Ca²⁺ (TC e LT) ──────────────────────────────────
        let i_t = if self.tipo.g_t() > 0.0 {
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
            self.tipo.g_t() * self.extras.m_t.powi(2) * self.extras.h_t * (v - E_CA)
        } else { 0.0 };

        i_nap + i_m + i_a + i_bk + i_t
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

        let (a, b, c, d) = self.tipo.parametros();
        self.v = (self.v + dt_ms * (dv_base + input_q)).clamp(-100.0, 100.0);
        self.u += dt_ms * a * (b * self.v - self.u);

        let spiked = if self.v >= self.threshold {
            self.v = c;
            self.u += d;
            self.threshold += THRESHOLD_DELTA;
            self.refr_count = (2.0 / dt_ms.max(0.1)).round() as u16;
            true
        } else {
            false
        };

        // Ca²⁺ AHP (SK) simplificado
        self.ca_intra *= (-dt_ms / self.tipo.tau_ca_ms()).exp();
        if spiked { self.ca_intra = (self.ca_intra + CA_POR_SPIKE).min(CA_MAX); }

        // BCM homeostático
        let bcm_decay = (-dt_ms / TAU_BCM_MS).exp();
        let sv = if spiked { 1.0 } else { 0.0 };
        self.activity_avg = self.activity_avg * bcm_decay + sv * (1.0 - bcm_decay);

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
        let tb = self.tipo.threshold_padrao();
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

        let (a, b, c, d) = self.tipo.parametros();
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
                self.threshold += THRESHOLD_DELTA;
                self.refr_count = (2.0 / dt_int).round() as u16;
                spiked = true;
                break;
            }
        }

        // Ca²⁺ AHP (SK) + BK rápido
        self.ca_intra *= (-dt_ms / self.tipo.tau_ca_ms()).exp();
        self.extras.q_bk *= (-dt_ms / TAU_BK_MS).exp();
        if spiked {
            self.ca_intra = (self.ca_intra + CA_POR_SPIKE).min(CA_MAX);
            self.extras.q_bk = (self.extras.q_bk + BK_PER_SPIKE).min(1.0);
        }

        // BCM homeostático
        let bcm_decay = (-dt_ms / TAU_BCM_MS).exp();
        let sv = if spiked { 1.0 } else { 0.0 };
        self.activity_avg = self.activity_avg * bcm_decay + sv * (1.0 - bcm_decay);

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
        let tb = self.tipo.threshold_padrao();
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
            .filter(|(_, n)| !n.tipo.e_inibitorico())
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
            if !self.neuronios[prox].tipo.e_inibitorico() {
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
                            if !n_.tipo.e_inibitorico() {
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

        // ── Parallel update: injeta estado global, roda cada neurônio ───────
        let ngf_div   = self.ngf_divisive;
        let chin_p    = self.chin_paused;
        let astro_cap = self.astrocito.ca_nmda_max();
        let lc_active = self.lc_burst_active; // captura para SNR no closure

        let spikes: Vec<bool> = self.neuronios.par_iter_mut().enumerate().map(|(i, n_)| {
            let ext = inputs.get(i).copied().unwrap_or(0.0);
            let lat = lateral_current.get(i).copied().unwrap_or(0.0);
            // Normalização divisiva NGF: input_efetivo = input / (1 + pool_divisivo)
            let raw_div = (ext + lat) / (1.0 + ngf_div);
            // LC_N SNR: sinapses fracas de RS são silenciadas durante burst de NA.
            // Biológico: NE → α1 → reduz condutância de fundo → melhora relação sinal/ruído.
            // Aplicado aqui (não em Pre-tick C) porque stp_efficacy é recalculado em update().
            let input_div = if lc_active
                && n_.tipo == TipoNeuronal::RS
                && n_.extras.stp.fator() < 0.5
            {
                raw_div * 0.3 // silencia sinapse fraca — só sinapses fortes passam
            } else {
                raw_div
            };
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
