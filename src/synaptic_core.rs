// =============================================================================
// src/synaptic_core.rs — Selene V3.0 — Modelo Unificado de Neurônio Biológico
// =============================================================================
//
// MODELO EVOLUTIVO DE 7 CAMADAS — TODOS OS MECANISMOS BIOLÓGICOS RELEVANTES:
//
//  ┌──────────────────────────────────────────────────────────────────────────┐
//  │ Camada 1 — IZHIKEVICH (todos os tipos)                                  │
//  │   dv/dt = 0.04v² + 5v + 140 − u + I_eff                                │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Camada 2 — HODGKIN-HUXLEY (TC e RZ) + I_T Ca²⁺ (TC e LT)              │
//  │   I_Na = g_Na·m³h·(v−E_Na)   I_K = g_K·n⁴·(v−E_K)   I_L              │
//  │   I_T  = g_T·m_T²·h_T·(v−E_Ca)  ← burst talâmico e LT rebound        │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Camada 3 — NOVOS CANAIS IÔNICOS (todos os tipos, escalonado por tipo)   │
//  │   I_NaP = g_NaP · m_inf(v) · (v − E_Na)  ← Na⁺ persistente, sem inat.│
//  │   I_M   = g_M · w · (v − E_K)            ← M-current (KCNQ), lento   │
//  │   I_A   = g_A · a³ · b · (v − E_K)       ← A-type K⁺, atrasa 1º spike│
//  │   I_BK  = g_BK · q_bk · (v − E_K)        ← BK channels, AHP rápido  │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Camada 4 — SHORT-TERM PLASTICITY (Tsodyks-Markram 1997)                │
//  │   x: recursos disponíveis  u: probabilidade de utilização               │
//  │   STD (RS/FS/RZ): depressão por depleção de vesículas                   │
//  │   STF (CH/LT):    facilitação por acúmulo de Ca²⁺ pré-sináptico        │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Camada 5 — Ca²⁺ DUAL: AHP (SK, adaptação) + NMDA (LTP trigger)        │
//  │   ca_intra: SK channels → AHP → adaptação de frequência                 │
//  │   ca_nmda:  NMDA Ca²⁺  → LTP → plasticidade de longo prazo             │
//  │   BK (fast AHP): q_bk surge após spike → hyperpolarização rápida       │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Camada 6 — STDP 3 FATORES (dopamina como gate de consolidação)          │
//  │   elig_trace: correlação pré-pós sem consolidação imediata              │
//  │   ΔW = η × elig_trace × (mod_dopa − 1.0).max(0)  ← só com dopamina   │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Camada 7 — ACh COMO 4º NEUROMODULADOR                                   │
//  │   ACh bloqueia I_M (KCNQ): mais disparo durante atenção                 │
//  │   ACh amplifica ca_nmda × 1.2: facilita LTP                             │
//  └──────────────────────────────────────────────────────────────────────────┘
//
// NEUROMODULAÇÃO (NeuroChem):
//   dopamina  ↑  →  g_K_mod  ↓  →  repolarização lenta  →  mais disparo
//   serotonina ↑  →  g_L_mod  ↓  →  menos vazamento      →  mais excitável
//   cortisol  ↑  →  g_Na_mod ↓  →  Na⁺ reduzido         →  limiar mais alto
//   ACh       ↑  →  I_M bloqueado                        →  atenção/LTP
//
// REFERÊNCIAS BIOLÓGICAS:
//   I_NaP: Alzheimer & ten Bruggencate (1988) — Nav1.6
//   I_M:   Adams et al. (1982) — KCNQ2/3, spike-freq adaptation
//   I_A:   Connor & Stevens (1971) — Kv4.x, delays first spike
//   I_T:   Destexhe et al. (1994) — Cav3.x, TC burst mode
//   I_BK:  Barrett et al. (1982) — fast Ca²⁺-activated K⁺
//   STP:   Tsodyks & Markram (1997) — resource depletion/facilitation
//   3-STDP: Frémaux & Gerstner (2016) — dopamine-gated eligibility
// =============================================================================

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(clippy::excessive_precision)]

use serde::{Deserialize, Serialize};
use half::f16;
use rayon::prelude::*;
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

    /// Threshold de disparo padrão (mV).
    #[inline]
    pub fn threshold_padrao(&self) -> f32 {
        match self {
            TipoNeuronal::TC => 25.0,
            TipoNeuronal::FS => 25.0,
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

    /// Constante de tempo de remoção de Ca²⁺ intracelular (ms) — específica por tipo.
    #[inline]
    pub fn tau_ca_ms(&self) -> f32 {
        match self {
            TipoNeuronal::RS => 80.0,
            TipoNeuronal::IB => 90.0,
            TipoNeuronal::CH => 120.0,
            TipoNeuronal::FS => 20.0,
            TipoNeuronal::LT => 50.0,
            TipoNeuronal::TC => 60.0,
            TipoNeuronal::RZ => 60.0,
        }
    }

    /// Atividade média alvo para plasticidade homeostática BCM.
    #[inline]
    pub fn bcm_theta(&self) -> f32 {
        match self {
            TipoNeuronal::RS => 0.10,
            TipoNeuronal::IB => 0.08,
            TipoNeuronal::CH => 0.15,
            TipoNeuronal::FS => 0.25,
            TipoNeuronal::LT => 0.07,
            TipoNeuronal::TC => 0.05,
            TipoNeuronal::RZ => 0.12,
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

/// Motor de cálculo HH base (Hodgkin & Huxley, 1952).
struct HH;

impl HH {
    #[inline]
    fn alpha_m(v: f32) -> f32 {
        let dv = v + 40.0;
        if dv.abs() < 1e-4 { 1.0 }
        else { 0.1 * dv / (1.0 - (-dv / 10.0_f32).exp()) }
    }
    #[inline] fn beta_m(v: f32) -> f32 { 4.0 * (-(v + 65.0) / 18.0_f32).exp() }
    #[inline] fn alpha_h(v: f32) -> f32 { 0.07 * (-(v + 65.0) / 20.0_f32).exp() }
    #[inline] fn beta_h(v: f32) -> f32 { 1.0 / (1.0 + (-(v + 35.0) / 10.0_f32).exp()) }
    #[inline]
    fn alpha_n(v: f32) -> f32 {
        let dv = v + 55.0;
        if dv.abs() < 1e-4 { 0.1 }
        else { 0.01 * dv / (1.0 - (-dv / 10.0_f32).exp()) }
    }
    #[inline] fn beta_n(v: f32) -> f32 { 0.125 * (-(v + 65.0) / 80.0_f32).exp() }

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

        let i_na = params.g_na * estado.g_na_mod * m.powi(3) * h * (v - params.e_na);
        let i_k  = params.g_k  * estado.g_k_mod  * n.powi(4)     * (v - params.e_k);
        let i_l  = params.g_l  * estado.g_l_mod                  * (v - params.e_l);

        let i_h = if params.g_h > 0.0 {
            let alpha_q = 0.001 * (-0.1 * (v + 75.0)).exp().min(10.0);
            let beta_q  = 0.001 * (0.1  * (v + 75.0)).exp().min(10.0);
            let q_new = estado.q_ih + dt_sub * (alpha_q * (1.0 - estado.q_ih) - beta_q * estado.q_ih);
            estado.q_ih = q_new.clamp(0.0, 1.0);
            params.g_h * estado.q_ih * (v - (-30.0))
        } else { 0.0 };

        i_na + i_k + i_l + i_h
    }
}

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
        }
    }

    fn g_t(&self) -> f32 {
        match self {
            TipoNeuronal::TC  => 8.0,
            TipoNeuronal::LT  => 10.0,
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
}

impl EstadoCanaisExtras {
    pub fn para_tipo(tipo: TipoNeuronal) -> Self {
        Self {
            w_m:          0.047,
            a_ka:         0.36,
            b_ka:         0.10,
            m_t:          0.01,
            h_t:          0.018,
            q_bk:         0.0,
            ca_nmda:      0.0,
            elig_trace:   0.0,
            mod_ach:      1.0,
            stp_efficacy: 1.0,
            stp:          SinapseSTP::para_tipo(tipo),
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
            return false;
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
        let input_stp = input_q; // input direto: sem depleção interna

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

        // ── 6. I_eff final ────────────────────────────────────────────────
        let i_eff = input_stp - (i_hh + i_extra) * HH_SCALE;

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

        // ── 11. STDP V3: 3 fatores com Ca²⁺ NMDA e eligibility trace ─────
        if spiked {
            let mg_unblock = 1.0 / (1.0 + 0.28 * (-0.062 * self.v).exp());
            if self.trace_pre > 0.05 {
                let ach_ltp_boost = if self.extras.mod_ach > 1.0 { 1.2 } else { 1.0 };
                let nmda_in = NMDA_CA_RATE * self.trace_pre * mg_unblock * ach_ltp_boost;
                self.extras.ca_nmda = (self.extras.ca_nmda + nmda_in).min(2.0);
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

            let dopa_burst = (self.mod_dopa - 1.0).max(0.0);
            let delta_dopa3 = DOPA_GATE * dopa_burst * self.extras.elig_trace;
            if dopa_burst > 0.0 {
                self.extras.elig_trace *= 1.0 - dopa_burst * 0.1;
            }

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
            let tau_ht = if v < -80.0 {
                (((v + 467.0) / 66.6).clamp(-20.0, 20.0).exp()).clamp(5.0, 500.0)
            } else {
                (28.0 + (-(v + 22.0) / 10.5).clamp(-20.0, 20.0).exp()).clamp(5.0, 500.0)
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
        self.mod_dopa = dopamina;
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
    pub neuronios:     Vec<NeuronioHibrido>,
    pub escala_camada: f32,
    pub nome:          String,
    pub lateral_w:     Vec<Vec<(usize, f32)>>,
    pub prev_spikes:   Vec<bool>,
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
            lateral_w:   Vec::new(),
            prev_spikes: vec![false; n],
        }
    }

    pub fn init_lateral_inhibition(&mut self, n_vizinhos: usize, peso_inhib: f32) {
        let n = self.neuronios.len();
        self.lateral_w = vec![Vec::new(); n];

        let fs_idx: Vec<usize> = self.neuronios.iter().enumerate()
            .filter(|(_, n)| n.tipo == TipoNeuronal::FS)
            .map(|(i, _)| i).collect();
        let rs_idx: Vec<usize> = self.neuronios.iter().enumerate()
            .filter(|(_, n)| !n.tipo.e_inibitorico())
            .map(|(i, _)| i).collect();

        if fs_idx.is_empty() || rs_idx.is_empty() { return; }

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
        for &rs in &rs_idx {
            let prox = (rs + 1) % n;
            if !self.neuronios[prox].tipo.e_inibitorico() {
                self.lateral_w[rs].push((prox, 0.8));
            }
        }
    }

    pub fn update(&mut self, inputs: &[f32], dt: f32, t_ms: f32) -> Vec<bool> {
        let esc = self.escala_camada;
        let n = self.neuronios.len();

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

        let spikes: Vec<bool> = self.neuronios.par_iter_mut().enumerate().map(|(i, n_)| {
            let ext = inputs.get(i).copied().unwrap_or(0.0);
            let lat = lateral_current.get(i).copied().unwrap_or(0.0);
            n_.update(ext + lat, dt, t_ms, esc)
        }).collect();

        if self.prev_spikes.len() != n { self.prev_spikes = vec![false; n]; }
        self.prev_spikes.copy_from_slice(&spikes);
        spikes
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
                TipoNeuronal::RS => s.tipo_rs += 1,
                TipoNeuronal::IB => s.tipo_ib += 1,
                TipoNeuronal::CH => s.tipo_ch += 1,
                TipoNeuronal::FS => s.tipo_fs += 1,
                TipoNeuronal::LT => s.tipo_lt += 1,
                TipoNeuronal::TC => s.tipo_tc += 1,
                TipoNeuronal::RZ => s.tipo_rz += 1,
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
