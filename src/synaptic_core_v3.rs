// =============================================================================
// src/synaptic_core_v3.rs — Selene V3.0 — Neurônio Biológico Completo
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
//  │   I_NaP = g_NaP · m_inf(v) · (v − E_Na)  ← Na⁺ persistente, sem inat. │
//  │   I_M   = g_M · w · (v − E_K)            ← M-current (KCNQ), lento    │
//  │   I_A   = g_A · a³ · b · (v − E_K)       ← A-type K⁺, atrasa 1º spike │
//  │   I_BK  = g_BK · q_bk · (v − E_K)        ← BK channels, AHP rápido   │
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
//  │   ΔW = η × elig_trace × (mod_dopa − 1.0).max(0)  ← só com dopamina    │
//  │   Sem dopamina: correlações acumulam mas NÃO alteram pesos              │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Camada 7 — ACh COMO 4º NEUROMODULADOR                                   │
//  │   ACh bloqueia I_M (KCNQ): mais disparo durante atenção                 │
//  │   ACh amplifica ca_nmda × 1.2: facilita LTP                             │
//  └──────────────────────────────────────────────────────────────────────────┘
//
// COMPATIBILIDADE:
//   NeuronioHibridoV3::update() assinatura idêntica à V2
//   CamadaHibridaV3::update() assinatura idêntica à V2
//   Re-usa TipoNeuronal, PrecisionType, PesoNeuronio, ModeloDinamico da V2
//
// REFERÊNCIAS BIOLÓGICAS:
//   I_NaP: Alzheimer & ten Bruggencate (1988) — Nav1.6, g≈1.5 mS/cm²
//   I_M:   Adams et al. (1982) — KCNQ2/3, spike-freq adaptation
//   I_A:   Connor & Stevens (1971) — Kv4.x, delays first spike
//   I_T:   Destexhe et al. (1994) — Cav3.x, TC burst mode
//   I_BK:  Barrett et al. (1982) — fast Ca²⁺-activated K⁺
//   STP:   Tsodyks & Markram (1997) — resource depletion/facilitation
//   3-STDP: Frémaux & Gerstner (2016) — dopamine-gated eligibility
//   NMDA-Ca²⁺: Jahr & Stevens (1990) — Mg²⁺ unblock model
// =============================================================================

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(clippy::excessive_precision)]

use serde::{Deserialize, Serialize};
use rayon::prelude::*;

// Re-usa e re-exporta tipos públicos da V2 — não duplica código existente
pub use crate::synaptic_core::{
    TipoNeuronal, PrecisionType, PesoNeuronio, Int4Par, ModeloDinamico,
    EstadoHH, ParametrosHH,
    CamadaStats,
};
use crate::config::Config;

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 1 — CONSTANTES
// ─────────────────────────────────────────────────────────────────────────────

/// Escala HH → Izhikevich (calibrada para 20–150 Hz fisiológico).
const HH_SCALE: f32 = 0.008;

/// Potenciais de reversão (mV)
const E_NA: f32 = 50.0;
const E_K:  f32 = -77.0;
const E_CA: f32 = 120.0;   // Ca²⁺ (simplificado — Nernst ≈ +135 mV)
const E_H:  f32 = -30.0;   // HCN (I_h)

// STDP (mesmos valores da V2 para consistência)
const TAU_STDP_MS:    f32 = 20.0;
const LTP_RATE:       f32 = 0.012;
const LTD_RATE:       f32 = 0.006;
const PESO_MAX:       f32 = 2.5;
const PESO_MIN:       f32 = 0.0;
const THRESHOLD_DELTA: f32 = 0.5;   // reduzido para evitar acumulação excessiva a alta freq
const THRESHOLD_DECAY: f32 = 0.985; // tau ≈ 330ms (mais rápido, evita >50mV de acumulação)

// Ca²⁺ AHP (SK channels)
const CA_POR_SPIKE: f32 = 2.0;
const CA_MAX:       f32 = 12.0;
const G_AHP:        f32 = 1.8;

// BCM homeostático
const TAU_BCM_MS: f32 = 5000.0;
const BCM_RATE:   f32 = 0.002;

// NOVOS — Plasticidade 3 fatores
/// Constante de tempo da eligibility trace (ms) — ~500ms de janela
const TAU_ELIG_MS:    f32 = 500.0;
/// Taxa de acúmulo da eligibility trace por evento STDP+NMDA
const ELIG_RATE:      f32 = 0.02;
/// Gate dopaminérgico: taxa de consolidação via eligibility × dopamina
const DOPA_GATE:      f32 = 0.008;

// NOVOS — NMDA Ca²⁺ (para gating de LTP)
/// Taxa de influxo de Ca²⁺ NMDA por evento coincidente
const NMDA_CA_RATE:    f32 = 0.4;
/// Constante de tempo de remoção do Ca²⁺ NMDA (ms) — extrusão rápida
const TAU_NMDA_CA_MS:  f32 = 50.0;

// NOVOS — BK channels (fast AHP)
/// Ativação de BK por spike (fração 0-1)
const BK_PER_SPIKE:  f32 = 0.6;
/// Constante de tempo do BK (ms) — rápido, contrasta com SK (~80ms)
const TAU_BK_MS:     f32 = 5.0;

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 2 — EXTENSÃO DE TipoNeuronal: CONDUTÂNCIAS DOS NOVOS CANAIS
// ─────────────────────────────────────────────────────────────────────────────

/// Condutâncias dos novos canais iônicos por tipo neuronal.
/// Calibradas para produzir efeitos biologicamente plausíveis com HH_SCALE=0.008.
pub trait TipoNeuronalV3 {
    /// Condutância I_NaP (Na⁺ persistente, mS/cm²).
    /// Amplifica inputs sub-limiares (~0.5 Izh units perto do threshold).
    fn g_nap(&self) -> f32;

    /// Condutância I_M (M-current, KCNQ2/3, mS/cm²).
    /// Spike-frequency adaptation. Bloqueado por ACh.
    fn g_m(&self) -> f32;

    /// Condutância I_A (A-type K⁺, Kv4.x, mS/cm²).
    /// Atrasa o primeiro spike; filtra inputs de alta frequência.
    fn g_a(&self) -> f32;

    /// Condutância I_T (T-type Ca²⁺, Cav3.x, mS/cm²).
    /// Burst talâmico e rebound em LT. Zero para outros tipos.
    fn g_t(&self) -> f32;

    /// Condutância I_BK (BK channels, fast Ca²⁺-activated K⁺, mS/cm²).
    /// AHP rápido pós-spike (tau≈5ms vs SK tau≈80ms).
    fn g_bk(&self) -> f32;
}

impl TipoNeuronalV3 for TipoNeuronal {
    fn g_nap(&self) -> f32 {
        // Base: ~1-2% do g_Na (120 mS/cm²) → biologicamente correto
        match self {
            TipoNeuronal::RS  => 1.5,   // pirâmide: amplificação moderada
            TipoNeuronal::IB  => 2.0,   // IB tem I_NaP maior → plateaus
            TipoNeuronal::CH  => 1.0,   // chattering: menos NaP
            TipoNeuronal::FS  => 0.3,   // interneurônio: mínimo
            TipoNeuronal::LT  => 1.2,
            TipoNeuronal::TC  => 0.8,   // talâmico: moderado
            TipoNeuronal::RZ  => 1.5,   // Purkinje: alta (burst rápido)
        }
    }

    fn g_m(&self) -> f32 {
        // M-current: adaptação de frequência. Valores biológicos: 2-4 mS/cm² (Adams 1982).
        // FS praticamente não adapta → g_M mínimo (~0.3).
        match self {
            TipoNeuronal::RS  => 3.0,   // pirâmide: adaptação moderada
            TipoNeuronal::IB  => 4.0,   // IB adapta mais
            TipoNeuronal::CH  => 2.0,
            TipoNeuronal::FS  => 0.3,   // FS: sem adaptação (< 10% redução)
            TipoNeuronal::LT  => 3.0,
            TipoNeuronal::TC  => 1.5,
            TipoNeuronal::RZ  => 1.0,
        }
    }

    fn g_a(&self) -> f32 {
        // I_A: transiente, controla delay do primeiro spike e timing.
        // Com b_ka=0.10 em repouso, g_A_efetivo ≈ g_A * 0.10 (inativado em repouso).
        // LT mantém alto para rebound burst característico após hiperpolarização.
        match self {
            TipoNeuronal::RS  => 8.0,   // b_ka=0.10 → efetivo ≈ 0.8 mS/cm² near threshold
            TipoNeuronal::IB  => 8.0,
            TipoNeuronal::CH  => 6.0,
            TipoNeuronal::FS  => 0.5,   // FS praticamente sem I_A (resposta instantânea)
            TipoNeuronal::LT  => 20.0,  // LT: I_A forte para rebound burst
            TipoNeuronal::TC  => 8.0,
            TipoNeuronal::RZ  => 4.0,
        }
    }

    fn g_t(&self) -> f32 {
        // T-type Ca²⁺: apenas TC (burst sono) e LT (rebound burst)
        match self {
            TipoNeuronal::TC  => 8.0,   // Destexhe (1994): g_T ≈ 2-4 mS/cm², calibrado para HH_SCALE
            TipoNeuronal::LT  => 10.0,  // LT tem I_T mais forte → rebound proeminente
            _                 => 0.0,
        }
    }

    fn g_bk(&self) -> f32 {
        // BK: AHP rápido. Valores biológicos: 1-3 mS/cm² (Barrett 1982).
        // RZ (Purkinje) tem BK forte para timing preciso de cerebeloso.
        match self {
            TipoNeuronal::RS  => 2.0,
            TipoNeuronal::IB  => 2.0,
            TipoNeuronal::CH  => 2.0,
            TipoNeuronal::FS  => 1.0,   // FS: repolarização via I_K; BK mínimo
            TipoNeuronal::LT  => 2.0,
            TipoNeuronal::TC  => 2.0,
            TipoNeuronal::RZ  => 5.0,   // Purkinje: BK forte (timing cerebelar)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 3 — HH V3: MOTOR HH INTERNO (duplicado para adicionar I_T)
// ─────────────────────────────────────────────────────────────────────────────

/// Motor HH interno da V3 — inclui I_T Ca²⁺ para TC/LT.
/// Funções α/β idênticas às originais (HH 1952).
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

    /// Integra portões HH (m,h,n,q_ih) e retorna I_hh total.
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
        let i_l  = params.g_l  * estado.g_l_mod  * (v - e_l_param(params));

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

/// Helper: extrai E_L do ParametrosHH (não é campo público na V2, usa e_l).
#[inline]
fn e_l_param(p: &ParametrosHH) -> f32 { p.e_l }

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 4 — SHORT-TERM PLASTICITY (Tsodyks-Markram 1997)
// ─────────────────────────────────────────────────────────────────────────────

/// Tipo de plasticidade de curto prazo da sinapse.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TipoSTP {
    /// Depressão: depleção de vesículas domina. RS, IB, FS, RZ.
    Depression,
    /// Facilitação: acúmulo de Ca²⁺ pré-sináptico domina. CH, LT.
    Facilitation,
    /// Misto: depressão e facilitação se equilibram. TC.
    Mixed,
}

/// Estado de plasticidade sináptica de curto prazo (Tsodyks-Markram).
///
/// `x`: recursos disponíveis (0-1). Começa em 1.0. Depleta com spikes.
/// `u_stp`: probabilidade de utilização. Sobe com spike (facilitação) ou fixo.
/// `tau_rec`: constante de recuperação (ms) — ~400-800ms para STD.
/// `tau_fac`: constante de facilitação (ms) — 0 para STD puro.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinapseSTP {
    pub x:       f32,    // recursos (0-1)
    pub u_stp:   f32,    // utilização atual
    pub u0:      f32,    // utilização basal
    tau_rec:     f32,    // recovery (ms)
    tau_fac:     f32,    // facilitation (ms), 0 = sem facilitação
    pub tipo:    TipoSTP,
}

impl SinapseSTP {
    /// Cria STP configurado para o tipo neuronal dado.
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

    /// Atualiza STP e retorna eficácia atual (produto u×x).
    ///
    /// Chamado a CADA tick. `spiked` indica se o neurônio disparou NESTE tick.
    /// A eficácia calculada será usada no PRÓXIMO tick (pré-computada).
    ///
    /// Equações Tsodyks-Markram:
    ///   Recuperação:   x += dt * (1-x) / tau_rec
    ///   Facilitação:   u_stp → u0 entre spikes
    ///   No spike:      eficácia = u*x; x -= u*x; u += u0*(1-u)
    pub fn tick(&mut self, spiked: bool, dt_ms: f32) -> f32 {
        // Recuperação de recursos entre spikes
        self.x += dt_ms * (1.0 - self.x) / self.tau_rec;
        self.x = self.x.clamp(0.0, 1.0);

        // Decaimento da facilitação (u retorna a u0)
        if self.tau_fac > 0.0 {
            self.u_stp += dt_ms * (self.u0 - self.u_stp) / self.tau_fac;
            self.u_stp = self.u_stp.clamp(0.01, 1.0);
        }

        if spiked {
            let eficacia = self.u_stp * self.x;
            // Depleção de recursos
            self.x -= self.u_stp * self.x;
            self.x = self.x.clamp(0.0, 1.0);
            // Facilitação: Ca²⁺ pré-sináptico aumenta u para próximo spike
            if self.tau_fac > 0.0 {
                self.u_stp += self.u0 * (1.0 - self.u_stp);
                self.u_stp = self.u_stp.clamp(0.0, 1.0);
            }
            eficacia
        } else {
            self.u_stp * self.x  // eficácia disponível sem spike
        }
    }

    /// Eficácia normalizada (0-1). 1.0 = sem modificação sináptica.
    /// Aplicada ao input_current antes do processamento neuronal.
    /// Para STD: começa em 1.0 (u0=0.45, x=1.0 → u0*x=0.45... mas escalonamos para 1.0)
    /// Retorna fator relativo ao estado inicial (efficacy / initial_efficacy).
    pub fn fator(&self) -> f32 {
        let efic = self.u_stp * self.x;
        let inicial = self.u0;  // eficácia inicial = u0 * 1.0
        if inicial < 1e-6 { 1.0 } else { (efic / inicial).clamp(0.05, 3.0) }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 5 — ESTADO DOS CANAIS EXTRAS (por neurônio)
// ─────────────────────────────────────────────────────────────────────────────

/// Estado dos novos canais iônicos e mecanismos de plasticidade V3.
///
/// Mantido separado do NeuronioHibrido original para clareza.
/// Armazenado em Box para evitar aumento excessivo do tamanho da struct principal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstadoCanaisExtras {
    // ── Canais iônicos ────────────────────────────────────────────────────
    /// Gate do M-current (I_M, KCNQ). Ativação lenta (tau~100ms).
    pub w_m:     f32,
    /// Gate de ativação do I_A (fast). Abre com despolarização.
    pub a_ka:    f32,
    /// Gate de inativação do I_A. Fecha lentamente em voltagens altas.
    pub b_ka:    f32,
    /// Gate de ativação do I_T (T-type Ca²⁺). Apenas para TC e LT.
    pub m_t:     f32,
    /// Gate de inativação do I_T. Deinativa em hiperpolarização profunda.
    pub h_t:     f32,
    /// Gate BK (fast AHP K⁺). Bump a cada spike, decai rápido (tau≈5ms).
    pub q_bk:    f32,

    // ── Plasticidade V3 ───────────────────────────────────────────────────
    /// Ca²⁺ derivado de NMDA — sensor de coincidência pré-pós.
    /// Acumula quando pré dispara + pós despolarizado (Mg²⁺ unblock).
    /// Tau ≈ 50ms. Gateia eligibility trace para LTP.
    pub ca_nmda:    f32,
    /// Eligibility trace (regra de 3 fatores).
    /// Acumula correlações pré-pós × ca_nmda. Não consolida sem dopamina.
    /// Tau ≈ 500ms. Multiplica mod_dopa para update sináptico.
    pub elig_trace: f32,
    /// ACh — 4º neuromodulador (0.0 = sem efeito, 1.0 = basal, 2.0 = saturação).
    pub mod_ach:    f32,

    // ── STP ───────────────────────────────────────────────────────────────
    /// Eficácia STP pré-computada para ESTE tick (calculada no tick anterior).
    /// Aplicada ao input_current antes do processamento.
    pub stp_efficacy: f32,
    /// Estado STP (Tsodyks-Markram)
    pub stp: SinapseSTP,
}

impl EstadoCanaisExtras {
    /// Estado inicial de equilíbrio para o tipo dado.
    /// I_A: b_ka começa alta (deinativado em repouso -65mV)
    /// I_T: h_t começa ~0.5 em repouso -65mV (parcialmente deinativado)
    pub fn para_tipo(tipo: TipoNeuronal) -> Self {
        // Condições iniciais = estado de equilíbrio em v = -65mV:
        //   w_inf (-65): 1/(1+exp(-(-65+35)/10)) = 1/(1+exp(3))    ≈ 0.047
        //   a_inf (-65): 1/(1+exp(-(-65+60)/8.5)) = 1/(1+exp(0.59)) ≈ 0.357 (ativação I_A)
        //   b_inf (-65): 1/(1+exp((-65+78)/6)) = 1/(1+exp(2.17))   ≈ 0.104 (INATIVADO em repouso!)
        //   h_T_inf(-65): 1/(1+exp((-65+81)/4)) = 1/(1+exp(4))     ≈ 0.018
        // IMPORTANTE: b_ka em repouso ≈ 0.10 (inativado). b_ka=0.95 só aparece
        // após hiperpolarização profunda (-80mV). Inicializar errado causa I_A excessivo.
        Self {
            w_m:          0.047,  // M-current estado de equilíbrio em -65mV
            a_ka:         0.36,   // I_A ativação steady-state em -65mV
            b_ka:         0.10,   // I_A inativação em repouso (INATIVADO, não deinativado!)
            m_t:          0.01,   // I_T ativação mínima em -65mV
            h_t:          0.018,  // I_T inativação fechada em -65mV (h_inf ≈ 0.018)
            q_bk:         0.0,    // BK fechado em repouso
            ca_nmda:      0.0,
            elig_trace:   0.0,
            mod_ach:      1.0,    // ACh basal (sem efeito adicional)
            stp_efficacy: 1.0,    // começa sem depressão/facilitação
            stp:          SinapseSTP::para_tipo(tipo),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 6 — NEURÔNIO HÍBRIDO V3
// ─────────────────────────────────────────────────────────────────────────────

/// Neurônio com 7 camadas biológicas — extensão do NeuronioHibrido V2.
///
/// Mantém todos os campos originais para compatibilidade.
/// Adiciona `extras: Box<EstadoCanaisExtras>` com novos canais e plasticidade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronioHibridoV3 {
    // ── Campos originais da V2 (manter para compatibilidade) ─────────────
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
    pub ca_intra:      f32,   // SK Ca²⁺ — AHP (adaptação)
    pub mod_dopa:      f32,
    pub mod_sero:      f32,
    pub mod_cort:      f32,
    pub activity_avg:  f32,
    pub modelo:        ModeloDinamico,

    // ── NOVO V3: estado dos novos canais e plasticidade ────────────────
    pub extras: Box<EstadoCanaisExtras>,
}

impl NeuronioHibridoV3 {
    pub fn new(id: u32, tipo: TipoNeuronal, precisao: PrecisionType) -> Self {
        let peso = match precisao {
            PrecisionType::FP32 => PesoNeuronio::FP32(1.0),
            PrecisionType::FP16 => PesoNeuronio::FP16(half::f16::from_f32(1.0)),
            PrecisionType::INT8 => PesoNeuronio::INT8(100),
            PrecisionType::INT4 => PesoNeuronio::INT4(0x77u8), // alto=7, baixo=7 packed
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

    /// Atualiza um tick. Assinatura IDÊNTICA à V2 — drop-in replacement.
    ///
    /// Adiciona sobre a V2:
    ///   - STP: input modulado por eficácia sináptica
    ///   - Novos canais: I_NaP + I_M + I_A + I_BK + I_T em i_eff
    ///   - Ca²⁺ NMDA: acumula em coincidências pré-pós
    ///   - Eligibility trace: gateado por dopamina para 3-fator STDP
    ///   - BK AHP rápido: pós-spike hiperpolarização extra
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
            // Decai extras durante refratário
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
        // STP modela a depressão/facilitação do OUTPUT desta sinapse para outros
        // neurônios — não gatea o input direto recebido pelo neurônio.
        // A camada de rede usa stp_efficacy ao propagar spikes para vizinhos.
        // (V2 também não aplica STP ao input — mantém compatibilidade.)
        self.extras.stp_efficacy = self.extras.stp.fator(); // telemetria para output
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
        // i_extra = I_NaP + I_M + I_A + I_BK + I_T (HH convention: outward+)
        // I_NaP < 0 (inward, depolariza) → subtrair aumenta i_eff
        // I_M, I_A, I_BK > 0 (outward, hiperpolariza) → subtrair reduz i_eff
        // I_T < 0 (inward Ca²⁺, depolariza) → subtrair aumenta i_eff
        let i_eff = input_stp - (i_hh + i_extra) * HH_SCALE;

        // ── 7. Substeps Izhikevich (~1 ms cada) ──────────────────────────
        let n_sub  = (dt_ms.round() as usize).max(1);
        let dt_int = dt_ms / n_sub as f32;
        let (a, b, c, d) = self.tipo.parametros();
        let mut spiked = false;

        // Threshold efetivo: neuromodulação + Ca²⁺ AHP
        let neuro_thresh_offset = -(self.mod_dopa - 1.0) * 2.0
                                  + self.mod_cort * 4.5
                                  - (self.mod_sero - 1.0) * 0.8
                                  - (self.extras.mod_ach - 1.0) * 1.5; // ACh ↓ threshold
        // FS (parvalbumin): AHP de Ca²⁺ mínimo — não adapta biologicamente.
        // FS tem tau_ca=20ms mas sensibilidade SK muito menor que RS/IB.
        let g_ahp_scale = if self.tipo == TipoNeuronal::FS { 0.1 } else { 1.0 };
        let ahp_extra = G_AHP * self.ca_intra * g_ahp_scale;
        let threshold_efetivo = self.threshold + neuro_thresh_offset + ahp_extra;

        for _ in 0..n_sub {
            self.v += dt_int * (0.04 * self.v.powi(2) + 5.0 * self.v + 140.0 - self.u + i_eff);
            self.u += dt_int * a * (b * self.v - self.u);
            self.v = self.v.clamp(-100.0, 100.0); // Ca²⁺ AHP pode elevar threshold > 50

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
            // BK: bump imediato por spike → AHP rápido no próximo step
            self.extras.q_bk = (self.extras.q_bk + BK_PER_SPIKE).min(1.0);
        }
        // Serotonina acelera remoção do Ca²⁺ AHP
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
            // Ca²⁺ NMDA: Mg²⁺ desbloqueia com despolarização.
            // Coincidência: pré disparou recentemente (trace_pre > 0) E pós despolarizado.
            // mg_unblock aumenta com v mais positivo (remove bloqueio de Mg²⁺).
            let mg_unblock = 1.0 / (1.0 + 0.28 * (-0.062 * self.v).exp());
            if self.trace_pre > 0.05 {
                let ach_ltp_boost = if self.extras.mod_ach > 1.0 { 1.2 } else { 1.0 };
                let nmda_in = NMDA_CA_RATE * self.trace_pre * mg_unblock * ach_ltp_boost;
                self.extras.ca_nmda = (self.extras.ca_nmda + nmda_in).min(2.0);
            }

            // BCM: escala LTP/LTD baseado na atividade histórica
            let bcm_theta = self.tipo.bcm_theta();
            let bcm_mod = if self.activity_avg > bcm_theta {
                let excess = (self.activity_avg - bcm_theta) / bcm_theta.max(0.01);
                1.0 - BCM_RATE * excess.min(5.0)
            } else {
                let deficit = (bcm_theta - self.activity_avg) / bcm_theta.max(0.01);
                1.0 + BCM_RATE * deficit.min(5.0)
            };

            // Eligibility trace: acumula correlação pré-pós × NMDA Ca²⁺
            // Não consolida diretamente — precisa de dopamina
            let elig_bump = ELIG_RATE * self.extras.ca_nmda * bcm_mod.max(0.0);
            self.extras.elig_trace = (self.extras.elig_trace + elig_bump).min(1.0);

            // STDP padrão (correlação temporal direta, V2-compatível)
            let hz_atual = 1000.0 / dt_ms;
            let ltd_threshold = crate::config::janela_stdp_atual(hz_atual);
            let delta_ltp = LTP_RATE * self.trace_pre * bcm_mod.max(0.1);
            let delta_ltd = if self.trace_pre < ltd_threshold {
                -LTD_RATE * (1.0 - self.trace_pre) / bcm_mod.max(0.1)
            } else { 0.0 };

            // 3º fator — dopamina consolida a eligibility trace:
            // Se dopamina acima do basal → delta_w adicional proporcional a elig_trace
            // Isso implementa a regra de 3 fatores: Δw = elig × dopa_burst
            let dopa_burst = (self.mod_dopa - 1.0).max(0.0);
            let delta_dopa3 = DOPA_GATE * dopa_burst * self.extras.elig_trace;
            // Degradação parcial da eligibility após consolidação
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
        // tick() modifica x e u_stp (depleção/facilitação). O fator normalizado
        // será lido no início do próximo tick via fator().
        self.extras.stp.tick(spiked, dt_ms);

        spiked
    }

    /// Calcula correntes dos novos canais iônicos. Atualiza gates internamente.
    ///
    /// Retorna soma total em HH convention (outward positive, µA/cm²):
    ///   I_NaP < 0 (inward Na⁺, depolariza)
    ///   I_M   > 0 (outward K⁺, hiperpolariza)
    ///   I_A   > 0 (outward K⁺, atrasa disparos)
    ///   I_BK  > 0 (outward K⁺, AHP rápido)
    ///   I_T   < 0 (inward Ca²⁺, burst trigger para TC/LT)
    /// Calcula correntes dos novos canais iônicos. Atualiza gates internamente.
    ///
    /// INTEGRAÇÃO EXPONENCIAL: x(t+dt) = x_inf + (x - x_inf) * exp(-dt/tau)
    /// Evita o overflow de Euler quando dt >> tau (ex: dt=5ms, tau_a=1ms).
    /// Numericamente estável para qualquer relação dt/tau.
    fn calcular_canais_extras(&mut self, dt_ms: f32) -> f32 {
        let v = self.v;
        use TipoNeuronalV3; // traz o trait para escopo

        // ── I_NaP: Na⁺ persistente — sem inativação, m_inf instantâneo ──
        // Ativa entre -65 e -50mV → amplifica inputs sub-limiares
        let m_nap_inf = 1.0 / (1.0 + (-(v + 52.0) / 5.0).clamp(-30.0, 30.0).exp());
        let i_nap = self.tipo.g_nap() * m_nap_inf * (v - E_NA);

        // ── I_M: M-current (KCNQ) — K⁺ lento, tau ~100ms ────────────────
        // ACh bloqueia (multiplicador < 1.0)
        let w_inf_m = 1.0 / (1.0 + (-(v + 35.0) / 10.0).clamp(-30.0, 30.0).exp());
        let tau_w = {
            let ex = ((v + 35.0) / 40.0).clamp(-20.0, 20.0).exp();
            let ey = (-(v + 35.0) / 20.0).clamp(-20.0, 20.0).exp();
            (400.0 / (3.3 * (ex + ey).max(1e-8))).clamp(5.0, 1000.0)
        };
        let g_m_eff = self.tipo.g_m() * (1.0 - (self.extras.mod_ach - 1.0) * 0.35).clamp(0.1, 1.0);
        // Integração exponencial — estável para dt >> tau
        let decay_w = (-dt_ms / tau_w).exp();
        self.extras.w_m = w_inf_m + (self.extras.w_m - w_inf_m) * decay_w;
        self.extras.w_m = self.extras.w_m.clamp(0.0, 1.0);
        let i_m = g_m_eff * self.extras.w_m * (v - E_K);

        // ── I_A: A-type K⁺ — transiente, atrasa primeiro spike ──────────
        // Papel: em repouso b_ka≈0.10 (inativado), I_A≈mínimo.
        // Só é relevante após hiperpolarização profunda (b_ka≈0.90 deinativado).
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
        // Integração exponencial — crítica para tau_a~1ms com dt=5ms
        let decay_a = (-dt_ms / tau_a).exp();
        let decay_b = (-dt_ms / tau_b).exp();
        self.extras.a_ka = a_inf + (self.extras.a_ka - a_inf) * decay_a;
        self.extras.b_ka = b_inf + (self.extras.b_ka - b_inf) * decay_b;
        self.extras.a_ka = self.extras.a_ka.clamp(0.0, 1.0);
        self.extras.b_ka = self.extras.b_ka.clamp(0.0, 1.0);
        let i_a = self.tipo.g_a() * self.extras.a_ka.powi(3) * self.extras.b_ka * (v - E_K);

        // ── I_BK: BK channels — K⁺ ativado por Ca²⁺ + voltagem ──────────
        // q_bk é bumped no update() após spike; aqui só calculamos a corrente
        let i_bk = self.tipo.g_bk() * self.extras.q_bk * (v - E_K);

        // ── I_T: T-type Ca²⁺ — apenas TC e LT (g_t > 0) ─────────────────
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

    /// Aplica neuromodulação V3 — adiciona ACh ao set dopamina/serotonina/cortisol.
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
        // HH: modula condutâncias iônicas (mesmo comportamento da V2)
        if let ModeloDinamico::IzhikevichHH(ref mut estado) = self.modelo {
            estado.modular(dopamina, serotonina, cortisol);
        }
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
                // Extrai nibble alto com extensão de sinal (sem aceder campo privado de Int4Par)
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

    /// Peso atual como f32.
    pub fn peso_f32(&self, escala: f32) -> f32 {
        self.peso.valor_f32(escala)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 7 — CAMADA HÍBRIDA V3
// ─────────────────────────────────────────────────────────────────────────────

/// Camada de neurônios V3 com full-stack biológico.
/// API idêntica à CamadaHibrida V2 — drop-in replacement.
#[derive(Debug)]
pub struct CamadaHibridaV3 {
    pub neuronios:     Vec<NeuronioHibridoV3>,
    pub escala_camada: f32,
    pub nome:          String,
    pub lateral_w:     Vec<Vec<(usize, f32)>>,
    pub prev_spikes:   Vec<bool>,
}

impl CamadaHibridaV3 {
    /// Cria camada V3. Assinatura idêntica à CamadaHibrida::new().
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
            neuronios.push(NeuronioHibridoV3::new(i as u32, tipo, prec_cur));
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

    /// Inicializa inibição lateral FS→RS. Idêntico à V2.
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

    /// Processa um tick. Assinatura idêntica à V2.
    pub fn update(&mut self, inputs: &[f32], dt: f32, t_ms: f32) -> Vec<bool> {
        let esc = self.escala_camada;
        let n = self.neuronios.len();

        // Correntes laterais dos spikes do tick anterior
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

    /// Aplica neuromodulação V3 (com ACh) a todos os neurônios.
    /// Compatibilidade V2: modula sem ACh (acetilcolina=1.0 = neutro).
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

    /// Estatísticas da camada.
    /// Compatibilidade V2: retorna `CamadaStats` com os mesmos campos que a V2.
    /// Para as estatísticas estendidas V3, use `estatisticas_v3()`.
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
            s.bytes_total += std::mem::size_of::<NeuronioHibridoV3>();
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
// SEÇÃO 8 — ESTATÍSTICAS
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct CamadaStatsV3 {
    pub n_neurons:         usize,
    pub spike_rate:        f32,   // fração de neurônios que dispararam
    pub media_v:           f32,   // potencial médio
    pub media_w_m:         f32,   // gate M-current médio (adaptação)
    pub media_ca_nmda:     f32,   // Ca²⁺ NMDA médio (LTP ativo)
    pub media_elig:        f32,   // eligibility trace médio
    pub media_stp_efficacy: f32,  // eficácia STP média
    pub n_com_it_ativo:    usize, // neurônios com I_T ativo (burst TC/LT)
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 9 — COMPATIBILIDADE V2 (aliases e re-exports)
// ─────────────────────────────────────────────────────────────────────────────

/// Alias V2-compat: NeuronioHibrido → NeuronioHibridoV3
pub type NeuronioHibrido = NeuronioHibridoV3;
/// Alias V2-compat: CamadaHibrida → CamadaHibridaV3
pub type CamadaHibrida = CamadaHibridaV3;

// Tipos de base (TipoNeuronal, PrecisionType, etc.) já são re-exportados via
// `pub use crate::synaptic_core::{...}` no topo deste arquivo.
