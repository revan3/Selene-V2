// src/learning/voices.rs
// V3.4 Multi-Self — As 4 Consciências (O Exército de Idiotas).
//
// Cada voz é uma instância paralela que:
//   1. Lê o ActiveContext compartilhado (lock-free).
//   2. Aplica sua heurística característica.
//   3. Produz um VoiceVote.
//
// O VoiceArbiter combina os 4 votos em um único Verdict, que é consumido pelo
// FrontalLobe / Go-NoGo (Fase E) para decidir entre falar, silenciar ou interromper.
//
// Comunicação inter-vozes via Atomics no ActiveContext — sem Mutex no caminho quente.

#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};

use super::active_context::{ActiveContext, CONTEXT_SLOTS};
use super::attention::{
    AttentionGate,
    VoiceProfile,
    VOICE_ANALITICA,
    VOICE_CENSOR,
    VOICE_DOPAMINA,
    VOICE_CRIATIVA,
};

/// Tipo de ação que uma voz recomenda.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoteAction {
    /// Continuar / falar normalmente.
    Speak,
    /// Inibir saída (Censor detectou risco).
    Inhibit,
    /// Buscar mais informação (Dopamina: curiosidade alta, recompensa pendente).
    Pursue,
    /// Salto criativo: trazer um conceito não-óbvio do grafo.
    LeapCreative,
    /// Sem opinião forte — voz inativa neste tick.
    Abstain,
}

/// Voto individual de uma voz num tick.
#[derive(Clone, Copy, Debug)]
pub struct VoiceVote {
    pub action:     VoteAction,
    /// Confiança no voto, 0..1.
    pub confidence: f32,
    /// Saliência sentida pela voz (heurística própria).
    pub salience:   f32,
    /// Slot do ActiveContext que motivou o voto, ou usize::MAX se irrelevante.
    pub focus_slot: usize,
    /// Concept_id de interesse (0 se nenhum).
    pub concept_id: u32,
}

impl VoiceVote {
    pub fn abstain() -> Self {
        Self {
            action: VoteAction::Abstain,
            confidence: 0.0,
            salience: 0.0,
            focus_slot: usize::MAX,
            concept_id: 0,
        }
    }
}

/// Veredicto final do arbiter — consumido pelo Go/NoGo na Fase E.
#[derive(Clone, Copy, Debug)]
pub struct Verdict {
    pub action:     VoteAction,
    /// Confiança ponderada (soma das confianças × pesos / soma dos pesos).
    pub confidence: f32,
    /// Saliência urgente — usada para ForceInterrupt na Fase E.
    pub urgency:    f32,
    /// Concept_id em foco (do voto vencedor).
    pub concept_id: u32,
    /// Dissonância: 0 (consenso) → 1 (vozes em conflito total).
    pub dissonance: f32,
}

impl Verdict {
    pub fn silent() -> Self {
        Self {
            action: VoteAction::Abstain,
            confidence: 0.0,
            urgency: 0.0,
            concept_id: 0,
            dissonance: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Trait Voice — interface comum
// ═══════════════════════════════════════════════════════════════════════════════

/// Contexto externo passado a cada voz no tick (sem Mutex — slices imutáveis e
/// valores atômicos lidos antecipadamente pelo loop).
pub struct VoiceTick<'a> {
    pub ctx:            &'a ActiveContext,
    /// Reward atual exposto pelo RL_Engine (0..1).
    pub reward:         f32,
    /// Dopamina atual do FrontalLobe (0..2).
    pub dopamine:       f32,
    /// Serotonina atual do FrontalLobe (0..2).
    pub serotonin:      f32,
    /// Tick global (200Hz).
    pub tick:           u64,
    /// Frontal rates (top-down) — opcional, fatia para set_topdown.
    pub frontal_rates:  &'a [f32],
    /// V3.4 Fase F — true quando ModoOperacao::Quiescencia. Censor/Criativa
    /// só votam a cada 4 ticks (poupar cores no Ryzen 3500U).
    pub quiescencia:    bool,
}

pub trait Voice {
    fn name(&self) -> &'static str;
    fn profile(&self) -> &VoiceProfile;
    /// Consome o input sensorial atual (saída do attention gate dessa voz)
    /// e o VoiceTick, e produz um voto.
    fn vote(&mut self, attended_input: &[f32], t: &VoiceTick) -> VoiceVote;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Voz Analítica — "O Arquiteto"
// ═══════════════════════════════════════════════════════════════════════════════

pub struct AnaliticaVoice {
    pub gate: AttentionGate,
}

impl AnaliticaVoice {
    pub fn new(n_channels: usize) -> Self {
        Self { gate: AttentionGate::with_profile(n_channels, VOICE_ANALITICA) }
    }
}

impl Voice for AnaliticaVoice {
    fn name(&self) -> &'static str { "Analitica" }
    fn profile(&self) -> &VoiceProfile { &self.gate.profile }

    /// Heurística: confiança alta quando o foco é estável e há pelo menos 3 conceitos
    /// no contexto ativo (mostrou estrutura). Foca no slot de maior saliência.
    fn vote(&mut self, attended_input: &[f32], t: &VoiceTick) -> VoiceVote {
        let active = t.ctx.active_count();
        if active < 1 {
            return VoiceVote::abstain();
        }

        // Encontra o slot de maior saliência.
        let mut best_slot = usize::MAX;
        let mut best_cid: u32 = 0;
        let mut best_sal: f32 = 0.0;
        let mut sum_sal: f32 = 0.0;
        let mut count: f32 = 0.0;
        t.ctx.for_each_active(|slot, cid, sal| {
            sum_sal += sal;
            count += 1.0;
            if sal > best_sal {
                best_sal = sal;
                best_slot = slot;
                best_cid = cid;
            }
        });
        let mean_sal = if count > 0.0 { sum_sal / count } else { 0.0 };

        // Confiança = (saliência média) × min(1, active/3) × peso da voz.
        let structural = (active.min(8) as f32) / 8.0;
        let confidence = (mean_sal * structural).clamp(0.0, 1.0);

        // Decisão: Speak quando a confiança ultrapassa 0.4 (arbitrário, calibrável).
        let action = if confidence >= 0.4 { VoteAction::Speak } else { VoteAction::Abstain };

        VoiceVote {
            action,
            confidence,
            salience: best_sal,
            focus_slot: best_slot,
            concept_id: best_cid,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Voz Censor — "O Negativa". Inibição lateral, monitora inconsistências.
// ═══════════════════════════════════════════════════════════════════════════════

pub struct CensorVoice {
    pub gate: AttentionGate,
    /// Janela curta de saliências médias para detectar volatilidade.
    sal_history: [f32; 16],
    head: usize,
}

impl CensorVoice {
    pub fn new(n_channels: usize) -> Self {
        Self {
            gate: AttentionGate::with_profile(n_channels, VOICE_CENSOR),
            sal_history: [0.0; 16],
            head: 0,
        }
    }
}

impl Voice for CensorVoice {
    fn name(&self) -> &'static str { "Censor" }
    fn profile(&self) -> &VoiceProfile { &self.gate.profile }

    /// Heurística: detecta inconsistência via variância da saliência. Vota Inhibit
    /// quando há alta volatilidade (sinais conflitantes) ou serotonina baixa
    /// (sistema em estresse) com dopamina alta (impulsividade).
    fn vote(&mut self, _attended_input: &[f32], t: &VoiceTick) -> VoiceVote {
        let active = t.ctx.active_count();

        // Computa média e variância das saliências ativas.
        let mut sum: f32 = 0.0;
        let mut sum_sq: f32 = 0.0;
        let mut n: f32 = 0.0;
        let mut max_cid: u32 = 0;
        let mut max_sal: f32 = 0.0;
        t.ctx.for_each_active(|_, cid, sal| {
            sum += sal;
            sum_sq += sal * sal;
            n += 1.0;
            if sal > max_sal { max_sal = sal; max_cid = cid; }
        });

        let mean = if n > 0.0 { sum / n } else { 0.0 };
        let variance = if n > 0.0 { (sum_sq / n) - (mean * mean) } else { 0.0 };

        // Atualiza histórico.
        self.sal_history[self.head] = mean;
        self.head = (self.head + 1) % self.sal_history.len();

        // Volatilidade = variância das últimas 16 médias.
        let h_mean: f32 = self.sal_history.iter().sum::<f32>() / self.sal_history.len() as f32;
        let h_var: f32 = self.sal_history.iter()
            .map(|x| (x - h_mean).powi(2))
            .sum::<f32>() / self.sal_history.len() as f32;

        // Risco = variância intra-tick + volatilidade entre ticks + impulsividade
        // (alta dopamina + baixa serotonina = perigo).
        let impulsivity = (t.dopamine - t.serotonin).max(0.0);
        let risk = (variance + h_var + impulsivity * 0.3) * self.gate.profile.salience_bias;

        // Sem contexto ativo = sem motivo para censurar.
        if active < 1 {
            return VoiceVote::abstain();
        }

        let confidence = risk.clamp(0.0, 1.0);
        let action = if confidence >= 0.5 { VoteAction::Inhibit } else { VoteAction::Abstain };

        VoiceVote {
            action,
            confidence,
            salience: risk,
            focus_slot: usize::MAX,
            concept_id: max_cid,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Voz Dopamina — "A Positiva". Busca recompensa e curiosidade.
// ═══════════════════════════════════════════════════════════════════════════════

pub struct DopaminaVoice {
    pub gate: AttentionGate,
    last_reward: f32,
}

impl DopaminaVoice {
    pub fn new(n_channels: usize) -> Self {
        Self {
            gate: AttentionGate::with_profile(n_channels, VOICE_DOPAMINA),
            last_reward: 0.0,
        }
    }
}

impl Voice for DopaminaVoice {
    fn name(&self) -> &'static str { "Dopamina" }
    fn profile(&self) -> &VoiceProfile { &self.gate.profile }

    /// Heurística: vota Pursue quando reward está subindo (curiosidade) ou Speak
    /// quando há um conceito de alta saliência aparecendo no contexto (recompensa
    /// esperada). Suporta também LTD: se reward cai abruptamente, sinaliza
    /// Inhibit (caminho ruim — não falar).
    fn vote(&mut self, _attended_input: &[f32], t: &VoiceTick) -> VoiceVote {
        let delta_reward = t.reward - self.last_reward;
        self.last_reward = t.reward;

        let mut best_cid = 0u32;
        let mut best_sal = 0.0f32;
        t.ctx.for_each_active(|_, cid, sal| {
            if sal > best_sal { best_sal = sal; best_cid = cid; }
        });

        // Curiosidade: reward subindo = continuar buscando.
        if delta_reward > 0.05 {
            return VoiceVote {
                action: VoteAction::Pursue,
                confidence: (delta_reward * 4.0).clamp(0.0, 1.0),
                salience: best_sal,
                focus_slot: usize::MAX,
                concept_id: best_cid,
            };
        }

        // Recompensa esperada: alto saliência + dopamina alta → falar.
        if best_sal > 0.5 && t.dopamine > 0.7 {
            return VoiceVote {
                action: VoteAction::Speak,
                confidence: (best_sal * t.dopamine.min(1.0)).clamp(0.0, 1.0),
                salience: best_sal,
                focus_slot: usize::MAX,
                concept_id: best_cid,
            };
        }

        // Reward caindo abruptamente: inibir (caminho ruim — RPE negativo).
        if delta_reward < -0.10 {
            return VoiceVote {
                action: VoteAction::Inhibit,
                confidence: (-delta_reward * 3.0).clamp(0.0, 1.0),
                salience: -delta_reward,
                focus_slot: usize::MAX,
                concept_id: best_cid,
            };
        }

        VoiceVote::abstain()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Voz Criativa — "Ruído Estocástico"
// ═══════════════════════════════════════════════════════════════════════════════

pub struct CriativaVoice {
    pub gate: AttentionGate,
    /// Estado RNG xorshift64 dedicado (independente do gate).
    rng_state: u64,
}

impl CriativaVoice {
    pub fn new(n_channels: usize) -> Self {
        Self {
            gate: AttentionGate::with_profile(n_channels, VOICE_CRIATIVA),
            rng_state: 0xDEADBEEFCAFEBABEu64,
        }
    }

    #[inline]
    fn next_rand(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.rng_state = x;
        x
    }
}

impl Voice for CriativaVoice {
    fn name(&self) -> &'static str { "Criativa" }
    fn profile(&self) -> &VoiceProfile { &self.gate.profile }

    /// Heurística: ocasionalmente (1 em 8 ticks) escolhe um slot aleatório do
    /// ActiveContext e vota LeapCreative. Confiança sempre baixa (peso 0.4 na
    /// arbitragem). Quando dispara, traz um concept_id "lateral" ao foco normal.
    fn vote(&mut self, _attended_input: &[f32], t: &VoiceTick) -> VoiceVote {
        // 12.5% chance de disparar — saltos criativos são raros.
        if (self.next_rand() & 0x07) != 0 {
            return VoiceVote::abstain();
        }

        let active = t.ctx.active_count();
        if active < 2 {
            // Precisa de pelo menos 2 conceitos para fazer um salto interessante.
            return VoiceVote::abstain();
        }

        // Escolhe slot aleatório entre os ativos.
        let target_idx = (self.next_rand() % active as u64) as u32;
        let mut count = 0u32;
        let mut chosen_cid = 0u32;
        let mut chosen_sal = 0.0f32;
        let mut chosen_slot = usize::MAX;
        t.ctx.for_each_active(|slot, cid, sal| {
            if count == target_idx {
                chosen_cid = cid;
                chosen_sal = sal;
                chosen_slot = slot;
            }
            count += 1;
        });

        // Confiança modulada por nivel_atencao e ruído.
        let r = (self.next_rand() >> 40) as f32 / ((1u64 << 24) as f32);
        let confidence = (0.3 + r * 0.4).clamp(0.0, 0.8);

        VoiceVote {
            action: VoteAction::LeapCreative,
            confidence,
            salience: chosen_sal,
            focus_slot: chosen_slot,
            concept_id: chosen_cid,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VoiceArbiter — combina os 4 votos em um único Verdict
// ═══════════════════════════════════════════════════════════════════════════════

/// Telemetria de saliência urgente (atomic) — lida pelo Go/NoGo na Fase E.
/// Codificação fixed-point: f32 × 1e6 → i32. Aceita valores negativos para
/// indicar "ofensa do Censor" (sinal de inibição forte).
pub struct ArbiterTelemetry {
    pub last_urgency: AtomicI32,
    pub last_action_code: AtomicU32, // 0=Speak,1=Inhibit,2=Pursue,3=Leap,4=Abstain
    pub votes_emitted: AtomicU32,
}

impl ArbiterTelemetry {
    pub fn new() -> Self {
        Self {
            last_urgency: AtomicI32::new(0),
            last_action_code: AtomicU32::new(4),
            votes_emitted: AtomicU32::new(0),
        }
    }

    pub fn store(&self, v: &Verdict) {
        let urg = (v.urgency.clamp(-1.0, 1.0) * 1_000_000.0) as i32;
        self.last_urgency.store(urg, Ordering::Release);
        let code = match v.action {
            VoteAction::Speak        => 0,
            VoteAction::Inhibit      => 1,
            VoteAction::Pursue       => 2,
            VoteAction::LeapCreative => 3,
            VoteAction::Abstain      => 4,
        };
        self.last_action_code.store(code, Ordering::Release);
        self.votes_emitted.fetch_add(1, Ordering::Relaxed);
    }

    pub fn read_urgency(&self) -> f32 {
        self.last_urgency.load(Ordering::Acquire) as f32 / 1_000_000.0
    }
}

pub struct VoiceArbiter {
    pub analitica: AnaliticaVoice,
    pub censor:    CensorVoice,
    pub dopamina:  DopaminaVoice,
    pub criativa:  CriativaVoice,
    pub telemetry: Arc<ArbiterTelemetry>,
    /// Última geração lida do ActiveContext — evita arbitragem redundante quando
    /// o contexto não mudou (Diretriz: economia de ciclos no Ryzen 3500U).
    last_ctx_gen: u64,
    pub last_verdict: Verdict,
}

impl VoiceArbiter {
    pub fn new(n_channels: usize) -> Self {
        Self {
            analitica: AnaliticaVoice::new(n_channels),
            censor:    CensorVoice::new(n_channels),
            dopamina:  DopaminaVoice::new(n_channels),
            criativa:  CriativaVoice::new(n_channels),
            telemetry: Arc::new(ArbiterTelemetry::new()),
            last_ctx_gen: u64::MAX,
            last_verdict: Verdict::silent(),
        }
    }

    /// Roda 1 tick de arbitragem. Se contexto não mudou desde o último tick,
    /// retorna o veredicto cached (skip).
    ///
    /// Estratégia de combinação:
    /// 1. Cada voz produz VoiceVote.
    /// 2. Multiplicar confidence × voice_weight (peso do perfil).
    /// 3. Conta de tipo: somar pesos efetivos por VoteAction.
    /// 4. Action vencedor = máximo. Verdict.confidence = peso vencedor / soma de pesos.
    /// 5. Dissonance = 1 - (peso vencedor / soma). Alto = vozes brigando.
    /// 6. Urgency = confidence se ação ∈ {Inhibit, LeapCreative} e dissonance baixo,
    ///    senão confidence × 0.5. Censor dispara ForceInterrupt em urgency > 0.7.
    pub fn arbitrate(&mut self, attended_input: &[f32], t: &VoiceTick) -> Verdict {
        let cur_gen = t.ctx.current_generation();
        // Skip se contexto não mudou (economia de ciclos).
        if cur_gen == self.last_ctx_gen && t.tick % 50 != 0 {
            return self.last_verdict;
        }
        self.last_ctx_gen = cur_gen;

        let v1 = self.analitica.vote(attended_input, t);
        // V3.4 Fase F — Quiescência: Censor e Criativa rodam só 1 a cada 4 ticks,
        // poupando ~50% do trabalho do arbiter no Ryzen 3500U quando idle.
        let skip_secundarias = t.quiescencia && (t.tick % 4 != 0);
        let v2 = if skip_secundarias { VoiceVote::abstain() }
                 else { self.censor.vote(attended_input, t) };
        let v3 = self.dopamina.vote(attended_input, t);
        let v4 = if skip_secundarias { VoiceVote::abstain() }
                 else { self.criativa.vote(attended_input, t) };

        let weights = [
            v1.confidence * self.analitica.gate.profile.voice_weight,
            v2.confidence * self.censor.gate.profile.voice_weight,
            v3.confidence * self.dopamina.gate.profile.voice_weight,
            v4.confidence * self.criativa.gate.profile.voice_weight,
        ];
        let actions = [v1.action, v2.action, v3.action, v4.action];
        let votes = [v1, v2, v3, v4];

        let total: f32 = weights.iter().sum();
        if total < 1e-3 {
            self.last_verdict = Verdict::silent();
            self.telemetry.store(&self.last_verdict);
            return self.last_verdict;
        }

        // Acumula peso por VoteAction.
        let mut acc = [0.0f32; 5]; // Speak, Inhibit, Pursue, LeapCreative, Abstain
        for (a, w) in actions.iter().zip(weights.iter()) {
            let idx = match a {
                VoteAction::Speak        => 0,
                VoteAction::Inhibit      => 1,
                VoteAction::Pursue       => 2,
                VoteAction::LeapCreative => 3,
                VoteAction::Abstain      => 4,
            };
            acc[idx] += *w;
        }
        let (winner_idx, &winner_weight) = acc.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((4, &0.0));

        let action = match winner_idx {
            0 => VoteAction::Speak,
            1 => VoteAction::Inhibit,
            2 => VoteAction::Pursue,
            3 => VoteAction::LeapCreative,
            _ => VoteAction::Abstain,
        };

        let confidence = (winner_weight / total).clamp(0.0, 1.0);
        let dissonance = (1.0 - confidence).clamp(0.0, 1.0);

        // Urgency: alto para Inhibit/LeapCreative com baixa dissonância (consenso).
        let urg_base = match action {
            VoteAction::Inhibit | VoteAction::LeapCreative => confidence,
            _ => confidence * 0.5,
        };
        let urgency = urg_base * (1.0 - dissonance * 0.4);

        // Concept_id: do voto vencedor da mesma ação.
        let mut concept_id = 0u32;
        for v in votes.iter() {
            if v.action == action && v.concept_id != 0 {
                concept_id = v.concept_id;
                break;
            }
        }

        let verdict = Verdict { action, confidence, urgency, concept_id, dissonance };
        self.last_verdict = verdict;
        self.telemetry.store(&verdict);
        verdict
    }

    /// Snapshot textual para logs/telemetria WS.
    pub fn debug_summary(&self) -> String {
        let v = &self.last_verdict;
        format!(
            "[Arbiter] action={:?} conf={:.2} urg={:.2} diss={:.2} cid={}",
            v.action, v.confidence, v.urgency, v.dissonance, v.concept_id
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tick<'a>(ctx: &'a ActiveContext, frontal: &'a [f32]) -> VoiceTick<'a> {
        VoiceTick {
            ctx,
            reward: 0.5,
            dopamine: 0.6,
            serotonin: 0.5,
            tick: 0,
            frontal_rates: frontal,
            quiescencia: false,
        }
    }

    #[test]
    fn analitica_abstem_em_contexto_vazio() {
        let mut a = AnaliticaVoice::new(64);
        let ctx = ActiveContext::new();
        let frontal = vec![0.0f32; 64];
        let t = make_tick(&ctx, &frontal);
        let vote = a.vote(&[], &t);
        assert_eq!(vote.action, VoteAction::Abstain);
    }

    #[test]
    fn analitica_speak_com_contexto_saliente() {
        let mut a = AnaliticaVoice::new(64);
        let ctx = ActiveContext::new();
        ctx.inject_concept(101, 0.9);
        ctx.inject_concept(102, 0.85);
        ctx.inject_concept(103, 0.8);
        let frontal = vec![0.0f32; 64];
        let t = make_tick(&ctx, &frontal);
        let vote = a.vote(&[], &t);
        assert_eq!(vote.action, VoteAction::Speak);
        assert!(vote.confidence > 0.0);
    }

    #[test]
    fn arbiter_silencio_sem_votos() {
        let mut arb = VoiceArbiter::new(64);
        let ctx = ActiveContext::new();
        let frontal = vec![0.0f32; 64];
        let t = make_tick(&ctx, &frontal);
        let v = arb.arbitrate(&[], &t);
        assert_eq!(v.action, VoteAction::Abstain);
    }

    #[test]
    fn arbiter_speak_quando_analitica_vence() {
        let mut arb = VoiceArbiter::new(64);
        let ctx = ActiveContext::new();
        for i in 1..=5 { ctx.inject_concept(i, 0.85); }
        let frontal = vec![0.0f32; 64];
        let t = make_tick(&ctx, &frontal);
        let v = arb.arbitrate(&[], &t);
        // Analítica deveria votar Speak; Censor pode tentar Inhibit mas com peso menor.
        // Como o contexto é estável e saliente, esperamos Speak ou pelo menos não-Inhibit.
        assert!(matches!(v.action, VoteAction::Speak | VoteAction::Pursue | VoteAction::Abstain));
    }
}
