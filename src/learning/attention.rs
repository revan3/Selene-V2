// src/learning/attention.rs
// Mecanismo de Atenção Seletiva Bio-inspirado
//
// Modela o "spotlight" atencional que amplifica sinais salientes e suprime ruído.
//
// Dois componentes:
//   1. Atenção bottom-up (salience): novidade detectada por desvio da média corrente
//   2. Atenção top-down (goal-directed): bias do FrontalLobe sobre onde focar
//
// Saída: vetor de ganhos por canal (0..1) multiplicado pelo input sensorial.
//
// Base biológica:
//   - Colliculus superior: saliência visual rápida (bottom-up)
//   - Córtex pré-frontal: atenção dirigida a objetivos (top-down)
//   - Pulvinar talâmico: amplifica o canal atendido

#![allow(dead_code)]
#![allow(unused_variables)]

/// Constante de tempo para média móvel de atividade de fundo (ms).
/// Lento o suficiente para capturar a linha de base, rápido para seguir mudanças.
const TAU_BASELINE_MS: f32 = 500.0;
/// Quão forte é o boost no canal mais saliente (máximo) — perfil padrão V3.3.
const MAX_GAIN:        f32 = 2.5;
/// Ganho mínimo — nunca suprime completamente (1/MAX_GAIN aprox) — perfil padrão V3.3.
const MIN_GAIN:        f32 = 0.3;
/// Peso do componente top-down vs bottom-up — perfil padrão V3.3.
const TOPDOWN_WEIGHT:  f32 = 0.4;

// ═══════════════════════════════════════════════════════════════════════════════
// V3.4 Multi-Self — Perfis de voz para o "Exército de Idiotas"
// ═══════════════════════════════════════════════════════════════════════════════
//
// Cada voz é uma instância paralela de AttentionGate com pesos diferenciados.
// Os 4 votos competem na arbitragem (VoiceArbiter, Fase B) via Atomics no
// ActiveContext compartilhado — sem Mutex no caminho quente.

/// Configuração de uma voz do Multi-Self V3.4.
#[derive(Clone, Copy, Debug)]
pub struct VoiceProfile {
    /// Nome da voz (telemetria).
    pub name:           &'static str,
    /// Ganho máximo por canal (clamp superior).
    pub max_gain:       f32,
    /// Ganho mínimo por canal (clamp inferior — nunca silencia totalmente).
    pub min_gain:       f32,
    /// Peso de top-down (0..1). Alto = segue Frontal; baixo = segue saliência bottom-up.
    pub topdown_weight: f32,
    /// Peso da voz na arbitragem polifônica (Fase B).
    pub voice_weight:   f32,
    /// Multiplicador da saliência computada — tendência a reagir a desvios.
    /// > 1 = mais reativo (Censor, Criativa); < 1 = mais conservador (Analítica).
    pub salience_bias:  f32,
    /// Ruído estocástico injetado a cada attend (0..1). Apenas a Voz Criativa usa > 0.
    pub creative_noise: f32,
}

/// Voz Analítica — "O Arquiteto". Foco em precisão e concept_id, top-down forte.
pub const VOICE_ANALITICA: VoiceProfile = VoiceProfile {
    name:           "Analitica",
    max_gain:       2.5,
    min_gain:       0.3,
    topdown_weight: 0.6,   // dirigida pelo Frontal
    voice_weight:   1.0,   // peso máximo na arbitragem
    salience_bias:  0.7,   // menos reativa a ruído
    creative_noise: 0.0,
};

/// Voz Negativa — "O Censor". Inibição lateral, monitora riscos e inconsistências.
pub const VOICE_CENSOR: VoiceProfile = VoiceProfile {
    name:           "Censor",
    max_gain:       1.8,   // suprime mais que amplifica
    min_gain:       0.10,  // pode silenciar quase totalmente
    topdown_weight: 0.5,
    voice_weight:   0.8,
    salience_bias:  1.3,   // dispara em anomalias
    creative_noise: 0.0,
};

/// Voz Positiva — "A Dopamina". Busca recompensa e curiosidade no RL_Engine.
pub const VOICE_DOPAMINA: VoiceProfile = VoiceProfile {
    name:           "Dopamina",
    max_gain:       2.8,   // amplifica recompensa
    min_gain:       0.4,
    topdown_weight: 0.3,   // mais bottom-up
    voice_weight:   0.7,
    salience_bias:  1.1,
    creative_noise: 0.0,
};

/// Voz Criativa — "O Ruído Estocástico". Saltos aleatórios no grafo, traz "memes".
pub const VOICE_CRIATIVA: VoiceProfile = VoiceProfile {
    name:           "Criativa",
    max_gain:       3.0,
    min_gain:       0.2,
    topdown_weight: 0.1,   // ignora Frontal majoritariamente
    voice_weight:   0.4,
    salience_bias:  1.5,
    creative_noise: 0.15,  // 15% de ruído estocástico nos ganhos
};

impl VoiceProfile {
    /// Perfil padrão V3.3 (compatível com `AttentionGate::new`).
    pub const DEFAULT_V33: VoiceProfile = VoiceProfile {
        name:           "Default",
        max_gain:       MAX_GAIN,
        min_gain:       MIN_GAIN,
        topdown_weight: TOPDOWN_WEIGHT,
        voice_weight:   1.0,
        salience_bias:  1.0,
        creative_noise: 0.0,
    };
}

/// Gate de atenção multicanal.
///
/// Cada "canal" corresponde a um neurônio ou grupo de neurônios na entrada sensorial.
/// O gate calcula um vetor de ganhos que é aplicado ao input antes de processar.
pub struct AttentionGate {
    /// Média móvel de ativação por canal (linha de base).
    baseline: Vec<f32>,
    /// Ganhos atuais por canal (0..max_gain do perfil).
    pub gains: Vec<f32>,
    /// Bias top-down (sinal do FrontalLobe).
    topdown_bias: Vec<f32>,
    /// Índice do canal mais saliente (para telemetria).
    pub canal_foco: usize,
    /// Nível médio de atenção (0..1).
    pub nivel_atencao: f32,
    n_channels: usize,
    /// Perfil de voz (V3.4). Padrão = V3.3 (DEFAULT_V33).
    pub profile: VoiceProfile,
    /// Estado interno do gerador de ruído estocástico (xorshift64) — só usado se
    /// `profile.creative_noise > 0`. Determinístico para reprodutibilidade em testes.
    rng_state: u64,
}

impl AttentionGate {
    pub fn new(n_channels: usize) -> Self {
        Self::with_profile(n_channels, VoiceProfile::DEFAULT_V33)
    }

    /// Construtor V3.4: instancia uma voz com perfil específico.
    pub fn with_profile(n_channels: usize, profile: VoiceProfile) -> Self {
        // Seed única por nome de voz — evita 4 vozes gerarem o mesmo ruído.
        let seed = profile.name.bytes().fold(0xCBF29CE484222325u64, |a, b| {
            a.wrapping_mul(0x100000001B3).wrapping_add(b as u64)
        });
        Self {
            baseline:     vec![0.1; n_channels],
            gains:        vec![1.0; n_channels],
            topdown_bias: vec![0.0; n_channels],
            canal_foco:   0,
            nivel_atencao: 0.5,
            n_channels,
            profile,
            rng_state: seed.max(1),
        }
    }

    /// xorshift64 — barato, determinístico, sem alocação. Retorna f32 em [0, 1).
    #[inline]
    fn next_noise(&mut self) -> f32 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        ((x >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Aplica o gate ao input sensorial e atualiza estado interno.
    ///
    /// - `input`: firing rates da camada sensorial (ex: V1 ou cochlea)
    /// - `dt_ms`: passo de tempo em ms
    ///
    /// Retorna o input modulado pelo ganho atencional.
    pub fn attend(&mut self, input: &[f32], dt_ms: f32) -> Vec<f32> {
        let n = self.n_channels.min(input.len());
        let decay = (-dt_ms / TAU_BASELINE_MS).exp();
        // V3.4: parâmetros vêm do perfil de voz (constantes V3.3 ainda servem como fallback).
        let max_gain = self.profile.max_gain;
        let min_gain = self.profile.min_gain;
        let td_w     = self.profile.topdown_weight;
        let sal_bias = self.profile.salience_bias;

        // Modo lazy: se taxa global < 1% apenas atualiza baseline e aplica ganhos existentes.
        // Evita salience+softmax quando o sistema está em silêncio biológico.
        let global_rate = if n > 0 { input[..n].iter().sum::<f32>() / n as f32 } else { 0.0 };
        if global_rate < 0.01 {
            for i in 0..n {
                self.baseline[i] = self.baseline[i] * decay + input[i] * (1.0 - decay);
            }
            self.nivel_atencao *= 0.99;
            let mut out = Vec::with_capacity(input.len());
            for i in 0..input.len() {
                let gain = if i < n { self.gains[i] } else { 1.0 };
                out.push((input[i] * gain).clamp(0.0, 1.0));
            }
            return out;
        }

        // Atualiza linha de base (média exponencial)
        for i in 0..n {
            self.baseline[i] = self.baseline[i] * decay + input[i] * (1.0 - decay);
        }

        // Saliência bottom-up: desvio absoluto da linha de base × bias do perfil
        let mut salience = vec![0.0f32; n];
        for i in 0..n {
            salience[i] = (input[i] - self.baseline[i]).abs() * sal_bias;
        }

        // Normaliza saliência para [0, 1]
        let max_sal = salience.iter().cloned().fold(0.0f32, f32::max).max(0.001);
        for s in &mut salience { *s /= max_sal; }

        // Combina bottom-up e top-down (peso conforme perfil)
        let mut combined = vec![0.0f32; n];
        for i in 0..n {
            let td = if i < self.topdown_bias.len() { self.topdown_bias[i] } else { 0.0 };
            combined[i] = salience[i] * (1.0 - td_w) + td * td_w;
        }

        // Voz Criativa: ruído estocástico — saltos aleatórios no foco
        if self.profile.creative_noise > 0.0 {
            let amp = self.profile.creative_noise;
            for i in 0..n {
                let r = self.next_noise() * 2.0 - 1.0; // [-1, 1]
                combined[i] = (combined[i] + r * amp).max(0.0);
            }
        }

        // Softmax esparso: amplifica máximo, atenua o resto
        let max_c = combined.iter().cloned().fold(0.0f32, f32::max);
        self.canal_foco = combined.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        for i in 0..n {
            let normed = if max_c > 0.001 { combined[i] / max_c } else { 0.5 };
            self.gains[i] = min_gain + (max_gain - min_gain) * normed.powi(2);
        }

        // Nível médio de atenção (normalizado pelo max_gain do perfil)
        self.nivel_atencao = self.gains[0..n].iter().sum::<f32>() / n as f32 / max_gain;

        // Aplica ganhos ao input
        let mut out = Vec::with_capacity(input.len());
        for i in 0..input.len() {
            let gain = if i < n { self.gains[i] } else { 1.0 };
            out.push((input[i] * gain).clamp(0.0, 1.0));
        }
        out
    }

    /// Recebe bias top-down do FrontalLobe.
    /// `frontal_rates`: firing rates do frontal mapeados para os canais de atenção.
    pub fn set_topdown(&mut self, frontal_rates: &[f32]) {
        let n = self.n_channels.min(frontal_rates.len());
        // Normaliza para [0, 1]
        let max_f = frontal_rates[0..n].iter().cloned().fold(0.001f32, f32::max);
        for i in 0..n {
            self.topdown_bias[i] = frontal_rates[i] / max_f;
        }
    }

    /// Redimensiona para um número diferente de canais (ex: ao mudar resolução visual).
    pub fn resize(&mut self, n_new: usize) {
        self.baseline.resize(n_new, 0.1);
        self.gains.resize(n_new, 1.0);
        self.topdown_bias.resize(n_new, 0.0);
        self.n_channels = n_new;
    }
}
