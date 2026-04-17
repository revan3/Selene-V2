// src/brain_zones/cingulate.rs
// Córtex Cingulado Anterior (ACC) — Monitoramento de conflito e erro
//
// O ACC é o "árbitro" do cérebro: detecta quando duas respostas concorrentes
// competem (ex: o que o frontal quer fazer vs o que o límbico quer sentir),
// gera um sinal de conflito que recruta mais atenção e ajusta o comportamento.
//
// Biologicamente:
//   - dACC (dorsal): monitoramento cognitivo de conflito — RS + IB
//   - rACC (rostral): regulação emocional, dor social — RS + FS
//   - Caminho: detecta conflito → sinal para locus coeruleus → NA ↑ → foco ↑
//
// Composição neuronal:
//   conflict_layer: 60% RS + 40% IB — IB para burst quando conflito é alto
//   regulation_layer: 70% RS + 30% FS — FS para inibição emocional top-down

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::synaptic_core::{CamadaHibrida, PrecisionType, TipoNeuronal};
use crate::config::Config;
use rand::{Rng, thread_rng};

/// Decay do sinal de conflito entre ticks (EMA).
const CONFLICT_DECAY: f32 = 0.85;
/// Limiar acima do qual o conflito é "alto" e dispara ajuste comportamental.
const CONFLICT_THRESHOLD: f32 = 0.45;
/// Taxa de aprendizado do valor esperado de resposta.
const EXPECTED_LR: f32 = 0.05;

pub struct AnteriorCingulate {
    /// Camada dorsal — detecção de conflito cognitivo.
    pub conflict_layer: CamadaHibrida,
    /// Camada rostral — regulação emocional, dor social.
    pub regulation_layer: CamadaHibrida,

    /// Nível atual de conflito detectado [0.0, 1.0].
    /// Alto quando goal frontal e estado límbico divergem fortemente.
    pub conflict_signal: f32,

    /// Sinal de erro de resposta: discrepância entre ação esperada e ação executada.
    /// > 0 = surpresa positiva, < 0 = erro (pior que esperado).
    pub response_error: f32,

    /// Nível de "dor social": ativado por rejeição, punição verbal, emocao < -0.5.
    /// Mesmo circuito que dor física — modula amígdala e noradrenalina.
    pub social_pain: f32,

    /// Valor esperado de atividade frontal — baseline adaptativo (EMA).
    /// Quando atividade real desvia muito → conflict_signal sobe.
    expected_frontal: f32,

    /// Valor esperado de estado límbico — baseline adaptativo.
    expected_limbic: f32,

    /// Fator de ajuste para n_passos no graph-walk.
    /// Conflito alto → walk mais cauteloso (menos passos, mais focado).
    /// Conflito baixo → walk livre.
    pub adjustment_factor: f32,
}

impl AnteriorCingulate {
    pub fn new(n_neurons: usize, config: &Config) -> Self {
        // dACC: IB (40%) para burst de conflito intenso
        let conflict_dist = vec![
            (PrecisionType::FP16, 0.30),
            (PrecisionType::INT8, 0.55),
            (PrecisionType::INT4, 0.15),
        ];
        // rACC: FS (30%) para regulação emocional inibitória
        let reg_dist = vec![
            (PrecisionType::FP32, 0.05),
            (PrecisionType::FP16, 0.45),
            (PrecisionType::INT8, 0.50),
        ];

        let escala = 38.0 / 127.0;
        let n_sub = (n_neurons / 2).max(4);

        let conflict = CamadaHibrida::new(
            n_sub, "acc_conflict",
            TipoNeuronal::IB,               // IB dominante — burst de conflito
            Some((TipoNeuronal::RS, 0.60)), // 60% RS para integração sustentada
            Some(conflict_dist),
            escala,
        );
        let regulation = CamadaHibrida::new(
            n_sub, "acc_regulation",
            TipoNeuronal::RS,
            Some((TipoNeuronal::FS, 0.30)), // FS para inibição emocional
            Some(reg_dist),
            escala,
        );

        Self {
            conflict_layer: conflict,
            regulation_layer: regulation,
            conflict_signal: 0.0,
            response_error: 0.0,
            social_pain: 0.0,
            expected_frontal: 0.5,
            expected_limbic: 0.0,
            adjustment_factor: 1.0,
        }
    }

    /// Processa conflito entre o estado frontal (intenção) e límbico (emoção).
    ///
    /// `frontal_activity`: taxa de atividade média do frontal (0.0-1.0)
    /// `limbic_valence`: valência emocional atual (-1.0 a 1.0)
    /// `rpe`: Reward Prediction Error do RL (> 0 = melhor que esperado)
    ///
    /// Retorna `(conflict_signal, adjustment_factor)`.
    pub fn update(
        &mut self,
        frontal_activity: f32,
        limbic_valence: f32,
        rpe: f32,
        dt: f32,
        current_time: f32,
        config: &Config,
    ) -> (f32, f32) {
        let t_ms = current_time * 1000.0;
        let mut rng = thread_rng();

        // 1. Conflito cognitivo: divergência entre atividade frontal e baseline esperado
        let frontal_deviation = (frontal_activity - self.expected_frontal).abs();

        // 2. Conflito emocional: quando a emoção está muito fora do esperado
        let limbic_deviation = (limbic_valence - self.expected_limbic).abs();

        // 3. Conflito total = combinação ponderada (frontal tem mais peso)
        let raw_conflict = (frontal_deviation * 0.6 + limbic_deviation * 0.4).clamp(0.0, 1.0);

        // 4. Alimenta a camada de conflito (dACC) — IB dispara em burst com conflito alto
        let n = self.conflict_layer.neuronios.len();
        let conflict_input: Vec<f32> = (0..n)
            .map(|_| raw_conflict * 30.0 + rng.gen_range(-1.0..1.0))
            .collect();
        let conflict_spikes = self.conflict_layer.update(&conflict_input, dt, t_ms);
        let conflict_rate = conflict_spikes.iter().filter(|&&s| s).count() as f32 / n as f32;

        // 5. EMA do sinal de conflito
        self.conflict_signal = self.conflict_signal * CONFLICT_DECAY
            + conflict_rate * (1.0 - CONFLICT_DECAY);

        // 6. Sinal de erro de resposta (RPE negativo = erro, ativa ACC fortemente)
        self.response_error = self.response_error * 0.90 + rpe * 0.10;

        // 7. Camada de regulação (rACC) — inibe amígdala quando emoção negativa alta
        let n_reg = self.regulation_layer.neuronios.len();
        let reg_input_strength = if limbic_valence < -0.4 {
            // Dor social ativada — recruta mais regulação
            self.social_pain = (self.social_pain + 0.05 * limbic_valence.abs()).clamp(0.0, 1.0);
            limbic_valence.abs() * 25.0
        } else {
            self.social_pain *= 0.97; // decai quando emoção melhora
            limbic_valence.abs() * 10.0
        };
        let reg_input: Vec<f32> = (0..n_reg)
            .map(|_| reg_input_strength + rng.gen_range(-0.5..0.5))
            .collect();
        self.regulation_layer.update(&reg_input, dt, t_ms);

        // 8. Atualiza baselines adaptativos (EMA)
        self.expected_frontal += (frontal_activity - self.expected_frontal) * EXPECTED_LR;
        self.expected_limbic  += (limbic_valence - self.expected_limbic) * EXPECTED_LR;

        // 9. Fator de ajuste comportamental
        // Conflito alto → menos passos, mais foco (evita respostas impulsivas)
        // Conflito baixo → comportamento normal
        self.adjustment_factor = if self.conflict_signal > CONFLICT_THRESHOLD {
            (1.0 - (self.conflict_signal - CONFLICT_THRESHOLD) * 0.8).clamp(0.5, 1.0)
        } else {
            1.0
        };

        (self.conflict_signal, self.adjustment_factor)
    }

    /// Retorna o delta de noradrenalina gerado pelo ACC.
    /// Conflito alto → recruta locus coeruleus → NA ↑ (mais atenção e foco).
    /// Biologicamente: ACC→LC é uma das principais projeções eferentes do dACC.
    pub fn noradrenaline_drive(&self) -> f32 {
        if self.conflict_signal > CONFLICT_THRESHOLD {
            (self.conflict_signal - CONFLICT_THRESHOLD) * 0.3
        } else {
            0.0
        }
    }

    /// Retorna o sinal inibitório sobre a amígdala (rACC → BLA).
    /// Quanto mais o rACC está ativo (regulação emocional), mais inibe a amígdala.
    /// Biologicamente: rACC→amígdala é o substrato de regulação do medo por cognição.
    pub fn amygdala_inhibition(&self) -> f32 {
        // Usa conflict_signal como proxy da atividade de regulação (rACC co-ativa com dACC)
        let reg_activity = (1.0 - self.conflict_signal) * self.social_pain;
        (reg_activity * 0.5 + self.social_pain * 0.2).clamp(0.0, 0.6)
    }

    /// Notifica o ACC de uma punição social (rejeição, feedback negativo verbal).
    /// Aumenta `social_pain` diretamente — mesmo circuito que dor física.
    pub fn registrar_rejeicao(&mut self, intensidade: f32) {
        self.social_pain = (self.social_pain + intensidade * 0.4).clamp(0.0, 1.0);
        // Spike de conflito imediato
        self.conflict_signal = (self.conflict_signal + intensidade * 0.3).clamp(0.0, 1.0);
    }

    pub fn estatisticas(&self) -> CingulateStats {
        CingulateStats {
            conflict: self.conflict_layer.estatisticas(),
            regulation: self.regulation_layer.estatisticas(),
            conflict_signal: self.conflict_signal,
            social_pain: self.social_pain,
            adjustment_factor: self.adjustment_factor,
        }
    }
}

pub struct CingulateStats {
    pub conflict: crate::synaptic_core::CamadaStats,
    pub regulation: crate::synaptic_core::CamadaStats,
    pub conflict_signal: f32,
    pub social_pain: f32,
    pub adjustment_factor: f32,
}
