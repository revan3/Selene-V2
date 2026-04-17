// src/brain_zones/limbic.rs
// Sistema Límbico — Emoção, medo, recompensa
//
// Composição neuronal:
//   amygdala: 50% RS + 50% IB — resposta de medo em burst (biologicamente correto)
//   nucleus_accumbens: RS puro — recompensa e prazer
//
// IB (Intrinsic Bursting) é o tipo correto para amígdala porque:
// - Dispara em bursts iniciais intensos quando ativado
// - Biológico: células BLA (basolateral amygdala) são majorialmente IB
// - Resultado: respostas emocionais mais dinâmicas e realistas

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::synaptic_core::{CamadaHibrida, PrecisionType, TipoNeuronal};
use rand::{Rng, thread_rng};
use crate::config::Config;

pub struct LimbicSystem {
    pub amygdala: CamadaHibrida,
    pub nucleus_accumbens: CamadaHibrida,
    pub emotional_state: f32,
    pub arousal_level: f32,
    pub dopamine_mod: f32,
    pub habituation_counter: Vec<u32>,
}

impl LimbicSystem {
    pub fn new(n_neurons: usize, config: &Config) -> Self {
        let n_sub = (n_neurons / 2).max(1);

        // Amígdala: FP16 para precisão emocional, 50% IB para burst de medo
        let amy_dist = vec![
            (PrecisionType::FP32, 0.05),
            (PrecisionType::FP16, 0.55),
            (PrecisionType::INT8, 0.40),
        ];

        // Accumbens: INT8 para processamento de recompensa em massa
        let acc_dist = vec![
            (PrecisionType::FP16, 0.30),
            (PrecisionType::INT8, 0.60),
            (PrecisionType::INT4, 0.10),
        ];

        // Escala para correntes límbicas (~35pA)
        let escala = 35.0 / 127.0;

        let amy = CamadaHibrida::new(
            n_sub, "limbico_amygdala",
            TipoNeuronal::IB,               // IB dominante — burst emocional
            Some((TipoNeuronal::RS, 0.50)),
            Some(amy_dist),
            escala,
        );
        let acc = CamadaHibrida::new(
            n_sub, "limbico_accumbens",
            TipoNeuronal::RS,
            None,
            Some(acc_dist),
            escala,
        );

        Self {
            amygdala: amy,
            nucleus_accumbens: acc,
            emotional_state: 0.0,
            arousal_level: 1.0,
            dopamine_mod: 1.0,
            habituation_counter: vec![0; n_sub],
        }
    }

    pub fn evaluate(
        &mut self,
        sensory_valence: &[f32],
        reward_signal: f32,
        dt: f32,
        current_time: f32,
        config: &Config,
    ) -> (f32, f32) {
        let mut rng = thread_rng();
        let n = self.amygdala.neuronios.len();
        let t_ms = current_time * 1000.0;

        let mut amy_in = vec![0.0; n];
        for i in 0..n {
            let s = sensory_valence.get(i).copied().unwrap_or(0.0);
            let intensity = s.abs();
            let sign_weight = if s < 0.0 { 1.5 } else { 0.5 };
            let novelty_bonus = if intensity > 0.7 { 8.0 } else { 0.0 };
            let mut input = (intensity * 12.0 * sign_weight) + novelty_bonus;
            input += rng.gen_range(-1.0..1.0);

            if intensity > 0.4 {
                self.habituation_counter[i] = self.habituation_counter[i].saturating_add(1);
                if self.habituation_counter[i] > 8 {
                    input *= 0.6;
                }
            } else {
                self.habituation_counter[i] = 0;
            }
            amy_in[i] = input;
        }

        let amy_spikes = self.amygdala.update(&amy_in, dt, t_ms);

        let fear_factor = amy_spikes.iter().filter(|&&s| s).count() as f32 / n as f32;
        let fear_inhibition = (1.0 - fear_factor * 0.5).max(0.2);

        let acc_input = reward_signal * 15.0 * self.dopamine_mod * fear_inhibition;
        let acc_in = vec![acc_input; self.nucleus_accumbens.neuronios.len()];
        let acc_spikes = self.nucleus_accumbens.update(&acc_in, dt, t_ms);

        let pleasure_score = acc_spikes.iter().filter(|&&s| s).count() as f32
            / self.nucleus_accumbens.neuronios.len() as f32;

        self.emotional_state = (self.emotional_state * 0.92) + (pleasure_score - fear_factor) * 0.6;
        self.emotional_state = self.emotional_state.clamp(-1.0, 1.0);

        self.arousal_level = (0.8 + fear_factor * 2.0 + pleasure_score + self.dopamine_mod * 0.2)
            .clamp(0.5, 3.5);

        self.dopamine_mod = (1.0 + self.emotional_state * 0.4).clamp(0.5, 2.5);

        (self.emotional_state, self.arousal_level)
    }

    pub fn set_dopamine(&mut self, level: f32) {
        self.dopamine_mod = level.clamp(0.3, 3.0);
    }

    pub fn estatisticas(&self) -> LimbicoStats {
        LimbicoStats {
            amygdala: self.amygdala.estatisticas(),
            accumbens: self.nucleus_accumbens.estatisticas(),
            emotional_state: self.emotional_state,
            arousal: self.arousal_level,
        }
    }
}

pub struct LimbicoStats {
    pub amygdala: crate::synaptic_core::CamadaStats,
    pub accumbens: crate::synaptic_core::CamadaStats,
    pub emotional_state: f32,
    pub arousal: f32,
}
