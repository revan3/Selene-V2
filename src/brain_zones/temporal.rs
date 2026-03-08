// src/brain_zones/temporal.rs
// Córtex Temporal — Reconhecimento auditivo, linguagem, memória semântica
//
// Composição neuronal:
//   recognition_layer: 70% RS + 30% CH
//   CH para reconhecimento rápido de padrões fonéticos repetitivos.

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::synaptic_core::{CamadaHibrida, PrecisionType, TipoNeuronal};
use rand::{Rng, thread_rng};
use crate::config::Config;

pub struct TemporalLobe {
    pub recognition_layer: CamadaHibrida,
    pub auditory_buffer: Vec<f32>,
    pub semantic_memory: Vec<f32>,
    pub novelty_detection: f32,
    pub habituation_counter: Vec<u32>,
    pub learning_rate: f32,
    pub noise_std_base: f32,
}

impl TemporalLobe {
    pub fn new(n_neurons: usize, learning_rate: f32, noise_std_base: f32, config: &Config) -> Self {
        // Temporal: FP16 dominante para reconhecimento de padrões preciso
        // 30% CH para reconhecimento rápido de padrões repetitivos (palavras, ritmos)
        let dist = vec![
            (PrecisionType::FP32, 0.05),
            (PrecisionType::FP16, 0.55),
            (PrecisionType::INT8, 0.40),
        ];

        let escala = 45.0 / 127.0;

        let recognition = CamadaHibrida::new(
            n_neurons, "temporal_recog",
            TipoNeuronal::RS,
            Some((TipoNeuronal::CH, 0.30)),
            Some(dist),
            escala,
        );

        Self {
            recognition_layer: recognition,
            auditory_buffer: vec![0.0; n_neurons],
            semantic_memory: vec![0.0; n_neurons],
            novelty_detection: 1.0,
            habituation_counter: vec![0; n_neurons],
            learning_rate,
            noise_std_base,
        }
    }

    pub fn process(
        &mut self,
        stimulus_in: &[f32],
        context_bias: &[f32],
        dt: f32,
        current_time: f32,
        config: &Config,
    ) -> Vec<f32> {
        let n = self.recognition_layer.neuronios.len();
        let mut rng = thread_rng();
        let mut combined_input = vec![0.0; n];
        let t_ms = current_time * 1000.0;

        for i in 0..n {
            let s = stimulus_in.get(i).copied().unwrap_or(0.0);
            let recurrence = self.auditory_buffer[i] * 0.75;
            let semantic_match = (s - self.semantic_memory[i]).abs();
            let hab_factor = if self.habituation_counter[i] > 20 { 0.5 } else { 1.0 };
            let surprise = (1.0 + semantic_match * self.novelty_detection) * hab_factor;
            let noise = rng.gen_range(-self.noise_std_base..self.noise_std_base) * surprise;

            combined_input[i] = (s + recurrence + context_bias.get(i).copied().unwrap_or(0.0) + noise) * surprise;
        }

        let spikes = self.recognition_layer.update(&combined_input, dt, t_ms);

        let mut output = vec![0.0; n];
        for i in 0..n {
            if spikes[i] {
                self.auditory_buffer[i] = 1.5;
                let s = stimulus_in.get(i).copied().unwrap_or(0.0);
                self.semantic_memory[i] = self.semantic_memory[i] * (1.0 - self.learning_rate)
                    + s * self.learning_rate;
                output[i] = 1.0;
                self.habituation_counter[i] = 0;
            } else {
                self.auditory_buffer[i] *= 0.88;
                output[i] = self.auditory_buffer[i];
                let s = stimulus_in.get(i).copied().unwrap_or(0.0);
                if s > 0.1 {
                    self.habituation_counter[i] = self.habituation_counter[i].saturating_add(1);
                }
            }
        }

        // Normalização para range 0-1
        let max_val = output.iter().copied().fold(0.0f32, f32::max);
        if max_val > 0.0 {
            output.iter_mut().for_each(|v| *v /= max_val);
        }

        output
    }

    pub fn set_novelty_sensitivity(&mut self, level: f32) {
        self.novelty_detection = level.clamp(0.1, 5.0);
    }

    pub fn estatisticas(&self) -> TemporalStats {
        TemporalStats { recognition: self.recognition_layer.estatisticas() }
    }
}

pub struct TemporalStats {
    pub recognition: crate::synaptic_core::CamadaStats,
}
