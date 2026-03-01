// src/brain_zones/temporal.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use crate::synaptic_core::{CamadaHibrida, PrecisionType};
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
        // Temporal: mais FP16 para reconhecimento de padrões
        let mut recog_dist = config.precision_distribution.clone();
        recog_dist.push((PrecisionType::FP16, 0.55));
        
        let recognition = CamadaHibrida::new(n_neurons, "temporal_recog", Some(recog_dist));

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

    pub fn process(&mut self, stimulus_in: &[f32], context_bias: &[f32], dt: f32, current_time: f32, config: &Config) -> Vec<f32> {
        let n = self.recognition_layer.neuronios.len();
        let mut rng = thread_rng();
        let mut combined_input = vec![0.0; n];

        for i in 0..n {
            let recurrence = self.auditory_buffer[i] * 0.75;
            let semantic_match = (stimulus_in[i] - self.semantic_memory[i]).abs();
            
            let hab_factor = if self.habituation_counter[i] > 20 { 0.5 } else { 1.0 };
            let surprise = (1.0 + (semantic_match * self.novelty_detection)) * hab_factor;

            let noise_std = self.noise_std_base * surprise;
            let noise = rng.gen_range(-noise_std..noise_std);

            combined_input[i] = (stimulus_in[i] + recurrence + context_bias[i] + noise) * surprise;
        }

        let spikes = self.recognition_layer.update(&combined_input, dt, current_time);

        let mut output = vec![0.0; n];
        for i in 0..n {
            if spikes[i] {
                self.auditory_buffer[i] = 1.5;
                self.semantic_memory[i] = self.semantic_memory[i] * (1.0 - self.learning_rate) 
                                        + stimulus_in[i] * self.learning_rate;
                output[i] = 1.0;
                self.habituation_counter[i] = 0;
            } else {
                self.auditory_buffer[i] *= 0.88;
                output[i] = self.auditory_buffer[i];

                if stimulus_in[i] > 0.1 {
                    self.habituation_counter[i] = self.habituation_counter[i].saturating_add(1);
                }
            }
        }

        let max_val = output.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_val > 0.0 {
            for val in output.iter_mut() {
                *val /= max_val;
            }
        }

        output
    }

    pub fn set_novelty_sensitivity(&mut self, level: f32) {
        self.novelty_detection = level.clamp(0.1, 5.0);
    }
    
    pub fn estatisticas(&self) -> TemporalStats {
        TemporalStats {
            recognition: self.recognition_layer.estatisticas(),
        }
    }
}

pub struct TemporalStats {
    pub recognition: crate::synaptic_core::CamadaStats,
}