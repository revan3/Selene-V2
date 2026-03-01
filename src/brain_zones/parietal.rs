// src/brain_zones/parietal.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use crate::synaptic_core::{CamadaHibrida, PrecisionType};
use rand::{Rng, thread_rng};
use crate::config::Config;

pub struct ParietalLobe {
    pub integration_layer: CamadaHibrida,
    pub spatial_map: Vec<f32>,
    pub attention_map: Vec<f32>,
    pub attention_global: f32,
    pub noise_std: f32,
}

impl ParietalLobe {
    pub fn new(n_neurons: usize, noise_std: f32, config: &Config) -> Self {
        // Parietal: equilíbrio entre FP16 e INT8
        let mut integ_dist = config.precision_distribution.clone();
        integ_dist.push((PrecisionType::FP16, 0.40));
        integ_dist.push((PrecisionType::INT8, 0.40));
        
        let integration = CamadaHibrida::new(n_neurons, "parietal_integ", Some(integ_dist));

        let mut rng = thread_rng();
        let mut spatial = vec![0.0; n_neurons];
        let mut attention = vec![1.0; n_neurons];
        
        for i in 0..n_neurons {
            spatial[i] = rng.gen_range(0.0..0.2);
            attention[i] = rng.gen_range(0.9..1.1);
        }

        Self {
            integration_layer: integration,
            spatial_map: spatial,
            attention_map: attention,
            attention_global: 1.0,
            noise_std,
        }
    }

    pub fn integrate(&mut self, visual_in: &[f32], sensory_in: &[f32], dt: f32, current_time: f32, config: &Config) -> Vec<f32> {
        let n = self.integration_layer.neuronios.len();
        let mut rng = thread_rng();
        let mut combined_input = vec![0.0; n];

        let vis_weight = 0.6 * self.attention_global;
        let sen_weight = 0.4 * self.attention_global;

        for i in 0..n {
            let noise = rng.gen_range(-self.noise_std..self.noise_std);
            let fused = (visual_in[i] * vis_weight + sensory_in[i] * sen_weight) * self.attention_map[i];
            combined_input[i] = fused + noise;
        }

        let spikes = self.integration_layer.update(&combined_input, dt, current_time);

        let mut output = vec![0.0; n];
        for i in 0..n {
            if spikes[i] {
                self.spatial_map[i] = (1.0 + self.attention_map[i] * 0.5).min(2.0);
                output[i] = self.spatial_map[i];
            } else {
                self.spatial_map[i] *= 0.92;
                output[i] = self.spatial_map[i];
            }
            
            self.attention_map[i] = self.attention_map[i] * 0.99 + 1.0 * 0.01;
        }

        let max_val = output.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_val > 1.0 {
            for val in output.iter_mut() {
                *val /= max_val;
            }
        }

        output
    }

    pub fn set_attention(&mut self, global_level: f32, spatial_modulation: Option<&[f32]>) {
        self.attention_global = global_level.clamp(0.5, 3.0);
        
        if let Some(modulation) = spatial_modulation {
            for (i, &val) in modulation.iter().enumerate() {
                if i < self.attention_map.len() {
                    self.attention_map[i] = (self.attention_map[i] * val).clamp(0.1, 5.0);
                }
            }
        }
    }
    
    pub fn estatisticas(&self) -> ParietalStats {
        ParietalStats {
            integration: self.integration_layer.estatisticas(),
        }
    }
}

pub struct ParietalStats {
    pub integration: crate::synaptic_core::CamadaStats,
}