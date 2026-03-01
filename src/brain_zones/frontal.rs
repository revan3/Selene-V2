// src/brain_zones/frontal.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use crate::synaptic_core::{CamadaHibrida, PrecisionType};
use rand::{Rng, thread_rng};
use crate::config::Config;

#[derive(Debug)]
pub struct FrontalLobe {
    pub executive_layer: CamadaHibrida,
    pub inhibitory_layer: CamadaHibrida,
    pub dopamine_level: f32,
    pub working_memory_trace: Vec<f32>,
    pub inhibition_strength: f32,
    pub noise_std: f32,
}

impl FrontalLobe {
    pub fn new(n_executive: usize, inhibition_ratio: f32, noise_std: f32, config: &Config) -> Self {
        let n_inhib = (n_executive as f32 * inhibition_ratio) as usize;
        
        // Distribuição específica para frontal (mais FP16 para precisão)
        let mut exec_dist = config.precision_distribution.clone();
        exec_dist.push((PrecisionType::FP16, 0.50)); // 50% FP16 para executivos
        
        let mut inhib_dist = config.precision_distribution.clone();
        inhib_dist.push((PrecisionType::INT8, 0.60)); // 60% INT8 para inibitórios
        
        let executive_layer = CamadaHibrida::new(n_executive, "frontal_exec", Some(exec_dist));
        let inhibitory_layer = CamadaHibrida::new(n_inhib, "frontal_inhib", Some(inhib_dist));

        Self {
            executive_layer,
            inhibitory_layer,
            dopamine_level: 1.0,
            working_memory_trace: vec![0.0; n_executive],
            inhibition_strength: 6.5,
            noise_std,
        }
    }

    pub fn decide(&mut self, sensory_input: &[f32], goal_bias: &[f32], dt: f32, current_time: f32, config: &Config) -> Vec<f32> {
        let n = self.executive_layer.neuronios.len();
        let mut rng = thread_rng();
        let mut combined_input = vec![0.0; n];

        let gain = 1.0 + self.dopamine_level * 0.8;
        
        for i in 0..n {
            let noise = rng.gen_range(-self.noise_std..self.noise_std);
            combined_input[i] = (sensory_input[i] + (goal_bias[i] * 1.5) + self.working_memory_trace[i] + noise) * gain;
        }

        let executive_spikes = self.executive_layer.update(&combined_input, dt, current_time);

        let active_count = executive_spikes.iter().filter(|&&s| s).count() as f32;
        let inhibition_input = vec![active_count * 0.4; self.inhibitory_layer.neuronios.len()];
        let inhibitory_spikes = self.inhibitory_layer.update(&inhibition_input, dt, current_time);

        let mut output_voltages = vec![0.0; n];
        let n_inhib = self.inhibitory_layer.neuronios.len();

        for i in 0..n {
            if executive_spikes[i] {
                self.working_memory_trace[i] = 25.0 * self.dopamine_level;
            }
            self.working_memory_trace[i] *= 0.96;

            if inhibitory_spikes[i % n_inhib] {
                let drop = self.inhibition_strength * (1.0 + self.dopamine_level * 0.2);
                if let Some(neuronio) = self.executive_layer.neuronios.get_mut(i) {
                    neuronio.v -= drop;
                }
            }

            output_voltages[i] = if executive_spikes[i] { 1.0 } else { 0.0 };
        }

        output_voltages
    }

    pub fn set_dopamine(&mut self, level: f32) {
        self.dopamine_level = level.clamp(0.3, 2.5);
    }
    
    pub fn estatisticas(&self) -> FrontalStats {
        FrontalStats {
            executive: self.executive_layer.estatisticas(),
            inhibitory: self.inhibitory_layer.estatisticas(),
            dopamine: self.dopamine_level,
        }
    }
}

pub struct FrontalStats {
    pub executive: crate::synaptic_core::CamadaStats,
    pub inhibitory: crate::synaptic_core::CamadaStats,
    pub dopamine: f32,
}