// src/brain_zones/occipital.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use crate::synaptic_core::{CamadaHibrida, PrecisionType};
use rand::{Rng, thread_rng};
use crate::config::Config;

pub struct OccipitalLobe {
    pub v1_primary_layer: CamadaHibrida,
    pub v2_feature_layer: CamadaHibrida,
    pub contrast_threshold: f32,
    pub flicker_buffer: Vec<f32>,
    pub orientation_pref: Vec<f32>,
    pub noise_std_base: f32,
}

impl OccipitalLobe {
    pub fn new(n_neurons: usize, noise_std_base: f32, config: &Config) -> Self {
        let n_v1 = (n_neurons as f32 * 0.7) as usize;
        let n_v2 = n_neurons - n_v1;

        // V1: mais INT8/INT4 para alta densidade (processamento visual bruto)
        let mut v1_dist = config.precision_distribution.clone();
        v1_dist.push((PrecisionType::INT8, 0.60));
        v1_dist.push((PrecisionType::INT4, 0.20));
        
        // V2: mais FP16 para integração de características
        let mut v2_dist = config.precision_distribution.clone();
        v2_dist.push((PrecisionType::FP16, 0.50));
        
        let v1 = CamadaHibrida::new(n_v1, "occipital_v1", Some(v1_dist));
        let v2 = CamadaHibrida::new(n_v2, "occipital_v2", Some(v2_dist));

        let mut rng = thread_rng();
        let mut ori_pref = vec![0.0; n_v1];
        for i in 0..n_v1 {
            ori_pref[i] = rng.gen_range(0.0..std::f32::consts::PI);
        }

        Self {
            v1_primary_layer: v1,
            v2_feature_layer: v2,
            contrast_threshold: 0.15,
            flicker_buffer: vec![0.0; n_v1],
            orientation_pref: ori_pref,
            noise_std_base,
        }
    }

    pub fn visual_sweep(&mut self, retinal_input: &[f32], dt: f32, top_down_bias: Option<&[f32]>, current_time: f32, config: &Config) -> Vec<f32> {
        let n_v1 = self.v1_primary_layer.neuronios.len();
        let mut rng = thread_rng();
        let mut v1_input = vec![0.0; n_v1];

        for i in 0..n_v1 {
            let pixel_val = retinal_input.get(i).cloned().unwrap_or(0.0);
            
            let motion = (pixel_val - self.flicker_buffer[i]).abs();
            let contrast_gain = if pixel_val.abs() > self.contrast_threshold { 1.3 } else { 0.4 };
            let ori_match = 1.0 + (pixel_val.sin() - self.orientation_pref[i].sin()).abs() * 0.5;
            
            let noise = rng.gen_range(-self.noise_std_base..self.noise_std_base);

            v1_input[i] = (pixel_val + motion * 2.5) * contrast_gain * ori_match + noise;
            
            self.flicker_buffer[i] = pixel_val;
        }

        if let Some(bias) = top_down_bias {
            for i in 0..n_v1.min(bias.len()) {
                v1_input[i] *= 1.0 + bias[i] * 0.6;
            }
        }

        let v1_spikes = self.v1_primary_layer.update(&v1_input, dt, current_time);

        let n_v2 = self.v2_feature_layer.neuronios.len();
        let mut v2_input = vec![0.0; n_v2];
        for i in 0..n_v1 {
            if v1_spikes[i] {
                let center = (i as f32 / n_v1 as f32 * n_v2 as f32) as isize;
                for offset in -2..=2 {
                    let idx = (center + offset).rem_euclid(n_v2 as isize) as usize;
                    v2_input[idx] += 2.0; 
                }
            }
        }

        let v2_spikes = self.v2_feature_layer.update(&v2_input, dt, current_time);

        v2_spikes
            .chunks(20)
            .map(|chunk| {
                let count = chunk.iter().filter(|&&s| s).count() as f32;
                (count / chunk.len() as f32) * 100.0
            })
            .collect()
    }

    pub fn set_sensitivity(&mut self, threshold: f32) {
        self.contrast_threshold = threshold.clamp(0.01, 0.9);
    }
    
    pub fn estatisticas(&self) -> OccipitalStats {
        OccipitalStats {
            v1: self.v1_primary_layer.estatisticas(),
            v2: self.v2_feature_layer.estatisticas(),
        }
    }
}

pub struct OccipitalStats {
    pub v1: crate::synaptic_core::CamadaStats,
    pub v2: crate::synaptic_core::CamadaStats,
}