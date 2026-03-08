// src/brain_zones/parietal.rs
// Córtex Parietal — Integração sensorial, atenção espacial
//
// Composição: RS + LT (Low-Threshold para atenção seletiva)

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::synaptic_core::{CamadaHibrida, PrecisionType, TipoNeuronal};
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
        let dist = vec![
            (PrecisionType::FP32, 0.05),
            (PrecisionType::FP16, 0.40),
            (PrecisionType::INT8, 0.45),
            (PrecisionType::INT4, 0.10),
        ];
        let escala = 40.0 / 127.0;

        let integration = CamadaHibrida::new(
            n_neurons, "parietal_integ",
            TipoNeuronal::RS,
            Some((TipoNeuronal::LT, 0.20)), // LT para atenção de baixo limiar
            Some(dist),
            escala,
        );

        let mut rng = thread_rng();
        let spatial: Vec<f32>  = (0..n_neurons).map(|_| rng.gen_range(0.0..0.2)).collect();
        let attention: Vec<f32> = (0..n_neurons).map(|_| rng.gen_range(0.9..1.1)).collect();

        Self {
            integration_layer: integration,
            spatial_map: spatial,
            attention_map: attention,
            attention_global: 1.0,
            noise_std,
        }
    }

    pub fn integrate(
        &mut self,
        visual_in: &[f32],
        sensory_in: &[f32],
        dt: f32,
        current_time: f32,
        config: &Config,
    ) -> Vec<f32> {
        let n = self.integration_layer.neuronios.len();
        let mut rng = thread_rng();
        let t_ms = current_time * 1000.0;
        let mut combined = vec![0.0; n];

        let vis_w = 0.6 * self.attention_global;
        let sen_w = 0.4 * self.attention_global;

        for i in 0..n {
            let noise = rng.gen_range(-self.noise_std..self.noise_std);
            let fused = (visual_in.get(i).copied().unwrap_or(0.0) * vis_w
                + sensory_in.get(i).copied().unwrap_or(0.0) * sen_w)
                * self.attention_map[i];
            combined[i] = fused + noise;
        }

        let spikes = self.integration_layer.update(&combined, dt, t_ms);

        let mut output = vec![0.0; n];
        for i in 0..n {
            if spikes[i] {
                self.spatial_map[i] = (self.spatial_map[i] * 0.9 + 0.1).clamp(0.0, 1.0);
                output[i] = 1.0;
            } else {
                self.spatial_map[i] *= 0.995;
                output[i] = self.spatial_map[i];
            }
        }
        output
    }

    pub fn set_attention(&mut self, level: f32) {
        self.attention_global = level.clamp(0.1, 3.0);
    }

    pub fn estatisticas(&self) -> ParietalStats {
        ParietalStats { integration: self.integration_layer.estatisticas() }
    }
}

pub struct ParietalStats {
    pub integration: crate::synaptic_core::CamadaStats,
}
