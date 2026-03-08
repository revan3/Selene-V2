// src/brain_zones/occipital.rs
// Córtex Occipital — Processamento visual primário (V1) e secundário (V2)
//
// Composição neuronal:
//   v1_primary_layer: 60% RS + 40% CH — detecção de bordas e contraste
//   v2_feature_layer: 70% CH + 30% RS — integração de características visuais
//
// CH (Chattering) é ideal para processamento visual porque dispara em bursts
// rápidos repetitivos — o equivalente às células simples de V1 que respondem
// a estímulos orientados com alta taxa de disparo.

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::synaptic_core::{CamadaHibrida, PrecisionType, TipoNeuronal};
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

        // V1: processamento bruto — INT8/INT4 para alta densidade
        // CH (40%) para células de alta frequência (bordas, movimento)
        let v1_dist = vec![
            (PrecisionType::FP16, 0.10),
            (PrecisionType::INT8, 0.60),
            (PrecisionType::INT4, 0.30),
        ];

        // V2: integração de características — FP16 para precisão
        // CH dominante (70%) para reconhecimento rápido de padrões
        let v2_dist = vec![
            (PrecisionType::FP32, 0.05),
            (PrecisionType::FP16, 0.55),
            (PrecisionType::INT8, 0.40),
        ];

        // Escala para correntes visuais típicas (~40pA)
        let escala = 40.0 / 127.0;

        let v1 = CamadaHibrida::new(
            n_v1, "occipital_v1",
            TipoNeuronal::RS,
            Some((TipoNeuronal::CH, 0.40)), // 40% CH para detecção rápida
            Some(v1_dist),
            escala,
        );
        let v2 = CamadaHibrida::new(
            n_v2, "occipital_v2",
            TipoNeuronal::CH,               // CH dominante em V2
            Some((TipoNeuronal::RS, 0.30)),
            Some(v2_dist),
            escala,
        );

        let mut rng = thread_rng();
        let ori_pref: Vec<f32> = (0..n_v1)
            .map(|_| rng.gen_range(0.0..std::f32::consts::PI))
            .collect();

        Self {
            v1_primary_layer: v1,
            v2_feature_layer: v2,
            contrast_threshold: 0.15,
            flicker_buffer: vec![0.0; n_v1],
            orientation_pref: ori_pref,
            noise_std_base,
        }
    }

    pub fn visual_sweep(
        &mut self,
        retinal_input: &[f32],
        dt: f32,
        top_down_bias: Option<&[f32]>,
        current_time: f32,
        config: &Config,
    ) -> Vec<f32> {
        let n_v1 = self.v1_primary_layer.neuronios.len();
        let mut rng = thread_rng();
        let mut v1_input = vec![0.0; n_v1];
        let t_ms = current_time * 1000.0;

        for i in 0..n_v1 {
            let pixel = retinal_input.get(i).copied().unwrap_or(0.0);
            let motion = (pixel - self.flicker_buffer[i]).abs();
            let contrast_gain = if pixel.abs() > self.contrast_threshold { 1.3 } else { 0.4 };
            let ori_match = 1.0 + (pixel.sin() - self.orientation_pref[i].sin()).abs() * 0.5;
            let noise = rng.gen_range(-self.noise_std_base..self.noise_std_base);

            v1_input[i] = (pixel + motion * 2.5) * contrast_gain * ori_match + noise;
            self.flicker_buffer[i] = pixel;
        }

        // Atenção top-down modula V1
        if let Some(bias) = top_down_bias {
            for i in 0..n_v1.min(bias.len()) {
                v1_input[i] *= 1.0 + bias[i] * 0.6;
            }
        }

        let v1_spikes = self.v1_primary_layer.update(&v1_input, dt, t_ms);

        // V1 → V2: propagação lateral de spikes
        let n_v2 = self.v2_feature_layer.neuronios.len();
        let mut v2_input = vec![0.0; n_v2];
        for i in 0..n_v1 {
            if v1_spikes[i] {
                let center = (i as f32 / n_v1 as f32 * n_v2 as f32) as isize;
                for offset in -2..=2isize {
                    let idx = (center + offset).rem_euclid(n_v2 as isize) as usize;
                    v2_input[idx] += 2.0;
                }
            }
        }

        let v2_spikes = self.v2_feature_layer.update(&v2_input, dt, t_ms);

        // Agrega em features (taxa de disparo por janela de 20 neurônios)
        v2_spikes.chunks(20).map(|chunk| {
            let count = chunk.iter().filter(|&&s| s).count() as f32;
            (count / chunk.len() as f32) * 100.0
        }).collect()
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
