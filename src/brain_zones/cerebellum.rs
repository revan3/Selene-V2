// src/brain_zones/cerebellum.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use crate::synaptic_core::{CamadaHibrida, PrecisionType};
use crate::config::Config;
use std::collections::HashMap;
use uuid::Uuid;

pub struct Cerebellum {
    pub purkinje_layer: CamadaHibrida,
    pub granular_layer: CamadaHibrida,
    pub error_signal: Vec<f32>,
    pub ltd_factor: Vec<f32>,
    // IDs dos neurônios (para referência futura)
    purkinje_ids: Vec<Uuid>,
    granular_ids: Vec<Uuid>,
}

impl Cerebellum {
    pub fn new(n_purkinje: usize, n_granular: usize, config: &Config) -> Self {
        // Distribuição específica para cerebelo (mais INT8/INT4)
        let mut purkinje_dist = config.precision_distribution.clone();
        purkinje_dist.push((PrecisionType::INT8, 0.70)); // 70% INT8 para Purkinje
        
        let mut granular_dist = config.precision_distribution.clone();
        granular_dist.push((PrecisionType::INT4, 0.30)); // 30% INT4 para granulares
        
        let mut purkinje = CamadaHibrida::new(n_purkinje, "cerebelo_purkinje", Some(purkinje_dist));
        let mut granular = CamadaHibrida::new(n_granular, "cerebelo_granular", Some(granular_dist));

        // Ajustes biofísicos específicos do cerebelo
        for neuronio in &mut purkinje.neuronios {
            neuronio.g_scale_hh = 3.8;
            neuronio.threshold = 20.0;
            neuronio.d = 2.0;
        }
        
        for neuronio in &mut granular.neuronios {
            neuronio.threshold = 15.0;
        }

        // Gerar IDs para todos os neurônios
        let mut purkinje_ids = Vec::with_capacity(n_purkinje);
        for i in 0..n_purkinje {
            purkinje_ids.push(Uuid::new_v4());
        }
        
        let mut granular_ids = Vec::with_capacity(n_granular);
        for i in 0..n_granular {
            granular_ids.push(Uuid::new_v4());
        }

        Self {
            purkinje_layer: purkinje,
            granular_layer: granular,
            error_signal: vec![0.0; n_purkinje],
            ltd_factor: vec![1.0; n_purkinje],
            purkinje_ids,
            granular_ids,
        }
    }

    pub fn coordinate(&mut self, intention: &[f32], sensory_feedback: &[f32], dt: f32, current_time: f32, config: &Config) -> Vec<f32> {
        // 1. Granulares (Feedback Sensorial)
        let granular_spikes = self.granular_layer.update(sensory_feedback, dt, current_time);

        // 2. Fan-in/out: Granular -> Purkinje com modulação LTD
        let mut purkinje_input = vec![0.0; self.purkinje_layer.neuronios.len()];
        let p_len = purkinje_input.len();

        for (i, &spike) in granular_spikes.iter().enumerate() {
            if spike {
                let base = (i * 3) % p_len;
                for offset in 0..3 {
                    let idx = (base + offset) % p_len;
                    purkinje_input[idx] += 3.5 * self.ltd_factor[idx];
                }
            }
        }

        // 3. Climbing Fibers (Sinal de Erro)
        for i in 0..p_len {
            self.ltd_factor[i] = (self.ltd_factor[i] + 0.0001).min(1.0);

            if self.error_signal[i].abs() > 0.3 {
                purkinje_input[i] += 18.0;
                self.ltd_factor[i] *= 0.95;
            }
            
            if i < intention.len() {
                purkinje_input[i] += intention[i];
            }
        }

        // 4. Processamento
        let purkinje_spikes = self.purkinje_layer.update(&purkinje_input, dt, current_time);

        // 5. Output Rate (Normalizado)
        purkinje_spikes
            .chunks(10)
            .map(|chunk| {
                let active = chunk.iter().filter(|&&s| s).count() as f32;
                (active / chunk.len() as f32) * 50.0
            })
            .collect()
    }

    pub fn update_error(&mut self, new_error: &[f32]) {
        for (i, &err) in new_error.iter().enumerate().take(self.error_signal.len()) {
            self.error_signal[i] = 0.7 * self.error_signal[i] + 0.3 * err;
        }
    }
    
    pub fn get_purkinje_id(&self, indice: usize) -> Option<Uuid> {
        self.purkinje_ids.get(indice).copied()
    }
    
    pub fn get_granular_id(&self, indice: usize) -> Option<Uuid> {
        self.granular_ids.get(indice).copied()
    }
    
    pub fn estatisticas(&self) -> CerebeloStats {
        CerebeloStats {
            purkinje: self.purkinje_layer.estatisticas(),
            granular: self.granular_layer.estatisticas(),
            ltd_medio: self.ltd_factor.iter().sum::<f32>() / self.ltd_factor.len() as f32,
        }
    }
}

pub struct CerebeloStats {
    pub purkinje: crate::synaptic_core::CamadaStats,
    pub granular: crate::synaptic_core::CamadaStats,
    pub ltd_medio: f32,
}