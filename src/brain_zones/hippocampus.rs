// src/brain_zones/hippocampus.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use crate::synaptic_core::{CamadaHibrida, PrecisionType};
use rand::{Rng, thread_rng};
use crate::storage::ConexaoSinaptica;
use uuid::Uuid;
use crate::brain_zones::RegionType;
use crate::config::{Config, ModoOperacao};
use chrono;
use std::collections::HashMap;

#[derive(Debug)]
pub struct HippocampusV2 {
    pub ca1_encoding: CamadaHibrida,
    pub ca3_recurrent: CamadaHibrida,
    pub ltp_matrix: Vec<f32>,
    pub consolidation_rate: f32,
    pub prev_ca3_spikes: Vec<bool>,
    pub theta_phase: f32,
    conexoes_recentes: Vec<ConexaoSinaptica>,
    // Mapa para manter IDs consistentes dos neurônios
    neuron_ids: HashMap<(usize, String), Uuid>, // (índice, "CA1" ou "CA3") -> UUID
}

impl HippocampusV2 {
    pub fn new(n_neurons: usize, config: &Config) -> Self {
        let n_sub = n_neurons / 2;
        
        // Distribuição específica para hipocampo (mais FP16 para plasticidade)
        let mut ca1_dist = config.precision_distribution.clone();
        ca1_dist.push((PrecisionType::FP16, 0.60)); // 60% FP16 para CA1 (encoding)
        
        let mut ca3_dist = config.precision_distribution.clone();
        ca3_dist.push((PrecisionType::INT8, 0.70)); // 70% INT8 para CA3 (recorrência)
        
        let ca1 = CamadaHibrida::new(n_sub, "hipocampo_ca1", Some(ca1_dist));
        let ca3 = CamadaHibrida::new(n_sub, "hipocampo_ca3", Some(ca3_dist));

        // Gerar IDs consistentes para todos os neurônios
        let mut neuron_ids = HashMap::new();
        for i in 0..n_sub {
            neuron_ids.insert((i, "CA1".to_string()), Uuid::new_v4());
            neuron_ids.insert((i, "CA3".to_string()), Uuid::new_v4());
        }

        Self {
            ca1_encoding: ca1,
            ca3_recurrent: ca3,
            ltp_matrix: vec![0.5; n_sub],
            consolidation_rate: 0.01,
            prev_ca3_spikes: vec![false; n_sub],
            theta_phase: 0.0,
            conexoes_recentes: Vec::with_capacity(1000),
            neuron_ids,
        }
    }

    // Método para obter ID consistente de um neurônio
    fn get_neuron_id(&self, indice: usize, regiao: &str) -> Uuid {
        *self.neuron_ids.get(&(indice, regiao.to_string())).unwrap_or(&Uuid::nil())
    }

    pub fn memorize_with_connections(&mut self, pattern_in: &[f32], emotional_weight: f32, dt: f32, current_time: f32, config: &Config) 
        -> (Vec<f32>, Vec<ConexaoSinaptica>) {
        
        let n = self.ca1_encoding.neuronios.len();
        let mut conexoes = Vec::new();
        
        let mut ca1_in = vec![0.0; n];
        for i in 0..n {
            ca1_in[i] = pattern_in.get(i).cloned().unwrap_or(0.0) * self.ltp_matrix[i];
        }
        
        let ca1_spikes = self.ca1_encoding.update(&ca1_in, dt, current_time);
        
        for i in 0..n {
            if ca1_spikes[i] {
                for j in 0..n/10 {
                    if rand::random::<f32>() > 0.7 {
                        let de_id = self.get_neuron_id(i, "CA1");
                        let para_id = self.get_neuron_id(j, "CA3");
                        
                        let conexao = ConexaoSinaptica {
                            id: Uuid::new_v4(),
                            de_neuronio: de_id,
                            para_neuronio: para_id,
                            peso: emotional_weight * rand::random::<f32>(),
                            criada_em: current_time as f64,
                            ultimo_uso: Some(chrono::Utc::now().timestamp() as f64),
                            total_usos: 1,
                            emocao_media: emotional_weight,
                            contexto_criacao: None,
                        };
                        conexoes.push(conexao);
                    }
                }
            }
        }
        
        let mut ca3_in = vec![0.0; n];
        for i in 0..n {
            if self.prev_ca3_spikes[i] {
                ca3_in[i] += 6.0;
            }
            if ca1_spikes[i] {
                ca3_in[i] += 20.0;
            }
        }
        
        let ca3_spikes = self.ca3_recurrent.update(&ca3_in, dt, current_time);
        
        let mut output = vec![0.0; n];
        for i in 0..n {
            output[i] = if ca3_spikes[i] { 1.0 } else { 0.0 };
            self.prev_ca3_spikes[i] = ca3_spikes[i];
        }
        
        self.conexoes_recentes.extend(conexoes.clone());
        
        (output, conexoes)
    }
    
    pub fn memorize(&mut self, pattern_in: &[f32], emotional_weight: f32, dt: f32) -> Vec<f32> {
        let (output, _) = self.memorize_with_connections(pattern_in, emotional_weight, dt, 0.0, &Config::new(ModoOperacao::Humano));
        output
    }
    
    pub fn consolidate_recent(&mut self) -> Vec<ConexaoSinaptica> {
        let conexoes = self.conexoes_recentes.clone();
        self.conexoes_recentes.clear();
        conexoes
    }
    
    pub fn estatisticas(&self) -> HipocampoStats {
        HipocampoStats {
            ca1: self.ca1_encoding.estatisticas(),
            ca3: self.ca3_recurrent.estatisticas(),
            conexoes_recentes: self.conexoes_recentes.len(),
        }
    }
}

pub struct HipocampoStats {
    pub ca1: crate::synaptic_core::CamadaStats,
    pub ca3: crate::synaptic_core::CamadaStats,
    pub conexoes_recentes: usize,
}