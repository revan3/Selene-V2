// src/brain_zones/hippocampus.rs
// Hipocampo — Memória episódica, consolidação, navegação espacial
//
// Composição neuronal:
//   ca1_encoding: 80% RS + 20% LT — encoding com neurônios de baixo limiar
//   ca3_recurrent: 70% RS + 30% RZ — recorrência com detecção de padrões rítmicos (ondas theta)
//
// MUDANÇA: removido HashMap<(usize, String), Uuid> por neurônio.
// IDs agora são u32 simples gerados pela CamadaHibrida. Economia: ~40KB para 512 neurônios.
// ConexaoSinaptica usa ID composto (camada + índice) em vez de UUID por instância.

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::synaptic_core::{CamadaHibrida, PrecisionType, TipoNeuronal};
use rand::{Rng, thread_rng};
use crate::storage::ConexaoSinaptica;
use uuid::Uuid;
use crate::brain_zones::RegionType;
use crate::config::{Config, ModoOperacao};
use chrono;

#[derive(Debug)]
pub struct HippocampusV2 {
    pub ca1_encoding: CamadaHibrida,
    pub ca3_recurrent: CamadaHibrida,
    pub ltp_matrix: Vec<f32>,
    pub consolidation_rate: f32,
    pub prev_ca3_spikes: Vec<bool>,
    pub theta_phase: f32,          // fase da onda theta (~8Hz)
    conexoes_recentes: Vec<ConexaoSinaptica>,
}

impl HippocampusV2 {
    pub fn new(n_neurons: usize, config: &Config) -> Self {
        let n_sub = (n_neurons / 2).max(1);

        // CA1: FP16 dominante para encoding preciso, LT para baixo limiar
        let ca1_dist = vec![
            (PrecisionType::FP32, 0.05),
            (PrecisionType::FP16, 0.60),
            (PrecisionType::INT8, 0.35),
        ];

        // CA3: INT8 para recorrência em massa, RZ para ondas theta
        let ca3_dist = vec![
            (PrecisionType::FP16, 0.20),
            (PrecisionType::INT8, 0.70),
            (PrecisionType::INT4, 0.10),
        ];

        let escala = 40.0 / 127.0;

        let ca1 = CamadaHibrida::new(
            n_sub, "hipocampo_ca1",
            TipoNeuronal::RS,
            Some((TipoNeuronal::LT, 0.20)),
            Some(ca1_dist),
            escala,
        );
        let ca3 = CamadaHibrida::new(
            n_sub, "hipocampo_ca3",
            TipoNeuronal::RS,
            Some((TipoNeuronal::RZ, 0.30)), // RZ para ondas theta
            Some(ca3_dist),
            escala,
        );

        Self {
            ca1_encoding: ca1,
            ca3_recurrent: ca3,
            ltp_matrix: vec![0.5; n_sub],
            consolidation_rate: 0.01,
            prev_ca3_spikes: vec![false; n_sub],
            theta_phase: 0.0,
            conexoes_recentes: Vec::with_capacity(1000),
        }
    }

    pub fn memorize_with_connections(
        &mut self,
        pattern_in: &[f32],
        emotional_weight: f32,
        dt: f32,
        current_time: f32,
        config: &Config,
    ) -> (Vec<f32>, Vec<ConexaoSinaptica>) {
        let n = self.ca1_encoding.neuronios.len();
        let t_ms = current_time * 1000.0;
        let mut conexoes = Vec::new();

        let ca1_in: Vec<f32> = (0..n)
            .map(|i| pattern_in.get(i).copied().unwrap_or(0.0) * self.ltp_matrix[i])
            .collect();

        let ca1_spikes = self.ca1_encoding.update(&ca1_in, dt, t_ms);

        // Cria conexões CA1→CA3 (índices compactos, sem UUID por neurônio)
        for i in 0..n {
            if ca1_spikes[i] {
                for j in 0..(n / 10) {
                    if rand::random::<f32>() > 0.7 {
                        // UUID gerado apenas para a *conexão*, não para o neurônio
                        let conexao = ConexaoSinaptica {
                            id: Uuid::new_v4(),
                            de_neuronio: Uuid::from_u128(i as u128),   // índice como UUID
                            para_neuronio: Uuid::from_u128(j as u128),
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

        // Atualiza onda theta e entrada recorrente de CA3
        self.theta_phase = (self.theta_phase + dt * 8.0 * 2.0 * std::f32::consts::PI) % (2.0 * std::f32::consts::PI);
        let theta_mod = (self.theta_phase.sin() * 0.5 + 0.5).max(0.0);

        let ca3_in: Vec<f32> = (0..n)
            .map(|i| {
                let recurrent = if self.prev_ca3_spikes[i] { 6.0 } else { 0.0 };
                let from_ca1 = if ca1_spikes[i] { 20.0 } else { 0.0 };
                (recurrent + from_ca1) * theta_mod
            })
            .collect();

        let ca3_spikes = self.ca3_recurrent.update(&ca3_in, dt, t_ms);

        let output: Vec<f32> = (0..n).map(|i| {
            self.prev_ca3_spikes[i] = ca3_spikes[i];
            if ca3_spikes[i] { 1.0 } else { 0.0 }
        }).collect();

        self.conexoes_recentes.extend(conexoes.clone());
        (output, conexoes)
    }

    pub fn memorize(&mut self, pattern_in: &[f32], emotional_weight: f32, dt: f32) -> Vec<f32> {
        let (out, _) = self.memorize_with_connections(
            pattern_in, emotional_weight, dt, 0.0,
            &Config::new(ModoOperacao::Humano),
        );
        out
    }

    pub fn consolidate_recent(&mut self) -> Vec<ConexaoSinaptica> {
        std::mem::take(&mut self.conexoes_recentes)
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
