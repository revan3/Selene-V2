// src/brain_zones/frontal.rs
// Córtex Pré-Frontal — Decisão executiva, working memory, controle inibitório
//
// Composição neuronal:
//   executive_layer: 80% RS (decisão) + 20% FS (inibição lateral)
//   inhibitory_layer: 100% FS (interneurônios GABAérgicos)
//
// O FS (Fast Spiking) implementa a inibição lateral real:
// quando muitos executivos disparam juntos, os FS são ativados
// e reduzem o potencial dos vizinhos — evitando hiperativação.

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::synaptic_core::{CamadaHibrida, PrecisionType, TipoNeuronal};
use rand::{Rng, thread_rng};
use crate::config::Config;

#[derive(Debug)]
pub struct FrontalLobe {
    /// Camada executiva: RS (decisão) + FS (auto-inibição)
    pub executive_layer: CamadaHibrida,
    /// Camada inibitória pura: FS GABAérgico
    pub inhibitory_layer: CamadaHibrida,
    pub dopamine_level: f32,
    pub working_memory_trace: Vec<f32>,
    pub inhibition_strength: f32,
    pub noise_std: f32,
}

impl FrontalLobe {
    pub fn new(n_executive: usize, inhibition_ratio: f32, noise_std: f32, config: &Config) -> Self {
        let n_inhib = (n_executive as f32 * inhibition_ratio).max(1.0) as usize;

        // Executivos: FP16 dominante para working memory precisa, 20% FS para auto-inibição
        let exec_dist = vec![
            (PrecisionType::FP32, 0.05),
            (PrecisionType::FP16, 0.55),
            (PrecisionType::INT8, 0.30),
            (PrecisionType::INT4, 0.10),
        ];
        // Inibitórios: INT8 — alta densidade, precisão não crítica
        let inhib_dist = vec![
            (PrecisionType::FP16, 0.20),
            (PrecisionType::INT8, 0.60),
            (PrecisionType::INT4, 0.20),
        ];

        // Escala para correntes típicas do frontal (~50pA)
        let escala = 50.0 / 127.0;

        let executive_layer = CamadaHibrida::new(
            n_executive, "frontal_exec",
            TipoNeuronal::RS,
            Some((TipoNeuronal::FS, 0.20)), // 20% FS para inibição lateral
            Some(exec_dist),
            escala,
        );
        let inhibitory_layer = CamadaHibrida::new(
            n_inhib, "frontal_inhib",
            TipoNeuronal::FS, // 100% Fast Spiking GABAérgico
            None,
            Some(inhib_dist),
            escala,
        );

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

        // Dopamina modula o ganho: alta dopamina → mais responsivo a objetivos
        let gain = 1.0 + self.dopamine_level * 0.8;
        // current_time em ms para STDP
        let t_ms = current_time * 1000.0;

        for i in 0..n {
            let noise = rng.gen_range(-self.noise_std..self.noise_std);
            combined_input[i] = (
                sensory_input.get(i).copied().unwrap_or(0.0)
                + goal_bias.get(i).copied().unwrap_or(0.0) * 1.5
                + self.working_memory_trace[i]
                + noise
            ) * gain;
        }

        let executive_spikes = self.executive_layer.update(&combined_input, dt, t_ms);

        // Inibição lateral: conta quantos executivos dispararam e ativa os FS proporcionalmente
        let active_count = executive_spikes.iter().filter(|&&s| s).count() as f32;
        let inhibition_input = vec![active_count * 0.4; self.inhibitory_layer.neuronios.len()];
        let inhibitory_spikes = self.inhibitory_layer.update(&inhibition_input, dt, t_ms);

        let mut output_voltages = vec![0.0; n];
        let n_inhib = self.inhibitory_layer.neuronios.len();

        for i in 0..n {
            // Atualiza working memory trace
            if executive_spikes[i] {
                self.working_memory_trace[i] = 25.0 * self.dopamine_level;
            }
            self.working_memory_trace[i] *= 0.96;

            // FS dispara → hiperpolariza o executivo correspondente
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
