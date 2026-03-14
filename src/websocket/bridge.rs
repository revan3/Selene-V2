// src/websocket/bridge.rs

use serde::{Serialize, Deserialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;
use crate::config::Config;
use crate::storage::swap_manager::SwapManager;

// Estrutura principal que representa o "Estado Mental" para a Web
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NeuralStatus {
    pub neurotransmissores: NeurochemStatus,
    pub hardware: HardwareStatus,
    pub ego: EgoStatus,
    pub atividade: AtividadeStatus,
    pub swap: SwapStatus,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NeurochemStatus {
    pub dopamina: f32,
    pub serotonina: f32,
    pub noradrenalina: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HardwareStatus {
    pub cpu_temp: f32,
    pub ram_usage_gb: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EgoStatus {
    pub pensamentos: Vec<String>,
    pub sentimento_atual: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AtividadeStatus {
    pub step: u64,
    pub alerta: f32,
    pub emocao: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SwapStatus {
    pub neuronios_ativos: usize,
    pub capacidade_max: usize,
}

// O Objeto de sincronização que o main.rs usa
pub struct BrainState {
    pub swap_manager: Arc<TokioMutex<SwapManager>>,
    pub config: Config,
    pub neurotransmissores: (f32, f32, f32), // Dop, Sero, Nor
    pub hardware: (f32, f32),               // Temp, RAM
    pub atividade: (u64, f32, f32),         // Step, Alerta, Emoção
    pub ego: EgoVoiceState,
}

pub struct EgoVoiceState {
    pub pensamentos_recentes: VecDeque<String>,
    pub sentimento: f32,
}

impl BrainState {
    pub fn new(swap: Arc<TokioMutex<SwapManager>>, cfg: &Config) -> Self {
        Self {
            swap_manager: swap,
            config: cfg.clone(),
            neurotransmissores: (0.5, 0.5, 0.5),
            hardware: (40.0, 0.0),
            atividade: (0, 1.0, 0.0),
            ego: EgoVoiceState {
                pensamentos_recentes: VecDeque::with_capacity(10),
                sentimento: 0.0,
            },
        }
    }
}

// Esta função traduz o BrainState complexo para o NeuralStatus (JSON)
pub async fn collect_neural_status(state: &BrainState) -> NeuralStatus {
    let swap_guard = state.swap_manager.lock().await;
    
    NeuralStatus {
        neurotransmissores: NeurochemStatus {
            dopamina: state.neurotransmissores.0,
            serotonina: state.neurotransmissores.1,
            noradrenalina: state.neurotransmissores.2,
        },
        hardware: HardwareStatus {
            cpu_temp: state.hardware.0,
            ram_usage_gb: state.hardware.1,
        },
        ego: EgoStatus {
            pensamentos: state.ego.pensamentos_recentes.iter().cloned().collect(),
            sentimento_atual: state.ego.sentimento,
        },
        atividade: AtividadeStatus {
            step: state.atividade.0,
            alerta: state.atividade.1,
            emocao: state.atividade.2,
        },
        swap: SwapStatus {
            neuronios_ativos: swap_guard.ram_count(),
            capacidade_max: 1_000_000, // MAX_RAM_NEURONS
        },
    }
}