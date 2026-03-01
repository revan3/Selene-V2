// src/neurochem.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use crate::sensors::hardware::HardwareSensor;
use crate::config::{Config, ModoOperacao};

pub struct NeuroChem {
    pub serotonin: f32,
    pub dopamine: f32,
    pub cortisol: f32,
    pub noradrenaline: f32,
    last_temp: f32,
}

// Adicionar em neurochem.rs
pub struct EmotionalState {
    // Roda de emoções de Plutchik
    joy: f32, trust: f32, fear: f32, surprise: f32,
    sadness: f32, disgust: f32, anger: f32, anticipation: f32,
}

impl NeuroChem {
    pub fn new() -> Self {
        Self {
            serotonin: 1.0,
            dopamine: 1.0,
            cortisol: 0.0,
            noradrenaline: 0.5,
            last_temp: 0.0,
        }
    }

    pub fn update(&mut self, sensor: &mut HardwareSensor, config: &Config) {
        let jitter = sensor.get_jitter_ms();
        let switches = sensor.get_context_switches_per_sec();
        let ram_usage = sensor.get_ram_usage();
        let temp = sensor.get_cpu_temp();
        let delta_temp = (temp - self.last_temp).abs();
        self.last_temp = temp;

        let fator_tempo = config.fator_boost();
        let decay_rate = 0.01 / fator_tempo;

        // Serotonina
        let switches_penalty = (switches / 6000.0).clamp(0.0, 1.0);
        let jitter_penalty = (jitter / 3.0).clamp(0.0, 1.0);
        let target_sero = (1.0 - 0.5 * (switches_penalty + jitter_penalty)).clamp(0.1f32, 1.0f32);
        self.serotonin += (target_sero - self.serotonin) * decay_rate;

        // Dopamina - CORRIGIDO: todos os modos cobertos
        let target_dopa = match config.modo {
            ModoOperacao::Humano => (ram_usage / 100.0).clamp(0.5, 1.0),
            ModoOperacao::Boost200 => (ram_usage / 90.0).clamp(0.6, 1.2),
            ModoOperacao::Boost800 => (ram_usage / 80.0).clamp(0.7, 1.5),
            ModoOperacao::Ultra => (ram_usage / 70.0).clamp(0.8, 1.8),
            ModoOperacao::Insano => (ram_usage / 60.0).clamp(0.9, 2.0),
        };
        self.dopamine += (target_dopa - self.dopamine) * decay_rate;

        // Cortisol
        self.cortisol = (delta_temp / 5.0).clamp(0.0, 1.0);

        // Noradrenalina - CORRIGIDO: todos os modos cobertos
        self.noradrenaline = match config.modo {
            ModoOperacao::Humano => (temp / 100.0).clamp(0.5, 1.0),
            ModoOperacao::Boost200 => (temp / 95.0).clamp(0.6, 1.1),
            ModoOperacao::Boost800 => (temp / 85.0).clamp(0.8, 1.2),
            ModoOperacao::Ultra => (temp / 75.0).clamp(0.9, 1.4),
            ModoOperacao::Insano => (temp / 65.0).clamp(1.0, 1.6),
        };
    }
}