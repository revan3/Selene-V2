// src/thalamus/mod.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use crate::config::Config;

/// Tálamo: gateway sensorial que filtra inputs baseado em arousal
pub struct Thalamus {
    pub filter_strength: f32,   // 0.0 a 1.0 – quão seletivo
    pub attention_gate: f32,     // vindo do límbico (arousal)
    pub history: Vec<f32>,       // histórico recente para adaptação
}

impl Thalamus {
    pub fn new() -> Self {
        Self {
            filter_strength: 0.5,
            attention_gate: 1.0,
            history: Vec::with_capacity(100),
        }
    }
    
    /// Filtra e roteia input sensorial baseado no estado de alerta
    pub fn relay(&mut self, sensory_input: &[f32], arousal: f32, config: &Config) -> Vec<f32> {
        self.attention_gate = arousal;
        
        // Aplica filtro: inputs menos relevantes são atenuados
        let filtered: Vec<f32> = sensory_input.iter()
            .map(|&v| v * self.attention_gate * self.filter_strength)
            .collect();
            
        // Guarda histórico para adaptação
        self.history.extend(filtered.iter().copied());
        if self.history.len() > 1000 {
            self.history.drain(0..500);
        }
        
        filtered
    }
    
    /// Ajusta a seletividade do tálamo baseado em erro (aprendizado)
    pub fn adapt_filter(&mut self, error_signal: f32) {
        self.filter_strength = (self.filter_strength + error_signal * 0.01)
            .clamp(0.1, 1.0);
    }
    
    /// Estatísticas de filtragem
    pub fn stats(&self) -> ThalamusStats {
        let media = if !self.history.is_empty() {
            self.history.iter().sum::<f32>() / self.history.len() as f32
        } else {
            0.0
        };
        
        ThalamusStats {
            filter_strength: self.filter_strength,
            attention_gate: self.attention_gate,
            avg_input: media,
        }
    }
}

pub struct ThalamusStats {
    pub filter_strength: f32,
    pub attention_gate: f32,
    pub avg_input: f32,
}