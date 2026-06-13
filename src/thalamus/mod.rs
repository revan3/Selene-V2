// src/thalamus/mod.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use crate::config::Config;
use crate::synaptic_core::{CamadaHibrida, TipoNeuronal};

/// Tálamo: gateway sensorial que filtra inputs baseado em arousal
pub struct Thalamus {
    pub filter_strength: f32,   // 0.0 a 1.0 – quão seletivo
    pub attention_gate: f32,     // vindo do límbico (arousal)
    pub history: Vec<f32>,       // histórico recente para adaptação
    // ── V4.6.1: substrato spiking real — relé Thalamo-Cortical (TC, HH) ──────
    // TC tem casa anatômica aqui (antes só em occipital como proxy). I_T Ca²⁺ →
    // burst (sono) ↔ tônico (vigília). Roda junto ao filtro abstrato.
    pub relay_layer: CamadaHibrida,
    tick_ms: f32,
}

impl Thalamus {
    pub fn new() -> Self {
        Self {
            filter_strength: 0.5,
            attention_gate: 1.0,
            history: Vec::with_capacity(100),
            relay_layer: CamadaHibrida::new(
                64, "thalamus_tc_relay", TipoNeuronal::TC, None, None, 1.0,
            ),
            tick_ms: 0.0,
        }
    }

    /// Taxa de disparo do relé TC (fração de neurônios ativos no último tick).
    pub fn tc_spike_rate(&self) -> f32 {
        self.relay_layer.estatisticas_v3().spike_rate
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

        // V4.6.1 — alimenta o relé spiking TC com o input filtrado (escala p/ pA).
        self.tick_ms += config.dt_simulacao * 1000.0;
        let drive: Vec<f32> = filtered.iter().take(64).map(|&v| v * 18.0).collect();
        let _ = self.relay_layer.update(&drive, config.dt_simulacao, self.tick_ms);

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