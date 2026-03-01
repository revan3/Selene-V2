// src/config.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModoOperacao {
    Humano,      // 100 Hz - Eficiente, padrão
    Boost200,    // 200 Hz - Equilíbrio perfeito
    Boost800,    // 800 Hz - Performance
    Ultra,       // 3200 Hz - Máxima performance
    Insano,      // 6400 Hz - Extreme
}

#[derive(Debug, Clone)]
pub struct Config {
    pub modo: ModoOperacao,
    pub frequencia_base_hz: f32,
    pub dt_simulacao: f32,
    pub energia_watts: f32,
    pub tempo_refratario_ms: f32,
    pub taxa_aprendizado: f32,
    
    // Parâmetros para neurônios híbridos
    pub precision_distribution: Vec<(crate::synaptic_core::PrecisionType, f32)>,
    pub use_mixed_precision: bool,
    pub swap_threshold_seconds: u64,  // Tempo sem uso para swap
}

impl Config {
    pub fn new(modo: ModoOperacao) -> Self {
        match modo {
            ModoOperacao::Humano => Self {
                modo,
                frequencia_base_hz: 100.0,
                dt_simulacao: 0.01,                 // 10ms
                energia_watts: 15.0,
                tempo_refratario_ms: 20.0,
                taxa_aprendizado: 0.001,
                precision_distribution: vec![
                    (crate::synaptic_core::PrecisionType::FP32, 0.05),
                    (crate::synaptic_core::PrecisionType::FP16, 0.35),
                    (crate::synaptic_core::PrecisionType::INT8, 0.50),
                    (crate::synaptic_core::PrecisionType::INT4, 0.10),
                ],
                use_mixed_precision: true,
                swap_threshold_seconds: 3600, // 1 hora
            },
            
            // NOVO MODO: 200Hz - Equilíbrio perfeito
            ModoOperacao::Boost200 => Self {
                modo,
                frequencia_base_hz: 200.0,
                dt_simulacao: 0.005,                // 5ms = 200Hz ✓
                energia_watts: 25.0,                 // Consumo moderado
                tempo_refratario_ms: 10.0,            // Metade do humano
                taxa_aprendizado: 0.002,              // 2x mais rápido
                precision_distribution: vec![
                    (crate::synaptic_core::PrecisionType::FP32, 0.04),
                    (crate::synaptic_core::PrecisionType::FP16, 0.36),
                    (crate::synaptic_core::PrecisionType::INT8, 0.50),
                    (crate::synaptic_core::PrecisionType::INT4, 0.10),
                ],
                use_mixed_precision: true,
                swap_threshold_seconds: 2700, // 45 minutos
            },
            
            ModoOperacao::Boost800 => Self {
                modo,
                frequencia_base_hz: 800.0,
                dt_simulacao: 0.00125,              // 1.25ms
                energia_watts: 45.0,
                tempo_refratario_ms: 2.5,
                taxa_aprendizado: 0.008,
                precision_distribution: vec![
                    (crate::synaptic_core::PrecisionType::FP32, 0.03),
                    (crate::synaptic_core::PrecisionType::FP16, 0.27),
                    (crate::synaptic_core::PrecisionType::INT8, 0.60),
                    (crate::synaptic_core::PrecisionType::INT4, 0.10),
                ],
                use_mixed_precision: true,
                swap_threshold_seconds: 1800, // 30 minutos
            },
            
            ModoOperacao::Ultra => Self {
                modo,
                frequencia_base_hz: 3200.0,
                dt_simulacao: 0.0003125,            // 0.3125ms
                energia_watts: 80.0,
                tempo_refratario_ms: 0.625,
                taxa_aprendizado: 0.032,
                precision_distribution: vec![
                    (crate::synaptic_core::PrecisionType::FP32, 0.01),
                    (crate::synaptic_core::PrecisionType::FP16, 0.19),
                    (crate::synaptic_core::PrecisionType::INT8, 0.70),
                    (crate::synaptic_core::PrecisionType::INT4, 0.10),
                ],
                use_mixed_precision: true,
                swap_threshold_seconds: 900, // 15 minutos
            },
            
            ModoOperacao::Insano => Self {
                modo,
                frequencia_base_hz: 6400.0,
                dt_simulacao: 0.00015625,           // 0.15625ms
                energia_watts: 120.0,
                tempo_refratario_ms: 0.3125,
                taxa_aprendizado: 0.064,
                precision_distribution: vec![
                    (crate::synaptic_core::PrecisionType::FP32, 0.005),
                    (crate::synaptic_core::PrecisionType::FP16, 0.095),
                    (crate::synaptic_core::PrecisionType::INT8, 0.80),
                    (crate::synaptic_core::PrecisionType::INT4, 0.10),
                ],
                use_mixed_precision: true,
                swap_threshold_seconds: 300, // 5 minutos
            },
        }
    }
    
    pub fn fator_boost(&self) -> f32 {
        match self.modo {
            ModoOperacao::Humano => 1.0,
            ModoOperacao::Boost200 => 2.0,   // 2x humano
            ModoOperacao::Boost800 => 8.0,
            ModoOperacao::Ultra => 32.0,
            ModoOperacao::Insano => 64.0,
        }
    }
}