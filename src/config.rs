// src/config.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use std::time::Duration;

/// Hz de referência para calibração STDP e cap de neurônios (hardware S145)
pub const HZ_REFERENCIA: f32 = 200.0;
/// Limiar LTD padrão (trace_pre abaixo disto → depressão anti-Hebbiana)
pub const BASE_JANELA_STDP: f32 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModoOperacao {
    Humano,      // 100 Hz - Eficiente, padrão
    Boost200,    // 200 Hz - Equilíbrio perfeito
    Boost800,    // 800 Hz - Performance
    Ultra,       // 3200 Hz - Máxima performance
    Insano,      // 6400 Hz - Extreme
    // Modos dinâmicos v2.3 — tier-aware
    Economia,    // 5–50 Hz  | CPU ≤ 25% | apenas em hardware < 16 GB
    Normal,      // 5–100 Hz | CPU ≤ 50% | hardware ≥ 8 GB
    Turbo,       // 200–300 Hz | CPU ≤ 90% | requer RAM ≥ 20 GB + temp < 85 °C
    // V3.4 Fase F — Cenário C (Quiescência) para Ryzen 3500U (4 cores físicos).
    // Vozes Censor/Criativa do Multi-Self rodam apenas 1/4 dos ticks; loop
    // principal cai para 5–15 Hz quando ocioso. Não satura os 4 núcleos.
    Quiescencia, // 5–15 Hz  | CPU ≤ 15% | poupar cores em idle (Ryzen 3500U-class)
}

/// Retorna o Hz alvo para o modo e carga de CPU atual.
/// `carga_cpu` em 0.0–1.0 (ex: 0.5 = 50%).
pub fn hz_alvo(modo: ModoOperacao, carga_cpu: f32) -> u32 {
    match modo {
        // Cenário C — Quiescência: 5–15 Hz quando idle. Ryzen 3500U respira.
        ModoOperacao::Quiescencia => (15.0 * (1.0 - carga_cpu)).max(5.0) as u32,
        ModoOperacao::Economia => (50.0 * (1.0 - carga_cpu)).max(5.0) as u32,
        ModoOperacao::Normal | ModoOperacao::Humano =>
            (100.0 * (1.0 - carga_cpu)).max(5.0) as u32,
        ModoOperacao::Turbo | ModoOperacao::Boost200 =>
            (300.0 * (1.0 - carga_cpu)).max(200.0) as u32,
        ModoOperacao::Boost800 =>
            (800.0 * (1.0 - carga_cpu)).max(200.0) as u32,
        ModoOperacao::Ultra =>
            (3200.0 * (1.0 - carga_cpu)).max(200.0) as u32,
        ModoOperacao::Insano =>
            (6400.0 * (1.0 - carga_cpu)).max(200.0) as u32,
    }
}

/// Escala o limiar LTD do STDP proporcionalmente ao Hz atual.
/// A ticks lentos (Hz baixo), reduz o limiar para evitar
/// anti-depressão excessiva causada por decay de trace entre ticks longos.
///
/// `hz_atual`: frequência real do tick loop (Hz)
/// Retorna o limiar LTD ajustado (usado em `if trace_pre < limiar`)
pub fn janela_stdp_atual(hz_atual: f32) -> f32 {
    BASE_JANELA_STDP * (hz_atual / HZ_REFERENCIA).clamp(0.1, 3.0)
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

            ModoOperacao::Economia => Self {
                modo,
                frequencia_base_hz: 25.0,           // centro da faixa 5–50 Hz
                dt_simulacao: 0.04,                 // 40ms
                energia_watts: 8.0,
                tempo_refratario_ms: 40.0,
                taxa_aprendizado: 0.0005,
                precision_distribution: vec![
                    (crate::synaptic_core::PrecisionType::FP32, 0.02),
                    (crate::synaptic_core::PrecisionType::FP16, 0.28),
                    (crate::synaptic_core::PrecisionType::INT8, 0.60),
                    (crate::synaptic_core::PrecisionType::INT4, 0.10),
                ],
                use_mixed_precision: true,
                swap_threshold_seconds: 7200, // 2 horas
            },

            ModoOperacao::Normal => Self {
                modo,
                frequencia_base_hz: 50.0,           // centro da faixa 5–100 Hz
                dt_simulacao: 0.02,                 // 20ms
                energia_watts: 18.0,
                tempo_refratario_ms: 20.0,
                taxa_aprendizado: 0.001,
                precision_distribution: vec![
                    (crate::synaptic_core::PrecisionType::FP32, 0.04),
                    (crate::synaptic_core::PrecisionType::FP16, 0.36),
                    (crate::synaptic_core::PrecisionType::INT8, 0.50),
                    (crate::synaptic_core::PrecisionType::INT4, 0.10),
                ],
                use_mixed_precision: true,
                swap_threshold_seconds: 3600, // 1 hora
            },

            ModoOperacao::Turbo => Self {
                modo,
                frequencia_base_hz: 250.0,          // centro da faixa 200–300 Hz
                dt_simulacao: 0.004,                // 4ms
                energia_watts: 35.0,
                tempo_refratario_ms: 8.0,
                taxa_aprendizado: 0.003,
                precision_distribution: vec![
                    (crate::synaptic_core::PrecisionType::FP32, 0.04),
                    (crate::synaptic_core::PrecisionType::FP16, 0.36),
                    (crate::synaptic_core::PrecisionType::INT8, 0.50),
                    (crate::synaptic_core::PrecisionType::INT4, 0.10),
                ],
                use_mixed_precision: true,
                swap_threshold_seconds: 1800, // 30 minutos
            },

            // V3.4 Fase F — Cenário C (Quiescência) para Ryzen 3500U-class.
            // Hz baixo, alta proporção INT8/INT4 para reduzir ALU pressure.
            ModoOperacao::Quiescencia => Self {
                modo,
                frequencia_base_hz: 10.0,           // centro da faixa 5–15 Hz
                dt_simulacao: 0.10,                 // 100ms
                energia_watts: 4.0,                 // mínimo absoluto
                tempo_refratario_ms: 100.0,
                taxa_aprendizado: 0.0002,
                precision_distribution: vec![
                    (crate::synaptic_core::PrecisionType::FP32, 0.01),
                    (crate::synaptic_core::PrecisionType::FP16, 0.19),
                    (crate::synaptic_core::PrecisionType::INT8, 0.60),
                    (crate::synaptic_core::PrecisionType::INT4, 0.20),
                ],
                use_mixed_precision: true,
                swap_threshold_seconds: 14400, // 4 horas — pouca atividade, swap raro
            },
        }
    }

    pub fn fator_boost(&self) -> f32 {
        match self.modo {
            ModoOperacao::Humano  => 1.0,
            ModoOperacao::Quiescencia => 0.10, // mínimo absoluto — tick raro
            ModoOperacao::Economia => 0.25,
            ModoOperacao::Normal  => 0.5,
            ModoOperacao::Boost200 | ModoOperacao::Turbo => 2.0,
            ModoOperacao::Boost800 => 8.0,
            ModoOperacao::Ultra   => 32.0,
            ModoOperacao::Insano  => 64.0,
        }
    }
}