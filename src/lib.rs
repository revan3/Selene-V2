// src/lib.rs
#![allow(unused)]

use pyo3::prelude::*;
use std::time::Instant;

// Módulos existentes
pub mod synaptic_core;
pub mod brain_zones;
pub mod sensors;
pub mod storage;
pub mod neurochem;
pub mod sleep_manager;
pub mod config;
pub mod sleep_cycle;
pub mod websocket;
pub mod compressor;
pub mod ego;
pub mod encoding;
pub mod thalamus;
pub mod interoception;
pub mod basal_ganglia;
pub mod brainstem;
pub mod learning;
pub mod meta;

// Re-exportações
pub use sleep_manager::SleepManagerV2 as SleepManager;
pub use storage::NeuralEnactiveMemory;
pub use storage::NeuralEnactiveMemoryV2;
pub use storage::firing_rates_to_spike_bits;
pub use storage::spike_bits_to_firing_rates;
pub use storage::backup_to_hdd;
pub use storage::exportar_linguagem;

// Brain zones
pub use brain_zones::frontal::FrontalLobe;
pub use brain_zones::occipital::OccipitalLobe;
pub use brain_zones::parietal::ParietalLobe;
pub use brain_zones::temporal::TemporalLobe;
pub use brain_zones::limbic::LimbicSystem;
pub use brain_zones::hippocampus::HippocampusV2;
pub use brain_zones::cerebellum::Cerebellum;
pub use brain_zones::corpus_callosum::CorpusCallosum;

// Sensors
pub use sensors::hardware::HardwareSensor;
pub use sensors::audio;
pub use sensors::camera::VisualTransducer;

// Storage
pub use storage::BrainStorage;
pub use storage::memory_tier::MemoryTier;
pub use storage::swap_manager::SwapManager;

// Neurochem
pub use neurochem::NeuroChem;

// Estado global para o kernel Python
static mut LAST_CHEM_STATE: Option<Vec<Vec<f64>>> = None;
static mut LAST_TICK_DURATION: f64 = 0.0;

/// Função principal do kernel Python
#[pyfunction]
pub fn process_brain_cycle(
    visual_input: Vec<f64>,
    cpu_temp: f64,
    cpu_freq_rel: f64,
    ram_usage: f64,
    user_threshold: f64 
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>, f64, f64)> {
    
    let start_tick = Instant::now();
    
    // Placeholder - implementação real virá depois
    let v_final = vec![0.0; visual_input.len()];
    let s_fro = vec![0.0; 100];
    let novo_fluxo = vec![vec![0.5, 0.5, 0.5]; 9];
    
    unsafe { 
        LAST_TICK_DURATION = start_tick.elapsed().as_micros() as f64;
    }

    Ok((v_final, s_fro, novo_fluxo, unsafe { LAST_TICK_DURATION }, 0.0))
}

/// Função de saudação
#[pyfunction]
fn hello_from_rust() -> String {
    "Olá da Selene via Rust!".to_string()
}

#[pymodule]
fn selene_kernel(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_rust, m)?)?;
    m.add_function(wrap_pyfunction!(process_brain_cycle, m)?)?;
    Ok(())
}