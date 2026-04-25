// src/lib.rs
#![allow(unused)]

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
pub mod glia;
pub mod synthesis;

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

