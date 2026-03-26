// src/brain_zones/mod.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
pub mod frontal;
pub mod occipital;
pub mod parietal;
pub mod temporal;
pub mod limbic;
pub mod hippocampus;
pub mod cerebellum;
pub mod corpus_callosum;
pub mod mirror_neurons;

// Re-exportações
pub use frontal::FrontalLobe;
pub use occipital::OccipitalLobe;
pub use parietal::ParietalLobe;
pub use temporal::TemporalLobe;
pub use limbic::LimbicSystem;
pub use hippocampus::HippocampusV2;
pub use cerebellum::Cerebellum;
pub use corpus_callosum::CorpusCallosum;

use serde::{Serialize, Deserialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionType {
    Frontal,
    Parietal,
    Temporal,
    Occipital,
    Limbic,
    Hippocampus,
    Cerebellum,
    Brainstem,
    CorpusCallosum,
}

pub fn step(_dt: f32, _current_time: f32) {
    // Placeholder
}

impl RegionType {
    pub fn from_usize(value: usize) -> RegionType {
        match value {
            0 => RegionType::Frontal,
            1 => RegionType::Parietal,
            2 => RegionType::Temporal,
            3 => RegionType::Occipital,
            4 => RegionType::Limbic,
            5 => RegionType::Hippocampus,
            6 => RegionType::Cerebellum,
            7 => RegionType::Brainstem,
            8 => RegionType::CorpusCallosum,
            _ => RegionType::Frontal,
        }
    }
    
    pub fn nome(&self) -> &'static str {
        match self {
            RegionType::Frontal => "FRONTAL",
            RegionType::Parietal => "PARIETAL",
            RegionType::Temporal => "TEMPORAL",
            RegionType::Occipital => "OCCIPITAL",
            RegionType::Limbic => "LÍMBICO",
            RegionType::Hippocampus => "HIPOCAMPO",
            RegionType::Cerebellum => "CEREBELO",
            RegionType::Brainstem => "TRONCO",
            RegionType::CorpusCallosum => "CALOSO",
        }
    }
}