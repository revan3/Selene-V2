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
pub mod depth_stack;
pub mod cingulate;
pub mod orbitofrontal;
pub mod language;
pub mod amygdala;
// V4.3 — memória episódica avançada (não substitui HippocampusV2; complementa)
pub mod dentate_gyrus;
pub mod ca3_attractor;
pub mod memory_engrams;
pub mod hippocampal_index;

// Re-exportações
pub use frontal::FrontalLobe;
pub use occipital::OccipitalLobe;
pub use parietal::ParietalLobe;
pub use temporal::TemporalLobe;
pub use limbic::LimbicSystem;
pub use hippocampus::HippocampusV2;
pub use cerebellum::Cerebellum;
pub use corpus_callosum::CorpusCallosum;
pub use cingulate::AnteriorCingulate;
pub use orbitofrontal::OrbitalFrontal;
pub use language::LanguageAreas;
pub use amygdala::Amygdala;
// V4.3 re-exports
pub use dentate_gyrus::{DentateGyrus, SparsePattern};
pub use ca3_attractor::CA3Attractor;
pub use memory_engrams::{EngramStore, Engram, EngramId};
pub use hippocampal_index::{HippocampalIndex, HippocampalIndexConfig, HippocampalIndexStats};

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
    Amygdala,
    Cingulate,
    OrbitofrontalCortex,
    Thalamus,
    Striatum,
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
            9 => RegionType::Amygdala,
            10 => RegionType::Cingulate,
            11 => RegionType::OrbitofrontalCortex,
            12 => RegionType::Thalamus,
            13 => RegionType::Striatum,
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
            RegionType::Amygdala => "AMÍGDALA",
            RegionType::Cingulate => "CINGULADO",
            RegionType::OrbitofrontalCortex => "OFC",
            RegionType::Thalamus => "TÁLAMO",
            RegionType::Striatum => "ESTRATO",
        }
    }
}