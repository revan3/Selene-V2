// src/encoding/mod.rs
//
// Text ↔ spike encoding pipeline for Selene.
//
//  spike_codec   — word → SpikePattern ([u64;8]) encoder/decoder
//  helix_store   — mmap-based persistent spike vocabulary (HelixStore)
//  phoneme       — PT-BR G2P + formant parameter table

pub mod spike_codec;
pub mod helix_store;
pub mod phoneme;
pub mod fonetico;
pub mod fft_encoder;
pub mod espectro_visual;

// Convenient re-exports
pub use spike_codec::{
    SpikePattern,
    N_NEURONS,
    K_ACTIVE,
    encode,
    similarity,
    popcount,
    decode,
    decode_top_n,
    superimpose,
    intersect,
};

pub use helix_store::HelixStore;

pub use phoneme::{
    Phoneme,
    FormantParams,
    formant_table,
    word_to_phonemes,
    sentence_to_formants,
};
