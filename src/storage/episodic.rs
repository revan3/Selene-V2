// src/storage/episodic.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
pub struct EpisodicMemory {
    episodes: Vec<Episode>,
    temporal_context: VecDeque<Event>,
}