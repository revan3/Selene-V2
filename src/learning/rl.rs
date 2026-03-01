// src/learning/rl.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
pub struct ReinforcementLearning {
    reward_history: VecDeque<f32>,
    policy: HashMap<(RegionType, RegionType), f32>,
}