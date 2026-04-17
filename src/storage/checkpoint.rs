// src/storage/checkpoint.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
pub struct CheckpointSystem {
    last_save: Instant,
    auto_save_interval: Duration,
}

impl CheckpointSystem {
    pub async fn auto_save(&mut self, brain: &BrainState) {
        if self.last_save.elapsed() > self.auto_save_interval {
            self.save(brain).await;
        }
    }
}