// src/telemetry/broadcaster.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use tokio::sync::broadcast;
use crate::brain_zones::*;
use crate::neurochem::NeuroChem;

pub struct Telemetry {
    tx: broadcast::Sender<BrainSnapshot>,
}

impl Telemetry {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(100);
        Self { tx }
    }
    
    pub async fn broadcast_loop(&self, brain: Arc<Mutex<BrainState>>) {
        let mut interval = tokio::time::interval(Duration::from_millis(100));
        loop {
            interval.tick().await;
            let brain = brain.lock().await;
            let snapshot = BrainSnapshot::from(&brain);
            let _ = self.tx.send(snapshot);
        }
    }
}