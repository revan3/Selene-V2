// src/storage/memory_tier.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use memmap2::MmapMut;
use std::fs::{File, OpenOptions};
use std::path::Path;
use crate::storage::BrainStorage;
use crate::storage::NeuralEnactiveMemory;
use std::sync::Arc;
use crate::brain_zones::RegionType;


pub struct MemoryTier {
    pub l1_ram: Vec<f32>,
    l3_nvme: Option<MmapMut>,
    db: Arc<BrainStorage>,
}

impl MemoryTier {
    pub fn dummy() -> Self {
        Self {
            l1_ram: vec![0.0; 1024],
            l3_nvme: None,
            db: Arc::new(BrainStorage::dummy()),
        }
    }
    
    pub async fn new(db: Arc<BrainStorage>, nvme_path: &Path, size: usize) -> Result<Self, std::io::Error> {
        // Implementação real seria mais complexa
        Ok(Self::dummy())
    }

    pub async fn prioritize_and_save(&mut self, memory: NeuralEnactiveMemory) -> Result<(), Box<dyn std::error::Error>> {
        let weight = memory.emotion_state.abs();
    
        if weight > 0.8 {
            let rates = memory.visual_rates();
            self.write_l1(&rates);
            println!("🧠 Memória importante mantida em RAM");
        } else if weight > 0.4 {
            let rates = memory.visual_rates();
            self.write_l1(&rates);
            self.flush_to_l3();
            println!("💾 Memória de média importância salva no NVMe");
        } else {
            self.flush_to_l4(&memory).await?;
            println!("📀 Memória de baixa importância arquivada no DB");
        }
        Ok(())
    }

    pub fn write_l1(&mut self, data: &[f32]) {
        let len = data.len().min(self.l1_ram.len());
        self.l1_ram[..len].copy_from_slice(&data[..len]);
    }

    pub fn flush_to_l3(&mut self) {
        if let Some(mmap) = &mut self.l3_nvme {  // ← mutável diretamente
            for (i, &value) in self.l1_ram.iter().enumerate() {
                let bytes = value.to_ne_bytes();
                let start = i * 4;
                if start + 4 <= mmap.len() {
                    mmap[start..start+4].copy_from_slice(&bytes);
                }
            }
        }
    }

    pub async fn flush_to_l4(&self, memory: &NeuralEnactiveMemory) -> Result<(), surrealdb::Error> {
        self.db.save_snapshot(memory.clone()).await?;  // ← ADICIONADO '?'
        Ok(())
    }
}