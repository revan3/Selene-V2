// src/sleep_manager.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use std::thread;
use std::time::Duration;
use crate::sensors::hardware::HardwareSensor;
use crate::storage::memory_tier::MemoryTier;
use crate::storage::BrainStorage;  // ← IMPORTANTE!
use crate::brain_zones::hippocampus::HippocampusV2;
use crate::brain_zones::RegionType;

pub struct SleepManagerV2 {
    adenosina: f32,
    threshold: f32,
    
    // Referências para consolidação
    hippocampus: Option<*mut HippocampusV2>,
    storage: Option<*mut BrainStorage>,
}

impl SleepManagerV2 {
    pub fn new() -> Self {
        Self {
            adenosina: 0.0,
            threshold: 0.8,
            hippocampus: None,
            storage: None,
        }
    }

    pub fn register_brain_parts(&mut self, 
                                 hip: *mut HippocampusV2, 
                                 store: *mut BrainStorage) {
        self.hippocampus = Some(hip);
        self.storage = Some(store);
    }

    pub fn monitor(&mut self, sensor: &mut HardwareSensor, memory_tier: &mut MemoryTier) {
        loop {
            let cpu_temp = sensor.get_cpu_temp();
            self.adenosina += (cpu_temp / 100.0) * 0.01;
            self.adenosina = self.adenosina.clamp(0.0, 1.0);

            if self.adenosina > self.threshold {
                println!("😴 Selene dormindo... Hora de consolidar memórias!");
                
                // 1. Salva memórias voláteis
                memory_tier.flush_to_l3();
                
                // 2. CONSOLIDAÇÃO DAS CONEXÕES (SONHO!)
                if let Some(hip_ptr) = self.hippocampus {
                    unsafe {
                        let hip = &mut *hip_ptr;
                        let conexoes = hip.consolidate_recent();
                        
                        if let Some(store_ptr) = self.storage {
                            let store = &mut *store_ptr;
                            
                            // Salva cada conexão no DB
                            for conexao in conexoes {
                                println!("   💾 Conexão consolidada: {:?} -> {:?} (peso: {:.2})",
                                         conexao.de_neuronio, conexao.para_neuronio, conexao.peso);
                            }
                        }
                    }
                }

                // 3. "Dorme" por alguns segundos
                self.adenosina = 0.0;
                thread::sleep(Duration::from_secs(5));
                println!("🔆 Selene acordou! Memórias consolidadas.");
            }
            
            thread::sleep(Duration::from_millis(1000));
        }
    }
}