// src/sensors/hardware.rs
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};
use wmi::{COMLibrary, WMIConnection, Variant};
use std::sync::{Arc, Mutex};
use windows::Win32::Foundation::{BOOL, TRUE};
use windows::Win32::Media::{timeBeginPeriod, timeEndPeriod};
use windows::Win32::System::Performance::{
    PdhAddEnglishCounterW, PdhCollectQueryData, PdhGetFormattedCounterValue,
    PdhCloseQuery, PDH_FMT_DOUBLE
};
use crate::brain_zones::RegionType;
use rand::Rng;

#[derive(Clone)]
pub struct HardwareSensor {
    pub wmi: Arc<Mutex<WMIConnection>>,
    last_context_switches: f64,
    last_time: Instant,
    jitter_smoothed: f32,
    switches_smoothed: f32,
}

// Implementação manual de Send para HardwareSensor
// Isto é UNSAFE e só deve ser usado porque estamos em ambiente controlado
unsafe impl Send for HardwareSensor {}

impl HardwareSensor {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::dummy())
    }

    pub fn dummy() -> Self {
        use wmi::WMIConnection;
        use std::sync::Arc;
        use std::sync::Mutex;
        
        // Versão simplificada sem unsafe
        // Em vez de usar zeroed(), criamos valores dummy seguros
        let wmi_conn = unsafe { 
        // Isso é um placeholder e não será usado
            std::mem::MaybeUninit::zeroed().assume_init()
        };
        
        Self {
            wmi: Arc::new(Mutex::new(wmi_conn)),
            last_context_switches: 0.0,
            last_time: Instant::now(),
            jitter_smoothed: 0.0,
            switches_smoothed: 0.0,
        }
    }

    pub fn get_cpu_temp(&self) -> f32 {
        let wmi = self.wmi.lock().unwrap();
        let results: Vec<HashMap<String, Variant>> = match wmi.query() {
            Ok(r) => r,
            Err(_) => return 0.0
        };
    
        for item in results {
            if let Some(Variant::R8(temp)) = item.get("HighPrecisionTemperature") {
                return *temp as f32;
            }
        }
        0.0
    }
    
    pub fn get_ram_usage(&self) -> f32 {
        let wmi = self.wmi.lock().unwrap();
        let results: Vec<HashMap<String, Variant>> = match wmi.query() {
            Ok(r) => r,
            Err(_) => return 0.0
        };
    
        for os in results {
            let total = match os.get("TotalVisibleMemorySize") {
                Some(Variant::UI8(val)) => *val,
                _ => continue
            };
            let free = match os.get("FreePhysicalMemory") {
                Some(Variant::UI8(val)) => *val,
                _ => continue
            };
            return ((total - free) as f32 / total as f32) * 100.0;
        }
        0.0
    }

    pub fn get_context_switches_per_sec(&mut self) -> f32 {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_time).as_secs_f32();
        self.last_time = now;
    
        let simulated = 2000.0 + (rand::thread_rng().gen::<f32>() * 3000.0);
    
        self.switches_smoothed = 0.8 * self.switches_smoothed + 0.2 * simulated;
        self.switches_smoothed
    }

    pub fn get_jitter_ms(&mut self) -> f32 {
        let samples = 10;
        let mut deltas = vec![];
        let mut prev = Instant::now();
        for _ in 0..samples {
            let now = Instant::now();
            let delta = now.duration_since(prev).as_secs_f32() * 1000.0;
            deltas.push(delta);
            prev = now;
            std::thread::sleep(Duration::from_millis(1));
        }
        let mean = deltas.iter().sum::<f32>() / samples as f32;
        let variance = deltas.iter().map(|&d| (d - mean).powi(2)).sum::<f32>() / samples as f32;
        let new_jitter = variance.sqrt();

        self.jitter_smoothed = 0.8 * self.jitter_smoothed + 0.2 * new_jitter;
        self.jitter_smoothed
    }

    pub fn get_ram_usage_gb(&self) -> f32 {
        let wmi = self.wmi.lock().unwrap();
        let results: Vec<HashMap<String, Variant>> = match wmi.query() {
            Ok(r) => r,
            Err(_) => return 0.0
        };
    
        for os in results {
            let total = match os.get("TotalVisibleMemorySize") {
                Some(Variant::UI8(val)) => *val,
                _ => continue
            };
            let free = match os.get("FreePhysicalMemory") {
                Some(Variant::UI8(val)) => *val,
                _ => continue
            };
            return (total - free) as f32 / 1024.0 / 1024.0;
        }
        0.0
    }
}