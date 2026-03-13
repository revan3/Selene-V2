// src/brainstem/mod.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use crate::config::Config;
use crate::sensors::audio::AudioSignal;
/// Tronco Cerebral (Formação Reticular): regulação de alerta e sono/vigília
pub struct Brainstem {
    pub noradrenaline: f32,      // 0.3 a 1.2
    pub adenosina: f32,           // 0.0 a 1.0 (fadiga)
    pub alertness: f32,           // 0.0 a 1.0
    pub horas_acordado: f32,      // em horas simuladas
}

impl Brainstem {
    pub fn new() -> Self {
        Self {
            noradrenaline: 0.8,
            adenosina: 0.0,
            alertness: 1.0,
            horas_acordado: 0.0,
        }
    }
    
    /// Atualiza estado de alerta baseado em adenosina e tempo acordado
    pub fn update(&mut self, adenosina: f32, dt: f32) {
        self.adenosina = adenosina;
        self.horas_acordado += dt / 3600.0; // converte segundos para horas
        
        // Quanto mais adenosina, menos alerta
        let base_alert = 1.0 - self.adenosina;
        
        // Compensação: mais tempo acordado aumenta noradrenalina (tentativa de manter alerta)
        let compensation = (self.horas_acordado / 16.0).clamp(0.0, 0.5);
        
        self.noradrenaline = (base_alert + compensation).clamp(0.3, 1.2);
        self.alertness = self.noradrenaline;
        
        // Se alerta muito baixo, pode induzir sono
        if self.alertness < 0.4 {
            println!("😴 Tronco cerebral: alerta baixo ({:.2}), induzindo sono...", self.alertness);
        }
    }
    
    /// Modula qualquer input baseado no alerta atual
    pub fn modulate(&self, input: &AudioSignal) -> Vec<f32> {
        self.modulate(&input.data)  // delega para &[f32]
    }
    
    pub fn stats(&self) -> BrainstemStats {
        BrainstemStats {
            noradrenaline: self.noradrenaline,
            adenosina: self.adenosina,
            alertness: self.alertness,
            horas_acordado: self.horas_acordado,
        }
    }
}

pub struct BrainstemStats {
    pub noradrenaline: f32,
    pub adenosina: f32,
    pub alertness: f32,
    pub horas_acordado: f32,
}