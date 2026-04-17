// src/interoception/mod.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use crate::config::Config;

/// Ínsula e Cingulado Anterior: integram sensações corporais ao self
#[derive(Debug)] 
pub struct Interoception {
    pub fadiga: f32,           // 0.0 a 1.0 (do SleepManager)
    pub temperatura: f32,       // °C (do HardwareSensor)
    pub arousal: f32,           // do LimbicSystem
    pub dor_simulada: f32,      // 0.0 a 1.0 (opcional)
    pub historico: Vec<f32>,    // histórico de sensações
}

impl Interoception {
    pub fn new() -> Self {
        Self {
            fadiga: 0.0,
            temperatura: 36.0,
            arousal: 0.5,
            dor_simulada: 0.0,
            historico: Vec::with_capacity(100),
        }
    }
    
    /// Atualiza sinais corporais
    pub fn update(&mut self, fadiga: f32, temp: f32, arousal: f32) {
        self.fadiga = fadiga;
        self.temperatura = temp;
        self.arousal = arousal;
        
        // Simples modelo de "dor" baseado em extremos
        self.dor_simulada = (self.temperatura - 36.0).abs() / 10.0 + self.fadiga * 0.3;
        self.dor_simulada = self.dor_simulada.clamp(0.0, 1.0);
        
        // Guarda sensação integrada
        self.historico.push(self.sentir());
        if self.historico.len() > 1000 {
            self.historico.drain(0..500);
        }
    }
    
    /// Integra todos os sinais em uma "sensação corporal" única
    pub fn sentir(&self) -> f32 {
        self.fadiga * 0.3 +
        self.arousal * 0.5 +
        self.temperatura / 100.0 * 0.2 +
        self.dor_simulada * 0.2
    }
    
    /// Influencia o ego com base na sensação corporal
    pub fn influenciar_ego(&self) -> (String, f32) {
        let sensacao = self.sentir();
        let descricao = match sensacao {
            s if s < 0.2 => "Sinto-me muito bem!".to_string(),
            s if s < 0.4 => "Sinto-me normal.".to_string(),
            s if s < 0.6 => "Estou um pouco cansada.".to_string(),
            s if s < 0.8 => "Sinto desconforto.".to_string(),
            _ => "Sinto-me mal.".to_string(),
        };
        (descricao, sensacao)
    }
    
    /// Estatísticas interoceptivas
    pub fn stats(&self) -> InteroceptionStats {
        InteroceptionStats {
            fadiga: self.fadiga,
            temperatura: self.temperatura,
            arousal: self.arousal,
            sensacao_integrada: self.sentir(),
        }
    }
}
#[derive(Debug)]
pub struct InteroceptionStats {
    pub fadiga: f32,
    pub temperatura: f32,
    pub arousal: f32,
    pub sensacao_integrada: f32,
}