// src/neurochem.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use crate::sensors::hardware::HardwareSensor;
use crate::config::{Config, ModoOperacao};

pub struct NeuroChem {
    pub serotonin: f32,
    pub dopamine: f32,
    pub cortisol: f32,
    pub noradrenaline: f32,
    /// Acetilcolina (ACh) — neuromodulador de atenção e codificação de memória.
    /// Alta ACh = foco atencional aguçado, hipocampo mais receptivo, memória de trabalho estável.
    /// Produzida pelo núcleo basal de Meynert; degrada com fadiga (adenosina alta).
    /// Análogo biológico: inibidores de colinesterase melhoram memória em Alzheimer.
    pub acetylcholine: f32,
    last_temp: f32,
}

/// Roda de emoções de Plutchik — derivada dos neurotransmissores.
/// Usada para colorir os pensamentos e respostas da Selene.
pub struct EmotionalState {
    pub joy: f32,          // dopamina alta + serotonina alta
    pub trust: f32,        // serotonina alta + cortisol baixo
    pub fear: f32,         // noradrenalina alta + cortisol alto
    pub surprise: f32,     // noradrenalina spike (delta grande)
    pub sadness: f32,      // dopamina baixa + serotonina baixa
    pub disgust: f32,      // cortisol alto + dopamina baixa
    pub anger: f32,        // noradrenalina alta + dopamina alta
    pub anticipation: f32, // dopamina moderada + noradrenalina moderada
}

impl EmotionalState {
    /// Deriva o estado emocional de Plutchik dos neurotransmissores atuais.
    pub fn from_neurochem(dopa: f32, sero: f32, cort: f32, nor: f32) -> Self {
        Self {
            joy:          (dopa * sero).clamp(0.0, 1.0),
            trust:        (sero * (1.0 - cort)).clamp(0.0, 1.0),
            fear:         (nor * cort).clamp(0.0, 1.0),
            surprise:     (nor * (1.0 - sero) * 0.5).clamp(0.0, 1.0),
            sadness:      ((1.0 - dopa) * (1.0 - sero)).clamp(0.0, 1.0),
            disgust:      (cort * (1.0 - dopa)).clamp(0.0, 1.0),
            anger:        (nor * dopa * (1.0 - sero)).clamp(0.0, 1.0),
            anticipation: (dopa * nor * 0.5).clamp(0.0, 1.0),
        }
    }

    /// Retorna a emoção dominante como string legível.
    pub fn dominante(&self) -> &'static str {
        let arr = [
            (self.joy,          "alegria"),
            (self.trust,        "confiança"),
            (self.fear,         "medo"),
            (self.surprise,     "surpresa"),
            (self.sadness,      "tristeza"),
            (self.disgust,      "repulsa"),
            (self.anger,        "raiva"),
            (self.anticipation, "antecipação"),
        ];
        arr.iter().max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, name)| *name)
            .unwrap_or("neutro")
    }
}

impl NeuroChem {
    pub fn new() -> Self {
        Self {
            serotonin: 1.0,
            dopamine: 1.0,
            cortisol: 0.0,
            noradrenaline: 0.5,
            acetylcholine: 0.7, // baseline saudável de ACh
            last_temp: 0.0,
        }
    }

    pub fn update(&mut self, sensor: &mut HardwareSensor, config: &Config) {
        let jitter = sensor.get_jitter_ms();
        let switches = sensor.get_context_switches_per_sec();
        let ram_usage = sensor.get_ram_usage();
        let temp = sensor.get_cpu_temp();
        let delta_temp = (temp - self.last_temp).abs();
        self.last_temp = temp;

        let fator_tempo = config.fator_boost();
        let decay_rate = 0.01 / fator_tempo;

        // Serotonina
        let switches_penalty = (switches / 6000.0).clamp(0.0, 1.0);
        let jitter_penalty = (jitter / 3.0).clamp(0.0, 1.0);
        let target_sero = (1.0 - 0.5 * (switches_penalty + jitter_penalty)).clamp(0.1f32, 1.0f32);
        self.serotonin += (target_sero - self.serotonin) * decay_rate;

        // Dopamina - CORRIGIDO: todos os modos cobertos
        let target_dopa = match config.modo {
            ModoOperacao::Economia => (ram_usage / 110.0).clamp(0.3, 0.8),
            ModoOperacao::Humano | ModoOperacao::Normal => (ram_usage / 100.0).clamp(0.5, 1.0),
            ModoOperacao::Boost200 | ModoOperacao::Turbo => (ram_usage / 90.0).clamp(0.6, 1.2),
            ModoOperacao::Boost800 => (ram_usage / 80.0).clamp(0.7, 1.5),
            ModoOperacao::Ultra => (ram_usage / 70.0).clamp(0.8, 1.8),
            ModoOperacao::Insano => (ram_usage / 60.0).clamp(0.9, 2.0),
        };
        self.dopamine += (target_dopa - self.dopamine) * decay_rate;

        // Cortisol
        self.cortisol = (delta_temp / 5.0).clamp(0.0, 1.0);

        // Noradrenalina - CORRIGIDO: todos os modos cobertos
        self.noradrenaline = match config.modo {
            ModoOperacao::Economia => (temp / 110.0).clamp(0.3, 0.8),
            ModoOperacao::Humano | ModoOperacao::Normal => (temp / 100.0).clamp(0.5, 1.0),
            ModoOperacao::Boost200 | ModoOperacao::Turbo => (temp / 95.0).clamp(0.6, 1.1),
            ModoOperacao::Boost800 => (temp / 85.0).clamp(0.8, 1.2),
            ModoOperacao::Ultra => (temp / 75.0).clamp(0.9, 1.4),
            ModoOperacao::Insano => (temp / 65.0).clamp(1.0, 1.6),
        };

        // Acetilcolina — modulada por fadiga (adenosina inibe ACh no tálamo/hipocampo)
        // e por noradrenalina (arousal alto libera mais ACh no córtex).
        // Alta RAM (processamento intenso) → mais demanda colinérgica → degrada levemente.
        let adenosina_proxy = (jitter / 5.0).clamp(0.0, 1.0); // jitter alto ≈ sistema estressado
        let target_ach = (0.8 - adenosina_proxy * 0.4 + self.noradrenaline * 0.2)
            .clamp(0.2, 1.2);
        self.acetylcholine += (target_ach - self.acetylcholine) * decay_rate;
    }
}