// src/meta/consciousness.rs
// Módulo de metacognição: auto-monitoramento do estado interno da Selene.
// Detecta foco, confusão, deriva atencional e senso de self.
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::brain_zones::RegionType;

/// Estado metacognitivo atual — como a Selene percebe seu próprio funcionamento.
pub struct MetaCognitive {
    pub attention_focus: RegionType,
    pub current_goal: String,
    pub self_awareness: f32,   // 0-1: quão "consciente" está do próprio estado
    pub confusion_level: f32,  // 0-1: grau de conflito/incerteza interna
    pub focus_stability: f32,  // 0-1: estabilidade do foco ao longo do tempo
    last_arousal: f32,
}

impl MetaCognitive {
    pub fn new() -> Self {
        Self {
            attention_focus: RegionType::Frontal,
            current_goal: "aprender".to_string(),
            self_awareness: 0.5,
            confusion_level: 0.0,
            focus_stability: 1.0,
            last_arousal: 0.5,
        }
    }

    /// Atualiza o estado metacognitivo com base nos sinais neurais atuais.
    /// `arousal`: nível de alerta do sistema límbico
    /// `emocao`: valência emocional atual
    /// `n_vocab`: tamanho do vocabulário (proxy de conhecimento)
    pub fn observe(&mut self, arousal: f32, emocao: f32, n_vocab: usize) {
        // Self-awareness cresce com vocabulário (mais conceitos = mais auto-modelo)
        let target_awareness = (n_vocab as f32 / 5000.0).clamp(0.1, 0.95);
        self.self_awareness += (target_awareness - self.self_awareness) * 0.01;

        // Confusão: alta quando arousal muda muito rápido (instabilidade)
        let delta_arousal = (arousal - self.last_arousal).abs();
        self.confusion_level = (delta_arousal * 3.0).clamp(0.0, 1.0);
        self.last_arousal = arousal;

        // Estabilidade do foco: inverso da confusão modulado pela emoção positiva
        self.focus_stability = (1.0 - self.confusion_level) * (0.5 + emocao.max(0.0) * 0.5);

        // Região de atenção dominante baseada no arousal
        self.attention_focus = if arousal > 2.0 {
            RegionType::Limbic      // alta excitação → límbico domina
        } else if arousal > 1.2 {
            RegionType::Frontal     // foco moderado → frontal executivo
        } else {
            RegionType::Hippocampus // calmo → consolidação de memória
        };
    }

    pub fn descricao(&self) -> String {
        format!(
            "awareness={:.2} confusão={:.2} foco={} estabilidade={:.2}",
            self.self_awareness,
            self.confusion_level,
            self.attention_focus.nome(),
            self.focus_stability,
        )
    }
}