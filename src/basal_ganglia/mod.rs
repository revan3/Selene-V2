// src/basal_ganglia/mod.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use std::collections::VecDeque;
use crate::config::Config;

/// Representa um hábito procedural
#[derive(Debug, Clone)]
pub struct Habito {
    pub padrao_entrada: Vec<f32>,    // situação que dispara
    pub resposta_motora: Vec<f32>,    // ação associada
    pub forca: f32,                   // 0.0 a 1.0
    pub recompensa_acumulada: f32,
    pub vezes_executado: u32,
}

/// Núcleos da Base: formação de hábitos e modulação de recompensa
pub struct BasalGanglia {
    pub habitos: Vec<Habito>,
    pub dopamine_mod: f32,
    pub learning_rate: f32,
    pub historico_recompensas: VecDeque<f32>,
}

impl BasalGanglia {
    pub fn new(config: &Config) -> Self {
        Self {
            habitos: Vec::with_capacity(100),
            dopamine_mod: 1.0,
            learning_rate: config.taxa_aprendizado * 2.0, // hábitos aprendem mais rápido
            historico_recompensas: VecDeque::with_capacity(100),
        }
    }
    
    /// Atualiza hábitos com base no estado atual e recompensa
    pub fn update_habits(&mut self, current_state: &[f32], acao_executada: &[f32], recompensa: f32) {
        self.historico_recompensas.push_back(recompensa);
        if self.historico_recompensas.len() > 100 {
            self.historico_recompensas.pop_front();
        }
        
        // CORRIGIDO: clonar antes de iterar para evitar borrow checker
        let habitos_clone = self.habitos.clone();
        
        // Verifica se estado atual corresponde a algum hábito existente
        for (i, habito) in habitos_clone.iter().enumerate() {
            let similaridade = self.cosseno_similaridade(current_state, &habito.padrao_entrada);
            if similaridade > 0.8 {
                // Reforça ou enfraquece baseado na recompensa
                self.habitos[i].forca += recompensa * self.learning_rate;
                self.habitos[i].forca = self.habitos[i].forca.clamp(0.0, 1.0);
                self.habitos[i].recompensa_acumulada += recompensa;
                self.habitos[i].vezes_executado += 1;
                
                // Atualiza padrão com média ponderada
                for (j, &valor) in acao_executada.iter().enumerate() {
                    if j < self.habitos[i].resposta_motora.len() {
                        self.habitos[i].resposta_motora[j] = 
                            self.habitos[i].resposta_motora[j] * 0.9 + valor * 0.1;
                    }
                }
                return;
            }
        }
        
        // Se não existe, pode criar um novo hábito se a recompensa for alta
        if recompensa > 0.7 && self.habitos.len() < 1000 {
            self.habitos.push(Habito {
                padrao_entrada: current_state.to_vec(),
                resposta_motora: acao_executada.to_vec(),
                forca: 0.3,
                recompensa_acumulada: recompensa,
                vezes_executado: 1,
            });
        }
    }
    
    /// Sugere uma ação baseada em hábitos existentes
    pub fn suggest_action(&self, current_state: &[f32]) -> Option<Vec<f32>> {
        let mut melhor_habito = None;
        let mut melhor_similaridade = 0.7; // threshold mínimo
        
        for habito in &self.habitos {
            let similaridade = self.cosseno_similaridade(current_state, &habito.padrao_entrada);
            if similaridade > melhor_similaridade && habito.forca > 0.5 {
                melhor_similaridade = similaridade;
                melhor_habito = Some(habito.resposta_motora.clone());
            }
        }
        
        melhor_habito
    }
    
    fn cosseno_similaridade(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.is_empty() || b.is_empty() { return 0.0; }
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
    }
    
    pub fn stats(&self) -> BasalGangliaStats {
        let forca_media = if !self.habitos.is_empty() {
            self.habitos.iter().map(|h| h.forca).sum::<f32>() / self.habitos.len() as f32
        } else {
            0.0
        };
        
        BasalGangliaStats {
            num_habitos: self.habitos.len(),
            forca_media,
            recompensa_media: self.historico_recompensas.iter().sum::<f32>() / 
                self.historico_recompensas.len().max(1) as f32,
            dopamine_mod: self.dopamine_mod,
        }
    }
}

pub struct BasalGangliaStats {
    pub num_habitos: usize,
    pub forca_media: f32,
    pub recompensa_media: f32,
    pub dopamine_mod: f32,
}