// src/basal_ganglia/mod.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use std::collections::VecDeque;
use crate::config::Config;
use crate::synaptic_core::{CamadaHibrida, TipoNeuronal};

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
    // ── V4.6.1: substrato spiking real — estriado (MSN, GABAérgico) ──────────
    // MSN tem casa anatômica aqui (estriado dorsal). D1/D2 sensíveis à dopamina
    // (dopamine_mod). Roda junto à lógica de hábitos abstrata.
    pub striatum: CamadaHibrida,
    dt: f32,
    tick_ms: f32,
}

impl BasalGanglia {
    pub fn new(config: &Config) -> Self {
        let mut striatum = CamadaHibrida::new(
            128, "striatum_msn", TipoNeuronal::MSN,
            Some((TipoNeuronal::FS, 0.10)), // alguns FS (interneurônios estriatais)
            None, 1.0,
        );
        striatum.init_lateral_inhibition(4, 2.0); // MSN inibição lateral (winner-take-all)
        Self {
            habitos: Vec::with_capacity(100),
            dopamine_mod: 1.0,
            learning_rate: config.taxa_aprendizado * 2.0, // hábitos aprendem mais rápido
            historico_recompensas: VecDeque::with_capacity(100),
            striatum,
            dt: config.dt_simulacao,
            tick_ms: 0.0,
        }
    }

    /// Taxa de disparo do estriado MSN (fração de neurônios ativos no último tick).
    pub fn msn_spike_rate(&self) -> f32 {
        self.striatum.estatisticas_v3().spike_rate
    }
    
    /// V4.2 — Recebe o RPE (Reward Prediction Error) do `ReinforcementLearning`
    /// e modula `dopamine_mod`. Biologicamente: SNpc → BG via projeção
    /// dopaminérgica nigrostriatal. RPE positivo amplifica plasticidade
    /// dos hábitos; RPE negativo a suprime.
    ///
    /// Fórmula: `dopamine_mod = (1.0 + rpe * 0.5).clamp(0.5, 2.0)`
    /// Chamar APÓS `rl.update()` no loop neural, ANTES de `update_habits()`.
    pub fn aplicar_rpe(&mut self, rpe: f32) {
        let alvo = (1.0 + rpe * 0.5).clamp(0.5, 2.0);
        // EMA suave (τ ≈ 20 ticks) — evita oscilação caótica do gate
        self.dopamine_mod = self.dopamine_mod * 0.95 + alvo * 0.05;
    }

    /// Atualiza hábitos com base no estado atual e recompensa.
    /// V4.2: `effective_lr = learning_rate × dopamine_mod` — RPE do RL
    /// modula a taxa de plasticidade dos hábitos via `aplicar_rpe()`.
    pub fn update_habits(&mut self, current_state: &[f32], acao_executada: &[f32], recompensa: f32) {
        self.historico_recompensas.push_back(recompensa);
        if self.historico_recompensas.len() > 100 {
            self.historico_recompensas.pop_front();
        }

        // V4.6.1 — estriado spiking: dopamina (D1/D2) + drive do estado atual.
        // MSN dispara esparso (down-state) → seleção de ação winner-take-all.
        self.tick_ms += self.dt * 1000.0;
        self.striatum.modular_neuro(self.dopamine_mod, 1.0, 0.0);
        let drive: Vec<f32> = current_state.iter().take(128).map(|&v| v * 22.0).collect();
        let _ = self.striatum.update(&drive, self.dt, self.tick_ms);

        // V4.2: dopamine_mod modula a taxa efetiva (vem de aplicar_rpe).
        let effective_lr = self.learning_rate * self.dopamine_mod;

        // CORRIGIDO: clonar antes de iterar para evitar borrow checker
        let habitos_clone = self.habitos.clone();

        // Verifica se estado atual corresponde a algum hábito existente
        for (i, habito) in habitos_clone.iter().enumerate() {
            let similaridade = self.cosseno_similaridade(current_state, &habito.padrao_entrada);
            if similaridade > 0.8 {
                // Reforça ou enfraquece baseado na recompensa × dopamine_mod
                self.habitos[i].forca += recompensa * effective_lr;
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