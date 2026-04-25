// src/learning/ontogeny.rs
// Sistema de ontogenia — desenvolvimento cognitivo graduado da Selene.
// Controla quais capacidades estão ativas em cada fase do desenvolvimento.
// Fase Neonatal: apenas escuta passiva, zero resposta verbal.
// Fase PreVerbal: reações (sons, emoções) mas sem palavras.
// Fase PalavraUnica: resposta de 1-2 palavras máximo.
// Fase Frase: frases curtas até 5 palavras.
// Fase Discurso: conversação completa (modo atual padrão).

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::fmt;

// ── Estágios de desenvolvimento ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DevStage {
    Neonatal   = 0,  // Só escuta. Zero output verbal.
    PreVerbal  = 1,  // Reações emocionais/sons, sem palavras.
    PalavraUnica = 2, // Até 2 palavras por resposta.
    Frase      = 3,  // Até 5 palavras por resposta.
    Discurso   = 4,  // Conversação livre (modo normal).
}

impl fmt::Display for DevStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DevStage::Neonatal     => write!(f, "Neonatal"),
            DevStage::PreVerbal    => write!(f, "PreVerbal"),
            DevStage::PalavraUnica => write!(f, "PalavraUnica"),
            DevStage::Frase        => write!(f, "Frase"),
            DevStage::Discurso     => write!(f, "Discurso"),
        }
    }
}

impl DevStage {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Neonatal"     => Some(DevStage::Neonatal),
            "PreVerbal"    => Some(DevStage::PreVerbal),
            "PalavraUnica" => Some(DevStage::PalavraUnica),
            "Frase"        => Some(DevStage::Frase),
            "Discurso"     => Some(DevStage::Discurso),
            _              => None,
        }
    }

    // Retorna o máximo de palavras permitido por resposta neste estágio.
    // None = sem limite.
    pub fn max_palavras(&self) -> Option<usize> {
        match self {
            DevStage::Neonatal     => Some(0),
            DevStage::PreVerbal    => Some(0),
            DevStage::PalavraUnica => Some(2),
            DevStage::Frase        => Some(5),
            DevStage::Discurso     => None,
        }
    }

    // Pode gerar resposta verbal?
    pub fn pode_responder(&self) -> bool {
        matches!(self, DevStage::PalavraUnica | DevStage::Frase | DevStage::Discurso)
    }

    // Emite reações emocionais (emoções no ack, sons)?
    pub fn emite_reacao_emocional(&self) -> bool {
        !matches!(self, DevStage::Neonatal)
    }

    // Próximo estágio (para auto-progressão)
    pub fn proximo(&self) -> Option<DevStage> {
        match self {
            DevStage::Neonatal     => Some(DevStage::PreVerbal),
            DevStage::PreVerbal    => Some(DevStage::PalavraUnica),
            DevStage::PalavraUnica => Some(DevStage::Frase),
            DevStage::Frase        => Some(DevStage::Discurso),
            DevStage::Discurso     => None,
        }
    }
}

// ── Thresholds para auto-progressão ─────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageThresholds {
    pub vocab_min:       usize,  // Vocabulário mínimo
    pub edges_min:       usize,  // Arestas semânticas mínimas
    pub avg_reward_min:  f32,    // Recompensa média mínima (aprendizado)
    pub horas_escuta_min: f32,   // Horas em modo escuta antes de progredir
}

impl StageThresholds {
    fn para_stage(stage: DevStage) -> Self {
        match stage {
            // Para sair do Neonatal → precisa de 30 palavras ouvidas, 0.5h
            DevStage::Neonatal => StageThresholds {
                vocab_min: 30, edges_min: 10, avg_reward_min: 0.0, horas_escuta_min: 0.5,
            },
            // Para sair do PreVerbal → 100 palavras, 20 arestas, 2h
            DevStage::PreVerbal => StageThresholds {
                vocab_min: 100, edges_min: 20, avg_reward_min: 0.05, horas_escuta_min: 2.0,
            },
            // Para sair do PalavraUnica → 300 palavras, 100 arestas, 5h
            DevStage::PalavraUnica => StageThresholds {
                vocab_min: 300, edges_min: 100, avg_reward_min: 0.1, horas_escuta_min: 5.0,
            },
            // Para sair do Frase → 800 palavras, 500 arestas, 15h
            DevStage::Frase => StageThresholds {
                vocab_min: 800, edges_min: 500, avg_reward_min: 0.15, horas_escuta_min: 15.0,
            },
            // Discurso não progride automaticamente (é o estágio final)
            DevStage::Discurso => StageThresholds {
                vocab_min: 0, edges_min: 0, avg_reward_min: 0.0, horas_escuta_min: 0.0,
            },
        }
    }
}

// ── Métricas de desenvolvimento ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevMetrics {
    pub vocab_count:      usize,
    pub graph_edges:      usize,
    pub avg_reward:       f32,   // Média móvel da recompensa
    pub horas_escuta:     f32,   // Horas acumuladas em escuta (apenas passiva)
    pub total_palavras_ouvidas: u64,
    pub total_interacoes: u64,
}

impl DevMetrics {
    pub fn new() -> Self {
        DevMetrics {
            vocab_count: 0,
            graph_edges: 0,
            avg_reward: 0.0,
            horas_escuta: 0.0,
            total_palavras_ouvidas: 0,
            total_interacoes: 0,
        }
    }

    pub fn update_reward(&mut self, r: f32) {
        // EMA com alpha=0.05
        self.avg_reward = self.avg_reward * 0.95 + r * 0.05;
    }

    pub fn add_escuta_dt(&mut self, dt_secs: f32) {
        self.horas_escuta += dt_secs / 3600.0;
    }
}

impl Default for DevMetrics {
    fn default() -> Self { Self::new() }
}

// ── Estado de Ontogenia ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntogenyState {
    pub stage:             DevStage,
    pub metrics:           DevMetrics,
    pub auto_progress:     bool,   // Progride automaticamente quando métricas atingidas?
    pub stage_start_ts:    f64,    // Unix timestamp do início do estágio atual
    pub progressoes:       Vec<(String, f64)>, // Histórico de progressões (stage, ts)
}

impl OntogenyState {
    pub fn new() -> Self {
        OntogenyState {
            stage: DevStage::Discurso, // Padrão: modo adulto (não interrompe comportamento atual)
            metrics: DevMetrics::new(),
            auto_progress: true,
            stage_start_ts: 0.0,
            progressoes: Vec::new(),
        }
    }

    // Carrega de disco — retorna estado padrão se arquivo não existir
    pub fn carregar(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|_| OntogenyState::new()),
            Err(_) => OntogenyState::new(),
        }
    }

    // Salva em disco
    pub fn salvar(&self, path: &str) {
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, json);
        }
    }

    pub fn salvar_async<'a>(&'a self, path: &'a str) -> impl std::future::Future<Output = ()> + 'a {
        let json = serde_json::to_string_pretty(self).unwrap_or_default();
        let path = path.to_string();
        async move {
            let _ = tokio::fs::write(path, json).await;
        }
    }

    // Reseta para Neonatal — não apaga memória neural (isso é feito pelo caller)
    pub fn reset_para_neonatal(&mut self) {
        let ts = now_secs();
        self.progressoes.push((format!("{} → Neonatal (reset)", self.stage), ts));
        self.stage = DevStage::Neonatal;
        self.metrics = DevMetrics::new();
        self.stage_start_ts = ts;
        log::info!("[ONTOGENY] Reset para Neonatal — novo ciclo de vida iniciado.");
    }

    // Força um estágio específico (override manual)
    pub fn set_stage(&mut self, novo: DevStage) {
        let ts = now_secs();
        self.progressoes.push((format!("{} → {} (manual)", self.stage, novo), ts));
        self.stage = novo;
        self.stage_start_ts = ts;
        log::info!("[ONTOGENY] Estágio forçado para {}", novo);
    }

    // Atualiza métricas e verifica auto-progressão
    // Retorna true se houve progressão de estágio
    pub fn tick(
        &mut self,
        vocab_count: usize,
        graph_edges: usize,
        reward: Option<f32>,
        dt_secs_escuta: Option<f32>,
    ) -> bool {
        self.metrics.vocab_count = vocab_count;
        self.metrics.graph_edges = graph_edges;
        if let Some(r) = reward { self.metrics.update_reward(r); }
        if let Some(dt) = dt_secs_escuta { self.metrics.add_escuta_dt(dt); }

        if !self.auto_progress { return false; }
        if let Some(proximo) = self.stage.proximo() {
            let thr = StageThresholds::para_stage(self.stage);
            if self.metrics.vocab_count >= thr.vocab_min
                && self.metrics.graph_edges >= thr.edges_min
                && self.metrics.avg_reward >= thr.avg_reward_min
                && self.metrics.horas_escuta >= thr.horas_escuta_min
            {
                let ts = now_secs();
                log::info!(
                    "[ONTOGENY] Progressão automática: {} → {} | vocab={} edges={} reward={:.3} horas={:.1}",
                    self.stage, proximo,
                    self.metrics.vocab_count, self.metrics.graph_edges,
                    self.metrics.avg_reward, self.metrics.horas_escuta
                );
                self.progressoes.push((format!("{} → {} (auto)", self.stage, proximo), ts));
                self.stage = proximo;
                self.stage_start_ts = ts;
                return true;
            }
        }
        false
    }

    // Filtra uma resposta gerada pelo walk conforme o estágio atual.
    // Retorna None se o estágio não permite resposta verbal.
    // Retorna Some(reply_truncado) se há limite de palavras.
    pub fn filtrar_resposta(&self, reply: &str) -> Option<String> {
        if !self.stage.pode_responder() {
            return None;
        }
        match self.stage.max_palavras() {
            None => Some(reply.to_string()),
            Some(0) => None,
            Some(max) => {
                let palavras: Vec<&str> = reply.split_whitespace().collect();
                if palavras.len() <= max {
                    Some(reply.to_string())
                } else {
                    Some(palavras[..max].join(" "))
                }
            }
        }
    }

    // Métricas prontas para serializar para o frontend
    pub fn to_json(&self) -> serde_json::Value {
        let thr = StageThresholds::para_stage(self.stage);
        let proximo = self.stage.proximo().map(|s| s.to_string());
        serde_json::json!({
            "stage":            self.stage.to_string(),
            "auto_progress":    self.auto_progress,
            "metrics": {
                "vocab_count":           self.metrics.vocab_count,
                "graph_edges":           self.metrics.graph_edges,
                "avg_reward":            self.metrics.avg_reward,
                "horas_escuta":          self.metrics.horas_escuta,
                "total_palavras_ouvidas": self.metrics.total_palavras_ouvidas,
                "total_interacoes":      self.metrics.total_interacoes,
            },
            "thresholds": {
                "vocab_min":        thr.vocab_min,
                "edges_min":        thr.edges_min,
                "avg_reward_min":   thr.avg_reward_min,
                "horas_escuta_min": thr.horas_escuta_min,
            },
            "proximo_estagio":  proximo,
            "historico":        self.progressoes.len(),
        })
    }
}

impl Default for OntogenyState {
    fn default() -> Self { Self::new() }
}

// ─────────────────────────────────────────────────────────────────────────────

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}
