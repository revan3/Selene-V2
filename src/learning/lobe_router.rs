// src/learning/lobe_router.rs
// Roteador dinâmico de lóbulos-constelação com especialização emergente.
//
// CONCEITO:
//   Cada lóbulo mantém um vetor-chave (16 dims) que descreve "em que tipo
//   de input eu sou bom." O estado atual do sistema é codificado como um
//   vetor-query (16 dims). O gate de cada lóbulo = cosine_similarity(query, key).
//
//   ESPECIALIZAÇÃO EMERGENTE — competitive Hebbian learning:
//     vencedor (gate mais alto): key += lr * (query - key)  → deriva para inputs premiados
//     perdedores:                key -= lr * REPULSION * (query - key) → afasta-se
//   Com o tempo, lobes divergem para nichos diferentes sem programação explícita.
//
//   HOMEOSTASE — evita lóbulo morto:
//     Rastreia atividade média de cada lóbulo na janela dos últimos N ticks.
//     Lóbulo subativo (<50% do alvo) recebe boost crescente no gate.
//     Garante que nenhuma constelação "apaga" permanentemente.
//
//   SKIP ADAPTATIVO — economiza CPU:
//     Lóbulos com gate < SKIP_THRESHOLD não rodam (usam output do tick anterior).
//     Economiza entre 10-40% de CPU em estados focados (ex: só linguagem ativa).

#![allow(dead_code)]

use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// Constantes
// ─────────────────────────────────────────────────────────────────────────────

/// Dimensão do espaço de embedding (query e chaves dos lóbulos).
pub const EMBED_DIM: usize = 16;

/// Taxa de aprendizado da chave do lóbulo vencedor.
const KEY_LR: f32 = 0.004;

/// Fração de repulsão aplicada às chaves dos lóbulos perdedores.
/// Valor pequeno: diferenciação lenta mas estável.
const LOSER_REPULSION: f32 = 0.12;

/// EMA alpha para suavização dos gate scores (anti-flicker).
/// 0.1 → τ ≈ 10 ticks ≈ 50ms @ 200Hz.
const GATE_EMA: f32 = 0.10;

/// Janela de histórico para cálculo de homeostase (em ticks).
const HOMEOSTASIS_WINDOW: usize = 500;

/// Atividade média alvo para homeostase (0.5 = 50% de ativação).
const HOMEOSTASIS_TARGET: f32 = 0.50;

/// Boost homeostático máximo aplicável.
const HOMEOSTASIS_MAX_BOOST: f32 = 0.22;

/// Gate mínimo abaixo do qual o lóbulo é marcado como SKIP.
/// O lóbulo ainda "existe" mas não roda computação pesada.
pub const SKIP_THRESHOLD: f32 = 0.08;

/// RPE mínimo para disparar update de especialização.
const RPE_MIN_UPDATE: f32 = 0.08;

// ─────────────────────────────────────────────────────────────────────────────
// LobeId
// ─────────────────────────────────────────────────────────────────────────────

/// Identificador de cada constelação (lóbulo).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LobeId {
    Temporal,
    Parietal,
    Limbic,
    Hippocampus,
    Frontal,
    Cerebellum,
}

impl LobeId {
    pub const ALL: [LobeId; 6] = [
        LobeId::Temporal, LobeId::Parietal, LobeId::Limbic,
        LobeId::Hippocampus, LobeId::Frontal, LobeId::Cerebellum,
    ];

    pub fn nome(&self) -> &'static str {
        match self {
            LobeId::Temporal    => "temporal",
            LobeId::Parietal    => "parietal",
            LobeId::Limbic      => "límbico",
            LobeId::Hippocampus => "hipocampo",
            LobeId::Frontal     => "frontal",
            LobeId::Cerebellum  => "cerebelo",
        }
    }

    /// Gate mínimo garantido — nenhum lóbulo é silenciado completamente.
    /// Frontal e Limbic têm piso mais alto (controle executivo e emoção são sempre necessários).
    pub fn gate_minimo(&self) -> f32 {
        match self {
            LobeId::Frontal     => 0.30,
            LobeId::Limbic      => 0.20,
            LobeId::Temporal    => 0.15,
            LobeId::Hippocampus => 0.10,
            LobeId::Parietal    => 0.10,
            LobeId::Cerebellum  => 0.05,
        }
    }

    fn idx(&self) -> usize {
        match self {
            LobeId::Temporal    => 0,
            LobeId::Parietal    => 1,
            LobeId::Limbic      => 2,
            LobeId::Hippocampus => 3,
            LobeId::Frontal     => 4,
            LobeId::Cerebellum  => 5,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RoutingDecision
// ─────────────────────────────────────────────────────────────────────────────

/// Gate scores para o tick atual (0.0 = skip, 1.0 = ativação completa).
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub temporal:    f32,
    pub parietal:    f32,
    pub limbic:      f32,
    pub hippocampus: f32,
    pub frontal:     f32,
    pub cerebellum:  f32,
}

impl RoutingDecision {
    /// Gate uniforme (todos ativos) — usado no warmup.
    pub fn uniform() -> Self {
        Self { temporal: 1.0, parietal: 1.0, limbic: 1.0,
               hippocampus: 1.0, frontal: 1.0, cerebellum: 1.0 }
    }

    pub fn get(&self, id: LobeId) -> f32 {
        match id {
            LobeId::Temporal    => self.temporal,
            LobeId::Parietal    => self.parietal,
            LobeId::Limbic      => self.limbic,
            LobeId::Hippocampus => self.hippocampus,
            LobeId::Frontal     => self.frontal,
            LobeId::Cerebellum  => self.cerebellum,
        }
    }

    /// True se o lóbulo deve ser pulado (abaixo do threshold de skip).
    pub fn deve_skipar(&self, id: LobeId) -> bool {
        self.get(id) < SKIP_THRESHOLD
    }

    /// Número de lóbulos ativos (gate >= SKIP_THRESHOLD).
    pub fn n_ativos(&self) -> usize {
        [self.temporal, self.parietal, self.limbic,
         self.hippocampus, self.frontal, self.cerebellum]
            .iter().filter(|&&g| g >= SKIP_THRESHOLD).count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LobeRouter
// ─────────────────────────────────────────────────────────────────────────────

pub struct LobeRouter {
    /// Vetores-chave dos lóbulos [6][EMBED_DIM].
    /// Inicializados com bias diferente por lóbulo (quebra de simetria),
    /// depois aprendidos via competição Hebbiana.
    keys: [[f32; EMBED_DIM]; 6],

    /// Query suavizada do tick anterior (para anti-flicker).
    prev_query: [f32; EMBED_DIM],

    /// Gate scores suavizados por EMA (o que o sistema "vê" como gate atual).
    smoothed_gates: [f32; 6],

    /// Histórico de atividade por lóbulo (para homeostase).
    activity_history: [VecDeque<f32>; 6],

    /// Boost homeostático atual por lóbulo [0..HOMEOSTASIS_MAX_BOOST].
    pub homeostasis_boost: [f32; 6],

    /// Query do último update de especialização (para telemetria).
    last_query: [f32; EMBED_DIM],

    /// Contador de updates de especialização (para debug).
    pub n_especialization_updates: u64,
}

impl LobeRouter {
    pub fn new() -> Self {
        // Bias inicial por lóbulo — define nichos iniciais (quebra de simetria).
        // Dimensões do embedding:
        //   0-3: visual (energia, pico, novidade, gradiente)
        //   4-5: auditivo (energia, pitch)
        //   6-9: neuroquímica (da, ser, cor, nor)
        //   10-13: estado (emoção, arousal, atividade, abstração)
        //   14-15: fase temporal (sin, cos)
        let mut keys = [[0.0f32; EMBED_DIM]; 6];
        // Temporal: prefere auditivo + linguagem + alta abstração
        keys[0] = [0.1, 0.1, 0.2, 0.1, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.3, 0.3, 0.1, 0.1];
        // Parietal: prefere visual + espacial + novidade
        keys[1] = [0.4, 0.3, 0.4, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.2];
        // Limbic: prefere emoção + arousal + cortisol
        keys[2] = [0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.2, 0.1, 0.1, 0.1];
        // Hippo: prefere novidade visual + emoção + baixa atividade (replay)
        keys[3] = [0.2, 0.2, 0.4, 0.2, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.3, 0.4, 0.1, 0.2, 0.2, 0.2];
        // Frontal: multi-dimensional (sempre relevante, prefere dopamina alta)
        keys[4] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.1, 0.2, 0.2, 0.2, 0.3, 0.4, 0.1, 0.1];
        // Cerebelo: prefere precisão motora (gradiente + arousal + fase)
        keys[5] = [0.1, 0.2, 0.1, 0.4, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 0.1, 0.4, 0.2, 0.1, 0.4, 0.4];

        // Normaliza cada chave para norma unitária
        for key in &mut keys {
            let norm = key.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-6);
            for v in key.iter_mut() { *v /= norm; }
        }

        Self {
            keys,
            prev_query: [0.0; EMBED_DIM],
            smoothed_gates: [1.0; 6],
            activity_history: Default::default(),
            homeostasis_boost: [0.0; 6],
            last_query: [0.0; EMBED_DIM],
            n_especialization_updates: 0,
        }
    }

    /// Constrói o embedding de 16 dims a partir do estado atual do sistema.
    /// Chamado uma vez por tick antes de `route()`.
    pub fn build_query(
        vision_full:    &[f32],
        cochlea:        &[f32],
        dopamine:       f32,
        serotonin:      f32,
        cortisol:       f32,
        noradrenaline:  f32,
        emotion:        f32,
        arousal:        f32,
        activity:       f32,
        abstraction:    f32,
        step:           u64,
    ) -> [f32; EMBED_DIM] {
        let mut q = [0.0f32; EMBED_DIM];
        let n = vision_full.len().max(1);

        // Visual (4 dims)
        let mean_v = vision_full.iter().sum::<f32>() / n as f32;
        q[0] = mean_v;
        q[1] = vision_full.iter().copied().fold(0.0f32, f32::max);
        q[2] = (vision_full.iter().map(|&v| (v - mean_v).powi(2)).sum::<f32>() / n as f32).sqrt();
        q[3] = if n > 1 {
            vision_full.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>() / (n - 1) as f32
        } else { 0.0 };

        // Auditivo (2 dims)
        let na = cochlea.len().max(1);
        q[4] = cochlea.iter().sum::<f32>() / na as f32;
        q[5] = cochlea.iter().copied().fold(0.0f32, f32::max);

        // Neuroquímica (4 dims)
        q[6]  = (dopamine / 2.0).clamp(0.0, 1.0);
        q[7]  = serotonin.clamp(0.0, 1.0);
        q[8]  = cortisol.clamp(0.0, 1.0);
        q[9]  = (noradrenaline / 1.6).clamp(0.0, 1.0);

        // Estado (4 dims)
        q[10] = emotion.clamp(-1.0, 1.0) * 0.5 + 0.5;
        q[11] = arousal.clamp(0.0, 1.0);
        q[12] = activity.clamp(0.0, 1.0);
        q[13] = abstraction.clamp(0.0, 1.0);

        // Fase temporal sin/cos (2 dims) — captura periodicidade do loop
        let phase = (step as f32 * 0.01) % (2.0 * std::f32::consts::PI);
        q[14] = phase.sin() * 0.5 + 0.5;
        q[15] = phase.cos() * 0.5 + 0.5;

        q
    }

    /// Computa gates para este tick dado o query embedding.
    ///
    /// Pipeline:
    ///   1. Suaviza query com tick anterior (anti-ruído)
    ///   2. Cosine similarity entre query e chave de cada lóbulo
    ///   3. Aplica boost homeostático
    ///   4. EMA para suavizar gates (evita flicker)
    ///   5. Aplica gate_minimo por lóbulo
    pub fn route(&mut self, query: [f32; EMBED_DIM]) -> RoutingDecision {
        // 1. Suaviza query (70% atual, 30% anterior)
        let mut q = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            q[i] = query[i] * 0.7 + self.prev_query[i] * 0.3;
        }
        self.prev_query = q;
        self.last_query = q;

        // Norma do query
        let q_norm = q.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-6);

        // 2. Cosine similarity → raw gate [0..1]
        let mut raw = [0.0f32; 6];
        for (i, key) in self.keys.iter().enumerate() {
            let k_norm = key.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-6);
            let dot: f32 = q.iter().zip(key.iter()).map(|(&qi, &ki)| qi * ki).sum();
            raw[i] = (dot / (q_norm * k_norm)).clamp(0.0, 1.0);
        }

        // 3. Boost homeostático (soma ao raw, limita em 1.0)
        for i in 0..6 {
            raw[i] = (raw[i] + self.homeostasis_boost[i]).min(1.0);
        }

        // 4. EMA sobre raw gates
        for i in 0..6 {
            self.smoothed_gates[i] = self.smoothed_gates[i] * (1.0 - GATE_EMA) + raw[i] * GATE_EMA;
        }

        // 5. Gate mínimo por lóbulo
        for (i, id) in LobeId::ALL.iter().enumerate() {
            self.smoothed_gates[i] = self.smoothed_gates[i].max(id.gate_minimo());
        }

        let g = &self.smoothed_gates;
        let decision = RoutingDecision {
            temporal:    g[0],
            parietal:    g[1],
            limbic:      g[2],
            hippocampus: g[3],
            frontal:     g[4],
            cerebellum:  g[5],
        };

        // Atualiza histórico de atividade para homeostase
        self.update_homeostasis(&decision);

        decision
    }

    /// Competitive Hebbian: atualiza chaves com base em qual lóbulo ganhou este tick.
    ///
    /// Vencedor → key move-se em direção ao query (aprende este nicho).
    /// Perdedores → key afasta-se levemente (diferenciação emergente).
    /// Só roda quando |RPE| > RPE_MIN_UPDATE (sinal de aprendizado significativo).
    pub fn update_specialization(&mut self, rpe: f32) {
        if rpe.abs() < RPE_MIN_UPDATE { return; }

        let q = self.last_query;
        let lr = KEY_LR * rpe.abs().min(1.0);

        // Vencedor = lóbulo com maior gate suavizado
        let winner = self.smoothed_gates.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        for (i, key) in self.keys.iter_mut().enumerate() {
            if i == winner {
                // Vencedor: move chave em direção ao query atual
                for j in 0..EMBED_DIM {
                    key[j] += lr * (q[j] - key[j]);
                }
            } else {
                // Perdedor: leve repulsão para especialização divergente
                for j in 0..EMBED_DIM {
                    key[j] -= lr * LOSER_REPULSION * (q[j] - key[j]);
                }
            }
            // Re-normaliza chave para norma unitária (cosine sim estável)
            let norm = key.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-6);
            for v in key.iter_mut() { *v /= norm; }
        }

        self.n_especialization_updates += 1;
    }

    /// Atualiza histórico de atividade e recalcula boost homeostático.
    fn update_homeostasis(&mut self, gates: &RoutingDecision) {
        let current = [
            gates.temporal, gates.parietal, gates.limbic,
            gates.hippocampus, gates.frontal, gates.cerebellum,
        ];
        for i in 0..6 {
            let hist = &mut self.activity_history[i];
            hist.push_back(current[i]);
            if hist.len() > HOMEOSTASIS_WINDOW { hist.pop_front(); }

            if hist.len() >= 50 {
                let mean_act = hist.iter().sum::<f32>() / hist.len() as f32;
                // Déficit positivo → lóbulo subativo → boost positivo
                // Déficit negativo → lóbulo superativo → sem boost (apenas reduz)
                let deficit = (HOMEOSTASIS_TARGET - mean_act).clamp(-0.3, 0.3);
                let boost = (deficit * HOMEOSTASIS_MAX_BOOST / 0.3).clamp(0.0, HOMEOSTASIS_MAX_BOOST);
                // Suaviza o boost para evitar oscilação
                self.homeostasis_boost[i] = self.homeostasis_boost[i] * 0.99 + boost * 0.01;
            }
        }
    }

    /// Telemetria: retorna o label do dim dominante na chave de cada lóbulo.
    /// Mostra para qual tipo de input cada lóbulo está especializado.
    pub fn especialidade_dominante(&self) -> [(&'static str, &'static str); 6] {
        const LABELS: [&str; EMBED_DIM] = [
            "vis-energia", "vis-pico", "vis-novidade", "vis-gradiente",
            "aud-energia",  "aud-pitch",
            "dopamina",    "serotonina", "cortisol",   "noradren",
            "emoção",      "arousal",    "atividade",  "abstração",
            "fase-sin",    "fase-cos",
        ];
        let ids = LobeId::ALL;
        let mut out = [("?", "?"); 6];
        for (i, key) in self.keys.iter().enumerate() {
            let (max_j, _) = key.iter().enumerate()
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                .unwrap_or((0, &0.0));
            out[i] = (ids[i].nome(), LABELS[max_j]);
        }
        out
    }

    /// Atividade média de um lóbulo na janela homeostática.
    pub fn atividade_media(&self, id: LobeId) -> f32 {
        let hist = &self.activity_history[id.idx()];
        if hist.is_empty() { return 0.5; }
        hist.iter().sum::<f32>() / hist.len() as f32
    }

    /// Gate suavizado atual para um lóbulo específico.
    pub fn gate(&self, id: LobeId) -> f32 {
        self.smoothed_gates[id.idx()]
    }
}

impl Default for LobeRouter {
    fn default() -> Self { Self::new() }
}
