// src/learning/multimodal.rs — Integração Multimodal Completa
//
// Implementa:
//   1. ConvergenciaMultimodal — une streams visual + audio + interoceptivo
//   2. Predição cross-modal: visual prediz audio esperado (e vice-versa)
//   3. Score de congruência audiovisual (base para efeito McGurk, atenção crossmodal)
//   4. Amplificação: sinal fraco + confirmação multimodal → sinal forte
//   5. Context multimodal: palavras + spike patterns mais recentes
#![allow(dead_code)]

// ── Constantes ────────────────────────────────────────────────────────────────
/// Dimensão dos vetores de feature visual (V1+V2 resumidos)
pub const DIM_VISUAL: usize = 16;
/// Dimensão dos vetores de feature auditiva (bandas FFT resumidas)
pub const DIM_AUDIO: usize = 16;
/// Gain de amplificação quando audiovisual é congruente
const AV_AMPLIFICACAO: f32 = 1.35;
/// Threshold de congruência para disparar binding (McGurk-like)
const BINDING_THRESHOLD: f32 = 0.45;
/// Taxa de aprendizado dos pesos de predição cross-modal
const PRED_LR: f32 = 0.012;
/// Decaimento do contexto ativo entre ticks
const CTX_DECAY: f32 = 0.95;

// ── Convergência Multimodal ───────────────────────────────────────────────────

/// Motor de integração audiovisual com predição cruzada.
pub struct ConvergenciaMultimodal {
    /// Pesos de predição visual→audio [DIM_VISUAL × DIM_AUDIO]
    v2a: Vec<f32>,
    /// Pesos de predição audio→visual [DIM_AUDIO × DIM_VISUAL]
    a2v: Vec<f32>,
    /// Último score de congruência AV
    pub binding_score: f32,
    /// Contexto multimodal ativo — vetor combinado [DIM_VISUAL + DIM_AUDIO]
    pub contexto: Vec<f32>,
    /// Última predição de audio a partir de visual
    pub pred_audio: Vec<f32>,
    /// Última predição de visual a partir de audio
    pub pred_visual: Vec<f32>,
    /// Sinal interoceptivo recente (hardware sensor)
    pub interoceptivo: Vec<f32>,
    /// Número de bindings realizados
    pub n_bindings: u64,
    /// Erro médio de predição cruzada (proxy de surpresa multimodal)
    pub erro_pred_medio: f32,
}

impl ConvergenciaMultimodal {
    pub fn novo() -> Self {
        // Pesos inicializados com pequeno ruído determinístico
        let v2a = init_weights(DIM_VISUAL * DIM_AUDIO, 0.05);
        let a2v = init_weights(DIM_AUDIO * DIM_VISUAL, 0.05);
        Self {
            v2a,
            a2v,
            binding_score: 0.0,
            contexto: vec![0.0; DIM_VISUAL + DIM_AUDIO],
            pred_audio: vec![0.0; DIM_AUDIO],
            pred_visual: vec![0.0; DIM_VISUAL],
            interoceptivo: Vec::new(),
            n_bindings: 0,
            erro_pred_medio: 0.5,
        }
    }

    // ── API principal ──────────────────────────────────────────────────────────

    /// Processa um frame com streams visual + audio.
    /// Retorna o sinal combinado e amplificado + score de congruência.
    pub fn processar(
        &mut self,
        visual: &[f32],
        audio: &[f32],
        emocao: f32,
    ) -> (Vec<f32>, f32) {
        let vis = normalizar(visual, DIM_VISUAL);
        let aud = normalizar(audio, DIM_AUDIO);

        // Predições cruzadas
        self.pred_audio  = self.prever_audio_de_visual(&vis);
        self.pred_visual = self.prever_visual_de_audio(&aud);

        // Congruência: quão bem as predições combinam com o sinal real
        let cong_va = cosseno_sim(&self.pred_audio, &aud);
        let cong_av = cosseno_sim(&self.pred_visual, &vis);
        self.binding_score = (cong_va + cong_av) * 0.5;

        // Amplificação AV: sinal congruente é mais forte
        let amplificado = if self.binding_score > BINDING_THRESHOLD {
            let g = 1.0 + (self.binding_score - BINDING_THRESHOLD) * AV_AMPLIFICACAO;
            let mut out = Vec::with_capacity(vis.len() + aud.len());
            out.extend(vis.iter().map(|&v| v * g));
            out.extend(aud.iter().map(|&v| v * g));
            out
        } else {
            let mut out = Vec::with_capacity(vis.len() + aud.len());
            out.extend_from_slice(&vis);
            out.extend_from_slice(&aud);
            out
        };

        // Atualiza contexto ativo com decaimento
        for (i, &v) in amplificado.iter().enumerate() {
            if i < self.contexto.len() {
                self.contexto[i] = self.contexto[i] * CTX_DECAY + v * (1.0 - CTX_DECAY);
            }
        }

        // Aprende com o erro de predição (Hebbian AV)
        let erro_v = erro_pred(&self.pred_audio, &aud);
        let erro_a = erro_pred(&self.pred_visual, &vis);
        self.atualizar_pesos(&vis, &aud, emocao);
        self.erro_pred_medio = self.erro_pred_medio * 0.98 + (erro_v + erro_a) * 0.5 * 0.02;

        if self.binding_score > BINDING_THRESHOLD {
            self.n_bindings += 1;
        }

        (amplificado, self.binding_score)
    }

    /// Prediz audio esperado a partir de visual.
    pub fn prever_audio_de_visual(&self, visual: &[f32]) -> Vec<f32> {
        matmul(&self.v2a, visual, DIM_VISUAL, DIM_AUDIO)
    }

    /// Prediz visual esperado a partir de audio.
    pub fn prever_visual_de_audio(&self, audio: &[f32]) -> Vec<f32> {
        matmul(&self.a2v, audio, DIM_AUDIO, DIM_VISUAL)
    }

    /// Score de congruência entre visual e audio atuais.
    /// 1.0 = perfeitamente congruentes; 0.0 = ortogonais.
    pub fn score_congruencia(&self, visual: &[f32], audio: &[f32]) -> f32 {
        let vis = normalizar(visual, DIM_VISUAL);
        let aud = normalizar(audio, DIM_AUDIO);
        let pa = self.prever_audio_de_visual(&vis);
        let pv = self.prever_visual_de_audio(&aud);
        (cosseno_sim(&pa, &aud) + cosseno_sim(&pv, &vis)) * 0.5
    }

    /// Define o sinal interoceptivo corrente (body state).
    pub fn atualizar_interoceptivo(&mut self, sinal: &[f32]) {
        self.interoceptivo = sinal.to_vec();
    }

    /// Contexto combinado reduzido a `n` dimensões via média de janelas.
    pub fn contexto_reduzido(&self, n: usize) -> Vec<f32> {
        if self.contexto.is_empty() { return vec![0.0; n]; }
        let step = (self.contexto.len() as f32 / n as f32).max(1.0);
        (0..n).map(|i| {
            let start = (i as f32 * step) as usize;
            let end   = ((i + 1) as f32 * step) as usize;
            let end   = end.min(self.contexto.len());
            if start >= end { return 0.0; }
            self.contexto[start..end].iter().sum::<f32>() / (end - start) as f32
        }).collect()
    }

    // ── Aprendizado ────────────────────────────────────────────────────────────

    fn atualizar_pesos(&mut self, vis: &[f32], aud: &[f32], emocao: f32) {
        let lr = PRED_LR * (1.0 + emocao.abs() * 0.5);

        // v2a: visual → audio (Hebbian: ajusta pesos vis×aud)
        for i in 0..DIM_VISUAL {
            for j in 0..DIM_AUDIO {
                let delta = lr * vis.get(i).copied().unwrap_or(0.0)
                    * (aud.get(j).copied().unwrap_or(0.0)
                       - self.pred_audio.get(j).copied().unwrap_or(0.0));
                let idx = i * DIM_AUDIO + j;
                if idx < self.v2a.len() {
                    self.v2a[idx] = (self.v2a[idx] + delta).clamp(-2.0, 2.0);
                }
            }
        }

        // a2v: audio → visual
        for i in 0..DIM_AUDIO {
            for j in 0..DIM_VISUAL {
                let delta = lr * aud.get(i).copied().unwrap_or(0.0)
                    * (vis.get(j).copied().unwrap_or(0.0)
                       - self.pred_visual.get(j).copied().unwrap_or(0.0));
                let idx = i * DIM_VISUAL + j;
                if idx < self.a2v.len() {
                    self.a2v[idx] = (self.a2v[idx] + delta).clamp(-2.0, 2.0);
                }
            }
        }
    }
}

// ── Math helpers ──────────────────────────────────────────────────────────────

fn normalizar(input: &[f32], dim: usize) -> Vec<f32> {
    if input.is_empty() { return vec![0.0; dim]; }
    if input.len() == dim {
        return input.to_vec();
    }
    // Resampling por média de janelas
    let step = input.len() as f32 / dim as f32;
    (0..dim).map(|i| {
        let s = (i as f32 * step) as usize;
        let e = ((i + 1) as f32 * step) as usize;
        let e = e.min(input.len());
        if s >= e { return 0.0; }
        input[s..e].iter().sum::<f32>() / (e - s) as f32
    }).collect()
}

fn matmul(weights: &[f32], input: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    (0..out_dim).map(|j| {
        (0..in_dim).map(|i| {
            weights.get(i * out_dim + j).copied().unwrap_or(0.0)
                * input.get(i).copied().unwrap_or(0.0)
        }).sum::<f32>().tanh()
    }).collect()
}

fn cosseno_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { (dot / (na * nb)).clamp(-1.0, 1.0) }
}

fn erro_pred(pred: &[f32], real: &[f32]) -> f32 {
    if pred.is_empty() || real.is_empty() { return 1.0; }
    let n = pred.len().min(real.len()) as f32;
    pred.iter().zip(real).map(|(p, r)| (p - r).abs()).sum::<f32>() / n
}

fn init_weights(n: usize, escala: f32) -> Vec<f32> {
    (0..n).map(|i| {
        let h = (i as u64).wrapping_mul(2654435761).wrapping_add(1013904223);
        ((h & 0xFFFF) as f32 / 32767.5 - 1.0) * escala
    }).collect()
}
