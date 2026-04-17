// src/brain_zones/depth_stack.rs
// Pilha de profundidade hierárquica — 3 círculos de compressão (chama branca)
//
// Conceito:
//   Cada camada de abstração comprime a anterior. O lóbulo processa três
//   "visões" simultâneas do mesmo sinal: bruto (D0), comprimido (D1), super-
//   comprimido (D2). A atenção entre profundidades é aprendida via RPE —
//   recompensa positiva fortalece as camadas mais abstratas (D2 cresce);
//   recompensa negativa ancora o sistema nas features brutas (D0 domina).
//
// Análogo ao conceito de "camadas de mana comprimidas":
//   D0 = mana bruta na lâmina (features raw do lóbulo)
//   D1 = primeira compressão (padrões de médio nível)
//   D2 = segunda compressão (estrutura abstrata pura)
//   saída = soma ponderada pelas atenções aprendidas
//
// Falha catastrófica evitada: sem perder controle de qualquer camada
// (attn[k] nunca vai a zero — clamp mínimo em 0.05 garante substrato ativo).

#![allow(dead_code)]

/// Número de camadas de compressão além da camada base (D0).
pub const N_DEPTH_LAYERS: usize = 2;

/// Clamp mínimo de atenção por profundidade — "substrato nunca apaga".
const ATTN_MIN: f32 = 0.05;
/// Clamp máximo de atenção por profundidade.
const ATTN_MAX: f32 = 0.85;
/// Taxa de update de atenção por RPE.
const ATTN_LR: f32 = 0.008;
/// Decaimento exponencial das camadas comprimidas entre ticks.
const DEPTH_DECAY: f32 = 0.85;

/// Pilha de 3 profundidades de abstração.
///
/// - `D0`: saída bruta do lóbulo (n neurônios)
/// - `D1`: max-pool de D0 (n/2 — half resolution)
/// - `D2`: max-pool de D1 (n/4 — quarter resolution)
///
/// A saída (`forward`) expande D1 e D2 de volta para n e soma ponderada.
pub struct DepthStack {
    n: usize,
    /// Saída comprimida nível 1 (n/2)
    pub d1: Vec<f32>,
    /// Saída comprimida nível 2 (n/4)
    pub d2: Vec<f32>,
    /// Pesos de atenção por camada: [D0_weight, D1_weight, D2_weight]
    /// Aprendidos via RPE: abstração recompensada → D2 cresce.
    pub attn: [f32; 3],
    /// Traço de RPE para atualização suave de atenção
    rpe_trace: f32,
}

impl DepthStack {
    pub fn new(n: usize) -> Self {
        let n1 = (n / 2).max(1);
        let n2 = (n / 4).max(1);
        Self {
            n,
            d1: vec![0.0; n1],
            d2: vec![0.0; n2],
            // Inicializa D0 dominante (substrato), D1/D2 latentes
            attn: [0.60, 0.25, 0.15],
            rpe_trace: 0.0,
        }
    }

    /// Processa D0 → comprime para D1, D2 → expande de volta → soma ponderada.
    ///
    /// Retorna vetor de tamanho n com a representação multi-profundidade.
    /// D0 nunca é alterado — apenas lido. D1/D2 atualizam com decaimento.
    pub fn forward(&mut self, d0: &[f32]) -> Vec<f32> {
        let n = d0.len().min(self.n);
        let n1 = self.d1.len();
        let n2 = self.d2.len();

        // Comprime D0 → D1 via max-pool com stride 2
        for j in 0..n1 {
            let i0 = j * 2;
            let i1 = (j * 2 + 1).min(n - 1);
            let max_val = d0.get(i0).copied().unwrap_or(0.0)
                .max(d0.get(i1).copied().unwrap_or(0.0));
            // Decaimento suave: D1 lembra levemente do estado anterior (memória de curto prazo)
            self.d1[j] = self.d1[j] * DEPTH_DECAY + max_val * (1.0 - DEPTH_DECAY);
        }

        // Comprime D1 → D2 via max-pool com stride 2
        for k in 0..n2 {
            let j0 = k * 2;
            let j1 = (k * 2 + 1).min(n1 - 1);
            let max_val = self.d1.get(j0).copied().unwrap_or(0.0)
                .max(self.d1.get(j1).copied().unwrap_or(0.0));
            self.d2[k] = self.d2[k] * DEPTH_DECAY + max_val * (1.0 - DEPTH_DECAY);
        }

        // Expande D1 e D2 de volta para n via nearest-neighbor e combina
        let [a0, a1, a2] = self.attn;
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let v0 = d0.get(i).copied().unwrap_or(0.0);
            // D1 expandido: cada posição i de D1 cobre 2 posições de D0
            let j1 = (i / 2).min(n1 - 1);
            let v1 = self.d1.get(j1).copied().unwrap_or(0.0);
            // D2 expandido: cada posição k de D2 cobre 4 posições de D0
            let k2 = (i / 4).min(n2 - 1);
            let v2 = self.d2.get(k2).copied().unwrap_or(0.0);

            out[i] = (v0 * a0 + v1 * a1 + v2 * a2).clamp(0.0, 1.0);
        }
        out
    }

    /// Atualiza pesos de atenção com base no RPE do tick atual.
    ///
    /// RPE positivo → abstração recompensada → D2 (e D1) ganham atenção.
    /// RPE negativo → sistema retorna às features brutas → D0 ganha atenção.
    /// Clamps garantem que nenhuma camada "apaga" completamente.
    pub fn update_attention(&mut self, rpe: f32) {
        // Traço suaviza oscilações rápidas de RPE
        self.rpe_trace = self.rpe_trace * 0.9 + rpe * 0.1;
        let signal = self.rpe_trace;

        if signal > 0.0 {
            // Recompensa: eleva abstração
            self.attn[2] = (self.attn[2] + ATTN_LR * signal).clamp(ATTN_MIN, ATTN_MAX);
            self.attn[1] = (self.attn[1] + ATTN_LR * signal * 0.5).clamp(ATTN_MIN, ATTN_MAX);
            self.attn[0] = (self.attn[0] - ATTN_LR * signal * 1.5).clamp(ATTN_MIN, ATTN_MAX);
        } else {
            // Punição: ancora no substrato bruto
            self.attn[0] = (self.attn[0] - ATTN_LR * signal * 1.5).clamp(ATTN_MIN, ATTN_MAX);
            self.attn[1] = (self.attn[1] + ATTN_LR * signal * 0.5).clamp(ATTN_MIN, ATTN_MAX);
            self.attn[2] = (self.attn[2] + ATTN_LR * signal).clamp(ATTN_MIN, ATTN_MAX);
        }

        // Re-normaliza para soma = 1.0
        let soma: f32 = self.attn.iter().sum();
        if soma > 0.0 {
            for a in &mut self.attn { *a /= soma; }
        }
    }

    /// Nível de abstração atual (0.0 = D0 domina, 1.0 = D2 domina).
    /// Usado para telemetria — mostra se o sistema está em modo bruto ou abstrato.
    pub fn abstraction_level(&self) -> f32 {
        self.attn[1] * 0.5 + self.attn[2] * 1.0
    }
}
