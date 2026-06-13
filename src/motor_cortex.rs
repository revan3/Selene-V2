// =============================================================================
// src/motor_cortex.rs — V4.6.1 — Córtex Motor (saída de ação discreta)
// =============================================================================
//
// Conceito A — primeira fatia: dá à Selene a SAÍDA motora que faltava para
// agir num computador (jogos, navegador) via o daemon `selene_agent.py`.
//
// Desenho ATOR-CRÍTICO biologicamente coerente:
//   • MotorCortex (aqui)   = ATOR  → escolhe 1 de 4 ações (Q-learning ε-greedy).
//   • rl.rs / dopamina     = CRÍTICO → valor de estado + RPE (já existe).
//   • A recompensa também é injetada em `recompensa_pendente` → dopamina → STDP,
//     para que o substrato neural aprenda em paralelo (não só a Q-table).
//
// BOOTSTRAP: esta Q-table é o atalho para fechar o loop sensório-motor já.
// Próximo passo (biologizar): a seleção de ação deve emergir da atividade do
// gânglio basal a partir do frame processado pelo occipital — não de um hash.
// =============================================================================

use std::collections::HashMap;

/// Ações discretas de um jogo de 4 teclas (Snake/2048/etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acao {
    Cima,
    Baixo,
    Esquerda,
    Direita,
}

impl Acao {
    pub const TODAS: [Acao; 4] = [Acao::Cima, Acao::Baixo, Acao::Esquerda, Acao::Direita];

    #[inline]
    pub fn idx(self) -> usize {
        match self {
            Acao::Cima => 0,
            Acao::Baixo => 1,
            Acao::Esquerda => 2,
            Acao::Direita => 3,
        }
    }

    #[inline]
    pub fn de_idx(i: usize) -> Acao {
        Acao::TODAS[i % 4]
    }

    /// Nome da tecla enviado ao daemon (mapeado para a seta correspondente).
    pub fn tecla(self) -> &'static str {
        match self {
            Acao::Cima => "up",
            Acao::Baixo => "down",
            Acao::Esquerda => "left",
            Acao::Direita => "right",
        }
    }
}

/// Máximo de estados na Q-table (anti-OOM; limpa quando excede).
const MAX_ESTADOS: usize = 100_000;

pub struct MotorCortex {
    /// Q(estado_hash) → valores das 4 ações.
    q: HashMap<u64, [f32; 4]>,
    pub epsilon: f32, // exploração (decai com a experiência)
    alpha: f32,       // taxa de aprendizagem
    gamma: f32,       // desconto temporal
    ultimo_estado: Option<u64>,
    ultima_acao: Option<usize>,
    rng: u64, // xorshift determinístico (sem dependência externa)
    pub passos: u64,
    pub recompensa_total: f32,
}

impl Default for MotorCortex {
    fn default() -> Self {
        Self::new()
    }
}

impl MotorCortex {
    pub fn new() -> Self {
        Self {
            q: HashMap::with_capacity(4096),
            epsilon: 0.30,
            alpha: 0.15,
            gamma: 0.95,
            ultimo_estado: None,
            ultima_acao: None,
            rng: 0x9E3779B97F4A7C15,
            passos: 0,
            recompensa_total: 0.0,
        }
    }

    #[inline]
    fn rand_f32(&mut self) -> f32 {
        // xorshift64* — barato e determinístico.
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let v = self.rng.wrapping_mul(0x2545F4914F6CDD1D);
        ((v >> 40) as f32) / (1u64 << 24) as f32 // [0,1)
    }

    /// Seleciona uma ação para o estado atual (ε-greedy) e memoriza o par
    /// (estado, ação) para o próximo `aprender`.
    pub fn selecionar(&mut self, estado: u64) -> Acao {
        let qs = *self.q.get(&estado).unwrap_or(&[0.0; 4]);
        let idx = if self.rand_f32() < self.epsilon {
            (self.rand_f32() * 4.0) as usize % 4 // exploração
        } else {
            // argmax (desempate pelo primeiro)
            let mut best = 0;
            for i in 1..4 {
                if qs[i] > qs[best] {
                    best = i;
                }
            }
            best
        };
        self.ultimo_estado = Some(estado);
        self.ultima_acao = Some(idx);
        Acao::de_idx(idx)
    }

    /// Atualiza a Q-table com a recompensa observada APÓS a última ação,
    /// tendo o `novo_estado` como resultado (Q-learning TD).
    pub fn aprender(&mut self, novo_estado: u64, recompensa: f32) {
        if let (Some(s), Some(a)) = (self.ultimo_estado, self.ultima_acao) {
            let q_next = self
                .q
                .get(&novo_estado)
                .map(|v| v.iter().copied().fold(f32::MIN, f32::max))
                .unwrap_or(0.0);
            let alvo = recompensa + self.gamma * q_next;
            let entry = self.q.entry(s).or_insert([0.0; 4]);
            entry[a] += self.alpha * (alvo - entry[a]);
            // Decai a exploração lentamente até um piso.
            self.epsilon = (self.epsilon * 0.9999).max(0.05);
        }
        self.passos += 1;
        self.recompensa_total += recompensa;

        if self.q.len() > MAX_ESTADOS {
            // Poda simples (bootstrap): zera e recomeça o mapeamento.
            self.q.clear();
        }
    }

    /// Reinicia a memória episódica do par estado/ação (fim de episódio).
    pub fn fim_episodio(&mut self) {
        self.ultimo_estado = None;
        self.ultima_acao = None;
    }

    pub fn n_estados(&self) -> usize {
        self.q.len()
    }
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn aprende_acao_recompensada() {
        let mut mc = MotorCortex::new();
        mc.epsilon = 0.0; // sem exploração para o teste ser determinístico
        let estado = 42u64;
        // Recompensa repetidamente a ação tomada em `estado` → o seu Q sobe.
        for _ in 0..50 {
            let _a = mc.selecionar(estado);
            mc.aprender(estado, 1.0);
        }
        let qs = mc.q.get(&estado).copied().unwrap();
        let max = qs.iter().copied().fold(f32::MIN, f32::max);
        assert!(max > 0.5, "Q da ação recompensada deve subir (got {max:?})");
        assert!(mc.recompensa_total >= 50.0);
    }

    #[test]
    fn acoes_mapeiam_para_teclas() {
        assert_eq!(Acao::Cima.tecla(), "up");
        assert_eq!(Acao::de_idx(3), Acao::Direita);
        assert_eq!(Acao::Esquerda.idx(), 2);
    }
}
