// =============================================================================
// src/learning/successor.rs — V4.3
// =============================================================================
//
// SUCCESSOR REPRESENTATION (SR) — "mapa preditivo" hippocampal
//
// Implementa Dayan 1993 / Stachenfeld 2017 ("The hippocampus as a predictive map").
//
// SR codifica, para cada par (s, s'), a probabilidade descontada de visitar s'
// no futuro dado que estamos em s agora:
//
//     M(s, s') = E[ Σ γ^t · 𝟙(s_t = s') | s_0 = s ]
//
// Diferença vs Q-learning:
//   • Q(s, a) → escalar específico de uma reward
//   • M(s, s') → matriz agnóstica à reward; ao mudar R, basta recombinar
//
// Update TD para SR (Dayan 1993):
//   M(s_prev, s') += α · [δ(s_curr, s') + γ · M(s_curr, s') − M(s_prev, s')]
//
// Em Selene:
//   • s ≡ concept_id (u32, FNV-1a hash de palavra)
//   • Compatível com codificação localista (1 conceito = 1 neurônio)
//   • Habilita planning (rollout virtual sobre M) e priority replay (need(s))
//
// =============================================================================

use std::collections::HashMap;

/// Fator de desconto — coerente com `rl.rs::GAMMA`.
const GAMMA: f32 = 0.95;

/// Taxa de aprendizado do TD da SR.
const ALPHA: f32 = 0.05;

/// Threshold de poda — entradas M(s, s') abaixo disso são removidas.
const PRUNE_THRESHOLD: f32 = 0.005;

/// Capacidade máxima de entradas (sparse).
const CAP_ENTRIES: usize = 500_000;

/// Successor Representation sparse sobre concept_ids u32.
///
/// Implementa a SR de Dayan/Stachenfeld para Selene. Mantém um mapa esparso
/// `(s, s') → M_ss'` populando apenas pares observados (não os 2^64 possíveis).
#[derive(Debug, Clone, Default)]
pub struct SuccessorRepresentation {
    matrix: HashMap<(u32, u32), f32>,
    /// Janela curta de estados recentes — usada para difusão da update
    /// (substitui o requisito teórico de "todos os pares" por uma vizinhança
    /// recente, computacionalmente viável).
    recentes: std::collections::VecDeque<u32>,
    /// Total de updates aplicadas (telemetria).
    total_updates: u64,
}

impl SuccessorRepresentation {
    pub fn new() -> Self {
        Self {
            matrix: HashMap::with_capacity(10_000),
            recentes: std::collections::VecDeque::with_capacity(64),
            total_updates: 0,
        }
    }

    /// Update da SR após transição `s_prev → s_curr`. Implementa TD-SR clássico:
    ///   • Diagonal (s_prev, s_curr): TD com δ = 1 (acabei de visitar s_curr a partir de s_prev)
    ///   • Off-diagonal (s_prev, s') para s' ≠ s_curr: propaga γ · M(s_curr, s') (bootstrap futuro)
    ///
    /// Para eficiência, só propaga para s' em uma janela DEDUPLICADA dos estados
    /// recentes (max 16 únicos). Isso evita o problema de "decay falso" quando
    /// uma sequência longa visita um estado raro: a diagonal não decai apenas
    /// porque o estado deixou de aparecer.
    pub fn update(&mut self, s_prev: u32, s_curr: u32) {
        // 1. Diagonal: TD com delta = 1 (visitou s_curr) e bootstrap γ · M(s_curr, s_curr)
        let m_curr_diag = self.matrix.get(&(s_curr, s_curr)).copied().unwrap_or(0.0);
        let m_prev_diag = self.matrix.get(&(s_prev, s_curr)).copied().unwrap_or(0.0);
        let novo_diag = m_prev_diag + ALPHA * (1.0 + GAMMA * m_curr_diag - m_prev_diag);
        if novo_diag.abs() > PRUNE_THRESHOLD {
            self.matrix.insert((s_prev, s_curr), novo_diag);
        }

        // 2. Off-diagonal: TD bootstrap γ · M(s_curr, s') sobre janela única.
        // Usamos um set deduplicado dos estados recentes (cap 16) — preserva
        // generalização sem causar decay artificial da diagonal.
        let alvos: std::collections::HashSet<u32> = self.recentes.iter()
            .copied()
            .filter(|sp| *sp != s_curr) // diagonal já tratada acima
            .collect();
        for sp in alvos {
            let m_prev = self.matrix.get(&(s_prev, sp)).copied().unwrap_or(0.0);
            let m_curr = self.matrix.get(&(s_curr, sp)).copied().unwrap_or(0.0);
            // delta = 0 (s_curr não é sp); só bootstrap
            let novo = m_prev + ALPHA * (GAMMA * m_curr - m_prev);
            if novo.abs() > PRUNE_THRESHOLD {
                self.matrix.insert((s_prev, sp), novo);
            }
        }

        // 3. Atualiza janela de recentes (cap 16 — pequena, focada no contexto atual)
        self.recentes.push_back(s_curr);
        if self.recentes.len() > 16 {
            self.recentes.pop_front();
        }

        self.total_updates += 1;

        if self.matrix.len() > CAP_ENTRIES {
            self.podar();
        }
    }

    /// Consulta direta da SR: M(s, s').
    pub fn get(&self, s: u32, sp: u32) -> f32 {
        self.matrix.get(&(s, sp)).copied().unwrap_or(0.0)
    }

    /// Computa "need(s)" — soma da SR partindo do estado atual.
    /// Usado por [[Priority Replay]] para estimar a importância de um update.
    pub fn need(&self, s_atual: u32, s_alvo: u32) -> f32 {
        // need = M(s_atual, s_alvo) — probabilidade descontada de visitar s_alvo
        self.get(s_atual, s_alvo)
    }

    /// Computa V(s) combinando SR com função de reward.
    /// V(s) = Σ_{s'} M(s, s') · R(s')
    pub fn value(&self, s: u32, rewards: &HashMap<u32, f32>) -> f32 {
        rewards.iter()
            .map(|(&sp, &r)| self.get(s, sp) * r)
            .sum()
    }

    /// Retorna os top-k estados mais prováveis de visitar a partir de `s`.
    /// Útil para planning (rollout virtual) e graph walk dirigido.
    pub fn top_k_next(&self, s: u32, k: usize) -> Vec<(u32, f32)> {
        let mut candidatos: Vec<(u32, f32)> = self.matrix.iter()
            .filter(|((src, _), _)| *src == s)
            .map(|((_, sp), &m)| (*sp, m))
            .collect();
        candidatos.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidatos.into_iter().take(k).collect()
    }

    /// Remove entradas com magnitude < threshold para manter sparsity.
    fn podar(&mut self) {
        self.matrix.retain(|_, v| v.abs() > PRUNE_THRESHOLD);
    }

    /// Número de entradas armazenadas (proxy de "experiência acumulada").
    pub fn n_entries(&self) -> usize {
        self.matrix.len()
    }

    /// Total de updates aplicadas.
    pub fn total_updates(&self) -> u64 {
        self.total_updates
    }

    /// Reset completo (esquece tudo). Use com cuidado.
    pub fn resetar(&mut self) {
        self.matrix.clear();
        self.recentes.clear();
        self.total_updates = 0;
    }

    // ─── Persistência ───────────────────────────────────────────────────────
    // Formato binário compacto: [n_entries: u64][src: u32][dst: u32][m: f32] × n

    /// Salva a SR em arquivo binário.
    pub async fn salvar_async(&self, caminho: &str) -> std::io::Result<()> {
        let mut buf = Vec::with_capacity(8 + self.matrix.len() * 12);
        buf.extend_from_slice(&(self.matrix.len() as u64).to_le_bytes());
        for ((src, dst), m) in &self.matrix {
            buf.extend_from_slice(&src.to_le_bytes());
            buf.extend_from_slice(&dst.to_le_bytes());
            buf.extend_from_slice(&m.to_le_bytes());
        }
        tokio::fs::write(caminho, buf).await?;
        log::info!("[SR] Persistida: {} entries → {}", self.matrix.len(), caminho);
        Ok(())
    }

    /// Restaura SR de arquivo salvo previamente.
    pub async fn carregar_async(&mut self, caminho: &str) -> std::io::Result<()> {
        let bytes = tokio::fs::read(caminho).await?;
        if bytes.len() < 8 { return Ok(()); }
        let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap_or([0u8; 8])) as usize;
        let mut pos = 8usize;
        let mut restaurados = 0usize;
        while pos + 12 <= bytes.len() && restaurados < n {
            let src = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap_or([0u8;4]));
            let dst = u32::from_le_bytes(bytes[pos+4..pos+8].try_into().unwrap_or([0u8;4]));
            let m   = f32::from_le_bytes(bytes[pos+8..pos+12].try_into().unwrap_or([0u8;4]));
            self.matrix.insert((src, dst), m);
            pos += 12;
            restaurados += 1;
        }
        log::info!("[SR] Restaurada: {} entries de {}", restaurados, caminho);
        Ok(())
    }
}

// =============================================================================
// Testes
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sr_nova_esta_vazia() {
        let sr = SuccessorRepresentation::new();
        assert_eq!(sr.n_entries(), 0);
        assert_eq!(sr.get(1, 2), 0.0);
    }

    #[test]
    fn update_cria_entrada_diagonal() {
        let mut sr = SuccessorRepresentation::new();
        sr.update(1, 2);
        // Após update, (1, 2) deve ter valor > 0 (delta=1 quando s_curr == s')
        assert!(sr.get(1, 2) > 0.0, "M(1,2) deve crescer após observar 1→2");
    }

    #[test]
    fn update_repetida_converge() {
        let mut sr = SuccessorRepresentation::new();
        // Simula 1000 transições 1→2 (mesma sequência)
        for _ in 0..1000 {
            sr.update(1, 2);
        }
        // M(1, 2) deve convergir para algo próximo de 1.0 (visita sempre 2)
        let m = sr.get(1, 2);
        assert!(m > 0.7, "M(1,2) após 1000 updates deve estar perto de 1.0; got {m}");
    }

    #[test]
    fn cadeia_propaga_via_recentes() {
        let mut sr = SuccessorRepresentation::new();
        // Cadeia 1 → 2 → 3 → 4
        for _ in 0..100 {
            sr.update(1, 2);
            sr.update(2, 3);
            sr.update(3, 4);
        }
        // M(1, 2) deve ser alta
        assert!(sr.get(1, 2) > 0.3);
        // M(1, 3) deve ser positiva (transitividade via janela recentes)
        assert!(sr.get(1, 3) > 0.0,
            "M(1, 3) deve ser >0 (propaga via janela); got {}", sr.get(1, 3));
    }

    #[test]
    fn top_k_next_retorna_destinos_prováveis() {
        // TD-SR decai entradas antigas — para validar "mais provável" intercalamos
        // visitas de forma que a frequência relativa recente reflita o esperado.
        let mut sr = SuccessorRepresentation::new();
        for _ in 0..25 {
            sr.update(1, 2);
            sr.update(1, 2);
            sr.update(1, 3);
            sr.update(1, 2);
            sr.update(1, 4);
        }
        let top = sr.top_k_next(1, 3);
        assert!(!top.is_empty(), "deve haver destinos no top-k");
        let estados: Vec<u32> = top.iter().map(|(s, _)| *s).collect();
        // Estado 2 (visitado 3× mais que cada outro) deve aparecer no top-3
        assert!(estados.contains(&2),
            "estado 2 (mais frequente) deve estar no top-3; got {:?}", estados);
        // Valor de M deve estar em ordem decrescente
        for i in 1..top.len() {
            assert!(top[i-1].1 >= top[i].1,
                "top-k mal ordenado: {} >= {}?", top[i-1].1, top[i].1);
        }
    }

    #[test]
    fn value_combina_sr_com_rewards() {
        let mut sr = SuccessorRepresentation::new();
        for _ in 0..100 { sr.update(1, 2); }
        let mut rewards = HashMap::new();
        rewards.insert(2, 1.0); // s' = 2 vale 1.0
        rewards.insert(3, 0.5); // s' = 3 vale 0.5 (não visitado)
        let v = sr.value(1, &rewards);
        assert!(v > 0.0, "V(1) deve ser positivo dado M(1,2)>0 e R(2)>0; got {v}");
    }

    #[test]
    fn poda_remove_entradas_pequenas() {
        let mut sr = SuccessorRepresentation::new();
        // Insere entrada que ficará pequena
        sr.matrix.insert((1, 2), 0.001);
        // E uma grande
        sr.matrix.insert((3, 4), 0.5);
        sr.podar();
        assert!(sr.matrix.get(&(1, 2)).is_none(),
            "Entrada pequena deveria ter sido podada");
        assert!(sr.matrix.get(&(3, 4)).is_some());
    }

    #[test]
    fn reset_zera_tudo() {
        let mut sr = SuccessorRepresentation::new();
        for _ in 0..10 { sr.update(1, 2); }
        sr.resetar();
        assert_eq!(sr.n_entries(), 0);
        assert_eq!(sr.total_updates(), 0);
    }
}
