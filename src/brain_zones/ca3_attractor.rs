// =============================================================================
// src/brain_zones/ca3_attractor.rs — V4.3
// =============================================================================
//
// CA3 ATTRACTOR — Pattern Completion via Hopfield-like dynamics
//
// O CA3 do hipocampo é um circuito altamente recurrent que implementa
// completion: cue parcial → padrão completo estável.
//
// Mecanismo:
//   • Sinapses recurrent entre cells (auto-associativas)
//   • Pesos aprendem via outer product de padrões armazenados (Hebbian)
//   • Recall: ativa cells do cue, deixa dynamics convergir para attractor
//
// Implementação simplificada para Selene:
//   • Wraps de [[Modern Hopfield Networks]] adaptado para SparsePattern
//   • Pesos crescem proporcional a freq de co-ocorrência (Hebbian online)
//   • Iteração: para cada cell, calcula input weighted dos vizinhos; top-k WTA
//   • Convergência: para quando padrão estabiliza ou max_iter
//
// Compatível com [[Dentate Gyrus]] (entrada esparsa) e
// [[Memory Engrams Tonegawa]] (recall via attractor).
//
// =============================================================================

use crate::brain_zones::dentate_gyrus::SparsePattern;
use std::collections::HashMap;

/// Limite de iterações do attractor (converge ou para).
const MAX_ITER: usize = 8;

/// Cap superior de pesos (estabilidade numérica).
const W_MAX: f32 = 5.0;

/// Aprendizado Hebbian — ganho por co-ocorrência.
const ETA: f32 = 0.05;

/// CA3 — rede recurrent que armazena padrões via Hebbian e completa via attractor.
#[derive(Debug, Default)]
pub struct CA3Attractor {
    /// Pesos sinápticos esparsos entre cells: (i, j) → w_ij.
    /// Armazena apenas (i < j) por simetria — sinaptizamos pares lookup-aware.
    weights: HashMap<(u32, u32), f32>,
    /// Cap superior de cells ativas durante completion (deve casar com DG).
    pub k_target: usize,
    /// Número de padrões armazenados (telemetria).
    n_stored: u64,
}

impl CA3Attractor {
    pub fn new(k_target: usize) -> Self {
        Self {
            weights: HashMap::with_capacity(4096),
            k_target: k_target.max(1),
            n_stored: 0,
        }
    }

    /// Armazena um padrão via Hebbian outer product.
    /// Para cada par (i, j) ativo no pattern, incrementa w_ij.
    pub fn store(&mut self, pattern: &SparsePattern) {
        let n = pattern.active.len();
        for i in 0..n {
            for j in (i+1)..n {
                let key = self.canonical_key(pattern.active[i], pattern.active[j]);
                let w = self.weights.entry(key).or_insert(0.0);
                *w = (*w + ETA).min(W_MAX);
            }
        }
        self.n_stored += 1;
    }

    /// Completion: dado cue parcial, itera attractor até convergir.
    /// Retorna pattern completo.
    pub fn complete(&self, cue: &SparsePattern) -> SparsePattern {
        if cue.active.is_empty() {
            return SparsePattern { active: vec![] };
        }

        let mut state = cue.clone();
        for _ in 0..MAX_ITER {
            let next = self.step(&state);
            if next == state {
                break; // convergiu
            }
            state = next;
        }
        state
    }

    /// Um passo da dynamics do attractor — winner-take-all sobre input dos vizinhos.
    fn step(&self, current: &SparsePattern) -> SparsePattern {
        // 1. Coleta todas as cells potencialmente ativas (vizinhas das ativas)
        let mut candidatos: HashMap<u32, f32> = HashMap::new();

        // Cells atualmente ativas mantêm self-input (estabilidade)
        for &c in &current.active {
            *candidatos.entry(c).or_insert(0.0) += 1.0;
        }

        // Cada par (ativa, vizinha) contribui peso w_ij à vizinha
        for &a in &current.active {
            // Note: como armazenamos apenas (i < j), precisamos checar ambas ordens
            for (&(i, j), &w) in &self.weights {
                if i == a {
                    *candidatos.entry(j).or_insert(0.0) += w;
                } else if j == a {
                    *candidatos.entry(i).or_insert(0.0) += w;
                }
            }
        }

        // 2. Winner-take-all: top-k candidatos
        let mut indexed: Vec<(u32, f32)> = candidatos.into_iter().collect();
        indexed.sort_by(|a, b|
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut active: Vec<u32> = indexed.into_iter()
            .take(self.k_target)
            .filter(|(_, a)| *a > 0.0)
            .map(|(i, _)| i)
            .collect();
        active.sort_unstable();
        SparsePattern { active }
    }

    /// Chave canônica (i, j) com i < j — economiza metade dos pesos por simetria.
    fn canonical_key(&self, a: u32, b: u32) -> (u32, u32) {
        if a < b { (a, b) } else { (b, a) }
    }

    /// Número de sinapses ativas (telemetria).
    pub fn n_synapses(&self) -> usize { self.weights.len() }

    /// Padrões armazenados.
    pub fn n_stored(&self) -> u64 { self.n_stored }

    /// Decay homeostático — sem aprendizado, pesos decaem (forgetting).
    /// Chamar periodicamente para evitar saturação.
    pub fn decay(&mut self, rate: f32) {
        let factor = 1.0 - rate.clamp(0.0, 1.0);
        self.weights.retain(|_, w| {
            *w *= factor;
            *w > 0.01 // remove pesos muito pequenos
        });
    }
}

// =============================================================================
// Testes
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn pat(active: &[u32]) -> SparsePattern {
        let mut v = active.to_vec();
        v.sort_unstable();
        SparsePattern { active: v }
    }

    #[test]
    fn ca3_novo_sem_sinapses() {
        let ca3 = CA3Attractor::new(5);
        assert_eq!(ca3.n_synapses(), 0);
        assert_eq!(ca3.n_stored(), 0);
    }

    #[test]
    fn store_cria_sinapses_entre_pares_ativos() {
        let mut ca3 = CA3Attractor::new(5);
        ca3.store(&pat(&[1, 2, 3]));
        // 3 cells → C(3, 2) = 3 pares
        assert_eq!(ca3.n_synapses(), 3);
        assert_eq!(ca3.n_stored(), 1);
    }

    #[test]
    fn store_repetido_aumenta_pesos() {
        let mut ca3 = CA3Attractor::new(5);
        for _ in 0..10 {
            ca3.store(&pat(&[1, 2]));
        }
        let key = ca3.canonical_key(1, 2);
        let w = ca3.weights.get(&key).copied().unwrap_or(0.0);
        // ETA=0.05, 10 stores → w ~0.5 (capped em W_MAX)
        assert!(w >= 0.4 && w <= W_MAX,
            "peso após 10 stores deve estar entre 0.4 e W_MAX; got {w}");
    }

    #[test]
    fn complete_de_cue_vazio_retorna_vazio() {
        let ca3 = CA3Attractor::new(5);
        let r = ca3.complete(&pat(&[]));
        assert!(r.active.is_empty());
    }

    #[test]
    fn complete_recall_simple_pattern() {
        let mut ca3 = CA3Attractor::new(4);
        // Armazena padrão [1, 2, 3, 4] muitas vezes
        for _ in 0..30 {
            ca3.store(&pat(&[1, 2, 3, 4]));
        }
        // Cue parcial [1, 2] deve completar para o padrão original
        let recall = ca3.complete(&pat(&[1, 2]));
        // Deve incluir as cells ativas no padrão original (overlap alto)
        let original = pat(&[1, 2, 3, 4]);
        let overlap = recall.overlap(&original);
        assert!(overlap >= 3,
            "recall de cue [1,2] deve recuperar pelo menos 3/4 cells do padrão original; got overlap={overlap}");
    }

    #[test]
    fn complete_estavel_apos_armazenamento() {
        let mut ca3 = CA3Attractor::new(3);
        for _ in 0..50 {
            ca3.store(&pat(&[1, 2, 3]));
        }
        // Recall do próprio padrão completo deve ser estável
        let original = pat(&[1, 2, 3]);
        let recall = ca3.complete(&original);
        assert_eq!(recall, original, "padrão completo deve ser ponto fixo");
    }

    #[test]
    fn decay_reduz_pesos() {
        let mut ca3 = CA3Attractor::new(5);
        for _ in 0..10 {
            ca3.store(&pat(&[1, 2]));
        }
        let key = ca3.canonical_key(1, 2);
        let w_antes = ca3.weights.get(&key).copied().unwrap();
        ca3.decay(0.5);
        let w_depois = ca3.weights.get(&key).copied().unwrap();
        assert!(w_depois < w_antes, "decay deve reduzir peso");
    }

    #[test]
    fn decay_remove_pesos_pequenos() {
        let mut ca3 = CA3Attractor::new(5);
        ca3.weights.insert((1, 2), 0.005); // muito pequeno
        ca3.weights.insert((3, 4), 1.0);   // saudável
        ca3.decay(0.0); // sem decay efetivo
        // Mesmo sem decay, retain remove os < 0.01
        assert!(ca3.weights.get(&(1, 2)).is_none(),
            "peso < 0.01 deve ser removido pelo retain");
        assert!(ca3.weights.get(&(3, 4)).is_some());
    }
}
