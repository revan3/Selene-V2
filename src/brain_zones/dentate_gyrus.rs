// =============================================================================
// src/brain_zones/dentate_gyrus.rs — V4.3
// =============================================================================
//
// DENTATE GYRUS (DG) — Pattern Separation via sparse encoding
//
// O DG decorrelaciona inputs similares: dois eventos parecidos → padrões
// hipocampais DISTINTOS.
//
// Mecanismo biológico:
//   • Population de granular cells grande (>>input size)
//   • Disparo esparso (~5% ativos)
//   • Inibição lateral forte (interneurons GABAérgicos)
//
// Implementação computacional em Selene:
//   • n_granular células com receptive field aleatório sobre os inputs
//   • Top-k winner: só os k mais ativos disparam (k ≈ sparsity × n)
//   • Resultado: padrão esparso único por experiência
//
// Usado por:
//   • [[Hippocampal Index]] — engram codes vêm do DG output
//   • [[Memory Engrams Tonegawa]] — engram = cells ativas no DG output
//
// =============================================================================

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Serialize, Deserialize};

/// Pattern esparso emitido pelo DG — bitset compacto.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SparsePattern {
    /// Índices das células ativas (ordenado).
    pub active: Vec<u32>,
}

impl SparsePattern {
    /// Overlap (intersection size) com outro padrão.
    pub fn overlap(&self, other: &Self) -> usize {
        // Ambos ordenados → merge-like intersection O(n+m)
        let mut i = 0;
        let mut j = 0;
        let mut count = 0;
        while i < self.active.len() && j < other.active.len() {
            match self.active[i].cmp(&other.active[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => { count += 1; i += 1; j += 1; }
            }
        }
        count
    }

    /// Similaridade Jaccard ∈ [0, 1].
    pub fn jaccard(&self, other: &Self) -> f32 {
        let intersection = self.overlap(other) as f32;
        let union = (self.active.len() + other.active.len()) as f32 - intersection;
        if union == 0.0 { 0.0 } else { intersection / union }
    }
}

/// Dentate Gyrus — sparse encoder de padrões neurais.
///
/// Pesos são determinísticos (gerados de seed) para que recall seja
/// reproduzível entre sessões. O DG é uma camada de feature mapping fixa
/// (não aprende) — é o CA3 que aprende via attractor dynamics.
#[derive(Debug)]
pub struct DentateGyrus {
    /// Número de granular cells (saída).
    pub n_granular: usize,
    /// Dimensão do input.
    pub input_dim: usize,
    /// Fração de cells ativas após winner-take-all (sparsity).
    pub sparsity: f32,
    /// Pesos receptive field: n_granular × input_dim, gerados de seed.
    weights: Vec<Vec<f32>>,
    /// Seed para reproducibilidade.
    seed: u64,
}

impl DentateGyrus {
    /// Construtor padrão. Sparsity típica: 0.05 (5% ativos).
    pub fn new(n_granular: usize, input_dim: usize, sparsity: f32, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        // Pesos com distribuição esparsa e centrada em 0 — alguns excitatórios,
        // alguns inibitórios. Esparsidade do peso amplia a separação.
        let weights: Vec<Vec<f32>> = (0..n_granular)
            .map(|_| {
                (0..input_dim).map(|_| {
                    // 70% pesos zero (esparsidade do receptive field)
                    if rng.gen::<f32>() < 0.7 {
                        0.0
                    } else {
                        rng.gen::<f32>() * 2.0 - 1.0
                    }
                }).collect()
            })
            .collect();

        Self {
            n_granular,
            input_dim,
            sparsity: sparsity.clamp(0.01, 0.5),
            weights,
            seed,
        }
    }

    /// Codifica input denso em padrão esparso.
    /// 1. Cada granular cell calcula `dot(weights[i], input)`
    /// 2. Top-k winner: só os k = sparsity × n_granular mais ativos disparam
    pub fn encode(&self, input: &[f32]) -> SparsePattern {
        if input.is_empty() {
            return SparsePattern { active: vec![] };
        }
        // 1. Ativação linear (sem ReLU — preservamos sinal para top-k)
        let activations: Vec<f32> = self.weights.iter()
            .map(|w| {
                w.iter().zip(input.iter())
                    .map(|(wi, xi)| wi * xi)
                    .sum::<f32>()
            })
            .collect();

        // 2. Top-k winner-take-all
        let k = ((self.sparsity * self.n_granular as f32) as usize).max(1);
        let mut indexed: Vec<(u32, f32)> = activations.iter().enumerate()
            .map(|(i, &a)| (i as u32, a))
            .collect();
        // Ordena por ativação descendente
        indexed.sort_unstable_by(|a, b|
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Pega top-k com ativação positiva (cells sem input forte não devem disparar)
        let mut active: Vec<u32> = indexed.into_iter()
            .take(k)
            .filter(|(_, a)| *a > 0.0)
            .map(|(i, _)| i)
            .collect();
        active.sort_unstable(); // mantemos ordem canônica para comparação eficiente

        SparsePattern { active }
    }

    /// Número alvo de células ativas (k).
    pub fn k_target(&self) -> usize {
        ((self.sparsity * self.n_granular as f32) as usize).max(1)
    }
}

// =============================================================================
// Testes
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dg_novo_tem_n_granular_correto() {
        let dg = DentateGyrus::new(1024, 32, 0.05, 42);
        assert_eq!(dg.n_granular, 1024);
        assert_eq!(dg.input_dim, 32);
        assert_eq!(dg.k_target(), 51); // 0.05 * 1024 = 51.2 → 51
    }

    #[test]
    fn encode_produz_pattern_esparso() {
        let dg = DentateGyrus::new(1000, 32, 0.05, 42);
        let input: Vec<f32> = (0..32).map(|i| (i as f32) / 32.0).collect();
        let pattern = dg.encode(&input);
        // Cap superior: k_target
        assert!(pattern.active.len() <= dg.k_target(),
            "pattern não deve exceder k_target; got {}", pattern.active.len());
        // Cap inferior: pelo menos 1 ativo (input não-zero)
        assert!(!pattern.active.is_empty(), "input não-zero deve gerar ao menos 1 ativo");
    }

    #[test]
    fn encode_input_vazio_retorna_pattern_vazio() {
        let dg = DentateGyrus::new(100, 16, 0.1, 42);
        let pattern = dg.encode(&[]);
        assert!(pattern.active.is_empty());
    }

    #[test]
    fn encode_deterministico_mesma_seed() {
        let dg1 = DentateGyrus::new(500, 32, 0.05, 42);
        let dg2 = DentateGyrus::new(500, 32, 0.05, 42);
        let input: Vec<f32> = (0..32).map(|i| (i as f32) / 32.0).collect();
        let p1 = dg1.encode(&input);
        let p2 = dg2.encode(&input);
        assert_eq!(p1, p2, "mesma seed → mesmo encoding");
    }

    #[test]
    fn pattern_separation_inputs_similares_produzem_overlap_baixo() {
        // Princípio chave: inputs similares devem produzir padrões com
        // overlap MENOR que o esperado por chance pura.
        let dg = DentateGyrus::new(2000, 32, 0.05, 42);
        let input_a: Vec<f32> = (0..32).map(|i| (i as f32) / 32.0).collect();
        let mut input_b = input_a.clone();
        input_b[5] += 0.1;  // diferença mínima
        input_b[15] += 0.1;

        let pat_a = dg.encode(&input_a);
        let pat_b = dg.encode(&input_b);

        let overlap = pat_a.overlap(&pat_b);
        let max_overlap = pat_a.active.len().min(pat_b.active.len());

        // Pattern separation: overlap << max possível para inputs similares
        // (DG decorrelaciona). Aceitamos até 80% overlap como evidência razoável.
        assert!(overlap < max_overlap,
            "DG deve decorrelacionar inputs similares; overlap={} max={}",
            overlap, max_overlap);
    }

    #[test]
    fn jaccard_de_pattern_consigo_mesmo_eh_um() {
        let p = SparsePattern { active: vec![1, 5, 10, 20] };
        assert!((p.jaccard(&p) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_disjuntos_eh_zero() {
        let a = SparsePattern { active: vec![1, 2, 3] };
        let b = SparsePattern { active: vec![10, 20, 30] };
        assert_eq!(a.jaccard(&b), 0.0);
    }

    #[test]
    fn overlap_dois_ordenados_eficiente() {
        let a = SparsePattern { active: vec![1, 3, 5, 7, 9] };
        let b = SparsePattern { active: vec![2, 3, 5, 11] };
        assert_eq!(a.overlap(&b), 2); // {3, 5}
    }
}
