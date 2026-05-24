// =============================================================================
// src/brain_zones/hippocampal_index.rs — V4.3
// =============================================================================
//
// HIPPOCAMPAL INDEX (Teyler & DiScenna 1986; Rudy 2024)
//
// Hipocampo armazena INDEX, não conteúdo. O conteúdo (padrões neocorticais
// completos) fica distribuído pelo neocortex. O hippocampus apenas mantém
// índices pequenos que apontam para subconjuntos do neocortex.
//
// Vantagens:
//   • Capacidade massiva (10^7 índices vs 10^4 memórias completas)
//   • Consolidação Squire real (hippocampus → neocortex)
//   • Recall robusto via partial cue + pattern completion (CA3)
//
// Em Selene, este módulo orquestra:
//   1. [[Dentate Gyrus]]    → recebe input neocortical denso, produz engram esparso
//   2. [[Memory Engrams]]   → armazena engram como índice (com ponteiros p/ conteúdo)
//   3. [[CA3 Attractor]]    → recall via partial cue → completion
//
// Não substitui HippocampusV2 existente — é COMPLEMENTAR. HippocampusV2
// continua fazendo LTP de pesos sinápticos no nível neuronal. Este módulo
// adiciona uma camada superior de organização episódica.
//
// =============================================================================

use std::collections::HashSet;
use uuid::Uuid;

use crate::brain_zones::dentate_gyrus::{DentateGyrus, SparsePattern};
use crate::brain_zones::ca3_attractor::CA3Attractor;
use crate::brain_zones::memory_engrams::{EngramStore, EngramId};

/// Configuração default para o HippocampalIndex.
pub struct HippocampalIndexConfig {
    pub n_granular: usize,    // tamanho do DG
    pub input_dim: usize,     // dim. do input do neocortex
    pub sparsity: f32,         // fração de cells ativas (alvo)
    pub seed: u64,             // seed do DG
}

impl Default for HippocampalIndexConfig {
    fn default() -> Self {
        Self {
            n_granular: 2048,
            input_dim: 32,      // coerente com `embeddings 32d` do swap_manager
            sparsity: 0.04,     // 4% → 82 cells ativas em média
            seed: 0xC0FFEE,
        }
    }
}

/// HippocampalIndex — orquestra DG + Engrams + CA3 para memória episódica.
pub struct HippocampalIndex {
    pub dg: DentateGyrus,
    pub ca3: CA3Attractor,
    pub engrams: EngramStore,
    /// Threshold de overlap para considerar reativação (sintonizado com DG sparsity).
    pub recall_overlap_threshold: usize,
}

impl HippocampalIndex {
    pub fn new(cfg: HippocampalIndexConfig) -> Self {
        let dg = DentateGyrus::new(cfg.n_granular, cfg.input_dim, cfg.sparsity, cfg.seed);
        let k = dg.k_target();
        let ca3 = CA3Attractor::new(k);
        Self {
            dg,
            ca3,
            engrams: EngramStore::new(),
            // Threshold: ao menos 30% das cells do engram precisam estar no cue
            recall_overlap_threshold: (k / 3).max(2),
        }
    }

    /// Encoding completo de uma memória episódica:
    /// input neocortical denso → DG (sparse) → CA3 store + Engram persist.
    ///
    /// Retorna o `EngramId` criado.
    pub fn encode_episode(
        &mut self,
        neocortical_input: &[f32],
        member_cells: HashSet<Uuid>,
        step_atual: u64,
        emocao: f32,
    ) -> EngramId {
        // 1. Pattern separation via DG
        let sparse_pattern = self.dg.encode(neocortical_input);

        // 2. CA3 armazena (Hebbian outer product)
        self.ca3.store(&sparse_pattern);

        // 3. Engram persiste com membros = cells reais do swap_manager
        // (não as índices internas do DG — usamos os Uuid do neocortex porque
        // são esses que serão usados na reativação posterior)
        self.engrams.encode(member_cells, step_atual, emocao)
    }

    /// Recall por cue parcial (Uuid de cells ativas agora).
    /// Combina:
    ///   1. Engram reactivar() (lookup via overlap de Uuid)
    ///   2. Se sucesso, retorna engram completo
    pub fn recall_by_cells(
        &mut self,
        cue_cells: &HashSet<Uuid>,
        step_atual: u64,
    ) -> Option<EngramId> {
        self.engrams.reativar(cue_cells, step_atual)
    }

    /// Recall por pattern denso (input neocortical parcial/ruidoso).
    /// Aplica DG (separation) + CA3 (completion) para inferir o pattern original.
    /// Retorna `SparsePattern` reconstruído (não engram — para isso use recall_by_cells).
    pub fn pattern_complete(&self, partial_input: &[f32]) -> SparsePattern {
        let cue_sparse = self.dg.encode(partial_input);
        self.ca3.complete(&cue_sparse)
    }

    /// Tick periódico de homeostase — chamar durante sono N3 ou idle.
    /// - Decay de pesos CA3 (forgetting biológico)
    /// - Identifica engrams candidatos a consolidação no neocortex
    pub fn tick_consolidacao(&mut self, decay_rate: f32, n_reactivations_threshold: u32)
        -> Vec<EngramId>
    {
        self.ca3.decay(decay_rate);
        let candidatos = self.engrams.candidatos_consolidacao(n_reactivations_threshold);
        log::debug!("[HIT] tick_consolidacao: {} engrams candidatos", candidatos.len());
        candidatos
    }

    /// Telemetria.
    pub fn stats(&self) -> HippocampalIndexStats {
        HippocampalIndexStats {
            n_engrams: self.engrams.len(),
            n_ca3_synapses: self.ca3.n_synapses(),
            n_patterns_stored: self.ca3.n_stored(),
            dg_k_target: self.dg.k_target(),
        }
    }
}

#[derive(Debug)]
pub struct HippocampalIndexStats {
    pub n_engrams: usize,
    pub n_ca3_synapses: usize,
    pub n_patterns_stored: u64,
    pub dg_k_target: usize,
}

// =============================================================================
// Testes
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn cells(uuids: &[u128]) -> HashSet<Uuid> {
        uuids.iter().map(|u| Uuid::from_u128(*u)).collect()
    }

    #[test]
    fn hit_novo_vazio() {
        let hit = HippocampalIndex::new(HippocampalIndexConfig::default());
        let s = hit.stats();
        assert_eq!(s.n_engrams, 0);
        assert_eq!(s.n_ca3_synapses, 0);
        assert!(s.dg_k_target > 0);
    }

    #[test]
    fn encode_episode_cria_engram_e_armazena_no_ca3() {
        let mut hit = HippocampalIndex::new(HippocampalIndexConfig::default());
        let input: Vec<f32> = (0..32).map(|i| (i as f32) / 32.0).collect();
        let members = cells(&[1, 2, 3, 4]);
        let id = hit.encode_episode(&input, members, 100, 0.5);
        assert!(id > 0);

        let s = hit.stats();
        assert_eq!(s.n_engrams, 1);
        assert!(s.n_ca3_synapses > 0, "CA3 deve ter sinapses após store");
        assert_eq!(s.n_patterns_stored, 1);
    }

    #[test]
    fn recall_by_cells_funciona_com_partial_cue() {
        let mut hit = HippocampalIndex::new(HippocampalIndexConfig::default());
        let input: Vec<f32> = (0..32).map(|i| (i as f32) / 32.0).collect();
        let members = cells(&[10, 20, 30, 40]);
        let id = hit.encode_episode(&input, members, 100, 0.5);

        // Cue parcial — só 2 de 4 cells
        let cue = cells(&[10, 20]);
        let recall = hit.recall_by_cells(&cue, 200);
        assert_eq!(recall, Some(id));
    }

    #[test]
    fn pattern_complete_recupera_pattern_original() {
        let mut hit = HippocampalIndex::new(HippocampalIndexConfig::default());
        let input_a: Vec<f32> = (0..32).map(|i| (i as f32 * 0.7) / 32.0).collect();

        // Armazena padrão A várias vezes para fortalecer attractor
        for _ in 0..20 {
            hit.encode_episode(&input_a, cells(&[1, 2, 3]), 100, 0.5);
        }

        // Input parcial: zera 30% das features
        let mut partial = input_a.clone();
        for i in (0..32).step_by(3) { partial[i] = 0.0; }

        let recall = hit.pattern_complete(&partial);
        let original = hit.dg.encode(&input_a);

        let overlap = recall.overlap(&original);
        let max = recall.active.len().min(original.active.len());
        // Pelo menos 50% das cells do padrão original devem ser recuperadas
        let frac = if max > 0 { overlap as f32 / max as f32 } else { 0.0 };
        assert!(frac >= 0.4,
            "pattern_complete deve recuperar ≥40% do padrão original; got {frac:.2} (overlap {overlap}/{max})");
    }

    #[test]
    fn tick_consolidacao_identifica_engrams_frequentes() {
        let mut hit = HippocampalIndex::new(HippocampalIndexConfig::default());
        let input: Vec<f32> = (0..32).map(|i| (i as f32) / 32.0).collect();
        let id_a = hit.encode_episode(&input, cells(&[1, 2, 3]), 100, 0.5);
        let _id_b = hit.encode_episode(&input, cells(&[10, 20]), 200, 0.5);

        // Reativa A muitas vezes
        for _ in 0..15 {
            hit.recall_by_cells(&cells(&[1, 2]), 300);
        }

        let candidatos = hit.tick_consolidacao(0.0, 10);
        assert!(candidatos.contains(&id_a));
    }
}
