// =============================================================================
// src/brain_zones/memory_engrams.rs — V4.3
// =============================================================================
//
// MEMORY ENGRAMS (Tonegawa 2012+) — ensembles celulares específicos por memória
//
// Cada memória episódica é codificada por um ensemble específico de neurônios
// hipocampais (Uuid). Reativar o ensemble = recall da memória.
//
// Discoveries-chave (Tonegawa Lab, MIT):
//   • 2012: engrams existem fisicamente; optogenética prova suficiência
//   • 2013: manipulação de engram cria falsas memórias
//   • 2016: em modelo Alzheimer's, engrams existem mas acesso bloqueado
//
// Implementação em Selene:
//   • Engram = HashSet<Uuid> com metadados (timestamp, emoção, n_recalls)
//   • Encoding: ao consolidar memória nova, marca subset ativo + boost STDP intra
//   • Reativação: cue partial → overlap scoring → WTA → recall completo
//   • Index reverso (cell → engrams) para reativação eficiente
//
// Compatível com [[Hippocampal Indexing Theory]]: engrams atuam como índices.
//
// =============================================================================

use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Threshold mínimo de overlap (cells em comum) para considerar reativação.
const MIN_OVERLAP: usize = 2;

/// Capacidade máxima de engrams (FIFO por idade quando excede).
const CAP_ENGRAMS: usize = 50_000;

/// Identificador interno de engram.
pub type EngramId = u64;

/// Engram episódico — um ensemble específico de neurônios + metadados.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Engram {
    pub id: EngramId,
    /// Conjunto de neurônios membros (Uuid do swap_manager).
    pub cell_ensemble: HashSet<Uuid>,
    /// Timestamp da criação (step do loop neural).
    pub encoding_step: u64,
    /// Valência emocional da experiência [-1.0, 1.0].
    pub emocao: f32,
    /// Vezes que o engram foi reativado (proxy de saliência).
    pub n_reactivations: u32,
    /// Última reativação (step) — usado para decay.
    pub last_reactivation_step: u64,
}

impl Engram {
    /// Quão saliente é este engram agora — combina frequência + recência.
    pub fn salience(&self, step_atual: u64) -> f32 {
        let recency = if self.last_reactivation_step > 0 {
            let dt = step_atual.saturating_sub(self.last_reactivation_step) as f32;
            (-dt / 10_000.0).exp() // decai com tau ≈ 10k steps
        } else { 0.0 };
        let frequency = (self.n_reactivations as f32).ln().max(0.0);
        frequency * 0.5 + recency * 0.5
    }
}

/// Store de engrams com index reverso (cell → engrams).
///
/// O índice reverso é o que torna reativação eficiente: dado um cue parcial,
/// achamos rapidamente engrams candidatos sem varrer todos.
#[derive(Debug, Default)]
pub struct EngramStore {
    /// Todos os engrams ativos.
    engrams: HashMap<EngramId, Engram>,
    /// Index reverso: cell → engrams que a contêm.
    cell_to_engrams: HashMap<Uuid, Vec<EngramId>>,
    next_id: EngramId,
}

impl EngramStore {
    pub fn new() -> Self {
        Self {
            engrams: HashMap::with_capacity(1024),
            cell_to_engrams: HashMap::with_capacity(4096),
            next_id: 1,
        }
    }

    /// Codifica novo engram para um conjunto de neurônios ativos.
    /// Retorna o ID gerado.
    pub fn encode(
        &mut self,
        active_cells: HashSet<Uuid>,
        encoding_step: u64,
        emocao: f32,
    ) -> EngramId {
        let id = self.next_id;
        self.next_id += 1;

        // Index reverso
        for c in &active_cells {
            self.cell_to_engrams.entry(*c).or_default().push(id);
        }

        let engram = Engram {
            id,
            cell_ensemble: active_cells,
            encoding_step,
            emocao,
            n_reactivations: 0,
            last_reactivation_step: 0,
        };
        self.engrams.insert(id, engram);

        // Poda LRU se passou da capacidade
        if self.engrams.len() > CAP_ENGRAMS {
            self.podar_lru();
        }

        id
    }

    /// Reativação por cue parcial — encontra engram com maior overlap.
    /// Implementa pattern completion: cue parcial → engram completo.
    /// Atualiza contadores de reativação no engram vencedor.
    pub fn reativar(&mut self, cue: &HashSet<Uuid>, step_atual: u64) -> Option<EngramId> {
        if cue.is_empty() { return None; }

        // Conta overlap usando o index reverso (eficiente)
        let mut scores: HashMap<EngramId, usize> = HashMap::new();
        for c in cue {
            if let Some(engram_ids) = self.cell_to_engrams.get(c) {
                for eid in engram_ids {
                    *scores.entry(*eid).or_insert(0) += 1;
                }
            }
        }

        // Winner-take-all com threshold mínimo de overlap
        let (vencedor_id, overlap) = scores.into_iter()
            .max_by_key(|(_, s)| *s)
            .filter(|(_, s)| *s >= MIN_OVERLAP)?;

        // Atualiza contadores
        if let Some(engram) = self.engrams.get_mut(&vencedor_id) {
            engram.n_reactivations += 1;
            engram.last_reactivation_step = step_atual;
            log::debug!("[ENGRAM] reativado id={} overlap={} reactivations={}",
                vencedor_id, overlap, engram.n_reactivations);
        }

        Some(vencedor_id)
    }

    /// Acesso ao engram completo (após reativação ou para introspecção).
    pub fn get(&self, id: EngramId) -> Option<&Engram> {
        self.engrams.get(&id)
    }

    /// Remove engram (consolidação completa → conteúdo já está no neocortex).
    pub fn remove(&mut self, id: EngramId) -> Option<Engram> {
        let engram = self.engrams.remove(&id)?;
        // Limpa index reverso
        for c in &engram.cell_ensemble {
            if let Some(v) = self.cell_to_engrams.get_mut(c) {
                v.retain(|&eid| eid != id);
                if v.is_empty() {
                    self.cell_to_engrams.remove(c);
                }
            }
        }
        Some(engram)
    }

    /// Engrams ordenados por saliência (frequência × recência).
    pub fn top_salient(&self, k: usize, step_atual: u64) -> Vec<&Engram> {
        let mut todos: Vec<&Engram> = self.engrams.values().collect();
        todos.sort_by(|a, b| b.salience(step_atual)
            .partial_cmp(&a.salience(step_atual))
            .unwrap_or(std::cmp::Ordering::Equal));
        todos.into_iter().take(k).collect()
    }

    /// Engrams candidatos a consolidação (alta reativação → migrar p/ neocortex).
    pub fn candidatos_consolidacao(&self, threshold: u32) -> Vec<EngramId> {
        self.engrams.values()
            .filter(|e| e.n_reactivations >= threshold)
            .map(|e| e.id)
            .collect()
    }

    /// Poda LRU: remove os mais antigos sem reativação.
    fn podar_lru(&mut self) {
        let n_remover = self.engrams.len() / 10; // remove 10%
        let mut por_idade: Vec<(EngramId, u64)> = self.engrams.iter()
            .map(|(id, e)| {
                let ultima = e.last_reactivation_step.max(e.encoding_step);
                (*id, ultima)
            })
            .collect();
        por_idade.sort_by_key(|(_, s)| *s); // mais antigos primeiro
        let ids_remover: Vec<EngramId> = por_idade.iter()
            .take(n_remover)
            .map(|(id, _)| *id)
            .collect();
        for id in ids_remover {
            self.remove(id);
        }
        log::info!("[ENGRAM] LRU pruned: {} engrams removidos", n_remover);
    }

    pub fn len(&self) -> usize { self.engrams.len() }
    pub fn is_empty(&self) -> bool { self.engrams.is_empty() }

    /// Itera sobre todos os engrams (referência).
    pub fn iter(&self) -> impl Iterator<Item = &Engram> {
        self.engrams.values()
    }

    /// Persistência — formato JSON estruturado (mesmo padrão de selene_*.json).
    pub async fn salvar_async(&self, caminho: &str) -> std::io::Result<()> {
        let payload = serde_json::json!({
            "selene_engrams_v1": {
                "n_engrams": self.engrams.len(),
                "next_id": self.next_id,
                "engrams": self.engrams.values().collect::<Vec<_>>(),
            }
        });
        let s = serde_json::to_string_pretty(&payload)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        // Rename atômico para evitar corrupção
        let tmp = format!("{caminho}.tmp");
        tokio::fs::write(&tmp, s).await?;
        tokio::fs::rename(&tmp, caminho).await?;
        log::info!("[ENGRAM] Persistidos: {} engrams → {}", self.engrams.len(), caminho);
        Ok(())
    }

    pub async fn carregar_async(&mut self, caminho: &str) -> std::io::Result<()> {
        let bytes = tokio::fs::read(caminho).await?;
        let val: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let root = val.get("selene_engrams_v1").ok_or_else(||
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing root key"))?;
        if let Some(next) = root.get("next_id").and_then(|v| v.as_u64()) {
            self.next_id = next;
        }
        if let Some(arr) = root.get("engrams").and_then(|v| v.as_array()) {
            for v in arr {
                if let Ok(engram) = serde_json::from_value::<Engram>(v.clone()) {
                    // Rebuild index reverso
                    for c in &engram.cell_ensemble {
                        self.cell_to_engrams.entry(*c).or_default().push(engram.id);
                    }
                    self.engrams.insert(engram.id, engram);
                }
            }
        }
        log::info!("[ENGRAM] Restaurados: {} engrams de {}",
            self.engrams.len(), caminho);
        Ok(())
    }
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
    fn store_novo_vazio() {
        let s = EngramStore::new();
        assert!(s.is_empty());
    }

    #[test]
    fn encode_cria_engram_com_id_unico() {
        let mut s = EngramStore::new();
        let id1 = s.encode(cells(&[1, 2, 3]), 100, 0.5);
        let id2 = s.encode(cells(&[4, 5, 6]), 200, -0.3);
        assert_ne!(id1, id2);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn reativar_encontra_engram_com_maior_overlap() {
        let mut s = EngramStore::new();
        let id_a = s.encode(cells(&[1, 2, 3, 4]), 100, 0.5);   // tem 1,2,3,4
        let _id_b = s.encode(cells(&[5, 6, 7, 8]), 200, 0.3);   // tem 5,6,7,8
        // Cue [1, 2, 9] → overlap com A = 2, com B = 0 → vence A
        let cue = cells(&[1, 2, 9]);
        let recall = s.reativar(&cue, 300);
        assert_eq!(recall, Some(id_a));
        // n_reactivations deve subir
        assert_eq!(s.get(id_a).unwrap().n_reactivations, 1);
        assert_eq!(s.get(id_a).unwrap().last_reactivation_step, 300);
    }

    #[test]
    fn reativar_falha_se_overlap_abaixo_threshold() {
        let mut s = EngramStore::new();
        s.encode(cells(&[1, 2, 3, 4]), 100, 0.5);
        let cue = cells(&[1, 9, 10]); // overlap = 1, abaixo de MIN_OVERLAP=2
        let recall = s.reativar(&cue, 200);
        assert!(recall.is_none(),
            "overlap=1 deve falhar (MIN_OVERLAP=2)");
    }

    #[test]
    fn reativar_cue_vazio_retorna_none() {
        let mut s = EngramStore::new();
        s.encode(cells(&[1, 2, 3]), 100, 0.5);
        let recall = s.reativar(&HashSet::new(), 200);
        assert!(recall.is_none());
    }

    #[test]
    fn remove_limpa_index_reverso() {
        let mut s = EngramStore::new();
        let id = s.encode(cells(&[1, 2, 3]), 100, 0.5);
        s.remove(id);
        // cell 1 não deve mais estar associada a engram nenhum
        assert!(s.cell_to_engrams.get(&Uuid::from_u128(1)).is_none()
            || s.cell_to_engrams.get(&Uuid::from_u128(1)).unwrap().is_empty());
    }

    #[test]
    fn top_salient_ordena_por_saliencia() {
        let mut s = EngramStore::new();
        let id_a = s.encode(cells(&[1, 2]), 100, 0.5);
        let id_b = s.encode(cells(&[3, 4]), 100, 0.5);
        // Reativa B muitas vezes
        for _ in 0..10 {
            s.reativar(&cells(&[3, 4]), 200);
        }
        let top = s.top_salient(2, 250);
        assert_eq!(top[0].id, id_b, "B reativado mais → mais saliente que A");
        assert_eq!(top[1].id, id_a);
    }

    #[test]
    fn candidatos_consolidacao_filtra_por_reativacoes() {
        let mut s = EngramStore::new();
        let id_a = s.encode(cells(&[1, 2]), 100, 0.5);
        let _id_b = s.encode(cells(&[3, 4]), 100, 0.5);
        for _ in 0..15 {
            s.reativar(&cells(&[1, 2]), 200);
        }
        let candidatos = s.candidatos_consolidacao(10);
        assert!(candidatos.contains(&id_a));
        assert_eq!(candidatos.len(), 1);
    }
}
