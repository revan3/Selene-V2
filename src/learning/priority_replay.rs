// =============================================================================
// src/learning/priority_replay.rs — V4.3
// =============================================================================
//
// PRIORITY REPLAY (Mattar & Daw 2018) — replay estratégico durante sono
//
// Substitui o critério atual "emotion > 0.5 → replay" por uma priorização
// formal baseada em EVB (Expected Value of Backup):
//
//     EVB(s) = need(s) × gain(s)
//
// Onde:
//   • need(s) ≈ probabilidade descontada de visitar s no futuro (vem da SR)
//   • gain(s) ≈ |TD error| do último update (proxy do quanto Q melhoraria)
//
// Mattar & Daw (2018) mostraram que priority replay com EVB é quase
// state-of-the-art em RL e explica observações biológicas de hippocampal
// replay (awake-SWRs marcando experiências futuras importantes).
//
// Em Selene:
//   • Durante VIGÍLIA: cada novo TD error → push no buffer com EVB calculado
//   • Durante N3/REM: pop top-k EVB e re-treinar Q-table
//   • Substitui o "replay por emotion > 0.5" em rem_semantico()
//   • Quando SR estiver disponível: usa need() real; senão usa |TD| puro
//
// =============================================================================

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::VecDeque;

use crate::learning::successor::SuccessorRepresentation;

/// Capacidade máxima do buffer (FIFO quando excede).
const CAP_BUFFER: usize = 10_000;

/// Item no priority queue — wrap de (EVB, estado, timestamp).
///
/// `Ord` implementado para BinaryHeap (max-heap por EVB) — mantém estabilidade
/// usando timestamp como tiebreaker (LIFO em caso de empate).
#[derive(Debug, Clone, Copy)]
pub struct EvbItem {
    pub evb: f32,
    pub estado: u32,
    pub td_error: f32,
    pub timestamp: u64,
}

impl PartialEq for EvbItem {
    fn eq(&self, other: &Self) -> bool {
        self.evb == other.evb && self.timestamp == other.timestamp
    }
}
impl Eq for EvbItem {}

impl PartialOrd for EvbItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for EvbItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // max-heap por EVB; em empate, mais recente primeiro (timestamp maior)
        match self.evb.partial_cmp(&other.evb).unwrap_or(Ordering::Equal) {
            Ordering::Equal => self.timestamp.cmp(&other.timestamp),
            o => o,
        }
    }
}

/// Buffer de priority replay baseado em EVB.
///
/// Durante vigília acumula experiências com EVB. Durante sono N3/REM, pop top-k
/// para re-treino. Substitui o critério "emotion > 0.5" do replay atual.
#[derive(Debug, Default)]
pub struct PriorityReplayBuffer {
    heap: BinaryHeap<EvbItem>,
    /// Histórico circular dos últimos N pops (telemetria + evita re-replay imediato).
    historico_pop: VecDeque<u32>,
    n_pushes_total: u64,
    n_pops_total: u64,
}

impl PriorityReplayBuffer {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::with_capacity(1024),
            historico_pop: VecDeque::with_capacity(256),
            n_pushes_total: 0,
            n_pops_total: 0,
        }
    }

    /// Adiciona experiência ao buffer. Usa SR para `need(s)` se disponível.
    ///
    /// - `s_atual`: estado de onde a transição partiu (para need)
    /// - `s_alvo`: estado que recebeu o TD update
    /// - `td_error`: TD error do update
    /// - `step`: tick atual (vira timestamp)
    /// - `sr`: SR opcional para need real (None → usa 1.0 como proxy)
    pub fn push(
        &mut self,
        s_atual: u32,
        s_alvo: u32,
        td_error: f32,
        step: u64,
        sr: Option<&SuccessorRepresentation>,
    ) {
        let need = sr.map(|s| s.need(s_atual, s_alvo).max(0.01)).unwrap_or(1.0);
        let gain = td_error.abs();
        let evb = need * gain;

        // Filtra ruído — só vale enfileirar se há sinal real
        if evb < 1e-4 { return; }

        self.heap.push(EvbItem {
            evb,
            estado: s_alvo,
            td_error,
            timestamp: step,
        });
        self.n_pushes_total += 1;

        // FIFO quando excede cap — remove o pior EVB (menor) reconstruindo o heap
        if self.heap.len() > CAP_BUFFER {
            // BinaryHeap não tem pop_min eficiente; convertemos, podamos e refazemos
            let mut itens: Vec<EvbItem> = self.heap.drain().collect();
            itens.sort_unstable_by(|a, b| b.cmp(a)); // ordem decrescente
            itens.truncate(CAP_BUFFER);
            self.heap.extend(itens);
        }
    }

    /// Pop top-k itens por EVB. Chamar durante N3/REM.
    /// Retorna lista ordenada (maior EVB primeiro).
    pub fn pop_top_k(&mut self, k: usize) -> Vec<EvbItem> {
        let mut out = Vec::with_capacity(k);
        for _ in 0..k {
            match self.heap.pop() {
                Some(item) => {
                    self.historico_pop.push_back(item.estado);
                    if self.historico_pop.len() > 256 {
                        self.historico_pop.pop_front();
                    }
                    self.n_pops_total += 1;
                    out.push(item);
                }
                None => break,
            }
        }
        out
    }

    /// Inspecionar próximo item sem remover.
    pub fn peek(&self) -> Option<&EvbItem> {
        self.heap.peek()
    }

    /// Reseta o buffer (esquece tudo pendente — use ao acordar de sono longo).
    pub fn clear(&mut self) {
        self.heap.clear();
        self.historico_pop.clear();
    }

    pub fn len(&self) -> usize { self.heap.len() }
    pub fn is_empty(&self) -> bool { self.heap.is_empty() }
    pub fn n_pushes_total(&self) -> u64 { self.n_pushes_total }
    pub fn n_pops_total(&self) -> u64 { self.n_pops_total }
}

// =============================================================================
// Testes
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_novo_vazio() {
        let b = PriorityReplayBuffer::new();
        assert!(b.is_empty());
        assert!(b.peek().is_none());
    }

    #[test]
    fn push_filtra_evb_baixo() {
        let mut b = PriorityReplayBuffer::new();
        b.push(1, 2, 1e-6, 0, None); // EVB ~1e-6, abaixo do threshold
        assert!(b.is_empty(), "EVB muito baixo não deve entrar no buffer");
    }

    #[test]
    fn pop_retorna_maior_evb_primeiro() {
        let mut b = PriorityReplayBuffer::new();
        b.push(1, 10, 0.1,  0, None); // evb = 0.1
        b.push(2, 20, 0.5,  1, None); // evb = 0.5 (maior)
        b.push(3, 30, 0.3,  2, None); // evb = 0.3
        let top = b.pop_top_k(3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].estado, 20, "estado 20 com maior EVB deve sair primeiro");
        assert_eq!(top[1].estado, 30);
        assert_eq!(top[2].estado, 10);
    }

    #[test]
    fn evb_usa_sr_quando_disponivel() {
        let mut sr = SuccessorRepresentation::new();
        // Cria padrão: estado 1 → estado 2 com alta frequência
        for _ in 0..100 { sr.update(1, 2); }
        // Estado 1 → estado 9 muito raro
        sr.update(1, 9);

        let mut b = PriorityReplayBuffer::new();
        // Mesmo TD error (0.5) para ambos, mas need diferente
        b.push(1, 2, 0.5, 0, Some(&sr));
        b.push(1, 9, 0.5, 1, Some(&sr));
        let top = b.pop_top_k(2);
        assert_eq!(top.len(), 2);
        // O estado 2 (alta need) deve ter saído primeiro
        assert_eq!(top[0].estado, 2,
            "Com SR, estado mais visitado (alta need) deve ter prioridade");
    }

    #[test]
    fn cap_buffer_descarta_pior_evb() {
        let mut b = PriorityReplayBuffer::new();
        // Insere acima da cap
        for i in 0..(CAP_BUFFER + 10) {
            b.push(i as u32, (i + 1) as u32, 0.1 + (i as f32 / 1e6), i as u64, None);
        }
        // Buffer não deve exceder a cap
        assert!(b.len() <= CAP_BUFFER,
            "Buffer não deve exceder CAP_BUFFER; got {}", b.len());
    }

    #[test]
    fn pop_de_buffer_vazio_retorna_vazio() {
        let mut b = PriorityReplayBuffer::new();
        let out = b.pop_top_k(5);
        assert!(out.is_empty());
    }

    #[test]
    fn telemetria_contadores_corretos() {
        let mut b = PriorityReplayBuffer::new();
        b.push(1, 2, 0.5, 0, None);
        b.push(3, 4, 0.5, 1, None);
        assert_eq!(b.n_pushes_total(), 2);
        let _ = b.pop_top_k(1);
        assert_eq!(b.n_pops_total(), 1);
        assert_eq!(b.len(), 1);
    }
}
