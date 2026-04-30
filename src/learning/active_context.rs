// src/learning/active_context.rs
// ActiveContext — janela de contexto compartilhada entre as 4 vozes do Multi-Self V3.4.
//
// Comunicação inter-vozes via primitivas atômicas (Bitset + AtomicU32) para evitar
// contenção de Mutex no loop crítico de 200Hz no Ryzen 3500U (4 cores físicos).
//
// Design:
//   - 64 slots fixos (CONTEXT_SLOTS) para concept_ids ativos no foco atencional.
//   - Cada slot tem `concept_id: AtomicU32` (0 = vazio) e `salience: AtomicU32`
//     (f32 codificado como fixed-point com escala 1e6).
//   - `active_mask: AtomicU64` é um bitmap — bit i marca slot i ativo (lookup O(1)).
//   - `generation: AtomicU64` incrementa a cada mutação. Vozes leem essa marca para
//     detectar "Repolarização Sináptica" (Diretriz 3) sem reler todo o array.
//
// Não há heap allocation no caminho quente. Todas as operações são lock-free.

#![allow(dead_code)]

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Número de slots de conceitos ativos no foco atencional simultâneo.
/// Limitado a 64 para caber no `active_mask: AtomicU64` (1 bit por slot).
pub const CONTEXT_SLOTS: usize = 64;

/// Escala fixed-point para codificar f32 saliência (0..1) em u32.
/// 1.0 ↔ 1_000_000. Suficiente para 6 dígitos de precisão.
const SALIENCE_SCALE: u32 = 1_000_000;

#[inline]
fn f32_to_fp(x: f32) -> u32 {
    (x.clamp(0.0, 1.0) * SALIENCE_SCALE as f32) as u32
}
#[inline]
fn fp_to_f32(x: u32) -> f32 {
    x as f32 / SALIENCE_SCALE as f32
}

/// Janela de contexto ativo lock-free, compartilhada por todas as 4 vozes.
pub struct ActiveContext {
    concept_ids: [AtomicU32; CONTEXT_SLOTS],
    salience:    [AtomicU32; CONTEXT_SLOTS],
    /// Bit i = slot i ativo. Operações atômicas via fetch_or / fetch_and.
    active_mask: AtomicU64,
    /// Contador de gerações — incrementa a cada `inject_concept` ou `clear_slot`.
    generation:  AtomicU64,
    /// Tick da última injeção lateral (Diretriz 2 — Escuta Ativa).
    last_lateral_tick: AtomicU64,
}

impl ActiveContext {
    pub fn new() -> Self {
        // Inicialização explícita pois `[T; N]` exige Copy para inicialização literal,
        // e AtomicU32 não é Copy. Workaround com std::array::from_fn (estável desde 1.63).
        Self {
            concept_ids:       std::array::from_fn(|_| AtomicU32::new(0)),
            salience:          std::array::from_fn(|_| AtomicU32::new(0)),
            active_mask:       AtomicU64::new(0),
            generation:        AtomicU64::new(0),
            last_lateral_tick: AtomicU64::new(0),
        }
    }

    /// Injeta um conceito no slot livre de menor índice. Se nenhum livre, substitui
    /// o slot de menor saliência (replacement-by-saliency, similar a LRU mas por relevância).
    /// Incrementa a `generation` para sinalizar mudança às vozes.
    /// Retorna o índice do slot usado, ou None se concept_id == 0 (reservado para vazio).
    pub fn inject_concept(&self, concept_id: u32, salience: f32) -> Option<usize> {
        if concept_id == 0 { return None; }
        let mask = self.active_mask.load(Ordering::Acquire);

        // 1) Procurar slot livre (bit zero).
        let free_idx = (0..CONTEXT_SLOTS).find(|&i| (mask & (1u64 << i)) == 0);
        let slot = match free_idx {
            Some(i) => i,
            None => {
                // Substituir slot de menor saliência.
                let mut min_sal = u32::MAX;
                let mut min_idx = 0;
                for i in 0..CONTEXT_SLOTS {
                    let s = self.salience[i].load(Ordering::Relaxed);
                    if s < min_sal { min_sal = s; min_idx = i; }
                }
                min_idx
            }
        };

        self.concept_ids[slot].store(concept_id, Ordering::Release);
        self.salience[slot].store(f32_to_fp(salience), Ordering::Release);
        self.active_mask.fetch_or(1u64 << slot, Ordering::AcqRel);
        self.generation.fetch_add(1, Ordering::AcqRel);
        Some(slot)
    }

    /// Marca um slot como livre. Não zera concept_id (custa um store extra a toa).
    pub fn clear_slot(&self, slot: usize) {
        if slot >= CONTEXT_SLOTS { return; }
        self.active_mask.fetch_and(!(1u64 << slot), Ordering::AcqRel);
        self.salience[slot].store(0, Ordering::Release);
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Lê todos os conceitos ativos como (concept_id, salience). Aloca Vec — usar
    /// fora do caminho quente. Para tick crítico, prefira `for_each_active`.
    pub fn read_active(&self) -> Vec<(u32, f32)> {
        let mask = self.active_mask.load(Ordering::Acquire);
        let mut out = Vec::with_capacity(mask.count_ones() as usize);
        for i in 0..CONTEXT_SLOTS {
            if (mask & (1u64 << i)) != 0 {
                let cid = self.concept_ids[i].load(Ordering::Acquire);
                let sal = fp_to_f32(self.salience[i].load(Ordering::Acquire));
                if cid != 0 { out.push((cid, sal)); }
            }
        }
        out
    }

    /// Itera sobre slots ativos sem heap allocation — adequado para o tick 200Hz.
    pub fn for_each_active<F: FnMut(usize, u32, f32)>(&self, mut f: F) {
        let mask = self.active_mask.load(Ordering::Acquire);
        for i in 0..CONTEXT_SLOTS {
            if (mask & (1u64 << i)) != 0 {
                let cid = self.concept_ids[i].load(Ordering::Acquire);
                if cid != 0 {
                    let sal = fp_to_f32(self.salience[i].load(Ordering::Acquire));
                    f(i, cid, sal);
                }
            }
        }
    }

    /// Conta quantos slots estão ativos (uso de popcount nativo).
    pub fn active_count(&self) -> u32 {
        self.active_mask.load(Ordering::Acquire).count_ones()
    }

    /// Marca de geração atual. Vozes guardam esse valor e comparam depois para
    /// detectar mudança sem reler todos os slots (Diretriz 3 — Recálculo em Voo).
    pub fn current_generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// True se houve qualquer mutação desde `last_gen`.
    pub fn changed_since(&self, last_gen: u64) -> bool {
        self.current_generation() != last_gen
    }

    /// Marca o tick atual como o momento de uma injeção lateral (Diretriz 2 —
    /// Escuta Ativa: novo fragmento de input chegou no meio do processamento).
    pub fn mark_lateral_injection(&self, tick: u64) {
        self.last_lateral_tick.store(tick, Ordering::Release);
    }

    /// Tick da última injeção lateral. 0 se nunca houve.
    pub fn last_lateral(&self) -> u64 {
        self.last_lateral_tick.load(Ordering::Acquire)
    }

    /// Limpa tudo. Custo O(CONTEXT_SLOTS) — usar entre conversações.
    pub fn clear_all(&self) {
        self.active_mask.store(0, Ordering::Release);
        for i in 0..CONTEXT_SLOTS {
            self.concept_ids[i].store(0, Ordering::Release);
            self.salience[i].store(0, Ordering::Release);
        }
        self.generation.fetch_add(1, Ordering::AcqRel);
    }
}

impl Default for ActiveContext {
    fn default() -> Self { Self::new() }
}

/// Hash FNV-1a 32-bit aplicado a um token. Mapeia para o range
/// [1, 0xEFFF_FFFF] — preserva 0 como "vazio" e reserva 0xF000_0000+ para
/// concept_ids virtuais da atenção (canal_foco do AttentionGate principal).
pub fn token_to_concept_id(token: &str) -> u32 {
    if token.is_empty() { return 0; }
    let mut hash: u32 = 0x811C9DC5;
    for byte in token.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x01000193);
    }
    // Mascara para [1, 0xEFFF_FFFF]; nunca retorna 0.
    let masked = hash & 0x0FFF_FFFF;
    if masked == 0 { 1 } else { masked }
}

/// Injeta uma sequência de tokens no contexto, com saliência decrescente
/// (token mais recente = saliência mais alta). Usada pela escuta ativa para
/// transformar uma frase em pontos de foco do ActiveContext.
pub fn inject_tokens(ctx: &ActiveContext, tokens: &[&str], base_salience: f32, tick: u64) -> usize {
    if tokens.is_empty() { return 0; }
    let n = tokens.len();
    let mut count = 0;
    for (i, token) in tokens.iter().enumerate() {
        let cid = token_to_concept_id(token);
        if cid == 0 { continue; }
        // Saliência: o último token recebe `base_salience`; os anteriores decaem
        // linearmente. Modela atenção recência-prioritária (recency bias).
        let pos = (i + 1) as f32 / n as f32; // 0..1
        let sal = (base_salience * pos).clamp(0.0, 1.0);
        if ctx.inject_concept(cid, sal).is_some() {
            count += 1;
        }
    }
    ctx.mark_lateral_injection(tick);
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injecao_basica() {
        let ctx = ActiveContext::new();
        let s = ctx.inject_concept(42, 0.8).unwrap();
        assert_eq!(ctx.active_count(), 1);
        let active = ctx.read_active();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].0, 42);
        assert!((active[0].1 - 0.8).abs() < 1e-3);
        ctx.clear_slot(s);
        assert_eq!(ctx.active_count(), 0);
    }

    #[test]
    fn substitui_menor_saliencia_quando_cheio() {
        let ctx = ActiveContext::new();
        for i in 0..CONTEXT_SLOTS {
            ctx.inject_concept((i + 1) as u32, 0.5 + (i as f32) * 0.001);
        }
        assert_eq!(ctx.active_count() as usize, CONTEXT_SLOTS);
        // Slot 0 (cid=1) tem saliência 0.5, menor de todos.
        ctx.inject_concept(9999, 0.95);
        // Total ainda CONTEXT_SLOTS (substituiu, não cresceu).
        assert_eq!(ctx.active_count() as usize, CONTEXT_SLOTS);
        let active = ctx.read_active();
        assert!(active.iter().any(|&(cid, _)| cid == 9999));
        assert!(!active.iter().any(|&(cid, _)| cid == 1));
    }

    #[test]
    fn generation_incrementa() {
        let ctx = ActiveContext::new();
        let g0 = ctx.current_generation();
        ctx.inject_concept(7, 0.5);
        assert!(ctx.changed_since(g0));
        let g1 = ctx.current_generation();
        assert!(!ctx.changed_since(g1));
    }

    #[test]
    fn concept_id_zero_rejeitado() {
        let ctx = ActiveContext::new();
        assert!(ctx.inject_concept(0, 0.9).is_none());
        assert_eq!(ctx.active_count(), 0);
    }
}
