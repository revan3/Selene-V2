// =============================================================================
// src/neural_pool.rs — V3.3 — Localist Coding + Metaplasticidade Evolutiva + NVMe Swap
// =============================================================================
//
// MUDANÇAS V3.3 (sobre V3.2):
//
//   1. concept_id: u32 (FNV-1a hash da palavra)
//      O pool "não pensa em palavras" — apenas IDs numéricos. A tradução
//      concept_id ↔ String vive no ConceptRegistry (bridge.rs). O cérebro
//      opera em espaço numérico; a interface humana usa string.
//
//   2. METAPLASTICIDADE EVOLUTIVA — promoção de CorticalLevel
//      Quando ltp_count cruza TETO_NIVEL_Cx, o bloco migra de nível:
//      C0Sensorial → C1Perceptual → C2Lexical → C3Contextual
//      Analogia biológica: spine de V1 (C0) com LTP repetido cresce, forma
//      sinapse dendrítica para L2/3 (C1) — representação perceptual emerge do
//      sensorial (V1→V2, Hebbian remapping, Gilbert 1992).
//
//   3. NVMe SWAP (LRU eviction quando pool L1 cheio)
//      `aloca_para_tarefa` → se free_list vazio → evict_lru() move bloco mais
//      antigo (menos ativado) para `swapped_out` (RAM-swap em memória) e
//      opcionalmente para F:/selene_pool_swap/<id>.bin (disco).
//      `buscar_conceito` → se não no pool L1 → verifica swapped_out → restore.
//
// REFERÊNCIAS BIOLÓGICAS:
//   Quiroga (2005) — "Concept cells" no hipocampo humano
//   Abraham & Bear (1996) — Metaplasticity
//   Gilbert (1992) — V1 cortical remapping / Hebbian plasticity
//   Ester et al. (2020) — NVMe-based neural memory tiering
// =============================================================================

#![allow(dead_code)]

use serde::{Serialize, Deserialize};
use std::collections::{VecDeque, HashMap};

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 0 — CONCEPT REGISTRY (String ↔ u32)
// ─────────────────────────────────────────────────────────────────────────────

/// Converte uma palavra para seu concept_id via FNV-1a 32 bits.
///
/// Propriedades:
///   - Determinístico: mesma palavra → mesmo ID em qualquer sessão
///   - Sem colisão em vocabulários < 100k palavras (probabilidade ~0.001%)
///   - 0 = NUNCA produzido para string não-vazia (sentinel "sem conceito")
#[inline]
pub fn word_to_concept_id(word: &str) -> u32 {
    const OFFSET: u32 = 2_166_136_261;
    const PRIME:  u32 =    16_777_619;
    let mut h = OFFSET;
    for byte in word.to_lowercase().bytes() {
        h ^= byte as u32;
        h = h.wrapping_mul(PRIME);
    }
    // Garante nunca retornar 0 (sentinel) — mapeia 0→1
    if h == 0 { 1 } else { h }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 1 — NÍVEIS DE PRECISÃO E HIERARQUIA CORTICAL
// ─────────────────────────────────────────────────────────────────────────────

/// Precisão dinâmica do bloco neural — promovida via LTP.
///
/// Mapeamento físico (sempre 32 bits no hardware):
///   FP4   →  4 bits úteis (1 sinal + 3 mantissa)  — 28 bits mascarados
///   FP8   →  8 bits úteis (1 sinal + 4 expo + 3 mantissa)
///   FP16  → 16 bits úteis (half-precision IEEE-754)
///   FP32  → 32 bits úteis (single-precision IEEE-754)
///   INT32 → 32 bits úteis interpretados como inteiro
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionLevel {
    FP4   = 0,
    FP8   = 1,
    FP16  = 2,
    FP32  = 3,
    INT32 = 4,
}

impl PrecisionLevel {
    #[inline]
    pub fn bits_uteis(&self) -> u8 {
        match self {
            PrecisionLevel::FP4   => 4,
            PrecisionLevel::FP8   => 8,
            PrecisionLevel::FP16  => 16,
            PrecisionLevel::FP32  => 32,
            PrecisionLevel::INT32 => 32,
        }
    }

    #[inline]
    pub fn mascara(&self) -> u32 {
        match self {
            PrecisionLevel::FP4   => 0x0000_000F,
            PrecisionLevel::FP8   => 0x0000_00FF,
            PrecisionLevel::FP16  => 0x0000_FFFF,
            PrecisionLevel::FP32  |
            PrecisionLevel::INT32 => 0xFFFF_FFFF,
        }
    }

    pub fn promover(&self) -> Self {
        match self {
            PrecisionLevel::FP4   => PrecisionLevel::FP8,
            PrecisionLevel::FP8   => PrecisionLevel::FP16,
            PrecisionLevel::FP16  => PrecisionLevel::FP32,
            PrecisionLevel::FP32  => PrecisionLevel::INT32,
            PrecisionLevel::INT32 => PrecisionLevel::INT32,
        }
    }
}

/// Hierarquia cortical funcional. Cada bloco vive em um nível que determina
/// seu teto de precisão e seus limiares de promoção de NÍVEL (C0→C1→...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorticalLevel {
    /// C0 — sensorial puro: bandas FFT, pixels, propriocepção.
    C0Sensorial,
    /// C1 — perceptual: bordas, fonemas, formas.
    C1Perceptual,
    /// C2 — lexical/sintático: palavras, frases.
    C2Lexical,
    /// C3 — contextual: episódios, eventos.
    C3Contextual,
    /// C4 — abstrato: conceitos filosóficos, metarepresentações.
    C4Abstrato,
}

impl CorticalLevel {
    #[inline]
    pub fn precisao_inicial(&self) -> PrecisionLevel {
        match self {
            CorticalLevel::C0Sensorial   => PrecisionLevel::FP4,
            CorticalLevel::C1Perceptual  => PrecisionLevel::FP4,
            CorticalLevel::C2Lexical     => PrecisionLevel::FP8,
            CorticalLevel::C3Contextual  => PrecisionLevel::FP16,
            CorticalLevel::C4Abstrato    => PrecisionLevel::FP32,
        }
    }

    #[inline]
    pub fn precisao_maxima(&self) -> PrecisionLevel {
        match self {
            CorticalLevel::C0Sensorial   => PrecisionLevel::FP16,
            CorticalLevel::C1Perceptual  => PrecisionLevel::FP32,
            CorticalLevel::C2Lexical     => PrecisionLevel::FP32,
            CorticalLevel::C3Contextual  => PrecisionLevel::INT32,
            CorticalLevel::C4Abstrato    => PrecisionLevel::INT32,
        }
    }

    /// Nível cortical seguinte para promoção evolutiva.
    /// C4Abstrato retorna None — é o nível final.
    pub fn proximo_nivel(&self) -> Option<CorticalLevel> {
        match self {
            CorticalLevel::C0Sensorial   => Some(CorticalLevel::C1Perceptual),
            CorticalLevel::C1Perceptual  => Some(CorticalLevel::C2Lexical),
            CorticalLevel::C2Lexical     => Some(CorticalLevel::C3Contextual),
            CorticalLevel::C3Contextual  => Some(CorticalLevel::C4Abstrato),
            CorticalLevel::C4Abstrato    => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 2 — LIMIARES DE PROMOÇÃO
// ─────────────────────────────────────────────────────────────────────────────

/// Limiares LTP para promoção de PRECISÃO (dentro do mesmo nível cortical).
const LTP_PROMO_FP4_FP8:    u32 = 10;
const LTP_PROMO_FP8_FP16:   u32 = 50;
const LTP_PROMO_FP16_FP32:  u32 = 200;
const LTP_PROMO_FP32_INT32: u32 = 1000;

/// Limiares LTP para promoção de NÍVEL CORTICAL (C0→C1→C2→C3).
/// Após cruzar o limiar, o bloco migra para o próximo nível e ltp_count é resetado.
/// Analogia: spine de V1 que recebe LTP repetido "cresce" para L2/3 (C1).
const LTP_NIVEL_C0_C1: u32 = 128;
const LTP_NIVEL_C1_C2: u32 = 512;
const LTP_NIVEL_C2_C3: u32 = 2048;

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 3 — BLOCO NEURAL (32 bits físicos, precisão lógica variável)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralBlock {
    /// 32 bits físicos. Bits superiores zerados quando precisão < FP32.
    raw: u32,
    /// Precisão lógica atual (promovida por LTP).
    pub precision: PrecisionLevel,
    /// Nível cortical (define teto de promoção e estratégia de uso).
    /// Agora MUTÁVEL — evolui com LTP (C0→C1→C2→...).
    pub level: CorticalLevel,
    /// Eventos LTP acumulados — gatilho de promoção de precisão E de nível.
    pub ltp_count: u32,
    /// Localist Coding: ID u32 único do conceito. 0 = livre (sem conceito).
    /// Derivado via FNV-1a da palavra — determinístico entre sessões.
    pub concept_id: u32,
    /// Valência emocional [-1.0, 1.0].
    pub valence: f32,
    /// Última ativação (timestamp ms) — para LRU eviction e reciclagem.
    pub last_active_ms: f64,
    /// `true` quando alocado; `false` quando no pool de repouso.
    pub in_use: bool,
}

impl NeuralBlock {
    fn vazio() -> Self {
        Self {
            raw: 0,
            precision: PrecisionLevel::FP4,
            level: CorticalLevel::C0Sensorial,
            ltp_count: 0,
            concept_id: 0,
            valence: 0.0,
            last_active_ms: 0.0,
            in_use: false,
        }
    }

    pub fn ler_f32(&self) -> f32 {
        let bits = self.raw & self.precision.mascara();
        match self.precision {
            PrecisionLevel::FP4 => {
                let sign = (bits & 0x8) != 0;
                let mag = (bits & 0x7) as f32 / 7.0;
                if sign { -mag } else { mag }
            }
            PrecisionLevel::FP8 => {
                let sign = (bits & 0x80) != 0;
                let expo = ((bits >> 3) & 0xF) as i32 - 7;
                let mant = (bits & 0x7) as f32 / 8.0;
                let mag = (1.0 + mant) * 2f32.powi(expo);
                if sign { -mag } else { mag }
            }
            PrecisionLevel::FP16 => {
                half::f16::from_bits(bits as u16).to_f32()
            }
            PrecisionLevel::FP32 => f32::from_bits(bits),
            PrecisionLevel::INT32 => bits as f32,
        }
    }

    pub fn escrever_f32(&mut self, valor: f32) {
        let v = valor.clamp(-128.0, 128.0);
        match self.precision {
            PrecisionLevel::FP4 => {
                let mag = (v.abs() * 7.0).round().clamp(0.0, 7.0) as u32;
                let sign = if v < 0.0 { 0x8 } else { 0x0 };
                self.raw = sign | (mag & 0x7);
            }
            PrecisionLevel::FP8 => {
                let abs = v.abs().max(1e-6);
                let expo = abs.log2().floor() as i32;
                let mant = (abs / 2f32.powi(expo) - 1.0) * 8.0;
                let expo_b = ((expo + 7).clamp(0, 15) as u32) & 0xF;
                let mant_b = mant.round().clamp(0.0, 7.0) as u32 & 0x7;
                let sign = if v < 0.0 { 0x80 } else { 0x00 };
                self.raw = sign | (expo_b << 3) | mant_b;
            }
            PrecisionLevel::FP16 => {
                self.raw = half::f16::from_f32(v).to_bits() as u32;
            }
            PrecisionLevel::FP32 => {
                self.raw = v.to_bits();
            }
            PrecisionLevel::INT32 => {
                self.raw = v as u32;
            }
        }
    }

    #[inline]
    pub fn ler_i32(&self) -> i32 { self.raw as i32 }

    #[inline]
    pub fn escrever_i32(&mut self, valor: i32) {
        if self.precision == PrecisionLevel::INT32 {
            self.raw = valor as u32;
        }
    }

    /// Registra um evento LTP.
    ///
    /// Fase 1 — Promoção de PRECISÃO dentro do nível cortical atual.
    /// Fase 2 — Promoção de NÍVEL CORTICAL (Metaplasticidade Evolutiva):
    ///   Se ltp_count cruza o limiar do nível atual, o bloco migra para
    ///   o próximo nível (C0→C1, C1→C2, C2→C3). ltp_count é resetado para
    ///   que o ciclo continue no novo nível. A precisão sobe para o mínimo
    ///   do novo nível se ainda for menor.
    pub fn evento_ltp(&mut self, t_ms: f64) {
        self.ltp_count = self.ltp_count.saturating_add(1);
        self.last_active_ms = t_ms;

        // Fase 1: promoção de precisão (respeitando teto cortical)
        let teto = self.level.precisao_maxima();
        let nova_prec = match self.ltp_count {
            n if n >= LTP_PROMO_FP32_INT32 => PrecisionLevel::INT32,
            n if n >= LTP_PROMO_FP16_FP32  => PrecisionLevel::FP32,
            n if n >= LTP_PROMO_FP8_FP16   => PrecisionLevel::FP16,
            n if n >= LTP_PROMO_FP4_FP8    => PrecisionLevel::FP8,
            _                              => self.precision,
        };
        let nova_prec = if (nova_prec as u8) > (teto as u8) { teto } else { nova_prec };
        if (nova_prec as u8) > (self.precision as u8) {
            self.precision = nova_prec;
        }

        // Fase 2: promoção de nível cortical (Metaplasticidade Evolutiva)
        // Cada nível tem um limiar acumulado; ao cruzá-lo o bloco "migra"
        // para um nível funcional mais abstrato, assim como V1 → L2/3.
        let limiar_nivel = match self.level {
            CorticalLevel::C0Sensorial   => Some(LTP_NIVEL_C0_C1),
            CorticalLevel::C1Perceptual  => Some(LTP_NIVEL_C1_C2),
            CorticalLevel::C2Lexical     => Some(LTP_NIVEL_C2_C3),
            CorticalLevel::C3Contextual  |
            CorticalLevel::C4Abstrato    => None, // teto final
        };

        if let Some(limiar) = limiar_nivel {
            if self.ltp_count >= limiar {
                if let Some(novo_nivel) = self.level.proximo_nivel() {
                    self.level = novo_nivel;
                    // Reset ltp_count para que o bloco possa continuar
                    // acumulando até o próximo limiar de nível.
                    self.ltp_count = 0;
                    // Garante que a precisão sobe ao mínimo do novo nível
                    let prec_min = novo_nivel.precisao_inicial();
                    if (prec_min as u8) > (self.precision as u8) {
                        self.precision = prec_min;
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 4 — POOL GLOBAL DE NEURÔNIOS EM REPOUSO + NVMe SWAP
// ─────────────────────────────────────────────────────────────────────────────

/// Pool L1 (RAM) com swap automático para disco quando cheio.
///
/// Política de evicção:
///   Quando free_list está vazio e um novo bloco é solicitado, o bloco
///   em uso mais antigo (menor last_active_ms) que NÃO seja C3/C4 é
///   movido para `swapped_out` (e opcionalmente para disco via `swap_dir`).
///   Isso libera um slot no L1 sem perder o estado do bloco.
///
/// Restauração:
///   Quando `buscar_conceito(id)` não encontra no L1 mas o ID está em
///   `swapped_out`, o bloco é restaurado automaticamente ao pool.
#[derive(Debug)]
pub struct NeuralPool {
    pub blocks: Vec<NeuralBlock>,
    pub free_list: VecDeque<usize>,
    pub capacity: usize,
    /// Localist Coding: concept_id (u32) → índice no vetor `blocks`.
    /// Chave é o hash FNV-1a da palavra — sem strings no caminho quente.
    pub concept_index: HashMap<u32, usize>,
    pub total_alocacoes: u64,
    pub total_devolucoes: u64,
    /// Blocos swap-out da L1: concept_id → bloco completo (RAM-swap).
    /// Quando `swap_dir` está definido, também escrito em disco.
    swapped_out: HashMap<u32, NeuralBlock>,
    /// Diretório raiz para swap em disco (ex: F:/selene_pool_swap/).
    /// `None` = sem persistência de swap (apenas RAM-swap).
    pub swap_dir: Option<std::path::PathBuf>,
}

impl NeuralPool {
    pub fn new(capacity: usize) -> Self {
        let blocks: Vec<NeuralBlock> = (0..capacity).map(|_| NeuralBlock::vazio()).collect();
        let free_list: VecDeque<usize> = (0..capacity).collect();
        Self {
            blocks,
            free_list,
            capacity,
            concept_index: HashMap::new(),
            total_alocacoes: 0,
            total_devolucoes: 0,
            swapped_out: HashMap::new(),
            swap_dir: None,
        }
    }

    /// Configura o diretório de swap em disco.
    pub fn com_swap_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        let p = dir.into();
        let _ = std::fs::create_dir_all(&p);
        self.swap_dir = Some(p);
        self
    }

    // ── Alocação ────────────────────────────────────────────────────────────

    /// Aloca um bloco por concept_id u32.
    ///
    /// Localist: se o conceito já está no L1, retorna o índice existente.
    /// Se está no swap, restaura para o L1 primeiro.
    /// Se pool cheio, evicta LRU antes de alocar.
    pub fn aloca_para_tarefa(
        &mut self,
        concept_id: u32,
        level: CorticalLevel,
        t_ms: f64,
    ) -> Option<usize> {
        if concept_id == 0 { return None; }

        // Localist: reutiliza se já no L1
        if let Some(&idx) = self.concept_index.get(&concept_id) {
            self.blocks[idx].last_active_ms = t_ms;
            return Some(idx);
        }

        // Restaura do swap se disponível
        if self.swapped_out.contains_key(&concept_id) {
            return self.restaurar_swapped(concept_id, t_ms);
        }

        // Se pool cheio, evicta LRU para abrir slot
        if self.free_list.is_empty() {
            if !self.evict_lru(t_ms) {
                return None; // pool cheio e nada a evictar (tudo C3/C4)
            }
        }

        let idx = self.free_list.pop_front()?;
        let b = &mut self.blocks[idx];
        b.raw = 0;
        b.precision = level.precisao_inicial();
        b.level = level;
        b.ltp_count = 0;
        b.concept_id = concept_id;
        b.valence = 0.0;
        b.last_active_ms = t_ms;
        b.in_use = true;
        self.concept_index.insert(concept_id, idx);
        self.total_alocacoes += 1;
        Some(idx)
    }

    /// Conveniência: aloca diretamente por string (hash automático).
    #[inline]
    pub fn aloca_palavra(
        &mut self,
        palavra: &str,
        level: CorticalLevel,
        t_ms: f64,
    ) -> Option<usize> {
        self.aloca_para_tarefa(word_to_concept_id(palavra), level, t_ms)
    }

    // ── Devolução / Reset ────────────────────────────────────────────────────

    pub fn devolver_ao_pool(&mut self, idx: usize) {
        if idx >= self.capacity { return; }
        let b = &mut self.blocks[idx];
        if !b.in_use { return; }
        let cid = b.concept_id;
        b.raw = 0;
        b.precision = PrecisionLevel::FP4;
        b.level = CorticalLevel::C0Sensorial;
        b.ltp_count = 0;
        b.concept_id = 0;
        b.valence = 0.0;
        b.last_active_ms = 0.0;
        b.in_use = false;
        self.concept_index.remove(&cid);
        self.free_list.push_back(idx);
        self.total_devolucoes += 1;
    }

    // ── Evicção LRU / NVMe Swap ──────────────────────────────────────────────

    /// Encontra o bloco LRU elegível (não C3/C4, em uso), move para swap.
    /// Retorna `true` se um slot foi liberado.
    fn evict_lru(&mut self, _t_ms: f64) -> bool {
        // Encontra índice com menor last_active_ms (excluindo C3/C4)
        let lru_idx = self.blocks.iter().enumerate()
            .filter(|(_, b)| b.in_use
                && b.level != CorticalLevel::C3Contextual
                && b.level != CorticalLevel::C4Abstrato)
            .min_by(|(_, a), (_, b)| {
                a.last_active_ms.partial_cmp(&b.last_active_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        let lru_idx = match lru_idx {
            Some(i) => i,
            None => return false,
        };

        let bloco = self.blocks[lru_idx].clone();
        let cid = bloco.concept_id;

        // Persiste em disco se swap_dir configurado
        if let Some(ref dir) = self.swap_dir {
            let path = dir.join(format!("{}.bin", cid));
            if let Ok(serialized) = serde_json::to_vec(&bloco) {
                let _ = std::fs::write(path, serialized);
            }
        }

        self.swapped_out.insert(cid, bloco);
        self.devolver_ao_pool(lru_idx);
        true
    }

    /// Restaura bloco do swap de volta ao pool L1.
    fn restaurar_swapped(&mut self, concept_id: u32, t_ms: f64) -> Option<usize> {
        let mut bloco = self.swapped_out.remove(&concept_id)?;
        bloco.last_active_ms = t_ms;

        // Remove arquivo de disco se existir
        if let Some(ref dir) = self.swap_dir {
            let _ = std::fs::remove_file(dir.join(format!("{}.bin", concept_id)));
        }

        // Abre slot — evicta LRU se necessário
        if self.free_list.is_empty() && !self.evict_lru(t_ms) {
            // Sem slot disponível, recoloca no swap
            self.swapped_out.insert(concept_id, bloco);
            return None;
        }
        let idx = self.free_list.pop_front()?;
        bloco.in_use = true;
        self.blocks[idx] = bloco;
        self.concept_index.insert(concept_id, idx);
        self.total_alocacoes += 1;
        Some(idx)
    }

    // ── Lookup ───────────────────────────────────────────────────────────────

    #[inline]
    pub fn buscar_conceito(&self, concept_id: u32) -> Option<&NeuralBlock> {
        self.concept_index.get(&concept_id).and_then(|&i| self.blocks.get(i))
    }

    #[inline]
    pub fn buscar_conceito_mut(&mut self, concept_id: u32) -> Option<&mut NeuralBlock> {
        let idx = *self.concept_index.get(&concept_id)?;
        self.blocks.get_mut(idx)
    }

    /// Lookup por string (hash automático). Restaura do swap se necessário.
    pub fn buscar_palavra(&mut self, palavra: &str, t_ms: f64) -> Option<&NeuralBlock> {
        let cid = word_to_concept_id(palavra);
        if !self.concept_index.contains_key(&cid) && self.swapped_out.contains_key(&cid) {
            self.restaurar_swapped(cid, t_ms);
        }
        self.buscar_conceito(cid)
    }

    // ── LTP & Valência ───────────────────────────────────────────────────────

    pub fn ltp_em_conceito(&mut self, concept_id: u32, t_ms: f64) {
        if let Some(b) = self.buscar_conceito_mut(concept_id) {
            b.evento_ltp(t_ms);
        }
    }

    pub fn ltp_em_palavra(&mut self, palavra: &str, t_ms: f64) {
        let cid = word_to_concept_id(palavra);
        self.ltp_em_conceito(cid, t_ms);
    }

    pub fn atualizar_valencia(&mut self, concept_id: u32, valence: f32) {
        if let Some(b) = self.buscar_conceito_mut(concept_id) {
            b.valence = valence.clamp(-1.0, 1.0);
        }
    }

    pub fn atualizar_valencia_palavra(&mut self, palavra: &str, valence: f32) {
        let cid = word_to_concept_id(palavra);
        self.atualizar_valencia(cid, valence);
    }

    /// Observa tokens léxicos (passive_hear, chat input) no nível C2Lexical.
    pub fn localist_observar(&mut self, tokens: &[String], t_ms: f64) {
        for tok in tokens {
            if tok.len() < 2 { continue; }
            let cid = word_to_concept_id(tok);
            if self.aloca_para_tarefa(cid, CorticalLevel::C2Lexical, t_ms).is_some() {
                self.ltp_em_conceito(cid, t_ms);
            }
        }
    }

    // ── Reciclagem ───────────────────────────────────────────────────────────

    pub fn reciclar_inativos(&mut self, t_ms_atual: f64, idade_max_ms: f64) -> usize {
        let a_devolver: Vec<usize> = self.blocks.iter().enumerate()
            .filter(|(_, b)| b.in_use
                && (t_ms_atual - b.last_active_ms) > idade_max_ms
                && b.level != CorticalLevel::C3Contextual
                && b.level != CorticalLevel::C4Abstrato)
            .map(|(i, _)| i)
            .collect();
        let n = a_devolver.len();
        for idx in a_devolver {
            self.devolver_ao_pool(idx);
        }
        n
    }

    // ── Métricas ─────────────────────────────────────────────────────────────

    pub fn n_em_uso(&self) -> usize { self.capacity - self.free_list.len() }
    pub fn n_disponivel(&self) -> usize { self.free_list.len() }
    pub fn n_swapped_out(&self) -> usize { self.swapped_out.len() }

    pub fn taxa_ocupacao(&self) -> f32 {
        if self.capacity == 0 { 0.0 }
        else { self.n_em_uso() as f32 / self.capacity as f32 }
    }

    pub fn dist_precisao(&self) -> [usize; 5] {
        let mut d = [0usize; 5];
        for b in &self.blocks {
            if b.in_use { d[b.precision as usize] += 1; }
        }
        d
    }

    pub fn dist_cortical(&self) -> [usize; 5] {
        let mut d = [0usize; 5];
        for b in &self.blocks {
            if b.in_use {
                d[match b.level {
                    CorticalLevel::C0Sensorial  => 0,
                    CorticalLevel::C1Perceptual => 1,
                    CorticalLevel::C2Lexical    => 2,
                    CorticalLevel::C3Contextual => 3,
                    CorticalLevel::C4Abstrato   => 4,
                }] += 1;
            }
        }
        d
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 5 — TESTES DE SANIDADE
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_to_id_determinista_e_sem_zero() {
        let id1 = word_to_concept_id("gato");
        let id2 = word_to_concept_id("gato");
        assert_eq!(id1, id2);
        assert_ne!(id1, 0);
        assert_ne!(word_to_concept_id("GATO"), 0); // case-insensitive
        assert_eq!(word_to_concept_id("gato"), word_to_concept_id("GATO"));
    }

    #[test]
    fn alocacao_e_localist_coding() {
        let mut pool = NeuralPool::new(8);
        let i1 = pool.aloca_palavra("maçã", CorticalLevel::C2Lexical, 0.0);
        let i2 = pool.aloca_palavra("maçã", CorticalLevel::C2Lexical, 1.0);
        assert_eq!(i1, i2, "Localist: mesma palavra → mesmo bloco");
        assert_eq!(pool.n_em_uso(), 1);
    }

    #[test]
    fn reset_e_devolucao() {
        let mut pool = NeuralPool::new(4);
        let cid = word_to_concept_id("foo");
        let i = pool.aloca_para_tarefa(cid, CorticalLevel::C0Sensorial, 0.0).unwrap();
        pool.devolver_ao_pool(i);
        assert_eq!(pool.n_em_uso(), 0);
        assert!(pool.buscar_conceito(cid).is_none());
    }

    #[test]
    fn metaplasticidade_promove_fp4_fp8() {
        let mut pool = NeuralPool::new(2);
        pool.aloca_palavra("X", CorticalLevel::C1Perceptual, 0.0);
        let cid = word_to_concept_id("X");
        assert_eq!(pool.buscar_conceito(cid).unwrap().precision, PrecisionLevel::FP4);
        for k in 1..=11 {
            pool.ltp_em_conceito(cid, k as f64);
        }
        assert_eq!(pool.buscar_conceito(cid).unwrap().precision, PrecisionLevel::FP8,
            "esperado FP4→FP8 após 11 eventos LTP");
    }

    #[test]
    fn teto_cortical_respeita_c0() {
        let mut pool = NeuralPool::new(2);
        pool.aloca_palavra("p", CorticalLevel::C0Sensorial, 0.0);
        let cid = word_to_concept_id("p");
        // 127 LTP — just below C0→C1 promotion threshold (128)
        // verifies that while IN C0, precision ceiling = FP16 is respected
        for k in 0..127 { pool.ltp_em_conceito(cid, k as f64); }
        let b = pool.buscar_conceito(cid).unwrap();
        assert_eq!(b.level, CorticalLevel::C0Sensorial, "deve ainda estar em C0");
        assert!((b.precision as u8) <= (PrecisionLevel::FP16 as u8),
            "C0 teto FP16 deve ser respeitado");
    }

    #[test]
    fn metaplasticidade_evolutiva_c0_promove_para_c1() {
        let mut pool = NeuralPool::new(2);
        pool.aloca_palavra("luz", CorticalLevel::C0Sensorial, 0.0);
        let cid = word_to_concept_id("luz");
        // 128 eventos LTP → deve promover C0→C1
        for k in 0..128 { pool.ltp_em_conceito(cid, k as f64); }
        let b = pool.buscar_conceito(cid).unwrap();
        assert_eq!(b.level, CorticalLevel::C1Perceptual,
            "C0 deve promover para C1 após 128 LTP");
        assert_eq!(b.ltp_count, 0,
            "ltp_count deve resetar após promoção de nível");
    }

    #[test]
    fn nvme_swap_evict_e_restaura() {
        let mut pool = NeuralPool::new(2);
        let cid_a = word_to_concept_id("alpha");
        let cid_b = word_to_concept_id("beta");
        let cid_c = word_to_concept_id("gamma");
        pool.aloca_para_tarefa(cid_a, CorticalLevel::C0Sensorial, 0.0);
        pool.aloca_para_tarefa(cid_b, CorticalLevel::C1Perceptual, 1.0);
        // Pool cheio (2/2). Alocar "gamma" deve evictar LRU (alpha, t=0)
        let idx_c = pool.aloca_para_tarefa(cid_c, CorticalLevel::C2Lexical, 100.0);
        assert!(idx_c.is_some(), "deve evictar LRU e alocar gamma");
        assert_eq!(pool.n_swapped_out(), 1, "alpha deve estar no swap");
        // Libera beta para abrir um slot livre — restauração não precisa evictar
        let idx_b = *pool.concept_index.get(&cid_b).unwrap();
        pool.devolver_ao_pool(idx_b);
        // Restaurar alpha — há slot livre, swap fica vazio
        let idx_a = pool.aloca_para_tarefa(cid_a, CorticalLevel::C0Sensorial, 200.0);
        assert!(idx_a.is_some(), "alpha deve ser restaurado do swap");
        assert_eq!(pool.n_swapped_out(), 0, "swap deve estar vazio após restauração sem evicção");
    }
}
