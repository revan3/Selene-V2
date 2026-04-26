// =============================================================================
// src/neural_pool.rs — V3.2 — Pool de Repouso + Metaplasticidade FP4→INT32
// =============================================================================
//
// IMPLEMENTA AS DIRETRIZES V3:
//
//   1. LOCALIST CODING (Grandmother Cell)
//      Cada bloco do pool tem `concept_id: Option<String>` único — o neurônio
//      É a representação física do conceito (1 neurônio = 1 ideia).
//
//   2. METAPLASTICIDADE POR RESOLUÇÃO (FP4 → FP8 → FP16 → FP32 → INT32)
//      Precisão promovida dinamicamente conforme LTP acumula. Implementado via
//      MASCARAMENTO de u32 — sem realocação de memória. Cada bloco é fisicamente
//      32 bits; o software lê apenas os bits relevantes para a precisão atual.
//
//   3. GROUNDING C0–C4 (Hierarquia Cortical)
//      `CorticalLevel` enum: sensorial puro → perceptual → lexical → contextual
//      → abstrato. Indexação por peso emocional via `valence_index`.
//
//   4. ALOCAÇÃO SOB DEMANDA (Pre-Frontal Cortex)
//      `aloca_para_tarefa()` extrai do pool. Sem alocação prévia desnecessária.
//
//   5. POOL DE REPOUSO (Templates INT32)
//      Vetor pré-alocado de blocos u32. Cada bloco PODE armazenar até INT32
//      completo, mas inicialmente opera como 8× FP4 mascarado.
//
//   6. RESET NEURAL + DEPRESSÃO SINÁPTICA
//      `devolver_ao_pool()` zera o bloco completamente e retorna ao free-list.
//
// REFERÊNCIAS BIOLÓGICAS:
//   Quiroga (2005) — "Concept cells" no hipocampo humano (Halle Berry neuron)
//   Abraham & Bear (1996) — Metaplasticity: the plasticity of synaptic plasticity
//   Hawkins (2021) — Thousand Brains: cada coluna cortical = grandmother cell
// =============================================================================

#![allow(dead_code)]

use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 1 — NÍVEIS DE PRECISÃO E HIERARQUIA CORTICAL
// ─────────────────────────────────────────────────────────────────────────────

/// Precisão dinâmica do bloco neural — promovida via LTP.
///
/// Mapeamento físico (sempre 32 bits no hardware):
///   FP4   →  4 bits úteis (1 sinal + 3 mantissa)  — 28 bits zerados/mascarados
///   FP8   →  8 bits úteis (1 sinal + 4 expo + 3 mantissa)
///   FP16  → 16 bits úteis (half-precision IEEE-754)
///   FP32  → 32 bits úteis (single-precision IEEE-754)
///   INT32 → 32 bits úteis interpretados como inteiro (templates complexos:
///           ponteiros para C3/C2, IDs de chunk, hashes de contexto)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionLevel {
    FP4   = 0,
    FP8   = 1,
    FP16  = 2,
    FP32  = 3,
    INT32 = 4,
}

impl PrecisionLevel {
    /// Bits úteis por nível.
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

    /// Máscara para extrair os bits úteis de um u32 cru.
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

    /// Promove para o próximo nível, se possível.
    pub fn promover(&self) -> Self {
        match self {
            PrecisionLevel::FP4   => PrecisionLevel::FP8,
            PrecisionLevel::FP8   => PrecisionLevel::FP16,
            PrecisionLevel::FP16  => PrecisionLevel::FP32,
            PrecisionLevel::FP32  => PrecisionLevel::INT32,
            PrecisionLevel::INT32 => PrecisionLevel::INT32, // teto
        }
    }
}

/// Hierarquia cortical funcional (Mountcastle 1957; Hawkins 2021).
/// Cada bloco do pool é alocado em um nível específico — define o tipo
/// de conteúdo que pode armazenar e a estratégia de promoção de precisão.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorticalLevel {
    /// C0 — sensorial puro: bandas FFT, pixels, propriocepção.
    /// Nasce como FP4. LTP promove até FP16.
    C0Sensorial,
    /// C1 — perceptual: bordas, fonemas, formas. Output do C0 agrupado.
    /// Nasce FP4. Promove até FP32.
    C1Perceptual,
    /// C2 — lexical/sintático: palavras, frases, gramática.
    /// Nasce FP8. Promove até FP32.
    C2Lexical,
    /// C3 — contextual: episódios, eventos, narrativas com referências cruzadas.
    /// Nasce FP16. Promove até INT32 (necessita ponteiros).
    C3Contextual,
    /// C4 — abstrato: conceitos filosóficos, metarepresentações.
    /// Nasce FP32. Tipicamente INT32 (alta complexidade semântica).
    C4Abstrato,
}

impl CorticalLevel {
    /// Precisão inicial recomendada para este nível cortical.
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

    /// Teto de promoção para este nível.
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
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 2 — BLOCO NEURAL (32 bits físicos, precisão lógica variável)
// ─────────────────────────────────────────────────────────────────────────────

/// Limites de promoção (eventos LTP acumulados) por nível atual.
const LTP_PROMO_FP4_FP8:   u32 = 10;
const LTP_PROMO_FP8_FP16:  u32 = 50;
const LTP_PROMO_FP16_FP32: u32 = 200;
const LTP_PROMO_FP32_INT32: u32 = 1000;

/// Bloco neural fisicamente sempre 32 bits — interpretação varia por `precision`.
/// Tamanho fixo em memória → sem realocação ao promover precisão.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralBlock {
    /// 32 bits físicos. Bits superiores zerados quando precisão < FP32.
    raw: u32,
    /// Precisão lógica atual (promovida por LTP).
    pub precision: PrecisionLevel,
    /// Nível cortical (define teto de promoção e estratégia de uso).
    pub level: CorticalLevel,
    /// Eventos LTP acumulados — gatilho de promoção de precisão.
    pub ltp_count: u32,
    /// Localist Coding (Grandmother Cell): ID único do conceito representado.
    /// `None` quando o bloco está no pool de repouso.
    pub concept_id: Option<String>,
    /// Valência emocional [-1.0, 1.0] — usada como índice secundário no banco.
    pub valence: f32,
    /// Última ativação (timestamp ms desde criação) — para depressão/reciclagem.
    pub last_active_ms: f64,
    /// `true` quando alocado para tarefa; `false` quando no pool de repouso.
    pub in_use: bool,
}

impl NeuralBlock {
    /// Bloco vazio em estado de repouso.
    fn vazio() -> Self {
        Self {
            raw: 0,
            precision: PrecisionLevel::FP4,
            level: CorticalLevel::C0Sensorial,
            ltp_count: 0,
            concept_id: None,
            valence: 0.0,
            last_active_ms: 0.0,
            in_use: false,
        }
    }

    /// Lê o valor armazenado conforme a precisão atual (sempre f32 para uso uniforme).
    pub fn ler_f32(&self) -> f32 {
        let bits = self.raw & self.precision.mascara();
        match self.precision {
            // FP4: 1 sinal + 3 mantissa, range simbólico [-7, +7] / 7
            PrecisionLevel::FP4 => {
                let sign = (bits & 0x8) != 0;
                let mag = (bits & 0x7) as f32 / 7.0;
                if sign { -mag } else { mag }
            }
            // FP8: 1 sinal + 4 expo + 3 mantissa (E4M3 simplificado)
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
            // INT32 não deve ser lido como f32 — usar ler_i32 ou ler_id
            PrecisionLevel::INT32 => bits as f32,
        }
    }

    /// Escreve um valor f32 conforme a precisão atual.
    /// Bits acima da precisão atual permanecem em zero (mascaramento natural).
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

    /// Lê como inteiro 32 bits (template/ponteiro/ID hash).
    #[inline]
    pub fn ler_i32(&self) -> i32 {
        self.raw as i32
    }

    /// Escreve um inteiro 32 bits (apenas válido em precisão INT32).
    #[inline]
    pub fn escrever_i32(&mut self, valor: i32) {
        if self.precision == PrecisionLevel::INT32 {
            self.raw = valor as u32;
        }
    }

    /// Registra um evento LTP — promove precisão se acumulação cruzar limiar.
    /// Respeita o teto cortical do nível.
    pub fn evento_ltp(&mut self, t_ms: f64) {
        self.ltp_count = self.ltp_count.saturating_add(1);
        self.last_active_ms = t_ms;
        let teto = self.level.precisao_maxima();
        let nova = match self.ltp_count {
            n if n >= LTP_PROMO_FP32_INT32 => PrecisionLevel::INT32,
            n if n >= LTP_PROMO_FP16_FP32  => PrecisionLevel::FP32,
            n if n >= LTP_PROMO_FP8_FP16   => PrecisionLevel::FP16,
            n if n >= LTP_PROMO_FP4_FP8    => PrecisionLevel::FP8,
            _                              => self.precision,
        };
        // Não excede o teto do nível cortical
        let nova = if (nova as u8) > (teto as u8) { teto } else { nova };
        if (nova as u8) > (self.precision as u8) {
            // Promoção: o bit pattern atual é preservado nos bits inferiores.
            // Os bits superiores (que estavam zerados) tornam-se utilizáveis.
            // Não há realocação — apenas mudança da máscara lógica.
            self.precision = nova;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEÇÃO 3 — POOL GLOBAL DE NEURÔNIOS EM REPOUSO
// ─────────────────────────────────────────────────────────────────────────────

/// Pool global pré-alocado de blocos neurais.
///
/// Modelo: pool fixo de N blocos u32 alocados uma única vez na inicialização.
/// Blocos no pool de repouso (`in_use = false`) ocupam memória mas não são
/// processados pelo loop neural. Quando uma tarefa precisa de capacidade
/// (ex: aprender uma nova palavra), `aloca_para_tarefa()` extrai um bloco;
/// quando termina, `devolver_ao_pool()` zera e devolve.
///
/// Indexação:
///   - `free_list`: FIFO de índices disponíveis
///   - `concept_index`: HashMap<String, usize> para Localist Coding rápido
///   - `valence_index`: lista ordenada por valência para busca por peso emocional
#[derive(Debug)]
pub struct NeuralPool {
    pub blocks: Vec<NeuralBlock>,
    pub free_list: VecDeque<usize>,
    pub capacity: usize,
    /// Localist Coding: concept_id → bloco_idx (busca O(1) por nome).
    pub concept_index: std::collections::HashMap<String, usize>,
    /// Total de alocações vitalícias (métrica).
    pub total_alocacoes: u64,
    /// Total de devoluções ao pool (métrica).
    pub total_devolucoes: u64,
}

impl NeuralPool {
    /// Cria um novo pool com capacidade fixa.
    /// Aloca todos os blocos imediatamente como NeuralBlock::vazio().
    pub fn new(capacity: usize) -> Self {
        let blocks: Vec<NeuralBlock> = (0..capacity).map(|_| NeuralBlock::vazio()).collect();
        let free_list: VecDeque<usize> = (0..capacity).collect();
        Self {
            blocks,
            free_list,
            capacity,
            concept_index: std::collections::HashMap::new(),
            total_alocacoes: 0,
            total_devolucoes: 0,
        }
    }

    /// Aloca um bloco para uma tarefa específica.
    ///
    /// Retorna o índice do bloco alocado, ou `None` se o pool está exausto.
    /// Implementa Localist Coding: o `concept_id` é registrado no `concept_index`.
    /// Se o conceito já existe (palavra repetida), retorna o índice existente
    /// — garante 1 conceito ↔ 1 neurônio físico.
    pub fn aloca_para_tarefa(
        &mut self,
        concept_id: String,
        level: CorticalLevel,
        t_ms: f64,
    ) -> Option<usize> {
        // Localist: se já existe, reusa
        if let Some(&idx) = self.concept_index.get(&concept_id) {
            self.blocks[idx].last_active_ms = t_ms;
            return Some(idx);
        }
        // Caso contrário, recruta do pool
        let idx = self.free_list.pop_front()?;
        let b = &mut self.blocks[idx];
        b.raw = 0;
        b.precision = level.precisao_inicial();
        b.level = level;
        b.ltp_count = 0;
        b.concept_id = Some(concept_id.clone());
        b.valence = 0.0;
        b.last_active_ms = t_ms;
        b.in_use = true;
        self.concept_index.insert(concept_id, idx);
        self.total_alocacoes += 1;
        Some(idx)
    }

    /// Devolve um bloco ao pool — Reset Neural completo.
    ///
    /// Zera todos os campos do bloco (raw, ltp_count, concept_id, valence)
    /// e o adiciona de volta ao final do free_list. Remove do concept_index.
    /// Garante: nenhum vazamento de memória semântica entre tarefas.
    pub fn devolver_ao_pool(&mut self, idx: usize) {
        if idx >= self.capacity { return; }
        let b = &mut self.blocks[idx];
        if !b.in_use { return; }

        if let Some(cid) = b.concept_id.take() {
            self.concept_index.remove(&cid);
        }
        b.raw = 0;
        b.precision = PrecisionLevel::FP4;
        b.level = CorticalLevel::C0Sensorial;
        b.ltp_count = 0;
        b.valence = 0.0;
        b.last_active_ms = 0.0;
        b.in_use = false;

        self.free_list.push_back(idx);
        self.total_devolucoes += 1;
    }

    /// Localist lookup: encontra o bloco que representa um conceito.
    #[inline]
    pub fn buscar_conceito(&self, concept_id: &str) -> Option<&NeuralBlock> {
        self.concept_index.get(concept_id).and_then(|&idx| self.blocks.get(idx))
    }

    #[inline]
    pub fn buscar_conceito_mut(&mut self, concept_id: &str) -> Option<&mut NeuralBlock> {
        let idx = *self.concept_index.get(concept_id)?;
        self.blocks.get_mut(idx)
    }

    /// Registra um evento LTP no bloco do conceito (promove precisão).
    pub fn ltp_em_conceito(&mut self, concept_id: &str, t_ms: f64) {
        if let Some(b) = self.buscar_conceito_mut(concept_id) {
            b.evento_ltp(t_ms);
        }
    }

    /// Atualiza a valência emocional de um conceito (indexação C3/C4).
    pub fn atualizar_valencia(&mut self, concept_id: &str, valence: f32) {
        if let Some(b) = self.buscar_conceito_mut(concept_id) {
            b.valence = valence.clamp(-1.0, 1.0);
        }
    }

    /// Recicla blocos inativos: devolve ao pool todos os blocos que não
    /// foram ativados em mais de `idade_max_ms`. Implementa depressão sináptica
    /// global: conceitos não usados são esquecidos para liberar capacidade.
    pub fn reciclar_inativos(&mut self, t_ms_atual: f64, idade_max_ms: f64) -> usize {
        let mut a_devolver: Vec<usize> = Vec::new();
        for (i, b) in self.blocks.iter().enumerate() {
            if b.in_use && (t_ms_atual - b.last_active_ms) > idade_max_ms {
                // Não recicla C3/C4 (memória de longo prazo)
                if b.level != CorticalLevel::C3Contextual
                    && b.level != CorticalLevel::C4Abstrato
                {
                    a_devolver.push(i);
                }
            }
        }
        let n = a_devolver.len();
        for idx in a_devolver {
            self.devolver_ao_pool(idx);
        }
        n
    }

    // ── Métricas ─────────────────────────────────────────────────────────

    pub fn n_em_uso(&self) -> usize {
        self.capacity - self.free_list.len()
    }

    pub fn n_disponivel(&self) -> usize {
        self.free_list.len()
    }

    pub fn taxa_ocupacao(&self) -> f32 {
        if self.capacity == 0 { 0.0 } else {
            self.n_em_uso() as f32 / self.capacity as f32
        }
    }

    /// Distribuição de blocos por nível de precisão atual.
    pub fn dist_precisao(&self) -> [usize; 5] {
        let mut d = [0usize; 5];
        for b in &self.blocks {
            if b.in_use { d[b.precision as usize] += 1; }
        }
        d
    }

    /// Distribuição de blocos por nível cortical.
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
// SEÇÃO 4 — TESTES DE SANIDADE
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alocacao_e_localist_coding() {
        let mut pool = NeuralPool::new(8);
        let i1 = pool.aloca_para_tarefa("maçã".into(), CorticalLevel::C2Lexical, 0.0);
        let i2 = pool.aloca_para_tarefa("maçã".into(), CorticalLevel::C2Lexical, 1.0);
        // Localist: mesma palavra → mesmo bloco
        assert_eq!(i1, i2);
        assert_eq!(pool.n_em_uso(), 1);
    }

    #[test]
    fn reset_e_devolucao() {
        let mut pool = NeuralPool::new(4);
        let i = pool.aloca_para_tarefa("foo".into(), CorticalLevel::C0Sensorial, 0.0).unwrap();
        pool.devolver_ao_pool(i);
        assert_eq!(pool.n_em_uso(), 0);
        assert_eq!(pool.buscar_conceito("foo").map(|b| b.in_use), None);
    }

    #[test]
    fn metaplasticidade_promove_fp4_fp8() {
        let mut pool = NeuralPool::new(2);
        // C1Perceptual: inicia em FP4, teto FP32 — permite testar a 1ª promoção
        let i = pool.aloca_para_tarefa("X".into(), CorticalLevel::C1Perceptual, 0.0).unwrap();
        assert_eq!(pool.blocks[i].precision, PrecisionLevel::FP4);
        for k in 1..=11 {
            pool.ltp_em_conceito("X", k as f64);
        }
        assert_eq!(pool.blocks[i].precision, PrecisionLevel::FP8,
            "esperado FP4 → FP8 após 11 LTP events");
    }

    #[test]
    fn teto_cortical_respeita_c0() {
        let mut pool = NeuralPool::new(2);
        let _ = pool.aloca_para_tarefa("p".into(), CorticalLevel::C0Sensorial, 0.0);
        // Acumula muitos LTP para tentar passar do teto FP16
        for k in 0..2000 { pool.ltp_em_conceito("p", k as f64); }
        let b = pool.buscar_conceito("p").unwrap();
        // Teto C0 = FP16
        assert!((b.precision as u8) <= (PrecisionLevel::FP16 as u8));
    }
}
