// =============================================================================
// src/learning/chunking.rs — Selene V2.2
// =============================================================================
//
// MOTOR DE CHUNKING — Emergência Hierárquica de Linguagem
//
// Princípio: neurônios que disparam juntos repetidamente (co-ativação STDP)
// formam uma unidade composta — "chunk". Assim letras → sílabas → palavras →
// frases emergem organicamente via aprendizado, sem serem pré-programadas.
//
// Campos reais do NeuronioHibrido usados aqui:
//   trace_pre  → força da ativação pré-sináptica recente (proxy de peso STDP)
//   threshold  → threshold adaptivo (sobe após spike — indica neurônio ativo)
//   peso       → PesoNeuronio::valor_f32(escala) → peso sináptico real
//   last_spike_ms → timing do último disparo (para janela temporal)
//
// Fluxo de integração no main.rs:
//
//   // 1. tick neural (já existe)
//   let spikes = camada_temporal.update(&inputs, dt, t_ms);
//
//   // 2. chunking detecta padrões
//   let emocao = limbic.emotional_state; // campo do seu limbic.rs
//   let novos = chunking.registrar_spikes(&spikes, &camada_temporal, emocao, t_ms);
//
//   // 3. persiste chunks novos no grafo de memória
//   for chunk in &novos {
//       memory.criar_conexao(chunk.para_conexao_sinaptica()).await;
//       println!("[CHUNK] {:?} → {} (valence={:.2})", chunk.tipo, chunk.simbolo, chunk.valence);
//   }
//
//   // 4. propaga RPE do RL para reforçar/enfraquecer chunks recentes
//   let rpe = rl.update(&padrao, neuro.dopamine, acao, &config);
//   chunking.aplicar_rpe(rpe);
//
// =============================================================================

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;
use crate::synaptic_core::{CamadaHibrida, NeuronioHibrido};
use crate::storage::memory_graph::ConexaoSinaptica;
use crate::brain_zones::RegionType;

// =============================================================================
// CONSTANTES
// =============================================================================

/// Co-ativações para promover padrão a chunk (LTP early-phase ~5 repetições).
const CHUNK_THRESHOLD: u32 = 5;

/// trace_pre mínimo médio para aceitar o chunk.
/// trace_pre do NeuronioHibrido começa em 0 e sobe até 1.0 com spikes.
/// 0.4 = neurônios disparando regularmente juntos, não por acaso.
const TRACE_MINIMO_CHUNK: f32 = 0.4;

/// Janela temporal para co-ativação (ms) — gamma band cortical.
const JANELA_MS: u64 = 20;

/// Máximo de neurônios por chunk (evita padrões gigantes sem semântica).
const MAX_POR_CHUNK: usize = 8;

/// Decaimento do traço de registro por tick sem reforço.
const DECAIMENTO_REGISTRO: f32 = 0.92;

// =============================================================================
// HIERARQUIA
// =============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum TipoChunk {
    /// 2-4 neurônios de letra → sílaba primitiva
    Primitivo,
    /// 2+ chunks primitivos → palavra composta
    Composto,
    /// 2+ chunks compostos + boundary → frase
    Sequencia,
}

// =============================================================================
// CHUNK
// =============================================================================

#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: Uuid,
    pub tipo: TipoChunk,

    /// Índices dos neurônios na CamadaHibrida que compõem este chunk.
    pub indices: Vec<usize>,

    /// Símbolo inferido. Começa como "chunk_i_j_k".
    /// Substituir por letra/sílaba real quando tiver mapa neurônio→char.
    pub simbolo: String,

    /// Média de trace_pre dos neurônios no momento da emergência.
    /// Representa a força STDP acumulada do padrão.
    pub forca_stdp: f32,

    /// Peso sináptico médio real (via PesoNeuronio::valor_f32).
    pub peso_medio: f32,

    /// Quantas co-ativações foram observadas antes da emergência.
    pub frequencia: u32,

    /// Valência emocional do limbic.rs no momento da criação.
    /// Positiva = contexto de recompensa → reforço dopaminérgico.
    /// Negativa = contexto de medo/dor → aprendizado de evitação.
    pub valence: f32,

    pub criado_em: Instant,
}

impl Chunk {
    /// Converte para ConexaoSinaptica para salvar no MemoryTierV2.
    ///
    /// emocao_media > 0.6 → conexoes_ativas (acesso rápido L1/L2)
    /// emocao_media < 0.6 → conexoes_dormentes (consolidado no sono REM)
    pub fn para_conexao_sinaptica(&self) -> ConexaoSinaptica {
        use std::time::{SystemTime, UNIX_EPOCH};
        let agora = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        ConexaoSinaptica {
            id: self.id,
            de_neuronio: Uuid::from_u128(
                self.indices.first().copied().unwrap_or(0) as u128
            ),
            para_neuronio: Uuid::from_u128(
                self.indices.last().copied().unwrap_or(0) as u128
            ),
            peso: self.peso_medio,
            criada_em: agora,
            ultimo_uso: Some(agora),
            total_usos: self.frequencia,
            emocao_media: self.valence.clamp(0.0, 1.0),
            contexto_criacao: None,
            contexto_semantico: crate::storage::memory_graph::ContextoSemantico::Hipotese,
            marcador_poda: 1.0,
        }
    }

    /// Vetor de ativação para ReinforcementLearning::codificar_estado().
    /// Mapeia índices dos neurônios para floats normalizados.
    pub fn para_padrao_rl(&self, n_total: usize) -> Vec<f32> {
        let mut padrao = vec![0.0f32; n_total.min(512)];
        for &idx in &self.indices {
            if idx < padrao.len() {
                padrao[idx] = self.forca_stdp;
            }
        }
        padrao
    }
}

// =============================================================================
// REGISTRO INTERNO
// =============================================================================

struct Registro {
    indices: Vec<usize>,
    contagem: u32,
    trace: f32,
    ultimo_disparo: Instant,
    soma_trace_pre: f32,   // acumula trace_pre para calcular média
    soma_peso: f32,        // acumula peso real para calcular média
}

// =============================================================================
// CHUNKING ENGINE
// =============================================================================

pub struct ChunkingEngine {
    registros: HashMap<String, Registro>,
    pub chunks: Vec<Chunk>,
    regiao: RegionType,
    tick_counter: u64,
}

impl ChunkingEngine {
    pub fn new(regiao: RegionType) -> Self {
        Self {
            registros: HashMap::new(),
            chunks: Vec::new(),
            regiao,
            tick_counter: 0,
        }
    }

    // -------------------------------------------------------------------------
    // ENTRADA PRINCIPAL
    // Chamar após CamadaHibrida::update() no loop principal.
    // -------------------------------------------------------------------------

    /// Registra os spikes deste tick e detecta emergência de chunks.
    ///
    /// # Parâmetros
    /// - `spikes`      : saída de CamadaHibrida::update() — true = disparou
    /// - `camada`      : referência à camada (lê trace_pre e peso dos neurônios)
    /// - `emocao`      : valência emocional do limbic.rs (-1.0 .. 1.0)
    /// - `t_ms`        : tempo atual em ms (mesmo parâmetro passado ao update)
    ///
    /// # Retorno
    /// Vec de novos chunks emergidos neste tick (geralmente vazio).
    pub fn registrar_spikes(
        &mut self,
        spikes: &[bool],
        camada: &CamadaHibrida,
        emocao: f32,
        t_ms: f32,
    ) -> Vec<Chunk> {
        self.tick_counter += 1;

        // Coleta índices dos neurônios que dispararam
        let mut ativos: Vec<usize> = spikes.iter()
            .enumerate()
            .filter(|(_, &s)| s)
            .map(|(i, _)| i)
            .collect();

        if ativos.len() < 2 {
            self.decair_registros();
            return Vec::new();
        }

        // Limita ao máximo, priorizando neurônios com trace_pre mais alto
        if ativos.len() > MAX_POR_CHUNK {
            ativos = self.top_por_trace(&ativos, camada, MAX_POR_CHUNK);
        }

        let chave = Self::chave(&ativos);
        let agora = Instant::now();

        // Métricas reais dos neurônios ativos
        let trace_medio = Self::media_trace_pre(&ativos, camada);
        let peso_medio  = Self::media_peso(&ativos, camada);

        // Atualiza registro
        let reg = self.registros.entry(chave.clone()).or_insert(Registro {
            indices: ativos.clone(),
            contagem: 0,
            trace: 0.0,
            ultimo_disparo: agora,
            soma_trace_pre: 0.0,
            soma_peso: 0.0,
        });

        // Verifica janela temporal (proporcional ao número de neurônios)
        let dentro_janela = reg.ultimo_disparo.elapsed()
            < Duration::from_millis(JANELA_MS * ativos.len() as u64);

        if dentro_janela || reg.trace > 0.1 {
            reg.contagem += 1;
            reg.trace = (reg.trace + 1.0).min(10.0);
            reg.soma_trace_pre += trace_medio;
            reg.soma_peso += peso_medio;
        } else {
            // Padrão voltou depois de longa pausa — reseta (novo aprendizado)
            reg.contagem = 1;
            reg.trace = 1.0;
            reg.soma_trace_pre = trace_medio;
            reg.soma_peso = peso_medio;
        }
        reg.ultimo_disparo = agora;

        let contagem       = reg.contagem;
        let forca_acum     = reg.soma_trace_pre / contagem as f32;
        let peso_acum      = reg.soma_peso / contagem as f32;

        // Limpeza periódica
        if self.tick_counter % 1000 == 0 {
            self.registros.retain(|_, r| r.trace > 0.05);
        }

        // Verifica emergência
        let mut novos = Vec::new();
        let ja_existe = self.chunks.iter()
            .any(|c| Self::chave(&c.indices) == chave);

        if contagem >= CHUNK_THRESHOLD
            && forca_acum >= TRACE_MINIMO_CHUNK
            && !ja_existe
        {
            let tipo    = self.inferir_tipo(&ativos);
            let simbolo = Self::inferir_simbolo(&ativos);

            let chunk = Chunk {
                id: Uuid::new_v4(),
                tipo,
                indices: ativos.clone(),
                simbolo,
                forca_stdp: forca_acum,
                peso_medio: peso_acum,
                frequencia: contagem,
                valence: emocao.clamp(-1.0, 1.0),
                criado_em: Instant::now(),
            };

            self.chunks.push(chunk.clone());
            novos.push(chunk);
        }

        novos
    }

    // -------------------------------------------------------------------------
    // REFORÇO POR RPE
    // Chamar após ReinforcementLearning::update()
    // -------------------------------------------------------------------------

    /// Propaga RPE para chunks criados nos últimos 5 segundos.
    ///
    /// RPE > 0 (surpresa positiva) → forca_stdp sobe, valence sobe.
    /// RPE < 0 (decepção)          → forca_stdp cai, valence cai.
    ///
    /// Isso conecta o aprendizado por reforço ao chunking:
    /// chunks que emergem antes de uma recompensa são reforçados,
    /// os que emergem antes de punição são enfraquecidos.
    pub fn aplicar_rpe(&mut self, rpe: f32) {
        let janela = Duration::from_secs(5);
        for chunk in self.chunks.iter_mut() {
            if chunk.criado_em.elapsed() < janela {
                chunk.forca_stdp = (chunk.forca_stdp + rpe * 0.05).clamp(0.0, 1.0);
                chunk.valence    = (chunk.valence    + rpe * 0.10).clamp(-1.0, 1.0);
            }
        }
    }

    // -------------------------------------------------------------------------
    // CONSULTAS
    // -------------------------------------------------------------------------

    pub fn chunks_por_tipo(&self, tipo: &TipoChunk) -> Vec<&Chunk> {
        self.chunks.iter().filter(|c| &c.tipo == tipo).collect()
    }

    pub fn stats(&self) -> ChunkingStats {
        ChunkingStats {
            primitivos: self.chunks_por_tipo(&TipoChunk::Primitivo).len(),
            compostos:  self.chunks_por_tipo(&TipoChunk::Composto).len(),
            sequencias: self.chunks_por_tipo(&TipoChunk::Sequencia).len(),
            padroes_monitorados: self.registros.len(),
            total_ticks: self.tick_counter,
        }
    }

    // -------------------------------------------------------------------------
    // INTERNOS
    // -------------------------------------------------------------------------

    fn chave(indices: &[usize]) -> String {
        let mut s = indices.to_vec();
        s.sort_unstable();
        s.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
    }

    fn inferir_tipo(&self, indices: &[usize]) -> TipoChunk {
        // Se algum índice já pertence a um chunk existente, este é de nível superior
        let tem_filho = indices.iter().any(|&idx| {
            self.chunks.iter().any(|c| c.indices.contains(&idx))
        });
        match (indices.len(), tem_filho) {
            (_, true)  => TipoChunk::Sequencia,
            (2..=4, _) => TipoChunk::Primitivo,
            _          => TipoChunk::Composto,
        }
    }

    fn inferir_simbolo(indices: &[usize]) -> String {
        // Placeholder até ter mapa neurônio→char no SurrealDB bootstrap.
        // Ex: quando letter_A for mapeado para índice 0, "chunk_0_5" vira "AS".
        format!(
            "chunk_{}",
            indices.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("_")
        )
    }

    /// Média de trace_pre dos neurônios ativos.
    /// trace_pre (campo real do NeuronioHibrido) representa a atividade
    /// pré-sináptica recente — o melhor proxy de "co-ativação STDP".
    fn media_trace_pre(indices: &[usize], camada: &CamadaHibrida) -> f32 {
        if indices.is_empty() { return 0.0; }
        let soma: f32 = indices.iter()
            .filter_map(|&i| camada.neuronios.get(i))
            .map(|n| n.trace_pre)
            .sum();
        soma / indices.len() as f32
    }

    /// Média do peso sináptico real (PesoNeuronio::valor_f32).
    fn media_peso(indices: &[usize], camada: &CamadaHibrida) -> f32 {
        if indices.is_empty() { return 0.0; }
        let esc = camada.escala_camada;
        let soma: f32 = indices.iter()
            .filter_map(|&i| camada.neuronios.get(i))
            .map(|n| n.peso.valor_f32(esc))
            .sum();
        soma / indices.len() as f32
    }

    /// Seleciona os N neurônios com trace_pre mais alto (mais correlacionados).
    fn top_por_trace(
        &self,
        indices: &[usize],
        camada: &CamadaHibrida,
        n: usize,
    ) -> Vec<usize> {
        let mut com_trace: Vec<(usize, f32)> = indices.iter()
            .filter_map(|&i| camada.neuronios.get(i).map(|n| (i, n.trace_pre)))
            .collect();
        com_trace.sort_by(|a, b| b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal));
        com_trace.into_iter().take(n).map(|(i, _)| i).collect()
    }

    fn decair_registros(&mut self) {
        for reg in self.registros.values_mut() {
            reg.trace *= DECAIMENTO_REGISTRO;
        }
    }
}

// =============================================================================
// STATS
// =============================================================================

#[derive(Debug)]
pub struct ChunkingStats {
    pub primitivos: usize,
    pub compostos:  usize,
    pub sequencias: usize,
    pub padroes_monitorados: usize,
    pub total_ticks: u64,
}

impl std::fmt::Display for ChunkingStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CHUNKS[prim={} comp={} seq={} | monit={} | ticks={}]",
            self.primitivos, self.compostos, self.sequencias,
            self.padroes_monitorados, self.total_ticks
        )
    }
}

// =============================================================================
// TESTES
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::synaptic_core::{CamadaHibrida, TipoNeuronal, PrecisionType};
    use crate::brain_zones::RegionType;

    fn camada(n: usize) -> CamadaHibrida {
        CamadaHibrida::new(n, "teste", TipoNeuronal::RS, None, None, 1.0)
    }

    fn spikes(ativos: &[usize], total: usize) -> Vec<bool> {
        (0..total).map(|i| ativos.contains(&i)).collect()
    }

    #[test]
    fn test_sem_spikes_sem_chunk() {
        let mut engine = ChunkingEngine::new(RegionType::Temporal);
        let c = camada(16);
        let s = vec![false; 16];
        assert!(engine.registrar_spikes(&s, &c, 0.5, 0.0).is_empty());
    }

    #[test]
    fn test_registro_acumula_contagem() {
        let mut engine = ChunkingEngine::new(RegionType::Temporal);
        let c = camada(16);
        let s = spikes(&[2, 5], 16);
        for _ in 0..3 {
            engine.registrar_spikes(&s, &c, 0.5, 0.0);
        }
        assert_eq!(engine.registros.len(), 1);
        let reg = engine.registros.values().next().unwrap();
        assert_eq!(reg.contagem, 3);
    }

    #[test]
    fn test_chunk_nao_duplicado() {
        let mut engine = ChunkingEngine::new(RegionType::Temporal);
        let c = camada(16);
        let s = spikes(&[1, 3], 16);
        // Dispara muito além do threshold
        for _ in 0..20 {
            engine.registrar_spikes(&s, &c, 0.5, 0.0);
        }
        // Não deve criar mais de 1 chunk para o mesmo padrão
        assert!(engine.chunks.len() <= 1);
    }

    #[test]
    fn test_rpe_positivo_aumenta_forca() {
        let mut engine = ChunkingEngine::new(RegionType::Temporal);
        // Injeta chunk manualmente para testar aplicar_rpe
        let chunk = Chunk {
            id: Uuid::new_v4(),
            tipo: TipoChunk::Primitivo,
            indices: vec![0, 1],
            simbolo: "ba".to_string(),
            forca_stdp: 0.5,
            peso_medio: 0.8,
            frequencia: 5,
            valence: 0.5,
            criado_em: Instant::now(),
        };
        engine.chunks.push(chunk);

        engine.aplicar_rpe(1.0);
        assert!(engine.chunks[0].forca_stdp > 0.5);
        assert!(engine.chunks[0].valence > 0.5);
    }

    #[test]
    fn test_para_conexao_emocao_media_alta() {
        let chunk = Chunk {
            id: Uuid::new_v4(),
            tipo: TipoChunk::Primitivo,
            indices: vec![0, 3],
            simbolo: "ma".to_string(),
            forca_stdp: 0.75,
            peso_medio: 0.9,
            frequencia: 5,
            valence: 0.85,
            criado_em: Instant::now(),
        };
        let conexao = chunk.para_conexao_sinaptica();
        // emocao > 0.6 → MemoryTierV2 coloca em conexoes_ativas
        assert!(conexao.emocao_media > 0.6);
        assert_eq!(conexao.peso, 0.9);
        assert_eq!(conexao.total_usos, 5);
    }

    #[test]
    fn test_stats_zeradas_inicio() {
        let engine = ChunkingEngine::new(RegionType::Temporal);
        let s = engine.stats();
        assert_eq!(s.primitivos, 0);
        assert_eq!(s.total_ticks, 0);
        println!("{}", s);
    }
}
