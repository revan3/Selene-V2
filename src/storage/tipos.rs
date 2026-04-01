// src/storage/tipos.rs
// Tipos base da arquitetura spike-first da Selene Brain 2.0.
// Toda entrada externa (texto/áudio/imagem) converte para spike patterns
// antes de entrar no núcleo Rust. As chaves de armazenamento são hashes
// determinísticos desses padrões (SpikeHash).
#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ─── Hierarquia fonética ───────────────────────────────────────────────────────

/// Nível da representação na hierarquia fonética.
/// Fonema → Sílaba → Palavra → Significado → Contexto
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CamadaFonetica {
    Fonema,
    Silaba,
    Palavra,
    Significado,
    Contexto,
}

// ─── Tom de voz ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TomVoz {
    Neutro,
    Alegre,
    Triste,
    Raiva,
    Medo,
    Surpresa,
}

/// Parâmetros de entonação para síntese de voz.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entonacao {
    pub tom:        TomVoz,
    pub intensidade: f32,   // [0.0, 1.0]
    pub duracao_ms:  u32,
}

// ─── Estado neuronal instantâneo ─────────────────────────────────────────────

/// Snapshot do estado de um único neurônio Izhikevich.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuroState {
    pub v:      f32,   // potencial de membrana (mV)
    pub u:      f32,   // variável de recuperação
    pub spikes: bool,  // disparou neste timestep?
    pub t:      f64,   // timestamp Unix (s)
}

// ─── SpikeHash ────────────────────────────────────────────────────────────────

/// Chave determinística de um padrão de spikes.
/// Implementada como string hex de 16 dígitos (FNV-1a 64-bit) para compatibilidade
/// com SurrealDB como campo de lookup indexado.
pub type SpikeHash = String;

/// Computa o SpikeHash de qualquer slice de bytes usando FNV-1a 64-bit.
pub fn hash_bytes(data: &[u8]) -> SpikeHash {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME:  u64 = 0x0000_0100_0000_01b3;
    let h = data.iter().fold(OFFSET, |acc, &b| acc.wrapping_mul(PRIME) ^ b as u64);
    format!("{:016x}", h)
}

/// Computa o SpikeHash de um padrão [u64; 8] (SpikePattern do spike_codec).
pub fn hash_pattern(pattern: &[u64; 8]) -> SpikeHash {
    let bytes: Vec<u8> = pattern.iter().flat_map(|&w| w.to_le_bytes()).collect();
    hash_bytes(&bytes)
}

// ─── SpikeRecord ──────────────────────────────────────────────────────────────

/// Registro persistente de um padrão de spikes.
/// Armazenado na tabela `spikes` do SurrealDB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeRecord {
    /// Hash determinístico do padrão (campo de lookup primário).
    pub hash:      SpikeHash,
    /// Padrão compactado em bitfield (1 bit/neurônio).
    pub pattern:   Vec<u8>,
    /// Número de neurônios representados no padrão.
    pub n_neurons: usize,
    /// Nível hierárquico a que este padrão pertence.
    pub camada:    CamadaFonetica,
    /// Timestamp de criação (Unix segundos).
    pub timestamp: f64,
    /// Rótulo legível opcional (palavra, fonema, etc.).
    pub label:     Option<String>,
}

// ─── Grafo de associações ─────────────────────────────────────────────────────

/// Tipo de aresta no grafo associativo de spikes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TipoAresta {
    Excitatorio,
    Inibitor,
    Modulatorio,
    Stdp,
}

/// Aresta ponderada entre dois padrões de spikes.
/// Armazenada na tabela `associacoes_grafo` do SurrealDB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociacaoGrafo {
    /// UUID v4 string — identificador único da aresta.
    pub id:         String,
    /// SpikeHash do nó de origem.
    pub de:         SpikeHash,
    /// SpikeHash do nó de destino.
    pub para:       SpikeHash,
    /// Tipo de conexão.
    pub tipo:       TipoAresta,
    /// Peso sináptico atual (STDP modifica este campo).
    pub peso:       f32,
    /// Último uso (Unix segundos).
    pub ultimo_uso: f64,
    /// Total de co-ativações registradas.
    pub total_usos: u32,
}
