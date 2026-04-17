// src/storage/episodic.rs
// Memória episódica persistente da Selene Brain 2.0.
//
// Armazena episódios (momentos de experiência com contexto perceptual) no
// SurrealDB, tabela `episodios`. Complementa o EventoEpisodico em memória
// (bridge.rs/VecDeque) com persistência de longo prazo.
#![allow(dead_code)]

use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;

use super::tipos::SpikeHash;

// ─── Struct principal ─────────────────────────────────────────────────────────

/// Episódio persistente: um momento de experiência com seus spike patterns associados.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episodio {
    /// UUID v4 string — identificador único.
    pub id:           String,
    /// Timestamp de ocorrência (Unix segundos).
    pub timestamp:    f64,
    /// Descrição textual breve do episódio.
    pub descricao:    String,
    /// Hashes dos spike patterns ativos no momento do episódio.
    pub spike_hashes: Vec<SpikeHash>,
    /// Valência emocional no momento (-1.0 = negativo, +1.0 = positivo).
    pub emocao:       f32,
    /// Contexto adicional (palavras do diálogo, rótulo de sensor, etc.).
    pub contexto:     Option<String>,
}

impl Episodio {
    /// Cria um novo Episodio com UUID v4 e timestamp atual.
    pub fn novo(
        descricao:    impl Into<String>,
        spike_hashes: Vec<SpikeHash>,
        emocao:       f32,
        contexto:     Option<String>,
    ) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        Self { id, timestamp, descricao: descricao.into(), spike_hashes, emocao, contexto }
    }
}

// ─── Persistência ─────────────────────────────────────────────────────────────

/// Persiste um episódio no banco de dados.
pub async fn registrar_episodio(db: &Surreal<Db>, ep: Episodio) -> surrealdb::Result<()> {
    let _: Vec<Episodio> = db.create("episodios").content(ep).await?;
    Ok(())
}

/// Retorna os `n` episódios mais recentes, ordenados do mais novo para o mais antigo.
pub async fn recuperar_episodios_recentes(db: &Surreal<Db>, n: usize) -> Vec<Episodio> {
    let res = db
        .query("SELECT * FROM episodios ORDER BY timestamp DESC LIMIT $n")
        .bind(("n", n as u64))
        .await;
    match res {
        Ok(mut r) => r.take(0).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

/// Recupera um episódio específico pelo id. Retorna None se não encontrado.
pub async fn recuperar_episodio(db: &Surreal<Db>, id: &str) -> Option<Episodio> {
    let mut res = db
        .query("SELECT * FROM episodios WHERE id = $id LIMIT 1")
        .bind(("id", id))
        .await
        .ok()?;
    let eps: Vec<Episodio> = res.take(0).ok()?;
    eps.into_iter().next()
}

/// Recupera episódios com valência emocional acima do limiar, os N mais recentes.
pub async fn recuperar_episodios_salientes(
    db: &Surreal<Db>,
    limiar_emocao: f32,
    n: usize,
) -> Vec<Episodio> {
    let res = db
        .query(
            "SELECT * FROM episodios \
             WHERE emocao > $lim OR emocao < $neg \
             ORDER BY timestamp DESC LIMIT $n",
        )
        .bind(("lim", limiar_emocao))
        .bind(("neg", -limiar_emocao))
        .bind(("n", n as u64))
        .await;
    match res {
        Ok(mut r) => r.take(0).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}
