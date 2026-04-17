// src/storage/spike_store.rs
// Camada de persistência spike-first para a Selene Brain 2.0.
//
// Mapeia os 5 "Column Families" do design document como tabelas SurrealDB:
//
//   spikes            — padrões de spikes indexados por hash
//   associacoes_grafo — arestas do grafo associativo (STDP)
//   fonemas           — mapeamento fonema/palavra → SpikeHash
//   pesos_stdp        — histórico de deltas de peso (STDP log)
//   episodios         — memória episódica persistente
//
// Todas as funções recebem `&Surreal<Db>` e retornam surrealdb::Result ou Option.
#![allow(dead_code)]

use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;

use super::tipos::{AssociacaoGrafo, SpikeHash, SpikeRecord};

// ─── Schema ───────────────────────────────────────────────────────────────────

/// Cria índices e tabelas necessárias. Idempotente — pode ser chamado a cada boot.
pub async fn inicializar_schema(db: &Surreal<Db>) -> surrealdb::Result<()> {
    // Tabela de spike records — indexed pelo campo hash
    let _ = db.query("DEFINE INDEX spike_hash_idx ON TABLE spikes FIELDS hash UNIQUE;").await;

    // Tabela do grafo associativo — indexed pela origem para get_associacoes()
    let _ = db.query("DEFINE INDEX assoc_de_idx   ON TABLE associacoes_grafo FIELDS de;").await;
    let _ = db.query("DEFINE INDEX assoc_para_idx ON TABLE associacoes_grafo FIELDS para;").await;

    // Fonemas — indexed pelo campo fonema para lookup direto
    let _ = db.query("DEFINE INDEX fonema_key_idx ON TABLE fonemas FIELDS fonema UNIQUE;").await;

    // Pesos STDP — indexed por (de, para) para filtro rápido
    let _ = db.query("DEFINE INDEX peso_de_idx    ON TABLE pesos_stdp FIELDS de;").await;
    let _ = db.query("DEFINE INDEX peso_para_idx  ON TABLE pesos_stdp FIELDS para;").await;

    // Episódios — indexed por timestamp para recuperar os N mais recentes
    let _ = db.query("DEFINE INDEX ep_ts_idx      ON TABLE episodios FIELDS timestamp;").await;

    Ok(())
}

// ─── Spikes ──────────────────────────────────────────────────────────────────

/// Persiste um SpikeRecord. Usa UPSERT para ser idempotente (hash é unique).
pub async fn put_spike(db: &Surreal<Db>, record: SpikeRecord) -> surrealdb::Result<()> {
    db.query(
        "INSERT INTO spikes $rec ON DUPLICATE KEY UPDATE \
         pattern = $rec.pattern, timestamp = $rec.timestamp, label = $rec.label RETURN NONE"
    )
    .bind(("rec", &record))
    .await?;
    Ok(())
}

/// Recupera um SpikeRecord pelo hash. Retorna None se não encontrado.
pub async fn get_spike(db: &Surreal<Db>, hash: &str) -> Option<SpikeRecord> {
    let mut res = db
        .query("SELECT * FROM spikes WHERE hash = $h LIMIT 1")
        .bind(("h", hash))
        .await
        .ok()?;
    let records: Vec<SpikeRecord> = res.take(0).ok()?;
    records.into_iter().next()
}

// ─── Associações do grafo ────────────────────────────────────────────────────

/// Persiste uma aresta AssociacaoGrafo (auto-create, sem dedup por hash).
pub async fn put_associacao(db: &Surreal<Db>, assoc: AssociacaoGrafo) -> surrealdb::Result<()> {
    let _: Vec<AssociacaoGrafo> = db.create("associacoes_grafo").content(assoc).await?;
    Ok(())
}

/// Retorna todas as arestas que partem de `from_hash`, ordenadas por peso desc.
pub async fn get_associacoes(db: &Surreal<Db>, from_hash: &str) -> Vec<AssociacaoGrafo> {
    let res = db
        .query("SELECT * FROM associacoes_grafo WHERE de = $h ORDER BY peso DESC")
        .bind(("h", from_hash))
        .await;
    match res {
        Ok(mut r) => r.take(0).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

// ─── Fonemas ─────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct FonemaEntry {
    fonema: String,
    hash:   String,
}

/// Persiste o mapeamento fonema/palavra → SpikeHash.
pub async fn put_fonema(db: &Surreal<Db>, fonema: &str, hash: &SpikeHash) -> surrealdb::Result<()> {
    db.query(
        "INSERT INTO fonemas { fonema: $f, hash: $h } \
         ON DUPLICATE KEY UPDATE hash = $h RETURN NONE"
    )
    .bind(("f", fonema))
    .bind(("h", hash.as_str()))
    .await?;
    Ok(())
}

/// Recupera o SpikeHash associado a um fonema/palavra. Retorna None se não encontrado.
pub async fn get_fonema(db: &Surreal<Db>, fonema: &str) -> Option<SpikeHash> {
    let mut res = db
        .query("SELECT hash FROM fonemas WHERE fonema = $f LIMIT 1")
        .bind(("f", fonema))
        .await
        .ok()?;
    let entries: Vec<FonemaEntry> = res.take(0).ok()?;
    entries.into_iter().next().map(|e| e.hash)
}

// ─── Pesos STDP ──────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct PesoStdp {
    de:        String,
    para:      String,
    delta:     f32,
    timestamp: f64,
}

/// Registra um delta de peso STDP entre dois padrões de spikes.
pub async fn registrar_peso(
    db: &Surreal<Db>,
    de:    &SpikeHash,
    para:  &SpikeHash,
    delta: f32,
) -> surrealdb::Result<()> {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    let entry = PesoStdp { de: de.clone(), para: para.clone(), delta, timestamp: ts };
    let _: Vec<PesoStdp> = db.create("pesos_stdp").content(entry).await?;
    Ok(())
}
