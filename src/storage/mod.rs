// src/storage/mod.rs
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;
use crate::brain_zones::RegionType;
use surrealdb::Surreal;
use surrealdb::engine::local::{Db, RocksDb};
use std::time::Duration;
use std::path::{Path, PathBuf};
use std::io;

pub mod memory_tier;
pub mod memory_graph;
pub mod swap_manager;
pub mod tipos;
pub mod spike_store;
pub mod episodic;
pub mod ondas;

// Re-exportações
pub use memory_graph::ConexaoSinaptica;

// ================== ESTRUTURAS DE DADOS ==================

#[derive(Debug, Clone)]
pub struct Experiencia {
    pub id: String,
    pub timestamp: f64,
    pub contexto: String,
    pub emocao: f32,
    pub neurons_ativos: Vec<usize>,
    pub conexoes_formadas: Vec<ConexaoSinaptica>,
    pub importancia: f32,
    pub ultimo_acesso: Option<f64>,
    pub frequencia_acesso: f32,
}

#[derive(Debug, Clone)]
pub struct Hipotese {
    pub descricao: String,
    pub conexoes_novas: Vec<ConexaoSinaptica>,
    pub probabilidade: f32,
    pub origem: Vec<String>,
    pub testada: bool,
    pub valida: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct BackupSistema {
    pub versao: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub memorias_consolidadas: Vec<Experiencia>,
    pub insights: Vec<Hipotese>,
    pub metricas: HashMap<String, f32>,
}

/// Converte um vetor de taxas de disparo (0..1) em spike train binário compacto (1 bit/neurônio).
/// Threshold padrão: 0.5 (neurônio disparou se taxa > 0.5)
pub fn firing_rates_to_spike_bits(rates: &[f32], threshold: f32) -> Vec<u8> {
    let n_bytes = (rates.len() + 7) / 8;
    let mut bits = vec![0u8; n_bytes];
    for (i, &r) in rates.iter().enumerate() {
        if r > threshold {
            bits[i / 8] |= 1 << (i % 8);
        }
    }
    bits
}

/// Reconstrói taxas de disparo aproximadas a partir de spike bits (0.0 ou 1.0)
pub fn spike_bits_to_firing_rates(bits: &[u8], n_neurons: usize) -> Vec<f32> {
    (0..n_neurons)
        .map(|i| if bits.get(i / 8).map(|b| b & (1 << (i % 8)) != 0).unwrap_or(false) { 1.0 } else { 0.0 })
        .collect()
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NeuralEnactiveMemory {
    pub timestamp: f64,
    pub emotion_state: f32,
    pub arousal_state: f32,
    /// Spike pattern visual compacto (1 bit/neurônio = 8x mais compacto que f32)
    pub visual_spikes: Vec<u8>,
    /// Spike pattern auditivo compacto
    pub auditory_spikes: Vec<u8>,
    /// Número original de neurônios (necessário para decodificação)
    pub n_neurons: usize,
    /// Intenção frontal — mantida em f32 (pequena, precisa de gradiente)
    pub frontal_intent: Vec<f32>,
    pub label: String,
}

impl NeuralEnactiveMemory {
    /// Cria a memória a partir de vetores de taxas de disparo (interface com o main loop)
    pub fn from_firing_rates(
        timestamp: f64,
        emotion_state: f32,
        arousal_state: f32,
        visual_rates: &[f32],
        auditory_rates: &[f32],
        frontal_intent: Vec<f32>,
        label: String,
    ) -> Self {
        let n_neurons = visual_rates.len().max(auditory_rates.len());
        Self {
            timestamp,
            emotion_state,
            arousal_state,
            visual_spikes: firing_rates_to_spike_bits(visual_rates, 0.5),
            auditory_spikes: firing_rates_to_spike_bits(auditory_rates, 0.5),
            n_neurons,
            frontal_intent,
            label,
        }
    }

    pub fn visual_rates(&self) -> Vec<f32> {
        spike_bits_to_firing_rates(&self.visual_spikes, self.n_neurons)
    }

    pub fn auditory_rates(&self) -> Vec<f32> {
        spike_bits_to_firing_rates(&self.auditory_spikes, self.n_neurons)
    }
}

/// Tipo legado mantido para compatibilidade com código existente que usa Vec<f32> diretamente.
/// Use NeuralEnactiveMemory::from_firing_rates() para novas memórias.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NeuralEnactiveMemoryRaw {
    pub timestamp: f64,
    pub emotion_state: f32,
    pub arousal_state: f32,
    pub visual_pattern: Vec<f32>,
    pub auditory_pattern: Vec<f32>,
    pub frontal_intent: Vec<f32>,
    pub label: String,
}

impl From<NeuralEnactiveMemoryRaw> for NeuralEnactiveMemory {
    fn from(r: NeuralEnactiveMemoryRaw) -> Self {
        Self::from_firing_rates(
            r.timestamp,
            r.emotion_state,
            r.arousal_state,
            &r.visual_pattern,
            &r.auditory_pattern,
            r.frontal_intent,
            r.label,
        )
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NeuralEnactiveMemoryV2 {
    pub timestamp: f64,
    pub emotion_state: f32,
    pub arousal_state: f32,
    pub visual_spikes: Vec<u8>,
    pub auditory_spikes: Vec<u8>,
    pub n_neurons: usize,
    pub frontal_intent: Vec<f32>,
    pub label: String,
    pub conexoes: Vec<ConexaoSinaptica>,
}

// ================== BRAIN STORAGE ==================

pub struct BrainStorage {
    pub db: Surreal<Db>,
    pub conexoes_ativas: HashMap<String, Vec<ConexaoSinaptica>>,
}

impl BrainStorage {
    /// CORREÇÃO FINAL: Removido block_on e corrigido motor de fallback
    pub fn dummy() -> Self {
        // Inicializa o driver síncronamente. 
        // Como é para um estado de 'dummy' (erro), não tentamos conectar a nada.
        let db = Surreal::init(); 
        
        Self {
            db,
            conexoes_ativas: HashMap::new(),
        }
    }
    
    pub async fn new() -> surrealdb::Result<Self> {
        const DB_PATH: &str = "selene_memories.db";

        match Self::try_open(DB_PATH).await {
            Ok(s) => {
                log::info!("Banco de memoria carregado com sucesso.");
                Ok(s)
            }
            Err(e) => {
                // Banco corrompido (timestamps futuros por drift de relogio, crash, etc.)
                // Preserva o banco antigo e inicia com um banco limpo.
                log::warn!("Banco corrompido ({}). Preservando backup e recriando...", e);
                let bkp = format!(
                    "selene_memories_bkp_{}.db",
                    chrono::Utc::now().format("%Y%m%d_%H%M%S")
                );
                let _ = std::fs::rename(DB_PATH, &bkp);
                log::info!("Banco antigo preservado em: {}", bkp);
                Self::try_open(DB_PATH).await
            }
        }
    }

    async fn try_open(path: &str) -> surrealdb::Result<Self> {
        let db = Surreal::new::<RocksDb>(path).await?;
        db.use_ns("selene_project").use_db("brain_v1").await?;

        // Índices vetoriais MTREE para recall por similaridade (spike bits)
        let _ = db.query("DEFINE INDEX vis_idx ON TABLE memories FIELDS visual_spikes MTREE DIST COSINE;").await;
        let _ = db.query("DEFINE INDEX aud_idx ON TABLE memories FIELDS auditory_spikes MTREE DIST COSINE;").await;
        // Índice para busca por emoção e timestamp (evita full table scan)
        let _ = db.query("DEFINE INDEX emo_idx ON TABLE memories FIELDS emotion_state;").await;
        let _ = db.query("DEFINE INDEX ts_idx  ON TABLE memories FIELDS timestamp;").await;
        // Índices de navegação sináptica
        let _ = db.query("DEFINE INDEX conexoes_origem  ON TABLE conexoes FIELDS de_neuronio;").await;
        let _ = db.query("DEFINE INDEX conexoes_destino ON TABLE conexoes FIELDS para_neuronio;").await;

        // Health-check: verifica se o MVCC consegue processar uma escrita.
        // Detecta o erro "ts is less than or equal to the latest ts" do NodeAgent do SurrealDB
        // que ocorre quando o relógio do sistema recuou desde a última sessão.
        let hc: surrealdb::Result<Vec<serde_json::Value>> = db
            .create("_healthcheck")
            .content(serde_json::json!({ "ok": true }))
            .await;
        if let Err(ref e) = hc {
            let msg = e.to_string();
            if msg.contains("ts is less than or equal") || msg.contains("timestamp") {
                return Err(hc.unwrap_err());
            }
        }
        // Limpa o registro de health-check (melhor esforço)
        let _ = db.query("DELETE _healthcheck").await;

        Ok(Self {
            db,
            conexoes_ativas: HashMap::new(),
        })
    }
    
    // ========== MÉTODOS DE PERSISTÊNCIA ==========
    
    pub async fn save_snapshot(&self, memory: NeuralEnactiveMemory) -> surrealdb::Result<()> {
        let _: Vec<NeuralEnactiveMemory> = self.db
            .create("memories")
            .content(memory)
            .await?;
        Ok(())
    }
    
    pub async fn save_memory_with_connections(&self, memory: NeuralEnactiveMemoryV2) -> surrealdb::Result<()> {
        let _: Vec<NeuralEnactiveMemoryV2> = self.db
            .create("memories")
            .content(&memory)
            .await?;
        
        for conexao in &memory.conexoes {
            let _: Vec<ConexaoSinaptica> = self.db
                .create("conexoes")
                .content(conexao)
                .await?;
        }
        
        Ok(())
    }
    
    pub async fn find_memories_by_emotion(&self, emotion_threshold: f32) -> Vec<NeuralEnactiveMemory> {
        let response = self.db
            .query("SELECT * FROM memories WHERE emotion_state > $th ORDER BY timestamp DESC LIMIT 5")
            .bind(("th", emotion_threshold))
            .await;

        match response {
            Ok(mut resp) => resp.take(0).unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }
    
    /// Busca memória com padrão auditivo similar usando distância cosine nos spike bits
    pub async fn find_similar_memory(&self, current_audio_spikes: Vec<u8>) -> Option<NeuralEnactiveMemory> {
        let response = self.db
            .query("SELECT * FROM memories ORDER BY vector::distance::cosine(auditory_spikes, $audio) ASC LIMIT 1")
            .bind(("audio", current_audio_spikes))
            .await;

        match response {
            Ok(mut resp) => {
                let memories: Vec<NeuralEnactiveMemory> = resp.take(0).unwrap_or_default();
                memories.into_iter().next()
            },
            Err(_) => None,
        }
    }

    /// Busca memória com padrão visual similar usando distância cosine nos spike bits
    pub async fn recall_full_context(&self, current_visual_spikes: Vec<u8>) -> Option<NeuralEnactiveMemory> {
        let response = self.db
            .query("SELECT * FROM memories ORDER BY vector::distance::cosine(visual_spikes, $val) ASC LIMIT 1")
            .bind(("val", current_visual_spikes))
            .await;

        match response {
            Ok(mut resp) => {
                let memories: Vec<NeuralEnactiveMemory> = resp.take(0).unwrap_or_default();
                memories.into_iter().next()
            },
            Err(_) => None,
        }
    }

    /// Recall multimodal: combina distância visual + auditiva nos spike bits
    pub async fn recall_multimodal(&self, vision_spikes: Vec<u8>, audio_spikes: Vec<u8>) -> Option<NeuralEnactiveMemory> {
        let response = self.db
            .query("SELECT *, (vector::distance::cosine(visual_spikes, $vis) + vector::distance::cosine(auditory_spikes, $aud)) as total_dist
                    FROM memories ORDER BY total_dist ASC LIMIT 1")
            .bind(("vis", vision_spikes))
            .bind(("aud", audio_spikes))
            .await;

        match response {
            Ok(mut resp) => {
                let memories: Vec<NeuralEnactiveMemory> = resp.take(0).unwrap_or_default();
                memories.into_iter().next()
            },
            Err(_) => None,
        }
    }

    // ========== PERSISTÊNCIA SINÁPTICA ==========

    /// Persiste uma nova conexão sináptica no DB (usada pelo REM e pelo loop neural).
    pub async fn save_conexao(&self, conexao: &ConexaoSinaptica) -> surrealdb::Result<()> {
        let _: Vec<ConexaoSinaptica> = self.db
            .create("conexoes")
            .content(conexao)
            .await?;
        Ok(())
    }

    /// Atualiza peso e marcador_poda de uma sinapse existente no DB.
    pub async fn update_conexao_peso(
        &self,
        id: uuid::Uuid,
        peso: f32,
        marcador_poda: f32,
        ultimo_uso: f64,
        total_usos: u32,
    ) -> surrealdb::Result<()> {
        self.db
            .query("UPDATE conexoes SET peso = $peso, marcador_poda = $mp, ultimo_uso = $uso, total_usos = $n WHERE id = $id")
            .bind(("id",  id.to_string()))
            .bind(("peso", peso))
            .bind(("mp",   marcador_poda))
            .bind(("uso",  ultimo_uso))
            .bind(("n",    total_usos))
            .await?;
        Ok(())
    }

    /// Remove uma sinapse do DB (chamada durante a poda).
    pub async fn delete_conexao(&self, id: uuid::Uuid) -> surrealdb::Result<()> {
        self.db
            .query("DELETE conexoes WHERE id = $id")
            .bind(("id", id.to_string()))
            .await?;
        Ok(())
    }

    /// Remove um lote de sinapses do DB em uma única query (mais eficiente que N deletes).
    pub async fn delete_conexoes_batch(&self, ids: &[uuid::Uuid]) -> surrealdb::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let lista: Vec<String> = ids.iter().map(|u| format!("'{}'", u)).collect();
        let query = format!("DELETE conexoes WHERE id IN [{}]", lista.join(", "));
        self.db.query(query).await?;
        Ok(())
    }
}

// ========== BACKUP ==========

/// Copia o diretório do RocksDB para o HDD frio (D:) com timestamp.
/// Retorna o caminho do backup criado.
///
/// Layout no HDD:
///   D:/Selene_Backup_RAM/backup_YYYYMMDD_HHMMSS/  ← snapshot do RocksDB
///   D:/Selene_Archive/                            ← neurônios individuais JSON
pub async fn backup_to_hdd(
    db_path: &str,
    backup_root: &str,
) -> io::Result<PathBuf> {
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
    let dest = PathBuf::from(backup_root).join(format!("backup_{}", timestamp));

    std::fs::create_dir_all(&dest)?;

    // Copia arquivos do diretório RocksDB para o destino.
    // Arquivos de lock são pulados (mantidos abertos exclusivamente pelo DB).
    // Erros por arquivo são ignorados — o backup é melhor-esforço.
    let mut copiados = 0usize;
    let mut pulados  = 0usize;
    for entry in std::fs::read_dir(db_path)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_file() { continue; }

        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Pula LOCK e arquivos .lock — mantidos abertos pelo RocksDB/SurrealDB
        if name_str == "LOCK" || name_str.ends_with(".lock") {
            pulados += 1;
            continue;
        }

        let dest_file = dest.join(&name);
        match std::fs::copy(entry.path(), &dest_file) {
            Ok(_)  => copiados += 1,
            Err(e) => {
                log::warn!("[Backup] Pulando {} — {}", name_str, e);
                pulados += 1;
            }
        }
    }
    log::info!("[Backup] {copiados} arquivos copiados, {pulados} pulados.");

    // Grava manifesto do backup
    let manifest = format!(
        "Selene Brain 2.0 — Backup\nTimestamp: {}\nFonte: {}\nDestino: {}\n",
        timestamp,
        db_path,
        dest.display()
    );
    std::fs::write(dest.join("BACKUP_MANIFEST.txt"), manifest)?;

    log::info!("💾 [Backup] Snapshot salvo em: {}", dest.display());
    Ok(dest)
}

/// Versão síncrona do backup (para uso no handler de shutdown)
pub fn backup_to_hdd_sync(db_path: &str, backup_root: &str) -> io::Result<PathBuf> {
    let rt = tokio::runtime::Handle::try_current();
    match rt {
        Ok(handle) => {
            let db = db_path.to_string();
            let br = backup_root.to_string();
            // bloqueia na handle tokio existente de forma síncrona
            std::thread::spawn(move || {
                tokio::runtime::Runtime::new()
                    .unwrap()
                    .block_on(backup_to_hdd(&db, &br))
            }).join().unwrap_or_else(|_| Err(io::Error::new(io::ErrorKind::Other, "thread panic")))
        }
        Err(_) => {
            // fora de contexto async — bloqueia diretamente
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(backup_to_hdd(db_path, backup_root))
        }
    }
}

// ========== CONVERSÕES ==========

impl NeuralEnactiveMemoryV2 {
    pub fn visual_rates(&self) -> Vec<f32> {
        spike_bits_to_firing_rates(&self.visual_spikes, self.n_neurons)
    }
    pub fn auditory_rates(&self) -> Vec<f32> {
        spike_bits_to_firing_rates(&self.auditory_spikes, self.n_neurons)
    }
}

impl From<NeuralEnactiveMemory> for NeuralEnactiveMemoryV2 {
    fn from(original: NeuralEnactiveMemory) -> Self {
        Self {
            timestamp: original.timestamp,
            emotion_state: original.emotion_state,
            arousal_state: original.arousal_state,
            visual_spikes: original.visual_spikes,
            auditory_spikes: original.auditory_spikes,
            n_neurons: original.n_neurons,
            frontal_intent: original.frontal_intent,
            label: original.label,
            conexoes: Vec::new(),
        }
    }
}

/// Serializa o modelo de linguagem da Selene para JSON exportável.
/// Contém vocabulário (palavra→valência), grafo de associações e padrões de frase.
/// Este arquivo é o "cérebro linguístico" — independente do estado neural (RocksDB).
pub fn exportar_linguagem(
    vocabulario: &HashMap<String, f32>,
    associacoes: &HashMap<String, Vec<(String, f32)>>,
    frases: &[Vec<String>],
) -> String {
    let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();
    let n_associacoes: usize = associacoes.values().map(|v| v.len()).sum();

    // Converte associacoes para formato JSON [[w2, peso], ...]
    let assoc_json: HashMap<&String, Vec<serde_json::Value>> = associacoes.iter()
        .map(|(w1, vizinhos)| {
            let pares: Vec<serde_json::Value> = vizinhos.iter()
                .map(|(w2, peso)| serde_json::json!([w2, peso]))
                .collect();
            (w1, pares)
        })
        .collect();

    let payload = serde_json::json!({
        "selene_linguagem_v1": {
            "metadata": {
                "versao":          "1.0",
                "criado_em":       timestamp,
                "n_palavras":      vocabulario.len(),
                "n_associacoes":   n_associacoes,
                "n_frases_padrao": frases.len(),
                "descricao":       "Modelo de linguagem emergente da Selene Brain 2.0"
            },
            "vocabulario":    vocabulario,
            "associacoes":    assoc_json,
            "frases_padrao":  frases
        }
    });

    serde_json::to_string_pretty(&payload).unwrap_or_default()
}
