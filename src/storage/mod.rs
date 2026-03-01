// src/storage/mod.rs
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use uuid::Uuid;
use crate::brain_zones::RegionType;
use surrealdb::Surreal;
use surrealdb::engine::local::{Db, RocksDb};
use std::time::Duration;

pub mod memory_tier;
pub mod memory_graph;
pub mod swap_manager;

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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NeuralEnactiveMemory {
    pub timestamp: f64,
    pub emotion_state: f32,
    pub arousal_state: f32,
    pub visual_pattern: Vec<f32>,
    pub auditory_pattern: Vec<f32>,
    pub frontal_intent: Vec<f32>,
    pub label: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NeuralEnactiveMemoryV2 {
    pub timestamp: f64,
    pub emotion_state: f32,
    pub arousal_state: f32,
    pub visual_pattern: Vec<f32>,
    pub auditory_pattern: Vec<f32>,
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
    pub fn dummy() -> Self {
        unimplemented!("dummy não deve ser usado em produção");
    }
    
    pub async fn new() -> surrealdb::Result<Self> {
        let db = Surreal::new::<RocksDb>("selene_memories.db").await?;
        db.use_ns("selene_project").use_db("brain_v1").await?;
        
        // Criar índices
        let _ = db.query("DEFINE INDEX vis_idx ON TABLE memories FIELDS visual_pattern MTREE(1024);").await;
        let _ = db.query("DEFINE INDEX aud_idx ON TABLE memories FIELDS auditory_pattern MTREE(1024);").await;
        let _ = db.query("DEFINE INDEX conexoes_destino ON TABLE conexoes FIELDS para_regiao, para_indice;").await;
        
        Ok(Self {
            db,
            conexoes_ativas: HashMap::new(),
        })
    }
    
    // ========== MÉTODOS PRINCIPAIS ==========
    
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
    
    /// Busca memórias por limiar emocional
    pub async fn find_memories_by_emotion(&self, emotion_threshold: f32) -> Vec<NeuralEnactiveMemoryV2> {
        let response = self.db
            .query("SELECT * FROM memories WHERE emotion_state > $th ORDER BY timestamp DESC LIMIT 5")
            .bind(("th", emotion_threshold))
            .await;
    
        match response {
            Ok(mut resp) => resp.take(0).unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }
    
    /// Busca memória por similaridade auditiva
    pub async fn find_similar_memory(&self, current_audio: Vec<f32>) -> Option<NeuralEnactiveMemory> {
        let response = self.db
            .query("SELECT * FROM memories ORDER BY vector::distance::cosine(auditory_pattern, $audio) ASC LIMIT 1")
            .bind(("audio", current_audio))
            .await;
        
        match response {
            Ok(mut resp) => {
                let memories: Vec<NeuralEnactiveMemory> = resp.take(0).unwrap_or_default();
                memories.into_iter().next()
            },
            Err(_) => None,
        }
    }
    
    /// Busca memória por contexto visual
    pub async fn recall_full_context(&self, current_visual: Vec<f32>) -> Option<NeuralEnactiveMemory> {
        let response = self.db
            .query("SELECT * FROM memories ORDER BY vector::distance::cosine(visual_pattern, $val) ASC LIMIT 1")
            .bind(("val", current_visual))
            .await;
        
        match response {
            Ok(mut resp) => {
                let memories: Vec<NeuralEnactiveMemory> = resp.take(0).unwrap_or_default();
                memories.into_iter().next()
            },
            Err(_) => None,
        }
    }
    
    /// Busca multimodal (visão + áudio)
    pub async fn recall_multimodal(&self, vision: Vec<f32>, audio: Vec<f32>) -> Option<NeuralEnactiveMemory> {
        let response = self.db
            .query("SELECT *, (vector::distance::cosine(visual_pattern, $vis) + vector::distance::cosine(auditory_pattern, $aud)) as total_dist 
                    FROM memories ORDER BY total_dist ASC LIMIT 1")
            .bind(("vis", vision))
            .bind(("aud", audio))
            .await;
        
        match response {
            Ok(mut resp) => {
                let memories: Vec<NeuralEnactiveMemory> = resp.take(0).unwrap_or_default();
                memories.into_iter().next()
            },
            Err(_) => None,
        }
    }
}

// ========== CONVERSÕES ==========

impl From<NeuralEnactiveMemory> for NeuralEnactiveMemoryV2 {
    fn from(original: NeuralEnactiveMemory) -> Self {
        Self {
            timestamp: original.timestamp,
            emotion_state: original.emotion_state,
            arousal_state: original.arousal_state,
            visual_pattern: original.visual_pattern,
            auditory_pattern: original.auditory_pattern,
            frontal_intent: original.frontal_intent,
            label: original.label,
            conexoes: Vec::new(),
        }
    }
}