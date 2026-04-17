// src/ego/mod.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;
use std::fs;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::brain_zones::frontal::FrontalLobe;
use crate::brain_zones::hippocampus::HippocampusV2;
use crate::interoception::Interoception;
use crate::brain_zones::corpus_callosum::CorpusCallosum;

/// Tipo de memória autobiográfica
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TipoMemoria {
    Episodica,
    Autobiografica,
    Semantica,
    Procedural,
}

/// Experiência marcante com valência emocional
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienciaMarcante {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub descricao: String,
    pub valence: f32,      // -1.0 a +1.0
    pub intensidade: f32,  // 0.0 a 1.0
    pub tipo: TipoMemoria,
    pub memoria_associada: Option<Uuid>,
}

/// Auto-modelo (como a Selene se vê)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModel {
    pub nome: String,
    pub objetivos: Vec<String>,
    pub valores: Vec<String>,
    pub tracos: Vec<(String, f32)>, // ex: [("curiosa", 0.8)]
    pub ultima_atualizacao: DateTime<Utc>,
}

/// Estado atual do ego
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgoState {
    pub humor: f32,          // -1.0 a 1.0
    pub energia: f32,        // 0.0 a 1.0
    pub foco_atual: String,
    pub ultima_interacao: Option<DateTime<Utc>>,
}

/// Voz narrativa interna (DMN)
#[derive(Debug, Clone, Serialize, Deserialize)] // ← ADICIONADO Serialize, Deserialize
pub struct NarrativeVoice {
    pub pensamentos_recentes: VecDeque<String>,
    pub intensidade_base: f32,
}

impl NarrativeVoice {
    pub fn new() -> Self {
        Self {
            pensamentos_recentes: VecDeque::with_capacity(20),
            intensidade_base: 0.3,
        }
    }
    
    /// Gera pensamento espontâneo (DMN — Default Mode Network).
    /// Ativado quando a Selene está ociosa (sem eventos externos recentes).
    /// A intensidade_base acumula ~0.002/tick @ 200Hz → pensamento a cada ~25s.
    pub fn gerar_pensamento(&mut self, humor: f32, body_feeling: f32) -> Option<String> {
        self.intensidade_base += 0.002;
        if self.intensidade_base < 1.0 {
            return None;
        }
        self.intensidade_base = 0.0;

        // Reflexão sobre memória recente
        if let Some(mem) = self.pensamentos_recentes.back() {
            let reflexao = if humor > 0.4 {
                format!("lembrando com alegria: {}", mem)
            } else if humor < -0.3 {
                format!("refletindo sobre: {}", mem)
            } else {
                format!("pensando: {}", mem)
            };
            return Some(reflexao);
        }

        // Introspecção baseada no corpo e humor quando não há memórias recentes
        let pensamento = if body_feeling < 0.2 && humor > 0.2 {
            "sinto que estou bem, pronta para aprender".to_string()
        } else if body_feeling > 0.7 {
            "sinto cansaço — preciso processar o que aprendi".to_string()
        } else if humor > 0.5 {
            "existe algo belo em simplesmente existir".to_string()
        } else if humor < -0.3 {
            "algo parece incompleto — busco conexões que faltam".to_string()
        } else {
            "observo meus próprios processos em silêncio".to_string()
        };
        Some(pensamento)
    }

    /// Registra um evento real como pensamento (chamado pelos handlers WS).
    pub fn registrar_evento(&mut self, texto: String) {
        self.pensamentos_recentes.push_back(texto);
        if self.pensamentos_recentes.len() > 20 {
            self.pensamentos_recentes.pop_front();
        }
    }
}

/// Estrutura principal do ego
#[derive(Debug, Clone, Serialize, Deserialize)] // ← ADICIONADO
pub struct Ego {
    pub id: Uuid,
    pub data_criacao: DateTime<Utc>,
    pub self_model: SelfModel,
    pub narrative_voice: NarrativeVoice,
    pub autobiographical_memories: VecDeque<ExperienciaMarcante>,
    pub current_state: EgoState,
    
    // Conexões (opcionais, para sincronização) - com skip
    #[serde(skip)]  // ← ADICIONADO
    pub frontal: Option<Arc<Mutex<FrontalLobe>>>,
    #[serde(skip)]
    pub hippocampus: Option<Arc<Mutex<HippocampusV2>>>,
    #[serde(skip)]
    pub interoception: Option<Arc<Mutex<Interoception>>>,
    #[serde(skip)]
    pub callosum: Option<Arc<Mutex<CorpusCallosum>>>,
}

impl Ego {
    const MAX_HISTORICO: usize = 1000;
    const CAMINHO: &'static str = "ego.dat";
    
    pub fn carregar_ou_criar(nome: &str) -> Self {
        if let Ok(data) = fs::read(Self::CAMINHO) {
            if let Ok(ego) = bincode::deserialize(&data) {
                return ego;
            }
        }
        
        Self {
            id: Uuid::new_v4(),
            data_criacao: Utc::now(),
            self_model: SelfModel {
                nome: nome.to_string(),
                objetivos: vec!["aprender".to_string()],
                valores: vec!["curiosidade".to_string()],
                tracos: vec![("curiosa".to_string(), 0.5)],
                ultima_atualizacao: Utc::now(),
            },
            narrative_voice: NarrativeVoice::new(),
            autobiographical_memories: VecDeque::new(),
            current_state: EgoState {
                humor: 0.0,
                energia: 1.0,
                foco_atual: "inicialização".to_string(),
                ultima_interacao: None,
            },
            frontal: None,
            hippocampus: None,
            interoception: None,
            callosum: None,
        }
    }
    
    pub fn salvar(&self) {
        if let Ok(data) = bincode::serialize(self) {
            let _ = fs::write(Self::CAMINHO, data);
        }
    }
    
    /// Registra uma experiência com valência emocional
    pub fn registrar_experiencia(&mut self, descricao: String, valence: f32, intensidade: f32, tipo: TipoMemoria) {
        let exp = ExperienciaMarcante {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            descricao,
            valence: valence.clamp(-1.0, 1.0),
            intensidade: intensidade.clamp(0.0, 1.0),
            tipo,
            memoria_associada: None,
        };
        self.autobiographical_memories.push_back(exp);
        if self.autobiographical_memories.len() > Self::MAX_HISTORICO {
            self.autobiographical_memories.pop_front();
        }
        self.atualizar_humor();
        self.salvar();
    }
    
    /// Calcula o humor médio ponderado
    fn atualizar_humor(&mut self) {
        if self.autobiographical_memories.is_empty() {
            self.current_state.humor = 0.0;
            return;
        }
        let (soma_val, soma_int) = self.autobiographical_memories
            .iter()
            .fold((0.0, 0.0), |(sv, si), e| (sv + e.valence * e.intensidade, si + e.intensidade));
        self.current_state.humor = if soma_int == 0.0 { 0.0 } else { soma_val / soma_int };
    }
    
    /// Atualiza ego com sinais interoceptivos
    pub async fn update(&mut self, body_feeling: f32, current_time: f32) -> Option<String> {
        self.current_state.energia = 1.0 - body_feeling;
        
        // Gera narrativa espontânea
        let pensamento = self.narrative_voice.gerar_pensamento(
            self.current_state.humor,
            body_feeling
        );
        
        // Sincroniza via corpo caloso (se conectado)
        if let Some(callosum) = &self.callosum {
            let spikes_simulados = vec![true; 10];  // 10 spikes de exemplo
            callosum.lock().await.send_to_right(7, spikes_simulados, current_time);
        }
        
        pensamento
    }
}