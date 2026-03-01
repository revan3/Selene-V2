// src/storage/memory_graph.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;
use crate::brain_zones::RegionType;

// ================== FUNÇÕES AUXILIARES ==================

fn current_time() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

fn similaridade(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}

// ================== ESTRUTURAS DE DADOS ==================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NeuronParams {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
}

impl Default for NeuronParams {
    fn default() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
        }
    }
}

/// Um NEURÔNIO DIGITAL - a informação base
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NeuronioDigital {
    pub id: Uuid,
    pub regiao: RegionType,
    pub indice: usize,
    pub parametros: NeuronParams,
    pub historico_ativacoes: Vec<AtivacaoHistorica>,
    pub ultima_ativacao: Option<f64>,
    pub importancia_acumulada: f32,
}

/// Uma ativação específica na história deste neurônio
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AtivacaoHistorica {
    pub timestamp: f64,
    pub contexto: Uuid,
    pub intensidade: f32,
    pub emocao_associada: f32,
    pub outros_neurons_ativos: Vec<Uuid>,
}

impl AtivacaoHistorica {
    pub fn contexto_repr(&self) -> Vec<f32> {
        // Implementação simplificada - converte o contexto em um vetor de features
        vec![self.intensidade, self.emocao_associada]
    }
}

/// Uma CONEXÃO entre neurônios - a memória em si
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConexaoSinaptica {
    pub id: Uuid,
    pub de_neuronio: Uuid,
    pub para_neuronio: Uuid,
    pub peso: f32,
    pub criada_em: f64,
    pub ultimo_uso: Option<f64>,
    pub total_usos: u32,
    pub emocao_media: f32,
    pub contexto_criacao: Option<Uuid>,
}

/// O GRAFO COMPLETO - todas as conexões que já existiram
#[derive(Serialize, Deserialize, Debug)]
pub struct GrafoNeuralCompleto {
    pub neuronios: HashMap<Uuid, NeuronioDigital>,
    pub conexoes: HashMap<Uuid, ConexaoSinaptica>,
    pub conexoes_por_origem: HashMap<Uuid, Vec<Uuid>>,
    pub conexoes_por_destino: HashMap<Uuid, Vec<Uuid>>,
}

impl GrafoNeuralCompleto {
    pub fn new() -> Self {
        Self {
            neuronios: HashMap::new(),
            conexoes: HashMap::new(),
            conexoes_por_origem: HashMap::new(),
            conexoes_por_destino: HashMap::new(),
        }
    }
}

/// GERENCIADOR DE MEMÓRIA POR CAMADAS
pub struct MemoryTierV2 {
    pub conexoes_ativas: HashMap<Uuid, ConexaoSinaptica>,
    pub conexoes_dormentes: HashMap<Uuid, ConexaoSinaptica>,
    pub grafo_completo: Arc<GrafoNeuralCompleto>,
    cache_importancia: HashMap<Uuid, f32>,
}

impl MemoryTierV2 {
    pub fn new() -> Self {
        Self {
            conexoes_ativas: HashMap::new(),
            conexoes_dormentes: HashMap::new(),
            grafo_completo: Arc::new(GrafoNeuralCompleto::new()),
            cache_importancia: HashMap::new(),
        }
    }
    
    pub async fn criar_conexao(&mut self, conexao: ConexaoSinaptica) {
        // 1. Salva NO GRAFO COMPLETO (nunca será apagada)
        Arc::get_mut(&mut self.grafo_completo).unwrap().conexoes.insert(conexao.id, conexao.clone());
        
        // 2. Se for importante o suficiente, vai para ativas
        if conexao.emocao_media > 0.6 {
            self.conexoes_ativas.insert(conexao.id, conexao);
        } else {
            self.conexoes_dormentes.insert(conexao.id, conexao);
        }
    }
    
    pub async fn ativar_conexao(&mut self, id_conexao: Uuid) -> Option<&ConexaoSinaptica> {
        if let Some(conexao) = self.conexoes_dormentes.remove(&id_conexao) {
            let mut conexao = conexao;
            conexao.ultimo_uso = Some(current_time());
            conexao.total_usos += 1;
            
            self.conexoes_ativas.insert(id_conexao, conexao);
            return self.conexoes_ativas.get(&id_conexao);
        }
        None
    }
    
    pub async fn ciclo_rem(&mut self) {
        println!("💤 Iniciando ciclo REM...");
        
        // 1. PEGA conexões "esquecidas" (não usadas há muito tempo)
        let grafo = self.grafo_completo.clone();
        let esquecidas: Vec<_> = grafo.conexoes
            .values()
            .filter(|c| {
                c.ultimo_uso.map(|u| u < current_time() - 86400.0).unwrap_or(false)
            })
            .cloned()
            .collect();
        
        // 2. RECOMBINAÇÃO: Cria NOVAS conexões a partir de antigas
        for par in esquecidas.chunks(2) {
            if par.len() == 2 {
                let nova_conexao = ConexaoSinaptica {
                    id: Uuid::new_v4(),
                    de_neuronio: par[0].de_neuronio,
                    para_neuronio: par[1].para_neuronio,
                    peso: (par[0].peso + par[1].peso) / 2.0,
                    criada_em: current_time(),
                    ultimo_uso: None,
                    total_usos: 0,
                    emocao_media: (par[0].emocao_media + par[1].emocao_media) / 2.0,
                    contexto_criacao: None,
                };
                
                Arc::get_mut(&mut self.grafo_completo).unwrap().conexoes.insert(nova_conexao.id, nova_conexao);
                println!("   ✨ Nova conexão criada no REM!");
            }
        }
        
        // 3. REFORÇO: Fortalece conexões relacionadas a emoções fortes
        for conexao in Arc::get_mut(&mut self.grafo_completo).unwrap().conexoes.values_mut() {
            if conexao.emocao_media > 0.8 {
                conexao.peso = (conexao.peso + 0.01).min(1.0);
            }
        }
        
        println!("💤 Ciclo REM concluído!");
    }
    
    pub async fn buscar_por_contexto(&self, contexto: &[f32], regiao: RegionType) -> Vec<Uuid> {
        self.grafo_completo.neuronios
            .values()
            .filter(|n| n.regiao == regiao)
            .filter(|n| {
                n.historico_ativacoes.iter().any(|h| {
                    similaridade(contexto, &h.contexto_repr()) > 0.7
                })
            })
            .map(|n| n.id)
            .collect()
    }
}