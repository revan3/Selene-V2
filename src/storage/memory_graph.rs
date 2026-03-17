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

/// Contexto semântico de uma conexão — determina regras de poda
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ContextoSemantico {
    Realidade,   // conexão validada com o mundo real
    Fantasia,    // conexão válida apenas em contexto imaginativo
    Sonho,       // gerada pelo ciclo REM — hipótese criativa
    Hipotese,    // aguardando validação contextual
    Habito,      // padrão repetido consolidado
}

impl Default for ContextoSemantico {
    fn default() -> Self { ContextoSemantico::Hipotese }
}

/// Uma CONEXÃO entre neurônios - a memória em si
///
/// REGRA DE PODA:
///   peso >= 0.0  → conexão permanece no mapa sináptico
///   peso <  0.0  → SINAPSE apagada (neurônios permanecem intactos)
///
/// O contexto determina se a poda é aplicada:
///   - Realidade: poda agressiva (conexões absurdas são removidas)
///   - Fantasia/Sonho: poda suave (conexões "impossíveis" são permitidas)
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
    /// Contexto semântico — define regras de poda e validade
    pub contexto_semantico: ContextoSemantico,
    /// Acumulador de invalidações contextuais (decrementado quando a conexão
    /// é contestada; quando atinge < 0.0 a sinapse é podada no N2)
    pub marcador_poda: f32,
}

impl ConexaoSinaptica {
    /// Retorna true se esta sinapse deve ser podada
    pub fn deve_podar(&self) -> bool {
        self.peso < 0.0 || self.marcador_poda < 0.0
    }

    /// Penaliza a conexão em contexto de realidade.
    /// Se `contexto_semantico` for Fantasia ou Sonho, a penalidade é amortecida.
    pub fn penalizar_contexto_real(&mut self, magnitude: f32) {
        match self.contexto_semantico {
            ContextoSemantico::Realidade | ContextoSemantico::Hipotese => {
                self.marcador_poda -= magnitude;
                self.peso = (self.peso - magnitude * 0.1).max(-1.0);
            }
            ContextoSemantico::Fantasia | ContextoSemantico::Sonho => {
                // Conexões de fantasia/sonho recebem apenas 10% da penalidade
                self.marcador_poda -= magnitude * 0.1;
            }
            ContextoSemantico::Habito => {
                // Hábitos são resistentes à poda
                self.marcador_poda -= magnitude * 0.05;
            }
        }
    }

    /// Reforça a conexão (usado no REM e na consolidação N1)
    pub fn reforcar(&mut self, magnitude: f32) {
        self.peso = (self.peso + magnitude).min(2.5);
        self.marcador_poda = (self.marcador_poda + magnitude * 0.5).min(1.0);
    }
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
    
    /// Ciclo REM: recombina conexões esquecidas e retorna as novas conexões
    /// criadas para que o sleep cycle as persista no DB.
    pub async fn ciclo_rem(&mut self) -> Vec<ConexaoSinaptica> {
        println!("💤 Iniciando ciclo REM...");

        // 1. Pega conexões "esquecidas" (não usadas há mais de 24h)
        let grafo = self.grafo_completo.clone();
        let esquecidas: Vec<_> = grafo.conexoes
            .values()
            .filter(|c| {
                c.ultimo_uso.map(|u| u < current_time() - 86400.0).unwrap_or(false)
            })
            .cloned()
            .collect();

        // 2. Recombinação: cria novas conexões a partir de pares de esquecidas
        let mut novas: Vec<ConexaoSinaptica> = Vec::new();
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
                    contexto_semantico: ContextoSemantico::Sonho,
                    marcador_poda: 0.5,
                };
                Arc::get_mut(&mut self.grafo_completo).unwrap()
                    .conexoes.insert(nova_conexao.id, nova_conexao.clone());
                novas.push(nova_conexao);
            }
        }

        // 3. Reforço: fortalece conexões emocionalmente salientes
        for conexao in Arc::get_mut(&mut self.grafo_completo).unwrap().conexoes.values_mut() {
            if conexao.emocao_media > 0.8 {
                conexao.peso = (conexao.peso + 0.01).min(1.0);
            }
        }

        println!("💤 Ciclo REM concluído! {} nova(s) conexão(ões) criada(s)", novas.len());
        novas
    }

    /// FASE N2 — Poda sináptica contextual
    ///
    /// Remove SINAPSES (não neurônios) cujo marcador_poda < 0 ou peso < 0.
    /// Conexões de Fantasia/Sonho têm limiar mais alto para sobreviver.
    /// Retorna os IDs das sinapses podadas (para o sleep cycle deletar no DB).
    pub fn podar_sinapses(&mut self) -> Vec<Uuid> {
        let grafo = Arc::get_mut(&mut self.grafo_completo).expect("grafo em uso");

        // Coleta IDs a podar (sem modificar o HashMap durante iteração)
        let a_podar: Vec<Uuid> = grafo.conexoes
            .values()
            .filter(|c| {
                if c.deve_podar() { return true; }
                // Hipóteses nunca usadas e antigas (>7 dias) são removidas
                if c.contexto_semantico == ContextoSemantico::Hipotese
                    && c.total_usos == 0
                    && c.ultimo_uso.map(|u| current_time() - u > 604_800.0).unwrap_or(true)
                {
                    return true;
                }
                false
            })
            .map(|c| c.id)
            .collect();

        for id in &a_podar {
            grafo.conexoes.remove(id);
            for idx in grafo.conexoes_por_origem.values_mut() {
                idx.retain(|x| x != id);
            }
            for idx in grafo.conexoes_por_destino.values_mut() {
                idx.retain(|x| x != id);
            }
            self.conexoes_ativas.remove(id);
            self.conexoes_dormentes.remove(id);
        }

        if !a_podar.is_empty() {
            println!("✂️  [N2] {} sinapses podadas (neurônios intactos)", a_podar.len());
        }
        a_podar
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