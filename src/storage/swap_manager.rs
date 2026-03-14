// src/storage/swap_manager.rs
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;
// CORREÇÃO E0425 + E0063: importar TipoNeuronal e adicionar next_id
use crate::synaptic_core::{NeuronioHibrido, PrecisionType, TipoNeuronal};
use crate::brain_zones::RegionType;

// Limites
const RAM_PERCENT_FOR_NEURONS: f32 = 0.8;      // 80% da RAM para neurônios ativos
const LIMITE_BIOLOGICO: usize = 946_000_000;   // 1,1% de 86B neurônios humanos
const SYNAPSES_PER_NEURON: usize = 8500;       // Média biológica

pub struct SwapManager {
    // RAM: neurônios ativos
    pub ram: HashMap<Uuid, NeuronioHibrido>,
    
    // SSD/HDD: neurônios dormentes (mapeados por ID)
    pub ssd: HashMap<Uuid, NeuronioHibrido>,
    
    // Índices para busca rápida
    pub indices: HashMap<String, Vec<Uuid>>,  // contexto -> neurônios
    
    // Estatísticas de acesso
    pub ultimo_acesso: HashMap<Uuid, f64>,
    pub frequencia_acesso: HashMap<Uuid, u32>,
    
    // Limites
    pub max_ram_neurons: usize,
    pub swap_threshold_seconds: u64,

    // CORREÇÃO E0063: campo next_id estava faltando no struct
    // Usado como ID numérico incremental para NeuronioHibrido::new(id, tipo, precisao)
    pub next_id: u32,
    
    // Contadores (para interface)
    pub total_neurons_criados: usize,
    pub total_neurogenese_eventos: usize,
}

impl SwapManager {
    pub fn ram_count(&self) -> usize { self.ram.len() }
    pub fn total_count(&self) -> usize { self.ram.len() + self.ssd.len() }
    pub fn synapses_ativas(&self) -> usize { 
        self.ram.len() * SYNAPSES_PER_NEURON 
    }

    pub fn new(max_ram_neurons: usize, swap_threshold_seconds: u64) -> Self {
        Self {
            ram: HashMap::with_capacity(max_ram_neurons),
            ssd: HashMap::new(),
            indices: HashMap::new(),
            ultimo_acesso: HashMap::new(),
            frequencia_acesso: HashMap::new(),
            max_ram_neurons,
            swap_threshold_seconds,
            // CORREÇÃO E0063: inicializar next_id
            next_id: 0,
            total_neurons_criados: 0,
            total_neurogenese_eventos: 0,
        }
    }
    
    /// Adiciona um neurônio existente ao sistema
    pub async fn adicionar_neuronio(&mut self, neuronio: NeuronioHibrido, contexto: &str) {
        let id = Uuid::new_v4();
        
        // Adiciona aos índices
        self.indices.entry(contexto.to_string())
            .or_insert_with(Vec::new)
            .push(id);
        
        // Tenta colocar na RAM primeiro
        if self.ram.len() < self.max_ram_neurons {
            self.ultimo_acesso.insert(id, current_time());
            self.ram.insert(id, neuronio);
        } else {
            // Se RAM cheia, vai direto para SSD
            self.ssd.insert(id, neuronio);
        }
        
        self.total_neurons_criados += 1;
    }
    
    /// CRIA um novo neurônio sob demanda (neurogênese)
    pub async fn criar_neuronio(
        &mut self,
        regiao: RegionType,
        precisao: PrecisionType,
        contexto: &str,
    ) -> Option<Uuid> {
        // Verifica limite físico: 80% da RAM disponível para neurônios ativos
        let max_ativos = (self.max_ram_neurons as f32 * RAM_PERCENT_FOR_NEURONS) as usize;
        if self.ram.len() >= max_ativos {
            println!("⚠️ Limite físico de RAM atingido ({} neurônios ativos).", self.ram.len());
            return None;
        }

        // Verifica limite biológico (1,1% do cérebro humano)
        let total_neurons = self.ram.len() + self.ssd.len();
        if total_neurons >= LIMITE_BIOLOGICO {
            println!("⚠️ Limite biológico de neurônios atingido ({}).", LIMITE_BIOLOGICO);
            return None;
        }

        // CORREÇÃO E0425: NeuronioHibrido::new exige (id: u32, tipo: TipoNeuronal, precisao: PrecisionType)
        // O campo .id não existia na versão anterior — usamos next_id incremental como u32
        let nid = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);

        // Seleciona o tipo neuronal com base na região
        let tipo = tipo_para_regiao(&regiao);

        let neuronio = NeuronioHibrido::new(nid, tipo, precisao);

        // UUID para indexação no HashMap (independente do id numérico interno)
        let id = Uuid::new_v4();

        // Adiciona aos índices
        self.indices.entry(contexto.to_string())
            .or_insert_with(Vec::new)
            .push(id);

        // Decide se coloca na RAM ou SSD
        if self.ram.len() < max_ativos {
            self.ultimo_acesso.insert(id, current_time());
            self.ram.insert(id, neuronio);
            println!("🧠 Novo neurônio criado e ativado na RAM (região {:?})", regiao);
        } else {
            self.ssd.insert(id, neuronio);
            println!("🧠 Novo neurônio criado no SSD (região {:?})", regiao);
        }

        self.total_neurons_criados += 1;
        self.total_neurogenese_eventos += 1;
        Some(id)
    }
    
    /// Ativa neurônios por contexto (busca na RAM ou faz swap do SSD)
    pub async fn ativar_por_contexto(&mut self, contexto: &str) -> Vec<Uuid> {
        let mut ativados = Vec::new();
    
        if let Some(ids) = self.indices.get(contexto).cloned() {
            for id in ids {
                // Verifica se está na RAM
                if self.ram.contains_key(&id) {
                    self.ultimo_acesso.insert(id, current_time());
                    *self.frequencia_acesso.entry(id).or_insert(0) += 1;
                    ativados.push(id);
                } 
                // Se não está na RAM, busca do SSD
                else if let Some(neuronio) = self.ssd.remove(&id) {
                    // Faz swap (remove o mais antigo se necessário)
                    if self.ram.len() >= self.max_ram_neurons {
                        self.fazer_swap_para_ssd().await;
                    }
                
                    // Adiciona à RAM
                    self.ultimo_acesso.insert(id, current_time());
                    self.ram.insert(id, neuronio);
                    ativados.push(id);
                }
            }
        }
    
        ativados
    }
    
    pub fn get_neuronio(&self, id: Uuid) -> Option<&NeuronioHibrido> {
        self.ram.get(&id).or_else(|| self.ssd.get(&id))
    }
    
    /// Swap LRU: move o neurônio menos usado recentemente para SSD
    async fn fazer_swap_para_ssd(&mut self) {
        let mais_antigo = self.ultimo_acesso
            .iter()
            .min_by_key(|(_, &tempo)| tempo as u64)
            .map(|(&id, _)| id);
        
        if let Some(id) = mais_antigo {
            if let Some(neuronio) = self.ram.remove(&id) {
                self.ssd.insert(id, neuronio);
                self.ultimo_acesso.remove(&id);
                println!("🔄 Swap: neurônio {} movido para SSD", id);
            }
        }
    }
    
    /// Move neurônios inativos (acima do threshold) para SSD
    pub async fn limpar_neurônios_inativos(&mut self) {
        let agora = current_time();
        let limite = agora - self.swap_threshold_seconds as f64;
        
        let inativos: Vec<Uuid> = self.ultimo_acesso
            .iter()
            .filter(|(_, &tempo)| tempo < limite)
            .map(|(&id, _)| id)
            .collect();
        
        for id in inativos {
            if let Some(neuronio) = self.ram.remove(&id) {
                self.ssd.insert(id, neuronio);
                self.ultimo_acesso.remove(&id);
                println!("💤 Neurônio {} movido para SSD (inativo)", id);
            }
        }
    }
    
    /// Retorna contagens para a interface
    pub fn get_counts(&self) -> (usize, usize, usize) {
        (self.ram.len(), self.ssd.len(), self.ram.len() + self.ssd.len())
    }
    
    /// Estima o número total de sinapses
    pub fn estimar_sinapses(&self) -> (usize, usize) {
        let ativas = self.ram.len() * SYNAPSES_PER_NEURON;
        let totais = (self.ram.len() + self.ssd.len()) * SYNAPSES_PER_NEURON;
        (ativas, totais)
    }
    
    /// Estatísticas detalhadas
    pub fn estatisticas(&self) -> SwapStats {
        let (ativas, totais) = self.estimar_sinapses();
        SwapStats {
            ram: self.ram.len(),
            ssd: self.ssd.len(),
            total: self.ram.len() + self.ssd.len(),
            synapses_ativas: ativas,
            synapses_totais: totais,
            total_indices: self.indices.len(),
            acesso_medio: self.frequencia_acesso.values().sum::<u32>() as f32 / self.ram.len().max(1) as f32,
            neurogenese_eventos: self.total_neurogenese_eventos,
            limite_fisico: (self.max_ram_neurons as f32 * RAM_PERCENT_FOR_NEURONS) as usize,
            limite_biologico: LIMITE_BIOLOGICO,
        }
    }
}

pub struct SwapStats {
    pub ram: usize,
    pub ssd: usize,
    pub total: usize,
    pub synapses_ativas: usize,
    pub synapses_totais: usize,
    pub total_indices: usize,
    pub acesso_medio: f32,
    pub neurogenese_eventos: usize,
    pub limite_fisico: usize,
    pub limite_biologico: usize,
}

fn current_time() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

/// Seleciona o TipoNeuronal mais adequado para cada região cerebral.
/// RS (Regular Spiking) é o padrão seguro para qualquer região.
fn tipo_para_regiao(regiao: &RegionType) -> TipoNeuronal {
    match regiao {
        RegionType::Limbic    => TipoNeuronal::IB,  // Intrinsic Bursting — resposta emocional
        RegionType::Occipital => TipoNeuronal::CH,  // Chattering — detecção visual rápida
        RegionType::Temporal  => TipoNeuronal::RS,  // Regular Spiking — reconhecimento
        RegionType::Parietal  => TipoNeuronal::RS,  // Regular Spiking — integração espacial
        RegionType::Frontal   => TipoNeuronal::RS,  // Regular Spiking — decisão
        _                     => TipoNeuronal::RS,  // Padrão seguro para demais regiões
    }
}