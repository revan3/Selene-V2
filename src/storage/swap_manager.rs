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

// ── CAP DINÂMICO POR TIER DE RAM (docx v2.3 §03) ──────────────────────────
/// Calcula o cap de neurônios ativos baseado no tier de hardware.
/// Retorna (cap_neuronios, modo_liberado_str).
pub fn calcular_cap(ram_total_gb: f64, ram_livre_gb: f64) -> (usize, &'static str) {
    if ram_total_gb >= 20.0 && ram_livre_gb >= 6.0 {
        (500_000, "Todos + Turbo")
    } else if ram_total_gb >= 16.0 && ram_livre_gb >= 4.0 {
        (200_000, "Economia + Normal")
    } else if ram_total_gb >= 8.0 && ram_livre_gb >= 2.0 {
        (50_000, "Só Economia")
    } else {
        (10_000, "Só Economia (RAM crítica)")
    }
}

// ── Constantes para aprendizado semântico ─────────────────────────────────
const LTP_CONCEITO:   f32 = 0.018;  // taxa de potenciação inter-conceito
const PESO_MAX_CONCEITO: f32 = 2.5; // peso máximo de sinapse conceitual
const I_BASE_CONCEITO: f32 = 12.0;  // corrente base para spike RS (pA)

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
    pub next_id: u32,

    // Contadores (para interface)
    pub total_neurons_criados: usize,
    pub total_neurogenese_eventos: usize,

    // ── Aprendizado semântico biológico ──────────────────────────────────
    /// Mapa palavra → UUID do neurônio conceitual correspondente
    pub palavra_para_id: HashMap<String, Uuid>,
    /// Correntes de excitação pendentes por neurônio (decaem a cada tick)
    pub correntes: HashMap<Uuid, f32>,
    /// Sinapses associativas entre conceitos: (pre, pos) → peso
    pub sinapses_conceito: HashMap<(Uuid, Uuid), f32>,
    /// Último neurônio aprendido (para STDP sequencial)
    pub ultimo_conceito_id: Option<Uuid>,
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
            next_id: 0,
            total_neurons_criados: 0,
            total_neurogenese_eventos: 0,
            palavra_para_id: HashMap::new(),
            correntes: HashMap::new(),
            sinapses_conceito: HashMap::new(),
            ultimo_conceito_id: None,
        }
    }
    
    /// Adiciona um neurônio existente ao sistema.
    /// Retorna `false` se o cap biológico foi atingido (neurônio descartado).
    pub async fn adicionar_neuronio(&mut self, neuronio: NeuronioHibrido, contexto: &str) -> bool {
        let total = self.ram.len() + self.ssd.len();
        if total >= LIMITE_BIOLOGICO {
            println!("⚠️ Cap biológico atingido ({} neurônios). Neurônio descartado.", LIMITE_BIOLOGICO);
            return false;
        }

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
            // Se RAM cheia, vai direto para SSD (dormente)
            self.ssd.insert(id, neuronio);
        }

        self.total_neurons_criados += 1;
        true
    }

    /// Coloca `excesso` neurônios ativos (RAM) em estado dormente (SSD),
    /// preservando pesos, conexões e histórico STDP.
    /// Seleciona os menos recentemente usados (LRU).
    pub fn dormir_excesso(&mut self, excesso: usize) {
        if excesso == 0 { return; }

        // Coleta candidatos da RAM ordenados por último acesso (mais antigos primeiro)
        let mut candidatos: Vec<(Uuid, f64)> = self.ultimo_acesso
            .iter()
            .filter(|(id, _)| self.ram.contains_key(id))
            .map(|(id, t)| (*id, *t))
            .collect();
        candidatos.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let n = excesso.min(candidatos.len());
        for (id, _) in candidatos.into_iter().take(n) {
            if let Some(neuronio) = self.ram.remove(&id) {
                self.ultimo_acesso.remove(&id);
                self.ssd.insert(id, neuronio);
            }
        }
        if n > 0 {
            println!("💤 {} neurônios enviados para estado dormente (cap RAM atingido).", n);
        }
    }

    /// Verifica o cap de RAM e dorme o excesso se necessário.
    /// Deve ser chamado dentro do tick loop.
    /// `ram_total_gb` e `ram_livre_gb`: valores atuais do sistema.
    pub fn verificar_cap_ram(&mut self, ram_total_gb: f64, ram_livre_gb: f64) {
        let (cap, _modo) = calcular_cap(ram_total_gb, ram_livre_gb);
        let ativos = self.ram.len();
        if ativos > cap {
            self.dormir_excesso(ativos - cap);
        }
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

    // ═══════════════════════════════════════════════════════════════════
    // APRENDIZADO SEMÂNTICO BIOLÓGICO
    // ═══════════════════════════════════════════════════════════════════

    /// Cria (ou recupera) o neurônio conceitual para uma palavra e injeta
    /// corrente proporcional à valência.
    pub fn aprender_conceito(&mut self, palavra: &str, valence: f32) -> Uuid {
        let chave = palavra.to_lowercase();

        let id = if let Some(&existing) = self.palavra_para_id.get(&chave) {
            self.ultimo_acesso.insert(existing, current_time());
            *self.frequencia_acesso.entry(existing).or_insert(0) += 1;
            existing
        } else {
            let nid = self.next_id;
            self.next_id = self.next_id.wrapping_add(1);
            let neuronio = NeuronioHibrido::new(nid, TipoNeuronal::RS, PrecisionType::FP32);
            let uuid = Uuid::new_v4();
            self.ram.insert(uuid, neuronio);
            self.ultimo_acesso.insert(uuid, current_time());
            self.frequencia_acesso.insert(uuid, 1);
            self.indices.entry("conceito".to_string())
                .or_insert_with(Vec::new).push(uuid);
            self.palavra_para_id.insert(chave, uuid);
            self.total_neurons_criados += 1;
            uuid
        };

        let corrente = valence.abs() * I_BASE_CONCEITO + 4.0;
        let entry = self.correntes.entry(id).or_insert(0.0);
        *entry = entry.max(corrente);

        // STDP associativo sequencial: pre → post LTP
        if let Some(pre_id) = self.ultimo_conceito_id {
            if pre_id != id {
                let peso = self.sinapses_conceito.entry((pre_id, id)).or_insert(0.0);
                *peso = (*peso + LTP_CONCEITO).clamp(0.0, PESO_MAX_CONCEITO);
            }
        }
        self.ultimo_conceito_id = Some(id);
        id
    }

    /// Executa um tick nos neurônios conceituais (chamado a cada tick do loop principal).
    pub fn tick_semantico(&mut self, dt_s: f32, current_time_ms: f32) {
        let ids: Vec<Uuid> = self.palavra_para_id.values().copied().collect();
        let mut spikes: Vec<Uuid> = Vec::new();

        for &id in &ids {
            if let Some(n) = self.ram.get_mut(&id) {
                let i_ext = self.correntes.get(&id).copied().unwrap_or(0.0);
                if n.update(i_ext, dt_s, current_time_ms, 1.0) {
                    spikes.push(id);
                }
            }
        }

        // STDP inter-conceito: post dispara → fortalece sinapses com pré recente
        for &post_id in &spikes {
            for &pre_id in &ids {
                if pre_id == post_id { continue; }
                let pre_trace = self.ram.get(&pre_id).map(|n| n.trace_pos).unwrap_or(0.0);
                if pre_trace > 0.15 {
                    let peso = self.sinapses_conceito.entry((pre_id, post_id)).or_insert(0.0);
                    *peso = (*peso + LTP_CONCEITO * pre_trace).clamp(0.0, PESO_MAX_CONCEITO);
                }
            }
        }

        // Propaga spikes pelas sinapses semânticas
        let sinapses_snap: Vec<((Uuid, Uuid), f32)> = self.sinapses_conceito
            .iter().map(|(&k, &v)| (k, v)).collect();
        for ((pre_id, post_id), peso) in sinapses_snap {
            if spikes.contains(&pre_id) {
                let entry = self.correntes.entry(post_id).or_insert(0.0);
                *entry += peso * 6.0;
            }
        }

        // Decai correntes (meia-vida ~8 ticks a 200Hz)
        for v in self.correntes.values_mut() {
            *v *= 0.88;
            if *v < 0.3 { *v = 0.0; }
        }
    }

    /// Executa N ciclos de treino STDP (consolidação). Retorna (spikes, avg_delta_peso, n_sinapses).
    pub fn treinar_semantico(
        &mut self,
        n_ciclos: u32,
        dt_s: f32,
        valencias: &std::collections::HashMap<String, f32>,
    ) -> (u32, f32, usize) {
        let mut total_spikes: u32 = 0;
        let mut total_delta: f32 = 0.0;
        let mut n_amostras: u32 = 0;
        let mut t_ms: f32 = 0.0;

        for (palavra, &id) in &self.palavra_para_id {
            if let Some(&val) = valencias.get(palavra.as_str()) {
                self.correntes.insert(id, (val.abs() * I_BASE_CONCEITO + 4.0).min(20.0));
            }
        }

        let ids: Vec<Uuid> = self.palavra_para_id.values().copied().collect();

        for ciclo in 0..n_ciclos {
            if ciclo % 50 == 0 {
                for (palavra, &id) in &self.palavra_para_id {
                    if let Some(&val) = valencias.get(palavra.as_str()) {
                        let entry = self.correntes.entry(id).or_insert(0.0);
                        *entry = entry.max((val.abs() * 8.0 + 2.0).min(18.0));
                    }
                }
            }

            let mut spikes_ciclo: Vec<Uuid> = Vec::new();
            for &id in &ids {
                if let Some(n) = self.ram.get_mut(&id) {
                    let peso_antes = if let crate::synaptic_core::PesoNeuronio::FP32(v) = n.peso { v } else { 1.0 };
                    let i_ext = self.correntes.get(&id).copied().unwrap_or(0.0);
                    if n.update(i_ext, dt_s, t_ms, 1.0) {
                        total_spikes += 1;
                        spikes_ciclo.push(id);
                        let peso_depois = if let crate::synaptic_core::PesoNeuronio::FP32(v) = n.peso { v } else { 1.0 };
                        total_delta += (peso_depois - peso_antes).abs();
                        n_amostras += 1;
                    }
                }
            }

            for &post_id in &spikes_ciclo {
                for &pre_id in &ids {
                    if pre_id == post_id { continue; }
                    let pre_trace = self.ram.get(&pre_id).map(|n| n.trace_pos).unwrap_or(0.0);
                    if pre_trace > 0.15 {
                        let peso = self.sinapses_conceito.entry((pre_id, post_id)).or_insert(0.0);
                        *peso = (*peso + LTP_CONCEITO * pre_trace).clamp(0.0, PESO_MAX_CONCEITO);
                    }
                }
            }

            for v in self.correntes.values_mut() {
                *v *= 0.88;
                if *v < 0.3 { *v = 0.0; }
            }
            t_ms += dt_s * 1000.0;
        }

        let avg_delta = if n_amostras > 0 { total_delta / n_amostras as f32 } else { 0.0 };
        (total_spikes, avg_delta, self.sinapses_conceito.len())
    }

    /// Número de sinapses semânticas com peso > 0.01.
    pub fn sinapses_semanticas_ativas(&self) -> usize {
        self.sinapses_conceito.values().filter(|&&p| p > 0.01).count()
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

impl SwapManager {
    // --- NOVA FUNÇÃO PARA MEMÓRIA INFINITA ---
    pub fn arquivar_para_hdd(&mut self, id: &Uuid) -> std::io::Result<()> {
        if let Some(neuronio) = self.ssd.remove(id) {
            let path_hdd = format!("D:/Selene_Archive/{}.json", id);
            let dados = serde_json::to_string(&neuronio).unwrap();

            // Grava o neurônio no HDD (D:) e libera espaço no NVMe/RAM
            std::fs::write(path_hdd, dados)?;
            println!("📦 [Cold Storage] Neurônio {} movido para o HDD (D:)", id);
        }
        Ok(())
    }
}
