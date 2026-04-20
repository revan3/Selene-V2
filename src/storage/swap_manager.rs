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

// ── CAMADA ZERO: primitivas fixas (criadas UMA VEZ, nunca substituídas) ───────

/// 35 fonemas do PT-BR — tags geradas via format!("ph:{:?}", Phoneme).to_lowercase()
pub const FONEMAS_PRIMITIVOS: &[&str] = &[
    "ph:a",  "ph:e",  "ph:i",  "ph:o",  "ph:u",
    "ph:an", "ph:en", "ph:in", "ph:on", "ph:un",
    "ph:p",  "ph:b",  "ph:t",  "ph:d",  "ph:k",  "ph:g",
    "ph:f",  "ph:v",  "ph:s",  "ph:z",  "ph:sh", "ph:zh",
    "ph:ch", "ph:dj",
    "ph:m",  "ph:n",  "ph:nh",
    "ph:l",  "ph:lh", "ph:r",  "ph:rr",
    "ph:w",  "ph:y",
    "ph:sil",
];

/// 20 primitivas visuais — bandas espectrais, luminância, bordas, frequência, movimento
pub const VISUAIS_PRIMITIVOS: &[&str] = &[
    // 7 bandas espectrais (fotorreceptores S/M/L)
    "vis:band:violeta", "vis:band:azul",    "vis:band:ciano",
    "vis:band:verde",   "vis:band:amarelo", "vis:band:laranja", "vis:band:vermelho",
    // 5 níveis de luminância (luz → sombra)
    "vis:lum:muito_escuro", "vis:lum:escuro", "vis:lum:medio",
    "vis:lum:claro",        "vis:lum:muito_claro",
    // 4 orientações de borda (formas)
    "vis:borda:0", "vis:borda:45", "vis:borda:90", "vis:borda:135",
    // 3 frequências espaciais (textura)
    "vis:freq:baixa", "vis:freq:media", "vis:freq:alta",
    // 1 detector de movimento
    "vis:movimento:sim",
];

// ── One-shot learning ──────────────────────────────────────────────────────

/// Peso temporário de aprendizado rápido (episódico).
#[derive(Debug, Clone)]
pub struct FastWeight {
    /// Peso inicial alto — decai se não reforçado.
    pub peso: f32,
    /// Timestamp de criação (segundos desde epoch).
    pub t_criacao: f64,
    /// Número de reforços recebidos após o primeiro encontro.
    pub reforcos: u32,
    /// Valência emocional no momento do aprendizado.
    pub valence: f32,
}

// ── Graph versioning ───────────────────────────────────────────────────────

/// Snapshot do estado semântico em um ponto no tempo.
#[derive(Debug, Clone)]
pub struct GraphSnapshot {
    /// Tick do loop principal em que o snapshot foi criado.
    pub tick: u64,
    /// Timestamp (segundos desde epoch).
    pub timestamp: f64,
    /// Snapshot do grafo: palavra → vizinhos com pesos.
    pub grafo: HashMap<String, Vec<(String, f32)>>,
    /// Snapshot das valências.
    pub valencias: HashMap<String, f32>,
    /// Número de sinapses no momento.
    pub n_sinapses: usize,
}

const MAX_SNAPSHOTS: usize = 5;
const FAST_WEIGHT_INICIAL: f32 = 0.75;
const FAST_WEIGHT_DECAY: f32 = 0.98;  // decai ~2% por tick semântico
const FAST_WEIGHT_TTL_S: f64 = 300.0; // 5 minutos antes de descartar sem reforço
const FAST_WEIGHT_CONSOLIDAR_REFORCOS: u32 = 2; // 2 reforços = consolidação permanente
const EMBED_DIM: usize = 32;

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
const LTP_CONCEITO:      f32 = 0.018;  // taxa de potenciação inter-conceito
const PESO_MAX_CONCEITO: f32 = 2.5;   // peso máximo de sinapse conceitual
const I_BASE_CONCEITO:   f32 = 12.0;  // corrente base para spike RS (pA)

/// Número de neurônios por conceito (codificação em população).
/// Biologicamente: ~50–200 neurônios representam um conceito cortical.
/// Aqui usamos 20 — rico o suficiente para gradação e ambivalência,
/// leve o suficiente para rodar em hardware de consumidor.
const POPULACAO_N: usize = 20;

/// Limite máximo de sinapses conceituais em memória (LRU eviction acima disso).
/// Previne crescimento ilimitado de sinapses_conceito — ~160 MB estimado.
const SINAPSES_CAP: usize = 500_000;

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
    /// Mapa palavra → população de POPULACAO_N neurônios conceituais.
    /// Cada conceito é representado por um conjunto distribuído de neurônios
    /// com limiares ligeiramente diferentes — codificação em população.
    pub palavra_para_id: HashMap<String, Vec<Uuid>>,
    /// Valência acumulada por neurônio individual.
    /// Permite que o mesmo conceito tenha valências diferentes em cada
    /// neurônio da população — base para ambivalência emocional genuína.
    pub valencia_neuronio: HashMap<Uuid, f32>,
    /// Correntes de excitação pendentes por neurônio (decaem a cada tick)
    pub correntes: HashMap<Uuid, f32>,
    /// Sinapses associativas entre conceitos: (pre_canonico, post_canonico) → peso.
    /// Opera entre canônicos (pop[0]) para evitar explosão O(N²×vocab²).
    pub sinapses_conceito: HashMap<(Uuid, Uuid), f32>,
    /// Último neurônio aprendido (para STDP sequencial — fonemas/visuais)
    pub ultimo_conceito_id: Option<Uuid>,
    /// Última população conceitual ativada (para STDP entre conceitos)
    pub ultimo_conceito_pop: Option<Vec<Uuid>>,

    // ── Camada 0: primitivas fixas ────────────────────────────────────────
    /// Mapa tag-fonema → UUID (ex: "ph:a" → uuid). Nunca cresce após inicializar.
    pub fonemas_para_id: HashMap<String, Uuid>,
    /// Mapa tag-visual → UUID (ex: "vis:band:azul" → uuid). Nunca cresce após inicializar.
    pub visuais_para_id: HashMap<String, Uuid>,

    // ── One-shot learning ─────────────────────────────────────────────────
    /// Fast-weights: conceitos vistos pela primeira vez recebem peso temporário alto.
    /// Consolidam em sinapses permanentes se reforçados; caso contrário decaem.
    pub fast_weights: HashMap<String, FastWeight>,

    // ── Graph versioning ──────────────────────────────────────────────────
    /// Snapshots periódicos do grafo semântico (máx 5).
    /// Protege contra sessões ruins sobrescreverem bom aprendizado.
    pub snapshots: std::collections::VecDeque<GraphSnapshot>,

    // ── Embeddings vetoriais ──────────────────────────────────────────────
    /// Representação vetorial de 32 dimensões por conceito.
    /// Permite busca por similaridade semântica (cosine similarity).
    pub embeddings: HashMap<String, [f32; 32]>,

    /// Ticks consecutivos sem spike — usado para coasting (pular STDP quando ocioso).
    ticks_sem_spike: u32,

    /// Templates cognitivos — padrões relacionais com slots efêmeros.
    /// A topologia (relações entre slots) persiste; o conteúdo dos slots é efêmero.
    pub template_store: crate::learning::templates::TemplateStore,

    // ── Cache de grafo_palavras ───────────────────────────────────────────
    /// Cache do grafo de palavras — reconstruído apenas quando sinapses mudam.
    grafo_cache: Option<HashMap<String, Vec<(String, f32)>>>,
    /// Sinaliza que sinapses mudaram e o cache precisa ser reconstruído.
    grafo_dirty: bool,
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
            valencia_neuronio: HashMap::new(),
            correntes: HashMap::new(),
            sinapses_conceito: HashMap::new(),
            ultimo_conceito_id: None,
            ultimo_conceito_pop: None,
            fonemas_para_id: HashMap::new(),
            visuais_para_id: HashMap::new(),
            fast_weights: HashMap::new(),
            snapshots: std::collections::VecDeque::with_capacity(MAX_SNAPSHOTS),
            embeddings: HashMap::new(),
            ticks_sem_spike: 0,
            template_store: crate::learning::templates::TemplateStore::novo(),
            grafo_cache: None,
            grafo_dirty: true,
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

    /// Retorna o UUID canônico de um conceito (primeiro da população).
    /// Usado para STDP inter-conceito sem explosão O(N²×vocab²).
    pub fn canonico(&self, palavra: &str) -> Option<Uuid> {
        self.palavra_para_id.get(palavra)?.first().copied()
    }

    /// Cria (ou recupera) a população conceitual para uma palavra e injeta
    /// corrente proporcional à valência em todos os neurônios da população.
    ///
    /// **Codificação em população**: cada conceito é representado por POPULACAO_N
    /// neurônios com limiares ligeiramente diferentes (ruído biológico de inicialização).
    /// A valência é distribuída com pequena variação por neurônio — permitindo que
    /// o mesmo conceito tenha representações emocionais ligeiramente distintas
    /// (base para ambivalência genuína após múltiplas experiências contraditórias).
    pub fn aprender_conceito(&mut self, palavra: &str, valence: f32) -> Vec<Uuid> {
        let chave = palavra.to_lowercase();

        let populacao: Vec<Uuid> = if let Some(pop) = self.palavra_para_id.get(&chave).cloned() {
            // Conceito já existe — atualiza acesso de toda a população
            for &id in &pop {
                self.ultimo_acesso.insert(id, current_time());
                *self.frequencia_acesso.entry(id).or_insert(0) += 1;
            }
            // One-shot: se estava em fast_weights, reforça
            if let Some(fw) = self.fast_weights.get_mut(&chave) {
                fw.reforcos += 1;
                fw.peso = (fw.peso + 0.1).min(1.0);
            }
            pop
        } else {
            // Novo conceito — one-shot: registra fast_weight antes de criar população
            self.fast_weights.insert(chave.clone(), FastWeight {
                peso: FAST_WEIGHT_INICIAL,
                t_criacao: current_time(),
                reforcos: 0,
                valence,
            });
            // Inicializa embedding determinístico a partir do hash da palavra
            let emb = embedding_from_hash(&chave);
            self.embeddings.insert(chave.clone(), emb);
            // Cria população de POPULACAO_N neurônios
            let mut pop = Vec::with_capacity(POPULACAO_N);
            for i in 0..POPULACAO_N {
                let nid = self.next_id;
                self.next_id = self.next_id.wrapping_add(1);
                let mut neuronio = NeuronioHibrido::new(nid, TipoNeuronal::RS, PrecisionType::FP32);
                // Ruído de inicialização: ±0.25 por neurônio
                // Cada neurônio da população tem um limiar ligeiramente diferente —
                // simula variabilidade biológica dentro de uma coluna cortical.
                let ruido = ((i as f32 * 7.3 + valence.abs() * 13.1) % 0.5) - 0.25;
                if let crate::synaptic_core::PesoNeuronio::FP32(ref mut v) = neuronio.peso {
                    *v = (*v + ruido).clamp(0.4, 1.6);
                }
                let uuid = Uuid::new_v4();
                self.ram.insert(uuid, neuronio);
                self.ultimo_acesso.insert(uuid, current_time());
                self.frequencia_acesso.insert(uuid, 1);
                // Valência inicial com pequena variação por neurônio
                let val_ruido = valence + ((i as f32 * 3.7 + 1.1) % 0.3) - 0.15;
                self.valencia_neuronio.insert(uuid, val_ruido.clamp(-1.0, 1.0));
                pop.push(uuid);
            }
            self.indices.entry("conceito".to_string())
                .or_insert_with(Vec::new)
                .extend(pop.iter().copied());
            self.palavra_para_id.insert(chave, pop.clone());
            self.total_neurons_criados += POPULACAO_N;
            pop
        };

        // Injeta corrente em toda a população + atualiza valência por neurônio
        let corrente = valence.abs() * I_BASE_CONCEITO + 4.0;
        for &id in &populacao {
            let entry = self.correntes.entry(id).or_insert(0.0);
            *entry = entry.max(corrente);
            // Atualiza valência com média ponderada (a experiência acumula)
            let v = self.valencia_neuronio.entry(id).or_insert(0.0);
            *v = (*v * 0.85 + valence * 0.15).clamp(-1.0, 1.0);
        }

        // STDP entre canônicos: pop_anterior[0] → pop_atual[0]
        // Manter STDP entre canônicos evita criação de O(N²×vocab²) sinapses.
        if let Some(ref pre_pop) = self.ultimo_conceito_pop.clone() {
            if let (Some(&pre_id), Some(&post_id)) = (pre_pop.first(), populacao.first()) {
                if pre_id != post_id {
                    let peso = self.sinapses_conceito.entry((pre_id, post_id)).or_insert(0.0);
                    *peso = (*peso + LTP_CONCEITO).clamp(0.0, PESO_MAX_CONCEITO);
                }
            }
        }
        self.ultimo_conceito_pop = Some(populacao.clone());
        self.grafo_dirty = true;

        // LRU eviction: remove sinapses mais fracas quando acima do cap
        if self.sinapses_conceito.len() > SINAPSES_CAP {
            let mut pesos: Vec<((Uuid, Uuid), f32)> = self.sinapses_conceito
                .iter()
                .map(|(&k, &v)| (k, v))
                .collect();
            pesos.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            // Remove os 5% mais fracos
            let n_remover = SINAPSES_CAP / 20;
            for (k, _) in pesos.iter().take(n_remover) {
                self.sinapses_conceito.remove(k);
            }
            self.grafo_dirty = true;
        }

        populacao
    }

    /// Fração da população de um conceito que está atualmente excitada (0..1).
    /// Representa quão "presente" o conceito está na mente da Selene agora.
    pub fn ativacao_populacao(&self, palavra: &str) -> f32 {
        let Some(pop) = self.palavra_para_id.get(palavra) else { return 0.0 };
        if pop.is_empty() { return 0.0; }
        let ativos = pop.iter()
            .filter(|id| self.correntes.get(id).copied().unwrap_or(0.0) > 1.0)
            .count();
        ativos as f32 / pop.len() as f32
    }

    /// Valência média da população de um conceito, ponderada pela ativação.
    /// Reflete a "memória emocional" acumulada — pode ser ambivalente se o conceito
    /// foi vivenciado em contextos contraditórios (parte da pop positiva, parte negativa).
    pub fn valencia_populacao(&self, palavra: &str) -> Option<f32> {
        let pop = self.palavra_para_id.get(palavra)?;
        if pop.is_empty() { return None; }
        let vals: Vec<f32> = pop.iter()
            .filter_map(|id| self.valencia_neuronio.get(id).copied())
            .collect();
        if vals.is_empty() { return None; }
        Some(vals.iter().sum::<f32>() / vals.len() as f32)
    }

    /// Lookup inverso: UUID do canônico → palavra.
    /// Construído sob demanda — não cacheado (vocab é pequeno, ~1k palavras).
    pub fn id_para_palavra(&self) -> HashMap<Uuid, String> {
        self.palavra_para_id.iter()
            .filter_map(|(palavra, pop)| pop.first().map(|&id| (id, palavra.clone())))
            .collect()
    }

    /// Constrói um grafo de palavras a partir das sinapses_conceito neurais.
    /// Substitui o antigo grafo_associacoes (HashMap de strings) — a fonte de
    /// verdade é agora o STDP entre populações neurais, não co-ocorrência textual.
    /// Filtro mínimo: apenas palavras com ≥2 chars entram no grafo.
    pub fn grafo_palavras(&mut self) -> HashMap<String, Vec<(String, f32)>> {
        if !self.grafo_dirty {
            if let Some(ref cache) = self.grafo_cache {
                return cache.clone();
            }
        }
        let id_to_word = self.id_para_palavra();
        let mut grafo: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        for (&(pre, post), &peso) in &self.sinapses_conceito {
            if let (Some(w_pre), Some(w_post)) = (id_to_word.get(&pre), id_to_word.get(&post)) {
                if w_pre.chars().count() >= 2 && w_post.chars().count() >= 2 {
                    grafo.entry(w_pre.clone()).or_default().push((w_post.clone(), peso));
                }
            }
        }
        for vizinhos in grafo.values_mut() {
            vizinhos.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        self.grafo_cache = Some(grafo.clone());
        self.grafo_dirty = false;
        grafo
    }

    /// Constrói o mapa palavra→valência a partir das populações neurais.
    /// Substitui o antigo palavra_valencias do BrainState — a valência
    /// agora reflete a memória emocional acumulada via Hebbian learning.
    pub fn valencias_palavras(&self) -> HashMap<String, f32> {
        self.palavra_para_id.keys()
            .filter_map(|p| self.valencia_populacao(p).map(|v| (p.clone(), v)))
            .collect()
    }

    /// Retorna uma sequência de palavras-guia baseada no template que melhor
    /// corresponde aos conceitos fornecidos. Os slots do template são preenchidos
    /// com as palavras mais frequentes no histórico de cada slot (restrição emergente).
    /// Retorna (tokens_scaffold, Some(uuid)) se há match útil, ou (vec![], None) caso contrário.
    pub fn template_scaffold(&self, conceitos: &[String]) -> (Vec<String>, Option<uuid::Uuid>) {
        let matches = self.template_store.reconhecer(conceitos);
        if let Some((id, score)) = matches.first() {
            if *score < 0.3 { return (Vec::new(), None); }
            if let Some(template) = self.template_store.templates.get(id) {
                let scaffold: Vec<String> = template.slots.iter()
                    .flat_map(|s| s.restricao_emergente(1).into_iter().next())
                    .collect();
                if scaffold.len() >= 2 { return (scaffold, Some(*id)); }
            }
        }
        (Vec::new(), None)
    }

    /// Injeta uma lista de pares causais (causa, efeito, peso) como sinapses
    /// de alta prioridade no swap. Usado na migração do grafo_causal legado.
    pub fn importar_causal(&mut self, pares: Vec<(String, String, f32)>) {
        for (causa, efeito, peso) in pares {
            let pop_c = self.aprender_conceito(&causa, 0.3);
            let pop_e = self.aprender_conceito(&efeito, 0.3);
            if let (Some(&pre), Some(&post)) = (pop_c.first(), pop_e.first()) {
                let entry = self.sinapses_conceito.entry((pre, post)).or_insert(0.0);
                *entry = (*entry + peso * 0.5).clamp(0.0, PESO_MAX_CONCEITO);
            }
        }
        self.grafo_dirty = true;
    }

    /// Retorna os N conceitos mais ativados no momento (por fração da população excitada).
    /// Útil para introspecção: "o que a Selene está pensando agora?"
    pub fn conceitos_ativos_top(&self, n: usize) -> Vec<(String, f32)> {
        let mut scores: Vec<(String, f32)> = self.palavra_para_id.keys()
            .map(|p| (p.clone(), self.ativacao_populacao(p)))
            .filter(|(_, a)| *a > 0.0)
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(n);
        scores
    }

    /// Executa um tick nos neurônios conceituais (chamado a cada tick do loop principal).
    ///
    /// Estratégia de população:
    /// 1. Todos os POPULACAO_N neurônios de cada conceito recebem update (rica representação interna)
    /// 2. STDP inter-conceito opera APENAS entre canônicos (pop[0]) → O(vocab²), não O((vocab×N)²)
    /// 3. Propagação sináptica vai do canônico pré para o canônico pós (sinapses já registradas)
    pub fn tick_semantico(&mut self, dt_s: f32, current_time_ms: f32) {
        // Coasting: pula STDP completo quando ocioso (sem correntes e sem spikes recentes)
        let tem_corrente = !self.correntes.is_empty();
        if !tem_corrente && self.ticks_sem_spike >= 15 {
            return;
        }

        // Todos os neurônios conceituais (população completa)
        let ids: Vec<Uuid> = self.palavra_para_id.values()
            .flat_map(|pop| pop.iter().copied())
            .collect();
        // Canônicos (pop[0] de cada conceito) — usados no STDP inter-conceito
        let canonicos: Vec<Uuid> = self.palavra_para_id.values()
            .filter_map(|pop| pop.first().copied())
            .collect();
        let mut spikes: Vec<Uuid> = Vec::new();

        for &id in &ids {
            if let Some(n) = self.ram.get_mut(&id) {
                let i_ext = self.correntes.get(&id).copied().unwrap_or(0.0);
                if n.update(i_ext, dt_s, current_time_ms, 1.0) {
                    spikes.push(id);
                }
            }
        }

        // STDP inter-conceito assimétrico (biologicamente correto):
        // LTP: pre disparou ANTES de post → pré→post fortalece (causal)
        // LTD: pre disparou DEPOIS de post → pré→post enfraquece (anti-causal)
        const LTD_CONCEITO: f32 = LTP_CONCEITO * 0.7; // LTD menor que LTP — net potentiação

        // LTP: post dispara agora, pre tem trace (disparou antes)
        for &post_id in spikes.iter().filter(|id| canonicos.contains(id)) {
            for &pre_id in &canonicos {
                if pre_id == post_id { continue; }
                let pre_trace = self.ram.get(&pre_id).map(|n| n.trace_pos).unwrap_or(0.0);
                if pre_trace > 0.15 {
                    let peso = self.sinapses_conceito.entry((pre_id, post_id)).or_insert(0.0);
                    *peso = (*peso + LTP_CONCEITO * pre_trace).clamp(0.0, PESO_MAX_CONCEITO);
                    self.grafo_dirty = true;
                }
            }
        }

        // LTD: pre dispara agora, post tem trace (disparou antes → pre chegou tarde → anti-causal)
        for &pre_id in spikes.iter().filter(|id| canonicos.contains(id)) {
            for &post_id in &canonicos {
                if pre_id == post_id { continue; }
                let post_trace = self.ram.get(&post_id).map(|n| n.trace_pos).unwrap_or(0.0);
                if post_trace > 0.15 {
                    // Enfraquece sinapse pre→post (pré chegou depois — não causou o spike de post)
                    if let Some(peso) = self.sinapses_conceito.get_mut(&(pre_id, post_id)) {
                        *peso = (*peso - LTD_CONCEITO * post_trace).max(0.0);
                        self.grafo_dirty = true;
                    }
                }
            }
        }

        // Propaga spikes pelas sinapses semânticas (canônico pre → canônico post)
        // A corrente injetada no canônico se espalha pela população via tick na próxima chamada
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

        if spikes.is_empty() {
            self.ticks_sem_spike = self.ticks_sem_spike.saturating_add(1);
        } else {
            self.ticks_sem_spike = 0;
        }

        // P4.2 — Plasticidade homeostática (synaptic scaling): a cada 100 ticks.
        // Conceitos silenciosos aumentam sua sensibilidade global; conceitos muito ativos
        // reduzem — mantém ativação média da população em nível biológico (~20%).
        // Turrigiano (2008): sem homeostase há runaway excitation ou silêncio total.
        const HOMEOSTASE_ALVO: f32 = 0.20; // 20% dos neurônios da população ativos
        const HOMEOSTASE_TAXA: f32 = 0.001;
        if self.ticks_sem_spike == 0 && ids.len() > 0 {
            let ativacao_atual = spikes.len() as f32 / ids.len().max(1) as f32;
            let delta_escala = (HOMEOSTASE_ALVO - ativacao_atual) * HOMEOSTASE_TAXA;
            if delta_escala.abs() > 1e-6 {
                for &id in &ids {
                    if let Some(n) = self.ram.get_mut(&id) {
                        if let crate::synaptic_core::PesoNeuronio::FP32(ref mut v) = n.peso {
                            *v = (*v + delta_escala).clamp(0.1, 2.5);
                        }
                    }
                }
            }
        }

        // P4.3 — Sparse coding (L1 regularization): a cada 50 ticks.
        // Suprime neurônios muito ativos (>80% disparando) — mantém ~20% de esparsidade.
        // Yamins (2021): representações esparsas reduzem interferência entre conceitos.
        const SPARSE_REG: f32 = 0.003; // taxa de supressão
        if self.ticks_sem_spike == 0 && spikes.len() > ids.len() * 4 / 5 {
            // Mais de 80% da população disparando → aplica L1 (reduz pesos)
            for &id in &spikes {
                if let Some(n) = self.ram.get_mut(&id) {
                    if let crate::synaptic_core::PesoNeuronio::FP32(ref mut v) = n.peso {
                        *v = (*v * (1.0 - SPARSE_REG)).max(0.1);
                    }
                }
            }
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

        // Injeta corrente inicial em toda a população de cada conceito
        for (palavra, pop) in &self.palavra_para_id {
            if let Some(&val) = valencias.get(palavra.as_str()) {
                let corrente = (val.abs() * I_BASE_CONCEITO + 4.0).min(20.0);
                for &id in pop {
                    self.correntes.insert(id, corrente);
                }
            }
        }

        // Todos os neurônios (população completa) + canônicos separados para STDP
        let ids: Vec<Uuid> = self.palavra_para_id.values()
            .flat_map(|pop| pop.iter().copied())
            .collect();
        let canonicos: Vec<Uuid> = self.palavra_para_id.values()
            .filter_map(|pop| pop.first().copied())
            .collect();

        for ciclo in 0..n_ciclos {
            // A cada 50 ciclos re-injeta corrente (mantém ativação durante treino longo)
            if ciclo % 50 == 0 {
                for (palavra, pop) in &self.palavra_para_id {
                    if let Some(&val) = valencias.get(palavra.as_str()) {
                        let reforco = (val.abs() * 8.0 + 2.0).min(18.0);
                        for &id in pop {
                            let entry = self.correntes.entry(id).or_insert(0.0);
                            *entry = entry.max(reforco);
                        }
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

            // STDP apenas entre canônicos (eficiência O(vocab²))
            for &post_id in spikes_ciclo.iter().filter(|id| canonicos.contains(id)) {
                for &pre_id in &canonicos {
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

    // ═══════════════════════════════════════════════════════════════════
    // CAMADA ZERO — primitivas sensoriais fixas
    // ═══════════════════════════════════════════════════════════════════

    /// Cria neurônio de camada zero (primitivo fixo).
    /// RS para fonemas (auditivo regular), CH para visuais (burst rápido).
    fn criar_primitivo(&mut self, tag: &str) -> Uuid {
        let nid = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let tipo = if tag.starts_with("vis:") { TipoNeuronal::CH } else { TipoNeuronal::RS };
        let neuronio = NeuronioHibrido::new(nid, tipo, PrecisionType::FP32);
        let uuid = Uuid::new_v4();
        self.ram.insert(uuid, neuronio);
        self.ultimo_acesso.insert(uuid, current_time());
        self.frequencia_acesso.insert(uuid, 0);
        self.total_neurons_criados += 1;
        uuid
    }

    /// Inicializa todos os neurônios de camada zero (35 fonemas + 20 visuais).
    /// Idempotente — só cria o que ainda não existe.
    /// Deve ser chamado logo após `carregar_estado()` no startup.
    pub fn inicializar_camada_zero(&mut self) {
        let mut criados = 0usize;

        for &tag in FONEMAS_PRIMITIVOS {
            if !self.fonemas_para_id.contains_key(tag) {
                let uuid = self.criar_primitivo(tag);
                self.fonemas_para_id.insert(tag.to_string(), uuid);
                criados += 1;
            }
        }

        for &tag in VISUAIS_PRIMITIVOS {
            if !self.visuais_para_id.contains_key(tag) {
                let uuid = self.criar_primitivo(tag);
                self.visuais_para_id.insert(tag.to_string(), uuid);
                criados += 1;
            }
        }

        if criados > 0 {
            println!(
                "🔵 [Camada 0] {} primitivos criados ({} fonemas, {} visuais)",
                criados,
                self.fonemas_para_id.len(),
                self.visuais_para_id.len(),
            );
        } else {
            println!(
                "🔵 [Camada 0] Restaurada — {} fonemas + {} visuais já presentes",
                self.fonemas_para_id.len(),
                self.visuais_para_id.len(),
            );
        }
    }

    /// Aprende uma palavra como sequência de fonemas (camada 1 emergente).
    ///
    /// Não cria neurônios novos — apenas reforça as sinapses STDP entre
    /// os neurônios primitivos de fonema que compõem a palavra.
    /// A representação da palavra existe APENAS nas sinapses — nunca como neurônio próprio.
    ///
    /// `tags`: lista de tags no formato "ph:a", "ph:k", etc.
    pub fn aprender_sequencia_fonemas(&mut self, tags: &[String], valence: f32) {
        let corrente_base = valence.abs() * I_BASE_CONCEITO + 4.0;
        let mut anterior: Option<Uuid> = None;

        for tag in tags {
            let Some(&id) = self.fonemas_para_id.get(tag.as_str()) else { continue };

            // Injeta corrente proporcional à valência
            let entry = self.correntes.entry(id).or_insert(0.0);
            *entry = entry.max(corrente_base);

            // STDP sequencial: pre → post forma o bigrama fonêmico
            if let Some(pre_id) = anterior {
                let peso = self.sinapses_conceito.entry((pre_id, id)).or_insert(0.0);
                *peso = (*peso + LTP_CONCEITO).clamp(0.0, PESO_MAX_CONCEITO);
            }

            self.ultimo_acesso.insert(id, current_time());
            *self.frequencia_acesso.entry(id).or_insert(0) += 1;
            anterior = Some(id);
        }

        // Propaga contexto sináptico para o próximo aprendizado
        self.ultimo_conceito_id = anterior;
    }

    /// Ativa uma primitiva visual injetando corrente e reforçando sinapses.
    /// Só funciona se `inicializar_camada_zero()` já foi chamado.
    pub fn ativar_primitiva_visual(&mut self, tag: &str, intensidade: f32) {
        let Some(&id) = self.visuais_para_id.get(tag) else { return };

        let corrente = intensidade.clamp(0.0, 1.0) * I_BASE_CONCEITO + 2.0;
        let entry = self.correntes.entry(id).or_insert(0.0);
        *entry = entry.max(corrente);

        // STDP com último primitivo ativado (cross-modal binding)
        if let Some(pre_id) = self.ultimo_conceito_id {
            if pre_id != id {
                let peso = self.sinapses_conceito.entry((pre_id, id)).or_insert(0.0);
                *peso = (*peso + LTP_CONCEITO * intensidade).clamp(0.0, PESO_MAX_CONCEITO);
            }
        }

        self.ultimo_acesso.insert(id, current_time());
        *self.frequencia_acesso.entry(id).or_insert(0) += 1;
        self.ultimo_conceito_id = Some(id);
    }

    /// Converte métricas visuais agregadas em ativações de primitivas de camada zero.
    ///
    /// Parâmetros (todos normalizados 0..1):
    /// - `lum_media`:   luminância média da cena (claridade / escuridão)
    /// - `freq_media`:  frequência espacial média (textura / bordas / formas)
    /// - `taxa_var`:    taxa de variação temporal (movimento entre frames)
    /// - `nm_medio`:    comprimento de onda dominante estimado em nm (cor)
    pub fn processar_visual_simples(
        &mut self,
        lum_media:  f32,
        freq_media: f32,
        taxa_var:   f32,
        nm_medio:   f32,
    ) {
        // Luminância → nível
        let lum_tag = match lum_media {
            l if l < 0.15 => "vis:lum:muito_escuro",
            l if l < 0.35 => "vis:lum:escuro",
            l if l < 0.60 => "vis:lum:medio",
            l if l < 0.80 => "vis:lum:claro",
            _              => "vis:lum:muito_claro",
        };
        self.ativar_primitiva_visual(lum_tag, lum_media);

        // Banda espectral dominante → cor
        let band_tag = match nm_medio as u32 {
            0..=449   => "vis:band:violeta",
            450..=494 => "vis:band:azul",
            495..=519 => "vis:band:ciano",
            520..=564 => "vis:band:verde",
            565..=589 => "vis:band:amarelo",
            590..=624 => "vis:band:laranja",
            _          => "vis:band:vermelho",
        };
        self.ativar_primitiva_visual(band_tag, lum_media);

        // Frequência espacial → textura / formas
        if freq_media > 0.05 {
            let freq_tag = match freq_media {
                f if f < 0.25 => "vis:freq:baixa",
                f if f < 0.55 => "vis:freq:media",
                _              => "vis:freq:alta",
            };
            self.ativar_primitiva_visual(freq_tag, freq_media);

            // Orientação de borda: aproximada pela razão freq/lum
            // (heurística: alta freq + baixa lum → bordas horizontais; inverso → verticais)
            let ratio = (freq_media / (lum_media + 0.01)).clamp(0.0, 4.0) / 4.0;
            let borda_tag = match ratio {
                r if r < 0.25 => "vis:borda:0",
                r if r < 0.50 => "vis:borda:45",
                r if r < 0.75 => "vis:borda:90",
                _              => "vis:borda:135",
            };
            self.ativar_primitiva_visual(borda_tag, freq_media);
        }

        // Movimento
        if taxa_var > 0.04 {
            self.ativar_primitiva_visual("vis:movimento:sim", taxa_var.min(1.0));
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


impl SwapManager {
    // ═══════════════════════════════════════════════════════════════════
    // ONE-SHOT LEARNING — Fast weights episódicos
    // ═══════════════════════════════════════════════════════════════════

    /// Consolida ou descarta fast_weights a cada tick semântico.
    ///
    /// Consolidação: reforços ≥ threshold → synapse permanente com peso alto
    /// Descarte: TTL expirado sem reforço suficiente → remove conceito raro
    pub fn consolidar_fast_weights(&mut self) {
        let agora = current_time();
        let mut consolidar: Vec<(String, f32)> = Vec::new();
        let mut descartar: Vec<String> = Vec::new();

        for (palavra, fw) in &mut self.fast_weights {
            // Decaimento natural
            fw.peso *= FAST_WEIGHT_DECAY;

            if fw.reforcos >= FAST_WEIGHT_CONSOLIDAR_REFORCOS {
                // Consolidação: fast_weight vira sinapse permanente forte
                consolidar.push((palavra.clone(), fw.peso * 1.5));
            } else if agora - fw.t_criacao > FAST_WEIGHT_TTL_S && fw.peso < 0.2 {
                // TTL expirado e peso baixo → descarta
                descartar.push(palavra.clone());
            }
        }

        // Consolida: reforça sinapses canônicas do conceito com ele mesmo
        for (palavra, peso_boost) in consolidar {
            if let Some(pop) = self.palavra_para_id.get(&palavra).cloned() {
                if let Some(&can) = pop.first() {
                    let entry = self.sinapses_conceito.entry((can, can)).or_insert(0.0);
                    *entry = (*entry + peso_boost * 0.3).clamp(0.0, PESO_MAX_CONCEITO);
                }
            }
            self.fast_weights.remove(&palavra);
            log::debug!("⚡ [ONE-SHOT] '{}' consolidado em sinapse permanente", palavra);
        }

        for palavra in descartar {
            self.fast_weights.remove(&palavra);
        }
    }

    /// Retorna os N conceitos aprendidos mais recentemente via one-shot.
    pub fn conceitos_one_shot_recentes(&self, n: usize) -> Vec<(&str, f32)> {
        let mut v: Vec<(&str, f32)> = self.fast_weights.iter()
            .map(|(k, fw)| (k.as_str(), fw.peso))
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v.truncate(n);
        v
    }

    // ═══════════════════════════════════════════════════════════════════
    // GRAPH VERSIONING — Snapshots do grafo semântico
    // ═══════════════════════════════════════════════════════════════════

    /// Cria um snapshot do estado atual do grafo.
    /// Chamado periodicamente (ex: a cada 1000 ticks ou no início do sono).
    pub fn criar_snapshot(&mut self, tick: u64) {
        if self.snapshots.len() >= MAX_SNAPSHOTS {
            self.snapshots.pop_front(); // descarta o mais antigo
        }
        let snap = GraphSnapshot {
            tick,
            timestamp: current_time(),
            grafo: self.grafo_palavras(),
            valencias: self.valencias_palavras(),
            n_sinapses: self.sinapses_semanticas_ativas(),
        };
        self.snapshots.push_back(snap);
        log::info!("📸 [SNAPSHOT] tick={} vocab={} sinapses={}",
            tick, self.palavra_para_id.len(), self.sinapses_semanticas_ativas());
    }

    /// Lista snapshots disponíveis: (índice, tick, n_sinapses).
    pub fn snapshots_disponiveis(&self) -> Vec<(usize, u64, usize)> {
        self.snapshots.iter().enumerate()
            .map(|(i, s)| (i, s.tick, s.n_sinapses))
            .collect()
    }

    /// Restaura valências de um snapshot anterior.
    /// Não remove conceitos novos — apenas restaura pesos de valência.
    /// Útil para reverter aprendizado problemático de uma sessão.
    pub fn restaurar_valencias_snapshot(&mut self, idx: usize) -> bool {
        let Some(snap) = self.snapshots.get(idx).cloned() else { return false };
        let mut restaurados = 0usize;
        for (palavra, val) in &snap.valencias {
            if let Some(pop) = self.palavra_para_id.get(palavra).cloned() {
                for &id in &pop {
                    if let Some(v) = self.valencia_neuronio.get_mut(&id) {
                        *v = *val;
                        restaurados += 1;
                    }
                }
            }
        }
        log::info!("🔄 [RESTAURAR] snapshot tick={} → {} valências restauradas",
            snap.tick, restaurados);
        true
    }

    // ═══════════════════════════════════════════════════════════════════
    // EMBEDDINGS VETORIAIS — Busca por similaridade semântica
    // ═══════════════════════════════════════════════════════════════════

    /// Similaridade de cosseno entre dois conceitos [-1.0, 1.0].
    /// Retorna None se qualquer conceito não tiver embedding.
    pub fn similaridade_cosseno(&self, a: &str, b: &str) -> Option<f32> {
        let ea = self.embeddings.get(a)?;
        let eb = self.embeddings.get(b)?;
        let dot: f32 = ea.iter().zip(eb.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = ea.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = eb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-6 || nb < 1e-6 { return None; }
        Some((dot / (na * nb)).clamp(-1.0, 1.0))
    }

    /// Top-K vizinhos semânticos de uma palavra (por cosine similarity).
    /// Exclui a própria palavra e retorna apenas similaridades > threshold.
    pub fn vizinhos_semanticos(&self, palavra: &str, k: usize, threshold: f32) -> Vec<(String, f32)> {
        let Some(ea) = self.embeddings.get(palavra) else { return Vec::new() };
        let mut scores: Vec<(String, f32)> = self.embeddings.iter()
            .filter(|(w, _)| w.as_str() != palavra)
            .filter_map(|(w, eb)| {
                let dot: f32 = ea.iter().zip(eb.iter()).map(|(x, y)| x * y).sum();
                let na: f32 = ea.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = eb.iter().map(|x| x * x).sum::<f32>().sqrt();
                if na < 1e-6 || nb < 1e-6 { return None; }
                let sim = (dot / (na * nb)).clamp(-1.0, 1.0);
                if sim > threshold { Some((w.clone(), sim)) } else { None }
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    /// Atualiza embeddings de dois conceitos co-ativados (aprendizado associativo).
    /// Biologicamente: co-ativação → representações aproximam-se no espaço vetorial.
    pub fn atualizar_embeddings_coativacao(&mut self, palavra_a: &str, palavra_b: &str, lr: f32) {
        let ea = self.embeddings.get(palavra_a).copied();
        let eb = self.embeddings.get(palavra_b).copied();
        if let (Some(mut a), Some(mut b)) = (ea, eb) {
            for i in 0..EMBED_DIM {
                let delta_a = lr * (b[i] - a[i]);
                let delta_b = lr * (a[i] - b[i]);
                a[i] = (a[i] + delta_a).clamp(-1.0, 1.0);
                b[i] = (b[i] + delta_b).clamp(-1.0, 1.0);
            }
            self.embeddings.insert(palavra_a.to_string(), a);
            self.embeddings.insert(palavra_b.to_string(), b);
        }
    }

    /// Inclui vizinhos semânticos no grafo retornado.
    /// Usado para enriquecer o graph_walk com similaridade vetorial.
    pub fn grafo_com_semantica(&mut self, top_k: usize, threshold: f32) -> HashMap<String, Vec<(String, f32)>> {
        let mut grafo = self.grafo_palavras();
        // Para cada palavra, adiciona vizinhos semânticos com peso 0.3×similaridade
        let palavras: Vec<String> = self.embeddings.keys().cloned().collect();
        for palavra in &palavras {
            let vizinhos = self.vizinhos_semanticos(palavra, top_k, threshold);
            let entry = grafo.entry(palavra.clone()).or_insert_with(Vec::new);
            for (viz, sim) in vizinhos {
                // Só adiciona se não existir aresta direta já
                if !entry.iter().any(|(w, _)| w == &viz) {
                    entry.push((viz, sim * 0.3));
                }
            }
        }
        grafo
    }
}

/// Gera embedding determinístico de 32 dimensões a partir do hash da string.
/// Sem dependência externa — inicialização barata e reprodutível.
fn embedding_from_hash(palavra: &str) -> [f32; EMBED_DIM] {
    let mut emb = [0.0f32; EMBED_DIM];
    // FNV-1a hash variant para cada dimensão
    let bytes = palavra.as_bytes();
    for (i, slot) in emb.iter_mut().enumerate() {
        let mut h: u64 = 14695981039346656037u64
            .wrapping_add(i as u64 * 2654435761);
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        // Mapeia para [-1.0, 1.0]
        *slot = ((h & 0xFFFF) as f32 / 32767.5) - 1.0;
    }
    emb
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
    /// Arquiva um neurônio do SSD para o cold storage em disco (D:/Selene_Archive/).
    /// Cria o diretório se não existir.
    pub fn arquivar_para_hdd(&mut self, id: &Uuid) -> std::io::Result<()> {
        if let Some(neuronio) = self.ssd.remove(id) {
            let dir = "D:/Selene_Archive";
            std::fs::create_dir_all(dir)?;
            let path = format!("{}/{}.json", dir, id);
            let dados = serde_json::to_string(&neuronio).unwrap_or_default();
            std::fs::write(&path, dados)?;
            log::debug!("[ColdStorage] Neurônio {} arquivado → {}", id, path);
        }
        Ok(())
    }

    /// Restaura um neurônio do cold storage para a RAM.
    /// Retorna true se encontrado e carregado com sucesso.
    pub fn restaurar_do_hdd(&mut self, id: &Uuid) -> bool {
        let path = format!("D:/Selene_Archive/{}.json", id);
        match std::fs::read_to_string(&path) {
            Ok(dados) => {
                match serde_json::from_str::<NeuronioHibrido>(&dados) {
                    Ok(mut neuronio) => {
                        // Garante invariante tipo↔modelo após deserialização
                        neuronio.modelo = crate::synaptic_core::ModeloDinamico::para_tipo(neuronio.tipo);
                        self.ram.insert(*id, neuronio);
                        self.ultimo_acesso.insert(*id, current_time());
                        *self.frequencia_acesso.entry(*id).or_insert(0) += 1;
                        log::info!("[ColdStorage] Neurônio {} restaurado do HDD", id);
                        true
                    }
                    Err(e) => {
                        log::warn!("[ColdStorage] Falha ao deserializar {}: {}", id, e);
                        false
                    }
                }
            }
            Err(_) => false,
        }
    }

    /// Consolida neurônios dormentes do SSD para RAM se há espaço disponível.
    /// Prioriza os de maior frequência de acesso (mais "lembrados").
    /// Chamado automaticamente quando a RAM está abaixo de 60% de capacidade.
    pub fn consolidar_ssd_para_ram(&mut self) {
        let ram_usada = self.ram.len();
        let cap = self.max_ram_neurons;
        if ram_usada >= (cap as f32 * 0.60) as usize { return; }

        let slots_livres = (cap as f32 * 0.60) as usize - ram_usada;

        // Ordena SSD por frequência de acesso (mais acessados primeiro)
        let mut candidatos: Vec<(Uuid, u32)> = self.ssd.keys()
            .map(|id| (*id, *self.frequencia_acesso.get(id).unwrap_or(&0)))
            .collect();
        candidatos.sort_by(|a, b| b.1.cmp(&a.1));

        let mut consolidados = 0usize;
        for (id, _) in candidatos.iter().take(slots_livres) {
            if let Some(neuronio) = self.ssd.remove(id) {
                self.ram.insert(*id, neuronio);
                self.ultimo_acesso.insert(*id, current_time());
                consolidados += 1;
            }
        }
        if consolidados > 0 {
            log::info!("[Consolidação] {} neurônios SSD→RAM", consolidados);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // PERSISTÊNCIA DO ESTADO SEMÂNTICO
    // ═══════════════════════════════════════════════════════════════════

    /// Salva o estado semântico do SwapManager para disco.
    ///
    /// Persiste:
    /// - `palavra_para_id`  → garante que palavras já aprendidas não recebam
    ///   novo UUID ao reiniciar (mantém identidade dos neurônios).
    /// - `sinapses_conceito` → preserva os pesos STDP acumulados entre conceitos.
    /// - `next_id` / `total_neurons_criados` → contadores para continuidade.
    ///
    /// Não persiste: estado Izhikevich (v, u) dos neurônios — é transitório.
    /// Os neurônios serão recriados pelo `aprender_conceito` na próxima sessão
    /// de treino, mas com os UUIDs e pesos sinápticos corretos restaurados.
    pub fn salvar_estado(&self, caminho: &str) -> std::io::Result<()> {
        // Serializa palavra_para_id como {palavra: [uuid1, uuid2, ...]} (população)
        let palavras: serde_json::Map<String, serde_json::Value> = self.palavra_para_id
            .iter()
            .map(|(p, pop)| {
                let uuids: Vec<serde_json::Value> = pop.iter()
                    .map(|u| serde_json::Value::String(u.to_string()))
                    .collect();
                (p.clone(), serde_json::Value::Array(uuids))
            })
            .collect();

        // Serializa sinapses_conceito como [{pre, post, peso}]
        let sinapses: Vec<serde_json::Value> = self.sinapses_conceito
            .iter()
            .filter(|(_, &p)| p > 0.005) // descarta sinapses praticamente nulas
            .map(|((pre, post), &peso)| serde_json::json!({
                "pre":  pre.to_string(),
                "post": post.to_string(),
                "peso": peso,
            }))
            .collect();

        // Serializa fonemas_para_id
        let fonemas_json: serde_json::Map<String, serde_json::Value> = self.fonemas_para_id
            .iter()
            .map(|(t, u)| (t.clone(), serde_json::Value::String(u.to_string())))
            .collect();

        // Serializa visuais_para_id
        let visuais_json: serde_json::Map<String, serde_json::Value> = self.visuais_para_id
            .iter()
            .map(|(t, u)| (t.clone(), serde_json::Value::String(u.to_string())))
            .collect();

        // Serializa valencia_neuronio como {uuid: val}
        let valencias_json: serde_json::Map<String, serde_json::Value> = self.valencia_neuronio
            .iter()
            .map(|(id, &val)| (id.to_string(), serde_json::Value::from(val as f64)))
            .collect();

        let payload = serde_json::json!({
            "selene_swap_v2": {
                "next_id":           self.next_id,
                "total_criados":     self.total_neurons_criados,
                "populacao_n":       POPULACAO_N,
                "n_palavras":        self.palavra_para_id.len(),
                "n_sinapses":        sinapses.len(),
                "palavra_para_id":   palavras,
                "sinapses_conceito": sinapses,
                "fonemas_para_id":   fonemas_json,
                "visuais_para_id":   visuais_json,
                "valencia_neuronio": valencias_json,
            }
        });

        let json = serde_json::to_string_pretty(&payload)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(caminho, json)
    }

    /// Restaura o estado semântico a partir do arquivo salvo por `salvar_estado`.
    ///
    /// Após a restauração, `aprender_conceito` vai reconhecer as palavras já
    /// aprendidas (via `palavra_para_id`) e reutilizar seus UUIDs, garantindo
    /// continuidade dos pesos sinápticos.
    pub fn carregar_estado(&mut self, caminho: &str) {
        let content = match std::fs::read_to_string(caminho) {
            Ok(c) => c,
            Err(_) => return, // arquivo não existe ainda — normal na primeira execução
        };
        let json: serde_json::Value = match serde_json::from_str(&content) {
            Ok(j) => j,
            Err(e) => { log::warn!("[SwapState] JSON inválido em {}: {}", caminho, e); return; }
        };

        // Suporta v2 (populações) e v1 legado (UUID único por palavra)
        let (swap, is_v2) = if let Some(s) = json.get("selene_swap_v2") {
            (s, true)
        } else if let Some(s) = json.get("selene_swap_v1") {
            (s, false)
        } else {
            return;
        };

        // Restaura next_id e contadores
        if let Some(n) = swap["next_id"].as_u64() { self.next_id = n as u32; }
        if let Some(n) = swap["total_criados"].as_u64() { self.total_neurons_criados = n as usize; }

        // Restaura palavra_para_id
        let mut n_palavras = 0usize;
        if let Some(obj) = swap["palavra_para_id"].as_object() {
            for (palavra, val) in obj {
                let uuids: Vec<Uuid> = if is_v2 {
                    // v2: array de UUIDs (população)
                    val.as_array()
                        .map(|arr| arr.iter()
                            .filter_map(|v| v.as_str().and_then(|s| Uuid::parse_str(s).ok()))
                            .collect())
                        .unwrap_or_default()
                } else {
                    // v1 legado: UUID único — envolve em Vec
                    val.as_str()
                        .and_then(|s| Uuid::parse_str(s).ok())
                        .map(|u| vec![u])
                        .unwrap_or_default()
                };

                if uuids.is_empty() { continue; }

                // Cria placeholder em RAM para cada neurônio da população
                for &uuid in &uuids {
                    self.ram.entry(uuid).or_insert_with(|| {
                        let nid = self.next_id;
                        self.next_id = self.next_id.wrapping_add(1);
                        NeuronioHibrido::new(nid, TipoNeuronal::RS, PrecisionType::FP32)
                    });
                }
                self.palavra_para_id.insert(palavra.clone(), uuids);
                n_palavras += 1;
            }
        }

        // Restaura valencia_neuronio (v2 apenas)
        if is_v2 {
            if let Some(obj) = swap["valencia_neuronio"].as_object() {
                for (uuid_str, val) in obj {
                    if let (Ok(uuid), Some(v)) = (Uuid::parse_str(uuid_str), val.as_f64()) {
                        self.valencia_neuronio.insert(uuid, v as f32);
                    }
                }
            }
        }

        // Restaura sinapses_conceito
        let mut n_sinapses = 0usize;
        if let Some(arr) = swap["sinapses_conceito"].as_array() {
            for s in arr {
                if let (Some(pre_str), Some(post_str), Some(peso)) = (
                    s["pre"].as_str(), s["post"].as_str(), s["peso"].as_f64()
                ) {
                    if let (Ok(pre), Ok(post)) = (
                        Uuid::parse_str(pre_str), Uuid::parse_str(post_str)
                    ) {
                        self.sinapses_conceito.insert((pre, post), peso as f32);
                        n_sinapses += 1;
                    }
                }
            }
        }

        // Restaura fonemas_para_id (camada 0 fonêmica)
        let mut n_fonemas = 0usize;
        if let Some(obj) = swap["fonemas_para_id"].as_object() {
            for (tag, val) in obj {
                if let Some(uuid_str) = val.as_str() {
                    if let Ok(uuid) = Uuid::parse_str(uuid_str) {
                        self.fonemas_para_id.insert(tag.clone(), uuid);
                        self.ram.entry(uuid).or_insert_with(|| {
                            let nid = self.next_id;
                            self.next_id = self.next_id.wrapping_add(1);
                            NeuronioHibrido::new(nid, TipoNeuronal::RS, PrecisionType::FP32)
                        });
                        n_fonemas += 1;
                    }
                }
            }
        }

        // Restaura visuais_para_id (camada 0 visual)
        let mut n_visuais = 0usize;
        if let Some(obj) = swap["visuais_para_id"].as_object() {
            for (tag, val) in obj {
                if let Some(uuid_str) = val.as_str() {
                    if let Ok(uuid) = Uuid::parse_str(uuid_str) {
                        self.visuais_para_id.insert(tag.clone(), uuid);
                        self.ram.entry(uuid).or_insert_with(|| {
                            let nid = self.next_id;
                            self.next_id = self.next_id.wrapping_add(1);
                            NeuronioHibrido::new(nid, TipoNeuronal::CH, PrecisionType::FP32)
                        });
                        n_visuais += 1;
                    }
                }
            }
        }

        println!(
            "🧠 [SwapState] Restaurado: {} conceituais | {} fonemas | {} visuais | {} sinapses",
            n_palavras, n_fonemas, n_visuais, n_sinapses
        );
    }
}
