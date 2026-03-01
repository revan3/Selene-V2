// src/websocket/bridge.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use warp::Filter;
use std::sync::Arc;
use tokio::sync::broadcast;
use serde::{Serialize, Deserialize};
use crate::brain_zones::frontal::FrontalLobe;
use crate::brain_zones::occipital::OccipitalLobe;
use crate::brain_zones::parietal::ParietalLobe;
use crate::brain_zones::temporal::TemporalLobe;
use crate::brain_zones::limbic::LimbicSystem;
use crate::brain_zones::hippocampus::HippocampusV2;
use crate::neurochem::NeuroChem;
use crate::ego::Ego;
use crate::interoception::Interoception;
use crate::basal_ganglia::BasalGanglia;
use crate::brainstem::Brainstem;
use crate::storage::swap_manager::SwapManager;

// ================== ESTRUTURAS DE DADOS ==================

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NeuralStatus {
    pub regions: Vec<RegionStatus>,
    pub global: GlobalMetrics,
    pub logs: Vec<LogEntry>,
    
    // NOVAS MÉTRICAS
    pub total_neurons: u64,
    pub active_neurons: u64,
    pub total_synapses: u64,
    pub active_synapses: u64,
    pub humor: i32,
    pub energia: u32,
    pub ultimo_pensamento: String,
    pub sensacao_corporal: f32,
    pub num_habitos: usize,
    pub alertness: f32,
    pub neurogenese_eventos: usize,
    pub limite_fisico: usize,
    pub limite_biologico: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RegionStatus {
    pub id: String,
    pub temp: f32,
    pub cpu: f32,
    pub ram: f32,
    pub gpu: f32,
    pub disk: f32,
    pub freq: f32,
    pub lat: f32,
    pub spikes: f32,
    pub chem: Vec<ChemLevel>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChemLevel {
    pub key: String,
    pub val: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GlobalMetrics {
    pub cpu_avg: f32,
    pub ram_avg: f32,
    pub gpu_avg: f32,
    pub disk_avg: f32,
    pub freq_avg: f32,
    pub lat_avg: f32,
    pub total_synapses: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LogEntry {
    pub time: String,
    pub level: String,
    pub msg: String,
}

// ================== ESTADO DO CÉREBRO (EXPANDIDO) ==================

pub struct BrainState {
    // Lobos existentes
    pub frontal: FrontalLobe,
    pub occipital: OccipitalLobe,
    pub parietal: ParietalLobe,
    pub temporal: TemporalLobe,
    pub limbic: LimbicSystem,
    pub hippocampus: HippocampusV2,
    
    // Novos módulos
    pub ego: Ego,
    pub interoception: Interoception,
    pub basal_ganglia: BasalGanglia,
    pub brainstem: Brainstem,
    pub swap_manager: Arc<tokio::sync::Mutex<SwapManager>>,
    
    // Métricas globais
    pub global_cpu_avg: f32,
    pub global_ram_avg: f32,
    pub global_freq_avg: f32,
    pub global_lat_avg: f32,
    pub total_synapses: f32,
}

impl BrainState {
    pub fn new(swap_manager: Arc<tokio::sync::Mutex<SwapManager>>) -> Self {
        Self {
            frontal: FrontalLobe::new(1024, 0.2, 0.1),
            occipital: OccipitalLobe::new(1024, 0.2),
            parietal: ParietalLobe::new(1024, 0.2),
            temporal: TemporalLobe::new(1024, 0.005, 0.2),
            limbic: LimbicSystem::new(512),
            hippocampus: HippocampusV2::new(512),
            ego: Ego::carregar_ou_criar("Selene"),
            interoception: Interoception::new(),
            basal_ganglia: BasalGanglia::new(),
            brainstem: Brainstem::new(),
            swap_manager,
            global_cpu_avg: 0.0,
            global_ram_avg: 0.0,
            global_freq_avg: 0.0,
            global_lat_avg: 0.0,
            total_synapses: 0.0,
        }
    }
}

// ================== FUNÇÕES DO SERVIDOR ==================

pub async fn start_websocket_server(brain: Arc<tokio::sync::Mutex<BrainState>>) {
    let (tx, _) = broadcast::channel::<NeuralStatus>(100);
    
    let ws_route = warp::path("ws")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
            let tx = tx.clone();
            let brain = brain.clone();
            ws.on_upgrade(move |socket| handle_connection(socket, tx, brain))
        });
    
    let static_route = warp::path("static")
        .and(warp::fs::dir("./static"));
    
    let index_route = warp::path::end()
        .and(warp::fs::file("./static/index.html"));
    
    let routes = index_route.or(static_route).or(ws_route);
    
    println!("🌐 Servidor web rodando em http://localhost:3030");
    
    // Thread para broadcast periódico
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(500));
        loop {
            interval.tick().await;
            let brain_locked = brain.lock().await;
            let status = collect_neural_status(&brain_locked).await;
            let _ = tx.send(status);
        }
    });
    
    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
}

async fn handle_connection(
    ws: warp::ws::WebSocket, 
    _tx: broadcast::Sender<NeuralStatus>,
    brain: Arc<tokio::sync::Mutex<BrainState>>
) {
    println!("   ✅ Cliente conectado ao WebSocket");
    let (mut ws_tx, mut ws_rx) = ws.split();
    
    while let Some(result) = ws_rx.next().await {
        if let Ok(msg) = result {
            if let Ok(text) = msg.to_str() {
                println!("   📩 Mensagem recebida: {}", text);
                // Processa comandos (chat, etc.)
                process_command(text, brain.clone()).await;
                
                // Envia confirmação
                let _ = ws_tx.send(warp::ws::Message::text("Comando recebido")).await;
            }
        }
    }
}

// ================== FUNÇÕES AUXILIARES ==================

async fn process_command(cmd: &str, brain: Arc<tokio::sync::Mutex<BrainState>>) {
    let mut brain = brain.lock().await;
    
    if cmd.contains("como você está") || cmd.contains("humor") {
        let resposta = format!(
            "Meu humor atual é {:.1} (de -100 a 100) e minha energia é {:.1}%. Último pensamento: {}",
            brain.ego.current_state.humor * 100.0,
            brain.ego.current_state.energia * 100.0,
            brain.ego.narrative_voice.pensamentos_recentes.back().unwrap_or(&"nenhum".to_string())
        );
        println!("   💬 Resposta: {}", resposta);
    } else if cmd.contains("estatísticas") || cmd.contains("stats") {
        let stats = brain.swap_manager.lock().await.estatisticas();
        println!(
            "   📊 Neurônios: {} ativos, {} totais | Sinapses: {}M ativas",
            stats.ram, stats.total, stats.synapses_ativas / 1_000_000
        );
    }
}

async fn collect_neural_status(brain: &BrainState) -> NeuralStatus {
    let swap_stats = brain.swap_manager.lock().await.estatisticas();
    
    NeuralStatus {
        regions: vec![
            RegionStatus {
                id: "frontal".into(),
                temp: 45.0,
                cpu: 50.0,
                ram: 30.0,
                gpu: 0.0,
                disk: 0.0,
                freq: 100.0,
                lat: 10.0,
                spikes: 75.0,
                chem: vec![
                    ChemLevel { key: "dopa".into(), val: 72.0 },
                    ChemLevel { key: "nora".into(), val: 58.0 },
                    ChemLevel { key: "sero".into(), val: 65.0 },
                ],
            },
        ],
        global: GlobalMetrics {
            cpu_avg: 45.0,
            ram_avg: 35.0,
            gpu_avg: 0.0,
            disk_avg: 0.0,
            freq_avg: 100.0,
            lat_avg: 12.0,
            total_synapses: 150.0,
        },
        logs: vec![
            LogEntry {
                time: chrono::Local::now().format("%H:%M").to_string(),
                level: "ok".to_string(),
                msg: "Sistema online".to_string(),
            }
        ],
        
        // Métricas expandidas
        total_neurons: swap_stats.total as u64,
        active_neurons: swap_stats.ram as u64,
        total_synapses: swap_stats.synapses_totais as u64,
        active_synapses: swap_stats.synapses_ativas as u64,
        humor: (brain.ego.current_state.humor * 100.0) as i32,
        energia: (brain.ego.current_state.energia * 100.0) as u32,
        ultimo_pensamento: brain.ego.narrative_voice.pensamentos_recentes
            .back().cloned().unwrap_or_else(|| "processando...".to_string()),
        sensacao_corporal: brain.interoception.sentir(),
        num_habitos: brain.basal_ganglia.stats().num_habitos,
        alertness: brain.brainstem.stats().alertness,
        neurogenese_eventos: swap_stats.neurogenese_eventos,
        limite_fisico: swap_stats.limite_fisico,
        limite_biologico: swap_stats.limite_biologico,
    }
}