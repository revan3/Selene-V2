// src/websocket.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use warp::Filter;
use std::sync::Arc;
use tokio::sync::broadcast;
use serde::{Serialize, Deserialize};
use futures_util::{SinkExt, StreamExt, stream::{SplitSink, SplitStream}};
use crate::brain_zones::{frontal::FrontalLobe, occipital::OccipitalLobe, parietal::ParietalLobe};
use crate::brain_zones::RegionType;
use crate::config::Config;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NeuralStatus {
    pub regions: Vec<RegionStatus>,
    pub global: GlobalMetrics,
    pub logs: Vec<LogEntry>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RegionStatus {
    pub id: String,
    pub name: String,
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
    pub label: String,
    pub val: f32,
    pub cls: String,
    pub fill: String,
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

pub struct BrainState {
    // Aqui você coloca referências para seu cérebro
    pub frontal: Arc<FrontalLobe>,
    pub occipital: Arc<OccipitalLobe>,
    pub parietal: Arc<ParietalLobe>,
    // ... etc
}

impl BrainState {
    pub fn new() -> Self {
        // Nota: isso é temporário - depois você conecta com o cérebro real
        // Precisa de um Config aqui - isso será ajustado depois
        let config = crate::config::Config::new(crate::config::ModoOperacao::Humano);
        
        Self {
            frontal: Arc::new(FrontalLobe::new(1024, 0.2, 0.1, &config)),
            occipital: Arc::new(OccipitalLobe::new(1024, 0.2, &config)),
            parietal: Arc::new(ParietalLobe::new(1024, 0.2, &config)),
        }
    }
}

pub async fn start_websocket_server(brain: Arc<BrainState>) {
    let (tx, _) = broadcast::channel::<NeuralStatus>(100);
    
    // Servir arquivos estáticos
    let static_files = warp::path("static")
        .and(warp::fs::dir("./static"));
    
    // Servir index.html na raiz
    let index = warp::path::end()
        .and(warp::fs::file("./static/index.html"));
    
    // WebSocket
    let ws_route = warp::path("ws")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
            let tx = tx.clone();
            ws.on_upgrade(move |socket| handle_connection(socket, tx))
        });
    
    let routes = index.or(static_files).or(ws_route);
    
    println!("🌐 Servidor rodando em http://localhost:3030");
    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
}

async fn handle_connection(ws: warp::ws::WebSocket, _tx: broadcast::Sender<NeuralStatus>) {
    println!("   ✅ Cliente conectado ao WebSocket");
    
    // CORRIGIDO: adicionando tipos explícitos
    let (mut _ws_tx, mut ws_rx): (SplitSink<_, _>, SplitStream<_>) = ws.split();
    
    while let Some(result) = ws_rx.next().await {
        if let Ok(msg) = result {
            if let Ok(text) = msg.to_str() {
                println!("📩 Mensagem: {}", text);
            }
        }
    }
}