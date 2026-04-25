// src/bin/neural_viewer.rs
// Visualizador nativo do grafo semântico da Selene.
// Usa egui + egui_graphs + petgraph para renderizar conceitos e sinapses em tempo real.
// Compila apenas com: cargo run --bin neural_viewer --features gui
//
// Protocolo: conecta via WebSocket ws://127.0.0.1:3030, envia {"action":"vocab_snapshot"},
// recebe JSON com lista de nós e arestas, reconstrui o grafo e repinta a cada frame.

#![cfg(feature = "gui")]

use eframe::egui;
use egui_graphs::{Graph, GraphView, SettingsInteraction, SettingsStyle};
use petgraph::stable_graph::{NodeIndex, StableGraph};
use petgraph::Directed;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use futures_util::{SinkExt, StreamExt};
use tokio::runtime::Runtime;

// ── Tipos de nó e aresta para o petgraph ──────────────────────────────────
#[derive(Clone, Debug)]
struct ConceptNode {
    label: String,
    valence: f32,
}

#[derive(Clone, Debug)]
struct SynapseEdge {
    weight: f32,
}

type SemanticGraph = StableGraph<ConceptNode, SynapseEdge, Directed>;

// ── Estado compartilhado entre thread WS e thread GUI ─────────────────────
#[derive(Default)]
struct SharedState {
    nodes: Vec<(String, f32)>,                       // (palavra, valência)
    edges: Vec<(String, String, f32)>,               // (pre, post, peso)
    updated: bool,
}

struct NeuralViewerApp {
    shared: Arc<Mutex<SharedState>>,
    g: Graph<ConceptNode, SynapseEdge, Directed>,
    index_map: HashMap<String, NodeIndex>,
    rt: Runtime,
}

impl NeuralViewerApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let shared = Arc::new(Mutex::new(SharedState::default()));
        let shared_ws = Arc::clone(&shared);

        let rt = Runtime::new().expect("tokio runtime");
        rt.spawn(async move {
            ws_loop(shared_ws).await;
        });

        let mut app = Self {
            shared,
            g: Graph::from(&StableGraph::default()),
            index_map: HashMap::new(),
            rt,
        };
        app
    }

    fn rebuild_graph(&mut self, nodes: &[(String, f32)], edges: &[(String, String, f32)]) {
        let mut sg: SemanticGraph = StableGraph::new();
        let mut idx_map: HashMap<String, NodeIndex> = HashMap::new();

        for (label, valence) in nodes {
            let idx = sg.add_node(ConceptNode { label: label.clone(), valence: *valence });
            idx_map.insert(label.clone(), idx);
        }
        for (pre, post, weight) in edges {
            if let (Some(&a), Some(&b)) = (idx_map.get(pre), idx_map.get(post)) {
                sg.add_edge(a, b, SynapseEdge { weight: *weight });
            }
        }

        self.g = Graph::from(&sg);
        self.index_map = idx_map;
    }
}

impl eframe::App for NeuralViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Verifica se há dados novos do WebSocket
        let (nodes, edges) = {
            let mut state = self.shared.lock().unwrap();
            if state.updated {
                state.updated = false;
                (state.nodes.clone(), state.edges.clone())
            } else {
                (vec![], vec![])
            }
        };
        if !nodes.is_empty() {
            self.rebuild_graph(&nodes, &edges);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Selene — Grafo Semântico");
            ui.label(format!("Nós: {} | Arestas: {}", self.g.node_count(), self.g.edge_count()));
            ui.separator();

            let interaction = SettingsInteraction::default()
                .with_dragging_enabled(true)
                .with_node_clicking_enabled(true);
            let style = SettingsStyle::default().with_labels_always(true);

            ui.add(&mut GraphView::<_, _, _, _, egui_graphs::DefaultNodeShape, egui_graphs::DefaultEdgeShape>::new(&mut self.g)
                .with_interactions(&interaction)
                .with_styles(&style));
        });

        ctx.request_repaint_after(std::time::Duration::from_millis(500));
    }
}

async fn ws_loop(shared: Arc<Mutex<SharedState>>) {
    loop {
        match tokio_tungstenite::connect_async("ws://127.0.0.1:3030").await {
            Ok((mut ws, _)) => {
                let request = serde_json::json!({ "action": "vocab_snapshot" }).to_string();
                let _ = ws.send(tokio_tungstenite::tungstenite::Message::Text(request)).await;

                while let Some(Ok(msg)) = ws.next().await {
                    if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                            let nodes: Vec<(String, f32)> = v["nodes"].as_array()
                                .unwrap_or(&vec![])
                                .iter()
                                .filter_map(|n| {
                                    let label = n["label"].as_str()?.to_string();
                                    let val = n["valence"].as_f64().unwrap_or(0.0) as f32;
                                    Some((label, val))
                                })
                                .collect();
                            let edges: Vec<(String, String, f32)> = v["edges"].as_array()
                                .unwrap_or(&vec![])
                                .iter()
                                .filter_map(|e| {
                                    let pre = e["pre"].as_str()?.to_string();
                                    let post = e["post"].as_str()?.to_string();
                                    let w = e["weight"].as_f64().unwrap_or(0.0) as f32;
                                    Some((pre, post, w))
                                })
                                .collect();
                            if !nodes.is_empty() {
                                let mut state = shared.lock().unwrap();
                                state.nodes = nodes;
                                state.edges = edges;
                                state.updated = true;
                            }
                        }
                        // Pede próximo snapshot após receber
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                        let request = serde_json::json!({ "action": "vocab_snapshot" }).to_string();
                        let _ = ws.send(tokio_tungstenite::tungstenite::Message::Text(request)).await;
                    }
                }
            }
            Err(e) => {
                eprintln!("[NeuralViewer] WS conexão falhou: {}. Retentando em 3s...", e);
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            }
        }
    }
}

fn main() {
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(1200.0, 800.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Selene — Grafo Semântico Neural",
        options,
        Box::new(|cc| Box::new(NeuralViewerApp::new(cc))),
    ).expect("eframe falhou");
}
