// src/websocket/mod.rs
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};
use warp::Filter;

// Re-exportar para o main.rs conseguir acessar
pub mod bridge;
pub mod server;

pub use bridge::{BrainState, NeuralStatus};

pub async fn start_websocket_server(brain_state: Arc<Mutex<BrainState>>) {
    // Canal de broadcast para enviar status neural para todos os clientes conectados
    let (tx, _) = broadcast::channel::<NeuralStatus>(100);

    // Clona antes dos closures que consomem por move
    let tx_telemetry = tx.clone();
    let brain_state_telemetry = brain_state.clone();

    // Rota WebSocket
    let ws_route = warp::path("selene")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
            // Clona o sender para esta conexão
            let tx = tx.clone();
            // Clona o Arc<Mutex<BrainState>> para esta conexão
            let brain = Arc::clone(&brain_state);

            ws.on_upgrade(move |socket| {
                // Cria um Receiver específico para esta conexão WebSocket
                let rx = tx.subscribe();
                server::handle_connection(socket, rx, brain)
            })
        });

    // Serve neural_interface.html na raiz (/)
    let index = warp::path::end().and(warp::fs::file("neural_interface.html"));

    // Serve selene_mobile_ui.html em /mobile
    let mobile = warp::path("mobile").and(warp::fs::file("selene_mobile_ui.html"));

    // Combina as rotas
    let routes = index.or(mobile).or(ws_route);

    println!("✨ Servidor Neural rodando em http://127.0.0.1:3030");
    println!("   → WebSocket em  ws://127.0.0.1:3030/selene");
    println!("   → Desktop       http://127.0.0.1:3030/");
    println!("   → Mobile        http://127.0.0.1:3030/mobile");

    // Task de telemetria: envia status do cérebro a cada ~500ms para todos os clientes
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(500));
        loop {
            interval.tick().await;

            // Tenta adquirir o lock sem bloquear forever (try_lock)
            if let Ok(brain_guard) = brain_state_telemetry.try_lock() {
                let status = bridge::collect_neural_status(&brain_guard).await;
                let _ = tx_telemetry.send(status);  // envia para todos os subscribers
            } else {
                // Opcional: log se estiver muito contended
                // eprintln!("Não consegui lock do brain state (contention)");
            }
        }
    });

    // Inicia o servidor Warp
    warp::serve(routes)
        .run(([127, 0, 0, 1], 3030))
        .await;
}