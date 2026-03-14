// src/websocket/mod.rs
use std::sync::Arc;
use tokio::sync::Mutex;
use warp::Filter;

// Re-exportar para o main.rs conseguir ver
pub mod bridge;
pub mod server;

pub use bridge::{BrainState, NeuralStatus};

pub async fn start_websocket_server(brain_state: Arc<Mutex<BrainState>>) {
    let (tx, _) = tokio::sync::broadcast::channel::<NeuralStatus>(100);
    let tx_for_ws = tx.clone();
    let brain_for_ws = Arc::clone(&brain_state);

    // Rota WebSocket
    let ws_route = warp::path("selene")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
            let tx = tx_for_ws.clone();
            let brain = Arc::clone(&brain_for_ws);
            ws.on_upgrade(move |socket| server::handle_connection(socket, tx, brain))
        });

    // Rota Estática (Interface)
    let static_files = warp::fs::dir("interface");
    let index = warp::path::end().and(warp::fs::file("interface/index.html"));

    let routes = index.or(ws_route).or(static_files);

    println!("✨ Servidor Neural em http://127.0.0.1:3030");
    
    // Loop de Telemetria (Envia dados para o HTML a cada 500ms)
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(500));
        loop {
            interval.tick().await;
            if let Ok(brain) = brain_state.try_lock() {
                let status = bridge::collect_neural_status(&brain).await;
                let _ = tx.send(status);
            }
        }
    });

    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
}