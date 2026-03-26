// src/websocket/mod.rs
use std::net::UdpSocket;
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};
use warp::Filter;

/// Descobre o IP local da máquina abrindo uma conexão UDP fictícia (não envia dados)
fn local_ip_address() -> Option<String> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    let addr = socket.local_addr().ok()?;
    Some(addr.ip().to_string())
}

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

    // PWA: manifest.json e service worker (necessários para instalação no celular)
    let manifest = warp::path("manifest.json").and(warp::fs::file("selene_manifest.json"));
    let sw      = warp::path("sw.js").and(warp::fs::file("selene_sw.js"));

    // Combina as rotas
    let routes = index.or(mobile).or(manifest).or(sw).or(ws_route);

    // Descobre o IP local para exibir no console
    let local_ip = local_ip_address();

    println!("✨ Servidor Neural rodando em http://127.0.0.1:3030");
    println!("   → WebSocket em  ws://127.0.0.1:3030/selene");
    println!("   → Desktop       http://127.0.0.1:3030/");
    println!("   → Mobile        http://127.0.0.1:3030/mobile");
    if let Some(ip) = &local_ip {
        println!("   → Celular (LAN) http://{}:3030/mobile", ip);
        println!("   → WebSocket LAN ws://{}:3030/selene", ip);
    }

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

    // Inicia o servidor Warp (0.0.0.0 = todas as interfaces — permite acesso pelo celular na mesma rede Wi-Fi)
    warp::serve(routes)
        .run(([0, 0, 0, 0], 3030))
        .await;
}