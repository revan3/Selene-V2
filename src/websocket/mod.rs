// src/websocket/mod.rs
use std::net::UdpSocket;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{broadcast, Mutex};
use warp::Filter;

/// Teto de conexões WebSocket simultâneas (defesa em profundidade contra
/// flood de conexões — cada conexão faz tokio::spawn de um handler).
/// O bind é 127.0.0.1 por padrão, então o vetor real só existe com SELENE_LAN=1;
/// ainda assim limitamos para nunca esgotar o executor por engano.
const MAX_CONEXOES_WS: usize = 64;

/// Contador global de conexões WebSocket ativas. Incrementado ao aceitar,
/// decrementado quando o handler retorna (conexão fechada).
static CONEXOES_WS_ATIVAS: AtomicUsize = AtomicUsize::new(0);

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

/// Rejeição de conexão sem token válido (V4.6.1 — segurança de rede).
#[derive(Debug)]
struct NaoAutorizado;
impl warp::reject::Reject for NaoAutorizado {}

/// Converte rejeições em respostas HTTP limpas (401 para token inválido).
async fn tratar_rejeicao(
    err: warp::Rejection,
) -> Result<impl warp::Reply, std::convert::Infallible> {
    use warp::http::StatusCode;
    if err.find::<NaoAutorizado>().is_some() {
        return Ok(warp::reply::with_status(
            "401 — token inválido ou ausente (defina ?token=…)",
            StatusCode::UNAUTHORIZED,
        ));
    }
    Ok(warp::reply::with_status("404 — não encontrado", StatusCode::NOT_FOUND))
}

pub async fn start_websocket_server(brain_state: Arc<Mutex<BrainState>>) {
    // Canal de broadcast para enviar status neural para todos os clientes conectados
    let (tx, _) = broadcast::channel::<NeuralStatus>(100);

    // Clona antes dos closures que consomem por move
    let tx_telemetry = tx.clone();
    let brain_state_telemetry = brain_state.clone();

    // ── Segurança de rede (V4.6.1) ──────────────────────────────────────────
    // Token opcional: se SELENE_TOKEN estiver definido, o /selene exige ?token=…
    let token = std::env::var("SELENE_TOKEN").ok().filter(|t| !t.is_empty());
    let token_ws = token.clone();

    // Rota WebSocket (com guarda de token antes do upgrade)
    let ws_route = warp::path("selene")
        .and(warp::query::<std::collections::HashMap<String, String>>())
        .and(warp::ws())
        .and_then(move |q: std::collections::HashMap<String, String>, ws: warp::ws::Ws| {
            let tx = tx.clone();
            let brain = Arc::clone(&brain_state);
            let token = token_ws.clone();
            async move {
                if let Some(expected) = &token {
                    let ok = q.get("token").map(|t| t == expected).unwrap_or(false);
                    if !ok {
                        return Err(warp::reject::custom(NaoAutorizado));
                    }
                }
                Ok::<_, warp::Rejection>(ws.on_upgrade(move |socket| {
                    let rx = tx.subscribe();
                    async move {
                        // Reserva um slot de forma atômica; se exceder o teto,
                        // devolve o slot e fecha a conexão sem gastar recursos.
                        let n = CONEXOES_WS_ATIVAS.fetch_add(1, Ordering::SeqCst);
                        if n >= MAX_CONEXOES_WS {
                            CONEXOES_WS_ATIVAS.fetch_sub(1, Ordering::SeqCst);
                            eprintln!("⚠️  [WS] Limite de {} conexões atingido — recusando nova conexão",
                                MAX_CONEXOES_WS);
                            return;
                        }
                        server::handle_connection(socket, rx, brain).await;
                        CONEXOES_WS_ATIVAS.fetch_sub(1, Ordering::SeqCst);
                    }
                }))
            }
        });

    // Serve neural_interface.html na raiz (/)
    let index = warp::path::end().and(warp::fs::file("neural_interface.html"));

    // Serve selene_mobile_ui.html em /mobile
    let mobile = warp::path("mobile").and(warp::fs::file("selene_mobile_ui.html"));

    // V4.6.1 — Visualização 3D dos neurônios (cor por tipo, por zona) em /neural
    let neural = warp::path("neural").and(warp::fs::file("selene_neural_viz.html"));

    // PWA: manifest.json e service worker (necessários para instalação no celular)
    let manifest = warp::path("manifest.json").and(warp::fs::file("selene_manifest.json"));
    let sw      = warp::path("sw.js").and(warp::fs::file("selene_sw.js"));

    // Serve arquivos estáticos do diretório de trabalho (d3.v7.min.js, etc.)
    // Deve ser o último da cadeia — só alcançado se nenhuma outra rota bater.
    let static_dir = warp::fs::dir(".");

    // Combina as rotas + handler de rejeição (401/404 limpos)
    let routes = index.or(mobile).or(neural).or(manifest).or(sw).or(ws_route).or(static_dir)
        .recover(tratar_rejeicao);

    // ── Bind seguro (V4.6.1) ────────────────────────────────────────────────
    // Padrão: 127.0.0.1 (só esta máquina). LAN só com SELENE_LAN=1 (e idealmente
    // SELENE_TOKEN). Antes ligava em 0.0.0.0 sem auth — exposto à rede toda.
    let lan = std::env::var("SELENE_LAN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let bind_ip: [u8; 4] = if lan { [0, 0, 0, 0] } else { [127, 0, 0, 1] };

    println!("✨ Servidor Neural em http://127.0.0.1:3030  (WebSocket: ws://127.0.0.1:3030/selene)");
    if token.is_some() {
        println!("   🔒 Token exigido: conecte com ?token=… (SELENE_TOKEN ativo)");
    } else {
        println!("   ⚠️  Sem token (defina SELENE_TOKEN para exigir autenticação)");
    }
    if lan {
        let local_ip = local_ip_address();
        if let Some(ip) = &local_ip {
            println!("   🌐 LAN ativa (SELENE_LAN=1) → http://{}:3030/mobile  ws://{}:3030/selene", ip, ip);
        }
        if token.is_none() {
            println!("   ‼️  LAN SEM TOKEN: qualquer dispositivo na rede pode controlar a Selene. Defina SELENE_TOKEN!");
        }
    } else {
        println!("   → LAN desligada. Para acesso pela rede: SELENE_LAN=1 (+ SELENE_TOKEN)");
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

    // Inicia o servidor Warp no endereço resolvido (127.0.0.1 por padrão).
    warp::serve(routes)
        .run((bind_ip, 3030))
        .await;
}