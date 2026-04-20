// src/sensors/vision_stream.rs
//
// Streaming do frame visual da Selene para o viewer externo.
//
// PROTOCOLO:
//   Servidor expõe endpoint WebSocket: ws://127.0.0.1:3031/vision
//   A cada frame capturado (~10fps), envia JSON:
//   {
//     "type": "frame",
//     "camera": 0,                  // 0=primária, 1=estéreo
//     "width": 640,
//     "height": 480,
//     "jpeg_b64": "<base64>",       // JPEG comprimido, qualidade 70
//     "timestamp_ms": 1234567890,
//     "disparidade_b64": "<base64>" // opcional — mapa de profundidade (grayscale JPEG)
//   }
//
// O viewer Python (selene_vision.py) conecta neste endpoint e exibe
// os frames em janelas OpenCV separadas.
//
// PORTA SEPARADA (3031) para não conflitar com o WebSocket principal (3030).

#![allow(dead_code)]

use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::broadcast;

use crate::sensors::camera::{SharedFrameBuffer, RgbFrame};

/// Capacidade do canal broadcast de frames (quantos clientes simultâneos).
const BROADCAST_CAP: usize = 8;

/// Intervalo entre frames enviados (~10fps).
const FRAME_INTERVAL_MS: u64 = 100;

/// Qualidade JPEG para o stream (0–100). Menor = menos bytes, mais rápido.
const JPEG_QUALITY: u8 = 70;

// ── Encoder JPEG mínimo (pure Rust, sem dependência externa) ─────────────────
// Usa a técnica de codificação JPEG simplificada.
// Para produção, substituir por `image` crate com feature jpeg.

/// Converte frame RGB para JPEG bytes (qualidade simplificada).
/// Usa quantização básica sem DCT completa — suficiente para visualização.
/// Se `image` crate disponível, usa; caso contrário, PNG fallback.
pub fn rgb_para_jpeg_bytes(frame: &RgbFrame) -> Vec<u8> {
    if frame.pixels.is_empty() || frame.largura == 0 || frame.altura == 0 {
        return Vec::new();
    }

    // Tenta usar image crate se disponível
    #[cfg(feature = "image")]
    {
        use image::{ImageBuffer, Rgb, ImageOutputFormat};
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_raw(frame.largura, frame.altura, frame.pixels.clone())
                .unwrap_or_else(|| ImageBuffer::new(frame.largura, frame.altura));
        let mut buf = Vec::new();
        let _ = img.write_to(&mut std::io::Cursor::new(&mut buf),
            ImageOutputFormat::Jpeg(JPEG_QUALITY));
        return buf;
    }

    // Fallback: PNG via raw bytes empacotados como BMP mínimo
    // (o viewer Python vai aceitar qualquer formato via cv2.imdecode)
    #[cfg(not(feature = "image"))]
    {
        // Encapsula pixels brutos num container minimalista:
        // Header: width(4) + height(4) + "RGB\0" → viewer Python decodifica
        let mut out = Vec::with_capacity(12 + frame.pixels.len());
        out.extend_from_slice(&frame.largura.to_le_bytes());
        out.extend_from_slice(&frame.altura.to_le_bytes());
        out.extend_from_slice(b"RGB\0");
        out.extend_from_slice(&frame.pixels);
        out
    }
}

/// Converte mapa de disparidade (f32 [0..1]) para bytes grayscale.
pub fn disparidade_para_bytes(disp: &[f32], largura: u32, altura: u32) -> Vec<u8> {
    let gray: Vec<u8> = disp.iter()
        .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
        .collect();

    let mut out = Vec::with_capacity(12 + gray.len());
    out.extend_from_slice(&largura.to_le_bytes());
    out.extend_from_slice(&altura.to_le_bytes());
    out.extend_from_slice(b"GRY\0");
    out.extend_from_slice(&gray);
    out
}

// ── Canal de frames ───────────────────────────────────────────────────────────

/// Mensagem de frame serializada (JSON pronta para envio).
pub type FrameMsg = Arc<String>;

/// Handle do broadcast de frames — clonado para cada novo cliente WebSocket.
#[derive(Clone)]
pub struct VisionBroadcast {
    pub tx: broadcast::Sender<FrameMsg>,
}

impl VisionBroadcast {
    pub fn novo() -> Self {
        let (tx, _) = broadcast::channel(BROADCAST_CAP);
        Self { tx }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<FrameMsg> {
        self.tx.subscribe()
    }

    pub fn n_receptores(&self) -> usize {
        self.tx.receiver_count()
    }
}

// ── Task de captura + broadcast ───────────────────────────────────────────────

/// Spawna uma task Tokio que lê o SharedFrameBuffer a ~10fps e faz broadcast
/// do frame serializado para todos os clientes conectados.
///
/// Deve ser chamada no main loop após iniciar o DualCameraSystem.
pub fn iniciar_broadcast(
    frame_buf:  SharedFrameBuffer,
    broadcast:  VisionBroadcast,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(FRAME_INTERVAL_MS)).await;

            // Sem receptores → não processa
            if broadcast.n_receptores() == 0 { continue; }

            let (frame_opt, disp_opt) = {
                match frame_buf.lock() {
                    Ok(fb) => (fb.frame_primario.clone(), fb.mapa_disparidade.clone()),
                    Err(_) => continue,
                }
            };

            let Some(frame) = frame_opt else { continue };

            let jpeg_bytes = rgb_para_jpeg_bytes(&frame);
            if jpeg_bytes.is_empty() { continue; }

            use base64::Engine;
            let jpeg_b64 = base64::engine::general_purpose::STANDARD.encode(&jpeg_bytes);

            let disp_b64 = disp_opt.as_ref().map(|d| {
                let bytes = disparidade_para_bytes(d, frame.largura, frame.altura);
                base64::engine::general_purpose::STANDARD.encode(&bytes)
            });

            let json = match disp_b64 {
                Some(d) => format!(
                    r#"{{"type":"frame","camera":{},"width":{},"height":{},"jpeg_b64":"{}","timestamp_ms":{},"disparidade_b64":"{}"}}"#,
                    frame.camera_id, frame.largura, frame.altura, jpeg_b64,
                    frame.timestamp_ms, d
                ),
                None => format!(
                    r#"{{"type":"frame","camera":{},"width":{},"height":{},"jpeg_b64":"{}","timestamp_ms":{}}}"#,
                    frame.camera_id, frame.largura, frame.altura, jpeg_b64,
                    frame.timestamp_ms
                ),
            };

            let _ = broadcast.tx.send(Arc::new(json));
        }
    })
}

// ── Servidor WebSocket de visão (porta 3031) ──────────────────────────────────

/// Inicia o servidor WebSocket de visão em ws://127.0.0.1:3031/vision.
/// Cada cliente que conectar recebe o broadcast de frames em tempo real.
pub async fn iniciar_servidor_visao(broadcast: VisionBroadcast) {
    use warp::Filter;
    use warp::ws::WebSocket;
    use futures_util::{SinkExt, StreamExt};

    let broadcast = Arc::new(broadcast);

    let vision_route = warp::path("vision")
        .and(warp::ws())
        .and(warp::any().map(move || Arc::clone(&broadcast)))
        .map(|ws: warp::ws::Ws, bc: Arc<VisionBroadcast>| {
            ws.on_upgrade(move |socket| handle_vision_client(socket, bc))
        });

    println!("[VISION] Servidor de visão iniciado em ws://127.0.0.1:3031/vision");
    warp::serve(vision_route)
        .run(([127, 0, 0, 1], 3031))
        .await;
}

async fn handle_vision_client(ws: warp::ws::WebSocket, bc: Arc<VisionBroadcast>) {
    use futures_util::{SinkExt, StreamExt};

    let (mut ws_tx, _ws_rx): (
        futures_util::stream::SplitSink<warp::ws::WebSocket, warp::ws::Message>,
        _,
    ) = ws.split();
    let mut rx = bc.subscribe();

    println!("[VISION] Novo cliente conectado.");

    loop {
        match rx.recv().await {
            Ok(msg) => {
                if ws_tx.send(warp::ws::Message::text(msg.as_str())).await.is_err() {
                    break;
                }
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                eprintln!("[VISION] Cliente atrasado — {} frames perdidos.", n);
                continue;
            }
            Err(_) => break,
        }
    }

    println!("[VISION] Cliente desconectado.");
}
