// src/websocket/server.rs (você já tem o esqueleto!)
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use warp::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};

pub async�a handle_connection(ws: WebSocket, telemetry: broadcast::Receiver<BrainSnapshot>) {
    let (mut ws_tx, mut ws_rx) = ws.split();
    let mut rx = telemetry;
    
    // Envia dados do cérebro a cada 100ms
    while let Ok(snapshot) = rx.recv().await {
        let json = serde_json::to_string(&snapshot).unwrap();
        if ws_tx.send(Message::text(json)).await.is_err() {
            break;
        }
    }
}