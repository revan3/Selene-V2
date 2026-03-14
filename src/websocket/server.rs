// src/websocket/server.rs
// Responsável pelo gerenciamento das conexões WebSocket e transmissão de dados neurais

#![allow(unused_imports, unused_variables, dead_code)]

use warp::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::broadcast;
use crate::websocket::bridge::NeuralStatus;

/// Gerencia a conexão de cada cliente (browser) que se conecta à Selene
pub async fn handle_connection(ws: WebSocket, mut telemetry_rx: broadcast::Receiver<NeuralStatus>) {
    // Divide o WebSocket em transmissor (tx) e receptor (rx)
    let (mut ws_tx, mut ws_rx) = ws.split();
    
    println!("   ✅ Conexão WebSocket estabelecida.");

    // Loop de processamento assíncrono
    loop {
        tokio::select! {
            // 1. Monitora o canal de broadcast da telemetria vindo do bridge.rs
            result = telemetry_rx.recv() => {
                match result {
                    // Caso receba um novo snapshot do cérebro com sucesso
                    Ok(status) => {
                        // Serializa o struct NeuralStatus para uma string JSON
                        if let Ok(json) = serde_json::to_string(&status) {
                            // Tenta enviar para o navegador
                            if ws_tx.send(Message::text(json)).await.is_err() {
                                // Se o envio falhar (browser fechou), sai do loop
                                break;
                            }
                        }
                    },
                    // Caso o consumidor esteja lento (Lagged), apenas ignora e pega o próximo
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        log::warn!("⚠️  WebSocket detectou lag de {} mensagens. Sincronizando...", n);
                        continue;
                    },
                    // Erro crítico no canal (Closed)
                    Err(_) => {
                        eprintln!("❌ Canal de telemetria fechado no servidor.");
                        break;
                    }
                }
            }

            // 2. Monitora mensagens vindas do Browser (para futuros comandos)
            Some(result) = ws_rx.next() => {
                match result {
                    Ok(msg) => {
                        if msg.is_close() {
                            break;
                        }
                        // Aqui você pode adicionar lógica para processar comandos do browser
                        // Ex: Alterar níveis de dopamina via interface
                    }
                    Err(e) => {
                        log::error!("Erro na leitura do WebSocket: {}", e);
                        break;
                    }
                }
            }
        }
    }

    println!("   ❌ Conexão WebSocket encerrada.");
}