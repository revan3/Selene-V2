// src/websocket/server.rs
// Responsável pelo gerenciamento das conexões WebSocket e transmissão de dados neurais

#![allow(unused_imports, unused_variables, dead_code)]

use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};

use warp::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};

use crate::websocket::bridge::NeuralStatus;
use serde_json::Value;

use crate::websocket::bridge::BrainState;

/// Gerencia a conexão de cada cliente (browser) que se conecta à Selene
pub async fn handle_connection(
    ws: WebSocket,
    mut telemetry_rx: broadcast::Receiver<NeuralStatus>,
    brain: Arc<Mutex<BrainState>>,  // ← terceiro parâmetro adicionado (Opção B)
) {
    // Divide o WebSocket em transmissor (tx) e receptor (rx)
    let (mut ws_tx, mut ws_rx) = ws.split();

    println!("   ✅ Conexão WebSocket estabelecida.");

    loop {
        tokio::select! {
            // 1. Monitora o canal de broadcast da telemetria vinda do bridge.rs
            result = telemetry_rx.recv() => {
                match result {
                    Ok(status) => {
                        // Serializa o struct NeuralStatus para JSON
                        if let Ok(json) = serde_json::to_string(&status) {
                            // Tenta enviar para o navegador
                            if ws_tx.send(Message::text(json)).await.is_err() {
                                // Cliente provavelmente desconectou
                                break;
                            }
                        }
                    }

                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        log::warn!("⚠️ WebSocket detectou lag de {} mensagens. Continuando...", n);
                        // Continua recebendo sem quebrar o loop
                    }

                    Err(_) => {
                        // Canal fechado (sender foi dropado)
                        eprintln!("❌ Canal de telemetria foi fechado.");
                        break;
                    }
                }
            }

            // 2. Monitora mensagens enviadas pelo browser
            Some(result) = ws_rx.next() => {
                match result {
                    Ok(msg) => {
                        if let Ok(text) = msg.to_str() {
                            // Tenta parsear como JSON (para comandos estruturados)
                            if let Ok(json) = serde_json::from_str::<Value>(text) {
                                if json["action"].as_str() == Some("run_script") {
                                    if let Some(script_name) = json["script"].as_str() {
                                        // Lista branca: APENAS esses 3 scripts são permitidos
                                        const ALLOWED_SCRIPTS: [&str; 3] = [
                                            "generate_lexicon.py",
                                            "selene_exam.py",
                                            "selene_tutor.py",
                                        ];

                                        if ALLOWED_SCRIPTS.contains(&script_name) {
                                            let full_path = format!("scripts/{}", script_name);
                                            println!("🚀 [SISTEMA] Executando script permitido: {}", full_path);

                                            // Executa de forma assíncrona (não bloqueia o ws)
                                            let _ = tokio::process::Command::new("python")
                                                .arg(&full_path)
                                                .spawn()
                                                .map_err(|e| {
                                                    log::error!("Falha ao executar {}: {}", full_path, e);
                                                });
                                        } else {
                                            log::warn!(
                                                "Tentativa de executar script NÃO permitido: {}",
                                                script_name
                                            );

                                            // Opcional: avisa o cliente
                                            let error_msg = format!(
                                                r#"{{"error": "Script não permitido. Use apenas: {}, {} ou {}"}}"#,
                                                ALLOWED_SCRIPTS[0], ALLOWED_SCRIPTS[1], ALLOWED_SCRIPTS[2]
                                            );
                                            let _ = ws_tx.send(Message::text(error_msg)).await;
                                        }
                                    }
                                }
                            } else {
                                // Mensagem simples de chat (não é JSON)
                                println!("💬 [CHAT] Selene recebeu: {}", text);
                            }
                        }
                    }

                    Err(e) => {
                        log::error!("Erro ao ler mensagem do WebSocket: {}", e);
                        break;
                    }
                }
            }
        }
    }

    println!("   ❌ Conexão WebSocket encerrada.");
}