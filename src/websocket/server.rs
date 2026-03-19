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
                                match json["action"].as_str() {
                                Some("shutdown") => {
                                    println!("🛑 [SISTEMA] Shutdown solicitado pela interface neural.");
                                    brain.lock().await.shutdown_requested = true;
                                    let ack = r#"{"event":"shutdown_ack","msg":"Iniciando desligamento gracioso..."}"#;
                                    let _ = ws_tx.send(Message::text(ack)).await;
                                }

                                Some("toggle_sensor") => {
                                    let sensor  = json["sensor"].as_str().unwrap_or("");
                                    let active  = json["active"].as_bool().unwrap_or(false);
                                    let state   = brain.lock().await;
                                    match sensor {
                                        "audio" => state.sensor_flags.set_audio(active),
                                        "video" => state.sensor_flags.set_video(active),
                                        _ => log::warn!("[WS] Sensor desconhecido: {}", sensor),
                                    }
                                    let ack = serde_json::json!({
                                        "event": "sensor_ack",
                                        "sensor": sensor,
                                        "active": active
                                    }).to_string();
                                    let _ = ws_tx.send(Message::text(ack)).await;
                                    println!("[SENSOR] {} → {}", sensor, if active { "ATIVO" } else { "INATIVO" });
                                }

                                Some("chat") => {
                                    // Mensagem de chat vinda da interface mobile/desktop
                                    let mensagem = json["message"].as_str().unwrap_or("").to_string();
                                    if !mensagem.is_empty() {
                                        println!("💬 [CHAT] Recebido: {}", mensagem);
                                        let state = brain.lock().await;
                                        let (dopa, sero, _nor) = state.neurotransmissores;
                                        let (_temp, _ram) = state.hardware;
                                        let (step, alerta, emocao) = state.atividade;
                                        // Resposta introspectiva baseada no estado atual
                                        let reply = format!(
                                            "[Tick #{step}] Processando: «{msg}» — \
                                            Estado emocional: {emo:.2} | Alerta: {al:.2} | \
                                            Dopamina: {d:.3} | Serotonina: {s:.3}",
                                            step = step, msg = mensagem,
                                            emo = emocao, al = alerta, d = dopa, s = sero
                                        );
                                        let resp = serde_json::json!({
                                            "event": "chat_reply",
                                            "message": reply,
                                            "emotion": emocao,
                                            "arousal": alerta,
                                        }).to_string();
                                        let _ = ws_tx.send(Message::text(resp)).await;
                                    }
                                }

                                Some("train") => {
                                    // Inicia ciclo de aprendizado via comando externo
                                    let epochs   = json["epochs"].as_u64().unwrap_or(6) as u32;
                                    let ltp      = json["ltp"].as_f64().unwrap_or(0.012) as f32;
                                    let ltd      = json["ltd"].as_f64().unwrap_or(0.003) as f32;
                                    println!("🧠 [TRAIN] Iniciando treino — epochs={} ltp={} ltd={}", epochs, ltp, ltd);

                                    // Emite progresso simulado por época
                                    for ep in 1..=epochs {
                                        let loss = 1.2 - ep as f32 * 0.08 + 0.02 * (ep as f32 * 0.7).sin();
                                        let acc  = (0.5 + ep as f32 * 0.04).min(0.97);
                                        let ev = serde_json::json!({
                                            "epoch": ep,
                                            "epochs": epochs,
                                            "loss": loss,
                                            "acc": acc,
                                        }).to_string();
                                        let _ = ws_tx.send(Message::text(ev)).await;
                                        tokio::time::sleep(tokio::time::Duration::from_millis(400)).await;
                                    }

                                    let done = serde_json::json!({
                                        "status": "done",
                                        "message": format!("Treino concluído: {} épocas", epochs)
                                    }).to_string();
                                    let _ = ws_tx.send(Message::text(done)).await;
                                    println!("✅ [TRAIN] Treino finalizado ({} épocas)", epochs);
                                }

                                Some("run_script") => {
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

                                            let _ = tokio::process::Command::new("python")
                                                .arg(&full_path)
                                                .spawn()
                                                .map_err(|e| {
                                                    log::error!("Falha ao executar {}: {}", full_path, e);
                                                });
                                        } else {
                                            log::warn!("Tentativa de executar script NÃO permitido: {}", script_name);
                                            let error_msg = format!(
                                                r#"{{"error": "Script não permitido. Use apenas: {}, {} ou {}"}}"#,
                                                ALLOWED_SCRIPTS[0], ALLOWED_SCRIPTS[1], ALLOWED_SCRIPTS[2]
                                            );
                                            let _ = ws_tx.send(Message::text(error_msg)).await;
                                        }
                                    }
                                }

                                _ => {
                                    log::debug!("[WS] Ação desconhecida: {:?}", json["action"]);
                                }
                            } // fim match json["action"]
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