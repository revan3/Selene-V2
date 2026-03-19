// src/main.rs
// Arquivo principal: Inicia o programa e integra todos os módulos da Selene 2.0

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

// Declaração de módulos
mod synaptic_core;
mod brain_zones;
mod sensors;
mod storage;
mod neurochem;
mod sleep_manager;
mod config;
mod sleep_cycle;
mod websocket;
mod compressor;
mod ego;
mod thalamus;
mod interoception;
mod basal_ganglia;
mod brainstem;
mod learning;

// Imports necessários
use std::sync::mpsc::{channel, Sender, Receiver};
use std::{thread, panic};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::path::Path;
use std::fs::File;
use std::io::Write;
use tokio::sync::Mutex as TokioMutex;
use warp::Filter;

// Imports para Log e Debug
use simplelog::*;

// Imports dos módulos da Selene
use crate::brain_zones::RegionType;
use crate::config::{Config, ModoOperacao};
use crate::sleep_cycle::CicloSono;
use crate::websocket::bridge::{BrainState, NeuralStatus};

// Imports dos lobos
use brain_zones::{
    frontal::FrontalLobe, 
    occipital::OccipitalLobe, 
    parietal::ParietalLobe,
    temporal::TemporalLobe, 
    limbic::LimbicSystem, 
    hippocampus::HippocampusV2 as Hippocampus,
    corpus_callosum::CorpusCallosum,
};

// Imports dos novos módulos
use compressor::salient::SalientCompressor;
use ego::Ego;
use thalamus::Thalamus;
use interoception::Interoception;
use basal_ganglia::BasalGanglia;
use brainstem::Brainstem;
use storage::swap_manager::SwapManager;

// Imports dos sensores
use sensors::camera::VisualTransducer;
use sensors::audio;
use sensors::hardware::HardwareSensor;
use sensors::SensorFlags;

// Imports de storage
use storage::{BrainStorage, NeuralEnactiveMemory};
use storage::memory_tier::MemoryTier;
use storage::memory_graph::MemoryTierV2;

// Chunking engine
use learning::chunking::ChunkingEngine;

// Outros imports
use neurochem::NeuroChem;
use sleep_manager::SleepManagerV2 as SleepManager;

// ================== CONSTANTES ==================
const SWAP_THRESHOLD_SECONDS: u64 = 3600;
const COMPRESSOR_MAX_POINTS: usize = 16;

/// Limite de recursos do sistema que a Selene pode consumir (70%)
const LIMITE_RECURSO_PCT: f32 = 0.70;
/// Bytes aproximados por neurônio em RAM (struct + sinapses)
const BYTES_POR_NEURONIO: usize = 256;
/// Frequência mínima em modo ocioso (Hz)
const FREQ_OCIOSA_HZ: u64 = 5;
/// Frequência máxima em plena atividade (Hz)
const FREQ_ATIVA_HZ: u64 = 200;
/// Ticks consecutivos abaixo do threshold para entrar em modo ocioso
const TICKS_PARA_OCIOSO: u32 = 60;
/// Nível mínimo de sinal para considerar "há atividade"
const ATIVIDADE_MINIMA: f32 = 0.015;

/// Calcula o número máximo de neurônios com base em 70% da RAM disponível.
fn calcular_max_neurons() -> usize {
    use sysinfo::System;
    let mut sys = System::new_all();
    sys.refresh_memory();
    let ram_total = sys.total_memory() as f64;
    let ram_disponivel = sys.available_memory() as f64;
    // Usa o menor entre 70% do total e 95% do disponível
    let limite = (ram_total * LIMITE_RECURSO_PCT as f64)
        .min(ram_disponivel * 0.95) as usize;
    let max_neurons = limite / BYTES_POR_NEURONIO;
    println!("🧬 RAM total: {:.1}GB | Disponível: {:.1}GB | Limite Selene (70%): {:.1}GB",
        ram_total / 1e9, ram_disponivel / 1e9,
        limite as f64 / 1e9);
    println!("🧬 Neurônios máximos calculados: {}", max_neurons);
    max_neurons.max(10_000)  // mínimo de 10k neurônios
}

// ================== FUNÇÃO PRINCIPAL ==================
fn main() {
    // 1. Inicializa o Sistema de Logs
    let _ = CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::Info, ConfigBuilder::new().build(), TerminalMode::Mixed, ColorChoice::Auto),
            WriteLogger::new(LevelFilter::Debug, ConfigBuilder::new().build(), File::create("selene_debug.log").unwrap()),
        ]
    );

    // 2. Hook de Pânico
    panic::set_hook(Box::new(|panic_info| {
        unsafe { windows::Win32::Media::timeEndPeriod(1) };
        
        let path_memoria_ativa = "F:/Selene/Memory_Active";
        let path_memoria_fria = "D:/Selene_Archive";
        let path_backup = "D:/Selene_Backup_RAM";

        let _ = std::fs::create_dir_all(path_memoria_ativa);
        let _ = std::fs::create_dir_all(path_memoria_fria);
        let _ = std::fs::create_dir_all(path_backup);

        let mut file = File::create("selene_crash_report.txt").unwrap_or_else(|_| panic!("Não foi possível criar crash report"));

        let message = panic_info.payload()
            .downcast_ref::<&str>().map(|s| *s)
            .or_else(|| panic_info.payload().downcast_ref::<String>().map(|s| s.as_str()))
            .unwrap_or("Causa do pânico desconhecida");

        let (loc_file, loc_line) = panic_info.location()
            .map(|l| (l.file().to_string(), l.line()))
            .unwrap_or_else(|| ("unknown".to_string(), 0));

        let report = format!(
            "========================================\n\
             🧠 SELENE BRAIN CRASH REPORT\n\
             ========================================\n\
             Data/Hora: {:?}\n\
             Erro: {}\n\
             Local: {}:{}\n\n\
             Possível causa: Verifique se o arquivo NVME ou o Banco de Dados está acessível.\n\
             ========================================\n",
            Instant::now(), message, loc_file, loc_line
        );

        let _ = file.write_all(report.as_bytes());
        eprintln!("\n❌ CRASH DETECTADO: Relatório salvo em 'selene_crash_report.txt'");
    }));

    // 3. Inicia o Runtime Tokio
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async_main());
}

// ================== FUNÇÃO PRINCIPAL ASSÍNCRONA ==================
async fn async_main() {
    let config = Config::new(ModoOperacao::Boost200);
    let dt = config.dt_simulacao;
    let n_neurons = 1024usize;
    
    println!("\n{}", "=".repeat(60));
    println!("🧠 SELENE BRAIN 2.0 - SISTEMA NEURAL BIO-INSPIRADO");
    println!("{}", "=".repeat(60));
    println!("📊 Configuração:");
    println!("   - Modo: {:?}", config.modo);
    println!("   - Frequência: {} Hz", config.frequencia_base_hz);
    println!("   - Energia: {}W", config.energia_watts);
    println!("   - Taxa aprendizado: {}", config.taxa_aprendizado);
    println!("   - Compressão: {} pontos/spike", COMPRESSOR_MAX_POINTS);

    // --- 1. SETUP DE HARDWARE E QUÍMICA ---
    println!("\n🔧 Inicializando hardware...");
    let sensor = match HardwareSensor::new() {
        Ok(s) => Arc::new(TokioMutex::new(s)),
        Err(e) => {
            println!("⚠️  Erro ao acessar sensores: {}. Usando simulação.", e);
            Arc::new(TokioMutex::new(HardwareSensor::dummy()))
        }
    };
    
    let mut neuro = NeuroChem::new();

    // --- 2. SETUP DE STORAGE ---
    println!("💾 Inicializando banco de memória...");
    let storage_db = match BrainStorage::new().await {
        Ok(s) => Arc::new(s),
        Err(e) => {
            println!("⚠️  Erro ao iniciar DB: {}. Modo memória apenas.", e);
            Arc::new(BrainStorage::dummy())
        }
    };

    // --- 3. HIERARQUIA DE MEMÓRIA ---
    println!("📀 Configurando memória em camadas...");
    let memory_tier = match MemoryTier::new(
        Arc::clone(&storage_db),
        Path::new("nvme_buffer.bin"),
        n_neurons
    ).await {
        Ok(m) => Arc::new(TokioMutex::new(m)),
        Err(e) => {
            println!("⚠️  Erro ao criar MemoryTier: {}. Usando dummy.", e);
            Arc::new(TokioMutex::new(MemoryTier::dummy()))
        }
    };

    // --- 4. SETUP DO SWAP MANAGER ---
    println!("🧬 Inicializando Swap Manager com neurogênese...");
    let max_neurons_dinamico = calcular_max_neurons();
    let swap_manager = Arc::new(TokioMutex::new(
        SwapManager::new(max_neurons_dinamico, SWAP_THRESHOLD_SECONDS)
    ));

    // --- 5. SETUP DE COMUNICAÇÃO ---
    let (tx_vision, rx_vision) = channel();
    let (tx_audio, rx_audio) = channel();
    let (tx_feedback, rx_feedback): (Sender<NeuralEnactiveMemory>, Receiver<NeuralEnactiveMemory>) = channel();

    // --- 6. SETUP DO COMPRESSOR ---
    let compressor = Arc::new(SalientCompressor::new(0.1, COMPRESSOR_MAX_POINTS));

    // --- 7. GESTÃO DE HOMEOSTASE ---
    let sleep_manager = SleepManager::new();
    let sensor_for_sleep = Arc::clone(&sensor);
    let memory_for_sleep = Arc::clone(&memory_tier);

    // Task de monitoramento simples
    // FIX exit-code-101 (docx v2.3 §07): spawn_blocking + Runtime::new() dentro de runtime
    // ativo causava panic com nested tokio runtime. Substituído por tokio::spawn.
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        loop {
            interval.tick().await;
            if let Ok(sensor_guard) = sensor_for_sleep.try_lock() {
                let _ = sensor_guard.get_cpu_temp();
            }
            if let Ok(_memory_guard) = memory_for_sleep.try_lock() {
                // vazio intencional
            }
            log::debug!("Sleep manager heart-beat");
        }
    });

    // --- 8. INSTANCIAÇÃO DOS LOBOS ---
    println!("🧠 Criando lobos cerebrais...");
    let mut occipital = OccipitalLobe::new(n_neurons, 0.2, &config);
    let mut parietal = ParietalLobe::new(n_neurons, 0.2, &config);
    let mut temporal = TemporalLobe::new(n_neurons, 0.005, 0.2, &config);
    let mut limbic = LimbicSystem::new(n_neurons / 2, &config);
    let hippocampus = Hippocampus::new(n_neurons / 2, &config);
    let mut frontal = FrontalLobe::new(n_neurons, 0.2, 0.1, &config);
    let mut corpus_callosum = CorpusCallosum::new(10.0, 8);

    // --- 9. INSTANCIAÇÃO DOS NOVOS MÓDULOS ---
    println!("🌱 Inicializando módulos avançados...");
    let mut ego = Ego::carregar_ou_criar("Selene");
    let mut thalamus = Thalamus::new();
    let mut interoception = Interoception::new();
    let mut basal_ganglia = BasalGanglia::new(&config);
    let mut brainstem = Brainstem::new();

    // --- 10. DISPARO DOS SENTIDOS (desativados por padrão) ---
    println!("📷 Inicializando sensores (DESATIVADOS — ative via interface)...");
    let sensor_flags = SensorFlags::new_desativados();

    let video_flag = sensor_flags.video_ativo.clone();
    let mut camera = VisualTransducer::new(n_neurons, video_flag);
    thread::spawn(move || camera.run(tx_vision));

    let audio_flag = sensor_flags.audio_ativo.clone();
    let tx_audio_clone = tx_audio.clone();
    thread::spawn(move || audio::start_listening(n_neurons, tx_audio_clone, audio_flag));

    // --- 11. INICIAR SERVIDOR WEB INTEGRADO ---
    println!("🌐 Iniciando interface neural integrada...");
    
    let brain_state = Arc::new(TokioMutex::new(BrainState::new(Arc::clone(&swap_manager), &config, sensor_flags)));
    let state_for_server = Arc::clone(&brain_state);

    let _server_handle = tokio::spawn(async move {
        websocket::start_websocket_server(state_for_server).await;
    });

    println!("\n✨ --- SELENE BRAIN 2.0: BIO-HARDWARE SYSTEM ONLINE --- ✨\n");

    // --- 12. ESTADO INICIAL ---
    let mut mental_imagery_visual = vec![0.0f32; n_neurons];
    let mut mental_imagery_auditory = vec![0.0f32; n_neurons];
    let internal_goal = vec![0.5f32; n_neurons];
    let start_time = Instant::now();
    let mut step: u64 = 0;
    let mut current_time = 0.0f32;

    // --- 12b. CHUNKING ENGINE ---
    let mut chunking = ChunkingEngine::new(RegionType::Temporal);

    // --- 12c. CONTROLE DE FREQUÊNCIA ADAPTIVA ---
    // Média ponderada exponencial de atividade recente (0.0 = ocioso, 1.0 = pleno)
    let mut atividade_recente: f32 = 0.0;
    // Contador de ciclos consecutivos abaixo do threshold
    let mut ciclos_ociosos: u32 = 0;
    // Frequência atual do loop em Hz
    let mut freq_hz: u64 = FREQ_ATIVA_HZ;

    // --- 13. CICLO DIA/NOITE ---
    let grafo_sinaptico = Arc::new(TokioMutex::new(MemoryTierV2::new()));
    let mut ciclo_sono = CicloSono::new()
        .with_grafo(Arc::clone(&grafo_sinaptico))
        .with_db(Arc::clone(&storage_db));
    let mut tempo_acordado = Duration::from_secs(0);
    let dia_duracao = Duration::from_secs(16 * 60 * 60);

    // --- 14. LOOP NEURAL PRINCIPAL ---
    loop {
        current_time += dt;

        // VERIFICAÇÃO DE SHUTDOWN — checar antes de qualquer processamento
        {
            if let Ok(state) = brain_state.try_lock() {
                if state.shutdown_requested {
                    println!("\n🛑 [MAIN] Shutdown solicitado. Encerrando loop neural...");
                    drop(state);
                    sleep_cycle::CicloSono::shutdown_gracioso(&mut *memory_tier.lock().await).await;
                    println!("✅ Selene encerrada com segurança.");
                    std::process::exit(0);
                }
            }
        }

        if tempo_acordado >= dia_duracao {
            println!("\n🌙 Hora de dormir! Iniciando ciclo de sono...");
            let body_feeling = interoception.sentir();
            if let Some(pensamento) = ego.update(body_feeling, current_time).await {
                println!("    💭 Pensamento antes de dormir: {}", pensamento);
            }
            ciclo_sono.dormir(&mut *memory_tier.lock().await, &config).await;
            tempo_acordado = Duration::from_secs(0);
            println!("☀️ Novo dia começou!\n");
            continue;
        }

        step += 1;
        let loop_start = Instant::now();
        let elapsed = start_time.elapsed().as_secs_f32();

        // A. Atualiza sinais corporais
        {
            let sensor_lock = sensor.lock().await;
            let cpu_temp = sensor_lock.get_cpu_temp();
            let adenosina = 0.1;
            interoception.update(adenosina, cpu_temp, neuro.noradrenaline);
            brainstem.update(adenosina, dt);
        }

        // B. Bioquímica
        neuro.update(&mut *sensor.lock().await, &config);

        // C. Filtragem sensorial
        let raw_retina = rx_vision.try_recv().unwrap_or_else(|_| vec![0.0f32; n_neurons]);
        let audio_signal = rx_audio.try_recv().ok();
        let raw_cochlea: Vec<f32> = match audio_signal {
            Some(sig) => {
                let mut v = sig.bandas.clone();
                v.push(sig.energia);
                v.push(sig.pitch_dominante);
                if v.len() != n_neurons {
                    let ratio = v.len() as f32 / n_neurons as f32;
                    (0..n_neurons).map(|i| {
                        let idx = (i as f32 * ratio) as usize;
                        v.get(idx).copied().unwrap_or(0.0)
                    }).collect()
                } else { v }
            }
            None => vec![0.0f32; n_neurons],
        };

        let retina_input = thalamus.relay(&raw_retina, neuro.noradrenaline, &config);
        let cochlea_input = brainstem.modulate(&raw_cochlea);

        // D. Feedback (recall de memória → imaginação mental)
        if let Ok(memory) = rx_feedback.try_recv() {
            mental_imagery_visual = memory.visual_rates();
            mental_imagery_auditory = memory.auditory_rates();
        }

        // E. Modulação por serotonina
        let stability_factor = neuro.serotonin;
        let hybrid_visual: Vec<f32> = retina_input.iter().enumerate()
            .map(|(i, &v)| {
                let base = (v * 0.7) + (mental_imagery_visual.get(i).unwrap_or(&0.0) * 0.3);
                base * stability_factor
            })
            .collect();

        // F. Processamento neural
        let vision_features = occipital.visual_sweep(&hybrid_visual, dt, Some(&parietal.spatial_map), current_time, &config);
        
        let chunk_size = n_neurons / vision_features.len().max(1);
        let mut vision_full = vec![0.0f32; n_neurons];
        for (i, &feature) in vision_features.iter().enumerate() {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(n_neurons);
            for j in start..end { vision_full[j] = feature / 100.0; }
        }

        let spatial_focus = parietal.integrate(&vision_full, &vec![0.0f32; n_neurons], dt, current_time, &config);
        let recognized = temporal.process(&vision_full, &spatial_focus, dt, current_time, &config);

        // F1. CHUNKING — detecta padrões STDP emergentes na camada temporal
        {
            let spikes_temporais: Vec<bool> = temporal.recognition_layer.neuronios.iter()
                .map(|n| n.last_spike_ms > 0.0 && n.last_spike_ms >= (current_time * 1000.0) - dt * 1000.0)
                .collect();
            let novos_chunks = chunking.registrar_spikes(
                &spikes_temporais,
                &temporal.recognition_layer,
                0.0, // emoção ainda não calculada neste tick — será aplicada via RPE abaixo
                current_time * 1000.0,
            );
            if !novos_chunks.is_empty() {
                for c in &novos_chunks {
                    log::info!("🔤 Chunk emergido: {:?} '{}' (força={:.2}, valence={:.2})",
                        c.tipo, c.simbolo, c.forca_stdp, c.valence);
                }
            }
        }

        let safe_len = 10.min(cochlea_input.len());
        let (emotion, arousal) = limbic.evaluate(&cochlea_input[0..safe_len], 0.0, dt, current_time, &config);

        frontal.set_dopamine(neuro.dopamine + emotion);
        let action = frontal.decide(&recognized, &internal_goal, dt, current_time, &config);

        // G. Núcleos da Base
        basal_ganglia.update_habits(&vision_full, &action, emotion);
        if let Some(_habit_action) = basal_ganglia.suggest_action(&vision_full) {
            log::info!("🔄 Hábito detectado");
        }

        // G1. Propaga RPE (sinal dopaminérgico) para reforçar/enfraquecer chunks recentes
        {
            let rpe = neuro.dopamine - 0.5; // desvio da baseline dopaminérgica
            chunking.aplicar_rpe(rpe);
        }

        // H. Comunicação entre hemisférios
        if step % 100 == 0 {
            let spikes_esquerdo = vec![true; n_neurons / 10];
            corpus_callosum.send_to_right(0, spikes_esquerdo, current_time);
        }
        if let Some(spikes) = corpus_callosum.receive_at_right(0, current_time) {
            log::debug!("🔗 Caloso sync: {} spikes", spikes.len());
        }

        // I. Memória
        if emotion.abs() > 0.6 {
            let snapshot = NeuralEnactiveMemory::from_firing_rates(
                elapsed as f64,
                emotion,
                arousal,
                &vision_full,
                &cochlea_input,
                action.clone(),
                format!("exp_{:.2}_{}", emotion, step),
            );
            if emotion.abs() > 0.8 {
                ego.registrar_experiencia(format!("Emoção Forte {:.2}", emotion), emotion, arousal, ego::TipoMemoria::Autobiografica);
            }
            let mem_clone = Arc::clone(&memory_tier);
            tokio::spawn(async move {
                let _ = mem_clone.lock().await.prioritize_and_save(snapshot).await;
            });
        }

        // J. Atualização do ego
        let body_feeling = interoception.sentir();
        if let Some(pensamento) = ego.update(body_feeling, current_time).await {
            if step % 100 == 0 { println!("    💭 Pensamento: {}", pensamento); }
            
            // Correção: removido narrative_voice (não existe)
            if let Ok(mut state) = brain_state.try_lock() {
                state.ego.pensamentos_recentes.push_back(pensamento.clone());
                if state.ego.pensamentos_recentes.len() > 10 {
                    state.ego.pensamentos_recentes.pop_front();
                }
            }
        }

        // K. Telemetria
        if step % 500 == 0 {
            let swap_guard = swap_manager.lock().await;
            let chunk_stats = chunking.stats();
            println!("🧪 [BIO] Sero: {:.2} | Dop: {:.2} | RAM: {:.1}GB | Tempo: {:?}",
                neuro.serotonin, neuro.dopamine, sensor.lock().await.get_ram_usage_gb(), tempo_acordado
            );
            println!("   🧬 Neurônios: {} ativos | Hábitos: {} | Alerta: {:.2}",
                swap_guard.ram_count(), basal_ganglia.stats().num_habitos, brainstem.stats().alertness
            );
            println!("   🔤 {} | Freq: {}Hz | Atividade: {:.3}",
                chunk_stats, freq_hz, atividade_recente
            );
        }

        // L. Decaimento e Framerate Adaptivo
        for v in mental_imagery_visual.iter_mut() { *v *= 0.95; }
        for a in mental_imagery_auditory.iter_mut() { *a *= 0.95; }

        // L1. Calcula nível de atividade do tick atual
        let atividade_tick: f32 = {
            let spikes_ativos = temporal.recognition_layer.neuronios.iter()
                .filter(|n| n.last_spike_ms > 0.0 && n.last_spike_ms >= (current_time * 1000.0) - dt * 1000.0)
                .count();
            spikes_ativos as f32 / n_neurons as f32
        };
        // EMA rápida: α=0.1 para suavizar sem lag excessivo
        atividade_recente = atividade_recente * 0.90 + atividade_tick * 0.10;

        // L2. Controle de frequência adaptiva
        if atividade_recente < ATIVIDADE_MINIMA {
            ciclos_ociosos += 1;
        } else {
            ciclos_ociosos = 0;
        }

        freq_hz = if ciclos_ociosos >= TICKS_PARA_OCIOSO {
            // Idle: oscila entre 5 Hz e 55 Hz proporcionalmente à atividade residual
            let idle_boost = (atividade_recente / ATIVIDADE_MINIMA).min(1.0);
            (FREQ_OCIOSA_HZ as f32 + idle_boost * 50.0) as u64
        } else {
            // Ativa: escala linear de 55 Hz → 200 Hz com a atividade
            let ratio = (atividade_recente / 0.1).min(1.0);
            (55 + (ratio * (FREQ_ATIVA_HZ - 55) as f32) as u64).min(FREQ_ATIVA_HZ)
        };

        // L3. Throttle por RAM — se uso de RAM acima de 70%, adiciona pausa extra
        // (dá tempo ao OS para paginar e evita swap thrashing)
        let extra_sleep_ms: u64 = {
            if let Ok(s) = sensor.try_lock() {
                let ram_pct = s.get_ram_usage(); // 0-100
                if ram_pct > LIMITE_RECURSO_PCT * 100.0 {
                    let excesso = (ram_pct - LIMITE_RECURSO_PCT * 100.0) / 100.0;
                    (excesso * 50.0) as u64  // até 50ms extras
                } else { 0 }
            } else { 0 }
        };

        // L4. Sleep não-bloqueante com tokio
        let frame_duration = loop_start.elapsed();
        let periodo_ms = 1000 / freq_hz;
        let restante_ms = periodo_ms.saturating_sub(frame_duration.as_millis() as u64);
        let total_sleep_ms = restante_ms + extra_sleep_ms;
        if total_sleep_ms > 0 {
            tokio::time::sleep(Duration::from_millis(total_sleep_ms)).await;
        }
        tempo_acordado += Duration::from_millis(periodo_ms.max(1));

        // O. Limpeza periódica
        if step % 10000 == 0 {
            let mut swap = swap_manager.lock().await;
            let _ = swap.limpar_neurônios_inativos();
        }

        // P. Cap dinâmico de neurônios por RAM (docx v2.3 §03 — BLOCKER)
        // Verificado a cada 500 ticks para não sobrecarregar o lock
        if step % 500 == 0 {
            if let Ok(sensor_guard) = sensor.try_lock() {
                use sysinfo::System;
                let mut sys = System::new();
                sys.refresh_memory();
                let ram_total_gb = sys.total_memory() as f64 / 1e9;
                let ram_livre_gb = sys.available_memory() as f64 / 1e9;
                drop(sensor_guard);
                let mut swap = swap_manager.lock().await;
                swap.verificar_cap_ram(ram_total_gb, ram_livre_gb);
            }
        }
    }
}