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

// Imports necessários
use std::sync::mpsc::{channel, Sender, Receiver};
use std::{thread, panic};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::path::Path;
use std::fs::File;
use std::io::Write;
use tokio::sync::Mutex as TokioMutex;
use warp::Filter; // Adicionado para as rotas

// Imports para Log e Debug
use simplelog::*;

// Imports dos módulos da Selene
use crate::brain_zones::RegionType;
use crate::config::{Config, ModoOperacao};
use crate::sleep_cycle::CicloSono;
use crate::websocket::bridge::{BrainState, NeuralStatus}; // Ajustado para refletir o bridge.rs

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

// Imports de storage
use storage::{BrainStorage, NeuralEnactiveMemory};
use storage::memory_tier::MemoryTier;

// Outros imports
use neurochem::NeuroChem;
use sleep_manager::SleepManagerV2 as SleepManager;
use windows::Win32::Media::timeEndPeriod;

// ================== CONSTANTES ==================
const MAX_RAM_NEURONS: usize = 1_000_000;
const SWAP_THRESHOLD_SECONDS: u64 = 3600;
const COMPRESSOR_MAX_POINTS: usize = 16;

// ================== FUNÇÃO PRINCIPAL ==================
fn main() {
    // 1. Inicializa o Sistema de Logs (Gera selene_debug.log)
    let _ = CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::Info, ConfigBuilder::new().build(), TerminalMode::Mixed, ColorChoice::Auto),
            WriteLogger::new(LevelFilter::Debug, ConfigBuilder::new().build(), File::create("selene_debug.log").unwrap()),
        ]
    );

    // 2. Hook de Pânico (Gera selene_crash_report.txt em caso de erro 101)
    panic::set_hook(Box::new(|panic_info| {
        // Tenta fechar o período de tempo do Windows se necessário
        unsafe { timeEndPeriod(1) };
        
        let mut file = File::create("selene_crash_report.txt").unwrap();
        let message = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            s
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.as_str()
        } else {
            "Causa do pânico desconhecida"
        };
        
        let location = panic_info.location().unwrap_or_else(|| panic::Location::caller());
        
        let report = format!(
            "========================================\n\
             🧠 SELENE BRAIN CRASH REPORT\n\
             ========================================\n\
             Data/Hora: {:?}\n\
             Erro: {}\n\
             Local: {}:{}\n\n\
             Possível causa: Verifique se o arquivo NVME ou o Banco de Dados está acessível.\n\
             ========================================",
            Instant::now(), message, location.file(), location.line()
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
    // CONFIGURAÇÃO INICIAL
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

    // --- 4. SETUP DO SWAP MANAGER (NEUROGÊNESE) ---
    println!("🧬 Inicializando Swap Manager com neurogênese...");
    let swap_manager = Arc::new(TokioMutex::new(
        SwapManager::new(MAX_RAM_NEURONS, SWAP_THRESHOLD_SECONDS)
    ));

    // --- 5. SETUP DE COMUNICAÇÃO ---
    let (tx_vision, rx_vision) = channel();
    let (tx_audio, rx_audio) = channel();
    let (tx_feedback, rx_feedback): (Sender<NeuralEnactiveMemory>, Receiver<NeuralEnactiveMemory>) = channel();

    // --- 6. SETUP DO COMPRESSOR (PONTOS SALIENTES) ---
    let compressor = Arc::new(SalientCompressor::new(0.1, COMPRESSOR_MAX_POINTS));

    // --- 7. GESTÃO DE HOMEOSTASE (Sleep Manager) ---
    let sleep_manager = SleepManager::new();
    let sensor_for_sleep = Arc::clone(&sensor);
    let memory_for_sleep = Arc::clone(&memory_tier);

    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            loop {
                interval.tick().await;
                if let Ok(sensor_guard) = sensor_for_sleep.try_lock() {
                    let _cpu_temp = sensor_guard.get_cpu_temp();
                }
                if let Ok(memory_guard) = memory_for_sleep.try_lock() {
                }
                log::debug!("Sleep manager heart-beat");
            }
        });
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

    // --- 10. DISPARO DOS SENTIDOS ---
    println!("📷 Iniciando sensores...");
    let mut camera = VisualTransducer::new(n_neurons);
    thread::spawn(move || camera.run(tx_vision));

    let tx_audio_clone = tx_audio.clone();
    thread::spawn(move || audio::start_listening(n_neurons, tx_audio_clone));

    // --- 11. INICIAR SERVIDOR WEB INTEGRADO ---
    println!("🌐 Iniciando interface neural integrada...");
    
    // CORREÇÃO: Usando a config e swap_manager que já foram criados no início (itens 1 e 4)
    let brain_state = Arc::new(TokioMutex::new(BrainState::new(Arc::clone(&swap_manager), &config)));
    let state_for_server = Arc::clone(&brain_state);

    let server_handle = tokio::spawn(async move {
        // Criar o canal de broadcast para transmitir o NeuralStatus
        let (tx, _) = tokio::sync::broadcast::channel::<NeuralStatus>(100);
        let tx_broadcast = tx.clone();

        // Rota do WebSocket em /selene
        let ws_route = warp::path("selene")
            .and(warp::ws())
            .map(move |ws: warp::ws::Ws| {
                let rx_sub = tx_broadcast.subscribe();
                ws.on_upgrade(move |socket| crate::websocket::server::handle_connection(socket, rx_sub))
            });

        // Servir arquivos estáticos da pasta "interface"
        let static_files = warp::fs::dir("interface");

        // Habilitar CORS para evitar bloqueios do navegador
        let cors = warp::cors()
            .allow_any_origin()
            .allow_methods(vec!["GET", "POST", "OPTIONS"])
            .allow_headers(vec!["Content-Type"]);

        let routes = ws_route.or(static_files).with(cors);

        // Worker para coletar o estado e enviar para o broadcast
        let s_loop = Arc::clone(&state_for_server);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(500));
            loop {
                interval.tick().await;
                if let Ok(brain) = s_loop.try_lock() {
                    let status = crate::websocket::bridge::collect_neural_status(&brain).await;
                    let _ = tx.send(status);
                }
            }
        });

        println!("✨ Interface Online em: http://127.0.0.1:3030");
        warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
    });

    println!("\n✨ --- SELENE BRAIN 2.0: BIO-HARDWARE SYSTEM ONLINE --- ✨\n");

    // --- 12. ESTADO INICIAL ---
    let mut mental_imagery_visual = vec![0.0f32; n_neurons];
    let mut mental_imagery_auditory = vec![0.0f32; n_neurons];
    let internal_goal = vec![0.5f32; n_neurons];
    let start_time = Instant::now();
    let mut step: u64 = 0;
    let mut current_time = 0.0f32;

    // --- 13. CICLO DIA/NOITE ---
    let mut ciclo_sono = CicloSono::new();
    let mut tempo_acordado = Duration::from_secs(0);
    let dia_duracao = Duration::from_secs(16 * 60 * 60);

    // --- 14. LOOP NEURAL PRINCIPAL ---
    loop {
        current_time += dt;

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

        // D. Feedback
        if let Ok(memory) = rx_feedback.try_recv() {
            mental_imagery_visual = memory.visual_pattern;
            mental_imagery_auditory = memory.auditory_pattern;
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
        let safe_len = 10.min(cochlea_input.len());
        let (emotion, arousal) = limbic.evaluate(&cochlea_input[0..safe_len], 0.0, dt, current_time, &config);

        frontal.set_dopamine(neuro.dopamine + emotion);
        let action = frontal.decide(&recognized, &internal_goal, dt, current_time, &config);

        // G. Núcleos da Base
        basal_ganglia.update_habits(&vision_full, &action, emotion);
        if let Some(_habit_action) = basal_ganglia.suggest_action(&vision_full) {
            log::info!("🔄 Hábito detectado");
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
            let snapshot = NeuralEnactiveMemory {
                timestamp: elapsed as f64,
                emotion_state: emotion,
                arousal_state: arousal,
                visual_pattern: vision_full.clone(),
                auditory_pattern: cochlea_input.clone(),
                frontal_intent: action.clone(),
                label: format!("exp_{:.2}_{}", emotion, step),
            };
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
            
            // Sincronizando o pensamento com o BrainState para o WebSocket
            if let Ok(mut state) = brain_state.try_lock() {
                state.ego.narrative_voice.pensamentos_recentes.push_back(pensamento);
                if state.ego.narrative_voice.pensamentos_recentes.len() > 10 {
                    state.ego.narrative_voice.pensamentos_recentes.pop_front();
                }
            }
        }

        // K. Telemetria
        if step % 500 == 0 {
            let swap_guard = swap_manager.lock().await;
            println!("🧪 [BIO] Sero: {:.2} | Dop: {:.2} | RAM: {:.1}GB | Tempo: {:?}",
                neuro.serotonin, neuro.dopamine, sensor.lock().await.get_ram_usage_gb(), tempo_acordado
            );
            println!("   🧬 Neurônios: {} ativos | Hábitos: {} | Alerta: {:.2}",
                swap_guard.ram_count(), basal_ganglia.stats().num_habitos, brainstem.stats().alertness
            );
        }

        // L. Decaimento e Framerate
        for v in mental_imagery_visual.iter_mut() { *v *= 0.95; }
        for a in mental_imagery_auditory.iter_mut() { *a *= 0.95; }

        let frame_duration = loop_start.elapsed();
        if frame_duration < Duration::from_millis(5) {
            thread::sleep(Duration::from_millis(5) - frame_duration);
        }
        tempo_acordado += Duration::from_millis(5);

        // O. Limpeza periódica
        if step % 10000 == 0 {
            let mut swap = swap_manager.lock().await;
            let _ = swap.limpar_neurônios_inativos(); 
        }
    }
}