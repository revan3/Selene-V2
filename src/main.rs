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
mod encoding;
mod thalamus;
mod interoception;
mod basal_ganglia;
mod brainstem;
mod learning;
mod meta;
mod glia;
mod synthesis;
mod neural_pool;

// Imports necessários
use std::sync::mpsc::{channel, Sender, Receiver};
use std::{thread, panic};
use std::sync::Arc;
use rayon::prelude::*;
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
    cerebellum::Cerebellum,
    mirror_neurons::MirrorNeurons,
    cingulate::AnteriorCingulate,
    orbitofrontal::OrbitalFrontal,
    language::LanguageAreas,
    amygdala::Amygdala,
};
// Q-Learning TD-lambda
use learning::rl::ReinforcementLearning;
// Sinapses inter-lobe e atenção seletiva
use learning::inter_lobe::BrainConnections;
use learning::attention::AttentionGate;
use learning::lobe_router::{LobeRouter, LobeId};

// Imports dos novos módulos
use compressor::salient::SalientCompressor;
use ego::Ego;
use thalamus::Thalamus;
use interoception::Interoception;
use basal_ganglia::BasalGanglia;
use brainstem::Brainstem;
use storage::swap_manager::SwapManager;
use storage::checkpoint::CheckpointSystem;

// Imports dos sensores
use sensors::camera::{DualCameraSystem, novo_frame_buffer};
use sensors::vision_stream::{VisionBroadcast, iniciar_broadcast, iniciar_servidor_visao};
use sensors::audio;
use sensors::hardware::HardwareSensor;
use sensors::SensorFlags;

// Imports de storage
use storage::{BrainStorage, NeuralEnactiveMemory};
use storage::memory_tier::MemoryTier;
use storage::memory_graph::MemoryTierV2;

// Chunking engine
use learning::chunking::ChunkingEngine;

// Metacognição
use meta::MetaCognitive;

// Glia
use glia::GliaLayer;

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
    let term_config = ConfigBuilder::new()
        .add_filter_ignore_str("surrealdb")
        .add_filter_ignore_str("surrealdb_core")
        .add_filter_ignore_str("kvs")
        .add_filter_ignore_str("hyper")
        .add_filter_ignore_str("warp")
        .add_filter_ignore_str("tungstenite")
        .add_filter_ignore_str("tracing")
        .build();

    let debug_config = ConfigBuilder::new()
        .add_filter_ignore_str("hyper")
        .add_filter_ignore_str("warp")
        .add_filter_ignore_str("tungstenite")
        .add_filter_ignore_str("tokio_tungstenite")
        .add_filter_ignore_str("tracing")
        .add_filter_ignore_str("surrealdb")
        .add_filter_ignore_str("surrealdb_core")
        .add_filter_ignore_str("kvs")
        .build();

    let _ = CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::Info, term_config, TerminalMode::Mixed, ColorChoice::Auto),
            WriteLogger::new(LevelFilter::Debug, debug_config, File::create("selene_debug.log").unwrap()),
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
    // Escala neurônios com RAM disponível: mín 1024, máx 8192 (>8k aumenta latência)
    let n_neurons = {
        let max_dyn = calcular_max_neurons();
        max_dyn.min(8192).max(1024)
    };
    
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
    // Restaura pesos sinápticos STDP da sessão anterior (se existir)
    if let Ok(mut sw) = swap_manager.try_lock() {
        sw.carregar_estado("selene_swap_state.json");
        println!("💾 SwapManager restaurado: {} conceitos | {} sinapses",
            sw.palavra_para_id.len(), sw.sinapses_conceito.len());
    }

    // --- 5. SETUP DE COMUNICAÇÃO ---

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
    let mut hippocampus = Hippocampus::new(n_neurons / 2, &config);
    hippocampus.load_ltp("selene_hippo_ltp.json"); // restaura pesos sinápticos persistidos
    // Restaura estado semântico do swap_manager (palavra_para_id + sinapses_conceito)
    {
        let mut swap = swap_manager.lock().await;
        swap.carregar_estado("selene_swap_state.json");
        // Garante que todos os 55 neurônios primitivos existam (idempotente).
        // Deve ser chamado DEPOIS de carregar_estado para não duplicar UUIDs
        // que foram persistidos em sessões anteriores.
        swap.inicializar_camada_zero();
    }
    let mut frontal = FrontalLobe::new(n_neurons, 0.2, 0.1, &config);
    let mut corpus_callosum = CorpusCallosum::new(10.0, 8);

    // --- 8b. CEREBELO E RL ---
    println!("🏃 Inicializando Cerebelo e Aprendizado por Reforço...");
    let mut cerebelo = Cerebellum::new(n_neurons / 4, n_neurons / 2, &config);
    // RL com persistência: restaura Q-table de sessões anteriores
    let mut rl = ReinforcementLearning::restaurar_ou_novo("selene_qtable.bin");
    // Sinapses inter-lobe com STDP — vias de longa distância entre regiões
    let mut brain_conn = BrainConnections::new(n_neurons);
    // Gate de atenção seletiva (bottom-up saliência + top-down frontal)
    let mut attention = AttentionGate::new(n_neurons);
    // Roteador dinâmico de lóbulos — decide quais constelações ativar por tick.
    // Especialização emerge via competitive Hebbian learning nas chaves de cada lóbulo.
    let mut lobe_router = LobeRouter::new();
    println!("🗺️  LobeRouter online: 6 constelações, embedding 16d, homeostase ativa.");

    // --- 9. INSTANCIAÇÃO DOS NOVOS MÓDULOS ---
    println!("🌱 Inicializando módulos avançados...");
    let mut ego = Ego::carregar_ou_criar("Selene");
    let mut thalamus = Thalamus::new();
    let mut interoception = Interoception::new();
    let mut basal_ganglia = BasalGanglia::new(&config);
    let mut brainstem = Brainstem::new();
    // Fix 6: Neurônios espelho — ressonância motora com input do usuário.
    // Pré-configurados com padrões de emoções e ações básicas; aprendem com uso.
    let mut mirror = MirrorNeurons::new();
    println!("🪞 Mirror neurons: {} padrões pré-configurados.", mirror.n_padroes());

    // --- 9b. NOVOS LOBOS COGNITIVOS ---
    // ACC: monitoramento de conflito, dor social, ajuste comportamental por erro.
    let mut cingulate = AnteriorCingulate::new(n_neurons / 4, &config);
    // OFC: valor esperado por contexto, reversal learning.
    let mut ofc = OrbitalFrontal::new(n_neurons / 3, &config);
    // Áreas de linguagem: Wernicke (compreensão) + Broca (produção).
    let mut language = LanguageAreas::new(n_neurons / 3, &config);
    // Amígdala separada: BLA (condicionamento) + CeA (output arousal).
    let mut amygdala = Amygdala::new(n_neurons / 4, &config);
    println!("🗣️  Áreas de linguagem online: Wernicke + Broca.");
    println!("⚖️  Cingulado anterior + OFC online: conflito, reversal learning.");
    println!("😨 Amígdala (BLA + CeA) online: condicionamento de medo e arousal.");

    // --- 10. DISPARO DOS SENTIDOS (desativados por padrão) ---
    println!("📷 Inicializando sensores (DESATIVADOS — ative via interface)...");
    let sensor_flags = SensorFlags::new_desativados();

    let video_flag = sensor_flags.video_ativo.clone();
    let estereo_flag = sensor_flags.video_ativo.clone(); // reutiliza mesma flag; câmera 1 inativa por ora
    let dual_cam = DualCameraSystem::novo(false); // stereo_disponivel=false (1 câmera física)
    let frame_buf_clone = std::sync::Arc::clone(&dual_cam.frame_buf);
    let (rx_vision_dual, _rx_estereo) = dual_cam.iniciar(n_neurons, video_flag, estereo_flag);
    // Vision stream broadcast — envia frames para o visualizador Python
    let vision_broadcast = VisionBroadcast::novo();
    let vision_bc_clone  = vision_broadcast.clone();
    iniciar_broadcast(frame_buf_clone, vision_bc_clone);
    tokio::spawn(async move { iniciar_servidor_visao(vision_broadcast).await });
    // Redireciona o canal de visão neural para o receptor do DualCamSystem
    let rx_vision = rx_vision_dual;

    let audio_flag = sensor_flags.audio_ativo.clone();
    let tx_audio_clone = tx_audio.clone();
    thread::spawn(move || audio::start_listening(n_neurons, tx_audio_clone, audio_flag));

    // --- 11. INICIAR SERVIDOR WEB INTEGRADO ---
    println!("🌐 Iniciando interface neural integrada...");
    
    let brain_state = Arc::new(TokioMutex::new(BrainState::new(Arc::clone(&swap_manager), &config, sensor_flags)));
    // Injeta o storage de ondas e inicializa o schema wave-first
    {
        let mut bs = brain_state.lock().await;
        bs.storage = Arc::clone(&storage_db);
    }
    // Handle direto para o contador de spikes — sem precisar locker BrainState no loop.
    let neuronios_ativos_handle: Arc<std::sync::atomic::AtomicUsize> = {
        let bs = brain_state.lock().await;
        Arc::clone(&bs.neuronios_ativos)
    };
    let _ = crate::storage::ondas::inicializar_schema_ondas(&storage_db.db).await;
    let state_for_server = Arc::clone(&brain_state);

    let _server_handle = tokio::spawn(async move {
        websocket::start_websocket_server(state_for_server).await;
    });

    // ── Eternal Hole: ciclos de pensamento interno autônomo ───────────────
    // Consciente (50Hz): caminha a partir do neural_context atual → enriquece respostas.
    // Inconsciente (10Hz): deriva livre pelo grafo → cria associações transitivas.
    // Ambos usam try_lock — não bloqueiam o loop neural nem o chat handler.
    crate::learning::pensamento::iniciar_ciclos_pensamento(Arc::clone(&brain_state));

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

    // --- 12e. CAMADA GLIAL (astrocytes tripartite) ---
    let mut glia = GliaLayer::new();

    // --- 12d. METACOGNIÇÃO ---
    let mut metacognitive = MetaCognitive::new();
    // Firing rates do tick anterior — usados pelas projeções inter-lobe
    let mut prev_v1_rates:       Vec<f32> = vec![0.0; n_neurons];
    let mut prev_temporal_rates: Vec<f32> = vec![0.0; n_neurons];
    let mut prev_parietal_rates: Vec<f32> = vec![0.0; n_neurons];
    let mut prev_frontal_rates:  Vec<f32> = vec![0.0; n_neurons];
    let mut prev_limbic_rates:   Vec<f32> = vec![0.0; n_neurons / 2];
    let mut prev_hippo_rates:    Vec<f32> = vec![0.0; n_neurons / 2];
    // Luminância média do frame anterior — para calcular taxa de variação (movimento)
    let mut prev_lum_media: f32 = 0.0;
    // Contador para salvar RL periodicamente (a cada ~60s @ 200Hz = 12000 ticks)
    let mut rl_save_counter: u64 = 0;

    // ── Checkpoint automático (a cada 4h de tempo real) ───────────────────
    let mut checkpoint_system = CheckpointSystem::new();

    // ── Reward signal: histórico de contexto para medir consistência ──────
    // Guarda snapshots do neural_context a cada 1000 ticks.
    // Se 2+ palavras aparecem em ≥3 dos últimos 10 snapshots → dopamina +0.04.
    let mut contexto_historico: std::collections::VecDeque<Vec<String>> =
        std::collections::VecDeque::with_capacity(10);

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
        .with_db(Arc::clone(&storage_db))
        .with_brain_state(Arc::clone(&brain_state));
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
            // Extinção do medo durante sono (hipocampo + vmPFC consolida segurança)
            amygdala.extinção_durante_sono();
            // Snapshot de fim de dia antes de continuar
            if let Ok(mut sw) = swap_manager.try_lock() {
                sw.criar_snapshot(step as u64);
            }
            tempo_acordado = Duration::from_secs(0);
            println!("☀️ Novo dia começou!\n");
            continue;
        }

        step += 1;
        let loop_start = Instant::now();
        let elapsed = start_time.elapsed().as_secs_f32();

        // A. Atualiza sinais corporais
        // Adenosina acumula com o tempo acordado (pressão de sono biológica):
        // 0.0 ao acordar → ~0.9 após 16h, revertido pelo sono.
        {
            let sensor_lock = sensor.lock().await;
            let cpu_temp = sensor_lock.get_cpu_temp();
            let adenosina = (tempo_acordado.as_secs_f32() / (16.0 * 3600.0)).clamp(0.0, 0.95);
            interoception.update(adenosina, cpu_temp, neuro.noradrenaline);
            brainstem.update(adenosina, dt);
        }

        // A1. Efeito tátil — aplica delta neuromodulador do toque à bioquímica.
        // O sinal decai dentro do próprio efeito_toque() a cada tick.
        // Carinho:  da+, sero+, na-, cortisol-
        // Beliscão: da-, sero-, na+, cortisol+
        {
            let (d_da, d_ser, d_na, d_cor) = interoception.efeito_toque();
            if d_da.abs() > 0.001 || d_ser.abs() > 0.001 {
                neuro.dopamine      = (neuro.dopamine      + d_da ).clamp(0.3, 2.5);
                neuro.serotonin     = (neuro.serotonin     + d_ser).clamp(0.0, 2.0);
                neuro.noradrenaline = (neuro.noradrenaline + d_na ).clamp(0.0, 2.0);
                neuro.cortisol      = (neuro.cortisol      + d_cor).clamp(0.0, 2.0);
            }
        }

        // B. Bioquímica
        neuro.update(&mut *sensor.lock().await, &config);

        // B0. Atualiza buffer perceptual no BrainState — snapshot do estado atual
        // para que o chat handler possa fazer grounding binding contextualizado.
        if let Ok(mut bs) = brain_state.try_lock() {
            bs.ultimo_estado_corpo = [
                neuro.dopamine, neuro.serotonin, neuro.noradrenaline,
                neuro.acetylcholine, neuro.cortisol,
            ];
        }

        // B1. Neuromodulação global — propaga dopamina/serotonina/cortisol para TODOS os lobos.
        if step % 5 == 0 {
            let (da, ser, cor, ach) = (neuro.dopamine, neuro.serotonin, neuro.cortisol, neuro.acetylcholine);
            occipital.v1_primary_layer.modular_neuro_v3(da, ser, cor, ach);
            occipital.v2_feature_layer.modular_neuro_v3(da, ser, cor, ach);
            temporal.recognition_layer.modular_neuro_v3(da, ser, cor, ach);
            frontal.executive_layer.modular_neuro_v3(da, ser, cor, ach);
            frontal.inhibitory_layer.modular_neuro_v3(da, ser, cor, ach);
            limbic.amygdala.modular_neuro_v3(da, ser, cor, ach);
            limbic.nucleus_accumbens.modular_neuro_v3(da, ser, cor, ach);
            // Fix 7a: ACh modula hipocampo — alta ACh = codificação mais nítida.
            // Biologicamente: projeções colinérgicas do núcleo basal → CA1/CA3.
            // Efeito: ACh alta reduz cortisol efetivo no hipocampo (neurônios mais sensíveis).
            let cor_hippo = (cor * (1.0 - neuro.acetylcholine * 0.3)).clamp(0.0, 1.0);
            hippocampus.ca1_encoding.modular_neuro(da, ser * neuro.acetylcholine.clamp(0.5, 1.2), cor_hippo);
            hippocampus.ca3_recurrent.modular_neuro(da, ser, cor_hippo);
            parietal.integration_layer.modular_neuro_v3(da, ser, cor, ach);
            cerebelo.purkinje_layer.modular_neuro_v3(da, ser, cor, ach);
            cerebelo.granular_layer.modular_neuro_v3(da, ser, cor, ach);
            // Propaga serotonina para working memory do frontal (afeta decay WM)
            frontal.set_serotonin(neuro.serotonin);
            frontal.set_dopamine(neuro.dopamine);
            // Neuromodulação das projeções inter-lobe: dopamina aumenta plasticidade
            brain_conn.modular_all(da, cor);

            // Fix 7b: Parietal → Tálamo (atenção espacial).
            // O lóbulo parietal sabe "onde olhar" — passa esse sinal ao tálamo para
            // filtrar inputs em favor da região espacialmente saliente.
            // Biologicamente: PPC → pulvinar → tálamo sensorial.
            let parietal_salience = parietal.spatial_map.iter()
                .map(|&v| v.abs()).sum::<f32>() / parietal.spatial_map.len().max(1) as f32;
            thalamus.adapt_filter(parietal_salience * 0.05 - 0.01); // abre filtro onde há saliência

            // Fix 7c: Frontal → inibição preditiva no temporal.
            // Quando frontal tem WM carregada (alta dopamina_level), suprime inputs
            // já previstos — só o INESPERADO passa. Isso implementa predictive coding.
            // Biologicamente: projeções descendentes do PFC → camadas supragranulares do temporal.
            let wm_snaps = frontal.wm_snapshots();
            if !wm_snaps.is_empty() {
                let wm_certeza: f32 = wm_snaps.iter().map(|(sal, _)| sal).sum::<f32>()
                    / wm_snaps.len() as f32;
                // Alta certeza WM → temporal suprime entradas previstas levemente
                let inibicao_preditiva = (wm_certeza * neuro.dopamine * 0.15).clamp(0.0, 0.25);
                temporal.recognition_layer.modular_neuro_v3(
                    da * (1.0 - inibicao_preditiva),
                    ser,
                    cor + inibicao_preditiva * 0.5,
                    ach,
                );
            }
        }

        // B2. Metacognição ativa — retroalimenta o sistema com base no estado observado
        let meta_feedback = {
            let n_vocab = swap_manager.try_lock()
                .map(|sw| sw.palavra_para_id.len()).unwrap_or(100);
            metacognitive.observe(neuro.noradrenaline, atividade_recente, n_vocab);
            metacognitive.retroalimentar()
        };

        // Aplica ganho_frontal: escala o sinal dopaminérgico do frontal.
        // Alta confusão → mais dopamina no frontal → foco executivo reforçado.
        neuro.dopamine = (neuro.dopamine * meta_feedback.ganho_frontal).clamp(0.0, 2.0);

        // Aplica threshold_offset nas projeções inter-lobe via cortisol proxy.
        // threshold_offset < 0 (self_awareness alta) → cortisol menor → neurônios mais sensíveis.
        // threshold_offset > 0 (baixa awareness) → cortisol maior → filtra ruído.
        let cortisol_ajustado = (neuro.cortisol - meta_feedback.threshold_offset * 0.1).clamp(0.0, 2.0);

        // Aplica plasticidade_mod ao STDP inter-lobe: baixa estabilidade → menos plasticidade.
        // Isso evita que Selene "aprenda errado" quando está confusa/instável.
        brain_conn.modular_all(neuro.dopamine, cortisol_ajustado * (2.0 - meta_feedback.plasticidade_mod));

        // Sinaliza replay hippocampal quando idle e foco está no hipocampo
        let idle_replay_agora = meta_feedback.habilitar_replay && atividade_recente < 0.005;

        // C. Filtragem sensorial
        let raw_retina = rx_vision.try_recv().unwrap_or_else(|_| vec![0.0f32; n_neurons]);
        let audio_signal = rx_audio.try_recv().ok();

        // ── Reconhecimento auditivo via microfone (córtex auditivo secundário) ──
        // Pipeline nativo: cpal → FFT → acumulador → palavra_completa → cérebro.
        // Não usa WS; a interface só envia start_mic/stop_mic para toggle da flag.
        // Mantém-se audio_raw WS como porta aberta para mobile/remoto.
        if let Some(ref sig) = audio_signal {
            if let Some(ref bandas_palavra) = sig.palavra_completa {
                use crate::encoding::spike_codec::{bands_to_spike_pattern, similarity as spike_sim};
                let audio_pat = bands_to_spike_pattern(bandas_palavra);

                // ── Dedup: descarta reconhecimentos idênticos em < 300ms ──────
                let agora_ms = step as f64 * (1000.0 / 200.0); // ~5ms por step @ 200Hz
                let repetido = {
                    if let Ok(bs) = brain_state.try_lock() {
                        let mut h: u64 = 14695981039346656037;
                        for b in &audio_pat { h = h.wrapping_mul(1099511628211) ^ (*b as u64); }
                        h == bs.ultimo_audio_hash && (agora_ms - bs.ultimo_audio_ts_ms) < 300.0
                    } else { false }
                };

                if !repetido {
                    // ── Busca no spike_vocab ─────────────────────────────────
                    let (melhor, sim_max) = {
                        if let Ok(bs) = brain_state.try_lock() {
                            let mut m: Option<String> = None;
                            let mut s_max = 0.0f32;
                            for (chave, pat_ref) in &bs.spike_vocab {
                                if let Some(p) = chave.strip_prefix("audio:") {
                                    let s = spike_sim(&audio_pat, pat_ref);
                                    if s > s_max { s_max = s; m = Some(p.to_string()); }
                                }
                            }
                            (m, s_max)
                        } else { (None, 0.0) }
                    };

                    if let Ok(mut bs) = brain_state.try_lock() {
                        // Grava hash/ts para dedup
                        let mut h: u64 = 14695981039346656037;
                        for b in &audio_pat { h = h.wrapping_mul(1099511628211) ^ (*b as u64); }
                        bs.ultimo_audio_hash = h;
                        bs.ultimo_audio_ts_ms = agora_ms;

                        // ── Fase C: grava frames FFT brutos para síntese neural ──
                        // Converte Vec<f32> em [f32;32] e armazena por palavra reconhecida
                        if let Some(ref palavra) = melhor {
                            if bandas_palavra.len() >= 32 {
                                let mut frame = [0f32; 32];
                                frame.copy_from_slice(&bandas_palavra[..32]);
                                bs.audio_frames
                                    .entry(palavra.clone())
                                    .and_modify(|frames: &mut Vec<[f32; 32]>| {
                                        if frames.len() >= 16 { frames.remove(0); } // janela 16
                                        frames.push(frame);
                                    })
                                    .or_insert_with(|| vec![frame]);
                            }
                        }

                        if sim_max >= 0.55 {
                            if let Some(ref palavra) = melhor {
                                println!("🎙️ [MIC] Reconheceu «{}» (sim={:.2})", palavra, sim_max);
                                if bs.neural_context.len() >= 20 { bs.neural_context.pop_front(); }
                                bs.neural_context.push_back(palavra.clone());
                                let vpad = bs.ultimo_padrao_visual;
                                let emocao = bs.emocao_bias;
                                bs.grounding_bind(&[palavra.clone()], vpad, audio_pat, emocao, sim_max, 0.0);
                                // Atualiza ontogeny: conta palavra ouvida
                                bs.ontogeny.metrics.total_palavras_ouvidas += 1;
                            }
                        } else {
                            // Palavra nova: aprende semanticamente (mesmo pipeline do audio_raw WS)
                            let chave_audio = format!("audio:palavra_{}", step % 10000);
                            bs.inserir_spike_vocab(chave_audio, audio_pat);
                            let swap_arc = bs.swap_manager.clone();
                            drop(bs);
                            if let Ok(mut sw) = swap_arc.try_lock() {
                                // Valência neutra para palavras ouvidas mas não reconhecidas
                                let valence = sig.energia * 0.1;
                                sw.aprender_conceito("_audio_novo", valence);
                            };
                        }
                    }
                }
            }

            // Atualiza ontogeny: acumula horas de escuta (~5ms por step onde há som)
            if sig.energia > 0.01 {
                if let Ok(mut bs) = brain_state.try_lock() {
                    bs.ontogeny.metrics.add_escuta_dt(0.005); // ~5ms por frame com som
                }
            }
        }

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

        // E. Atenção seletiva + modulação por serotonina
        // Top-down bias: frontal direciona onde olhar (supressão de V1 durante deliberação)
        attention.set_topdown(&prev_frontal_rates);
        let retina_attended = attention.attend(&retina_input, dt * 1000.0);

        // Top-down suppression: quando frontal está deliberando (foco interno), atenua V1
        let suppression_mean = frontal.suppression_signal.iter().sum::<f32>()
            / frontal.suppression_signal.len().max(1) as f32;

        // Correntes inter-lobe do tick anterior modulam os inputs deste tick
        let inter_lobe_currents = brain_conn.project_all(
            &prev_v1_rates, &prev_temporal_rates, &prev_parietal_rates,
            &prev_frontal_rates, &prev_limbic_rates, &prev_hippo_rates,
        );

        let stability_factor = neuro.serotonin;
        // Fix B: alertness do brainstem modula TODOS os inputs sensoriais.
        // Biologicamente: formação reticular ascendente (ARAS) controla o "volume" global
        // do processamento cortical. Cansaço (adenosina alta) → alertness baixo → inputs atenuados.
        // Mantemos mínimo de 0.3 para não zerar completamente (o cérebro nunca desliga totalmente).
        let alertness_gain = brainstem.stats().alertness.max(0.3);

        // Fix 4: input tônico espontâneo (disparo basal biológico).
        // Neurônios corticais nunca ficam completamente silenciosos — ruído de fundo
        // mantém excitabilidade mínima mesmo sem câmera/microfone ativos.
        // Amplitude: 0.04 (abaixo do limiar de spike ~0.1, mas acima de zero).
        let tonico: f32 = 0.04 * ((step.wrapping_mul(1664525).wrapping_add(1013904223)) as f32 / u32::MAX as f32);

        let hybrid_visual: Vec<f32> = retina_attended.iter().enumerate()
            .map(|(i, &v)| {
                let base = (v * 0.7) + (mental_imagery_visual.get(i).unwrap_or(&0.0) * 0.3) + tonico;
                // Top-down suppression atenua visual quando frontal está focado internamente
                let suppressed = base * (1.0 - suppression_mean.clamp(0.0, 0.5));
                // Adiciona corrente inter-lobe (feedback de hipocampo e frontal)
                let inter = inter_lobe_currents.para_temporal.get(i).copied().unwrap_or(0.0) * 0.1;
                // Alertness do brainstem: cansaço atenua processamento visual
                (suppressed + inter).clamp(0.0, 1.0) * stability_factor * alertness_gain
            })
            .collect();

        // B_percept: Atualiza padrões sensoriais no BrainState para grounding.
        // Features visuais e bandas auditivas são convertidas em SpikePattern
        // e ficam disponíveis para o chat handler vincular palavras a percepções.
        if step % 10 == 0 {
            use crate::encoding::spike_codec::{features_to_spike_pattern, bands_to_spike_pattern};
            // Nota: vision_features ainda não calculado aqui; usamos prev_v1_rates
            // como proxy (rates do tick anterior — 1-tick de latência, biologicamente plausível).
            let vpad = features_to_spike_pattern(
                &prev_v1_rates.iter().take(8).map(|&v| v * 100.0).collect::<Vec<_>>()
            );
            let apad = bands_to_spike_pattern(
                &cochlea_input.iter().take(32).copied().collect::<Vec<_>>()
            );
            if let Ok(mut bs) = brain_state.try_lock() {
                bs.ultimo_padrao_visual = vpad;
                bs.ultimo_padrao_audio  = apad;
            }
        }

        // ── PRÉ-WAVE: cochlea_alertado + query de roteamento ─────────────────
        let safe_len = 10.min(cochlea_input.len());
        let cochlea_alertado: Vec<f32> = cochlea_input[0..safe_len].iter()
            .map(|&v| v * alertness_gain)
            .collect();

        // Constrói query de roteamento com o estado atual do sistema.
        // O router usa isso para decidir quais constelações ativar neste tick.
        let abstraction_level = temporal.depth_stack.abstraction_level();
        let route_query = LobeRouter::build_query(
            &hybrid_visual,
            &cochlea_alertado,
            neuro.dopamine, neuro.serotonin, neuro.cortisol, neuro.noradrenaline,
            0.0, // emotion ainda não calculada — será 0.0 no primeiro sub-tick
            atividade_recente, // arousal proxy
            atividade_recente,
            abstraction_level,
            step,
        );
        let routing = lobe_router.route(route_query);

        // ── WAVE 1 [paralelo]: occipital + limbic ─────────────────────────────
        // Limbic: se gate baixíssimo (< SKIP), usa emoção neutra (0.0, 0.1).
        // Na prática o gate_minimo do limbic é 0.20 então nunca vai skippar.
        // FIX inter-lobe: frontal→limbic (para_limbic) é a corrente de regulação
        // top-down do PFC sobre a amígdala — agora aplicada como reward_signal.
        //
        // Sensibilidade occipital: cortisol alto (estresse) → limiar de contraste maior
        // (filtra ruído visual). Cortisol baixo (calmo) → mais sensível a variações sutis.
        // Mapeia cortisol [0.0, 2.0] → contrast_threshold [0.10, 0.40].
        occipital.set_sensitivity(0.10 + neuro.cortisol * 0.15);
        let limbic_reward_inter = {
            let n = inter_lobe_currents.para_limbic.len().max(1) as f32;
            inter_lobe_currents.para_limbic.iter().sum::<f32>() / n * 0.1
        };
        let (vision_features, (emotion, arousal)) = rayon::join(
            || occipital.visual_sweep(&hybrid_visual, dt, Some(&parietal.spatial_map), current_time, &config),
            || {
                if routing.deve_skipar(LobeId::Limbic) {
                    (0.0f32, 0.1f32) // gate baixo: emoção neutra
                } else {
                    let (e, a) = limbic.evaluate(&cochlea_alertado, limbic_reward_inter, dt, current_time, &config);
                    // Escala output pelo gate: gate alto = influência total
                    (e * routing.limbic, a * routing.limbic)
                }
            },
        );

        // ── Camada 0 visual: ativa primitivas sensoriais no swap_manager ─────
        // Derivado de raw_retina (luminância) e vision_features (saída V2 do occipital).
        // Taxa de variação temporal = diferença entre luminâncias médias de frames consecutivos.
        {
            let n_ret = raw_retina.len().max(1) as f32;
            let lum_media = raw_retina.iter().sum::<f32>() / n_ret;
            let taxa_var = (lum_media - prev_lum_media).abs();
            let freq_media = {
                let n_vf = vision_features.len().max(1) as f32;
                (vision_features.iter().sum::<f32>() / n_vf / 100.0).clamp(0.0, 1.0)
            };
            // nm_medio: sem RGB disponível, usa luminância para estimar escala cinza (≈ 550nm)
            // Quando pipeline RGB for adicionado à câmera, substituir por nm real.
            let nm_medio = 550.0_f32;
            if let Ok(mut swap) = swap_manager.try_lock() {
                swap.processar_visual_simples(lum_media, freq_media, taxa_var, nm_medio);
            }
            prev_lum_media = lum_media;
        }

        // Reconstrói vision_full (sequencial — rápido, só indexação)
        let chunk_size = n_neurons / vision_features.len().max(1);
        let mut vision_full = vec![0.0f32; n_neurons];
        for (i, &feature) in vision_features.iter().enumerate() {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(n_neurons);
            for j in start..end { vision_full[j] = feature / 100.0; }
        }

        // ── WAVE 2 [paralelo]: parietal + temporal ────────────────────────────
        // Skip adaptativo: lóbulos com gate baixo reutilizam output anterior.
        // Temporal usa prev_parietal_rates (1-tick latência) — paralelo + biológico.
        // FIX inter-lobe: V1→parietal (para_parietal) substitui zero_n — corrente de
        // localização espacial (onde está o objeto) agora chega corretamente ao parietal.
        //
        // Atenção parietal: arousal alto → foco atencional ampliado (atenção vigilante).
        // Arousal [0.5, 3.5] → attention_global [0.5, 2.5] via escala linear.
        parietal.set_attention(0.3 + arousal * 0.6);
        let parietal_inter: Vec<f32> = inter_lobe_currents.para_parietal.iter()
            .map(|&v| v * 0.1).collect();
        let (new_parietal_rates, recognized) = rayon::join(
            || {
                if routing.deve_skipar(LobeId::Parietal) {
                    prev_parietal_rates.clone() // economiza CPU, usa output anterior
                } else {
                    parietal.integrate(&vision_full, &parietal_inter, dt, current_time, &config)
                }
            },
            || {
                if routing.deve_skipar(LobeId::Temporal) {
                    prev_temporal_rates.clone()
                } else {
                    let out = temporal.process(&vision_full, &prev_parietal_rates, dt, current_time, &config);
                    // Robustez: normaliza output se norma explodir
                    let max_v = out.iter().copied().fold(0.0f32, f32::max);
                    if max_v > 2.0 {
                        out.iter().map(|&v| v / max_v).collect()
                    } else {
                        out
                    }
                }
            },
        );
        // Atualiza prev_parietal_rates com output real deste tick
        {
            let p_len = prev_parietal_rates.len().min(new_parietal_rates.len());
            prev_parietal_rates[..p_len].copy_from_slice(&new_parietal_rates[..p_len]);
        }

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
                    // Fix 5a: Persiste chunk no grafo neural (fire-and-forget, não bloqueia o loop)
                    let conexao = c.para_conexao_sinaptica();
                    let grafo_clone = Arc::clone(&grafo_sinaptico);
                    tokio::spawn(async move {
                        grafo_clone.lock().await.criar_conexao(conexao).await;
                    });
                }
                // Fix 5b: Propaga chunks ao BrainState — contexto neural + valência no vocabulário.
                // IMPORTANTE: símbolos "chunk_X_Y_Z" são placeholders neurais sem valor semântico.
                // Filtramos esses e usamos o chunk apenas para reforçar palavras JÁ no contexto
                // (via valência emergente), criando um binding neuron-pattern → word real.
                // Classify chunks without any lock
                let mut placeholder_valences: Vec<f32> = Vec::new();
                let mut real_simbolos: Vec<(String, f32)> = Vec::new();
                for c in &novos_chunks {
                    if c.simbolo.starts_with("chunk_") || c.simbolo.contains('_') {
                        placeholder_valences.push(c.valence);
                    } else {
                        real_simbolos.push((c.simbolo.clone(), c.valence));
                    }
                }

                // Snapshot neural_context for placeholder boosting + push real symbols
                let palavras_ativas: Vec<String> = if let Ok(mut bs) = brain_state.try_lock() {
                    let snap: Vec<String> = bs.neural_context.iter().take(3).cloned().collect();
                    for (sim, _) in &real_simbolos {
                        bs.neural_context.push_back(sim.clone());
                    }
                    while bs.neural_context.len() > 20 { bs.neural_context.pop_front(); }
                    snap
                } else { Vec::new() };

                // Single swap lock for all updates
                if let Ok(mut sw) = swap_manager.try_lock() {
                    for val in &placeholder_valences {
                        for palavra in palavras_ativas.iter() {
                            if sw.palavra_para_id.contains_key(palavra.as_str()) {
                                sw.aprender_conceito(palavra, val * 0.10);
                            }
                        }
                    }
                    for (simbolo, valence) in &real_simbolos {
                        sw.aprender_conceito(simbolo, *valence);
                    }
                }
            }
        }

        // emotion e arousal já calculados em Wave 1 acima (limbic paralelo com occipital).

        // ── FASE 1a: Emoção (Plutchik) → viés do vocabulário de fala ──────────
        // EmotionalState deriva joy/fear/sadness dos neurotransmissores atuais.
        // O resultado (emocao_bias) desloca o alvo emocional da caminhada no grafo,
        // fazendo o vocabulário espelhar o estado interno: alegria → palavras positivas,
        // medo → palavras negativas, tristeza → moderação negativa.
        let plutchik = neurochem::EmotionalState::from_neurochem(
            neuro.dopamine, neuro.serotonin, neuro.cortisol, neuro.noradrenaline,
        );
        let emocao_bias = plutchik.joy * 0.3 - plutchik.fear * 0.4 - plutchik.sadness * 0.25;

        // ── FASE 1b: Hipocampo → Grafo linguístico ───────────────────────────
        // Hipocampo roda quando há emoção significativa (≥ 0.35).
        // CA3 recorrente com onda theta codifica o padrão reconhecido em memória
        // episódica. Conexões geradas são propagadas ao grafo de associações via
        // brain_state quando o chunk acompanhou emoção forte.
        if emotion.abs() >= 0.35 {
            // FIX inter-lobe: temporal→hippo + parietal→hippo (para_hippo) agora é adicionado
            // ao padrão temporal antes de memorizar — o hipocampo recebe contexto espacial
            // e temporal integrado, não apenas output puro do temporal.
            let hippo_input: Vec<f32> = recognized.iter().enumerate()
                .map(|(i, &r)| {
                    let h = inter_lobe_currents.para_hippo.get(i).copied().unwrap_or(0.0);
                    (r + h * 0.1).clamp(0.0, 1.0)
                })
                .collect();
            let (hippo_out, conexoes_hippo) = hippocampus.memorize_with_connections(
                &hippo_input, emotion, dt, current_time, &config,
            );
            // Fix 1: Consolida memória no grafo de linguagem.
            // O hipocampo sinaliza que AGORA é um momento emocionalmente saliente.
            // Os chunks ativos em neural_context são os "elementos da experiência".
            // Reforçamos as associações entre eles — conceitos co-ativos durante emoção
            // ficam mais fortemente ligados, exatamente como na memória episódica real.
            // Hippocampo reforça co-ativações no swap (sinapses_conceito) em vez de grafo_associacoes.
            // Emoções intensas fortalecem sinapses entre palavras co-ativas — LTP episódico.
            if !conexoes_hippo.is_empty() {
                if let Ok(bs) = brain_state.try_lock() {
                    let ctx: Vec<String> = bs.neural_context.iter().cloned().collect();
                    drop(bs);
                    let valence_hippo = emotion.abs().clamp(0.05, 1.0);
                    if let Ok(mut sw) = swap_manager.try_lock() {
                        // Aprende conceitos co-ativos — STDP sequencial entre pares do contexto
                        for i in 0..ctx.len().min(6) {
                            sw.aprender_conceito(&ctx[i], valence_hippo * emotion.signum());
                        }
                        // P1-B: consolida conexoes_hippo como pares causais no swap.
                        // Mapeia UUID canônico → palavra via lookup inverso,
                        // depois reforça a sinapse conceitual com o peso emocional.
                        // Isso fecha o loop hipocampo→semântica: episódios emocionais
                        // fortalecem diretamente as sinapses entre os conceitos envolvidos.
                        let id_to_word = sw.id_para_palavra();
                        let pares: Vec<(String, String, f32)> = conexoes_hippo.iter()
                            .filter_map(|c| {
                                let w_pre  = id_to_word.get(&c.de_neuronio)?;
                                let w_post = id_to_word.get(&c.para_neuronio)?;
                                let peso = (c.peso * c.emocao_media.abs()).clamp(0.05, 1.0);
                                Some((w_pre.clone(), w_post.clone(), peso))
                            })
                            .collect();
                        if !pares.is_empty() {
                            sw.importar_causal(pares);
                            log::debug!("[Hippo→Swap] {} pares causais consolidados (emocao={:.2})",
                                conexoes_hippo.len(), emotion);
                        }
                    }
                }
            }

            // ── FASE 1c: Saída do hipocampo → canal de imaginação mental ─────
            // Quando a emoção é intensa (≥ 0.6), o padrão hipocampal é enviado de
            // volta como "eco de memória" — alimenta mental_imagery_visual/auditory,
            // fazendo a Selene "lembrar" de experiências parecidas enquanto processa.
            if emotion.abs() >= 0.6 {
                // frontal_intent não disponível aqui ainda — usamos o padrão hipocampal
                // como proxy (o hipocampo já modulou o padrão com emoção e theta)
                let memory_echo = NeuralEnactiveMemory::from_firing_rates(
                    elapsed as f64,
                    emotion,
                    arousal,
                    &hippo_out,
                    &hippo_out,
                    hippo_out.clone(),
                    format!("hippo_echo_{}", step),
                );
                let _ = tx_feedback.send(memory_echo);
            }
        }

        frontal.set_dopamine(neuro.dopamine + emotion);
        // Corrente inter-lobe para o frontal (limbic bias + parietal + temporal)
        let frontal_inter: Vec<f32> = inter_lobe_currents.para_frontal.iter()
            .map(|&v| v * 0.08).collect();
        let recognized_with_inter: Vec<f32> = recognized.iter().enumerate()
            .map(|(i, &r)| r + frontal_inter.get(i).copied().unwrap_or(0.0))
            .collect();
        let action = frontal.decide(&recognized_with_inter, &internal_goal, dt, current_time, &config);

        // P2 — goal_queue ciclo feedback: avalia se o output atual satisfez o goal.
        // RPE positivo → goal cumprido → dopamina sobe → reforço das sinapses usadas.
        // Fecha o loop detect→plan→execute→avaliar que estava ausente.
        {
            let (goal_rpe, goal_desc) = frontal.avaliar_goal(&action);
            if goal_rpe.abs() > 0.01 {
                neuro.dopamine = (neuro.dopamine + goal_rpe).clamp(0.3, 2.0);
                if let Ok(mut bs) = brain_state.try_lock() {
                    bs.ultimo_rpe = goal_rpe;
                }
                if let Some(desc) = goal_desc {
                    log::debug!("🎯 [Goal] concluído='{}' rpe={:.2} dopa={:.2}", desc, goal_rpe, neuro.dopamine);
                }
            }
        }

        // F1c. Replay hipocampal em idle — consolida memórias sem estímulo externo
        // Biologicamente: sharp-wave ripples durante waking rest consolidam episódios em semântica
        if idle_replay_agora && step % 10 == 0 {
            let _ = hippocampus.memorize_with_connections(
                &prev_hippo_rates, 0.1, dt, current_time, &config,
            );
        }

        // F2. CEREBELO — corrige o output do frontal
        // climbing_error = discrepância entre o que o frontal decidiu e o que o temporal reconheceu.
        // O cerebelo aprende a compensar esse erro ao longo do tempo (LTD cerebelar).
        // Output Purkinje é inibitório (-1.0 ou 0.0): suprime componentes do action com erro alto.
        let climbing_error: Vec<f32> = action.iter().zip(recognized.iter())
            .map(|(a, r)| (a - r).clamp(-1.0, 1.0))
            .collect();
        let cerebelo_out = cerebelo.compute_motor_output(
            &recognized, &climbing_error, dt, current_time, &config
        );
        // Aplica correção cerebelar ao action: expande n_purkinje → n_neurons via interpolação
        // e escala para correção suave (±0.05). Biologicamente: cerebelo → núcleos cerebelares
        // → tálamo → córtex motor (frontal), refinando timing e amplitude do comando motor.
        let action: Vec<f32> = {
            let n = action.len();
            let n_cerb = cerebelo_out.len();
            action.iter().enumerate().map(|(i, &a)| {
                let cerb_idx = (i * n_cerb / n).min(n_cerb.saturating_sub(1));
                let correcao = cerebelo_out.get(cerb_idx).copied().unwrap_or(0.0) * 0.05;
                (a + correcao).clamp(0.0, 1.0)
            }).collect()
        };

        // F3. RL com persistência periódica
        {
            // Absorve recompensa/punição externa (reward/punish do jogo ou interface WS).
            // Sem isso o sinal do jogo nunca chega à Q-table — o loop usa neuro.dopamine
            // local e não lê bs.neurotransmissores diretamente.
            if let Ok(mut bs) = brain_state.try_lock() {
                if bs.recompensa_pendente.abs() > 0.001 {
                    neuro.dopamine = (neuro.dopamine + bs.recompensa_pendente).clamp(0.3, 2.0);
                    bs.recompensa_pendente = 0.0;
                }
            }

            let action_scalar = action.iter().sum::<f32>() / action.len().max(1) as f32;
            let rl_rpe = rl.update(&recognized, neuro.dopamine, action_scalar, &config);
            // Fix 5: floor em 0.3 previne espiral dopaminérgica.
            // RPE negativo contínuo deprimia dopamina → reward sempre negativo → ciclo vicioso.
            // Baseline biológica: neurônios dopaminérgicos da SNpc nunca param completamente.
            neuro.dopamine = (neuro.dopamine + rl_rpe * 0.04).clamp(0.3, 2.0);
            // Fix 7: Thalamus aprende a filtrar com base no erro do RL.
            // RPE positivo (recompensa inesperada) → abre o filtro (mais sensível).
            // RPE negativo (punição) → fecha o filtro (filtra ruído para focar).
            thalamus.adapt_filter(rl_rpe * 0.1);
            // Fix 1: Propaga RPE ao BrainState para LTD/LTP no grafo semântico (server.rs).
            // O grafo de palavras aprende quais associações produzem bons/maus resultados.
            //
            // CICLO DE DECISÃO: além do RPE, propaga Q-values por palavra e goal do frontal.
            // Isso fecha o loop: experiência → RL → Q-value → influencia fala.
            if let Ok(mut bs) = brain_state.try_lock() {
                bs.ultimo_rpe = rl_rpe;

                // Q-values por palavra: palavras ativas no neural_context são
                // responsáveis pelo padrão temporal atual. Associa o Q-value
                // calculado a cada uma delas com decaimento exponencial (0.9).
                // RPE forte (|rpe|>0.1) → atualiza mais agressivamente.
                if rl_rpe.abs() > 0.05 {
                    let q_atual = rl.valor_de(&recognized);
                    let palavras_ativas: Vec<String> = bs.neural_context.iter()
                        .filter(|w| w.len() >= 2)
                        .cloned()
                        .collect();
                    for palavra in palavras_ativas {
                        let entry = bs.palavra_qvalores.entry(palavra).or_insert(0.0);
                        // Média exponencial: 90% histórico + 10% novo sinal
                        *entry = *entry * 0.90 + q_atual * 0.10;
                        // Clamp para evitar deriva em sessões longas
                        *entry = entry.clamp(-2.0, 2.0);
                    }
                }

                // Frontal goal words: extrai tokens da descrição do goal atual.
                // O chat handler injeta essas palavras no contexto do walk,
                // fazendo respostas serem influenciadas pela intenção do PFC.
                bs.frontal_goal_words = frontal.goal_queue.front()
                    .map(|g| {
                        g.descricao.split_whitespace()
                            .filter(|w| w.len() >= 3)
                            .map(|w| w.to_lowercase())
                            .collect()
                    })
                    .unwrap_or_default();

                // Habituation level: média do contador de habituação da amígdala,
                // normalizada para [0.0, 1.0]. Alta habituação → Selene busca novidade.
                let hab_sum: u32 = limbic.habituation_counter.iter().sum();
                let hab_n = limbic.habituation_counter.len().max(1) as f32;
                // Contador satura em ~20 (ver limbic.rs: habituado após >8 ticks)
                bs.habituation_nivel = ((hab_sum as f32 / hab_n) / 20.0).clamp(0.0, 1.0);

                // Novos campos cognitivos
                bs.acc_conflict    = cingulate.conflict_signal;
                bs.acc_social_pain = cingulate.social_pain;
                bs.ofc_value_bias  = ofc.value_bias;
                bs.oxytocin_level  = neuro.oxytocin;
                bs.amygdala_fear   = amygdala.fear_signal;
                bs.amygdala_extinction = amygdala.extinction_trace;
                // Wernicke: consome um lote da fila FIFO por tick (evita starvation)
                if let Some(tokens) = bs.pending_wernicke_tokens.pop_front() {
                    let valencias_w = if let Ok(sw) = swap_manager.try_lock() {
                        sw.valencias_palavras()
                    } else {
                        std::collections::HashMap::new()
                    };
                    let score = language.wernicke_process(
                        &tokens, &valencias_w, dt, current_time, &config,
                    );
                    bs.wernicke_comprehension = score;
                }
                let frontal_goal_signal = frontal.goal_queue.front()
                    .map(|g| g.prioridade).unwrap_or(0.0);
                let (fluency, _syntax) = language.broca_plan(
                    frontal_goal_signal, emotion,
                    bs.wernicke_comprehension,
                    dt, current_time, &config,
                );
                bs.broca_fluency = fluency;
            }
            // DepthStack: atualiza atenção de abstração com base no RPE.
            // RPE positivo → camadas mais abstratas (D2) ganham mais atenção.
            // RPE negativo → sistema ancora no substrato bruto (D0).
            temporal.apply_rpe(rl_rpe);

            // ── F4. NOVOS LOBOS COGNITIVOS ────────────────────────────────
            // ACC: detecta conflito entre atividade frontal e estado límbico.
            // Roda a cada tick (peso leve) — sinal de conflito modula NA e WM.
            {
                let frontal_activity = action.iter().sum::<f32>() / action.len().max(1) as f32;
                let (conflict_sig, adj_factor) = cingulate.update(
                    frontal_activity, emotion, rl_rpe, dt, current_time, &config
                );
                // ACC → noradrenalina: conflito alto recruta locus coeruleus
                let na_drive = cingulate.noradrenaline_drive();
                if na_drive > 0.0 {
                    neuro.noradrenaline = (neuro.noradrenaline + na_drive).clamp(0.0, 2.0);
                }
                // ACC → amígdala: rACC inibe amígdala quando dor social alta
                // (aplicado como redução no reward_signal do próximo tick via dopamina)
                let amy_inhib = cingulate.amygdala_inhibition();
                if amy_inhib > 0.1 {
                    neuro.dopamine = (neuro.dopamine - amy_inhib * 0.05).clamp(0.3, 2.0);
                }
                // Propaga adj_factor e social_pain ao BrainState
                if let Ok(mut bs) = brain_state.try_lock() {
                    bs.ultimo_estado_corpo[2] = (bs.ultimo_estado_corpo[2]
                        + na_drive).clamp(0.0, 2.0); // NA atualizado
                    // social_pain disponível para server.rs via campo dedicado abaixo
                }
            }

            // OFC: atualiza valor esperado do contexto + detecta reversal.
            // Roda a cada 10 ticks (mapa de valor muda lentamente).
            if step % 10 == 0 {
                let ctx_ofc: Vec<String> = if let Ok(bs) = brain_state.try_lock() {
                    bs.neural_context.iter().cloned().collect()
                } else { Vec::new() };
                let (val_bias, reversal_sig, ltd_boost) = ofc.update(
                    &ctx_ofc, rl_rpe, dt, current_time, &config
                );
                // Reversal → acelera LTD no swap para arestas do contexto atual
                if reversal_sig > 0.3 {
                    if let Ok(mut sw) = swap_manager.try_lock() {
                        let pares_penalizar: Vec<(String, String, f32)> = ctx_ofc.iter()
                            .zip(ctx_ofc.iter().skip(1))
                            .map(|(a, b)| (a.clone(), b.clone(), -(ltd_boost * 0.2).clamp(0.02, 0.3)))
                            .collect();
                        if !pares_penalizar.is_empty() {
                            sw.importar_causal(pares_penalizar);
                        }
                    }
                }
                // OFC exporta valências como pares causais reflexivos a cada 100 ticks.
                // Palavras com valor positivo reforçam self-connection (word→word, peso positivo).
                // Palavras com valor negativo recebem peso negativo (extinção de arestas).
                if step % 100 == 0 {
                    let pares_valor = ofc.export_value_pairs(0.2);
                    if !pares_valor.is_empty() {
                        if let Ok(mut sw) = swap_manager.try_lock() {
                            let pares_causal: Vec<(String, String, f32)> = pares_valor.iter()
                                .map(|(w, v)| (w.clone(), w.clone(), (v * 0.1).clamp(-0.3, 0.3)))
                                .collect();
                            sw.importar_causal(pares_causal);
                        }
                    }
                }
            }

            // Cerebelo → PFC (via tálamo ventrolateral): timing cognitivo.
            // Projeção anatômica: Cerebelo dentado → tálamo VL → PFC.
            // Biologicamente contribui para: timing de linguagem, predição sequencial.
            // Aplica 5% do sinal cerebelar como boost ao working_memory_trace do frontal.
            if step % 5 == 0 {
                let n_cerb = cerebelo_out.len();
                let wm_len = frontal.working_memory_trace.len();
                for i in 0..wm_len {
                    let cerb_idx = i * n_cerb / wm_len.max(1);
                    let mag = cerebelo_out.get(cerb_idx).copied().unwrap_or(0.0).abs();
                    frontal.working_memory_trace[i] =
                        (frontal.working_memory_trace[i] + mag * 0.05).clamp(0.0, 1.0);
                }
            }

            // Amígdala (BLA + CeA): atualiza condicionamento emocional.
            // Roda a cada 5 ticks — muda mais lentamente que o tick neural.
            if step % 5 == 0 {
                let amy_inhib_acc = cingulate.amygdala_inhibition();
                let (fear_sig, arousal_boost) = amygdala.update(
                    emotion, amy_inhib_acc, neuro.noradrenaline, dt, current_time, &config
                );
                // CeA → cortisol: medo ativa eixo HPA
                let cortisol_drive = amygdala.cortisol_drive();
                if cortisol_drive > 0.0 {
                    neuro.cortisol = (neuro.cortisol + cortisol_drive).clamp(0.0, 1.5);
                }
                // CeA → noradrenalina: arousal de medo ativa LC
                if arousal_boost > 0.1 {
                    neuro.noradrenaline = (neuro.noradrenaline + arousal_boost * 0.15).clamp(0.0, 2.0);
                }
                // Rejeição severa → amígdala registra aversão
                if rl_rpe < -0.4 {
                    amygdala.registrar_aversao((-rl_rpe - 0.4).clamp(0.0, 0.6));
                }
            }

            // Snapshot do grafo a cada 1000 ticks (preserva estado semântico bom)
            if step % 1000 == 0 && step > 0 {
                if let Ok(mut sw) = swap_manager.try_lock() {
                    sw.criar_snapshot(step as u64);
                    // One-shot: consolida ou descarta fast_weights
                    sw.consolidar_fast_weights();
                }
            }

            // Embeddings: atualiza co-ativações semânticas a cada 50 ticks
            if step % 50 == 0 {
                if let Ok(bs) = brain_state.try_lock() {
                    let ctx: Vec<String> = bs.neural_context.iter().cloned().collect();
                    drop(bs);
                    if ctx.len() >= 2 {
                        if let Ok(mut sw) = swap_manager.try_lock() {
                            for i in 0..ctx.len().saturating_sub(1) {
                                sw.atualizar_embeddings_coativacao(&ctx[i], &ctx[i+1], 0.008);
                            }
                        }
                    }
                }
            }

            // Ocitocina: interações positivas (RPE > 0.2) liberam ocitocina.
            // D1/D2 são atualizados automaticamente no neuro.update(), mas podemos
            // propagar o efeito D1 ao frontal: D1 alto = WM mais estável.
            if rl_rpe > 0.2 {
                neuro.registrar_interacao_positiva(rl_rpe.clamp(0.0, 1.0));
            } else if rl_rpe < -0.2 {
                neuro.registrar_rejeicao((-rl_rpe).clamp(0.0, 1.0));
                cingulate.registrar_rejeicao((-rl_rpe).clamp(0.0, 1.0));
            }
            // D1 → WM boost: dopamina alta via D1 aumenta estabilidade da WM
            {
                let d1_boost = (neuro.d1_signal - 0.5).max(0.0) * 0.02;
                if d1_boost > 0.0 {
                    let wm_len = frontal.working_memory_trace.len();
                    for v in frontal.working_memory_trace.iter_mut().take(wm_len) {
                        *v = (*v + d1_boost).clamp(0.0, 1.0);
                    }
                }
            }
            // LobeRouter: especialização competitiva — vencedor aproxima embedding do query,
            // perdedores afastam. O RPE amplifica ou atenua a magnitude do update.
            lobe_router.update_specialization(rl_rpe);
            rl_save_counter += 1;
            // Salva Q-table a cada ~60s (12000 ticks @ 200Hz)
            if rl_save_counter >= 12000 {
                rl_save_counter = 0;
                if let Err(e) = rl.salvar_em_arquivo("selene_qtable.bin") {
                    log::warn!("[RL] Falha ao salvar Q-table: {}", e);
                }
            }
        }

        // F4. STDP inter-lobe (a cada 10 ticks — não precisa ser a cada tick)
        if step % 10 == 0 {
            brain_conn.stdp_update_all(
                &prev_v1_rates, &recognized, &prev_parietal_rates,
                &action, &prev_limbic_rates, &prev_hippo_rates,
                dt * 1000.0 * 10.0,
            );
        }

        // P2-B: Hebbian temporal → swap (a cada 200 ticks ≈ 1s).
        // Exporta pares de neurônios temporais com conexão Hebbiana forte (≥0.2)
        // e os converte em co-ativações semânticas no swap_manager.
        // Os índices de neurônio são mapeados para palavras via neural_context:
        // se o contexto tem N palavras, o neurônio i corresponde à palavra i%N.
        // Isso propaga o aprendizado implícito de co-ocorrência do temporal
        // para o grafo explícito de conceitos — fechando o loop temporal→semântica.
        if step % 200 == 0 {
            let pares_hebb = temporal.hebbian_pares_fortes(0.2);
            if !pares_hebb.is_empty() {
                if let Ok(bs) = brain_state.try_lock() {
                    let ctx: Vec<String> = bs.neural_context.iter().cloned().collect();
                    drop(bs);
                    if ctx.len() >= 2 {
                        let n_ctx = ctx.len();
                        let pares_swap: Vec<(String, String, f32)> = pares_hebb.iter()
                            .take(10) // limite por tick para não sobrecarregar
                            .filter_map(|&(i, j, peso)| {
                                let w1 = ctx.get(i % n_ctx)?;
                                let w2 = ctx.get(j % n_ctx)?;
                                if w1 == w2 || w1.len() < 2 || w2.len() < 2 { return None; }
                                Some((w1.clone(), w2.clone(), peso * 0.3))
                            })
                            .collect();
                        if !pares_swap.is_empty() {
                            if let Ok(mut sw) = swap_manager.try_lock() {
                                sw.importar_causal(pares_swap.clone());
                            }
                            log::debug!("[Hebbian→Swap] {} pares exportados do temporal", pares_swap.len());
                        }
                    }
                }
            }
        }

        // Salva firing rates para o próximo tick (projeções inter-lobe usam t-1)
        let v1_len = vision_full.len().min(n_neurons);
        prev_v1_rates[..v1_len].copy_from_slice(&vision_full[..v1_len]);
        let t_len = recognized.len().min(n_neurons);
        prev_temporal_rates[..t_len].copy_from_slice(&recognized[..t_len]);
        let a_len = action.len().min(n_neurons);
        prev_frontal_rates[..a_len].copy_from_slice(&action[..a_len]);
        let l_len = (n_neurons / 2).min(cochlea_input.len());
        prev_limbic_rates[..l_len].copy_from_slice(&cochlea_input[..l_len]);
        // Hippo rates: subsamplea temporal (proxy do output do hipocampo quando não rodou este tick)
        // Quando hippocampus.memorize_with_connections() rodou, hippo_out está disponível localmente
        // mas é scoped ao if-block acima. Aqui usamos temporal D1 como proxy inter-tick.
        {
            let h_len = prev_hippo_rates.len();
            let d1 = &temporal.depth_stack.d1;
            let src_len = d1.len().min(h_len);
            prev_hippo_rates[..src_len].copy_from_slice(&d1[..src_len]);
        }

        // P3 — depth_stack D1/D2 → neural_context (a cada 100 ticks ≈ 0.5s).
        // Os neurônios mais ativos em D1 (abstração média do temporal) são mapeados
        // para palavras via swap_manager e injetados no neural_context do BrainState.
        // Isso faz a caminhada de resposta refletir o que o temporal "está abstraindo"
        // agora, não só os tokens crus do input.
        if step % 100 == 0 {
            let d1_top = temporal.d1_top_indices(6);
            if !d1_top.is_empty() {
                if let Ok(sw) = swap_manager.try_lock() {
                    let id_to_word = sw.id_para_palavra();
                    // Mapeia índice D1 → UUID canônico via palavra_para_id (por ordem de inserção)
                    let palavras_d1: Vec<String> = {
                        let all_palavras: Vec<&String> = sw.palavra_para_id.keys().collect();
                        d1_top.iter()
                            .filter_map(|&idx| all_palavras.get(idx % all_palavras.len().max(1)).copied())
                            .filter(|w| w.len() >= 3)
                            .cloned()
                            .collect()
                    };
                    drop(sw);
                    if !palavras_d1.is_empty() {
                        if let Ok(mut bs) = brain_state.try_lock() {
                            for w in &palavras_d1 {
                                if !bs.neural_context.contains(w) {
                                    bs.neural_context.push_back(w.clone());
                                }
                            }
                            while bs.neural_context.len() > 20 {
                                bs.neural_context.pop_front();
                            }
                        }
                        log::debug!("[D1→Ctx] {} palavras abstratas injetadas: {:?}", palavras_d1.len(), palavras_d1);

                        // P1.1 — PatternEngine: grava episódio de pensamento espontâneo
                        if emotion.abs() > 0.15 {
                            use crate::learning::pattern_engine::FonteEpisodio;
                            let neuro_snap = [neuro.dopamine, neuro.serotonin, neuro.noradrenaline,
                                              neuro.acetylcholine, neuro.cortisol];
                            if let Ok(mut bs) = brain_state.try_lock() {
                                let tms = current_time * 1000.0;
                                bs.pattern_engine.gravar(
                                    tms as f64, FonteEpisodio::Pensamento,
                                    palavras_d1.clone(),
                                    format!("pens_step_{}", step),
                                    None, emotion, neuro_snap,
                                );
                            }
                        }
                    }
                }
            }
        }

        // ── FASE 1d: Onda dominante → profundidade da caminhada no grafo ──────
        // Derivado do estado neurológico atual — a onda modula quantos passos
        // o graph-walk percorre, refletindo o estado atencional:
        //   delta (ocioso profundo): 6 passos — respostas curtas e simples
        //   theta/alpha (repouso alerta): 9 passos — respostas contemplativas
        //   beta (foco ativo): 10 passos — respostas precisas
        //   gamma (aprendizado intenso): 13 passos — respostas ricas e expansivas
        let n_passos_walk: usize = if neuro.dopamine > 1.1 && atividade_recente > 0.04 {
            13  // gamma
        } else if neuro.dopamine > 0.75 && atividade_recente > 0.01 {
            10  // beta
        } else if neuro.serotonin > 0.65 {
            9   // alpha
        } else if neuro.serotonin > 0.4 {
            8   // theta
        } else {
            6   // delta
        };

        // G. Núcleos da Base
        basal_ganglia.update_habits(&vision_full, &action, emotion);
        // Fix 4: hábito sugerido → frontal.planejar() com prioridade proporcional à força
        // Os núcleos da base agora influenciam decisões do frontal: comportamentos
        // reforçados no passado (alta recompensa) entram como goals de alta prioridade.
        if let Some(habit_action) = basal_ganglia.suggest_action(&vision_full) {
            let prioridade = basal_ganglia.habitos.iter()
                .map(|h| h.forca)
                .fold(0.0f32, f32::max)
                .clamp(0.3, 0.9);
            frontal.planejar(habit_action, prioridade, "habito_basal");
            log::debug!("🔄 Hábito → frontal goal (prioridade={:.2})", prioridade);
        }

        // Fix 6: Mirror neurons — decaimento por tick + aprendizado da ação atual.
        // O output motor do frontal (action) ensina: "quando executo 'action', é assim que parece."
        // Esse template é depois ativado quando Selene OBSERVA palavras similares no input do usuário.
        {
            mirror.decay();
            // A cada 50 ticks: aprende do output motor + observa neural_context + propaga ressonância
            if step % 50 == 0 {
                if let Ok(mut bs) = brain_state.try_lock() {
                    let ctx_words: Vec<String> = bs.neural_context.iter().cloned().collect();
                    for palavra in ctx_words.iter().take(3) {
                        mirror.learn_from_action(palavra, &action);
                    }
                    // Observa e propaga ressonância
                    let ressonancia = mirror.observe(&ctx_words);
                    bs.mirror_resonance = ressonancia;
                    if mirror.is_resonating() {
                        let empatia = mirror.empatia_bias(bs.emocao_bias);
                        bs.emocao_bias = (bs.emocao_bias + empatia).clamp(-1.0, 1.0);
                    }
                }
                // P2-A: mirror.wm_signal() → frontal working memory.
                // Quando há ressonância ativa, o sinal espelho (8 dims da ativação motora)
                // entra na WM como goal de baixa prioridade — "simulação interna da ação
                // observada" que colore o planejamento frontal com o estado empático atual.
                if mirror.is_resonating() {
                    let wm_sig = mirror.wm_signal();
                    // Expande 8 → n_neurons via repetição periódica para compatibilidade
                    let n = action.len();
                    let padrao_wm: Vec<f32> = (0..n)
                        .map(|i| wm_sig[i % wm_sig.len()] * mirror.resonance_score * 0.4)
                        .collect();
                    frontal.planejar(padrao_wm, mirror.resonance_score * 0.4, "mirror_resonance");
                    log::debug!("🪞 [Mirror→WM] ressonância={:.2} → goal frontal", mirror.resonance_score);
                }
            }
        }

        // G1. Propaga RPE (sinal dopaminérgico) para reforçar/enfraquecer chunks recentes
        {
            let rpe = neuro.dopamine - 0.5; // desvio da baseline dopaminérgica
            chunking.aplicar_rpe(rpe);
        }

        // P3 — spatial_map parietal → grounding de saliência (a cada 150 ticks).
        // Os neurônios parietais mais salientes indicam "onde a atenção espacial está".
        // Mapeamos seus índices para palavras no swap e incrementamos o grounding score
        // dessas palavras — palavras que "aparecem" enquanto a atenção é alta ganham
        // prioridade como âncoras em gerar_resposta_emergente.
        if step % 150 == 0 {
            let top_parietal: Vec<usize> = {
                let mut indexed: Vec<(usize, f32)> = parietal.spatial_map.iter()
                    .enumerate()
                    .filter(|(_, &v)| v > 0.15)
                    .map(|(i, &v)| (i, v))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.into_iter().take(4).map(|(i, _)| i).collect()
            };
            if !top_parietal.is_empty() {
                if let Ok(sw) = swap_manager.try_lock() {
                    let all_palavras: Vec<String> = sw.palavra_para_id.keys().cloned().collect();
                    let palavras_parietal: Vec<String> = top_parietal.iter()
                        .filter_map(|&idx| all_palavras.get(idx % all_palavras.len().max(1)).cloned())
                        .filter(|w| w.len() >= 3)
                        .collect();
                    drop(sw);
                    if !palavras_parietal.is_empty() {
                        if let Ok(mut bs) = brain_state.try_lock() {
                            for w in &palavras_parietal {
                                let g = bs.grounding.entry(w.clone()).or_insert(0.0);
                                *g = (*g + 0.05).min(1.0);
                            }
                        }
                        log::debug!("[Parietal→Grounding] {:?}", palavras_parietal);
                    }
                }
            }
        }

        // G2. Grounding semântico — vincula palavras do neural_context ao contexto perceptual.
        // A cada 100 ticks (~0.5s): registra um EventoEpisodico com o estado perceptual atual.
        // Isso implementa a camada de significado: palavras co-ativadas com percepções reais
        // ganham grounding score maior e são preferidas como âncoras nas respostas.
        if step % 100 == 0 {
            use crate::encoding::spike_codec::{features_to_spike_pattern, bands_to_spike_pattern};
            let vpad = features_to_spike_pattern(
                &prev_v1_rates.iter().take(8).map(|&v| v * 100.0).collect::<Vec<_>>()
            );
            let apad = bands_to_spike_pattern(
                &cochlea_input.iter().take(32).copied().collect::<Vec<_>>()
            );
            if let Ok(mut bs) = brain_state.try_lock() {
                let palavras: Vec<String> = bs.neural_context.iter()
                    .filter(|w| w.len() > 2)
                    .cloned()
                    .take(8)
                    .collect();
                if !palavras.is_empty() {
                    let tms = current_time * 1000.0;
                    bs.grounding_bind(&palavras, vpad, apad, emotion, atividade_recente, tms as f64);

                    // P1.1 — PatternEngine: grava episódio visual quando há percepção real
                    use crate::learning::pattern_engine::FonteEpisodio;
                    let neuro_snap = [neuro.dopamine, neuro.serotonin, neuro.noradrenaline,
                                      neuro.acetylcholine, neuro.cortisol];
                    let visual_ativo = bs.sensor_flags.video_ativo.load(std::sync::atomic::Ordering::Relaxed);
                    let audio_ativo  = bs.sensor_flags.audio_ativo.load(std::sync::atomic::Ordering::Relaxed);
                    if visual_ativo && emotion.abs() > 0.1 {
                        let ctx_vis = palavras.clone();
                        bs.pattern_engine.gravar(
                            tms as f64, FonteEpisodio::Visual,
                            ctx_vis, format!("vis_step_{}", step),
                            None, emotion, neuro_snap,
                        );
                    }
                    if audio_ativo && emotion.abs() > 0.05 {
                        let ctx_amb = palavras.clone();
                        bs.pattern_engine.gravar(
                            tms as f64, FonteEpisodio::Ambiente,
                            ctx_amb, format!("amb_step_{}", step),
                            None, emotion, neuro_snap,
                        );
                    }
                }
                // Propaga RPE ao grounding (predições corretas solidificam o grounding)
                let rpe_atual = bs.ultimo_rpe;
                bs.grounding_rpe(rpe_atual);
            }
        }

        // Decaimento global de grounding a cada 1000 ticks (~5s)
        if step % 1000 == 0 {
            if let Ok(mut bs) = brain_state.try_lock() {
                bs.grounding_decay();
            }
        }

        // H. Comunicação entre hemisférios — dados reais de spikes
        // Hemisfério esquerdo (temporal/linguagem) → direito (parietal/espacial)
        // Hemisfério direito (parietal/visual) → esquerdo (frontal/execução)
        //
        // Latência dinâmica: arousal alto → transmissão mais rápida (neurônios mais excitáveis).
        // Arousal [0.5, 3.5] → latência [4ms, 20ms] (inversamente proporcional).
        // Conectividade: serotonina alta = calma → coerência inter-hemisférica maior.
        // Serotonina baixa = estresse → corpus callosum menos confiável (sonolência/fadiga).
        {
            let latencia = (20.0 - (arousal - 0.5) / 3.0 * 16.0).clamp(4.0, 20.0);
            corpus_callosum.set_latency(latencia);
            let conectividade = (neuro.serotonin * 0.6).clamp(0.4, 1.0);
            corpus_callosum.set_connectivity(conectividade);
        }
        if step % 50 == 0 {
            // Esquerdo → Direito: padrão temporal (linguagem/reconhecimento)
            let spikes_temporal: Vec<bool> = temporal.recognition_layer.neuronios.iter()
                .take(n_neurons / 10)
                .map(|n| n.last_spike_ms > 0.0 && n.last_spike_ms >= (current_time * 1000.0) - 50.0)
                .collect();
            corpus_callosum.send_to_right(0, spikes_temporal, current_time);

            // Direito → Esquerdo: padrão parietal (atenção espacial)
            let spikes_parietal: Vec<bool> = parietal.integration_layer.neuronios.iter()
                .take(n_neurons / 10)
                .map(|n| n.last_spike_ms > 0.0 && n.last_spike_ms >= (current_time * 1000.0) - 50.0)
                .collect();
            corpus_callosum.send_to_left(0, spikes_parietal, current_time);
        }
        // Frontal recebe o padrão vindo do hemisfério direito (atenção espacial → decisão)
        if let Some(spikes_parietal_echo) = corpus_callosum.receive_at_left(0, current_time) {
            // Converte spikes booleanos em corrente contínua e injeta no frontal
            let corrente_calosa: Vec<f32> = spikes_parietal_echo.iter()
                .map(|&s| if s { 0.15 } else { 0.0 })
                .collect();
            // Adiciona como bias ao working_memory_trace do frontal
            let wm_len = frontal.working_memory_trace.len();
            for (i, &c) in corrente_calosa.iter().enumerate() {
                if i < wm_len {
                    frontal.working_memory_trace[i] = (frontal.working_memory_trace[i] + c).clamp(0.0, 1.0);
                }
            }
        }
        // Parietal recebe o padrão do hemisfério esquerdo (linguagem → atenção espacial)
        if let Some(spikes_temporal_echo) = corpus_callosum.receive_at_right(0, current_time) {
            let n_spikes_ativos = spikes_temporal_echo.iter().filter(|&&s| s).count();
            log::debug!("🔗 Caloso T→P: {} spikes ativos cruzando", n_spikes_ativos);
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

        // J. Atualização do ego + tick semântico dos neurônios conceituais
        // Pensamentos espontâneos (DMN) são propagados ao brain_state para
        // aparecerem na telemetria da interface neural.
        let body_feeling = interoception.sentir();
        if let Some(pensamento) = ego.update(body_feeling, current_time).await {
            if let Ok(mut brain_guard) = brain_state.try_lock() {
                brain_guard.ego.pensamentos_recentes.push_back(pensamento);
                if brain_guard.ego.pensamentos_recentes.len() > 10 {
                    brain_guard.ego.pensamentos_recentes.pop_front();
                }
            }
        }
        // Fix C: Interoception → ego → linguagem.
        // Executa a cada ~1000 ticks (~5s @ 200Hz) para não saturar o neural_context.
        // Palavras corporais têm baixa prioridade: só entram se houver espaço livre (≤10 slots).
        if step % 1000 == 0 {
            let (descricao_corpo, valencia_corpo) = interoception.influenciar_ego();
            // Propaga o pensamento ao ego (via NarrativeVoice)
            ego.narrative_voice.pensamentos_recentes.push_back(descricao_corpo.clone());
            if ego.narrative_voice.pensamentos_recentes.len() > 8 {
                ego.narrative_voice.pensamentos_recentes.pop_front();
            }
            if let Ok(mut bs) = brain_state.try_lock() {
                // Só adiciona palavras corporais se o contexto não estiver saturado por
                // palavras da conversa (≤ 10 slots ocupados = há espaço para cor corporal)
                if bs.neural_context.len() <= 10 {
                    for palavra in descricao_corpo.split_whitespace() {
                        let p = palavra.to_lowercase()
                            .trim_matches(|c: char| !c.is_alphabetic())
                            .to_string();
                        if p.len() > 2 && !bs.neural_context.contains(&p) {
                            bs.neural_context.push_back(p);
                        }
                    }
                    while bs.neural_context.len() > 20 {
                        bs.neural_context.pop_front();
                    }
                }
                // Valência corporal modula emocao_bias: desconforto → bias negativo
                let bias_corporal = (0.5 - valencia_corpo) * 0.15; // range ±0.075
                bs.emocao_bias = (bs.emocao_bias + bias_corporal).clamp(-1.0, 1.0);
            }
        }

        // ── Reward signal — coerência narrativa ───────────────────────────
        // Cada 1000 ticks (~5s): snapshot do neural_context.
        // Se 2+ palavras aparecem em ≥3 dos 10 últimos snapshots, o tema está
        // estável → pequeno boost de dopamina (recompensa por coerência interna).
        // Cap em 1.5 para não saturar o sistema dopaminérgico.
        if step % 1000 == 0 && step > 0 {
            if let Ok(bs) = brain_state.try_lock() {
                let snapshot: Vec<String> = bs.neural_context.iter().cloned().collect();
                drop(bs);

                if !snapshot.is_empty() {
                    contexto_historico.push_back(snapshot);
                    if contexto_historico.len() > 10 {
                        contexto_historico.pop_front();
                    }

                    // Conta frequência de cada palavra nos últimos snapshots
                    if contexto_historico.len() >= 3 {
                        let mut freq: std::collections::HashMap<&str, u32> =
                            std::collections::HashMap::new();
                        for snap in &contexto_historico {
                            for w in snap {
                                *freq.entry(w.as_str()).or_insert(0) += 1;
                            }
                        }
                        // Palavras estáveis: aparecem em ≥3 snapshots
                        let palavras_estaveis = freq.values().filter(|&&c| c >= 3).count();
                        if palavras_estaveis >= 2 {
                            // Coerência detectada → boost de dopamina
                            let boost = (palavras_estaveis as f32 * 0.015).min(0.06);
                            neuro.dopamine = (neuro.dopamine + boost).min(1.5);
                            log::debug!(
                                "[Reward] Coerência: {} palavras estáveis → dopamina +{:.3} ({:.2})",
                                palavras_estaveis, boost, neuro.dopamine
                            );
                        }
                    }
                }
            }
        }

        // Fix 6: Propaga estado da WM + goal do frontal ao neural_context.
        // O que o frontal está "segurando" na working memory e planejando
        // agora influencia o tema da linguagem. Executa a cada 200 ticks (~1s)
        // para não sobrecarregar o neural_context com dados de WM a cada tick.
        if step % 200 == 0 {
            // Goal atual: palavras da descrição viram sementes de tópico
            if let Some(goal) = frontal.goal_queue.front() {
                if !goal.descricao.is_empty() {
                    if let Ok(mut bs) = brain_state.try_lock() {
                        for palavra in goal.descricao.split_whitespace() {
                            let p = palavra.to_lowercase();
                            if p.len() > 2 && !bs.neural_context.contains(&p) {
                                bs.neural_context.push_back(p);
                            }
                        }
                        while bs.neural_context.len() > 20 {
                            bs.neural_context.pop_front();
                        }
                    }
                }
            }
            // WM slots ativos: slots com alta saliência sinalizam foco atual do frontal.
            // Usamos a saliência como indicador: slot muito saliente → frontal focado nisso.
            // O padrão médio > 0.3 indica ativação real (não ruído de baseline).
            let wm_snap = frontal.wm_snapshots();
            let wm_ativo = wm_snap.iter().any(|(sal, med)| *sal > 0.5 && *med > 0.3);
            if wm_ativo {
                if let Ok(mut bs) = brain_state.try_lock() {
                    // WM ativo: move palavras com representação neural para o final da fila
                    let swap_vocab: std::collections::HashSet<String> = if let Ok(sw) = bs.swap_manager.try_lock() {
                        sw.palavra_para_id.keys().cloned().collect()
                    } else {
                        std::collections::HashSet::new()
                    };
                    let boost: Vec<String> = bs.neural_context.iter()
                        .filter(|w| swap_vocab.contains(w.as_str()))
                        .cloned()
                        .collect();
                    for w in &boost {
                        // Remove a ocorrência existente e reinsere no final (sem duplicar)
                        if let Some(pos) = bs.neural_context.iter().position(|x| x == w) {
                            bs.neural_context.remove(pos);
                            bs.neural_context.push_back(w.clone());
                        }
                    }
                    while bs.neural_context.len() > 20 {
                        bs.neural_context.pop_front();
                    }
                }
            }
        }

        // Atualiza camada glial (step%5: 40Hz — suficiente para Ca²⁺ lento).
        if step % 5 == 0 {
            let regional_activity = GliaLayer::activity_from_firing_rates(&[
                prev_frontal_rates.iter().sum::<f32>() / prev_frontal_rates.len().max(1) as f32,
                prev_parietal_rates.iter().sum::<f32>() / prev_parietal_rates.len().max(1) as f32,
                prev_temporal_rates.iter().sum::<f32>() / prev_temporal_rates.len().max(1) as f32,
                prev_v1_rates.iter().sum::<f32>() / prev_v1_rates.len().max(1) as f32,
                prev_limbic_rates.iter().sum::<f32>() / prev_limbic_rates.len().max(1) as f32,
                prev_hippo_rates.iter().sum::<f32>() / prev_hippo_rates.len().max(1) as f32,
                0.0, // Cerebellum
                brainstem.stats().alertness,
                0.0, // CorpusCallosum
            ]);
            glia.update(&regional_activity, dt * 5.0);
            if let Ok(mut swap) = swap_manager.try_lock() {
                swap.glio_factor = glia.global_glio_factor();
            }
        }

        // Roda Izhikevich+STDP nos neurônios conceituais aprendidos via WS.
        // try_lock() em vez de lock().await — salta o tick semântico se o chat handler
        // estiver com o swap. O STDP tolera alguns ticks pulados sem degradação.
        if let Ok(mut swap) = swap_manager.try_lock() {
            swap.tick_semantico(dt, current_time * 1000.0);
            let conceitos_ativos = swap.conceitos_ativos_top(8);
            drop(swap);
            if !conceitos_ativos.is_empty() {
                if let Ok(mut bs) = brain_state.try_lock() {
                    let dopa_bias = neuro.dopamine - 1.0;
                    for (palavra, ativacao) in &conceitos_ativos {
                        let q_boost = dopa_bias * ativacao * 0.05;
                        let entry = bs.palavra_qvalores.entry(palavra.clone()).or_insert(0.0);
                        *entry = (*entry * 0.95 + q_boost).clamp(-2.0, 2.0);
                    }
                }
            }
        }

        // K. Telemetria
        if step % 500 == 0 {
            // Fase 1e: atualiza metacognição com estado neural atual
            let n_vocab = swap_manager.try_lock()
                .map(|sw| sw.palavra_para_id.len()).unwrap_or(0);
            metacognitive.observe(arousal, emotion, n_vocab);

            let chunk_stats = chunking.stats();
            println!("🧪 [BIO] Sero: {:.2} | Dop: {:.2} | Cort: {:.2} | Emoção: {} | Onda: {}p",
                neuro.serotonin, neuro.dopamine, neuro.cortisol,
                plutchik.dominante(), n_passos_walk,
            );
            let spikes_vivos = temporal.recognition_layer.neuronios.iter()
                .filter(|n| n.last_spike_ms > 0.0
                    && n.last_spike_ms >= (current_time * 1000.0) - 500.0)
                .count();
            neuronios_ativos_handle.store(spikes_vivos, std::sync::atomic::Ordering::Relaxed);
            let (ram_cache, ram_gb) = swap_manager.try_lock()
                .map(|sw| (sw.ram_count(), 0.0f32))
                .unwrap_or((0, 0.0));
            let ram_gb = sensor.try_lock()
                .map(|s| s.get_ram_usage_gb())
                .unwrap_or(ram_gb);
            println!("   🧬 Neurônios disparando: {} | RAM cache: {} | Hábitos: {} | Alerta: {:.2} | RAM: {:.1}GB",
                spikes_vivos, ram_cache, basal_ganglia.stats().num_habitos,
                brainstem.stats().alertness, ram_gb,
            );
            println!("   🧠 META: {} | Vocab: {} palavras",
                metacognitive.descricao(), n_vocab,
            );
            println!("   🔤 {} | Freq: {}Hz | Atividade: {:.3}",
                chunk_stats, freq_hz, atividade_recente
            );
            let abs_level = temporal.depth_stack.abstraction_level();
            let abs_str = if abs_level > 0.6 { "D2-abstrato" } else if abs_level > 0.35 { "D1-médio" } else { "D0-bruto" };
            let hebb_ativos: usize = temporal.hebbian_traces.iter().filter(|&&t| t > 0.3).count();
            println!("   🌀 DepthStack: {abs_str} ({abs_level:.2}) | Hebb ativos: {hebb_ativos} | Galaxy: {}/{}",
                brain_conn.hippo_frontal.peso_medio.abs() > 0.05,
                brain_conn.parietal_hippo.peso_medio.abs() > 0.05,
            );
            println!("   🎯 RL: {} | Cerebelo LTD: {:.3}",
                rl, cerebelo.ltd_factor.iter().sum::<f32>() / cerebelo.ltd_factor.len().max(1) as f32
            );
            // LobeRouter: gate scores e especialização emergente por lóbulo
            {
                use learning::lobe_router::LobeId;
                let ids = LobeId::ALL;
                let gates: Vec<String> = ids.iter().map(|id| {
                    format!("{}:{:.2}", id.nome(), lobe_router.gate(*id))
                }).collect();
                println!("   🛰️  Router gates: {} | skips:{}/{}", gates.join(" | "),
                    6 - ids.iter().filter(|id| lobe_router.gate(**id) >= 0.08).count(), 6);
                let specs = lobe_router.especialidade_dominante();
                let spec_str: Vec<String> = specs.iter()
                    .map(|(nome, dim)| format!("{}→{}", nome, dim))
                    .collect();
                println!("   🧭 Especializ: {} | updates:{}", spec_str.join(" | "),
                    lobe_router.n_especialization_updates);
            }
        }

        // ── Checkpoint automático (timer interno: só salva a cada 4h reais) ─
        // Chamado aqui (step % 5000) para garantir que linguagem.json já foi
        // escrito no disco antes de o checkpoint copiar os arquivos.
        if step % 5000 == 0 && step > 0 {
            if checkpoint_system.tick() {
                // Força save do hipocampo junto ao checkpoint
                hippocampus.save_ltp("selene_hippo_ltp.json");
                if let Err(e) = rl.salvar_em_arquivo("selene_qtable.bin") {
                    log::warn!("[Checkpoint] Falha ao salvar Q-table: {}", e);
                }
            }
        }

        // Fix 3 / Fix 5: Export automático do modelo de linguagem a cada 5000 ticks (~25s).
        // Fonte de verdade: swap_manager (sinapses_conceito) em vez de grafo_associacoes.
        if step % 5000 == 0 && step > 0 {
            // Coleta dados do swap (fora do lock do brain_state)
            let (valencias_export, grafo_export) = if let Ok(mut sw) = swap_manager.try_lock() {
                (sw.valencias_palavras(), sw.grafo_palavras())
            } else {
                (std::collections::HashMap::<String, f32>::new(),
                 std::collections::HashMap::<String, Vec<(String, f32)>>::new())
            };
            let causal_export: std::collections::HashMap<String, Vec<(String, f32)>> = std::collections::HashMap::new();
            let export_payload = if !valencias_export.is_empty() {
                if let Ok(bs) = brain_state.try_lock() {
                    let n_assoc: usize = grafo_export.values().map(|v| v.len()).sum();
                    let ctx_vec: Vec<String> = bs.neural_context.iter().cloned().collect();
                    let json = crate::storage::exportar_linguagem(
                        &valencias_export,
                        &grafo_export,
                        &bs.frases_padrao,
                        &causal_export,
                        &bs.grounding,
                        &bs.emocao_palavras,
                        &bs.auto_learn_contagem,
                        &ctx_vec,
                    );
                    let tracos_json = serde_json::to_string_pretty(&bs.ego.tracos).ok();
                    let pens: Vec<&String> = bs.ego.pensamentos_recentes.iter().collect();
                    let pens_json = serde_json::to_string_pretty(&pens).ok();
                    // Autobiografia: sentimento + memorias_autobiograficas
                    let memorias_vec: Vec<(&str, f32)> = bs.ego.memorias_autobiograficas
                        .iter().map(|(d, v)| (d.as_str(), *v)).collect();
                    let autobiografia_json = serde_json::to_string_pretty(&serde_json::json!({
                        "sentimento": bs.ego.sentimento,
                        "memorias": memorias_vec,
                    })).ok();
                    // HypothesisEngine serializado para escrita fora do lock
                    let hypotheses_json = serde_json::to_string_pretty(&bs.hypothesis_engine).ok();
                    let stats = (valencias_export.len(), n_assoc, 0usize, bs.grounding.len());
                    Some((json, tracos_json, pens_json, autobiografia_json, hypotheses_json, stats))
                } else { None }
            } else { None };

            // Fase 2: escreve fora do lock — tokio::fs não bloqueia o executor
            if let Some((json, tracos_json, pens_json, autobiografia_json, hypotheses_json, (nv, na, nc, ng))) = export_payload {
                if let Err(e) = tokio::fs::write("selene_linguagem.json", json).await {
                    log::warn!("[AUTO-EXPORT] Falha ao salvar linguagem: {}", e);
                } else {
                    println!("💾 [AUTO-EXPORT] Linguagem salva: {} palavras, {} assoc, {} causal, {} grounded",
                        nv, na, nc, ng);
                }
                if let Some(tj) = tracos_json {
                    let _ = tokio::fs::write("selene_ego.json", tj).await;
                }
                if let Some(pj) = pens_json {
                    let _ = tokio::fs::write("selene_memoria_ego.json", pj).await;
                }
                if let Some(aj) = autobiografia_json {
                    let _ = tokio::fs::write("selene_autobiografia.json", aj).await;
                }
                if let Some(hj) = hypotheses_json {
                    let _ = tokio::fs::write("selene_hypotheses.json", hj).await;
                }
            }
            // Auto-save do estado semântico do swap_manager
            if let Ok(swap) = swap_manager.try_lock() {
                if let Err(e) = swap.salvar_estado("selene_swap_state.json") {
                    log::warn!("[SwapState] Falha ao salvar: {}", e);
                }
            }

            // ── Atualiza e salva métricas de ontogenia ───────────────────────────
            {
                let (vocab_n, edges_n) = if let Ok(mut sw) = swap_manager.try_lock() {
                    let v = sw.palavra_para_id.len();
                    let e: usize = sw.sinapses_conceito.len();
                    (v, e)
                } else { (0, 0) };
                let progressou = if let Ok(mut bs) = brain_state.try_lock() {
                    let rpe = bs.ultimo_rpe;
                    bs.ontogeny.tick(vocab_n, edges_n, Some(rpe), None)
                } else { false };
                if let Ok(bs) = brain_state.try_lock() {
                    bs.ontogeny.salvar("selene_ontogeny.json");
                    if progressou {
                        println!("🧒 [ONTOGENY] Progressão automática → {}", bs.ontogeny.stage);
                    }
                }
            }
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

        // Lê sinal de atividade WS e atualiza brain_state com estado neural real
        // Isso garante 200Hz durante sessões de treinamento mesmo sem spikes neurais
        // e permite que o chat handler use step/neurotransmissores/emocao atualizados.
        let ws_sinal = {
            if let Ok(mut brain_guard) = brain_state.try_lock() {
                let v = brain_guard.ws_atividade;
                brain_guard.ws_atividade = (v * 0.985).max(0.0);
                // Atualiza estado neural real no brain_state para o chat handler usar
                let emocao_real = (neuro.serotonin - 0.5).clamp(-1.0, 1.0);
                brain_guard.atividade = (step, brainstem.stats().alertness, emocao_real);
                brain_guard.neurotransmissores = (neuro.dopamine, neuro.serotonin, neuro.noradrenaline);
                // Fase 1a + 1d: propaga viés emocional e profundidade de walk para o chat handler
                brain_guard.emocao_bias = emocao_bias;
                brain_guard.n_passos_walk = n_passos_walk;
                v
            } else { 0.0 }
        };

        // EMA rápida: α=0.1; usa o maior entre spikes neurais e sinal WS
        let atividade_combinada = atividade_tick.max(ws_sinal);
        atividade_recente = atividade_recente * 0.90 + atividade_combinada * 0.10;

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

        // O. Limpeza periódica + persistência do hipocampo (LTP)
        if step % 10000 == 0 {
            let mut swap = swap_manager.lock().await;
            let _ = swap.limpar_neurônios_inativos();
            hippocampus.save_ltp("selene_hippo_ltp.json");
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