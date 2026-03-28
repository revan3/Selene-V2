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

// Metacognição
use meta::MetaCognitive;

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
    let mut hippocampus = Hippocampus::new(n_neurons / 2, &config);
    hippocampus.load_ltp("selene_hippo_ltp.json"); // restaura pesos sinápticos persistidos
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

    // --- 12d. METACOGNIÇÃO ---
    let mut metacognitive = MetaCognitive::new();
    // Firing rates do tick anterior — usados pelas projeções inter-lobe
    let mut prev_v1_rates:       Vec<f32> = vec![0.0; n_neurons];
    let mut prev_temporal_rates: Vec<f32> = vec![0.0; n_neurons];
    let mut prev_parietal_rates: Vec<f32> = vec![0.0; n_neurons];
    let mut prev_frontal_rates:  Vec<f32> = vec![0.0; n_neurons];
    let mut prev_limbic_rates:   Vec<f32> = vec![0.0; n_neurons / 2];
    let mut prev_hippo_rates:    Vec<f32> = vec![0.0; n_neurons / 2];
    // Contador para salvar RL periodicamente (a cada ~60s @ 200Hz = 12000 ticks)
    let mut rl_save_counter: u64 = 0;

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
        // Adenosina acumula com o tempo acordado (pressão de sono biológica):
        // 0.0 ao acordar → ~0.9 após 16h, revertido pelo sono.
        {
            let sensor_lock = sensor.lock().await;
            let cpu_temp = sensor_lock.get_cpu_temp();
            let adenosina = (tempo_acordado.as_secs_f32() / (16.0 * 3600.0)).clamp(0.0, 0.95);
            interoception.update(adenosina, cpu_temp, neuro.noradrenaline);
            brainstem.update(adenosina, dt);
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
            let n_vocab = brain_state.try_lock()
                .map(|s| s.palavra_valencias.len()).unwrap_or(100);
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
        // Quando o acumulador interno do audio.rs detecta fim de palavra,
        // AudioSignal.palavra_completa fica Some(bandas_medias).
        // Aqui fazemos a busca no spike_vocab e injetamos em neural_context —
        // mesmo mecanismo do WebSocket audio_raw, mas para microfone físico.
        if let Some(ref sig) = audio_signal {
            if let Some(ref bandas_palavra) = sig.palavra_completa {
                use crate::encoding::spike_codec::{bands_to_spike_pattern, similarity as spike_sim};
                let audio_pat = bands_to_spike_pattern(bandas_palavra);
                if let Ok(mut bs) = brain_state.try_lock() {
                    let mut melhor: Option<String> = None;
                    let mut sim_max: f32 = 0.0;
                    for (chave, pat_ref) in &bs.spike_vocab {
                        if let Some(palavra) = chave.strip_prefix("audio:") {
                            let s = spike_sim(&audio_pat, pat_ref);
                            if s > sim_max { sim_max = s; melhor = Some(palavra.to_string()); }
                        }
                    }
                    if sim_max >= 0.55 {
                        if let Some(ref palavra) = melhor {
                            println!("🎙️ [MIC] Reconheceu «{}» (sim={:.2})", palavra, sim_max);
                            if bs.neural_context.len() >= 20 { bs.neural_context.pop_front(); }
                            bs.neural_context.push_back(palavra.clone());
                            let vpad = bs.ultimo_padrao_visual;
                            let apad = audio_pat;
                            let emocao = bs.emocao_bias;
                            bs.grounding_bind(&[palavra.clone()], vpad, apad, emocao, sim_max, 0.0);
                        }
                    }
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
        let (vision_features, (emotion, arousal)) = rayon::join(
            || occipital.visual_sweep(&hybrid_visual, dt, Some(&parietal.spatial_map), current_time, &config),
            || {
                if routing.deve_skipar(LobeId::Limbic) {
                    (0.0f32, 0.1f32) // gate baixo: emoção neutra
                } else {
                    let (e, a) = limbic.evaluate(&cochlea_alertado, 0.0, dt, current_time, &config);
                    // Escala output pelo gate: gate alto = influência total
                    (e * routing.limbic, a * routing.limbic)
                }
            },
        );

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
        let zero_n = vec![0.0f32; n_neurons];
        let (new_parietal_rates, recognized) = rayon::join(
            || {
                if routing.deve_skipar(LobeId::Parietal) {
                    prev_parietal_rates.clone() // economiza CPU, usa output anterior
                } else {
                    parietal.integrate(&vision_full, &zero_n, dt, current_time, &config)
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
                if let Ok(mut bs) = brain_state.try_lock() {
                    for c in &novos_chunks {
                        let eh_placeholder = c.simbolo.starts_with("chunk_")
                            || c.simbolo.contains('_');
                        if eh_placeholder {
                            // Chunk sem símbolo real: usa valência para reforçar palavras
                            // do contexto atual que já têm entrada no vocabulário.
                            // Isso cria o binding neuron-pattern → word de forma emergente.
                            let palavras_ativas: Vec<String> = bs.neural_context.iter()
                                .filter(|w| bs.palavra_valencias.contains_key(w.as_str()))
                                .cloned()
                                .collect();
                            for palavra in palavras_ativas.iter().take(3) {
                                let val_atual = bs.palavra_valencias.get(palavra).copied().unwrap_or(0.0);
                                let val_nova = val_atual * 0.90 + c.valence * 0.10;
                                bs.palavra_valencias.insert(palavra.clone(), val_nova);
                            }
                        } else {
                            // Símbolo real (mapeado futuramente a letra/sílaba/palavra):
                            // injeta diretamente no contexto e vocabulário
                            bs.neural_context.push_back(c.simbolo.clone());
                            let val_atual = bs.palavra_valencias.get(&c.simbolo).copied().unwrap_or(0.0);
                            let val_nova = val_atual * 0.85 + c.valence * 0.15;
                            bs.palavra_valencias.insert(c.simbolo.clone(), val_nova);
                        }
                    }
                    while bs.neural_context.len() > 20 {
                        bs.neural_context.pop_front();
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
            let (hippo_out, conexoes_hippo) = hippocampus.memorize_with_connections(
                &recognized, emotion, dt, current_time, &config,
            );
            // Fix 1: Consolida memória no grafo de linguagem.
            // O hipocampo sinaliza que AGORA é um momento emocionalmente saliente.
            // Os chunks ativos em neural_context são os "elementos da experiência".
            // Reforçamos as associações entre eles — conceitos co-ativos durante emoção
            // ficam mais fortemente ligados, exatamente como na memória episódica real.
            if !conexoes_hippo.is_empty() {
                if let Ok(mut bs) = brain_state.try_lock() {
                    let ctx: Vec<String> = bs.neural_context.iter().cloned().collect();
                    let peso_reforco = (emotion.abs() * 0.06).clamp(0.01, 0.15);
                    for i in 0..ctx.len().min(6) {
                        for j in (i + 1)..ctx.len().min(6) {
                            // Reforça a→b
                            let entry_ab = bs.grafo_associacoes.entry(ctx[i].clone()).or_insert_with(Vec::new);
                            if let Some(p) = entry_ab.iter_mut().find(|(w, _)| w == &ctx[j]) {
                                p.1 = (p.1 + peso_reforco).min(1.0);
                            } else if entry_ab.len() < 50 {
                                entry_ab.push((ctx[j].clone(), peso_reforco));
                            }
                            // Reforça b→a
                            let entry_ba = bs.grafo_associacoes.entry(ctx[j].clone()).or_insert_with(Vec::new);
                            if let Some(p) = entry_ba.iter_mut().find(|(w, _)| w == &ctx[i]) {
                                p.1 = (p.1 + peso_reforco).min(1.0);
                            } else if entry_ba.len() < 50 {
                                entry_ba.push((ctx[i].clone(), peso_reforco));
                            }
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
            if let Ok(mut bs) = brain_state.try_lock() {
                bs.ultimo_rpe = rl_rpe;
            }
            // DepthStack: atualiza atenção de abstração com base no RPE.
            // RPE positivo → camadas mais abstratas (D2) ganham mais atenção.
            // RPE negativo → sistema ancora no substrato bruto (D0).
            temporal.apply_rpe(rl_rpe);
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
            }
        }

        // G1. Propaga RPE (sinal dopaminérgico) para reforçar/enfraquecer chunks recentes
        {
            let rpe = neuro.dopamine - 0.5; // desvio da baseline dopaminérgica
            chunking.aplicar_rpe(rpe);
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
                    // WM ativo: move palavras já no contexto com conexões no grafo para o
                    // final da fila (aumenta saliência temporal sem duplicar).
                    let boost: Vec<String> = bs.neural_context.iter()
                        .filter(|w| bs.grafo_associacoes.contains_key(w.as_str()))
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

        // Roda Izhikevich+STDP nos neurônios conceituais aprendidos via WS
        {
            let mut swap = swap_manager.lock().await;
            swap.tick_semantico(dt, current_time * 1000.0);
        }

        // K. Telemetria
        if step % 500 == 0 {
            // Fase 1e: atualiza metacognição com estado neural atual
            let n_vocab = brain_state.try_lock()
                .map(|s| s.palavra_valencias.len()).unwrap_or(0);
            metacognitive.observe(arousal, emotion, n_vocab);

            let swap_guard = swap_manager.lock().await;
            let chunk_stats = chunking.stats();
            println!("🧪 [BIO] Sero: {:.2} | Dop: {:.2} | Cort: {:.2} | Emoção: {} | Onda: {}p",
                neuro.serotonin, neuro.dopamine, neuro.cortisol,
                plutchik.dominante(), n_passos_walk,
            );
            // ram_count() = neurônios em cache quente (RAM swap) — não é "0 disparando".
            // O contador real de spikes é derivado da atividade recente do temporal.
            let spikes_vivos = temporal.recognition_layer.neuronios.iter()
                .filter(|n| n.last_spike_ms > 0.0
                    && n.last_spike_ms >= (current_time * 1000.0) - 500.0)
                .count();
            println!("   🧬 Neurônios disparando: {} | RAM cache: {} | Hábitos: {} | Alerta: {:.2} | RAM: {:.1}GB",
                spikes_vivos, swap_guard.ram_count(), basal_ganglia.stats().num_habitos,
                brainstem.stats().alertness, sensor.lock().await.get_ram_usage_gb(),
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

        // Fix 3: Export automático do modelo de linguagem a cada 5000 ticks (~25s).
        // Garante que o grafo e vocabulário aprendidos em RAM não se percam ao fechar.
        // Sem isso, reiniciar a Selene perde todas as associações construídas na sessão.
        if step % 5000 == 0 && step > 0 {
            if let Ok(bs) = brain_state.try_lock() {
                let n_assoc: usize = bs.grafo_associacoes.values().map(|v| v.len()).sum();
                if n_assoc > 0 {
                    let json = crate::storage::exportar_linguagem(
                        &bs.palavra_valencias,
                        &bs.grafo_associacoes,
                        &bs.frases_padrao,
                    );
                    if let Err(e) = std::fs::write("selene_linguagem.json", json) {
                        log::warn!("[AUTO-EXPORT] Falha ao salvar linguagem: {}", e);
                    } else {
                        println!("💾 [AUTO-EXPORT] Linguagem salva: {} palavras, {} associações",
                            bs.palavra_valencias.len(), n_assoc);
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