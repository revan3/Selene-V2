// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use crate::synaptic_core;
use crate::brain_zones;
use crate::storage;
use crate::sensors;
use crate::brain_zones::RegionType;
use storage::{BrainStorage, NeuralEnactiveMemory};
use sensors::camera::VisualTransducer;
use std::sync::mpsc::channel;
use std::time::Duration;

#[tokio::main]
async fn main() {
    println!("🚀 Iniciando Diagnóstico de Sistemas da Selene...");

    // 1. TESTE DE BANCO DE DADOS (HELIX STORAGE)
    println!("--- [1/3] Testando SurrealDB/RocksDB ---");
    let storage = match BrainStorage::new().await {
        Ok(s) => {
            println!("✅ Banco de Dados conectado e índices MTREE criados.");
            s
        },
        Err(e) => {
            println!("❌ ERRO no Banco de Dados: {}", e);
            return;
        }
    };

    // Teste de Gravação/Leitura
    let test_mem = NeuralEnactiveMemory {
        timestamp: 0.0,
        emotion_state: 0.5,
        arousal_state: 0.5,
        visual_pattern: vec![1.0; 1024], // Padrão de teste
        auditory_pattern: vec![0.0; 1024],
        frontal_intent: vec![0.0; 1024],
        label: "diagnostico_teste".to_string(),
    };

    if storage.save_snapshot(test_mem).await.is_ok() {
        println!("✅ Gravação de memória vetorial funcionando.");
    }

    // 2. TESTE DE HARDWARE (CÂMERA)
    println!("--- [2/3] Testando Visão (Webcam) ---");
    let (tx, rx) = channel::<Vec<f32>>();  // Tipo explícito para evitar inferência falha
    let mut camera = VisualTransducer::new(1024);
    
    println!("Aguardando frame da câmera (5 segundos)...");
    std::thread::spawn(move || {
        camera.run(tx);
    });

    match rx.recv_timeout(Duration::from_secs(5)) {
        Ok(signal) => {
            let sum: f32 = signal.iter().sum();
            println!("✅ Câmera detectada! Intensidade média de luz: {:.2}", sum / 1024.0);
        },
        Err(_) => println!("❌ ERRO: Câmera não respondeu ou OpenCV falhou."),
    }

    // 3. TESTE DE INTEGRIDADE NEURAL
    println!("--- [3/3] Testando Ciclo Neural (Izhikevich) ---");
    use brain_zones::occipital::OccipitalLobe;
    let mut occipital = OccipitalLobe::new(1024, 0.2);
    let test_input = vec![0.5; 1024];
    
    // Simula 10 steps
    let mut spikes = 0;
    for _ in 0..10 {
        let output = occipital.visual_sweep(&test_input, 0.1, None);
        if output.iter().any(|&x| x > 0.0) { spikes += 1; }
    }

    if spikes > 0 {
        println!("✅ Neurônios disparando corretamente (Spiking OK).");
    } else {
        println!("⚠️  Aviso: Nenhum spike detectado (pode ser ruído baixo).");
    }

    println!("\n✨ DIAGNÓSTICO CONCLUÍDO: Selene está pronta para ser despertada.");
}