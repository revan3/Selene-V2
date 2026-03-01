// src/sensors/audio.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use spectrum_analyzer::{samples_fft_to_spectrum, FrequencyLimit};
use std::sync::mpsc::Sender;
use crate::brain_zones::RegionType;

// src/sensors/audio.rs - substitua a função start_listening por:

pub fn start_listening(n_neurons: usize, tx: Sender<Vec<f32>>) {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use spectrum_analyzer::{samples_fft_to_spectrum, FrequencyLimit};
    
    let host = cpal::default_host();
    let device = host.default_input_device().expect("Sem microfone");
    let config = device.default_input_config().unwrap();

    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _| {
            // Versão simplificada - apenas calcula energia média
            let energy: f32 = data.iter().map(|&x| x * x).sum::<f32>() / data.len() as f32;
            
            let mut neural_input = vec![0.0; n_neurons];
            // Distribui a energia pelos neurônios
            for i in 0..n_neurons.min(10) {
                neural_input[i] = energy * 10.0;
            }
            let _ = tx.send(neural_input);
        },
        |err| eprintln!("Erro áudio: {}", err),
        None
    ).unwrap();

    stream.play().unwrap();
    std::thread::park();
}