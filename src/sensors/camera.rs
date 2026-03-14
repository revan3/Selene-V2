// src/sensors/camera.rs
//
// CORREÇÕES (E0268 + E0308):
//
// Versão anterior tinha `continue` e mismatched types em match arms FORA de loop.
// A câmera agora usa nokhwa (pure-Rust, sem OpenCV).
// `continue` está corretamente dentro do `loop { ... }` de captura.
//
// Dependência no Cargo.toml:
//   nokhwa = { version = "0.10", features = ["input-native"] }

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},  // CameraIndex está em utils, não na raiz
};
use std::sync::mpsc::Sender;
use std::time::Duration;
use crate::brain_zones::RegionType;

pub struct VisualTransducer {
    resolution: usize,
}

impl VisualTransducer {
    pub fn new(resolution: usize) -> Self {
        Self { resolution }
    }

    /// Inicia o loop de captura. Deve ser chamada dentro de `thread::spawn`.
    /// Se não houver câmera, usa modo placeholder (fosfeno — ruído visual leve).
    pub fn run(&mut self, tx: Sender<Vec<f32>>) {
        let format = RequestedFormat::new::<RgbFormat>(
            RequestedFormatType::AbsoluteHighestFrameRate
        );

        // CORREÇÃO E0268 + E0308:
        // O `continue` e os tipos errados estavam em match arms numa função,
        // não dentro de um loop. Agora o match está DENTRO do loop correto.
        let mut cam = match Camera::new(CameraIndex::Index(0), format) {
            Ok(c) => c,
            Err(e) => {
                println!("[CAMERA] Câmera não encontrada: {e}. Usando placeholder.");
                // Não há `continue` aqui — simplesmente retorna o placeholder
                return self.run_placeholder(tx);
            }
        };

        // Abre o stream (sem `continue` fora de loop)
        if let Err(e) = cam.open_stream() {
            println!("[CAMERA] Falha ao abrir stream: {e}. Usando placeholder.");
            return self.run_placeholder(tx);
        }

        println!("[CAMERA] Stream ativo — formato: {:?}", cam.camera_format());

        // ── Loop de captura — `continue` é válido AQUI dentro do loop ──────
        loop {
            // CORREÇÃO E0268: `continue` agora está dentro do loop, não em match arm solto
            let frame = match cam.frame() {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("[CAMERA] Erro ao capturar frame: {e}");
                    continue;  // ✅ válido: estamos dentro do loop
                }
            };

            // CORREÇÃO E0308: frame.buffer() retorna &[u8] — tipo correto
            let neural_signal = self.frame_para_neuronio(frame.buffer());

            if tx.send(neural_signal).is_err() {
                println!("[CAMERA] Canal fechado. Encerrando.");
                break;
            }
        }
    }

    /// Converte buffer RGB para vetor de luminância neural (0.0..1.0).
    fn frame_para_neuronio(&self, buffer: &[u8]) -> Vec<f32> {
        let pixels: Vec<f32> = buffer
            .chunks(3)
            .map(|rgb| {
                let r = rgb[0] as f32 / 255.0;
                let g = rgb[1] as f32 / 255.0;
                let b = rgb[2] as f32 / 255.0;
                // Luminância perceptual ITU-R BT.601
                0.299 * r + 0.587 * g + 0.114 * b
            })
            .collect();

        self.reamostrar(&pixels, self.resolution)
    }

    /// Reamostra vetor para `target` pontos (nearest-neighbor).
    fn reamostrar(&self, src: &[f32], target: usize) -> Vec<f32> {
        if src.is_empty() {
            return vec![0.0; target];
        }
        let ratio = src.len() as f32 / target as f32;
        (0..target)
            .map(|i| {
                let idx = (i as f32 * ratio) as usize;
                src[idx.min(src.len() - 1)]
            })
            .collect()
    }

    /// Modo sem câmera — envia ruído baixo simulando fosfeno (~30 Hz).
    fn run_placeholder(&self, tx: Sender<Vec<f32>>) {
        println!("[CAMERA] Modo fosfeno ativo (sem câmera, 30Hz).");
        loop {
            let fosfeno: Vec<f32> = (0..self.resolution)
                .map(|_| rand::random::<f32>() * 0.05)
                .collect();
            if tx.send(fosfeno).is_err() {
                break;
            }
            std::thread::sleep(Duration::from_millis(33));
        }
    }
}