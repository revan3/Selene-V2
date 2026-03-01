// src/sensors/camera.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
// use opencv::{prelude::*, videoio, imgproc, core};  // Comentado até instalar OpenCV nativo
use std::sync::mpsc::Sender;
use std::time::Duration;
use crate::brain_zones::RegionType;
pub struct VisualTransducer {
    // cam: videoio::VideoCapture,  // Comentado
    resolution: usize,
}

impl VisualTransducer {
    pub fn new(resolution: usize) -> Self {
        // let cam = videoio::VideoCapture::new(0, videoio::CAP_ANY).expect("Não foi possível acessar a Webcam");
        Self { resolution }  // cam comentado temporariamente
    }

    pub fn run(&mut self, tx: Sender<Vec<f32>>) {
        // Placeholder até câmera funcionar
        println!("[CAMERA] Placeholder - câmera desabilitada temporariamente (opencv não encontrado)");
        loop {
            // Simula envio de frame vazio
            let signal = vec![0.5; self.resolution];
            let _ = tx.send(signal);
            std::thread::sleep(Duration::from_millis(500));  // Evita loop infinito rápido
        }
    }
}