// src/sensors/camera.rs
//
// Sistema de visão dual-câmera da Selene.
//
// ARQUITETURA:
//   Câmera 0 (primária)  — visão principal. Sempre ativa quando vídeo ligado.
//   Câmera 1 (estéreo)   — opcional. Quando presente, calcula mapa de disparidade
//                          para estimar profundidade (visão estéreo como olhos humanos).
//
// SHARED FRAME BUFFER:
//   Arc<Mutex<Option<RgbFrame>>> compartilhado com:
//     - O cérebro da Selene (processamento neural)
//     - O vision_stream (broadcast WebSocket para o viewer externo)
//   O frame bruto RGB é preservado, não só a luminância neural.
//
// DISPARIDADE:
//   Quando câmera 1 está ativa, o DisparityEngine calcula diferença pixel-a-pixel
//   entre frames L e R (após alinhamento simples por correlação).
//   Saída: mapa de profundidade normalizado [0..1] para o parietal.

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::time::Duration;

// ── Frame RGB compartilhado ───────────────────────────────────────────────────

/// Frame RGB bruto capturado da câmera.
/// Preservado no SharedFrameBuffer para o viewer e para o processamento neural.
#[derive(Debug, Clone)]
pub struct RgbFrame {
    pub pixels:    Vec<u8>,   // R,G,B,R,G,B...
    pub largura:   u32,
    pub altura:    u32,
    pub camera_id: u8,        // 0 = primária, 1 = estéreo
    pub timestamp_ms: u64,
}

impl RgbFrame {
    pub fn vazio(camera_id: u8) -> Self {
        Self { pixels: Vec::new(), largura: 0, altura: 0, camera_id, timestamp_ms: 0 }
    }

    pub fn n_pixels(&self) -> usize { (self.largura * self.altura) as usize }

    /// Luminância perceptual ITU-R BT.601 por pixel → [0..1]
    pub fn para_luminancia(&self) -> Vec<f32> {
        self.pixels.chunks(3)
            .map(|rgb| {
                let r = rgb[0] as f32 / 255.0;
                let g = rgb[1] as f32 / 255.0;
                let b = rgb[2] as f32 / 255.0;
                0.299 * r + 0.587 * g + 0.114 * b
            })
            .collect()
    }
}

/// Buffer compartilhado entre câmera, cérebro e vision_stream.
/// Contém o frame mais recente de cada câmera.
pub type SharedFrameBuffer = Arc<Mutex<FrameBuffer>>;

#[derive(Debug, Default)]
pub struct FrameBuffer {
    pub frame_primario: Option<RgbFrame>,
    pub frame_estereo:  Option<RgbFrame>,
    /// Mapa de disparidade (profundidade) quando câmera 1 está ativa.
    pub mapa_disparidade: Option<Vec<f32>>,
}

pub fn novo_frame_buffer() -> SharedFrameBuffer {
    Arc::new(Mutex::new(FrameBuffer::default()))
}

// ── Mapa de Disparidade (visão estéreo) ──────────────────────────────────────

/// Calcula mapa de disparidade simples entre dois frames de mesma resolução.
/// Resultado: Vec<f32> normalizado [0..1] onde 1.0 = objeto próximo.
/// Usa subtração absoluta de luminância como aproximação rápida.
pub fn calcular_disparidade(esquerda: &RgbFrame, direita: &RgbFrame) -> Vec<f32> {
    if esquerda.pixels.is_empty() || direita.pixels.is_empty() {
        return Vec::new();
    }
    let lum_e = esquerda.para_luminancia();
    let lum_d = direita.para_luminancia();
    let n = lum_e.len().min(lum_d.len());

    let disp: Vec<f32> = (0..n)
        .map(|i| (lum_e[i] - lum_d[i]).abs())
        .collect();

    // Normaliza para [0..1]
    let max = disp.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        disp.iter().map(|&d| d / max).collect()
    } else {
        disp
    }
}

// ── Transdutor Visual (câmera única) ─────────────────────────────────────────

pub struct VisualTransducer {
    resolution: usize,
    camera_id:  u8,
    ativo:      Arc<AtomicBool>,
    /// Buffer compartilhado — atualizado a cada frame capturado.
    frame_buf:  SharedFrameBuffer,
}

impl VisualTransducer {
    pub fn new(
        resolution: usize,
        ativo:      Arc<AtomicBool>,
        frame_buf:  SharedFrameBuffer,
    ) -> Self {
        Self { resolution, camera_id: 0, ativo, frame_buf }
    }

    pub fn com_camera_id(mut self, id: u8) -> Self {
        self.camera_id = id;
        self
    }

    /// Inicia o loop de captura. Deve ser chamada dentro de `thread::spawn`.
    pub fn run(&mut self, tx: Sender<Vec<f32>>) {
        while !self.ativo.load(Ordering::Relaxed) {
            if tx.send(vec![0.0f32; self.resolution]).is_err() { return; }
            std::thread::sleep(Duration::from_millis(100));
        }
        println!("[CAMERA {}] Sensor ativado.", self.camera_id);

        let format = RequestedFormat::new::<RgbFormat>(
            RequestedFormatType::AbsoluteHighestFrameRate
        );

        let mut cam = match Camera::new(CameraIndex::Index(self.camera_id as u32), format) {
            Ok(c) => c,
            Err(e) => {
                println!("[CAMERA {}] Não encontrada: {e}. Modo fosfeno.", self.camera_id);
                return self.run_placeholder(tx);
            }
        };

        if let Err(e) = cam.open_stream() {
            println!("[CAMERA {}] Stream falhou: {e}. Modo fosfeno.", self.camera_id);
            return self.run_placeholder(tx);
        }

        let fmt = cam.camera_format();
        println!("[CAMERA {}] Stream ativo — {:?}", self.camera_id, fmt);

        loop {
            if !self.ativo.load(Ordering::Relaxed) {
                if tx.send(vec![0.0f32; self.resolution]).is_err() { break; }
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }

            let frame = match cam.frame() {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("[CAMERA {}] Erro: {e}", self.camera_id);
                    continue;
                }
            };

            let buf = frame.buffer();
            let (largura, altura) = (fmt.width(), fmt.height());
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            let rgb_frame = RgbFrame {
                pixels:       buf.to_vec(),
                largura:      largura as u32,
                altura:       altura as u32,
                camera_id:    self.camera_id,
                timestamp_ms: ts,
            };

            // Atualiza shared buffer
            if let Ok(mut fb) = self.frame_buf.lock() {
                match self.camera_id {
                    0 => fb.frame_primario = Some(rgb_frame.clone()),
                    1 => {
                        // Se temos os dois frames, calcula disparidade
                        let disp = fb.frame_primario.as_ref()
                            .map(|prim| calcular_disparidade(prim, &rgb_frame));
                        fb.frame_estereo   = Some(rgb_frame.clone());
                        fb.mapa_disparidade = disp;
                    }
                    _ => {}
                }
            }

            // Sinal neural: luminância reamostrada
            let neural_signal = self.frame_para_neuronio(&rgb_frame);
            if tx.send(neural_signal).is_err() { break; }
        }
    }

    fn frame_para_neuronio(&self, frame: &RgbFrame) -> Vec<f32> {
        let lum = frame.para_luminancia();
        self.reamostrar(&lum, self.resolution)
    }

    fn reamostrar(&self, src: &[f32], target: usize) -> Vec<f32> {
        if src.is_empty() { return vec![0.0; target]; }
        let ratio = src.len() as f32 / target as f32;
        (0..target)
            .map(|i| {
                let idx = (i as f32 * ratio) as usize;
                src[idx.min(src.len() - 1)]
            })
            .collect()
    }

    fn run_placeholder(&self, tx: Sender<Vec<f32>>) {
        println!("[CAMERA {}] Modo fosfeno ativo.", self.camera_id);
        loop {
            if self.ativo.load(Ordering::Relaxed) {
                let fosfeno: Vec<f32> = (0..self.resolution)
                    .map(|_| rand::random::<f32>() * 0.05)
                    .collect();
                if tx.send(fosfeno).is_err() { break; }
                std::thread::sleep(Duration::from_millis(33)); // ~30 Hz
            } else {
                if tx.send(vec![0.0f32; self.resolution]).is_err() { break; }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}

// ── Sistema Dual-Câmera ───────────────────────────────────────────────────────

/// Gerencia câmera primária + câmera estéreo opcional.
/// A câmera estéreo só é iniciada se `stereo_disponivel == true`.
pub struct DualCameraSystem {
    pub frame_buf:          SharedFrameBuffer,
    pub stereo_disponivel:  bool,
}

impl DualCameraSystem {
    pub fn novo(stereo_disponivel: bool) -> Self {
        Self {
            frame_buf:         novo_frame_buffer(),
            stereo_disponivel,
        }
    }

    /// Spawna threads de captura para câmera(s) ativa(s).
    /// Retorna (tx_primario, tx_estereo_opcional).
    pub fn iniciar(
        &self,
        resolution:       usize,
        ativo_primario:   Arc<AtomicBool>,
        ativo_estereo:    Arc<AtomicBool>,
    ) -> (
        std::sync::mpsc::Receiver<Vec<f32>>,
        Option<std::sync::mpsc::Receiver<Vec<f32>>>,
    ) {
        // Câmera primária
        let (tx0, rx0) = std::sync::mpsc::channel();
        let buf0 = Arc::clone(&self.frame_buf);
        std::thread::spawn(move || {
            let mut cam = VisualTransducer::new(resolution, ativo_primario, buf0)
                .com_camera_id(0);
            cam.run(tx0);
        });

        // Câmera estéreo (opcional)
        let rx1 = if self.stereo_disponivel {
            let (tx1, rx1) = std::sync::mpsc::channel();
            let buf1 = Arc::clone(&self.frame_buf);
            std::thread::spawn(move || {
                let mut cam = VisualTransducer::new(resolution, ativo_estereo, buf1)
                    .com_camera_id(1);
                cam.run(tx1);
            });
            Some(rx1)
        } else {
            None
        };

        (rx0, rx1)
    }

    /// Retorna clone do frame primário mais recente (para vision_stream).
    pub fn frame_atual(&self) -> Option<RgbFrame> {
        self.frame_buf.lock().ok()?.frame_primario.clone()
    }

    /// Retorna mapa de disparidade quando estéreo ativo.
    pub fn disparidade_atual(&self) -> Option<Vec<f32>> {
        self.frame_buf.lock().ok()?.mapa_disparidade.clone()
    }
}
