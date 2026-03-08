// =============================================================================
// src/sensors/camera.rs
// =============================================================================
//
// VISÃO DA SELENE — Transdutor Visual
//
// Analogia biológica:
//   Retina → Nervo óptico → Tálamo (CGLd) → Córtex Visual Primário (V1)
//
// O que este módulo faz:
//   1. Abre a webcam via `nokhwa` (pure-Rust, sem OpenCV)
//   2. Captura frames RGB em tempo real
//   3. Converte pixels para luminância normalizada (0.0 a 1.0)
//   4. Reamosta o frame inteiro para N neurônios
//   5. Envia o vetor neural pelo canal `tx` para o OccipitalLobe
//
// Por que luminância e não RGB completo?
//   O V1 biológico é dominado por detectores de borda e contraste em escala de cinza.
//   Cor é processada em camadas superiores (V4). Para a Selene v2, luminância é
//   suficiente para detectar movimento, bordas e padrões gerais.
//   Cor pode ser adicionada no OccipitalLobe quando necessário.
//
// Dependência necessária no Cargo.toml:
//   nokhwa = { version = "0.10", features = ["input-native"] }
//
// Por que nokhwa e não OpenCV?
//   OpenCV exige CMake + LLVM + configuração manual de PATH no Windows.
//   nokhwa é pure-Rust, compila com `cargo build` sem dependências externas.
//   Suporte: DirectShow (Windows), V4L2 (Linux), AVFoundation (macOS).
//
// =============================================================================

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use nokhwa::{
    Camera,
    CameraIndex,
    pixel_format::RgbFormat,
    utils::{RequestedFormat, RequestedFormatType},
};
use std::sync::mpsc::Sender;
use std::time::Duration;

// -----------------------------------------------------------------------------
// Estrutura principal
// -----------------------------------------------------------------------------

/// Transdutor visual — encapsula câmera e converte frames para sinais neurais.
///
/// # Campos
/// - `resolution`: número de neurônios no OccipitalLobe que receberão o sinal.
///   Deve ser igual a `brain.occipital.neurons.len()`.
pub struct VisualTransducer {
    /// Número de neurônios-destino no OccipitalLobe.
    /// O frame inteiro da câmera será reamostrado para este tamanho.
    resolution: usize,
}

impl VisualTransducer {
    // -------------------------------------------------------------------------
    // Construção
    // -------------------------------------------------------------------------

    /// Cria um novo transdutor visual.
    ///
    /// # Parâmetros
    /// - `resolution`: tamanho do vetor de saída (= número de neurônios do Occipital).
    ///   Valor típico: 512 ou 1024 neurônios.
    pub fn new(resolution: usize) -> Self {
        Self { resolution }
    }

    // -------------------------------------------------------------------------
    // Loop principal de captura
    // -------------------------------------------------------------------------

    /// Inicia o loop de captura de vídeo e envia frames pelo canal `tx`.
    ///
    /// Esta função **bloqueia a thread** — deve ser chamada dentro de
    /// `std::thread::spawn` no main.rs:
    ///
    /// ```rust
    /// let (vis_tx, vis_rx) = mpsc::channel();
    /// std::thread::spawn(move || {
    ///     VisualTransducer::new(config.n_neurons).run(vis_tx);
    /// });
    /// ```
    ///
    /// Se nenhuma câmera for encontrada, cai automaticamente para `run_placeholder`,
    /// que envia ruído visual leve (simula "olhos fechados mas não cegos").
    pub fn run(&mut self, tx: Sender<Vec<f32>>) {
        // Tenta abrir câmera no índice 0 (primeira câmera disponível).
        // RequestedFormatType::AbsoluteHighestFrameRate = nokhwa escolhe a maior
        // resolução com o maior FPS suportados pelo hardware.
        let format = RequestedFormat::new::<RgbFormat>(
            RequestedFormatType::AbsoluteHighestFrameRate
        );

        let mut cam = match Camera::new(CameraIndex::Index(0), format) {
            Ok(c)  => c,
            Err(e) => {
                // Câmera não encontrada — pode ser laptop sem webcam, VM, etc.
                // Em vez de panicar, usamos placeholder para que o resto do cérebro
                // continue funcionando normalmente (Selene "de olhos fechados").
                println!("[CAMERA] Câmera não encontrada: {e}");
                println!("[CAMERA] Usando modo placeholder (ruído baixo).");
                return self.run_placeholder(tx);
            }
        };

        // Abre o stream de captura (começa a receber frames da câmera).
        cam.open_stream().expect("[CAMERA] Falha ao abrir stream de vídeo");

        // Log informativo: mostra resolução e FPS que foram negociados com o hardware.
        println!(
            "[CAMERA] Stream ativo — formato: {:?}",
            cam.camera_format()
        );

        // ── Loop de captura ──────────────────────────────────────────────────
        loop {
            // Captura o próximo frame disponível.
            // `frame()` retorna dados brutos RGB (3 bytes por pixel).
            let frame = match cam.frame() {
                Ok(f)  => f,
                Err(e) => {
                    // Erros de frame são comuns (câmera ocupada, lag de USB, etc.)
                    // Simplesmente tenta de novo sem derrubar o sistema.
                    eprintln!("[CAMERA] Erro ao capturar frame: {e}");
                    continue;
                }
            };

            // Converte o frame bruto para vetor de luminância neural.
            let neural_signal = self.frame_para_neuronio(frame.buffer());

            // Envia pelo canal para o OccipitalLobe.
            // Se o receptor foi dropado (sistema encerrando), simplesmente para.
            if tx.send(neural_signal).is_err() {
                println!("[CAMERA] Canal fechado. Encerrando captura.");
                break;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Conversão de frame para sinal neural
    // -------------------------------------------------------------------------

    /// Converte buffer RGB bruto da câmera para vetor de ativações neurais.
    ///
    /// # Processo
    /// 1. Agrupa pixels em triplas RGB
    /// 2. Calcula luminância perceptual de cada pixel
    /// 3. Reamosta para `self.resolution` pontos
    ///
    /// # Por que luminância perceptual (ITU-R BT.601)?
    ///   O olho humano é 3x mais sensível ao verde do que ao vermelho,
    ///   e quase insensível ao azul em comparação. Os coeficientes
    ///   0.299R + 0.587G + 0.114B refletem essa sensibilidade.
    ///   Usar média simples (R+G+B)/3 daria percepção errada de brilho.
    fn frame_para_neuronio(&self, buffer: &[u8]) -> Vec<f32> {
        // Extrai luminância de cada pixel (tripla RGB → 1 valor 0..1)
        let pixels: Vec<f32> = buffer
            .chunks(3)  // cada tripla = [R, G, B]
            .map(|rgb| {
                // Normaliza cada canal de 0..255 para 0.0..1.0
                let r = rgb[0] as f32 / 255.0;
                let g = rgb[1] as f32 / 255.0;
                let b = rgb[2] as f32 / 255.0;

                // Fórmula de luminância perceptual ITU-R BT.601
                // Verde domina (0.587) porque o olho tem mais cones M (verde)
                0.299 * r + 0.587 * g + 0.114 * b
            })
            .collect();

        // Reamosta o vetor de pixels para o número de neurônios disponíveis.
        // Ex: frame 640×480 = 307.200 pixels → 512 neurônios
        self.reamostrar(&pixels, self.resolution)
    }

    // -------------------------------------------------------------------------
    // Reamostragem (downsampling)
    // -------------------------------------------------------------------------

    /// Reamosta um vetor de tamanho arbitrário para `target` pontos.
    ///
    /// Usa interpolação de vizinho mais próximo (nearest-neighbor).
    /// Para visão neural, isso é suficiente — não precisamos de bicúbico
    /// porque o OccipitalLobe já faz integração espacial internamente.
    ///
    /// # Parâmetros
    /// - `src`: vetor original (pode ter centenas de milhares de elementos)
    /// - `target`: tamanho desejado (= número de neurônios)
    ///
    /// # Retorno
    /// Vetor de `target` floats em 0.0..1.0
    fn reamostrar(&self, src: &[f32], target: usize) -> Vec<f32> {
        if src.is_empty() {
            // Frame vazio — retorna vetor nulo (sem estímulo visual)
            return vec![0.0; target];
        }

        // Razão de amostragem: quantos pixels da câmera por neurônio
        let ratio = src.len() as f32 / target as f32;

        (0..target)
            .map(|i| {
                // Para cada neurônio i, calcula qual pixel da câmera corresponde
                let pixel_idx = (i as f32 * ratio) as usize;
                src[pixel_idx.min(src.len() - 1)]
            })
            .collect()
    }

    // -------------------------------------------------------------------------
    // Modo placeholder (sem câmera)
    // -------------------------------------------------------------------------

    /// Modo de operação sem câmera física.
    ///
    /// Em vez de enviar zeros (silêncio total), envia ruído muito baixo (0.0..0.1).
    /// Isso simula o "fosfeno" biológico — os neurônios visuais nunca ficam
    /// completamente silenciosos, mesmo em total escuridão.
    ///
    /// Frequência: 30 Hz (frame a cada ~33ms), similar a uma câmera real.
    fn run_placeholder(&self, tx: Sender<Vec<f32>>) {
        println!("[CAMERA] Modo fosfeno ativo (ruído visual baixo, 30Hz).");

        loop {
            // Ruído muito baixo simula atividade espontânea do V1 no escuro.
            // Amplitude 0.0..0.1 — abaixo do threshold de disparo na maioria das configs.
            let fosfeno: Vec<f32> = (0..self.resolution)
                .map(|_| rand::random::<f32>() * 0.1)
                .collect();

            if tx.send(fosfeno).is_err() {
                break; // Sistema encerrando
            }

            // ~30 FPS para não saturar a CPU
            std::thread::sleep(Duration::from_millis(33));
        }
    }
}

// =============================================================================
// NOTAS PARA IMPLEMENTAÇÃO FUTURA
// =============================================================================
//
// 1. DETECÇÃO DE MOVIMENTO (V1 → MT/V5)
//    Guarde o frame anterior e calcule `delta = |frame_atual - frame_anterior|`.
//    Spike de movimento = delta.mean() > threshold.
//    Envie pelo canal separado para o parietal (atenção visual espacial).
//
// 2. DETECÇÃO DE FACES (FaceNet / tflite)
//    Quando uma face aparecer no frame, gere spike no sistema límbico diretamente.
//    Rosto sorrindo → dopamina sobe. Rosto agressivo → amígdala dispara.
//
// 3. CÂMERAS MÚLTIPLAS
//    Para visão binocular (profundidade), instancie dois VisualTransducer
//    com CameraIndex::Index(0) e CameraIndex::Index(1).
//    A disparidade entre eles → sinal de profundidade para o parietal.
//
// 4. CÂMERA VIRTUAL PARA TESTES
//    Use CameraIndex::String("v4l2loopback") no Linux com um vídeo sintético
//    para testar o pipeline sem hardware físico.
//
// =============================================================================
