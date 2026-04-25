// src/synthesis/cpal_output.rs
// Output de áudio nativo via cpal — Selene fala diretamente pelo speaker do servidor.
//
// Arquitetura (thread-safe no Windows/WASAPI):
//   AudioOutput: struct Send+Sync que armazena apenas um SyncSender<Vec<f32>>.
//   Um thread std dedicado cria e owna a cpal::Stream (!Send no Windows).
//   O thread recebe chunks de PCM pelo canal e os empurra para o buffer da stream.
//   O callback cpal drena o buffer a cada bloco de hardware (~5ms).
//
// Porta mobile aberta:
//   Mobile recebe voz via WS (voz_params JSON). AudioOutput só ativo localmente.

#![allow(dead_code)]

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

pub struct AudioOutput {
    /// Canal para enviar chunks PCM ao thread de áudio.
    pcm_tx: std::sync::mpsc::SyncSender<Vec<f32>>,
    /// Taxa de amostragem negociada com o device.
    pub sample_rate: u32,
}

impl AudioOutput {
    /// Tenta abrir o device de saída padrão.
    /// Retorna None se não há device disponível (headless, container, etc.).
    pub fn try_new() -> Option<Self> {
        // Canal para o thread de áudio reportar a taxa de amostragem escolhida.
        let (sr_tx, sr_rx) = std::sync::mpsc::channel::<u32>();
        // Canal para enviar PCM ao thread (capacidade 16 — ~1.5s de buffer de envio).
        let (pcm_tx, pcm_rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(16);

        std::thread::Builder::new()
            .name("cpal-audio-out".into())
            .spawn(move || {
                // Toda a criação cpal acontece aqui dentro: Stream nunca precisa ser Send.
                let host = cpal::default_host();
                let device = match host.default_output_device() {
                    Some(d) => d,
                    None    => { let _ = sr_tx.send(0); return; }
                };
                let config = match device.default_output_config() {
                    Ok(c)  => c,
                    Err(_) => { let _ = sr_tx.send(0); return; }
                };
                let sr = config.sample_rate().0;

                let buffer: Arc<Mutex<VecDeque<f32>>> =
                    Arc::new(Mutex::new(VecDeque::with_capacity(sr as usize * 2)));
                let buf_cb = buffer.clone();
                let channels = config.channels() as usize;

                let stream_result = match config.sample_format() {
                    cpal::SampleFormat::F32 => build_output_stream::<f32>(&device, &config.into(), buf_cb, channels),
                    cpal::SampleFormat::I16 => build_output_stream::<i16>(&device, &config.into(), buf_cb, channels),
                    cpal::SampleFormat::U16 => build_output_stream::<u16>(&device, &config.into(), buf_cb, channels),
                    _ => { let _ = sr_tx.send(0); return; }
                };

                let stream = match stream_result {
                    Ok(s)  => s,
                    Err(_) => { let _ = sr_tx.send(0); return; }
                };

                if stream.play().is_err() { let _ = sr_tx.send(0); return; }

                log::info!("[VOICE] AudioOutput nativo: device='{}' sr={}Hz",
                    device.name().unwrap_or_default(), sr);

                let _ = sr_tx.send(sr); // confirma sample_rate

                // Loop de recepção: empurra chunks PCM para o buffer da stream.
                // Termina quando o canal pcm_tx (no AudioOutput) é dropado.
                while let Ok(pcm) = pcm_rx.recv() {
                    if let Ok(mut buf) = buffer.lock() {
                        let cap = sr as usize * 3; // 3s cap
                        if buf.len() + pcm.len() > cap {
                            let ex = (buf.len() + pcm.len()).saturating_sub(cap);
                            buf.drain(..ex);
                        }
                        buf.extend(pcm);
                    }
                }
                // _stream dropped aqui — para o áudio graciosamente
            })
            .ok()?;

        // Aguarda confirmação da sample_rate (timeout 2s)
        let sr = sr_rx.recv_timeout(std::time::Duration::from_secs(2)).ok()?;
        if sr == 0 { return None; }

        Some(Self { pcm_tx, sample_rate: sr })
    }

    /// Empurra amostras PCM f32 para reprodução.
    /// Reamostra linearmente se src_rate != sample_rate do device.
    pub fn enqueue(&self, mut pcm: Vec<f32>, src_rate: u32) {
        if src_rate != self.sample_rate && src_rate > 0 {
            let ratio  = self.sample_rate as f64 / src_rate as f64;
            let n_out  = (pcm.len() as f64 * ratio) as usize;
            let mut rs = Vec::with_capacity(n_out);
            for i in 0..n_out {
                let sf  = i as f64 / ratio;
                let si  = sf as usize;
                let frc = (sf - si as f64) as f32;
                let s0  = pcm.get(si    ).copied().unwrap_or(0.0);
                let s1  = pcm.get(si + 1).copied().unwrap_or(s0);
                rs.push(s0 + frc * (s1 - s0));
            }
            pcm = rs;
        }
        // Se o canal estiver cheio, descarta silenciosamente (prefere-se latência baixa).
        let _ = self.pcm_tx.try_send(pcm);
    }

    /// Envia sinal de silêncio (chunk vazio) para limpar o buffer do thread.
    pub fn silencio(&self) {
        let _ = self.pcm_tx.try_send(vec![0.0f32; self.sample_rate as usize / 100]);
    }
}

fn build_output_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    buf: Arc<Mutex<VecDeque<f32>>>,
    channels: usize,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>,
{
    device.build_output_stream(
        config,
        move |data: &mut [T], _| {
            let mut guard = match buf.lock() {
                Ok(g)  => g,
                Err(_) => return,
            };
            for frame in data.chunks_mut(channels) {
                let sample = guard.pop_front().unwrap_or(0.0);
                let s = T::from_sample(sample);
                for ch in frame.iter_mut() { *ch = s; }
            }
        },
        |err| log::warn!("[VOICE] cpal output error: {}", err),
        None,
    )
}
