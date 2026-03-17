// src/sensors/sensor_control.rs
//
// Controle runtime de sensores — ativo/inativo sem reiniciar threads.
//
// Design:
//   - Cada sensor tem um AtomicBool compartilhado (Arc<AtomicBool>)
//   - Sensor desativado → envia zeros (silêncio/escuridão) ao loop neural
//   - Sensor ativado   → captura dados reais do hardware
//   - Toggle via WebSocket sem parar/reiniciar o processo
//
// Por padrão AMBOS os sensores iniciam DESATIVADOS.
// O usuário ativa explicitamente via interface neural.

#![allow(dead_code)]

use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

/// Estado compartilhado de todos os sensores de entrada.
#[derive(Clone)]
pub struct SensorFlags {
    pub audio_ativo: Arc<AtomicBool>,
    pub video_ativo: Arc<AtomicBool>,
}

impl SensorFlags {
    /// Cria com ambos os sensores DESATIVADOS por padrão.
    pub fn new_desativados() -> Self {
        Self {
            audio_ativo: Arc::new(AtomicBool::new(false)),
            video_ativo: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn audio_ativo(&self) -> bool {
        self.audio_ativo.load(Ordering::Relaxed)
    }

    pub fn video_ativo(&self) -> bool {
        self.video_ativo.load(Ordering::Relaxed)
    }

    pub fn set_audio(&self, ativo: bool) {
        self.audio_ativo.store(ativo, Ordering::Relaxed);
        log::info!("[SENSOR] Áudio: {}", if ativo { "ATIVADO" } else { "DESATIVADO" });
    }

    pub fn set_video(&self, ativo: bool) {
        self.video_ativo.store(ativo, Ordering::Relaxed);
        log::info!("[SENSOR] Vídeo: {}", if ativo { "ATIVADO" } else { "DESATIVADO" });
    }
}
