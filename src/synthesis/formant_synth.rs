// src/synthesis/formant_synth.rs
// Síntese de formantes Klatt simplificada — gera PCM diretamente em Rust.
// Não depende de browser, TTS externo, ou serviço online.
//
// Pipeline por fonema:
//   voiced  → onda dente-de-serra @ F0 Hz
//   unvoiced → ruído branco pseudo-aleatório
//   → filtro biquad bandpass F1 (vocalização)
//   → filtro biquad bandpass F2 (timbre)
//   → filtro biquad bandpass F3 (brilho)
//   → envelope linear (fade 8ms início/fim)
//   → normalização e concatenação
//
// Jitter de F0 (~0.5%) e shimmer de amplitude (~1%) tornam a voz menos sintética.

#![allow(dead_code)]

use crate::encoding::phoneme::FormantParams;

pub const SAMPLE_RATE: u32 = 44_100;

// ── Filtro Biquad Bandpass ──────────────────────────────────────────────────

struct Biquad {
    b0: f32, b1: f32, b2: f32,
    a1: f32, a2: f32,
    x1: f32, x2: f32,
    y1: f32, y2: f32,
}

impl Biquad {
    /// Bandpass RBJ com frequência central `fc` e largura de banda `bw` (Hz).
    fn bandpass(fc: f32, bw: f32, sr: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * fc / sr;
        let alpha = w0.sin() * (2.0_f32.ln() / 2.0 * bw / sr * w0 / w0.sin()).sinh();
        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * w0.cos();
        let a2 = 1.0 - alpha;
        Self { b0: b0/a0, b1: b1/a0, b2: b2/a0, a1: a1/a0, a2: a2/a0,
               x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0 }
    }

    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
              - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1; self.x1 = x;
        self.y2 = self.y1; self.y1 = y;
        y
    }
}

// ── Gerador de excitação ─────────────────────────────────────────────────────

struct Excitation {
    phase: f32,     // fase da onda dente-de-serra (0..1)
    jitter: f32,    // variação suave de F0
    noise_seed: u64,
}

impl Excitation {
    fn new() -> Self { Self { phase: 0.0, jitter: 0.0, noise_seed: 0x9e3779b97f4a7c15 } }

    /// Sample voiced (dente-de-serra com jitter de F0).
    fn voiced(&mut self, f0: f32, dt: f32) -> f32 {
        // Micro-variação de F0 (jitter ~0.5%)
        self.jitter = self.jitter * 0.99 + (self.lcg() as f32 / u64::MAX as f32 - 0.5) * 0.005;
        let f0_j = f0 * (1.0 + self.jitter);
        self.phase = (self.phase + f0_j * dt) % 1.0;
        // Dente-de-serra: 2*phase - 1, com suavização de descida
        if self.phase < 0.9 {
            2.0 * self.phase - 1.0
        } else {
            // descida suave (anti-aliasing)
            1.0 - (self.phase - 0.9) / 0.1 * 2.0
        }
    }

    /// Sample unvoiced (ruído branco pseudo-aleatório).
    fn unvoiced(&mut self) -> f32 {
        (self.lcg() as f32 / u64::MAX as f32) * 2.0 - 1.0
    }

    fn lcg(&mut self) -> u64 {
        self.noise_seed = self.noise_seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.noise_seed
    }
}

// ── Síntese principal ────────────────────────────────────────────────────────

/// Sintetiza lista de FormantParams em PCM f32 @ SAMPLE_RATE.
/// Cada fonema produz `dur_ms` ms de áudio.
pub fn sintetizar(formants: &[FormantParams]) -> Vec<f32> {
    let sr = SAMPLE_RATE as f32;
    let dt = 1.0 / sr;
    let fade_n = (0.008 * sr) as usize; // 8ms fade

    let mut out: Vec<f32> = Vec::new();
    let mut exc = Excitation::new();
    let mut shimmer_seed: u64 = 0xdeadbeefcafe;

    for fp in formants {
        if fp.dur_ms <= 0.0 { continue; }
        let n_samples = ((fp.dur_ms / 1000.0) * sr) as usize;
        if n_samples == 0 { continue; }

        let f0   = fp.f0.max(50.0);
        let f1   = fp.f1.max(100.0);
        let f2   = fp.f2.max(f1 + 50.0);
        let f3   = fp.f3.max(f2 + 50.0);
        let gain = fp.energy.clamp(0.0, 1.0);

        // Bandas biológicas PT-BR (Hz):
        // F1: jaw opening (300-800), bw=80
        // F2: tongue front-back (700-2300), bw=120
        // F3: lip rounding (2000-3000), bw=160
        let mut bp1 = Biquad::bandpass(f1, 80.0, sr);
        let mut bp2 = Biquad::bandpass(f2, 120.0, sr);
        let mut bp3 = Biquad::bandpass(f3, 160.0, sr);

        let mut frame: Vec<f32> = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            // Excitação
            let x = if fp.voiced {
                exc.voiced(f0, dt)
            } else {
                exc.unvoiced() * 0.3
            };

            // Filtros em série: F1 → F2 → F3
            let y = bp3.process(bp2.process(bp1.process(x)));

            // Shimmer de amplitude (~1% variação por amostra)
            shimmer_seed = shimmer_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let shimmer = 1.0 + (shimmer_seed as f32 / u64::MAX as f32 - 0.5) * 0.01;

            // Envelope linear (fade in/out 8ms)
            let env = if i < fade_n {
                i as f32 / fade_n as f32
            } else if i >= n_samples.saturating_sub(fade_n) {
                (n_samples - i) as f32 / fade_n as f32
            } else {
                1.0
            };

            frame.push(y * gain * shimmer * env);
        }

        out.extend(frame);
    }

    // Normaliza para -1..1 evitando clip
    let peak = out.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.01 {
        let scale = 0.85 / peak;
        for s in &mut out { *s *= scale; }
    }

    out
}

/// Sintetiza uma frase a partir de frames FFT brutos gravados (voz neural).
/// Usa vocoder simples: bandas FFT como envelope espectral sobre excitação.
/// Se `frames` for vazio, retorna Vec vazio (caller usa fallback Klatt).
pub fn sintetizar_neural(frames: &[[f32; 32]], f0_hz: f32, dur_ms_total: f32) -> Vec<f32> {
    if frames.is_empty() { return Vec::new(); }

    let sr = SAMPLE_RATE as f32;
    let dt = 1.0 / sr;
    let n_total = ((dur_ms_total / 1000.0) * sr) as usize;
    if n_total == 0 { return Vec::new(); }

    let fade_n = (0.008 * sr) as usize;
    let mut exc = Excitation::new();
    let mut out = Vec::with_capacity(n_total);

    // Interpola entre frames ao longo do tempo
    let samples_per_frame = n_total.max(1) / frames.len().max(1);

    for i in 0..n_total {
        let frame_idx = (i / samples_per_frame.max(1)).min(frames.len() - 1);
        let bands = &frames[frame_idx];

        // Excitação periódica (voz humana = quasi-periódica)
        let x = exc.voiced(f0_hz, dt);

        // Cada banda FFT modula um filtro ressonante na sua frequência central
        // Frequências centrais das 32 bandas (escala log 80-8000 Hz)
        let freq_min = 80.0f32;
        let freq_max = 8000.0f32;
        let mut y = 0.0f32;
        for (b, &amp) in bands.iter().enumerate() {
            if amp < 0.01 { continue; }
            let t = b as f32 / 31.0;
            let fc = freq_min * (freq_max / freq_min).powf(t);
            // Ressonador simples: cos modulation (IIR-free, lighter)
            let phase = 2.0 * std::f32::consts::PI * fc * i as f32 / sr;
            y += x * amp * phase.cos();
        }

        // Normalização por número de bandas ativas
        let n_ativas = bands.iter().filter(|&&v| v > 0.01).count().max(1);
        y /= n_ativas as f32;

        // Envelope
        let env = if i < fade_n {
            i as f32 / fade_n as f32
        } else if i >= n_total.saturating_sub(fade_n) {
            (n_total - i) as f32 / fade_n as f32
        } else { 1.0 };

        out.push(y * env * 0.7);
    }

    // Normaliza
    let peak = out.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.01 {
        let scale = 0.85 / peak;
        for s in &mut out { *s *= scale; }
    }

    out
}
