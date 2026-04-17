// src/encoding/fft_encoder.rs
// Pipeline áudio → primitivas de onda para Selene Brain 2.0.
//
// Entrada:  &[(f32, f32)] — pares (frequência_hz, amplitude) do FFT
//           Compatível com spectrum-analyzer crate e qualquer outro backend.
//
// Saída:    PrimitivaOnda — parâmetros físicos extraídos sem texto
//
// Pipeline:
//   1. Bandas cocleares (33 bandas log-espaçadas, escala Bark aproximada)
//   2. Detecção de formantes F1/F2/F3 via peak-picking suavizado
//   3. Estimativa de F0 por Harmonic Product Spectrum (HPS simplificado)
//   4. Classificação de onset (Plosiva/Fricativa/Nasal/Vogal...)
//   5. Cálculo de VOT e delta de formantes
//   6. Construção de PrimitivaOnda
//
// Melhoria: delta features (taxa de variação de formantes) são incluídas.
// Melhoria: prosódia — F0 contour exportado separadamente.
// Melhoria: estado vocal (voiced/unvoiced) por energy ratio.
#![allow(dead_code)]

use std::time::{SystemTime, UNIX_EPOCH};
use crate::storage::ondas::{PrimitivaOnda, TipoOnset};

// ─── Bandas cocleares ─────────────────────────────────────────────────────────

/// 33 bandas cocleares log-espaçadas de 20 Hz a 20 kHz.
/// O número 33 corresponde aos 33 fonemas do português.
pub const N_BANDAS: usize = 33;
pub const FREQ_MIN: f32   = 20.0;
pub const FREQ_MAX: f32   = 20_000.0;

/// Frequências centrais das 33 bandas cocleares (Hz).
/// Calculadas por interpolação logarítmica: f[i] = FREQ_MIN * r^i
/// onde r = (FREQ_MAX/FREQ_MIN)^(1/(N_BANDAS-1))
pub fn bandas_cocleares() -> [f32; N_BANDAS] {
    let mut bandas = [0f32; N_BANDAS];
    let ratio = (FREQ_MAX / FREQ_MIN).powf(1.0 / (N_BANDAS as f32 - 1.0));
    for i in 0..N_BANDAS {
        bandas[i] = FREQ_MIN * ratio.powi(i as i32);
    }
    bandas
}

/// Energia de cada banda coclear extraída do espectro FFT.
/// Integra a amplitude de todos os bins dentro de cada banda.
pub fn extrair_bandas_cocleares(fft: &[(f32, f32)]) -> [f32; N_BANDAS] {
    let centros = bandas_cocleares();
    let mut bandas = [0f32; N_BANDAS];
    let mut contagens = [0u32; N_BANDAS];

    for &(freq, amp) in fft {
        // Encontra banda mais próxima pelo log de frequência
        let log_f   = freq.max(1.0).log10();
        let log_min = FREQ_MIN.log10();
        let log_max = FREQ_MAX.log10();
        let idx = ((log_f - log_min) / (log_max - log_min) * (N_BANDAS - 1) as f32)
            .round()
            .clamp(0.0, (N_BANDAS - 1) as f32) as usize;
        bandas[idx] += amp;
        contagens[idx] += 1;
    }

    // Normaliza cada banda pela contagem de bins e pelo máximo global
    for i in 0..N_BANDAS {
        if contagens[i] > 0 {
            bandas[i] /= contagens[i] as f32;
        }
    }
    let max = bandas.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        for b in &mut bandas { *b /= max; }
    }
    bandas
}

// ─── Detecção de formantes ────────────────────────────────────────────────────

/// Parâmetros de formante extraídos de um frame de áudio.
#[derive(Debug, Clone, Default)]
pub struct Formantes {
    /// Frequência fundamental em Hz. None = frame não-voiced.
    pub f0:    Option<f32>,
    /// 1º formante (abertura vocal): 200–900 Hz.
    pub f1:    Option<f32>,
    /// 2º formante (posição da língua): 700–2500 Hz.
    pub f2:    Option<f32>,
    /// 3º formante (arredondamento labial): 2000–3500 Hz.
    pub f3:    Option<f32>,
    /// Taxa de mudança de F1 em Hz/ms (delta feature — detecta onset).
    pub delta_f1: f32,
    /// Taxa de mudança de F2 em Hz/ms (delta feature — locus de consoantes).
    pub delta_f2: f32,
    /// True se o frame tem energia voiced suficiente.
    pub voiced: bool,
    /// Energia total normalizada [0.0, 1.0].
    pub energia: f32,
    /// Razão de energia acima de 3kHz (índice de fricação).
    pub hf_ratio: f32,
}

/// Suaviza o espectro com janela de ±`raio` Hz para remover harmônicos
/// e deixar os picos de formante mais visíveis.
fn suavizar_espectro(fft: &[(f32, f32)], raio_hz: f32) -> Vec<(f32, f32)> {
    fft.iter().map(|&(f_centro, _)| {
        let soma: f32 = fft.iter()
            .filter(|&&(f, _)| (f - f_centro).abs() <= raio_hz)
            .map(|&(_, a)| a)
            .sum();
        let cnt = fft.iter()
            .filter(|&&(f, _)| (f - f_centro).abs() <= raio_hz)
            .count()
            .max(1);
        (f_centro, soma / cnt as f32)
    }).collect()
}

/// Encontra o pico de maior amplitude dentro de [f_min, f_max] Hz.
fn pico_em(fft: &[(f32, f32)], f_min: f32, f_max: f32) -> Option<f32> {
    fft.iter()
        .filter(|&&(f, _)| f >= f_min && f <= f_max)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|&(f, _)| f)
}

/// Energia relativa acima de `threshold_hz`.
fn energia_acima(fft: &[(f32, f32)], threshold_hz: f32) -> f32 {
    let total: f32 = fft.iter().map(|&(_, a)| a).sum::<f32>().max(1e-9);
    let alta:  f32 = fft.iter().filter(|&&(f, _)| f >= threshold_hz).map(|&(_, a)| a).sum();
    alta / total
}

/// Energia total normalizada pelo número de bins.
fn energia_total(fft: &[(f32, f32)]) -> f32 {
    if fft.is_empty() { return 0.0; }
    let soma: f32 = fft.iter().map(|&(_, a)| a).sum();
    (soma / fft.len() as f32).min(1.0)
}

/// Estimativa de F0 por Harmonic Product Spectrum (HPS) simplificado.
/// Aplica downsampling do espectro 2× e 3× e multiplica — o F0 aparece como pico.
fn estimar_f0(fft: &[(f32, f32)]) -> Option<f32> {
    if fft.len() < 6 { return None; }

    // Filtra para região de voz humana (70–400 Hz)
    let region: Vec<(f32, f32)> = fft.iter()
        .filter(|&&(f, _)| f >= 70.0 && f <= 400.0)
        .cloned()
        .collect();
    if region.is_empty() { return None; }

    // HPS: para cada candidato F0, verifica se existe harmônico em 2×F0 e 3×F0
    let best = region.iter().max_by(|a, b| {
        let score_a = hps_score(fft, a.0);
        let score_b = hps_score(fft, b.0);
        score_a.partial_cmp(&score_b).unwrap()
    });

    best.map(|&(f, _)| f)
}

fn hps_score(fft: &[(f32, f32)], f0: f32) -> f32 {
    let amp = |target: f32| -> f32 {
        fft.iter()
            .filter(|&&(f, _)| (f - target).abs() < target * 0.05) // 5% tolerância
            .map(|&(_, a)| a)
            .sum()
    };
    amp(f0) * amp(f0 * 2.0) * amp(f0 * 3.0)
}

/// Extrai todos os parâmetros de formante de um frame FFT.
/// `prev_f1` e `prev_f2`: valores do frame anterior para cálculo de delta.
/// `duracao_ms`: duração do frame em ms (para normalizar deltas Hz/ms).
pub fn extrair_formantes(
    fft:       &[(f32, f32)],
    prev_f1:   Option<f32>,
    prev_f2:   Option<f32>,
    duracao_ms: f32,
) -> Formantes {
    if fft.is_empty() {
        return Formantes::default();
    }

    let energia = energia_total(fft);
    let hf_ratio = energia_acima(fft, 3000.0);
    let voiced = energia > 0.05 && hf_ratio < 0.6;

    // Suaviza com janela 40Hz para remover harmônicos individuais
    let suave = suavizar_espectro(fft, 40.0);

    let f0 = if voiced { estimar_f0(fft) } else { None };
    let f1 = pico_em(&suave, 200.0, 900.0);
    let f2 = pico_em(&suave, 700.0, 2500.0);
    let f3 = pico_em(&suave, 2000.0, 3500.0);

    // Delta features — taxa de variação por ms
    let dur = duracao_ms.max(1.0);
    let delta_f1 = match (f1, prev_f1) {
        (Some(n), Some(p)) => (n - p) / dur,
        _ => 0.0,
    };
    let delta_f2 = match (f2, prev_f2) {
        (Some(n), Some(p)) => (n - p) / dur,
        _ => 0.0,
    };

    Formantes { f0, f1, f2, f3, delta_f1, delta_f2, voiced, energia, hf_ratio }
}

// ─── Classificação de onset ───────────────────────────────────────────────────

/// Contexto de silêncio necessário para detectar VOT de oclusivas.
#[derive(Debug, Clone, Default)]
pub struct ContextoOnset {
    /// Energia média dos últimos N frames (para detectar silêncio pré-burst).
    pub energia_media_anterior: f32,
    /// F0 estava presente no frame anterior.
    pub voiced_anterior: bool,
    /// VOT acumulado em ms (cresce enquanto burst sem vozeamento).
    pub vot_acumulado_ms: f32,
}

/// Classifica o tipo de onset com base nos parâmetros espectrais e contexto.
pub fn classificar_onset(f: &Formantes, ctx: &ContextoOnset) -> (TipoOnset, f32) {
    // Silêncio
    if f.energia < 0.02 {
        return (TipoOnset::Silencio, 0.0);
    }

    // Fricativa: alta energia acima de 3kHz, sem F0
    if f.hf_ratio > 0.55 && !f.voiced {
        return (TipoOnset::Fricativa, 0.0);
    }

    // Nasal: energia concentrada em baixas frequências (~250Hz), anti-formante visível
    // Detectado por: energia baixa em F2, F0 presente, hf_ratio muito baixo
    if f.voiced && f.hf_ratio < 0.1 && f.f2.map(|v| v < 1000.0).unwrap_or(false) {
        return (TipoOnset::Nasal, 0.0);
    }

    // Oclusiva: burst de energia após silêncio
    let burst = ctx.energia_media_anterior < 0.05 && f.energia > 0.2;

    if burst {
        if f.voiced {
            // Sonora: vozeamento imediato (VOT negativo ou zero)
            return (TipoOnset::OclusivaSonora, ctx.vot_acumulado_ms);
        } else {
            // Surda: silêncio pós-burst antes do vozeamento (VOT positivo)
            return (TipoOnset::OclusivaSurda, ctx.vot_acumulado_ms);
        }
    }

    // Lateral: energia em F3 específica com F2 moderado
    if f.voiced && f.f3.map(|v| v > 2500.0).unwrap_or(false) && f.hf_ratio < 0.3 {
        return (TipoOnset::Lateral, 0.0);
    }

    // Aproximante: transição suave (delta_f2 alto mas sem burst)
    if f.delta_f2.abs() > 5.0 && !burst {
        return (TipoOnset::Aproximante, 0.0);
    }

    // Default: vogal pura
    (TipoOnset::Vogal, 0.0)
}

// ─── Pipeline principal ───────────────────────────────────────────────────────

/// Estado incremental do encoder entre frames.
/// Mantém o histórico mínimo necessário para cálculo de deltas e VOT.
#[derive(Debug, Clone, Default)]
pub struct EstadoEncoder {
    pub prev_f1:     Option<f32>,
    pub prev_f2:     Option<f32>,
    pub ctx_onset:   ContextoOnset,
    pub ultima_primitiva: Option<SpikeHashRef>,
}

/// Referência ao hash da última primitiva (para bigramas).
pub type SpikeHashRef = String;

/// Converte dados FFT em PrimitivaOnda.
///
/// Parâmetros:
/// - `fft`: pares (freq_hz, amplitude) do spectrum-analyzer ou equivalente
/// - `estado`: estado incremental (atualizado in-place)
/// - `duracao_ms`: duração do frame de áudio
/// - `timestamp`: timestamp Unix em segundos
///
/// Melhoria implementada: delta features embutidas na primitiva,
/// classificação de onset com VOT, prosódia via F0.
pub fn fft_para_primitiva(
    fft:        &[(f32, f32)],
    estado:     &mut EstadoEncoder,
    duracao_ms: u32,
    timestamp:  f64,
) -> PrimitivaOnda {
    let f = extrair_formantes(fft, estado.prev_f1, estado.prev_f2, duracao_ms as f32);
    let (onset, vot) = classificar_onset(&f, &estado.ctx_onset);

    // Atualiza contexto para próximo frame
    estado.prev_f1 = f.f1;
    estado.prev_f2 = f.f2;
    estado.ctx_onset.voiced_anterior = f.voiced;
    estado.ctx_onset.energia_media_anterior = f.energia;
    if onset == TipoOnset::OclusivaSurda && !f.voiced {
        estado.ctx_onset.vot_acumulado_ms += duracao_ms as f32;
    } else {
        estado.ctx_onset.vot_acumulado_ms = 0.0;
    }

    PrimitivaOnda::sonora(
        f.f0, f.f1, f.f2, f.f3,
        f.delta_f1, f.delta_f2,
        vot, onset,
        f.energia, f.hf_ratio,
        duracao_ms, timestamp,
    )
}

// ─── Contorno de prosódia ─────────────────────────────────────────────────────

/// Contorno de F0 ao longo de um enunciado.
/// Captura a "melodia" da fala: afirmação, interrogação, ênfase.
#[derive(Debug, Clone)]
pub struct ContornoProsodia {
    /// Sequência de (timestamp, F0_hz). None = frame não-voiced.
    pub contorno: Vec<(f64, Option<f32>)>,
    pub duracao_total_ms: u32,
}

impl ContornoProsodia {
    pub fn novo() -> Self {
        Self { contorno: Vec::new(), duracao_total_ms: 0 }
    }

    pub fn adicionar_frame(&mut self, timestamp: f64, f0: Option<f32>, duracao_ms: u32) {
        self.contorno.push((timestamp, f0));
        self.duracao_total_ms += duracao_ms;
    }

    /// Classifica o padrão prosódico por tendência de F0 final.
    /// Subida final = interrogação. Descida = afirmação. Plano = neutro.
    pub fn tipo_prosodico(&self) -> ProsodiaDetectada {
        let voiced: Vec<f32> = self.contorno.iter()
            .filter_map(|(_, f)| *f)
            .collect();
        if voiced.len() < 4 { return ProsodiaDetectada::Neutro; }

        let n = voiced.len();
        let media_inicial = voiced[..n/3].iter().sum::<f32>() / (n/3) as f32;
        let media_final   = voiced[2*n/3..].iter().sum::<f32>() / (n - 2*n/3) as f32;
        let delta = media_final - media_inicial;

        if delta > 20.0      { ProsodiaDetectada::Interrogacao }
        else if delta < -20.0 { ProsodiaDetectada::Afirmacao }
        else                  { ProsodiaDetectada::Neutro }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProsodiaDetectada {
    Afirmacao,
    Interrogacao,
    Neutro,
}

// ─── Testes ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fft_vogal_a() -> Vec<(f32, f32)> {
        // Simula /a/: F1≈800Hz, F2≈1200Hz, voiced, baixo hf_ratio
        let mut v: Vec<(f32, f32)> = (0..200).map(|i| {
            let f = i as f32 * 100.0 + 50.0;
            let amp = if (f - 800.0).abs() < 150.0 { 0.9 }
                     else if (f - 1200.0).abs() < 150.0 { 0.7 }
                     else if f < 400.0 { 0.3 }
                     else { 0.05 };
            (f, amp)
        }).collect();
        v
    }

    #[test]
    fn extrai_f1_vogal_a() {
        let fft = fft_vogal_a();
        let f = extrair_formantes(&fft, None, None, 25.0);
        assert!(f.f1.is_some());
        let f1 = f.f1.unwrap();
        assert!(f1 > 600.0 && f1 < 1000.0, "F1={} fora de 600-1000Hz", f1);
    }

    #[test]
    fn mesmo_fft_mesmo_hash() {
        let fft = fft_vogal_a();
        let mut e1 = EstadoEncoder::default();
        let mut e2 = EstadoEncoder::default();
        let p1 = fft_para_primitiva(&fft, &mut e1, 25, 0.0);
        let p2 = fft_para_primitiva(&fft, &mut e2, 25, 0.0);
        assert_eq!(p1.hash, p2.hash);
    }

    #[test]
    fn bandas_cocleares_primeira_e_ultima() {
        let b = bandas_cocleares();
        assert!((b[0] - FREQ_MIN).abs() < 1.0);
        assert!((b[N_BANDAS-1] - FREQ_MAX).abs() < 100.0);
    }
}
