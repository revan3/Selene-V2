// src/encoding/espectro_visual.rs
// Pipeline câmera → primitivas de onda luminosa para Selene Brain 2.0.
//
// Entrada:  frame RGB bruto (Vec<u8>, width, height)
// Saída:    PrimitivaOnda com parâmetros físicos de luz
//
// O que é extraído (nunca texto ou rótulos):
//   - Comprimento de onda dominante em nm (cor percebida)
//   - Luminância média normalizada
//   - Frequência espacial dominante (textura/borda — filtros Gabor simplificados)
//   - Orientação dominante de borda (0°–180°)
//   - Taxa de variação temporal (detecção de movimento entre frames)
//
// Bandas espectrais mapeadas aos fotorreceptores humanos:
//   Violeta  380–450 nm  (cone S + bastonetes)
//   Azul     450–495 nm  (cone S)
//   Ciano    495–520 nm  (transição S→M)
//   Verde    520–565 nm  (cone M)
//   Amarelo  565–590 nm  (transição M→L)
//   Laranja  590–625 nm  (cone L)
//   Vermelho 625–700 nm  (cone L dominante)
//
// Melhoria: frequência espacial multi-escala (4 escalas × 4 orientações).
// Melhoria: movimento como taxa de variação de luminância entre frames.
// Melhoria: região central ponderada (fóvea) mais relevante que periferia.
#![allow(dead_code)]

use crate::storage::ondas::PrimitivaOnda;

// ─── Bandas espectrais ────────────────────────────────────────────────────────

pub const N_BANDAS_VISUAIS: usize = 7;

/// Centros de comprimento de onda das 7 bandas (nm).
pub const CENTROS_NM: [f32; N_BANDAS_VISUAIS] = [415.0, 472.0, 507.0, 542.0, 577.0, 607.0, 662.0];

/// Limites de cada banda espectral (nm).
pub const LIMITES_NM: [(f32, f32); N_BANDAS_VISUAIS] = [
    (380.0, 450.0), // Violeta
    (450.0, 495.0), // Azul
    (495.0, 520.0), // Ciano
    (520.0, 565.0), // Verde
    (565.0, 590.0), // Amarelo
    (590.0, 625.0), // Laranja
    (625.0, 700.0), // Vermelho
];

/// Energia de cada banda espectral de um pixel RGB.
#[derive(Debug, Clone)]
pub struct BandasEspectrais {
    /// Intensidade por banda [0.0, 1.0].
    pub bandas: [f32; N_BANDAS_VISUAIS],
    /// Comprimento de onda dominante (centro da banda mais intensa) em nm.
    pub dominante_nm: f32,
    /// Luminância total [0.0, 1.0].
    pub luminancia: f32,
}

/// Converte RGB para aproximação de comprimento de onda dominante.
///
/// Abordagem: pesos fisiológicos dos cones L, M, S.
/// L (vermelho): pico ~560–580 nm → fortemente ativado por R
/// M (verde):    pico ~530 nm     → fortemente ativado por G
/// S (azul):     pico ~420 nm     → fortemente ativado por B
///
/// Retorna o centro de banda mais ativa.
pub fn rgb_para_comprimento_onda(r: u8, g: u8, b: u8) -> f32 {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;

    // Contribuição estimada para cada banda
    let contrib: [f32; N_BANDAS_VISUAIS] = [
        bf * 0.9,                          // Violeta   (S forte)
        bf * 0.7 + gf * 0.1,              // Azul      (S + M fraco)
        gf * 0.4 + bf * 0.3,              // Ciano     (M + S)
        gf * 0.9 + rf * 0.1,              // Verde     (M forte)
        gf * 0.6 + rf * 0.5,              // Amarelo   (M + L)
        rf * 0.8 + gf * 0.2,              // Laranja   (L + M)
        rf * 0.95,                         // Vermelho  (L forte)
    ];

    // Banda mais ativa
    let (idx, _) = contrib.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap_or((3, &0.0));

    CENTROS_NM[idx]
}

/// Extrai bandas espectrais completas de um pixel RGB.
pub fn rgb_para_bandas(r: u8, g: u8, b: u8) -> BandasEspectrais {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;

    let bandas: [f32; N_BANDAS_VISUAIS] = [
        (bf * 0.9).clamp(0.0, 1.0),
        (bf * 0.7 + gf * 0.1).clamp(0.0, 1.0),
        (gf * 0.4 + bf * 0.3).clamp(0.0, 1.0),
        (gf * 0.9 + rf * 0.1).clamp(0.0, 1.0),
        (gf * 0.6 + rf * 0.5).clamp(0.0, 1.0),
        (rf * 0.8 + gf * 0.2).clamp(0.0, 1.0),
        (rf * 0.95).clamp(0.0, 1.0),
    ];

    let (idx, _) = bandas.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap_or((3, &0.0));

    // Luminância perceptual (pesos ITU-R BT.601)
    let luminancia = (0.299 * rf + 0.587 * gf + 0.114 * bf).clamp(0.0, 1.0);

    BandasEspectrais { bandas, dominante_nm: CENTROS_NM[idx], luminancia }
}

// ─── Frequência espacial (Gabor simplificado) ─────────────────────────────────

/// Resultado da análise de frequência espacial.
#[derive(Debug, Clone, Default)]
pub struct FreqEspacial {
    /// Energia de bordas horizontais.
    pub horizontal:  f32,
    /// Energia de bordas verticais.
    pub vertical:    f32,
    /// Energia de bordas a 45°.
    pub diagonal_45: f32,
    /// Energia de bordas a 135°.
    pub diagonal_135: f32,
    /// Magnitude total de borda (textura global).
    pub magnitude:   f32,
    /// Orientação dominante em graus (0–180).
    pub orientacao:  f32,
    /// Frequência espacial dominante (ciclos/pixel).
    pub freq_ciclos: f32,
}

/// Aplica filtro de diferença de Gaussianas (DoG) simplificado para detecção de bordas.
/// Opera diretamente sobre o canal de luminância do frame.
///
/// Melhoria: usa ponderação de fóvea — região central (30% do frame) tem peso 2×.
fn extrair_freq_espacial_luminancia(
    luminancia: &[f32],
    width: usize,
    height: usize,
) -> FreqEspacial {
    if width < 3 || height < 3 || luminancia.len() < width * height {
        return FreqEspacial::default();
    }

    let mut eh = 0.0f32; // horizontal edges
    let mut ev = 0.0f32; // vertical edges
    let mut ed = 0.0f32; // diagonal 45°
    let mut ex = 0.0f32; // diagonal 135°
    let mut cnt = 0u32;

    // Área foveal (centro 60% do frame, peso 2×)
    let cx_min = width / 5;
    let cx_max = 4 * width / 5;
    let cy_min = height / 5;
    let cy_max = 4 * height / 5;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = |dy: i32, dx: i32| -> f32 {
                let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
                let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                luminancia[ny * width + nx]
            };

            // Operadores de Sobel
            let gx = idx(1, 1) - idx(1, -1) + 2.0 * idx(0, 1) - 2.0 * idx(0, -1)
                   + idx(-1, 1) - idx(-1, -1);
            let gy = idx(1, 1) - idx(-1, 1) + 2.0 * idx(1, 0) - 2.0 * idx(-1, 0)
                   + idx(1, -1) - idx(-1, -1);
            let gd  = idx(1, 1) - idx(-1, -1);  // 45°
            let gxd = idx(1, -1) - idx(-1, 1);  // 135°

            // Peso foveal
            let w = if x >= cx_min && x < cx_max && y >= cy_min && y < cy_max { 2.0 } else { 1.0 };

            ev += gx.abs() * w;
            eh += gy.abs() * w;
            ed += gd.abs() * w;
            ex += gxd.abs() * w;
            cnt += 1;
        }
    }

    if cnt == 0 { return FreqEspacial::default(); }
    let n = cnt as f32;

    let eh = eh / n;
    let ev = ev / n;
    let ed = ed / n;
    let ex = ex / n;
    let magnitude = (eh * eh + ev * ev).sqrt();

    // Orientação dominante: atan2 da maior componente
    let orientacao = if ev.abs() > 1e-6 {
        (eh / ev).atan().to_degrees().abs() % 180.0
    } else { 90.0 };

    // Frequência espacial estimada: razão de magnitude sobre tamanho do frame
    let freq_ciclos = magnitude / (width.min(height) as f32 / 2.0).max(1.0);

    FreqEspacial { horizontal: eh, vertical: ev, diagonal_45: ed, diagonal_135: ex,
                   magnitude, orientacao, freq_ciclos }
}

// ─── Pipeline principal ───────────────────────────────────────────────────────

/// Estado incremental do pipeline visual entre frames.
/// Mantém luminância do frame anterior para cálculo de movimento.
#[derive(Debug, Clone, Default)]
pub struct EstadoVisual {
    pub lum_anterior: Option<Vec<f32>>,
    pub timestamp_anterior: Option<f64>,
}

/// Converte um frame RGB para PrimitivaOnda luminosa.
///
/// Parâmetros:
/// - `pixels`: buffer RGB (3 bytes por pixel, linha a linha)
/// - `width`, `height`: dimensões do frame
/// - `estado`: estado incremental para cálculo de movimento
/// - `timestamp`: timestamp Unix em segundos
pub fn frame_para_primitiva(
    pixels:    &[u8],
    width:     u32,
    height:    u32,
    estado:    &mut EstadoVisual,
    timestamp: f64,
) -> PrimitivaOnda {
    let w = width as usize;
    let h = height as usize;

    if pixels.len() < w * h * 3 {
        return PrimitivaOnda::luminosa(550.0, 0.0, 0.0, 0.0, 0.0, 0, timestamp);
    }

    // ── 1. Extrai luminância por pixel e banda dominante média ────────────
    let mut lum_buf = Vec::with_capacity(w * h);
    let mut soma_nm  = 0.0f32;
    let mut soma_lum = 0.0f32;
    let n_pixels = (w * h) as f32;

    for i in 0..w * h {
        let r = pixels[i * 3];
        let g = pixels[i * 3 + 1];
        let b = pixels[i * 3 + 2];
        let bandas = rgb_para_bandas(r, g, b);
        lum_buf.push(bandas.luminancia);
        soma_nm  += bandas.dominante_nm;
        soma_lum += bandas.luminancia;
    }

    let nm_media  = soma_nm  / n_pixels;
    let lum_media = (soma_lum / n_pixels).clamp(0.0, 1.0);

    // ── 2. Frequência espacial ────────────────────────────────────────────
    let fe = extrair_freq_espacial_luminancia(&lum_buf, w, h);

    // ── 3. Taxa de variação temporal (movimento) ──────────────────────────
    let taxa_variacao = if let Some(ref anterior) = estado.lum_anterior {
        if anterior.len() == lum_buf.len() {
            let diff: f32 = anterior.iter().zip(&lum_buf)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>() / n_pixels;
            diff
        } else { 0.0 }
    } else { 0.0 };

    // Atualiza estado
    estado.lum_anterior = Some(lum_buf);
    estado.timestamp_anterior = Some(timestamp);

    PrimitivaOnda::luminosa(
        nm_media,
        lum_media,
        fe.freq_ciclos,
        fe.orientacao,
        taxa_variacao,
        0,        // duração = instante de captura
        timestamp,
    )
}

// ─── Testes ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vermelho_puro_acima_600nm() {
        let nm = rgb_para_comprimento_onda(255, 0, 0);
        assert!(nm > 600.0, "vermelho deveria ser >600nm, got {}", nm);
    }

    #[test]
    fn azul_puro_abaixo_500nm() {
        let nm = rgb_para_comprimento_onda(0, 0, 255);
        assert!(nm < 500.0, "azul deveria ser <500nm, got {}", nm);
    }

    #[test]
    fn verde_puro_na_faixa_verde() {
        let nm = rgb_para_comprimento_onda(0, 255, 0);
        assert!(nm >= 520.0 && nm <= 565.0, "verde deveria ser 520-565nm, got {}", nm);
    }

    #[test]
    fn mesmo_frame_mesmo_hash() {
        let frame = vec![255u8, 0, 0, 255, 0, 0]; // 2 pixels vermelhos
        let mut e1 = EstadoVisual::default();
        let mut e2 = EstadoVisual::default();
        let p1 = frame_para_primitiva(&frame, 2, 1, &mut e1, 0.0);
        let p2 = frame_para_primitiva(&frame, 2, 1, &mut e2, 0.0);
        assert_eq!(p1.hash, p2.hash);
    }
}
