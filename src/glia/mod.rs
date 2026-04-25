// src/glia/mod.rs
// Módulo glial — sinapse tripartite com dinâmica de cálcio lento.
// Uma instância por RegionType (9 astrocytes). Modula STDP via glio_factor.

use crate::brain_zones::RegionType;

const NUM_REGIONS: usize = 9;

/// Astrocyte com dinâmica de cálcio lenta (τ ≈ 200ms).
/// Recebe atividade regional (firing rate normalizado [0,1]) e libera
/// gliotransmissores que modulam LTP/LTD: glio_factor ∈ [-1, +1].
#[derive(Clone, Debug)]
pub struct Astrocyte {
    pub region: RegionType,
    /// Concentração intracelular de Ca²⁺ livre [0, 1].
    pub calcium: f32,
    /// Limiar de disparo para liberação de gliotransmissor.
    threshold: f32,
    /// τ de subida do cálcio (s).
    tau_rise: f32,
    /// τ de decaimento do cálcio (s).
    tau_decay: f32,
    /// Último valor de gliotransmissor liberado (D-serina / glutamato).
    pub gliotransmitter: f32,
}

impl Astrocyte {
    pub fn new(region: RegionType) -> Self {
        Self {
            region,
            calcium: 0.0,
            threshold: 0.4,
            tau_rise: 0.15,   // 150ms subida
            tau_decay: 0.80,  // 800ms decaimento
            gliotransmitter: 0.0,
        }
    }

    /// `activity` = firing rate normalizado da região [0, 1].
    /// `dt` em segundos.
    pub fn update(&mut self, activity: f32, dt: f32) {
        // Dinâmica de cálcio: I_ext → Ca²⁺ sobe; sem estímulo → decai
        let i_ext = activity.clamp(0.0, 1.0);
        let tau = if i_ext > self.calcium { self.tau_rise } else { self.tau_decay };
        self.calcium += dt * (i_ext - self.calcium) / tau;
        self.calcium = self.calcium.clamp(0.0, 1.0);

        // Liberação de gliotransmissor proporcional ao excesso acima do limiar
        if self.calcium > self.threshold {
            self.gliotransmitter = (self.calcium - self.threshold) / (1.0 - self.threshold);
        } else {
            // Decai quando abaixo do limiar
            self.gliotransmitter = (self.gliotransmitter - dt / self.tau_decay).max(0.0);
        }
    }

    /// glio_factor ∈ [-1, +1]: facilita LTP quando positivo, deprime quando negativo.
    pub fn glio_factor(&self) -> f32 {
        // Alta atividade → D-serina libera NR2B → LTP facilitado (+)
        // Baixa atividade prolongada → LTD dominante (−)
        let gt = self.gliotransmitter;
        if self.calcium > self.threshold {
            gt  // [0, 1] → LTP facilitado
        } else {
            -(1.0 - self.calcium / self.threshold.max(0.01)) * 0.3 // depressão leve
        }
    }
}

/// Camada glial completa — 9 astrocytes (um por RegionType).
pub struct GliaLayer {
    pub astrocytes: [Astrocyte; NUM_REGIONS],
}

impl GliaLayer {
    pub fn new() -> Self {
        use RegionType::*;
        let regions = [
            Frontal, Parietal, Temporal, Occipital, Limbic,
            Hippocampus, Cerebellum, Brainstem, CorpusCallosum,
        ];
        Self {
            astrocytes: regions.map(Astrocyte::new),
        }
    }

    /// Atualiza todos os astrocytes com a atividade de cada região.
    /// `regional_activity`: slice com um firing rate por região (mesma ordem de RegionType).
    pub fn update(&mut self, regional_activity: &[f32; NUM_REGIONS], dt: f32) {
        for (i, ast) in self.astrocytes.iter_mut().enumerate() {
            ast.update(regional_activity[i], dt);
        }
    }

    /// Retorna o glio_factor global — média ponderada de todas as regiões.
    /// Hipocampo tem peso 2× (central para plasticidade sináptica).
    pub fn global_glio_factor(&self) -> f32 {
        let hippocampus_idx = 5; // RegionType::Hippocampus
        let mut sum = 0.0f32;
        let mut weight_total = 0.0f32;
        for (i, ast) in self.astrocytes.iter().enumerate() {
            let w = if i == hippocampus_idx { 2.0 } else { 1.0 };
            sum += ast.glio_factor() * w;
            weight_total += w;
        }
        (sum / weight_total).clamp(-1.0, 1.0)
    }

    /// Atividade regional derivada dos firing rates do loop neural.
    /// Normaliza pelo pico máximo observado.
    pub fn activity_from_firing_rates(rates: &[f32]) -> [f32; NUM_REGIONS] {
        let mut out = [0.0f32; NUM_REGIONS];
        let chunks = rates.len() / NUM_REGIONS;
        if chunks == 0 {
            return out;
        }
        for (i, chunk) in rates.chunks(chunks).take(NUM_REGIONS).enumerate() {
            let mean = chunk.iter().sum::<f32>() / chunk.len() as f32;
            out[i] = (mean / 80.0).clamp(0.0, 1.0); // normaliza para ~80 Hz pico
        }
        out
    }
}
