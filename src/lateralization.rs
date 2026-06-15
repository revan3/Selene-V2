// =============================================================================
// src/lateralization.rs — Cérebro Lateralizado SELETIVO (hemisférios especializados)
// =============================================================================
//
// Implementa a lateralização *especializada* (não cópia) discutida na análise:
//
//   HEMISFÉRIO ESQUERDO  → linguagem / sequencial
//       Temporal + Frontal | janela temporal CURTA | input FINO | aprende RÁPIDO
//       viés dopaminérgico maior (explorador)
//
//   HEMISFÉRIO DIREITO   → espacial / holístico
//       Occipital + Parietal | janela LONGA | input HOLÍSTICO (suavizado) | aprende DEVAGAR
//
//   CORPO CALOSO → canal ESTREITO e assíncrono: cada lado envia apenas um RESUMO
//       escalar do tick anterior ao outro (não o estado bruto). Isso evita o
//       gargalo de banda e preserva a especialização.
//
// Princípio central (validado na discussão): o ganho vem da DIFERENÇA entre os
// lados, não da duplicação. Sistemas globais (neuroquímica, sono, tronco) NÃO são
// lateralizados — ficam de fora deste módulo, compartilhados.
//
// ATENÇÃO de API (verificada no código real):
//   temporal.process / frontal.decide  → recebem current_time em SEGUNDOS
//   occipital.visual_sweep / parietal.integrate → recebem t em MILISSEGUNDOS
// =============================================================================

use crate::config::Config;
use crate::brain_zones::{
    temporal::TemporalLobe,
    frontal::FrontalLobe,
    occipital::OccipitalLobe,
    parietal::ParietalLobe,
};

/// Parâmetros que tornam os hemisférios DIFERENTES (a fonte da especialização).
#[derive(Debug, Clone, Copy)]
pub struct PerfilHemisferio {
    /// Janela temporal de integração do contexto caloso (ms). Esq curto, dir longo.
    pub janela_temporal_ms: f32,
    /// Granularidade do input: 1.0 = fino (detalhe); <1.0 = holístico (suaviza).
    pub granularidade: f32,
    /// Taxa de aprendizado das zonas. Esq rápido, dir devagar.
    pub taxa_aprendizado: f32,
    /// Viés dopaminérgico (modula o frontal). Esq explorador, dir estável.
    pub vies_dopamina: f32,
}

impl PerfilHemisferio {
    /// Esquerdo: linguagem/sequencial — rápido, fino, explorador.
    pub fn esquerdo() -> Self {
        Self { janela_temporal_ms: 25.0, granularidade: 1.0, taxa_aprendizado: 0.02, vies_dopamina: 1.15 }
    }
    /// Direito: espacial/holístico — lento, grosso, estável.
    pub fn direito() -> Self {
        Self { janela_temporal_ms: 200.0, granularidade: 0.5, taxa_aprendizado: 0.008, vies_dopamina: 0.95 }
    }
}

/// Fator de transmissão do corpo caloso (canal estreito: só uma fração cruza).
const GANHO_CALOSO: f32 = 0.3;

pub struct CerebroLateralizado {
    n: usize,
    pub perfil_esq: PerfilHemisferio,
    pub perfil_dir: PerfilHemisferio,

    // Hemisfério esquerdo (linguagem/sequencial)
    temporal: TemporalLobe,
    frontal:  FrontalLobe,
    // Hemisfério direito (espacial/holístico)
    occipital: OccipitalLobe,
    parietal:  ParietalLobe,

    // Corpo caloso: resumos escalares do tick anterior (assíncrono).
    resumo_esq: f32,
    resumo_dir: f32,
}

impl CerebroLateralizado {
    pub fn novo(n: usize, config: &Config) -> Self {
        let perfil_esq = PerfilHemisferio::esquerdo();
        let perfil_dir = PerfilHemisferio::direito();
        Self {
            n,
            perfil_esq,
            perfil_dir,
            // Esquerdo aprende rápido (taxa alta no learning_rate do temporal).
            temporal: TemporalLobe::new(n, perfil_esq.taxa_aprendizado, 0.2, config),
            frontal:  FrontalLobe::new(n, 0.2, 0.1, config),
            occipital: OccipitalLobe::new(n, 0.2, config),
            parietal:  ParietalLobe::new(n, 0.2, config),
            resumo_esq: 0.0,
            resumo_dir: 0.0,
        }
    }

    /// Suaviza um vetor por média móvel — simula processamento HOLÍSTICO.
    /// `granularidade` 1.0 = sem suavização; menor = janela de blur maior.
    fn aplicar_granularidade(entrada: &[f32], granularidade: f32) -> Vec<f32> {
        if granularidade >= 0.999 {
            return entrada.to_vec();
        }
        // janela de blur cresce conforme granularidade cai
        let janela = ((1.0 - granularidade) * 8.0).round().max(1.0) as usize;
        let n = entrada.len();
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let lo = i.saturating_sub(janela);
            let hi = (i + janela + 1).min(n);
            let soma: f32 = entrada[lo..hi].iter().sum();
            out[i] = soma / (hi - lo) as f32;
        }
        out
    }

    /// Reconstrói um vetor de tamanho `n` a partir das features do occipital.
    fn reconstruir_visao(features: &[f32], n: usize) -> Vec<f32> {
        let chunk = (n / features.len().max(1)).max(1);
        let mut full = vec![0.0f32; n];
        for (i, &f) in features.iter().enumerate() {
            let start = i * chunk;
            let end = (start + chunk).min(n);
            for j in start..end { full[j] = f / 100.0; }
        }
        full
    }

    /// Um passo de processamento dos dois hemisférios.
    /// Retorna (saída_esquerda, saída_direita), ambas de tamanho `n`.
    pub fn tick(&mut self, input: &[f32], dt: f32, t_seg: f32, config: &Config) -> (Vec<f32>, Vec<f32>) {
        let t_ms = t_seg * 1000.0;

        // ── Corpo caloso: bias que cada lado recebe do OUTRO (tick anterior) ──
        let bias_para_esq = vec![self.resumo_dir * GANHO_CALOSO; self.n];
        let bias_para_dir = vec![self.resumo_esq * GANHO_CALOSO; self.n];

        // ── HEMISFÉRIO ESQUERDO: input fino → Temporal → Frontal ──────────────
        self.frontal.set_dopamine(self.perfil_esq.vies_dopamina);
        let input_fino = Self::aplicar_granularidade(input, self.perfil_esq.granularidade);
        let temporal_out = self.temporal.process(&input_fino, &bias_para_esq, dt, t_seg, config);
        let saida_esq = self.frontal.decide(&temporal_out, &bias_para_esq, dt, t_seg, config);

        // ── HEMISFÉRIO DIREITO: input holístico → Occipital → Parietal ────────
        let input_holistico = Self::aplicar_granularidade(input, self.perfil_dir.granularidade);
        let features = self.occipital.visual_sweep(
            &input_holistico, dt, Some(&self.parietal.spatial_map), t_ms, config,
        );
        let vision_full = Self::reconstruir_visao(&features, self.n);
        // O bias caloso entra como "propriocepção" do parietal (contexto do esquerdo).
        let saida_dir = self.parietal.integrate(&vision_full, &bias_para_dir, dt, t_ms, config);

        // ── Atualiza resumos do corpo caloso para o PRÓXIMO tick ──────────────
        self.resumo_esq = media_abs(&saida_esq);
        self.resumo_dir = media_abs(&saida_dir);

        (saida_esq, saida_dir)
    }

    /// Resumos atuais de cada lado (para inspeção/medição de especialização).
    pub fn resumos(&self) -> (f32, f32) { (self.resumo_esq, self.resumo_dir) }
}

/// Média do valor absoluto — resumo escalar que cruza o corpo caloso.
fn media_abs(v: &[f32]) -> f32 {
    if v.is_empty() { return 0.0; }
    v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32
}
