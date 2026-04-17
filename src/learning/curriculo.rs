// src/learning/curriculo.rs
// Currículo de aprendizado fonético hierárquico para Selene Brain 2.0.
//
// Mapeia o modelo de aquisição de linguagem humana em 12 fases,
// do som puro à palavra emergente. Cada fase é definida em termos de
// parâmetros físicos de onda — sem texto, sem símbolos.
//
// Ordem biológica:
//   0  — Bandas de frequência pura (hardware coclear)
//   1  — Vogais: estados estáveis de formante
//   2  — Vogais nasais: anti-formante + murmúrio
//   3  — CV Labiais: onset labial + vogal
//   4  — CV Dentais/Alveolares
//   5  — CV Velares
//   6  — CV Fricativas
//   7  — CV Africadas e complexas
//   8  — CVC: sílaba fechada
//   9  — CVCV: reduplicação (primeiras "palavras")
//   10 — Clusters CCV
//   11 — Padrões de alta frequência (emergência de palavras)
//
// Melhoria: cada sílaba inclui parâmetros de prosódia esperados (F0 médio,
// duração típica) para que o sistema aprenda ritmo junto com fonemas.
// Melhoria: campo `prioridade` para ordenar apresentação dentro da fase.
// Melhoria: função `maturidade_para_progredir` define critério automático.
#![allow(dead_code)]

use crate::storage::ondas::TipoOnset;

// ─── Estágios ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EstagioCurriculo {
    Fase0BandasPuras      = 0,
    Fase1VogaisPuras      = 1,
    Fase2VogaisNasais     = 2,
    Fase3LabiaisCV        = 3,
    Fase4DentaisCV        = 4,
    Fase5VelaresCV        = 5,
    Fase6FricativasCV     = 6,
    Fase7AfricadasCV      = 7,
    Fase8CVC              = 8,
    Fase9CVCV             = 9,
    Fase10ClustersCCV     = 10,
    Fase11AltaFrequencia  = 11,
}

impl EstagioCurriculo {
    /// Retorna o estágio seguinte. None se já estamos no último.
    pub fn proximo(&self) -> Option<Self> {
        match self {
            Self::Fase0BandasPuras     => Some(Self::Fase1VogaisPuras),
            Self::Fase1VogaisPuras     => Some(Self::Fase2VogaisNasais),
            Self::Fase2VogaisNasais    => Some(Self::Fase3LabiaisCV),
            Self::Fase3LabiaisCV       => Some(Self::Fase4DentaisCV),
            Self::Fase4DentaisCV       => Some(Self::Fase5VelaresCV),
            Self::Fase5VelaresCV       => Some(Self::Fase6FricativasCV),
            Self::Fase6FricativasCV    => Some(Self::Fase7AfricadasCV),
            Self::Fase7AfricadasCV     => Some(Self::Fase8CVC),
            Self::Fase8CVC             => Some(Self::Fase9CVCV),
            Self::Fase9CVCV            => Some(Self::Fase10ClustersCCV),
            Self::Fase10ClustersCCV    => Some(Self::Fase11AltaFrequencia),
            Self::Fase11AltaFrequencia => None,
        }
    }

    /// Número mínimo de exposições para considerar o estágio consolidado.
    pub fn exposicoes_minimas(&self) -> u32 {
        match self {
            Self::Fase0BandasPuras     => 10,
            Self::Fase1VogaisPuras     => 50,
            Self::Fase2VogaisNasais    => 40,
            Self::Fase3LabiaisCV       => 60,
            Self::Fase4DentaisCV       => 60,
            Self::Fase5VelaresCV       => 60,
            Self::Fase6FricativasCV    => 80,
            Self::Fase7AfricadasCV     => 80,
            Self::Fase8CVC             => 100,
            Self::Fase9CVCV            => 100,
            Self::Fase10ClustersCCV    => 120,
            Self::Fase11AltaFrequencia => 200,
        }
    }
}

// ─── Parâmetros de sílaba ─────────────────────────────────────────────────────

/// Parâmetros acústicos esperados de uma sílaba, em termos de onda.
/// `escrita` é apenas para referência humana — NUNCA entra no DB.
#[derive(Debug, Clone)]
pub struct Silaba {
    /// Referência humana (ex: "ba") — não entra no treinamento.
    pub referencia_humana: &'static str,
    /// Tipo de onset consonantal.
    pub onset_tipo:   TipoOnset,
    /// Frequência característica do onset em Hz.
    /// Para oclusivas: frequência do burst. Para fricativas: centro do ruído.
    pub onset_freq_hz: f32,
    /// F1 esperado da vogal (Hz).
    pub vogal_f1: f32,
    /// F2 esperado da vogal (Hz).
    pub vogal_f2: f32,
    /// F3 esperado da vogal (Hz).
    pub vogal_f3: f32,
    /// VOT esperado em ms. Negativo = pré-vozeado.
    pub vot_ms: f32,
    /// Frequência fundamental média esperada (Hz). 0 = não-voiced.
    pub f0_medio_hz: f32,
    /// Duração típica da sílaba em ms.
    pub duracao_ms: u32,
    /// Estágio de aprendizado a que pertence.
    pub estagio: EstagioCurriculo,
    /// Prioridade dentro do estágio (menor = apresentado primeiro).
    pub prioridade: u8,
}

// ─── Parâmetros de vogal ──────────────────────────────────────────────────────

/// Retorna (F1, F2, F3) para as vogais orais do português (Hz).
/// Baseado em dados acústicos de falantes nativos de PT-BR.
pub fn formantes_vogal_oral(vogal: char) -> (f32, f32, f32) {
    match vogal {
        'a' => (800.0, 1200.0, 2550.0),
        'e' => (500.0, 1700.0, 2500.0),
        'i' => (300.0, 2300.0, 3100.0),
        'o' => (500.0,  900.0, 2500.0),
        'u' => (300.0,  800.0, 2400.0),
        _   => (500.0, 1400.0, 2500.0), // vogal média default
    }
}

/// Retorna (F1, F2, F3) para as vogais nasais do português.
/// Vogais nasais têm F1 reduzido e formante nasal extra ~250Hz.
pub fn formantes_vogal_nasal(vogal: char) -> (f32, f32, f32) {
    match vogal {
        'a' => (600.0, 1100.0, 2500.0), // ã
        'e' => (400.0, 1600.0, 2400.0), // ẽ
        'i' => (280.0, 2100.0, 3000.0), // ĩ
        'o' => (400.0,  850.0, 2400.0), // õ
        'u' => (280.0,  750.0, 2300.0), // ũ
        _   => (400.0, 1200.0, 2400.0),
    }
}

// ─── Banco de sílabas por estágio ────────────────────────────────────────────

/// Retorna todas as sílabas de um estágio, ordenadas por prioridade.
pub fn silabas_da_fase(estagio: EstagioCurriculo) -> Vec<Silaba> {
    let mut v: Vec<Silaba> = CATALOGO_SILABAS.iter()
        .filter(|s| s.estagio == estagio)
        .cloned()
        .collect();
    v.sort_by_key(|s| s.prioridade);
    v
}

/// Retorna todas as sílabas até e incluindo o estágio informado.
/// Útil para treinamento cumulativo (estágio atual + revisão dos anteriores).
pub fn silabas_ate_fase(estagio: EstagioCurriculo) -> Vec<Silaba> {
    let mut v: Vec<Silaba> = CATALOGO_SILABAS.iter()
        .filter(|s| s.estagio <= estagio)
        .cloned()
        .collect();
    v.sort_by(|a, b| a.estagio.cmp(&b.estagio).then(a.prioridade.cmp(&b.prioridade)));
    v
}

// ─── Catálogo completo ────────────────────────────────────────────────────────
//
// F0 médio de voz feminina adulta ~220Hz; masculina ~120Hz.
// Usamos 170Hz (média) como default neutro.

const F0: f32 = 170.0; // Hz — frequência fundamental média neutra

/// Catálogo estático de sílabas com parâmetros acústicos.
static CATALOGO_SILABAS: &[Silaba] = &[

    // ── Fase 1: Vogais puras ──────────────────────────────────────────────
    Silaba { referencia_humana: "a",
        onset_tipo: TipoOnset::Vogal, onset_freq_hz: 0.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 200,
        estagio: EstagioCurriculo::Fase1VogaisPuras, prioridade: 0 },

    Silaba { referencia_humana: "e",
        onset_tipo: TipoOnset::Vogal, onset_freq_hz: 0.0,
        vogal_f1: 500.0, vogal_f2: 1700.0, vogal_f3: 2500.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 180,
        estagio: EstagioCurriculo::Fase1VogaisPuras, prioridade: 1 },

    Silaba { referencia_humana: "i",
        onset_tipo: TipoOnset::Vogal, onset_freq_hz: 0.0,
        vogal_f1: 300.0, vogal_f2: 2300.0, vogal_f3: 3100.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 160,
        estagio: EstagioCurriculo::Fase1VogaisPuras, prioridade: 2 },

    Silaba { referencia_humana: "o",
        onset_tipo: TipoOnset::Vogal, onset_freq_hz: 0.0,
        vogal_f1: 500.0, vogal_f2: 900.0, vogal_f3: 2500.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 180,
        estagio: EstagioCurriculo::Fase1VogaisPuras, prioridade: 3 },

    Silaba { referencia_humana: "u",
        onset_tipo: TipoOnset::Vogal, onset_freq_hz: 0.0,
        vogal_f1: 300.0, vogal_f2: 800.0, vogal_f3: 2400.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 160,
        estagio: EstagioCurriculo::Fase1VogaisPuras, prioridade: 4 },

    // ── Fase 2: Vogais nasais ─────────────────────────────────────────────
    Silaba { referencia_humana: "ã",
        onset_tipo: TipoOnset::Nasal, onset_freq_hz: 250.0,
        vogal_f1: 600.0, vogal_f2: 1100.0, vogal_f3: 2500.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 220,
        estagio: EstagioCurriculo::Fase2VogaisNasais, prioridade: 0 },

    Silaba { referencia_humana: "õ",
        onset_tipo: TipoOnset::Nasal, onset_freq_hz: 250.0,
        vogal_f1: 400.0, vogal_f2: 850.0, vogal_f3: 2400.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 200,
        estagio: EstagioCurriculo::Fase2VogaisNasais, prioridade: 1 },

    // ── Fase 3: CV Labiais ────────────────────────────────────────────────
    // /ma/: murmúrio nasal 250Hz → vogal /a/
    Silaba { referencia_humana: "ma",
        onset_tipo: TipoOnset::Nasal, onset_freq_hz: 250.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 250,
        estagio: EstagioCurriculo::Fase3LabiaisCV, prioridade: 0 },

    Silaba { referencia_humana: "pa",
        onset_tipo: TipoOnset::OclusivaSurda, onset_freq_hz: 300.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: 60.0, f0_medio_hz: F0, duracao_ms: 230,
        estagio: EstagioCurriculo::Fase3LabiaisCV, prioridade: 1 },

    Silaba { referencia_humana: "ba",
        onset_tipo: TipoOnset::OclusivaSonora, onset_freq_hz: 200.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: -20.0, f0_medio_hz: F0, duracao_ms: 220,
        estagio: EstagioCurriculo::Fase3LabiaisCV, prioridade: 2 },

    Silaba { referencia_humana: "me",
        onset_tipo: TipoOnset::Nasal, onset_freq_hz: 250.0,
        vogal_f1: 500.0, vogal_f2: 1700.0, vogal_f3: 2500.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 220,
        estagio: EstagioCurriculo::Fase3LabiaisCV, prioridade: 3 },

    Silaba { referencia_humana: "mi",
        onset_tipo: TipoOnset::Nasal, onset_freq_hz: 250.0,
        vogal_f1: 300.0, vogal_f2: 2300.0, vogal_f3: 3100.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 200,
        estagio: EstagioCurriculo::Fase3LabiaisCV, prioridade: 4 },

    Silaba { referencia_humana: "mo",
        onset_tipo: TipoOnset::Nasal, onset_freq_hz: 250.0,
        vogal_f1: 500.0, vogal_f2: 900.0, vogal_f3: 2500.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 220,
        estagio: EstagioCurriculo::Fase3LabiaisCV, prioridade: 5 },

    Silaba { referencia_humana: "mu",
        onset_tipo: TipoOnset::Nasal, onset_freq_hz: 250.0,
        vogal_f1: 300.0, vogal_f2: 800.0, vogal_f3: 2400.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 200,
        estagio: EstagioCurriculo::Fase3LabiaisCV, prioridade: 6 },

    // ── Fase 4: CV Dentais/Alveolares ─────────────────────────────────────
    Silaba { referencia_humana: "ta",
        onset_tipo: TipoOnset::OclusivaSurda, onset_freq_hz: 3500.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: 55.0, f0_medio_hz: F0, duracao_ms: 220,
        estagio: EstagioCurriculo::Fase4DentaisCV, prioridade: 0 },

    Silaba { referencia_humana: "da",
        onset_tipo: TipoOnset::OclusivaSonora, onset_freq_hz: 2500.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: -10.0, f0_medio_hz: F0, duracao_ms: 210,
        estagio: EstagioCurriculo::Fase4DentaisCV, prioridade: 1 },

    Silaba { referencia_humana: "na",
        onset_tipo: TipoOnset::Nasal, onset_freq_hz: 300.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 230,
        estagio: EstagioCurriculo::Fase4DentaisCV, prioridade: 2 },

    Silaba { referencia_humana: "la",
        onset_tipo: TipoOnset::Lateral, onset_freq_hz: 2700.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 220,
        estagio: EstagioCurriculo::Fase4DentaisCV, prioridade: 3 },

    // ── Fase 5: CV Velares ────────────────────────────────────────────────
    // /ka/ (escrita: ca/que/qui) — todas têm o mesmo onset velar
    Silaba { referencia_humana: "ka",
        onset_tipo: TipoOnset::OclusivaSurda, onset_freq_hz: 2000.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: 70.0, f0_medio_hz: F0, duracao_ms: 230,
        estagio: EstagioCurriculo::Fase5VelaresCV, prioridade: 0 },

    Silaba { referencia_humana: "ga",
        onset_tipo: TipoOnset::OclusivaSonora, onset_freq_hz: 1800.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: -15.0, f0_medio_hz: F0, duracao_ms: 210,
        estagio: EstagioCurriculo::Fase5VelaresCV, prioridade: 1 },

    Silaba { referencia_humana: "ke",
        onset_tipo: TipoOnset::OclusivaSurda, onset_freq_hz: 2200.0,
        vogal_f1: 500.0, vogal_f2: 1700.0, vogal_f3: 2500.0,
        vot_ms: 65.0, f0_medio_hz: F0, duracao_ms: 210,
        estagio: EstagioCurriculo::Fase5VelaresCV, prioridade: 2 },

    // ── Fase 6: CV Fricativas ─────────────────────────────────────────────
    Silaba { referencia_humana: "sa",
        onset_tipo: TipoOnset::Fricativa, onset_freq_hz: 5500.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: 0.0, f0_medio_hz: 0.0, duracao_ms: 280,
        estagio: EstagioCurriculo::Fase6FricativasCV, prioridade: 0 },

    Silaba { referencia_humana: "za",
        onset_tipo: TipoOnset::Fricativa, onset_freq_hz: 5000.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 260,
        estagio: EstagioCurriculo::Fase6FricativasCV, prioridade: 1 },

    Silaba { referencia_humana: "fa",
        onset_tipo: TipoOnset::Fricativa, onset_freq_hz: 7000.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: 0.0, f0_medio_hz: 0.0, duracao_ms: 270,
        estagio: EstagioCurriculo::Fase6FricativasCV, prioridade: 2 },

    Silaba { referencia_humana: "va",
        onset_tipo: TipoOnset::Fricativa, onset_freq_hz: 6500.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 260,
        estagio: EstagioCurriculo::Fase6FricativasCV, prioridade: 3 },

    // ── Fase 7: CV Africadas e complexas ─────────────────────────────────
    Silaba { referencia_humana: "xa",  // /ʃa/
        onset_tipo: TipoOnset::Fricativa, onset_freq_hz: 3500.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: 0.0, f0_medio_hz: 0.0, duracao_ms: 270,
        estagio: EstagioCurriculo::Fase7AfricadasCV, prioridade: 0 },

    Silaba { referencia_humana: "ja",  // /ʒa/
        onset_tipo: TipoOnset::Fricativa, onset_freq_hz: 3200.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 260,
        estagio: EstagioCurriculo::Fase7AfricadasCV, prioridade: 1 },

    Silaba { referencia_humana: "lha",  // /ʎa/
        onset_tipo: TipoOnset::Lateral, onset_freq_hz: 3000.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 250,
        estagio: EstagioCurriculo::Fase7AfricadasCV, prioridade: 2 },

    Silaba { referencia_humana: "nha",  // /ɲa/
        onset_tipo: TipoOnset::Nasal, onset_freq_hz: 350.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 260,
        estagio: EstagioCurriculo::Fase7AfricadasCV, prioridade: 3 },

    Silaba { referencia_humana: "ra",   // /ɾa/ — tepe (r fraco)
        onset_tipo: TipoOnset::Aproximante, onset_freq_hz: 1500.0,
        vogal_f1: 800.0, vogal_f2: 1200.0, vogal_f3: 2550.0,
        vot_ms: 0.0, f0_medio_hz: F0, duracao_ms: 200,
        estagio: EstagioCurriculo::Fase7AfricadasCV, prioridade: 4 },
];

// ─── Critério de progressão ───────────────────────────────────────────────────

/// Avalia se a Selene está pronta para avançar para o próximo estágio.
///
/// `contagens`: número de exposições a cada sílaba do estágio atual.
/// `estagio`: estágio atual.
///
/// Retorna true se o mínimo de exposições foi atingido para >= 80% das sílabas.
pub fn pronta_para_progredir(contagens: &[u32], estagio: EstagioCurriculo) -> bool {
    if contagens.is_empty() { return false; }
    let minimo = estagio.exposicoes_minimas();
    let acima = contagens.iter().filter(|&&c| c >= minimo).count();
    acima >= (contagens.len() * 4) / 5  // >= 80%
}

// ─── Testes ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fase1_tem_5_vogais() {
        let s = silabas_da_fase(EstagioCurriculo::Fase1VogaisPuras);
        assert_eq!(s.len(), 5, "deve ter exatamente 5 vogais");
    }

    #[test]
    fn vogais_ordenadas_por_prioridade() {
        let s = silabas_da_fase(EstagioCurriculo::Fase1VogaisPuras);
        for i in 1..s.len() {
            assert!(s[i].prioridade >= s[i-1].prioridade);
        }
    }

    #[test]
    fn progressao_estagio_funciona() {
        let e = EstagioCurriculo::Fase3LabiaisCV;
        assert_eq!(e.proximo(), Some(EstagioCurriculo::Fase4DentaisCV));
    }

    #[test]
    fn ultimo_estagio_sem_proximo() {
        assert_eq!(EstagioCurriculo::Fase11AltaFrequencia.proximo(), None);
    }

    #[test]
    fn formantes_vogal_a_corretos() {
        let (f1, f2, _) = formantes_vogal_oral('a');
        assert!(f1 > 700.0 && f1 < 900.0, "F1 /a/ = {}", f1);
        assert!(f2 > 1000.0 && f2 < 1400.0, "F2 /a/ = {}", f2);
    }
}
