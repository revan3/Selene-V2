// src/brain_zones/language.rs
// Áreas de Linguagem — Wernicke (compreensão) + Broca (produção)
//
// No cérebro humano estas áreas são fundamentalmente distintas:
//
//   ÁREA DE WERNICKE (Temporal posterior — BA 22, 39, 40):
//     Compreensão semântica e fonológica.
//     Lesão → Afasia de Wernicke: fala fluente mas sem sentido, não entende input.
//     Neurônios: RS + CH (reconhecimento rápido de padrões fonéticos/semânticos)
//
//   ÁREA DE BROCA (Frontal inferior — BA 44, 45):
//     Planejamento motor da fala, sintaxe, sequenciamento.
//     Lesão → Afasia de Broca: entende mas não articula, fala telegráfica.
//     Neurônios: RS + FS (sincronização gamma para articulação)
//
//   FASCÍCULO ARQUEADO: conecta Wernicke → Broca (compreendido → planejado)
//
// Wire:
//   Wernicke recebe o input tokenizado → gera comprehension_score + semantic_buffer
//   Broca recebe o semantic_buffer + goal frontal → gera fluency_signal + syntax_template
//   fluency_signal modula n_passos no server.rs
//   comprehension_score modula emocao_bias (compreensão alta = mais confiança)

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::synaptic_core::{CamadaHibrida, PrecisionType, TipoNeuronal};
use crate::config::Config;
use rand::{Rng, thread_rng};

/// Decaimento do semantic_buffer entre turnos (EMA).
const SEMANTIC_DECAY: f32 = 0.88;
/// Taxa de aprendizado do vocabulary familiarity por palavra.
const FAMILIARITY_LR: f32 = 0.02;
/// Limite do mapa de familiaridade de vocabulário.
const FAMILIARITY_MAX: usize = 2048;

pub struct LanguageAreas {
    // ── Área de Wernicke ────────────────────────────────────────────────────
    /// Camada de reconhecimento semântico — RS + CH (detecção rápida de padrões).
    pub wernicke_layer: CamadaHibrida,

    /// Score de compreensão do último input [0.0, 1.0].
    /// Alto = palavras familiares, estrutura conhecida.
    /// Baixo = input com muitas palavras desconhecidas ou muito longo.
    pub comprehension_score: f32,

    /// Buffer semântico: representação comprimida do input compreendido.
    /// Passado ao Broca como "o que foi entendido antes de produzir resposta".
    /// Biologicamente: Wernicke → fascículo arqueado → Broca.
    pub semantic_buffer: Vec<f32>,

    /// Familiaridade de vocabulário: palavra → n_exposições.
    /// Palavras mais frequentes → comprehension_score maior.
    familiarity_map: std::collections::HashMap<String, u32>,

    // ── Área de Broca ───────────────────────────────────────────────────────
    /// Camada de planejamento motor verbal — RS + FS (sincronização gamma).
    pub broca_layer: CamadaHibrida,

    /// Sinal de fluência verbal [0.0, 1.0].
    /// Alto = Selene tem muito para dizer, planejamento motor rico.
    /// Baixo = incerteza sintática ou vocabulário insuficiente.
    /// Modula n_passos no graph-walk: fluência alta → walk mais longo.
    pub fluency_signal: f32,

    /// Template sintático atual: vetor de pesos que influencia a ordem
    /// dos tokens no graph-walk (começo vs fim de sentença, pergunta vs afirmação).
    /// [0] = probabilidade de estrutura interrogativa
    /// [1] = probabilidade de estrutura declarativa
    /// [2] = probabilidade de estrutura exclamativa
    /// [3] = comprimento relativo preferido (0=curto, 1=longo)
    pub syntax_template: [f32; 4],

    /// Número de tokens processados desde a última resposta (telemetria).
    pub tokens_processed: u64,
}

impl LanguageAreas {
    pub fn new(n_neurons: usize, config: &Config) -> Self {
        // Wernicke: CH (40%) para reconhecimento rápido de padrões fonéticos/semânticos
        let wernicke_dist = vec![
            (PrecisionType::FP16, 0.40),
            (PrecisionType::INT8, 0.50),
            (PrecisionType::INT4, 0.10),
        ];
        // Broca: FS (30%) para sincronização gamma e timing articulatório
        let broca_dist = vec![
            (PrecisionType::FP32, 0.05),
            (PrecisionType::FP16, 0.45),
            (PrecisionType::INT8, 0.40),
            (PrecisionType::INT4, 0.10),
        ];

        let escala_w = 42.0 / 127.0; // Wernicke: limiar um pouco mais baixo
        let escala_b = 38.0 / 127.0;
        let n_w = (n_neurons / 3).max(8);
        let n_b = (n_neurons / 4).max(6);

        let mut wernicke = CamadaHibrida::new(
            n_w, "wernicke",
            TipoNeuronal::RS,
            Some((TipoNeuronal::CH, 0.40)), // CH para reconhecimento rápido
            Some(wernicke_dist),
            escala_w,
        );
        // Inibição lateral em Wernicke: compressão para apenas o padrão mais forte
        wernicke.init_lateral_inhibition(3, 2.0);

        let mut broca = CamadaHibrida::new(
            n_b, "broca",
            TipoNeuronal::RS,
            Some((TipoNeuronal::FS, 0.30)), // FS para sincronização gamma articulatória
            Some(broca_dist),
            escala_b,
        );
        // Inibição lateral em Broca: seleção de um padrão motor por vez
        broca.init_lateral_inhibition(4, 3.0);

        Self {
            wernicke_layer: wernicke,
            comprehension_score: 0.5,
            semantic_buffer: vec![0.0; n_w],
            familiarity_map: std::collections::HashMap::with_capacity(256),
            broca_layer: broca,
            fluency_signal: 0.5,
            syntax_template: [0.3, 0.5, 0.1, 0.5], // padrão: maioria declarativa
            tokens_processed: 0,
        }
    }

    /// Wernicke: processa tokens de input e gera comprehension_score + semantic_buffer.
    ///
    /// `tokens`: palavras do input em minúsculas
    /// `valencias`: mapa palavra→valência do swap_manager (indica se palavra é conhecida)
    ///
    /// Retorna `comprehension_score`.
    pub fn wernicke_process(
        &mut self,
        tokens: &[String],
        valencias: &std::collections::HashMap<String, f32>,
        dt: f32,
        current_time: f32,
        config: &Config,
    ) -> f32 {
        if tokens.is_empty() {
            self.comprehension_score *= SEMANTIC_DECAY;
            return self.comprehension_score;
        }

        let t_ms = current_time * 1000.0;
        let mut rng = thread_rng();
        let n = self.wernicke_layer.neuronios.len();

        // 1. Calcula familiaridade do input
        let n_known = tokens.iter()
            .filter(|w| valencias.contains_key(w.as_str())
                || self.familiarity_map.contains_key(w.as_str()))
            .count();
        let familiarity = if tokens.is_empty() { 0.5 }
            else { n_known as f32 / tokens.len() as f32 };

        // 2. Atualiza mapa de familiaridade
        for token in tokens {
            if self.familiarity_map.len() >= FAMILIARITY_MAX {
                // Remove o menos familiar
                if let Some(min_key) = self.familiarity_map.iter()
                    .min_by_key(|(_, &v)| v).map(|(k, _)| k.clone()) {
                    self.familiarity_map.remove(&min_key);
                }
            }
            let count = self.familiarity_map.entry(token.clone()).or_insert(0);
            *count = count.saturating_add(1);
        }

        // 3. Input para Wernicke: intensidade proporcional à familiaridade + valência
        let val_mean: f32 = if tokens.is_empty() { 0.0 } else {
            tokens.iter()
                .filter_map(|w| valencias.get(w.as_str()).copied())
                .sum::<f32>() / tokens.len() as f32
        };
        let input_strength = familiarity * 30.0 + val_mean.abs() * 10.0;
        let wernicke_input: Vec<f32> = (0..n)
            .map(|i| {
                let token_idx = i * tokens.len() / n;
                let tok_familiarity = tokens.get(token_idx)
                    .and_then(|w| self.familiarity_map.get(w.as_str()))
                    .copied()
                    .unwrap_or(0) as f32 / 10.0;
                input_strength * (0.5 + tok_familiarity * 0.5) + rng.gen_range(-2.0..2.0)
            })
            .collect();

        let wernicke_spikes = self.wernicke_layer.update(&wernicke_input, dt, t_ms);

        // 4. Atualiza semantic_buffer com taxa de disparo
        for i in 0..n.min(self.semantic_buffer.len()) {
            let spike_val = if *wernicke_spikes.get(i).unwrap_or(&false) { 1.0 } else { 0.0 };
            self.semantic_buffer[i] = self.semantic_buffer[i] * SEMANTIC_DECAY + spike_val * (1.0 - SEMANTIC_DECAY);
        }

        // 5. comprehension_score: taxa de disparo + familiaridade
        let spike_rate = wernicke_spikes.iter().filter(|&&s| s).count() as f32 / n as f32;
        let raw_comprehension = spike_rate * 0.5 + familiarity * 0.5;
        self.comprehension_score = self.comprehension_score * 0.7 + raw_comprehension * 0.3;
        self.tokens_processed += tokens.len() as u64;

        self.comprehension_score
    }

    /// Broca: planeja a resposta com base no semantic_buffer + contexto frontal.
    ///
    /// `frontal_goal_signal`: atividade do goal atual do frontal (0.0-1.0)
    /// `emocao`: emoção atual (-1.0 a 1.0) — afeta template sintático
    ///
    /// Retorna `(fluency_signal, syntax_template)`.
    pub fn broca_plan(
        &mut self,
        frontal_goal_signal: f32,
        emocao: f32,
        comprehension_score: f32,
        dt: f32,
        current_time: f32,
        config: &Config,
    ) -> (f32, [f32; 4]) {
        let t_ms = current_time * 1000.0;
        let mut rng = thread_rng();
        let n = self.broca_layer.neuronios.len();

        // 1. Broca recebe o semantic_buffer (de Wernicke via fascículo arqueado)
        let buf_strength: f32 = self.semantic_buffer.iter().sum::<f32>()
            / self.semantic_buffer.len().max(1) as f32;

        // 2. Input para Broca: intenção frontal + compreensão semântica + emoção
        let broca_input: Vec<f32> = (0..n)
            .map(|i| {
                let sem_idx = i * self.semantic_buffer.len() / n;
                let sem = self.semantic_buffer.get(sem_idx).copied().unwrap_or(0.0);
                (sem * 15.0 + frontal_goal_signal * 10.0 + comprehension_score * 8.0)
                    + rng.gen_range(-1.5..1.5)
            })
            .collect();

        let broca_spikes = self.broca_layer.update(&broca_input, dt, t_ms);
        let broca_rate = broca_spikes.iter().filter(|&&s| s).count() as f32 / n as f32;

        // 3. fluency_signal: taxa de disparo do Broca + qualidade do semantic_buffer
        let raw_fluency = broca_rate * 0.6 + buf_strength * 0.4;
        self.fluency_signal = self.fluency_signal * 0.80 + raw_fluency * 0.20;

        // 4. Atualiza syntax_template baseado em emoção e goal
        // Emoção positiva → mais exclamativo; negativa → mais declarativo/quieto
        // Goal ativo → mais declarativo (assertivo); sem goal → mais interrogativo
        let emocao_n = emocao.clamp(-1.0, 1.0);
        let goal_strength = frontal_goal_signal.clamp(0.0, 1.0);

        self.syntax_template[0] = (0.3 - goal_strength * 0.2 + (1.0 - comprehension_score) * 0.3)
            .clamp(0.0, 1.0); // interrogativo
        self.syntax_template[1] = (0.5 + goal_strength * 0.3 - emocao_n.abs() * 0.1)
            .clamp(0.0, 1.0); // declarativo
        self.syntax_template[2] = (0.1 + emocao_n.max(0.0) * 0.4)
            .clamp(0.0, 1.0); // exclamativo
        self.syntax_template[3] = (0.5 + self.fluency_signal * 0.5)
            .clamp(0.0, 1.0); // comprimento (0=curto, 1=longo)

        (self.fluency_signal, self.syntax_template)
    }

    /// Retorna delta de n_passos baseado nas áreas de linguagem.
    ///
    /// fluência alta + compreensão alta → mais passos (resposta rica)
    /// fluência baixa OU compreensão baixa → menos passos (resposta concisa/cautelosa)
    pub fn walk_length_delta(&self) -> f32 {
        let combined = self.fluency_signal * 0.6 + self.comprehension_score * 0.4;
        // Mapeamento não-linear: [0.0, 1.0] → [-2.0, +3.0]
        (combined - 0.5) * 5.0
    }

    /// Retorna se Selene está em modo "interrogativo" (quer perguntar).
    /// Ativado quando comprehension_score baixo + syntax_template interrogativo alto.
    pub fn quer_perguntar(&self) -> bool {
        self.comprehension_score < 0.45 && self.syntax_template[0] > 0.5
    }

    pub fn estatisticas(&self) -> LanguageStats {
        LanguageStats {
            wernicke: self.wernicke_layer.estatisticas(),
            broca: self.broca_layer.estatisticas(),
            comprehension_score: self.comprehension_score,
            fluency_signal: self.fluency_signal,
            tokens_processed: self.tokens_processed,
        }
    }
}

pub struct LanguageStats {
    pub wernicke: crate::synaptic_core::CamadaStats,
    pub broca: crate::synaptic_core::CamadaStats,
    pub comprehension_score: f32,
    pub fluency_signal: f32,
    pub tokens_processed: u64,
}
