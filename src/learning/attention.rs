// src/learning/attention.rs
// Mecanismo de Atenção Seletiva Bio-inspirado
//
// Modela o "spotlight" atencional que amplifica sinais salientes e suprime ruído.
//
// Dois componentes:
//   1. Atenção bottom-up (salience): novidade detectada por desvio da média corrente
//   2. Atenção top-down (goal-directed): bias do FrontalLobe sobre onde focar
//
// Saída: vetor de ganhos por canal (0..1) multiplicado pelo input sensorial.
//
// Base biológica:
//   - Colliculus superior: saliência visual rápida (bottom-up)
//   - Córtex pré-frontal: atenção dirigida a objetivos (top-down)
//   - Pulvinar talâmico: amplifica o canal atendido

#![allow(dead_code)]
#![allow(unused_variables)]

/// Constante de tempo para média móvel de atividade de fundo (ms).
/// Lento o suficiente para capturar a linha de base, rápido para seguir mudanças.
const TAU_BASELINE_MS: f32 = 500.0;
/// Quão forte é o boost no canal mais saliente (máximo).
const MAX_GAIN:        f32 = 2.5;
/// Ganho mínimo — nunca suprime completamente (1/MAX_GAIN aprox).
const MIN_GAIN:        f32 = 0.3;
/// Peso do componente top-down vs bottom-up.
const TOPDOWN_WEIGHT:  f32 = 0.4;

/// Gate de atenção multicanal.
///
/// Cada "canal" corresponde a um neurônio ou grupo de neurônios na entrada sensorial.
/// O gate calcula um vetor de ganhos que é aplicado ao input antes de processar.
pub struct AttentionGate {
    /// Média móvel de ativação por canal (linha de base).
    baseline: Vec<f32>,
    /// Ganhos atuais por canal (0..MAX_GAIN).
    pub gains: Vec<f32>,
    /// Bias top-down (sinal do FrontalLobe).
    topdown_bias: Vec<f32>,
    /// Índice do canal mais saliente (para telemetria).
    pub canal_foco: usize,
    /// Nível médio de atenção (0..1).
    pub nivel_atencao: f32,
    n_channels: usize,
}

impl AttentionGate {
    pub fn new(n_channels: usize) -> Self {
        Self {
            baseline:     vec![0.1; n_channels],
            gains:        vec![1.0; n_channels],
            topdown_bias: vec![0.0; n_channels],
            canal_foco:   0,
            nivel_atencao: 0.5,
            n_channels,
        }
    }

    /// Aplica o gate ao input sensorial e atualiza estado interno.
    ///
    /// - `input`: firing rates da camada sensorial (ex: V1 ou cochlea)
    /// - `dt_ms`: passo de tempo em ms
    ///
    /// Retorna o input modulado pelo ganho atencional.
    pub fn attend(&mut self, input: &[f32], dt_ms: f32) -> Vec<f32> {
        let n = self.n_channels.min(input.len());
        let decay = (-dt_ms / TAU_BASELINE_MS).exp();

        // Atualiza linha de base (média exponencial)
        for i in 0..n {
            self.baseline[i] = self.baseline[i] * decay + input[i] * (1.0 - decay);
        }

        // Saliência bottom-up: desvio absoluto da linha de base
        let mut salience = vec![0.0f32; n];
        for i in 0..n {
            salience[i] = (input[i] - self.baseline[i]).abs();
        }

        // Normaliza saliência para [0, 1]
        let max_sal = salience.iter().cloned().fold(0.0f32, f32::max).max(0.001);
        for s in &mut salience { *s /= max_sal; }

        // Combina bottom-up e top-down
        let mut combined = vec![0.0f32; n];
        for i in 0..n {
            let td = if i < self.topdown_bias.len() { self.topdown_bias[i] } else { 0.0 };
            combined[i] = salience[i] * (1.0 - TOPDOWN_WEIGHT) + td * TOPDOWN_WEIGHT;
        }

        // Softmax esparso: amplifica máximo, atenua o resto
        let max_c = combined.iter().cloned().fold(0.0f32, f32::max);
        self.canal_foco = combined.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        for i in 0..n {
            // Ganho: canal saliente recebe MAX_GAIN, canais suprimidos recebem MIN_GAIN
            let normed = if max_c > 0.001 { combined[i] / max_c } else { 0.5 };
            self.gains[i] = MIN_GAIN + (MAX_GAIN - MIN_GAIN) * normed.powi(2);
        }

        // Nível médio de atenção
        self.nivel_atencao = self.gains[0..n].iter().sum::<f32>() / n as f32 / MAX_GAIN;

        // Aplica ganhos ao input
        let mut out = Vec::with_capacity(input.len());
        for i in 0..input.len() {
            let gain = if i < n { self.gains[i] } else { 1.0 };
            out.push((input[i] * gain).clamp(0.0, 1.0));
        }
        out
    }

    /// Recebe bias top-down do FrontalLobe.
    /// `frontal_rates`: firing rates do frontal mapeados para os canais de atenção.
    pub fn set_topdown(&mut self, frontal_rates: &[f32]) {
        let n = self.n_channels.min(frontal_rates.len());
        // Normaliza para [0, 1]
        let max_f = frontal_rates[0..n].iter().cloned().fold(0.001f32, f32::max);
        for i in 0..n {
            self.topdown_bias[i] = frontal_rates[i] / max_f;
        }
    }

    /// Redimensiona para um número diferente de canais (ex: ao mudar resolução visual).
    pub fn resize(&mut self, n_new: usize) {
        self.baseline.resize(n_new, 0.1);
        self.gains.resize(n_new, 1.0);
        self.topdown_bias.resize(n_new, 0.0);
        self.n_channels = n_new;
    }
}
