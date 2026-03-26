// src/brain_zones/temporal.rs
// Córtex Temporal — Reconhecimento auditivo, linguagem, memória semântica
//
// Composição neuronal:
//   recognition_layer: 55% RS + 30% CH + 15% FS
//   CH para reconhecimento rápido de padrões fonéticos repetitivos.
//   FS (15%) para oscilações gamma auditivas e inibição lateral entre padrões concorrentes.

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::synaptic_core::{CamadaHibrida, PrecisionType, TipoNeuronal};
use crate::brain_zones::depth_stack::DepthStack;
use rand::{Rng, thread_rng};
use crate::config::Config;

/// Número máximo de conexões Hebbianas por neurônio (esparsidade online).
/// Cada neurônio rastreia até K co-ativações recentes — as mais fortes.
const HEBBIAN_K: usize = 8;
/// Taxa de decaimento do traço Hebbiano por tick.
const HEBBIAN_TRACE_DECAY: f32 = 0.92;
/// Força do update Hebbiano por co-ativação.
const HEBBIAN_LR: f32 = 0.015;

pub struct TemporalLobe {
    pub recognition_layer: CamadaHibrida,
    pub auditory_buffer: Vec<f32>,
    pub semantic_memory: Vec<f32>,
    pub novelty_detection: f32,
    pub habituation_counter: Vec<u32>,
    pub learning_rate: f32,
    pub noise_std_base: f32,
    /// Traço de elegibilidade Hebbiana por neurônio (decai entre ticks).
    /// Neurônios que dispararam recentemente têm traço alto — prontos para potenciação.
    pub hebbian_traces: Vec<f32>,
    /// Pesos Hebbianos esparsos: hebbian_weights[i] = lista de (j, peso).
    /// i→j significa "quando i dispara após j disparar, reforça input de j para i".
    /// Implementa aprendizado online dentro do lóbulo temporal a cada tick.
    hebbian_weights: Vec<Vec<(usize, f32)>>,
    /// Pilha de profundidade — 3 círculos de compressão (chama branca).
    /// D0 = output bruto do lóbulo, D1 = comprimido (n/2), D2 = super-comprimido (n/4).
    /// A saída multi-profundidade enriquece o sinal enviado ao frontal e hipocampo.
    pub depth_stack: DepthStack,
}

impl TemporalLobe {
    pub fn new(n_neurons: usize, learning_rate: f32, noise_std_base: f32, config: &Config) -> Self {
        // Temporal: FP16 dominante para reconhecimento de padrões preciso
        // 30% CH para reconhecimento rápido de padrões repetitivos (palavras, ritmos)
        let dist = vec![
            (PrecisionType::FP32, 0.05),
            (PrecisionType::FP16, 0.55),
            (PrecisionType::INT8, 0.40),
        ];

        let escala = 45.0 / 127.0;

        let mut recognition = CamadaHibrida::new(
            n_neurons, "temporal_recog",
            TipoNeuronal::RS,
            Some((TipoNeuronal::CH, 0.30)),
            Some(dist),
            escala,
        );
        // Converte os últimos 15% (metade dos CH) para FS:
        // interneurônios parvalbumin+ que dirigem oscilações gamma auditivas
        // e implementam inibição lateral entre padrões fonéticos concorrentes.
        let n_temp = recognition.neuronios.len();
        let fs_start = (n_temp as f32 * 0.85) as usize;
        for nr in &mut recognition.neuronios[fs_start..] {
            nr.tipo = TipoNeuronal::FS;
            nr.threshold = 25.0; // FS dispara com limiar menor
        }
        recognition.init_lateral_inhibition(4, 2.5); // 4 vizinhos, força moderada

        Self {
            recognition_layer: recognition,
            auditory_buffer: vec![0.0; n_neurons],
            semantic_memory: vec![0.0; n_neurons],
            novelty_detection: 1.0,
            habituation_counter: vec![0; n_neurons],
            learning_rate,
            noise_std_base,
            hebbian_traces: vec![0.0; n_neurons],
            hebbian_weights: vec![Vec::new(); n_neurons],
            depth_stack: DepthStack::new(n_neurons),
        }
    }

    /// Atualiza traços e pesos Hebbianos com base nos spikes do tick atual.
    ///
    /// Regra: se i dispara E j tinha traço alto (disparou recentemente),
    /// fortalece a conexão j→i (j previu i). Isso é STDP simplificado online.
    /// Complexidade: O(n × K) por tick — não cresce com o número de sinapses.
    pub fn hebbian_update(&mut self, spikes: &[bool]) {
        let n = self.hebbian_traces.len();
        // 1. Decaimento dos traços
        for t in &mut self.hebbian_traces {
            *t *= HEBBIAN_TRACE_DECAY;
        }
        // 2. Para cada neurônio que disparou: potencia com seus pré-sinápticos ativos
        for i in 0..n.min(spikes.len()) {
            if !spikes[i] { continue; }
            // Procura neurônios j≠i com traço alto (j disparou recentemente)
            // Amostra janela local para eficiência: ±16 neurônios vizinhos
            let start = i.saturating_sub(16);
            let end = (i + 16).min(n);
            for j in start..end {
                if j == i { continue; }
                let trace_j = self.hebbian_traces[j];
                if trace_j < 0.1 { continue; }
                // Potencia j→i
                let conns = &mut self.hebbian_weights[i];
                if let Some(p) = conns.iter_mut().find(|(jj, _)| *jj == j) {
                    p.1 = (p.1 + HEBBIAN_LR * trace_j).min(0.5);
                } else if conns.len() < HEBBIAN_K {
                    conns.push((j, HEBBIAN_LR * trace_j));
                } else {
                    // Substitui a conexão mais fraca se a nova for mais forte
                    let (min_idx, min_w) = conns.iter().enumerate()
                        .min_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap())
                        .map(|(i, &(_, w))| (i, w))
                        .unwrap_or((0, 1.0));
                    if HEBBIAN_LR * trace_j > min_w {
                        conns[min_idx] = (j, HEBBIAN_LR * trace_j);
                    }
                }
            }
            // Bump do traço do neurônio que disparou
            if i < self.hebbian_traces.len() {
                self.hebbian_traces[i] = (self.hebbian_traces[i] + 1.0).min(2.0);
            }
        }
    }

    /// Retorna a corrente Hebbiana adicional para o neurônio i.
    /// Neurônios fortemente associados aos pré-sinápticos ativos recebem boost.
    fn hebbian_input(&self, i: usize) -> f32 {
        let conns = &self.hebbian_weights[i];
        if conns.is_empty() { return 0.0; }
        conns.iter()
            .map(|&(j, w)| self.hebbian_traces.get(j).copied().unwrap_or(0.0) * w)
            .sum::<f32>()
            .clamp(0.0, 0.3)
    }

    pub fn process(
        &mut self,
        stimulus_in: &[f32],
        context_bias: &[f32],
        dt: f32,
        current_time: f32,
        config: &Config,
    ) -> Vec<f32> {
        let n = self.recognition_layer.neuronios.len();
        let mut rng = thread_rng();
        let mut combined_input = vec![0.0; n];
        let t_ms = current_time * 1000.0;

        for i in 0..n {
            let s = stimulus_in.get(i).copied().unwrap_or(0.0);
            let recurrence = self.auditory_buffer[i] * 0.75;
            let semantic_match = (s - self.semantic_memory[i]).abs();
            let hab_factor = if self.habituation_counter[i] > 20 { 0.5 } else { 1.0 };
            let surprise = (1.0 + semantic_match * self.novelty_detection) * hab_factor;
            let noise = rng.gen_range(-self.noise_std_base..self.noise_std_base) * surprise;
            // Hebbiano: boost para neurônios co-ativos com pré-sinápticos recentes.
            // Aprende padrões temporais dentro do lóbulo a cada tick (não só via RPE).
            let hebb = self.hebbian_input(i);

            combined_input[i] = (s + recurrence + context_bias.get(i).copied().unwrap_or(0.0) + noise + hebb) * surprise;
        }

        let spikes = self.recognition_layer.update(&combined_input, dt, t_ms);
        // Atualiza pesos Hebbianos com os spikes deste tick — aprendizado online imediato.
        self.hebbian_update(&spikes);

        let mut output = vec![0.0; n];
        for i in 0..n {
            if spikes[i] {
                self.auditory_buffer[i] = 1.5;
                let s = stimulus_in.get(i).copied().unwrap_or(0.0);
                self.semantic_memory[i] = self.semantic_memory[i] * (1.0 - self.learning_rate)
                    + s * self.learning_rate;
                output[i] = 1.0;
                self.habituation_counter[i] = 0;
            } else {
                self.auditory_buffer[i] *= 0.88;
                output[i] = self.auditory_buffer[i];
                let s = stimulus_in.get(i).copied().unwrap_or(0.0);
                if s > 0.1 {
                    self.habituation_counter[i] = self.habituation_counter[i].saturating_add(1);
                }
            }
        }

        // Normalização para range 0-1
        let max_val = output.iter().copied().fold(0.0f32, f32::max);
        if max_val > 0.0 {
            output.iter_mut().for_each(|v| *v /= max_val);
        }

        // Aplica pilha de profundidade — enriquece saída com representações multi-nível.
        // D1 captura padrões de médio prazo, D2 captura estrutura abstrata estável.
        // A combinação ponderada (attn) reflete o estado de abstração atual do sistema.
        self.depth_stack.forward(&output)
    }

    /// Propaga RPE para atualizar pesos de atenção da DepthStack.
    /// Chamado de main.rs após o cálculo de RPE do RL.
    pub fn apply_rpe(&mut self, rpe: f32) {
        self.depth_stack.update_attention(rpe);
    }

    pub fn set_novelty_sensitivity(&mut self, level: f32) {
        self.novelty_detection = level.clamp(0.1, 5.0);
    }

    pub fn estatisticas(&self) -> TemporalStats {
        TemporalStats { recognition: self.recognition_layer.estatisticas() }
    }
}

pub struct TemporalStats {
    pub recognition: crate::synaptic_core::CamadaStats,
}
