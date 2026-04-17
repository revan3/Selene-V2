// src/brain_zones/amygdala.rs
// Amígdala — Processamento emocional, condicionamento de medo, modulação do arousal
//
// Estrutura biológica:
//   BLA (basolateral): aprende associações emocionais — RS + IB
//     - Recebe input do tálamo e córtex
//     - Codifica valência e saliência emocional
//     - Base do condicionamento de medo e extinção
//
//   CeA (central): output motor do medo — RS + FS
//     - Projeta para hipotálamo, PAG, LC
//     - Gera respostas autonômicas (arousal, freeze, fight/flight)
//     - FS para inibição lateral (gate de saída)
//
// Integração:
//   - ACC (rACC) inibe BLA via `acc_inhibition` — regulação emocional top-down
//   - Fear signal propaga para neuroquímica via arousal_boost
//   - Extinção durante sono/calma reduz fear_signal gradualmente

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::synaptic_core::{CamadaHibrida, PrecisionType, TipoNeuronal};
use crate::config::Config;
use rand::{Rng, thread_rng};

/// Decay da extinção do medo entre ticks.
const EXTINCTION_DECAY: f32 = 0.995;
/// Taxa de aprendizado de condicionamento (uma experiência negativa intensa fica).
const FEAR_LR: f32 = 0.12;
/// Taxa de extinção ativa (durante calma/sono).
const EXTINCTION_LR: f32 = 0.04;
/// Threshold de fear_signal para gerar arousal_boost.
const FEAR_THRESHOLD: f32 = 0.35;

pub struct Amygdala {
    /// BLA — codificação de valência emocional e associação estímulo-emoção.
    pub bla_layer: CamadaHibrida,
    /// CeA — output motor: arousal, freeze, fight/flight.
    pub cea_layer: CamadaHibrida,

    /// Sinal de medo atual [0.0, 1.0] — integração temporal do BLA.
    pub fear_signal: f32,

    /// Traço de extinção [0.0, 1.0] — inibe fear_signal quando ativo.
    /// Cresce durante calma e sono; decresce com re-exposição ao estímulo.
    pub extinction_trace: f32,

    /// Bias de valência aprendida [-1.0, 1.0].
    /// Negativo = contexto historicamente aversivo; positivo = apetitivo.
    pub valence_bias: f32,

    /// Sinal de arousal gerado pela CeA → modula noradrenalina e cortisol.
    pub arousal_boost: f32,

    /// Nível de condicionamento contextual [0.0, 1.0].
    /// Cresce quando medo ocorre repetidamente no mesmo contexto.
    pub context_conditioning: f32,

    /// Baseline adaptativo de valência — EMA do input de valência.
    expected_valence: f32,
}

impl Amygdala {
    pub fn new(n_neurons: usize, config: &Config) -> Self {
        // BLA: IB dominante para burst de medo intenso, RS para integração
        let bla_dist = vec![
            (PrecisionType::FP32, 0.05),
            (PrecisionType::FP16, 0.55),
            (PrecisionType::INT8, 0.40),
        ];
        // CeA: RS + FS para inibição lateral no output
        let cea_dist = vec![
            (PrecisionType::FP16, 0.40),
            (PrecisionType::INT8, 0.50),
            (PrecisionType::INT4, 0.10),
        ];

        let escala = 36.0 / 127.0;
        let n_sub = (n_neurons / 2).max(4);

        let bla = CamadaHibrida::new(
            n_sub, "amygdala_bla",
            TipoNeuronal::IB,
            Some((TipoNeuronal::RS, 0.50)),
            Some(bla_dist),
            escala,
        );
        let cea = CamadaHibrida::new(
            n_sub, "amygdala_cea",
            TipoNeuronal::RS,
            Some((TipoNeuronal::FS, 0.30)),
            Some(cea_dist),
            escala,
        );

        Self {
            bla_layer: bla,
            cea_layer: cea,
            fear_signal: 0.0,
            extinction_trace: 0.3, // começa com alguma extinção de base
            valence_bias: 0.0,
            arousal_boost: 0.0,
            context_conditioning: 0.0,
            expected_valence: 0.0,
        }
    }

    /// Atualiza a amígdala.
    ///
    /// `input_valence`: valência emocional do input atual (-1.0 a 1.0)
    ///   Negativo = aversivo; positivo = apetitivo
    /// `acc_inhibition`: sinal inibitório do rACC [0.0, 0.6]
    ///   Modulação top-down — reduz fear_signal
    /// `arousal`: nível de arousal atual (para modular entrada BLA)
    ///
    /// Retorna `(fear_signal, arousal_boost)`
    pub fn update(
        &mut self,
        input_valence: f32,
        acc_inhibition: f32,
        arousal: f32,
        dt: f32,
        current_time: f32,
        config: &Config,
    ) -> (f32, f32) {
        let t_ms = current_time * 1000.0;
        let mut rng = thread_rng();

        // 1. Input de ameaça: valência negativa ativa o BLA
        //    Positivo inibe (sinal de segurança)
        let threat_input = (-input_valence).max(0.0); // 0..1
        let safety_input = input_valence.max(0.0);    // 0..1

        // 2. BLA: input escalado pelo arousal (alta excitação amplifica medo)
        let n_bla = self.bla_layer.neuronios.len();
        let bla_strength = threat_input * 28.0 * (1.0 + arousal * 0.3);
        let bla_input: Vec<f32> = (0..n_bla)
            .map(|_| bla_strength + rng.gen_range(-1.5..1.5))
            .collect();
        let bla_spikes = self.bla_layer.update(&bla_input, dt, t_ms);
        let bla_rate = bla_spikes.iter().filter(|&&s| s).count() as f32 / n_bla as f32;

        // 3. Extinção: ACC inibe + sinal de segurança → extinção ativa
        let extinction_drive = acc_inhibition * 0.5 + safety_input * 0.3;
        self.extinction_trace = (self.extinction_trace
            + extinction_drive * EXTINCTION_LR
            - threat_input * 0.02)
            .clamp(0.0, 1.0);
        self.extinction_trace *= EXTINCTION_DECAY;

        // 4. Fear signal: BLA rate - extinção
        let raw_fear = (bla_rate - self.extinction_trace * 0.6).max(0.0);
        self.fear_signal = (self.fear_signal * 0.88 + raw_fear * 0.12).clamp(0.0, 1.0);

        // 5. Condicionamento contextual: medo repetido consolida
        if self.fear_signal > FEAR_THRESHOLD {
            self.context_conditioning = (self.context_conditioning + 0.01).clamp(0.0, 1.0);
        } else {
            self.context_conditioning *= 0.998;
        }

        // 6. CeA: output de arousal — mediado pelo fear_signal
        let n_cea = self.cea_layer.neuronios.len();
        let cea_strength = self.fear_signal * 22.0;
        let cea_input: Vec<f32> = (0..n_cea)
            .map(|_| cea_strength + rng.gen_range(-1.0..1.0))
            .collect();
        let cea_spikes = self.cea_layer.update(&cea_input, dt, t_ms);
        let cea_rate = cea_spikes.iter().filter(|&&s| s).count() as f32 / n_cea as f32;

        // 7. Arousal boost: CeA → hipotálamo/LC → NA + cortisol
        self.arousal_boost = if self.fear_signal > FEAR_THRESHOLD {
            (self.fear_signal - FEAR_THRESHOLD) * 0.4 + cea_rate * 0.2
        } else {
            self.arousal_boost * 0.90 // decai quando não há medo
        };
        self.arousal_boost = self.arousal_boost.clamp(0.0, 0.5);

        // 8. Valence bias: EMA de valências recentes
        self.valence_bias = self.valence_bias * 0.97 + input_valence * 0.03;
        self.expected_valence = self.expected_valence * 0.99 + input_valence * 0.01;

        (self.fear_signal, self.arousal_boost)
    }

    /// Registra um evento aversivo intenso (choque, rejeição severa).
    /// Escala diretamente fear_signal e context_conditioning.
    pub fn registrar_aversao(&mut self, intensidade: f32) {
        self.fear_signal = (self.fear_signal + intensidade * 0.35).clamp(0.0, 1.0);
        self.context_conditioning = (self.context_conditioning + intensidade * 0.15).clamp(0.0, 1.0);
        // Reduz extinção — o medo override a segurança aprendida
        self.extinction_trace *= 1.0 - intensidade * 0.3;
    }

    /// Fase de extinção ativa — chamada durante sono ou períodos de calma prolongada.
    /// O hipocampo + vmPFC consolidam a extinção: o medo não some, mas é suprimido.
    pub fn extinção_durante_sono(&mut self) {
        self.extinction_trace = (self.extinction_trace + 0.05).clamp(0.0, 1.0);
        self.fear_signal *= 0.92;
        self.arousal_boost *= 0.80;
    }

    /// Delta de cortisol gerado pela amígdala.
    /// BLA ativa o eixo HPA → cortisol ↑ quando medo alto.
    pub fn cortisol_drive(&self) -> f32 {
        if self.fear_signal > FEAR_THRESHOLD {
            (self.fear_signal - FEAR_THRESHOLD) * 0.25
        } else {
            0.0
        }
    }

    pub fn estatisticas(&self) -> AmygdalaStats {
        AmygdalaStats {
            bla: self.bla_layer.estatisticas(),
            cea: self.cea_layer.estatisticas(),
            fear_signal: self.fear_signal,
            extinction_trace: self.extinction_trace,
            valence_bias: self.valence_bias,
            arousal_boost: self.arousal_boost,
            context_conditioning: self.context_conditioning,
        }
    }
}

pub struct AmygdalaStats {
    pub bla: crate::synaptic_core::CamadaStats,
    pub cea: crate::synaptic_core::CamadaStats,
    pub fear_signal: f32,
    pub extinction_trace: f32,
    pub valence_bias: f32,
    pub arousal_boost: f32,
    pub context_conditioning: f32,
}
