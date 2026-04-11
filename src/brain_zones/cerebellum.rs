// src/brain_zones/cerebellum.rs
// Cerebelo — Controle motor, timing, aprendizado de procedimentos
//
// Composição neuronal:
//   purkinje_layer: 80% RS + 20% RZ — células de Purkinje com detecção rítmica
//   granular_layer: 100% RZ — células granulares detectam padrões temporais
//
// RZ (Resonator) é o tipo correto para granulares do cerebelo porque:
// - Detecta frequências específicas de input (propriocepção, timing motor)
// - Biologicamente: células granulares têm ressonância temporal
// - Resultado: aprendizado de sequências motoras e timing preciso

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::synaptic_core::{CamadaHibrida, ModeloDinamico, PrecisionType, TipoNeuronal};
use crate::config::Config;

pub struct Cerebellum {
    pub purkinje_layer: CamadaHibrida,
    pub granular_layer: CamadaHibrida,
    pub error_signal: Vec<f32>,
    pub ltd_factor: Vec<f32>,
}

impl Cerebellum {
    pub fn new(n_purkinje: usize, n_granular: usize, config: &Config) -> Self {
        // Purkinje: INT8 dominante, RZ minoritário para timing
        let purkinje_dist = vec![
            (PrecisionType::FP16, 0.20),
            (PrecisionType::INT8, 0.70),
            (PrecisionType::INT4, 0.10),
        ];
        // Granulares: INT4 para alta densidade, 100% RZ
        let granular_dist = vec![
            (PrecisionType::INT8, 0.70),
            (PrecisionType::INT4, 0.30),
        ];

        let escala_p = 30.0 / 127.0;
        let escala_g = 20.0 / 127.0;

        let mut purkinje = CamadaHibrida::new(
            n_purkinje, "cerebelo_purkinje",
            TipoNeuronal::RS,
            Some((TipoNeuronal::RZ, 0.20)),
            Some(purkinje_dist),
            escala_p,
        );

        // Ajustes biofísicos das células de Purkinje:
        // threshold menor (20mV) para disparos mais frequentes
        // Últimos 15% → FS (células basket/stellate): inibição lateral entre Purkinje
        let n_purk = purkinje.neuronios.len();
        let basket_start = (n_purk as f32 * 0.85) as usize;
        for (i, n) in purkinje.neuronios.iter_mut().enumerate() {
            if i < basket_start {
                n.threshold = 20.0;
            } else {
                n.tipo = TipoNeuronal::FS; // células basket/stellate GABAérgicas
                n.modelo = ModeloDinamico::Izhikevich; // FS não usa HH — corrige invariante tipo/modelo
                n.threshold = 25.0;
            }
        }
        // Basket cells inibem Purkinje vizinhas → timing preciso de output motor
        purkinje.init_lateral_inhibition(3, 4.0);

        let granular = CamadaHibrida::new(
            n_granular, "cerebelo_granular",
            TipoNeuronal::RZ, // 100% RZ — detecção rítmica
            None,
            Some(granular_dist),
            escala_g,
        );

        Self {
            purkinje_layer: purkinje,
            granular_layer: granular,
            error_signal: vec![0.0; n_purkinje],
            ltd_factor: vec![1.0; n_purkinje],
        }
    }

    pub fn compute_motor_output(
        &mut self,
        mossy_fiber_input: &[f32],
        climbing_fiber_error: &[f32],
        dt: f32,
        current_time: f32,
        config: &Config,
    ) -> Vec<f32> {
        let t_ms = current_time * 1000.0;
        let n_g = self.granular_layer.neuronios.len();
        let n_p = self.purkinje_layer.neuronios.len();

        // Fibras musgosas → granulares
        let granular_spikes = self.granular_layer.update(
            &mossy_fiber_input[..mossy_fiber_input.len().min(n_g)],
            dt, t_ms,
        );

        // Granulares → Purkinje (expansão: muitos-para-um)
        let mut purkinje_input = vec![0.0; n_p];
        for (i, &spiked) in granular_spikes.iter().enumerate() {
            if spiked {
                let target = i * n_p / n_g;
                if target < n_p { purkinje_input[target] += 3.0; }
            }
        }

        // Acumula erro via EMA — memória de erros recentes escala a taxa de LTD
        // Sem isso, error_signal fica sempre zero e o cerebelo aprende na mesma taxa
        // independentemente de o erro estar piorando ou melhorando ao longo do tempo.
        let n_err = n_p.min(climbing_fiber_error.len());
        for i in 0..n_err {
            self.error_signal[i] = self.error_signal[i] * 0.95 + climbing_fiber_error[i].abs() * 0.05;
        }

        // LTD cerebelar: taxa adaptativa — erros acumulados aceleram o aprendizado
        for i in 0..n_err {
            let err = climbing_fiber_error[i].abs();
            let lr = 0.01 * (1.0 + self.error_signal[i] * 3.0).min(4.0);
            if err > 0.1 {
                self.ltd_factor[i] = (self.ltd_factor[i] - err * lr).max(0.1);
            } else {
                self.ltd_factor[i] = (self.ltd_factor[i] + 0.001).min(1.0);
            }
            purkinje_input[i] *= self.ltd_factor[i];
        }

        let purkinje_spikes = self.purkinje_layer.update(&purkinje_input, dt, t_ms);

        // Purkinje → output (sinal inibitório para núcleos cerebelares)
        purkinje_spikes.iter()
            .map(|&s| if s { -1.0 } else { 0.0 }) // Purkinje é inibitório
            .collect()
    }

    pub fn estatisticas(&self) -> CerebellumStats {
        CerebellumStats {
            purkinje: self.purkinje_layer.estatisticas(),
            granular: self.granular_layer.estatisticas(),
        }
    }
}

pub struct CerebellumStats {
    pub purkinje: crate::synaptic_core::CamadaStats,
    pub granular: crate::synaptic_core::CamadaStats,
}
