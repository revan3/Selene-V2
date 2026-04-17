// src/brain_zones/orbitofrontal.rs
// Córtex Orbitofrontal (OFC) — Valor esperado e Reversal Learning
//
// O OFC mantém um mapa de "valor esperado" por contexto (palavra/conceito) e
// detecta quando esse valor mudou — ou seja, quando algo que antes era bom
// agora é ruim, ou vice-versa (reversal learning).
//
// Biologicamente:
//   - OFC medial: valor positivo esperado, apetitivo
//   - OFC lateral: valor negativo esperado, aversivo — sinaliza quando punição
//     ocorre onde recompensa era esperada → acelera extinção da associação
//
// Composição neuronal:
//   value_layer: 70% RS + 30% LT — LT para resposta a estímulos de baixo valor
//   extinction_layer: 60% RS + 40% FS — FS para supressão ativa de associações antigas

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::synaptic_core::{CamadaHibrida, PrecisionType, TipoNeuronal};
use crate::config::Config;
use std::collections::HashMap;
use rand::{Rng, thread_rng};

/// Máximo de contextos rastreados no mapa de valor esperado.
const MAX_VALUE_MAP: usize = 512;
/// Taxa de aprendizado do valor esperado (EMA).
const VALUE_LR: f32 = 0.08;
/// Taxa de extinção quando reversal é detectado — mais rápida que aprendizado normal.
const EXTINCTION_LR: f32 = 0.25;
/// Limiar de discrepância para detectar reversal (|valor_esperado - outcome| > threshold).
const REVERSAL_THRESHOLD: f32 = 0.4;

pub struct OrbitalFrontal {
    /// Camada de valor — integra histórico de recompensa/punição por contexto.
    pub value_layer: CamadaHibrida,
    /// Camada de extinção — suprime associações que reverteram.
    pub extinction_layer: CamadaHibrida,

    /// Mapa contexto → valor esperado [-1.0, 1.0].
    /// Chave: palavra/conceito em minúsculas.
    /// Valor positivo = contexto associado a recompensa.
    /// Valor negativo = contexto associado a punição.
    value_map: HashMap<String, f32>,

    /// Sinal de reversal ativo [0.0, 1.0].
    /// Alto quando outcome recente contradiz o valor esperado armazenado.
    pub reversal_signal: f32,

    /// Bias de valor: soma ponderada dos valores esperados do contexto atual.
    /// Positivo = contexto prevê recompensa → Selene mais confiante/expansiva.
    /// Negativo = contexto prevê punição → Selene mais cautelosa.
    pub value_bias: f32,

    /// Fator de LTD adicional quando reversal é alto.
    /// Propagado ao swap_manager para acelerar o enfraquecimento de arestas
    /// associadas a contextos que mudaram de valor.
    pub ltd_boost: f32,
}

impl OrbitalFrontal {
    pub fn new(n_neurons: usize, config: &Config) -> Self {
        let val_dist = vec![
            (PrecisionType::FP32, 0.05),
            (PrecisionType::FP16, 0.50),
            (PrecisionType::INT8, 0.45),
        ];
        let ext_dist = vec![
            (PrecisionType::FP16, 0.30),
            (PrecisionType::INT8, 0.55),
            (PrecisionType::INT4, 0.15),
        ];

        let escala = 35.0 / 127.0;
        let n_sub = (n_neurons / 3).max(4);

        let value = CamadaHibrida::new(
            n_sub, "ofc_value",
            TipoNeuronal::RS,
            Some((TipoNeuronal::LT, 0.30)), // LT para resposta a estímulos de baixo valor
            Some(val_dist),
            escala,
        );
        let extinction = CamadaHibrida::new(
            n_sub, "ofc_extinction",
            TipoNeuronal::RS,
            Some((TipoNeuronal::FS, 0.40)), // FS para supressão ativa
            Some(ext_dist),
            escala,
        );

        Self {
            value_layer: value,
            extinction_layer: extinction,
            value_map: HashMap::with_capacity(128),
            reversal_signal: 0.0,
            value_bias: 0.0,
            ltd_boost: 0.0,
        }
    }

    /// Atualiza o mapa de valor e detecta reversals.
    ///
    /// `context`: palavras do contexto atual
    /// `outcome`: RPE do tick atual (> 0 = melhor que previsto, < 0 = pior)
    /// `dt`, `current_time`: para a camada neural
    ///
    /// Retorna `(value_bias, reversal_signal, ltd_boost)`.
    pub fn update(
        &mut self,
        context: &[String],
        outcome: f32,
        dt: f32,
        current_time: f32,
        config: &Config,
    ) -> (f32, f32, f32) {
        let t_ms = current_time * 1000.0;
        let mut rng = thread_rng();

        // 1. Calcula valor esperado para o contexto atual
        let expected: f32 = if context.is_empty() {
            0.0
        } else {
            let sum: f32 = context.iter()
                .filter_map(|w| self.value_map.get(w.as_str()).copied())
                .sum();
            (sum / context.len() as f32).clamp(-1.0, 1.0)
        };

        // 2. Discrepância entre valor esperado e outcome real
        let discrepancy = outcome - expected;

        // 3. Detecta reversal: discrepância acima do threshold e com sinal oposto
        let is_reversal = discrepancy.abs() > REVERSAL_THRESHOLD
            && (expected > 0.1 && outcome < -0.1 || expected < -0.1 && outcome > 0.1);

        // 4. Atualiza mapa de valor para palavras do contexto
        let lr = if is_reversal { EXTINCTION_LR } else { VALUE_LR };
        for word in context {
            // Limita o tamanho do mapa
            if self.value_map.len() >= MAX_VALUE_MAP && !self.value_map.contains_key(word.as_str()) {
                // Remove o menos importante (menor |valor|)
                if let Some(key_to_remove) = self.value_map.iter()
                    .min_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                    .map(|(k, _)| k.clone())
                {
                    self.value_map.remove(&key_to_remove);
                }
            }
            let v = self.value_map.entry(word.clone()).or_insert(0.0);
            *v += (outcome - *v) * lr;
            *v = v.clamp(-1.0, 1.0);
        }

        // 5. Alimenta camada de valor neural
        let n_val = self.value_layer.neuronios.len();
        let val_input: Vec<f32> = (0..n_val)
            .map(|_| expected.abs() * 20.0 + outcome * 10.0 + rng.gen_range(-1.0..1.0))
            .collect();
        self.value_layer.update(&val_input, dt, t_ms);

        // 6. Camada de extinção ativa quando reversal detectado
        let n_ext = self.extinction_layer.neuronios.len();
        let ext_strength = if is_reversal { discrepancy.abs() * 25.0 } else { 0.0 };
        let ext_input: Vec<f32> = (0..n_ext)
            .map(|_| ext_strength + rng.gen_range(-0.5..0.5))
            .collect();
        self.extinction_layer.update(&ext_input, dt, t_ms);

        // 7. Atualiza sinais de saída (EMA)
        let reversal_raw = if is_reversal { discrepancy.abs().clamp(0.0, 1.0) } else { 0.0 };
        self.reversal_signal = self.reversal_signal * 0.80 + reversal_raw * 0.20;
        self.value_bias = self.value_bias * 0.90 + expected * 0.10;
        // LTD boost: só quando reversal ativo — acelera esquecimento de arestas erradas
        self.ltd_boost = if self.reversal_signal > 0.3 {
            (self.reversal_signal - 0.3) * 0.5
        } else {
            self.ltd_boost * 0.95 // decai quando sem reversal
        };

        (self.value_bias, self.reversal_signal, self.ltd_boost)
    }

    /// Retorna o valor esperado para uma palavra específica.
    /// Usado pelo gerar_resposta_emergente para biesar palavras com histórico positivo.
    pub fn expected_value(&self, word: &str) -> f32 {
        self.value_map.get(word).copied().unwrap_or(0.0)
    }

    /// Exporta o mapa de valor como pares (palavra, valor) para palavras com |valor| ≥ min_abs.
    /// Usado pelo swap_manager para enriquecer valências com experiência afetiva real.
    pub fn export_value_pairs(&self, min_abs: f32) -> Vec<(String, f32)> {
        self.value_map.iter()
            .filter(|(_, &v)| v.abs() >= min_abs)
            .map(|(k, &v)| (k.clone(), v))
            .collect()
    }

    /// Número de contextos com valor aprendido (telemetria).
    pub fn n_learned_contexts(&self) -> usize {
        self.value_map.len()
    }

    pub fn estatisticas(&self) -> OFCStats {
        OFCStats {
            value: self.value_layer.estatisticas(),
            extinction: self.extinction_layer.estatisticas(),
            reversal_signal: self.reversal_signal,
            value_bias: self.value_bias,
            n_contexts: self.value_map.len(),
        }
    }
}

pub struct OFCStats {
    pub value: crate::synaptic_core::CamadaStats,
    pub extinction: crate::synaptic_core::CamadaStats,
    pub reversal_signal: f32,
    pub value_bias: f32,
    pub n_contexts: usize,
}
