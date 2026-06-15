// =============================================================================
// src/ternary.rs — Ternarização de Peso Efetivo {-1, 0, +1}
// =============================================================================
//
// Conceito (discutido na análise de lateralização/quantização):
//   - PESO LATENTE em ponto flutuante (FP): onde STDP/BCM aprendem, com a
//     gradação contínua que a plasticidade precisa.
//   - PESO EFETIVO ternário {-escala, 0, +escala}: usado na PROPAGAÇÃO.
//
// Por que vale:
//   * Multiplicação vira escolha de sinal + soma (sem multiplicador) → casa com NPU.
//   * O nível "0" dá ESPARSIDADE de graça (conexões zeradas são puladas).
//   * ~1,58 bit por peso efetivo (log2 3).
//
// Por que NÃO ternarizar direto o peso aprendido:
//   * 3 níveis não cabem o aprendizado fino. Mantendo o latente FP, o STDP
//     continua aprendendo de forma contínua; só a propagação é ternarizada.
// =============================================================================

/// Banda morta em torno de zero: |latente| abaixo disto → efetivo 0 (esparsidade).
pub const TERNARY_THRESHOLD_PADRAO: f32 = 0.05;

/// Bounds do peso latente (mesma faixa usada pelas sinapses do core).
const PESO_LATENTE_MIN: f32 = -2.5;
const PESO_LATENTE_MAX: f32 = 2.5;

/// Peso com latente contínuo + efetivo ternário cacheado.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PesoTernarizado {
    latente:   f32, // contínuo — STDP/BCM atualizam aqui
    escala:    f32, // α: magnitude dos níveis ±
    threshold: f32, // banda morta em torno de 0
    efetivo:   f32, // cache: -escala, 0, ou +escala
}

impl PesoTernarizado {
    /// Cria com latente/escala/threshold explícitos e já ternariza o efetivo.
    pub fn novo(latente: f32, escala: f32, threshold: f32) -> Self {
        let mut p = Self {
            latente: latente.clamp(PESO_LATENTE_MIN, PESO_LATENTE_MAX),
            escala: escala.max(0.0),
            threshold: threshold.max(0.0),
            efetivo: 0.0,
        };
        p.reternarizar();
        p
    }

    /// Construtor padrão: escala 1.0, threshold padrão.
    pub fn padrao(latente: f32) -> Self {
        Self::novo(latente, 1.0, TERNARY_THRESHOLD_PADRAO)
    }

    /// Aplica um delta de aprendizado (STDP/BCM) ao LATENTE e re-ternariza.
    /// O latente preserva a gradação fina; o efetivo segue o sinal.
    pub fn aprender(&mut self, delta: f32) {
        self.latente = (self.latente + delta).clamp(PESO_LATENTE_MIN, PESO_LATENTE_MAX);
        self.reternarizar();
    }

    /// Recalcula o efetivo a partir do latente (banda morta → 0).
    #[inline]
    fn reternarizar(&mut self) {
        self.efetivo = if self.latente.abs() < self.threshold {
            0.0
        } else if self.latente > 0.0 {
            self.escala
        } else {
            -self.escala
        };
    }

    /// Peso usado na PROPAGAÇÃO (ternário).
    #[inline] pub fn efetivo(&self) -> f32 { self.efetivo }
    /// Peso latente (precisão total, para inspeção/aprendizado).
    #[inline] pub fn latente(&self) -> f32 { self.latente }
    /// `true` se o efetivo é 0 (conexão silenciada → pode ser pulada).
    #[inline] pub fn e_zero(&self) -> bool { self.efetivo == 0.0 }
}

/// Ternariza um vetor de pesos latentes → efetivos (utilitário sem estado).
pub fn ternarizar_vetor(latentes: &[f32], escala: f32, threshold: f32) -> Vec<f32> {
    latentes.iter().map(|&w| {
        if w.abs() < threshold { 0.0 }
        else if w > 0.0 { escala }
        else { -escala }
    }).collect()
}

/// Fração de zeros (esparsidade) — métrica de eficiência da ternarização.
pub fn esparsidade(efetivos: &[f32]) -> f32 {
    if efetivos.is_empty() { return 0.0; }
    let zeros = efetivos.iter().filter(|&&w| w == 0.0).count();
    zeros as f32 / efetivos.len() as f32
}

/// Produto escalar usando pesos ternários: nenhuma multiplicação real —
/// soma quando efetivo=+α, subtrai quando −α, pula quando 0.
/// Demonstra o ganho de hardware (NPU/INT) da ternarização.
pub fn dot_ternario(entradas: &[f32], pesos: &[PesoTernarizado]) -> f32 {
    let n = entradas.len().min(pesos.len());
    let mut acc = 0.0f32;
    for i in 0..n {
        let w = pesos[i].efetivo();
        if w > 0.0 { acc += entradas[i] * w; }
        else if w < 0.0 { acc -= entradas[i] * (-w); }
        // w == 0.0 → pulado (esparsidade)
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banda_morta_zera_pesos_pequenos() {
        let p = PesoTernarizado::padrao(0.01); // abaixo do threshold 0.05
        assert!(p.e_zero());
        assert_eq!(p.efetivo(), 0.0);
    }

    #[test]
    fn sinal_preservado() {
        assert_eq!(PesoTernarizado::padrao(0.8).efetivo(), 1.0);
        assert_eq!(PesoTernarizado::padrao(-0.8).efetivo(), -1.0);
    }

    #[test]
    fn aprendizado_no_latente_cruza_threshold() {
        let mut p = PesoTernarizado::padrao(0.0);
        assert!(p.e_zero());
        for _ in 0..100 { p.aprender(0.01); } // empurra latente p/ cima
        assert_eq!(p.efetivo(), 1.0, "latente deveria ter cruzado o threshold");
        assert!(p.latente() <= 2.5, "latente não pode estourar o clamp");
    }
}
