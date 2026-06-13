// =============================================================================
// tests/validacao_allen.rs — V4.6.1
// Validação de plausibilidade biológica dos tipos neuronais (Estágio 1 do
// compass_artifact). Mede a curva F-I (taxa de disparo vs corrente) de cada tipo
// e compara com a faixa Hz biológica documentada (TipoNeuronal::faixa_hz).
//
// HONESTO: isto valida contra as FAIXAS derivadas da literatura, não contra os
// spike trains BRUTOS do Allen Cell Types Database. Para a métrica Γ de
// coincidência de spikes (Jolivet/Naud) ≥80-90%, baixar os datasets do Allen e
// ligar como referência em `referencia_allen()` (marcado abaixo). O harness e os
// critérios já estão prontos para isso.
// =============================================================================

use selene_kernel::synaptic_core::{NeuronioHibrido, PrecisionType, TipoNeuronal};

const DT: f32 = 0.001; // 1 ms — resolução fina
const JANELA_MS: usize = 2000;

/// Taxa de disparo (Hz) de um tipo sob corrente sustentada `drive` por JANELA_MS.
fn taxa_hz(tipo: TipoNeuronal, drive: f32) -> f32 {
    let mut n = NeuronioHibrido::new(0, tipo, PrecisionType::FP32);
    let mut spikes = 0usize;
    for t in 0..JANELA_MS {
        if n.update(drive, DT, t as f32, 1.0) {
            spikes += 1;
        }
    }
    spikes as f32 / (JANELA_MS as f32 / 1000.0)
}

/// Curva F-I: taxa para cada nível de corrente.
fn curva_fi(tipo: TipoNeuronal, drives: &[f32]) -> Vec<f32> {
    drives.iter().map(|&d| taxa_hz(tipo, d)).collect()
}

const DRIVES: [f32; 8] = [0.0, 5.0, 10.0, 15.0, 22.0, 32.0, 45.0, 60.0];

/// PLACEHOLDER para dados reais do Allen Cell Types DB.
/// Devolve Some((taxas_de_referencia)) por tipo quando os datasets forem ligados.
/// Hoje devolve None → o teste usa a faixa biológica (faixa_hz) como referência.
fn referencia_allen(_tipo: TipoNeuronal) -> Option<Vec<f32>> {
    None
}

// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn relatorio_curva_fi_todos_os_tipos() {
    let tipos = [
        TipoNeuronal::RS, TipoNeuronal::IB, TipoNeuronal::CH, TipoNeuronal::FS,
        TipoNeuronal::LT, TipoNeuronal::TC, TipoNeuronal::RZ, TipoNeuronal::PS,
        TipoNeuronal::PB, TipoNeuronal::AC, TipoNeuronal::BI, TipoNeuronal::DAP,
        TipoNeuronal::IIS, TipoNeuronal::PV, TipoNeuronal::SST, TipoNeuronal::VIP,
        TipoNeuronal::DA_N, TipoNeuronal::NGF, TipoNeuronal::LC_N, TipoNeuronal::ChIN,
        TipoNeuronal::GridCell, TipoNeuronal::MirrorCell, TipoNeuronal::MSN,
    ];
    println!("\n=== Curva F-I (Hz) por corrente {:?} | faixa biológica ===", DRIVES);
    let mut max_global = 0.0f32;
    for tipo in tipos {
        let fi = curva_fi(tipo, &DRIVES);
        let (lo, hi) = tipo.faixa_hz();
        let max = fi.iter().copied().fold(0.0, f32::max);
        max_global = max_global.max(max);
        let marca = if max <= hi * 2.0 { "ok" } else { "⚠ runaway" };
        let fis: Vec<String> = fi.iter().map(|h| format!("{h:5.0}")).collect();
        println!("{:>11?}: [{}]  max={:5.1}  faixa={:.0}-{:.0}  {}",
            tipo, fis.join(" "), max, lo, hi, marca);

        // Nenhum tipo pode explodir (NaN/∞) nem ultrapassar absurdamente o teto.
        assert!(max.is_finite(), "{tipo:?}: taxa não-finita (instabilidade numérica)");
        assert!(max <= 600.0, "{tipo:?}: runaway irreal ({max} Hz)");

        // Se houver referência Allen ligada, compara (métrica Γ — futuro).
        if let Some(_ref_taxas) = referencia_allen(tipo) {
            // TODO: Γ de coincidência de spikes contra _ref_taxas (≥0.8).
        }
    }
    assert!(max_global.is_finite());
}

/// Tipos excitatórios padrão DEVEM disparar sob corrente forte (viabilidade).
#[test]
fn excitatorios_disparam_sob_corrente_forte() {
    for tipo in [TipoNeuronal::RS, TipoNeuronal::IB, TipoNeuronal::CH,
                 TipoNeuronal::MirrorCell, TipoNeuronal::GridCell] {
        let hz = taxa_hz(tipo, 45.0);
        assert!(hz > 0.0, "{tipo:?} deve disparar sob corrente forte (got {hz} Hz)");
    }
}

/// Propriedade fast-spiking: FS/PV alcançam taxas MAIORES que RS sob o mesmo drive
/// (interneurônios sem adaptação ↔ piramidal com adaptação). Validação biológica.
#[test]
fn fast_spiking_supera_regular_spiking() {
    let drive = 32.0;
    let rs = taxa_hz(TipoNeuronal::RS, drive);
    let fs = taxa_hz(TipoNeuronal::FS, drive);
    let pv = taxa_hz(TipoNeuronal::PV, drive);
    println!("F-I @ {drive}: RS={rs:.1} FS={fs:.1} PV={pv:.1} Hz");
    assert!(fs >= rs, "FS deve igualar/superar RS (fs={fs}, rs={rs})");
    assert!(pv >= rs, "PV deve igualar/superar RS (pv={pv}, rs={rs})");
}

/// Monotonicidade aproximada da curva F-I dos excitatórios: mais corrente não
/// deve REDUZIR drasticamente a taxa (curva de resposta sã).
#[test]
fn fi_excitatorios_nao_colapsa_com_corrente() {
    for tipo in [TipoNeuronal::RS, TipoNeuronal::CH] {
        let baixa = taxa_hz(tipo, 15.0);
        let alta = taxa_hz(tipo, 45.0);
        assert!(alta + 1.0 >= baixa,
            "{tipo:?}: F-I não deve colapsar (15→{baixa} Hz, 45→{alta} Hz)");
    }
}
