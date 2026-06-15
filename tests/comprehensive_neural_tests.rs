// =============================================================================
// tests/comprehensive_neural_tests.rs — Testes Neurais Simplificados
// =============================================================================

#![allow(unused_imports)]

use selene_kernel::{
    synaptic_core::{NeuronioHibrido, TipoNeuronal, PrecisionType},
    brain_zones::{
        occipital::OccipitalLobe,
        parietal::ParietalLobe,
        temporal::TemporalLobe,
        frontal::FrontalLobe,
        limbic::LimbicSystem,
        hippocampus::HippocampusV2 as Hippocampus,
        cerebellum::Cerebellum,
    },
    config::{Config, ModoOperacao},
};

const N: usize = 64;
const DT: f32 = 0.005;

// ─────────────────────────────────────────────────────────────────────────────
// L1: Validação de Tipos Neuronais
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn l1_neuron_type_rs_init() {
    let n = NeuronioHibrido::new(1, TipoNeuronal::RS, PrecisionType::FP32);
    assert!(!n.v.is_nan());
    assert!(!n.u.is_nan());
}

#[test]
fn l1_neuron_type_fs_init() {
    let n = NeuronioHibrido::new(2, TipoNeuronal::FS, PrecisionType::FP32);
    assert!(!n.v.is_nan());
}

#[test]
fn l1_neuron_type_tc_init() {
    let n = NeuronioHibrido::new(3, TipoNeuronal::TC, PrecisionType::FP32);
    assert!(!n.v.is_nan());
}

#[test]
fn l1_neuron_type_pv_init() {
    let n = NeuronioHibrido::new(4, TipoNeuronal::PV, PrecisionType::FP32);
    assert!(!n.v.is_nan());
}

#[test]
fn l1_neuron_type_sst_init() {
    let n = NeuronioHibrido::new(5, TipoNeuronal::SST, PrecisionType::FP32);
    assert!(!n.v.is_nan());
}

#[test]
fn l1_neuron_type_vip_init() {
    let n = NeuronioHibrido::new(6, TipoNeuronal::VIP, PrecisionType::FP32);
    assert!(!n.v.is_nan());
}

#[test]
fn l1_precision_types_all_valid() {
    let types = [
        PrecisionType::FP32,
        PrecisionType::FP16,
        PrecisionType::INT8,
        PrecisionType::INT4,
    ];

    for (i, precision) in types.iter().enumerate() {
        let n = NeuronioHibrido::new(i as u32, TipoNeuronal::RS, *precision);
        assert!(!n.v.is_nan(), "Precision {:?} iniciou com NaN", precision);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// L3: Testes de Zonas Cerebrais Isoladas
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn l3_occipital_visual_sweep_bounds() {
    let config = Config::new(ModoOperacao::Normal);
    let mut occipital = OccipitalLobe::new(N, 0.2, &config);

    let visual_input = vec![0.3f32; N];
    let features = occipital.visual_sweep(&visual_input, DT, None, 0.0, &config);

    assert!(!features.is_empty());
    for &f in &features {
        assert!(!f.is_nan() && !f.is_infinite());
        assert!(f >= 0.0 && f <= 1000.0);
    }
}

#[test]
fn l3_parietal_integration_no_nan() {
    let config = Config::new(ModoOperacao::Normal);
    let mut parietal = ParietalLobe::new(N, 0.2, &config);

    let visual = vec![0.4f32; N];
    let proprioception = vec![0.2f32; N];

    let output = parietal.integrate(&visual, &proprioception, DT, 0.0, &config);

    assert_eq!(output.len(), N);
    for &v in &output {
        assert!(!v.is_nan());
    }
}

#[test]
fn l3_temporal_lobe_creation() {
    let config = Config::new(ModoOperacao::Normal);
    let _temporal = TemporalLobe::new(N, DT, 0.2, &config);
    // Simples teste de inicialização sem crash
}

#[test]
fn l3_frontal_lobe_creation() {
    let config = Config::new(ModoOperacao::Normal);
    let _frontal = FrontalLobe::new(N, 0.2, 0.1, &config);
    // Teste de inicialização
}

#[test]
fn l3_limbic_system_creation() {
    let config = Config::new(ModoOperacao::Normal);
    let _limbic = LimbicSystem::new(N / 2, &config);
    // Teste de inicialização
}

#[test]
fn l3_hippocampus_creation() {
    let config = Config::new(ModoOperacao::Normal);
    let _hippo = Hippocampus::new(N / 2, &config);
    // Teste de inicialização
}

#[test]
fn l3_cerebellum_creation() {
    let config = Config::new(ModoOperacao::Normal);
    let _cerebelo = Cerebellum::new(N / 4, N / 2, &config);
    // Teste de inicialização
}

// ─────────────────────────────────────────────────────────────────────────────
// L4: Nota sobre Hemisférios
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn l4_hemispheres_architecture_design() {
    // NOTA: Selene v4.6.1 implementa single-brain (não L/R explícitos)
    // Este teste documenta que não há testes de hemisférios porque
    // a arquitetura não os implementa estruturalmente.
    // Lateralização é emergente via aprendizado.
    let config = Config::new(ModoOperacao::Normal);
    let _f = FrontalLobe::new(N, 0.2, 0.1, &config);
    // Single instance — sem duplicação L/R
}

// ─────────────────────────────────────────────────────────────────────────────
// L5: Integração Completa (Pipeline Simples)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn l5_full_pipeline_stability() {
    let config = Config::new(ModoOperacao::Normal);
    let mut occ = OccipitalLobe::new(N, 0.2, &config);
    let mut par = ParietalLobe::new(N, 0.2, &config);

    let mut had_nan = false;
    for _ in 0..20 {
        let input = vec![0.3f32; N];
        let _features = occ.visual_sweep(&input, DT, Some(&par.spatial_map), 0.0, &config);
        let _spatial = par.integrate(&input, &vec![0.0f32; N], DT, 0.0, &config);

        if _spatial.iter().any(|v| v.is_nan()) {
            had_nan = true;
        }
    }

    assert!(!had_nan, "Pipeline contém NaN após 20 ticks");
}

// ─────────────────────────────────────────────────────────────────────────────
// Testes de Segurança (Security-Critical Paths)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn security_neuron_voltage_bounds() {
    for _ in 0..10 {
        let n = NeuronioHibrido::new(1, TipoNeuronal::RS, PrecisionType::FP32);
        assert!(!n.v.is_nan(), "Voltagem não deve ser NaN");
        assert!(n.v >= -100.0 && n.v <= 50.0, "Voltagem fora de bounds biológicos");
    }
}

#[test]
fn security_floating_point_not_infinite() {
    let config = Config::new(ModoOperacao::Normal);
    let occ = OccipitalLobe::new(32, 0.2, &config);

    // Verify V1 neurons have finite values
    for neuron in &occ.v1_primary_layer.neuronios {
        assert!(neuron.v.is_finite(), "Neuron voltagem é infinita");
    }
}

#[test]
fn security_no_uninitialized_state() {
    let n = NeuronioHibrido::new(1, TipoNeuronal::RS, PrecisionType::FP32);

    // Verificar que campos críticos não estão em estado indeterminado
    assert!(!n.v.is_nan());
    assert!(!n.u.is_nan());
    assert!(!n.trace_pre.is_nan());
    assert!(!n.trace_pos.is_nan());
}

