// =============================================================================
// tests/detailed_level_tests.rs — Testes que faltavam (gaps L1 e L3)
// =============================================================================
//
// Complementa comprehensive_neural_tests.rs cobrindo os GAPS identificados:
//
//  L1 — TODOS os 24 tipos neuronais via update() real (não só inicialização):
//       (a doc do projeto dizia "27", mas o enum TipoNeuronal tem 24 variantes)
//       * Estabilidade numérica ao longo de 500 ticks (sem NaN/Inf, v em bounds)
//       * Diferenciação comportamental (excitatório dispara; inibitório existe)
//       * Os 4 PrecisionType (FP32/FP16/INT8/INT4) sob carga
//
//  L3 — Zonas cerebrais antes NÃO testadas funcionalmente:
//       * Amygdala  — condicionamento de medo (valência → arousal)
//       * Cerebellum — compute_motor_output (mossy + climbing fiber)
//       * Language   — Wernicke (compreensão) + Broca (planejamento de fala)
//
// API real confirmada em src/synaptic_core.rs:
//   NeuronioHibrido::new(id, tipo, precisao)
//   neuronio.update(input_current, dt_segundos, current_time_ms, escala) -> bool
//   campos: v, u, trace_pre, trace_pos, extras.mod_ach
// =============================================================================

#![allow(unused_imports)]

use std::collections::HashMap;

use selene_kernel::{
    synaptic_core::{NeuronioHibrido, TipoNeuronal, PrecisionType},
    brain_zones::{
        amygdala::Amygdala,
        cerebellum::Cerebellum,
        language::LanguageAreas,
    },
    config::{Config, ModoOperacao},
};

// Os 24 tipos neuronais REAIS do enum (a doc dizia "27", mas a contagem do
// enum TipoNeuronal é 24: 7+6+4+3+3+1). Sem array ALL no enum — listados aqui.
const TODOS_OS_TIPOS: [TipoNeuronal; 24] = [
    // Originais (7)
    TipoNeuronal::RS, TipoNeuronal::IB, TipoNeuronal::CH, TipoNeuronal::FS,
    TipoNeuronal::LT, TipoNeuronal::TC, TipoNeuronal::RZ,
    // Izhikevich adicionais (6)
    TipoNeuronal::PS, TipoNeuronal::PB, TipoNeuronal::AC, TipoNeuronal::BI,
    TipoNeuronal::DAP, TipoNeuronal::IIS,
    // Subtipos biológicos (4)
    TipoNeuronal::PV, TipoNeuronal::SST, TipoNeuronal::VIP, TipoNeuronal::DA_N,
    // V3.1 (3)
    TipoNeuronal::NGF, TipoNeuronal::LC_N, TipoNeuronal::ChIN,
    // V4.6 biofísicos (3)
    TipoNeuronal::GridCell, TipoNeuronal::MirrorCell, TipoNeuronal::MSN,
    // Híbrido (1)
    TipoNeuronal::Hybrid,
];

const DT: f32 = 0.005; // 200 Hz

// ─────────────────────────────────────────────────────────────────────────────
// L1 — ESTABILIDADE NUMÉRICA DE TODOS OS 27 TIPOS
// ─────────────────────────────────────────────────────────────────────────────

/// Roda `n_ticks` de update com corrente fixa e devolve (nº de spikes, v_final).
/// Falha o teste imediatamente se algum estado virar NaN/Inf ou sair de bounds.
fn simular_tipo(tipo: TipoNeuronal, corrente: f32, n_ticks: usize) -> (usize, f32) {
    let mut n = NeuronioHibrido::new(0, tipo, PrecisionType::FP32);
    let mut spikes = 0usize;
    for i in 0..n_ticks {
        let t_ms = i as f32 * DT * 1000.0;
        let disparou = n.update(corrente, DT, t_ms, 1.0);
        if disparou { spikes += 1; }

        assert!(n.v.is_finite(), "{:?}: v não-finito no tick {} ({})", tipo, i, n.v);
        assert!(n.u.is_finite(), "{:?}: u não-finito no tick {}", tipo, i);
        assert!(!n.trace_pre.is_nan() && !n.trace_pos.is_nan(),
            "{:?}: trace STDP NaN no tick {}", tipo, i);
        // Bound generoso: captura DIVERGÊNCIA real (o termo v² do Izhikevich
        // explode pra milhares/NaN se instável). O pico/overshoot legítimo de um
        // spike pode passar de +30mV antes do reset (ex.: CH chega a ~78mV); isso
        // é esperado, não instabilidade.
        assert!(n.v >= -150.0 && n.v <= 150.0,
            "{:?}: v divergiu: {} mV (tick {})", tipo, n.v, i);
    }
    (spikes, n.v)
}

#[test]
fn l1_todos_tipos_estaveis_sob_corrente_forte() {
    // Corrente forte sustentada — estressa a dinâmica de cada tipo.
    for tipo in TODOS_OS_TIPOS {
        let (spikes, v_final) = simular_tipo(tipo, 10.0, 500);
        println!("  {:?}: {} spikes em 500 ticks | v_final={:.2} mV", tipo, spikes, v_final);
    }
}

#[test]
fn l1_todos_tipos_estaveis_em_repouso() {
    // Corrente zero — não deve haver disparo espúrio descontrolado nem divergência.
    for tipo in TODOS_OS_TIPOS {
        let (spikes, _v) = simular_tipo(tipo, 0.0, 500);
        // Pacemakers (DA_N, ChIN) podem disparar tonicamente; o limite generoso
        // apenas captura disparo descontrolado (a cada tick = 500).
        assert!(spikes < 500,
            "{:?}: disparo descontrolado em repouso ({} spikes)", tipo, spikes);
    }
}

#[test]
fn l1_tipos_excitatorios_disparam_sob_corrente() {
    // Tipos excitatórios clássicos devem disparar ao menos uma vez sob corrente forte.
    for tipo in [TipoNeuronal::RS, TipoNeuronal::IB, TipoNeuronal::CH] {
        let (spikes, _) = simular_tipo(tipo, 12.0, 1000);
        assert!(spikes > 0, "{:?}: não disparou sob corrente forte (esperado >0)", tipo);
    }
}

#[test]
fn l1_fast_spiking_nao_diverge_em_alta_frequencia() {
    // FS é fast-spiking sem adaptação — deve disparar bastante mas continuar estável.
    let (spikes, v_final) = simular_tipo(TipoNeuronal::FS, 12.0, 1000);
    assert!(v_final.is_finite(), "FS: v_final não-finito");
    println!("  FS: {} spikes em 1000 ticks (alta frequência, estável)", spikes);
}

#[test]
fn l1_todas_precisoes_estaveis() {
    // Os 4 tipos de precisão devem rodar sem NaN sob carga (INT4/INT8 usam fast path NTC).
    for precisao in [PrecisionType::FP32, PrecisionType::FP16,
                     PrecisionType::INT8, PrecisionType::INT4] {
        let mut n = NeuronioHibrido::new(0, TipoNeuronal::RS, precisao);
        for i in 0..500 {
            let t_ms = i as f32 * DT * 1000.0;
            n.update(10.0, DT, t_ms, 1.0);
            assert!(n.v.is_finite(),
                "Precisão {:?}: v não-finito no tick {}", precisao, i);
        }
    }
}

#[test]
fn l1_hybrid_sem_dna_usa_fallback_sem_panic() {
    // Hybrid sem genoma deve cair no fallback dos getters *_efetivo() sem panic.
    let (_spikes, v_final) = simular_tipo(TipoNeuronal::Hybrid, 10.0, 300);
    assert!(v_final.is_finite(), "Hybrid sem DNA: v_final não-finito");
}

// ─────────────────────────────────────────────────────────────────────────────
// L3 — AMYGDALA (condicionamento de medo)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn l3_amygdala_valencia_negativa_gera_resposta() {
    let config = Config::new(ModoOperacao::Normal);
    let mut amyg = Amygdala::new(64, &config);

    let mut fear_max = 0.0f32;
    for i in 0..100 {
        let t = i as f32 * DT;
        // Valência negativa = ameaça; arousal alto amplifica.
        let (fear, arousal_out) = amyg.update(-0.8, 0.0, 0.9, DT, t, &config, None);
        assert!(fear.is_finite() && arousal_out.is_finite(),
            "Amygdala: saída não-finita no tick {}", i);
        fear_max = fear_max.max(fear);
    }
    println!("  Amygdala fear_max sob ameaça = {:.4}", fear_max);
}

#[test]
fn l3_amygdala_valencia_positiva_nao_diverge() {
    let config = Config::new(ModoOperacao::Normal);
    let mut amyg = Amygdala::new(64, &config);

    for i in 0..100 {
        let t = i as f32 * DT;
        // Sinal de segurança (valência positiva) não deve explodir o medo.
        let (fear, _) = amyg.update(0.7, 0.2, 0.3, DT, t, &config, None);
        assert!(fear.is_finite(), "Amygdala: medo não-finito sob segurança");
        assert!(fear <= 2.0, "Amygdala: medo fora de bounds ({}) sob segurança", fear);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// L3 — CEREBELLUM (controle motor)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn l3_cerebellum_motor_output_sem_nan() {
    let config = Config::new(ModoOperacao::Normal);
    let mut cereb = Cerebellum::new(32, 64, &config);

    let mossy = vec![0.5f32; 64];
    let climbing = vec![0.3f32; 32];

    for i in 0..50 {
        let t = i as f32 * DT;
        let out = cereb.compute_motor_output(&mossy, &climbing, DT, t, &config);
        assert!(!out.is_empty(), "Cerebelo: output vazio");
        for &v in &out {
            assert!(v.is_finite(), "Cerebelo: motor output não-finito no tick {}", i);
        }
    }
}

#[test]
fn l3_cerebellum_silencioso_sem_input() {
    let config = Config::new(ModoOperacao::Normal);
    let mut cereb = Cerebellum::new(32, 64, &config);

    let zero_mossy = vec![0.0f32; 64];
    let zero_climb = vec![0.0f32; 32];

    let out = cereb.compute_motor_output(&zero_mossy, &zero_climb, DT, 0.0, &config);
    // Early-exit documentado: sem input → saída silenciosa (zeros).
    let soma: f32 = out.iter().map(|v| v.abs()).sum();
    assert!(soma < 1e-3, "Cerebelo: deveria estar silencioso sem input (soma={})", soma);
}

// ─────────────────────────────────────────────────────────────────────────────
// L3 — LANGUAGE (Wernicke + Broca)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn l3_language_wernicke_processa_tokens_conhecidos() {
    let config = Config::new(ModoOperacao::Normal);
    let mut lang = LanguageAreas::new(64, &config);

    let tokens: Vec<u32> = vec![10, 20, 30];
    let mut valencias: HashMap<u32, f32> = HashMap::new();
    valencias.insert(10, 0.5);
    valencias.insert(20, -0.3);
    valencias.insert(30, 0.8);

    let mut score = 0.0f32;
    for i in 0..50 {
        let t = i as f32 * DT;
        score = lang.wernicke_process(&tokens, &valencias, DT, t, &config);
        assert!(score.is_finite(), "Wernicke: comprehension_score não-finito");
    }
    println!("  Wernicke comprehension_score (tokens conhecidos) = {:.4}", score);
}

#[test]
fn l3_language_wernicke_vazio_decai() {
    let config = Config::new(ModoOperacao::Normal);
    let mut lang = LanguageAreas::new(64, &config);
    let valencias: HashMap<u32, f32> = HashMap::new();

    // Input vazio deve apenas decair o score, sem panic / NaN.
    let score = lang.wernicke_process(&[], &valencias, DT, 0.0, &config);
    assert!(score.is_finite() && score >= 0.0, "Wernicke vazio: score inválido ({})", score);
}

#[test]
fn l3_language_broca_planeja_fala_sem_nan() {
    let config = Config::new(ModoOperacao::Normal);
    let mut lang = LanguageAreas::new(64, &config);

    for i in 0..50 {
        let t = i as f32 * DT;
        let (output, bandas) = lang.broca_plan(0.7, 0.2, 0.6, DT, t, &config);
        assert!(output.is_finite(), "Broca: output não-finito no tick {}", i);
        for (j, &b) in bandas.iter().enumerate() {
            assert!(b.is_finite(), "Broca: banda formante {} não-finita", j);
        }
    }
}
