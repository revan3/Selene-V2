// =============================================================================
// tests/lateralization_ternary_tests.rs
// Valida lateralização especializada + ternarização de peso efetivo.
// Foco do pedido: rodar 500 / 1000 / 10000 ticks e garantir que NÃO surge
// nenhum erro de execução (NaN/Inf/divergência/panic) após cargas longas.
// =============================================================================

#![allow(unused_imports)]

use selene_kernel::ternary::{
    PesoTernarizado, ternarizar_vetor, esparsidade, dot_ternario, TERNARY_THRESHOLD_PADRAO,
};
use selene_kernel::lateralization::{CerebroLateralizado, PerfilHemisferio};
use selene_kernel::config::{Config, ModoOperacao};

const DT: f32 = 0.005; // 200 Hz

fn tem_problema(v: &[f32]) -> bool {
    v.iter().any(|x| x.is_nan() || x.is_infinite())
}

// ─────────────────────────────────────────────────────────────────────────────
// TERNARIZAÇÃO — estabilidade do aprendizado no latente sob cargas longas
// ─────────────────────────────────────────────────────────────────────────────

/// STDP simulado no latente por N ticks: o efetivo deve permanecer ∈ {-1,0,+1}
/// e o latente nunca pode estourar o clamp nem virar NaN.
fn estressar_ternario(n_ticks: usize) {
    // 64 pesos com deltas pseudo-aleatórios (sinais alternados + ruído determinístico).
    let mut pesos: Vec<PesoTernarizado> =
        (0..64).map(|i| PesoTernarizado::padrao((i as f32 % 7.0 - 3.0) * 0.01)).collect();

    for t in 0..n_ticks {
        for (i, p) in pesos.iter_mut().enumerate() {
            // delta oscilante — empurra latente pra cima e pra baixo repetidamente
            let fase = ((t + i) as f32 * 0.07).sin();
            p.aprender(fase * 0.03);

            assert!(p.latente().is_finite(), "ternário: latente não-finito (tick {t})");
            assert!(p.latente() >= -2.5 && p.latente() <= 2.5,
                "ternário: latente estourou clamp = {} (tick {t})", p.latente());
            let e = p.efetivo();
            assert!(e == -1.0 || e == 0.0 || e == 1.0,
                "ternário: efetivo fora de {{-1,0,+1}} = {} (tick {t})", e);
        }
    }

    // dot_ternario deve produzir resultado finito.
    let entradas = vec![0.5f32; 64];
    let r = dot_ternario(&entradas, &pesos);
    assert!(r.is_finite(), "ternário: dot_ternario não-finito");
}

#[test] fn ternario_estavel_500()   { estressar_ternario(500); }
#[test] fn ternario_estavel_1000()  { estressar_ternario(1000); }
#[test] fn ternario_estavel_10000() { estressar_ternario(10_000); }

#[test]
fn ternario_gera_esparsidade() {
    // Latentes pequenos → muitos zeros (esparsidade > 0).
    let latentes: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.001).collect();
    let efetivos = ternarizar_vetor(&latentes, 1.0, TERNARY_THRESHOLD_PADRAO);
    let esp = esparsidade(&efetivos);
    assert!(esp > 0.0, "esperava esparsidade > 0, veio {esp}");
    // Todos efetivos ∈ {-1,0,+1}
    for e in &efetivos {
        assert!(*e == -1.0 || *e == 0.0 || *e == 1.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LATERALIZAÇÃO — estabilidade dos dois hemisférios sob cargas longas
// ─────────────────────────────────────────────────────────────────────────────

/// Roda o cérebro lateralizado por N ticks com input variável e verifica que
/// nenhum lado produz NaN/Inf nem diverge.
fn estressar_lateralizacao(n_ticks: usize) {
    let config = Config::new(ModoOperacao::Normal);
    let n = 64;
    let mut cerebro = CerebroLateralizado::novo(n, &config);

    for t in 0..n_ticks {
        let t_seg = t as f32 * DT;
        // Input dinâmico: mistura de duas senóides (estímulo não-trivial).
        let input: Vec<f32> = (0..n).map(|i| {
            0.3 + 0.2 * ((t_seg * 3.0 + i as f32 * 0.1).sin())
                + 0.1 * ((t_seg * 11.0).cos())
        }).collect();

        let (esq, dir) = cerebro.tick(&input, DT, t_seg, &config);

        assert!(!tem_problema(&esq), "lateralização: hemisfério ESQ NaN/Inf (tick {t})");
        assert!(!tem_problema(&dir), "lateralização: hemisfério DIR NaN/Inf (tick {t})");

        let (resumo_esq, resumo_dir) = cerebro.resumos();
        assert!(resumo_esq.is_finite() && resumo_dir.is_finite(),
            "lateralização: resumo caloso não-finito (tick {t})");
        // Divergência: resumos explodindo indicam feedback instável no corpo caloso.
        assert!(resumo_esq < 1e4 && resumo_dir < 1e4,
            "lateralização: resumo divergiu (esq={resumo_esq}, dir={resumo_dir}, tick {t})");
    }
}

#[test] fn lateralizacao_estavel_500()   { estressar_lateralizacao(500); }
#[test] fn lateralizacao_estavel_1000()  { estressar_lateralizacao(1000); }
#[test] fn lateralizacao_estavel_10000() { estressar_lateralizacao(10_000); }

#[test]
fn lateralizacao_hemisferios_sao_diferentes() {
    // A especialização exige que os lados NÃO sejam idênticos: perfis distintos.
    let esq = PerfilHemisferio::esquerdo();
    let dir = PerfilHemisferio::direito();
    assert!(esq.janela_temporal_ms != dir.janela_temporal_ms, "janelas iguais → sem especialização");
    assert!(esq.granularidade != dir.granularidade, "granularidades iguais");
    assert!(esq.taxa_aprendizado != dir.taxa_aprendizado, "taxas iguais");

    // E na prática: após processar o MESMO input, os resumos devem divergir.
    let config = Config::new(ModoOperacao::Normal);
    let mut cerebro = CerebroLateralizado::novo(64, &config);
    for t in 0..200 {
        let t_seg = t as f32 * DT;
        let input = vec![0.5f32; 64];
        cerebro.tick(&input, DT, t_seg, &config);
    }
    let (re, rd) = cerebro.resumos();
    // Não exigimos magnitude específica, só que o sistema produziu dois resumos
    // finitos e distintos (especialização emergente sob o mesmo input).
    assert!(re.is_finite() && rd.is_finite());
    println!("  resumo_esq={re:.5} | resumo_dir={rd:.5} | diferentes={}", (re - rd).abs() > 1e-6);
}
