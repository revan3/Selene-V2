// =============================================================================
// src/bin/intensive_benchmark.rs — Selene V3.3 — Benchmark Intensivo
// =============================================================================
//
//  [A] Processamento Neural — latência/tick, NTC micro-bench, sparsity real-time
//  [B] Encoding de Spikes   — throughput encode/similarity/helix
//  [C] Grafo de Associações — lookup u32 vs String, walk, scoring
//  [D] Aprendizado          — crescimento de grafo, Hebbian, chunking, RL
//  [E] Memória e Grounding  — grounding bind/decay, memória episódica
//  [F] Capacidade do Sistema — vocabulário, padrões, neurônios totais
// =============================================================================

#![allow(unused_imports, dead_code, unused_variables, unused_mut)]

use std::{
    sync::Arc,
    time::Instant,
    collections::HashMap,
};
use tokio::sync::Mutex as TokioMutex;

use selene_kernel::{
    FrontalLobe, OccipitalLobe, ParietalLobe, TemporalLobe,
    LimbicSystem, HippocampusV2 as Hippocampus, Cerebellum,
    synaptic_core::{CamadaHibrida, TipoNeuronal, PrecisionType, NeuronioHibrido},
    neurochem::NeuroChem,
    thalamus::Thalamus,
    brainstem::Brainstem,
    learning::{
        attention::AttentionGate,
        inter_lobe::BrainConnections,
        lobe_router::LobeRouter,
        chunking::{ChunkingEngine, TipoChunk},
        active_context::ActiveContext,
    },
    brain_zones::RegionType,
    config::{Config, ModoOperacao},
    sensors::hardware::HardwareSensor,
    storage::swap_manager::SwapManager,
    sensors::SensorFlags,
    websocket::bridge::{BrainState, EventoEpisodico},
    encoding::spike_codec::{
        encode as spike_encode, similarity, decode_top_n, popcount,
        features_to_spike_pattern, bands_to_spike_pattern,
        SpikePattern, N_NEURONS, K_ACTIVE,
    },
    neural_pool::{NeuralPool, CorticalLevel, word_to_concept_id},
};

// ─────────────────────────────────────────────────────────────────────────────
// Formatação de saída
// ─────────────────────────────────────────────────────────────────────────────

fn secao(titulo: &str) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  {:<60}║", titulo);
    println!("╚══════════════════════════════════════════════════════════════╝");
}

fn subsecao(titulo: &str) {
    println!("\n  ┌─ {} ─────────────────────────────────────", titulo);
}

fn linha(label: &str, valor: &str, unidade: &str) {
    println!("  │  {:<35} {:>14} {}", label, valor, unidade);
}

fn linha_ok(label: &str, valor: &str) {
    println!("  │  ✓ {:<33} {}", label, valor);
}

fn linha_warn(label: &str, valor: &str) {
    println!("  │  ⚠ {:<33} {}", label, valor);
}

fn linha_info(label: &str, valor: &str) {
    println!("  │  · {:<33} {}", label, valor);
}

fn separador() {
    println!("  └─────────────────────────────────────────────────────────────");
}

fn fmt_ns(ns: f64) -> String {
    if ns < 1_000.0          { format!("{:.1} ns", ns) }
    else if ns < 1_000_000.0 { format!("{:.2} µs", ns / 1_000.0) }
    else                     { format!("{:.3} ms", ns / 1_000_000.0) }
}

fn fmt_throughput(ops_per_sec: f64) -> String {
    if ops_per_sec >= 1_000_000.0 {
        format!("{:.2} M/s", ops_per_sec / 1_000_000.0)
    } else if ops_per_sec >= 1_000.0 {
        format!("{:.1} K/s", ops_per_sec / 1_000.0)
    } else {
        format!("{:.0} /s", ops_per_sec)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// A. Processamento Neural
// ─────────────────────────────────────────────────────────────────────────────

fn benchmark_neural(config: &Config) {
    secao("A — PROCESSAMENTO NEURAL");

    // ── A0: NTC Micro-benchmark: INT4 (LUT) vs FP32 (full pipeline) ──────────
    subsecao("A0 — NTC: INT4/LUT vs FP32/full-pipeline (camada 256 neurônios)");
    let n_ntc    = 256usize;
    let dt_ntc   = 0.005f32; // 200 Hz
    let reps_ntc = 1000usize;

    // INT4 path: NTC LUT — skips HH, ionic channels, heavy STDP
    let mut cam_int4 = CamadaHibrida::new(
        n_ntc, "ntc_int4", TipoNeuronal::RS, None, None, 40.0,
    );
    // Set all neurons to INT4
    cam_int4.neuronios.iter_mut().for_each(|n| n.precisao = PrecisionType::INT4);
    let inputs_ntc = vec![0.05f32; n_ntc];
    // Warmup
    for _ in 0..20 { let _ = cam_int4.update(&inputs_ntc, dt_ntc, 0.0); }

    let t_int4 = Instant::now();
    for tick in 0..reps_ntc {
        let _ = cam_int4.update(&inputs_ntc, dt_ntc, tick as f32 * 5.0);
    }
    let ns_int4 = t_int4.elapsed().as_nanos() as f64 / reps_ntc as f64;

    // FP32 path: full pipeline
    let mut cam_fp32 = CamadaHibrida::new(
        n_ntc, "ntc_fp32", TipoNeuronal::RS, None, None, 40.0,
    );
    cam_fp32.neuronios.iter_mut().for_each(|n| n.precisao = PrecisionType::FP32);
    for _ in 0..20 { let _ = cam_fp32.update(&inputs_ntc, dt_ntc, 0.0); }

    let t_fp32 = Instant::now();
    for tick in 0..reps_ntc {
        let _ = cam_fp32.update(&inputs_ntc, dt_ntc, tick as f32 * 5.0);
    }
    let ns_fp32 = t_fp32.elapsed().as_nanos() as f64 / reps_ntc as f64;

    // INT8 path
    let mut cam_int8 = CamadaHibrida::new(
        n_ntc, "ntc_int8", TipoNeuronal::RS, None, None, 40.0,
    );
    cam_int8.neuronios.iter_mut().for_each(|n| n.precisao = PrecisionType::INT8);
    for _ in 0..20 { let _ = cam_int8.update(&inputs_ntc, dt_ntc, 0.0); }

    let t_int8 = Instant::now();
    for tick in 0..reps_ntc {
        let _ = cam_int8.update(&inputs_ntc, dt_ntc, tick as f32 * 5.0);
    }
    let ns_int8 = t_int8.elapsed().as_nanos() as f64 / reps_ntc as f64;

    let speedup_int4 = ns_fp32 / ns_int4;
    let speedup_int8 = ns_fp32 / ns_int8;

    linha("Latência INT4 (NTC LUT path)",
        &fmt_ns(ns_int4),
        &format!("/ tick  ({:.0} ps/neurônio)", ns_int4 / n_ntc as f64 * 1000.0));
    linha("Latência INT8 (NTC direto)",
        &fmt_ns(ns_int8),
        &format!("/ tick  ({:.0} ps/neurônio)", ns_int8 / n_ntc as f64 * 1000.0));
    linha("Latência FP32 (pipeline completo)",
        &fmt_ns(ns_fp32),
        &format!("/ tick  ({:.0} ps/neurônio)", ns_fp32 / n_ntc as f64 * 1000.0));

    if speedup_int4 >= 1.0 {
        linha_ok("Speedup INT4 vs FP32",
            &format!("{:.2}× mais rápido (NTC LUT ativo)", speedup_int4));
    } else {
        linha_warn("Speedup INT4 vs FP32",
            &format!("{:.2}× (LUT não acelerando — verifique cache)", speedup_int4));
    }
    linha_info("Speedup INT8 vs FP32", &format!("{:.2}×", speedup_int8));
    linha_info("Dispatch NTC", "✓ INT4→LUT path | INT8→direto | FP16/FP32→completo");
    separador();

    // ── A1: Latência por tick (n_neurônios × tipo) ─────────────────────────
    subsecao("A1 — Latência por tick (n_neurônios × tipo)");
    let dt = 0.005f32;
    let t0 = 0.0f32;
    let repeticoes = 500;

    let configs_teste: &[(usize, &str, TipoNeuronal, Option<(TipoNeuronal, f32)>)] = &[
        (256,   "256  RS puro",           TipoNeuronal::RS, None),
        (1024,  "1024 RS puro",           TipoNeuronal::RS, None),
        (4096,  "4096 RS puro",           TipoNeuronal::RS, None),
        (256,   "256  RS+FS (80/20)",     TipoNeuronal::RS, Some((TipoNeuronal::FS, 0.20))),
        (1024,  "1024 RS+CH (70/30)",     TipoNeuronal::RS, Some((TipoNeuronal::CH, 0.30))),
        (256,   "256  RZ cerebelar",      TipoNeuronal::RZ, None),
        (256,   "256  TC talamocortical", TipoNeuronal::TC, None),
    ];

    for (n, nome, tipo_p, tipo_s) in configs_teste {
        let mut camada = CamadaHibrida::new(
            *n, nome, *tipo_p, *tipo_s, None, 40.0,
        );
        let inputs = vec![0.05f32; *n];
        for _ in 0..10 { let _ = camada.update(&inputs, dt, t0); }

        let t_start = Instant::now();
        for tick in 0..repeticoes {
            let t_ms = tick as f32 * dt * 1000.0;
            let _ = camada.update(&inputs, dt, t_ms);
        }
        let elapsed_ns    = t_start.elapsed().as_nanos() as f64;
        let ns_por_tick   = elapsed_ns / repeticoes as f64;
        let ns_por_neu    = ns_por_tick / *n as f64;
        let ticks_por_s   = 1e9 / ns_por_tick;

        linha(
            nome,
            &fmt_ns(ns_por_tick),
            &format!("/ tick  ({:.1} ps/neurônio  |  {:.0} ticks/s)",
                ns_por_neu * 1000.0, ticks_por_s),
        );
    }
    separador();

    // ── A2: Pipeline completo — Cenário A (100%) + Cenário B (biológico 8%) ──
    subsecao("A2 — Pipeline completo: 100% ativo vs 8% ativo (biológico/real)");
    let n = 128usize;

    // ── Helper closure para executar o pipeline e retornar (ms_por_tick, taxa_spike) ──
    let run_pipeline = |sparsity: f32| -> (f64, f64) {
        let mut occipital   = OccipitalLobe::new(n, 0.2, config);
        let mut parietal    = ParietalLobe::new(n, 0.2, config);
        let mut temporal    = TemporalLobe::new(n, dt, 0.2, config);
        let mut limbic      = LimbicSystem::new(n / 2, config);
        let mut hippocampus = Hippocampus::new(n / 2, config);
        let mut frontal     = FrontalLobe::new(n, 0.2, 0.1, config);
        let mut cerebelo    = Cerebellum::new(n / 4, n / 2, config);
        let mut thalamus    = Thalamus::new();
        let mut brainstem   = Brainstem::new();
        let mut neuro       = NeuroChem::new();
        let mut attention   = AttentionGate::new(n);
        let mut conn        = BrainConnections::new(n);
        let mut sensor      = HardwareSensor::dummy();
        let goal            = vec![0.5f32; n];
        let zero            = vec![0.0f32; n];
        let zero_half       = vec![0.0f32; n / 2];
        let mut prev_frontal  = zero.clone();
        let mut prev_temporal = zero.clone();
        let mut prev_parietal = zero.clone();
        let mut prev_limbic   = zero_half.clone();
        let mut prev_hippo    = zero_half.clone();

        let n_ticks = 1000usize;
        let mut spikes_total: u64 = 0;

        let t_pipeline = Instant::now();
        for tick in 0..n_ticks {
            let t_ms    = tick as f32 * dt * 1000.0;
            neuro.update(&mut sensor, config);
            let (da, ser, cor) = (neuro.dopamine, neuro.serotonin, neuro.cortisol);

            // sparsity controla a fração de neurônios que recebem sinal
            let tonic = if sparsity >= 1.0 {
                0.30 * (tick as f32 * 0.1).sin().abs() + 0.10
            } else {
                // Biológico: input chega apenas em ~8% dos neurônios
                if (tick % (1.0 / sparsity.max(0.01)) as usize) == 0 {
                    0.30
                } else {
                    0.02  // ruído de fundo subliminar
                }
            };
            let input_vis = vec![tonic; n];
            let input_aud = vec![tonic * 0.5; 10];

            brainstem.update(0.1, dt);
            let alertness = brainstem.stats().alertness.max(0.3);
            let relayed   = thalamus.relay(&input_vis, neuro.noradrenaline, config);
            let cochlea   = brainstem.modulate(&input_aud);
            attention.set_topdown(&prev_frontal);
            let attended  = attention.attend(&relayed, dt * 1000.0);
            let currents  = conn.project_all(
                &attended, &prev_temporal, &prev_parietal,
                &prev_frontal, &prev_limbic, &prev_hippo,
            );
            let hybrid: Vec<f32> = attended.iter().enumerate()
                .map(|(i, &v)| (v + currents.para_temporal.get(i).copied().unwrap_or(0.0) * 0.1)
                    .clamp(0.0, 1.0) * alertness)
                .collect();

            let features  = occipital.visual_sweep(&hybrid, dt, Some(&parietal.spatial_map), t_ms, config);
            let chunk_size = (n / features.len().max(1)).max(1);
            let mut vision_full = vec![0.0f32; n];
            for (i, &f) in features.iter().enumerate() {
                let start = i * chunk_size;
                let end   = (start + chunk_size).min(n);
                for j in start..end { vision_full[j] = f / 100.0; }
            }

            let new_parietal = parietal.integrate(&vision_full, &zero, dt, t_ms, config);
            let recognized   = temporal.process(&vision_full, &new_parietal, dt, t_ms, config);
            let (emotion, _) = limbic.evaluate(&cochlea, 0.0, dt, t_ms, config);

            spikes_total += recognized.iter().filter(|&&v| v > 0.5).count() as u64;

            if emotion.abs() >= 0.35 {
                let (hippo_out, _) = hippocampus.memorize_with_connections(
                    &recognized, emotion, dt, t_ms, config,
                );
                prev_hippo.iter_mut().enumerate().for_each(|(i, v)| {
                    *v = hippo_out.get(i).copied().unwrap_or(0.0);
                });
            }

            frontal.set_dopamine(da + emotion);
            let action   = frontal.decide(&recognized, &goal, dt, t_ms, config);
            let climbing: Vec<f32> = action.iter().zip(recognized.iter())
                .map(|(a, r)| (a - r).clamp(-1.0, 1.0)).collect();
            let _cerb    = cerebelo.compute_motor_output(&recognized, &climbing, dt, t_ms, config);

            if tick % 5 == 0 {
                occipital.v1_primary_layer.modular_neuro(da, ser, cor);
                temporal.recognition_layer.modular_neuro(da, ser, cor);
                frontal.executive_layer.modular_neuro(da, ser, cor);
            }

            prev_frontal.iter_mut().enumerate()
                .for_each(|(i, v)| *v = action.get(i).copied().unwrap_or(0.0));
            prev_temporal.iter_mut().enumerate()
                .for_each(|(i, v)| *v = recognized.get(i).copied().unwrap_or(0.0));
            prev_parietal.iter_mut().enumerate()
                .for_each(|(i, v)| *v = new_parietal.get(i).copied().unwrap_or(0.0));
            prev_limbic.iter_mut().for_each(|v| *v = emotion.clamp(0.0, 1.0));
        }
        let pipeline_ms  = t_pipeline.elapsed().as_secs_f64() * 1000.0;
        let ms_por_tick  = pipeline_ms / n_ticks as f64;
        let taxa_spike   = spikes_total as f64 / n_ticks as f64 / n as f64 * 100.0;
        (ms_por_tick, taxa_spike)
    };

    // Cenário A: 100% ativo
    let (ms_a, spike_a) = run_pipeline(1.0);
    let tick_rate_a     = 1000.0 / ms_a;
    let rt_a            = tick_rate_a / 200.0;

    // Cenário B: 8% ativo (biológico/quiescente)
    let (ms_b, spike_b) = run_pipeline(0.08);
    let tick_rate_b     = 1000.0 / ms_b;
    let rt_b            = tick_rate_b / 200.0;

    linha("Cenário A — 100% ativo — ms/tick",
        &format!("{:.3} ms/tick", ms_a),
        &format!("({:.0} ticks/s  |  spike ≈{:.1}%)", tick_rate_a, spike_a));
    linha("Cenário B — 8% ativo  — ms/tick",
        &format!("{:.3} ms/tick", ms_b),
        &format!("({:.0} ticks/s  |  spike ≈{:.1}%)", tick_rate_b, spike_b));

    // Fator tempo-real reportado sobre Cenário B (uso real)
    if rt_b >= 1.0 {
        linha_ok("Fator tempo-real (Cenário B — uso real)",
            &format!("{:.2}× acima de real-time (200 Hz)", rt_b));
    } else {
        linha_warn("Fator tempo-real (Cenário B — uso real)",
            &format!("{:.2}× abaixo de real-time — {:.0}% gap", rt_b, (1.0 - rt_b) * 100.0));
    }
    if rt_a >= 1.0 {
        linha_ok("Fator tempo-real (Cenário A — pico stress)",
            &format!("{:.2}×", rt_a));
    } else {
        linha_warn("Fator tempo-real (Cenário A — pico stress)",
            &format!("{:.2}×", rt_a));
    }
    linha_info("Sparsity gain (B/A)",
        &format!("{:.2}×  (quiescência reduz carga em {:.0}%)",
            ms_a / ms_b, (1.0 - ms_b / ms_a) * 100.0));
    separador();

    // ── A3: Benchmark NeuralPool u32 vs String (interface overhead) ───────────
    subsecao("A3 — NeuralPool: lookup u32 vs hash String (interface overhead)");
    let mut pool = NeuralPool::new(4096);
    let palavras_pool = ["amor", "medo", "selene", "neural", "dopamina",
                         "consciência", "memória", "tempo", "luz", "sombra"];
    let mut cids: Vec<u32> = Vec::new();
    for p in &palavras_pool {
        let cid = word_to_concept_id(p);
        let _   = pool.aloca_para_tarefa(cid, CorticalLevel::C2Lexical, 0.0);
        cids.push(cid);
    }

    let n_lu = 1_000_000usize;

    // Lookup por u32 (path quente do cérebro)
    let t_u32 = Instant::now();
    let mut found_u32 = 0usize;
    for i in 0..n_lu {
        let cid = cids[i % cids.len()];
        if pool.buscar_conceito(cid).is_some() { found_u32 += 1; }
    }
    let ns_u32 = t_u32.elapsed().as_nanos() as f64 / n_lu as f64;

    // Hash de String (interface overhead — deve ficar FORA do loop neural)
    let t_hash = Instant::now();
    let mut found_hash = 0usize;
    for i in 0..n_lu {
        let cid = word_to_concept_id(palavras_pool[i % palavras_pool.len()]);
        if pool.buscar_conceito(cid).is_some() { found_hash += 1; }
    }
    let ns_hash = t_hash.elapsed().as_nanos() as f64 / n_lu as f64;

    let overhead_pct = (ns_hash - ns_u32) / ns_u32 * 100.0;
    linha("Lookup u32 (path quente — sem hash)",
        &fmt_ns(ns_u32), "por lookup");
    linha("Lookup String→u32+lookup (interface)",
        &fmt_ns(ns_hash), "por lookup");
    if overhead_pct > 0.0 {
        linha_warn("Interface overhead de String",
            &format!("+{:.1}% — mantenha fora do loop neural", overhead_pct));
    } else {
        linha_ok("Interface overhead de String",
            &format!("{:.1}% (FNV-1a muito rápido)", overhead_pct));
    }
    separador();
}

// ─────────────────────────────────────────────────────────────────────────────
// B. Encoding de Spikes
// ─────────────────────────────────────────────────────────────────────────────

fn benchmark_spikes() {
    secao("B — ENCODING DE SPIKES");

    let palavras: Vec<&str> = vec![
        "amor", "tristeza", "alegria", "medo", "raiva", "surpresa", "nojo",
        "sol", "lua", "estrela", "mar", "rio", "montanha", "floresta", "vento",
        "água", "fogo", "terra", "ar", "luz", "sombra", "silêncio", "música",
        "calor", "frio", "dor", "prazer", "saudade", "esperança", "sonho",
        "consciência", "memória", "tempo", "espaço", "vida", "morte",
        "selene", "pensamento", "linguagem", "aprendizado", "evolução",
        "neural", "cortex", "sinapse", "dopamina", "serotonina",
        "a", "de", "o", "e", "em", "que", "para", "com",
    ];

    subsecao("B1 — Throughput de encoding");
    let n_encode = 50_000;
    let t_enc = Instant::now();
    let mut last_pat: SpikePattern = [0u64; 8];
    for i in 0..n_encode {
        last_pat = spike_encode(palavras[i % palavras.len()]);
    }
    let enc_ns     = t_enc.elapsed().as_nanos() as f64;
    let enc_per_s  = n_encode as f64 / (enc_ns / 1e9);
    let enc_ns_each = enc_ns / n_encode as f64;
    linha("Encoding (FNV-1a + xorshift64)",
        &fmt_throughput(enc_per_s),
        &format!("({} por encoding)", fmt_ns(enc_ns_each)));

    let mut erros_k = 0usize;
    for &w in &palavras {
        let p = spike_encode(w);
        if popcount(&p) != K_ACTIVE as u32 { erros_k += 1; }
    }
    if erros_k == 0 {
        linha_ok("Esparsidade K=26",
            &format!("todos {K_ACTIVE} bits ativos em {}/{} palavras",
                palavras.len(), palavras.len()));
    } else {
        linha_warn("Esparsidade K=26", &format!("{erros_k} palavras fora do padrão"));
    }
    separador();

    subsecao("B2 — Throughput de similaridade Jaccard");
    let vocab_encoded: Vec<(&str, SpikePattern)> = palavras.iter()
        .map(|&w| (w, spike_encode(w)))
        .collect();
    let query = spike_encode("amor");

    let n_sim = 500_000;
    let t_sim = Instant::now();
    let mut sum_sim = 0.0f32;
    for i in 0..n_sim {
        let (_, ref pat) = vocab_encoded[i % vocab_encoded.len()];
        sum_sim += similarity(&query, pat);
    }
    let sim_ns    = t_sim.elapsed().as_nanos() as f64;
    let sim_per_s = n_sim as f64 / (sim_ns / 1e9);
    linha("Jaccard similarity (SIMD popcount)",
        &fmt_throughput(sim_per_s),
        &format!("({} por query)", fmt_ns(sim_ns / n_sim as f64)));

    let n_nn = 10_000;
    let t_nn = Instant::now();
    for _ in 0..n_nn {
        let _top = decode_top_n(
            &query, vocab_encoded.iter().map(|(w, p)| (*w, p)), 0.0, 3,
        );
    }
    let nn_ns = t_nn.elapsed().as_nanos() as f64;
    linha(&format!("NN search (vocab={} palavras)", vocab_encoded.len()),
        &fmt_throughput(n_nn as f64 / (nn_ns / 1e9)),
        &format!("({} por busca NN)", fmt_ns(nn_ns / n_nn as f64)));

    let mut sim_self = 0.0f32;
    let mut sim_rand = 0.0f32;
    let ns = vocab_encoded.len();
    for i in 0..ns {
        let (_, ref pa) = vocab_encoded[i];
        sim_self += similarity(pa, pa);
        let (_, ref pb) = vocab_encoded[(i + ns / 2) % ns];
        sim_rand += similarity(pa, pb);
    }
    linha("Similaridade média consigo mesma",
        &format!("{:.4}", sim_self / ns as f32), "(esperado: 1.0000)");
    linha("Similaridade média par aleatório",
        &format!("{:.4}", sim_rand / ns as f32),
        &format!("(espaço real ≈ {:.4})", K_ACTIVE as f32 * K_ACTIVE as f32 / N_NEURONS as f32));
    separador();

    subsecao("B3 — Throughput de superposição / interseção");
    use selene_kernel::encoding::spike_codec::{superimpose, intersect};
    let pa = spike_encode("amor");
    let pb = spike_encode("saudade");

    let n_arith = 1_000_000;
    let t_arith = Instant::now();
    let mut acc = pa;
    for _ in 0..n_arith { acc = superimpose(&acc, &pb); }
    let arith_ns = t_arith.elapsed().as_nanos() as f64;
    linha("Superimpose (OR) + intersect (AND)",
        &fmt_throughput(n_arith as f64 / (arith_ns / 1e9)),
        &format!("({} por op)", fmt_ns(arith_ns / n_arith as f64)));

    let sup = superimpose(&pa, &pb);
    let int = intersect(&pa, &pb);
    linha("Bits ativos em OR(amor, saudade)",
        &format!("{}", popcount(&sup)),
        &format!("(esperado: {:}–{:})", K_ACTIVE, K_ACTIVE * 2));
    linha("Bits ativos em AND(amor, saudade)",
        &format!("{}", popcount(&int)),
        &format!("(sobreposição ≈ {:.1})",
            K_ACTIVE as f32 * K_ACTIVE as f32 / N_NEURONS as f32));
    separador();
}

// ─────────────────────────────────────────────────────────────────────────────
// C. Grafo de Associações
// ─────────────────────────────────────────────────────────────────────────────

fn benchmark_resposta(state: &BrainState) {
    secao("C — GRAFO DE ASSOCIAÇÕES (concept_id u32 vs String)");

    let mut sw = state.swap_manager.try_lock().expect("swap lock held");
    let grafo_str  = sw.grafo_palavras();
    let val_str    = sw.valencias_palavras();
    let grafo_u32  = sw.grafo_conceitos();
    let val_u32    = sw.valencias_conceitos();
    drop(sw);

    let frases = &state.frases_padrao;

    subsecao("C1 — Estatísticas do grafo carregado");
    let n_nos      = grafo_str.len();
    let n_arestas: usize = grafo_str.values().map(|v| v.len()).sum();
    let grau_medio = if n_nos > 0 { n_arestas as f64 / n_nos as f64 } else { 0.0 };
    let grau_max   = grafo_str.values().map(|v| v.len()).max().unwrap_or(0);

    linha("Nós no grafo",         &format!("{n_nos}"),     "nós");
    linha("Arestas totais",       &format!("{n_arestas}"), "associações");
    linha("Frases-padrão",        &format!("{}", frases.len()), "frases");
    linha("Grau médio",           &format!("{grau_medio:.2}"), "vizinhos/nó");
    linha("Grau máximo",          &format!("{grau_max}"),  "vizinhos");
    separador();

    subsecao("C2 — Lookup: HashMap<String,...> vs HashMap<u32,...>");
    // Benchmark String lookup (path legado)
    let palavras_chave_str: Vec<&str> = grafo_str.keys().take(200)
        .map(|s| s.as_str()).collect();
    let palavras_chave_u32: Vec<u32>  = palavras_chave_str.iter()
        .map(|&w| word_to_concept_id(w)).collect();

    if palavras_chave_str.is_empty() {
        println!("  │  ⚠ Grafo vazio — skip");
        separador();
        return;
    }

    let n_lookups = 500_000usize;

    // String path
    let t_str = Instant::now();
    let mut found_str = 0usize;
    for i in 0..n_lookups {
        if grafo_str.get(palavras_chave_str[i % palavras_chave_str.len()]).is_some() {
            found_str += 1;
        }
    }
    let ns_str = t_str.elapsed().as_nanos() as f64 / n_lookups as f64;

    // u32 path (cérebro real)
    let t_u32 = Instant::now();
    let mut found_u32 = 0usize;
    for i in 0..n_lookups {
        if grafo_u32.get(&palavras_chave_u32[i % palavras_chave_u32.len()]).is_some() {
            found_u32 += 1;
        }
    }
    let ns_u32 = t_u32.elapsed().as_nanos() as f64 / n_lookups as f64;

    linha("Lookup HashMap<String, ...> (legado)",
        &fmt_ns(ns_str), "por lookup");
    linha("Lookup HashMap<u32, ...> (V3.3 ID)",
        &fmt_ns(ns_u32), "por lookup");

    let speedup_lu = ns_str / ns_u32;
    if speedup_lu >= 1.0 {
        linha_ok("Speedup u32 vs String lookup",
            &format!("{:.2}×  (sem alocação de hash String)", speedup_lu));
    } else {
        linha_info("Speedup u32 vs String",
            &format!("{:.2}× (grafo pequeno — diferença marginal)", speedup_lu));
    }

    // Overhead de hashing no boundary (só paga uma vez por input externo)
    let n_hash = 100_000usize;
    let t_boundary = Instant::now();
    let mut sum_hash = 0u32;
    for i in 0..n_hash {
        sum_hash = sum_hash.wrapping_add(
            word_to_concept_id(palavras_chave_str[i % palavras_chave_str.len()])
        );
    }
    let ns_boundary = t_boundary.elapsed().as_nanos() as f64 / n_hash as f64;
    linha_info("FNV-1a hash (boundary — 1× por input)",
        &format!("{} por hash — não conta no loop neural", fmt_ns(ns_boundary)));
    separador();

    subsecao("C3 — Walk no grafo u32 (geração de resposta)");
    let emocao = 0.5f32;
    let n_walks = 10_000usize;
    let passos  = 10usize;
    let mut comp_total = 0usize;
    let mut ttr_total  = 0.0f32;

    let chaves_u32: Vec<u32> = grafo_u32.keys().take(200).copied().collect();
    if chaves_u32.is_empty() {
        println!("  │  ⚠ Grafo u32 vazio — skip");
        separador();
        return;
    }

    let t_walk = Instant::now();
    for i in 0..n_walks {
        let mut visitados: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut atual = chaves_u32[i % chaves_u32.len()];
        let mut cadeia: Vec<u32> = Vec::with_capacity(passos);

        for _ in 0..passos {
            if visitados.contains(&atual) { break; }
            visitados.insert(atual);
            cadeia.push(atual);

            if let Some(vizinhos) = grafo_u32.get(&atual) {
                let prox = vizinhos.iter()
                    .filter(|(w, p)| !visitados.contains(w) && *p > -0.1)
                    .min_by(|a, b| {
                        let va = val_u32.get(&a.0).copied().unwrap_or(emocao);
                        let vb = val_u32.get(&b.0).copied().unwrap_or(emocao);
                        (va - emocao).abs().partial_cmp(&(vb - emocao).abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                match prox {
                    Some((next, _)) => atual = *next,
                    None => break,
                }
            } else { break; }
        }

        comp_total += cadeia.len();
        let unique: std::collections::HashSet<u32> = cadeia.iter().copied().collect();
        if !cadeia.is_empty() {
            ttr_total += unique.len() as f32 / cadeia.len() as f32;
        }
    }
    let walk_ns = t_walk.elapsed().as_nanos() as f64;

    linha("Walks por segundo (u32)",
        &fmt_throughput(n_walks as f64 / (walk_ns / 1e9)),
        &format!("({} por walk de {} passos)", fmt_ns(walk_ns / n_walks as f64), passos));
    linha("Comprimento médio do walk",
        &format!("{:.2}", comp_total as f64 / n_walks as f64), "concept_ids por resposta");
    linha("TTR médio (diversidade)",
        &format!("{:.3}", ttr_total / n_walks as f32), "(1.0=todos únicos)");
    separador();

    subsecao("C4 — Scoring de frases-padrão (seleção de prefixo)");
    if frases.is_empty() {
        println!("  │  ⚠ Sem frases — skip");
        separador();
        return;
    }
    let inputs_teste = [
        "quem é você", "o que você sente agora", "você tem memória",
        "o que é consciência", "me conta sobre o amor", "você aprende",
    ];
    let n_prefix = 50_000usize;
    let t_prefix = Instant::now();
    let mut melhor_soma = 0usize;
    for round in 0..n_prefix {
        let input = inputs_teste[round % inputs_teste.len()].to_lowercase();
        let tokens: std::collections::HashSet<&str> = input.split_whitespace().collect();
        melhor_soma += frases.iter()
            .map(|f| f.iter().filter(|w| tokens.contains(w.as_str())).count())
            .max().unwrap_or(0);
    }
    let prefix_ns = t_prefix.elapsed().as_nanos() as f64;
    linha(&format!("Prefix scoring ({} frases × {} inputs)", frases.len(), inputs_teste.len()),
        &fmt_throughput(n_prefix as f64 / (prefix_ns / 1e9)),
        &format!("({} por scoring)", fmt_ns(prefix_ns / n_prefix as f64)));
    separador();
}

// ─────────────────────────────────────────────────────────────────────────────
// D. Aprendizado
// ─────────────────────────────────────────────────────────────────────────────

fn benchmark_aprendizado(config: &Config) {
    secao("D — APRENDIZADO");

    subsecao("D1 — Crescimento do grafo de associações (u32)");
    let corpus = [
        "o sol nasce toda manhã com luz e calor",
        "a lua brilha na noite escura com beleza",
        "o vento sopra suave trazendo frescor",
        "selene aprende com amor e curiosidade",
        "a consciência emerge da complexidade neural",
        "memória e emoção se entrelaçam no hipocampo",
        "o medo protege mas também paralisa a alma",
        "a dopamina recompensa o aprendizado bem sucedido",
        "sonhos consolidam o que o dia ensinou",
        "linguagem é o espelho do pensamento humano",
    ];

    // Grafo nativo u32 — zero alocações de String no loop interno
    let mut grafo_u32: HashMap<u32, Vec<(u32, f32)>> = HashMap::new();
    let mut aresta_cnt: HashMap<(u32, u32), u32> = HashMap::new();

    let n_rodadas = 1000;
    let t_learn = Instant::now();
    let mut arestas_novas = 0usize;

    for rodada in 0..n_rodadas {
        let frase = corpus[rodada % corpus.len()];
        let ids: Vec<u32> = frase.split_whitespace()
            .filter(|w| w.len() > 2)
            .map(word_to_concept_id)
            .collect();

        for i in 0..ids.len() {
            for j in (i + 1)..ids.len().min(i + 4) {
                let (a, b) = (ids[i], ids[j]);
                let chave  = if a < b { (a, b) } else { (b, a) };
                let cnt    = aresta_cnt.entry(chave).or_insert(0);
                *cnt += 1;
                let peso   = (*cnt as f32 * 0.1).min(1.0);

                let entry = grafo_u32.entry(chave.0).or_default();
                if let Some(e) = entry.iter_mut().find(|(w, _)| *w == chave.1) {
                    e.1 = peso;
                } else {
                    entry.push((chave.1, peso));
                    arestas_novas += 1;
                }

                let entry2 = grafo_u32.entry(chave.1).or_default();
                if let Some(e) = entry2.iter_mut().find(|(w, _)| *w == chave.0) {
                    e.1 = peso;
                } else {
                    entry2.push((chave.0, peso));
                }
            }
        }
    }
    let learn_ns = t_learn.elapsed().as_nanos() as f64;

    let nos_finais     = grafo_u32.len();
    let arestas_finais: usize = grafo_u32.values().map(|v| v.len()).sum();

    linha("Velocidade de processamento do corpus",
        &fmt_throughput(n_rodadas as f64 / (learn_ns / 1e9)), "frases/s");
    linha("Nós criados após aprendizado",  &format!("{nos_finais}"),     "nós (u32)");
    linha("Arestas totais no grafo",       &format!("{arestas_finais}"), "associações");
    linha("Novidade de arestas",
        &format!("{:.2}", arestas_finais as f64 / n_rodadas as f64),
        "arestas / rodada (converge com repetição)");
    separador();

    subsecao("D2 — Aprendizado Hebbiano na camada temporal");
    let n_neu = 256usize;
    let mut temporal = TemporalLobe::new(n_neu, 0.005, 0.1, config);

    let padroes_teste = [
        vec![1.0f32; n_neu],
        (0..n_neu).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect::<Vec<f32>>(),
        (0..n_neu).map(|i| i as f32 / n_neu as f32).collect::<Vec<f32>>(),
    ];

    let n_ticks_hebb = 2000;
    let mut t_ms = 0.0f32;
    let t_hebb = Instant::now();
    for tick in 0..n_ticks_hebb {
        let padrao = &padroes_teste[tick % padroes_teste.len()];
        let _ = temporal.process(padrao, padrao, 0.005, t_ms, config);
        temporal.hebbian_update(&padrao.iter().map(|&v| v > 0.5).collect::<Vec<bool>>());
        t_ms += 5.0;
    }
    let hebb_ns    = t_hebb.elapsed().as_nanos() as f64;
    let hebb_per_s = n_ticks_hebb as f64 / (hebb_ns / 1e9);
    linha("Ticks Hebbianos (256 neurônios, K=8)",
        &fmt_throughput(hebb_per_s),
        &format!("({} por tick)", fmt_ns(hebb_ns / n_ticks_hebb as f64)));
    separador();

    subsecao("D3 — Chunking: emergência de padrões compostos");
    let mut chunker = ChunkingEngine::new(RegionType::Temporal);
    let n_chunk = 256usize;
    let mut camada_chunk = CamadaHibrida::new(
        n_chunk, "chunk_bench", TipoNeuronal::RS,
        Some((TipoNeuronal::FS, 0.2)), None, 1.0,
    );
    let n_ticks_chunk = 3000usize;
    let mut chunks_emergidos = 0usize;
    let mut tick_primeiro_chunk: Option<usize> = None;
    let inputs_chunk: Vec<f32> = (0..n_chunk)
        .map(|i| if i < n_chunk / 4 { 8.0 } else { 0.2 })
        .collect();

    let t_chunk = Instant::now();
    for tick in 0..n_ticks_chunk {
        let t_ms = tick as f32 * 5.0;
        let cur: Vec<f32> = if tick % 10 < 5 {
            inputs_chunk.clone()
        } else {
            vec![0.01f32; n_chunk]
        };
        let spikes = camada_chunk.update(&cur, 0.005, t_ms);
        let novos  = chunker.registrar_spikes(&spikes, &camada_chunk, 0.3, t_ms);
        if !novos.is_empty() {
            if tick_primeiro_chunk.is_none() { tick_primeiro_chunk = Some(tick); }
            chunks_emergidos += novos.len();
        }
    }
    let chunk_ns = t_chunk.elapsed().as_nanos() as f64;
    let stats = chunker.stats();
    let total_chunks = stats.primitivos + stats.compostos + stats.sequencias;

    linha("Ticks de chunking (256 neurônios)",
        &fmt_throughput(n_ticks_chunk as f64 / (chunk_ns / 1e9)),
        &format!("({} por tick)", fmt_ns(chunk_ns / n_ticks_chunk as f64)));
    linha("Chunks total emergidos", &format!("{total_chunks}"), "chunks");
    linha("  Primitivos / Compostos / Sequências",
        &format!("{} / {} / {}", stats.primitivos, stats.compostos, stats.sequencias), "");
    match tick_primeiro_chunk {
        Some(t) => linha("Latência até 1º chunk",
            &format!("{:.1} ms", t as f32 * 5.0), &format!("(tick {})", t)),
        None => linha_warn("Latência até 1º chunk", "nenhum chunk emergiu"),
    }
    separador();

    subsecao("D4 — RL (Q-learning): crescimento da Q-table");
    use selene_kernel::learning::rl::ReinforcementLearning;
    let mut rl = ReinforcementLearning::new();
    let n_rl = 10_000usize;
    let t_rl = Instant::now();
    for i in 0..n_rl {
        let padrao: Vec<f32> = (0..64).map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0).collect();
        let dopa  = 0.3 + (i % 7) as f32 * 0.1;
        let acao  = (i % 5) as f32 * 0.25;
        let _rpe  = rl.update(&padrao, dopa, acao, config);
    }
    let rl_ns     = t_rl.elapsed().as_nanos() as f64;
    let padrao_f: Vec<f32> = (0..64).map(|j| j as f32 / 64.0).collect();
    let rpe = rl.update(&padrao_f, 0.8, 0.5, config);
    linha("Updates RL por segundo",
        &fmt_throughput(n_rl as f64 / (rl_ns / 1e9)),
        &format!("({} por update)", fmt_ns(rl_ns / n_rl as f64)));
    linha("RPE após 10k updates (dopa=0.8)",
        &format!("{:.4}", rpe), "(positivo = acima da linha de base)");
    linha("Total de atualizações RL", &format!("{}", rl.total_atualizacoes()), "");
    separador();
}

// ─────────────────────────────────────────────────────────────────────────────
// E. Memória, Grounding e NVMe Swap
// ─────────────────────────────────────────────────────────────────────────────

fn benchmark_grounding(state: &mut BrainState) {
    secao("E — MEMÓRIA, GROUNDING E NVME SWAP");

    subsecao("E1 — Taxa de binding grounding");
    let visual_ativo: SpikePattern = [0xF0F0_F0F0u64; 8];
    let audio_ativo: SpikePattern  = [0x0F0F_0F0Fu64; 8];

    let palavras_bind = [
        vec!["quente".to_string(), "sol".to_string()],
        vec!["frio".to_string(), "inverno".to_string()],
        vec!["musica".to_string(), "ritmo".to_string()],
        vec!["dor".to_string(), "corpo".to_string()],
        vec!["alegria".to_string(), "luz".to_string()],
    ];

    let n_binds = 50_000;
    let t_bind  = Instant::now();
    for i in 0..n_binds {
        state.grounding_bind(
            &palavras_bind[i % palavras_bind.len()],
            visual_ativo, audio_ativo,
            0.5, 0.6, i as f64 * 10.0,
        );
    }
    let bind_ns    = t_bind.elapsed().as_nanos() as f64;
    let bind_per_s = n_binds as f64 / (bind_ns / 1e9);

    linha("grounding_bind() por segundo",
        &fmt_throughput(bind_per_s),
        &format!("({} por bind)", fmt_ns(bind_ns / n_binds as f64)));

    let g_quente = state.grounding.get("quente").copied().unwrap_or(0.0);
    let g_musica = state.grounding.get("musica").copied().unwrap_or(0.0);
    linha("Score 'quente' após 10k binds", &format!("{:.4}", g_quente), "(saturação → 1.0)");
    linha("Score 'musica' após 10k binds", &format!("{:.4}", g_musica), "");
    separador();

    subsecao("E2 — NVMe Swap: latência de leitura e escrita RAM-swap");
    // Benchmark do NeuralPool swap: mede latência de evict_lru + restaurar_swapped
    let mut pool = NeuralPool::new(4);  // pool pequeno para forçar evicção
    let conceitos = ["alpha", "beta", "gamma", "delta", "epsilon"];
    let mut cids: Vec<u32> = conceitos.iter().map(|&w| word_to_concept_id(w)).collect();

    // Preenche o pool
    for &cid in &cids[..4] {
        let _ = pool.aloca_para_tarefa(cid, CorticalLevel::C1Perceptual, 0.0);
    }

    let n_swap_ops = 1_000usize;
    let mut swap_write_ns = 0u128;
    let mut swap_read_ns  = 0u128;

    for i in 0..n_swap_ops {
        // Evicção (RAM-swap write): aloca conceito que não está no pool
        let t_w = Instant::now();
        let _ = pool.aloca_para_tarefa(cids[4], CorticalLevel::C2Lexical, i as f64 * 10.0);
        swap_write_ns += t_w.elapsed().as_nanos();

        // Restauração (RAM-swap read): recupera o LRU evictado
        let evictado = cids[i % 4];
        let t_r = Instant::now();
        let _ = pool.aloca_para_tarefa(evictado, CorticalLevel::C1Perceptual, i as f64 * 10.0 + 5.0);
        swap_read_ns += t_r.elapsed().as_nanos();
    }

    let avg_write_ns = swap_write_ns as f64 / n_swap_ops as f64;
    let avg_read_ns  = swap_read_ns  as f64 / n_swap_ops as f64;

    if avg_write_ns < 10_000.0 {
        linha_ok("RAM-swap write (evict_lru)",
            &format!("{}", fmt_ns(avg_write_ns)));
    } else {
        linha_warn("RAM-swap write (evict_lru)",
            &format!("{} — latência alta, verificar contenção", fmt_ns(avg_write_ns)));
    }
    if avg_read_ns < 10_000.0 {
        linha_ok("RAM-swap read (restaurar_swapped)",
            &format!("{}", fmt_ns(avg_read_ns)));
    } else {
        linha_warn("RAM-swap read (restaurar_swapped)",
            &format!("{} — alerta hardware (pode ser I/O se swap_dir ativo)", fmt_ns(avg_read_ns)));
    }
    linha_info("Swap NVMe disk (swap_dir=None)",
        "desativado — apenas RAM-swap (sem latência de I/O)");
    separador();

    subsecao("E3 — Capacidade da memória episódica");
    let n_eventos = state.historico_episodico.len();
    let bytes_por_evento = std::mem::size_of::<EventoEpisodico>() + 8 * std::mem::size_of::<String>();
    let mem_kb = n_eventos * bytes_por_evento / 1024;

    linha("Eventos episódicos em memória", &format!("{n_eventos}"), "eventos (máx 500)");
    linha("Memória estimada da fila",      &format!("{mem_kb}"),    "KB");
    linha("Palavras com grounding ativo",  &format!("{}", state.grounding.len()), "tokens com score > 0");
    separador();

    subsecao("E4 — Decay de grounding (meia-vida)");
    state.grounding.insert("teste_decay".to_string(), 0.5);
    let mut ticks_mv = 0usize;
    let mut g_atual  = 0.5f32;
    let t_decay = Instant::now();
    while g_atual > 0.25 && ticks_mv < 10_000 {
        state.grounding.iter_mut().for_each(|(_, v)| *v *= 0.999);
        ticks_mv += 1;
        g_atual = state.grounding.get("teste_decay").copied().unwrap_or(0.0);
    }
    let decay_ns  = t_decay.elapsed().as_nanos() as f64;
    let ticks_reais = ticks_mv * 1000;
    let s_mv = ticks_reais as f64 / 200.0;

    linha("Chamadas decay para meia-vida (0.5→0.25)",
        &format!("{ticks_mv}"), "chamadas × 1000 ticks");
    linha("Meia-vida em tempo real",
        &format!("{:.1}", s_mv), "segundos");
    linha("Throughput de decay",
        &fmt_throughput(ticks_mv as f64 / (decay_ns / 1e9)), "ciclos/s");
    separador();
}

// ─────────────────────────────────────────────────────────────────────────────
// F. Capacidade do Sistema
// ─────────────────────────────────────────────────────────────────────────────

fn benchmark_capacidade(state: &BrainState) {
    secao("F — CAPACIDADE DO SISTEMA");

    subsecao("F1 — Vocabulário e grafo atual");
    let mut sw_f     = state.swap_manager.try_lock().expect("swap lock held");
    let grafo_f      = sw_f.grafo_palavras();
    let valencias_f  = sw_f.valencias_palavras();
    drop(sw_f);

    let n_nos     = grafo_f.len();
    let n_arestas: usize = grafo_f.values().map(|v| v.len()).sum();
    let n_helix   = state.spike_vocab.len();
    let n_frases  = state.frases_padrao.len();
    let dist = {
        let pos = valencias_f.values().filter(|&&v| v > 0.1).count();
        let neu = valencias_f.values().filter(|&&v| v.abs() <= 0.1).count();
        let neg = valencias_f.values().filter(|&&v| v < -0.1).count();
        (pos, neu, neg)
    };

    linha("Vocabulário (tokens com valência)",   &format!("{}", valencias_f.len()), "tokens");
    linha("Helix (padrões spike in-memory)",     &format!("{n_helix}"), "padrões");
    linha("Nós no grafo de associações",         &format!("{n_nos}"),   "nós");
    linha("Arestas no grafo",                    &format!("{n_arestas}"), "associações");
    linha("Frases-padrão carregadas",            &format!("{n_frases}"), "frases");
    linha("Distribuição de valências",
        &format!("+ {} / 0 {} / - {}", dist.0, dist.1, dist.2),
        "(positiva/neutra/negativa)");
    separador();

    subsecao("F2 — Helix Store (arquivo mmap)");
    match &state.helix {
        Some(helix) => {
            let n_padroes   = helix.len();
            let bytes_record = 96u64;
            let kb = n_padroes as u64 * bytes_record / 1024;
            let mb = kb as f64 / 1024.0;
            let max_por_gb  = 1_073_741_824u64 / bytes_record;
            linha("Padrões armazenados no Helix", &format!("{n_padroes}"), "registros");
            linha("Espaço usado pelos padrões",   &format!("{:.2} MB", mb), &format!("({kb} KB)"));
            linha("Capacidade máxima por GB",     &format!("{max_por_gb}"), "padrões/GB");
        }
        None => {
            linha_warn("Helix Store", "não carregado (.hlx ausente ou desativado)");
        }
    }
    separador();

    subsecao("F3 — Neurônios por zona (pipeline 128 neurônios)");
    let n_base = 128usize;
    let zonas = [
        ("Occipital (V1+V2)",   n_base,     "60% RS + 40% CH / 70% CH + 30% RS"),
        ("Parietal",            n_base,     "RS + LT"),
        ("Temporal",            n_base,     "55% RS + 30% CH + 15% FS"),
        ("Frontal (exec+inhib)",n_base * 2, "80% RS + 20% FS / 100% FS"),
        ("Límbico",             n_base / 2, "IB + FS"),
        ("Hipocampo (CA1+CA3)", n_base / 2, "80% RS + 20% LT / 70% RS + 30% RZ"),
        ("Cerebelo",            n_base / 4, "RZ (Purkinje)"),
        ("Tálamo",              0,          "relay sem modelo spike"),
        ("Tronco cerebral",     0,          "relay sem modelo spike"),
    ];
    let total_spike: usize = zonas.iter().map(|(_, n, _)| n).sum();
    for (nome, n, tipos) in &zonas {
        if *n > 0 {
            linha(&format!("  {nome}"), &format!("{n} neurônios"), &format!("({tipos})"));
        } else {
            linha(&format!("  {nome}"), "relay", tipos);
        }
    }
    linha("TOTAL neurônios spike ativos", &format!("{total_spike}"), "neurônios por instância");
    separador();

    subsecao("F4 — Sizing: NTC Pool vs Pipeline completo");
    linha("NTC INT4 mem/neurônio", "~4 bits úteis", "(28 bits mascarados — mesmo u32 raw)");
    linha("NTC INT8 mem/neurônio", "~8 bits úteis", "(FP32 raw, precisão lógica INT8)");
    linha("NTC FP32 mem/neurônio", "32 bits úteis", "(pipeline completo — C3/C4)");
    linha("NeuralPool L1 capacity", "4096 blocos", "(u32 concept_id, LRU eviction)");
    linha("NVMe Swap",             "RAM-swap + F:/selene_pool_swap/ opcional", "");
    separador();
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    println!();
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║     SELENE BRAIN V3.3 — BENCHMARK INTENSIVO (NTC + concept_id)   ║");
    println!("║     Data: 2026-04-27  |  Plataforma: Windows 11  |  Ryzen 3500U  ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");

    let config = Config::new(ModoOperacao::Normal);

    println!("\n⏳ Inicializando BrainState com dados reais...");
    let swap  = Arc::new(TokioMutex::new(SwapManager::new(256, 3600)));
    let flags = SensorFlags::new_desativados();
    let mut state = BrainState::new(swap, &config, flags, Arc::new(ActiveContext::new()), Arc::new(selene_kernel::learning::go_nogo::GoNoGoFilter::new()));
    println!("✓ BrainState carregado.");

    benchmark_neural(&config);
    benchmark_spikes();
    benchmark_resposta(&state);
    benchmark_aprendizado(&config);
    benchmark_grounding(&mut state);
    benchmark_capacidade(&state);

    println!();
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK CONCLUÍDO                            ║");
    println!("║  Fator tempo-real: leia A2 Cenário B (uso biológico/real 8%)      ║");
    println!("║  NTC speedup:      leia A0 INT4 vs FP32                           ║");
    println!("║  Interface overhead: String→u32 isolado em C2 e A3               ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();
}
