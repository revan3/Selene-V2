// =============================================================================
// src/bin/intensive_benchmark.rs
// Benchmark intensivo — performance, capacidade e velocidade de aprendizado
// =============================================================================
//
//  [A] Processamento Neural  — latência/tick por tamanho de rede e tipo de neurônio
//  [B] Encoding de Spikes    — throughput encode/similarity/helix
//  [C] Geração de Respostas  — ops de grafo, walk, diversidade lexical
//  [D] Aprendizado           — crescimento de grafo, Hebbian, chunking, RL
//  [E] Memória e Grounding   — grounding bind/decay, memória episódica
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
    synaptic_core::{CamadaHibrida, TipoNeuronal, PrecisionType},
    neurochem::NeuroChem,
    thalamus::Thalamus,
    brainstem::Brainstem,
    learning::{
        attention::AttentionGate,
        inter_lobe::BrainConnections,
        lobe_router::LobeRouter,
        chunking::{ChunkingEngine, TipoChunk},
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

fn separador() {
    println!("  └─────────────────────────────────────────────────────────────");
}

fn fmt_ns(ns: f64) -> String {
    if ns < 1_000.0       { format!("{:.1} ns", ns) }
    else if ns < 1_000_000.0 { format!("{:.2} µs", ns / 1_000.0) }
    else                  { format!("{:.3} ms", ns / 1_000_000.0) }
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

    subsecao("A1 — Latência por tick (n_neurônios × tipo)");
    let dt = 0.005f32;  // 200 Hz
    let t0 = 0.0f32;
    let repeticoes = 500;

    let configs_teste: &[(usize, &str, TipoNeuronal, Option<(TipoNeuronal, f32)>)] = &[
        (256,   "256  RS puro",          TipoNeuronal::RS, None),
        (1024,  "1024 RS puro",          TipoNeuronal::RS, None),
        (4096,  "4096 RS puro",          TipoNeuronal::RS, None),
        (256,   "256  RS+FS (80/20)",    TipoNeuronal::RS, Some((TipoNeuronal::FS, 0.20))),
        (1024,  "1024 RS+CH (70/30)",    TipoNeuronal::RS, Some((TipoNeuronal::CH, 0.30))),
        (256,   "256  RZ cerebelar",     TipoNeuronal::RZ, None),
        (256,   "256  TC talamocortical",TipoNeuronal::TC, None),
    ];

    for (n, nome, tipo_p, tipo_s) in configs_teste {
        let mut camada = CamadaHibrida::new(
            *n, nome, *tipo_p, *tipo_s,
            None,   // distribuição padrão
            40.0,
        );
        let inputs = vec![0.05f32; *n];

        // aquecimento
        for _ in 0..10 {
            let _ = camada.update(&inputs, dt, t0);
        }

        let t_start = Instant::now();
        for tick in 0..repeticoes {
            let t_ms = tick as f32 * dt * 1000.0;
            let _ = camada.update(&inputs, dt, t_ms);
        }
        let elapsed_ns = t_start.elapsed().as_nanos() as f64;
        let ns_por_tick  = elapsed_ns / repeticoes as f64;
        let ns_por_neuronio = ns_por_tick / *n as f64;
        let ticks_por_s = 1e9 / ns_por_tick;

        linha(
            &format!("{nome}"),
            &fmt_ns(ns_por_tick),
            &format!("/ tick  ({:.1} ps/neurônio  |  {:.0} ticks/s)",
                ns_por_neuronio * 1000.0, ticks_por_s),
        );
    }
    separador();

    subsecao("A2 — Pipeline completo 128 neurônios: latência ponto-a-ponta");
    let n = 128usize;
    let mut occipital  = OccipitalLobe::new(n, 0.2, config);
    let mut parietal   = ParietalLobe::new(n, 0.2, config);
    let mut temporal   = TemporalLobe::new(n, dt, 0.2, config);
    let mut limbic     = LimbicSystem::new(n / 2, config);
    let mut hippocampus = Hippocampus::new(n / 2, config);
    let mut frontal    = FrontalLobe::new(n, 0.2, 0.1, config);
    let mut cerebelo   = Cerebellum::new(n / 4, n / 2, config);
    let mut thalamus   = Thalamus::new();
    let mut brainstem  = Brainstem::new();
    let mut neuro      = NeuroChem::new();
    let mut attention  = AttentionGate::new(n);
    let mut conn       = BrainConnections::new(n);
    let mut sensor     = HardwareSensor::dummy();
    let goal = vec![0.5f32; n];
    let zero = vec![0.0f32; n];
    let zero_half = vec![0.0f32; n / 2];
    let mut prev_frontal  = zero.clone();
    let mut prev_temporal = zero.clone();
    let mut prev_parietal = zero.clone();
    let mut prev_limbic   = zero_half.clone();
    let mut prev_hippo    = zero_half.clone();

    let n_ticks_pipeline = 1000usize;
    let mut spikes_total: u64 = 0;

    let t_pipeline = Instant::now();
    for tick in 0..n_ticks_pipeline {
        let t_ms = tick as f32 * dt * 1000.0;
        neuro.update(&mut sensor, config);
        let (da, ser, cor) = (neuro.dopamine, neuro.serotonin, neuro.cortisol);
        let tonic = 0.04 * (tick as f32 * 0.1).sin().abs();
        let input_vis = vec![tonic; n];
        let input_aud = vec![tonic * 0.5; 10];

        brainstem.update(0.1, dt);
        let alertness = brainstem.stats().alertness.max(0.3);
        let relayed = thalamus.relay(&input_vis, neuro.noradrenaline, config);
        let cochlea = brainstem.modulate(&input_aud);
        attention.set_topdown(&prev_frontal);
        let attended = attention.attend(&relayed, dt * 1000.0);
        let currents = conn.project_all(
            &attended, &prev_temporal, &prev_parietal,
            &prev_frontal, &prev_limbic, &prev_hippo,
        );
        let hybrid: Vec<f32> = attended.iter().enumerate()
            .map(|(i, &v)| (v + currents.para_temporal.get(i).copied().unwrap_or(0.0) * 0.1)
                .clamp(0.0, 1.0) * alertness)
            .collect();

        let features = occipital.visual_sweep(&hybrid, dt, Some(&parietal.spatial_map), t_ms, config);
        let chunk_size = (n / features.len().max(1)).max(1);
        let mut vision_full = vec![0.0f32; n];
        for (i, &f) in features.iter().enumerate() {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(n);
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
        let action = frontal.decide(&recognized, &goal, dt, t_ms, config);
        let climbing: Vec<f32> = action.iter().zip(recognized.iter())
            .map(|(a, r)| (a - r).clamp(-1.0, 1.0)).collect();
        let _cerb_out = cerebelo.compute_motor_output(&recognized, &climbing, dt, t_ms, config);

        if tick % 5 == 0 {
            occipital.v1_primary_layer.modular_neuro(da, ser, cor);
            temporal.recognition_layer.modular_neuro(da, ser, cor);
            frontal.executive_layer.modular_neuro(da, ser, cor);
        }

        prev_frontal.iter_mut().enumerate().for_each(|(i, v)| *v = action.get(i).copied().unwrap_or(0.0));
        prev_temporal.iter_mut().enumerate().for_each(|(i, v)| *v = recognized.get(i).copied().unwrap_or(0.0));
        prev_parietal.iter_mut().enumerate().for_each(|(i, v)| *v = new_parietal.get(i).copied().unwrap_or(0.0));
        prev_limbic.iter_mut().for_each(|v| *v = emotion.clamp(0.0, 1.0));
    }
    let pipeline_ms = t_pipeline.elapsed().as_secs_f64() * 1000.0;
    let ms_por_tick = pipeline_ms / n_ticks_pipeline as f64;
    let tick_rate   = 1000.0 / ms_por_tick;
    let taxa_spike  = spikes_total as f64 / n_ticks_pipeline as f64 / n as f64 * 100.0;

    linha("Pipeline 1000 ticks / 128 neurônios",
        &format!("{:.3} ms/tick", ms_por_tick),
        &format!("({:.0} ticks/s  |  taxa spike ≈{:.1}%)", tick_rate, taxa_spike));
    let realtime_ratio = tick_rate / 200.0; // 200 Hz = real-time
    if realtime_ratio >= 1.0 {
        linha_ok("Fator tempo-real", &format!("{:.1}× acima de real-time (200 Hz)", realtime_ratio));
    } else {
        linha_warn("Fator tempo-real", &format!("{:.2}× (abaixo de real-time)", realtime_ratio));
    }
    separador();
}

// ─────────────────────────────────────────────────────────────────────────────
// B. Encoding de Spikes
// ─────────────────────────────────────────────────────────────────────────────

fn benchmark_spikes() {
    secao("B — ENCODING DE SPIKES");

    // Vocabulário de teste representativo
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
    let enc_ns = t_enc.elapsed().as_nanos() as f64;
    let enc_per_s = n_encode as f64 / (enc_ns / 1e9);
    let enc_ns_each = enc_ns / n_encode as f64;
    linha("Encoding (FNV-1a + xorshift64)",
        &fmt_throughput(enc_per_s),
        &format!("({} por encoding)", fmt_ns(enc_ns_each)));

    // Verifica que cada palavra gera exatamente K_ACTIVE bits
    let mut erros_k = 0usize;
    for &w in &palavras {
        let p = spike_encode(w);
        let k = popcount(&p);
        if k != K_ACTIVE as u32 { erros_k += 1; }
    }
    if erros_k == 0 {
        linha_ok("Esparsidade K=26", &format!("todos {K_ACTIVE} bits ativos em {}/{} palavras",
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

    // Similaridade par-a-par no vocabulário
    let n_sim = 500_000;
    let t_sim = Instant::now();
    let mut sum_sim = 0.0f32;
    for i in 0..n_sim {
        let (_, ref pat) = vocab_encoded[i % vocab_encoded.len()];
        sum_sim += similarity(&query, pat);
    }
    let sim_ns = t_sim.elapsed().as_nanos() as f64;
    let sim_per_s = n_sim as f64 / (sim_ns / 1e9);
    linha("Jaccard similarity (SIMD popcount)",
        &fmt_throughput(sim_per_s),
        &format!("({} por query)", fmt_ns(sim_ns / n_sim as f64)));

    // Nearest neighbor search no vocabulário completo
    let t_nn = Instant::now();
    let n_nn = 10_000;
    for _ in 0..n_nn {
        let _top = decode_top_n(
            &query,
            vocab_encoded.iter().map(|(w, p)| (*w, p)),
            0.0, 3,
        );
    }
    let nn_ns = t_nn.elapsed().as_nanos() as f64;
    linha(&format!("NN search (vocab={} palavras)", vocab_encoded.len()),
        &fmt_throughput(n_nn as f64 / (nn_ns / 1e9)),
        &format!("({} por busca NN)", fmt_ns(nn_ns / n_nn as f64)));

    // Estatísticas de similaridade entre pares semânticos vs aleatórios
    let mut sim_self_sum = 0.0f32;
    let mut sim_rand_sum = 0.0f32;
    let n_stats = vocab_encoded.len();
    for i in 0..n_stats {
        let (_, ref pa) = vocab_encoded[i];
        sim_self_sum += similarity(pa, pa);
        let j = (i + n_stats / 2) % n_stats;
        let (_, ref pb) = vocab_encoded[j];
        sim_rand_sum += similarity(pa, pb);
    }
    let sim_self_avg = sim_self_sum / n_stats as f32;
    let sim_rand_avg = sim_rand_sum / n_stats as f32;
    linha("Similaridade média palavra consigo mesma",
        &format!("{:.4}", sim_self_avg), "(esperado: 1.0000)");
    linha("Similaridade média par aleatório",
        &format!("{:.4}", sim_rand_avg),
        &format!("(espaço real ≈ {:.4})", K_ACTIVE as f32 * K_ACTIVE as f32 / N_NEURONS as f32));
    separador();

    subsecao("B3 — Throughput de superposição / interseção de padrões");
    use selene_kernel::encoding::spike_codec::{superimpose, intersect};
    let pa = spike_encode("amor");
    let pb = spike_encode("saudade");

    let n_arith = 1_000_000;
    let t_arith = Instant::now();
    let mut acc = pa;
    for _ in 0..n_arith {
        acc = superimpose(&acc, &pb);
    }
    let arith_ns = t_arith.elapsed().as_nanos() as f64;
    linha("Superimpose (OR) + intersect (AND)",
        &fmt_throughput(n_arith as f64 / (arith_ns / 1e9)),
        &format!("({} por op)", fmt_ns(arith_ns / n_arith as f64)));

    let sup_amor_saudade = superimpose(&pa, &pb);
    let int_amor_saudade = intersect(&pa, &pb);
    linha("Bits ativos em OR(amor, saudade)",
        &format!("{}", popcount(&sup_amor_saudade)),
        &format!("(esperado: {:}–{:})", K_ACTIVE, K_ACTIVE * 2));
    linha("Bits ativos em AND(amor, saudade)",
        &format!("{}", popcount(&int_amor_saudade)),
        &format!("(sobreposição esperada ≈ {:.1})",
            K_ACTIVE as f32 * K_ACTIVE as f32 / N_NEURONS as f32));
    separador();
}

// ─────────────────────────────────────────────────────────────────────────────
// C. Geração de Respostas (operações do grafo)
// ─────────────────────────────────────────────────────────────────────────────

fn benchmark_resposta(state: &BrainState) {
    secao("C — GERAÇÃO DE RESPOSTAS (GRAFO DE ASSOCIAÇÕES)");

    let grafo    = &state.grafo_associacoes;
    let valencias = &state.palavra_valencias;
    let frases   = &state.frases_padrao;

    subsecao("C1 — Estatísticas do grafo carregado");
    let n_palavras = valencias.len();
    let n_nos      = grafo.len();
    let n_arestas: usize = grafo.values().map(|v| v.len()).sum();
    let n_frases   = frases.len();
    let grau_medio = if n_nos > 0 { n_arestas as f64 / n_nos as f64 } else { 0.0 };
    let grau_max   = grafo.values().map(|v| v.len()).max().unwrap_or(0);
    let grau_min   = grafo.values().map(|v| v.len()).min().unwrap_or(0);
    let nos_isolados = grafo.values().filter(|v| v.is_empty()).count();

    linha("Vocabulário total (tokens)", &format!("{n_palavras}"), "palavras");
    linha("Nós no grafo (com arestas)",  &format!("{n_nos}"), "nós");
    linha("Arestas totais",              &format!("{n_arestas}"), "associações");
    linha("Frases-padrão",               &format!("{n_frases}"), "frases");
    linha("Grau médio",                  &format!("{grau_medio:.2}"), "vizinhos/nó");
    linha("Grau máximo",                 &format!("{grau_max}"), "vizinhos");
    linha("Grau mínimo",                 &format!("{grau_min}"), "vizinhos");
    linha("Nós isolados (0 vizinhos)",   &format!("{nos_isolados}"), "nós");
    separador();

    subsecao("C2 — Velocidade de lookup no grafo (HashMap)");
    let palavras_chave: Vec<&str> = grafo.keys()
        .take(200)
        .map(|s| s.as_str())
        .collect();
    if palavras_chave.is_empty() {
        println!("  │  ⚠ Grafo vazio — skip");
        separador();
        return;
    }

    let n_lookups = 500_000;
    let t_lookup = Instant::now();
    let mut found = 0usize;
    for i in 0..n_lookups {
        let word = palavras_chave[i % palavras_chave.len()];
        if grafo.get(word).is_some() { found += 1; }
    }
    let lu_ns = t_lookup.elapsed().as_nanos() as f64;
    linha("Lookup HashMap<String, Vec<(String,f32)>>",
        &fmt_throughput(n_lookups as f64 / (lu_ns / 1e9)),
        &format!("({} por lookup)", fmt_ns(lu_ns / n_lookups as f64)));
    separador();

    subsecao("C3 — Simulação de walk no grafo (geração de resposta)");
    // Simula um walk de comprimento n_passos escolhendo sempre o vizinho
    // de menor distância emocional (como gerar_resposta_emergente)
    let emocao = 0.5f32;
    let n_walks = 10_000;
    let passos_por_walk = 10;
    let mut comprimento_total = 0usize;
    let mut diversidade_total = 0.0f32;

    let t_walk = Instant::now();
    for i in 0..n_walks {
        let mut visitados: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut atual = palavras_chave[i % palavras_chave.len()];
        let mut cadeia: Vec<&str> = Vec::with_capacity(passos_por_walk);

        for _ in 0..passos_por_walk {
            if visitados.contains(atual) { break; }
            visitados.insert(atual);
            cadeia.push(atual);

            if let Some(vizinhos) = grafo.get(atual) {
                let prox = vizinhos.iter()
                    .filter(|(w, p)| !visitados.contains(w.as_str()) && *p > -0.1)
                    .min_by(|a, b| {
                        let va = valencias.get(a.0.as_str()).copied().unwrap_or(emocao);
                        let vb = valencias.get(b.0.as_str()).copied().unwrap_or(emocao);
                        (va - emocao).abs().partial_cmp(&(vb - emocao).abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                match prox {
                    Some(entry) => atual = entry.0.as_str(),
                    None => break,
                }
            } else { break; }
        }

        comprimento_total += cadeia.len();
        // TTR: type-token ratio = unique words / total words
        let unique: std::collections::HashSet<&str> = cadeia.iter().copied().collect();
        if !cadeia.is_empty() {
            diversidade_total += unique.len() as f32 / cadeia.len() as f32;
        }
    }
    let walk_ns = t_walk.elapsed().as_nanos() as f64;
    let walk_per_s = n_walks as f64 / (walk_ns / 1e9);
    let comp_medio  = comprimento_total as f64 / n_walks as f64;
    let ttr_medio   = diversidade_total / n_walks as f32;

    linha("Walks por segundo",
        &fmt_throughput(walk_per_s),
        &format!("({} por walk de {} passos)", fmt_ns(walk_ns / n_walks as f64), passos_por_walk));
    linha("Comprimento médio do walk",
        &format!("{:.2}", comp_medio),
        "tokens por resposta");
    linha("TTR médio (diversidade lexical)",
        &format!("{:.3}", ttr_medio),
        "(1.0 = todos únicos, 0.0 = loop)");
    separador();

    subsecao("C4 — Scoring de frases-padrão (seleção de prefixo)");
    if frases.is_empty() {
        println!("  │  ⚠ Sem frases — skip");
        separador();
        return;
    }
    let inputs_teste = [
        "quem é você",
        "o que você sente agora",
        "você tem memória",
        "o que é consciência",
        "me conta sobre o amor",
        "você aprende",
    ];
    let t_prefix = Instant::now();
    let mut melhor_soma = 0usize;
    let n_prefix_rounds = 50_000;
    for round in 0..n_prefix_rounds {
        let input = inputs_teste[round % inputs_teste.len()].to_lowercase();
        let tokens: std::collections::HashSet<&str> = input
            .split_whitespace()
            .collect();
        let best_score = frases.iter()
            .map(|frase| frase.iter().filter(|w| tokens.contains(w.as_str())).count())
            .max()
            .unwrap_or(0);
        melhor_soma += best_score;
    }
    let prefix_ns = t_prefix.elapsed().as_nanos() as f64;
    linha(&format!("Prefix scoring ({} frases × {} inputs)", frases.len(), inputs_teste.len()),
        &fmt_throughput(n_prefix_rounds as f64 / (prefix_ns / 1e9)),
        &format!("({} por scoring)", fmt_ns(prefix_ns / n_prefix_rounds as f64)));

    // Analisa distribuição de sobreposição das frases com inputs reais
    let mut total_unique_words = 0usize;
    let mut frase_len_sum = 0usize;
    for frase in frases {
        frase_len_sum += frase.len();
        let unique: std::collections::HashSet<&str> = frase.iter().map(|w| w.as_str()).collect();
        total_unique_words += unique.len();
    }
    linha("Comprimento médio das frases",
        &format!("{:.2}", frase_len_sum as f64 / frases.len().max(1) as f64),
        "tokens/frase");
    separador();
}

// ─────────────────────────────────────────────────────────────────────────────
// D. Aprendizado
// ─────────────────────────────────────────────────────────────────────────────

fn benchmark_aprendizado(config: &Config) {
    secao("D — APRENDIZADO");

    subsecao("D1 — Crescimento do grafo de associações");
    // Simula o aprendizado de co-ocorrências de palavras
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

    let mut grafo_aprendido: HashMap<String, Vec<(String, f32)>> = HashMap::new();
    let mut aresta_contagem: HashMap<(String, String), u32> = HashMap::new();

    let n_rodadas = 1000;
    let t_learn = Instant::now();
    let mut arestas_novas = 0usize;

    for rodada in 0..n_rodadas {
        let frase = corpus[rodada % corpus.len()];
        let palavras: Vec<String> = frase.split_whitespace()
            .filter(|w| w.len() > 2)
            .map(|w| w.to_lowercase())
            .collect();

        // Janela deslizante de co-ocorrência (como o real)
        for i in 0..palavras.len() {
            for j in (i + 1)..palavras.len().min(i + 4) {
                let (a, b) = (&palavras[i], &palavras[j]);
                let chave = if a < b {
                    (a.clone(), b.clone())
                } else {
                    (b.clone(), a.clone())
                };
                let cnt = aresta_contagem.entry(chave.clone()).or_insert(0);
                *cnt += 1;
                let peso = (*cnt as f32 * 0.1).min(1.0);

                // Atualiza grafo bidirecional
                let entry = grafo_aprendido.entry(chave.0.clone()).or_default();
                if let Some(e) = entry.iter_mut().find(|(w, _)| w == &chave.1) {
                    e.1 = peso;
                } else {
                    entry.push((chave.1.clone(), peso));
                    arestas_novas += 1;
                }

                let entry2 = grafo_aprendido.entry(chave.1.clone()).or_default();
                if let Some(e) = entry2.iter_mut().find(|(w, _)| w == &chave.0) {
                    e.1 = peso;
                } else {
                    entry2.push((chave.0.clone(), peso));
                }
            }
        }
    }
    let learn_ns = t_learn.elapsed().as_nanos() as f64;

    let nos_finais    = grafo_aprendido.len();
    let arestas_finais: usize = grafo_aprendido.values().map(|v| v.len()).sum();
    let arestas_por_rodada = arestas_finais as f64 / n_rodadas as f64;

    linha("Velocidade de processamento do corpus",
        &fmt_throughput(n_rodadas as f64 / (learn_ns / 1e9)),
        "frases/s");
    linha("Nós criados após aprendizado",    &format!("{nos_finais}"), "nós únicos");
    linha("Arestas totais no grafo",         &format!("{arestas_finais}"), "associações");
    linha("Novidade de arestas",
        &format!("{:.2}", arestas_por_rodada),
        "arestas novas / rodada (converge com repetição)");
    separador();

    subsecao("D2 — Aprendizado Hebbiano na camada temporal");
    let n_neu = 256usize;
    let mut temporal = TemporalLobe::new(n_neu, 0.005, 0.1, config);

    // Mede convergência dos pesos Hebbianos
    let padroes_teste = [
        vec![1.0f32; n_neu],                              // tônico
        (0..n_neu).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect::<Vec<f32>>(), // alternado
        (0..n_neu).map(|i| (i as f32 / n_neu as f32)).collect::<Vec<f32>>(),           // gradiente
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
    let hebb_ns = t_hebb.elapsed().as_nanos() as f64;
    let hebb_per_s = n_ticks_hebb as f64 / (hebb_ns / 1e9);

    linha("Ticks Hebbianos (256 neurônios, K=8)",
        &fmt_throughput(hebb_per_s),
        &format!("({} por tick)", fmt_ns(hebb_ns / n_ticks_hebb as f64)));
    separador();

    subsecao("D3 — Chunking: emergência de padrões compostos");
    let mut chunker = ChunkingEngine::new(RegionType::Temporal);
    let n_chunk = 256usize;
    let mut camada_chunk = CamadaHibrida::new(
        n_chunk, "chunk_bench",
        TipoNeuronal::RS,
        Some((TipoNeuronal::FS, 0.2)),
        None, 40.0,
    );

    let n_ticks_chunk = 3000usize;
    let mut chunks_emergidos = 0usize;
    let mut ticks_ate_primeiro_chunk: Option<usize> = None;
    let inputs_chunk: Vec<f32> = (0..n_chunk)
        .map(|i| if i < n_chunk / 4 { 0.8 } else { 0.02 })
        .collect();

    let t_chunk = Instant::now();
    for tick in 0..n_ticks_chunk {
        let t_ms = tick as f32 * 5.0;
        // Padrão periódico: activo a cada 10 ticks para criar co-ativação
        let current_input: Vec<f32> = if tick % 10 < 5 {
            inputs_chunk.clone()
        } else {
            vec![0.01f32; n_chunk]
        };
        let spikes = camada_chunk.update(&current_input, 0.005, t_ms);
        let novos = chunker.registrar_spikes(&spikes, &camada_chunk, 0.3, t_ms);

        if !novos.is_empty() {
            if ticks_ate_primeiro_chunk.is_none() {
                ticks_ate_primeiro_chunk = Some(tick);
            }
            chunks_emergidos += novos.len();
        }
    }
    let chunk_ns = t_chunk.elapsed().as_nanos() as f64;

    let stats = chunker.stats();
    let total_chunks = stats.primitivos + stats.compostos + stats.sequencias;

    linha("Ticks de chunking (256 neurônios)",
        &fmt_throughput(n_ticks_chunk as f64 / (chunk_ns / 1e9)),
        &format!("({} por tick)", fmt_ns(chunk_ns / n_ticks_chunk as f64)));
    linha("Chunks total emergidos",
        &format!("{total_chunks}"), "chunks");
    linha("  Primitivos / Compostos / Sequências",
        &format!("{} / {} / {}", stats.primitivos, stats.compostos, stats.sequencias), "");
    match ticks_ate_primeiro_chunk {
        Some(t) => linha("Latência até 1º chunk",
            &format!("{:.1} ms", t as f32 * 5.0),
            &format!("(tick {})", t)),
        None => linha_warn("Latência até 1º chunk", "nenhum chunk emergiu"),
    }
    separador();

    subsecao("D4 — RL (Q-learning): crescimento da Q-table");
    use selene_kernel::learning::rl::ReinforcementLearning;

    let mut rl = ReinforcementLearning::new();
    let n_rl = 10_000usize;
    let t_rl = Instant::now();

    for i in 0..n_rl {
        // Padrões variados para criar estados distintos na Q-table
        let padrao: Vec<f32> = (0..64).map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0).collect();
        let dopamina = 0.3 + (i % 7) as f32 * 0.1;
        let acao = (i % 5) as f32 * 0.25;
        let _rpe = rl.update(&padrao, dopamina, acao, config);
    }
    let rl_ns = t_rl.elapsed().as_nanos() as f64;
    let rl_per_s = n_rl as f64 / (rl_ns / 1e9);

    // RPE final
    let padrao_final: Vec<f32> = (0..64).map(|j| j as f32 / 64.0).collect();
    let rpe = rl.update(&padrao_final, 0.8, 0.5, config);

    linha("Updates RL por segundo",
        &fmt_throughput(rl_per_s),
        &format!("({} por update)", fmt_ns(rl_ns / n_rl as f64)));
    linha("RPE após 10k updates (dopa=0.8)",
        &format!("{:.4}", rpe),
        "(positivo = recompensa acima da linha de base)");
    linha("Total de atualizações RL",
        &format!("{}", rl.total_atualizacoes()), "");
    separador();
}

// ─────────────────────────────────────────────────────────────────────────────
// E. Memória e Grounding
// ─────────────────────────────────────────────────────────────────────────────

fn benchmark_grounding(state: &mut BrainState) {
    secao("E — MEMÓRIA E GROUNDING SEMÂNTICO");

    subsecao("E1 — Taxa de binding grounding");
    let visual_ativo: SpikePattern  = [0xF0F0_F0F0u64; 8];
    let audio_ativo: SpikePattern   = [0x0F0F_0F0Fu64; 8];
    let visual_zero: SpikePattern   = [0u64; 8];
    let audio_zero: SpikePattern    = [0u64; 8];

    let palavras_bind = [
        vec!["quente".to_string(), "sol".to_string()],
        vec!["frio".to_string(), "inverno".to_string()],
        vec!["musica".to_string(), "ritmo".to_string()],
        vec!["dor".to_string(), "corpo".to_string()],
        vec!["alegria".to_string(), "luz".to_string()],
    ];

    let n_binds = 50_000;
    let t_bind = Instant::now();
    for i in 0..n_binds {
        let palavras = &palavras_bind[i % palavras_bind.len()];
        state.grounding_bind(
            palavras,
            visual_ativo, audio_ativo,
            0.5, 0.6, i as f64 * 10.0,
        );
    }
    let bind_ns = t_bind.elapsed().as_nanos() as f64;
    let bind_per_s = n_binds as f64 / (bind_ns / 1e9);

    linha("grounding_bind() por segundo",
        &fmt_throughput(bind_per_s),
        &format!("({} por bind)", fmt_ns(bind_ns / n_binds as f64)));

    // Verificar scores acumulados
    let g_quente = state.grounding.get("quente").copied().unwrap_or(0.0);
    let g_musica = state.grounding.get("musica").copied().unwrap_or(0.0);
    linha("Score grounding 'quente' após 10k binds",
        &format!("{:.4}", g_quente),
        "(esperado: próximo de 1.0 por saturação)");
    linha("Score grounding 'musica' após 10k binds",
        &format!("{:.4}", g_musica),
        "");
    separador();

    subsecao("E2 — Capacidade da memória episódica");
    let n_eventos = state.historico_episodico.len();
    let bytes_por_evento =
        std::mem::size_of::<EventoEpisodico>() +
        8 * std::mem::size_of::<String>(); // palavras Vec estimado
    let mem_episodica_kb = n_eventos * bytes_por_evento / 1024;
    let n_grounded = state.grounding.len();

    linha("Eventos episódicos em memória",
        &format!("{n_eventos}"), "eventos (máx 500)");
    linha("Memória estimada da fila episódica",
        &format!("{mem_episodica_kb}"), "KB");
    linha("Palavras com grounding ativo",
        &format!("{n_grounded}"), "tokens com score > 0");
    separador();

    subsecao("E3 — Decay de grounding (meia-vida)");
    // Mede quantos ticks de decay são necessários para reduzir 0.5 → 0.25
    state.grounding.insert("teste_decay".to_string(), 0.5);
    let mut ticks_para_meia_vida = 0usize;
    let mut g_atual = 0.5f32;

    // grounding_decay() faz *= 0.999 a cada chamada (equivale a cada 1000 ticks reais)
    // Meia-vida = ln(0.5) / ln(0.999) ≈ 692 chamadas de decay
    let t_decay = Instant::now();
    while g_atual > 0.25 && ticks_para_meia_vida < 10_000 {
        state.grounding.iter_mut().for_each(|(_, v)| *v *= 0.999);
        ticks_para_meia_vida += 1;
        g_atual = state.grounding.get("teste_decay").copied().unwrap_or(0.0);
    }
    let decay_ns = t_decay.elapsed().as_nanos() as f64;

    // Ticks reais = ticks_para_meia_vida × 1000 (decay chama a cada 1000 ticks reais)
    let ticks_reais = ticks_para_meia_vida * 1000;
    let segundos_meia_vida = ticks_reais as f64 / 200.0; // 200 Hz

    linha("Chamadas decay para meia-vida (0.5→0.25)",
        &format!("{ticks_para_meia_vida}"),
        "chamadas × 1000 ticks");
    linha("Meia-vida em ticks reais (200 Hz)",
        &format!("{ticks_reais}"), "ticks");
    linha("Meia-vida em tempo real",
        &format!("{:.1}", segundos_meia_vida),
        "segundos");
    linha("Throughput de decay",
        &fmt_throughput(ticks_para_meia_vida as f64 / (decay_ns / 1e9)),
        "ciclos/s");
    separador();
}

// ─────────────────────────────────────────────────────────────────────────────
// F. Capacidade do Sistema
// ─────────────────────────────────────────────────────────────────────────────

fn benchmark_capacidade(state: &BrainState) {
    secao("F — CAPACIDADE DO SISTEMA");

    subsecao("F1 — Vocabulário e grafo atual");
    let n_vocab    = state.palavra_valencias.len();
    let n_nos      = state.grafo_associacoes.len();
    let n_arestas: usize = state.grafo_associacoes.values().map(|v| v.len()).sum();
    let n_frases   = state.frases_padrao.len();
    let n_helix    = state.spike_vocab.len();

    let distribuicao_valencas = {
        let positivas = state.palavra_valencias.values().filter(|&&v| v > 0.1).count();
        let neutras   = state.palavra_valencias.values().filter(|&&v| v.abs() <= 0.1).count();
        let negativas = state.palavra_valencias.values().filter(|&&v| v < -0.1).count();
        (positivas, neutras, negativas)
    };

    let frase_palavras_total: usize = state.frases_padrao.iter().map(|f| f.len()).sum();

    linha("Vocabulário (tokens com valência)",    &format!("{n_vocab}"), "tokens");
    linha("Helix (padrões spike in-memory)",      &format!("{n_helix}"), "padrões");
    linha("Nós no grafo de associações",           &format!("{n_nos}"), "nós");
    linha("Arestas no grafo",                     &format!("{n_arestas}"), "associações");
    linha("Frases-padrão carregadas",              &format!("{n_frases}"), "frases");
    linha("Total palavras nas frases",             &format!("{frase_palavras_total}"), "tokens");
    linha("Distribuição de valências",
        &format!("+ {} / 0 {} / - {}",
            distribuicao_valencas.0, distribuicao_valencas.1, distribuicao_valencas.2),
        "(positiva/neutra/negativa)");
    separador();

    subsecao("F2 — Helix Store (arquivo mmap)");
    match &state.helix {
        Some(helix) => {
            let n_padroes = helix.len();
            let bytes_por_record = 96u64; // RECORD_SIZE do helix_store.rs
            let bytes_dados = n_padroes as u64 * bytes_por_record;
            let kb = bytes_dados / 1024;
            let mb = kb as f64 / 1024.0;

            // Capacidade teórica
            let max_por_gb = 1_073_741_824u64 / bytes_por_record;

            linha("Padrões armazenados no Helix",
                &format!("{n_padroes}"), "registros");
            linha("Espaço usado pelos padrões",
                &format!("{:.2} MB", mb),
                &format!("({kb} KB)"));
            linha("Throughput de nearest neighbor (helix)",
                "O(n) Jaccard scan", "por consulta");
            linha("Capacidade máxima por GB de disco",
                &format!("{max_por_gb}"),
                "padrões/GB");
            linha("Escalonamento esperado",
                &format!("{}M padrões/GB", max_por_gb / 1_000_000),
                "padrões por GB");
        }
        None => {
            linha_warn("Helix Store", "não carregado (arquivo .hlx ausente ou desativado)");
        }
    }
    separador();

    subsecao("F3 — Neurônios por zona (pipeline 128 neurônios)");
    // Contagens estimadas baseadas nas definições dos lobos (config N=128)
    let n_base = 128usize;
    struct ZonaInfo { nome: &'static str, n: usize, tipos: &'static str }
    let zonas = [
        ZonaInfo { nome: "Occipital (V1+V2)",    n: n_base,     tipos: "60% RS + 40% CH / 70% CH + 30% RS" },
        ZonaInfo { nome: "Parietal",              n: n_base,     tipos: "RS + LT" },
        ZonaInfo { nome: "Temporal",              n: n_base,     tipos: "55% RS + 30% CH + 15% FS" },
        ZonaInfo { nome: "Frontal (exec+inhib)",  n: n_base * 2, tipos: "80% RS + 20% FS / 100% FS" },
        ZonaInfo { nome: "Límbico",               n: n_base / 2, tipos: "IB + FS" },
        ZonaInfo { nome: "Hipocampo (CA1+CA3)",   n: n_base / 2, tipos: "80% RS + 20% LT / 70% RS + 30% RZ" },
        ZonaInfo { nome: "Cerebelo",              n: n_base / 4, tipos: "RZ (Purkinje)" },
        ZonaInfo { nome: "Tálamo",                n: 0,          tipos: "relay sem modelo spike" },
        ZonaInfo { nome: "Tronco cerebral",       n: 0,          tipos: "relay sem modelo spike" },
    ];
    let total_spike: usize = zonas.iter().map(|z| z.n).sum();
    for z in &zonas {
        if z.n > 0 {
            linha(&format!("  {}", z.nome),
                &format!("{} neurônios", z.n),
                &format!("({})", z.tipos));
        } else {
            linha(&format!("  {}", z.nome), "relay", z.tipos);
        }
    }
    linha("TOTAL neurônios spike ativos", &format!("{total_spike}"), "neurônios por instância");
    separador();

    subsecao("F4 — Capacidade de criação de novos neurônios");
    // A arquitetura Selene usa CamadaHibrida com tamanho fixo definido em new()
    // Novos "neurônios" semânticos são criados via grafo (novos nós/arestas)
    // e padrões spike (helix). Aqui medimos a velocidade de cada mecanismo.

    linha("Modelo de crescimento neural",
        "grafo de associações", "(nós/arestas = abstração de sinapses)");
    linha("Criação de novo nó semântico",
        "HashMap::insert O(1)", "(amortizado)");
    linha("Criação de nova sinapse",
        "Vec::push O(1)", "(por nó)");
    linha("Limite prático do grafo",
        "memória RAM disponível",
        &format!("(atual: {} nós, {} arestas)", n_nos, n_arestas));

    let bits_por_padrao = N_NEURONS;
    let bytes_por_padrao = bits_por_padrao / 8;
    let vocab_mem_kb = n_helix * bytes_por_padrao / 1024;
    linha("Memória dos padrões spike (in-memory)",
        &format!("{:.2} KB", vocab_mem_kb as f64),
        &format!("({} padrões × {} bytes)", n_helix, bytes_por_padrao));

    let max_helix_1gb = 1_073_741_824usize / 96;
    linha("Máx padrões Helix em 1 GB (disco mmap)",
        &format!("{:.1}M", max_helix_1gb as f64 / 1_000_000.0),
        "padrões");
    separador();
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        SELENE BRAIN 2.0 — BENCHMARK INTENSIVO                   ║");
    println!("║        Data: 2026-03-27  |  Plataforma: Windows 11 x64          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let config = Config::new(ModoOperacao::Normal);

    // Carrega BrainState com dados reais (linguagem, helix, etc.)
    println!("\n⏳ Inicializando BrainState com dados reais...");
    let swap  = Arc::new(TokioMutex::new(SwapManager::new(256, 3600)));
    let flags = SensorFlags::new_desativados();
    let mut state = BrainState::new(swap, &config, flags);
    println!("✓ BrainState carregado.");

    // ── Seções de benchmark ──────────────────────────────────────────────────
    benchmark_neural(&config);
    benchmark_spikes();
    benchmark_resposta(&state);
    benchmark_aprendizado(&config);
    benchmark_grounding(&mut state);
    benchmark_capacidade(&state);

    // ── Resumo final ─────────────────────────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                   BENCHMARK CONCLUÍDO                           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
}
