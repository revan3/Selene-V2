// src/bin/benchmark.rs — Selene Brain 2.0
// Benchmark de velocidade neural e aprendizado por chunking.
//
// Execução:
//   cargo run --bin benchmark --release
//
// Métricas:
//   1. TICK SPEED      — ticks/s brutos da CamadaHibrida (1024 neurônios)
//   2. CHUNK DETECT    — ticks até detectar o 1º chunk num padrão fixo
//   3. STDP RPE        — variação de forca_stdp após ciclos positivo/negativo
//   4. SCALE           — throughput para 4k, 16k, 64k neurônios

#![allow(unused_imports)]
#![allow(dead_code)]

use std::time::{Duration, Instant};
use selene_kernel::synaptic_core::{CamadaHibrida, TipoNeuronal};
use selene_kernel::brain_zones::RegionType;
use selene_kernel::learning::chunking::ChunkingEngine;

// ── helpers ──────────────────────────────────────────────────────────────────

fn camada(n: usize) -> CamadaHibrida {
    CamadaHibrida::new(n, "bench", TipoNeuronal::RS, None, None, 1.0)
}

fn spikes_fixos(ativos: &[usize], total: usize) -> Vec<bool> {
    (0..total).map(|i| ativos.contains(&i)).collect()
}

fn hr(n: usize) -> String {
    if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1e6) }
    else if n >= 1_000 { format!("{:.1}k", n as f64 / 1e3) }
    else { n.to_string() }
}

fn sep() { println!("{}", "─".repeat(60)); }

// ── BENCH 1: tick speed bruta ─────────────────────────────────────────────

fn bench_tick_speed(n: usize, duracao_secs: f64) -> f64 {
    let mut camada = camada(n);
    let input = vec![0.3f32; n];
    let dt = 0.005f32;

    let start = Instant::now();
    let mut ticks: u64 = 0;
    while start.elapsed().as_secs_f64() < duracao_secs {
        camada.update(&input, dt, ticks as f32 * dt * 1000.0);
        ticks += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();
    ticks as f64 / elapsed
}

// ── BENCH 2: ticks até detectar 1º chunk ────────────────────────────────────

fn bench_chunk_detection(n: usize) -> (u64, Duration) {
    let mut engine = ChunkingEngine::new(RegionType::Temporal);
    let mut camada = camada(n);
    // Padrão fixo: neurônios 10, 20, 30 disparam sempre juntos
    let padrao: Vec<usize> = vec![10, 20, 30];
    let dt = 0.005f32;
    let start = Instant::now();

    for tick in 0u64.. {
        let t_ms = tick as f32 * dt * 1000.0;
        // Injeta sinal forte nos neurônios do padrão
        let mut input = vec![0.0f32; n];
        for &idx in &padrao { input[idx] = 2.0; }
        let spikes_raw = camada.update(&input, dt, t_ms);
        let novos = engine.registrar_spikes(&spikes_raw, &camada, 0.5, t_ms);
        if !novos.is_empty() {
            return (tick, start.elapsed());
        }
        // Timeout de segurança: 100k ticks
        if tick > 100_000 { return (tick, start.elapsed()); }
    }
    unreachable!()
}

// ── BENCH 3: convergência STDP com RPE ──────────────────────────────────────

fn bench_rpe_convergence(n: usize) -> (f32, f32) {
    let mut engine = ChunkingEngine::new(RegionType::Temporal);
    let mut camada = camada(n);
    let padrao: Vec<usize> = vec![5, 15];
    let dt = 0.005f32;

    // Fase 1: treina o chunk
    for tick in 0u64..10_000 {
        let t_ms = tick as f32 * dt * 1000.0;
        let mut input = vec![0.0f32; n];
        for &idx in &padrao { input[idx] = 2.0; }
        let spikes_raw = camada.update(&input, dt, t_ms);
        engine.registrar_spikes(&spikes_raw, &camada, 0.5, t_ms);
    }

    let forca_antes = engine.chunks.first().map(|c| c.forca_stdp).unwrap_or(0.0);

    // Fase 2: RPE positivo (recompensa)
    for _ in 0..50 { engine.aplicar_rpe(1.0); }
    let forca_pos = engine.chunks.first().map(|c| c.forca_stdp).unwrap_or(0.0);

    // Fase 3: RPE negativo (punição)
    for _ in 0..100 { engine.aplicar_rpe(-1.0); }
    let forca_neg = engine.chunks.first().map(|c| c.forca_stdp).unwrap_or(0.0);

    println!("   força STDP: antes={:.4}  após recompensa={:.4}  após punição={:.4}",
        forca_antes, forca_pos, forca_neg);

    (forca_pos - forca_antes, forca_pos - forca_neg)
}

// ── BENCH 4: escala de neurônios ─────────────────────────────────────────────

fn bench_scale() {
    let tamanhos = [1_024usize, 4_096, 16_384, 65_536];
    println!("   {:>8}  {:>14}  {:>16}", "Neurônios", "Ticks/s", "ns/tick");
    println!("   {:>8}  {:>14}  {:>16}", "─────────", "──────────────", "────────────────");
    for &n in &tamanhos {
        let tps = bench_tick_speed(n, 1.0);
        let ns_per_tick = 1e9 / tps;
        println!("   {:>8}  {:>14.0}  {:>14.1} ns", hr(n), tps, ns_per_tick);
    }
}

// ── MAIN ─────────────────────────────────────────────────────────────────────

fn main() {
    println!("\n{}", "═".repeat(60));
    println!("  🧠 SELENE BRAIN 2.0 — BENCHMARK DE APRENDIZADO E VELOCIDADE");
    println!("{}\n", "═".repeat(60));

    // ── 1. Tick speed (1024 neurônios, 3s) ──────────────────────────────────
    sep();
    println!("📊 BENCH 1 — Velocidade Neural Bruta (n=1024, t=3s)");
    sep();
    let tps = bench_tick_speed(1024, 3.0);
    let ns = 1e9 / tps;
    println!("   Ticks/segundo  : {:.0}", tps);
    println!("   Tempo/tick     : {:.2} µs  ({:.0} ns)", ns / 1000.0, ns);
    println!("   Equivalente    : {:.1}x faster que 200 Hz", tps / 200.0);
    let headroom_pct = ((tps - 200.0) / tps * 100.0).max(0.0);
    println!("   Headroom livre : {:.1}%", headroom_pct);

    // ── 2. Chunk detection ────────────────────────────────────────────────
    sep();
    println!("📊 BENCH 2 — Detecção de Padrão / Emergência de Chunk (n=1024)");
    sep();
    let (ticks, dur) = bench_chunk_detection(1024);
    if ticks < 100_000 {
        println!("   Ticks até 1º chunk : {}", ticks);
        println!("   Tempo real         : {:.2} ms", dur.as_secs_f64() * 1000.0);
        println!("   Velocidade aprend. : {:.0} padrões/s", 1.0 / dur.as_secs_f64());
    } else {
        println!("   ⚠️  Chunk não emergiu em 100k ticks (trace_pre muito baixo)");
        println!("   Isso é esperado: neurônios precisam de sinal > threshold para disparar.");
    }

    // ── 3. Convergência STDP/RPE ──────────────────────────────────────────
    sep();
    println!("📊 BENCH 3 — Convergência STDP + Reforço por RPE (n=1024)");
    sep();
    let (delta_pos, delta_pun) = bench_rpe_convergence(1024);
    println!("   ΔForça (recompensa) : {:+.4}", delta_pos);
    println!("   ΔForça (punição)    : {:+.4}", -delta_pun);
    let aprendeu = delta_pos > 0.0;
    let esqueceu = delta_pun < 0.0;
    println!("   Reforço positivo : {}", if aprendeu { "✅ funcional" } else { "❌ sem efeito" });
    println!("   Extinção negativa: {}", if esqueceu { "✅ funcional" } else { "❌ sem efeito" });

    // ── 4. Escala ─────────────────────────────────────────────────────────
    sep();
    println!("📊 BENCH 4 — Escala de Neurônios (1s cada)");
    sep();
    bench_scale();

    // ── Resumo ────────────────────────────────────────────────────────────
    println!("\n{}", "═".repeat(60));
    println!("  ✅ Benchmark concluído");
    println!("{}\n", "═".repeat(60));
}
