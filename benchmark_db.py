#!/usr/bin/env python3
# benchmark_db.py — Benchmark de desempenho do banco de dados da Selene
#
# Mede:
#   - Velocidade de leitura/parse do grafo e vocabulário
#   - Throughput de lookup de palavras
#   - Velocidade de busca de links por nó
#   - Taxa de frames FFT que podem ser processados por segundo (simulação)
#   - Tamanho e fragmentação do RocksDB
#   - Estimativa de capacidade: quantos minutos de áudio cabem no DB atual
#
# Uso:
#   python benchmark_db.py              # benchmark completo
#   python benchmark_db.py --rapido     # apenas métricas rápidas
#   python benchmark_db.py --n 50000    # n iterações para lookups

import json
import os
import sys
import time
import random
import struct
from pathlib import Path

# ─── Configuração ─────────────────────────────────────────────────────────────

BASE       = Path(__file__).parent
DB_PATH    = BASE / "selene_memories.db"
GRAFO_PATH = BASE / "grafo_selene.json"
LANG_PATH  = BASE / "selene_linguagem.json"
HIPPO_PATH = BASE / "selene_hippo_ltp.json"

N_ITER    = 100_000
RAPIDO    = "--rapido" in sys.argv

for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--n" and i + 1 < len(sys.argv):
        try: N_ITER = int(sys.argv[i + 1])
        except: pass

# ─── Utilitários ──────────────────────────────────────────────────────────────

def fmt(n_bytes: int) -> str:
    if n_bytes < 1024:     return f"{n_bytes} B"
    if n_bytes < 1024**2:  return f"{n_bytes/1024:.1f} KB"
    return f"{n_bytes/1024**2:.2f} MB"

def linha(c="─", n=60):
    print(c * n)

def medir(label: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"  {label:<45} {dt*1000:>8.2f} ms")
    return result, dt

# ─── 1. Estado dos arquivos ───────────────────────────────────────────────────

def bench_arquivos():
    linha("═")
    print("  ESTADO DO BANCO DE DADOS")
    linha("═")

    arquivos = {
        "selene_memories.db (RocksDB)": DB_PATH,
        "grafo_selene.json":            GRAFO_PATH,
        "selene_linguagem.json":        LANG_PATH,
        "selene_hippo_ltp.json":        HIPPO_PATH,
        "selene_qtable.bin":            BASE / "selene_qtable.bin",
    }

    total = 0
    for nome, path in arquivos.items():
        if path.is_dir():
            sz = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            nf = sum(1 for _ in path.rglob("*") if _.is_file())
            total += sz
            print(f"  {nome:<35} {fmt(sz):>10}  ({nf} arquivos)")
        elif path.is_file():
            sz = path.stat().st_size
            total += sz
            print(f"  {nome:<35} {fmt(sz):>10}")
        else:
            print(f"  {nome:<35} {'(não encontrado)':>10}")

    linha()
    print(f"  {'TOTAL':35} {fmt(total):>10}")

    if DB_PATH.is_dir():
        ssts = list(DB_PATH.glob("*.sst"))
        logs = list(DB_PATH.glob("LOG.old.*"))
        print(f"\n  RocksDB: {len(ssts)} SST file(s), {len(logs)} LOG.old (fragmentação)")
        if logs:
            print(f"  ⚠  {len(logs)} logs antigos — execute limpar_db.py para compactar")

# ─── 2. Benchmark de leitura ─────────────────────────────────────────────────

def bench_leitura():
    linha()
    print("  BENCHMARK DE LEITURA / PARSE")
    linha()

    # Grafo
    if GRAFO_PATH.exists():
        with open(GRAFO_PATH, "rb") as f:
            raw = f.read()
        def parse_grafo():
            return json.loads(raw)
        (grafo, _) = medir("Parse grafo_selene.json", parse_grafo)[0], None
        grafo, dt_grafo = medir("Parse grafo_selene.json", parse_grafo)
    else:
        grafo = {"nodes": [], "links": []}
        print("  grafo_selene.json não encontrado")

    # Vocabulário
    if LANG_PATH.exists():
        with open(LANG_PATH, "rb") as f:
            raw_lang = f.read()
        def parse_lang():
            return json.loads(raw_lang)
        lang, dt_lang = medir("Parse selene_linguagem.json", parse_lang)
    else:
        lang = {}
        print("  selene_linguagem.json não encontrado")

    # Hippo
    if HIPPO_PATH.exists():
        with open(HIPPO_PATH, "rb") as f:
            raw_hippo = f.read()
        hippo, _ = medir("Parse selene_hippo_ltp.json", lambda: json.loads(raw_hippo))
    else:
        hippo = {}

    return grafo, lang

# ─── 3. Benchmark de lookup ───────────────────────────────────────────────────

def bench_lookup(grafo: dict, lang: dict):
    linha()
    print(f"  BENCHMARK DE LOOKUP  ({N_ITER:,} iterações)")
    linha()

    # Prepara índice de nós → links
    nodes = [n["id"] for n in grafo.get("nodes", [])]
    links = grafo.get("links", [])
    if not nodes:
        print("  Grafo vazio — pulando lookups.")
        return

    # Índice adjacência
    t0 = time.perf_counter()
    adjacencia: dict[str, list] = {}
    for l in links:
        src = l.get("source", "")
        tgt = l.get("target", "")
        adjacencia.setdefault(src, []).append(tgt)
        adjacencia.setdefault(tgt, []).append(src)
    dt_idx = time.perf_counter() - t0
    print(f"  {'Construção índice adjacência':<45} {dt_idx*1000:>8.2f} ms")

    # Lookup aleatório de nó → vizinhos
    t0 = time.perf_counter()
    n_total = 0
    for _ in range(N_ITER):
        no = nodes[_ % len(nodes)]
        vizinhos = adjacencia.get(no, [])
        n_total += len(vizinhos)
    dt_lookup = time.perf_counter() - t0
    ops = N_ITER / dt_lookup
    print(f"  {'Lookup nó→vizinhos (dict)':<45} {dt_lookup*1000:>8.2f} ms  →  {ops:,.0f} ops/s")

    # Lookup no vocabulário
    chave = next(iter(lang), None)
    if chave:
        vocab = lang[chave].get("vocabulario", {})
        palavras = list(vocab.keys())
        if palavras:
            t0 = time.perf_counter()
            for i in range(N_ITER):
                _ = vocab.get(palavras[i % len(palavras)], 0.0)
            dt_vocab = time.perf_counter() - t0
            ops_v = N_ITER / dt_vocab
            print(f"  {'Lookup palavra→peso (dict)':<45} {dt_vocab*1000:>8.2f} ms  →  {ops_v:,.0f} ops/s")

    # Busca de link específico (source == X)
    if links:
        amostra = [l["source"] for l in links[:100]]
        t0 = time.perf_counter()
        n = min(N_ITER // 10, 10_000)
        for i in range(n):
            src = amostra[i % len(amostra)]
            _ = [l for l in links if l.get("source") == src]
        dt_scan = time.perf_counter() - t0
        ops_s = n / dt_scan
        print(f"  {'Scan linear links por source':<45} {dt_scan*1000:>8.2f} ms  →  {ops_s:,.0f} ops/s  (scan, lento)")

# ─── 4. Simulação de throughput FFT ──────────────────────────────────────────

def bench_fft_simulado():
    linha()
    print("  SIMULAÇÃO: THROUGHPUT DE FRAMES FFT")
    linha()

    try:
        import numpy as np
        FRAME_SAMPLES = 551   # 25ms @ 22050Hz
        N_BINS = 128
        N_FRAMES = 10_000

        # Gera frames aleatórios
        frames = [np.random.randn(FRAME_SAMPLES).astype(np.float32) for _ in range(N_FRAMES)]

        t0 = time.perf_counter()
        resultados = []
        for frame in frames:
            hann   = np.hanning(len(frame))
            fft_m  = np.abs(np.fft.rfft(frame * hann))
            mx     = fft_m.max()
            if mx > 1e-9:
                fft_m /= mx
            resultados.append(fft_m[:N_BINS].tolist())
        dt = time.perf_counter() - t0

        fps = N_FRAMES / dt
        dur_audio_s = N_FRAMES * 0.025
        print(f"  {N_FRAMES:,} frames FFT processados em {dt*1000:.1f} ms")
        print(f"  Throughput: {fps:,.0f} frames/s  ({fps*0.025:.1f}x tempo real)")
        print(f"  = capaz de processar {fps*0.025*60:.0f} min de áudio por minuto real")

        # Benchmark de serialização JSON (custo do send)
        sample = resultados[0]
        t0 = time.perf_counter()
        for _ in range(10_000):
            _ = json.dumps({"action": "learn_audio_fft", "fft": sample, "duracao_ms": 25})
        dt_json = time.perf_counter() - t0
        ops_json = 10_000 / dt_json
        print(f"  Serialização JSON por frame: {dt_json/10:.2f} ms/frame  ({ops_json:,.0f} ops/s)")

    except ImportError:
        print("  numpy não disponível — pulando benchmark FFT")

# ─── 5. Capacidade estimada ───────────────────────────────────────────────────

def bench_capacidade(grafo: dict, lang: dict):
    linha()
    print("  CAPACIDADE E ESTIMATIVAS")
    linha()

    nodes = grafo.get("nodes", [])
    links = grafo.get("links", [])

    chave = next(iter(lang), None)
    vocab = lang.get(chave, {}).get("vocabulario", {}) if chave else {}

    print(f"  Nós no grafo:              {len(nodes):>8,}")
    print(f"  Links no grafo:            {len(links):>8,}")
    print(f"  Palavras no vocabulário:   {len(vocab):>8,}")
    if links and nodes:
        grau_medio = len(links) * 2 / len(nodes)
        print(f"  Grau médio por nó:         {grau_medio:>8.1f} conexões")
    if len(vocab) > 0 and len(nodes) > 0:
        cobertura = len(vocab) / len(nodes) * 100
        print(f"  Cobertura vocab/grafo:     {cobertura:>8.1f}%")

    # Estimativa: frames de 25ms @ 22050Hz com 128 bins float32
    bytes_por_frame_bin   = 128 * 8   # 128 floats JSON ~= 8 bytes cada
    frames_por_min_audio  = 60 / 0.025  # 2400 frames/min
    bytes_por_min_audio   = bytes_por_frame_min = frames_por_min_audio * bytes_por_frame_bin
    if DB_PATH.is_dir():
        db_sz = sum(f.stat().st_size for f in DB_PATH.rglob("*") if f.is_file())
        mins_estimados = db_sz / bytes_por_min_audio
        print(f"\n  RocksDB atual:             {fmt(db_sz):>10}")
        print(f"  Áudio equivalente aprox:   {mins_estimados:>8.1f} min @ 128 bins/frame")
        print(f"  (1 min de áudio ≈ {fmt(int(bytes_por_min_audio))} no DB)")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    linha("═")
    print("  BENCHMARK — SELENE BRAIN 2.0  DATABASE")
    linha("═")

    bench_arquivos()
    grafo, lang = bench_leitura()

    if not RAPIDO:
        bench_lookup(grafo, lang)
        bench_fft_simulado()

    bench_capacidade(grafo, lang)

    linha("═")
    print("  Benchmark concluído.")
    linha("═")

if __name__ == "__main__":
    main()
