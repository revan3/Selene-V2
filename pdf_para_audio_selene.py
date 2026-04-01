#!/usr/bin/env python3
# pdf_para_audio_selene.py — Lê PDF em voz alta para a Selene Brain 2.0
#
# Extrai texto de PDF, sintetiza com espeak-ng e envia FFT para Selene.
# NUNCA envia texto — apenas parâmetros físicos de frequência.
#
# Fluxo:
#   PDF → texto (PyMuPDF) → limpeza → espeak-ng WAV → FFT frames → WebSocket
#
# Requisitos:
#   pip install pymupdf websockets scipy numpy
#   espeak-ng instalado e no PATH
#   (Windows: https://github.com/espeak-ng/espeak-ng/releases)
#
# Uso:
#   python pdf_para_audio_selene.py livro.pdf
#   python pdf_para_audio_selene.py livro.pdf --pagina 10       # começa na pág 10
#   python pdf_para_audio_selene.py livro.pdf --ate 50          # até pág 50
#   python pdf_para_audio_selene.py livro.pdf --lang en         # inglês
#   python pdf_para_audio_selene.py livro.pdf --pausa 0.005     # mais rápido
#   python pdf_para_audio_selene.py livro.pdf --host 192.168.1.10

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly
except ImportError:
    print("Instale: pip install numpy scipy")
    sys.exit(1)

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Instale: pip install pymupdf")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("Instale: pip install websockets")
    sys.exit(1)

# ─── Configuração ─────────────────────────────────────────────────────────────

HOST         = "127.0.0.1"
PORTA        = 3030
WS_URL       = f"ws://{HOST}:{PORTA}/selene"
FRAME_MS     = 25
PAUSA        = 0.010
SILENCIO_DB  = -42.0
MAX_BINS     = 128
SAMPLE_RATE  = 22050
LANG_TTS     = "pt-br"
PAG_INICIO   = 1
PAG_FIM      = None      # None = até o fim
CHUNK_CHARS  = 500       # caracteres por chunk de síntese (frases curtas = melhor TTS)

for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--host"    and i+1 < len(sys.argv): HOST = sys.argv[i+1]; WS_URL = f"ws://{HOST}:{PORTA}/selene"
    if arg == "--lang"    and i+1 < len(sys.argv): LANG_TTS = sys.argv[i+1]
    if arg == "--pausa"   and i+1 < len(sys.argv):
        try: PAUSA = float(sys.argv[i+1])
        except: pass
    if arg == "--pagina"  and i+1 < len(sys.argv):
        try: PAG_INICIO = int(sys.argv[i+1])
        except: pass
    if arg == "--ate"     and i+1 < len(sys.argv):
        try: PAG_FIM = int(sys.argv[i+1])
        except: pass
    if arg == "--chunk"   and i+1 < len(sys.argv):
        try: CHUNK_CHARS = int(sys.argv[i+1])
        except: pass

# ─── Extração de texto do PDF ─────────────────────────────────────────────────

def limpar_texto(texto: str) -> str:
    """Remove ruídos comuns de PDFs: hifenação, números de página, cabeçalhos."""
    # Junta hifenação no final de linha (pa-\nlavra → palavra)
    texto = re.sub(r"-\n(\w)", r"\1", texto)
    # Normaliza quebras de linha múltiplas em parágrafo
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    # Remove linhas que são só números (número de página)
    texto = re.sub(r"^\s*\d+\s*$", "", texto, flags=re.MULTILINE)
    # Remove sequências de espaços excessivos
    texto = re.sub(r"[ \t]{2,}", " ", texto)
    # Remove caracteres de controle
    texto = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", texto)
    return texto.strip()

def extrair_texto_pdf(caminho: str, pag_inicio: int, pag_fim: int | None) -> list[tuple[int, str]]:
    """
    Retorna lista de (numero_pagina, texto) por página.
    pag_inicio e pag_fim são 1-based.
    """
    doc   = fitz.open(caminho)
    total = len(doc)
    fim   = min(pag_fim, total) if pag_fim else total
    inicio = max(1, pag_inicio)

    paginas = []
    for num in range(inicio, fim + 1):
        page = doc[num - 1]
        texto = page.get_text("text")
        texto = limpar_texto(texto)
        if len(texto.strip()) > 10:   # ignora páginas praticamente vazias
            paginas.append((num, texto))

    doc.close()
    return paginas

def dividir_em_chunks(texto: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Divide texto em chunks por frases completas, respeitando max_chars."""
    # Divide por pontuação forte
    frases = re.split(r'(?<=[.!?;:])\s+', texto)
    chunks = []
    atual  = ""
    for frase in frases:
        if not frase.strip():
            continue
        if len(atual) + len(frase) + 1 <= max_chars:
            atual = (atual + " " + frase).strip()
        else:
            if atual:
                chunks.append(atual)
            # frase muito longa? divide por vírgula
            if len(frase) > max_chars:
                sub = re.split(r'(?<=,)\s+', frase)
                sub_atual = ""
                for s in sub:
                    if len(sub_atual) + len(s) + 1 <= max_chars:
                        sub_atual = (sub_atual + " " + s).strip()
                    else:
                        if sub_atual: chunks.append(sub_atual)
                        sub_atual = s
                if sub_atual: chunks.append(sub_atual)
                atual = ""
            else:
                atual = frase
    if atual:
        chunks.append(atual)
    return chunks

# ─── TTS via espeak-ng ────────────────────────────────────────────────────────

def sintetizar_espeak(texto: str, lang: str) -> tuple | None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "espeak-ng",
            "-v", lang,
            "-s", "150",    # palavras/minuto
            "-p", "50",     # pitch neutro
            "-a", "180",
            "-w", tmp_path,
            texto,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode != 0:
            return None
        sr, data = wavfile.read(tmp_path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, int(sr)
    except Exception:
        return None
    finally:
        try: os.unlink(tmp_path)
        except: pass

# ─── FFT ──────────────────────────────────────────────────────────────────────

def energia_db(frame: np.ndarray) -> float:
    rms = np.sqrt(np.mean(frame ** 2))
    return -100.0 if rms < 1e-10 else 20.0 * np.log10(rms)

def frame_para_bins(frame: np.ndarray, sr: int) -> list | None:
    if energia_db(frame) < SILENCIO_DB:
        return None
    n, hann = len(frame), np.hanning(len(frame))
    fft_mag  = np.abs(np.fft.rfft(frame * hann))
    freqs    = np.fft.rfftfreq(n, 1 / sr)
    mx = fft_mag.max()
    if mx < 1e-9:
        return None
    fft_norm = fft_mag / mx
    mask = (freqs >= 80.0) & (freqs <= 8000.0)
    f_f, a_f = freqs[mask], fft_norm[mask]
    if len(f_f) > MAX_BINS:
        idx  = np.round(np.linspace(0, len(f_f) - 1, MAX_BINS)).astype(int)
        f_f  = f_f[idx]; a_f = a_f[idx]
    return [[float(f), float(a)] for f, a in zip(f_f, a_f)]

def amostras_para_frames(samples: np.ndarray, sr: int):
    frame_size = int(sr * FRAME_MS / 1000)
    if frame_size < 8:
        return
    for start in range(0, len(samples) - frame_size, frame_size):
        bins = frame_para_bins(samples[start:start + frame_size], sr)
        if bins:
            yield bins

def re_amostrar(samples: np.ndarray, sr_orig: int, sr_alvo: int) -> np.ndarray:
    if sr_orig == sr_alvo:
        return samples
    from math import gcd
    g = gcd(sr_alvo, sr_orig)
    return resample_poly(samples, sr_alvo // g, sr_orig // g).astype(np.float32)

# ─── Barra de progresso ───────────────────────────────────────────────────────

def barra(n: int, total: int, w: int = 35) -> str:
    p = n / total if total else 0
    c = int(p * w)
    return f"[{'=' * c}{' ' * (w - c)}] {n}/{total} ({p*100:.0f}%)"

# ─── Processamento de um PDF ──────────────────────────────────────────────────

async def processar_pdf(caminho: str):
    nome = Path(caminho).name
    print(f"\n{'='*60}")
    print(f"  PDF: {nome}")
    print(f"{'='*60}")

    # Extrai texto
    print(f"  Extraindo texto (pág {PAG_INICIO}→{PAG_FIM or 'fim'})...", end="", flush=True)
    try:
        paginas = extrair_texto_pdf(caminho, PAG_INICIO, PAG_FIM)
    except Exception as e:
        print(f"\n  Erro ao abrir PDF: {e}")
        return

    total_chars = sum(len(t) for _, t in paginas)
    total_pags  = len(paginas)
    print(f" {total_pags} páginas | {total_chars:,} caracteres")

    if total_pags == 0:
        print("  Nenhum texto extraído (PDF digitalizado/imagem?).")
        return

    # Conta chunks totais
    todos_chunks: list[tuple[int, str]] = []
    for num_pag, texto in paginas:
        for chunk in dividir_em_chunks(texto):
            todos_chunks.append((num_pag, chunk))

    print(f"  {len(todos_chunks)} chunks de texto | lang: {LANG_TTS}")

    async with websockets.connect(
        WS_URL, ping_interval=None, max_queue=512, open_timeout=15
    ) as ws:
        print(f"  Conectado! Lendo PDF em voz alta para Selene...")
        t0       = time.time()
        enviados = 0
        acks     = 0
        erros    = 0
        chunks_processados = 0

        for num_pag, chunk in todos_chunks:
            chunks_processados += 1

            resultado = sintetizar_espeak(chunk, LANG_TTS)
            if resultado is None:
                erros += 1
                if erros <= 3:
                    print(f"\n  Chunk falhou na síntese (chunk {chunks_processados})")
                continue

            samples, sr = resultado
            if sr != SAMPLE_RATE:
                samples = re_amostrar(samples, sr, SAMPLE_RATE)
                sr = SAMPLE_RATE

            for bins in amostras_para_frames(samples, sr):
                try:
                    msg = json.dumps({
                        "action":     "learn_audio_fft",
                        "fft":        bins,
                        "duracao_ms": FRAME_MS,
                        "referencia": f"{nome}:p{num_pag}",
                    })
                    await ws.send(msg)
                    for _ in range(3):
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                            if json.loads(raw).get("event") == "audio_ack":
                                acks += 1
                                break
                        except (asyncio.TimeoutError, json.JSONDecodeError):
                            break
                    enviados += 1
                    await asyncio.sleep(PAUSA)
                except websockets.exceptions.ConnectionClosed:
                    raise
                except Exception as e:
                    erros += 1
                    if erros <= 3:
                        print(f"\n  Erro frame: {e}")

            # Progresso a cada 10 chunks
            if chunks_processados % 10 == 0:
                elapsed = time.time() - t0
                print(f"\r  Pág {num_pag}  {barra(chunks_processados, len(todos_chunks))}"
                      f"  {enviados} frames", end="", flush=True)

        elapsed = time.time() - t0
        print(f"\n  Concluído: {enviados} frames | {acks} acks | {erros} erros | {elapsed:.1f}s")
        print(f"  Velocidade: {chunks_processados/elapsed*60:.0f} chunks/min")

        try:
            await ws.send(json.dumps({"action": "export_linguagem"}))
        except Exception:
            pass

# ─── Main ─────────────────────────────────────────────────────────────────────

async def main_async():
    pdfs = [a for a in sys.argv[1:] if not a.startswith("--") and a.endswith(".pdf")]
    if not pdfs:
        print("Uso: python pdf_para_audio_selene.py livro.pdf")
        print("     python pdf_para_audio_selene.py livro.pdf --pagina 10 --ate 50")
        print("     python pdf_para_audio_selene.py livro.pdf --lang en")
        sys.exit(1)

    for pdf in pdfs:
        if not Path(pdf).exists():
            print(f"Arquivo não encontrado: {pdf}")
            continue
        backoff = 1.0
        for tentativa in range(3):
            try:
                await processar_pdf(pdf)
                break
            except (ConnectionRefusedError, OSError) as e:
                print(f"\n  Selene não acessível ({e}). Aguardando {backoff:.0f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            except websockets.exceptions.ConnectionClosed:
                print(f"\n  Conexão perdida. Tentativa {tentativa+1}/3...")
                await asyncio.sleep(2.0)

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\nLeitura interrompida pelo usuário.")
