#!/usr/bin/env python3
# audiolivro_selene.py — Aprende com arquivos de áudio (audiolivros, podcasts)
#
# Lê arquivos WAV/MP3/OGG/FLAC, extrai FFT em frames de 25ms,
# envia cada frame como primitiva de onda para Selene.
# Detecta silêncio automaticamente e pula frames vazios.
# NUNCA envia texto — apenas parâmetros físicos de frequência.
#
# Requisitos:
#   pip install websockets scipy numpy soundfile
#   Para MP3: pip install pydub  (e instalar ffmpeg: https://ffmpeg.org)
#
# Uso:
#   python audiolivro_selene.py arquivo.wav
#   python audiolivro_selene.py arquivo.mp3
#   python audiolivro_selene.py pasta/de/audios/
#   python audiolivro_selene.py arquivo.wav --host 192.168.1.10
#   python audiolivro_selene.py arquivo.wav --pausa 0.01   (mais rápido)
#   python audiolivro_selene.py arquivo.wav --inicio 300   (começa no segundo 300)

import asyncio
import json
import os
import sys
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
    import websockets
except ImportError:
    print("Instale: pip install websockets")
    sys.exit(1)

# ─── Configuração ─────────────────────────────────────────────────────────────

HOST         = "127.0.0.1"
PORTA        = 3030
WS_URL       = f"ws://{HOST}:{PORTA}/selene"
FRAME_MS     = 25           # tamanho de frame FFT (ms)
PAUSA_FRAME  = 0.015        # pausa entre frames (s) — ~66 frames/s máx
SILENCIO_DB  = -45.0        # threshold de silêncio em dB (frames mais silenciosos são pulados)
SAMPLE_RATE_ALVO = 22050    # taxa de re-amostragem alvo
MAX_BINS     = 128          # número máximo de bins FFT por frame (compressão)
INICIO_SEG   = 0.0          # começa neste segundo do arquivo
BARRA        = 40           # largura da barra de progresso

for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--host" and i + 1 < len(sys.argv):
        HOST   = sys.argv[i + 1]
        WS_URL = f"ws://{HOST}:{PORTA}/selene"
    if arg == "--pausa" and i + 1 < len(sys.argv):
        try: PAUSA_FRAME = float(sys.argv[i + 1])
        except: pass
    if arg == "--inicio" and i + 1 < len(sys.argv):
        try: INICIO_SEG = float(sys.argv[i + 1])
        except: pass

# ─── Carregamento de áudio ────────────────────────────────────────────────────

def carregar_wav(caminho: str) -> tuple:
    """Carrega WAV e retorna (samples float32 mono, sample_rate)."""
    sr, data = wavfile.read(caminho)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    # Mono: se estéreo, mistura canais
    if data.ndim > 1:
        data = data.mean(axis=1)

    return data, int(sr)

def tentar_carregar_mp3(caminho: str) -> tuple:
    """Tenta carregar MP3 via pydub. Requer ffmpeg instalado."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("Para MP3, instale: pip install pydub\nE instale o ffmpeg: https://ffmpeg.org")

    audio  = AudioSegment.from_file(caminho)
    audio  = audio.set_channels(1).set_sample_width(2)
    sr     = audio.frame_rate
    raw    = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return raw, sr

def carregar_audio(caminho: str) -> tuple:
    """Carrega qualquer formato de áudio suportado."""
    ext = Path(caminho).suffix.lower()

    # Tenta soundfile primeiro (suporta WAV, OGG, FLAC, etc.)
    try:
        import soundfile as sf
        data, sr = sf.read(caminho, dtype='float32', always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, int(sr)
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback para WAV nativo
    if ext == ".wav":
        return carregar_wav(caminho)

    # MP3/AAC/etc via pydub
    return tentar_carregar_mp3(caminho)

def re_amostrar(samples: np.ndarray, sr_origem: int, sr_alvo: int) -> np.ndarray:
    """Re-amostra para a taxa alvo se necessário."""
    if sr_origem == sr_alvo:
        return samples
    from math import gcd
    g = gcd(sr_alvo, sr_origem)
    up   = sr_alvo  // g
    down = sr_origem // g
    return resample_poly(samples, up, down).astype(np.float32)

# ─── Extração de FFT ──────────────────────────────────────────────────────────

def energia_db(frame: np.ndarray) -> float:
    """Energia em dB (RMS). Silêncio ≈ -inf dB."""
    rms = np.sqrt(np.mean(frame ** 2))
    if rms < 1e-10:
        return -100.0
    return 20.0 * np.log10(rms)

def frame_para_fft(frame: np.ndarray, sample_rate: int) -> list:
    """
    Converte um frame de amostras em lista de bins FFT [[freq_hz, amp], ...].
    Retorna None se o frame estiver abaixo do limiar de silêncio.
    """
    if energia_db(frame) < SILENCIO_DB:
        return None

    n = len(frame)
    hann = np.hanning(n)
    fft_mag  = np.abs(np.fft.rfft(frame * hann))
    freqs    = np.fft.rfftfreq(n, 1 / sample_rate)

    # Normaliza
    mx = fft_mag.max()
    if mx < 1e-9:
        return None
    fft_norm = fft_mag / mx

    # Filtra 20–8000 Hz (faixa relevante para voz)
    mask = (freqs >= 20.0) & (freqs <= 8000.0)
    f_filt = freqs[mask]
    a_filt = fft_norm[mask]

    # Sub-amostra para MAX_BINS
    if len(f_filt) > MAX_BINS:
        idx = np.round(np.linspace(0, len(f_filt) - 1, MAX_BINS)).astype(int)
        f_filt = f_filt[idx]
        a_filt = a_filt[idx]

    return [[float(f), float(a)] for f, a in zip(f_filt, a_filt)]

def amostras_para_frames(samples: np.ndarray, sample_rate: int, frame_ms: int = FRAME_MS):
    """Gerador: produz (índice, bins_fft) para cada frame não-silencioso."""
    frame_size = int(sample_rate * frame_ms / 1000)
    if frame_size < 8:
        return

    total_frames = (len(samples) - frame_size) // frame_size
    for i, start in enumerate(range(0, len(samples) - frame_size, frame_size)):
        frame = samples[start:start + frame_size]
        bins  = frame_para_fft(frame, sample_rate)
        if bins:
            yield i, total_frames, bins

# ─── WebSocket ────────────────────────────────────────────────────────────────

def barra_progresso(n: int, total: int) -> str:
    p     = n / total if total else 0
    cheio = int(p * BARRA)
    return f"[{'=' * cheio}{' ' * (BARRA - cheio)}] {n}/{total} ({p*100:.0f}%)"

async def processar_arquivo(caminho: str):
    nome = os.path.basename(caminho)
    print(f"\n{'='*55}")
    print(f"  ARQUIVO: {nome}")
    print(f"{'='*55}")

    # Carrega e prepara o áudio
    print(f"  Carregando...", end="", flush=True)
    try:
        samples, sr = carregar_audio(caminho)
    except Exception as e:
        print(f"\n  Erro ao carregar: {e}")
        return

    # Recorta a partir do segundo de início
    if INICIO_SEG > 0:
        inicio_amostra = int(INICIO_SEG * sr)
        samples = samples[inicio_amostra:]

    # Re-amostra se necessário
    if sr != SAMPLE_RATE_ALVO:
        samples = re_amostrar(samples, sr, SAMPLE_RATE_ALVO)
        sr = SAMPLE_RATE_ALVO

    dur_s = len(samples) / sr
    print(f" {dur_s:.1f}s  |  SR: {sr}Hz")

    # Conta frames não-silenciosos
    total_frames = sum(1 for _ in amostras_para_frames(samples, sr))
    if total_frames == 0:
        print("  Nenhum frame de áudio detectado (arquivo silencioso?).")
        return
    print(f"  {total_frames} frames de áudio (silêncio removido). Conectando...")

    async with websockets.connect(
        WS_URL,
        ping_interval=None,
        max_queue=512,
        open_timeout=15,
    ) as ws:
        print(f"  Conectado! Enviando para Selene...")
        enviados  = 0
        acks      = 0
        erros     = 0
        t_inicio  = time.time()

        for frame_idx, total, bins in amostras_para_frames(samples, sr):
            try:
                msg = json.dumps({
                    "action":     "learn_audio_fft",
                    "fft":        bins,
                    "duracao_ms": FRAME_MS,
                    "referencia": nome,   # nome do arquivo — apenas para log
                })
                await ws.send(msg)

                # Aguarda ack (ignora telemetria)
                for _ in range(4):
                    try:
                        raw  = await asyncio.wait_for(ws.recv(), timeout=0.8)
                        if json.loads(raw).get("event") == "audio_ack":
                            acks += 1
                            break
                    except (asyncio.TimeoutError, json.JSONDecodeError):
                        break

                enviados += 1

                # Atualiza barra a cada 100 frames
                if enviados % 100 == 0:
                    elapsed  = time.time() - t_inicio
                    fps      = enviados / elapsed if elapsed > 0 else 0
                    restante = (total_frames - enviados) / fps if fps > 0 else 0
                    print(f"\r  {barra_progresso(enviados, total_frames)}"
                          f"  {fps:.0f}fps  ETA:{restante:.0f}s", end="", flush=True)

                await asyncio.sleep(PAUSA_FRAME)

            except websockets.exceptions.ConnectionClosed:
                print(f"\n  Conexão perdida no frame {frame_idx}. Tentando reconectar...")
                raise

            except Exception as e:
                erros += 1
                if erros <= 3:
                    print(f"\n  Erro frame {frame_idx}: {e}")

        elapsed = time.time() - t_inicio
        print(f"\n  Concluído: {enviados} frames | {acks} acks | {erros} erros | {elapsed:.1f}s")

        # Solicita export ao concluir
        try:
            await ws.send(json.dumps({"action": "export_linguagem"}))
        except Exception:
            pass

# ─── Entrada ─────────────────────────────────────────────────────────────────

EXTENSOES_SUPORTADAS = {".wav", ".ogg", ".flac", ".mp3", ".aac", ".m4a", ".opus"}

def listar_audios(caminho: str) -> list:
    p = Path(caminho)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        arquivos = []
        for ext in EXTENSOES_SUPORTADAS:
            arquivos.extend(sorted(p.glob(f"**/*{ext}")))
        return [str(a) for a in arquivos]
    return []

async def main_async():
    alvos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not alvos:
        print("Uso: python audiolivro_selene.py arquivo.wav")
        print("     python audiolivro_selene.py pasta/de/audios/")
        print("     python audiolivro_selene.py arquivo.mp3 --host 192.168.1.10")
        print("     python audiolivro_selene.py arquivo.wav --inicio 300  (começa no segundo 300)")
        sys.exit(1)

    for alvo in alvos:
        arquivos = listar_audios(alvo)
        if not arquivos:
            print(f"Nenhum arquivo de áudio encontrado em: {alvo}")
            continue
        if len(arquivos) > 1:
            print(f"Pasta: {len(arquivos)} arquivos encontrados.")

        for caminho in arquivos:
            backoff = 1.0
            for tentativa in range(3):
                try:
                    await processar_arquivo(caminho)
                    break
                except (ConnectionRefusedError, OSError) as e:
                    print(f"\n  Selene não acessível ({e}). Aguardando {backoff:.0f}s...")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                except websockets.exceptions.ConnectionClosed:
                    print(f"\n  Conexão perdida. Tentativa {tentativa + 1}/3...")
                    await asyncio.sleep(2.0)

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\nAudiolivro interrompido pelo usuário.")
