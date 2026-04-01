#!/usr/bin/env python3
# treinar_fonemas.py — Treinamento fonético via TTS para Selene Brain 2.0
#
# Sintetiza sílabas do currículo fonético com espeak-ng, extrai FFT
# em frames de 25ms e envia como primitivas de onda para a Selene.
# NUNCA envia texto — apenas parâmetros físicos de frequência.
#
# Requisitos:
#   pip install websockets scipy numpy soundfile
#   espeak-ng instalado: https://github.com/espeak-ng/espeak-ng/releases
#   (Windows: baixe o .msi e instale; depois adicione ao PATH)
#
# Uso:
#   python treinar_fonemas.py                        # vogais (fase 1)
#   python treinar_fonemas.py --fase 3               # labiais CV
#   python treinar_fonemas.py --todas                # todas as 12 fases
#   python treinar_fonemas.py --fase 3 --rep 200     # 200 repetições
#   python treinar_fonemas.py --host 192.168.1.10    # host remoto

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import windows as sig_windows
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
FRAME_MS     = 25          # tamanho de frame FFT (ms)
PAUSA_FRAME  = 0.02        # pausa entre frames (s)
PAUSA_SILABA = 0.3         # pausa entre sílabas (s)
REPETICOES   = 50          # exposições por sílaba (padrão)
ESPEAK_RATE  = 90          # velocidade da fala (palavras/min) — mais devagar = mais claro
FASE_PADRAO  = 1           # fase padrão se não especificada

for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--host" and i + 1 < len(sys.argv):
        HOST    = sys.argv[i + 1]
        WS_URL  = f"ws://{HOST}:{PORTA}/selene"
    if arg == "--rep" and i + 1 < len(sys.argv):
        try: REPETICOES = int(sys.argv[i + 1])
        except: pass

FASE_ALVO = None
if "--todas" in sys.argv:
    FASE_ALVO = list(range(0, 12))
else:
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--fase" and i + 1 < len(sys.argv):
            try: FASE_ALVO = [int(sys.argv[i + 1])]
            except: pass
if FASE_ALVO is None:
    FASE_ALVO = [FASE_PADRAO]

# ─── Currículo fonético (espelha curriculo.rs) ────────────────────────────────
# Formato: (referencia_humana, voz_espeak, fase)
# `referencia_humana` é apenas para log — nunca entra no DB.

CURRICULO = [
    # Fase 1 — Vogais puras
    ("a",   "[[a]]",   1),
    ("e",   "[[e]]",   1),
    ("i",   "[[i]]",   1),
    ("o",   "[[o]]",   1),
    ("u",   "[[u]]",   1),
    # Fase 2 — Vogais nasais
    ("ã",   "[[a~]]",  2),
    ("õ",   "[[o~]]",  2),
    ("ẽ",   "[[e~]]",  2),
    # Fase 3 — CV Labiais
    ("ma",  "ma",  3), ("me",  "me",  3), ("mi",  "mi",  3),
    ("mo",  "mo",  3), ("mu",  "mu",  3),
    ("pa",  "pa",  3), ("pe",  "pe",  3), ("pi",  "pi",  3),
    ("po",  "po",  3), ("pu",  "pu",  3),
    ("ba",  "ba",  3), ("be",  "be",  3), ("bi",  "bi",  3),
    ("bo",  "bo",  3), ("bu",  "bu",  3),
    # Fase 4 — CV Dentais
    ("ta",  "ta",  4), ("te",  "te",  4), ("ti",  "ti",  4),
    ("to",  "to",  4), ("tu",  "tu",  4),
    ("da",  "da",  4), ("de",  "de",  4), ("di",  "di",  4),
    ("do",  "do",  4), ("du",  "du",  4),
    ("na",  "na",  4), ("ne",  "ne",  4), ("ni",  "ni",  4),
    ("no",  "no",  4), ("nu",  "nu",  4),
    ("la",  "la",  4), ("le",  "le",  4), ("li",  "li",  4),
    ("lo",  "lo",  4), ("lu",  "lu",  4),
    # Fase 5 — CV Velares
    ("ka",  "ca",  5), ("ke",  "que", 5), ("ki",  "qui", 5),
    ("ko",  "co",  5), ("ku",  "cu",  5),
    ("ga",  "ga",  5), ("ge",  "gue", 5), ("gi",  "gui", 5),
    ("go",  "go",  5), ("gu",  "gu",  5),
    # Fase 6 — CV Fricativas
    ("sa",  "sa",  6), ("se",  "se",  6), ("si",  "si",  6),
    ("so",  "so",  6), ("su",  "su",  6),
    ("fa",  "fa",  6), ("fe",  "fe",  6), ("fi",  "fi",  6),
    ("fo",  "fo",  6), ("fu",  "fu",  6),
    ("va",  "va",  6), ("ve",  "ve",  6), ("vi",  "vi",  6),
    ("vo",  "vo",  6), ("vu",  "vu",  6),
    # Fase 7 — CV Complexas
    ("xa",  "xa",  7), ("ja",  "ja",  7),
    ("lha", "lha", 7), ("nha", "nha", 7),
    ("ra",  "ra",  7), ("rra", "rra", 7),
    # Fase 8 — CVC
    ("mas", "mas", 8), ("par", "par", 8), ("sol", "sol", 8),
    ("com", "com", 8), ("fim", "fim", 8), ("bem", "bem", 8),
    # Fase 9 — CVCV
    ("mama", "mama", 9), ("papa", "papa", 9), ("bebe", "bebê", 9),
    ("coco", "coco", 9), ("lola", "lola", 9),
    # Fase 10 — Clusters CCV
    ("bra",  "bra",  10), ("tre",  "tre",  10), ("flu",  "flu",  10),
    ("pra",  "pra",  10), ("gra",  "gra",  10),
    # Fase 11 — Palavras de alta frequência
    ("mãe",  "mãe",  11), ("pai",  "pai",  11), ("água", "água", 11),
    ("casa", "casa", 11), ("amor", "amor", 11), ("não",  "não",  11),
    ("sim",  "sim",  11), ("vida", "vida", 11),
]

# ─── Síntese de áudio ─────────────────────────────────────────────────────────

def espeak_disponivel() -> bool:
    try:
        subprocess.run(["espeak-ng", "--version"], capture_output=True, timeout=3)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def sintetizar_espeak(texto: str, tmp_path: str) -> bool:
    """Sintetiza texto PT-BR com espeak-ng e salva em WAV. Retorna True se ok."""
    try:
        resultado = subprocess.run(
            ["espeak-ng", "-v", "pt-br", "-s", str(ESPEAK_RATE),
             "--punct", "-w", tmp_path, texto],
            capture_output=True, timeout=10
        )
        return resultado.returncode == 0 and os.path.exists(tmp_path)
    except Exception as e:
        print(f"   espeak-ng falhou: {e}")
        return False

def sintetizar_formante(f1: float, f2: float, f0: float, dur_ms: int, sample_rate: int = 22050) -> np.ndarray:
    """
    Síntese de formante simplificada (fallback quando espeak-ng não disponível).
    Gera pulsos glotais → filtra com bandpasss em F1 e F2.
    """
    from scipy.signal import butter, lfilter

    n = int(sample_rate * dur_ms / 1000)
    t = np.linspace(0, dur_ms / 1000, n)

    if f0 > 0:
        # Voiced: trem de pulsos glotais (onda de dente de serra)
        period = int(sample_rate / f0)
        excitacao = np.zeros(n)
        for k in range(0, n, period):
            if k < n: excitacao[k] = 1.0
        # Suaviza levemente
        excitacao = np.convolve(excitacao, np.ones(5)/5, mode='same')
    else:
        # Unvoiced: ruído branco
        excitacao = np.random.randn(n) * 0.3

    def bandpass(sig, low, high, fs):
        b, a = butter(3, [low / (fs/2), high / (fs/2)], btype='band')
        return lfilter(b, a, sig)

    # Filtra em F1 e F2
    out = np.zeros(n)
    if 200 < f1 < 1000:
        out += bandpass(excitacao, max(100, f1 - 150), min(1100, f1 + 150), sample_rate) * 0.6
    if 700 < f2 < 2800:
        out += bandpass(excitacao, max(500, f2 - 200), min(3000, f2 + 200), sample_rate) * 0.4

    # Normaliza
    mx = np.abs(out).max()
    if mx > 0: out /= mx
    return (out * 0.8).astype(np.float32)

# Parâmetros de formante aproximados por letra (fallback TTS)
FORMANTES_FALLBACK = {
    "a":  (800, 1200, 170, 200), "e": (500, 1700, 170, 180),
    "i":  (300, 2300, 170, 160), "o": (500,  900, 170, 180),
    "u":  (300,  800, 170, 160),
    "ma": (350, 1100, 170, 250), "pa": (400, 1100,   0, 230),
    "ba": (380, 1050, 170, 220), "na": (300, 1100, 170, 230),
    "ta": (350, 1800,   0, 220), "da": (350, 1800, 170, 210),
    "ka": (350, 1500,   0, 230), "ga": (350, 1500, 170, 210),
    "sa": (400, 1200,   0, 280), "fa": (400, 1200,   0, 270),
}

def gerar_audio_silaba(referencia: str, texto_tts: str, use_espeak: bool) -> tuple:
    """
    Retorna (samples: np.ndarray, sample_rate: int).
    Usa espeak-ng se disponível, senão síntese por formante.
    """
    sr = 22050
    if use_espeak:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            if sintetizar_espeak(texto_tts, tmp):
                sr_file, data = wavfile.read(tmp)
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                # Se estéreo, pega canal mono
                if data.ndim > 1:
                    data = data[:, 0]
                return data, sr_file
        finally:
            try: os.unlink(tmp)
            except: pass

    # Fallback: síntese por formante
    chave = referencia[:3].lower()
    f1, f2, f0, dur_ms = FORMANTES_FALLBACK.get(chave, (600, 1300, 170, 200))
    return sintetizar_formante(f1, f2, f0, dur_ms, sr), sr

# ─── Extração de FFT ──────────────────────────────────────────────────────────

def audio_para_frames_fft(samples: np.ndarray, sample_rate: int, frame_ms: int = 25):
    """
    Divide áudio em frames, aplica janela de Hann, calcula FFT.
    Retorna lista de frames, cada frame = [[freq_hz, amp], ...].
    Retorna apenas bins 20Hz–8000Hz (faixa relevante para voz).
    """
    frame_size = int(sample_rate * frame_ms / 1000)
    if frame_size < 8:
        return []

    hann = np.hanning(frame_size)
    frames = []
    for start in range(0, len(samples) - frame_size, frame_size):
        frame = samples[start:start + frame_size]
        frame_w = frame * hann

        fft_mag = np.abs(np.fft.rfft(frame_w))
        freqs   = np.fft.rfftfreq(frame_size, 1 / sample_rate)

        # Normaliza pela energia máxima
        mx = fft_mag.max()
        if mx < 1e-9:
            continue  # Frame silencioso — pula

        fft_norm = fft_mag / mx

        # Filtra 20–8000 Hz e sub-amostra para no máximo 128 bins
        mask = (freqs >= 20) & (freqs <= 8000)
        f_filtrado = freqs[mask]
        a_filtrado = fft_norm[mask]

        # Sub-amostragem para reduzir tamanho do payload
        if len(f_filtrado) > 128:
            indices = np.round(np.linspace(0, len(f_filtrado)-1, 128)).astype(int)
            f_filtrado = f_filtrado[indices]
            a_filtrado = a_filtrado[indices]

        frames.append([[float(f), float(a)] for f, a in zip(f_filtrado, a_filtrado)])

    return frames

# ─── WebSocket ────────────────────────────────────────────────────────────────

async def enviar_frames(ws, frames: list, referencia: str, frame_ms: int = 25) -> int:
    """Envia frames FFT para Selene. Retorna número de acks recebidos."""
    acks = 0
    for fft_bins in frames:
        msg = json.dumps({
            "action":     "learn_audio_fft",
            "fft":        fft_bins,
            "duracao_ms": frame_ms,
            "referencia": referencia,  # apenas para log — não entra no DB
        })
        await ws.send(msg)

        # Aguarda ack com timeout (ignora telemetria no meio)
        for _ in range(5):
            try:
                raw  = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(raw)
                if data.get("event") == "audio_ack":
                    acks += 1
                    break
            except (asyncio.TimeoutError, json.JSONDecodeError):
                break

        await asyncio.sleep(PAUSA_FRAME)

    return acks

async def treinar(silabas_da_fase: list, fase: int):
    print(f"\n{'='*55}")
    print(f"  FASE {fase} — {len(silabas_da_fase)} sílabas × {REPETICOES} repetições")
    print(f"  Conectando em {WS_URL}")
    print(f"{'='*55}")

    use_espeak = espeak_disponivel()
    if use_espeak:
        print("  TTS: espeak-ng (qualidade fonética alta)")
    else:
        print("  TTS: síntese por formante (fallback — instale espeak-ng para melhor resultado)")

    async with websockets.connect(
        WS_URL,
        ping_interval=None,
        max_queue=256,
        open_timeout=15,
    ) as ws:
        print("  Conectado!\n")
        total_frames = 0
        total_acks   = 0

        for rep in range(1, REPETICOES + 1):
            for referencia, texto_tts, _ in silabas_da_fase:
                try:
                    samples, sr = gerar_audio_silaba(referencia, texto_tts, use_espeak)
                    frames = audio_para_frames_fft(samples, sr, FRAME_MS)
                    if not frames:
                        continue

                    acks = await enviar_frames(ws, frames, referencia, FRAME_MS)
                    total_frames += len(frames)
                    total_acks   += acks

                    await asyncio.sleep(PAUSA_SILABA)

                except Exception as e:
                    print(f"   Erro em '{referencia}': {e}")

            if rep % 10 == 0:
                pct = total_acks / max(total_frames, 1) * 100
                print(f"  Rep {rep:4d}/{REPETICOES} | Frames: {total_frames} | Acks: {total_acks} ({pct:.0f}%)")

        print(f"\n  Fase {fase} concluída: {total_frames} frames → {total_acks} primitivas armazenadas")

# ─── Entrada ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for fase in FASE_ALVO:
        silabas = [(r, t, f) for r, t, f in CURRICULO if f == fase]
        if not silabas:
            print(f"Nenhuma sílaba definida para fase {fase}")
            continue
        try:
            asyncio.run(treinar(silabas, fase))
        except KeyboardInterrupt:
            print("\n\nTreinamento interrompido pelo usuário.")
            break
        except ConnectionRefusedError:
            print(f"\nErro: Selene não está rodando em {WS_URL}")
            print("Execute start_remote.bat ou cargo run --release primeiro.")
            sys.exit(1)
