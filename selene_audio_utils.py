#!/usr/bin/env python3
# selene_audio_utils.py — Módulo compartilhado: escrita + áudio para Selene
#
# Importável por todos os scripts de treinamento.
# Cada palavra/frase é enviada de duas formas simultâneas:
#   1. Escrita  → learn / learn_frase / associate  (grafo simbólico)
#   2. Áudio    → learn_audio_fft + grounding_fonetico  (grounding físico)
#
# Se espeak-ng não estiver instalado, apenas o modo escrita é usado (degradado).
#
# Uso típico em outros scripts:
#   from selene_audio_utils import DualTrainer
#   dt = DualTrainer(ws)
#   await dt.palavra("amor", valencia=0.95)
#   await dt.frase(["eu", "sinto", "que", "existo"], valencia=0.85)

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time

try:
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False

# ─── Parâmetros globais ───────────────────────────────────────────────────────

FRAME_MS     = 25
PAUSA        = 0.006
SILENCIO_DB  = -42.0
MAX_BINS     = 128
SAMPLE_RATE  = 22050
LANG_DEFAULT = "pt-br"

# Dígrafos do Português — tratados como unidade fonética
DIGRAFOS = {"lh", "nh", "ch", "rr", "ss", "qu", "gu", "sc", "xc", "rh"}

# ─── Verificação de espeak-ng ─────────────────────────────────────────────────

def _espeak_disponivel() -> bool:
    """Verifica se espeak-ng está no PATH ou em local padrão Windows."""
    # Tenta PATH
    try:
        r = subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True, timeout=5
        )
        if r.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Tenta caminho padrão Windows
    win_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    if os.path.exists(win_path):
        return True
    return False

def _espeak_cmd() -> str:
    """Retorna o comando espeak-ng correto."""
    try:
        r = subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True, timeout=5
        )
        if r.returncode == 0:
            return "espeak-ng"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    win_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    if os.path.exists(win_path):
        return win_path
    return "espeak-ng"

ESPEAK_OK  = _espeak_disponivel() and _NUMPY_OK
ESPEAK_CMD = _espeak_cmd() if ESPEAK_OK else "espeak-ng"

if not ESPEAK_OK:
    print("[selene_audio_utils] ⚠  espeak-ng ou numpy não encontrado — modo escrita apenas.")
    print("                        Instale espeak-ng: https://github.com/espeak-ng/espeak-ng/releases")

# ─── Decomposição de grafema ──────────────────────────────────────────────────

def decompor_grafema(texto: str) -> list:
    """
    Decompõe um grafema nas suas unidades fonéticas escritas.
    Dígrafos (lh, nh, ch, rr...) são tratados como unidade única.
    Ex: "lha" → ["lh","a"] | "bra" → ["b","r","a"] | "amor" → ["a","m","o","r"]
    """
    letras = texto.lower().strip()
    unidades = []
    i = 0
    while i < len(letras):
        if i + 1 < len(letras) and letras[i:i+2] in DIGRAFOS:
            unidades.append(letras[i:i+2])
            i += 2
        elif letras[i].isalpha():
            unidades.append(letras[i])
            i += 1
        else:
            i += 1
    return unidades

# ─── TTS via espeak-ng ────────────────────────────────────────────────────────

def sintetizar_espeak(texto: str, lang: str = LANG_DEFAULT):
    """Sintetiza texto com espeak-ng. Retorna (samples float32, sr) ou None."""
    if not ESPEAK_OK:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            ESPEAK_CMD,
            "-v", lang,
            "-s", "140",
            "-p", "50",
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
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

# ─── FFT ──────────────────────────────────────────────────────────────────────

def _energia_db(frame) -> float:
    rms = float(np.sqrt(np.mean(frame ** 2)))
    return -100.0 if rms < 1e-10 else 20.0 * np.log10(rms)

def _frame_para_bins(frame, sr: int):
    if _energia_db(frame) < SILENCIO_DB:
        return None
    n    = len(frame)
    hann = np.hanning(n)
    fft  = np.abs(np.fft.rfft(frame * hann))
    freq = np.fft.rfftfreq(n, 1 / sr)
    mx   = fft.max()
    if mx < 1e-9:
        return None
    fft = fft / mx
    mask = (freq >= 80.0) & (freq <= 8000.0)
    f_f, a_f = freq[mask], fft[mask]
    if len(f_f) > MAX_BINS:
        idx = np.round(np.linspace(0, len(f_f) - 1, MAX_BINS)).astype(int)
        f_f, a_f = f_f[idx], a_f[idx]
    return [[float(f), float(a)] for f, a in zip(f_f, a_f)]

def _amostras_para_frames(samples, sr: int):
    frame_size = int(sr * FRAME_MS / 1000)
    if frame_size < 8:
        return
    for start in range(0, len(samples) - frame_size, frame_size):
        bins = _frame_para_bins(samples[start:start + frame_size], sr)
        if bins:
            yield bins

def _re_amostrar(samples, sr_orig: int, sr_alvo: int = SAMPLE_RATE):
    if sr_orig == sr_alvo:
        return samples
    from math import gcd
    g = gcd(sr_alvo, sr_orig)
    return resample_poly(samples, sr_alvo // g, sr_orig // g).astype(np.float32)

# ─── Envio de áudio via WebSocket ─────────────────────────────────────────────

async def _enviar_fft_frames(ws, samples, sr: int, referencia: str, rep: int = 1):
    """Envia frames FFT de um áudio sintetizado. Retorna nº de frames enviados."""
    if sr != SAMPLE_RATE:
        samples = _re_amostrar(samples, sr)
        sr = SAMPLE_RATE
    frames = list(_amostras_para_frames(samples, sr))
    if not frames:
        return 0
    enviados = 0
    for _ in range(rep):
        for bins in frames:
            await ws.send(json.dumps({
                "action":     "learn_audio_fft",
                "fft":        bins,
                "duracao_ms": FRAME_MS,
                "referencia": referencia,
            }))
            for _ in range(3):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.4)
                    if json.loads(raw).get("event") == "audio_ack":
                        break
                except (asyncio.TimeoutError, json.JSONDecodeError):
                    break
            enviados += 1
            await asyncio.sleep(PAUSA)
    return enviados

async def _enviar_grounding(ws, grafema: str, fonte: str = "dual_trainer"):
    """Envia grounding_fonetico e aguarda ack."""
    letras = decompor_grafema(grafema)
    await ws.send(json.dumps({
        "action":  "grounding_fonetico",
        "grafema": grafema,
        "letras":  letras,
        "fonte":   fonte,
    }))
    for _ in range(3):
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
            if json.loads(raw).get("event") == "grounding_ack":
                break
        except (asyncio.TimeoutError, json.JSONDecodeError):
            break

# ─── DualTrainer ─────────────────────────────────────────────────────────────

class DualTrainer:
    """
    Treinador dual escrita + áudio.

    Cada chamada envia:
      [escrita]  learn / learn_frase / associate
      [áudio]    learn_audio_fft (frames FFT) + grounding_fonetico (binding)

    Se espeak-ng não estiver disponível, apenas o modo escrita é executado.
    """

    def __init__(self, ws, lang: str = LANG_DEFAULT,
                 rep_audio: int = 3, delay: float = 0.030):
        self.ws        = ws
        self.lang      = lang
        self.rep_audio = rep_audio   # repetições de áudio por item
        self.delay     = delay       # pausa entre envios de escrita

    # ── Primitivas de escrita ──────────────────────────────────────────────

    async def _aprender_escrita(self, word: str, valencia: float = 0.5,
                                contexto: str = "Geral", strength: float = 0.80):
        await self.ws.send(json.dumps({
            "action":   "learn",
            "word":     word,
            "context":  contexto,
            "valence":  round(valencia, 3),
            "strength": round(strength, 3),
        }))
        await asyncio.sleep(self.delay)

    async def _frase_escrita(self, tokens: list):
        await self.ws.send(json.dumps({
            "action": "learn_frase",
            "words":  tokens,
        }))
        await asyncio.sleep(self.delay)

    async def _associar(self, w1: str, w2: str, peso: float = 0.85):
        await self.ws.send(json.dumps({
            "action":  "associate",
            "word1":   w1,
            "word2":   w2,
            "weight":  round(peso, 3),
        }))
        await asyncio.sleep(self.delay)

    # ── Primitivas de áudio ────────────────────────────────────────────────

    async def _audio_palavra(self, texto: str, rep: int = None):
        """Sintetiza texto, envia FFT + grounding. Silencioso se sem espeak."""
        if not ESPEAK_OK:
            return
        resultado = sintetizar_espeak(texto, self.lang)
        if resultado is None:
            return
        samples, sr = resultado
        rep = rep or self.rep_audio
        await _enviar_fft_frames(self.ws, samples, sr,
                                 referencia=f"dual:{texto}", rep=rep)
        await _enviar_grounding(self.ws, texto)

    async def _audio_frase(self, tokens: list, rep: int = None):
        """Sintetiza a frase inteira + grounding por palavra."""
        if not ESPEAK_OK:
            return
        frase = " ".join(tokens)
        resultado = sintetizar_espeak(frase, self.lang)
        if resultado is None:
            return
        samples, sr = resultado
        rep = rep or self.rep_audio
        # Envia FFT da frase completa
        await _enviar_fft_frames(self.ws, samples, sr,
                                 referencia=f"dual_frase:{frase[:30]}", rep=rep)
        # Grounding de cada palavra da frase
        for token in tokens:
            if len(token) > 1:
                await _enviar_grounding(self.ws, token)

    # ── API pública ────────────────────────────────────────────────────────

    async def palavra(self, word: str, valencia: float = 0.5,
                      contexto: str = "Geral", strength: float = 0.80,
                      rep_audio: int = None):
        """Ensina uma palavra: escrita (learn) + áudio (FFT + grounding)."""
        await self._aprender_escrita(word, valencia, contexto, strength)
        await self._audio_palavra(word, rep=rep_audio)

    async def frase(self, tokens: list, valencia: float = 0.5,
                    contexto: str = "Geral", rep_audio: int = None):
        """
        Ensina uma frase:
          - learn para cada token com valencia
          - learn_frase (bigrams sequenciais)
          - áudio da frase inteira + grounding por palavra
        """
        for tok in tokens:
            if len(tok) > 1:
                await self._aprender_escrita(tok, valencia, contexto)
        await self._frase_escrita(tokens)
        await self._audio_frase(tokens, rep=rep_audio)

    async def associacao(self, w1: str, w2: str, peso: float = 0.85):
        """Cria associação w1→w2 (escrita). Sem áudio necessário."""
        await self._associar(w1, w2, peso)

    async def exportar(self):
        """Solicita export do estado da linguagem."""
        try:
            await self.ws.send(json.dumps({"action": "export_linguagem"}))
        except Exception:
            pass


# ─── Função utilitária standalone ────────────────────────────────────────────

async def treinar_lista_palavras(ws, palavras: list, lang: str = LANG_DEFAULT,
                                 rep_audio: int = 3, delay: float = 0.030):
    """
    Conveniência: treina uma lista de (palavra, valencia) ou [palavra, ...].
    Cada item recebe escrita + áudio automaticamente.
    """
    dt = DualTrainer(ws, lang=lang, rep_audio=rep_audio, delay=delay)
    for item in palavras:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            word, val = item[0], item[1]
            ctx = item[2] if len(item) > 2 else "Geral"
        else:
            word, val, ctx = str(item), 0.5, "Geral"
        await dt.palavra(word, val, ctx)

async def treinar_lista_frases(ws, frases: list, lang: str = LANG_DEFAULT,
                               rep_audio: int = 2, delay: float = 0.030):
    """
    Conveniência: treina uma lista de frases (listas de tokens ou strings).
    """
    dt = DualTrainer(ws, lang=lang, rep_audio=rep_audio, delay=delay)
    for frase in frases:
        if isinstance(frase, str):
            tokens = [t for t in re.split(r'\W+', frase.lower()) if len(t) > 1]
        else:
            tokens = [str(t) for t in frase]
        if len(tokens) >= 2:
            await dt.frase(tokens)
