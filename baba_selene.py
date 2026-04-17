#!/usr/bin/env python3
# baba_selene.py — Treinamento fonético bê-á-bá para Selene Brain 2.0
#
# Ensina Selene a distinguir sons individuais do Português Brasileiro:
#   - Fase 1: vogais puras  (a, e, i, o, u)
#   - Fase 2: vogais nasais (ã, ẽ, ĩ, õ, ũ)
#   - Fase 3: consoantes isoladas com vogal apoio (ba, pa, da, ta...)
#   - Fase 4: sílabas CVC  (bal, par, dor, ...)
#   - Fase 5: pares mínimos (bato/pato, calo/galo, faca/vaca...)
#   - Fase 6: dígrafos e ditongos (lh, nh, ch, ai, ei, oi, au...)
#   - Fase 7: encontros consonantais (bra, cla, flo, pré...)
#   - Fase 8: alfabeto completo (a, bê, cê, dê, e, efe, ...)
#
# Loop de grounding fonético:
#   espeak-ng sintetiza "ba"
#     → WAV → FFT frames → learn_audio_fft  (Selene ouve o som)
#     → grounding_fonetico{"grafema":"ba","letras":["b","a"]}  (Selene aprende a escrita)
#
# Como funciona o grounding:
#   O SpikePattern do áudio fica salvo em ultimo_padrao_audio após cada frame FFT.
#   Quando chega o grounding_fonetico, o Rust faz grounding_bind(palavras, audio_spike)
#   criando a associação permanente: padrão de onda ↔ grafema ↔ letras.
#   Com repetição, Selene aprende que "aquele som" = "aquelas letras".
#
# STT opcional (vosk):
#   Se o modelo vosk-model-small-pt estiver na pasta, o script verifica se
#   espeak disse o que deveria — util para detectar erros de síntese.
#   Instale: pip install vosk
#   Baixe:   https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip
#
# Requisitos:
#   pip install websockets scipy numpy
#   espeak-ng instalado e no PATH
#   (Windows: https://github.com/espeak-ng/espeak-ng/releases)
#
# Uso:
#   python baba_selene.py                  # todas as 8 fases
#   python baba_selene.py --fase 1         # só vogais
#   python baba_selene.py --fase 5         # pares mínimos
#   python baba_selene.py --rep 50         # 50 repetições por sílaba
#   python baba_selene.py --host 192.168.1.10
#   python baba_selene.py --pausa 0.005    # mais rápido

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
except ImportError:
    print("Instale: pip install numpy scipy")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("Instale: pip install websockets")
    sys.exit(1)

# ─── Configuração ─────────────────────────────────────────────────────────────

HOST        = "127.0.0.1"
PORTA       = 3030
WS_URL      = f"ws://{HOST}:{PORTA}/selene"
FRAME_MS    = 25
PAUSA       = 0.008
SILENCIO_DB = -40.0
MAX_BINS    = 128
SAMPLE_RATE = 22050
REPETICOES  = 30       # repetições por unidade fonética
FASE_ALVO   = None     # None = todas

for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--host"  and i + 1 < len(sys.argv): HOST  = sys.argv[i+1]; WS_URL = f"ws://{HOST}:{PORTA}/selene"
    if arg == "--pausa" and i + 1 < len(sys.argv):
        try: PAUSA = float(sys.argv[i+1])
        except: pass
    if arg == "--rep"   and i + 1 < len(sys.argv):
        try: REPETICOES = int(sys.argv[i+1])
        except: pass
    if arg == "--fase"  and i + 1 < len(sys.argv):
        try: FASE_ALVO = int(sys.argv[i+1])
        except: pass

# ─── Currículo fonético do Português Brasileiro ───────────────────────────────
#
# Progressão pedagógica:
#   0  Letras isoladas       — cada letra do alfabeto como som (a-z)
#   1  Letras acentuadas     — á, â, ã, à, é, ê, í, ó, ô, õ, ú, ü, ç
#   2  Sílabas com A         — ba, ca, da, fa, ga... (todas as consoantes + A)
#   3  Sílabas com E         — be, ce, de, fe, ge...
#   4  Sílabas com I         — bi, ci, di, fi, gi...
#   5  Sílabas com O         — bo, co, do, fo, go...
#   6  Sílabas com U         — bu, cu, du, fu, gu...
#   7  Vogais puras          — a, e, i, o, u
#   8  Vogais nasais         — ã, ẽ, ĩ, õ, ũ, an, en, in, on, un
#   9  Sílabas CVC           — bal, par, dor, tal...
#  10  Pares mínimos         — pato/bato, calo/galo, faca/vaca...
#  11  Dígrafos e ditongos   — lh, nh, ch, ai, ei, oi, au, eu, ão...
#  12  Encontros consonantais— bla, bra, cra, dra, fra, pra, tra...
#  13  Nomes das letras      — a, bê, cê, dê, efe, gê, agá...

CURRICULO = {
    0: {
        "nome": "Letras isoladas (sons do alfabeto A-Z)",
        "descricao": "Cada letra do alfabeto pronunciada como som isolado",
        "itens": [
            ("iso_a", "a",  "pt-br"), ("iso_b", "b",  "pt-br"),
            ("iso_c", "c",  "pt-br"), ("iso_d", "d",  "pt-br"),
            ("iso_e", "e",  "pt-br"), ("iso_f", "f",  "pt-br"),
            ("iso_g", "g",  "pt-br"), ("iso_h", "h",  "pt-br"),
            ("iso_i", "i",  "pt-br"), ("iso_j", "j",  "pt-br"),
            ("iso_k", "k",  "pt-br"), ("iso_l", "l",  "pt-br"),
            ("iso_m", "m",  "pt-br"), ("iso_n", "n",  "pt-br"),
            ("iso_o", "o",  "pt-br"), ("iso_p", "p",  "pt-br"),
            ("iso_q", "q",  "pt-br"), ("iso_r", "r",  "pt-br"),
            ("iso_s", "s",  "pt-br"), ("iso_t", "t",  "pt-br"),
            ("iso_u", "u",  "pt-br"), ("iso_v", "v",  "pt-br"),
            ("iso_w", "w",  "pt-br"), ("iso_x", "x",  "pt-br"),
            ("iso_y", "y",  "pt-br"), ("iso_z", "z",  "pt-br"),
        ],
    },
    1: {
        "nome": "Letras acentuadas do Português",
        "descricao": "Vogais com acento agudo, circunflexo, til, grave e cedilha",
        "itens": [
            # Acento agudo — eleva/abre a vogal
            ("ac_a_agudo",  "á",  "pt-br"),
            ("ac_e_agudo",  "é",  "pt-br"),
            ("ac_i_agudo",  "í",  "pt-br"),
            ("ac_o_agudo",  "ó",  "pt-br"),
            ("ac_u_agudo",  "ú",  "pt-br"),
            # Acento circunflexo — fecha a vogal
            ("ac_a_circ",   "â",  "pt-br"),
            ("ac_e_circ",   "ê",  "pt-br"),
            ("ac_o_circ",   "ô",  "pt-br"),
            # Til — nasaliza a vogal
            ("ac_a_til",    "ã",  "pt-br"),
            ("ac_o_til",    "õ",  "pt-br"),
            # Acento grave — crase (contração)
            ("ac_a_grave",  "à",  "pt-br"),
            # Cedilha — altera o som de C para /s/
            ("cedilha",     "ç",  "pt-br"),
            # Trema (raro, ainda presente em nomes)
            ("ac_u_trema",  "ü",  "pt-br"),
            # Sílabas acentuadas comuns — som em contexto
            ("sil_ca",      "cá", "pt-br"),
            ("sil_pe",      "pé", "pt-br"),
            ("sil_vo",      "vó", "pt-br"),
            ("sil_fe",      "fé", "pt-br"),
            ("sil_po",      "pô", "pt-br"),
            ("sil_ca_circ", "câ", "pt-br"),
            ("sil_ce",      "cê", "pt-br"),
            ("sil_co",      "cô", "pt-br"),
            ("sil_ca_til",  "cã", "pt-br"),
            ("sil_co_til",  "cõ", "pt-br"),
            ("sil_aca",     "ça", "pt-br"),
            ("sil_ace",     "çe", "pt-br"),
            ("sil_aci",     "çi", "pt-br"),
            ("sil_aco",     "ço", "pt-br"),
            ("sil_acu",     "çu", "pt-br"),
        ],
    },
    2: {
        "nome": "Sílabas com A (consoante + A)",
        "descricao": "Todas as consoantes combinadas com a vogal A",
        "itens": [
            ("ba", "ba", "pt-br"), ("ca", "ca", "pt-br"),
            ("da", "da", "pt-br"), ("fa", "fa", "pt-br"),
            ("ga", "ga", "pt-br"), ("ha", "ha", "pt-br"),
            ("ja", "ja", "pt-br"), ("la", "la", "pt-br"),
            ("ma", "ma", "pt-br"), ("na", "na", "pt-br"),
            ("pa", "pa", "pt-br"), ("ra", "ra", "pt-br"),
            ("sa", "sa", "pt-br"), ("ta", "ta", "pt-br"),
            ("va", "va", "pt-br"), ("xa", "xa", "pt-br"),
            ("za", "za", "pt-br"),
        ],
    },
    3: {
        "nome": "Sílabas com E (consoante + E)",
        "descricao": "Todas as consoantes combinadas com a vogal E",
        "itens": [
            ("be", "be", "pt-br"), ("ce", "ce", "pt-br"),
            ("de", "de", "pt-br"), ("fe", "fe", "pt-br"),
            ("ge", "ge", "pt-br"), ("he", "he", "pt-br"),
            ("je", "je", "pt-br"), ("le", "le", "pt-br"),
            ("me", "me", "pt-br"), ("ne", "ne", "pt-br"),
            ("pe", "pe", "pt-br"), ("re", "re", "pt-br"),
            ("se", "se", "pt-br"), ("te", "te", "pt-br"),
            ("ve", "ve", "pt-br"), ("xe", "xe", "pt-br"),
            ("ze", "ze", "pt-br"),
        ],
    },
    4: {
        "nome": "Sílabas com I (consoante + I)",
        "descricao": "Todas as consoantes combinadas com a vogal I",
        "itens": [
            ("bi", "bi", "pt-br"), ("ci", "ci", "pt-br"),
            ("di", "di", "pt-br"), ("fi", "fi", "pt-br"),
            ("gi", "gi", "pt-br"), ("hi", "hi", "pt-br"),
            ("ji", "ji", "pt-br"), ("li", "li", "pt-br"),
            ("mi", "mi", "pt-br"), ("ni", "ni", "pt-br"),
            ("pi", "pi", "pt-br"), ("ri", "ri", "pt-br"),
            ("si", "si", "pt-br"), ("ti", "ti", "pt-br"),
            ("vi", "vi", "pt-br"), ("xi", "xi", "pt-br"),
            ("zi", "zi", "pt-br"),
        ],
    },
    5: {
        "nome": "Sílabas com O (consoante + O)",
        "descricao": "Todas as consoantes combinadas com a vogal O",
        "itens": [
            ("bo", "bo", "pt-br"), ("co", "co", "pt-br"),
            ("do", "do", "pt-br"), ("fo", "fo", "pt-br"),
            ("go", "go", "pt-br"), ("ho", "ho", "pt-br"),
            ("jo", "jo", "pt-br"), ("lo", "lo", "pt-br"),
            ("mo", "mo", "pt-br"), ("no", "no", "pt-br"),
            ("po", "po", "pt-br"), ("ro", "ro", "pt-br"),
            ("so", "so", "pt-br"), ("to", "to", "pt-br"),
            ("vo", "vo", "pt-br"), ("xo", "xo", "pt-br"),
            ("zo", "zo", "pt-br"),
        ],
    },
    6: {
        "nome": "Sílabas com U (consoante + U)",
        "descricao": "Todas as consoantes combinadas com a vogal U",
        "itens": [
            ("bu", "bu", "pt-br"), ("cu", "cu", "pt-br"),
            ("du", "du", "pt-br"), ("fu", "fu", "pt-br"),
            ("gu", "gu", "pt-br"), ("hu", "hu", "pt-br"),
            ("ju", "ju", "pt-br"), ("lu", "lu", "pt-br"),
            ("mu", "mu", "pt-br"), ("nu", "nu", "pt-br"),
            ("pu", "pu", "pt-br"), ("ru", "ru", "pt-br"),
            ("su", "su", "pt-br"), ("tu", "tu", "pt-br"),
            ("vu", "vu", "pt-br"), ("xu", "xu", "pt-br"),
            ("zu", "zu", "pt-br"),
        ],
    },
    7: {
        "nome": "Vogais puras",
        "descricao": "Sons vocálicos fundamentais — base de toda fonologia",
        "itens": [
            ("a_vogal", "a", "pt-br"),
            ("e_vogal", "e", "pt-br"),
            ("i_vogal", "i", "pt-br"),
            ("o_vogal", "o", "pt-br"),
            ("u_vogal", "u", "pt-br"),
        ],
    },
    8: {
        "nome": "Vogais nasais",
        "descricao": "Nasalização — característica marcante do PB",
        "itens": [
            ("an_nasal", "ã",  "pt-br"),
            ("em_nasal", "ẽ",  "pt-br"),
            ("im_nasal", "ĩ",  "pt-br"),
            ("om_nasal", "õ",  "pt-br"),
            ("um_nasal", "ũ",  "pt-br"),
            ("an_sil",   "an", "pt-br"),
            ("en_sil",   "en", "pt-br"),
            ("in_sil",   "in", "pt-br"),
            ("on_sil",   "on", "pt-br"),
            ("un_sil",   "un", "pt-br"),
        ],
    },
    9: {
        "nome": "Sílabas CVC (consoante + vogal + consoante)",
        "descricao": "Sílabas fechadas — coda consonantal",
        "itens": [
            ("bal", "bal", "pt-br"), ("par", "par", "pt-br"),
            ("dor", "dor", "pt-br"), ("tal", "tal", "pt-br"),
            ("mar", "mar", "pt-br"), ("nal", "nal", "pt-br"),
            ("sol", "sol", "pt-br"), ("cor", "cor", "pt-br"),
            ("lar", "lar", "pt-br"), ("ver", "ver", "pt-br"),
            ("fil", "fil", "pt-br"), ("gol", "gol", "pt-br"),
            ("bar", "bar", "pt-br"), ("pes", "pes", "pt-br"),
            ("bis", "bis", "pt-br"), ("dom", "dom", "pt-br"),
        ],
    },
    10: {
        "nome": "Pares mínimos",
        "descricao": "Contraste fonêmico — distingue sons próximos",
        "itens": [
            ("pato",  "pato",  "pt-br"), ("bato",  "bato",  "pt-br"),
            ("calo",  "calo",  "pt-br"), ("galo",  "galo",  "pt-br"),
            ("faca",  "faca",  "pt-br"), ("vaca",  "vaca",  "pt-br"),
            ("cama",  "cama",  "pt-br"), ("gama",  "gama",  "pt-br"),
            ("lata",  "lata",  "pt-br"), ("rata",  "rata",  "pt-br"),
            ("soma",  "soma",  "pt-br"), ("zona",  "zona",  "pt-br"),
            ("chave", "chave", "pt-br"), ("grave", "grave", "pt-br"),
            ("mala",  "mala",  "pt-br"), ("bala",  "bala",  "pt-br"),
            ("pena",  "pena",  "pt-br"), ("vena",  "vena",  "pt-br"),
            ("tio",   "tio",   "pt-br"), ("dio",   "dio",   "pt-br"),
        ],
    },
    11: {
        "nome": "Dígrafos e ditongos",
        "descricao": "Sons compostos característicos do PB",
        "itens": [
            # Dígrafos consonantais
            ("lha",  "lha",  "pt-br"), ("nha",  "nha",  "pt-br"),
            ("cha",  "cha",  "pt-br"), ("rra",  "rra",  "pt-br"),
            ("ssa",  "ssa",  "pt-br"),
            # Ditongos crescentes
            ("ia",   "ia",   "pt-br"), ("ie",   "ie",   "pt-br"),
            ("io",   "io",   "pt-br"), ("ua",   "ua",   "pt-br"),
            ("ue",   "ue",   "pt-br"), ("ui",   "ui",   "pt-br"),
            # Ditongos decrescentes
            ("ai",   "ai",   "pt-br"), ("ei",   "ei",   "pt-br"),
            ("oi",   "oi",   "pt-br"), ("au",   "au",   "pt-br"),
            ("eu",   "eu",   "pt-br"),
            # Ditongos nasais
            ("aen",  "ãe",   "pt-br"), ("aon",  "ão",   "pt-br"),
            ("oin",  "õe",   "pt-br"),
        ],
    },
    12: {
        "nome": "Encontros consonantais",
        "descricao": "Grupos consonantais — onset complexo",
        "itens": [
            # com L: bl, cl, fl, gl, pl
            ("bla", "bla", "pt-br"), ("cla", "cla", "pt-br"),
            ("fla", "fla", "pt-br"), ("gla", "gla", "pt-br"),
            ("pla", "pla", "pt-br"),
            # com R: br, cr, dr, fr, gr, pr, tr
            ("bra", "bra", "pt-br"), ("cra", "cra", "pt-br"),
            ("dra", "dra", "pt-br"), ("fra", "fra", "pt-br"),
            ("gra", "gra", "pt-br"), ("pra", "pra", "pt-br"),
            ("tra", "tra", "pt-br"),
            # variações com outras vogais
            ("bre", "bre", "pt-br"), ("pri", "pri", "pt-br"),
            ("fro", "fro", "pt-br"), ("cru", "cru", "pt-br"),
            ("gre", "gre", "pt-br"), ("tre", "tre", "pt-br"),
        ],
    },
    13: {
        "nome": "Nomes das letras (alfabeto completo)",
        "descricao": "Os 26 nomes das letras como sons únicos",
        "itens": [
            ("letra_a", "a",       "pt-br"),
            ("letra_b", "bê",      "pt-br"),
            ("letra_c", "cê",      "pt-br"),
            ("letra_d", "dê",      "pt-br"),
            ("letra_e", "e",       "pt-br"),
            ("letra_f", "efe",     "pt-br"),
            ("letra_g", "gê",      "pt-br"),
            ("letra_h", "agá",     "pt-br"),
            ("letra_i", "i",       "pt-br"),
            ("letra_j", "jota",    "pt-br"),
            ("letra_k", "ká",      "pt-br"),
            ("letra_l", "ele",     "pt-br"),
            ("letra_m", "eme",     "pt-br"),
            ("letra_n", "ene",     "pt-br"),
            ("letra_o", "o",       "pt-br"),
            ("letra_p", "pê",      "pt-br"),
            ("letra_q", "quê",     "pt-br"),
            ("letra_r", "erre",    "pt-br"),
            ("letra_s", "esse",    "pt-br"),
            ("letra_t", "tê",      "pt-br"),
            ("letra_u", "u",       "pt-br"),
            ("letra_v", "vê",      "pt-br"),
            ("letra_w", "dáblio",  "pt-br"),
            ("letra_x", "xis",     "pt-br"),
            ("letra_y", "ípsilon", "pt-br"),
            ("letra_z", "zê",      "pt-br"),
        ],
    },
}

# ─── TTS via espeak-ng ────────────────────────────────────────────────────────

def sintetizar_espeak(texto: str, lang: str = "pt-br") -> tuple | None:
    """Sintetiza texto com espeak-ng e retorna (samples float32, sample_rate)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "espeak-ng",
            "-v", lang,
            "-s", "140",     # velocidade normal de fala
            "-p", "50",      # pitch neutro
            "-a", "180",     # amplitude
            "-w", tmp_path,
            texto,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0:
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
    n       = len(frame)
    hann    = np.hanning(n)
    fft_mag = np.abs(np.fft.rfft(frame * hann))
    freqs   = np.fft.rfftfreq(n, 1 / sr)
    mx = fft_mag.max()
    if mx < 1e-9:
        return None
    fft_norm = fft_mag / mx
    mask = (freqs >= 80.0) & (freqs <= 8000.0)
    f_f, a_f = freqs[mask], fft_norm[mask]
    if len(f_f) > MAX_BINS:
        idx  = np.round(np.linspace(0, len(f_f) - 1, MAX_BINS)).astype(int)
        f_f  = f_f[idx]
        a_f  = a_f[idx]
    return [[float(f), float(a)] for f, a in zip(f_f, a_f)]

def amostras_para_frames(samples: np.ndarray, sr: int):
    frame_size = int(sr * FRAME_MS / 1000)
    if frame_size < 8:
        return
    for start in range(0, len(samples) - frame_size, frame_size):
        frame = samples[start:start + frame_size]
        bins  = frame_para_bins(frame, sr)
        if bins:
            yield bins

# ─── Grounding fonético ───────────────────────────────────────────────────────

# Dígrafos do Português: sequências de letras que representam UM único fonema.
# Tratados como unidade no grounding — não separados em letras individuais.
DIGRAFOS = {"lh", "nh", "ch", "rr", "ss", "qu", "gu", "sc", "xc"}

def decompor_grafema(texto: str) -> list[str]:
    """
    Decompõe um grafema em unidades fonéticas:
    - Dígrafos (lh, nh, ch, rr...) → 1 unidade
    - Demais letras → 1 unidade por letra
    Exemplos:
      "ba"  → ["b", "a"]
      "lha" → ["lh", "a"]
      "bra" → ["b", "r", "a"]
      "pato"→ ["p", "a", "t", "o"]
    """
    letras = texto.lower()
    unidades = []
    i = 0
    while i < len(letras):
        # Tenta dígrafo de 2 caracteres
        if i + 1 < len(letras) and letras[i:i+2] in DIGRAFOS:
            unidades.append(letras[i:i+2])
            i += 2
        elif letras[i].isalpha():
            unidades.append(letras[i])
            i += 1
        else:
            i += 1  # ignora acentos/pontuação não-alfabéticos
    return unidades


# ─── STT opcional via vosk ────────────────────────────────────────────────────

def _vosk_disponivel():
    """Verifica se vosk e modelo PT estão disponíveis."""
    try:
        import vosk
        from pathlib import Path
        modelos = list(Path(".").glob("vosk-model*pt*")) + list(Path(".").glob("vosk-model*br*"))
        return bool(modelos), str(modelos[0]) if modelos else None
    except ImportError:
        return False, None

_VOSK_OK, _VOSK_MODELO = _vosk_disponivel()
_vosk_recognizer = None  # lazy init

def _stt_vosk(wav_path: str) -> str | None:
    """Reconhece áudio WAV com vosk. Retorna texto ou None."""
    global _vosk_recognizer
    try:
        import vosk, wave, json as _json
        if _vosk_recognizer is None:
            model = vosk.Model(_VOSK_MODELO)
            _vosk_recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
        with wave.open(wav_path, "rb") as wf:
            data = wf.readframes(wf.getnframes())
        _vosk_recognizer.AcceptWaveform(data)
        result = _json.loads(_vosk_recognizer.Result())
        return result.get("text", "").strip() or None
    except Exception:
        return None


# ─── WebSocket ────────────────────────────────────────────────────────────────

async def _enviar_learn(ws, texto: str, letras: list, pausa: float = 0.020):
    """Envia componente escrito: learn grafema, learn cada letra, learn_frase letras."""
    await ws.send(json.dumps({
        "action":   "learn",
        "word":     texto,
        "context":  "fonética",
        "valence":  0.5,
        "strength": 0.85,
    }))
    await asyncio.sleep(pausa)
    for letra in letras:
        await ws.send(json.dumps({
            "action":   "learn",
            "word":     letra,
            "context":  "fonética",
            "valence":  0.5,
            "strength": 0.80,
        }))
        await asyncio.sleep(pausa)
    if len(letras) >= 2:
        await ws.send(json.dumps({
            "action": "learn_frase",
            "words":  letras,
        }))
        await asyncio.sleep(pausa)


async def enviar_fonema(ws, fonema_id: str, texto: str, lang: str, rep: int) -> int:
    """
    Sintetiza e envia um fonema N vezes com grounding fonético + escrita simultânea.

    Loop por repetição:
      1. Envia todos os frames FFT do fonema (Selene ouve o som)
         → ultimo_padrao_audio fica atualizado no Rust
      2. Envia grounding_fonetico com grafema + letras decompostas
         → Rust faz grounding_bind(audio_spike, [grafema, "b", "a", ...])
      3. Envia learn (escrita): grafema, cada letra, learn_frase das letras
         → grafo_associacoes recebe o grafema e as letras como nós simbólicos

    Após a 1ª repetição, se vosk estiver disponível, verifica o STT
    para confirmar que espeak produziu o fonema correto.
    """
    resultado = sintetizar_espeak(texto, lang)
    if resultado is None:
        print(f"\n    ⚠  '{texto}' — espeak-ng falhou (está instalado e no PATH?)")
        return 0

    samples, sr = resultado
    if sr != SAMPLE_RATE:
        from math import gcd
        g    = gcd(SAMPLE_RATE, sr)
        from scipy.signal import resample_poly
        samples = resample_poly(samples, SAMPLE_RATE // g, sr // g).astype(np.float32)
        sr = SAMPLE_RATE

    frames = list(amostras_para_frames(samples, sr))
    if not frames:
        print(f"\n    ⚠  '{texto}' — silêncio total após síntese")
        return 0

    letras = decompor_grafema(texto)
    enviados = 0

    for rep_idx in range(rep):
        # 1. Envia frames FFT (Selene ouve o som)
        for bins in frames:
            msg = json.dumps({
                "action":     "learn_audio_fft",
                "fft":        bins,
                "duracao_ms": FRAME_MS,
                "referencia": f"baba:{fonema_id}",
            })
            await ws.send(msg)
            for _ in range(3):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    if json.loads(raw).get("event") == "audio_ack":
                        break
                except (asyncio.TimeoutError, json.JSONDecodeError):
                    break
            enviados += 1
            await asyncio.sleep(PAUSA)

        # 2. Grounding fonético: conecta padrão de onda → letras
        grounding_msg = json.dumps({
            "action":  "grounding_fonetico",
            "grafema": texto,
            "letras":  letras,
            "fonte":   "baba_curriculum",
        })
        await ws.send(grounding_msg)
        for _ in range(3):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                if json.loads(raw).get("event") == "grounding_ack":
                    break
            except (asyncio.TimeoutError, json.JSONDecodeError):
                break

        # 3. Componente escrito: ensina grafema + letras no grafo simbólico
        await _enviar_learn(ws, texto, letras)

        # 4. STT opcional — só na 1ª repetição para não atrasar o treino
        if rep_idx == 0 and _VOSK_OK:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                from scipy.io import wavfile as _wf
                import numpy as _np
                _wf.write(tmp_path, sr, (samples * 32767).astype(_np.int16))
                reconhecido = _stt_vosk(tmp_path)
                if reconhecido:
                    sim = sum(c in reconhecido for c in texto) / max(len(texto), 1)
                    marca = "✓" if sim > 0.5 else "≈"
                    print(f"    STT: '{reconhecido}' {marca}", end="")
            except Exception:
                pass
            finally:
                try: os.unlink(tmp_path)
                except: pass

    return enviados

# ─── Main ─────────────────────────────────────────────────────────────────────

async def main_async():
    fases = [FASE_ALVO] if FASE_ALVO else sorted(CURRICULO.keys())

    print("=" * 60)
    print("  BÊ-Á-BÁ — TREINAMENTO FONÉTICO DA SELENE BRAIN 2.0")
    print("=" * 60)
    print(f"  Fases: {fases}  |  Repetições/fonema: {REPETICOES}")
    print(f"  WebSocket: {WS_URL}")
    print("=" * 60)

    for fase_num in fases:
        fase = CURRICULO.get(fase_num)
        if not fase:
            print(f"\nFase {fase_num} não encontrada.")
            continue

        print(f"\n{'─'*60}")
        print(f"  FASE {fase_num}: {fase['nome'].upper()}")
        print(f"  {fase['descricao']}")
        print(f"  {len(fase['itens'])} unidades fonéticas × {REPETICOES} reps")
        print(f"{'─'*60}")

        backoff = 1.0
        for tentativa in range(3):
            try:
                async with websockets.connect(
                    WS_URL, ping_interval=None, max_queue=512, open_timeout=15
                ) as ws:
                    print(f"  Conectado!")
                    t0 = time.time()
                    total_frames = 0

                    for idx, (fid, texto, lang) in enumerate(fase["itens"]):
                        print(f"\r  [{idx+1}/{len(fase['itens'])}] '{texto}' ({fid})...", end="", flush=True)
                        n = await enviar_fonema(ws, fid, texto, lang, REPETICOES)
                        total_frames += n

                    elapsed = time.time() - t0
                    print(f"\n  Fase {fase_num} concluída: {total_frames} frames em {elapsed:.1f}s")

                    # Solicita export
                    try:
                        await ws.send(json.dumps({"action": "export_linguagem"}))
                    except Exception:
                        pass
                break

            except (ConnectionRefusedError, OSError) as e:
                print(f"\n  Selene não acessível ({e}). Aguardando {backoff:.0f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            except websockets.exceptions.ConnectionClosed:
                print(f"\n  Conexão perdida. Tentativa {tentativa+1}/3...")
                await asyncio.sleep(2.0)

    print("\n" + "=" * 60)
    print("  Treinamento bê-á-bá concluído!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\nTreinamento interrompido pelo usuário.")
