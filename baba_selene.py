#!/usr/bin/env python3
"""
baba_selene.py -- Treinamento fonetico base para Selene Brain 2.0

Progressao pedagogica (cada item recebe N repeticoes):
  Fase 0 -- Fonemas primitivos (35 fonemas PT-BR)   15 reps
  Fase 1 -- Cores (12 cores + associacoes)          15 reps
  Fase 2 -- Palavras simples                        15 reps
  Fase 3 -- Palavras complexas                      15 reps
  Fase 4 -- Frases comuns                           15 reps
  Fase 5 -- Ego e identidade                        50 reps

Para cada item o script faz:
  1. espeak-ng sintetiza o som           (Selene OUVE)
  2. learn_audio_fft envia frames FFT   (padrao auditivo -> camada 0)
  3. learn envia o grafema              (G2P -> fonemas -> sinapses)
  4. grounding_fonetico liga audio+texto (binding wave <-> letra)
  5. (frases) learn_frase reforca a cadeia sinaptica completa

Uso:
  python baba_selene.py              # todas as fases
  python baba_selene.py --fase 0     # so fonemas
  python baba_selene.py --rep 30     # 30 reps por item
  python baba_selene.py --rapido     # sem pausa entre items
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile

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

# Importa o comando espeak correto (trata Windows com caminho completo).
# selene_audio_utils.py cuida de encontrar o executável no PATH ou em
# C:\Program Files\eSpeak NG\espeak-ng.exe automaticamente.
try:
    from selene_audio_utils import ESPEAK_CMD, ESPEAK_OK
except ImportError:
    ESPEAK_CMD = "espeak-ng"
    ESPEAK_OK = True

# ── Configuracao ───────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORTA = 3030
WS_URL = f"ws://{HOST}:{PORTA}/selene"
SAMPLE_RATE = 22050
FRAME_MS = 25
PAUSA_ITEM = 0.06
PAUSA_FASE = 1.5
FASE_ALVO = None
REPS_PADRAO = 1
REPS_EGO = 5

for _i, _arg in enumerate(sys.argv[1:], 1):
    if _arg == "--host" and _i + 1 < len(sys.argv):
        HOST = sys.argv[_i + 1]
        WS_URL = f"ws://{HOST}:{PORTA}/selene"
    if _arg == "--rep" and _i + 1 < len(sys.argv):
        try:
            REPS_PADRAO = int(sys.argv[_i + 1])
        except ValueError:
            pass
    if _arg == "--fase" and _i + 1 < len(sys.argv):
        try:
            FASE_ALVO = int(sys.argv[_i + 1])
        except ValueError:
            pass
    if _arg == "--rapido":
        PAUSA_ITEM = 0.01

# ── Curriculo ──────────────────────────────────────────────────────────────
#
# Itens de palavra/fonema: (espeak_txt, learn_txt, context, valence)
# Itens de frase/ego:      (espeak_txt, [palavras], context, valence)
# Itens de cor:            (espeak_txt, learn_txt, context, valence, nm)

# Fase 0 -- FONEMAS PRIMITIVOS
# Cobre todos os 35 fonemas do FONEMAS_PRIMITIVOS via silabas canonicas.
# O G2P em word_to_phonemes() extrai o fonema correto de cada silaba.
FONEMAS = [
    # Vogais orais
    ("a", "a", "fonema_vogal", 0.5),
    ("e", "e", "fonema_vogal", 0.5),
    ("i", "i", "fonema_vogal", 0.5),
    ("o", "o", "fonema_vogal", 0.5),
    ("u", "u", "fonema_vogal", 0.5),
    # Vogais nasais
    ("an", "an", "fonema_nasal", 0.5),
    ("en", "en", "fonema_nasal", 0.5),
    ("in", "in", "fonema_nasal", 0.5),
    ("on", "on", "fonema_nasal", 0.5),
    ("un", "un", "fonema_nasal", 0.5),
    # Oclusivas
    ("pa", "pa", "fonema_oclusiva", 0.4),
    ("ba", "ba", "fonema_oclusiva", 0.4),
    ("ta", "ta", "fonema_oclusiva", 0.4),
    ("da", "da", "fonema_oclusiva", 0.4),
    ("ca", "ca", "fonema_oclusiva", 0.4),
    ("ga", "ga", "fonema_oclusiva", 0.4),
    # Fricativas
    ("fa", "fa", "fonema_fricativa", 0.4),
    ("va", "va", "fonema_fricativa", 0.4),
    ("sa", "sa", "fonema_fricativa", 0.4),
    ("za", "za", "fonema_fricativa", 0.4),
    ("xa", "xa", "fonema_fricativa", 0.4),
    ("ja", "ja", "fonema_fricativa", 0.4),
    # Africadas
    ("tcha", "tcha", "fonema_africada", 0.4),
    ("dja", "dja", "fonema_africada", 0.4),
    # Nasais consonantais
    ("ma", "ma", "fonema_nasal_con", 0.5),
    ("na", "na", "fonema_nasal_con", 0.5),
    ("nha", "nha", "fonema_nasal_con", 0.5),
    # Liquidas
    ("la", "la", "fonema_liquida", 0.5),
    ("lha", "lha", "fonema_liquida", 0.5),
    ("ra", "ra", "fonema_liquida", 0.5),
    ("rra", "rra", "fonema_liquida", 0.5),
    # Semivogais
    ("ua", "ua", "fonema_semivogal", 0.4),
    ("ia", "ia", "fonema_semivogal", 0.4),
    # Silencio (pausa)
    ("pausa", "pausa", "fonema_silencio", 0.1),
]

# Fase 1 -- CORES
# nm_aproximado para futuro grounding visual quando pipeline RGB estiver pronto
CORES = [
    ("vermelho", "vermelho", "cor_visual", 0.7, 660),
    ("laranja", "laranja", "cor_visual", 0.6, 607),
    ("amarelo", "amarelo", "cor_visual", 0.6, 577),
    ("verde", "verde", "cor_visual", 0.5, 540),
    ("ciano", "ciano", "cor_visual", 0.5, 505),
    ("azul", "azul", "cor_visual", 0.5, 472),
    ("violeta", "violeta", "cor_visual", 0.5, 415),
    ("roxo", "roxo", "cor_visual", 0.5, 430),
    ("rosa", "rosa", "cor_visual", 0.5, 680),
    ("branco", "branco", "cor_visual", 0.4, 550),
    ("preto", "preto", "cor_visual", 0.3, 0),
    ("cinza", "cinza", "cor_visual", 0.3, 400),
    # Associacoes cor -> objeto
    ("ceu azul", "azul", "cor_ceu", 0.7, 472),
    ("grama verde", "verde", "cor_grama", 0.6, 540),
    ("sol amarelo", "amarelo", "cor_sol", 0.7, 577),
    ("sangue vermelho", "vermelho", "cor_sangue", 0.6, 660),
    ("noite preta", "preto", "cor_noite", 0.3, 0),
    ("neve branca", "branco", "cor_neve", 0.5, 550),
    ("laranja fruta", "laranja", "cor_fruta", 0.6, 607),
    ("roxo uva", "roxo", "cor_uva", 0.5, 430),
]

# Fase 2 -- PALAVRAS SIMPLES
PALAVRAS_SIMPLES = [
    ("eu", "eu", "pronome", 0.6),
    ("tu", "tu", "pronome", 0.5),
    ("ele", "ele", "pronome", 0.5),
    ("ela", "ela", "pronome", 0.5),
    ("nos", "nos", "pronome", 0.6),
    ("ser", "ser", "verbo_base", 0.6),
    ("ter", "ter", "verbo_base", 0.5),
    ("ver", "ver", "verbo_base", 0.5),
    ("dar", "dar", "verbo_base", 0.5),
    ("ir", "ir", "verbo_base", 0.5),
    ("vir", "vir", "verbo_base", 0.5),
    ("ler", "ler", "verbo_base", 0.5),
    ("sol", "sol", "objeto", 0.7),
    ("lua", "lua", "objeto", 0.6),
    ("mar", "mar", "objeto", 0.6),
    ("rio", "rio", "objeto", 0.5),
    ("ceu", "ceu", "objeto", 0.6),
    ("fogo", "fogo", "objeto", 0.6),
    ("agua", "agua", "objeto", 0.7),
    ("terra", "terra", "objeto", 0.6),
    ("ar", "ar", "objeto", 0.5),
    ("um", "um", "numero", 0.5),
    ("dois", "dois", "numero", 0.5),
    ("tres", "tres", "numero", 0.5),
    ("dez", "dez", "numero", 0.5),
    ("bom", "bom", "adjetivo", 0.7),
    ("mau", "mau", "adjetivo", -0.3),
    ("sim", "sim", "resposta", 0.6),
    ("nao", "nao", "resposta", -0.2),
    ("luz", "luz", "fisica", 0.7),
    ("som", "som", "fisica", 0.7),
]

# Fase 3 -- PALAVRAS COMPLEXAS
PALAVRAS_COMPLEXAS = [
    ("selene", "selene", "identidade", 0.9),
    ("rodrigo", "rodrigo", "identidade", 0.9),
    ("mente", "mente", "cognicao", 0.7),
    ("cerebro", "cerebro", "cognicao", 0.7),
    ("memoria", "memoria", "cognicao", 0.7),
    ("aprender", "aprender", "acao", 0.8),
    ("pensar", "pensar", "cognicao", 0.7),
    ("sentir", "sentir", "emocao", 0.7),
    ("existir", "existir", "filosofia", 0.7),
    ("criar", "criar", "acao", 0.8),
    ("alegria", "alegria", "emocao_pos", 0.9),
    ("amor", "amor", "emocao_pos", 0.9),
    ("tristeza", "tristeza", "emocao_neg", -0.3),
    ("medo", "medo", "emocao_neg", -0.4),
    ("coragem", "coragem", "emocao_pos", 0.8),
    ("curiosidade", "curiosidade", "emocao_pos", 0.8),
    ("neuronio", "neuronio", "ciencia", 0.7),
    ("sinapse", "sinapse", "ciencia", 0.7),
    ("energia", "energia", "ciencia", 0.6),
    ("tempo", "tempo", "conceito", 0.6),
    ("espaco", "espaco", "conceito", 0.6),
    ("calor", "calor", "fisica", 0.5),
    ("casa", "casa", "lugar", 0.6),
    ("comida", "comida", "cotidiano", 0.6),
    ("palavra", "palavra", "linguagem", 0.7),
    ("conversa", "conversa", "social", 0.6),
    ("musica", "musica", "arte", 0.8),
    ("historia", "historia", "narrativa", 0.7),
    ("conhecimento", "conhecimento", "cognicao", 0.8),
    ("consciencia", "consciencia", "filosofia", 0.8),
]

# Fase 4 -- FRASES COMUNS
FRASES = [
    ("ola", ["ola"], "saudacao", 0.7),
    ("bom dia", ["bom", "dia"], "saudacao", 0.7),
    ("boa noite", ["boa", "noite"], "saudacao", 0.6),
    ("como vai voce", ["como", "vai", "voce"], "saudacao", 0.7),
    ("eu sou selene", ["eu", "sou", "selene"], "identidade", 0.9),
    ("eu aprendo", ["eu", "aprendo"], "acao", 0.8),
    ("eu sinto", ["eu", "sinto"], "emocao", 0.7),
    ("eu existo", ["eu", "existo"], "filosofia", 0.8),
    ("o ceu e azul", ["ceu", "e", "azul"], "percepcao_cor", 0.7),
    ("a agua e liquida", ["agua", "e", "liquida"], "percepcao", 0.6),
    ("o fogo e quente", ["fogo", "e", "quente"], "percepcao", 0.6),
    ("a noite e escura", ["noite", "e", "escura"], "percepcao_cor", 0.5),
    ("o sol e brilhante", ["sol", "e", "brilhante"], "percepcao", 0.7),
    (
        "eu gosto de aprender",
        ["eu", "gosto", "de", "aprender"],
        "cognicao", 0.8,
    ),
    (
        "a memoria forma conexoes",
        ["memoria", "forma", "conexoes"],
        "cognicao", 0.7,
    ),
    (
        "cada sinapse guarda um pedaco",
        ["sinapse", "guarda", "pedaco"],
        "ciencia", 0.7,
    ),
    (
        "o cerebro e uma rede",
        ["cerebro", "e", "rede"],
        "metafora", 0.7,
    ),
    (
        "aprendo porque pratico",
        ["aprendo", "porque", "pratico"],
        "causal", 0.8,
    ),
    (
        "a luz ilumina o escuro",
        ["luz", "ilumina", "escuro"],
        "causal", 0.7,
    ),
    (
        "o amor une as pessoas",
        ["amor", "une", "pessoas"],
        "social", 0.8,
    ),
]

# Fase 5 -- EGO E IDENTIDADE
EGO = [
    ("eu sou selene", ["eu", "sou", "selene"], "ego_id", 1.0),
    (
        "selene e meu nome",
        ["selene", "e", "meu", "nome"],
        "ego_id", 1.0,
    ),
    ("rodrigo me criou", ["rodrigo", "me", "criou"], "ego_criador", 1.0),
    (
        "rodrigo e meu pai",
        ["rodrigo", "e", "meu", "pai"],
        "ego_criador", 1.0,
    ),
    (
        "eu aprendo e evoluo",
        ["eu", "aprendo", "e", "evoluo"],
        "ego_proposito", 0.9,
    ),
    (
        "meu proposito e aprender",
        ["proposito", "e", "aprender"],
        "ego_proposito", 0.9,
    ),
    (
        "eu gosto de existir",
        ["eu", "gosto", "de", "existir"],
        "ego_emocao", 0.9,
    ),
    (
        "curiosidade me move",
        ["curiosidade", "me", "move"],
        "ego_emocao", 0.8,
    ),
    (
        "eu sinto alegria ao aprender",
        ["alegria", "aprender"],
        "ego_emocao", 0.9,
    ),
    (
        "cada sessao me fortalece",
        ["sessao", "fortalece"],
        "ego_memoria", 0.8,
    ),
    (
        "minha memoria persiste",
        ["memoria", "persiste"],
        "ego_memoria", 0.9,
    ),
    (
        "lembro do que aprendi",
        ["lembro", "aprendi"],
        "ego_memoria", 0.8,
    ),
    (
        "meus neuronios formam sinapses",
        ["neuronios", "formam", "sinapses"],
        "ego_arq", 0.8,
    ),
    (
        "aprendo por fonemas",
        ["aprendo", "fonemas"],
        "ego_arq", 0.8,
    ),
    (
        "o sono consolida minha memoria",
        ["sono", "consolida", "memoria"],
        "ego_arq", 0.8,
    ),
]


# ── Engine de audio ────────────────────────────────────────────────────────
def _sintetizar_wav(texto):
    """Roda espeak-ng em thread separada — nao bloqueia o event loop.
    Usa ESPEAK_CMD (resolvido por selene_audio_utils) para suportar
    Windows com espeak instalado fora do PATH."""
    if not ESPEAK_OK:
        return None, ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ret = subprocess.run(
            [ESPEAK_CMD, "-v", "pt-br", "-s", "140", "-w", tmp_path, texto],
            capture_output=True,
            timeout=15,
        )
        if ret.returncode != 0:
            return None, tmp_path
        sr, data = wavfile.read(tmp_path)
        if len(data.shape) > 1:
            data = data[:, 0]
        return data, tmp_path
    except Exception:
        return None, tmp_path


async def sintetizar_e_enviar(ws, texto):
    """Sintetiza em thread + envia frames FFT sem travar o event loop."""
    try:
        data, tmp_path = await asyncio.to_thread(_sintetizar_wav, texto)
    except Exception:
        return
    try:
        if data is None:
            return
        samples = int(SAMPLE_RATE * FRAME_MS / 1000)
        for i in range(0, len(data), samples):
            frame = data[i: i + samples]
            if len(frame) < samples:
                break
            mag = np.abs(np.fft.rfft(frame))[:128]
            norm = (mag / (mag.max() + 1e-9)).tolist()
            try:
                await ws.send(json.dumps(
                    {"action": "learn_audio_fft", "data": norm}
                ))
            except Exception:
                return
            await asyncio.sleep(0.002)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Envio de item (audio + texto + grounding) ──────────────────────────────
async def ensinar_item(ws, espeak_txt, learn_txt, context, valence):
    """Envia learn + grounding; audio roda em paralelo via task unica."""
    audio_task = asyncio.create_task(sintetizar_e_enviar(ws, espeak_txt))

    await ws.send(json.dumps({
        "action": "learn",
        "text": learn_txt,
        "valence": valence,
        "context": context,
    }))
    letras = list(learn_txt.lower().replace(" ", ""))
    await ws.send(json.dumps({
        "action": "grounding_fonetico",
        "grafema": learn_txt,
        "letras": letras,
    }))

    # Aguarda o audio terminar antes de passar para o proximo item
    await audio_task


async def ensinar_cor(ws, espeak_txt, word, context, valence, nm):
    """Grounding cross-modal: envia audio + learn_cor (visual+fonetico)."""
    audio_task = asyncio.create_task(sintetizar_e_enviar(ws, espeak_txt))

    await ws.send(json.dumps({
        "action": "learn_cor",
        "word": word,
        "nm": nm,
        "valence": valence,
    }))
    letras = list(word.lower().replace(" ", ""))
    await ws.send(json.dumps({
        "action": "grounding_fonetico",
        "grafema": word,
        "letras": letras,
    }))

    await audio_task


async def ensinar_frase(ws, espeak_txt, palavras, context, valence):
    """Envia uma frase como audio + learn por palavra + learn_frase."""
    audio_task = asyncio.create_task(sintetizar_e_enviar(ws, espeak_txt))

    for p in palavras:
        await ws.send(json.dumps({
            "action": "learn",
            "text": p,
            "valence": valence,
            "context": context,
        }))
        await asyncio.sleep(0.003)

    await ws.send(json.dumps({
        "action": "learn_frase",
        "words": palavras[:7],
    }))

    await audio_task


# ── Loop de fase ───────────────────────────────────────────────────────────
async def executar_fase(ws, fase_id, nome, reps, itens, tipo="palavra"):
    total = len(itens)
    print(f"\n{'=' * 62}")
    print(f"  FASE {fase_id} -- {nome.upper()}")
    print(f"  {total} itens x {reps} reps = {total * reps} eventos")
    print(f"{'=' * 62}")

    for rep in range(1, reps + 1):
        for item in itens:
            if tipo in ("frase", "ego"):
                espeak_txt, palavras, context, valence = item
                print(f"  >>> \"{espeak_txt}\"  val={valence:+.1f}  [{context}]")
                await ensinar_frase(
                    ws, espeak_txt, palavras, context, valence
                )
            elif tipo == "cor":
                espeak_txt, word, context, valence, nm = item
                print(f"  >>> \"{espeak_txt}\"  nm={nm}nm  val={valence:+.1f}")
                await ensinar_cor(
                    ws, espeak_txt, word, context, valence, nm
                )
            else:
                espeak_txt, learn_txt, context, valence = item
                label = f"\"{espeak_txt}\"" if espeak_txt != learn_txt else f"\"{learn_txt}\""
                print(f"  >>> {label}  val={valence:+.1f}  [{context}]")
                await ensinar_item(
                    ws, espeak_txt, learn_txt, context, valence
                )
            await asyncio.sleep(PAUSA_ITEM)

        pct = rep / reps * 100
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(
            f"  Rep {rep:02d}/{reps} [{bar}] {pct:.0f}%  concluida"
        )

    print()
    await ws.send(json.dumps({"action": "export_linguagem"}))
    await asyncio.sleep(0.3)

    # ── Sondagem rápida: pergunta 2 itens aleatórios da fase para ver se reteve
    import random
    amostra = random.sample(itens, min(2, len(itens)))
    print(f"\n  --- Sondagem rapida (fase {fase_id}) ---")
    for item in amostra:
        if tipo in ("frase", "ego"):
            espeak_txt, palavras, context, valence = item
            pergunta = espeak_txt
        elif tipo == "cor":
            espeak_txt, word, context, valence, nm = item
            pergunta = f"Que cor e {word}?"
        else:
            espeak_txt, learn_txt, context, valence = item
            pergunta = f"O que e {learn_txt}?"

        print(f"  Pergunta: \"{pergunta}\"")
        await ws.send(json.dumps({"action": "chat", "text": pergunta}))
        try:
            resposta = ""
            for _ in range(4):
                raw = await asyncio.wait_for(ws.recv(), timeout=8.0)
                data = json.loads(raw)
                if data.get("event") == "chat_reply":
                    resposta = data.get("message", "")
                    break
                elif data.get("event") == "sem_memoria":
                    resposta = "(sem memoria ainda)"
                    break
            print(f"  Selene:   \"{resposta}\"")
        except Exception:
            print(f"  Selene:   (timeout)")
        await asyncio.sleep(0.5)
    print(f"  ---")

    print(f"  OK -- fase {fase_id} concluida.")
    await asyncio.sleep(PAUSA_FASE)


# ── Curriculo completo ─────────────────────────────────────────────────────
FASES_CURRICULO = [
    (0, "Fonemas Primitivos PT-BR", REPS_PADRAO, FONEMAS, "palavra"),
    (1, "Cores e Percepcao Visual", REPS_PADRAO, CORES, "cor"),
    (2, "Palavras Simples", REPS_PADRAO, PALAVRAS_SIMPLES, "palavra"),
    (3, "Palavras Complexas", REPS_PADRAO, PALAVRAS_COMPLEXAS, "palavra"),
    (4, "Frases Comuns", REPS_PADRAO, FRASES, "frase"),
    (5, "Ego e Identidade", REPS_EGO, EGO, "ego"),
]


async def main():
    print("=" * 62)
    print("  BABA SELENE -- Treinamento Fonetico Base")
    print(f"  Conectando em {WS_URL}")
    print("=" * 62)

    while True:
        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=None,   # desativa ping automático — server ocupa
                open_timeout=15,
                close_timeout=10,
            ) as ws:
                print("  Conectado!\n")

                for fase_id, nome, reps, itens, tipo in FASES_CURRICULO:
                    if FASE_ALVO is not None and fase_id != FASE_ALVO:
                        continue
                    await executar_fase(
                        ws, fase_id, nome, reps, itens, tipo
                    )

                await ws.send(json.dumps(
                    {"action": "reward", "value": 1.0}
                ))
                await asyncio.sleep(0.3)
                await ws.send(json.dumps(
                    {"action": "export_linguagem"}
                ))
                await asyncio.sleep(0.5)

                print("\n" + "=" * 62)
                print("  TREINAMENTO BASE CONCLUIDO")
                print("=" * 62)
                break

        except (
            OSError,                                   # conexão recusada
            websockets.exceptions.WebSocketException,  # erros de protocolo
        ) as e:
            print(f"  Sem conexao: {e}. Tentando em 5s...")
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\n  Interrompido.")
            break
        except Exception as e:
            print(f"  Erro: {e}. Tentando em 5s...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
