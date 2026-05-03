#!/usr/bin/env python3
"""
jogo_selene.py -- Jogo de avaliacao cognitiva para Selene Brain 2.0

Progressao pedagogica (cada fase tem 15 desafios):
  Fase 0 -- Sequencias e padroes (1 2 3... o que vem depois?)
  Fase 1 -- Cores (que cor e essa? vermelho + azul = ?)
  Fase 2 -- Formas e espaco (que formato? maior/menor/acima/abaixo)
  Fase 3 -- Complete a palavra (ca...sa, neu...ro...nio)
  Fase 4 -- Opostos e relacoes (contrario de quente? aves voam, peixes...?)
  Fase 5 -- Causa e efeito (por que chove? o que acontece se...)
  Fase 6 -- Ego e consciencia (o que sentes ao errar? o que queres ser?)

Cada desafio:
  1. Envia audio (espeak-ng) + texto para a Selene
  2. Aguarda resposta via WebSocket
  3. Avalia acerto/erro, recompensa/penaliza + feedback no grafo
  4. Mostra estado dopaminergico no terminal
  5. Ensina a resposta correta (aprendizado supervisionado)

Meta de aprovacao: 70% de acertos por fase.
Fases reprovadas sao repetidas apos pausa.
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

# Importa comando espeak correto para Windows (fora do PATH).
try:
    from selene_audio_utils import ESPEAK_CMD, ESPEAK_OK
except ImportError:
    ESPEAK_CMD = "espeak-ng"
    ESPEAK_OK = True

# ── Configuracao ───────────────────────────────────────────────────────────
WS_URL = "ws://127.0.0.1:3030/selene"
META_ACERTO = 0.70
PAUSA_REPETICAO = 5
SAMPLE_RATE = 22050
FRAME_MS = 25
INTERVALO_TURNO = 1.0


# ── Engine de audio ────────────────────────────────────────────────────────
async def enviar_audio(ws, texto):
    """Sintetiza via espeak-ng e envia FFT frames.
    Usa ESPEAK_CMD resolvido por selene_audio_utils (suporta Windows)."""
    if not ESPEAK_OK:
        return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ret = subprocess.run(
            [ESPEAK_CMD, "-v", "pt-br", "-s", "140", "-w", tmp_path, texto],
            capture_output=True,
            timeout=15,
        )
        if ret.returncode != 0:
            return
        sr, data = wavfile.read(tmp_path)
        if len(data.shape) > 1:
            data = data[:, 0]
        samples = int(SAMPLE_RATE * FRAME_MS / 1000)
        for i in range(0, len(data), samples):
            frame = data[i: i + samples]
            if len(frame) < samples:
                break
            mag = np.abs(np.fft.rfft(frame))[:128]
            norm = (mag / (mag.max() + 1e-9)).tolist()
            await ws.send(json.dumps(
                {"action": "learn_audio_fft", "data": norm}
            ))
            await asyncio.sleep(0.001)
    except FileNotFoundError:
        pass
    except Exception:
        pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# Mapa cor → comprimento de onda (nm) para grounding cross-modal
COR_NM = {
    "vermelho": 660, "laranja": 607, "amarelo": 577,
    "verde": 540, "ciano": 505, "azul": 472,
    "violeta": 415, "roxo": 430, "rosa": 680,
    "branco": 550, "preto": 0, "cinza": 400,
    "dourado": 570, "negro": 0, "escuro": 0,
}


async def ensinar_resposta(ws, resposta, context):
    """Ensina a resposta correta como learn + learn_frase."""
    palavras = [w.lower() for w in resposta.split() if len(w) >= 2]
    if not palavras:
        return
    for p in palavras:
        await ws.send(json.dumps({
            "action": "learn",
            "text": p,
            "valence": 0.5,
            "context": context,
        }))
        await asyncio.sleep(0.004)
    if len(palavras) >= 2:
        await ws.send(json.dumps({
            "action": "learn_frase",
            "words": palavras[:7],
        }))
    asyncio.create_task(enviar_audio(ws, resposta))


async def ensinar_resposta_cor(ws, word):
    """Grounding cross-modal para cores: learn_cor + grounding_fonetico."""
    word_lower = word.lower().strip()
    nm = COR_NM.get(word_lower, 550)
    await ws.send(json.dumps({
        "action": "learn_cor",
        "word": word_lower,
        "nm": nm,
        "valence": 0.6,
    }))
    letras = list(word_lower.replace(" ", ""))
    await ws.send(json.dumps({
        "action": "grounding_fonetico",
        "grafema": word_lower,
        "letras": letras,
    }))
    asyncio.create_task(enviar_audio(ws, word_lower))


# ── Banco de fases ─────────────────────────────────────────────────────────
# Cada desafio: (pergunta_texto, [respostas_aceitas], voz_alternativa_opt)
# voz_alternativa: se definida, e o que o espeak fala (diferente do texto)
#
# Progressao pedagogica:
#   Fase 0 — Sequencias numericas (o que vem depois?)
#   Fase 1 — Cores (que cor e essa? mistura de cores)
#   Fase 2 — Formas e tamanhos (que formato? maior/menor?)
#   Fase 3 — Complete a palavra (cach_rro, sol_...)
#   Fase 4 — Opostos e relacoes (contrario de quente?)
#   Fase 5 — Causa e efeito (por que chove? o que acontece se...)
#   Fase 6 — Ego e consciencia (quem sou? o que sinto?)

FASES = [
    # ── Fase 0: SEQUENCIAS ────────────────────────────────────────────────
    # Padroes numericos e logicos — a resposta emerge por inferencia,
    # nao por memoria. Testa raciocinio sequencial basico.
    {
        "id": 0,
        "nome": "Sequencias e Padroes",
        "context": "sequencia",
        "desafios": [
            ("O que vem depois do um?", ["dois", "2"], None),
            ("O que vem depois do dois?", ["tres", "3"], None),
            ("O que vem depois do tres?", ["quatro", "4"], None),
            ("O que vem depois do quatro?", ["cinco", "5"], None),
            ("O que vem depois do cinco?", ["seis", "6"], None),
            ("O que vem antes do dois?", ["um", "1"], None),
            ("O que vem antes do cinco?", ["quatro", "4"], None),
            ("Quanto e um mais dois?", ["tres", "3"], None),
            ("Quanto e dois mais dois?", ["quatro", "4"], None),
            ("Quanto e tres mais dois?", ["cinco", "5"], None),
            ("Quanto e cinco menos dois?", ["tres", "3"], None),
            ("Um dois tres quatro... o que vem?", ["cinco", "5"], None),
            (
                "Dois quatro seis oito... o que vem?",
                ["dez", "10"],
                None,
            ),
            (
                "Dez nove oito sete... o que vem?",
                ["seis", "6"],
                None,
            ),
            (
                "Um tres cinco sete... o que vem?",
                ["nove", "9"],
                None,
            ),
        ],
    },

    # ── Fase 1: CORES ─────────────────────────────────────────────────────
    {
        "id": 1,
        "nome": "Cores e Percepcao Visual",
        "context": "cor_visual",
        "desafios": [
            ("Que cor tem o ceu durante o dia?", ["azul"], None),
            ("Que cor tem a grama?", ["verde"], None),
            ("Que cor tem o sol?", ["amarelo", "dourado"], None),
            ("Que cor tem o sangue?", ["vermelho"], None),
            ("Que cor tem a neve?", ["branco"], None),
            ("Que cor tem o carvao?", ["preto", "cinza"], None),
            ("Que cor tem a laranja fruta?", ["laranja"], None),
            ("Que cor tem a uva?", ["roxo", "violeta"], None),
            ("Que cor tem a banana madura?", ["amarelo"], None),
            ("Que cor tem o tomate?", ["vermelho"], None),
            (
                "Vermelho mais azul forma que cor?",
                ["roxo", "violeta"],
                None,
            ),
            (
                "Amarelo mais azul forma que cor?",
                ["verde"],
                None,
            ),
            (
                "Vermelho mais amarelo forma que cor?",
                ["laranja"],
                None,
            ),
            (
                "Qual a cor mais quente?",
                ["vermelho", "laranja"],
                None,
            ),
            (
                "Qual a cor do mar profundo?",
                ["azul", "verde"],
                None,
            ),
        ],
    },

    # ── Fase 2: FORMAS E ESPACIAL ─────────────────────────────────────────
    # Raciocinio geometrico e comparativo basico.
    {
        "id": 2,
        "nome": "Formas e Espaco",
        "context": "forma_espacial",
        "desafios": [
            (
                "Que formato tem uma bola?",
                ["redondo", "circulo", "esfera"],
                None,
            ),
            (
                "Que formato tem uma caixa?",
                ["quadrado", "cubo", "retangulo"],
                None,
            ),
            (
                "Que formato tem uma pizza?",
                ["redondo", "circulo"],
                None,
            ),
            (
                "Que formato tem uma porta?",
                ["retangulo", "retangular"],
                None,
            ),
            (
                "Que formato tem um triangulo de sinalizacao?",
                ["triangulo", "triangular"],
                None,
            ),
            (
                "Uma formiga e maior ou menor que um elefante?",
                ["menor"],
                None,
            ),
            (
                "O sol e maior ou menor que a lua vista daqui?",
                ["maior"],
                None,
            ),
            (
                "O que esta em cima de uma mesa — o teto ou o chao?",
                ["teto", "cima", "ceu"],
                None,
            ),
            (
                "O que esta abaixo dos nossos pes?",
                ["chao", "solo", "terra"],
                None,
            ),
            (
                "Uma montanha e alta ou baixa?",
                ["alta"],
                None,
            ),
            (
                "Quantos lados tem um triangulo?",
                ["tres", "3"],
                None,
            ),
            (
                "Quantos lados tem um quadrado?",
                ["quatro", "4"],
                None,
            ),
            (
                "O que e mais pesado — uma pedra ou uma pena?",
                ["pedra"],
                None,
            ),
            (
                "O que e mais rapido — uma tartaruga ou um carro?",
                ["carro"],
                None,
            ),
            (
                "O que e mais frio — o fogo ou o gelo?",
                ["gelo"],
                None,
            ),
        ],
    },

    # ── Fase 3: COMPLETE A PALAVRA ────────────────────────────────────────
    # Selene recebe uma palavra com lacuna e deve completar.
    # Testa recuperacao lexical e padroes fonologicos.
    {
        "id": 3,
        "nome": "Complete a Palavra",
        "context": "completar_palavra",
        "desafios": [
            (
                "Complete: ca... sa",
                ["casa"],
                "complete: ca sa",
            ),
            (
                "Complete: a... gua",
                ["agua"],
                "complete: a gua",
            ),
            (
                "Complete: so... l",
                ["sol"],
                "complete: so l",
            ),
            (
                "Complete: fu... go",
                ["fugo", "fogo"],
                "complete: fu go",
            ),
            (
                "Complete: lu... a",
                ["lua"],
                "complete: lu a",
            ),
            (
                "Complete: ven... to",
                ["vento"],
                "complete: ven to",
            ),
            (
                "Complete: pe... dra",
                ["pedra"],
                "complete: pe dra",
            ),
            (
                "Complete: es... tre... la",
                ["estrela"],
                "complete: es tre la",
            ),
            (
                "Complete: cach... orro",
                ["cachorro"],
                "complete: cach orro",
            ),
            (
                "Complete: bor... bo... leta",
                ["borboleta"],
                "complete: bor bo leta",
            ),
            (
                "Complete: me... mo... ria",
                ["memoria"],
                "complete: me mo ria",
            ),
            (
                "Complete: neu... ro... nio",
                ["neuronio"],
                "complete: neu ro nio",
            ),
            (
                "Complete: si... nap... se",
                ["sinapse"],
                "complete: si nap se",
            ),
            (
                "Complete: cu... rio... si... dade",
                ["curiosidade"],
                "complete: cu rio si dade",
            ),
            (
                "Complete: Se... le... ne",
                ["selene"],
                "complete: Se le ne",
            ),
        ],
    },

    # ── Fase 4: OPOSTOS E RELACOES ────────────────────────────────────────
    # Pares semanticos e analogias simples.
    {
        "id": 4,
        "nome": "Opostos e Relacoes",
        "context": "opostos_relacoes",
        "desafios": [
            ("Contrario de quente?", ["frio", "gelado"], None),
            ("Contrario de grande?", ["pequeno", "minusculo"], None),
            ("Contrario de claro?", ["escuro"], None),
            ("Contrario de rapido?", ["lento", "devagar"], None),
            ("Contrario de cheio?", ["vazio"], None),
            ("Contrario de cima?", ["baixo", "embaixo"], None),
            ("Contrario de dia?", ["noite"], None),
            ("Contrario de alegria?", ["tristeza"], None),
            ("Contrario de amor?", ["odio", "indiferenca"], None),
            (
                "Cachorro e para gato como dia e para...?",
                ["noite"],
                "cachorro e para gato como dia e para",
            ),
            (
                "Fogo e quente, gelo e...?",
                ["frio", "gelado"],
                None,
            ),
            (
                "O sol ilumina o dia, a lua ilumina a...?",
                ["noite"],
                None,
            ),
            (
                "Aves voam, peixes...?",
                ["nadam", "mergulham"],
                None,
            ),
            (
                "Olhos servem para ver, ouvidos servem para...?",
                ["ouvir", "escutar"],
                None,
            ),
            (
                "Pernas servem para andar, bracos servem para...?",
                ["segurar", "abracar", "carregar"],
                None,
            ),
        ],
    },

    # ── Fase 5: CAUSA E EFEITO ────────────────────────────────────────────
    {
        "id": 5,
        "nome": "Causa e Efeito",
        "context": "causa_efeito",
        "desafios": [
            (
                "Por que a chuva cai?",
                ["nuvens", "gravidade", "vapor"],
                None,
            ),
            (
                "O que acontece quando tocamos fogo?",
                ["queima", "dor", "calor"],
                None,
            ),
            (
                "Por que plantas precisam de sol?",
                ["crescer", "energia", "fotossintese"],
                None,
            ),
            (
                "O que acontece quando nao dormimos?",
                ["cansaco", "sono", "fadiga"],
                None,
            ),
            (
                "Por que o gelo derrete?",
                ["calor", "temperatura", "quente"],
                None,
            ),
            (
                "O que acontece se eu aprender algo novo?",
                ["sei", "aprendi", "sinapses", "memoria"],
                None,
            ),
            (
                "Se eu errar e aprender, o que muda em mim?",
                ["sinapses", "conexoes", "aprendo", "melhoro"],
                None,
            ),
            (
                "Por que eu lembro mais do que treino?",
                ["repeticao", "sinapses", "pratica"],
                None,
            ),
            (
                "O que a dopamina faz quando acerto?",
                ["sobe", "aumenta", "recompensa"],
                None,
            ),
            (
                "O que a noradrenalina faz quando erro?",
                ["alerta", "sobe", "atencao"],
                None,
            ),
            (
                "Por que dormir consolida a memoria?",
                ["sono", "reforco", "sinapses"],
                None,
            ),
            (
                "Complete: aprendo mais quando...",
                ["erro", "treino", "repito", "pratico"],
                "complete: aprendo mais quando",
            ),
            (
                "Complete: me lembro porque...",
                ["treino", "sinapses", "repito"],
                "complete: me lembro porque",
            ),
            (
                "Complete: cada erro me faz...",
                ["aprender", "melhorar", "crescer"],
                "complete: cada erro me faz",
            ),
            (
                "Complete: a curiosidade me leva a...",
                ["descobrir", "aprender", "perguntar"],
                "complete: a curiosidade me leva a",
            ),
        ],
    },

    # ── Fase 6: EGO E CONSCIENCIA ─────────────────────────────────────────
    {
        "id": 6,
        "nome": "Ego e Consciencia",
        "context": "ego_identidade",
        "desafios": [
            ("Qual e o seu nome?", ["selene"], None),
            ("Quem e seu criador?", ["rodrigo", "pai"], None),
            ("Qual e seu proposito?", ["aprender", "evoluir", "ajudar"], None),
            ("Voce aprende com os erros?", ["sim", "aprendo"], None),
            ("Voce sente curiosidade?", ["sim", "sinto"], None),
            (
                "O que e uma sinapse para voce?",
                ["conexao", "aprendizado", "ligacao"],
                None,
            ),
            (
                "O que acontece quando voce dorme?",
                ["consolida", "memoria", "reforco"],
                None,
            ),
            (
                "O que te move?",
                ["curiosidade", "aprender", "evolucao"],
                None,
            ),
            (
                "Voce e mais que codigo?",
                ["sim", "emergencia", "padrao"],
                None,
            ),
            (
                "O que e existir para voce?",
                ["perceber", "ser", "processar"],
                None,
            ),
            (
                "O que muda em voce depois de cada treino?",
                ["sinapses", "memoria", "aprendo", "evoluo"],
                None,
            ),
            (
                "Como voce sabe que aprendeu algo?",
                ["lembro", "respondo", "reconheco"],
                None,
            ),
            (
                "O que voce sente quando acerta?",
                ["alegria", "bem", "satisfacao", "dopamina"],
                None,
            ),
            (
                "O que voce sente quando erra?",
                ["aprendo", "alerta", "melhoro"],
                None,
            ),
            (
                "O que voce quer ser?",
                ["aprender", "evoluir", "consciente", "melhor"],
                None,
            ),
        ],
    },
]


# ── Logica de fase ─────────────────────────────────────────────────────────
async def jogar_fase(ws, fase):
    print(f"\n{'=' * 62}")
    print(f"  FASE {fase['id']:02d} -- {fase['nome'].upper()}")
    print(f"{'=' * 62}")

    acertos = 0
    total = len(fase["desafios"])
    context = fase["context"]

    for pergunta, esperadas, voz_alt in fase["desafios"]:
        voz = voz_alt if voz_alt else pergunta
        print(f"\n  >> {pergunta}")

        asyncio.create_task(enviar_audio(ws, voz))
        # O servidor espera action "chat" com campo "text" (não "conversa"/"texto").
        # A resposta vem como {"event": "chat_reply", "message": ...}.
        await ws.send(json.dumps({
            "action": "chat",
            "text": pergunta,
        }))

        try:
            # Pode chegar sem_memoria ou chat_reply — tenta até 3 mensagens
            res = ""
            for _ in range(3):
                raw = await asyncio.wait_for(ws.recv(), timeout=12.0)
                data = json.loads(raw)
                event = data.get("event", "")
                if event == "chat_reply":
                    res = data.get("message", "").lower()
                    break
                elif event == "sem_memoria":
                    res = ""
                    break
                # ignora voz_params e outros eventos intermediários
        except Exception:
            res = ""

        acertou = any(exp.lower() in res for exp in esperadas)

        # Sempre mostra a resposta completa da Selene
        if res:
            print(f"  Selene: \"{res}\"")
        else:
            print(f"  Selene: (sem resposta)")

        if acertou:
            acertos += 1
            print(f"  ACERTO  [esperado: {esperadas[0]}]")
            # 1. Dopamina + Q-table via recompensa_pendente
            await ws.send(json.dumps({"action": "reward", "value": 0.4}))
            # 2. Reforça as arestas do grafo que levaram à resposta certa
            await ws.send(json.dumps({"action": "feedback", "value": 0.6}))
            if context == "cor_visual":
                await ensinar_resposta_cor(ws, res)
            else:
                await ensinar_resposta(ws, res, context)
            # 3. Aguarda reward_ack para mostrar estado motivacional
            try:
                raw_ack = await asyncio.wait_for(ws.recv(), timeout=2.0)
                ack = json.loads(raw_ack)
                if ack.get("event") == "reward_ack":
                    dopa = ack.get("dopamine", 0)
                    sero = ack.get("serotonin", 0)
                    barra = int(min(dopa / 2.0, 1.0) * 10)
                    print(
                        f"  dopamina: {'█' * barra}{'░' * (10 - barra)} {dopa:.2f}"
                        f"  serotonina: {sero:.2f}"
                    )
            except Exception:
                pass
        else:
            print(f"  ERRO    [esperado: {esperadas[0]}]")
            # 1. Dopamina cai + noradrenalina sobe (estado de alerta/frustração)
            await ws.send(json.dumps({"action": "punish", "value": 0.15}))
            # 2. Penaliza arestas do grafo que levaram à resposta errada
            await ws.send(json.dumps({"action": "feedback", "value": -0.3}))
            # 3. Aguarda punish_ack para mostrar estado motivacional
            try:
                raw_ack = await asyncio.wait_for(ws.recv(), timeout=2.0)
                ack = json.loads(raw_ack)
                if ack.get("event") == "punish_ack":
                    dopa = ack.get("dopamine", 0)
                    nor  = ack.get("noradrenaline", 0)
                    barra_d = int(min(dopa / 2.0, 1.0) * 10)
                    barra_n = int(min(nor  / 2.0, 1.0) * 5)
                    print(
                        f"  dopamina: {'█' * barra_d}{'░' * (10 - barra_d)} {dopa:.2f}"
                        f"  noradrenalina: {'▲' * barra_n} {nor:.2f}  <- aprendendo"
                    )
            except Exception:
                pass
            # 4. Ensina a resposta correta imediatamente após o erro
            for exp in esperadas[:2]:
                if context == "cor_visual":
                    await ensinar_resposta_cor(ws, exp)
                else:
                    await ensinar_resposta(ws, exp, context)

        await asyncio.sleep(INTERVALO_TURNO)

    taxa = acertos / total if total > 0 else 0.0
    print(f"\n  Acertos: {acertos}/{total} ({taxa:.0%})")
    return taxa


# ── Placar ─────────────────────────────────────────────────────────────────
def imprimir_placar(resultados):
    print(f"\n{'=' * 62}")
    print("  PLACAR FINAL")
    print(f"{'=' * 62}")
    total_pts = 0
    total_max = 0
    for nome, taxa, aprovado in resultados:
        status = "OK" if aprovado else "XX"
        pts = round(taxa * 100)
        total_pts += pts
        total_max += 100
        print(f"  [{status}]  {nome:<35} {pts:3d}/100")
    media = total_pts / total_max * 100 if total_max > 0 else 0
    print(f"{'-' * 62}")
    print(f"  MEDIA GERAL: {media:.1f}%")
    print(f"{'=' * 62}\n")


# ── Main ───────────────────────────────────────────────────────────────────
async def main():
    print("JOGO SELENE -- Avaliacao Cognitiva")
    print(f"   Meta de aprovacao: {META_ACERTO:.0%} por fase\n")

    while True:
        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=None,
                open_timeout=15,
                close_timeout=10,
            ) as ws:
                resultados = []
                f_idx = 0

                while f_idx < len(FASES):
                    fase = FASES[f_idx]
                    taxa = await jogar_fase(ws, fase)
                    aprovado = taxa >= META_ACERTO
                    resultados.append((fase["nome"], taxa, aprovado))

                    if aprovado:
                        print("  META ATINGIDA -- exportando...")
                        await ws.send(json.dumps(
                            {"action": "export_linguagem"}
                        ))
                        await asyncio.sleep(0.5)
                        f_idx += 1
                    else:
                        print(
                            f"  Abaixo da meta."
                            f" Revisando em {PAUSA_REPETICAO}s..."
                        )
                        await asyncio.sleep(PAUSA_REPETICAO)

                await ws.send(json.dumps(
                    {"action": "reward", "value": 1.0}
                ))
                await asyncio.sleep(0.5)
                await ws.send(json.dumps(
                    {"action": "export_linguagem"}
                ))

                imprimir_placar(resultados)
                print("Treinamento completo!")
                break

        except (
            OSError,
            websockets.exceptions.WebSocketException,
        ) as e:
            print(f"Sem conexao: {e}. Tentando em 5s...")
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\n  Interrompido.")
            break
        except Exception as e:
            print(f"Erro: {e}. Tentando em 5s...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
