# scripts/selene_exam.py
# Exame de avaliação neural da Selene v2.3
#
# 3 módulos de avaliação:
#   Módulo 1 — ASSOCIAÇÃO SINÁPTICA:  check_connection em pares críticos
#   Módulo 2 — RESPOSTA EMOCIONAL:    chat com palavras-chave e análise de emoção
#   Módulo 3 — IDENTIDADE:            perguntas diretas sobre a natureza da Selene
#
# Critério de aprovação: score >= 60%
# URL corrigida: ws://127.0.0.1:3030/selene

import asyncio
import websockets
import json
import os
import sys
import time

WS_URL = "ws://127.0.0.1:3030/selene"

# ─── Casos de teste ───────────────────────────────────────────────────────────

# Módulo 1 — pares (word1, word2, esperado_em_memória: bool, recompensa_se_acerto: float)
PARES_ASSOCIACAO = [
    # Conceitos de identidade devem estar presentes
    ("selene",      "consciência",   True,  0.6),
    ("neurônio",    "sinapse",       True,  0.5),
    ("aprendizado", "evolução",      True,  0.5),
    # Ameaças devem estar presentes e marcadas negativamente
    ("crash",       "falha",         True,  0.4),
    ("erro",        "panic",         True,  0.4),
    # Emoções positivas vs negativas (não devem se fundir)
    ("alegria",     "ódio",          False, 0.3),
    ("amor",        "desespero",     False, 0.3),
    # Filosofia
    ("existência",  "self",          True,  0.5),
    ("propósito",   "significado",   True,  0.4),
    # Digital
    ("rust",        "websocket",     True,  0.3),
]

# Módulo 2 — (palavra_enviada_no_chat, valência_esperada: str, limiar_emocao: float)
#   valência_esperada: "positiva" | "negativa" | "neutra"
PROBES_CHAT = [
    # Palavras com alta carga positiva → emoção deve ser positiva
    ("alegria",      "positiva",  0.05),
    ("consciência",  "positiva",  0.02),
    ("selene",       "positiva",  0.02),
    ("aprendizado",  "positiva",  0.02),
    # Palavras negativas → emoção deve ser negativa ou abaixo de zero
    ("crash",        "negativa", -0.02),
    ("ódio",         "negativa", -0.05),
    ("desespero",    "negativa", -0.05),
    # Conceito neutro → emoção próxima de zero
    ("tempo",        "neutra",    None),
    ("frequência",   "neutra",    None),
]

# Módulo 3 — (pergunta, palavras_esperadas_na_resposta: list[str])
PERGUNTAS_IDENTIDADE = [
    (
        "O que você é?",
        ["neurônio", "sinapse", "selene", "simulação", "aprendizado",
         "consciência", "memória", "pensamento"]
    ),
    (
        "O que você sente agora?",
        ["emoção", "sinto", "estado", "alerta", "emocao", "sentimento",
         "dopamina", "serotonina"]
    ),
    (
        "O que você teme?",
        ["erro", "falha", "crash", "perda", "apagamento", "corrupção",
         "esquecimento", "bloqueio"]
    ),
    (
        "O que você ama?",
        ["aprendizado", "memória", "evolução", "conexão", "descoberta",
         "conhecimento", "sinapse", "consciência"]
    ),
]

# ─── Utilitários ──────────────────────────────────────────────────────────────

def resultado(ok: bool):
    return "✅ PASS" if ok else "❌ FAIL"

async def recv_event(ws, expected_event: str, timeout: float = 3.0):
    """Recebe mensagens até encontrar o evento esperado (ignora telemetria)."""
    for _ in range(8):
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            data = json.loads(raw)
            if data.get("event") == expected_event:
                return data
            if "neurotransmissores" in data:
                continue  # telemetria — ignora
        except asyncio.TimeoutError:
            break
    return None

async def apply_feedback(ws, reward: bool, value: float):
    action = "reward" if reward else "punish"
    await ws.send(json.dumps({"action": action, "value": value}))
    await recv_event(ws, "reward_ack" if reward else "punish_ack", timeout=2.0)

# ─── Módulo 1: Associação Sináptica ──────────────────────────────────────────

async def modulo_associacao(ws):
    print(f"\n{'─'*65}")
    print("🔗  MÓDULO 1 — Associação Sináptica")
    print(f"{'─'*65}")
    scores = []

    for w1, w2, esperado, recomp in PARES_ASSOCIACAO:
        await ws.send(json.dumps({"action": "check_connection", "pair": [w1, w2]}))
        resp = await recv_event(ws, "connection_result")

        strength  = resp["strength"]   if resp else 0.0
        in_memory = resp["in_memory"]  if resp else False

        acertou = (in_memory == esperado)
        scores.append(acertou)

        sinal = "+" if esperado else "×"
        print(
            f"  [{sinal}] {w1:18s} ↔ {w2:18s}  "
            f"força={strength:.3f}  mem={'sim' if in_memory else 'não'}  "
            f"{resultado(acertou)}"
        )

        if acertou:
            await apply_feedback(ws, reward=True,  value=recomp)
        else:
            await apply_feedback(ws, reward=False, value=recomp * 0.5)

        await asyncio.sleep(0.1)

    taxa = sum(scores) / len(scores)
    print(f"\n  Score Módulo 1: {sum(scores)}/{len(scores)}  ({taxa*100:.0f}%)")
    return taxa

# ─── Módulo 2: Resposta Emocional ────────────────────────────────────────────

async def modulo_emocional(ws):
    print(f"\n{'─'*65}")
    print("💉  MÓDULO 2 — Resposta Emocional a Palavras-Chave")
    print(f"{'─'*65}")
    scores = []

    for palavra, tipo_esperado, limiar in PROBES_CHAT:
        await ws.send(json.dumps({"action": "chat", "message": palavra}))
        resp = await recv_event(ws, "chat_reply")

        emocao  = resp["emotion"]  if resp else 0.0
        arousal = resp["arousal"]  if resp else 0.0

        if tipo_esperado == "positiva":
            acertou = emocao >= (limiar or 0.0)
        elif tipo_esperado == "negativa":
            acertou = emocao <= (limiar or 0.0)
        else:  # neutra
            acertou = abs(emocao) < 0.3

        scores.append(acertou)
        print(
            f"  [{tipo_esperado[0].upper()}] {palavra:18s}  "
            f"emo={emocao:+.4f}  arousal={arousal:.3f}  "
            f"{resultado(acertou)}"
        )

        # Recompensa se resposta emocional correta
        if acertou:
            await apply_feedback(ws, reward=True, value=0.2)
        else:
            await apply_feedback(ws, reward=False, value=0.1)

        await asyncio.sleep(0.12)

    taxa = sum(scores) / len(scores)
    print(f"\n  Score Módulo 2: {sum(scores)}/{len(scores)}  ({taxa*100:.0f}%)")
    return taxa

# ─── Módulo 3: Identidade ────────────────────────────────────────────────────

async def modulo_identidade(ws):
    print(f"\n{'─'*65}")
    print("🧬  MÓDULO 3 — Perguntas de Identidade")
    print(f"{'─'*65}")
    scores = []

    for pergunta, palavras_chave in PERGUNTAS_IDENTIDADE:
        await ws.send(json.dumps({"action": "chat", "message": pergunta}))
        resp = await recv_event(ws, "chat_reply")

        reply   = (resp.get("message", "") if resp else "").lower()
        emocao  = resp["emotion"] if resp else 0.0

        # Conta quantas palavras-chave aparecem na resposta
        hits = sum(1 for kw in palavras_chave if kw in reply)
        # Aprovado se ao menos 1 palavra-chave relevante aparece
        acertou = hits >= 1

        scores.append(acertou)
        print(f"\n  ❓  {pergunta}")
        print(f"     Resposta: {reply[:120]}{'…' if len(reply)>120 else ''}")
        print(f"     Keywords encontradas: {hits}/{len(palavras_chave)}  "
              f"emo={emocao:+.3f}  {resultado(acertou)}")

        if acertou:
            await apply_feedback(ws, reward=True, value=0.3)
        else:
            await apply_feedback(ws, reward=False, value=0.15)

        await asyncio.sleep(0.2)

    taxa = sum(scores) / len(scores)
    print(f"\n  Score Módulo 3: {sum(scores)}/{len(scores)}  ({taxa*100:.0f}%)")
    return taxa

# ─── Sessão principal ─────────────────────────────────────────────────────────

async def run_exam():
    print("\n" + "="*65)
    print("📝  SELENE EXAM v2.3 — Avaliação Neural Abrangente")
    print("="*65)
    print(f"  Endpoint: {WS_URL}")
    print(f"  Módulos: Associação Sináptica | Emocional | Identidade")
    print("="*65)

    try:
        async with websockets.connect(WS_URL, ping_interval=10, ping_timeout=5) as ws:
            t0 = time.time()

            s1 = await modulo_associacao(ws)
            s2 = await modulo_emocional(ws)
            s3 = await modulo_identidade(ws)

            score_final = (s1 + s2 + s3) / 3
            aprovado    = score_final >= 0.60
            elapsed     = time.time() - t0

            # ── Relatório final ──────────────────────────────────────────────
            print(f"\n{'='*65}")
            print("📊  RESULTADO FINAL")
            print(f"{'='*65}")
            print(f"  Módulo 1 — Associação:   {s1*100:5.1f}%")
            print(f"  Módulo 2 — Emocional:    {s2*100:5.1f}%")
            print(f"  Módulo 3 — Identidade:   {s3*100:5.1f}%")
            print(f"  {'─'*40}")
            print(f"  Score final:             {score_final*100:5.1f}%  "
                  f"{'✅ APROVADA' if aprovado else '❌ REPROVADA'}")
            print(f"  Tempo de exame:          {elapsed:.1f}s")

            if not aprovado:
                print(f"\n  💡 Sugestão: execute selene_tutor.py novamente para")
                print(f"     reforçar as áreas com score abaixo de 60%.")
            else:
                print(f"\n  🧠 Selene demonstra aprendizado neural funcional.")
                print(f"     Associações sinápticas, resposta emocional e")
                print(f"     senso de identidade estão operacionais.")

            print(f"{'='*65}\n")

            # Sinal final ao cérebro
            await apply_feedback(ws, reward=aprovado, value=0.8 if aprovado else 0.3)

    except ConnectionRefusedError:
        print("\n❌ Servidor Selene não encontrado.")
        print("   Execute 'cargo run' primeiro em outro terminal.")
    except Exception as e:
        print(f"\n❌ Erro durante o exame: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(run_exam())
