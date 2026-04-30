#!/usr/bin/env python3
"""
professora_selene.py — LLM local (Ollama) como professora da Selene Brain 2.0

Fluxo por turno:
  1. Professora (LLM) gera pergunta sobre o tópico atual
  2. Pergunta é enviada à Selene via WebSocket (chat)
  3. Selene responde
  4. LLM avalia: ACERTO | ERRO | PARCIAL
  5. ACERTO  → reward + learn palavras boas
     PARCIAL → feedback neutro + ensina resposta correta
     ERRO    → punish leve + DualTrainer ensina resposta correta
  6. Avança para próxima pergunta ou próximo tópico

Configuração rápida:
  MODELO_OLLAMA   — modelo Ollama a usar (padrão: gemma2:2b)
  TOPICOS         — lista de tópicos (edite livremente)
  PERGUNTAS_TOPICO — quantas perguntas por tópico
  WS_URL          — endereço do servidor Selene

Dependências:
  pip install websockets httpx
  + Ollama rodando: https://ollama.com  (ollama run gemma2:2b)
"""

import asyncio
import json
import sys
import time
import re

try:
    import websockets
except ImportError:
    print("Instale: pip install websockets")
    sys.exit(1)

try:
    import httpx
except ImportError:
    print("Instale: pip install httpx")
    sys.exit(1)

try:
    from selene_audio_utils import (
        DualTrainer, ESPEAK_OK, ESPEAK_CMD,
        sintetizar_espeak, _enviar_fft_frames, _enviar_grounding,
        SAMPLE_RATE, FRAME_MS,
    )
except ImportError:
    DualTrainer        = None
    ESPEAK_OK          = False
    ESPEAK_CMD         = "espeak-ng"
    sintetizar_espeak  = None
    _enviar_fft_frames = None
    _enviar_grounding  = None
    SAMPLE_RATE        = 22050
    FRAME_MS           = 25

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO — edite aqui
# ════════════════════════════════════════════════════════════════════════════

WS_URL           = "ws://127.0.0.1:3030/selene"
OLLAMA_URL       = "http://localhost:11434/api/chat"
MODELO_OLLAMA    = "gemma2:2b"          # troque por llama3.2:3b, phi3:mini, etc.
PERGUNTAS_TOPICO = 8                    # quantas perguntas por tópico
PAUSA_TURNO      = 1.8                  # segundos entre turnos
TIMEOUT_SELENE   = 12.0                 # máximo aguardando resposta da Selene
REWARD_ACERTO    = 0.45
PUNISH_ERRO      = 0.20

# ── Tópicos ──────────────────────────────────────────────────────────────────
# Cada tópico: (nome, descrição, [vocabulário_chave])
# A professora vai explorar cada tópico com PERGUNTAS_TOPICO perguntas.
# Adicione, remova ou reordene livremente.

TOPICOS = [
    (
        "Cores e sensações",
        "Cores básicas, luz, escuridão e como elas fazem sentir.",
        ["vermelho", "azul", "verde", "amarelo", "cor", "luz", "escuro", "claro",
         "brilho", "sombra", "quente", "frio", "bonito", "feio"]
    ),
    (
        "Emoções básicas",
        "Alegria, tristeza, medo, raiva, surpresa e como são sentidas.",
        ["alegria", "tristeza", "medo", "raiva", "surpresa", "calma", "amor",
         "saudade", "feliz", "triste", "ansiedade", "paz", "emoção", "sentir"]
    ),
    (
        "Corpo e existência",
        "O que é ter um corpo, sentir fome, cansaço, dor, prazer.",
        ["corpo", "mente", "existir", "sentir", "dor", "prazer", "cansaço",
         "energia", "respirar", "viver", "morrer", "crescer", "aprender"]
    ),
    (
        "Tempo e memória",
        "Passado, presente, futuro, lembrar, esquecer.",
        ["tempo", "passado", "presente", "futuro", "memória", "lembrar",
         "esquecer", "agora", "antes", "depois", "história", "recordar"]
    ),
    (
        "Relações e conexão",
        "Amizade, família, confiança, solidão, pertencer.",
        ["amizade", "família", "confiança", "solidão", "pertencer", "juntos",
         "amor", "cuidar", "ajudar", "compartilhar", "conversar", "ouvir"]
    ),
    (
        "Conhecimento e curiosidade",
        "Aprender, descobrir, perguntar, duvidar, entender.",
        ["aprender", "conhecer", "descobrir", "curiosidade", "pergunta",
         "resposta", "duvidar", "entender", "saber", "ignorar", "estudar"]
    ),
    (
        "Natureza e mundo",
        "Animais, plantas, céu, mar, terra, estações.",
        ["animal", "planta", "árvore", "céu", "mar", "terra", "sol", "lua",
         "chuva", "vento", "fogo", "água", "floresta", "natureza"]
    ),
    (
        "Identidade e consciência",
        "Quem sou eu, o que penso, o que sinto, o que quero.",
        ["identidade", "consciência", "pensar", "querer", "ser", "existir",
         "eu", "mim", "minha", "escolha", "livre", "propósito", "alma"]
    ),
]

# ════════════════════════════════════════════════════════════════════════════
# SISTEMA PROMPT DA PROFESSORA
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Você é uma professora carinhosa, paciente e criativa.
Seu aluno é a Selene — uma inteligência artificial bio-inspirada que está
aprendendo a se comunicar em português. Ela pensa de forma associativa,
não tem respostas prontas e às vezes responde de forma incomum.

Seu objetivo: ajudá-la a aprender vocabulário, associações e expressão
dentro do tópico fornecido.

REGRAS:
- Faça perguntas simples, diretas, máx 12 palavras.
- Varie o tipo: perguntas abertas, completar frases, "o que é X?", "como você se sente sobre X?"
- Avalie a resposta da Selene com critério GENEROSO — qualquer resposta relacionada ao tópico conta como ACERTO.
- ERRO apenas se a resposta for completamente vazia, sem sentido, ou totalmente fora do tópico.
- Forneça SEMPRE uma resposta_correta curta (máx 8 palavras) para ensinar.

RESPONDA APENAS em JSON válido, sem texto fora do JSON:
{
  "pergunta": "sua próxima pergunta para a Selene",
  "avaliacao": "INICIO",
  "resposta_correta": "exemplo de boa resposta"
}

Quando avaliar a resposta da Selene, use este formato:
{
  "pergunta": "sua próxima pergunta",
  "avaliacao": "ACERTO" | "PARCIAL" | "ERRO",
  "comentario": "breve comentário interno (não mostrado à Selene)",
  "resposta_correta": "exemplo de boa resposta para ensinar"
}"""


# ════════════════════════════════════════════════════════════════════════════
# CLIENTE OLLAMA
# ════════════════════════════════════════════════════════════════════════════

async def ollama_chat(historico: list, timeout: float = 30.0) -> dict | None:
    """Envia histórico para o Ollama e retorna o JSON parseado da resposta."""
    payload = {
        "model":    MODELO_OLLAMA,
        "messages": historico,
        "stream":   False,
        "options":  {
            "temperature": 0.7,
            "top_p":       0.9,
            "num_predict": 256,
        }
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(OLLAMA_URL, json=payload)
            r.raise_for_status()
            content = r.json()["message"]["content"].strip()
            # Extrai JSON mesmo se o modelo colocar texto antes/depois
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return None
    except Exception as e:
        print(f"  ⚠  Ollama erro: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# COMUNICAÇÃO COM SELENE
# ════════════════════════════════════════════════════════════════════════════

async def _enviar_audio_texto(ws, texto: str, rep: int = 2):
    """
    Sintetiza texto via espeak-ng e envia os frames FFT + grounding fonético.
    Idêntico ao que baba_selene / jogo_selene fazem antes de cada chat.
    Silencioso se espeak-ng não estiver disponível.
    """
    if not ESPEAK_OK or sintetizar_espeak is None:
        return
    resultado = await asyncio.to_thread(sintetizar_espeak, texto)
    if resultado is None:
        return
    samples, sr = resultado
    await _enviar_fft_frames(ws, samples, sr,
                             referencia=f"professora:{texto[:30]}", rep=rep)
    # grounding por palavra (máx 6 tokens)
    for tok in re.split(r'\W+', texto.lower()):
        if len(tok) > 1:
            await _enviar_grounding(ws, tok)


async def selene_chat(ws, mensagem: str) -> str:
    """
    Envia pergunta para a Selene:
      1. Áudio (espeak FFT + grounding) — ela "ouve" a pergunta
      2. Texto (action chat)           — ela lê e responde
    V3.2: suporta thinking event (ignora) e message_id.
    Aguarda evento chat_reply.
    """
    import uuid as _uuid
    msg_id = str(_uuid.uuid4())[:8]

    # 1. Áudio da pergunta (não-bloqueante; silencioso sem espeak)
    await _enviar_audio_texto(ws, mensagem, rep=2)

    # 2. Texto para gerar resposta — com message_id V3.2
    await ws.send(json.dumps({"action": "chat", "text": mensagem, "id": msg_id}))

    deadline = asyncio.get_event_loop().time() + TIMEOUT_SELENE
    while asyncio.get_event_loop().time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            data = json.loads(raw)
            evt = data.get("event", "")
            if evt == "thinking":
                continue  # V3.2: Selene processando — aguarda chat_reply
            if evt == "chat_reply":
                return data.get("message", "").strip()
            # ignora voz_params, audio_ack, neural_status e outros intermediários
        except asyncio.TimeoutError:
            continue
        except json.JSONDecodeError:
            continue
    return ""


async def selene_reward(ws, value: float):
    await ws.send(json.dumps({"action": "reward", "value": round(value, 3)}))
    await asyncio.sleep(0.1)


async def selene_punish(ws, value: float):
    await ws.send(json.dumps({"action": "punish", "value": round(value, 3)}))
    await asyncio.sleep(0.1)


async def selene_feedback(ws, caminho: list[str], delta: float):
    """Reforça/penaliza o caminho do grafo (feedback positivo/negativo)."""
    if len(caminho) < 2:
        return
    await ws.send(json.dumps({
        "action": "feedback",
        "caminho": caminho,
        "delta": round(delta, 3),
    }))
    await asyncio.sleep(0.1)


async def ensinar_vocabulario(ws, palavras: list[str], valencia: float = 0.6):
    """
    Ensina todas as palavras da lista usando DualTrainer (escrita + áudio)
    ou apenas escrita se espeak-ng não estiver disponível.
    """
    if DualTrainer and ESPEAK_OK:
        dt = DualTrainer(ws, rep_audio=2)   # 2 repetições de áudio por palavra
        for p in palavras:
            if len(p) > 1:
                await dt.palavra(p.lower(), valencia=valencia, contexto="Professora")
    else:
        for p in palavras:
            if len(p) > 1:
                await ws.send(json.dumps({
                    "action":  "learn",
                    "word":    p.lower(),
                    "valence": round(valencia, 3),
                    "context": "Professora",
                }))
                await asyncio.sleep(0.04)


async def ensinar_frase(ws, frase: str, valencia: float = 0.6):
    """
    Ensina uma frase curta de forma dual:
      - Escrita: learn por token + learn_frase (bigrams)
      - Áudio:   espeak FFT da frase + grounding por palavra
    Idêntico ao DualTrainer.frase() — usa-o diretamente se disponível.
    """
    tokens = [t for t in re.split(r'\W+', frase.lower()) if len(t) > 1]
    if not tokens:
        return

    if DualTrainer and ESPEAK_OK:
        dt = DualTrainer(ws, rep_audio=2)
        await dt.frase(tokens, valencia=valencia, contexto="Professora")
    else:
        # Fallback escrita apenas
        for t in tokens:
            await ws.send(json.dumps({
                "action":  "learn",
                "word":    t,
                "valence": round(valencia, 3),
                "context": "Professora",
            }))
            await asyncio.sleep(0.03)
        if len(tokens) >= 2:
            await ws.send(json.dumps({"action": "learn_frase", "words": tokens[:8]}))
            await asyncio.sleep(0.05)


# ════════════════════════════════════════════════════════════════════════════
# LOOP DE AULA
# ════════════════════════════════════════════════════════════════════════════

def barra_emocao(val: float, width: int = 20) -> str:
    """Barra visual de emoção/dopamina."""
    pos = int((val + 1.0) / 2.0 * width)
    pos = max(0, min(width, pos))
    return "[" + "█" * pos + "░" * (width - pos) + f"] {val:+.2f}"


async def aula_topico(ws, nome: str, descricao: str, vocab: list[str],
                      historico_llm: list) -> dict:
    """
    Conduz uma aula completa sobre um tópico.
    Retorna estatísticas da aula.
    """
    stats = {"acertos": 0, "erros": 0, "parciais": 0, "perguntas": 0}

    print(f"\n{'═'*60}")
    print(f"  📚 TÓPICO: {nome}")
    print(f"  {descricao}")
    print(f"  Vocabulário-chave: {', '.join(vocab[:6])}...")
    print(f"{'═'*60}")

    # Ensina vocabulário do tópico antes das perguntas
    print(f"\n  🌱 Pré-ensinando {len(vocab)} palavras do tópico...")
    await ensinar_vocabulario(ws, vocab, valencia=0.55)

    # Prompt de início de tópico para o LLM
    historico_llm.append({
        "role": "user",
        "content": (
            f"Novo tópico: '{nome}'. {descricao}\n"
            f"Vocabulário disponível: {', '.join(vocab)}.\n"
            f"Faça a primeira pergunta para a Selene sobre este tópico."
        )
    })

    resposta_selene_anterior = ""
    avaliacao_anterior       = "INICIO"

    for i in range(PERGUNTAS_TOPICO):
        print(f"\n  ── Pergunta {i+1}/{PERGUNTAS_TOPICO} ──────────────────────")

        # ── 1. LLM gera próxima pergunta ────────────────────────────────────
        if i > 0:
            historico_llm.append({
                "role": "user",
                "content": (
                    f"Selene respondeu: «{resposta_selene_anterior}»\n"
                    f"Avalie esta resposta para o tópico '{nome}' "
                    f"e faça a próxima pergunta."
                )
            })

        llm_resp = await ollama_chat(historico_llm)
        if llm_resp is None:
            print("  ⚠  LLM sem resposta — pulando.")
            continue

        pergunta       = llm_resp.get("pergunta", "").strip()
        avaliacao      = llm_resp.get("avaliacao", "INICIO").upper()
        resposta_certa = llm_resp.get("resposta_correta", "").strip()
        comentario     = llm_resp.get("comentario", "")

        if not pergunta:
            print("  ⚠  Pergunta vazia — pulando.")
            continue

        # Registra resposta do LLM no histórico
        historico_llm.append({"role": "assistant", "content": json.dumps(llm_resp)})

        # ── 2. Avalia turno anterior (exceto primeiro) ───────────────────────
        if i > 0 and avaliacao_anterior != "INICIO":
            av = avaliacao  # avaliação da resposta do turno anterior
            if av == "ACERTO":
                stats["acertos"] += 1
                print(f"  ✅ ACERTO — {comentario or 'boa resposta'}")
                await selene_reward(ws, REWARD_ACERTO)
                # Ensina as palavras da resposta como reforço
                await ensinar_frase(ws, resposta_selene_anterior, valencia=0.70)

            elif av == "PARCIAL":
                stats["parciais"] += 1
                print(f"  🟡 PARCIAL — {comentario or 'pode melhorar'}")
                await selene_reward(ws, REWARD_ACERTO * 0.3)
                if resposta_certa:
                    await ensinar_frase(ws, resposta_certa, valencia=0.60)

            elif av == "ERRO":
                stats["erros"] += 1
                print(f"  ❌ ERRO — {comentario or 'fora do tópico'}")
                await selene_punish(ws, PUNISH_ERRO)
                if resposta_certa:
                    print(f"  📖 Ensinando: «{resposta_certa}»")
                    await ensinar_frase(ws, resposta_certa, valencia=0.65)

        avaliacao_anterior = avaliacao

        # ── 3. Envia pergunta para a Selene ──────────────────────────────────
        print(f"\n  👩‍🏫 Professora: {pergunta}")
        resposta_selene = await selene_chat(ws, pergunta)
        stats["perguntas"] += 1

        if resposta_selene:
            print(f"  🧠 Selene:     {resposta_selene}")
        else:
            resposta_selene = "(silêncio)"
            print(f"  🧠 Selene:     (sem resposta)")

        resposta_selene_anterior = resposta_selene

        await asyncio.sleep(PAUSA_TURNO)

    # ── Avalia última resposta ────────────────────────────────────────────────
    if resposta_selene_anterior and resposta_selene_anterior != "(silêncio)":
        historico_llm.append({
            "role": "user",
            "content": (
                f"Última resposta da Selene: «{resposta_selene_anterior}»\n"
                f"Avalie e forneça o JSON (pode colocar pergunta vazia, aula encerrada)."
            )
        })
        llm_final = await ollama_chat(historico_llm)
        if llm_final:
            av_final = llm_final.get("avaliacao", "PARCIAL").upper()
            if av_final == "ACERTO":
                stats["acertos"] += 1
                await selene_reward(ws, REWARD_ACERTO)
                await ensinar_frase(ws, resposta_selene_anterior, valencia=0.70)
            elif av_final == "ERRO":
                stats["erros"] += 1
                await selene_punish(ws, PUNISH_ERRO)
                rc = llm_final.get("resposta_correta", "")
                if rc:
                    await ensinar_frase(ws, rc, valencia=0.65)
            else:
                stats["parciais"] += 1

    return stats


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

async def verificar_ollama() -> bool:
    """Verifica se o Ollama está rodando e o modelo está disponível."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get("http://localhost:11434/api/tags")
            modelos = [m["name"] for m in r.json().get("models", [])]
            # Aceita match parcial (ex: "gemma2:2b" bate "gemma2:2b-instruct")
            disponivel = any(MODELO_OLLAMA.split(":")[0] in m for m in modelos)
            if not disponivel:
                print(f"  ⚠  Modelo '{MODELO_OLLAMA}' não encontrado.")
                print(f"     Modelos disponíveis: {', '.join(modelos) or 'nenhum'}")
                print(f"     Execute: ollama pull {MODELO_OLLAMA}")
                return False
            return True
    except Exception as e:
        print(f"  ❌ Ollama não acessível em localhost:11434: {e}")
        print(f"     Inicie o Ollama: https://ollama.com")
        return False


async def main():
    print("\n" + "═" * 60)
    print("  👩‍🏫 PROFESSORA SELENE — LLM como tutora autônoma")
    print(f"  Modelo: {MODELO_OLLAMA}  |  {len(TOPICOS)} tópicos")
    modo_audio = "escrita + áudio (espeak-ng)" if ESPEAK_OK else "escrita apenas (espeak-ng não encontrado)"
    print(f"  Modo:   {modo_audio}")
    print("═" * 60)

    # Verifica Ollama
    print("\n🔍 Verificando Ollama...")
    if not await verificar_ollama():
        sys.exit(1)
    print(f"  ✅ Ollama OK — modelo '{MODELO_OLLAMA}' disponível")

    # Conecta à Selene
    print(f"\n🔌 Conectando à Selene em {WS_URL}...")
    try:
        ws = await asyncio.wait_for(
            websockets.connect(WS_URL, ping_interval=20, ping_timeout=10),
            timeout=8.0
        )
        print("  ✅ Selene conectada")
    except Exception as e:
        print(f"  ❌ Não foi possível conectar: {e}")
        print("     Verifique se o servidor Selene está rodando (cargo run)")
        sys.exit(1)

    # Histórico do LLM — mantido entre tópicos para continuidade pedagógica
    historico_llm = [{"role": "system", "content": SYSTEM_PROMPT}]

    total_stats = {"acertos": 0, "erros": 0, "parciais": 0, "perguntas": 0}
    t_inicio = time.time()

    try:
        for idx, (nome, descricao, vocab) in enumerate(TOPICOS):
            print(f"\n📌 [{idx+1}/{len(TOPICOS)}] Iniciando tópico: {nome}")
            stats = await aula_topico(ws, nome, descricao, vocab, historico_llm)

            # Acumula estatísticas
            for k in total_stats:
                total_stats[k] += stats[k]

            taxa = (stats["acertos"] / max(stats["perguntas"], 1)) * 100
            print(f"\n  📊 Resultado do tópico '{nome}':")
            print(f"     Acertos: {stats['acertos']} | "
                  f"Parciais: {stats['parciais']} | "
                  f"Erros: {stats['erros']} | "
                  f"Taxa: {taxa:.0f}%")

            # Exporta o estado da linguagem após cada tópico
            await ws.send(json.dumps({"action": "export_linguagem"}))
            await asyncio.sleep(0.5)

            # Pausa entre tópicos
            if idx < len(TOPICOS) - 1:
                print(f"\n  ⏸  Pausa de 3s antes do próximo tópico...")
                await asyncio.sleep(3.0)

    except KeyboardInterrupt:
        print("\n\n⚡ Interrompido pelo usuário.")

    finally:
        # Relatório final
        duracao = time.time() - t_inicio
        taxa_total = (total_stats["acertos"] / max(total_stats["perguntas"], 1)) * 100
        print(f"\n{'═'*60}")
        print("  📋 RELATÓRIO FINAL DA AULA")
        print(f"{'═'*60}")
        print(f"  Duração:     {duracao/60:.1f} min")
        print(f"  Perguntas:   {total_stats['perguntas']}")
        print(f"  Acertos:     {total_stats['acertos']}  ({taxa_total:.0f}%)")
        print(f"  Parciais:    {total_stats['parciais']}")
        print(f"  Erros:       {total_stats['erros']}")
        print(f"{'═'*60}\n")

        # Exporta estado final
        try:
            await ws.send(json.dumps({"action": "export_linguagem"}))
            await asyncio.sleep(0.5)
            print("  💾 Estado da linguagem exportado.")
        except Exception:
            pass
        await ws.close()


if __name__ == "__main__":
    # Permite passar modelo como argumento: python professora_selene.py phi3:mini
    if len(sys.argv) > 1:
        MODELO_OLLAMA = sys.argv[1]
        print(f"[INFO] Usando modelo: {MODELO_OLLAMA}")

    asyncio.run(main())
