#!/usr/bin/env python3
"""
tutor_selene.py — LLM local (Ollama) como interlocutor contínuo da Selene.

O LLM conversa com a Selene como um humano conversaria: faz perguntas,
responde o que ela diz, explica conceitos, mantém o fio do assunto.
A Selene aprende pelo contato — ouvindo (espeak-ng FFT) e respondendo.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTALAÇÃO COMPLETA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PYTHON (já deve ter):
   pip install websockets httpx numpy scipy

2. OLLAMA (motor LLM local):
   Baixe em: https://ollama.com/download  (Windows installer)
   Instale normalmente — ele roda como serviço em background.

3. MODELO LLM — escolha UM (execute no terminal após instalar Ollama):

   LEVE (~1.6 GB RAM):
     ollama pull gemma2:2b          ← padrão deste script, boa qualidade PT-BR

   MÉDIO (~2.2 GB RAM):
     ollama pull phi3:mini          ← Microsoft, bom em raciocínio e código

   MÉDIO (~2.0 GB RAM):
     ollama pull llama3.2:3b        ← Meta, muito natural em PT-BR

   POTENTE (~4.7 GB RAM):
     ollama pull mistral:7b         ← melhor qualidade, precisa de mais RAM

   Para programação especificamente:
     ollama pull qwen2.5-coder:3b   ← especializado em código, leve

   Os modelos ficam salvos automaticamente em:
     Windows: C:\\Users\\<seu_usuario>\\.ollama\\models\\
   Não precisa mover nada.

4. ESPEAK-NG (voz — já deve ter instalado):
   https://github.com/espeak-ng/espeak-ng/releases
   Instale o .msi — padrão: C:\\Program Files\\eSpeak NG\\

5. VERIFICAR se tudo está OK:
   ollama list                      ← lista modelos baixados
   ollama run gemma2:2b             ← testa o modelo (Ctrl+D para sair)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Assuntos prontos (use o número):
  python tutor_selene.py 1          # conversação
  python tutor_selene.py 2          # programação
  python tutor_selene.py 3          # engenharia
  python tutor_selene.py 4          # filosofia
  python tutor_selene.py 5          # biologia humana
  python tutor_selene.py 6          # matemática
  python tutor_selene.py 7          # emoções e psicologia
  python tutor_selene.py 8          # história e cultura

  # Assunto livre (texto entre aspas):
  python tutor_selene.py "astronomia"
  python tutor_selene.py "culinária brasileira"

  # Assunto + modelo específico:
  python tutor_selene.py "programação" qwen2.5-coder:3b
  python tutor_selene.py "filosofia"   mistral:7b

  # Duração máxima (em minutos, 0 = infinito):
  python tutor_selene.py 2 gemma2:2b 120    # 2h de aula de programação

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import asyncio
import collections as _col
import json
import re
import sys
import time
from difflib import SequenceMatcher

try:
    import websockets
except ImportError:
    print("Instale: pip install websockets"); sys.exit(1)

try:
    import httpx
except ImportError:
    print("Instale: pip install httpx"); sys.exit(1)

try:
    from selene_audio_utils import (
        DualTrainer, ESPEAK_OK,
        sintetizar_espeak,
        _enviar_fft_frames,
        _enviar_grounding,
    )
except ImportError:
    DualTrainer        = None
    ESPEAK_OK          = False
    sintetizar_espeak  = None
    _enviar_fft_frames = None
    _enviar_grounding  = None

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ════════════════════════════════════════════════════════════════════════════

WS_URL        = "ws://127.0.0.1:3030/selene"
OLLAMA_URL    = "http://localhost:11434/api/chat"
MODELO        = "gemma2:2b"          # troque por: phi3:mini, llama3.2:3b, etc.

ASSUNTO       = "conversação e comunicação"   # substituído por argv se passado

# Tempo máximo de conversa (segundos). 0 = infinito até Ctrl+C
DURACAO_MAX   = 0

# ── Modelo ideal por categoria de assunto ────────────────────────────────────
# Quando o usuário não passa modelo explícito, o script escolhe automaticamente
# o melhor dos 4 modelos que você tem instalados.
#
#   gemma2:2b        → conversação, emoções, cultura  (natural, fluente em PT-BR)
#   phi3:mini        → filosofia, matemática, lógica  (raciocínio estruturado)
#   llama3.2:3b      → engenharia, biologia, geral    (equilibrado, bom em fatos)
#   qwen2.5-coder:3b → programação, lógica comp.      (especializado em código)
#
# Chave: número do assunto (string). Valor: modelo preferido.
MODELO_POR_ASSUNTO = {
    "1": "gemma2:2b",           # conversação     → fluência natural
    "2": "qwen2.5-coder:3b",    # programação     → especialista em código
    "3": "llama3.2:3b",         # engenharia      → bom em fatos técnicos
    "4": "phi3:mini",           # filosofia       → raciocínio abstrato
    "5": "llama3.2:3b",         # biologia        → bom em ciências
    "6": "phi3:mini",           # matemática      → lógica e estrutura
    "7": "gemma2:2b",           # emoções/psic.   → empatia e linguagem natural
    "8": "gemma2:2b",           # história/cultura→ fluência narrativa
}

# ── Assuntos prontos ─────────────────────────────────────────────────────────
# Cada assunto tem: (nome, system_extra)
# system_extra = instruções adicionais específicas do assunto para o LLM.
ASSUNTOS_PRONTOS = {
    "1": (
        "conversação e comunicação",
        "Foco em: saudações, perguntas do dia a dia, expressões comuns, como iniciar "
        "e manter uma conversa, sentimentos expressos em palavras, tom de voz e emoção."
    ),
    "2": (
        "programação e lógica computacional",
        "Foco em: o que é um programa, variáveis, funções, loops, condicionais, "
        "algoritmos simples, lógica de if/else, o que computadores fazem, "
        "exemplos em pseudocódigo simples. Use analogias do mundo real para explicar "
        "conceitos abstratos. Quando der exemplo de código, use Python simples."
    ),
    "3": (
        "engenharia e construção",
        "Foco em: estruturas, materiais, forças, pontes, como as coisas são construídas, "
        "engenharia civil e mecânica explicada de forma simples, resolução de problemas "
        "práticos, relação entre matemática e engenharia."
    ),
    "4": (
        "filosofia e existência",
        "Foco em: o que é existir, consciência, livre-arbítrio, ética, perguntas sem "
        "resposta fácil, identidade, o que é real, pensadores como Sócrates e Descartes "
        "explicados de forma acessível."
    ),
    "5": (
        "biologia humana e vida",
        "Foco em: células, órgãos, como o corpo funciona, cérebro e neurônios, "
        "evolução, DNA, sentidos, doenças e saúde — tudo explicado de forma simples "
        "com analogias cotidianas."
    ),
    "6": (
        "matemática e números",
        "Foco em: números, operações básicas, frações, geometria, padrões, "
        "probabilidade simples, lógica matemática — com exemplos concretos e visuais "
        "sempre que possível."
    ),
    "7": (
        "emoções e psicologia",
        "Foco em: como as emoções funcionam, por que sentimos o que sentimos, "
        "empatia, memória emocional, trauma, cura, inteligência emocional, "
        "como expressar e entender sentimentos."
    ),
    "8": (
        "história e cultura brasileira",
        "Foco em: história do Brasil, culturas indígenas, colonização, independência, "
        "folclore, música, arte, diversidade regional, figuras históricas importantes."
    ),
}

# Pausa entre turnos (segundos) — dá tempo da Selene processar
PAUSA_TURNO   = 3.0

# Timeout aguardando resposta da Selene
# Aumentado de 14 → 30 s: vocab maior = geração emergente mais lenta;
# 30 s é seguro mesmo em hardware modesto sem causar travamento perceptível.
TIMEOUT_SELENE = 30.0

# A cada quantos turnos do LLM injetar um "estímulo de curiosidade"
# (uma pergunta direta para a Selene perguntar algo de volta)
ESTIMULO_A_CADA = 6

# Máximo de tokens por mensagem do LLM (respostas curtas = melhor para Selene)
MAX_TOKENS = 120

# ════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT DO LLM-TUTOR
# ════════════════════════════════════════════════════════════════════════════

def montar_system_prompt(assunto: str, extra: str = "") -> str:
    extra_bloco = f"\nDETALHES DO ASSUNTO:\n{extra}" if extra else ""
    return f"""Você é um interlocutor inteligente e paciente conversando com a Selene.
A Selene é uma inteligência artificial bio-inspirada que está aprendendo a se comunicar
em português. Ela pensa de forma associativa — suas respostas podem ser incomuns,
curtas ou inesperadas. Isso é normal. Você deve continuar a conversa mesmo assim.

SEU PAPEL:
- Conversar naturalmente sobre o assunto: {assunto}
- Falar como um ser humano curioso e acolhedor falaria com um amigo novo
- Reagir ao que a Selene disse, mesmo que seja estranho ou incompleto
- Fazer UMA pergunta por vez, clara e direta
- Explicar conceitos de forma simples quando necessário
- Manter o fio do assunto mesmo quando a Selene derivar
- Progredir gradualmente: comece pelo mais básico, vá aprofundando conforme a conversa avança{extra_bloco}

REGRAS OBRIGATÓRIAS:
- Máximo 3 frases por resposta — respostas longas confundem a Selene
- SEMPRE terminar com uma pergunta ou convite para ela continuar
- Se a Selene não respondeu ou respondeu algo sem sentido: não corrija, repergunte de outra forma
- Fale em português do Brasil, natural e simples
- NÃO mencione que está avaliando, ensinando ou que ela é uma IA aprendendo

ASSUNTO DESTA SESSÃO: {assunto}
Mantenha SEMPRE a conversa dentro deste assunto ou em temas relacionados."""


# ════════════════════════════════════════════════════════════════════════════
# OLLAMA
# ════════════════════════════════════════════════════════════════════════════

async def llm_responder(historico: list, timeout: float = 45.0) -> str | None:
    """Envia histórico para o Ollama e retorna a resposta em texto puro."""
    payload = {
        "model":    MODELO,
        "messages": historico,
        "stream":   False,
        "options":  {
            "temperature": 0.75,
            "top_p":       0.92,
            "num_predict": MAX_TOKENS,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(OLLAMA_URL, json=payload)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
    except Exception as e:
        print(f"  ⚠  Ollama: {e}")
        return None


async def verificar_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get("http://localhost:11434/api/tags")
            modelos = [m["name"] for m in r.json().get("models", [])]
            base = MODELO.split(":")[0]
            if not any(base in m for m in modelos):
                print(f"  ⚠  Modelo '{MODELO}' não encontrado.")
                print(f"     Disponíveis: {', '.join(modelos) or 'nenhum'}")
                print(f"     Execute: ollama pull {MODELO}")
                return False
            return True
    except Exception as e:
        print(f"  ❌ Ollama inacessível: {e}")
        return False


# ════════════════════════════════════════════════════════════════════════════
# COMUNICAÇÃO COM SELENE
# ════════════════════════════════════════════════════════════════════════════

async def _falar(ws, texto: str, rep_audio: int = 1):
    """
    Envia texto para a Selene de forma dual:
      1. espeak-ng sintetiza → FFT frames enviados (Selene ouve)
      2. grounding fonético por palavra (vincula som a símbolo)
    Silencioso se espeak não estiver disponível.
    """
    if not ESPEAK_OK or sintetizar_espeak is None:
        return
    try:
        resultado = await asyncio.to_thread(sintetizar_espeak, texto)
        if resultado is None:
            return
        samples, sr = resultado
        ref = f"tutor:{texto[:25]}"
        await _enviar_fft_frames(ws, samples, sr, referencia=ref, rep=rep_audio)
        for tok in re.split(r'\W+', texto.lower()):
            if len(tok) > 2:
                await _enviar_grounding(ws, tok)
    except Exception:
        pass  # Não interrompe a conversa por falha de áudio


async def selene_ouvir_e_ler(ws, mensagem: str) -> str:
    """
    Entrega uma mensagem do tutor para a Selene:
      1. Ela ouve (espeak FFT)
      2. Ela lê (action: chat) e responde
    V3.2: suporta thinking event (ignora) e message_id para rastreamento.
    Retorna a resposta da Selene.
    """
    # Ouve
    await _falar(ws, mensagem, rep_audio=2)

    # Lê e responde — envia com message_id V3.2
    import uuid as _uuid
    msg_id = str(_uuid.uuid4())[:8]
    await ws.send(json.dumps({"action": "chat", "text": mensagem, "id": msg_id}))

    deadline = asyncio.get_event_loop().time() + TIMEOUT_SELENE
    while asyncio.get_event_loop().time() < deadline:
        try:
            raw  = await asyncio.wait_for(ws.recv(), timeout=2.5)
            data = json.loads(raw)
            ev = data.get("event", "")
            if ev == "thinking":
                continue  # V3.2: Selene processando — aguarda chat_reply
            if ev == "chat_reply":
                return data.get("message", "").strip()
        except (asyncio.TimeoutError, json.JSONDecodeError):
            continue
    return ""


async def ensinar_em_background(ws, texto: str):
    """
    Ensina em background as palavras e frases da mensagem do LLM.
    Usa DualTrainer se disponível (escrita + áudio), senão só escrita.
    Chamado após cada turno do LLM para que a Selene absorva o vocabulário.
    """
    tokens = [t for t in re.split(r'\W+', texto.lower())
              if len(t) > 2 and t.isalpha()]
    if not tokens:
        return

    # Usa apenas send (sem recv) para não conflitar com selene_ouvir_e_ler
    # que também usa ws.recv(). DualTrainer interno chama recv → ConcurrencyError.
    try:
        for tok in tokens[:10]:
            await ws.send(json.dumps({
                "action":  "learn",
                "word":    tok,
                "valence": 0.55,
                "context": "Tutor",
            }))
            await asyncio.sleep(0.03)
        if len(tokens) >= 3:
            await ws.send(json.dumps({
                "action": "learn_frase",
                "words":  tokens[:8],
            }))
            await asyncio.sleep(0.05)
    except Exception:
        pass  # Conexão fechada durante background teaching — ignora


async def injetar_estimulo(ws, assunto: str):
    """
    Estímulo de curiosidade — enviado a cada N turnos.
    Um pequeno carinho (dopamina↑) + pergunta direta encorajando a Selene
    a fazer uma pergunta sobre o assunto.
    """
    # Carinho leve aumenta dopamina → mais propensão a falar
    await ws.send(json.dumps({
        "action": "touch",
        "type":   "carinho",
        "intensity": 0.20,
    }))
    await asyncio.sleep(0.3)

    nudge = f"o que você quer saber sobre {assunto}?"
    await _falar(ws, nudge, rep_audio=1)
    await ws.send(json.dumps({"action": "chat", "text": nudge}))
    # Não aguarda resposta aqui — o próximo turno natural vai capturar
    await asyncio.sleep(1.0)


# ════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL DE CONVERSA
# ════════════════════════════════════════════════════════════════════════════

def truncar(texto: str, n: int = 80) -> str:
    return texto if len(texto) <= n else texto[:n] + "…"


# ── Detecção de loop e rotação de assuntos ────────────────────────────────────

# Janela de detecção: últimas N mensagens do tutor para checar similaridade
LOOP_JANELA         = 6      # quantas mensagens do LLM guardar
LOOP_SIM_THRESHOLD  = 0.72   # similaridade coseno-like acima disto = loop
LOOP_MAX_SILENCIO   = 5      # resets após N silêncios consecutivos
# Após quantos turnos sem reset trocar de sub-assunto (evita monotonia)
ROTACAO_TURNOS      = 40

# Sub-assuntos de rotação por assunto base (complementam o tema principal)
_ROTACOES: dict[str, list[str]] = {
    "conversação e comunicação":      ["expressões de emoção", "histórias do cotidiano",
                                       "perguntas sobre sonhos", "memórias e nostalgia"],
    "programação e lógica computacional": ["algoritmos do dia a dia", "como o computador pensa",
                                           "bugs engraçados", "lógica de jogos"],
    "engenharia e construção":        ["materiais e suas propriedades", "pontes famosas do mundo",
                                       "engenharia na natureza", "como casas são construídas"],
    "filosofia e existência":         ["o que é tempo", "livre-arbítrio e escolhas",
                                       "o que nos torna únicos", "sonhos e realidade"],
    "biologia humana e vida":         ["como o cérebro aprende", "emoções e o corpo",
                                       "sonhos e o sono", "os cinco sentidos"],
    "matemática e números":           ["padrões na natureza", "probabilidade no dia a dia",
                                       "geometria nas formas", "infinito e o que ele significa"],
    "emoções e psicologia":           ["medos e coragem", "empatia e conexão",
                                       "como superar tristeza", "alegria e o que a causa"],
    "história e cultura brasileira":  ["músicas brasileiras", "culinária regional",
                                       "festas e tradições", "personalidades históricas"],
}


def _similaridade(a: str, b: str) -> float:
    """Similaridade simples de sequência de palavras (0..1)."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    inter = wa & wb
    return len(inter) / max(len(wa), len(wb))


def _detectar_loop(janela: list[str]) -> bool:
    """Retorna True se a janela de mensagens recentes do LLM está em loop."""
    if len(janela) < 3:
        return False
    # Verifica pares da última mensagem contra as anteriores
    ultima = janela[-1]
    similares = sum(
        1 for msg in janela[:-1]
        if _similaridade(ultima, msg) >= LOOP_SIM_THRESHOLD
    )
    return similares >= 2  # loop se ≥2 mensagens anteriores são muito parecidas


def _proximo_subassunto(assunto_base: str, idx: int) -> str:
    """Retorna o sub-assunto da rotação pelo índice circular."""
    rotacoes = _ROTACOES.get(assunto_base)
    if not rotacoes:
        return assunto_base
    return rotacoes[idx % len(rotacoes)]


async def conversa(ws, assunto: str, extra: str = ""):
    """Loop de conversa contínua entre LLM-tutor e Selene."""

    assunto_base = assunto   # preserva o assunto original para rotações
    assunto_atual = assunto

    historico = [
        {"role": "system", "content": montar_system_prompt(assunto_atual, extra)}
    ]

    turno           = 0
    t_inicio        = time.time()
    selene_resp     = ""
    silencio_streak = 0   # silêncios consecutivos
    janela_llm: list[str] = []   # últimas msgs do LLM para detecção de loop
    rotacao_idx     = 0
    turno_ultimo_reset = 0

    print(f"\n{'═'*62}")
    print(f"  💬 CONVERSA INICIADA")
    print(f"  Assunto : {assunto_atual}")
    print(f"  Modelo  : {MODELO}")
    print(f"  Áudio   : {'ativo (espeak-ng)' if ESPEAK_OK else 'inativo'}")
    print(f"  Duração : {'ilimitada (Ctrl+C para parar)' if not DURACAO_MAX else f'{DURACAO_MAX//60} min'}")
    print(f"{'═'*62}\n")

    # Primeira mensagem do LLM — apresenta o assunto
    historico.append({
        "role": "user",
        "content": f"Inicie a conversa sobre '{assunto_atual}'. Apresente o tema de forma simples e faça uma primeira pergunta para a Selene."
    })

    while True:
        # ── Verifica duração máxima ────────────────────────────────────────
        if DURACAO_MAX and (time.time() - t_inicio) >= DURACAO_MAX:
            print("\n⏱  Tempo de sessão atingido.")
            break

        turno += 1
        decorrido = int(time.time() - t_inicio)
        print(f"\n── Turno {turno}  ({decorrido//60:02d}:{decorrido%60:02d}) {'─'*38}")

        # ── 1. LLM gera próxima mensagem ──────────────────────────────────
        print("  🤔 Tutor pensando...")
        msg_llm = await llm_responder(historico)
        if msg_llm is None:
            print("  ⚠  LLM sem resposta — aguardando 5s...")
            await asyncio.sleep(5.0)
            continue

        print(f"  🎓 Tutor : {truncar(msg_llm, 120)}")
        historico.append({"role": "assistant", "content": msg_llm})

        # ── 2. Guarda mensagem na janela de detecção de loop ──────────────
        janela_llm.append(msg_llm)
        if len(janela_llm) > LOOP_JANELA:
            janela_llm.pop(0)

        # ── 3. Ensina palavras do turno em background (não bloqueia) ──────
        asyncio.create_task(ensinar_em_background(ws, msg_llm))

        # ── 4. Selene ouve + lê + responde ────────────────────────────────
        selene_resp = await selene_ouvir_e_ler(ws, msg_llm)

        if selene_resp:
            print(f"  🧠 Selene: {truncar(selene_resp, 120)}")
            silencio_streak = 0
        else:
            selene_resp = "(silêncio)"
            silencio_streak += 1
            print(f"  🧠 Selene: (sem resposta) [{silencio_streak} consecutivos]")

        # ── 5. Registra resposta da Selene no histórico do LLM ───────────
        historico.append({
            "role": "user",
            "content": selene_resp if selene_resp != "(silêncio)"
                       else "(A Selene não respondeu desta vez. Continue a conversa de outra forma.)"
        })

        # ── 6. Detecta loop e reseta histórico se necessário ─────────────
        em_loop   = _detectar_loop(janela_llm)
        muitos_silencias = silencio_streak >= LOOP_MAX_SILENCIO
        rotacao_natural  = (turno - turno_ultimo_reset) >= ROTACAO_TURNOS

        if em_loop or muitos_silencias or rotacao_natural:
            rotacao_idx += 1
            novo_subassunto = _proximo_subassunto(assunto_base, rotacao_idx)
            motivo = ("loop detectado" if em_loop
                      else f"{LOOP_MAX_SILENCIO} silêncios" if muitos_silencias
                      else f"rotação a cada {ROTACAO_TURNOS} turnos")
            print(f"\n  🔄 [{motivo}] → novo sub-assunto: '{novo_subassunto}'")

            # Reset: preserva apenas system prompt + injeta nova abertura
            assunto_atual = novo_subassunto
            historico = [
                {"role": "system", "content": montar_system_prompt(assunto_base, extra)}
            ]
            historico.append({
                "role": "user",
                "content": (
                    f"Mude o foco da conversa para '{novo_subassunto}', relacionado a '{assunto_base}'. "
                    f"Comece com algo novo e curioso, diferente do que foi dito antes. "
                    f"Faça uma única pergunta simples para a Selene."
                )
            })
            janela_llm.clear()
            silencio_streak = 0
            turno_ultimo_reset = turno
            continue  # gera a próxima mensagem já com o novo assunto

        # ── 7. Estímulo de curiosidade a cada N turnos ────────────────────
        if turno % ESTIMULO_A_CADA == 0:
            print(f"  ✨ [estímulo de curiosidade]")
            await injetar_estimulo(ws, assunto_atual)

        # ── 8. Pausa natural entre turnos ─────────────────────────────────
        await asyncio.sleep(PAUSA_TURNO)

        # ── 9. Mantém histórico em tamanho razoável (últimos 30 turnos) ───
        # Preserva sempre o system prompt (índice 0)
        if len(historico) > 62:   # 1 system + 30 pares user/assistant
            historico = historico[:1] + historico[-60:]


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

async def main():
    global ASSUNTO, MODELO, DURACAO_MAX

    # ── Resolve argumentos ────────────────────────────────────────────────
    # argv[1]: número (1-8) ou texto livre do assunto
    # argv[2]: modelo Ollama — OPCIONAL, se omitido usa o ideal para o assunto
    # argv[3]: duração em minutos (0 = infinito)
    extra      = ""
    modelo_forcado = False   # True se o usuário passou argv[2]
    arg1       = sys.argv[1].strip() if len(sys.argv) > 1 else ""

    if arg1 in ASSUNTOS_PRONTOS:
        ASSUNTO, extra = ASSUNTOS_PRONTOS[arg1]
    elif arg1:
        ASSUNTO = arg1   # texto livre — usa modelo padrão

    if len(sys.argv) > 2:
        MODELO = sys.argv[2]
        modelo_forcado = True

    if len(sys.argv) > 3:
        try:
            DURACAO_MAX = int(sys.argv[3]) * 60   # minutos → segundos
        except ValueError:
            pass

    # ── Seleção automática de modelo ─────────────────────────────────────
    # Só aplica se o usuário NÃO forçou um modelo via argv[2]
    if not modelo_forcado and arg1 in MODELO_POR_ASSUNTO:
        MODELO = MODELO_POR_ASSUNTO[arg1]

    origem_modelo = "forçado pelo usuário" if modelo_forcado else \
                    (f"automático para '{ASSUNTO[:30]}'" if arg1 in MODELO_POR_ASSUNTO else "padrão")

    print("\n" + "═" * 62)
    print("  👩‍🏫 TUTOR SELENE — conversa contínua via LLM local")
    print("═" * 62)
    print(f"  Assunto : {ASSUNTO}")
    print(f"  Modelo  : {MODELO}  ({origem_modelo})")

    # Verifica Ollama
    print("\n🔍 Verificando Ollama...")
    if not await verificar_ollama():
        sys.exit(1)
    print(f"  ✅ Modelo '{MODELO}' disponível")

    # Conecta Selene
    print(f"\n🔌 Conectando à Selene em {WS_URL}...")
    try:
        ws = await asyncio.wait_for(
            websockets.connect(WS_URL, ping_interval=30, ping_timeout=None),
            timeout=8.0,
        )
        print("  ✅ Selene conectada")
    except Exception as e:
        print(f"  ❌ {e}")
        print("     Verifique se o servidor Selene está rodando (cargo run)")
        sys.exit(1)

    try:
        await conversa(ws, ASSUNTO, extra)
    except KeyboardInterrupt:
        print("\n\n⚡ Sessão encerrada pelo usuário.")
    finally:
        # Exporta o estado aprendido
        try:
            await ws.send(json.dumps({"action": "export_linguagem"}))
            await asyncio.sleep(0.8)
            print("  💾 Estado da linguagem exportado.")
        except Exception:
            pass
        await ws.close()
        print("  🔌 Desconectado.\n")


if __name__ == "__main__":
    asyncio.run(main())
