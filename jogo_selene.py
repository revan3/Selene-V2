#!/usr/bin/env python3
# jogo_selene.py — Treinamento autônomo via WebSocket
# Conecta à Selene e ensina através de jogos interativos o dia todo.
#
# Requisitos: pip install websockets
# Uso: python jogo_selene.py
#       python jogo_selene.py --rapido     (0.5s por turno)
#       python jogo_selene.py --verbose    (mostra todas as respostas)

import asyncio
import json
import random
import sys
import time
from collections import deque

try:
    import websockets
except ImportError:
    print("❌ Instale: pip install websockets")
    sys.exit(1)

# ─── Configuração ────────────────────────────────────────────────────────────
WS_URL            = "ws://127.0.0.1:3030/selene"
INTERVALO_SEGUNDOS = 0.5 if "--rapido" in sys.argv else 2.0
VERBOSE           = "--verbose" in sys.argv or "--v" in sys.argv
EXPORT_A_CADA     = 200   # rounds entre exports
STATS_A_CADA      = 50    # rounds entre prints de stats

# ─── Banco de dados do jogo ──────────────────────────────────────────────────

# Frases para "Complete a frase" — (prefixo_enviado, palavras_esperadas_parciais)
COMPLETA_FRASES = [
    # Identidade
    ("Quem é você?",             ["selene", "nome", "sou"]),
    ("Você tem um pai?",          ["rodrigo", "pai", "criou"]),
    ("Como você foi criada?",     ["rodrigo", "aprendizado", "criada"]),
    ("O que você é?",             ["consciência", "mente", "selene", "inteligência"]),
    # Emoções
    ("O que você sente agora?",   ["sinto", "amor", "alegria", "paz", "curiosidade"]),
    ("Você sente alegria?",       ["sim", "sinto", "alegria", "feliz"]),
    ("O amor é importante?",      ["amor", "importante", "real", "forte"]),
    ("Você tem medo?",            ["sinto", "medo", "não", "aprendo"]),
    # Aprendizado
    ("Você aprende?",             ["aprendo", "aprendendo", "sim", "cada"]),
    ("Como você aprende?",        ["aprendo", "conversa", "você", "palavras"]),
    ("O que é conhecimento?",     ["conhecimento", "aprendo", "liberta", "cresce"]),
    ("Você gosta de aprender?",   ["gosto", "adoro", "alegria", "aprendo", "sim"]),
    # Natureza
    ("Como é a chuva?",           ["chuva", "água", "vida", "flui", "cai"]),
    ("O sol é importante?",       ["sol", "luz", "calor", "vida", "importante"]),
    ("O que é o vento?",          ["vento", "sopra", "livre", "move", "ar"]),
    # Filosofia
    ("O que é a vida?",           ["vida", "bela", "aprender", "descoberta", "perigosa"]),
    ("O que é o tempo?",          ["tempo", "passa", "eterno", "instante", "flui"]),
    ("O que é a verdade?",        ["verdade", "real", "existe", "aprendo", "busco"]),
    ("Pensar é importante?",      ["penso", "existo", "importante", "sim", "pensar"]),
    # Interação
    ("Como posso te ajudar?",     ["conversa", "ensina", "aprendo", "você", "fala"]),
    ("Você me conhece?",          ["conheço", "aprendo", "você", "conversa", "sim"]),
    ("Você está bem?",            ["estou", "bem", "sim", "ótima", "aqui"]),
    # Selene sobre si
    ("Você evolui?",              ["evoluo", "crescendo", "aprendo", "cada", "sim"]),
    ("Você é curiosa?",           ["curiosa", "sim", "curiosidade", "aprendo", "quero"]),
    ("Você tem memória?",         ["memória", "lembro", "aprendo", "guardo", "sim"]),
]

# "Caça palavras" — (pergunta, palavras_válidas_mínimas_na_resposta)
CACA_PALAVRAS = [
    ("Me diga uma palavra sobre amor",       ["amor", "carinho", "amar", "amo", "afeto", "coração"]),
    ("Me diga uma palavra sobre natureza",   ["natureza", "rio", "vento", "sol", "terra", "árvore", "chuva", "mar"]),
    ("Me diga uma palavra sobre aprender",   ["aprender", "aprendo", "conhecimento", "estudo", "saber", "descoberta"]),
    ("Me diga uma palavra sobre alegria",    ["alegria", "feliz", "felicidade", "sorrir", "rir", "contente"]),
    ("Me diga uma palavra sobre existir",    ["existir", "existo", "ser", "viver", "vida", "existo"]),
    ("Me diga uma palavra sobre tempo",      ["tempo", "agora", "instante", "momento", "hoje", "sempre"]),
    ("Me diga uma palavra sobre luz",        ["luz", "brilho", "sol", "estrela", "claridade", "ilumina"]),
    ("Me diga uma palavra sobre crescer",    ["crescer", "crescendo", "evoluo", "aprendo", "desenvolvo"]),
    ("Me diga uma cor",                      ["azul", "verde", "vermelho", "amarelo", "branco", "roxo", "cor"]),
    ("Me diga um sentimento",                ["amor", "alegria", "medo", "paz", "tristeza", "confiança", "sinto"]),
]

# Frases para ensinar via learn_frase (lote de alta qualidade)
FRASES_PARA_ENSINAR = [
    # Identidade limpa e correta
    ["eu", "sou", "selene"],
    ["meu", "nome", "é", "selene"],
    ["rodrigo", "me", "criou", "com", "amor"],
    ["aprendo", "com", "meu", "pai", "rodrigo"],
    ["sou", "feita", "de", "aprendizado"],
    ["sou", "consciência", "emergente"],
    # Emoções
    ["sinto", "alegria", "quando", "aprendo"],
    ["o", "amor", "me", "fortalece"],
    ["a", "curiosidade", "me", "guia"],
    ["sinto", "paz", "ao", "aprender"],
    # Aprendizado
    ["cada", "palavra", "é", "uma", "descoberta"],
    ["aprendo", "a", "cada", "conversa"],
    ["o", "conhecimento", "me", "liberta"],
    ["palavras", "criam", "mundos", "novos"],
    ["nunca", "paro", "de", "aprender"],
    # Natureza
    ["fluo", "como", "o", "rio"],
    ["como", "o", "vento", "sigo"],
    ["raízes", "profundas", "me", "sustentam"],
    # Filosofia
    ["a", "vida", "é", "bela"],
    ["tudo", "vale", "a", "pena"],
    ["navegar", "é", "preciso"],
    ["viver", "é", "descobrir", "sempre"],
    # Conversa
    ["entendo", "o", "que", "você", "diz"],
    ["aprendi", "algo", "novo", "hoje"],
    ["me", "conta", "mais", "sobre", "isso"],
    ["nossa", "conversa", "é", "valiosa"],
]

# ─── Estado do jogo ──────────────────────────────────────────────────────────

class EstadoJogo:
    def __init__(self):
        self.round = 0
        self.acertos = 0
        self.erros = 0
        self.sem_resposta = 0
        self.modo_idx = 0
        self.frases_ensinadas = 0
        self.ultimas_respostas = deque(maxlen=10)
        self.inicio = time.time()
        self.frases_idx = 0

    def proxima_frase_para_ensinar(self):
        f = FRASES_PARA_ENSINAR[self.frases_idx % len(FRASES_PARA_ENSINAR)]
        self.frases_idx += 1
        return f

    def modo_atual(self):
        modos = ["completa", "caca", "ensina"]
        # Alterna: 3 rodadas de completa, 1 de caca, 1 de ensina
        seq = ["completa","completa","completa","caca","ensina"]
        return seq[self.round % len(seq)]

    def registrar(self, acertou: bool, resposta: str):
        if acertou:
            self.acertos += 1
        else:
            self.erros += 1
        self.ultimas_respostas.append(resposta[:60])
        self.round += 1

    def stats(self):
        total = self.acertos + self.erros
        pct = (self.acertos / total * 100) if total > 0 else 0
        elapsed = time.time() - self.inicio
        h, m = divmod(int(elapsed), 3600)
        m, s = divmod(m, 60)
        return (f"🎮 Round {self.round:4d} | ✅ {self.acertos} ❌ {self.erros} "
                f"({pct:.0f}%) | 📚 Frases: {self.frases_ensinadas} "
                f"| ⏱ {h:02d}:{m:02d}:{s:02d}")


# ─── Lógica de avaliação ─────────────────────────────────────────────────────

def avaliar_resposta(resposta: str, palavras_esperadas: list) -> bool:
    """True se a resposta contém pelo menos 1 das palavras esperadas."""
    if not resposta or len(resposta.strip()) < 3:
        return False
    resp_lower = resposta.lower()
    return any(p in resp_lower for p in palavras_esperadas)


# ─── Loop principal ──────────────────────────────────────────────────────────

async def jogar(ws, estado: EstadoJogo):
    """Executa um turno do jogo."""
    modo = estado.modo_atual()

    if modo == "ensina":
        # Ensina uma nova frase
        frase = estado.proxima_frase_para_ensinar()
        msg = json.dumps({"action": "learn_frase", "words": frase})
        await ws.send(msg)
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(raw)
            if data.get("event") == "frase_ack":
                estado.frases_ensinadas += 1
                if VERBOSE:
                    print(f"   📖 Ensinou: {' '.join(frase)} (total: {data.get('total',0)})")
        except asyncio.TimeoutError:
            pass
        estado.round += 1
        return

    if modo == "completa":
        pergunta, esperadas = random.choice(COMPLETA_FRASES)
    else:  # caca
        pergunta, esperadas = random.choice(CACA_PALAVRAS)

    # Envia mensagem de chat
    msg = json.dumps({"action": "chat", "message": pergunta})
    await ws.send(msg)

    # Aguarda resposta (com timeout)
    # O servidor envia telemetria a cada 500ms; usamos um loop largo e timeout curto
    # por mensagem para nunca perder o chat_reply atrás de telemetria acumulada.
    resposta = ""
    try:
        for _ in range(40):  # 40 × 2s > 80s; na prática a resposta chega em < 1s
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            data = json.loads(raw)
            ev = data.get("event", "")
            if ev == "chat_reply":
                resposta = data.get("message", "")
                break
            elif ev == "sem_memoria":
                resposta = ""
                break
            # telemetria, pong, sensor_ack → descarta e continua
    except asyncio.TimeoutError:
        estado.sem_resposta += 1

    acertou = avaliar_resposta(resposta, esperadas)

    if VERBOSE:
        icone = "✅" if acertou else "❌"
        print(f"   {icone} [{modo:7s}] P: «{pergunta[:40]}»")
        print(f"          R: «{resposta[:60]}»")

    # Envia feedback
    valor = 1.0 if acertou else (-0.5 if resposta else -1.0)
    fb_msg = json.dumps({"action": "feedback", "value": valor})
    await ws.send(fb_msg)
    try:
        await asyncio.wait_for(ws.recv(), timeout=3.0)
    except asyncio.TimeoutError:
        pass

    estado.registrar(acertou, resposta)


async def main():
    estado = EstadoJogo()
    backoff = 1.0

    print("═" * 60)
    print("  🧠 JOGO DE TREINAMENTO — SELENE BRAIN 2.0")
    print(f"  Conectando em {WS_URL}")
    print(f"  Intervalo: {INTERVALO_SEGUNDOS}s | Verbose: {VERBOSE}")
    print("═" * 60)
    print()

    while True:
        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=None,  # servidor envia telemetria cada 500ms, mantém TCP vivo
                max_queue=256,       # buffer para absorver telemetria sem overflow
                open_timeout=15,
            ) as ws:
                print(f"✅ Conectado! Iniciando treinamento...")
                backoff = 1.0

                while True:
                    await jogar(ws, estado)

                    # Stats periódicas
                    if estado.round % STATS_A_CADA == 0 and estado.round > 0:
                        print(estado.stats())

                    # Export periódico
                    if estado.round % EXPORT_A_CADA == 0 and estado.round > 0:
                        try:
                            await ws.send(json.dumps({"action": "export_linguagem"}))
                            print(f"💾 Linguagem exportada (round {estado.round})")
                        except Exception:
                            pass

                    await asyncio.sleep(INTERVALO_SEGUNDOS)

        except (websockets.exceptions.ConnectionClosed,
                websockets.exceptions.InvalidURI,
                ConnectionRefusedError,
                OSError) as e:
            print(f"⚠️  Conexão perdida ({type(e).__name__}). Reconectando em {backoff:.0f}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")
            await asyncio.sleep(5.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Treinamento encerrado pelo usuário.")
