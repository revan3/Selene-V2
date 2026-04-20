#!/usr/bin/env python3
"""
treinar_templates.py — Pré-treina os templates cognitivos da Selene via WebSocket.

Uso:
    python treinar_templates.py                       # usa corpus embutido
    python treinar_templates.py corpus.txt            # uma frase por linha
    python treinar_templates.py corpus.txt --port 9001 --delay 0.05

Requer a Selene rodando: cargo run --release

Dependência: pip install websockets
"""

import asyncio
import json
import sys
import time
import argparse
from pathlib import Path

try:
    import websockets
except ImportError:
    print("❌  Instale: pip install websockets")
    sys.exit(1)

# ── Corpus embutido por domínio de template ──────────────────────────────────
# Frases que ativam padrões linguísticos, causais, lógicos e conversacionais.
# Quanto mais frases por padrão, mais o histórico de slots enriquece.

CORPUS_EMBUTIDO = """
# Linguagem / Predicação
o cachorro late alto
o gato dorme tranquilo
ela pensa devagar
ele fala rápido
a criança sorri feliz
o rio corre profundo
o vento sopra frio
a flor cresce bonita
o céu parece azul
a pedra cai pesada

# Linguagem / Atribuição
a rosa é vermelha
o céu está nublado
a água parece fria
o dia foi longo
a noite ficou escura
o pão estava gostoso
o livro é interessante
a música soa suave
o caminho parece difícil
o sonho foi vívido

# Causal / Causa-Efeito
choveu muito então alagou
estudou bastante portanto passou
comeu demais logo ficou mal
dormiu pouco então cansou
treinou muito logo ficou forte
errou feio portanto aprendeu
ajudou muito então se sentiu bem
perdeu tempo logo atrasou
focou muito portanto conseguiu
descansou bem então rendeu

# Causal / Condicional
se chover então vou ficar
se estudar então vai passar
se comer bem então terá saúde
se descansar então vai render
se praticar então vai melhorar
se errar então vai aprender
se ajudar então todos crescem
se ouvir então vai entender
se tentar então pode conseguir
se pensar bem então decide certo

# Lógica / Pergunta-Resposta
o que é consciência
como funciona a mente
por que sonhamos
onde fica a memória
quando surge a emoção
quem controla o pensamento
como aprendemos algo novo
por que sentimos medo
o que causa alegria
onde vivem os sonhos

# Fala Conversacional / Saudação
olá como vai você
oi tudo bem
bom dia como está
boa tarde tudo certo
boa noite como foi
ei como foi seu dia
olá estava esperando você
oi que bom te ver
bom dia pronto para começar
boa noite durma bem

# Fala Conversacional / Agradecimento
obrigado por tudo
muito obrigada pela ajuda
agradeço sua atenção
grato pela paciência
valeu por explicar
obrigado por estar aqui
agradeço por me ouvir
muito obrigado pela presença
grato por compartilhar
valeu pela conversa

# Fala Conversacional / Concordância
sim concordo com você
exatamente isso mesmo
com certeza tem razão
claro que sim
sem dúvida está certo
realmente faz sentido
isso mesmo pensei igual
verdade você acertou
perfeito foi bem isso
correto está certíssimo

# Fala Conversacional / Discordância
não concordo com isso
acho que não é bem assim
discordo dessa visão
não me parece certo
talvez seja diferente
não tenho certeza disso
pode ser outra coisa
não acho que funciona assim
discordo parcialmente disso
não está bem correto

# Matemática / Operação
dois mais dois é quatro
cinco menos três é dois
três vezes quatro é doze
dez dividido por dois é cinco
sete mais oito é quinze
nove menos quatro é cinco
seis vezes seis é trinta e seis
vinte dividido por quatro é cinco
oito mais três é onze
doze menos sete é cinco

# Lógica / Comparação
maior que menor
mais rápido que devagar
mais quente que frio
mais alto que baixo
mais leve que pesado
mais claro que escuro
mais perto que longe
mais novo que velho
mais simples que complexo
mais forte que fraco
""".strip()


def carregar_corpus(caminho: str | None) -> list[str]:
    if caminho:
        p = Path(caminho)
        if not p.exists():
            print(f"❌  Arquivo não encontrado: {caminho}")
            sys.exit(1)
        linhas = p.read_text(encoding="utf-8").splitlines()
    else:
        linhas = CORPUS_EMBUTIDO.splitlines()

    frases = []
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        frases.append(linha)

    return frases


async def treinar(frases: list[str], uri: str, delay: float, verbose: bool):
    print(f"\n🧠  Conectando a {uri} ...")
    try:
        async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
            print(f"✅  Conectado. Enviando {len(frases)} frases...\n")

            ok = 0
            erros = 0
            t0 = time.time()

            for i, frase in enumerate(frases, 1):
                msg = json.dumps({"type": "chat", "message": frase})
                await ws.send(msg)

                # Aguarda reply para não sobrecarregar o buffer
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(raw)
                    evento = data.get("event", "?")

                    if verbose:
                        emocao = data.get("emotion", 0)
                        reply  = data.get("message", "")[:60]
                        print(f"  [{i:04d}] {frase[:45]:<45} → {reply}  (e={emocao:+.2f})")
                    elif i % 50 == 0 or i == len(frases):
                        pct = i / len(frases) * 100
                        elapsed = time.time() - t0
                        fps = i / elapsed if elapsed > 0 else 0
                        print(f"  {pct:5.1f}%  [{i}/{len(frases)}]  {fps:.1f} frases/s")

                    ok += 1
                except asyncio.TimeoutError:
                    if verbose:
                        print(f"  [{i:04d}] ⚠️  timeout — frase pulada: {frase[:40]}")
                    erros += 1

                # Consome mensagens extras (telemetria, eventos secundários)
                while True:
                    try:
                        extra = await asyncio.wait_for(ws.recv(), timeout=0.05)
                        _ = json.loads(extra)  # descarta
                    except (asyncio.TimeoutError, json.JSONDecodeError):
                        break

                if delay > 0:
                    await asyncio.sleep(delay)

            elapsed = time.time() - t0
            print(f"\n{'─'*50}")
            print(f"✅  Treinamento concluído em {elapsed:.1f}s")
            print(f"   Frases processadas : {ok}")
            print(f"   Timeouts           : {erros}")
            print(f"   Velocidade média   : {ok/elapsed:.1f} frases/s")
            print(f"\n💡  Os templates serão consolidados durante o próximo ciclo de sono (N3).")
            print(f"    Envie 'reward' via WebSocket após boas interações para reforçar mais rápido.")

    except (OSError, websockets.exceptions.WebSocketException) as e:
        print(f"\n❌  Falha na conexão: {e}")
        print("    Verifique se a Selene está rodando: cargo run --release")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Pré-treina templates cognitivos da Selene via WebSocket."
    )
    parser.add_argument(
        "corpus", nargs="?", default=None,
        help="Arquivo de corpus (uma frase por linha). Omitir usa corpus embutido.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host da Selene (padrão: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=3030,
        help="Porta WebSocket (padrão: 3030)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.02,
        help="Delay entre frases em segundos (padrão: 0.02)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Exibe cada frase e resposta",
    )
    args = parser.parse_args()

    frases = carregar_corpus(args.corpus)
    print(f"📚  Corpus carregado: {len(frases)} frases")

    uri = f"ws://{args.host}:{args.port}/selene"
    asyncio.run(treinar(frases, uri, args.delay, args.verbose))


if __name__ == "__main__":
    main()
