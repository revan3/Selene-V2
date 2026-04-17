# scripts/selene_tutor.py
# Tutor de aprendizado da Selene v2.3
#
# Protocolo de 3 passes (spaced repetition bio-inspirada):
#   Pass 1 — APRESENTAÇÃO:   todas as palavras, 1x, intervalo 55ms
#   Pass 2 — REFORÇO:        palavras com |valence| >= 0.5, 2x, intervalo 80ms
#   Pass 3 — CONSOLIDAÇÃO:   palavras com |valence| >= 0.8, 3x, intervalo 150ms
#
# Entre categorias: dispara um mini-sweep STDP ("train" com 1 época)
# para consolidar o que foi aprendido antes de passar ao próximo grupo.
#
# URL corrigida: ws://127.0.0.1:3030/selene

import asyncio
import websockets
import json
import os
import sys
import time

WS_URL   = "ws://127.0.0.1:3030/selene"
LEXICON  = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "selene_lexicon.json"))

# ─── Thresholds de spaced repetition ────────────────────────────────────────
PASS2_THRESHOLD = 0.5   # |valence| >= este valor → reforço
PASS3_THRESHOLD = 0.8   # |valence| >= este valor → consolidação

# ─── Intervalos entre palavras por passe ─────────────────────────────────────
DELAY_PASS1 = 0.055   # 55ms  (~18 palavras/s, respeitando benchmark)
DELAY_PASS2 = 0.080   # 80ms  (reforço mais cuidadoso)
DELAY_PASS3 = 0.150   # 150ms (consolidação profunda)

# ─── Utilidades ──────────────────────────────────────────────────────────────

def bar(value, width=20, fill="█", empty="░"):
    """Barra de progresso visual para valores em [-1, 1]."""
    if value >= 0:
        n = int(abs(value) * width)
        return " " * (width - n) + fill * n + "│" + " " * width
    else:
        n = int(abs(value) * width)
        return " " * width + "│" + fill * n + empty * (width - n)

def sign(v):
    return "+" if v >= 0 else ""

def load_lexicon():
    if not os.path.exists(LEXICON):
        print("❌ selene_lexicon.json não encontrado.")
        print("   Execute primeiro: python scripts/generate_lexicon.py")
        sys.exit(1)
    with open(LEXICON, "r", encoding="utf-8") as f:
        return json.load(f)

# ─── Envio de uma palavra ────────────────────────────────────────────────────

async def ensinar_palavra(ws, entry, pass_num, word_idx, total_words):
    """Envia uma palavra e aguarda learn_ack, ignorando telemetria intercalada."""
    payload = {
        "action":    "learn",
        "text":      entry["word"],
        "valence":   entry["valence"],
        "context":   entry["context"],
        "intensity": entry["intensity"],
    }
    await ws.send(json.dumps(payload))

    # Aguarda especificamente o learn_ack (pode chegar depois de telemetria)
    ack = None
    for _ in range(5):
        raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
        data = json.loads(raw)
        if data.get("event") == "learn_ack":
            ack = data
            break
        # Se for telemetria (NeuralStatus), ignora e espera mais

    dopa  = ack["dopamine"]     if ack else 0.5
    sero  = ack["serotonin"]    if ack else 0.5
    emocao = ack["emotion"]     if ack else 0.0
    step  = ack["step"]         if ack else 0

    label_pass = ["", "APRESENTAÇÃO", "REFORÇO    ", "CONSOLIDAÇÃO"][pass_num]
    pct = f"{word_idx}/{total_words}"

    print(
        f"  [{label_pass}] {pct:>8}  {entry['word']:20s}  "
        f"val={sign(entry['valence'])}{entry['valence']:+.2f}  "
        f"dopa={dopa:.3f}  sero={sero:.3f}  emo={emocao:+.3f}  "
        f"ctx={entry['context']}"
    )

# ─── Mini-sweep STDP entre categorias ───────────────────────────────────────

async def stdp_sweep(ws, categoria):
    """Dispara 1 época de treino STDP para consolidar a categoria recém-aprendida."""
    payload = {"action": "train", "epochs": 1, "ltp": 0.012, "ltd": 0.003}
    await ws.send(json.dumps(payload))
    # Drena resposta do servidor (progress + done)
    for _ in range(3):
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            data = json.loads(raw)
            if data.get("status") == "done":
                break
        except asyncio.TimeoutError:
            break
    print(f"  ⚡ STDP sweep após '{categoria}' concluído")

# ─── Recompensa/punição no fim de cada passe ─────────────────────────────────

async def modular(ws, reward: bool, value: float):
    action = "reward" if reward else "punish"
    await ws.send(json.dumps({"action": action, "value": value}))
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
        data = json.loads(raw)
        event = data.get("event", "")
        if event == "reward_ack":
            print(f"  🏆 Recompensa aplicada → dopa={data['dopamine']:.3f}  sero={data['serotonin']:.3f}")
        elif event == "punish_ack":
            print(f"  ⚡ Penalidade aplicada  → dopa={data['dopamine']:.3f}  nor={data['noradrenaline']:.3f}")
    except asyncio.TimeoutError:
        pass

# ─── Loop principal ──────────────────────────────────────────────────────────

async def run_session():
    lexico = load_lexicon()

    # Flatlist completa com metadados de categoria
    all_entries = []
    for cat, words in lexico.items():
        for w in words:
            all_entries.append({**w, "categoria": cat})

    total = len(all_entries)
    reforco      = [e for e in all_entries if abs(e["valence"]) >= PASS2_THRESHOLD]
    consolidacao = [e for e in all_entries if abs(e["valence"]) >= PASS3_THRESHOLD]

    print("\n" + "="*70)
    print("🧠  SELENE TUTOR v2.3 — Sessão de Aprendizado Neural")
    print("="*70)
    print(f"  Léxico carregado: {total} conceitos em {len(lexico)} categorias")
    print(f"  Pass 1 — Apresentação:   {total} palavras")
    print(f"  Pass 2 — Reforço:        {len(reforco)} palavras  (|val| ≥ {PASS2_THRESHOLD})")
    print(f"  Pass 3 — Consolidação:   {len(consolidacao)} palavras  (|val| ≥ {PASS3_THRESHOLD})")
    print(f"  Endpoint: {WS_URL}")
    print("="*70)

    try:
        async with websockets.connect(WS_URL, ping_interval=10, ping_timeout=5) as ws:
            t_inicio = time.time()

            # ── PASS 1: APRESENTAÇÃO ─────────────────────────────────────────
            print(f"\n{'─'*70}")
            print("📖  PASS 1 — APRESENTAÇÃO (todas as palavras, uma vez)")
            print(f"{'─'*70}")

            for cat, words in lexico.items():
                print(f"\n  [Categoria: {cat}]")
                for i, entry in enumerate(words, 1):
                    await ensinar_palavra(ws, entry, 1, i, len(words))
                    await asyncio.sleep(DELAY_PASS1)
                await stdp_sweep(ws, cat)

            await modular(ws, reward=True, value=0.4)
            elapsed = time.time() - t_inicio
            print(f"\n  ✅ Pass 1 completo em {elapsed:.1f}s  ({total/elapsed:.1f} palavras/s)")

            # ── PASS 2: REFORÇO ──────────────────────────────────────────────
            print(f"\n{'─'*70}")
            print(f"🔁  PASS 2 — REFORÇO ({len(reforco)} palavras salientes, 2x)")
            print(f"{'─'*70}")

            for rep in range(2):
                print(f"\n  [Repetição {rep+1}/2]")
                for i, entry in enumerate(reforco, 1):
                    await ensinar_palavra(ws, entry, 2, i, len(reforco))
                    await asyncio.sleep(DELAY_PASS2)
                # Sweep após cada repetição
                await stdp_sweep(ws, f"reforco_rep{rep+1}")

            await modular(ws, reward=True, value=0.5)
            print(f"\n  ✅ Pass 2 completo")

            # ── PASS 3: CONSOLIDAÇÃO ─────────────────────────────────────────
            print(f"\n{'─'*70}")
            print(f"🧬  PASS 3 — CONSOLIDAÇÃO ({len(consolidacao)} conceitos críticos, 3x)")
            print(f"{'─'*70}")

            for rep in range(3):
                print(f"\n  [Repetição {rep+1}/3]")
                for i, entry in enumerate(consolidacao, 1):
                    await ensinar_palavra(ws, entry, 3, i, len(consolidacao))
                    await asyncio.sleep(DELAY_PASS3)
                await stdp_sweep(ws, f"consolidacao_rep{rep+1}")

            await modular(ws, reward=True, value=0.7)

            # ── RELATÓRIO FINAL ──────────────────────────────────────────────
            total_elapsed = time.time() - t_inicio
            total_eventos = total + len(reforco) * 2 + len(consolidacao) * 3

            print(f"\n{'='*70}")
            print("🎓  SESSÃO DE APRENDIZADO CONCLUÍDA")
            print(f"{'='*70}")
            print(f"  Total de eventos de aprendizado : {total_eventos}")
            print(f"  Tempo total                     : {total_elapsed:.1f}s")
            print(f"  Taxa média                      : {total_eventos/total_elapsed:.1f} eventos/s")
            print(f"\n  Próximo passo: python scripts/selene_exam.py")
            print(f"{'='*70}\n")

    except ConnectionRefusedError:
        print("\n❌ Não foi possível conectar ao servidor Selene.")
        print("   Certifique-se que 'cargo run' está rodando em outro terminal.")
    except FileNotFoundError:
        print("\n❌ selene_lexicon.json não encontrado.")
        print("   Execute primeiro: python scripts/generate_lexicon.py")
    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(run_session())
