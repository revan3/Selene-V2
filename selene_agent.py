#!/usr/bin/env python3
"""
Selene Agent — Corpo Digital (Conceito A, V4.6.1)
==================================================
Corre na máquina-MUNDO (ex.: o IdeaPad S145 ocioso). Fecha o loop sensório-motor:
  captura ecrã → envia observação+recompensa → recebe ação → injeta tecla.

A Selene (no Avell) decide a ação com o MotorCortex (ator Q-learning) + dopamina.

PRÉ-REQUISITO: Selene a correr e acessível em ws://HOST:PORTA/selene.
  pip install websockets mss pydirectinput pillow numpy
  (pydirectinput é melhor para jogos; cai para pyautogui se ausente)

USO (jogo de 4 teclas, ex. Snake/2048 no browser):
  1. Abre o jogo e posiciona-o num canto conhecido do ecrã.
  2. python selene_agent.py --region 100,100,400,400 --grid 12 --fps 4 \
         --host 192.168.0.10 --token MEU_TOKEN
  3. Clica no jogo para lhe dar foco. A Selene começa a jogar.

⚠️  SEGURANÇA: este daemon injeta teclado nesta máquina. Corre-o numa máquina
    dedicada/sacrificial. Liga-se só ao host/token que indicares.
"""

from __future__ import annotations  # compat 3.9 (anotações X | None)

import argparse
import asyncio
import json
import sys

try:
    import websockets
    import numpy as np
    from mss import mss
except ImportError:
    print("[FAIL] Instale: pip install websockets mss numpy pillow")
    sys.exit(1)

# Injeção de teclado: pydirectinput (jogos) → fallback pyautogui.
try:
    import pydirectinput as kb
    kb.PAUSE = 0.0
except ImportError:
    try:
        import pyautogui as kb
        kb.PAUSE = 0.0
    except ImportError:
        print("[FAIL] Instale: pip install pydirectinput  (ou pyautogui)")
        sys.exit(1)

TECLAS = {"up": "up", "down": "down", "left": "left", "right": "right"}


def capturar_grid(sct, regiao, n: int) -> np.ndarray:
    """Captura a região e reduz para uma grelha NxN em tons de cinza [0,1]."""
    raw = np.asarray(sct.grab(regiao))[:, :, :3]  # BGRA → BGR
    cinza = raw.mean(axis=2)  # luminância simples
    h, w = cinza.shape
    # Downsample por blocos (média) para NxN — barato e sem dependência de PIL.
    bh, bw = max(1, h // n), max(1, w // n)
    grid = cinza[: bh * n, : bw * n].reshape(n, bh, n, bw).mean(axis=(1, 3))
    return (grid / 255.0).astype(np.float32)


def compute_reward(prev: np.ndarray | None, atual: np.ndarray) -> float:
    """
    ⚠️  RECOMPENSA PLACEHOLDER (substitua por jogo!).
    Default: magnitude da mudança no ecrã (= "algo aconteceu / progresso").
    Prova o loop, mas é hackeável. Para Snake/2048: ler o score (delta) e
    devolver +score_delta; morrer/game-over → recompensa negativa.
    """
    if prev is None:
        return 0.0
    return float(np.clip(np.abs(atual - prev).mean() * 5.0, 0.0, 1.0))


async def jogar(url: str, regiao: dict, n_grid: int, fps: float):
    intervalo = 1.0 / max(0.5, fps)
    prev_grid = None
    passos = 0
    async with mss() as sct, websockets.connect(url, max_size=None, ping_interval=20) as ws:
        print(f"🎮 Agente ligado a {url}. Região={regiao} grelha={n_grid}x{n_grid} fps={fps}")
        print("   (clica no jogo para lhe dar foco)")
        while True:
            grid = capturar_grid(sct, regiao, n_grid)
            reward = compute_reward(prev_grid, grid)
            prev_grid = grid

            await ws.send(json.dumps({
                "action": "env_step",
                "reward": reward,
                "grid": grid.flatten().tolist(),
                "done": False,
            }))
            # Espera a ação da Selene
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(resp)
                if data.get("type") == "motor_action":
                    tecla = TECLAS.get(data.get("key", ""), None)
                    if tecla:
                        kb.press(tecla)
                        passos += 1
                        if passos % 20 == 0:
                            print(f"  passo {passos}: ação={tecla} reward={reward:.3f}", end="\r")
            except (asyncio.TimeoutError, json.JSONDecodeError):
                pass

            await asyncio.sleep(intervalo)


def main():
    ap = argparse.ArgumentParser(description="Selene Agent — corpo digital (loop sensório-motor).")
    ap.add_argument("--region", required=True,
                    help="Região de captura 'x,y,largura,altura' (ex.: 100,100,400,400)")
    ap.add_argument("--grid", type=int, default=12, help="Resolução da grelha NxN (default 12)")
    ap.add_argument("--fps", type=float, default=4.0, help="Passos por segundo (default 4)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="3030")
    ap.add_argument("--token", default="", help="SELENE_TOKEN, se exigido")
    args = ap.parse_args()

    try:
        x, y, w, h = (int(v) for v in args.region.split(","))
    except ValueError:
        print("[FAIL] --region deve ser 'x,y,largura,altura'")
        sys.exit(1)
    regiao = {"left": x, "top": y, "width": w, "height": h}

    url = f"ws://{args.host}:{args.port}/selene"
    if args.token:
        url += f"?token={args.token}"

    try:
        asyncio.run(jogar(url, regiao, args.grid, args.fps))
    except KeyboardInterrupt:
        print("\n🛑 Agente parado.")


if __name__ == "__main__":
    main()
