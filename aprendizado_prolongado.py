#!/usr/bin/env python3
"""
Aprendizado prolongado da Selene.

- Inicia a Selene com lateralização + ternarização LIGADAS.
- Liga câmera + microfone (deixe um áudio tocando perto do mic).
- Monitora progresso: step, Hz, estado cerebral, RAM, vocabulário e as taxas
  dos dois hemisférios (ESQ=temporal/áudio, DIR=occipital+parietal/vídeo).
- Resiliência: se a Selene cair, reinicia sozinha; se o WebSocket cair, reconecta.
- O gatilho de RAM da própria Selene faz sono profundo + flush quando a RAM lota
  (mantém só os neurônios ativos) — este script só observa e mantém vivo.

Uso:   venv/Scripts/python.exe aprendizado_prolongado.py
Parar: Ctrl+C  (encerra a Selene em seguida)
"""
import asyncio
import json
import os
import subprocess
import time

import websockets

SELENE_BIN = "./target/release/selene_brain.exe"
URI = "ws://127.0.0.1:3030/selene"
LOG = "aprendizado_prolongado.log"
MONITOR_S = 30  # intervalo do relatório de progresso
ENV = {
    **os.environ,
    "SELENE_LATERAL": "1",
    "SELENE_TERNARY": "1",
    "SELENE_HW": "avell",
}


def ts():
    return time.strftime("%H:%M:%S")


def iniciar_selene(logf):
    return subprocess.Popen(
        [SELENE_BIN], stdout=logf, stderr=subprocess.STDOUT, env=ENV
    )


def rate(regions, zona):
    rr = regions.get(zona)
    return rr.get("rate", 0.0) if isinstance(rr, dict) else 0.0


async def ativar_sensores(ws):
    await ws.send(json.dumps({"action": "set_sensor", "sensor": "audio", "ativo": True}))
    await asyncio.sleep(0.3)
    await ws.send(json.dumps({"action": "set_sensor", "sensor": "video", "ativo": True}))


async def sessao(ws):
    await ativar_sensores(ws)
    print(f"[{ts()}] sensores ON — aprendizado em curso (Ctrl+C para parar)")
    ultimo = 0.0
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        except asyncio.TimeoutError:
            continue
        try:
            m = json.loads(raw)
        except Exception:
            continue
        if not isinstance(m, dict) or "regions" not in m:
            continue
        agora = time.time()
        if agora - ultimo < MONITOR_S:
            continue
        ultimo = agora
        reg = m.get("regions", {})
        hw = m.get("hardware", {})
        swap = m.get("swap", {})
        vocab = (swap.get("vocab") or swap.get("conceitos")
                 or swap.get("palavras") or swap.get("vocabulario") or "?")
        print(
            f"[{ts()}] step={m.get('step', 0)} "
            f"{m.get('loop_hz', 0):.0f}Hz {m.get('brain_state', '?')} | "
            f"RAM={hw.get('ram_usage_gb', 0):.1f}GB vocab={vocab} | "
            f"ESQ temporal={rate(reg, 'temporal'):.2f} | "
            f"DIR occ={rate(reg, 'occipital'):.2f} par={rate(reg, 'parietal'):.2f}"
        )


async def main():
    logf = open(LOG, "a", buffering=1, encoding="utf-8", errors="replace")
    proc = None
    try:
        while True:
            if proc is None or proc.poll() is not None:
                print(f"[{ts()}] iniciando Selene (lateral+ternary ON)...")
                proc = iniciar_selene(logf)

            ws = None
            for _ in range(60):
                if proc.poll() is not None:
                    break
                try:
                    ws = await websockets.connect(
                        URI, max_size=None, ping_interval=None
                    )
                    break
                except Exception:
                    await asyncio.sleep(1)

            if ws is None:
                print(f"[{ts()}] não conectou; reiniciando Selene em 5s")
                if proc.poll() is None:
                    proc.terminate()
                proc = None
                await asyncio.sleep(5)
                continue

            print(f"[{ts()}] CONECTADO à Selene")
            try:
                await sessao(ws)
            except Exception as e:
                print(f"[{ts()}] conexão caiu ({type(e).__name__}); reconectando")
                try:
                    await ws.close()
                except Exception:
                    pass
                await asyncio.sleep(3)
    except KeyboardInterrupt:
        print(f"\n[{ts()}] encerrando aprendizado...")
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
        logf.close()


if __name__ == "__main__":
    asyncio.run(main())
