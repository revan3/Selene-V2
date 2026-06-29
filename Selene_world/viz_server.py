"""
viz_server.py — Servidor de visualização 3D do Selene-World.

Roda o Mundo e TRANSMITE o estado a cada ciclo via WebSocket para o
`selene_world_3d.html` (Three.js), que renderiza em 3D estilo Age of Empires.
Controle de velocidade por VEL (ciclos/segundo). Se a população extingue, o
mundo renasce automaticamente (uma nova "era").

Uso:  ../venv/Scripts/python.exe viz_server.py
      depois abra selene_world_3d.html no navegador.
"""
import asyncio
import json
import sys

import websockets

from orchestrator import Mundo

try:
    sys.stdout.reconfigure(encoding="utf-8")   # console Windows aceita emojis
except Exception:
    pass

HOST, PORTA = "127.0.0.1", 8765
# 1 passo = 1 DIA. 1x = 1 ano a cada 10 min → 1 dia a cada 600/365 ≈ 1.64s.
# 2x..5x dividem o tempo (5x ≈ 2min/ano). Vida de 80 anos: ~13h em 1x.
BASE_SEG = 600.0 / 365
estado = {"intervalo": BASE_SEG}
clientes = set()


def estado_json(mundo, m):
    recursos = [[x, y, r, round(q, 1)]
                for (x, y), cel in mundo.mapa.celulas.items()
                for r, q in cel.items() if q > 0.5]
    bots = [[b.id, b.x, b.y, b.tribo, round(b.energia),
             round(b.crenca_fogo, 2), b.otimizacoes]
            for b in mundo.pop]
    fogos = [[x, y, round(i, 2)] for (x, y), i in mundo.mapa.fogos.items()]
    cavernas = [[x, y, ti] for ti, (x, y) in mundo.mapa.cavernas.items()]
    abrigos = [[x, y] for (x, y) in mundo.mapa.estruturas]
    agua = [[x, y] for (x, y) in mundo.mapa.agua]
    return json.dumps({
        "t": m["t"], "pop": m["pop"], "lingua": round(m["lingua"], 3),
        "ger_max": m["ger_max"], "cpu": round(m["cpu"], 2),
        "ram": round(m["ram"], 2), "abundancia": round(m["abundancia"], 2),
        "fogo": round(m["fogo"], 3), "otim": m["otim"],
        "vida": round(m["vida_media"]), "luz": m["luz"],
        "tribos": mundo.tribo_stats(),
        "w": mundo.mapa.w, "h": mundo.mapa.h,
        "recursos": recursos, "bots": bots, "fogos": fogos,
        "cavernas": cavernas, "abrigos": abrigos, "agua": agua,
    })


async def handler(ws):
    clientes.add(ws)
    try:
        async for msg in ws:                       # cliente manda {"vel": 1..5}
            d = json.loads(msg)
            if "vel" in d:
                v = max(1, min(5, int(d["vel"])))
                estado["intervalo"] = BASE_SEG / v
    except Exception:
        pass
    finally:
        clientes.discard(ws)


async def simular():
    mundo = Mundo()
    dia = 0
    while True:
        m = mundo.passo(dia)
        if not mundo.pop:                      # extinção → nova era
            print("💀 extinção — renascendo o mundo")
            mundo, dia = Mundo(), 0
            continue
        msg = estado_json(mundo, m)
        for ws in list(clientes):
            try:
                await ws.send(msg)
            except Exception:
                clientes.discard(ws)
        dia += 1
        await asyncio.sleep(estado["intervalo"])   # velocidade controlada pelo cliente


async def main():
    async with websockets.serve(handler, HOST, PORTA):
        print(f"🌍 Selene-World 3D em ws://{HOST}:{PORTA}")
        print("   → abra selene_world_3d.html no navegador (Ctrl+C para parar)")
        await simular()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nencerrado.")
