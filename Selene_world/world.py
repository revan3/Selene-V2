"""
world.py — Loop do Selene-World 2D (survival + emergência de linguagem).

DOIS MODOS:
  • rápido (padrão): roda tudo e imprime a tabela + relatório no fim
  • AO VIVO  (--vel N): mostra o mapa animado em tempo real, N ciclos por segundo
    (1=câmera lenta pra ver eles vivendo, 3, 10, 30…). Ctrl+C interrompe.

Mapa:  f=comida  T=madeira  ^=ferro  .=vazio  @=bot  &=multidão
Uso:   python world.py [n_ciclos] [--vel N] [--render]
"""
import os
import sys
import time
from collections import defaultdict

from orchestrator import DIAS_POR_ANO, Mundo

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
GLIFO = {"comida": "f", "madeira": "T", "ferro": "^"}


def _habilitar_ansi():
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)   # virtual terminal
        except Exception:
            pass


def render(mundo):
    bots = defaultdict(int)
    for b in mundo.pop:
        bots[(b.x, b.y)] += 1
    linhas = []
    for y in range(mundo.mapa.h):
        cs = []
        for x in range(mundo.mapa.w):
            if (x, y) in bots:
                cs.append("@" if bots[(x, y)] == 1 else "&")
            elif (x, y) in mundo.mapa.fogos:
                cs.append("*")                         # 🔥 fogo
            else:
                rec = mundo.mapa.recursos_em((x, y))
                cs.append(GLIFO[max(rec, key=rec.get)] if rec else ".")
        linhas.append(" ".join(cs))
    return "\n".join(linhas)


def painel(mundo, m):
    """Mapa + barra de status para o modo AO VIVO."""
    ab = m["abundancia"]
    clima = "fartura 🌱" if ab > 0.6 else ("FOME 🥵" if ab < 0.35 else "normal")
    cab = (f"🌍 ciclo {m['t']:<4} pop {m['pop']:<3} língua {m['lingua']:.2f}  "
           f"🔥{m['fogo']:.0%}  ger {m['ger_max']:<3} idade~{m['idade_media']:.0f}  |  "
           f"CPU {m['cpu']*100:.0f}% RAM {m['ram']*100:.0f}% → {clima}")
    return "\033[H" + cab + "\n" + "─" * 58 + "\n" + render(mundo) + "\n\033[J"


def main(n_ciclos=60, vel=0):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"mundo_{time.strftime('%Y%m%d_%H%M%S')}.log")
    arq = open(log_path, "w", encoding="utf-8", buffering=1)
    cont = {"FESTA": 0, "CATASTROFE": 0, "FOGO": 0}

    def log(bot_id, evento, msg):
        if evento in cont:
            cont[evento] += 1
        arq.write(f"[{evento}] {msg}\n")

    mundo = Mundo(n_inicial=14, capacidade=55, log=log)

    if vel > 0:                                   # ── MODO AO VIVO ──
        _habilitar_ansi()
        sys.stdout.write("\033[2J")               # limpa a tela 1x
        try:
            for dia in range(n_ciclos * DIAS_POR_ANO):
                m = mundo.passo(dia)
                sys.stdout.write(painel(mundo, m))
                sys.stdout.flush()
                time.sleep(1.0 / vel)
        except KeyboardInterrupt:
            print("\n(interrompido)")
    else:                                         # ── MODO RÁPIDO ──
        print(f"\n🌍 Selene-World — {n_ciclos} anos (1 passo=1 dia) | log: {log_path}\n")
        print(f"{'ano':>3} {'pop':>3} {'vida':>5} {'g_max':>5} "
              f"{'língua':>6} {'fogo':>5} {'otim':>5} {'ram%':>5} {'m/n':>7}")
        print("-" * 66)
        for ano in range(n_ciclos):
            for d in range(DIAS_POR_ANO):
                m = mundo.passo(ano * DIAS_POR_ANO + d)
            print(f"{ano:>3} {m['pop']:>3} {m['vida_media']:>5.0f} {m['ger_max']:>5} "
                  f"{m['lingua']:>6.2f} {m['fogo']:>5.0%} {m['otim']:>5} "
                  f"{m['ram']*100:>5.0f} {m['mortos']:>3}/{m['nascidos']:<3}")

    arq.close()
    _relatorio(mundo, cont, log_path)


def _relatorio(mundo, cont, log_path):
    lp = [x['lingua'] for x in mundo.metricas]
    pops = [x['pop'] for x in mundo.metricas]
    print("\n── RELATÓRIO ──")
    print(f"População   : oscilou entre {min(pops)} e {max(pops)} bots (boom-bust)")
    print(f"Linguagem   : convergência {lp[0]:.2f} → {lp[-1]:.2f}  "
          f"({'📈 emergindo um código comum' if lp[-1] > lp[0] + 0.05 else 'ainda em Babel'})")
    fogo = [x['fogo'] for x in mundo.metricas]
    pico = max(fogo) if fogo else 0
    print(f"Tecnologia  : domínio do fogo {fogo[0]:.0%} → {fogo[-1]:.0%} "
          f"(pico {pico:.0%})  — {'🔥 decifraram o fogo!' if pico > 0.2 else 'ainda no escuro'}")
    print(f"Eventos     : 🎉 {cont['FESTA']} festas | 💀 {cont['CATASTROFE']} catástrofes "
          f"| ⚡ {cont['FOGO']} raios")
    # PONTO PRINCIPAL: a pré-escrita (código) que os bots evoluíram → catálogo p/ Selene
    print("\n🧬 PRÉ-ESCRITA evoluída (código que pode voltar pra Selene):")
    for st in mundo.tribo_stats():
        print(f"   {st['nome']:<6} pop {st['pop']:<3} otimizações {st['otim']:<3} "
              f"vida~{st['vida']:.0f} anos (base 80)")
    cat = mundo.gen.catalogo_uteis()
    if cat:
        print("   📋 otimizações catalogadas:")
        for (tar, tr, vezes, gm, _gmx) in cat:
            print(f"      {tar:<9} {tr[:32]:<32} {vezes}× ganho~{gm}%")
    else:
        print("   (nenhuma otimização adotada ainda — rode mais ciclos)")
    print(f"\n📄 Log: {log_path}\n")


def _arg_vel(argv):
    if "--vel" in argv:
        i = argv.index("--vel")
        if i + 1 < len(argv) and argv[i + 1].isdigit():
            return int(argv[i + 1])
        return 3
    return 0


if __name__ == "__main__":
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    n = int(nums[0]) if nums else 60
    main(n, vel=_arg_vel(sys.argv))
