"""
system_resources.py — O PULSO DA MÁQUINA.

Lê o estado REAL do computador (CPU/RAM) e traduz em abundância/escassez do
mundo dos bots. O ecossistema vira um espelho vivo do PC:
  • CPU ociosa + RAM livre → abundância (recursos crescem, festa)
  • CPU/RAM saturada (você joga/renderiza) → escassez (fome, catástrofe)

psutil é opcional: sem ele, cai num pulso SIMULADO (oscilante) — o mundo roda
em qualquer lugar, mas só "sente" o PC de verdade com psutil instalado.
"""
import math
import time

try:
    import psutil
    _PSUTIL = True
    psutil.cpu_percent(interval=None)        # 1ª leitura "esquenta" o medidor
except ImportError:
    _PSUTIL = False


def pulso_do_sistema():
    """Retorna dict com o estado do mundo derivado do PC:
       abundancia (0..1) alta = máquina ociosa → recursos regeneram rápido.
       pressao    (0..1) alta = máquina saturada → escassez/extinção."""
    if _PSUTIL:
        cpu = psutil.cpu_percent(interval=None) / 100.0
        ram = psutil.virtual_memory().percent / 100.0
        fonte = "real"
    else:
        # pulso simulado: respira numa senóide lenta + ruído (sem psutil)
        t = time.time() / 30.0
        base = 0.4 + 0.25 * math.sin(t)
        cpu = ram = min(1.0, max(0.0, base))
        fonte = "simulado"
    pressao = max(cpu, ram)                   # o recurso mais saturado manda
    return {
        "abundancia": 1.0 - pressao,
        "pressao": pressao,
        "cpu": cpu,
        "ram": ram,
        "fonte": fonte,
    }
