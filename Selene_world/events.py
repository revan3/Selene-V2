"""
events.py — Eventos do ecossistema: pressões EXTERNAS que moldam a evolução,
além da seleção contínua de cada ciclo. Inspirado em extinções em massa reais —
gargalos populacionais são o que mais acelera a evolução na natureza.

  🎉 FESTA       → abundância: todos ganham energia (mais reprodução/exploração)
  💀 CATÁSTROFE  → extinção em massa: os MENOS evoluídos morrem (fitness baixo)
  🪦 SUPERPOP    → Malthus: quando a população lota, escassez mata os fracos
"""
import random

CATASTROFES = ["seca", "praga", "inverno rigoroso", "chuva de meteoros",
               "colapso de recursos", "onda de calor"]
FESTAS = ["aniversário do mundo", "colheita farta", "solstício",
          "descoberta de recursos", "primavera"]


def festa(pop, log):
    """🎉 Abundância — energia extra pra toda a população (favorece reprodução)."""
    if not pop:
        return
    bonus = random.uniform(15.0, 45.0)
    for b in pop:
        b.energia += bonus
    log(0, "FESTA", f"🎉 {random.choice(FESTAS)}: +{bonus:.0f} energia para {len(pop)} bots")


def catastrofe(pop, fitness, severidade, motivo, log):
    """💀 Extinção em massa — mata os de MENOR fitness (os menos evoluídos).
    `severidade` = fração que morre (0.3 = some com os 30% piores). Preserva ≥2."""
    if len(pop) <= 2:
        return pop
    pop = sorted(pop, key=fitness, reverse=True)
    sobrev = max(2, round(len(pop) * (1.0 - severidade)))
    mortos = len(pop) - sobrev
    log(0, "CATASTROFE",
        f"💀 {motivo}: {mortos} menos evoluídos morreram | sobrevivem {sobrev} (os mais aptos)")
    return pop[:sobrev]


def por_superpopulacao(pop, fitness, limiar, log):
    """🪦 Gargalo malthusiano — se a população passou do que o mundo suporta,
    a escassez extermina os fracos. É a 'seleção por superpopulação' pedida."""
    if len(pop) <= limiar:
        return pop
    return catastrofe(pop, fitness, severidade=0.22,
                      motivo="superpopulação (escassez de recursos)", log=log)


LIMIAR_RAM = 0.85   # acima disso, a memória REAL do PC "colapsa" e a vida não cabe


def por_pressao_ram(pop, fitness, ram, log):
    """💥 EXTINÇÃO POR MEMÓRIA — o mundo está atado à RAM REAL do PC. Quando a
    máquina passa de LIMIAR_RAM, falta 'espaço' para a vida e os menos aptos morrem,
    proporcional ao excesso (em 100% de RAM, metade da população se extingue)."""
    if ram < LIMIAR_RAM or len(pop) <= 2:
        return pop
    excesso = (ram - LIMIAR_RAM) / (1.0 - LIMIAR_RAM)        # 0..1
    sev = min(0.5, excesso * 0.5)
    return catastrofe(pop, fitness, sev,
                      f"colapso de memória do sistema (RAM {ram * 100:.0f}%)", log)


def fogo_natural(mapa, log, prob=0.07):
    """⚡ Um raio (ou combustão espontânea) acende fogo numa célula — de preferência
    onde há madeira (combustível). É um FENÔMENO: ninguém 'fez' o fogo. Os bots terão
    de DECIFRÁ-LO sozinhos (aprender que aquece no frio, mas queima de perto)."""
    if random.random() >= prob:
        return
    candidatas = [pos for pos, cel in mapa.celulas.items() if cel.get("madeira", 0) > 5]
    pos = random.choice(candidatas) if candidatas else (
        random.randrange(mapa.w), random.randrange(mapa.h))
    mapa.acender_fogo(pos, random.uniform(0.9, 1.4))
    log(0, "FOGO", f"⚡ raio acende fogo em {pos}")


def evento_aleatorio(pop, fitness, log, prob_festa=0.14, prob_catastrofe=0.05):
    """Sorteia um evento ESPONTÂNEO por ciclo (além das pressões normais).
    Devolve a população (a catástrofe pode encolhê-la)."""
    r = random.random()
    if r < prob_festa:
        festa(pop, log)
    elif r < prob_festa + prob_catastrofe and len(pop) > 4:
        sev = random.uniform(0.2, 0.45)
        pop = catastrofe(pop, fitness, sev, random.choice(CATASTROFES), log)
    return pop
