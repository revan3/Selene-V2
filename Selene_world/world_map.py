"""
world_map.py — O grid 2D do Selene-World (estilo Age of Empires).

Células guardam recursos (comida, madeira, ferro) que se esgotam quando colhidos
e REGENERAM com o tempo. A taxa de regeneração é modulada pelo PULSO DO SISTEMA:
PC ocioso → recursos voltam rápido; PC saturado → mundo entra em escassez.
"""
import random

from system_resources import pulso_do_sistema

RECURSOS = ["comida", "madeira", "ferro"]
GLIFO = {"comida": "🌾", "madeira": "🌲", "ferro": "⛰", "fogo": "🔥", "vazio": "·"}


class Mapa:
    def __init__(self, w=24, h=12, densidade=0.36, seed=0):
        self.w, self.h = w, h
        rng = random.Random(seed)
        self.celulas = {(x, y): {} for y in range(h) for x in range(w)}
        # espalha "jazidas"; comida domina (a base da sobrevivência no mapa grande)
        sorteio = ["comida", "comida", "comida", "madeira", "ferro"]
        for _ in range(int(w * h * densidade)):
            pos = (rng.randrange(w), rng.randrange(h))
            r = rng.choice(sorteio)
            self.celulas[pos][r] = rng.randint(30, 80)
        # capacidade máxima de cada jazida (alvo do respawn)
        self.cap = {pos: dict(rec) for pos, rec in self.celulas.items()}
        self.fogos = {}                            # (x,y) -> intensidade (fenômeno transitório)
        self.cavernas = {}                         # tribo_id -> (x,y): abrigo + armazém
        self.ultimo_pulso = pulso_do_sistema()

    # ── FOGO: um fenômeno natural que os bots têm de decifrar sozinhos ────────
    def acender_fogo(self, pos, intensidade=1.0):
        self.fogos[pos] = max(self.fogos.get(pos, 0.0), intensidade)

    def decair_fogo(self, taxa=0.12):
        """O fogo se apaga sozinho — ninguém o mantém aceso (ainda)."""
        for pos in list(self.fogos):
            self.fogos[pos] -= taxa
            if self.fogos[pos] <= 0.05:
                del self.fogos[pos]

    def calor_em(self, x, y):
        """Calor sentido numa célula = soma dos fogos próximos, caindo com a distância.
        Em cima do fogo queima; a 1 célula aquece; a 3+ não chega."""
        total = 0.0
        for (fx, fy), inten in self.fogos.items():
            d = abs(fx - x) + abs(fy - y)
            total += inten * (1.0 if d == 0 else 0.5 if d == 1 else 0.2 if d == 2 else 0.0)
        return total

    def fogo_proximo(self, x, y, raio):
        """Posição do fogo mais perto dentro do raio (ou None)."""
        melhor, best_d = None, raio + 1
        for pos in self.fogos:
            d = abs(pos[0] - x) + abs(pos[1] - y)
            if d < best_d:
                melhor, best_d = pos, d
        return melhor

    def regenerar(self, escala=1.0, atualizar_pulso=True):
        """Recursos voltam conforme a abundância REAL do sistema. `escala` divide a
        taxa p/ o passo diário (1/365); `atualizar_pulso` evita ler o PC todo dia."""
        if atualizar_pulso:
            self.ultimo_pulso = pulso_do_sistema()
        taxa = (0.3 + self.ultimo_pulso["abundancia"] * 2.5) * escala
        for pos, base in self.cap.items():
            for r, qmax in base.items():
                if r == "ferro":              # METAIS são FINITOS (não regeneram, tipo
                    continue                  # AoE) — só comida e madeira crescem
                atual = self.celulas[pos].get(r, 0)
                if atual < qmax:
                    self.celulas[pos][r] = min(qmax, atual + taxa)
        return self.ultimo_pulso

    def coletar(self, pos, recurso, quanto):
        cel = self.celulas.get(pos, {})
        pego = min(cel.get(recurso, 0), quanto)
        if pego > 0:
            cel[recurso] -= pego
        return pego

    def recursos_em(self, pos):
        return {r: q for r, q in self.celulas.get(pos, {}).items() if q > 0.5}

    def dentro(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

    def glifo_celula(self, pos):
        if pos in self.fogos:
            return GLIFO["fogo"]               # o fogo domina a célula visualmente
        rec = self.recursos_em(pos)
        if not rec:
            return GLIFO["vazio"]
        return GLIFO[max(rec, key=rec.get)]    # o recurso dominante na célula
