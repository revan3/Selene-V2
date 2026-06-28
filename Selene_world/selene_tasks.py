"""
selene_tasks.py — Tarefas que ESPELHAM operações reais da Selene.

A graça: as otimizações que o lab descobre nestas tarefas são DIRETAMENTE
portáveis pro cérebro real (loop de neurônios, distância de spikes, decaimento de
neuroquímica). Usamos inteiros nos dados pra o gabarito bater EXATO (sem ruído de
float). Cada tarefa = (genoma_inicial, gerar(seed), esperado(dados)).
"""
import random

# ── 1. FIRING dos neurônios: soma das ativações² acima do limiar ──────────────
#    Espelha: varrer o pool e somar contribuições dos que dispararam.
#    Otimizações possíveis: loop→sum (vetorização) e v**2→v*v.
NEURONIO_FIRE = '''def processar(dados):
    total = 0
    for v in dados:
        if v > 50:
            total = total + v ** 2
    return total
'''


def _gerar_fire(seed):
    rng = random.Random(seed)
    return [rng.randint(0, 100) for _ in range(4000)]


def _esp_fire(dados):
    return sum(v * v for v in dados if v > 50)


# ── 2. DISTÂNCIA entre dois spike patterns (soma das diferenças²) ─────────────
#    Espelha: similaridade entre padrões de spike. Otimização: v**2→v*v (mas só
#    quando a base é simples — aqui (a-b) é composto, então o lab vai DESCARTAR,
#    mostrando que a otimização NÃO vale em todo contexto: catálogo honesto).
SPIKE_DIST = '''def processar(dados):
    a, b = dados
    d = 0
    for i in range(len(a)):
        d = d + (a[i] - b[i]) ** 2
    return d
'''


def _gerar_spike(seed):
    rng = random.Random(seed)
    n = 2500
    return ([rng.randint(0, 30) for _ in range(n)],
            [rng.randint(0, 30) for _ in range(n)])


def _esp_spike(dados):
    a, b = dados
    return sum((a[i] - b[i]) ** 2 for i in range(len(a)))


# ── 3. DECAIMENTO de neuroquímica (com operações redundantes) ────────────────
#    Espelha: aplicar decay a cada modulador. Otimização: remover *1 e +0 (peephole).
DECAY = '''def processar(dados):
    out = []
    for m in dados:
        out.append(m * 2 * 1 + 0)
    return out
'''


def _gerar_decay(seed):
    rng = random.Random(seed)
    return [rng.randint(0, 200) for _ in range(4000)]


def _esp_decay(dados):
    return [m * 2 for m in dados]


# nome → (genoma_inicial, gerar, esperado)
TAREFAS = {
    "neuronio_fire": (NEURONIO_FIRE, _gerar_fire, _esp_fire),
    "spike_dist":    (SPIKE_DIST, _gerar_spike, _esp_spike),
    "decay_neuroq":  (DECAY, _gerar_decay, _esp_decay),
}
