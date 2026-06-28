"""
tasks.py — O AMBIENTE do Selene-World: a tarefa que os bots resolvem.

Escolhida de propósito por ter implementações de eficiência MUITO diferente
(loop ingênuo vs comprehension vs truques) → a evolução tem o que otimizar.
`esperado()` é o gabarito: toda mutação é validada contra ele (mutação que
acelera mas QUEBRA o resultado é descartada).
"""
import random

# Genoma de partida: versão INGÊNUA de propósito (loop + if + acumulador).
# A evolução deve descobrir formas mais rápidas de chegar no mesmo resultado.
GENOMA_INICIAL = '''def processar(dados):
    resultado = 0
    for x in dados:
        if x % 2 == 0:
            resultado = resultado + x * x
    return resultado
'''


def gerar_lote(n=4000, seed=None):
    """Lote de entrada (números). `seed` fixa o lote → comparação justa no ciclo."""
    rng = random.Random(seed)
    return [rng.randint(0, 1000) for _ in range(n)]


def esperado(dados):
    """Gabarito de referência: soma dos quadrados dos números pares."""
    return sum(x * x for x in dados if x % 2 == 0)
