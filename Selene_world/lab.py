"""
lab.py — LABORATÓRIO DE EVOLUÇÃO DE CÓDIGO (o ponto principal do Selene-World).

Bots cujo GENOMA é código Python evoluem por auto-reescrita VALIDADA (reusa
safety.py = sandbox + mutations.py = operadores), em tarefas que ESPELHAM operações
reais da Selene (selene_tasks.py). Cada refatoração adotada é catalogada na
genealogia (SQLite). Reprodução = CASAMENTO: 2 bots maduros (idade≥3 + ≥1 evolução)
geram filho que herda o código do mais apto + 1 transformação (recombinação).

No fim imprime o CATÁLOGO: por operação da Selene, quais otimizações ajudaram e
quanto — pronto pra eu portar pro cérebro real.

⚠️ SANDBOX: todo código evoluído roda via safety.executar_seguro (processo isolado,
whitelist de AST, timeout). Loop infinito ou código perigoso NÃO toca seu sistema.

Uso:  ../venv/Scripts/python.exe lab.py [geracoes_por_tarefa]
"""
import random
import sys

import mutations
from genealogia import Genealogia
from safety import executar_seguro, validar_seguranca
from selene_tasks import TAREFAS


class CodeBot:
    _seq = 0

    def __init__(self, codigo, pai_a=None, pai_b=None, geracao=0):
        CodeBot._seq += 1
        self.id = CodeBot._seq
        self.codigo = codigo
        self.pai_a, self.pai_b = pai_a, pai_b
        self.geracao = geracao
        self.idade = 0
        self.latencia = None
        self.n_evolucoes = 0

    def avaliar(self, dados, esp):
        ok, res, lat, _ = executar_seguro(self.codigo, dados)
        self.latencia = lat if (ok and res == esp(dados)) else None
        return self.latencia

    def evoluir(self, dados, esp, gen, tarefa):
        if self.latencia is None:
            return
        prop = mutations.mutar(self.codigo, preferir_otimizacao=self.latencia > 5e-4)
        if not prop:
            return
        novo, desc = prop
        if not validar_seguranca(novo)[0]:
            return
        ok, res, lat, _ = executar_seguro(novo, dados)
        if not (ok and res == esp(dados)):
            return
        if lat < self.latencia:
            ganho = (1 - lat / self.latencia) * 100
            self.codigo, self.latencia = novo, lat
            self.n_evolucoes += 1
            gen.registrar_descoberta(self.id, self.geracao, tarefa, desc, ganho)

    def maduro(self):
        return self.idade >= 3 and self.n_evolucoes >= 1


def casar(a, b, dados, esp):
    base = a if (a.latencia or 9) <= (b.latencia or 9) else b
    cod = base.codigo
    prop = mutations.mutar(cod, preferir_otimizacao=True)
    if prop and validar_seguranca(prop[0])[0]:
        ok, res, _, _ = executar_seguro(prop[0], dados)
        if ok and res == esp(dados):
            cod = prop[0]
    return CodeBot(cod, a.id, b.id, max(a.geracao, b.geracao) + 1)


def rodar_tarefa(nome, genoma, gerar, esp, gen, geracoes, pop_size=8):
    pop = [CodeBot(genoma) for _ in range(pop_size)]
    for b in pop:
        gen.registrar_bot(b.id, None, None, 0, None, b.codigo)
    for g in range(geracoes):
        dados = gerar(g)
        for b in pop:
            b.idade += 1
            b.avaliar(dados, esp)
            b.evoluir(dados, esp, gen, nome)
        pop.sort(key=lambda b: b.latencia if b.latencia else 9)
        pop = pop[:pop_size]
        maduros = [b for b in pop if b.maduro()]
        while len(pop) < pop_size + 4 and len(maduros) >= 2:    # casamentos
            a, b = random.sample(maduros, 2)
            filho = casar(a, b, dados, esp)
            filho.avaliar(dados, esp)
            gen.registrar_bot(filho.id, a.id, b.id, filho.geracao,
                              (filho.latencia or 0) * 1000, filho.codigo)
            pop.append(filho)
            maduros = [x for x in pop if x.maduro()]
    vivos = [b for b in pop if b.latencia]
    campeao = min(vivos, key=lambda b: b.latencia) if vivos else None
    evs = campeao.n_evolucoes if campeao else 0
    print(f"  ✓ {nome:<16} campeão com {evs} otimização(ões) adotada(s)")
    return campeao


def main(geracoes=20):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    gen = Genealogia()
    print(f"\n🧪 Lab de Código — {len(TAREFAS)} tarefas (espelham a Selene), "
          f"{geracoes} gerações cada. Sandbox: safety.py (processo isolado + timeout).\n")
    campeoes = {}
    for nome, (genoma, gerar, esp) in TAREFAS.items():
        campeoes[nome] = rodar_tarefa(nome, genoma, gerar, esp, gen, geracoes)
    _catalogo(gen, campeoes)
    gen.fechar()


def _catalogo(gen, campeoes):
    print(f"\n{'=' * 72}")
    print(f"📋 CATÁLOGO DE OTIMIZAÇÕES ({gen.total_descobertas()} adoções) "
          f"— porte estas pra Selene:")
    print(f"{'-' * 72}")
    print(f"{'operação da Selene':<17}{'otimização descoberta':<40}"
          f"{'×':>4}{'ganho%':>9}")
    for (tar, tr, vezes, gm, _gmx) in gen.catalogo_uteis():
        print(f"{tar:<17}{tr[:39]:<40}{vezes:>4}{gm:>9}")
    print("\n💎 Código campeão de cada operação (já evoluído):")
    for nome, c in campeoes.items():
        if c:
            print(f"\n  ── {nome} ──")
            for ln in c.codigo.strip().splitlines():
                print(f"    {ln}")
    print(f"\n🗄️  Genealogia completa (SQLite): {gen.caminho}\n")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 20
    main(n)
