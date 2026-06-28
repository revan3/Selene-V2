"""
validar_evolucoes.py — VALIDA as otimizações que os bots descobrem na pré-escrita.

Para cada tarefa (espelho de operação da Selene) e cada operador de mutação:
  1. aplica a transformação no código-base;
  2. roda original e otimizado no SANDBOX (safety), REPS vezes, com o mesmo gabarito;
  3. confirma que o resultado é IDÊNTICO (correção) e mede o ganho de latência (mediana).

Assim separamos as otimizações CONFIÁVEIS (corretas + ganho real, ✅) das que a
evolução corretamente REJEITA (quebram o resultado, ❌ — ex. mutação de constante).
O que passar aqui é candidato a portar pra Selene real.

Uso:  ../venv/Scripts/python.exe validar_evolucoes.py
"""
import statistics
import sys

import mutations
from safety import executar_seguro, validar_seguranca
from selene_tasks import TAREFAS

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPS = 5   # repetições por medição → mediana (corta o ruído de spawn/medição única)


def medir(codigo, dados, esp):
    """Roda REPS vezes; retorna (correto_em_todas, latencia_mediana)."""
    lats, correto = [], True
    gab = esp(dados)
    for _ in range(REPS):
        ok, res, lat, _ = executar_seguro(codigo, dados)
        if not (ok and res == gab):
            correto = False
        lats.append(lat)
    return correto, statistics.median(lats)


def main():
    print("\n🔬 VALIDAÇÃO das evoluções sugeridas pelos bots "
          f"(mediana de {REPS} execuções no sandbox)\n")
    print(f"{'tarefa (op da Selene)':<17}{'otimização descoberta':<43}"
          f"{'ok?':<5}{'ganho':>7}")
    print("-" * 72)
    validas = []
    for nome, (base, gerar, esp) in TAREFAS.items():
        dados = gerar(7)
        _, lat_base = medir(base, dados, esp)
        achou = False
        for op in mutations.OPERADORES:
            try:
                prop = op(base)
            except Exception:
                prop = None
            if not prop:
                continue
            novo, desc = prop
            seguro = validar_seguranca(novo)[0]
            correto, lat_otim = medir(novo, dados, esp)
            ganho = (1 - lat_otim / lat_base) * 100 if lat_base else 0.0
            vale = correto and seguro and ganho > 1.0
            marca = "✅" if vale else ("⚠️ " if correto else "❌")
            if vale:
                validas.append((nome, desc, ganho))
                achou = True
            print(f"{nome:<17}{desc[:42]:<43}{marca:<5}{ganho:>6.1f}%")
        if not achou:
            print(f"{nome:<17}{'(nenhuma otimização aplicável)':<43}{'—':<5}")
    print("-" * 72)
    print(f"\n✅ {len(validas)} otimizações VÁLIDAS (corretas + ganho real) — "
          "candidatas a portar pra Selene:")
    for nome, desc, ganho in sorted(validas, key=lambda x: -x[2]):
        print(f"   • [{nome}] {desc}  (+{ganho:.0f}%)")
    print("\n❌/⚠️  = a evolução REJEITA (quebra o resultado ou não acelera) — "
          "prova que o lab só adota o que preserva a operação.\n")


if __name__ == "__main__":
    main()
