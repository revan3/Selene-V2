"""
safety.py — As 3 travas de segurança do Selene-World.

Os bots REESCREVEM e EXECUTAM o próprio código. Sem estas travas, um bot pode
travar a máquina (loop infinito) ou rodar algo perigoso. Toda execução de código
evoluído passa por aqui ANTES de rodar.

  Trava 1: sintaxe        → ast.parse()           (código é Python válido?)
  Trava 2: segurança      → whitelist no AST       (sem import/eval/open/...)
  Trava 3: execução       → processo + timeout     (mata loop infinito de verdade)
"""
import ast
import builtins
import multiprocessing as mp
import queue
import time

# ── Trava 2: o que o código de um bot NUNCA pode conter ──────────────────────
NOMES_PROIBIDOS = {
    "eval", "exec", "compile", "open", "__import__", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
}

# Builtins LIBERADOS p/ o código do bot (processamento puro, sem I/O/sistema)
BUILTINS_OK = [
    "range", "len", "sum", "min", "max", "abs", "sorted", "map", "filter",
    "list", "dict", "set", "tuple", "int", "float", "str", "bool",
    "enumerate", "zip", "round", "pow", "all", "any", "reversed",
]


def validar_sintaxe(codigo: str):
    """Trava 1 — `ast.parse`. Retorna (ok: bool, msg: str)."""
    try:
        ast.parse(codigo)
        return True, "sintaxe ok"
    except SyntaxError as e:
        return False, f"SINTAXE inválida (linha {e.lineno}): {e.msg}"


def validar_seguranca(codigo: str):
    """Trava 2 — percorre o AST e barra import/dunder/chamada perigosa.
    Sintaxe válida NÃO é segurança: `import os` compila perfeitamente."""
    ok, msg = validar_sintaxe(codigo)
    if not ok:
        return False, msg
    for no in ast.walk(ast.parse(codigo)):
        if isinstance(no, (ast.Import, ast.ImportFrom)):
            return False, "PROIBIDO: import (bots não importam módulos)"
        if isinstance(no, ast.Name) and no.id in NOMES_PROIBIDOS:
            return False, f"PROIBIDO: uso de '{no.id}'"
        if isinstance(no, ast.Attribute) and no.attr.startswith("__"):
            return False, f"PROIBIDO: acesso dunder '.{no.attr}'"
    return True, "seguro"


def _builtins_seguros():
    """Escopo de builtins restrito — o bot só enxerga funções de processamento."""
    return {n: getattr(builtins, n) for n in BUILTINS_OK if hasattr(builtins, n)}


def _alvo(codigo, dados, q):
    """Executado num PROCESSO separado — define `processar` e roda em `dados`."""
    try:
        escopo = {"__builtins__": _builtins_seguros()}
        exec(codigo, escopo)                       # nosec — escopo restrito + validado
        fn = escopo.get("processar")
        if fn is None:
            q.put(("erro", "função 'processar(dados)' não foi definida"))
            return
        resultado = fn(dados)                  # warmup + resultado de referência
        # Mede o MÍNIMO de 7 execuções (prática de benchmark): a execução é rápida e
        # 1 disparo é dominado por ruído de GC/scheduling — o mínimo é o sinal limpo.
        melhor = float("inf")
        for _ in range(7):
            t0 = time.perf_counter()
            fn(dados)
            melhor = min(melhor, time.perf_counter() - t0)
        q.put(("ok", resultado, melhor))
    except Exception as e:
        q.put(("erro", f"runtime {type(e).__name__}: {e}"))


def executar_seguro(codigo: str, dados, timeout: float = 2.0):
    """Trava 3 — valida e executa `processar(dados)` num processo com timeout.
    Retorna (ok: bool, resultado, latencia_s: float, msg: str).
    Um loop infinito é MORTO ao estourar o timeout (não trava o note)."""
    ok, msg = validar_seguranca(codigo)
    if not ok:
        return False, None, 0.0, msg
    q = mp.Queue()
    p = mp.Process(target=_alvo, args=(codigo, dados, q), daemon=True)
    p.start()
    # Lemos a fila COM timeout (em vez de join) — isso DRENA o resultado e libera o
    # filho na hora. Sem isso, um resultado grande enche o pipe e o filho TRAVA no
    # put (deadlock) → falso timeout. Um loop infinito não faz put → get estoura.
    try:
        r = q.get(timeout=timeout)
    except queue.Empty:
        r = None
    if p.is_alive():
        p.terminate()
    p.join()
    if r is None:
        return False, None, timeout, "TIMEOUT — possível loop infinito (bot descartado)"
    if r[0] == "ok":
        return True, r[1], r[2], "ok"
    return False, None, 0.0, r[1]
