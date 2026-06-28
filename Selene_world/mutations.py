"""
mutations.py — Operadores de AUTO-REESCRITA (mutação local, sem LLM).

Cada operador recebe o código-fonte de `processar` e devolve uma VARIANTE
(novo_codigo, descrição) ou None se não se aplica. A SELEÇÃO por fitness (no
orchestrator) decide quais variantes sobrevivem — é aí que mora a evolução.
Mutações são ESTRUTURADAS (preservam validade sintática) em vez de cegas: muta-
ção cega de texto quase sempre quebra. Toda variante ainda passa pelo safety.py.

⚠️ Anti-loop-infinito: nenhum operador AUMENTA a contagem de loops (for/while),
então a evolução não consegue introduzir um loop sem fim (risco #1 do blueprint).
"""
import ast
import random


def _conta_loops(arvore):
    return sum(isinstance(n, (ast.For, ast.While)) for n in ast.walk(arvore))


def _finalizar(arvore_original_src, arvore_nova, descricao):
    """Garante que a variante não introduziu loops e devolve (codigo, descricao)."""
    if _conta_loops(arvore_nova) > _conta_loops(ast.parse(arvore_original_src)):
        return None  # recusa: poderia ser loop infinito
    ast.fix_missing_locations(arvore_nova)
    return ast.unparse(arvore_nova), descricao


def mut_constante(codigo: str):
    """Perturba um literal numérico (exploração → gera DERIVA genética)."""
    arvore = ast.parse(codigo)
    consts = [n for n in ast.walk(arvore)
              if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
              and not isinstance(n.value, bool)]
    if not consts:
        return None
    alvo = random.choice(consts)
    antigo = alvo.value
    alvo.value = antigo + random.choice([-1, 1])
    return _finalizar(codigo, arvore, f"constante {antigo}→{alvo.value}")


class _LoopParaSum(ast.NodeTransformer):
    """Detecta `acc=0; for x in it: [if cond:] acc = acc + expr; return acc`
    e reescreve como `return sum(expr for x in it [if cond])`.
    Essa é a OTIMIZAÇÃO-CHAVE: vetoriza o loop acumulador (o truque que a
    evolução 'descobre' — e que talvez valha portar pra Selene real)."""
    def __init__(self):
        self.aplicou = None

    def visit_FunctionDef(self, node):
        corpo = node.body
        if len(corpo) < 3:
            return node
        a0, laco, ret = corpo[0], corpo[1], corpo[-1]
        if not (isinstance(a0, ast.Assign) and isinstance(laco, ast.For)
                and isinstance(ret, ast.Return)):
            return node
        if not (isinstance(a0.targets[0], ast.Name)):
            return node
        passo, cond = laco.body, None
        if len(passo) == 1 and isinstance(passo[0], ast.If):
            cond, passo = passo[0].test, passo[0].body
        if len(passo) != 1:
            return node
        st = passo[0]
        if isinstance(st, ast.AugAssign) and isinstance(st.op, ast.Add):
            expr = st.value
        elif (isinstance(st, ast.Assign) and isinstance(st.value, ast.BinOp)
              and isinstance(st.value.op, ast.Add)):
            expr = st.value.right                         # acc = acc + expr
        else:
            return node
        gen = ast.GeneratorExp(
            elt=expr,
            generators=[ast.comprehension(
                target=laco.target, iter=laco.iter,
                ifs=[cond] if cond else [], is_async=0)])
        node.body = [ast.Return(ast.Call(ast.Name("sum", ast.Load()), [gen], []))]
        self.aplicou = "loop acumulador → sum(comprehension)  [vetorização]"
        return node


def mut_loop_para_sum(codigo: str):
    arvore = ast.parse(codigo)
    t = _LoopParaSum()
    t.visit(arvore)
    if not t.aplicou:
        return None
    return _finalizar(codigo, arvore, t.aplicou)


def _neutro(n, v):
    return (isinstance(n, ast.Constant) and not isinstance(n.value, bool)
            and n.value == v)


class _PowMult(ast.NodeTransformer):
    """x ** 2 → x * x  (multiplicação é mais rápida que potência em Python).
    Só quando a base é um NOME simples — senão duplicar recomputaria a subexpressão
    (e ficaria mais LENTO; o lab descartaria, mostrando que a otimização é contextual)."""
    def __init__(self):
        self.aplicou = False

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if (isinstance(node.op, ast.Pow) and isinstance(node.left, ast.Name)
                and _neutro(node.right, 2)):
            self.aplicou = True
            return ast.BinOp(left=node.left, op=ast.Mult(),
                             right=ast.Name(node.left.id, ast.Load()))
        return node


def mut_pow_para_mult(codigo: str):
    arvore = ast.parse(codigo)
    t = _PowMult()
    t.visit(arvore)
    if not t.aplicou:
        return None
    return _finalizar(codigo, arvore, "x**2 → x*x (potência por multiplicação)")


class _Simplificar(ast.NodeTransformer):
    """Remove operações NEUTRAS: x*1→x, 1*x→x, x+0→x, 0+x→x (peephole)."""
    def __init__(self):
        self.aplicou = False

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Mult):
            if _neutro(node.right, 1):
                self.aplicou = True
                return node.left
            if _neutro(node.left, 1):
                self.aplicou = True
                return node.right
        if isinstance(node.op, ast.Add):
            if _neutro(node.right, 0):
                self.aplicou = True
                return node.left
            if _neutro(node.left, 0):
                self.aplicou = True
                return node.right
        return node


def mut_simplificar(codigo: str):
    arvore = ast.parse(codigo)
    t = _Simplificar()
    t.visit(arvore)
    if not t.aplicou:
        return None
    return _finalizar(codigo, arvore, "remove operação neutra (*1, +0)")


class _AppendParaListComp(ast.NodeTransformer):
    """out=[]; for x in it: [if c:] out.append(expr); return out
    → return [expr for x in it [if c]].  Comprehension é mais rápida que append em
    loop — otimização que a tribo do 'decay' (construção de lista) consegue descobrir."""
    def __init__(self):
        self.aplicou = None

    def visit_FunctionDef(self, node):
        corpo = node.body
        if len(corpo) < 3:
            return node
        a0, laco, ret = corpo[0], corpo[1], corpo[-1]
        if not (isinstance(a0, ast.Assign) and isinstance(laco, ast.For)
                and isinstance(ret, ast.Return)):
            return node
        if not (isinstance(a0.targets[0], ast.Name) and isinstance(a0.value, ast.List)
                and not a0.value.elts):
            return node                                   # precisa ser  out = []
        nome = a0.targets[0].id
        passo, cond = laco.body, None
        if len(passo) == 1 and isinstance(passo[0], ast.If):
            cond, passo = passo[0].test, passo[0].body
        if len(passo) != 1 or not isinstance(passo[0], ast.Expr):
            return node
        call = passo[0].value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "append" and len(call.args) == 1
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == nome):
            return node                                   # precisa ser  out.append(expr)
        if not (isinstance(ret.value, ast.Name) and ret.value.id == nome):
            return node                                   # precisa ser  return out
        comp = ast.ListComp(elt=call.args[0], generators=[ast.comprehension(
            target=laco.target, iter=laco.iter, ifs=[cond] if cond else [], is_async=0)])
        node.body = [ast.Return(comp)]
        self.aplicou = "loop append → list comprehension  [vetorização]"
        return node


def mut_append_para_listcomp(codigo: str):
    arvore = ast.parse(codigo)
    t = _AppendParaListComp()
    t.visit(arvore)
    if not t.aplicou:
        return None
    return _finalizar(codigo, arvore, t.aplicou)


# Otimizações ESTRUTURAIS primeiro; mut_constante (exploração) por último.
OPERADORES = [mut_loop_para_sum, mut_append_para_listcomp, mut_pow_para_mult,
              mut_simplificar, mut_constante]


def mutar(codigo: str, preferir_otimizacao: bool):
    """Aplica UMA mutação. `preferir_otimizacao` vem da auto-avaliação do bot:
    se a latência está alta, tenta primeiro a transformação que vetoriza o loop;
    senão, embaralha (exploração)."""
    ordem = list(OPERADORES)
    if not preferir_otimizacao:
        random.shuffle(ordem)
    for op in ordem:
        try:
            r = op(codigo)
        except Exception:
            r = None
        if r is not None:
            return r
    return None
