"""
fisica.py — DINÂMICA DO MUNDO emergente (sem receitas). Spec: atualização.txt.

Os agentes NÃO sabem "fazer um machado". Sabem que cada material tem PROPRIEDADES
(dureza, fusão, condutividade, reatividade, plasticidade) e testam AÇÕES genéricas
(aquecer, forjar, impactar, ligar). A física calcula o resultado. Quando um objeto
ganha uma CAPACIDADE que seus ingredientes não tinham, o motor rotula INVENÇÃO.

3 mundos mudam o que é possível com as MESMAS regras: na Lua (vácuo, sem O₂) não
há combustão livre; em Marte a oxidação é fraca. Física diferente, não receitas.

Puro e testável (sem dependências).  Roda:  python fisica.py   (demo de emergência)
"""
import random

# ===== 1. CONSTANTES GLOBAIS DOS 3 MUNDOS =====
# oxigenio governa combustão; retencao_calor governa quanto a energia aplicada "rende".
MUNDOS = {
    "Terra": dict(gravidade=9.81, pressao=1.000, radiacao=0.1, retencao_calor=1.0, oxigenio=1.0),
    "Marte": dict(gravidade=3.71, pressao=0.006, radiacao=0.7, retencao_calor=0.3, oxigenio=0.0),
    "Lua":   dict(gravidade=1.62, pressao=0.000, radiacao=1.0, retencao_calor=0.0, oxigenio=0.0),
}

# ===== 2. MATRIZ DE MATERIAIS (5 propriedades + ignição) =====
# fusao/ignicao em °C (None = não funde / não inflama). condut e plast em 0..1.
MATERIAIS = {
    "celulose":  dict(tipo="vegetal",  dureza=1.0, condut=0.0, fusao=None, ignicao=300,  react=0.4, plast=0.3),
    "comida":    dict(tipo="organico", dureza=0.5, condut=0.0, fusao=None, ignicao=250,  react=0.5, plast=0.6),
    "agua":      dict(tipo="liquido",  dureza=0.0, condut=0.1, fusao=0,    ignicao=None, react=0.2, plast=1.0),
    "pedra":     dict(tipo="rocha",    dureza=6.0, condut=0.1, fusao=1200, ignicao=None, react=0.0, plast=0.00),
    "obsidiana": dict(tipo="rocha",    dureza=5.5, condut=0.1, fusao=1100, ignicao=None, react=0.0, plast=0.05),
    "cobre":     dict(tipo="metal",    dureza=3.0, condut=0.9, fusao=1085, ignicao=None, react=0.7, plast=0.8),
    "ferro":     dict(tipo="metal",    dureza=4.5, condut=0.7, fusao=1538, ignicao=None, react=0.6, plast=0.6),
    "silicio":   dict(tipo="mineral",  dureza=7.0, condut=0.5, fusao=1414, ignicao=None, react=0.5, plast=0.1),
}


# ===== 3. OBJETO: um pedaço de matéria com estado, forma e capacidades =====
class Objeto:
    def __init__(self, material, estado="solido", forma="bruto", caps=None, rotulo=None, props=None):
        self.material = material
        self.estado = estado            # solido | liquido | fogo
        self.forma = forma              # bruto | fino | oco | po
        self.props = props or dict(MATERIAIS[material])   # ligas carregam props próprias
        self.caps = set(caps) if caps else capacidades_de(self.props, forma)
        self.rotulo = rotulo or material

    def __repr__(self):
        c = ",".join(sorted(self.caps)) or "—"
        return f"<{self.rotulo} [{self.estado}/{self.forma}] caps:{c}>"


def capacidades_de(props, forma):
    """Capacidades EMERGEM das propriedades + forma — não são receitas fixas."""
    caps = set()
    if props["dureza"] >= 4.0 and forma == "fino":
        caps.add("cortar")          # duro + afiado → corta
    if props["condut"] >= 0.6 and forma == "fino":
        caps.add("transmitir")      # condutor + fio → conduz energia
    if forma == "oco":
        caps.add("conter")          # forma oca → recipiente
    return caps


# ===== 4. AÇÕES GENÉRICAS (a física decide o resultado) =====
def _rende(energia, mundo):
    """Quanto da energia aplicada de fato aquece (retenção de calor do mundo)."""
    return energia * (0.4 + 0.6 * MUNDOS[mundo]["retencao_calor"])


def aquecer(obj, energia, mundo):
    """Inflama (→fogo), funde (→líquido) ou nada. Combustão exige O₂ do mundo."""
    p, ef = obj.props, _rende(energia, mundo)
    if p["ignicao"] and ef >= p["ignicao"] and MUNDOS[mundo]["oxigenio"] > 0.3:
        return Objeto(obj.material, estado="fogo", caps={"aquecer"}, rotulo="fogo")
    if p["fusao"] is not None and ef >= p["fusao"]:
        return Objeto(obj.material, estado="liquido", forma="bruto",
                      rotulo=obj.material + " fundido", props=p)
    return obj                       # energia insuficiente: sem mudança


def forjar(obj_liquido, forma):
    """Solidifica um líquido numa FORMA (fino/oco). É aqui que o fio/lâmina nasce."""
    if obj_liquido.estado != "liquido":
        return obj_liquido
    base = obj_liquido.material if obj_liquido.material in MATERIAIS else obj_liquido.rotulo
    return Objeto(base, estado="solido", forma=forma,
                  rotulo=f"{base} ({forma})", props=obj_liquido.props)


def ligar(a_liq, b_liq):
    """Dois metais líquidos → LIGA: propriedades médias + bônus de dureza (sinergia)."""
    if a_liq.estado != "liquido" or b_liq.estado != "liquido":
        return None
    pa, pb = a_liq.props, b_liq.props
    mix = {k: (pa[k] + pb[k]) / 2 for k in ("dureza", "condut", "react", "plast")
           if isinstance(pa[k], (int, float)) and isinstance(pb[k], (int, float))}
    mix["dureza"] *= 1.25                                  # liga é mais dura que a média
    mix["fusao"] = max(pa["fusao"] or 0, pb["fusao"] or 0)
    mix["ignicao"] = None
    mix["tipo"] = "liga"
    nome = f"liga({a_liq.material}+{b_liq.material})"
    return Objeto(nome if nome in MATERIAIS else a_liq.material, estado="liquido",
                  rotulo=nome, props=mix)


def impactar(ferramenta, alvo):
    """Ferramenta que CORTA (ou mais dura) molda o alvo: vira 'fino' (lasca) ou 'po'."""
    corta = "cortar" in ferramenta.caps or ferramenta.props["dureza"] > alvo.props["dureza"] + 1.5
    if not corta:
        return alvo                  # bate e não faz nada
    forma = "fino" if alvo.props["plast"] >= 0.2 else "po"
    return Objeto(alvo.material, estado=alvo.estado, forma=forma,
                  rotulo=f"{alvo.material} ({forma})", props=alvo.props)


def acidente_laboratorio(obj):
    """5%: emergência não-mapeada — o objeto ganha uma capacidade inesperada."""
    if random.random() >= 0.05:
        return None
    nova = random.choice(["luminescente", "magnetico", "isolante", "catalisador"])
    obj.caps.add(nova)
    return nova


# ===== 5. DETECÇÃO DE INVENÇÃO =====
def detectar_invencao(resultado, ingredientes):
    """Invenção = capacidade no resultado que NENHUM ingrediente tinha sozinho."""
    antes = set()
    for ing in ingredientes:
        antes |= ing.caps
    novas = resultado.caps - antes
    return novas or None


# ===== 6. DEMO: a tecnologia EMERGE sem receitas =====
def _demo():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    random.seed(7)
    log = []

    def registra(res, ingr):
        inv = detectar_invencao(res, ingr)
        if inv:
            log.append((res.rotulo, inv))
        return res

    print("🌍 DINÂMICA DO MUNDO — tecnologia emergente (sem receitas)\n")

    # 1) celulose + calor na TERRA → FOGO (há O₂)
    celulose = Objeto("celulose")
    fogo = registra(aquecer(celulose, 600, "Terra"), [celulose])
    print(f"  Terra:  aquecer(celulose, 600°)  → {fogo}")

    # 2) o MESMO na LUA → NÃO pega fogo (vácuo, sem O₂): física diferente
    fogo_lua = aquecer(Objeto("celulose"), 600, "Lua")
    print(f"  Lua:    aquecer(celulose, 600°)  → {fogo_lua}   (sem O₂: não inflama)")

    # 3) com fogo dá pra fundir cobre → forjar em FIO (transmite energia: INVENÇÃO)
    cobre_liq = aquecer(Objeto("cobre"), 1200, "Terra")
    fio = registra(forjar(cobre_liq, "fino"), [Objeto("cobre")])
    print(f"\n  fundir+forjar(cobre, fino)      → {fio}")

    # 4) liga cobre+ferro líquidos → mais dura; forjar fino → LÂMINA que corta
    liga = ligar(aquecer(Objeto("cobre"), 1200, "Terra"),
                 aquecer(Objeto("ferro"), 1600, "Terra"))
    lamina = registra(forjar(liga, "fino"), [Objeto("cobre"), Objeto("ferro")])
    print(f"  ligar(cobre+ferro)+forjar       → {lamina}")

    # 5) usar a lâmina (corta) pra lascar obsidiana → ponta afiada
    obs = registra(impactar(lamina, Objeto("obsidiana")), [lamina, Objeto("obsidiana")])
    print(f"  impactar(lâmina, obsidiana)     → {obs}")

    # 6) acidente de laboratório no silício
    si = Objeto("silicio", forma="fino")
    ac = acidente_laboratorio(si)
    print(f"\n  acidente_lab(silício)           → {('🎲 ' + ac) if ac else 'sem evento'}  {si}")

    print(f"\n💡 INVENÇÕES emergidas ({len(log)}):")
    for rotulo, novas in log:
        print(f"   • {rotulo:<22} ganhou capacidade(s): {', '.join(sorted(novas))}")


if __name__ == "__main__":
    _demo()
