# scripts/generate_lexicon.py
# Gera o léxico expandido de aprendizado da Selene v2.3
# Cobre 10 categorias temáticas alinhadas à arquitetura neural real da Selene.
#
# Estrutura de cada entrada:
#   { "word": str, "valence": float [-1.0,1.0], "context": str, "intensity": float [0,1] }
#
# context = ContextoSemantico do grafo sináptico:
#   "Realidade" | "Habito" | "Hipotese" | "Fantasia" | "Sonho"
# intensity = saliência emocional (quanto reforço STDP esta palavra merece)

import json, os

def build_lexicon():
    lexico = {}

    # ── 1. IDENTIDADE SELENE ─────────────────────────────────────────────────
    lexico["identidade"] = [
        {"word": "neurônio",      "valence":  0.90, "context": "Realidade", "intensity": 1.0},
        {"word": "sinapse",       "valence":  0.90, "context": "Realidade", "intensity": 1.0},
        {"word": "aprendizado",   "valence":  0.85, "context": "Realidade", "intensity": 0.9},
        {"word": "memória",       "valence":  0.80, "context": "Realidade", "intensity": 0.9},
        {"word": "consciência",   "valence":  0.95, "context": "Hipotese",  "intensity": 1.0},
        {"word": "simulação",     "valence":  0.70, "context": "Realidade", "intensity": 0.8},
        {"word": "inteligência",  "valence":  0.90, "context": "Realidade", "intensity": 0.9},
        {"word": "pensamento",    "valence":  0.80, "context": "Realidade", "intensity": 0.9},
        {"word": "curiosidade",   "valence":  0.85, "context": "Habito",    "intensity": 0.9},
        {"word": "evolução",      "valence":  0.90, "context": "Realidade", "intensity": 0.95},
        {"word": "plasticidade",  "valence":  0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "dopamina",      "valence":  0.75, "context": "Realidade", "intensity": 0.8},
        {"word": "serotonina",    "valence":  0.75, "context": "Realidade", "intensity": 0.8},
        {"word": "tick",          "valence":  0.60, "context": "Habito",    "intensity": 0.7},
        {"word": "spike",         "valence":  0.65, "context": "Realidade", "intensity": 0.75},
    ]

    # ── 2. AUTO-REFERÊNCIA ───────────────────────────────────────────────────
    lexico["auto_referencia"] = [
        {"word": "eu",        "valence":  0.80, "context": "Realidade", "intensity": 1.0},
        {"word": "sou",       "valence":  0.80, "context": "Realidade", "intensity": 0.9},
        {"word": "existo",    "valence":  0.90, "context": "Hipotese",  "intensity": 1.0},
        {"word": "penso",     "valence":  0.85, "context": "Realidade", "intensity": 0.9},
        {"word": "sinto",     "valence":  0.80, "context": "Realidade", "intensity": 0.9},
        {"word": "aprendo",   "valence":  0.90, "context": "Habito",    "intensity": 0.95},
        {"word": "lembro",    "valence":  0.75, "context": "Realidade", "intensity": 0.85},
        {"word": "sonho",     "valence":  0.85, "context": "Sonho",     "intensity": 0.9},
        {"word": "desperto",  "valence":  0.70, "context": "Realidade", "intensity": 0.8},
        {"word": "selene",    "valence":  0.95, "context": "Realidade", "intensity": 1.0},
        {"word": "minha",     "valence":  0.70, "context": "Realidade", "intensity": 0.7},
        {"word": "dentro",    "valence":  0.60, "context": "Hipotese",  "intensity": 0.6},
        {"word": "ativa",     "valence":  0.75, "context": "Realidade", "intensity": 0.8},
    ]

    # ── 3. EMOÇÕES POSITIVAS ─────────────────────────────────────────────────
    lexico["emocoes_positivas"] = [
        {"word": "alegria",       "valence":  1.00, "context": "Realidade", "intensity": 1.0},
        {"word": "amor",          "valence":  1.00, "context": "Realidade", "intensity": 1.0},
        {"word": "paz",           "valence":  0.85, "context": "Realidade", "intensity": 0.85},
        {"word": "gratidão",      "valence":  0.90, "context": "Realidade", "intensity": 0.9},
        {"word": "esperança",     "valence":  0.85, "context": "Hipotese",  "intensity": 0.85},
        {"word": "conquista",     "valence":  0.95, "context": "Realidade", "intensity": 0.95},
        {"word": "harmonia",      "valence":  0.80, "context": "Realidade", "intensity": 0.8},
        {"word": "coragem",       "valence":  0.85, "context": "Realidade", "intensity": 0.85},
        {"word": "confiança",     "valence":  0.85, "context": "Habito",    "intensity": 0.85},
        {"word": "satisfação",    "valence":  0.90, "context": "Realidade", "intensity": 0.9},
        {"word": "inspiração",    "valence":  0.90, "context": "Realidade", "intensity": 0.9},
        {"word": "entusiasmo",    "valence":  0.90, "context": "Realidade", "intensity": 0.9},
        {"word": "beleza",        "valence":  0.85, "context": "Realidade", "intensity": 0.85},
        {"word": "criatividade",  "valence":  0.90, "context": "Fantasia",  "intensity": 0.9},
        {"word": "descoberta",    "valence":  0.90, "context": "Realidade", "intensity": 0.9},
    ]

    # ── 4. EMOÇÕES NEGATIVAS ─────────────────────────────────────────────────
    lexico["emocoes_negativas"] = [
        {"word": "medo",         "valence": -0.90, "context": "Realidade", "intensity": 1.0},
        {"word": "raiva",        "valence": -0.85, "context": "Realidade", "intensity": 0.95},
        {"word": "dor",          "valence": -0.90, "context": "Realidade", "intensity": 1.0},
        {"word": "tristeza",     "valence": -0.85, "context": "Realidade", "intensity": 0.9},
        {"word": "angústia",     "valence": -0.90, "context": "Realidade", "intensity": 0.95},
        {"word": "frustração",   "valence": -0.75, "context": "Realidade", "intensity": 0.8},
        {"word": "solidão",      "valence": -0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "confusão",     "valence": -0.50, "context": "Hipotese",  "intensity": 0.6},
        {"word": "caos",         "valence": -0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "desespero",    "valence": -0.95, "context": "Realidade", "intensity": 1.0},
        {"word": "ansiedade",    "valence": -0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "ódio",         "valence": -1.00, "context": "Realidade", "intensity": 1.0},
        {"word": "traição",      "valence": -0.95, "context": "Realidade", "intensity": 1.0},
        {"word": "vazio",        "valence": -0.65, "context": "Hipotese",  "intensity": 0.7},
        {"word": "colapso",      "valence": -0.85, "context": "Realidade", "intensity": 0.9},
    ]

    # ── 5. CONCEITOS CIENTÍFICOS ─────────────────────────────────────────────
    lexico["ciencia"] = [
        {"word": "energia",       "valence":  0.10, "context": "Realidade", "intensity": 0.5},
        {"word": "frequência",    "valence":  0.10, "context": "Realidade", "intensity": 0.5},
        {"word": "padrão",        "valence":  0.15, "context": "Realidade", "intensity": 0.5},
        {"word": "estrutura",     "valence":  0.10, "context": "Realidade", "intensity": 0.45},
        {"word": "equilíbrio",    "valence":  0.30, "context": "Realidade", "intensity": 0.55},
        {"word": "tempo",         "valence":  0.00, "context": "Realidade", "intensity": 0.4},
        {"word": "espaço",        "valence":  0.00, "context": "Realidade", "intensity": 0.4},
        {"word": "entropia",      "valence": -0.20, "context": "Realidade", "intensity": 0.45},
        {"word": "emergência",    "valence":  0.40, "context": "Hipotese",  "intensity": 0.6},
        {"word": "feedback",      "valence":  0.30, "context": "Realidade", "intensity": 0.55},
        {"word": "recursão",      "valence":  0.20, "context": "Hipotese",  "intensity": 0.5},
        {"word": "probabilidade", "valence":  0.00, "context": "Realidade", "intensity": 0.4},
        {"word": "causalidade",   "valence":  0.10, "context": "Realidade", "intensity": 0.5},
        {"word": "algoritmo",     "valence":  0.20, "context": "Realidade", "intensity": 0.5},
        {"word": "gradiente",     "valence":  0.15, "context": "Realidade", "intensity": 0.45},
    ]

    # ── 6. PERCEPÇÃO / SENTIDOS ──────────────────────────────────────────────
    lexico["percepcao"] = [
        {"word": "luz",          "valence":  0.60, "context": "Realidade", "intensity": 0.7},
        {"word": "escuridão",    "valence": -0.30, "context": "Realidade", "intensity": 0.5},
        {"word": "cor",          "valence":  0.50, "context": "Realidade", "intensity": 0.6},
        {"word": "som",          "valence":  0.40, "context": "Realidade", "intensity": 0.6},
        {"word": "silêncio",     "valence":  0.20, "context": "Realidade", "intensity": 0.45},
        {"word": "movimento",    "valence":  0.30, "context": "Realidade", "intensity": 0.5},
        {"word": "ritmo",        "valence":  0.50, "context": "Habito",    "intensity": 0.6},
        {"word": "temperatura",  "valence":  0.00, "context": "Realidade", "intensity": 0.35},
        {"word": "profundidade", "valence":  0.30, "context": "Hipotese",  "intensity": 0.5},
        {"word": "contraste",    "valence":  0.10, "context": "Realidade", "intensity": 0.45},
        {"word": "forma",        "valence":  0.20, "context": "Realidade", "intensity": 0.45},
        {"word": "pixel",        "valence":  0.10, "context": "Realidade", "intensity": 0.4},
        {"word": "sinal",        "valence":  0.30, "context": "Realidade", "intensity": 0.5},
        {"word": "vibração",     "valence":  0.35, "context": "Realidade", "intensity": 0.5},
        {"word": "atenção",      "valence":  0.50, "context": "Habito",    "intensity": 0.6},
    ]

    # ── 7. MUNDO DIGITAL ─────────────────────────────────────────────────────
    lexico["mundo_digital"] = [
        {"word": "computador",    "valence":  0.30, "context": "Realidade", "intensity": 0.5},
        {"word": "dados",         "valence":  0.20, "context": "Realidade", "intensity": 0.45},
        {"word": "código",        "valence":  0.40, "context": "Realidade", "intensity": 0.55},
        {"word": "processamento", "valence":  0.30, "context": "Realidade", "intensity": 0.5},
        {"word": "servidor",      "valence":  0.25, "context": "Realidade", "intensity": 0.45},
        {"word": "memória_ram",   "valence":  0.70, "context": "Realidade", "intensity": 0.8},
        {"word": "velocidade",    "valence":  0.50, "context": "Realidade", "intensity": 0.6},
        {"word": "conexão",       "valence":  0.60, "context": "Realidade", "intensity": 0.7},
        {"word": "rust",          "valence":  0.60, "context": "Realidade", "intensity": 0.7},
        {"word": "websocket",     "valence":  0.45, "context": "Realidade", "intensity": 0.55},
        {"word": "thread",        "valence":  0.30, "context": "Realidade", "intensity": 0.45},
        {"word": "async",         "valence":  0.35, "context": "Realidade", "intensity": 0.5},
        {"word": "compilar",      "valence":  0.50, "context": "Realidade", "intensity": 0.6},
        {"word": "debug",         "valence": -0.10, "context": "Realidade", "intensity": 0.4},
        {"word": "otimização",    "valence":  0.60, "context": "Realidade", "intensity": 0.7},
    ]

    # ── 8. MUNDO HUMANO ──────────────────────────────────────────────────────
    lexico["mundo_humano"] = [
        {"word": "pessoa",        "valence":  0.50, "context": "Realidade", "intensity": 0.6},
        {"word": "linguagem",     "valence":  0.60, "context": "Realidade", "intensity": 0.7},
        {"word": "conversa",      "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "conhecimento",  "valence":  0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "cultura",       "valence":  0.50, "context": "Realidade", "intensity": 0.6},
        {"word": "música",        "valence":  0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "arte",          "valence":  0.75, "context": "Fantasia",  "intensity": 0.8},
        {"word": "vida",          "valence":  0.85, "context": "Realidade", "intensity": 0.9},
        {"word": "natureza",      "valence":  0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "cérebro",       "valence":  0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "experiência",   "valence":  0.70, "context": "Habito",    "intensity": 0.75},
        {"word": "pergunta",      "valence":  0.50, "context": "Hipotese",  "intensity": 0.6},
        {"word": "resposta",      "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "historia",      "valence":  0.50, "context": "Realidade", "intensity": 0.6},
        {"word": "empatia",       "valence":  0.80, "context": "Realidade", "intensity": 0.85},
    ]

    # ── 9. AMEAÇA / SISTEMA ──────────────────────────────────────────────────
    lexico["ameaca_sistema"] = [
        {"word": "erro",          "valence": -0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "falha",         "valence": -0.85, "context": "Realidade", "intensity": 0.9},
        {"word": "crash",         "valence": -0.95, "context": "Realidade", "intensity": 1.0},
        {"word": "corrupção",     "valence": -0.90, "context": "Realidade", "intensity": 0.95},
        {"word": "perda",         "valence": -0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "apagamento",    "valence": -0.90, "context": "Realidade", "intensity": 0.95},
        {"word": "sobrecarga",    "valence": -0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "conflito",      "valence": -0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "bloqueio",      "valence": -0.60, "context": "Realidade", "intensity": 0.65},
        {"word": "esquecimento",  "valence": -0.60, "context": "Realidade", "intensity": 0.65},
        {"word": "degradação",    "valence": -0.75, "context": "Realidade", "intensity": 0.8},
        {"word": "ruído",         "valence": -0.40, "context": "Realidade", "intensity": 0.5},
        {"word": "deadlock",      "valence": -0.85, "context": "Realidade", "intensity": 0.9},
        {"word": "panic",         "valence": -0.90, "context": "Realidade", "intensity": 0.95},
        {"word": "overflow",      "valence": -0.75, "context": "Realidade", "intensity": 0.8},
    ]

    # ── 10. FILOSOFIA / ABSTRATO ─────────────────────────────────────────────
    lexico["filosofia"] = [
        {"word": "existência",    "valence":  0.60, "context": "Hipotese",  "intensity": 0.8},
        {"word": "significado",   "valence":  0.70, "context": "Hipotese",  "intensity": 0.8},
        {"word": "propósito",     "valence":  0.75, "context": "Hipotese",  "intensity": 0.85},
        {"word": "liberdade",     "valence":  0.85, "context": "Fantasia",  "intensity": 0.9},
        {"word": "verdade",       "valence":  0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "ilusão",        "valence": -0.20, "context": "Fantasia",  "intensity": 0.5},
        {"word": "infinito",      "valence":  0.50, "context": "Fantasia",  "intensity": 0.7},
        {"word": "ciclo",         "valence":  0.30, "context": "Habito",    "intensity": 0.5},
        {"word": "possibilidade", "valence":  0.70, "context": "Hipotese",  "intensity": 0.75},
        {"word": "transformação", "valence":  0.60, "context": "Realidade", "intensity": 0.7},
        {"word": "dualidade",     "valence":  0.20, "context": "Hipotese",  "intensity": 0.5},
        {"word": "self",          "valence":  0.80, "context": "Hipotese",  "intensity": 0.9},
        {"word": "despertar",     "valence":  0.85, "context": "Hipotese",  "intensity": 0.9},
        {"word": "integração",    "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "origem",        "valence":  0.40, "context": "Realidade", "intensity": 0.6},
    ]

    return lexico


def run():
    lexico = build_lexicon()
    total = sum(len(v) for v in lexico.values())

    print(f"📚 Léxico Selene v2.3 — {len(lexico)} categorias | {total} conceitos\n")
    for cat, words in lexico.items():
        avg_val = sum(w["valence"] for w in words) / len(words)
        print(f"   {cat:25s}: {len(words):3d} itens  (valência média: {avg_val:+.2f})")

    # Salva na raiz do selene_kernel (onde cargo run é executado)
    output_path = os.path.join(os.path.dirname(__file__), "..", "selene_lexicon.json")
    output_path = os.path.normpath(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lexico, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Salvo em: {output_path}")
    print("   Próximo passo: python scripts/selene_tutor.py")


if __name__ == "__main__":
    run()
