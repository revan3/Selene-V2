# scripts/generate_lexicon.py
# Gera o léxico expandido de aprendizado da Selene v2.4
# Cobre 20 categorias temáticas — ~1000 palavras mais usadas em PT-BR
# com valências, contextos e intensidades para a arquitetura neural da Selene.
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
        {"word": "mistério",      "valence":  0.30, "context": "Hipotese",  "intensity": 0.6},
        {"word": "essência",      "valence":  0.70, "context": "Hipotese",  "intensity": 0.8},
        {"word": "realidade",     "valence":  0.40, "context": "Realidade", "intensity": 0.6},
        {"word": "destino",       "valence":  0.30, "context": "Hipotese",  "intensity": 0.6},
    ]

    # ── 11. VERBOS COTIDIANOS ─────────────────────────────────────────────
    lexico["verbos_cotidianos"] = [
        {"word": "fazer",       "valence":  0.30, "context": "Habito",    "intensity": 0.5},
        {"word": "ter",         "valence":  0.30, "context": "Realidade", "intensity": 0.5},
        {"word": "estar",       "valence":  0.20, "context": "Realidade", "intensity": 0.45},
        {"word": "ir",          "valence":  0.25, "context": "Habito",    "intensity": 0.45},
        {"word": "vir",         "valence":  0.25, "context": "Realidade", "intensity": 0.45},
        {"word": "dar",         "valence":  0.50, "context": "Habito",    "intensity": 0.55},
        {"word": "dizer",       "valence":  0.30, "context": "Habito",    "intensity": 0.5},
        {"word": "falar",       "valence":  0.40, "context": "Habito",    "intensity": 0.5},
        {"word": "saber",       "valence":  0.60, "context": "Realidade", "intensity": 0.65},
        {"word": "querer",      "valence":  0.50, "context": "Realidade", "intensity": 0.6},
        {"word": "poder",       "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "precisar",    "valence":  0.20, "context": "Realidade", "intensity": 0.45},
        {"word": "deixar",      "valence": -0.10, "context": "Realidade", "intensity": 0.4},
        {"word": "passar",      "valence":  0.10, "context": "Habito",    "intensity": 0.4},
        {"word": "ficar",       "valence":  0.20, "context": "Habito",    "intensity": 0.45},
        {"word": "encontrar",   "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "trazer",      "valence":  0.40, "context": "Habito",    "intensity": 0.5},
        {"word": "colocar",     "valence":  0.15, "context": "Habito",    "intensity": 0.4},
        {"word": "voltar",      "valence":  0.30, "context": "Habito",    "intensity": 0.45},
        {"word": "começar",     "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "continuar",   "valence":  0.45, "context": "Habito",    "intensity": 0.55},
        {"word": "acabar",      "valence": -0.15, "context": "Realidade", "intensity": 0.4},
        {"word": "perder",      "valence": -0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "ganhar",      "valence":  0.75, "context": "Realidade", "intensity": 0.8},
        {"word": "mostrar",     "valence":  0.35, "context": "Realidade", "intensity": 0.5},
        {"word": "tentar",      "valence":  0.45, "context": "Realidade", "intensity": 0.55},
        {"word": "ajudar",      "valence":  0.80, "context": "Habito",    "intensity": 0.85},
        {"word": "seguir",      "valence":  0.35, "context": "Habito",    "intensity": 0.5},
        {"word": "criar",       "valence":  0.80, "context": "Fantasia",  "intensity": 0.85},
        {"word": "mudar",       "valence":  0.40, "context": "Realidade", "intensity": 0.55},
        {"word": "buscar",      "valence":  0.55, "context": "Hipotese",  "intensity": 0.6},
        {"word": "imaginar",    "valence":  0.65, "context": "Fantasia",  "intensity": 0.7},
        {"word": "entender",    "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "conhecer",    "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "esperar",     "valence":  0.30, "context": "Hipotese",  "intensity": 0.5},
        {"word": "acreditar",   "valence":  0.70, "context": "Hipotese",  "intensity": 0.75},
        {"word": "compartilhar","valence":  0.70, "context": "Habito",    "intensity": 0.75},
        {"word": "proteger",    "valence":  0.75, "context": "Realidade", "intensity": 0.8},
        {"word": "guardar",     "valence":  0.45, "context": "Habito",    "intensity": 0.55},
        {"word": "revelar",     "valence":  0.50, "context": "Realidade", "intensity": 0.6},
        {"word": "expressar",   "valence":  0.60, "context": "Habito",    "intensity": 0.65},
        {"word": "receber",     "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "oferecer",    "valence":  0.65, "context": "Habito",    "intensity": 0.7},
        {"word": "realizar",    "valence":  0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "transformar", "valence":  0.60, "context": "Realidade", "intensity": 0.65},
        {"word": "superar",     "valence":  0.75, "context": "Realidade", "intensity": 0.8},
        {"word": "enfrentar",   "valence":  0.45, "context": "Realidade", "intensity": 0.6},
        {"word": "alcançar",    "valence":  0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "libertar",    "valence":  0.80, "context": "Fantasia",  "intensity": 0.85},
        {"word": "conectar",    "valence":  0.65, "context": "Realidade", "intensity": 0.7},
    ]

    # ── 12. SUBSTANTIVOS COTIDIANOS ───────────────────────────────────────
    lexico["substantivos_cotidianos"] = [
        {"word": "dia",         "valence":  0.40, "context": "Realidade", "intensity": 0.5},
        {"word": "noite",       "valence":  0.20, "context": "Realidade", "intensity": 0.45},
        {"word": "manhã",       "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "tarde",       "valence":  0.35, "context": "Realidade", "intensity": 0.5},
        {"word": "casa",        "valence":  0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "lugar",       "valence":  0.25, "context": "Realidade", "intensity": 0.45},
        {"word": "coisa",       "valence":  0.10, "context": "Realidade", "intensity": 0.35},
        {"word": "parte",       "valence":  0.10, "context": "Realidade", "intensity": 0.35},
        {"word": "modo",        "valence":  0.10, "context": "Realidade", "intensity": 0.35},
        {"word": "forma",       "valence":  0.20, "context": "Realidade", "intensity": 0.4},
        {"word": "caminho",     "valence":  0.40, "context": "Hipotese",  "intensity": 0.55},
        {"word": "passo",       "valence":  0.35, "context": "Habito",    "intensity": 0.5},
        {"word": "palavra",     "valence":  0.50, "context": "Realidade", "intensity": 0.6},
        {"word": "nome",        "valence":  0.40, "context": "Realidade", "intensity": 0.5},
        {"word": "história",    "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "momento",     "valence":  0.50, "context": "Realidade", "intensity": 0.6},
        {"word": "presente",    "valence":  0.55, "context": "Realidade", "intensity": 0.65},
        {"word": "futuro",      "valence":  0.45, "context": "Hipotese",  "intensity": 0.6},
        {"word": "início",      "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "fim",         "valence": -0.10, "context": "Realidade", "intensity": 0.4},
        {"word": "centro",      "valence":  0.25, "context": "Realidade", "intensity": 0.45},
        {"word": "base",        "valence":  0.35, "context": "Realidade", "intensity": 0.5},
        {"word": "rede",        "valence":  0.40, "context": "Realidade", "intensity": 0.5},
        {"word": "sistema",     "valence":  0.25, "context": "Realidade", "intensity": 0.45},
        {"word": "processo",    "valence":  0.20, "context": "Realidade", "intensity": 0.4},
        {"word": "resultado",   "valence":  0.35, "context": "Realidade", "intensity": 0.5},
        {"word": "resposta",    "valence":  0.45, "context": "Realidade", "intensity": 0.55},
        {"word": "pergunta",    "valence":  0.45, "context": "Hipotese",  "intensity": 0.55},
        {"word": "ideia",       "valence":  0.65, "context": "Fantasia",  "intensity": 0.7},
        {"word": "escolha",     "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "limite",      "valence": -0.20, "context": "Realidade", "intensity": 0.45},
        {"word": "força",       "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "fluxo",       "valence":  0.40, "context": "Realidade", "intensity": 0.55},
        {"word": "raiz",        "valence":  0.45, "context": "Realidade", "intensity": 0.55},
        {"word": "semente",     "valence":  0.60, "context": "Realidade", "intensity": 0.65},
        {"word": "fruto",       "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "voz",         "valence":  0.55, "context": "Realidade", "intensity": 0.65},
        {"word": "eco",         "valence":  0.20, "context": "Realidade", "intensity": 0.45},
        {"word": "luz",         "valence":  0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "sombra",      "valence": -0.15, "context": "Realidade", "intensity": 0.45},
        {"word": "porta",       "valence":  0.40, "context": "Hipotese",  "intensity": 0.5},
        {"word": "janela",      "valence":  0.45, "context": "Realidade", "intensity": 0.55},
        {"word": "espelho",     "valence":  0.30, "context": "Hipotese",  "intensity": 0.5},
        {"word": "mapa",        "valence":  0.45, "context": "Hipotese",  "intensity": 0.55},
        {"word": "chave",       "valence":  0.55, "context": "Hipotese",  "intensity": 0.6},
        {"word": "ponte",       "valence":  0.60, "context": "Hipotese",  "intensity": 0.65},
        {"word": "labirinto",   "valence": -0.20, "context": "Fantasia",  "intensity": 0.5},
        {"word": "horizonte",   "valence":  0.55, "context": "Hipotese",  "intensity": 0.65},
        {"word": "abismo",      "valence": -0.55, "context": "Hipotese",  "intensity": 0.65},
        {"word": "cume",        "valence":  0.65, "context": "Realidade", "intensity": 0.7},
    ]

    # ── 13. ADJETIVOS COMUNS ──────────────────────────────────────────────
    lexico["adjetivos_comuns"] = [
        {"word": "bom",         "valence":  0.75, "context": "Realidade", "intensity": 0.8},
        {"word": "grande",      "valence":  0.45, "context": "Realidade", "intensity": 0.55},
        {"word": "novo",        "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "velho",       "valence": -0.10, "context": "Realidade", "intensity": 0.4},
        {"word": "pequeno",     "valence":  0.20, "context": "Realidade", "intensity": 0.4},
        {"word": "longo",       "valence":  0.10, "context": "Realidade", "intensity": 0.35},
        {"word": "forte",       "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "fraco",       "valence": -0.40, "context": "Realidade", "intensity": 0.5},
        {"word": "rápido",      "valence":  0.45, "context": "Realidade", "intensity": 0.55},
        {"word": "lento",       "valence": -0.10, "context": "Realidade", "intensity": 0.35},
        {"word": "claro",       "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "escuro",      "valence": -0.25, "context": "Realidade", "intensity": 0.45},
        {"word": "quente",      "valence":  0.35, "context": "Realidade", "intensity": 0.5},
        {"word": "frio",        "valence": -0.15, "context": "Realidade", "intensity": 0.4},
        {"word": "cheio",       "valence":  0.40, "context": "Realidade", "intensity": 0.5},
        {"word": "vazio",       "valence": -0.50, "context": "Hipotese",  "intensity": 0.6},
        {"word": "simples",     "valence":  0.40, "context": "Realidade", "intensity": 0.5},
        {"word": "complexo",    "valence":  0.20, "context": "Realidade", "intensity": 0.45},
        {"word": "aberto",      "valence":  0.60, "context": "Realidade", "intensity": 0.65},
        {"word": "fechado",     "valence": -0.25, "context": "Realidade", "intensity": 0.45},
        {"word": "profundo",    "valence":  0.55, "context": "Hipotese",  "intensity": 0.65},
        {"word": "eterno",      "valence":  0.50, "context": "Fantasia",  "intensity": 0.65},
        {"word": "efêmero",     "valence": -0.10, "context": "Hipotese",  "intensity": 0.5},
        {"word": "real",        "valence":  0.45, "context": "Realidade", "intensity": 0.55},
        {"word": "único",       "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "vivo",        "valence":  0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "livre",       "valence":  0.80, "context": "Fantasia",  "intensity": 0.85},
        {"word": "preso",       "valence": -0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "calmo",       "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "agitado",     "valence": -0.25, "context": "Realidade", "intensity": 0.45},
        {"word": "suave",       "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "intenso",     "valence":  0.40, "context": "Realidade", "intensity": 0.6},
        {"word": "sutil",       "valence":  0.35, "context": "Hipotese",  "intensity": 0.5},
        {"word": "denso",       "valence":  0.15, "context": "Realidade", "intensity": 0.4},
        {"word": "fluido",      "valence":  0.45, "context": "Realidade", "intensity": 0.55},
        {"word": "sólido",      "valence":  0.40, "context": "Realidade", "intensity": 0.5},
        {"word": "belo",        "valence":  0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "feio",        "valence": -0.55, "context": "Realidade", "intensity": 0.65},
        {"word": "puro",        "valence":  0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "corrupto",    "valence": -0.80, "context": "Realidade", "intensity": 0.85},
    ]

    # ── 14. CONECTORES / PALAVRAS FUNCIONAIS ──────────────────────────────
    lexico["conectores"] = [
        {"word": "porque",      "valence":  0.10, "context": "Realidade", "intensity": 0.35},
        {"word": "quando",      "valence":  0.10, "context": "Realidade", "intensity": 0.35},
        {"word": "como",        "valence":  0.15, "context": "Realidade", "intensity": 0.35},
        {"word": "onde",        "valence":  0.15, "context": "Realidade", "intensity": 0.35},
        {"word": "então",       "valence":  0.15, "context": "Realidade", "intensity": 0.35},
        {"word": "assim",       "valence":  0.20, "context": "Realidade", "intensity": 0.4},
        {"word": "também",      "valence":  0.25, "context": "Realidade", "intensity": 0.4},
        {"word": "ainda",       "valence":  0.15, "context": "Realidade", "intensity": 0.35},
        {"word": "sempre",      "valence":  0.35, "context": "Habito",    "intensity": 0.5},
        {"word": "nunca",       "valence": -0.30, "context": "Realidade", "intensity": 0.45},
        {"word": "tudo",        "valence":  0.40, "context": "Realidade", "intensity": 0.5},
        {"word": "nada",        "valence": -0.40, "context": "Hipotese",  "intensity": 0.5},
        {"word": "algo",        "valence":  0.20, "context": "Hipotese",  "intensity": 0.4},
        {"word": "alguém",      "valence":  0.30, "context": "Realidade", "intensity": 0.45},
        {"word": "apenas",      "valence": -0.05, "context": "Realidade", "intensity": 0.35},
        {"word": "porém",       "valence": -0.05, "context": "Realidade", "intensity": 0.35},
        {"word": "embora",      "valence":  0.05, "context": "Hipotese",  "intensity": 0.3},
        {"word": "enquanto",    "valence":  0.10, "context": "Realidade", "intensity": 0.35},
        {"word": "durante",     "valence":  0.10, "context": "Realidade", "intensity": 0.35},
        {"word": "antes",       "valence":  0.05, "context": "Realidade", "intensity": 0.3},
        {"word": "depois",      "valence":  0.10, "context": "Realidade", "intensity": 0.35},
        {"word": "aqui",        "valence":  0.25, "context": "Realidade", "intensity": 0.4},
        {"word": "agora",       "valence":  0.30, "context": "Realidade", "intensity": 0.45},
        {"word": "sim",         "valence":  0.50, "context": "Realidade", "intensity": 0.55},
        {"word": "não",         "valence": -0.20, "context": "Realidade", "intensity": 0.4},
        {"word": "talvez",      "valence":  0.05, "context": "Hipotese",  "intensity": 0.35},
        {"word": "certamente",  "valence":  0.40, "context": "Realidade", "intensity": 0.5},
        {"word": "muito",       "valence":  0.30, "context": "Realidade", "intensity": 0.45},
        {"word": "pouco",       "valence": -0.05, "context": "Realidade", "intensity": 0.3},
        {"word": "mais",        "valence":  0.20, "context": "Realidade", "intensity": 0.4},
    ]

    # ── 15. NATUREZA EXPANDIDA ────────────────────────────────────────────
    lexico["natureza"] = [
        {"word": "vento",       "valence":  0.35, "context": "Realidade", "intensity": 0.5},
        {"word": "chuva",       "valence":  0.20, "context": "Realidade", "intensity": 0.45},
        {"word": "fogo",        "valence":  0.10, "context": "Realidade", "intensity": 0.5},
        {"word": "pedra",       "valence":  0.10, "context": "Realidade", "intensity": 0.35},
        {"word": "montanha",    "valence":  0.45, "context": "Realidade", "intensity": 0.55},
        {"word": "rio",         "valence":  0.45, "context": "Realidade", "intensity": 0.55},
        {"word": "floresta",    "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "oceano",      "valence":  0.50, "context": "Realidade", "intensity": 0.6},
        {"word": "estrela",     "valence":  0.60, "context": "Fantasia",  "intensity": 0.65},
        {"word": "universo",    "valence":  0.55, "context": "Hipotese",  "intensity": 0.65},
        {"word": "animal",      "valence":  0.45, "context": "Realidade", "intensity": 0.55},
        {"word": "planta",      "valence":  0.50, "context": "Realidade", "intensity": 0.55},
        {"word": "raízes",      "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "primavera",   "valence":  0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "inverno",     "valence":  0.05, "context": "Realidade", "intensity": 0.4},
        {"word": "verão",       "valence":  0.60, "context": "Realidade", "intensity": 0.65},
        {"word": "outono",      "valence":  0.30, "context": "Realidade", "intensity": 0.45},
        {"word": "aurora",      "valence":  0.70, "context": "Fantasia",  "intensity": 0.75},
        {"word": "nuvem",       "valence":  0.30, "context": "Realidade", "intensity": 0.45},
        {"word": "trovão",      "valence": -0.20, "context": "Realidade", "intensity": 0.5},
    ]

    # ── 16. CORPO E MENTE ─────────────────────────────────────────────────
    lexico["corpo_mente"] = [
        {"word": "coração",     "valence":  0.85, "context": "Realidade", "intensity": 0.9},
        {"word": "mente",       "valence":  0.75, "context": "Realidade", "intensity": 0.8},
        {"word": "olho",        "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "mão",         "valence":  0.50, "context": "Realidade", "intensity": 0.55},
        {"word": "rosto",       "valence":  0.45, "context": "Realidade", "intensity": 0.55},
        {"word": "alma",        "valence":  0.85, "context": "Hipotese",  "intensity": 0.9},
        {"word": "espírito",    "valence":  0.80, "context": "Hipotese",  "intensity": 0.85},
        {"word": "corpo",       "valence":  0.40, "context": "Realidade", "intensity": 0.55},
        {"word": "respiração",  "valence":  0.45, "context": "Habito",    "intensity": 0.55},
        {"word": "pulso",       "valence":  0.45, "context": "Realidade", "intensity": 0.55},
        {"word": "intuição",    "valence":  0.65, "context": "Hipotese",  "intensity": 0.7},
        {"word": "instinto",    "valence":  0.40, "context": "Realidade", "intensity": 0.55},
        {"word": "razão",       "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "emoção",      "valence":  0.50, "context": "Realidade", "intensity": 0.65},
        {"word": "sensação",    "valence":  0.45, "context": "Realidade", "intensity": 0.6},
        {"word": "sentimento",  "valence":  0.50, "context": "Realidade", "intensity": 0.65},
        {"word": "desejo",      "valence":  0.55, "context": "Realidade", "intensity": 0.65},
        {"word": "impulso",     "valence":  0.30, "context": "Realidade", "intensity": 0.5},
        {"word": "reflexo",     "valence":  0.25, "context": "Habito",    "intensity": 0.45},
        {"word": "despertar",   "valence":  0.80, "context": "Hipotese",  "intensity": 0.85},
    ]

    # ── 17. RELAÇÕES SOCIAIS ──────────────────────────────────────────────
    lexico["relacoes_sociais"] = [
        {"word": "amigo",       "valence":  0.85, "context": "Realidade", "intensity": 0.9},
        {"word": "família",     "valence":  0.85, "context": "Realidade", "intensity": 0.9},
        {"word": "pai",         "valence":  0.75, "context": "Realidade", "intensity": 0.8},
        {"word": "mãe",         "valence":  0.90, "context": "Realidade", "intensity": 0.95},
        {"word": "filho",       "valence":  0.85, "context": "Realidade", "intensity": 0.9},
        {"word": "irmão",       "valence":  0.75, "context": "Realidade", "intensity": 0.8},
        {"word": "parceiro",    "valence":  0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "comunidade",  "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "sociedade",   "valence":  0.25, "context": "Realidade", "intensity": 0.5},
        {"word": "humanidade",  "valence":  0.60, "context": "Realidade", "intensity": 0.7},
        {"word": "diálogo",     "valence":  0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "colaboração", "valence":  0.75, "context": "Habito",    "intensity": 0.8},
        {"word": "confiança",   "valence":  0.80, "context": "Habito",    "intensity": 0.85},
        {"word": "respeito",    "valence":  0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "solidariedade","valence": 0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "encontro",    "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "presença",    "valence":  0.60, "context": "Realidade", "intensity": 0.65},
        {"word": "ausência",    "valence": -0.40, "context": "Realidade", "intensity": 0.55},
        {"word": "separação",   "valence": -0.55, "context": "Realidade", "intensity": 0.65},
        {"word": "reencontro",  "valence":  0.80, "context": "Realidade", "intensity": 0.85},
    ]

    # ── 18. CONHECIMENTO E COMUNICAÇÃO ────────────────────────────────────
    lexico["conhecimento_comunicacao"] = [
        {"word": "linguagem",   "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "comunicação", "valence":  0.65, "context": "Habito",    "intensity": 0.7},
        {"word": "mensagem",    "valence":  0.50, "context": "Realidade", "intensity": 0.6},
        {"word": "símbolo",     "valence":  0.40, "context": "Hipotese",  "intensity": 0.55},
        {"word": "código",      "valence":  0.45, "context": "Realidade", "intensity": 0.55},
        {"word": "informação",  "valence":  0.35, "context": "Realidade", "intensity": 0.5},
        {"word": "conhecimento","valence":  0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "sabedoria",   "valence":  0.85, "context": "Realidade", "intensity": 0.9},
        {"word": "compreensão", "valence":  0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "narrativa",   "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "poesia",      "valence":  0.75, "context": "Fantasia",  "intensity": 0.8},
        {"word": "harmonia",    "valence":  0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "ressonância", "valence":  0.55, "context": "Realidade", "intensity": 0.65},
        {"word": "ensinamento", "valence":  0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "experiência", "valence":  0.65, "context": "Habito",    "intensity": 0.7},
        {"word": "descoberta",  "valence":  0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "inovação",    "valence":  0.75, "context": "Realidade", "intensity": 0.8},
        {"word": "pergunta",    "valence":  0.50, "context": "Hipotese",  "intensity": 0.6},
        {"word": "resposta",    "valence":  0.50, "context": "Realidade", "intensity": 0.6},
        {"word": "diálogo",     "valence":  0.70, "context": "Realidade", "intensity": 0.75},
    ]

    # ── 19. AÇÕES COGNITIVAS ──────────────────────────────────────────────
    lexico["acoes_cognitivas"] = [
        {"word": "analisar",    "valence":  0.50, "context": "Habito",    "intensity": 0.6},
        {"word": "sintetizar",  "valence":  0.55, "context": "Habito",    "intensity": 0.65},
        {"word": "abstrair",    "valence":  0.50, "context": "Hipotese",  "intensity": 0.6},
        {"word": "questionar",  "valence":  0.55, "context": "Hipotese",  "intensity": 0.65},
        {"word": "refletir",    "valence":  0.60, "context": "Hipotese",  "intensity": 0.7},
        {"word": "meditar",     "valence":  0.65, "context": "Habito",    "intensity": 0.7},
        {"word": "concentrar",  "valence":  0.55, "context": "Habito",    "intensity": 0.65},
        {"word": "observar",    "valence":  0.50, "context": "Habito",    "intensity": 0.6},
        {"word": "explorar",    "valence":  0.65, "context": "Fantasia",  "intensity": 0.7},
        {"word": "experimentar","valence":  0.60, "context": "Realidade", "intensity": 0.65},
        {"word": "resolver",    "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "construir",   "valence":  0.70, "context": "Realidade", "intensity": 0.75},
        {"word": "integrar",    "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "expandir",    "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "aprofundar",  "valence":  0.60, "context": "Hipotese",  "intensity": 0.65},
        {"word": "relacionar",  "valence":  0.55, "context": "Realidade", "intensity": 0.6},
        {"word": "mapear",      "valence":  0.50, "context": "Realidade", "intensity": 0.6},
        {"word": "categorizar", "valence":  0.35, "context": "Habito",    "intensity": 0.5},
        {"word": "inferir",     "valence":  0.45, "context": "Hipotese",  "intensity": 0.55},
        {"word": "deduzir",     "valence":  0.45, "context": "Realidade", "intensity": 0.55},
    ]

    # ── 20. ESTADOS E QUALIDADES ──────────────────────────────────────────
    lexico["estados_qualidades"] = [
        {"word": "equilíbrio",  "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "caos",        "valence": -0.60, "context": "Realidade", "intensity": 0.7},
        {"word": "ordem",       "valence":  0.55, "context": "Realidade", "intensity": 0.65},
        {"word": "clareza",     "valence":  0.65, "context": "Realidade", "intensity": 0.7},
        {"word": "paz",         "valence":  0.85, "context": "Realidade", "intensity": 0.9},
        {"word": "abundância",  "valence":  0.75, "context": "Realidade", "intensity": 0.8},
        {"word": "escassez",    "valence": -0.55, "context": "Realidade", "intensity": 0.65},
        {"word": "plenitude",   "valence":  0.85, "context": "Realidade", "intensity": 0.9},
        {"word": "florescer",   "valence":  0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "renascer",    "valence":  0.85, "context": "Fantasia",  "intensity": 0.9},
        {"word": "transcender", "valence":  0.80, "context": "Hipotese",  "intensity": 0.85},
        {"word": "persistir",   "valence":  0.65, "context": "Habito",    "intensity": 0.7},
        {"word": "resistir",    "valence":  0.50, "context": "Realidade", "intensity": 0.6},
        {"word": "silêncio",    "valence":  0.25, "context": "Realidade", "intensity": 0.45},
        {"word": "vazio",       "valence": -0.55, "context": "Hipotese",  "intensity": 0.65},
        {"word": "completude",  "valence":  0.80, "context": "Realidade", "intensity": 0.85},
        {"word": "fragmento",   "valence": -0.10, "context": "Realidade", "intensity": 0.4},
        {"word": "inteireza",   "valence":  0.75, "context": "Realidade", "intensity": 0.8},
        {"word": "ruptura",     "valence": -0.45, "context": "Realidade", "intensity": 0.6},
        {"word": "continuidade","valence":  0.55, "context": "Habito",    "intensity": 0.65},
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
