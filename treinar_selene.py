#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
treinar_selene.py -- Script de treinamento de templates: Fala + Matemática + Física + Química + Engenharia + Programação

Protocolo (3 passes com spaced-repetition bio-inspirada):
  Pass 1 — APRESENTAÇÃO:   todas as unidades, 1×, 60 ms entre itens
  Pass 2 — REFORÇO:        |valence| >= 0.4, 2×, 90 ms
  Pass 3 — CONSOLIDAÇÃO:   |valence| >= 0.7, 3×, 140 ms
  Entre módulos: sweep STDP ("train") para consolidar antes do próximo bloco.

Módulos de treinamento:
  [A] FALA — Fonologia e Prosódia
    A1  Aparelho fonador (lábios, língua, glote, palato)
    A2  Fonemas PT-BR (35 fonemas + alofones comuns)
    A3  Prosódia e entonação (perguntas, afirmações, emoção)
    A4  Sílabas e estrutura silábica (CV, CVC, CCVC…)
    A5  Fluência e coesão (conectivos, transições, pausas)
    A6  Metafonia e variações regionais

  [B] MATEMÁTICA — Fundamentos e Raciocínio
    B1  Números e quantidades (0–20 + centenas + frações)
    B2  Operações básicas (+ - × ÷) com frases semânticas
    B3  Geometria e formas (2D + 3D + propriedades)
    B4  Álgebra e variáveis (equações, incógnitas)
    B5  Probabilidade e estatística (básica)
    B6  Lógica matemática (e/ou/não, verdadeiro/falso, se-então)
    B7  Vocabulário matemático PT-BR formal

Uso:
  python treinar_selene.py                    # tudo
  python treinar_selene.py --modulo A         # só fala
  python treinar_selene.py --modulo B         # só matemática
  python treinar_selene.py --modulo C         # só física
  python treinar_selene.py --modulo D         # só química
  python treinar_selene.py --modulo E         # só engenharia
  python treinar_selene.py --modulo F         # só programação
  python treinar_selene.py --modulo C1        # só mecânica
  python treinar_selene.py --rep 5            # N repetições por passe
  python treinar_selene.py --host 192.168.1.x # servidor remoto
  python treinar_selene.py --rapido           # sem pausa entre módulos
  python treinar_selene.py --seco             # dry-run (não conecta, só lista)
"""

import asyncio
import json
import sys
import os
import time
import uuid as _uuid_mod

try:
    import websockets
except ImportError:
    print("❌ Instale: pip install websockets")
    sys.exit(1)

# ── DualTrainer V3.2 (audiovisual — FFT + grounding + escrita) ────────────────
# Importação opcional: se selene_audio_utils.py existir e espeak-ng estiver
# instalado, usa aprendizado dual (áudio + texto). Caso contrário, usa apenas
# texto (modo degradado, mas funcional).
try:
    from selene_audio_utils import DualTrainer, sintetizar_espeak
    _DUAL_OK = True
except ImportError:
    DualTrainer     = None  # type: ignore
    sintetizar_espeak = None  # type: ignore
    _DUAL_OK = False

ESPEAK_OK = _DUAL_OK and sintetizar_espeak is not None and sintetizar_espeak("a") is not None

# ── Configuração ──────────────────────────────────────────────────────────────
HOST    = "127.0.0.1"
PORTA   = "3030"
WS_URL  = f"ws://{HOST}:{PORTA}/selene"

DELAY_P1 = 0.060   # 60ms  — apresentação
DELAY_P2 = 0.090   # 90ms  — reforço
DELAY_P3 = 0.140   # 140ms — consolidação
PAUSA_MODULO = 1.2  # pausa entre módulos

PASS2_THRESHOLD = 0.4
PASS3_THRESHOLD = 0.7
REPS_PADRAO     = 3   # repetições por passe (multiplicador)
MODULO_ALVO     = None  # None = tudo
SECO            = False

# ── Parseia argumentos ────────────────────────────────────────────────────────
_args = sys.argv[1:]
_i = 0
while _i < len(_args):
    a = _args[_i]
    if a == "--host" and _i + 1 < len(_args):
        HOST = _args[_i + 1]; WS_URL = f"ws://{HOST}:{PORTA}/selene"; _i += 2
    elif a == "--porta" and _i + 1 < len(_args):
        PORTA = _args[_i + 1]; WS_URL = f"ws://{HOST}:{PORTA}/selene"; _i += 2
    elif a == "--modulo" and _i + 1 < len(_args):
        MODULO_ALVO = _args[_i + 1].upper(); _i += 2
    elif a == "--rep" and _i + 1 < len(_args):
        try: REPS_PADRAO = int(_args[_i + 1])
        except ValueError: pass
        _i += 2
    elif a == "--rapido":
        PAUSA_MODULO = 0.2; _i += 1
    elif a == "--seco":
        SECO = True; _i += 1
    else:
        _i += 1

# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO A — FALA
# ═══════════════════════════════════════════════════════════════════════════════

# Cada entrada: (palavra/frase, contexto_semântico, valência)
# valência: +1 = altamente desejável de aprender, -1 = aversivo (erros, dor)

MODULO_A1_FONADOR = [
    # (texto, contexto, valence)
    ("lábios",        "anatomia_fala", 0.6),
    ("língua",        "anatomia_fala", 0.6),
    ("dentes",        "anatomia_fala", 0.5),
    ("palato",        "anatomia_fala", 0.5),
    ("palato mole",   "anatomia_fala", 0.5),
    ("glote",         "anatomia_fala", 0.6),
    ("cordas vocais", "anatomia_fala", 0.7),
    ("faringe",       "anatomia_fala", 0.5),
    ("laringe",       "anatomia_fala", 0.5),
    ("cavidade oral", "anatomia_fala", 0.5),
    ("cavidade nasal","anatomia_fala", 0.5),
    ("pulmões",       "anatomia_fala", 0.6),
    ("diafragma",     "anatomia_fala", 0.6),
    ("articulação",   "anatomia_fala", 0.7),
    ("fonação",       "anatomia_fala", 0.7),
    ("ressonância",   "anatomia_fala", 0.6),
]

MODULO_A2_FONEMAS = [
    # Vogais
    ("a", "fonema_vogal", 0.8), ("e", "fonema_vogal", 0.8),
    ("i", "fonema_vogal", 0.8), ("o", "fonema_vogal", 0.8),
    ("u", "fonema_vogal", 0.8),
    ("é", "fonema_vogal", 0.7), ("ê", "fonema_vogal", 0.7),
    ("ó", "fonema_vogal", 0.7), ("ô", "fonema_vogal", 0.7),
    # Nasais
    ("ão", "fonema_nasal", 0.7), ("em", "fonema_nasal", 0.7),
    ("im", "fonema_nasal", 0.7), ("om", "fonema_nasal", 0.7),
    ("um", "fonema_nasal", 0.7),
    # Consoantes oclusivas
    ("p", "fonema_consoante", 0.7), ("b", "fonema_consoante", 0.7),
    ("t", "fonema_consoante", 0.7), ("d", "fonema_consoante", 0.7),
    ("k", "fonema_consoante", 0.7), ("g", "fonema_consoante", 0.7),
    # Fricativas
    ("f", "fonema_fricativa", 0.7), ("v", "fonema_fricativa", 0.7),
    ("s", "fonema_fricativa", 0.7), ("z", "fonema_fricativa", 0.7),
    ("x", "fonema_fricativa", 0.7), ("j", "fonema_fricativa", 0.7),
    # Líquidas e nasais
    ("l", "fonema_liquida", 0.7), ("r", "fonema_vibrante", 0.7),
    ("rr", "fonema_vibrante", 0.7), ("lh", "fonema_liquida", 0.7),
    ("nh", "fonema_nasal", 0.7), ("m", "fonema_nasal", 0.7),
    ("n", "fonema_nasal", 0.7),
]

MODULO_A3_PROSÓDIA = [
    # (frase, contexto, valence)
    ("Você entendeu?",              "prosódia_pergunta",    0.8),
    ("Sim, entendi.",               "prosódia_afirmação",   0.8),
    ("Não tenho certeza.",          "prosódia_dúvida",      0.7),
    ("Que ótima ideia!",            "prosódia_entusiasmo",  0.9),
    ("Espere um momento.",          "prosódia_pausa",       0.7),
    ("Por favor, repita.",          "prosódia_pedido",      0.7),
    ("Estou processando.",          "prosódia_neutro",      0.6),
    ("Claro, com prazer!",          "prosódia_afirmação",   0.9),
    ("Desculpe, não compreendi.",   "prosódia_erro",        0.5),
    ("Deixe-me pensar...",          "prosódia_reflexão",    0.7),
    ("Isso é correto.",             "prosódia_confirmação", 0.8),
    ("Isso é incorreto.",           "prosódia_correção",    0.6),
    ("Pode explicar melhor?",       "prosódia_pergunta",    0.8),
    ("Entendido, prosseguindo.",    "prosódia_confirmação", 0.8),
    ("Interessante observação.",    "prosódia_reflexão",    0.8),
]

MODULO_A4_SÍLABAS = [
    # Sílabas canônicas PT-BR com exemplos
    ("ca", "sílaba_cv", 0.7), ("ma", "sílaba_cv", 0.7), ("pa", "sílaba_cv", 0.7),
    ("bra", "sílaba_ccv", 0.7), ("tra", "sílaba_ccv", 0.7), ("fla", "sílaba_ccv", 0.7),
    ("ar",  "sílaba_vc",  0.7), ("er",  "sílaba_vc",  0.7), ("or",  "sílaba_vc",  0.7),
    ("com", "sílaba_cvc", 0.7), ("tem", "sílaba_cvc", 0.7), ("par", "sílaba_cvc", 0.7),
    ("trans", "sílaba_ccvcc", 0.6), ("pres", "sílaba_ccvc", 0.6),
    ("sílaba", "estrutura_silábica", 0.8),
    ("acento", "estrutura_silábica", 0.8),
    ("tônica", "estrutura_silábica", 0.7),
    ("átona",  "estrutura_silábica", 0.7),
    ("hiato",  "estrutura_silábica", 0.6),
    ("ditongo","estrutura_silábica", 0.6),
    ("tritongo","estrutura_silábica", 0.6),
]

MODULO_A5_FLUÊNCIA = [
    ("portanto",         "conectivo_conclusão",  0.8),
    ("portanto, conclui-se que", "conectivo_conclusão", 0.8),
    ("no entanto",       "conectivo_contraste",  0.8),
    ("além disso",       "conectivo_adição",     0.8),
    ("em primeiro lugar","conectivo_enumeração", 0.7),
    ("em seguida",       "conectivo_sequência",  0.8),
    ("finalmente",       "conectivo_conclusão",  0.8),
    ("por outro lado",   "conectivo_contraste",  0.8),
    ("ou seja",          "conectivo_reformulação",0.8),
    ("isto é",           "conectivo_reformulação",0.8),
    ("por exemplo",      "conectivo_exemplificação",0.8),
    ("assim sendo",      "conectivo_conclusão",  0.7),
    ("sendo assim",      "conectivo_conclusão",  0.7),
    ("em resumo",        "conectivo_resumo",     0.8),
    ("com base nisso",   "conectivo_base",       0.8),
    ("de acordo com",    "conectivo_referência", 0.7),
    ("vale ressaltar",   "conectivo_ênfase",     0.7),
    ("cabe destacar",    "conectivo_ênfase",     0.7),
]

MODULO_A6_VARIAÇÕES = [
    ("cê tá bom?",          "variação_coloquial", 0.5),
    ("você está bem?",      "variação_formal",    0.8),
    ("tô aqui",             "variação_coloquial", 0.5),
    ("estou aqui",          "variação_formal",    0.8),
    ("boa",                 "variação_gíria",     0.5),
    ("excelente",           "variação_formal",    0.8),
    ("pra mim",             "variação_coloquial", 0.5),
    ("para mim",            "variação_formal",    0.8),
    ("num sei",             "variação_coloquial", 0.4),
    ("não sei",             "variação_formal",    0.8),
]

# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO B — MATEMÁTICA
# ═══════════════════════════════════════════════════════════════════════════════

MODULO_B1_NÚMEROS = [
    # (expressão, contexto, valence)
    ("zero",    "número", 0.8), ("um",    "número", 0.9), ("dois",  "número", 0.9),
    ("três",    "número", 0.9), ("quatro","número", 0.9), ("cinco", "número", 0.9),
    ("seis",    "número", 0.9), ("sete",  "número", 0.9), ("oito",  "número", 0.9),
    ("nove",    "número", 0.9), ("dez",   "número", 0.9), ("onze",  "número", 0.8),
    ("doze",    "número", 0.8), ("treze", "número", 0.8), ("quatorze","número",0.8),
    ("quinze",  "número", 0.8), ("vinte", "número", 0.8), ("trinta","número", 0.8),
    ("cem",     "número", 0.8), ("mil",   "número", 0.8), ("milhão","número", 0.7),
    ("bilhão",  "número", 0.7),
    # Frações
    ("um meio",    "fração", 0.8), ("um terço",  "fração", 0.8),
    ("um quarto",  "fração", 0.8), ("três quartos","fração",0.8),
    ("dois terços","fração", 0.8), ("porcentagem","fração",  0.8),
    ("por cento",  "fração", 0.9),
    # Ordinais
    ("primeiro", "ordinal", 0.8), ("segundo", "ordinal", 0.8),
    ("terceiro", "ordinal", 0.8), ("último",  "ordinal", 0.8),
]

MODULO_B2_OPERAÇÕES = [
    ("mais",            "operação_aritmética", 0.9),
    ("menos",           "operação_aritmética", 0.9),
    ("vezes",           "operação_aritmética", 0.9),
    ("dividido por",    "operação_aritmética", 0.9),
    ("igual a",         "operação_aritmética", 0.9),
    ("dois mais dois é igual a quatro",      "equação_simples", 0.9),
    ("cinco vezes três é igual a quinze",    "equação_simples", 0.9),
    ("dez dividido por dois é igual a cinco","equação_simples", 0.9),
    ("a soma de três e sete é dez",          "equação_semântica", 0.8),
    ("o dobro de quatro é oito",             "equação_semântica", 0.8),
    ("a metade de dez é cinco",              "equação_semântica", 0.8),
    ("o quadrado de três é nove",            "potenciação", 0.8),
    ("a raiz quadrada de nove é três",       "radiciação", 0.8),
    ("potência",        "operação_avançada",   0.7),
    ("raiz quadrada",   "operação_avançada",   0.7),
    ("módulo",          "operação_avançada",   0.6),
    ("resto da divisão","operação_avançada",   0.7),
    ("máximo divisor comum", "operação_avançada", 0.6),
    ("mínimo múltiplo comum","operação_avançada", 0.6),
]

MODULO_B3_GEOMETRIA = [
    ("ponto",           "geometria_2d",  0.8),
    ("reta",            "geometria_2d",  0.8),
    ("segmento",        "geometria_2d",  0.8),
    ("ângulo",          "geometria_2d",  0.8),
    ("ângulo reto",     "geometria_2d",  0.8),
    ("ângulo agudo",    "geometria_2d",  0.7),
    ("ângulo obtuso",   "geometria_2d",  0.7),
    ("triângulo",       "geometria_2d",  0.9),
    ("quadrado",        "geometria_2d",  0.9),
    ("retângulo",       "geometria_2d",  0.9),
    ("círculo",         "geometria_2d",  0.9),
    ("circunferência",  "geometria_2d",  0.8),
    ("polígono",        "geometria_2d",  0.8),
    ("hexágono",        "geometria_2d",  0.7),
    ("pentágono",       "geometria_2d",  0.7),
    ("diagonal",        "geometria_2d",  0.7),
    ("perímetro",       "geometria_medida", 0.8),
    ("área",            "geometria_medida", 0.9),
    ("volume",          "geometria_3d",  0.8),
    ("cubo",            "geometria_3d",  0.8),
    ("esfera",          "geometria_3d",  0.8),
    ("cilindro",        "geometria_3d",  0.8),
    ("cone",            "geometria_3d",  0.7),
    ("pirâmide",        "geometria_3d",  0.7),
    ("paralelo",        "geometria_relação", 0.7),
    ("perpendicular",   "geometria_relação", 0.7),
]

MODULO_B4_ÁLGEBRA = [
    ("variável",         "álgebra", 0.8),
    ("incógnita",        "álgebra", 0.8),
    ("equação",          "álgebra", 0.9),
    ("inequação",        "álgebra", 0.7),
    ("expressão algébrica","álgebra",0.8),
    ("polinômio",        "álgebra", 0.7),
    ("coeficiente",      "álgebra", 0.7),
    ("expoente",         "álgebra", 0.7),
    ("fatoração",        "álgebra", 0.7),
    ("se x mais dois é igual a cinco, então x é três", "equação_verbal", 0.9),
    ("a soma de x e y é dez",                          "equação_verbal", 0.8),
    ("o produto de dois números é vinte",              "equação_verbal", 0.8),
    ("sistema de equações",   "álgebra_avançada", 0.7),
    ("progressão aritmética", "álgebra_avançada", 0.7),
    ("progressão geométrica", "álgebra_avançada", 0.7),
    ("função linear",         "álgebra_avançada", 0.7),
    ("função quadrática",     "álgebra_avançada", 0.7),
]

MODULO_B5_PROBABILIDADE = [
    ("probabilidade",       "probabilidade", 0.8),
    ("chance",              "probabilidade", 0.8),
    ("evento",              "probabilidade", 0.7),
    ("espaço amostral",     "probabilidade", 0.7),
    ("resultado favorável", "probabilidade", 0.7),
    ("cinquenta por cento de chance", "probabilidade_verbal", 0.8),
    ("a probabilidade de sair cara é um meio", "probabilidade_verbal", 0.8),
    ("média",               "estatística", 0.9),
    ("mediana",             "estatística", 0.8),
    ("moda",                "estatística", 0.8),
    ("desvio padrão",       "estatística", 0.7),
    ("amostra",             "estatística", 0.7),
    ("população",           "estatística", 0.7),
    ("correlação",          "estatística", 0.7),
    ("gráfico de barras",   "estatística_visual", 0.7),
    ("gráfico de pizza",    "estatística_visual", 0.7),
]

MODULO_B6_LÓGICA = [
    ("verdadeiro",          "lógica", 0.9),
    ("falso",               "lógica", 0.9),
    ("e",                   "lógica_conectivo", 0.9),
    ("ou",                  "lógica_conectivo", 0.9),
    ("não",                 "lógica_conectivo", 0.9),
    ("se então",            "lógica_condicional", 0.9),
    ("implica",             "lógica_condicional", 0.8),
    ("se e somente se",     "lógica_bicondicional", 0.7),
    ("negação",             "lógica_operação", 0.8),
    ("conjunção",           "lógica_operação", 0.7),
    ("disjunção",           "lógica_operação", 0.7),
    ("tautologia",          "lógica_avançada",  0.6),
    ("contradição",         "lógica_avançada",  0.6),
    ("todo",                "lógica_quantificador", 0.8),
    ("existe",              "lógica_quantificador", 0.8),
    ("se dois é par, então quatro é par",  "lógica_exemplo", 0.8),
    ("todos os quadrados são retângulos",  "lógica_exemplo", 0.8),
    ("nem todo número par é divisível por quatro", "lógica_exemplo", 0.8),
]

MODULO_B7_VOCABULÁRIO = [
    ("calcular",         "vocabulário_matemático", 0.9),
    ("resolver",         "vocabulário_matemático", 0.9),
    ("demonstrar",       "vocabulário_matemático", 0.8),
    ("provar",           "vocabulário_matemático", 0.8),
    ("definir",          "vocabulário_matemático", 0.8),
    ("teorema",          "vocabulário_matemático", 0.8),
    ("corolário",        "vocabulário_matemático", 0.7),
    ("lema",             "vocabulário_matemático", 0.7),
    ("axioma",           "vocabulário_matemático", 0.7),
    ("hipótese",         "vocabulário_matemático", 0.8),
    ("conclusão",        "vocabulário_matemático", 0.8),
    ("contraexemplo",    "vocabulário_matemático", 0.7),
    ("aproximação",      "vocabulário_matemático", 0.8),
    ("estimativa",       "vocabulário_matemático", 0.8),
    ("precisão",         "vocabulário_matemático", 0.8),
    ("exato",            "vocabulário_matemático", 0.8),
    ("infinito",         "vocabulário_matemático", 0.8),
    ("conjunto",         "vocabulário_matemático", 0.8),
    ("subconjunto",      "vocabulário_matemático", 0.7),
    ("interseção",       "vocabulário_matemático", 0.7),
    ("união",            "vocabulário_matemático", 0.7),
]

# ── Módulo C — Física ─────────────────────────────────────────────────────────

MODULO_C1_MECANICA = [
    ("força",            "física_mecânica", 0.9),
    ("massa",            "física_mecânica", 0.9),
    ("aceleração",       "física_mecânica", 0.9),
    ("velocidade",       "física_mecânica", 0.9),
    ("movimento",        "física_mecânica", 0.85),
    ("gravidade",        "física_mecânica", 0.9),
    ("atrito",           "física_mecânica", 0.8),
    ("inércia",          "física_mecânica", 0.85),
    ("trabalho",         "física_mecânica", 0.8),
    ("potência",         "física_mecânica", 0.8),
    ("cinemática",       "física_mecânica", 0.85),
    ("dinâmica",         "física_mecânica", 0.85),
    ("torque",           "física_mecânica", 0.8),
    ("impulso",          "física_mecânica", 0.8),
    ("momentum",         "física_mecânica", 0.85),
    ("a segunda lei de Newton diz que força é massa vezes aceleração", "física_mecânica", 0.95),
    ("a lei da gravitação universal descreve a força entre dois corpos com massa", "física_mecânica", 0.9),
    ("cinemática estuda o movimento sem considerar suas causas", "física_mecânica", 0.9),
]

MODULO_C2_TERMODINAMICA = [
    ("temperatura",      "física_termodinâmica", 0.9),
    ("calor",            "física_termodinâmica", 0.9),
    ("energia",          "física_termodinâmica", 0.95),
    ("entropia",         "física_termodinâmica", 0.9),
    ("pressão",          "física_termodinâmica", 0.85),
    ("volume",           "física_termodinâmica", 0.8),
    ("gás",              "física_termodinâmica", 0.8),
    ("condução",         "física_termodinâmica", 0.8),
    ("convecção",        "física_termodinâmica", 0.8),
    ("radiação térmica", "física_termodinâmica", 0.85),
    ("equilíbrio térmico", "física_termodinâmica", 0.85),
    ("a primeira lei da termodinâmica conserva energia em sistemas fechados", "física_termodinâmica", 0.95),
    ("a entropia de um sistema isolado nunca decresce espontaneamente", "física_termodinâmica", 0.9),
]

MODULO_C3_ELETROMAG = [
    ("carga elétrica",   "física_eletromag", 0.9),
    ("campo elétrico",   "física_eletromag", 0.95),
    ("campo magnético",  "física_eletromag", 0.95),
    ("corrente",         "física_eletromag", 0.85),
    ("tensão",           "física_eletromag", 0.85),
    ("resistência",      "física_eletromag", 0.85),
    ("indução",          "física_eletromag", 0.85),
    ("capacitor",        "física_eletromag", 0.8),
    ("indutor",          "física_eletromag", 0.8),
    ("ondas eletromagnéticas", "física_eletromag", 0.9),
    ("a lei de Coulomb descreve a força entre cargas elétricas", "física_eletromag", 0.9),
    ("as equações de Maxwell unificam eletricidade magnetismo e luz", "física_eletromag", 0.95),
]

MODULO_C4_ONDAS = [
    ("onda",             "física_óptica", 0.85),
    ("frequência",       "física_óptica", 0.85),
    ("comprimento de onda", "física_óptica", 0.9),
    ("amplitude",        "física_óptica", 0.8),
    ("reflexão",         "física_óptica", 0.85),
    ("refração",         "física_óptica", 0.85),
    ("difração",         "física_óptica", 0.8),
    ("interferência",    "física_óptica", 0.8),
    ("luz",              "física_óptica", 0.9),
    ("fóton",            "física_óptica", 0.9),
    ("espectro",         "física_óptica", 0.85),
    ("a luz viaja a trezentos mil quilômetros por segundo no vácuo", "física_óptica", 0.95),
]

MODULO_C5_QUANTICA = [
    ("mecânica quântica",        "física_teorica/quantica", 0.9),
    ("quantum",                  "física_teorica/quantica", 0.9),
    ("elétron",                  "física_teorica/quantica", 0.85),
    ("orbital",                  "física_teorica/quantica", 0.85),
    ("princípio da incerteza",   "física_teorica/quantica", 0.9),
    ("superposição",             "física_teorica/quantica", 0.9),
    ("emaranhamento",            "física_teorica/quantica", 0.9),
    ("colapso da função de onda","física_teorica/quantica", 0.9),
    ("relatividade",             "física_teorica/relatividade", 0.9),
    ("espaço-tempo",             "física_teorica/relatividade", 0.9),
    ("dilatação do tempo",       "física_teorica/relatividade", 0.9),
    ("o princípio da incerteza de Heisenberg limita a precisão simultânea de posição e momento", "física_teorica/quantica", 0.95),
    ("a teoria da relatividade especial de Einstein relaciona massa e energia via E igual a mc ao quadrado", "física_teorica/relatividade", 0.95),
]

# ── Módulo D — Química ─────────────────────────────────────────────────────────

MODULO_D1_ATOMO = [
    ("átomo",            "química_átomo", 0.9),
    ("próton",           "química_átomo", 0.9),
    ("nêutron",          "química_átomo", 0.9),
    ("elétron",          "química_átomo", 0.9),
    ("núcleo",           "química_átomo", 0.85),
    ("camada eletrônica","química_átomo", 0.85),
    ("número atômico",   "química_átomo", 0.9),
    ("massa atômica",    "química_átomo", 0.85),
    ("isótopo",          "química_átomo", 0.85),
    ("íon",              "química_átomo", 0.85),
    ("ligação química",  "química_átomo", 0.9),
    ("o átomo é formado por prótons nêutrons e elétrons", "química_átomo", 0.95),
]

MODULO_D2_PERIODICA = [
    ("tabela periódica",     "química_átomo", 0.95),
    ("período",              "química_átomo", 0.8),
    ("grupo",                "química_átomo", 0.8),
    ("metal",                "química_átomo", 0.8),
    ("não-metal",            "química_átomo", 0.8),
    ("metalóide",            "química_átomo", 0.8),
    ("eletronegatividade",   "química_átomo", 0.85),
    ("raio atômico",         "química_átomo", 0.8),
    ("valência",             "química_átomo", 0.85),
    ("nobre",                "química_átomo", 0.8),
    ("a tabela periódica organiza os elementos por número atômico crescente", "química_átomo", 0.9),
]

MODULO_D3_ORGANICA = [
    ("carbono",              "química_orgânica", 0.95),
    ("hidrocarboneto",       "química_orgânica", 0.9),
    ("benzeno",              "química_orgânica", 0.85),
    ("álcool",               "química_orgânica", 0.85),
    ("ácido orgânico",       "química_orgânica", 0.85),
    ("éster",                "química_orgânica", 0.8),
    ("amina",                "química_orgânica", 0.8),
    ("polímero",             "química_orgânica", 0.85),
    ("cadeia carbônica",     "química_orgânica", 0.9),
    ("isomeria",             "química_orgânica", 0.85),
    ("o carbono forma quatro ligações covalentes e é base da química orgânica", "química_orgânica", 0.95),
]

MODULO_D4_INORGANICA = [
    ("óxido",                "química_inorgânica", 0.85),
    ("ácido",                "química_inorgânica", 0.85),
    ("base",                 "química_inorgânica", 0.85),
    ("sal",                  "química_inorgânica", 0.85),
    ("pH",                   "química_inorgânica", 0.9),
    ("neutralização",        "química_inorgânica", 0.85),
    ("reação de oxidorredução", "química_inorgânica", 0.9),
    ("precipitação",         "química_inorgânica", 0.8),
    ("o pH sete é neutro abaixo é ácido e acima é básico", "química_inorgânica", 0.9),
]

MODULO_D5_BIOQUIMICA = [
    ("proteína",             "bioquímica", 0.9),
    ("aminoácido",           "bioquímica", 0.9),
    ("DNA",                  "bioquímica", 0.95),
    ("RNA",                  "bioquímica", 0.9),
    ("gene",                 "bioquímica", 0.9),
    ("enzima",               "bioquímica", 0.9),
    ("ATP",                  "bioquímica", 0.9),
    ("metabolismo",          "bioquímica", 0.85),
    ("lipídio",              "bioquímica", 0.8),
    ("carboidrato",          "bioquímica", 0.8),
    ("o DNA armazena a informação genética em sequências de bases nitrogenadas", "bioquímica", 0.95),
    ("as enzimas são catalisadores biológicos que aceleram reações químicas", "bioquímica", 0.9),
]

# ── Módulo E — Engenharia ──────────────────────────────────────────────────────

MODULO_E1_ELETRONICA = [
    ("transistor",           "engenharia_eletrônica", 0.95),
    ("diodo",                "engenharia_eletrônica", 0.9),
    ("amplificador",         "engenharia_eletrônica", 0.9),
    ("circuito integrado",   "engenharia_eletrônica", 0.9),
    ("microcontrolador",     "engenharia_eletrônica", 0.9),
    ("sinal analógico",      "engenharia_eletrônica", 0.85),
    ("sinal digital",        "engenharia_eletrônica", 0.85),
    ("conversor ADC",        "engenharia_eletrônica", 0.85),
    ("modulação",            "engenharia_eletrônica", 0.8),
    ("o transistor é o componente fundamental dos circuitos digitais modernos", "engenharia_eletrônica", 0.95),
]

MODULO_E2_ELETRICA = [
    ("resistência elétrica", "engenharia_elétrica", 0.9),
    ("circuito elétrico",    "engenharia_elétrica", 0.9),
    ("lei de Ohm",           "engenharia_elétrica", 0.95),
    ("potência elétrica",    "engenharia_elétrica", 0.9),
    ("motor elétrico",       "engenharia_elétrica", 0.85),
    ("gerador",              "engenharia_elétrica", 0.85),
    ("transformador",        "engenharia_elétrica", 0.85),
    ("corrente alternada",   "engenharia_elétrica", 0.85),
    ("corrente contínua",    "engenharia_elétrica", 0.85),
    ("a lei de Ohm diz que tensão é resistência vezes corrente", "engenharia_elétrica", 0.95),
]

MODULO_E3_MECANICA = [
    ("material",             "engenharia_mecânica_eng", 0.85),
    ("tensão mecânica",      "engenharia_mecânica_eng", 0.9),
    ("deformação",           "engenharia_mecânica_eng", 0.85),
    ("elasticidade",         "engenharia_mecânica_eng", 0.85),
    ("estática",             "engenharia_mecânica_eng", 0.85),
    ("dinâmica estrutural",  "engenharia_mecânica_eng", 0.85),
    ("fluido",               "engenharia_mecânica_eng", 0.8),
    ("viscosidade",          "engenharia_mecânica_eng", 0.8),
    ("fadiga",               "engenharia_mecânica_eng", 0.8),
]

MODULO_E4_CONTROLE = [
    ("sistema de controle",  "engenharia_controle", 0.9),
    ("realimentação",        "engenharia_controle", 0.9),
    ("PID",                  "engenharia_controle", 0.95),
    ("controlador",          "engenharia_controle", 0.9),
    ("planta",               "engenharia_controle", 0.8),
    ("erro",                 "engenharia_controle", 0.85),
    ("estabilidade",         "engenharia_controle", 0.85),
    ("função de transferência", "engenharia_controle", 0.9),
    ("o controlador PID usa proporcional integral e derivativo para minimizar o erro", "engenharia_controle", 0.95),
]

# ── Módulo F — Programação ─────────────────────────────────────────────────────

MODULO_F1_FUNDAMENTOS = [
    ("variável",             "programação_fundamentos", 0.9),
    ("tipo de dado",         "programação_fundamentos", 0.9),
    ("função",               "programação_fundamentos", 0.9),
    ("loop",                 "programação_fundamentos", 0.9),
    ("condicional",          "programação_fundamentos", 0.9),
    ("recursão",             "programação_fundamentos", 0.85),
    ("escopo",               "programação_fundamentos", 0.85),
    ("compilador",           "programação_fundamentos", 0.85),
    ("interpretador",        "programação_fundamentos", 0.85),
    ("depuração",            "programação_fundamentos", 0.8),
    ("uma variável armazena um valor que pode mudar durante a execução do programa", "programação_fundamentos", 0.9),
    ("a recursão é quando uma função chama a si mesma para resolver um subproblema menor", "programação_fundamentos", 0.9),
]

MODULO_F2_ALGORITMOS = [
    ("algoritmo",            "programação_algoritmos", 0.95),
    ("complexidade",         "programação_algoritmos", 0.9),
    ("big O",                "programação_algoritmos", 0.9),
    ("ordenação",            "programação_algoritmos", 0.9),
    ("busca",                "programação_algoritmos", 0.9),
    ("quicksort",            "programação_algoritmos", 0.9),
    ("mergesort",            "programação_algoritmos", 0.85),
    ("busca binária",        "programação_algoritmos", 0.9),
    ("programação dinâmica", "programação_algoritmos", 0.9),
    ("algoritmo guloso",     "programação_algoritmos", 0.85),
    ("o quicksort divide o array em partes menores e maiores que o pivô recursivamente", "programação_algoritmos", 0.95),
    ("a busca binária encontra elementos em tempo logarítmico em arrays ordenados", "programação_algoritmos", 0.9),
]

MODULO_F3_ESTRUTURAS = [
    ("estrutura de dados",   "programação_estruturas_dado", 0.95),
    ("array",                "programação_estruturas_dado", 0.9),
    ("lista ligada",         "programação_estruturas_dado", 0.9),
    ("pilha",                "programação_estruturas_dado", 0.9),
    ("fila",                 "programação_estruturas_dado", 0.9),
    ("árvore binária",       "programação_estruturas_dado", 0.9),
    ("árvore B",             "programação_estruturas_dado", 0.85),
    ("grafo",                "programação_estruturas_dado", 0.9),
    ("hash table",           "programação_estruturas_dado", 0.9),
    ("heap",                 "programação_estruturas_dado", 0.85),
    ("uma árvore binária de busca mantém elementos à esquerda menores e à direita maiores", "programação_estruturas_dado", 0.95),
]

MODULO_F4_BACKEND = [
    ("API REST",             "programação_backend", 0.95),
    ("servidor",             "programação_backend", 0.9),
    ("banco de dados",       "programação_backend", 0.9),
    ("SQL",                  "programação_backend", 0.9),
    ("autenticação",         "programação_backend", 0.85),
    ("middleware",           "programação_backend", 0.85),
    ("cache",                "programação_backend", 0.85),
    ("microserviço",         "programação_backend", 0.9),
    ("WebSocket",            "programação_backend", 0.9),
    ("uma API REST usa verbos HTTP para criar ler atualizar e deletar recursos", "programação_backend", 0.95),
]

MODULO_F5_FRONTEND = [
    ("HTML",                 "programação_frontend", 0.9),
    ("CSS",                  "programação_frontend", 0.9),
    ("JavaScript",           "programação_frontend", 0.9),
    ("DOM",                  "programação_frontend", 0.9),
    ("componente",           "programação_frontend", 0.85),
    ("estado",               "programação_frontend", 0.85),
    ("reatividade",          "programação_frontend", 0.85),
    ("responsividade",       "programação_frontend", 0.8),
    ("o DOM é a representação da página HTML como uma árvore de objetos manipulável por JavaScript", "programação_frontend", 0.9),
]

MODULO_F6_SISTEMAS = [
    ("sistema operacional",  "programação_sistemas_op", 0.95),
    ("processo",             "programação_sistemas_op", 0.9),
    ("thread",               "programação_concorr", 0.9),
    ("mutex",                "programação_concorr", 0.9),
    ("deadlock",             "programação_concorr", 0.9),
    ("memória virtual",      "programação_sistemas_op", 0.9),
    ("paginação",            "programação_sistemas_op", 0.85),
    ("sistema de arquivos",  "programação_sistemas_op", 0.85),
    ("um deadlock ocorre quando dois processos esperam um pelo recurso do outro indefinidamente", "programação_concorr", 0.95),
]

MODULO_F7_IA_ML = [
    ("inteligência artificial", "programação_ia_ml", 0.95),
    ("aprendizado de máquina",  "programação_ia_ml", 0.95),
    ("rede neural artificial",  "programação_ia_ml", 0.95),
    ("neurônio artificial",     "programação_ia_ml", 0.9),
    ("backpropagation",         "programação_ia_ml", 0.9),
    ("gradiente",               "programação_ia_ml", 0.9),
    ("overfitting",             "programação_ia_ml", 0.85),
    ("regularização",           "programação_ia_ml", 0.85),
    ("transformers",            "programação_ia_ml", 0.9),
    ("atenção",                 "programação_ia_ml", 0.9),
    ("embedding",               "programação_ia_ml", 0.9),
    ("o backpropagation calcula gradientes para ajustar pesos da rede neural via descida do gradiente", "programação_ia_ml", 0.95),
    ("transformers usam mecanismos de atenção para capturar dependências de longo alcance", "programação_ia_ml", 0.9),
]

# ── Mapa de módulos ────────────────────────────────────────────────────────────

GRUPO_NOMES = {
    "A": "FALA",
    "B": "MATEMÁTICA",
    "C": "FÍSICA",
    "D": "QUÍMICA",
    "E": "ENGENHARIA",
    "F": "PROGRAMAÇÃO",
}

MÓDULOS = {
    "A1": ("Fonador",        MODULO_A1_FONADOR,      "A"),
    "A2": ("Fonemas PT-BR",  MODULO_A2_FONEMAS,      "A"),
    "A3": ("Prosódia",       MODULO_A3_PROSÓDIA,     "A"),
    "A4": ("Sílabas",        MODULO_A4_SÍLABAS,      "A"),
    "A5": ("Fluência",       MODULO_A5_FLUÊNCIA,     "A"),
    "A6": ("Variações",      MODULO_A6_VARIAÇÕES,    "A"),
    "B1": ("Números",        MODULO_B1_NÚMEROS,      "B"),
    "B2": ("Operações",      MODULO_B2_OPERAÇÕES,    "B"),
    "B3": ("Geometria",      MODULO_B3_GEOMETRIA,    "B"),
    "B4": ("Álgebra",        MODULO_B4_ÁLGEBRA,      "B"),
    "B5": ("Probabilidade",  MODULO_B5_PROBABILIDADE,"B"),
    "B6": ("Lógica",         MODULO_B6_LÓGICA,       "B"),
    "B7": ("Vocabulário Mat",MODULO_B7_VOCABULÁRIO,  "B"),
    "C1": ("Mecânica",       MODULO_C1_MECANICA,     "C"),
    "C2": ("Termodinâmica",  MODULO_C2_TERMODINAMICA,"C"),
    "C3": ("Eletromag",      MODULO_C3_ELETROMAG,    "C"),
    "C4": ("Ondas/Óptica",   MODULO_C4_ONDAS,        "C"),
    "C5": ("Quântica",       MODULO_C5_QUANTICA,     "C"),
    "D1": ("Átomo",          MODULO_D1_ATOMO,        "D"),
    "D2": ("Periódica",      MODULO_D2_PERIODICA,    "D"),
    "D3": ("Orgânica",       MODULO_D3_ORGANICA,     "D"),
    "D4": ("Inorgânica",     MODULO_D4_INORGANICA,   "D"),
    "D5": ("Bioquímica",     MODULO_D5_BIOQUIMICA,   "D"),
    "E1": ("Eletrônica",     MODULO_E1_ELETRONICA,   "E"),
    "E2": ("Elétrica",       MODULO_E2_ELETRICA,     "E"),
    "E3": ("Mec. Estrutural",MODULO_E3_MECANICA,     "E"),
    "E4": ("Controle",       MODULO_E4_CONTROLE,     "E"),
    "F1": ("Fundamentos",    MODULO_F1_FUNDAMENTOS,  "F"),
    "F2": ("Algoritmos",     MODULO_F2_ALGORITMOS,   "F"),
    "F3": ("Estruturas",     MODULO_F3_ESTRUTURAS,   "F"),
    "F4": ("Backend",        MODULO_F4_BACKEND,      "F"),
    "F5": ("Frontend",       MODULO_F5_FRONTEND,     "F"),
    "F6": ("Sistemas/Conc.", MODULO_F6_SISTEMAS,     "F"),
    "F7": ("IA/ML",          MODULO_F7_IA_ML,        "F"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

ACK_EVENTS = {"learn_ack", "frase_ack", "associate_ack", "train_ack", "ok", "pong"}

# Eventos V3.2 que devem ser ignorados silenciosamente (não são ACK nem erro)
_SKIP_EVENTS = {"thinking", "neural_status", "voz_params", "pensamento_espontaneo",
                "curiosidade", "curiosidade_espontanea", "reacao_emocional"}

async def aguardar_ack(ws, timeout=3.0):
    """Drena mensagens do servidor até encontrar um ACK ou expirar o timeout.
    V3.2: ignora "thinking" e eventos de telemetria. Retorna True em timeout silencioso."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        remaining = timeout - (time.time() - t0)
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 0.4))
            d = json.loads(msg)
            ev = d.get("event", "")
            if ev in ACK_EVENTS:
                return True
            if ev == "erro":
                return False
            if ev in _SKIP_EVENTS:
                continue  # V3.2: thinking/neural_status — ignora e continua
        except asyncio.TimeoutError:
            break  # nenhuma msg chegou — ok, segue
        except Exception:
            raise  # propaga ConnectionClosed para o chamador
    return True  # timeout silencioso


def _novo_id() -> str:
    """Gera UUID v4 curto para rastreamento de mensagens (V3.2 message_id)."""
    return str(_uuid_mod.uuid4())[:8]

async def enviar_palavra(ws, texto, contexto, valence):
    """V3.2: usa DualTrainer (áudio FFT + texto) quando disponível; senão só texto."""
    if ESPEAK_OK and DualTrainer:
        dt = DualTrainer(ws)
        await dt.palavra(texto.lower(), valencia=valence, contexto=contexto)
        return
    # Modo texto (fallback)
    await ws.send(json.dumps({
        "action":  "learn",
        "word":    texto,
        "context": contexto,
        "valence": valence,
        "id":      _novo_id(),
    }))
    await aguardar_ack(ws)

async def enviar_frase(ws, frase, contexto, valence):
    """V3.2: usa DualTrainer (áudio FFT + texto) quando disponível; senão só texto."""
    palavras = frase.split()
    if ESPEAK_OK and DualTrainer:
        dt = DualTrainer(ws)
        await dt.frase(palavras, valencia=valence, contexto=contexto)
        return
    # Modo texto (fallback)
    for w in palavras:
        await ws.send(json.dumps({
            "action":  "learn",
            "word":    w,
            "context": contexto,
            "valence": valence,
            "id":      _novo_id(),
        }))
        await aguardar_ack(ws)
        await asyncio.sleep(0.02)
    await ws.send(json.dumps({"action": "learn_frase", "words": palavras}))
    await aguardar_ack(ws, timeout=2.0)
    for i in range(len(palavras) - 1):
        await ws.send(json.dumps({
            "action": "associate",
            "w1": palavras[i], "w2": palavras[i + 1],
            "weight": round(valence * 0.6, 2),
        }))
        await asyncio.sleep(0.015)

async def sweep_stdp(ws, epocas=1):
    await ws.send(json.dumps({"action": "train", "epochs": epocas}))
    await aguardar_ack(ws, timeout=5.0)

async def executar_módulo(ws, id_mod, nome, itens, group):
    """Executa os 3 passes de spaced-repetition para um módulo."""
    tem_frases = any(len(t.split()) > 1 for t, _, _ in itens)
    print(f"\n  ┌─ [{id_mod}] {nome} — {len(itens)} itens "
          f"({'misto' if tem_frases else 'palavras'}) ─────────")

    if SECO:
        for texto, ctx, val in itens:
            print(f"  │  {texto:<40} ctx={ctx}  val={val:+.1f}")
        print(f"  └─ [seco] {len(itens)} itens listados")
        return

    total_enviado = 0

    for passe, (threshold, delay, label) in enumerate([
        (0.0,              DELAY_P1, "APRESENTAÇÃO"),
        (PASS2_THRESHOLD,  DELAY_P2, "REFORÇO"),
        (PASS3_THRESHOLD,  DELAY_P3, "CONSOLIDAÇÃO"),
    ], start=1):
        filtrado = [(t, c, v) for t, c, v in itens if abs(v) >= threshold]
        if not filtrado:
            continue

        reps = passe  # pass 1 = 1×, pass 2 = 2×, pass 3 = 3×
        total_passe = len(filtrado) * reps * REPS_PADRAO
        print(f"  │  Pass {passe} [{label}] — {len(filtrado)} itens × {reps}×{REPS_PADRAO} = {total_passe} envios")

        for _ in range(reps * REPS_PADRAO):
            for texto, contexto, valence in filtrado:
                if len(texto.split()) > 1:
                    await enviar_frase(ws, texto, contexto, valence)
                else:
                    await enviar_palavra(ws, texto, contexto, valence)
                total_enviado += 1
                await asyncio.sleep(delay)

    await sweep_stdp(ws, epocas=2)
    print(f"  └─ [{id_mod}] concluido — {total_enviado} envios + STDP sweep")

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    # Seleciona módulos conforme filtro
    modulos_selecionados = []
    for id_mod, (nome, itens, grupo) in MÓDULOS.items():
        if MODULO_ALVO is None:
            modulos_selecionados.append((id_mod, nome, itens, grupo))
        elif MODULO_ALVO == id_mod:
            modulos_selecionados.append((id_mod, nome, itens, grupo))
        elif MODULO_ALVO in ("A", "B", "C", "D", "E", "F") and grupo == MODULO_ALVO:
            modulos_selecionados.append((id_mod, nome, itens, grupo))

    total_itens = sum(len(its) for _, _, its, _ in modulos_selecionados)

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║     TREINAR SELENE — Fala + Mat + Física + Quím + Eng + Prog ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Servidor : {WS_URL:<48}║")
    print(f"║  Módulos  : {len(modulos_selecionados):<3}  |  Itens: {total_itens:<5}  |  {'DRY-RUN' if SECO else 'LIVE':^10}          ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if SECO:
        print("\n[DRY-RUN] Listando itens sem conectar ao servidor...\n")
        grupo_atual = None
        for id_mod, nome, itens, grupo in modulos_selecionados:
            if grupo != grupo_atual:
                grp_nome = GRUPO_NOMES.get(grupo, grupo)
                print(f"\n{'═'*60}")
                print(f"  MÓDULO {grupo} — {grp_nome}")
                print(f"{'═'*60}")
                grupo_atual = grupo
            await executar_módulo(None, id_mod, nome, itens, grupo)
        total = sum(len(its) for _, _, its, _ in modulos_selecionados)
        print(f"\n  📊 Total: {total} itens em {len(modulos_selecionados)} módulos")
        return

    # ── Modo de aprendizado V3.2 ──────────────────────────────────────────────────
    modo = "DUAL (áudio FFT + texto)" if ESPEAK_OK else "TEXTO (espeak não disponível)"
    print(f"  🎙  Modo de aprendizado: {modo}")
    if not ESPEAK_OK:
        print("      → Para ativar modo audiovisual instale: espeak-ng + pip install numpy scipy")

    # ── Conexão com ping desativado (local) + retry automático ──────────────────
    # ping_interval=None: o cliente não envia pings; servidor V3.2 envia heartbeat 30s.
    # A biblioteca websockets responde pong automaticamente quando recebe ping.
    # close_timeout aumentado para dar tempo ao servidor de responder no shutdown.
    WS_KWARGS = dict(ping_interval=None, close_timeout=5, max_size=2**20)
    MAX_TENTATIVAS = 5

    print(f"\n  Conectando em {WS_URL}...")

    # ── Verificação de estágio ontogenético (V3.2) ────────────────────────────────
    try:
        async with websockets.connect(WS_URL, ping_interval=None, close_timeout=3) as _ws:
            await _ws.send(json.dumps({"action": "ontogeny_status"}))
            raw = await asyncio.wait_for(_ws.recv(), timeout=2.0)
            d = json.loads(raw)
            stage = d.get("stage", "?")
            vocab = d.get("vocab_n", "?")
            print(f"  🧠 Ontogenia: estágio={stage} | vocab={vocab}")
    except Exception:
        pass  # Selene não respondeu — não bloqueia o treinamento

    grupo_atual   = None
    t_inicio      = time.time()
    mod_idx       = 0   # checkpoint: retoma do módulo que falhou

    while mod_idx < len(modulos_selecionados):
        tentativa = 0
        conectado = False

        while tentativa < MAX_TENTATIVAS:
            try:
                async with websockets.connect(WS_URL, **WS_KWARGS) as ws:
                    if tentativa == 0 and mod_idx == 0:
                        print("  ✅ Conectado!\n")
                    else:
                        print(f"  🔄 Reconectado (mod {mod_idx+1}/{len(modulos_selecionados)}).\n")
                    conectado = True

                    while mod_idx < len(modulos_selecionados):
                        id_mod, nome, itens, grupo = modulos_selecionados[mod_idx]

                        if grupo != grupo_atual:
                            if grupo_atual is not None:
                                print(f"\n  Pausa inter-grupo + STDP consolidação...")
                                await sweep_stdp(ws, epocas=3)
                                await asyncio.sleep(PAUSA_MODULO * 2)
                            grp_nome = GRUPO_NOMES.get(grupo, grupo)
                            print(f"\n{'='*60}")
                            print(f"  MODULO {grupo} — {grp_nome}")
                            print(f"{'='*60}")
                            grupo_atual = grupo

                        await executar_módulo(ws, id_mod, nome, itens, grupo)
                        mod_idx += 1
                        await asyncio.sleep(PAUSA_MODULO)

                    # Todos os módulos concluídos
                    print("\n  Consolidacao final (STDP 5 epocas)...")
                    await sweep_stdp(ws, epocas=5)
                    break  # sai do while tentativa

            except (ConnectionRefusedError, OSError):
                print(f"\n  ❌ Não foi possível conectar em {WS_URL}")
                print("     Certifique-se de que a Selene está rodando (cargo run --release)")
                sys.exit(1)
            except KeyboardInterrupt:
                print("\n  ⚠️  Treinamento interrompido pelo usuário.")
                return
            except Exception as e:
                tentativa += 1
                if tentativa < MAX_TENTATIVAS:
                    espera = min(2 ** tentativa, 16)
                    print(f"\n  ⚠️  Conexão perdida ({e.__class__.__name__}). "
                          f"Tentativa {tentativa}/{MAX_TENTATIVAS} em {espera}s...")
                    await asyncio.sleep(espera)
                else:
                    print(f"\n  ❌ Falha após {MAX_TENTATIVAS} tentativas: {e}")
                    sys.exit(1)

        if conectado:
            break  # tudo ok, sai do loop externo

    elapsed = time.time() - t_inicio
    print(f"\n{'='*64}")
    print(f"  TREINAMENTO CONCLUIDO")
    print(f"  Tempo total  : {elapsed:.1f}s")
    print(f"  Modulos      : {len(modulos_selecionados)}")
    print(f"  Itens unicos : {total_itens}")
    print(f"{'='*64}\n")

if __name__ == "__main__":
    asyncio.run(main())
