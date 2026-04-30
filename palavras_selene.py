#!/usr/bin/env python3
# palavras_selene.py — Treinamento de vocabulário para Selene Brain 2.0
#
# Ensina Selene palavras do Português Brasileiro organizadas por categoria:
#   - Fase  0: Monossílabas comuns     (pé, mão, sol, mar, sim, não...)
#   - Fase  1: Corpo humano            (cabeça, olho, nariz, boca...)
#   - Fase  2: Família                 (pai, mãe, filho, avô...)
#   - Fase  3: Cores                   (vermelho, azul, verde...)
#   - Fase  4: Números                 (um, dois, três, dez, cem...)
#   - Fase  5: Animais                 (gato, cachorro, pássaro...)
#   - Fase  6: Comida e bebida         (arroz, feijão, pão, água...)
#   - Fase  7: Casa e objetos          (mesa, cadeira, porta, cama...)
#   - Fase  8: Natureza                (árvore, rio, montanha, chuva...)
#   - Fase  9: Verbos básicos          (ser, ter, fazer, amar, comer...)
#   - Fase 10: Emoções e sentimentos   (alegria, tristeza, medo, amor...)
#   - Fase 11: Vestuário               (camisa, calça, sapato, chapéu...)
#   - Fase 12: Tempo e calendário      (dia, mês, ano, manhã, hoje...)
#   - Fase 13: Lugares e construções   (cidade, escola, rua, parque...)
#   - Fase 14: Adjetivos               (bom, grande, bonito, rápido...)
#   - Fase 15: Pronomes e determinantes(eu, você, meu, este, que...)
#   - Fase 16: Perguntas e saudações   (oi, obrigado, como, onde...)
#   - Fase 17: Polissílabas comuns     (computador, borboleta, universo...)
#
# Loop de grounding fonético (idêntico ao baba_selene.py):
#   espeak-ng sintetiza "gato"
#     → WAV → FFT frames → learn_audio_fft  (Selene ouve a palavra)
#     → grounding_fonetico{"grafema":"gato","letras":["g","a","t","o"]}
#     → learn + learn_frase                  (Selene lê a palavra)
#
# Requisitos:
#   pip install websockets scipy numpy
#   espeak-ng instalado e no PATH
#
# Uso:
#   python palavras_selene.py                  # todas as fases
#   python palavras_selene.py --fase 1         # só corpo humano
#   python palavras_selene.py --fase 5         # só animais
#   python palavras_selene.py --rep 20         # 20 repetições por palavra
#   python palavras_selene.py --pausa 0.005    # mais rápido
#   python palavras_selene.py --host 192.168.1.10

import asyncio
import json
import os
import random
import subprocess
import sys
import tempfile
import time

try:
    import numpy as np
    from scipy.io import wavfile
except ImportError:
    print("Instale: pip install numpy scipy")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("Instale: pip install websockets")
    sys.exit(1)

# ─── Configuração ─────────────────────────────────────────────────────────────

HOST        = "127.0.0.1"
PORTA       = 3030
WS_URL      = f"ws://{HOST}:{PORTA}/selene"
FRAME_MS    = 25
PAUSA       = 0.008
SILENCIO_DB = -40.0
MAX_BINS    = 128
SAMPLE_RATE = 22050
REPETICOES  = 20    # repetições por palavra
FASE_ALVO   = None  # None = todas
VARIACAO    = False  # --variacao: pitch/speed aleatórios + ruído gaussiano

for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--host" and i + 1 < len(sys.argv):
        HOST = sys.argv[i+1]
        WS_URL = f"ws://{HOST}:{PORTA}/selene"
    if arg == "--pausa" and i + 1 < len(sys.argv):
        try: PAUSA = float(sys.argv[i+1])
        except: pass
    if arg == "--rep" and i + 1 < len(sys.argv):
        try: REPETICOES = int(sys.argv[i+1])
        except: pass
    if arg == "--fase" and i + 1 < len(sys.argv):
        try: FASE_ALVO = int(sys.argv[i+1])
        except: pass
    if arg == "--variacao":
        VARIACAO = True

# ─── Currículo de vocabulário do Português Brasileiro ─────────────────────────
#
# Progressão pedagógica:
#   0  Monossílabas comuns      — palavras de 1 sílaba de alta frequência
#   1  Corpo humano             — vocabulário corporal essencial
#   2  Família                  — relações familiares
#   3  Cores                    — espectro cromático completo
#   4  Números                  — cardinal e ordinal
#   5  Animais                  — domésticos, selvagens, insetos, aquáticos
#   6  Comida e bebida          — alimentos, frutas, legumes, bebidas
#   7  Casa e objetos           — cômodos, móveis, utensílios, tecnologia
#   8  Natureza                 — flora, fauna, clima, geografia
#   9  Verbos básicos           — ações fundamentais do cotidiano
#  10  Emoções e sentimentos    — estados afetivos e psicológicos
#  11  Vestuário e acessórios   — roupas, calçados, adornos
#  12  Tempo e calendário       — unidades, dias, meses, estações
#  13  Lugares e construções    — espaços urbanos e naturais
#  14  Adjetivos comuns         — qualidades e características
#  15  Pronomes e determinantes — estrutura gramatical essencial
#  16  Perguntas e saudações    — comunicação social básica
#  17  Polissílabas comuns      — palavras mais longas de alta frequência

CURRICULO = {
    0: {
        "nome": "Monossílabas comuns",
        "descricao": "Palavras de uma sílaba de altíssima frequência no PB",
        "itens": [
            # Pronomes e artigos
            ("mono_eu",   "eu",   "pt-br"), ("mono_tu",  "tu",  "pt-br"),
            ("mono_o",    "o",    "pt-br"), ("mono_a",   "a",   "pt-br"),
            ("mono_os",   "os",   "pt-br"), ("mono_as",  "as",  "pt-br"),
            ("mono_um",   "um",   "pt-br"), ("mono_uns", "uns", "pt-br"),
            # Preposições
            ("mono_em",   "em",   "pt-br"), ("mono_de",  "de",  "pt-br"),
            ("mono_do",   "do",   "pt-br"), ("mono_da",  "da",  "pt-br"),
            ("mono_no",   "no",   "pt-br"), ("mono_na",  "na",  "pt-br"),
            ("mono_com",  "com",  "pt-br"), ("mono_por", "por", "pt-br"),
            ("mono_sem",  "sem",  "pt-br"), ("mono_sob", "sob", "pt-br"),
            ("mono_ao",   "ao",   "pt-br"), ("mono_dos", "dos", "pt-br"),
            # Advérbios e conectivos
            ("mono_sim",  "sim",  "pt-br"), ("mono_nao", "não", "pt-br"),
            ("mono_ja",   "já",   "pt-br"), ("mono_bem", "bem", "pt-br"),
            ("mono_mal",  "mal",  "pt-br"), ("mono_mais","mais","pt-br"),
            ("mono_so",   "só",   "pt-br"), ("mono_la",  "lá",  "pt-br"),
            ("mono_ca",   "cá",   "pt-br"), ("mono_qui", "qui", "pt-br"),
            # Verbos monossílabos
            ("mono_ir",   "ir",   "pt-br"), ("mono_ser", "ser", "pt-br"),
            ("mono_ter",  "ter",  "pt-br"), ("mono_dar", "dar", "pt-br"),
            ("mono_ver",  "ver",  "pt-br"), ("mono_ler", "ler", "pt-br"),
            ("mono_rir",  "rir",  "pt-br"), ("mono_por", "pôr", "pt-br"),
            # Substantivos monossílabos
            ("mono_pe",   "pé",   "pt-br"), ("mono_mao", "mão", "pt-br"),
            ("mono_mae",  "mãe",  "pt-br"), ("mono_pai", "pai", "pt-br"),
            ("mono_sol",  "sol",  "pt-br"), ("mono_mar", "mar", "pt-br"),
            ("mono_ceu",  "céu",  "pt-br"), ("mono_cao", "cão", "pt-br"),
            ("mono_mel",  "mel",  "pt-br"), ("mono_sal", "sal", "pt-br"),
            ("mono_flor", "flor", "pt-br"), ("mono_dor", "dor", "pt-br"),
            ("mono_voz",  "voz",  "pt-br"), ("mono_luz", "luz", "pt-br"),
            ("mono_fim",  "fim",  "pt-br"), ("mono_vez", "vez", "pt-br"),
            ("mono_vez",  "vez",  "pt-br"), ("mono_mes", "mês", "pt-br"),
            ("mono_no",   "nó",   "pt-br"), ("mono_fe",  "fé",  "pt-br"),
        ],
    },
    1: {
        "nome": "Corpo humano",
        "descricao": "Partes do corpo — vocabulário corporal essencial",
        "itens": [
            ("corp_cabeca",      "cabeça",      "pt-br"),
            ("corp_cabelo",      "cabelo",      "pt-br"),
            ("corp_testa",       "testa",       "pt-br"),
            ("corp_sobrancelha", "sobrancelha", "pt-br"),
            ("corp_olho",        "olho",        "pt-br"),
            ("corp_pestana",     "pestana",     "pt-br"),
            ("corp_ouvido",      "ouvido",      "pt-br"),
            ("corp_orelha",      "orelha",      "pt-br"),
            ("corp_nariz",       "nariz",       "pt-br"),
            ("corp_boca",        "boca",        "pt-br"),
            ("corp_labio",       "lábio",       "pt-br"),
            ("corp_dente",       "dente",       "pt-br"),
            ("corp_lingua",      "língua",      "pt-br"),
            ("corp_queixo",      "queixo",      "pt-br"),
            ("corp_bochecha",    "bochecha",    "pt-br"),
            ("corp_rosto",       "rosto",       "pt-br"),
            ("corp_pescoco",     "pescoço",     "pt-br"),
            ("corp_nuca",        "nuca",        "pt-br"),
            ("corp_ombro",       "ombro",       "pt-br"),
            ("corp_braco",       "braço",       "pt-br"),
            ("corp_cotovelo",    "cotovelo",    "pt-br"),
            ("corp_pulso",       "pulso",       "pt-br"),
            ("corp_mao",         "mão",         "pt-br"),
            ("corp_dedo",        "dedo",        "pt-br"),
            ("corp_unha",        "unha",        "pt-br"),
            ("corp_peito",       "peito",       "pt-br"),
            ("corp_costas",      "costas",      "pt-br"),
            ("corp_barriga",     "barriga",     "pt-br"),
            ("corp_cintura",     "cintura",     "pt-br"),
            ("corp_quadril",     "quadril",     "pt-br"),
            ("corp_perna",       "perna",       "pt-br"),
            ("corp_joelho",      "joelho",      "pt-br"),
            ("corp_canela",      "canela",      "pt-br"),
            ("corp_tornozelo",   "tornozelo",   "pt-br"),
            ("corp_pe",          "pé",          "pt-br"),
            ("corp_calcanhar",   "calcanhar",   "pt-br"),
            ("corp_coracao",     "coração",     "pt-br"),
            ("corp_pulmao",      "pulmão",      "pt-br"),
            ("corp_estomago",    "estômago",    "pt-br"),
            ("corp_figado",      "fígado",      "pt-br"),
            ("corp_rim",         "rim",         "pt-br"),
            ("corp_osso",        "osso",        "pt-br"),
            ("corp_musculo",     "músculo",     "pt-br"),
            ("corp_pele",        "pele",        "pt-br"),
            ("corp_sangue",      "sangue",      "pt-br"),
            ("corp_veia",        "veia",        "pt-br"),
            ("corp_nervo",       "nervo",       "pt-br"),
            ("corp_cerebro",     "cérebro",     "pt-br"),
        ],
    },
    2: {
        "nome": "Família e relações",
        "descricao": "Membros da família e relações pessoais",
        "itens": [
            ("fam_pai",       "pai",       "pt-br"),
            ("fam_mae",       "mãe",       "pt-br"),
            ("fam_filho",     "filho",     "pt-br"),
            ("fam_filha",     "filha",     "pt-br"),
            ("fam_irmao",     "irmão",     "pt-br"),
            ("fam_irma",      "irmã",      "pt-br"),
            ("fam_avo",       "avô",       "pt-br"),
            ("fam_avo_f",     "avó",       "pt-br"),
            ("fam_tio",       "tio",       "pt-br"),
            ("fam_tia",       "tia",       "pt-br"),
            ("fam_primo",     "primo",     "pt-br"),
            ("fam_prima",     "prima",     "pt-br"),
            ("fam_marido",    "marido",    "pt-br"),
            ("fam_esposa",    "esposa",    "pt-br"),
            ("fam_sobrinho",  "sobrinho",  "pt-br"),
            ("fam_sobrinha",  "sobrinha",  "pt-br"),
            ("fam_neto",      "neto",      "pt-br"),
            ("fam_neta",      "neta",      "pt-br"),
            ("fam_bisavo",    "bisavô",    "pt-br"),
            ("fam_bisava",    "bisavó",    "pt-br"),
            ("fam_cunhado",   "cunhado",   "pt-br"),
            ("fam_cunhada",   "cunhada",   "pt-br"),
            ("fam_genro",     "genro",     "pt-br"),
            ("fam_nora",      "nora",      "pt-br"),
            ("fam_padrasto",  "padrasto",  "pt-br"),
            ("fam_madrasta",  "madrasta",  "pt-br"),
            ("fam_padrinho",  "padrinho",  "pt-br"),
            ("fam_madrinha",  "madrinha",  "pt-br"),
            ("fam_amigo",     "amigo",     "pt-br"),
            ("fam_amiga",     "amiga",     "pt-br"),
            ("fam_vizinho",   "vizinho",   "pt-br"),
            ("fam_colega",    "colega",    "pt-br"),
            ("fam_parceiro",  "parceiro",  "pt-br"),
            ("fam_namorado",  "namorado",  "pt-br"),
            ("fam_namorada",  "namorada",  "pt-br"),
            ("fam_bebe",      "bebê",      "pt-br"),
            ("fam_crianca",   "criança",   "pt-br"),
            ("fam_adolescente","adolescente","pt-br"),
            ("fam_adulto",    "adulto",    "pt-br"),
            ("fam_idoso",     "idoso",     "pt-br"),
        ],
    },
    3: {
        "nome": "Cores",
        "descricao": "Espectro cromático completo — cores básicas e secundárias",
        "itens": [
            ("cor_vermelho",  "vermelho",  "pt-br"),
            ("cor_azul",      "azul",      "pt-br"),
            ("cor_verde",     "verde",     "pt-br"),
            ("cor_amarelo",   "amarelo",   "pt-br"),
            ("cor_laranja",   "laranja",   "pt-br"),
            ("cor_roxo",      "roxo",      "pt-br"),
            ("cor_rosa",      "rosa",      "pt-br"),
            ("cor_branco",    "branco",    "pt-br"),
            ("cor_preto",     "preto",     "pt-br"),
            ("cor_cinza",     "cinza",     "pt-br"),
            ("cor_marrom",    "marrom",    "pt-br"),
            ("cor_bege",      "bege",      "pt-br"),
            ("cor_dourado",   "dourado",   "pt-br"),
            ("cor_prateado",  "prateado",  "pt-br"),
            ("cor_creme",     "creme",     "pt-br"),
            ("cor_turquesa",  "turquesa",  "pt-br"),
            ("cor_lilas",     "lilás",     "pt-br"),
            ("cor_vinho",     "vinho",     "pt-br"),
            ("cor_ciano",     "ciano",     "pt-br"),
            ("cor_magenta",   "magenta",   "pt-br"),
            ("cor_indigo",    "índigo",    "pt-br"),
            ("cor_oliva",     "oliva",     "pt-br"),
            ("cor_salmon",    "salmão",    "pt-br"),
            ("cor_coral",     "coral",     "pt-br"),
        ],
    },
    4: {
        "nome": "Números e quantidades",
        "descricao": "Números cardinais, ordinais e quantificadores",
        "itens": [
            ("num_zero",      "zero",      "pt-br"),
            ("num_um",        "um",        "pt-br"),
            ("num_dois",      "dois",      "pt-br"),
            ("num_tres",      "três",      "pt-br"),
            ("num_quatro",    "quatro",    "pt-br"),
            ("num_cinco",     "cinco",     "pt-br"),
            ("num_seis",      "seis",      "pt-br"),
            ("num_sete",      "sete",      "pt-br"),
            ("num_oito",      "oito",      "pt-br"),
            ("num_nove",      "nove",      "pt-br"),
            ("num_dez",       "dez",       "pt-br"),
            ("num_onze",      "onze",      "pt-br"),
            ("num_doze",      "doze",      "pt-br"),
            ("num_treze",     "treze",     "pt-br"),
            ("num_catorze",   "catorze",   "pt-br"),
            ("num_quinze",    "quinze",    "pt-br"),
            ("num_dezesseis", "dezesseis", "pt-br"),
            ("num_dezessete", "dezessete", "pt-br"),
            ("num_dezoito",   "dezoito",   "pt-br"),
            ("num_dezenove",  "dezenove",  "pt-br"),
            ("num_vinte",     "vinte",     "pt-br"),
            ("num_trinta",    "trinta",    "pt-br"),
            ("num_quarenta",  "quarenta",  "pt-br"),
            ("num_cinquenta", "cinquenta", "pt-br"),
            ("num_sessenta",  "sessenta",  "pt-br"),
            ("num_setenta",   "setenta",   "pt-br"),
            ("num_oitenta",   "oitenta",   "pt-br"),
            ("num_noventa",   "noventa",   "pt-br"),
            ("num_cem",       "cem",       "pt-br"),
            ("num_mil",       "mil",       "pt-br"),
            ("num_primeiro",  "primeiro",  "pt-br"),
            ("num_segundo",   "segundo",   "pt-br"),
            ("num_terceiro",  "terceiro",  "pt-br"),
            ("num_metade",    "metade",    "pt-br"),
            ("num_dobro",     "dobro",     "pt-br"),
        ],
    },
    5: {
        "nome": "Animais",
        "descricao": "Fauna doméstica, selvagem, marinha e insetos",
        "itens": [
            # Domésticos e fazenda
            ("ani_gato",      "gato",      "pt-br"),
            ("ani_cachorro",  "cachorro",  "pt-br"),
            ("ani_cavalo",    "cavalo",    "pt-br"),
            ("ani_vaca",      "vaca",      "pt-br"),
            ("ani_boi",       "boi",       "pt-br"),
            ("ani_porco",     "porco",     "pt-br"),
            ("ani_ovelha",    "ovelha",    "pt-br"),
            ("ani_cabra",     "cabra",     "pt-br"),
            ("ani_galinha",   "galinha",   "pt-br"),
            ("ani_galo",      "galo",      "pt-br"),
            ("ani_pato",      "pato",      "pt-br"),
            ("ani_coelho",    "coelho",    "pt-br"),
            ("ani_burro",     "burro",     "pt-br"),
            ("ani_peixe",     "peixe",     "pt-br"),
            ("ani_passaro",   "pássaro",   "pt-br"),
            ("ani_papagaio",  "papagaio",  "pt-br"),
            # Selvagens
            ("ani_leao",      "leão",      "pt-br"),
            ("ani_tigre",     "tigre",     "pt-br"),
            ("ani_urso",      "urso",      "pt-br"),
            ("ani_lobo",      "lobo",      "pt-br"),
            ("ani_raposa",    "raposa",    "pt-br"),
            ("ani_elefante",  "elefante",  "pt-br"),
            ("ani_girafa",    "girafa",    "pt-br"),
            ("ani_zebra",     "zebra",     "pt-br"),
            ("ani_rinoceronte","rinoceronte","pt-br"),
            ("ani_hipopotamo","hipopótamo","pt-br"),
            ("ani_macaco",    "macaco",    "pt-br"),
            ("ani_gorila",    "gorila",    "pt-br"),
            ("ani_chimpanze", "chimpanzé", "pt-br"),
            ("ani_veado",     "veado",     "pt-br"),
            ("ani_javali",    "javali",    "pt-br"),
            ("ani_cobra",     "cobra",     "pt-br"),
            ("ani_lagarto",   "lagarto",   "pt-br"),
            ("ani_crocodilo", "crocodilo", "pt-br"),
            ("ani_sapo",      "sapo",      "pt-br"),
            # Aves
            ("ani_aguia",     "águia",     "pt-br"),
            ("ani_corvo",     "corvo",     "pt-br"),
            ("ani_coruja",    "coruja",    "pt-br"),
            ("ani_tucan",     "tucano",    "pt-br"),
            ("ani_flamingo",  "flamingo",  "pt-br"),
            ("ani_pinguim",   "pinguim",   "pt-br"),
            # Marinhos
            ("ani_tubarao",   "tubarão",   "pt-br"),
            ("ani_baleia",    "baleia",    "pt-br"),
            ("ani_golfinho",  "golfinho",  "pt-br"),
            ("ani_polvo",     "polvo",     "pt-br"),
            ("ani_caranguejo","caranguejo","pt-br"),
            ("ani_camarao",   "camarão",   "pt-br"),
            # Insetos
            ("ani_abelha",    "abelha",    "pt-br"),
            ("ani_borboleta", "borboleta", "pt-br"),
            ("ani_mosca",     "mosca",     "pt-br"),
            ("ani_formiga",   "formiga",   "pt-br"),
            ("ani_aranha",    "aranha",    "pt-br"),
            ("ani_grilo",     "grilo",     "pt-br"),
            ("ani_besouros",  "besouro",   "pt-br"),
        ],
    },
    6: {
        "nome": "Comida e bebida",
        "descricao": "Alimentos, frutas, legumes, carnes e bebidas",
        "itens": [
            # Grãos e massas
            ("com_arroz",     "arroz",     "pt-br"),
            ("com_feijao",    "feijão",    "pt-br"),
            ("com_macarrao",  "macarrão",  "pt-br"),
            ("com_pao",       "pão",       "pt-br"),
            ("com_bolo",      "bolo",      "pt-br"),
            ("com_biscoito",  "biscoito",  "pt-br"),
            ("com_farinha",   "farinha",   "pt-br"),
            ("com_tapioca",   "tapioca",   "pt-br"),
            # Laticínios e ovos
            ("com_leite",     "leite",     "pt-br"),
            ("com_queijo",    "queijo",    "pt-br"),
            ("com_manteiga",  "manteiga",  "pt-br"),
            ("com_creme",     "creme",     "pt-br"),
            ("com_iogurte",   "iogurte",   "pt-br"),
            ("com_ovo",       "ovo",       "pt-br"),
            # Carnes e proteínas
            ("com_carne",     "carne",     "pt-br"),
            ("com_frango",    "frango",    "pt-br"),
            ("com_peixe",     "peixe",     "pt-br"),
            ("com_camarao",   "camarão",   "pt-br"),
            ("com_bacon",     "bacon",     "pt-br"),
            ("com_salsicha",  "salsicha",  "pt-br"),
            ("com_presunto",  "presunto",  "pt-br"),
            # Legumes e verduras
            ("com_alface",    "alface",    "pt-br"),
            ("com_tomate",    "tomate",    "pt-br"),
            ("com_pepino",    "pepino",    "pt-br"),
            ("com_cebola",    "cebola",    "pt-br"),
            ("com_alho",      "alho",      "pt-br"),
            ("com_batata",    "batata",    "pt-br"),
            ("com_cenoura",   "cenoura",   "pt-br"),
            ("com_brocolis",  "brócolis",  "pt-br"),
            ("com_couve",     "couve",     "pt-br"),
            ("com_espinafre", "espinafre", "pt-br"),
            ("com_milho",     "milho",     "pt-br"),
            ("com_ervilha",   "ervilha",   "pt-br"),
            ("com_cogumelo",  "cogumelo",  "pt-br"),
            ("com_abobora",   "abóbora",   "pt-br"),
            ("com_beterraba", "beterraba", "pt-br"),
            # Frutas
            ("com_maca",      "maçã",      "pt-br"),
            ("com_banana",    "banana",    "pt-br"),
            ("com_laranja",   "laranja",   "pt-br"),
            ("com_limao",     "limão",     "pt-br"),
            ("com_uva",       "uva",       "pt-br"),
            ("com_morango",   "morango",   "pt-br"),
            ("com_pera",      "pera",      "pt-br"),
            ("com_abacaxi",   "abacaxi",   "pt-br"),
            ("com_melancia",  "melancia",  "pt-br"),
            ("com_manga",     "manga",     "pt-br"),
            ("com_abacate",   "abacate",   "pt-br"),
            ("com_coco",      "coco",      "pt-br"),
            ("com_mamao",     "mamão",     "pt-br"),
            ("com_pessego",   "pêssego",   "pt-br"),
            ("com_ameixa",    "ameixa",    "pt-br"),
            ("com_framboesa", "framboesa", "pt-br"),
            ("com_mirtilo",   "mirtilo",   "pt-br"),
            ("com_goiaba",    "goiaba",    "pt-br"),
            ("com_acerola",   "acerola",   "pt-br"),
            # Bebidas
            ("beb_agua",      "água",      "pt-br"),
            ("beb_cafe",      "café",      "pt-br"),
            ("beb_cha",       "chá",       "pt-br"),
            ("beb_suco",      "suco",      "pt-br"),
            ("beb_leite",     "leite",     "pt-br"),
            ("beb_refri",     "refrigerante","pt-br"),
            ("beb_cerveja",   "cerveja",   "pt-br"),
            ("beb_vinho",     "vinho",     "pt-br"),
            ("beb_cachaça",   "cachaça",   "pt-br"),
            # Temperos
            ("tmp_sal",       "sal",       "pt-br"),
            ("tmp_acucar",    "açúcar",    "pt-br"),
            ("tmp_pimenta",   "pimenta",   "pt-br"),
            ("tmp_oleo",      "óleo",      "pt-br"),
            ("tmp_vinagre",   "vinagre",   "pt-br"),
            ("tmp_mostarda",  "mostarda",  "pt-br"),
        ],
    },
    7: {
        "nome": "Casa e objetos",
        "descricao": "Cômodos, móveis, utensílios domésticos e tecnologia",
        "itens": [
            # Partes da casa
            ("cas_casa",      "casa",      "pt-br"),
            ("cas_quarto",    "quarto",    "pt-br"),
            ("cas_sala",      "sala",      "pt-br"),
            ("cas_cozinha",   "cozinha",   "pt-br"),
            ("cas_banheiro",  "banheiro",  "pt-br"),
            ("cas_garagem",   "garagem",   "pt-br"),
            ("cas_jardim",    "jardim",    "pt-br"),
            ("cas_varanda",   "varanda",   "pt-br"),
            ("cas_corredor",  "corredor",  "pt-br"),
            ("cas_escada",    "escada",    "pt-br"),
            # Estruturas
            ("cas_porta",     "porta",     "pt-br"),
            ("cas_janela",    "janela",    "pt-br"),
            ("cas_parede",    "parede",    "pt-br"),
            ("cas_teto",      "teto",      "pt-br"),
            ("cas_chao",      "chão",      "pt-br"),
            # Móveis
            ("cas_mesa",      "mesa",      "pt-br"),
            ("cas_cadeira",   "cadeira",   "pt-br"),
            ("cas_sofa",      "sofá",      "pt-br"),
            ("cas_cama",      "cama",      "pt-br"),
            ("cas_armario",   "armário",   "pt-br"),
            ("cas_gaveta",    "gaveta",    "pt-br"),
            ("cas_espelho",   "espelho",   "pt-br"),
            ("cas_tapete",    "tapete",    "pt-br"),
            ("cas_cortina",   "cortina",   "pt-br"),
            ("cas_prateleira","prateleira","pt-br"),
            ("cas_estante",   "estante",   "pt-br"),
            # Eletrodomésticos
            ("cas_geladeira", "geladeira", "pt-br"),
            ("cas_fogao",     "fogão",     "pt-br"),
            ("cas_forno",     "forno",     "pt-br"),
            ("cas_microondas","micro-ondas","pt-br"),
            ("cas_maquina",   "máquina",   "pt-br"),
            ("cas_liquidificador","liquidificador","pt-br"),
            # Tecnologia
            ("cas_televisao", "televisão", "pt-br"),
            ("cas_celular",   "celular",   "pt-br"),
            ("cas_computador","computador","pt-br"),
            ("cas_tablet",    "tablet",    "pt-br"),
            ("cas_radio",     "rádio",     "pt-br"),
            # Utensílios
            ("cas_panela",    "panela",    "pt-br"),
            ("cas_faca",      "faca",      "pt-br"),
            ("cas_garfo",     "garfo",     "pt-br"),
            ("cas_colher",    "colher",    "pt-br"),
            ("cas_prato",     "prato",     "pt-br"),
            ("cas_copo",      "copo",      "pt-br"),
            ("cas_xicara",    "xícara",    "pt-br"),
            # Objetos pessoais
            ("cas_chave",     "chave",     "pt-br"),
            ("cas_relogio",   "relógio",   "pt-br"),
            ("cas_livro",     "livro",     "pt-br"),
            ("cas_caneta",    "caneta",    "pt-br"),
            ("cas_papel",     "papel",     "pt-br"),
            ("cas_tesoura",   "tesoura",   "pt-br"),
            ("cas_vela",      "vela",      "pt-br"),
            ("cas_lampada",   "lâmpada",   "pt-br"),
        ],
    },
    8: {
        "nome": "Natureza",
        "descricao": "Flora, fauna, clima, geografia e fenômenos naturais",
        "itens": [
            # Flora
            ("nat_arvore",    "árvore",    "pt-br"),
            ("nat_planta",    "planta",    "pt-br"),
            ("nat_flor",      "flor",      "pt-br"),
            ("nat_grama",     "grama",     "pt-br"),
            ("nat_folha",     "folha",     "pt-br"),
            ("nat_raiz",      "raiz",      "pt-br"),
            ("nat_galho",     "galho",     "pt-br"),
            ("nat_tronco",    "tronco",    "pt-br"),
            ("nat_fruto",     "fruto",     "pt-br"),
            ("nat_semente",   "semente",   "pt-br"),
            ("nat_musgo",     "musgo",     "pt-br"),
            ("nat_bambu",     "bambu",     "pt-br"),
            ("nat_cacto",     "cacto",     "pt-br"),
            ("nat_palmeira",  "palmeira",  "pt-br"),
            # Geografia
            ("nat_montanha",  "montanha",  "pt-br"),
            ("nat_colina",    "colina",    "pt-br"),
            ("nat_vale",      "vale",      "pt-br"),
            ("nat_planicie",  "planície",  "pt-br"),
            ("nat_deserto",   "deserto",   "pt-br"),
            ("nat_floresta",  "floresta",  "pt-br"),
            ("nat_selva",     "selva",     "pt-br"),
            ("nat_praia",     "praia",     "pt-br"),
            ("nat_oceano",    "oceano",    "pt-br"),
            ("nat_mar",       "mar",       "pt-br"),
            ("nat_lago",      "lago",      "pt-br"),
            ("nat_rio",       "rio",       "pt-br"),
            ("nat_riacho",    "riacho",    "pt-br"),
            ("nat_cachoeira", "cachoeira", "pt-br"),
            ("nat_pantano",   "pântano",   "pt-br"),
            ("nat_ilha",      "ilha",      "pt-br"),
            ("nat_peninsula", "península", "pt-br"),
            ("nat_continente","continente","pt-br"),
            # Solo e minerais
            ("nat_pedra",     "pedra",     "pt-br"),
            ("nat_areia",     "areia",     "pt-br"),
            ("nat_lama",      "lama",      "pt-br"),
            ("nat_terra",     "terra",     "pt-br"),
            ("nat_ouro",      "ouro",      "pt-br"),
            ("nat_prata",     "prata",     "pt-br"),
            ("nat_ferro",     "ferro",     "pt-br"),
            # Clima e fenômenos
            ("nat_sol",       "sol",       "pt-br"),
            ("nat_lua",       "lua",       "pt-br"),
            ("nat_estrela",   "estrela",   "pt-br"),
            ("nat_nuvem",     "nuvem",     "pt-br"),
            ("nat_chuva",     "chuva",     "pt-br"),
            ("nat_trovao",    "trovão",    "pt-br"),
            ("nat_relampago", "relâmpago", "pt-br"),
            ("nat_neve",      "neve",      "pt-br"),
            ("nat_granizo",   "granizo",   "pt-br"),
            ("nat_vento",     "vento",     "pt-br"),
            ("nat_neblina",   "neblina",   "pt-br"),
            ("nat_arco_iris", "arco-íris", "pt-br"),
            ("nat_vulcao",    "vulcão",    "pt-br"),
            ("nat_terremoto", "terremoto", "pt-br"),
        ],
    },
    9: {
        "nome": "Verbos básicos",
        "descricao": "Ações fundamentais do cotidiano — verbos de alta frequência",
        "itens": [
            # Ser e estado
            ("vrb_ser",       "ser",       "pt-br"),
            ("vrb_estar",     "estar",     "pt-br"),
            ("vrb_ter",       "ter",       "pt-br"),
            ("vrb_haver",     "haver",     "pt-br"),
            ("vrb_ficar",     "ficar",     "pt-br"),
            ("vrb_parecer",   "parecer",   "pt-br"),
            # Movimentos
            ("vrb_ir",        "ir",        "pt-br"),
            ("vrb_vir",       "vir",       "pt-br"),
            ("vrb_andar",     "andar",     "pt-br"),
            ("vrb_correr",    "correr",    "pt-br"),
            ("vrb_pular",     "pular",     "pt-br"),
            ("vrb_nadar",     "nadar",     "pt-br"),
            ("vrb_voar",      "voar",      "pt-br"),
            ("vrb_subir",     "subir",     "pt-br"),
            ("vrb_descer",    "descer",    "pt-br"),
            ("vrb_entrar",    "entrar",    "pt-br"),
            ("vrb_sair",      "sair",      "pt-br"),
            ("vrb_chegar",    "chegar",    "pt-br"),
            ("vrb_partir",    "partir",    "pt-br"),
            ("vrb_voltar",    "voltar",    "pt-br"),
            # Ações físicas
            ("vrb_abrir",     "abrir",     "pt-br"),
            ("vrb_fechar",    "fechar",    "pt-br"),
            ("vrb_pegar",     "pegar",     "pt-br"),
            ("vrb_soltar",    "soltar",    "pt-br"),
            ("vrb_jogar",     "jogar",     "pt-br"),
            ("vrb_colocar",   "colocar",   "pt-br"),
            ("vrb_tirar",     "tirar",     "pt-br"),
            ("vrb_levantar",  "levantar",  "pt-br"),
            ("vrb_sentar",    "sentar",    "pt-br"),
            ("vrb_deitar",    "deitar",    "pt-br"),
            ("vrb_dormir",    "dormir",    "pt-br"),
            ("vrb_acordar",   "acordar",   "pt-br"),
            # Nutrição
            ("vrb_comer",     "comer",     "pt-br"),
            ("vrb_beber",     "beber",     "pt-br"),
            ("vrb_cozinhar",  "cozinhar",  "pt-br"),
            ("vrb_mastigar",  "mastigar",  "pt-br"),
            # Comunicação
            ("vrb_falar",     "falar",     "pt-br"),
            ("vrb_dizer",     "dizer",     "pt-br"),
            ("vrb_contar",    "contar",    "pt-br"),
            ("vrb_perguntar", "perguntar", "pt-br"),
            ("vrb_responder", "responder", "pt-br"),
            ("vrb_ouvir",     "ouvir",     "pt-br"),
            ("vrb_escutar",   "escutar",   "pt-br"),
            ("vrb_ler",       "ler",       "pt-br"),
            ("vrb_escrever",  "escrever",  "pt-br"),
            ("vrb_cantar",    "cantar",    "pt-br"),
            # Cognição
            ("vrb_pensar",    "pensar",    "pt-br"),
            ("vrb_saber",     "saber",     "pt-br"),
            ("vrb_conhecer",  "conhecer",  "pt-br"),
            ("vrb_aprender",  "aprender",  "pt-br"),
            ("vrb_entender",  "entender",  "pt-br"),
            ("vrb_lembrar",   "lembrar",   "pt-br"),
            ("vrb_esquecer",  "esquecer",  "pt-br"),
            ("vrb_imaginar",  "imaginar",  "pt-br"),
            ("vrb_sonhar",    "sonhar",    "pt-br"),
            # Afeto
            ("vrb_amar",      "amar",      "pt-br"),
            ("vrb_gostar",    "gostar",    "pt-br"),
            ("vrb_odiar",     "odiar",     "pt-br"),
            ("vrb_sentir",    "sentir",    "pt-br"),
            ("vrb_querer",    "querer",    "pt-br"),
            ("vrb_precisar",  "precisar",  "pt-br"),
            ("vrb_ajudar",    "ajudar",    "pt-br"),
            ("vrb_cuidar",    "cuidar",    "pt-br"),
            ("vrb_proteger",  "proteger",  "pt-br"),
            # Social
            ("vrb_comprar",   "comprar",   "pt-br"),
            ("vrb_vender",    "vender",    "pt-br"),
            ("vrb_pagar",     "pagar",     "pt-br"),
            ("vrb_receber",   "receber",   "pt-br"),
            ("vrb_trabalhar", "trabalhar", "pt-br"),
            ("vrb_estudar",   "estudar",   "pt-br"),
            ("vrb_brincar",   "brincar",   "pt-br"),
            ("vrb_dançar",    "dançar",    "pt-br"),
            ("vrb_rir",       "rir",       "pt-br"),
            ("vrb_chorar",    "chorar",    "pt-br"),
        ],
    },
    10: {
        "nome": "Emoções e sentimentos",
        "descricao": "Estados afetivos, emocionais e psicológicos",
        "itens": [
            ("emo_alegria",      "alegria",      "pt-br"),
            ("emo_tristeza",     "tristeza",     "pt-br"),
            ("emo_raiva",        "raiva",        "pt-br"),
            ("emo_medo",         "medo",         "pt-br"),
            ("emo_surpresa",     "surpresa",     "pt-br"),
            ("emo_amor",         "amor",         "pt-br"),
            ("emo_odio",         "ódio",         "pt-br"),
            ("emo_vergonha",     "vergonha",     "pt-br"),
            ("emo_orgulho",      "orgulho",      "pt-br"),
            ("emo_inveja",       "inveja",       "pt-br"),
            ("emo_ciume",        "ciúme",        "pt-br"),
            ("emo_saudade",      "saudade",      "pt-br"),
            ("emo_ansiedade",    "ansiedade",    "pt-br"),
            ("emo_calma",        "calma",        "pt-br"),
            ("emo_paz",          "paz",          "pt-br"),
            ("emo_felicidade",   "felicidade",   "pt-br"),
            ("emo_esperanca",    "esperança",    "pt-br"),
            ("emo_confianca",    "confiança",    "pt-br"),
            ("emo_solidao",      "solidão",      "pt-br"),
            ("emo_gratidao",     "gratidão",     "pt-br"),
            ("emo_carinho",      "carinho",      "pt-br"),
            ("emo_compaixao",    "compaixão",    "pt-br"),
            ("emo_empolgacao",   "empolgação",   "pt-br"),
            ("emo_frustracao",   "frustração",   "pt-br"),
            ("emo_alivio",       "alívio",       "pt-br"),
            ("emo_culpa",        "culpa",        "pt-br"),
            ("emo_entusiasmo",   "entusiasmo",   "pt-br"),
            ("emo_coragem",      "coragem",      "pt-br"),
            ("emo_timidez",      "timidez",      "pt-br"),
            ("emo_nostalgia",    "nostalgia",    "pt-br"),
            ("emo_empatia",      "empatia",      "pt-br"),
            ("emo_curiosidade",  "curiosidade",  "pt-br"),
            ("emo_admiracao",    "admiração",    "pt-br"),
            ("emo_decepção",     "decepção",     "pt-br"),
        ],
    },
    11: {
        "nome": "Vestuário e acessórios",
        "descricao": "Roupas, calçados, adornos e acessórios pessoais",
        "itens": [
            # Roupas superiores
            ("ves_camisa",    "camisa",    "pt-br"),
            ("ves_camiseta",  "camiseta",  "pt-br"),
            ("ves_blusa",     "blusa",     "pt-br"),
            ("ves_sueter",    "suéter",    "pt-br"),
            ("ves_casaco",    "casaco",    "pt-br"),
            ("ves_jaqueta",   "jaqueta",   "pt-br"),
            ("ves_paleto",    "paletó",    "pt-br"),
            ("ves_gravata",   "gravata",   "pt-br"),
            # Roupas inferiores
            ("ves_calca",     "calça",     "pt-br"),
            ("ves_shorts",    "shorts",    "pt-br"),
            ("ves_saia",      "saia",      "pt-br"),
            ("ves_vestido",   "vestido",   "pt-br"),
            # Íntimas
            ("ves_meia",      "meia",      "pt-br"),
            ("ves_cueca",     "cueca",     "pt-br"),
            ("ves_calcinha",  "calcinha",  "pt-br"),
            ("ves_sutia",     "sutiã",     "pt-br"),
            ("ves_pijama",    "pijama",    "pt-br"),
            ("ves_roupao",    "roupão",    "pt-br"),
            # Calçados
            ("ves_sapato",    "sapato",    "pt-br"),
            ("ves_tenis",     "tênis",     "pt-br"),
            ("ves_sandalia",  "sandália",  "pt-br"),
            ("ves_bota",      "bota",      "pt-br"),
            ("ves_chinelo",   "chinelo",   "pt-br"),
            # Acessórios
            ("ves_chapeu",    "chapéu",    "pt-br"),
            ("ves_bone",      "boné",      "pt-br"),
            ("ves_oculos",    "óculos",    "pt-br"),
            ("ves_cinto",     "cinto",     "pt-br"),
            ("ves_bolsa",     "bolsa",     "pt-br"),
            ("ves_mochila",   "mochila",   "pt-br"),
            ("ves_carteira",  "carteira",  "pt-br"),
            ("ves_lenco",     "lenço",     "pt-br"),
            ("ves_luva",      "luva",      "pt-br"),
            ("ves_cachecol",  "cachecol",  "pt-br"),
            ("ves_colar",     "colar",     "pt-br"),
            ("ves_brinco",    "brinco",    "pt-br"),
            ("ves_anel",      "anel",      "pt-br"),
            ("ves_pulseira",  "pulseira",  "pt-br"),
            ("ves_relogio",   "relógio",   "pt-br"),
        ],
    },
    12: {
        "nome": "Tempo e calendário",
        "descricao": "Unidades de tempo, dias, meses, estações e expressões temporais",
        "itens": [
            # Unidades de tempo
            ("tmp_segundo",   "segundo",   "pt-br"),
            ("tmp_minuto",    "minuto",    "pt-br"),
            ("tmp_hora",      "hora",      "pt-br"),
            ("tmp_dia",       "dia",       "pt-br"),
            ("tmp_semana",    "semana",    "pt-br"),
            ("tmp_mes",       "mês",       "pt-br"),
            ("tmp_ano",       "ano",       "pt-br"),
            ("tmp_decada",    "década",    "pt-br"),
            ("tmp_seculo",    "século",    "pt-br"),
            # Partes do dia
            ("tmp_manha",     "manhã",     "pt-br"),
            ("tmp_tarde",     "tarde",     "pt-br"),
            ("tmp_noite",     "noite",     "pt-br"),
            ("tmp_madrugada", "madrugada", "pt-br"),
            ("tmp_amanhecer", "amanhecer", "pt-br"),
            ("tmp_anoitecer", "anoitecer", "pt-br"),
            # Advérbios temporais
            ("tmp_hoje",      "hoje",      "pt-br"),
            ("tmp_ontem",     "ontem",     "pt-br"),
            ("tmp_amanha",    "amanhã",    "pt-br"),
            ("tmp_agora",     "agora",     "pt-br"),
            ("tmp_antes",     "antes",     "pt-br"),
            ("tmp_depois",    "depois",    "pt-br"),
            ("tmp_sempre",    "sempre",    "pt-br"),
            ("tmp_nunca",     "nunca",     "pt-br"),
            ("tmp_cedo",      "cedo",      "pt-br"),
            ("tmp_logo",      "logo",      "pt-br"),
            # Dias da semana
            ("tmp_segunda",   "segunda",   "pt-br"),
            ("tmp_terca",     "terça",     "pt-br"),
            ("tmp_quarta",    "quarta",    "pt-br"),
            ("tmp_quinta",    "quinta",    "pt-br"),
            ("tmp_sexta",     "sexta",     "pt-br"),
            ("tmp_sabado",    "sábado",    "pt-br"),
            ("tmp_domingo",   "domingo",   "pt-br"),
            # Meses
            ("tmp_janeiro",   "janeiro",   "pt-br"),
            ("tmp_fevereiro", "fevereiro", "pt-br"),
            ("tmp_marco",     "março",     "pt-br"),
            ("tmp_abril",     "abril",     "pt-br"),
            ("tmp_maio",      "maio",      "pt-br"),
            ("tmp_junho",     "junho",     "pt-br"),
            ("tmp_julho",     "julho",     "pt-br"),
            ("tmp_agosto",    "agosto",    "pt-br"),
            ("tmp_setembro",  "setembro",  "pt-br"),
            ("tmp_outubro",   "outubro",   "pt-br"),
            ("tmp_novembro",  "novembro",  "pt-br"),
            ("tmp_dezembro",  "dezembro",  "pt-br"),
            # Estações
            ("tmp_primavera", "primavera", "pt-br"),
            ("tmp_verao",     "verão",     "pt-br"),
            ("tmp_outono",    "outono",    "pt-br"),
            ("tmp_inverno",   "inverno",   "pt-br"),
        ],
    },
    13: {
        "nome": "Lugares e construções",
        "descricao": "Espaços urbanos, rurais, públicos e naturais",
        "itens": [
            # Espaços urbanos
            ("lug_cidade",    "cidade",    "pt-br"),
            ("lug_aldeia",    "aldeia",    "pt-br"),
            ("lug_bairro",    "bairro",    "pt-br"),
            ("lug_rua",       "rua",       "pt-br"),
            ("lug_avenida",   "avenida",   "pt-br"),
            ("lug_praca",     "praça",     "pt-br"),
            ("lug_parque",    "parque",    "pt-br"),
            # Saúde e educação
            ("lug_hospital",  "hospital",  "pt-br"),
            ("lug_clinica",   "clínica",   "pt-br"),
            ("lug_escola",    "escola",    "pt-br"),
            ("lug_faculdade", "faculdade", "pt-br"),
            ("lug_universidade","universidade","pt-br"),
            ("lug_biblioteca","biblioteca","pt-br"),
            # Cultura e lazer
            ("lug_museu",     "museu",     "pt-br"),
            ("lug_cinema",    "cinema",    "pt-br"),
            ("lug_teatro",    "teatro",    "pt-br"),
            ("lug_estadio",   "estádio",   "pt-br"),
            ("lug_parque2",   "parque",    "pt-br"),
            # Comércio
            ("lug_shopping",  "shopping",  "pt-br"),
            ("lug_mercado",   "mercado",   "pt-br"),
            ("lug_supermercado","supermercado","pt-br"),
            ("lug_loja",      "loja",      "pt-br"),
            ("lug_feira",     "feira",     "pt-br"),
            # Serviços
            ("lug_banco",     "banco",     "pt-br"),
            ("lug_correio",   "correio",   "pt-br"),
            ("lug_hotel",     "hotel",     "pt-br"),
            ("lug_restaurante","restaurante","pt-br"),
            ("lug_cafe",      "café",      "pt-br"),
            ("lug_farmacia",  "farmácia",  "pt-br"),
            # Religioso e institucional
            ("lug_igreja",    "igreja",    "pt-br"),
            ("lug_templo",    "templo",    "pt-br"),
            ("lug_tribunal",  "tribunal",  "pt-br"),
            ("lug_delegacia", "delegacia", "pt-br"),
            # Transportes
            ("lug_aeroporto", "aeroporto", "pt-br"),
            ("lug_estacao",   "estação",   "pt-br"),
            ("lug_porto",     "porto",     "pt-br"),
            ("lug_rodovia",   "rodovia",   "pt-br"),
            ("lug_ponte",     "ponte",     "pt-br"),
            ("lug_tunel",     "túnel",     "pt-br"),
            # Rural e natureza
            ("lug_fazenda",   "fazenda",   "pt-br"),
            ("lug_sitio",     "sítio",     "pt-br"),
            ("lug_rancho",    "rancho",    "pt-br"),
            ("lug_floresta",  "floresta",  "pt-br"),
            ("lug_praia2",    "praia",     "pt-br"),
            ("lug_campo",     "campo",     "pt-br"),
        ],
    },
    14: {
        "nome": "Adjetivos comuns",
        "descricao": "Qualidades, características e estados mais usados no PB",
        "itens": [
            # Tamanho e peso
            ("adj_grande",    "grande",    "pt-br"),
            ("adj_pequeno",   "pequeno",   "pt-br"),
            ("adj_alto",      "alto",      "pt-br"),
            ("adj_baixo",     "baixo",     "pt-br"),
            ("adj_gordo",     "gordo",     "pt-br"),
            ("adj_magro",     "magro",     "pt-br"),
            ("adj_longo",     "longo",     "pt-br"),
            ("adj_curto",     "curto",     "pt-br"),
            ("adj_largo",     "largo",     "pt-br"),
            ("adj_estreito",  "estreito",  "pt-br"),
            # Aparência
            ("adj_bonito",    "bonito",    "pt-br"),
            ("adj_feio",      "feio",      "pt-br"),
            ("adj_novo",      "novo",      "pt-br"),
            ("adj_velho",     "velho",     "pt-br"),
            ("adj_jovem",     "jovem",     "pt-br"),
            ("adj_antigo",    "antigo",    "pt-br"),
            ("adj_moderno",   "moderno",   "pt-br"),
            # Velocidade e força
            ("adj_rapido",    "rápido",    "pt-br"),
            ("adj_lento",     "lento",     "pt-br"),
            ("adj_forte",     "forte",     "pt-br"),
            ("adj_fraco",     "fraco",     "pt-br"),
            # Condição física
            ("adj_quente",    "quente",    "pt-br"),
            ("adj_frio",      "frio",      "pt-br"),
            ("adj_morno",     "morno",     "pt-br"),
            ("adj_seco",      "seco",      "pt-br"),
            ("adj_umido",     "úmido",     "pt-br"),
            ("adj_limpo",     "limpo",     "pt-br"),
            ("adj_sujo",      "sujo",      "pt-br"),
            # Luz e cor
            ("adj_claro",     "claro",     "pt-br"),
            ("adj_escuro",    "escuro",    "pt-br"),
            ("adj_brilhante", "brilhante", "pt-br"),
            # Estado
            ("adj_cheio",     "cheio",     "pt-br"),
            ("adj_vazio",     "vazio",     "pt-br"),
            ("adj_aberto",    "aberto",    "pt-br"),
            ("adj_fechado",   "fechado",   "pt-br"),
            ("adj_duro",      "duro",      "pt-br"),
            ("adj_mole",      "mole",      "pt-br"),
            # Valor moral e lógico
            ("adj_bom",       "bom",       "pt-br"),
            ("adj_mau",       "mau",       "pt-br"),
            ("adj_certo",     "certo",     "pt-br"),
            ("adj_errado",    "errado",    "pt-br"),
            ("adj_verdadeiro","verdadeiro","pt-br"),
            ("adj_falso",     "falso",     "pt-br"),
            ("adj_justo",     "justo",     "pt-br"),
            ("adj_injusto",   "injusto",   "pt-br"),
            # Social e afetivo
            ("adj_feliz",     "feliz",     "pt-br"),
            ("adj_triste",    "triste",    "pt-br"),
            ("adj_doente",    "doente",    "pt-br"),
            ("adj_saudavel",  "saudável",  "pt-br"),
            ("adj_inteligente","inteligente","pt-br"),
            ("adj_simpatico", "simpático", "pt-br"),
            ("adj_antipático","antipático","pt-br"),
            ("adj_corajoso",  "corajoso",  "pt-br"),
            ("adj_medroso",   "medroso",   "pt-br"),
            ("adj_honesto",   "honesto",   "pt-br"),
            ("adj_gentil",    "gentil",    "pt-br"),
            # Dificuldade
            ("adj_facil",     "fácil",     "pt-br"),
            ("adj_dificil",   "difícil",   "pt-br"),
            ("adj_possivel",  "possível",  "pt-br"),
            # Quantidade
            ("adj_rico",      "rico",      "pt-br"),
            ("adj_pobre",     "pobre",     "pt-br"),
            ("adj_muito",     "muito",     "pt-br"),
            ("adj_pouco",     "pouco",     "pt-br"),
            ("adj_bastante",  "bastante",  "pt-br"),
            ("adj_suficiente","suficiente","pt-br"),
        ],
    },
    15: {
        "nome": "Pronomes, artigos e determinantes",
        "descricao": "Estrutura gramatical essencial — dêiticos, possessivos, relativos",
        "itens": [
            # Pessoais
            ("pro_eu",        "eu",        "pt-br"),
            ("pro_tu",        "tu",        "pt-br"),
            ("pro_voce",      "você",      "pt-br"),
            ("pro_ele",       "ele",       "pt-br"),
            ("pro_ela",       "ela",       "pt-br"),
            ("pro_nos",       "nós",       "pt-br"),
            ("pro_voces",     "vocês",     "pt-br"),
            ("pro_eles",      "eles",      "pt-br"),
            ("pro_elas",      "elas",      "pt-br"),
            # Possessivos
            ("pro_meu",       "meu",       "pt-br"),
            ("pro_minha",     "minha",     "pt-br"),
            ("pro_seu",       "seu",       "pt-br"),
            ("pro_sua",       "sua",       "pt-br"),
            ("pro_nosso",     "nosso",     "pt-br"),
            ("pro_nossa",     "nossa",     "pt-br"),
            ("pro_dele",      "dele",      "pt-br"),
            ("pro_dela",      "dela",      "pt-br"),
            # Demonstrativos
            ("pro_este",      "este",      "pt-br"),
            ("pro_esse",      "esse",      "pt-br"),
            ("pro_aquele",    "aquele",    "pt-br"),
            ("pro_esta",      "esta",      "pt-br"),
            ("pro_isso",      "isso",      "pt-br"),
            ("pro_isto",      "isto",      "pt-br"),
            ("pro_aquilo",    "aquilo",    "pt-br"),
            # Interrogativos e relativos
            ("pro_que",       "que",       "pt-br"),
            ("pro_quem",      "quem",      "pt-br"),
            ("pro_qual",      "qual",      "pt-br"),
            ("pro_quando",    "quando",    "pt-br"),
            ("pro_onde",      "onde",      "pt-br"),
            ("pro_como",      "como",      "pt-br"),
            ("pro_por_que",   "por que",   "pt-br"),
            ("pro_quanto",    "quanto",    "pt-br"),
            # Indefinidos
            ("pro_algum",     "algum",     "pt-br"),
            ("pro_nenhum",    "nenhum",    "pt-br"),
            ("pro_todo",      "todo",      "pt-br"),
            ("pro_cada",      "cada",      "pt-br"),
            ("pro_outro",     "outro",     "pt-br"),
            ("pro_mesmo",     "mesmo",     "pt-br"),
            ("pro_qualquer",  "qualquer",  "pt-br"),
            ("pro_alguem",    "alguém",    "pt-br"),
            ("pro_ninguem",   "ninguém",   "pt-br"),
            ("pro_tudo",      "tudo",      "pt-br"),
            ("pro_nada",      "nada",      "pt-br"),
        ],
    },
    16: {
        "nome": "Perguntas, saudações e expressões sociais",
        "descricao": "Comunicação social básica, cortesia e expressões cotidianas",
        "itens": [
            # Saudações
            ("sau_ola",        "olá",        "pt-br"),
            ("sau_oi",         "oi",         "pt-br"),
            ("sau_bom_dia",    "bom dia",    "pt-br"),
            ("sau_boa_tarde",  "boa tarde",  "pt-br"),
            ("sau_boa_noite",  "boa noite",  "pt-br"),
            ("sau_tchau",      "tchau",      "pt-br"),
            ("sau_adeus",      "adeus",      "pt-br"),
            ("sau_ate_logo",   "até logo",   "pt-br"),
            ("sau_ate_mais",   "até mais",   "pt-br"),
            # Cortesia
            ("crt_obrigado",   "obrigado",   "pt-br"),
            ("crt_obrigada",   "obrigada",   "pt-br"),
            ("crt_de_nada",    "de nada",    "pt-br"),
            ("crt_por_favor",  "por favor",  "pt-br"),
            ("crt_desculpa",   "desculpa",   "pt-br"),
            ("crt_perdao",     "perdão",     "pt-br"),
            ("crt_com_licenca","com licença","pt-br"),
            # Perguntas essenciais
            ("per_como",       "como",       "pt-br"),
            ("per_quando",     "quando",     "pt-br"),
            ("per_onde",       "onde",       "pt-br"),
            ("per_quem",       "quem",       "pt-br"),
            ("per_porque",     "por que",    "pt-br"),
            ("per_quanto",     "quanto",     "pt-br"),
            ("per_qual",       "qual",       "pt-br"),
            ("per_o_que",      "o que",      "pt-br"),
            # Respostas e confirmações
            ("res_sim",        "sim",        "pt-br"),
            ("res_nao",        "não",        "pt-br"),
            ("res_talvez",     "talvez",     "pt-br"),
            ("res_claro",      "claro",      "pt-br"),
            ("res_certeza",    "certeza",    "pt-br"),
            ("res_nao_sei",    "não sei",    "pt-br"),
            # Expressões comuns
            ("exp_tudo_bem",   "tudo bem",   "pt-br"),
            ("exp_que_otimo",  "que ótimo",  "pt-br"),
            ("exp_que_pena",   "que pena",   "pt-br"),
            ("exp_que_legal",  "que legal",  "pt-br"),
            ("exp_parabens",   "parabéns",   "pt-br"),
            ("exp_feliz_aniversario","feliz aniversário","pt-br"),
            ("exp_boa_sorte",  "boa sorte",  "pt-br"),
            ("exp_com_cuidado","com cuidado","pt-br"),
            ("exp_me_ajuda",   "me ajuda",   "pt-br"),
            ("exp_nao_entendi","não entendi","pt-br"),
            ("exp_repete",     "pode repetir","pt-br"),
        ],
    },
    17: {
        "nome": "Polissílabas e palavras longas comuns",
        "descricao": "Palavras de 3+ sílabas de alta frequência no cotidiano",
        "itens": [
            # Tecnologia e ciência
            ("pol_computador",   "computador",    "pt-br"),
            ("pol_televisao",    "televisão",     "pt-br"),
            ("pol_telefone",     "telefone",      "pt-br"),
            ("pol_celular",      "celular",       "pt-br"),
            ("pol_internet",     "internet",      "pt-br"),
            ("pol_tecnologia",   "tecnologia",    "pt-br"),
            ("pol_informatica",  "informática",   "pt-br"),
            ("pol_matematica",   "matemática",    "pt-br"),
            ("pol_biologia",     "biologia",      "pt-br"),
            ("pol_quimica",      "química",       "pt-br"),
            ("pol_fisica",       "física",        "pt-br"),
            ("pol_astronomia",   "astronomia",    "pt-br"),
            ("pol_fotografia",   "fotografia",    "pt-br"),
            ("pol_musica",       "música",        "pt-br"),
            ("pol_literatura",   "literatura",    "pt-br"),
            ("pol_filosofia",    "filosofia",     "pt-br"),
            ("pol_historia",     "história",      "pt-br"),
            ("pol_geografia",    "geografia",     "pt-br"),
            # Abstratos e sociais
            ("pol_felicidade",   "felicidade",    "pt-br"),
            ("pol_liberdade",    "liberdade",     "pt-br"),
            ("pol_igualdade",    "igualdade",     "pt-br"),
            ("pol_solidariedade","solidariedade", "pt-br"),
            ("pol_responsabilidade","responsabilidade","pt-br"),
            ("pol_possibilidade","possibilidade", "pt-br"),
            ("pol_oportunidade", "oportunidade",  "pt-br"),
            ("pol_universidade", "universidade",  "pt-br"),
            ("pol_comunicacao",  "comunicação",   "pt-br"),
            ("pol_informacao",   "informação",    "pt-br"),
            ("pol_conhecimento", "conhecimento",  "pt-br"),
            ("pol_aprendizado",  "aprendizado",   "pt-br"),
            ("pol_compreensao",  "compreensão",   "pt-br"),
            ("pol_inteligencia", "inteligência",  "pt-br"),
            ("pol_consciencia",  "consciência",   "pt-br"),
            ("pol_imaginacao",   "imaginação",    "pt-br"),
            ("pol_criatividade", "criatividade",  "pt-br"),
            ("pol_desenvolvimento","desenvolvimento","pt-br"),
            ("pol_transformacao","transformação", "pt-br"),
            ("pol_participacao", "participação",  "pt-br"),
            ("pol_colaboracao",  "colaboração",   "pt-br"),
            # Corpo e saúde
            ("pol_alimentacao",  "alimentação",   "pt-br"),
            ("pol_respiracao",   "respiração",    "pt-br"),
            ("pol_circulacao",   "circulação",    "pt-br"),
            ("pol_digestao",     "digestão",      "pt-br"),
            ("pol_imunidade",    "imunidade",     "pt-br"),
            ("pol_vacina",       "vacina",        "pt-br"),
            ("pol_medicina",     "medicina",      "pt-br"),
            ("pol_operacao",     "operação",      "pt-br"),
            ("pol_recuperacao",  "recuperação",   "pt-br"),
            # Profissões
            ("pol_professor",    "professor",     "pt-br"),
            ("pol_engenheiro",   "engenheiro",    "pt-br"),
            ("pol_medico",       "médico",        "pt-br"),
            ("pol_advogado",     "advogado",      "pt-br"),
            ("pol_arquiteto",    "arquiteto",     "pt-br"),
            ("pol_economista",   "economista",    "pt-br"),
            ("pol_jornalista",   "jornalista",    "pt-br"),
            ("pol_psicólogo",    "psicólogo",     "pt-br"),
            ("pol_veterinario",  "veterinário",   "pt-br"),
            # Ambiente e natureza
            ("pol_sustentabilidade","sustentabilidade","pt-br"),
            ("pol_biodiversidade","biodiversidade","pt-br"),
            ("pol_ecossistema",  "ecossistema",   "pt-br"),
            ("pol_fotossintese", "fotossíntese",  "pt-br"),
            ("pol_terremoto",    "terremoto",     "pt-br"),
            ("pol_vulcao",       "vulcão",        "pt-br"),
            ("pol_furacao",      "furacão",       "pt-br"),
            ("pol_atmosfera",    "atmosfera",     "pt-br"),
            ("pol_temperatura",  "temperatura",   "pt-br"),
            ("pol_precipitacao", "precipitação",  "pt-br"),
            # Alimentos
            ("pol_abacaxi",      "abacaxi",       "pt-br"),
            ("pol_borboleta",    "borboleta",     "pt-br"),
            ("pol_elefante",     "elefante",      "pt-br"),
            ("pol_crocodilo",    "crocodilo",     "pt-br"),
            ("pol_hipopotamo",   "hipopótamo",    "pt-br"),
            ("pol_girassol",     "girassol",      "pt-br"),
            ("pol_maracuja",     "maracujá",      "pt-br"),
            ("pol_pitanga",      "pitanga",       "pt-br"),
            ("pol_jabuticaba",   "jabuticaba",    "pt-br"),
            ("pol_sapoti",       "sapoti",        "pt-br"),
        ],
    },
}

# ─── TTS via espeak-ng ────────────────────────────────────────────────────────

def sintetizar_espeak(
    texto: str,
    lang: str = "pt-br",
    variacao: bool = False,
) -> tuple | None:
    """Sintetiza texto com espeak-ng e retorna (samples float32, sample_rate).

    Se variacao=True, randomiza pitch e velocidade para treino robusto.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    speed = "140"
    pitch = "50"
    if variacao:
        speed = str(random.randint(120, 180))
        pitch = str(random.randint(30, 70))

    try:
        cmd = [
            "espeak-ng",
            "-v", lang,
            "-s", speed,
            "-p", pitch,
            "-a", "180",
            "-w", tmp_path,
            texto,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0:
            return None

        sr, data = wavfile.read(tmp_path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, int(sr)
    except Exception:
        return None
    finally:
        try: os.unlink(tmp_path)
        except: pass

# ─── FFT ──────────────────────────────────────────────────────────────────────

def energia_db(frame: np.ndarray) -> float:
    rms = np.sqrt(np.mean(frame ** 2))
    return -100.0 if rms < 1e-10 else 20.0 * np.log10(rms)

def frame_para_bins(frame: np.ndarray, sr: int) -> list | None:
    if energia_db(frame) < SILENCIO_DB:
        return None
    n       = len(frame)
    hann    = np.hanning(n)
    fft_mag = np.abs(np.fft.rfft(frame * hann))
    freqs   = np.fft.rfftfreq(n, 1 / sr)
    mx = fft_mag.max()
    if mx < 1e-9:
        return None
    fft_norm = fft_mag / mx
    mask = (freqs >= 80.0) & (freqs <= 8000.0)
    f_f, a_f = freqs[mask], fft_norm[mask]
    if len(f_f) > MAX_BINS:
        idx  = np.round(np.linspace(0, len(f_f) - 1, MAX_BINS)).astype(int)
        f_f  = f_f[idx]
        a_f  = a_f[idx]
    return [[float(f), float(a)] for f, a in zip(f_f, a_f)]

def amostras_para_frames(samples: np.ndarray, sr: int):
    frame_size = int(sr * FRAME_MS / 1000)
    if frame_size < 8:
        return
    for start in range(0, len(samples) - frame_size, frame_size):
        frame = samples[start:start + frame_size]
        bins  = frame_para_bins(frame, sr)
        if bins:
            yield bins

# ─── Grounding fonético ───────────────────────────────────────────────────────

DIGRAFOS = {"lh", "nh", "ch", "rr", "ss", "qu", "gu", "sc", "xc"}

def decompor_grafema(texto: str) -> list[str]:
    """
    Decompõe uma palavra em unidades fonéticas:
    - Dígrafos (lh, nh, ch, rr...) → 1 unidade
    - Demais letras → 1 unidade por letra
    Exemplos:
      "gato" → ["g", "a", "t", "o"]
      "filho"→ ["f", "i", "lh", "o"]
      "carro"→ ["c", "a", "rr", "o"]
    """
    letras = texto.lower()
    unidades = []
    i = 0
    while i < len(letras):
        if i + 1 < len(letras) and letras[i:i+2] in DIGRAFOS:
            unidades.append(letras[i:i+2])
            i += 2
        elif letras[i].isalpha():
            unidades.append(letras[i])
            i += 1
        else:
            i += 1
    return unidades


# ─── WebSocket ────────────────────────────────────────────────────────────────

async def _enviar_learn(ws, texto: str, letras: list, pausa: float = 0.020):
    """Envia componente escrito: learn palavra, learn cada letra, learn_frase letras."""
    await ws.send(json.dumps({
        "action":   "learn",
        "word":     texto,
        "context":  "vocabulário",
        "valence":  0.7,
        "strength": 0.85,
    }))
    await asyncio.sleep(pausa)
    for letra in letras:
        await ws.send(json.dumps({
            "action":   "learn",
            "word":     letra,
            "context":  "vocabulário",
            "valence":  0.5,
            "strength": 0.80,
        }))
        await asyncio.sleep(pausa)
    if len(letras) >= 2:
        await ws.send(json.dumps({
            "action": "learn_frase",
            "words":  letras,
        }))
        await asyncio.sleep(pausa)


async def enviar_palavra(ws, palavra_id: str, texto: str, lang: str, rep: int) -> int:
    """
    Sintetiza e envia uma palavra N vezes com grounding fonético + escrita simultânea.

    Loop por repetição:
      1. Envia todos os frames FFT da palavra (Selene ouve o som)
         → ultimo_padrao_audio fica atualizado no Rust
      2. Envia grounding_fonetico com grafema + letras decompostas
         → Rust faz grounding_bind(audio_spike, [grafema, letras...])
      3. Envia learn (escrita): palavra, cada letra, learn_frase das letras
         → grafo_associacoes recebe a palavra e as letras como nós simbólicos
    """
    resultado = sintetizar_espeak(texto, lang, variacao=VARIACAO)
    if resultado is None:
        print(f"\n    ⚠  '{texto}' — espeak-ng falhou")
        return 0

    samples, sr = resultado
    if sr != SAMPLE_RATE:
        from math import gcd
        g = gcd(SAMPLE_RATE, sr)
        from scipy.signal import resample_poly
        samples = resample_poly(
            samples, SAMPLE_RATE // g, sr // g
        ).astype(np.float32)
        sr = SAMPLE_RATE

    frames = list(amostras_para_frames(samples, sr))
    if not frames:
        print(f"\n    ⚠  '{texto}' — silêncio total após síntese")
        return 0

    letras = decompor_grafema(texto)
    enviados = 0

    for rep_idx in range(rep):
        # Variação: ressintetiza a cada 5 repetições com pitch/speed diferente
        if VARIACAO and rep_idx > 0 and rep_idx % 5 == 0:
            novo = sintetizar_espeak(texto, lang, variacao=True)
            if novo is not None:
                s2, sr2 = novo
                if sr2 != SAMPLE_RATE:
                    from math import gcd as _gcd
                    from scipy.signal import resample_poly as _rsp
                    g2 = _gcd(SAMPLE_RATE, sr2)
                    s2 = _rsp(
                        s2, SAMPLE_RATE // g2, sr2 // g2
                    ).astype(np.float32)
                frames = list(amostras_para_frames(s2, SAMPLE_RATE))

        # 1. Envia frames FFT (Selene ouve o som)
        for bins in frames:
            # Variação: ruído gaussiano nos bins (SNR ~10dB)
            bins_env = bins
            if VARIACAO:
                snr_fator = random.uniform(0.02, 0.08)
                bins_env = [
                    [f, max(0.0, a + random.gauss(0.0, snr_fator))]
                    for f, a in bins
                ]
            msg = json.dumps({
                "action":     "learn_audio_fft",
                "fft":        bins_env,
                "duracao_ms": FRAME_MS,
                "referencia": f"palavras:{palavra_id}",
            })
            await ws.send(msg)
            for _ in range(3):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    if json.loads(raw).get("event") == "audio_ack":
                        break
                except (asyncio.TimeoutError, json.JSONDecodeError):
                    break
            enviados += 1
            await asyncio.sleep(PAUSA)

        # 2. Grounding fonético: conecta padrão de onda → letras
        grounding_msg = json.dumps({
            "action":  "grounding_fonetico",
            "grafema": texto,
            "letras":  letras,
            "fonte":   "palavras_curriculum",
        })
        await ws.send(grounding_msg)
        for _ in range(3):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                if json.loads(raw).get("event") == "grounding_ack":
                    break
            except (asyncio.TimeoutError, json.JSONDecodeError):
                break

        # 3. Componente escrito: ensina palavra + letras no grafo simbólico
        await _enviar_learn(ws, texto, letras)

    return enviados

# ─── Main ─────────────────────────────────────────────────────────────────────

async def main_async():
    fases = [FASE_ALVO] if FASE_ALVO else sorted(CURRICULO.keys())

    total_palavras = sum(len(CURRICULO[f]["itens"]) for f in fases if f in CURRICULO)

    print("=" * 60)
    print("  PALAVRAS — TREINAMENTO DE VOCABULÁRIO SELENE BRAIN 2.0")
    print("=" * 60)
    print(f"  Fases: {fases}  |  Repetições/palavra: {REPETICOES}")
    print(f"  Total de palavras: {total_palavras}")
    print(f"  WebSocket: {WS_URL}")
    print("=" * 60)

    for fase_num in fases:
        fase = CURRICULO.get(fase_num)
        if not fase:
            print(f"\nFase {fase_num} não encontrada.")
            continue

        print(f"\n{'─'*60}")
        print(f"  FASE {fase_num}: {fase['nome'].upper()}")
        print(f"  {fase['descricao']}")
        print(f"  {len(fase['itens'])} palavras × {REPETICOES} reps")
        print(f"{'─'*60}")

        backoff = 1.0
        for tentativa in range(3):
            try:
                async with websockets.connect(
                    WS_URL, ping_interval=None, max_queue=512, open_timeout=15
                ) as ws:
                    print(f"  Conectado!")
                    t0 = time.time()
                    total_frames = 0

                    for idx, (pid, texto, lang) in enumerate(fase["itens"]):
                        print(f"\r  [{idx+1}/{len(fase['itens'])}] '{texto}' ...", end="", flush=True)
                        n = await enviar_palavra(ws, pid, texto, lang, REPETICOES)
                        total_frames += n

                    elapsed = time.time() - t0
                    print(f"\n  Fase {fase_num} concluída: {total_frames} frames em {elapsed:.1f}s")

                    try:
                        await ws.send(json.dumps({"action": "export_linguagem"}))
                    except Exception:
                        pass
                break

            except (ConnectionRefusedError, OSError) as e:
                print(f"\n  Selene não acessível ({e}). Aguardando {backoff:.0f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            except websockets.exceptions.ConnectionClosed:
                print(f"\n  Conexão perdida. Tentativa {tentativa+1}/3...")
                await asyncio.sleep(2.0)

    print("\n" + "=" * 60)
    print("  Treinamento de vocabulário concluído!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\nTreinamento interrompido pelo usuário.")
