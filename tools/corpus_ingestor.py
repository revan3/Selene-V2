#!/usr/bin/env python3
# =============================================================================
# tools/corpus_ingestor.py
# Pipeline PDF → selene_linguagem.json
#
# Transforma uma biblioteca de PDFs em linguagem natural para a Selene.
# Não substitui o que ela já sabe — ACUMULA sobre o grafo existente.
#
# O que este script faz:
#   1. Extrai texto de todos os PDFs na pasta especificada
#   2. Lemmatiza o texto em português (spaCy pt_core_news_lg)
#   3. Filtra stop-words e tokens irrelevantes
#   4. Computa PPMI (co-ocorrências com peso de relevância real)
#   5. Atribui valências de sentimento (OpLexicon PT ou heurística)
#   6. Mescla com o grafo existente (selene_linguagem.json)
#   7. Salva o resultado — Selene carrega na próxima inicialização
#
# Dependências:
#   pip install pdfplumber spacy tqdm
#   python -m spacy download pt_core_news_lg
#
# Uso:
#   python corpus_ingestor.py --pdf-dir "C:/meus_livros" --output "."
#   python corpus_ingestor.py --pdf-dir "C:/meus_livros" --output "." --janela 7 --min-freq 3
#
# Argumentos:
#   --pdf-dir   : pasta com os PDFs (recursivo)
#   --output    : pasta de saída (onde estão os arquivos .json da Selene)
#   --janela    : tamanho da janela de co-ocorrência (padrão: 5)
#   --min-freq  : frequência mínima para incluir uma palavra (padrão: 3)
#   --max-vocab : tamanho máximo do vocabulário (padrão: 50000)
#   --max-vizinhos : máximo de vizinhos por nó no grafo (padrão: 30)
#   --modo      : "acumular" (default) ou "substituir" o grafo existente
# =============================================================================

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ─── Dependências opcionais ───────────────────────────────────────────────────
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("⚠  pdfplumber não encontrado. Instale: pip install pdfplumber")

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    print("⚠  spaCy não encontrado. Instale: pip install spacy && python -m spacy download pt_core_news_lg")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ─── Constantes ───────────────────────────────────────────────────────────────

# Stop-words espelhadas do STOP_WORDS do server.rs + adicionais para corpus
STOP_WORDS = {
    "eu","tu","ele","ela","nós","vós","eles","elas",
    "meu","minha","meus","minhas","seu","sua","seus","suas",
    "que","qual","quem","onde","como","quando","porque","pois","porém",
    "de","da","do","em","na","no","para","com","por","ao","às","aos",
    "um","uma","os","as","uns","umas","o","a",
    "me","te","se","nos","vos","lhe","lhes",
    "é","são","foi","era","ser","estar","ter","há",
    "sim","não","já","ainda","muito","mais","menos","bem","mal",
    "isso","este","esta","estes","estas","esse","essa","esses","essas",
    "aqui","ali","lá","então","também","só","até",
    # adicionais para corpus
    "também","ser","estar","ter","haver","ir","vir","fazer","dizer",
    "ver","dar","querer","poder","saber","ficar","passar","deixar",
    "falar","pôr","trazer","levar","tomar","chegar","começar","seguir",
    "encontrar","comer","morar","entrar","sair","abrir","fechar","pagar",
    "escrever","ler","ouvir","sentir","pensar","acreditar","achar",
    "dever","precisar","conseguir","tentar","continuar","voltar","usar",
    "mas","porém","contudo","todavia","entretanto","pois","porque","logo",
    "embora","apesar","mesmo","assim","além","ainda","tal","tanto","tão",
    "cada","todo","toda","todos","todas","outro","outra","outros","outras",
    "certo","certa","mesmo","próprio","própria","mesmo","si","que","aquilo",
}

# Valência base para categorias gramaticais sem léxico de sentimento
VALENCIAS_CATEGORIAS = {
    # Substantivos concretos tendem para neutro
    "NOUN": 0.0,
    # Adjetivos positivos/negativos — determinados pelo léxico
    "ADJ": 0.0,
    # Verbos tendem para neutro
    "VERB": 0.0,
    # Advérbios
    "ADV": 0.0,
}

# Léxico de sentimento embutido (palavras com valência forte e inequívoca)
# Expandido com termos de domínio relevante para literatura fantasia + ciência
LEXICO_SENTIMENTO_BASE = {
    # Positivas fortes
    "amor": 0.95, "alegria": 0.90, "feliz": 0.85, "felicidade": 0.90,
    "esperança": 0.80, "paz": 0.80, "bem": 0.75, "belo": 0.75,
    "bom": 0.70, "boa": 0.70, "lindo": 0.75, "linda": 0.75,
    "vitória": 0.85, "sucesso": 0.85, "conquista": 0.80, "prazer": 0.80,
    "luz": 0.70, "vida": 0.65, "cura": 0.75, "harmonia": 0.80,
    "coragem": 0.85, "herói": 0.80, "heroína": 0.80, "glória": 0.85,
    "sabedoria": 0.80, "conhecimento": 0.70, "aprendizado": 0.70,
    "descoberta": 0.75, "inovação": 0.75, "criatividade": 0.75,
    "amizade": 0.85, "companheiro": 0.75, "aliado": 0.70,
    "maravilha": 0.85, "incrível": 0.80, "extraordinário": 0.80,
    "abundância": 0.75, "riqueza": 0.65, "prosperidade": 0.75,
    "calma": 0.70, "serenidade": 0.75, "equilíbrio": 0.70,
    "evolução": 0.70, "progresso": 0.70, "avanço": 0.70,
    "consciência": 0.65, "despertar": 0.70, "iluminação": 0.80,

    # Negativas fortes
    "medo": -0.90, "terror": -0.95, "horror": -0.95, "pavor": -0.90,
    "ódio": -0.95, "raiva": -0.80, "fúria": -0.85, "ira": -0.80,
    "tristeza": -0.75, "dor": -0.80, "sofrimento": -0.85, "angústia": -0.85,
    "morte": -0.70, "destruição": -0.90, "caos": -0.75, "guerra": -0.80,
    "derrota": -0.80, "fracasso": -0.75, "falha": -0.65, "erro": -0.55,
    "traição": -0.90, "mentira": -0.80, "engano": -0.80, "manipulação": -0.85,
    "crueldade": -0.90, "violência": -0.90, "brutalidade": -0.90,
    "escuridão": -0.60, "trevas": -0.70, "mal": -0.85, "vilão": -0.75,
    "doença": -0.70, "veneno": -0.80, "corrupção": -0.85,
    "solidão": -0.60, "abandono": -0.75, "rejeição": -0.70,
    "desespero": -0.85, "desgraça": -0.85, "ruína": -0.80,

    # Fantasia — específicas
    "dragão": 0.60, "magia": 0.70, "feitiço": 0.55, "mágico": 0.65,
    "espada": 0.30, "escudo": 0.40, "guerreiro": 0.50, "mago": 0.55,
    "elfo": 0.55, "anão": 0.40, "hobbit": 0.60, "orc": -0.40,
    "monstro": -0.50, "demônio": -0.75, "diabo": -0.80,
    "reino": 0.50, "castelo": 0.45, "floresta": 0.55, "montanha": 0.45,
    "aventura": 0.75, "missão": 0.65, "destino": 0.60, "profecia": 0.55,
    "aliança": 0.70, "traidor": -0.85, "inimigo": -0.65, "vilania": -0.80,

    # Ciência — específicas
    "descoberta": 0.75, "hipótese": 0.60, "experimento": 0.60,
    "teoria": 0.55, "evidência": 0.60, "prova": 0.65, "ciência": 0.70,
    "inovação": 0.75, "tecnologia": 0.65, "inteligência": 0.70,
    "neurônio": 0.50, "sinapse": 0.50, "cérebro": 0.55, "mente": 0.60,
    "aprendizado": 0.70, "memória": 0.60, "pensamento": 0.65,
    "evolução": 0.70, "adaptação": 0.65, "sobrevivência": 0.60,
}


# ─── Extração de texto ────────────────────────────────────────────────────────

def extrair_texto_pdf(caminho_pdf: str) -> str:
    """Extrai texto bruto de um PDF usando pdfplumber."""
    if not HAS_PDF:
        return ""
    try:
        texto = []
        with pdfplumber.open(caminho_pdf) as pdf:
            for pagina in pdf.pages:
                t = pagina.extract_text()
                if t:
                    texto.append(t)
        return "\n".join(texto)
    except Exception as e:
        print(f"  ⚠  Erro ao ler {os.path.basename(caminho_pdf)}: {e}")
        return ""


def listar_pdfs(pasta: str) -> list:
    """Lista todos os PDFs recursivamente numa pasta."""
    pdfs = []
    for root, _, files in os.walk(pasta):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))
    return sorted(pdfs)


# ─── Processamento linguístico ────────────────────────────────────────────────

def carregar_spacy():
    """Carrega o modelo spaCy português (lg para melhor qualidade)."""
    if not HAS_SPACY:
        return None
    for modelo in ["pt_core_news_lg", "pt_core_news_md", "pt_core_news_sm"]:
        try:
            nlp = spacy.load(modelo, disable=["parser", "ner"])
            print(f"✓ spaCy carregado: {modelo}")
            return nlp
        except OSError:
            continue
    print("⚠  Nenhum modelo spaCy PT encontrado. Use tokenização simples.")
    return None


def tokenizar_simples(texto: str) -> list:
    """Tokenização e lemmatização simples sem spaCy (fallback)."""
    # Normaliza caracteres especiais
    texto = texto.lower()
    # Divide por não-alfanumérico mantendo acentuação
    tokens = re.findall(r"[a-záéíóúâêôãõçàü]{3,}", texto)
    return [t for t in tokens if t not in STOP_WORDS]


def processar_com_spacy(texto: str, nlp, min_len: int = 3) -> list:
    """Tokeniza, lemmatiza e filtra com spaCy."""
    tokens = []
    # Processa em chunks para textos grandes
    tamanho_chunk = 100000
    for i in range(0, len(texto), tamanho_chunk):
        chunk = texto[i:i + tamanho_chunk]
        doc = nlp(chunk)
        for token in doc:
            lema = token.lemma_.lower().strip()
            if (len(lema) >= min_len
                    and lema not in STOP_WORDS
                    and token.is_alpha
                    and not token.is_stop
                    and token.pos_ in ("NOUN", "VERB", "ADJ", "ADV", "PROPN")):
                tokens.append(lema)
    return tokens


# ─── PPMI ─────────────────────────────────────────────────────────────────────

def calcular_ppmi(
    tokens: list,
    janela: int = 5,
    min_freq: int = 3,
    max_vocab: int = 50000,
) -> tuple:
    """
    Calcula co-ocorrências com PPMI (Pointwise Mutual Information positivo).

    PPMI(w1, w2) = max(0, log2( P(w1,w2) / (P(w1) * P(w2)) ))

    Retorna (freq_unigrama, cooc_ppmi) onde:
      freq_unigrama: Counter palavra → frequência
      cooc_ppmi: dict (w1, w2) → ppmi_score
    """
    # Frequência unigrama
    freq = Counter(tokens)

    # Filtra vocabulário pelos mais frequentes
    vocab = {w for w, c in freq.most_common(max_vocab) if c >= min_freq}
    print(f"  Vocabulário filtrado: {len(vocab):,} palavras únicas")

    tokens_filtrados = [t for t in tokens if t in vocab]
    N = len(tokens_filtrados)
    if N == 0:
        return freq, {}

    # Frequência co-ocorrência na janela deslizante
    cooc_count: dict = defaultdict(int)
    for i, w1 in enumerate(tokens_filtrados):
        for j in range(max(0, i - janela), min(N, i + janela + 1)):
            if i != j:
                w2 = tokens_filtrados[j]
                if w1 != w2:
                    par = (min(w1, w2), max(w1, w2))
                    cooc_count[par] += 1

    total_cooc = sum(cooc_count.values())
    freq_vocab = {w: freq[w] for w in vocab}
    total_uni = sum(freq_vocab.values())

    # Calcula PPMI
    cooc_ppmi: dict = {}
    for (w1, w2), cnt in cooc_count.items():
        if cnt < 2:
            continue
        p_w1w2 = cnt / total_cooc
        p_w1 = freq_vocab.get(w1, 1) / total_uni
        p_w2 = freq_vocab.get(w2, 1) / total_uni
        pmi = math.log2(p_w1w2 / (p_w1 * p_w2 + 1e-12))
        if pmi > 0:
            cooc_ppmi[(w1, w2)] = min(pmi / 10.0, 1.0)  # normaliza para [0, 1]

    return freq_vocab, cooc_ppmi


# ─── Valências ────────────────────────────────────────────────────────────────

def calcular_valencia(palavra: str, nlp=None) -> float:
    """
    Retorna valência emocional da palavra em [-1.0, 1.0].
    Prioridade: léxico base → 0.0 (neutro)
    """
    return LEXICO_SENTIMENTO_BASE.get(palavra, 0.0)


def enriquecer_valencias_por_contexto(
    freq: dict,
    cooc_ppmi: dict,
    valencias_base: dict,
) -> dict:
    """
    Propaga valências via co-ocorrência: palavras que co-ocorrem muito
    com palavras com valência forte ganham uma valência derivada (±30% da vizinha).
    Limita a 2 iterações para não contaminar o vocabulário inteiro.
    """
    valencias = dict(valencias_base)
    palavras_com_valencia = set(valencias.keys())

    for _ in range(2):  # 2 iterações de propagação
        novas = {}
        # Constrói adjacência: palavra → [(vizinha, ppmi)]
        adj: dict = defaultdict(list)
        for (w1, w2), ppmi in cooc_ppmi.items():
            adj[w1].append((w2, ppmi))
            adj[w2].append((w1, ppmi))

        for palavra, vizinhos in adj.items():
            if palavra in palavras_com_valencia:
                continue
            # Valência derivada: média ponderada por PPMI das vizinhas com valência
            soma_val = 0.0
            soma_peso = 0.0
            for viz, ppmi in vizinhos:
                if viz in valencias and abs(valencias[viz]) > 0.3:
                    soma_val += valencias[viz] * ppmi * 0.30  # 30% de propagação
                    soma_peso += ppmi
            if soma_peso > 0.5:  # só propaga se vizinhança forte
                val_derivada = (soma_val / soma_peso)
                if abs(val_derivada) > 0.05:
                    novas[palavra] = round(val_derivada, 4)

        valencias.update(novas)
        palavras_com_valencia.update(novas.keys())

    return valencias


# ─── Frases padrão ────────────────────────────────────────────────────────────

def extrair_frases_padrao(
    texto: str,
    nlp,
    max_frases: int = 200,
    max_tokens_frase: int = 8,
    min_tokens_frase: int = 3,
) -> list:
    """
    Extrai frases curtas (3-8 tokens, após filtragem) do corpus para usar
    como frases_padrao. Filtra frases com stop-words apenas, prefere
    frases com substantivos e adjetivos.
    """
    frases = []
    # Divide o texto em sentenças simples (por pontuação)
    sentencas = re.split(r'[.!?;]\s+', texto)

    for s in sentencas:
        s = s.strip().lower()
        if len(s) < 10 or len(s) > 200:
            continue

        if nlp:
            doc = nlp(s)
            tokens_frase = [
                t.lemma_.lower() for t in doc
                if t.is_alpha and len(t.lemma_) >= 3
                and not t.is_stop and t.lemma_.lower() not in STOP_WORDS
                and t.pos_ in ("NOUN", "VERB", "ADJ", "PROPN")
            ]
        else:
            tokens_frase = tokenizar_simples(s)

        if min_tokens_frase <= len(tokens_frase) <= max_tokens_frase:
            frases.append(tokens_frase)

        if len(frases) >= max_frases:
            break

    # Remove duplicatas mantendo ordem
    vistas = set()
    frases_unicas = []
    for f in frases:
        chave = tuple(f)
        if chave not in vistas:
            vistas.add(chave)
            frases_unicas.append(f)

    return frases_unicas[:max_frases]


# ─── Grafo ────────────────────────────────────────────────────────────────────

def construir_grafo(
    cooc_ppmi: dict,
    max_vizinhos: int = 30,
) -> dict:
    """
    Converte o dicionário PPMI em grafo de adjacência:
    { palavra: [["vizinha", ppmi], ...] }
    Limita a max_vizinhos vizinhos por nó (os de maior PPMI).
    """
    adj: dict = defaultdict(list)
    for (w1, w2), ppmi in cooc_ppmi.items():
        adj[w1].append((w2, ppmi))
        adj[w2].append((w1, ppmi))

    grafo = {}
    for palavra, vizinhos in adj.items():
        vizinhos_ordenados = sorted(vizinhos, key=lambda x: -x[1])[:max_vizinhos]
        grafo[palavra] = [[v, round(p, 4)] for v, p in vizinhos_ordenados]

    return grafo


# ─── Mescla com JSON existente ────────────────────────────────────────────────

def carregar_json_existente(caminho: str) -> dict:
    """Carrega selene_linguagem.json existente ou retorna estrutura vazia."""
    if not os.path.exists(caminho):
        return {"selene_linguagem_v1": {
            "metadata": {},
            "vocabulario": {},
            "associacoes": {},
            "frases_padrao": [],
        }}
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def mesclar_grafo(existente: dict, novo: dict, modo: str = "acumular") -> dict:
    """
    Mescla o novo grafo com o existente.
    Modo "acumular": soma pesos (+ reforço) para arestas já existentes.
    Modo "substituir": sobrescreve completamente.
    """
    if modo == "substituir":
        return novo

    resultado = {k: dict(vs) for k, vss in existente.items() for k, vs in [(k, {v[0]: v[1] for v in vss})]}
    # Converte existente para dict fácil de manipular
    grafo_existente: dict = {}
    for palavra, vizinhos in existente.items():
        grafo_existente[palavra] = {v[0]: v[1] for v in vizinhos}

    for palavra, vizinhos_novos in novo.items():
        if palavra not in grafo_existente:
            grafo_existente[palavra] = {}
        for vizinho_item in vizinhos_novos:
            viz = vizinho_item[0]
            peso = vizinho_item[1]
            if viz in grafo_existente[palavra]:
                # Acumula: média ponderada (peso existente 70%, novo 30%)
                grafo_existente[palavra][viz] = round(
                    grafo_existente[palavra][viz] * 0.70 + peso * 0.30, 4
                )
            else:
                grafo_existente[palavra][viz] = peso

    # Converte de volta para formato lista
    return {
        palavra: [[v, p] for v, p in sorted(vizinhos.items(), key=lambda x: -x[1])]
        for palavra, vizinhos in grafo_existente.items()
    }


def mesclar_valencias(existentes: dict, novas: dict) -> dict:
    """Mescla valências: EMA (70% existente, 30% novo) para palavras já conhecidas."""
    resultado = dict(existentes)
    for palavra, val in novas.items():
        if palavra in resultado:
            resultado[palavra] = round(resultado[palavra] * 0.70 + val * 0.30, 4)
        else:
            resultado[palavra] = round(val, 4)
    return resultado


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline PDF → selene_linguagem.json para Selene Brain 2.0"
    )
    parser.add_argument("--pdf-dir", required=True,
        help="Pasta contendo os PDFs (recursivo)")
    parser.add_argument("--output", default=".",
        help="Pasta de saída com os arquivos JSON da Selene (padrão: .)")
    parser.add_argument("--janela", type=int, default=5,
        help="Tamanho da janela de co-ocorrência (padrão: 5)")
    parser.add_argument("--min-freq", type=int, default=3,
        help="Frequência mínima de uma palavra para entrar no vocab (padrão: 3)")
    parser.add_argument("--max-vocab", type=int, default=50000,
        help="Tamanho máximo do vocabulário (padrão: 50000)")
    parser.add_argument("--max-vizinhos", type=int, default=30,
        help="Máximo de vizinhos por nó no grafo (padrão: 30)")
    parser.add_argument("--modo", choices=["acumular", "substituir"],
        default="acumular",
        help="acumular (default): soma ao existente | substituir: sobrescreve")
    parser.add_argument("--max-paginas-pdf", type=int, default=None,
        help="Máximo de páginas por PDF (para testar com corpus grande)")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║     SELENE CORPUS INGESTOR — Pipeline PDF → JSON    ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── 1. Lista PDFs ────────────────────────────────────────────────────────
    pdfs = listar_pdfs(args.pdf_dir)
    if not pdfs:
        print(f"✗  Nenhum PDF encontrado em: {args.pdf_dir}")
        sys.exit(1)
    print(f"📚 {len(pdfs)} PDFs encontrados em '{args.pdf_dir}'")

    # ── 2. Carrega spaCy ─────────────────────────────────────────────────────
    nlp = carregar_spacy()

    # ── 3. Extrai e tokeniza texto de todos os PDFs ──────────────────────────
    print("\n📖 Extraindo texto e tokenizando...")
    todos_tokens: list = []
    todos_textos: list = []  # para frases_padrao

    iterador = tqdm(pdfs) if HAS_TQDM else pdfs
    for caminho_pdf in iterador:
        nome = os.path.basename(caminho_pdf)
        if not HAS_TQDM:
            print(f"  → {nome}")

        texto = extrair_texto_pdf(caminho_pdf)
        if not texto.strip():
            continue

        # Limita texto para processamento
        if args.max_paginas_pdf:
            # Aproximação: ~3000 chars por página
            max_chars = args.max_paginas_pdf * 3000
            texto = texto[:max_chars]

        todos_textos.append(texto)

        if nlp:
            tokens = processar_com_spacy(texto, nlp)
        else:
            tokens = tokenizar_simples(texto)

        todos_tokens.extend(tokens)

    if not todos_tokens:
        print("✗  Nenhum token extraído. Verifique os PDFs e as dependências.")
        sys.exit(1)

    print(f"✓ Total de tokens extraídos: {len(todos_tokens):,}")

    # ── 4. Computa PPMI ──────────────────────────────────────────────────────
    print("\n🔢 Calculando PPMI...")
    freq, cooc_ppmi = calcular_ppmi(
        todos_tokens,
        janela=args.janela,
        min_freq=args.min_freq,
        max_vocab=args.max_vocab,
    )
    print(f"✓ {len(cooc_ppmi):,} pares de co-ocorrência com PPMI > 0")

    # ── 5. Calcula valências ─────────────────────────────────────────────────
    print("\n💗 Calculando valências de sentimento...")
    vocab_palavras = set(freq.keys())
    valencias_base = {w: calcular_valencia(w, nlp) for w in vocab_palavras}
    valencias_com_vocab = {w: v for w, v in valencias_base.items() if v != 0.0}

    # Propaga valências por contexto
    valencias_finais = enriquecer_valencias_por_contexto(
        freq, cooc_ppmi, valencias_com_vocab
    )
    # Completa com 0.0 para palavras sem valência
    for w in vocab_palavras:
        if w not in valencias_finais:
            valencias_finais[w] = 0.0

    palavras_com_val = sum(1 for v in valencias_finais.values() if abs(v) > 0.05)
    print(f"✓ {len(valencias_finais):,} palavras no vocabulário")
    print(f"  {palavras_com_val:,} com valência emocional significativa (|v|>0.05)")

    # ── 6. Constrói grafo ────────────────────────────────────────────────────
    print("\n🕸  Construindo grafo de associações...")
    grafo_novo = construir_grafo(cooc_ppmi, max_vizinhos=args.max_vizinhos)
    n_arestas_novas = sum(len(v) for v in grafo_novo.values())
    print(f"✓ {len(grafo_novo):,} nós, {n_arestas_novas:,} arestas")

    # ── 7. Extrai frases padrão do corpus ────────────────────────────────────
    print("\n📝 Extraindo frases padrão do corpus...")
    frases_novas = []
    # Usa apenas os primeiros 3 textos para frases (evitar viés de um livro só)
    for texto in todos_textos[:5]:
        frases_texto = extrair_frases_padrao(texto, nlp, max_frases=50)
        frases_novas.extend(frases_texto)
    print(f"✓ {len(frases_novas)} frases padrão extraídas do corpus")

    # ── 8. Mescla com JSON existente ─────────────────────────────────────────
    caminho_lingua = os.path.join(args.output, "selene_linguagem.json")
    print(f"\n🔄 Mesclando com: {caminho_lingua} (modo={args.modo})")

    dados_existentes = carregar_json_existente(caminho_lingua)
    lingua_v1 = dados_existentes.get("selene_linguagem_v1", {})

    # Vocabulário existente
    vocab_existente = lingua_v1.get("vocabulario", {})
    vocab_mesclado = mesclar_valencias(vocab_existente, valencias_finais)

    # Grafo existente
    grafo_existente = lingua_v1.get("associacoes", {})
    grafo_mesclado = mesclar_grafo(grafo_existente, grafo_novo, modo=args.modo)

    # Frases: acumula (sem duplicatas)
    frases_existentes = lingua_v1.get("frases_padrao", [])
    chaves_existentes = {tuple(f) for f in frases_existentes}
    frases_mescladas = list(frases_existentes)
    for f in frases_novas:
        if tuple(f) not in chaves_existentes:
            frases_mescladas.append(f)
            chaves_existentes.add(tuple(f))

    n_arestas_total = sum(len(v) for v in grafo_mesclado.values())

    # ── 9. Salva resultado ───────────────────────────────────────────────────
    resultado = {
        "selene_linguagem_v1": {
            "metadata": {
                "versao": "2.0",
                "gerado_por": "corpus_ingestor.py",
                "n_palavras": len(vocab_mesclado),
                "n_associacoes": n_arestas_total,
                "n_frases_padrao": len(frases_mescladas),
                "descricao": "Linguagem emergente Selene Brain 2.0 — corpus + frases_padrao"
            },
            "vocabulario": vocab_mesclado,
            "associacoes": grafo_mesclado,
            "frases_padrao": frases_mescladas,
        }
    }

    with open(caminho_lingua, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    # Atualiza também o selene_lexicon.json com as valências
    caminho_lexicon = os.path.join(args.output, "selene_lexicon.json")
    lexicon_novo = [
        {"word": w, "valence": float(v)}
        for w, v in sorted(vocab_mesclado.items())
        if abs(v) > 0.01  # só palavras com valência significativa
    ]
    with open(caminho_lexicon, "w", encoding="utf-8") as f:
        json.dump(lexicon_novo, f, ensure_ascii=False, indent=2)

    print(f"\n╔══════════════════════════════════════════════════════╗")
    print(f"║  ✓ CONCLUÍDO                                         ║")
    print(f"╠══════════════════════════════════════════════════════╣")
    print(f"║  Vocabulário total:    {len(vocab_mesclado):>10,} palavras          ║")
    print(f"║  Arestas no grafo:     {n_arestas_total:>10,} associações       ║")
    print(f"║  Frases padrão:        {len(frases_mescladas):>10,} frases            ║")
    print(f"║  Palavras c/ valência: {palavras_com_val:>10,}                    ║")
    print(f"╠══════════════════════════════════════════════════════╣")
    print(f"║  Arquivos atualizados:                               ║")
    print(f"║    {caminho_lingua:<48} ║")
    print(f"║    {caminho_lexicon:<48} ║")
    print(f"╚══════════════════════════════════════════════════════╝")
    print(f"\n🚀 Reinicie a Selene para carregar o novo vocabulário.\n")


if __name__ == "__main__":
    main()
