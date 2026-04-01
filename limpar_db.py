#!/usr/bin/env python3
# limpar_db.py -- Limpeza e diagnostico do banco de dados da Selene
#
# O que faz:
#   1. Remove LOG.old.* do RocksDB (logs antigos inuteis)
#   2. Remove nos orfaos do grafo (sem nenhum link)
#   3. Compacta o grafo removendo links com peso < threshold
#   4. Remove associacoes quebradas no vocabulario
#   5. Exibe relatorio antes/depois com tamanhos
#
# Uso:
#   python limpar_db.py                    # limpeza padrao
#   python limpar_db.py --threshold 0.01   # remove links mais fracos
#   python limpar_db.py --dry-run          # apenas mostra o que removeria
#   python limpar_db.py --grafo-only       # limpa so o grafo
#   python limpar_db.py --db-only          # limpa so o RocksDB

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# -- Configuracao --------------------------------------------------------------

BASE = Path(__file__).parent
DB_PATH = BASE / "selene_memories.db"
GRAFO_PATH = BASE / "grafo_selene.json"
LANG_PATH = BASE / "selene_linguagem.json"
HIPPO_PATH = BASE / "selene_hippo_ltp.json"

THRESHOLD_LINK = 0.001
DRY_RUN = "--dry-run" in sys.argv
GRAFO_ONLY = "--grafo-only" in sys.argv
DB_ONLY = "--db-only" in sys.argv

for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--threshold" and i + 1 < len(sys.argv):
        try:
            THRESHOLD_LINK = float(sys.argv[i + 1])
        except ValueError:
            pass


# -- Utilitarios ---------------------------------------------------------------

def tamanho_dir(path):
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def fmt(n_bytes):
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes / 1024 ** 2:.2f} MB"


def sep(n=60):
    print("-" * n)


def sep2(n=60):
    print("=" * n)


# -- 1. Limpeza do RocksDB ----------------------------------------------------

def limpar_rocksdb():
    if not DB_PATH.exists():
        print("  selene_memories.db nao encontrado -- pulando.")
        return

    antes = tamanho_dir(DB_PATH)
    logs_antigos = sorted(DB_PATH.glob("LOG.old.*"))
    removidos = 0
    bytes_removidos = 0

    for log in logs_antigos:
        sz = log.stat().st_size
        if DRY_RUN:
            print(f"  [DRY-RUN] removeria: {log.name}  ({fmt(sz)})")
        else:
            log.unlink()
            bytes_removidos += sz
            removidos += 1

    depois = tamanho_dir(DB_PATH)
    print(f"  RocksDB: {fmt(antes)} -> {fmt(depois)}  (-{fmt(antes - depois)})")
    print(
        f"  LOG.old removidos: {removidos}/{len(logs_antigos)}"
        f"  |  {fmt(bytes_removidos)} liberados"
    )


# -- 2. Limpeza do grafo -------------------------------------------------------

def limpar_grafo():
    if not GRAFO_PATH.exists():
        print("  grafo_selene.json nao encontrado -- pulando.")
        return

    antes = GRAFO_PATH.stat().st_size
    with open(GRAFO_PATH, encoding="utf-8") as f:
        grafo = json.load(f)

    nodes_orig = len(grafo.get("nodes", []))
    links_orig = len(grafo.get("links", []))

    # Remove links muito fracos
    links_filtrados = [
        lk for lk in grafo.get("links", [])
        if lk.get("weight", 1.0) >= THRESHOLD_LINK
    ]

    # Descobre nos conectados
    nos_conectados = set()
    for lk in links_filtrados:
        nos_conectados.add(lk.get("source", ""))
        nos_conectados.add(lk.get("target", ""))

    # Remove nos orfaos
    nodes_filtrados = [
        nd for nd in grafo.get("nodes", [])
        if nd.get("id") in nos_conectados
    ]

    grafo_limpo = {"nodes": nodes_filtrados, "links": links_filtrados}
    links_removidos = links_orig - len(links_filtrados)
    nos_removidos = nodes_orig - len(nodes_filtrados)

    if DRY_RUN:
        print(
            f"  [DRY-RUN] grafo: {links_removidos} links fracos"
            f" + {nos_removidos} nos orfaos seriam removidos"
        )
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = GRAFO_PATH.with_suffix(f".bak_{ts}.json")
    shutil.copy2(GRAFO_PATH, backup)
    with open(GRAFO_PATH, "w", encoding="utf-8") as f:
        json.dump(grafo_limpo, f, ensure_ascii=False, separators=(",", ":"))
    depois = GRAFO_PATH.stat().st_size
    print(f"  Grafo: {fmt(antes)} -> {fmt(depois)}  (-{fmt(antes - depois)})")
    print(
        f"  Nos: {nodes_orig} -> {len(nodes_filtrados)}  (-{nos_removidos} orfaos)"
    )
    print(
        f"  Links: {links_orig} -> {len(links_filtrados)}"
        f"  (-{links_removidos} fracos, threshold={THRESHOLD_LINK})"
    )
    print(f"  Backup: {backup.name}")


# -- 3. Limpeza do vocabulario -------------------------------------------------

def limpar_vocabulario():
    if not LANG_PATH.exists():
        print("  selene_linguagem.json nao encontrado -- pulando.")
        return

    antes = LANG_PATH.stat().st_size
    with open(LANG_PATH, encoding="utf-8") as f:
        lang = json.load(f)

    chave = next(iter(lang))
    inner = lang[chave]
    vocab = inner.get("vocabulario", {})
    # assocs: {palavra: [[alvo, peso], ...]}
    assocs = inner.get("associacoes", {})

    vocab_orig = len(vocab)

    # Remove palavras sem peso valido
    vocab_limpo = {
        k: v for k, v in vocab.items()
        if isinstance(v, (int, float)) and v > 0
    }

    # Remove associacoes com alvo inexistente no vocab
    assocs_limpo = {}
    if isinstance(assocs, dict):
        for palavra, alvos in assocs.items():
            if palavra not in vocab_limpo:
                continue
            if isinstance(alvos, list):
                # cada item e [alvo_str, peso_float]
                alvos_ok = [
                    par for par in alvos
                    if isinstance(par, list)
                    and len(par) == 2
                    and par[0] in vocab_limpo
                    and par[1] > 0
                ]
                if alvos_ok:
                    assocs_limpo[palavra] = alvos_ok

    inner["vocabulario"] = vocab_limpo
    if assocs:
        inner["associacoes"] = assocs_limpo
    lang[chave] = inner

    vocab_removidas = vocab_orig - len(vocab_limpo)
    assoc_orig = sum(len(v) for v in assocs.values()) if assocs else 0
    assoc_depois = sum(len(v) for v in assocs_limpo.values()) if assocs_limpo else 0

    if DRY_RUN:
        print(
            f"  [DRY-RUN] vocab: {vocab_removidas} invalidas,"
            f" {assoc_orig - assoc_depois} assocs quebradas"
        )
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = LANG_PATH.with_suffix(f".bak_{ts}.json")
    shutil.copy2(LANG_PATH, backup)
    with open(LANG_PATH, "w", encoding="utf-8") as f:
        json.dump(lang, f, ensure_ascii=False, separators=(",", ":"))
    depois = LANG_PATH.stat().st_size
    print(f"  Vocabulario: {fmt(antes)} -> {fmt(depois)}  (-{fmt(antes - depois)})")
    print(f"  Palavras: {vocab_orig} -> {len(vocab_limpo)}  (-{vocab_removidas})")
    if assoc_orig > 0:
        print(
            f"  Assocs: {assoc_orig} -> {assoc_depois}"
            f"  (-{assoc_orig - assoc_depois})"
        )
    print(f"  Backup: {backup.name}")


# -- 4. Relatorio final --------------------------------------------------------

def relatorio_final():
    sep()
    print("ESTADO FINAL DOS ARQUIVOS:")
    arquivos = [
        DB_PATH, GRAFO_PATH, LANG_PATH, HIPPO_PATH,
        BASE / "selene_qtable.bin",
        BASE / "selene_ego.json",
        BASE / "selene_manifest.json",
        BASE / "memoria_categorias.json",
    ]
    total = 0
    for p in arquivos:
        if p.exists():
            sz = tamanho_dir(p)
            total += sz
            print(f"  {p.name:<35} {fmt(sz):>10}")
    sep()
    print(f"  {'TOTAL':<35} {fmt(total):>10}")


# -- Main ----------------------------------------------------------------------

def main():
    sep2()
    print("  LIMPEZA DO BANCO DE DADOS -- SELENE BRAIN 2.0")
    if DRY_RUN:
        print("  MODO: DRY-RUN (nenhum arquivo sera alterado)")
    sep2()

    if not DB_ONLY:
        print("\n[1/3] Grafo de associacoes:")
        limpar_grafo()

        print("\n[2/3] Vocabulario:")
        limpar_vocabulario()

    if not GRAFO_ONLY:
        print("\n[3/3] RocksDB (selene_memories.db):")
        limpar_rocksdb()

    print()
    relatorio_final()
    if DRY_RUN:
        print("\nDRY-RUN concluido -- nenhum arquivo foi alterado.")
    else:
        print("\nLimpeza concluida.")


if __name__ == "__main__":
    main()
