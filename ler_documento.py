#!/usr/bin/env python3
"""
Selene — Leitor de Documentos (V4.6.1)
=======================================
Alimenta a Selene com textos longos / livros / transcrições de audiobooks
SEM perda e EM ORDEM, usando o novo caminho de ingestão dedicado (handler WS
"ingest"), que NÃO sofre o dedup/rate-limit/descarte do microfone ambiente.

Converte o conteúdo em neurônios funcionais via o STDP REAL da Selene (ela
"lê/ouve" de verdade) — não sintetiza pesos artificiais.

PRÉ-REQUISITO: Selene a correr em ws://HOST:PORTA/selene.
  pip install websockets         (obrigatório)
  pip install pypdf ebooklib bs4 (opcional: .pdf / .epub)
  pip install faster-whisper     (opcional: --audio para audiobooks)

USO:
  python ler_documento.py livro.txt
  python ler_documento.py livro.pdf --token MEU_TOKEN
  python ler_documento.py capitulo.epub --host 192.168.0.10
  python ler_documento.py audiobook.mp3 --audio          # transcreve e ingere
"""

import argparse
import asyncio
import sys
from pathlib import Path

try:
    import websockets
except ImportError:
    print("[FAIL] Instale: pip install websockets")
    sys.exit(1)

import json

# Quantas frases por mensagem WS (batch). A fila do lado da Selene não descarta;
# o batch é só para dar progresso e evitar um frame WS gigante.
FRASES_POR_LOTE = 400


# ──────────────────────────────────────────────────────────────────────────
# Extração de texto por formato (todos com fallback gracioso)
# ──────────────────────────────────────────────────────────────────────────
def extrair_texto(caminho: Path) -> str:
    ext = caminho.suffix.lower()
    if ext in (".txt", ".md", ""):
        return caminho.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print("[FAIL] Para .pdf: pip install pypdf")
            sys.exit(1)
        reader = PdfReader(str(caminho))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if ext == ".epub":
        try:
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError:
            print("[FAIL] Para .epub: pip install ebooklib beautifulsoup4")
            sys.exit(1)
        livro = epub.read_epub(str(caminho))
        partes = []
        for item in livro.get_items():
            if item.get_type() == 9:  # DOCUMENT
                soup = BeautifulSoup(item.get_content(), "html.parser")
                partes.append(soup.get_text(" "))
        return "\n".join(partes)
    print(f"[FAIL] Formato não suportado: {ext}. Use .txt/.md/.pdf/.epub ou --audio.")
    sys.exit(1)


def transcrever_audio(caminho: Path) -> str:
    """Audiobook → texto via faster-whisper (opcional)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[FAIL] Para --audio: pip install faster-whisper")
        sys.exit(1)
    print(f"[whisper] Transcrevendo {caminho.name} (pode demorar)...")
    modelo = WhisperModel("small", device="auto", compute_type="int8")
    segmentos, _ = modelo.transcribe(str(caminho), language="pt")
    texto = " ".join(seg.text for seg in segmentos)
    print(f"[whisper] {len(texto)} caracteres transcritos.")
    return texto


def segmentar_frases(texto: str) -> list[str]:
    """Divide em frases preservando a ordem (a Selene re-segmenta também)."""
    import re
    brutos = re.split(r"[.!?;\n\r]+", texto)
    return [f.strip() for f in brutos if len(f.strip()) > 2]


# ──────────────────────────────────────────────────────────────────────────
# Envio
# ──────────────────────────────────────────────────────────────────────────
async def enviar(frases: list[str], url: str):
    total = len(frases)
    enviados = 0
    async with websockets.connect(url, max_size=None, ping_interval=20) as ws:
        for i in range(0, total, FRASES_POR_LOTE):
            lote = frases[i:i + FRASES_POR_LOTE]
            await ws.send(json.dumps({"action": "ingest", "content": ". ".join(lote)}))
            # Aguarda o ack (confirma enfileiramento sem perda)
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=10)
                data = json.loads(resp)
                if data.get("type") == "ingest_ack":
                    enviados += data.get("chunks_enfileirados", 0)
            except (asyncio.TimeoutError, json.JSONDecodeError):
                pass
            print(f"  → {min(i + FRASES_POR_LOTE, total)}/{total} frases enviadas...", end="\r")
    print(f"\n✅ {enviados} trechos enfileirados. A Selene vai ler em ordem (~200/s)."
          f"\n   Acompanhe os logs: [INGEST] e a compreensão de Wernicke.")


def main():
    ap = argparse.ArgumentParser(description="Alimenta a Selene com documentos/livros/audiobooks.")
    ap.add_argument("arquivo", help="Caminho do .txt/.md/.pdf/.epub (ou áudio com --audio)")
    ap.add_argument("--audio", action="store_true", help="Tratar como audiobook (transcreve via whisper)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="3030")
    ap.add_argument("--token", default="", help="SELENE_TOKEN, se a Selene exigir")
    args = ap.parse_args()

    caminho = Path(args.arquivo)
    if not caminho.exists():
        print(f"[FAIL] Arquivo não encontrado: {caminho}")
        sys.exit(1)

    texto = transcrever_audio(caminho) if args.audio else extrair_texto(caminho)
    frases = segmentar_frases(texto)
    if not frases:
        print("[FAIL] Nenhum conteúdo textual extraído.")
        sys.exit(1)

    url = f"ws://{args.host}:{args.port}/selene"
    if args.token:
        url += f"?token={args.token}"
    print(f"📖 {caminho.name}: {len(frases)} frases → {url}")
    asyncio.run(enviar(frases, url))


if __name__ == "__main__":
    main()
