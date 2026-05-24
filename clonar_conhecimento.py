"""
clonar_conhecimento.py — V4.5 (2026-05-24)

Cliente para extrair/clonar conhecimento da Selene entre instâncias ou para
outros agentes. Usa os handlers WS `export_knowledge` e `import_knowledge`.

ARQUITETURA DA CLONAGEM:

  Selene A          export_knowledge          arquivo JSON
  ┌──────────┐      ──────────────►           ┌──────────────────┐
  │ engrams  │                                │ knowledge_v1.json│
  │ + CA3    │                                │ engrams: [...]   │
  │ + grafo  │                                │ aliases: {...}   │
  └──────────┘                                └──────────────────┘
                                                     │
                                                     │ import_knowledge
                                                     ▼
                                              Selene B (ou outro agente)
                                              ┌──────────────────┐
                                              │ engrams importados│
                                              │ origem=Restaurado │
                                              └──────────────────┘

CONTEXTO:
  • Os engrams importados são marcados com `origem=Restaurado` (não Implantado)
  • Tag original é prefixada com "imported:" para rastreabilidade
  • Auditáveis via `python implantar_conhecimento.py --auditar`
  • CA3 weights NÃO são portados — re-emergem do uso futuro

USO:
    # Exportar todo o conhecimento da Selene atual
    python clonar_conhecimento.py --exportar selene_knowledge_2026-05-24.json

    # Importar conhecimento em outra instância (ou na mesma após reset)
    python clonar_conhecimento.py --importar selene_knowledge_2026-05-24.json

    # Inspecionar conteúdo de um arquivo de export (sem importar)
    python clonar_conhecimento.py --inspecionar selene_knowledge_2026-05-24.json

PRÉ-REQUISITO: Selene rodando em ws://127.0.0.1:3030/selene.
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import websockets

URL_SELENE = "ws://127.0.0.1:3030/selene"


class ClienteClone:
    def __init__(self, url: str = URL_SELENE):
        self.url = url
        self.ws = None

    async def conectar(self) -> bool:
        try:
            self.ws = await websockets.connect(self.url, ping_interval=20)
            print(f"✅ Conectado em {self.url}")
            return True
        except Exception as e:
            print(f"❌ Falha ao conectar: {e}")
            return False

    async def desconectar(self) -> None:
        if self.ws is not None:
            await self.ws.close()

    async def _enviar_aguardar(self, payload: dict, event: str,
                                timeout: float = 30.0) -> dict | None:
        if self.ws is None:
            return None
        await self.ws.send(json.dumps(payload))
        try:
            while True:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
                resp = json.loads(raw)
                ev = resp.get("event", "")
                if ev == event or ev == event.replace("_done", "_error"):
                    return resp
        except asyncio.TimeoutError:
            print(f"   ⏱️  timeout aguardando '{event}'")
            return None

    async def exportar(self, path: str) -> dict | None:
        return await self._enviar_aguardar(
            {"action": "export_knowledge", "path": path},
            "export_done",
        )

    async def importar(self, path: str) -> dict | None:
        return await self._enviar_aguardar(
            {"action": "import_knowledge", "path": path},
            "import_done",
        )


# ============================================================================
# COMANDOS
# ============================================================================

async def cmd_exportar(path: str) -> None:
    cli = ClienteClone()
    if not await cli.conectar():
        sys.exit(1)
    try:
        # Path absoluto para o servidor escrever no diretório correto
        abs_path = str(Path(path).resolve())
        print(f"📤 Exportando para: {abs_path}")
        resp = await cli.exportar(abs_path)
        if resp is None:
            print("❌ Sem resposta")
            return
        if resp.get("event") == "export_error":
            print(f"❌ Erro: {resp.get('reason', '?')}")
            return
        print("\n✅ EXPORT CONCLUÍDO")
        print(f"   Engrams exportados: {resp.get('n_engrams', 0)}")
        print(f"   Aliases concept→palavra: {resp.get('n_aliases', 0)}")
        print(f"   Arquivo: {resp.get('path', path)}")
        # Mostra tamanho do arquivo
        try:
            tamanho = Path(abs_path).stat().st_size
            print(f"   Tamanho: {tamanho / 1024:.1f} KB")
        except OSError:
            pass
    finally:
        await cli.desconectar()


async def cmd_importar(path: str) -> None:
    abs_path = str(Path(path).resolve())
    if not Path(abs_path).exists():
        print(f"❌ Arquivo não encontrado: {abs_path}")
        sys.exit(1)
    cli = ClienteClone()
    if not await cli.conectar():
        sys.exit(1)
    try:
        print(f"📥 Importando de: {abs_path}")
        print("   ATENÇÃO: engrams importados serão marcados como Restaurado")
        resp = await cli.importar(abs_path)
        if resp is None:
            print("❌ Sem resposta")
            return
        if resp.get("event") == "import_error":
            print(f"❌ Erro: {resp.get('reason', '?')}")
            return
        print("\n✅ IMPORT CONCLUÍDO")
        print(f"   Engrams importados: {resp.get('n_engrams_imported', 0)}")
        print("\n💡 Auditoria recomendada:")
        print("   python implantar_conhecimento.py --auditar")
    finally:
        await cli.desconectar()


def cmd_inspecionar(path: str) -> None:
    """Lê e analisa um arquivo de export sem conectar à Selene."""
    p = Path(path)
    if not p.exists():
        print(f"❌ Arquivo não encontrado: {path}")
        sys.exit(1)

    print(f"🔍 Inspecionando: {p.resolve()}")
    print(f"   Tamanho: {p.stat().st_size / 1024:.1f} KB\n")

    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON inválido: {e}")
        sys.exit(1)

    root = data.get("selene_knowledge_v1")
    if root is None:
        print("❌ Formato desconhecido (esperado: 'selene_knowledge_v1')")
        sys.exit(1)

    n_engrams = root.get("n_engrams", 0)
    n_ca3 = root.get("n_ca3_synapses", 0)
    aliases = root.get("concept_aliases", {})
    engrams = root.get("engrams", [])

    print(f"📊 RESUMO")
    print(f"   Engrams:         {n_engrams}")
    print(f"   Sinapses CA3:    {n_ca3}")
    print(f"   Aliases (cid→w): {len(aliases)}")
    print(f"   DG k_target:     {root.get('dg_k_target', '?')}")

    if not engrams:
        return

    # Distribuição por origem
    origens = Counter(e.get("origem", "?") for e in engrams)
    print(f"\n🏷️  POR ORIGEM")
    for orig, n in origens.most_common():
        print(f"   {orig:.<20} {n:>4}")

    # Distribuição por tag
    tags = Counter(e.get("tag", "") for e in engrams)
    print(f"\n📁 POR TAG (top 15)")
    for tag, n in tags.most_common(15):
        nome = tag if tag else "(sem tag)"
        print(f"   {nome:.<30} {n:>4}")

    # Estatísticas de células
    n_cells_list = [len(e.get("cell_ensemble", [])) for e in engrams]
    if n_cells_list:
        avg = sum(n_cells_list) / len(n_cells_list)
        print(f"\n📐 CÉLULAS POR ENGRAM")
        print(f"   Mín:  {min(n_cells_list)}")
        print(f"   Méd:  {avg:.1f}")
        print(f"   Máx:  {max(n_cells_list)}")
        print(f"   Total cells (com duplicatas): {sum(n_cells_list)}")

    # Valências
    valencias = [e.get("emocao", 0.0) for e in engrams]
    if valencias:
        avg_v = sum(valencias) / len(valencias)
        pos = sum(1 for v in valencias if v > 0)
        neg = sum(1 for v in valencias if v < 0)
        print(f"\n💚💔 VALÊNCIAS")
        print(f"   Positivas: {pos}  | Negativas: {neg}  | Neutras: {len(valencias) - pos - neg}")
        print(f"   Média: {avg_v:+.3f}")

    # Aliases mais úteis (palavras curtas e comuns)
    if aliases:
        print(f"\n🔤 PALAVRAS NO ALIAS (amostra 10)")
        palavras = list(aliases.values())[:10]
        print(f"   {', '.join(palavras)}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clona conhecimento da Selene (export/import/inspecionar).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--exportar", metavar="ARQUIVO",
                       help="Exporta knowledge atual para arquivo JSON")
    grupo.add_argument("--importar", metavar="ARQUIVO",
                       help="Importa knowledge de arquivo JSON (origem=Restaurado)")
    grupo.add_argument("--inspecionar", metavar="ARQUIVO",
                       help="Analisa arquivo de export sem conectar (read-only)")

    args = parser.parse_args()

    if args.exportar:
        asyncio.run(cmd_exportar(args.exportar))
    elif args.importar:
        asyncio.run(cmd_importar(args.importar))
    elif args.inspecionar:
        cmd_inspecionar(args.inspecionar)


if __name__ == "__main__":
    main()
