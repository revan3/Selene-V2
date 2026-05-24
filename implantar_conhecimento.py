"""
implantar_conhecimento.py — V4.4 (2026-05-24)

Cliente Python para implantar conhecimento artificial na Selene via WebSocket,
usando a técnica de engram tagging (estilo Tonegawa 2013, false memory in mice).

Cada "memória" é um grupo de palavras semanticamente relacionadas + uma valência
emocional. O servidor:
  1. Resolve cada palavra para concept_id (FNV-1a)
  2. Aloca população de neurônios (20 por conceito) se não existir
  3. Cria engram marcado como Implantado (auditável via list_implants)
  4. Boost STDP entre todos os pares de conceitos (cria sinapses)
  5. Persiste no HippocampalIndex.engrams

USO:
    # Implantar TUDO (todos os 9 domínios):
    python implantar_conhecimento.py --tudo

    # Implantar domínios específicos:
    python implantar_conhecimento.py --dominios matematica fisica programacao

    # Listar domínios disponíveis:
    python implantar_conhecimento.py --listar

    # Listar implantes existentes na Selene:
    python implantar_conhecimento.py --auditar

    # Purgar implantes (CUIDADO):
    python implantar_conhecimento.py --purgar matematica
    python implantar_conhecimento.py --purgar TODOS  # tudo

PRÉ-REQUISITO: Selene rodando em ws://127.0.0.1:3030/selene.
"""

import argparse
import asyncio
import json
import sys
from typing import Any

import websockets

URL_SELENE = "ws://127.0.0.1:3030/selene"

# ============================================================================
# CORPORA DE CONHECIMENTO
# ============================================================================
#
# Cada domínio define:
#   - "valencia": [-1.0, 1.0] — emoção associada a TODAS as memórias do domínio
#   - "as_if_repeated": quantas reativações simuladas (10+ parece consolidado)
#   - "memorias": lista de listas de palavras (cada lista interna = uma memória)
#
# Cada "memória" vira um engram. Palavras dentro de uma memória ficam STDP-conectadas.
# ============================================================================

CONHECIMENTO: dict[str, dict[str, Any]] = {

    # ------------------------------------------------------------------------
    "matematica": {
        "descricao": "Aritmética, álgebra básica, geometria, lógica numérica",
        "valencia": 0.30,
        "as_if_repeated": 12,
        "memorias": [
            # Números cardinais
            ["zero", "um", "dois", "três", "quatro", "cinco"],
            ["seis", "sete", "oito", "nove", "dez"],
            ["cem", "mil", "milhão", "bilhão"],
            # Operações básicas
            ["soma", "mais", "adicionar", "total", "junto"],
            ["subtração", "menos", "tirar", "diferença", "resto"],
            ["multiplicação", "vezes", "produto", "duplicar"],
            ["divisão", "dividir", "metade", "quociente", "partes"],
            # Comparação
            ["igual", "diferente", "maior", "menor"],
            ["par", "ímpar", "primo", "fração"],
            # Geometria
            ["quadrado", "círculo", "triângulo", "retângulo"],
            ["lado", "ângulo", "área", "perímetro"],
            ["ponto", "linha", "reta", "curva"],
            # Álgebra básica
            ["variável", "incógnita", "equação", "resolver", "valor"],
            # Lógica
            ["verdadeiro", "falso", "se", "então", "porque"],
        ],
    },

    # ------------------------------------------------------------------------
    "semantica": {
        "descricao": "Relações conceituais — sinônimos, antônimos, hierarquia",
        "valencia": 0.10,
        "as_if_repeated": 8,
        "memorias": [
            # Sinônimos
            ["feliz", "alegre", "contente", "satisfeito"],
            ["triste", "infeliz", "abatido", "melancólico"],
            ["grande", "enorme", "gigante", "vasto"],
            ["pequeno", "minúsculo", "diminuto", "pouco"],
            ["rápido", "veloz", "ligeiro", "ágil"],
            ["lento", "devagar", "vagaroso", "calmo"],
            # Antônimos chave
            ["claro", "escuro", "luz", "sombra"],
            ["quente", "frio", "morno", "gelado"],
            ["forte", "fraco", "potente", "débil"],
            ["novo", "velho", "antigo", "recente"],
            # Hierarquias (hipônimos/hiperônimos)
            ["animal", "cachorro", "gato", "pássaro", "peixe"],
            ["fruta", "maçã", "banana", "uva", "laranja"],
            ["cor", "vermelho", "azul", "verde", "amarelo"],
            ["móvel", "cadeira", "mesa", "sofá", "cama"],
            # Tempo
            ["passado", "presente", "futuro", "agora", "depois"],
            ["dia", "noite", "tarde", "manhã"],
        ],
    },

    # ------------------------------------------------------------------------
    "oratoria": {
        "descricao": "Conectivos, estruturas retóricas, marcadores discursivos",
        "valencia": 0.20,
        "as_if_repeated": 10,
        "memorias": [
            # Conectivos causais
            ["porque", "pois", "já", "que", "visto"],
            ["portanto", "logo", "assim", "consequência"],
            # Conectivos adversativos
            ["mas", "porém", "contudo", "entretanto", "embora"],
            # Conectivos temporais
            ["quando", "enquanto", "antes", "depois", "durante"],
            # Conectivos conclusivos
            ["então", "logo", "assim", "portanto", "concluindo"],
            # Marcadores de opinião
            ["acredito", "penso", "acho", "imagino", "suponho"],
            # Marcadores de certeza
            ["certamente", "definitivamente", "claro", "sem", "dúvida"],
            # Marcadores de incerteza
            ["talvez", "possivelmente", "provavelmente", "parece"],
            # Estrutura argumentativa
            ["primeiro", "segundo", "finalmente", "exemplo", "ilustrar"],
            # Saudações e cortesia
            ["olá", "bom", "dia", "tarde", "noite", "obrigado"],
            ["por", "favor", "desculpa", "com", "licença"],
        ],
    },

    # ------------------------------------------------------------------------
    "fisica": {
        "descricao": "Forças, energia, movimento, ondas, gravidade",
        "valencia": 0.25,
        "as_if_repeated": 10,
        "memorias": [
            # Mecânica
            ["força", "massa", "aceleração", "newton"],
            ["velocidade", "tempo", "distância", "movimento"],
            ["gravidade", "peso", "queda", "atração", "terra"],
            ["energia", "trabalho", "potência", "joule"],
            ["cinética", "potencial", "conservação"],
            # Ondas
            ["onda", "frequência", "amplitude", "comprimento"],
            ["som", "vibração", "eco", "ressonância"],
            ["luz", "fóton", "espectro", "cor", "reflexão"],
            # Eletromagnetismo
            ["carga", "elétron", "corrente", "voltagem"],
            ["campo", "magnético", "elétrico", "indução"],
            # Termodinâmica
            ["calor", "temperatura", "entropia", "equilíbrio"],
            ["sólido", "líquido", "gasoso", "plasma", "fase"],
            # Quântica básica
            ["átomo", "núcleo", "elétron", "próton", "nêutron"],
            ["incerteza", "quantum", "superposição"],
        ],
    },

    # ------------------------------------------------------------------------
    "quimica": {
        "descricao": "Elementos, ligações, reações, estados",
        "valencia": 0.20,
        "as_if_repeated": 10,
        "memorias": [
            # Elementos fundamentais
            ["hidrogênio", "oxigênio", "carbono", "nitrogênio"],
            ["ferro", "ouro", "prata", "cobre", "alumínio"],
            ["sódio", "potássio", "cálcio", "magnésio"],
            # Moléculas comuns
            ["água", "h2o", "hidrogênio", "oxigênio"],
            ["sal", "cloreto", "sódio", "cristal"],
            ["açúcar", "glicose", "carbono", "doce"],
            ["dióxido", "carbono", "co2", "respiração"],
            # Ligações
            ["ligação", "covalente", "iônica", "metálica"],
            ["ácido", "base", "ph", "neutro", "alcalino"],
            # Reações
            ["reação", "reagente", "produto", "catalisador"],
            ["oxidação", "redução", "combustão", "fogo"],
            # Estados
            ["solução", "soluto", "solvente", "diluir", "misturar"],
            ["cristal", "estrutura", "molécula", "polímero"],
        ],
    },

    # ------------------------------------------------------------------------
    "biologia": {
        "descricao": "Células, organismos, evolução, sistemas",
        "valencia": 0.30,
        "as_if_repeated": 10,
        "memorias": [
            # Célula
            ["célula", "núcleo", "membrana", "citoplasma"],
            ["mitocôndria", "energia", "atp", "respiração"],
            ["dna", "rna", "gene", "código", "proteína"],
            ["ribossomo", "proteína", "síntese", "tradução"],
            # Sistemas
            ["coração", "sangue", "artéria", "circulação"],
            ["pulmão", "respiração", "oxigênio", "ar"],
            ["cérebro", "neurônio", "sinapse", "memória"],
            ["estômago", "digestão", "intestino", "fígado"],
            ["músculo", "osso", "esqueleto", "articulação"],
            # Reprodução e desenvolvimento
            ["célula", "divisão", "mitose", "meiose"],
            ["embrião", "feto", "desenvolvimento", "gravidez"],
            # Evolução
            ["evolução", "seleção", "natural", "darwin", "adaptação"],
            ["espécie", "mutação", "hereditariedade", "ancestral"],
            # Ecologia
            ["ecossistema", "biodiversidade", "habitat", "cadeia"],
            ["planta", "fotossíntese", "clorofila", "sol"],
        ],
    },

    # ------------------------------------------------------------------------
    "programacao": {
        "descricao": "Variáveis, funções, controle de fluxo, estruturas",
        "valencia": 0.40,
        "as_if_repeated": 12,
        "memorias": [
            # Conceitos básicos
            ["variável", "valor", "tipo", "memória"],
            ["função", "parâmetro", "retorno", "chamada"],
            ["string", "número", "booleano", "lista"],
            # Controle de fluxo
            ["if", "else", "condição", "branch"],
            ["loop", "for", "while", "iteração", "repetir"],
            ["break", "continue", "return", "early"],
            # Estruturas de dados
            ["array", "lista", "elemento", "índice"],
            ["dicionário", "chave", "valor", "hash", "map"],
            ["conjunto", "set", "único", "membership"],
            ["pilha", "fila", "stack", "queue"],
            # Paradigmas
            ["objeto", "classe", "método", "instância"],
            ["herança", "polimorfismo", "encapsulamento"],
            ["função", "pura", "imutável", "side-effect"],
            ["recursão", "base", "indução", "fibonacci"],
            # Práticas
            ["bug", "erro", "debug", "log", "stack-trace"],
            ["teste", "assert", "tdd", "cobertura"],
            ["commit", "branch", "merge", "git", "pull-request"],
            # Linguagens
            ["python", "rust", "javascript", "typescript", "go"],
            ["compilar", "interpretar", "lint", "format"],
        ],
    },

    # ------------------------------------------------------------------------
    "informatica": {
        "descricao": "Hardware, software, redes, conceitos de uso",
        "valencia": 0.25,
        "as_if_repeated": 10,
        "memorias": [
            # Hardware
            ["computador", "cpu", "ram", "disco", "ssd"],
            ["teclado", "mouse", "monitor", "tela", "pixel"],
            ["placa", "vídeo", "gpu", "processador"],
            # Software
            ["sistema", "operacional", "windows", "linux", "macos"],
            ["aplicativo", "programa", "instalar", "executar"],
            ["arquivo", "pasta", "diretório", "caminho"],
            # Internet
            ["internet", "wifi", "rede", "conexão"],
            ["site", "url", "navegador", "browser"],
            ["email", "mensagem", "enviar", "receber"],
            ["download", "upload", "transferência"],
            # Conceitos
            ["nuvem", "cloud", "servidor", "remoto", "local"],
            ["backup", "cópia", "segurança", "restore"],
            ["senha", "login", "autenticação", "criptografia"],
            ["vírus", "malware", "antivírus", "firewall"],
        ],
    },

    # ------------------------------------------------------------------------
    "jogos": {
        "descricao": "Objetivos, regras, estratégia, mecânicas comuns",
        "valencia": 0.50,
        "as_if_repeated": 8,
        "memorias": [
            # Conceitos universais
            ["jogo", "regra", "objetivo", "vencer", "perder"],
            ["jogador", "personagem", "avatar", "controle"],
            ["pontos", "score", "ranking", "level"],
            # Mecânicas
            ["movimento", "ação", "ataque", "defesa"],
            ["vida", "energia", "mana", "stamina"],
            ["inventário", "item", "arma", "armadura", "poção"],
            ["loot", "drop", "recompensa", "tesouro"],
            # Estratégia
            ["estratégia", "tática", "plano", "decisão"],
            ["risco", "recompensa", "trade-off", "escolha"],
            ["timing", "reação", "antecipação"],
            # Tipos
            ["puzzle", "quebra-cabeça", "lógica", "solução"],
            ["aventura", "exploração", "mundo", "mapa"],
            ["luta", "combate", "boss", "chefão"],
            ["corrida", "velocidade", "pista", "primeiro"],
            ["estratégia", "construir", "exército", "recursos"],
            # Social
            ["multiplayer", "co-op", "versus", "equipe"],
            ["chat", "comunicação", "voz", "texto"],
        ],
    },
}


# ============================================================================
# CLIENTE WEBSOCKET
# ============================================================================

class ClienteImplante:
    """Cliente WebSocket assíncrono para implantar conhecimento na Selene."""

    def __init__(self, url: str = URL_SELENE):
        self.url = url
        self.ws: Any = None

    async def conectar(self) -> bool:
        try:
            self.ws = await websockets.connect(self.url, ping_interval=20)
            print(f"✅ Conectado em {self.url}")
            return True
        except Exception as e:
            print(f"❌ Falha ao conectar: {e}")
            print("   Verifique se a Selene está rodando: target/release-lowmem/selene_brain.exe")
            return False

    async def desconectar(self) -> None:
        if self.ws is not None:
            await self.ws.close()
            self.ws = None

    async def _enviar_aguardar(self, payload: dict, event_esperado: str,
                                timeout: float = 10.0) -> dict | None:
        """Envia payload e espera resposta com event matching."""
        if self.ws is None:
            return None
        await self.ws.send(json.dumps(payload))
        try:
            while True:
                resp_raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
                resp = json.loads(resp_raw)
                if resp.get("event") == event_esperado:
                    return resp
                # Ignora outros eventos (heartbeat, etc.)
        except asyncio.TimeoutError:
            print(f"   ⏱️  timeout aguardando '{event_esperado}'")
            return None

    async def implantar(self, words: list[str], valence: float, as_if_repeated: int,
                        tag: str, boost_stdp: bool = True) -> dict | None:
        payload = {
            "action": "implant_memory",
            "words": words,
            "valence": valence,
            "as_if_repeated": as_if_repeated,
            "tag": tag,
            "boost_stdp": boost_stdp,
        }
        return await self._enviar_aguardar(payload, "implant_done")

    async def listar_implantes(self) -> dict | None:
        return await self._enviar_aguardar({"action": "list_implants"}, "implants_list")

    async def purgar_implantes(self, tag: str | None) -> dict | None:
        payload: dict = {"action": "purge_implants"}
        if tag is not None:
            payload["tag"] = tag
        return await self._enviar_aguardar(payload, "implants_purged")


# ============================================================================
# COMANDOS CLI
# ============================================================================

async def cmd_implantar_dominios(dominios: list[str]) -> None:
    """Implanta todos os domínios especificados."""
    cli = ClienteImplante()
    if not await cli.conectar():
        sys.exit(1)
    try:
        total_engrams = 0
        total_cells = 0
        total_synapses = 0
        for dominio in dominios:
            if dominio not in CONHECIMENTO:
                print(f"⚠️  Domínio desconhecido: '{dominio}'. Use --listar.")
                continue
            cfg = CONHECIMENTO[dominio]
            print(f"\n📚 Implantando domínio: {dominio}")
            print(f"   Descrição: {cfg['descricao']}")
            print(f"   Memórias: {len(cfg['memorias'])} | Valência: {cfg['valencia']:+.2f}")
            for i, memoria in enumerate(cfg["memorias"], 1):
                resp = await cli.implantar(
                    words=memoria,
                    valence=cfg["valencia"],
                    as_if_repeated=cfg["as_if_repeated"],
                    tag=dominio,
                )
                if resp is None:
                    print(f"   {i}/{len(cfg['memorias'])} ❌ {memoria}")
                    continue
                total_engrams += 1
                total_cells += resp.get("cells_used", 0)
                total_synapses += resp.get("synapses_boosted", 0)
                novos = resp.get("concepts_created", [])
                novos_str = f" (+{len(novos)} novos)" if novos else ""
                print(f"   {i}/{len(cfg['memorias'])} ✅ engram #{resp['engram_id']} | "
                      f"{len(memoria)} palavras → {resp.get('cells_used', 0)} cells{novos_str}")
        print("\n" + "="*60)
        print(f"  TOTAL: {total_engrams} engrams | {total_cells} cells | {total_synapses} sinapses")
        print("="*60)
    finally:
        await cli.desconectar()


async def cmd_listar_implantes() -> None:
    """Lista implantes existentes na Selene (auditoria)."""
    cli = ClienteImplante()
    if not await cli.conectar():
        sys.exit(1)
    try:
        resp = await cli.listar_implantes()
        if resp is None:
            print("❌ Sem resposta")
            return
        print(f"\n📊 Engrams na Selene:")
        print(f"   Orgânicos:    {resp.get('organicos', 0)}")
        print(f"   Implantados:  {resp.get('implantados', 0)}")
        print(f"   Restaurados:  {resp.get('restaurados', 0)}")
        implantes = resp.get("implants", [])
        if not implantes:
            print("\n   (nenhum implante registrado)")
            return
        # Agrupa por tag
        por_tag: dict[str, list] = {}
        for imp in implantes:
            por_tag.setdefault(imp["tag"], []).append(imp)
        print(f"\n📁 Por tag:")
        for tag in sorted(por_tag.keys()):
            engrams = por_tag[tag]
            total_cells = sum(e["cells"] for e in engrams)
            print(f"   {tag:.<30} {len(engrams):>3} engrams | {total_cells:>4} cells")
    finally:
        await cli.desconectar()


async def cmd_purgar(tag: str) -> None:
    """Remove implantes (CUIDADO — irreversível)."""
    cli = ClienteImplante()
    if not await cli.conectar():
        sys.exit(1)
    try:
        tag_filter = None if tag.upper() == "TODOS" else tag
        print(f"⚠️  Purgando implantes: tag={tag_filter or '(TODOS)'}")
        resp = await cli.purgar_implantes(tag_filter)
        if resp:
            print(f"✅ {resp['removed']} engrams removidos")
        else:
            print("❌ Sem resposta")
    finally:
        await cli.desconectar()


def cmd_listar_dominios() -> None:
    """Lista domínios disponíveis."""
    print("\n📚 Domínios de conhecimento disponíveis:\n")
    total_memorias = 0
    for nome, cfg in CONHECIMENTO.items():
        n = len(cfg["memorias"])
        total_memorias += n
        print(f"  {nome:.<20} {n:>2} memórias | valência {cfg['valencia']:+.2f} | {cfg['descricao']}")
    print(f"\nTotal: {len(CONHECIMENTO)} domínios, {total_memorias} memórias\n")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Implanta conhecimento artificial na Selene via WebSocket.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--tudo", action="store_true",
                       help="Implanta TODOS os domínios")
    grupo.add_argument("--dominios", nargs="+", metavar="DOMINIO",
                       help="Lista de domínios para implantar")
    grupo.add_argument("--listar", action="store_true",
                       help="Lista domínios disponíveis (não conecta)")
    grupo.add_argument("--auditar", action="store_true",
                       help="Lista implantes existentes na Selene")
    grupo.add_argument("--purgar", metavar="TAG",
                       help="Remove implantes (use TODOS para tudo)")

    args = parser.parse_args()

    if args.listar:
        cmd_listar_dominios()
    elif args.tudo:
        asyncio.run(cmd_implantar_dominios(list(CONHECIMENTO.keys())))
    elif args.dominios:
        asyncio.run(cmd_implantar_dominios(args.dominios))
    elif args.auditar:
        asyncio.run(cmd_listar_implantes())
    elif args.purgar:
        asyncio.run(cmd_purgar(args.purgar))


if __name__ == "__main__":
    main()
