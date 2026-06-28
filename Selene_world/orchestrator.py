"""
orchestrator.py — O MUNDO 2D: mapa, quadro negro, população, clima, eventos.

Cada ciclo:
  1. regenera recursos conforme o PULSO DO PC (ocioso=abundância, saturado=fome)
  2. clima: PC saturado = "inverno" → bots gastam mais energia
  3. todos os bots agem (percebem/movem/coletam/comem/falam)
  4. ENCONTROS: bots na mesma célula alinham a linguagem (transmissão cultural)
  5. vida/morte (fome ou velhice ~100 anos) → reprodução → eventos (+ extinção por RAM real)
Mede a CONVERGÊNCIA DA LINGUAGEM: os bots estão criando um código comum?
"""
import math
import random
import statistics
from collections import defaultdict

import events
from blackboard import QuadroNegro
from bot_core import SeleneBot, genoma_aleatorio
from genealogia import Genealogia
from selene_tasks import TAREFAS
from world_map import RECURSOS, Mapa

# 2 TRIBOS, cada uma espelhando uma operação REAL da Selene na pré-escrita. Começam
# isoladas → evoluem o código por caminhos diferentes (dialetos de código). 4 únicos cada.
TRIBOS = [
    {"nome": "Ignis", "tarefa": "neuronio_fire"},   # disparo de neurônios
    {"nome": "Umbra", "tarefa": "decay_neuroq"},    # decaimento de neuroquímica
]
POR_TRIBO = 6

# ── escala de tempo: 1 PASSO = 1 DIA; envelhece 1 ano a cada 365 dias ──────────
DIAS_POR_ANO = 365
GASTO_BASE = 0.3          # energia/dia em repouso (clima ameno)
GASTO_CLIMA = 0.5         # extra/dia quando o PC satura (inverno)
REGEN_FATOR = 8.0         # comida volta rápido o bastante p/ o consumo DIÁRIO dos bots
PROB_FOGO_DIA = 0.008     # ~3 raios/ano
DIAS_SONO = 30            # 1 bot refatora a pré-escrita a cada 30 dias
DURACAO_DIA = 12          # passos por ciclo dia↔noite (oscila o nível de luz)
GASTO_NOITE = 0.12        # frio noturno suave (sobe quando houver cavernas p/ escape)


class Mundo:
    def __init__(self, w=48, h=30, n_inicial=14, capacidade=100, seed=0, log=None):
        self.log = log or (lambda *a: None)
        self.rng = random.Random(seed)
        self.mapa = Mapa(w, h, seed=seed)
        self.quadro = QuadroNegro()
        self.capacidade = capacidade
        self.gen = Genealogia()              # cataloga as otimizações (pra portar à Selene)
        self.pop = self._fundar_tribos(w, h)
        self._sono_idx = 0                   # round-robin: 1 bot refatora por mês
        self.ano = 0
        self.luz = 1.0                       # nível de luz (1=dia, 0=noite)
        self.metricas = []

    def _fundar_tribos(self, w, h):
        """Cada tribo nasce com 4 indivíduos ÚNICOS (genes distintos) mas a MESMA
        pré-escrita-base = espelho da operação da Selene daquela tribo."""
        pop = []
        for ti, tribo in enumerate(TRIBOS):
            base = TAREFAS[tribo["tarefa"]][0]      # genoma_inicial (código-base)
            cx = w // 4 if ti == 0 else 3 * w // 4  # cada tribo nasce AGRUPADA numa
            cy = h // 2                             # região (não isolada no mapa grande)
            self.mapa.cavernas[ti] = (cx, cy)       # caverna da tribo (abrigo noturno)
            for _ in range(POR_TRIBO):
                x = max(0, min(w - 1, cx + self.rng.randint(-3, 3)))
                y = max(0, min(h - 1, cy + self.rng.randint(-3, 3)))
                bot = SeleneBot(x, y, genoma_aleatorio(self.rng), log=self.log,
                                pre_escrita=base, tribo=ti)
                self.gen.registrar_bot(bot.id, None, None, 0, None, bot.pre_escrita)
                pop.append(bot)
        return pop

    def _fitness(self, bot):
        # quem evoluiu o próprio código (otimizações) é mais apto — Lei da Eficiência
        return (bot.energia + bot.acertos_fala * 5 + bot.idade * 0.5
                + bot.otimizacoes * 4)

    def passo(self, dia):
        """1 DIA: regenera um pouco, bots forrageiam/movem/comem, encontros, fome.
        A cada 365 dias dispara _ano() (envelhece, reprodução, eventos)."""
        ano_virou = dia > 0 and dia % DIAS_POR_ANO == 0
        self.luz = 0.5 + 0.5 * math.sin(dia * 2 * math.pi / DURACAO_DIA)  # dia↔noite
        eh_noite = self.luz < 0.35
        pulso = self.mapa.regenerar(escala=REGEN_FATOR / DIAS_POR_ANO,
                                    atualizar_pulso=(dia % 7 == 0))
        self.mapa.decair_fogo()                       # fogo dura poucos dias
        if self.rng.random() < PROB_FOGO_DIA:
            events.fogo_natural(self.mapa, self.log, prob=1.0)
        gasto = GASTO_BASE + pulso["pressao"] * GASTO_CLIMA  # clima do PC (sem noite)

        for bot in self.pop:                          # bots vivem o dia
            bot.agir(self.mapa, self.quadro, gasto, eh_noite)  # noite→busca abrigo
            bot.usar_materiais(self.mapa, gasto)       # madeira→fogo, ferro→ferramenta
        self._encontros()
        self.quadro.envelhecer()

        vivos = [b for b in self.pop if b.energia > 0]  # morte por FOME (todo dia)
        mortos = len(self.pop) - len(vivos)
        self.pop = vivos

        if self.pop and dia % DIAS_SONO == 0:         # SONO mensal: 1 bot refatora
            self._sono_idx = (self._sono_idx + 1) % len(self.pop)
            dorminhoco = self.pop[self._sono_idx]
            _, gerar, esp = TAREFAS[TRIBOS[dorminhoco.tribo]["tarefa"]]
            dorminhoco.sonhar(gerar(dia), esp, self.gen)

        nascidos = 0
        if ano_virou:                                 # vira o ANO
            self.ano = dia // DIAS_POR_ANO
            nascidos, mortos_idade = self._ano(pulso)
            mortos += mortos_idade

        m = self._coletar(self.ano, mortos, nascidos, pulso)
        self.metricas.append(m)
        return m

    def _ano(self, pulso):
        """1 ANO (a cada 365 dias): envelhece, morte por idade, reprodução, eventos."""
        for b in self.pop:
            b.envelhecer()
        vivos = [b for b in self.pop if b.vivo()]     # morte por VELHICE
        mortos_idade = len(self.pop) - len(vivos)
        self.pop = vivos

        nascidos = 0
        for bot in list(self.pop):
            if bot.pode_reproduzir() and len(self.pop) < self.capacidade * 2:
                self.pop.append(bot.reproduzir(self.mapa))
                nascidos += 1

        self.pop = events.evento_aleatorio(self.pop, self._fitness, self.log)
        self.pop = events.por_superpopulacao(
            self.pop, self._fitness, self.capacidade, self.log)
        self.pop = events.por_pressao_ram(            # extinção atada à RAM REAL do PC
            self.pop, self._fitness, pulso["ram"], self.log)
        return nascidos, mortos_idade

    def _encontros(self):
        """Naming game: bots na MESMA célula adotam o símbolo do 'líder' (quem mais
        acertou comunicando) para 1 recurso. É a transmissão cultural que faz a
        linguagem CONVERGIR — sem ela, só haveria deriva genética (lentíssima)."""
        por_celula = defaultdict(list)
        for b in self.pop:
            por_celula[(b.x, b.y)].append(b)
        for grupo in por_celula.values():
            if len(grupo) < 2:
                continue
            lider = max(grupo, key=lambda b: (b.acertos_fala, b.energia))
            for b in grupo:                       # adota o léxico do líder (cultura)
                if b is not lider:
                    for r in RECURSOS:
                        b.genoma["lexico"][r] = lider.genoma["lexico"][r]
                    # a LINGUAGEM passa a MEMÓRIA adiante: quem sabe do fogo ensina,
                    # acelerando a difusão do conhecimento além da herança genética
                    if lider.crenca_fogo > b.crenca_fogo:
                        b.crenca_fogo += 0.3 * (lider.crenca_fogo - b.crenca_fogo)

    def convergencia_linguagem(self):
        """0..1 — fração média da população que concorda no símbolo de cada recurso.
        1.0 = todos chamam cada recurso pelo MESMO símbolo (linguagem comum)."""
        if not self.pop:
            return 0.0
        acordos = []
        for r in RECURSOS:
            cont = defaultdict(int)
            for b in self.pop:
                cont[b.genoma["lexico"][r]] += 1
            acordos.append(max(cont.values()) / len(self.pop))
        return sum(acordos) / len(acordos)

    def _dominio_fogo(self):
        """Fração da população que JÁ DECIFROU o fogo (crença > 0.3). Sobe sozinho
        quando a tecnologia 'emerge' — é o avanço tecnológico, medido."""
        if not self.pop:
            return 0.0
        return sum(1 for b in self.pop if b.crenca_fogo > 0.3) / len(self.pop)

    def tribo_stats(self):
        """Por tribo: nº de bots, otimizações acumuladas e vida média (mérito)."""
        out = []
        for ti, tribo in enumerate(TRIBOS):
            membros = [b for b in self.pop if b.tribo == ti]
            out.append({
                "nome": tribo["nome"],
                "pop": len(membros),
                "otim": sum(b.otimizacoes for b in membros),
                "comida": int(sum(b.coletado["comida"] for b in membros)),
                "madeira": int(sum(b.coletado["madeira"] for b in membros)),
                "ferro": int(sum(b.coletado["ferro"] for b in membros)),
                "ferramentas": sum(1 for b in membros if b.tem_ferramenta),
                "vida": statistics.mean([b.genoma["vida_max"] for b in membros])
                if membros else 0,
            })
        return out

    def _coletar(self, t, mortos, nascidos, pulso):
        idades = [b.idade for b in self.pop]
        return {
            "t": t,
            "pop": len(self.pop),
            "mortos": mortos,
            "nascidos": nascidos,
            "idade_media": statistics.mean(idades) if idades else 0,
            "ger_max": max((b.geracao for b in self.pop), default=0),
            "lingua": self.convergencia_linguagem(),
            "fogo": self._dominio_fogo(),
            "otim": sum(b.otimizacoes for b in self.pop),
            "ferramentas": sum(1 for b in self.pop if b.tem_ferramenta),
            "fogos": len(self.mapa.fogos),
            "luz": round(self.luz, 2),
            "vida_media": statistics.mean(
                [b.genoma["vida_max"] for b in self.pop]) if self.pop else 0,
            "cpu": pulso["cpu"],
            "ram": pulso["ram"],
            "abundancia": pulso["abundancia"],
            "fonte": pulso["fonte"],
        }
