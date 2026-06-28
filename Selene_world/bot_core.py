"""
bot_core.py — O Selene-Bot como AGENTE de survival num mapa 2D.

Genoma = genes de comportamento + LÉXICO (um símbolo para cada recurso). O bot:
  • percebe recursos por perto (com ERRO DE SONDA — às vezes lê errado)
  • move-se, coleta, e COME pra não morrer de fome
  • FALA no quadro negro (anuncia recurso usando o SEU símbolo)
  • ouve o quadro e DECODIFICA com o próprio léxico

Ninguém programa o significado dos símbolos. Cada bot nasce com símbolos quase
aleatórios (Babel). A seleção + os encontros (alinhamento no orchestrator) fazem
a população convergir num código comum → a linguagem EMERGE.
"""
import copy
import random

import mutations
from safety import executar_seguro, validar_seguranca
from world_map import RECURSOS

ENERGIA_INICIAL = 140.0
# Escala de tempo: 1 ciclo = 1 ANO de vida. Todos começam com 80 anos (prazo padrão)
# e ganham anos por MÉRITO (evoluir código/tecnologia). Ninguém passa de 120.
MAX_IDADE = 120
VIDA_BASE = 80
FRIO_NOITE = 0.25      # energia/dia gasta à noite se EXPOSTO (caverna ou fogo poupa)


def genoma_aleatorio(rng):
    return {
        "papel": rng.choice(RECURSOS),                       # recurso que prioriza
        "vida_max": VIDA_BASE,                               # 80 anos padrão (+anos por mérito)
        "visao": rng.randint(3, 6),                          # raio de percepção (mapa grande)
        "confia_quadro": round(rng.uniform(0.2, 0.9), 2),    # quanto crê em boatos
        "tagarela": round(rng.uniform(0.3, 0.9), 2),         # propensão a falar
        "curiosidade_fogo": round(rng.uniform(-0.4, 0.6), 2),  # atração/medo inato
        "lexico": {r: rng.randint(0, 30) for r in RECURSOS},  # símbolo por recurso
    }


class SeleneBot:
    _seq = 0

    def __init__(self, x, y, genoma, geracao=0, pai=None, log=None,
                 pre_escrita="", tribo=0):
        SeleneBot._seq += 1
        self.id = SeleneBot._seq
        self.x, self.y = x, y
        self.genoma = genoma
        self.geracao = geracao
        self.pai = pai
        self.idade = 0
        self.energia = ENERGIA_INICIAL
        self.coletado = {r: 0 for r in RECURSOS}
        self.acertos_fala = 0          # vezes que um boato lido levou a recurso real
        self.crenca_fogo = 0.0         # MEMÓRIA aprendida: fogo é bom (+) ou ruim (-)?
        self.aqueceu = False           # se aproveitou o fogo neste ciclo (métrica)
        self.ultima_repro = -999       # idade do último filho (p/ espaçar a reprodução)
        self.tem_ferramenta = False    # forjou ferramenta (ferro+entendimento) → coleta+
        # PRÉ-ESCRITA: o "código genético" do bot — espelho de uma operação da Selene
        # que ele refatora no sono pra ficar mais eficiente (Lei da Reescrita).
        self.tribo = tribo
        self.pre_escrita = pre_escrita
        self.pre_lat = None            # latência atual da pré-escrita (eficiência)
        self.otimizacoes = 0           # quantas vezes melhorou o próprio código
        self.fogo_premiado = False     # já ganhou o +1 ano por decifrar o fogo?
        self.log = log or (lambda *a: None)
        self.rng = random.Random(self.id * 7919 + 1)
        self._alvo_por_boato = None    # (pos, recurso) — p/ creditar a comunicação
        self._alvo_pos = None          # destino do ano — p/ caminhar entre os anos (viz)

    # ── percepção LOCAL com erro de sonda ────────────────────────────────────
    def _perceber(self, mapa):
        out, v = [], self.genoma["visao"]
        for dy in range(-v, v + 1):
            for dx in range(-v, v + 1):
                pos = (self.x + dx, self.y + dy)
                if not mapa.dentro(*pos):
                    continue
                for r in mapa.recursos_em(pos):
                    if self.rng.random() < 0.15:        # erro de sonda: 15% falha
                        continue
                    out.append((abs(dx) + abs(dy), pos, r, "olho"))
        return out

    def _ouvir(self, quadro):
        out = []
        for (autor, mx, my, simb) in quadro.ler_perto(self.x, self.y, raio=5):
            if autor == self.id:
                continue
            for r, s in self.genoma["lexico"].items():      # decodifica c/ SEU léxico
                if s == simb:
                    out.append((abs(mx - self.x) + abs(my - self.y),
                                (mx, my), r, "boato"))
                    break
        return out

    # ── decisão + ação ───────────────────────────────────────────────────────
    def agir(self, mapa, quadro, clima_gasto, eh_noite=False):
        # 1 chamada = 1 DIA: forrageia/move/come. NÃO envelhece aqui (idade é por ano).
        self.energia -= self._termico(mapa, clima_gasto)   # fogo aquece OU queima

        if eh_noite and self._abrigar(mapa):               # noite → recolhe ao abrigo
            return

        if self._busca_fogo(mapa, clima_gasto):            # frio + atração → ao fogo
            return

        opcoes = self._perceber(mapa)
        if self.rng.random() < self.genoma["confia_quadro"]:
            opcoes += self._ouvir(quadro)
        # prioriza o recurso do papel, depois o mais perto
        opcoes.sort(key=lambda o: (o[2] != self.genoma["papel"], o[0]))

        if not opcoes:                                       # nada à vista → vagueia
            self._passo(self.rng.randint(-1, 1), self.rng.randint(-1, 1), mapa)
            return
        _, (ax, ay), recurso, fonte = opcoes[0]
        self._alvo_por_boato = (recurso, fonte == "boato")
        if (ax, ay) == (self.x, self.y):
            self._coletar(mapa, quadro, recurso)
        else:
            self._passo((ax > self.x) - (ax < self.x),
                        (ay > self.y) - (ay < self.y), mapa)

    def _passo(self, dx, dy, mapa):
        if mapa.dentro(self.x + dx, self.y + dy):
            self.x += dx
            self.y += dy

    def _coletar(self, mapa, quadro, recurso):
        capacidade = 22 if self.tem_ferramenta else 15   # ferramenta → coleta mais
        pego = mapa.coletar((self.x, self.y), recurso, capacidade)
        if pego <= 0:
            return
        self.coletado[recurso] += pego
        if recurso == "comida":
            self.energia += pego * 3.2                       # comida → energia (vida)
        # creditar a comunicação: vim por um boato e ACHEI o recurso?
        if self._alvo_por_boato and self._alvo_por_boato == (recurso, True):
            self.acertos_fala += 1
        # FALAR: anuncia o recurso no quadro, no SEU símbolo
        if self.rng.random() < self.genoma["tagarela"]:
            quadro.escrever(self.id, self.x, self.y, self.genoma["lexico"][recurso])

    # ── USO dos materiais: dar PROPÓSITO à coleta (não coletar por coletar) ──
    def usar_materiais(self, mapa, clima_gasto):
        """Madeira→fogo, ferro→ferramenta. Só quem DECIFROU o fogo (crença>0.3)
        sabe acendê-lo — entender o fogo vira o poder de PRODUZI-lo."""
        # FERRO → FERRAMENTA: coleta mais rápido a partir daí (uso permanente do ferro)
        if not self.tem_ferramenta and self.coletado["ferro"] >= 12:
            self.coletado["ferro"] -= 12
            self.tem_ferramenta = True
        # MADEIRA → FOGO: no frio, quem entende o fogo o PRODUZ pra se aquecer
        if (clima_gasto >= 0.55 and self.crenca_fogo > 0.3
                and self.coletado["madeira"] >= 8
                and mapa.calor_em(self.x, self.y) < 0.2
                and not mapa.fogo_proximo(self.x, self.y, 2)):
            self.coletado["madeira"] -= 8
            mapa.acender_fogo((self.x, self.y), 1.0)
        # DEGRADAÇÃO: madeira estocada apodrece, MENOS se guardada na caverna (armazém)
        cav = mapa.cavernas.get(self.tribo)
        perto_cav = cav and abs(cav[0] - self.x) + abs(cav[1] - self.y) <= 2
        if self.coletado["madeira"] > 0 and not perto_cav:
            self.coletado["madeira"] *= 0.97        # -3%/dia exposto ao tempo

    def _abrigar(self, mapa):
        """À noite recolhe à CAVERNA da tribo. Quem está na caverna OU perto do fogo
        escapa do FRIO NOTURNO; quem fica exposto paga o frio. True se já tratou o dia."""
        cav = mapa.cavernas.get(self.tribo)
        perto_cav = cav and abs(cav[0] - self.x) + abs(cav[1] - self.y) <= 1
        if perto_cav or mapa.calor_em(self.x, self.y) > 0.1:
            return True                             # abrigado: sem frio, fica recolhido
        self.energia -= FRIO_NOITE                  # exposto ao frio da noite
        if cav and self.energia >= 30:              # corre pra caverna (se não em fome)
            self._passo((cav[0] > self.x) - (cav[0] < self.x),
                        (cav[1] > self.y) - (cav[1] < self.y), mapa)
            return True
        return False                                # fome ou sem caverna: forrageia

    # ── FOGO: sentir, aprender e decidir (sem ninguém ensinar) ───────────────
    def _termico(self, mapa, clima_gasto):
        """Efeito FÍSICO do fogo na célula atual + APRENDIZADO (reforço). Devolve o
        gasto de energia já ajustado: o fogo POUPA energia no frio, mas QUEIMA de perto.
        O bot só sente o saldo e atualiza a crença — ninguém lhe diz o que é fogo."""
        self.aqueceu = False
        calor = mapa.calor_em(self.x, self.y)
        if calor <= 0.05:
            return clima_gasto
        if calor >= 0.8:                                   # em cima do fogo: QUEIMA
            self.energia -= calor * 8.0
            self._aprender_fogo(-1)
            return clima_gasto
        economia = clima_gasto * min(1.0, calor * 1.5)     # aquece: poupa o gasto do frio
        self.aqueceu = economia > 0.2
        self._aprender_fogo(+1)
        return clima_gasto - economia

    def _aprender_fogo(self, sinal):
        """Memória por experiência: aqueceu (+) sobe a crença; queimou (−) derruba.
        Ao DECIFRAR o fogo (crença > 0.3) pela 1ª vez, ganha +1 ano (tecnologia nova)."""
        self.crenca_fogo = max(-1.0, min(1.0, self.crenca_fogo + 0.15 * sinal))
        if not self.fogo_premiado and self.crenca_fogo > 0.3:
            self.fogo_premiado = True
            self.genoma["vida_max"] = min(MAX_IDADE, self.genoma["vida_max"] + 1)

    def _busca_fogo(self, mapa, clima_gasto):
        """Só no FRIO e sem pânico de fome: se o bot é atraído (gene + crença), vai até
        a BORDA do fogo (aquece sem queimar). Quem aprendeu busca; quem se queimou recua."""
        if clima_gasto < 0.55 or self.energia < 40:    # frio = gasto diário alto (0.3..0.8)
            return False
        if self.genoma["curiosidade_fogo"] + self.crenca_fogo <= 0.1:
            return False
        alvo = mapa.fogo_proximo(self.x, self.y, raio=self.genoma["visao"] + 2)
        if not alvo:
            return False
        calor = mapa.calor_em(self.x, self.y)
        dx = (alvo[0] > self.x) - (alvo[0] < self.x)
        dy = (alvo[1] > self.y) - (alvo[1] < self.y)
        if calor >= 0.8:                                   # queimando: recua
            self._passo(-dx, -dy, mapa)
        elif calor < 0.4 and mapa.calor_em(self.x + dx, self.y + dy) < 0.8:
            self._passo(dx, dy, mapa)                       # avança sem pisar no fogo
        return True                                         # na zona ótima: fica e aquece

    # ── SONO: refatorar a própria pré-escrita (a Lei da Reescrita) ───────────
    def sonhar(self, dados, esp, gen):
        """No sono, tenta REESCREVER a pré-escrita pra ficar mais eficiente (reusa o
        lab: muta + valida no sandbox + mede latência). Se melhorou de verdade, adota,
        GANHA +2 anos de vida (recompensa) e cataloga a otimização (pra portar à Selene).
        Todo código roda em processo isolado (safety) — variante perigosa não toca nada."""
        if not self.pre_escrita:
            return
        if self.pre_lat is None:                       # 1ª vez: mede a base...
            ok, res, lat, _ = executar_seguro(self.pre_escrita, dados)
            if not (ok and res == esp(dados)):
                return
            self.pre_lat = lat                         # ...e já segue p/ tentar otimizar
        prop = mutations.mutar(self.pre_escrita, preferir_otimizacao=True)
        if not prop or not validar_seguranca(prop[0])[0]:
            return
        novo, desc = prop
        ok, res, lat, _ = executar_seguro(novo, dados)
        if ok and res == esp(dados) and lat < self.pre_lat:
            ganho = (1 - lat / self.pre_lat) * 100
            self.pre_escrita, self.pre_lat = novo, lat
            self.otimizacoes += 1
            self.genoma["vida_max"] = min(MAX_IDADE, self.genoma["vida_max"] + 2)
            gen.registrar_descoberta(self.id, self.geracao,
                                     f"tribo{self.tribo}", desc, ganho)

    # ── ciclo de vida ────────────────────────────────────────────────────────
    def envelhecer(self):
        self.idade += 1                                      # chamado 1×/ano (365 dias)

    def vivo(self):
        teto = min(self.genoma["vida_max"], MAX_IDADE)       # ninguém passa de 120 anos
        return self.energia > 0 and self.idade < teto

    def pode_reproduzir(self):
        # puberdade aos 15 + intervalo de 6 anos = reprodução lenta (dá tempo de
        # desenvolver), mas viável o bastante p/ a população não definhar
        return (self.energia >= 130 and self.idade >= 15
                and self.idade - self.ultima_repro >= 6)

    def reproduzir(self, mapa):
        self.energia -= 70
        self.ultima_repro = self.idade                       # marca p/ espaçar o próximo
        fx = max(0, min(mapa.w - 1, self.x + self.rng.choice([-1, 1])))
        fy = max(0, min(mapa.h - 1, self.y + self.rng.choice([-1, 1])))
        # o filho HERDA a pré-escrita já evoluída (o código melhora ao longo das
        # gerações), mas a vida volta a 80 — longevidade é mérito de cada um.
        return SeleneBot(fx, fy, self._mutar(), self.geracao + 1, self.id, self.log,
                         pre_escrita=self.pre_escrita, tribo=self.tribo)

    def _mutar(self):
        g = copy.deepcopy(self.genoma)
        g["vida_max"] = VIDA_BASE                            # nasce com 80 (mérito reinicia)
        if self.rng.random() < 0.3:
            g["confia_quadro"] = round(min(1, max(0,
                g["confia_quadro"] + self.rng.uniform(-0.1, 0.1))), 2)
        if self.rng.random() < 0.3:
            g["tagarela"] = round(min(1, max(0,
                g["tagarela"] + self.rng.uniform(-0.1, 0.1))), 2)
        if self.rng.random() < 0.3:                          # predisposição ao fogo evolui
            g["curiosidade_fogo"] = round(min(1, max(-1,
                g["curiosidade_fogo"] + self.rng.uniform(-0.15, 0.15))), 2)
        if self.rng.random() < 0.2:                          # mutação da LINGUAGEM
            r = self.rng.choice(RECURSOS)
            g["lexico"][r] = (g["lexico"][r] + self.rng.choice([-1, 1])) % 256
        return g
