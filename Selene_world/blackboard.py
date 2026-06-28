"""
blackboard.py — O QUADRO NEGRO Compartilhado (memética → linguagem).

Bots escrevem e leem mensagens CURTAS perto de onde estão. É o berço da
linguagem: não programamos significado nenhum. Damos só:
  • um canal (este quadro),
  • a capacidade de emitir/ler símbolos (no genoma do bot),
  • e PRESSÃO (escassez + erro de percepção).
Se um símbolo passar a prever recurso de forma confiável, quem o usa/entende
prospera → a seleção faz a população convergir num código comum. A linguagem
EMERGE; não é escrita por nós.

Mensagem = (autor_id, x, y, simbolo)  — "no lugar (x,y) tem algo que chamo de S".
"""


class QuadroNegro:
    def __init__(self, capacidade=60, validade=3):
        self.capacidade = capacidade
        self.validade = validade            # ciclos até a mensagem "apagar"
        self.mensagens = []                 # [(autor_id, x, y, simbolo, idade)]

    def escrever(self, autor_id, x, y, simbolo):
        self.mensagens.append([autor_id, x, y, int(simbolo) % 256, 0])
        if len(self.mensagens) > self.capacidade:
            self.mensagens.pop(0)

    def ler_perto(self, x, y, raio=5):
        """Comunicação é LOCAL: só enxerga o que foi escrito por perto."""
        return [(a, mx, my, s) for (a, mx, my, s, _) in self.mensagens
                if abs(mx - x) <= raio and abs(my - y) <= raio]

    def envelhecer(self):
        """Mensagens velhas somem (informação perde validade)."""
        for m in self.mensagens:
            m[4] += 1
        self.mensagens = [m for m in self.mensagens if m[4] <= self.validade]
