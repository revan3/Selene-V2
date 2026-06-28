# 🌍 Selene-World

Ecossistema de **Vida Artificial (ALife)**: agentes ("Selene-bots") que vivem num
mapa 2D, competem por recursos, **evoluem** e — sob pressão de escassez — começam
a **criar a própria linguagem** pra coordenar.

100% Python puro (só `psutil` opcional). **Independente da Selene real** — não
precisa do cérebro neural nem do WebSocket. Roda em qualquer máquina com Python 3.9+.

## Como rodar

```bash
python world.py            # 60 ciclos
python world.py 200        # 200 ciclos (a linguagem precisa de tempo p/ convergir)
python world.py 200 --render   # mostra o mapa ASCII a cada 12 ciclos
```

Mapa: `f`=comida · `T`=madeira · `^`=ferro · `.`=vazio · `@`=bot · `&`=multidão.
Cada execução grava um log detalhado em `logs/`.

## Os 3 indicadores que importam

- **População** — oscila (boom-bust): reprodução faz crescer, fome/catástrofes cortam.
- **Pulso do PC** (`cpu%`/`ram%`) — o mundo **respira com a sua máquina**: PC ocioso =
  abundância; PC saturado = escassez/fome. (Use o PC pesado e veja os bots sofrerem.)
- **Língua** (0→1) — convergência da linguagem. `0` = Babel (cada bot inventa seus
  símbolos); perto de `1` = todos chamam cada recurso pelo **mesmo** símbolo.

## Arquitetura

| Módulo | Papel |
|--------|-------|
| `system_resources.py` | lê CPU/RAM reais → abundância/escassez do mundo (a "joia") |
| `world_map.py` | grid 2D, recursos com respawn modulado pelo PC |
| `blackboard.py` | quadro negro: bots escrevem/leem símbolos perto (berço da língua) |
| `bot_core.py` | o agente: genoma + léxico, percepção com erro, mover/comer/falar |
| `events.py` | 🎉 festas (abundância) · 💀 catástrofes/superpopulação (extinção) |
| `orchestrator.py` | o mundo: clima, encontros (alinham a língua), vida/morte/reprodução |
| `world.py` | loop + mapa ASCII + relatório (população, língua, pulso do PC) |

## Como a linguagem emerge (sem ninguém programar significado)

1. Cada bot nasce com um **léxico aleatório** (símbolo → recurso). Início = Babel.
2. Ao achar recurso, o bot **fala** no quadro (usando o *seu* símbolo).
3. Quem **ouve** decodifica com o *próprio* léxico e vai até lá. Se acha o recurso,
   a comunicação funcionou → ganha vantagem (vive mais, reproduz).
4. **Encontros** no mapa alinham léxicos (transmissão cultural); a **seleção** favorece
   quem comunica bem. → A população **converge** num código comum: a língua nasce.

A **mutação de percepção** (erro de sonda) é o motor: como cada bot vê com falhas,
vale a pena confiar no que os outros dizem — e isso pressiona por um código confiável.

## Roadmap (próximas camadas)

- [ ] **SQLite** — árvore genealógica + performance por bot (ancestralidade rastreada)
- [ ] **Clima dinâmico** detalhado (estações que mudam consumo e respawn)
- [ ] **Papéis interdependentes** (minerador × coletor → cooperação obrigatória)
- [ ] **Genoma executável + Sono** — código de decisão que a Selene refatora (reusa
      `safety.py`/`mutations.py` do v1 abstrato, com `ast.parse` + sandbox)
- [ ] Servidor headless 24/7 + sync com a Selene real (refatoração dos mais aptos)
