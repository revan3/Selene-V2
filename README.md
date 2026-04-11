# Selene Brain V2 — Sistema Neural Bio-Inspirado

> **Simulação de cérebro artificial em Rust com neurônios Izhikevich, precisão mista, STDP e 7 tipos neuronais biológicos.**

---

## Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura do Sistema](#arquitetura-do-sistema)
3. [Núcleo Neural — synaptic_core](#núcleo-neural--synaptic_core)
4. [Tipos de Neurônio](#tipos-de-neurônio)
5. [Precisão Mista](#precisão-mista)
6. [Regiões Cerebrais](#regiões-cerebrais)
7. [Neuroquímica](#neuroquímica)
8. [Memória e Storage](#memória-e-storage)
9. [Interface WebSocket](#interface-websocket)
10. [Como Compilar e Rodar](#como-compilar-e-rodar)
11. [Estrutura de Arquivos](#estrutura-de-arquivos)
12. [Roadmap](#roadmap)

---

## Visão Geral

Selene é uma simulação de sistema nervoso artificial que replica aspectos centrais da neurobiologia computacional:

- **Neurônios Izhikevich** com 7 tipos funcionais distintos (RS, IB, CH, FS, LT, TC, RZ)
- **Precisão mista** (FP32/FP16/INT8/INT4) por neurônio, economia de ~60% de memória
- **STDP** (Spike-Timing Dependent Plasticity) com LTP e LTD anti-Hebbiana
- **14 regiões cerebrais** com composição neuronal específica por área
- **Neuroquímica dinâmica** (serotonina, dopamina, cortisol, noradrenalina, ocitocina, D1/D2)
- **Memória hierárquica** L1→L4 (RAM → NVMe → SwapManager/SurrealDB)
- **Interface WebSocket** em tempo real para monitoramento e chat neural
- **Ciclo de sono/vigília** com consolidação de memórias em REM
- **Motor de hipóteses** com feedback STDP e aprendizado causal
- **Linguagem bidirecional** com Broca (produção) e Wernicke (compreensão)

```
Sensores → Tálamo → Regiões Cerebrais → Neuroquímica → Memória → WebSocket
```

---

## Arquitetura do Sistema

```
╔══════════════════════════════════════════════════════════════════════╗
║                        SELENE BRAIN V2                               ║
║                                                                      ║
║  ┌─────────────┐    ┌──────────────┐    ┌────────────────────────┐  ║
║  │  WebSocket  │◄──►│  BrainState  │◄──►│  main.rs tick loop     │  ║
║  │  server.rs  │    │  bridge.rs   │    │  (~1ms / tick)          │  ║
║  └─────────────┘    └──────────────┘    └────────────────────────┘  ║
║         │                                          │                  ║
║  ┌──────▼──────────────────────────────────────────▼──────────────┐ ║
║  │                    REGIÕES CEREBRAIS                            │ ║
║  │  Frontal │ Parietal │ Temporal │ Occipital │ Limbic             │ ║
║  │  Hippoc. │ Cerebelo │ Caloso   │ ACC       │ OFC                │ ║
║  │  Language│ Mirror   │ Depth    │ Interocp. │                    │ ║
║  └─────────────────────────────────────────────────────────────────┘ ║
║         │                                                             ║
║  ┌──────▼──────────────────────────────────────────────────────────┐ ║
║  │            SYNAPTIC CORE — CamadaHibrida                        │ ║
║  │   NeuronioHibrido × N   |   STDP   |   Precisão Mista           │ ║
║  └─────────────────────────────────────────────────────────────────┘ ║
║         │                                                             ║
║  ┌──────▼──────────────────────────────────────────────────────────┐ ║
║  │            STORAGE — SwapManager + SurrealDB                    │ ║
║  │   L1: RAM  │  L2: NVMe  │  L3: SurrealDB  │  L4: Checkpoint     │ ║
║  └─────────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Núcleo Neural — synaptic_core

Cada neurônio em Selene é um `NeuronioHibrido` que implementa o modelo de Izhikevich com parâmetros (a, b, c, d) específicos por tipo:

```
v' = 0.04v² + 5v + 140 - u + I
u' = a(bv - u)
se v ≥ 30mV: spike → v = c, u = u + d
```

A `CamadaHibrida` agrupa N neurônios em composição mista (tipo primário + tipo secundário), executa STDP automático e retorna vetor de spikes a cada tick.

### STDP (Spike-Timing Dependent Plasticity)

- **LTP**: pré dispara antes do pós (Δt < 0) → reforço sináptico
- **LTD**: pós dispara antes do pré (Δt > 0) → enfraquecimento sináptico
- Janela temporal: ±20ms, τ = 10ms
- Peso sináptico: clampado em [−1.0, 1.0]

---

## Tipos de Neurônio

| Tipo | Nome | Comportamento | Onde predomina |
|------|------|---------------|----------------|
| **RS** | Regular Spiking | Disparo tônico sustentado | Córtex, frontal, hipocampo |
| **IB** | Intrinsic Bursting | Burst inicial + adaptação | ACC conflict, dACC |
| **CH** | Chattering | Bursts repetitivos rápidos | Wernicke, temporal |
| **FS** | Fast Spiking | Inibição rápida (interneurônio) | ACC regulation, Broca, límbico |
| **LT** | Low-Threshold | Dispara com correntes fracas | Tálamo sensorial |
| **TC** | Thalamocortical | Oscilações sono/vigília | Córtex cingulado, ciclo REM |
| **RZ** | Resonator | Ressoa em frequências específicas | Occipital visual |

---

## Precisão Mista

Cada neurônio carrega seus pesos em uma das 4 precisões:

| Precisão | Bits | Uso típico | Memória relativa |
|----------|------|------------|------------------|
| FP32 | 32 | Neurônios críticos de decisão | 100% |
| FP16 | 16 | Camadas intermediárias | 50% |
| INT8 | 8 | Processamento de volume | 25% |
| INT4 | 4 | Compressão máxima | 12.5% |

Distribuição por região otimizada para economizar ~60% de memória vs FP32 puro.

---

## Regiões Cerebrais

### 1. Lobo Frontal (`frontal.rs`)
- **Composição**: RS (60%) + IB (20%) + FS (20%)
- **Função**: Planejamento, tomada de decisão, working memory
- **Conexões**: Recebe cerebelo→PFC (5% saída cerebelar injetada a cada 5 ticks), D1 boost dopaminérgico

### 2. Lobo Parietal (`parietal.rs`)
- **Composição**: RS (70%) + CH (30%)
- **Função**: Integração sensorial, atenção espacial
- **Conexão**: `attention_weight` modula entrada do graph-walk

### 3. Lobo Temporal (`temporal.rs`)
- **Composição**: RS (50%) + CH (30%) + FS (20%)
- **Função**: Processamento auditivo, semântica, memória de trabalho
- **Conexão**: `apply_rpe()` recebe erro de predição de recompensa

### 4. Lobo Occipital (`occipital.rs`)
- **Composição**: RS (50%) + RZ (30%) + LT (20%)
- **Função**: Processamento visual, padrões
- **Conexão**: `sensitivity` modula limiar de detecção visual

### 5. Sistema Límbico (`limbic.rs`)
- **Composição**: RS (40%) + FS (40%) + IB (20%)
- **Função**: Emoção, motivação, valência afetiva
- **Campos**: `habituation_nivel` — reduz `n_passos` com exposição repetida

### 6. Hipocampo (`hippocampus.rs`)
- **Composição**: RS (60%) + CH (20%) + LT (20%)
- **Função**: Consolidação de memória episódica, HippocampusV2 com motor de hipóteses
- **Conexões**: `hipoteses_confiaveis()` → STDP; `gaps_conhecimento()` → neural_context; `proximo_topico_previsto()` → push_front

### 7. Cerebelo (`cerebellum.rs`)
- **Composição**: RS (70%) + FS (20%) + LT (10%)
- **Função**: Coordenação motora, predição de erro temporal
- **Conexão**: Saída cerebelar injetada no `working_memory_trace` frontal (cerebelo→PFC)

### 8. Corpo Caloso (`corpus_callosum.rs`)
- **Composição**: RS (80%) + CH (20%)
- **Função**: Transferência inter-hemisférica, sincronização
- **Dinâmica**: Latência 4–20ms controlada por arousal; serotonina modula conectividade

### 9. Córtex Cingulado Anterior — ACC (`cingulate.rs`)
- **conflict_layer**: IB (40%) + RS (60%) — burst quando conflito é alto (dACC)
- **regulation_layer**: RS (70%) + FS (30%) — inibição emocional (rACC)
- **Função**: Monitoramento de conflito frontal vs límbico, dor social, ajuste comportamental
- **Saídas**: `noradrenaline_drive()` → NA ↑ quando conflito > 0.45; `amygdala_inhibition()` → inibe BLA; `adjustment_factor` → reduz `n_passos`

### 10. Córtex Orbitofrontal — OFC (`orbitofrontal.rs`)
- **value_layer**: RS + IB — encoding de valor por contexto
- **extinction_layer**: RS + FS — aprendizado de extinção reversal
- **Função**: Mapa de valor por contexto (até 512 entradas), reversal learning
- **Dinâmica**: Extinção 3× mais rápida que aprendizado (EXTINCTION_LR = 0.25 vs VALUE_LR = 0.08)

### 11. Áreas de Linguagem (`language.rs`)
- **wernicke_layer**: RS (60%) + CH (40%) — compreensão semântica
- **broca_layer**: RS (70%) + FS (30%) — produção e fluência
- **Função**: Compreensão (familiarity scoring), produção (fluency_signal + syntax_template)
- **Interface**: `quer_perguntar()` → `comprehension < 0.45 && syntax_template[0] > 0.5`

### 12. Neurônios Espelho (`mirror_neurons.rs`)
- **Composição**: RS + IB
- **Função**: Reconhecimento de ação/intenção, empatia computacional

### 13. Profundidade de Processamento (`depth_stack.rs`)
- **Função**: Pilha de contexto com múltiplas camadas de profundidade cognitiva

### 14. Interoceptor (`interoception/`)
- **Função**: Monitoramento de estados internos (fome energética, fadiga, estado homeostático)

---

## Neuroquímica

| Neurotransmissor | Campo | Função | Range |
|------------------|-------|---------|-------|
| Dopamina | `dopamine` | Recompensa, motivação, RPE | 0.0–2.0 |
| Serotonina | `serotonin` | Humor, regulação social, latência calosa | 0.0–1.5 |
| Noradrenalina | `noradrenaline` | Atenção, arousal, locus coeruleus | 0.0–1.5 |
| Cortisol | `cortisol` | Estresse, memória de medo | 0.0–1.0 |
| Acetilcolina | `acetylcholine` | Aprendizado, plasticidade sináptica | 0.0–1.5 |
| Ocitocina | `oxytocin` | Vínculo social, confiança; cresce com RPE > 0 | 0.0–1.0 |
| D1 (receptor) | `d1_signal` | Alta dopamina → excitação PFC (sigmoid) | 0.0–1.0 |
| D2 (receptor) | `d2_signal` | Baixa dopamina → filtragem estriatal (sigmoid) | 0.0–1.0 |

### Cascata Neuroquímica por Evento

```
RPE > 0.2  → dopamina ↑ → D1 ↑ → PFC boost + oxytocina ↑ (interação positiva)
RPE < −0.2 → cortisol ↑ → ACC.registrar_rejeicao() + social_pain ↑
conflito > 0.45 → ACC → noradrenaline_drive() → LC → NA ↑ → atenção ↑
rACC ativo → amygdala_inhibition() → BLA ↓ (regulação do medo)
```

---

## Memória e Storage

### Hierarquia L1–L4

```
L1: NeuronioHibrido.historico_spikes   (RAM, ~1ms acesso)
L2: working_memory_trace frontal        (RAM, deque circular)
L3: SwapManager — grafo causal          (NVMe, ~10ms)
L4: SurrealDB checkpoint                (disco, persistência)
```

### SwapManager (Grafo Causal)

- Pares `(palavra_A, palavra_B, peso)` armazenados como arestas direcionadas
- `importar_causal()`: adiciona/reforça arestas
- `graph_walk()`: navegação probabilística no grafo semântico
- **Reversal LTD**: OFC detecta inversão de valor → `importar_causal` com peso negativo

### Motor de Hipóteses (HippocampusV2)

- Aprende padrões de co-ocorrência entre conceitos
- `hipoteses_confiaveis()` → pares com confiança > threshold → STDP
- `gaps_conhecimento()` → conceitos sub-representados → injetados no neural_context
- `proximo_topico_previsto()` → antecipa próximo tópico → `push_front` no contexto

---

## Interface WebSocket

### Conexão
```
ws://localhost:9001
```

### Mensagens de Entrada (cliente → Selene)

| Tipo | Payload | Descrição |
|------|---------|-----------|
| `chat` | `{"type":"chat","content":"texto"}` | Envia mensagem para Selene |
| `ping` | `{"type":"ping"}` | Verifica conexão |
| `set_mode` | `{"type":"set_mode","mode":"contemplativo"}` | Altera modo operacional |

### Mensagens de Saída (Selene → cliente)

| Tipo | Descrição |
|------|-----------|
| `neural_state` | Estado completo do cérebro a cada tick |
| `chat_response` | Resposta emergente do graph-walk |
| `spike_data` | Vetor de spikes das camadas ativas |

### Campos do BrainState

```json
{
  "emocao": 0.3,
  "arousal": 0.7,
  "dopamina": 1.2,
  "serotonina": 0.8,
  "noradrenalina": 0.6,
  "cortisol": 0.2,
  "oxytocin_level": 0.65,
  "acc_conflict": 0.15,
  "acc_social_pain": 0.05,
  "ofc_value_bias": 0.4,
  "wernicke_comprehension": 0.82,
  "broca_fluency": 0.75,
  "habituation_nivel": 0.1,
  "modo_operacional": "contemplativo",
  "ciclo_sono": "vigilia"
}
```

---

## Como Compilar e Rodar

### Pré-requisitos

- Rust 1.75+ (`rustup update stable`)
- Cargo
- SurrealDB (opcional, para persistência L4)

### Compilar

```bash
cargo build --release
```

### Rodar

```bash
cargo run --release
```

O servidor WebSocket sobe em `ws://0.0.0.0:9001`.

### Abrir Interface

Abrir `index.html` no navegador (ou usar `neural_interface.html` para monitoramento).

### Benchmarks

```bash
cargo run --bin intensive_benchmark --release
cargo run --bin system_test --release
```

---

## Estrutura de Arquivos

```
selene_kernel/
├── src/
│   ├── main.rs                    — Loop principal de ticks (~1ms)
│   ├── neurochem.rs               — Neuroquímica dinâmica (8 moléculas)
│   ├── config.rs                  — Configuração global
│   ├── synaptic_core/             — Núcleo neural
│   │   ├── mod.rs                 — CamadaHibrida, NeuronioHibrido
│   │   └── stdp.rs                — STDP, LTP/LTD
│   ├── brain_zones/               — 14 regiões cerebrais
│   │   ├── mod.rs
│   │   ├── frontal.rs             — Lobo frontal, working memory
│   │   ├── parietal.rs            — Integração sensorial, atenção
│   │   ├── temporal.rs            — Auditivo, semântica, RPE
│   │   ├── occipital.rs           — Visual, padrões
│   │   ├── limbic.rs              — Emoção, motivação
│   │   ├── hippocampus.rs         — Memória episódica, hipóteses
│   │   ├── cerebellum.rs          — Coordenação, erro temporal
│   │   ├── corpus_callosum.rs     — Inter-hemisférico, latência dinâmica
│   │   ├── cingulate.rs           — ACC: conflito, dor social
│   │   ├── orbitofrontal.rs       — OFC: valor por contexto, reversal
│   │   ├── language.rs            — Broca + Wernicke
│   │   ├── mirror_neurons.rs      — Empatia, intenção
│   │   └── depth_stack.rs         — Profundidade cognitiva
│   ├── learning/                  — Aprendizado
│   │   ├── mod.rs                 — RL, STDP supervisor
│   │   ├── pensamento.rs          — Motor de pensamento emergente
│   │   └── narrativa.rs           — Narrativa e contexto temporal
│   ├── storage/                   — Persistência
│   │   ├── mod.rs
│   │   ├── swap_manager.rs        — Grafo causal em NVMe
│   │   └── checkpoint.rs          — Checkpoints periódicos
│   ├── interoception/             — Estados internos homeostáticos
│   └── websocket/
│       ├── server.rs              — Handler WebSocket + chat
│       └── bridge.rs              — BrainState compartilhado
├── Cargo.toml
└── selene_memories.db/            — Banco SurrealDB local
```

---

## Roadmap

### V2.x (em andamento)
- [x] 7 tipos neuronais Izhikevich
- [x] Precisão mista FP32/FP16/INT8/INT4
- [x] STDP com janela temporal
- [x] 14 regiões cerebrais
- [x] Neuroquímica dinâmica (8 moléculas incluindo oxitocina, D1/D2)
- [x] ACC — monitoramento de conflito e dor social
- [x] OFC — reversal learning e mapa de valor
- [x] Áreas de linguagem (Broca + Wernicke)
- [x] Motor de hipóteses com feedback STDP
- [x] Cerebelo→PFC projeção
- [x] Corpus callosum com latência dinâmica
- [x] SwapManager — grafo causal semântico

### V3 (planejado)
- [ ] One-shot learning episódico
- [ ] Graph versioning (snapshots temporais do grafo)
- [ ] Embeddings vetoriais para busca semântica
- [ ] Mirror neurons → empatia computacional completa
- [ ] Tálamo como gate sensorial explícito
- [ ] Gânglios da base (D1/D2 → Go/NoGo pathway completo)
- [ ] Amígdala como estrutura separada (BLA + CeA)
- [ ] Integração multimodal (texto + áudio + visão)

---

*Selene Brain V2 — Criado por Rodrigo Luz*
