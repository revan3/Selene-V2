# Selene Brain V2 — Sistema Neural Bio-Inspirado

> **Simulação de cérebro artificial em Rust com 17 tipos neuronais Izhikevich + biológicos, precisão mista FP32/FP16/INT8/INT4, STDP assimétrico, 14 regiões cerebrais, 11 neurotransmissores dinâmicos, 19 templates cognitivos e aprendizado preditivo via motor de hipóteses.**

---

## Índice

1. [Visão Geral](#visão-geral)
2. [Estado Atual](#estado-atual)
3. [Arquitetura do Sistema](#arquitetura-do-sistema)
4. [Núcleo Neural — synaptic_core](#núcleo-neural--synaptic_core)
5. [Tipos de Neurônio (17 implementados)](#tipos-de-neurônio-17-implementados)
6. [Tipos de Neurônio Ainda Faltando](#tipos-de-neurônio-ainda-faltando)
7. [Precisão Mista](#precisão-mista)
8. [Regiões Cerebrais (14)](#regiões-cerebrais)
9. [Neuroquímica (11 moléculas)](#neuroquímica-11-moléculas)
10. [Sistema de Templates Cognitivos](#sistema-de-templates-cognitivos)
11. [Aprendizado Coerente (CLS)](#aprendizado-coerente-cls)
12. [Memória e Storage](#memória-e-storage)
13. [Motor de Hipóteses](#motor-de-hipóteses)
14. [Interface WebSocket](#interface-websocket)
15. [Interface Neural (Desktop/Mobile)](#interface-neural-desktopmobile)
16. [Como Compilar e Rodar](#como-compilar-e-rodar)
17. [Estrutura de Arquivos](#estrutura-de-arquivos)
18. [Roadmap](#roadmap)

---

## Visão Geral

Selene é uma simulação de sistema nervoso artificial que replica aspectos centrais da neurobiologia computacional moderna:

- **17 tipos neuronais** (7 Izhikevich originais + 6 adicionais + 4 subtipos biológicos) com 7 camadas biológicas por neurônio
- **Precisão mista** (FP32/FP16/INT8/INT4) por neurônio — economia de ~60% de memória
- **STDP assimétrico** — LTP quando pré dispara ANTES de pós (causal), LTD caso contrário
- **14 regiões cerebrais** com composição neuronal específica por área
- **11 neurotransmissores dinâmicos** — dopamina, serotonina, noradrenalina, cortisol, acetilcolina, ocitocina, histamina, adenosina, endocanabinoide, D1, D2
- **Memória hierárquica** L1→L4 (RAM → NVMe → SwapManager → SurrealDB)
- **Interface WebSocket** em `ws://127.0.0.1:3030/selene`
- **Ciclo de sono N1–N4** com consolidação, poda, REM semântico e backup; acionável via interface
- **Motor de hipóteses preditivo** com feedback STDP e aprendizado causal
- **Linguagem bidirecional** com Broca (produção) e Wernicke (compreensão)
- **19 templates cognitivos** com loop de treinamento completo
- **PatternEngine** integrado ao loop neural (episódios visuais, auditivos e de pensamento)
- **Plasticidade homeostática** (synaptic scaling, ~20% esparsidade alvo)
- **Replay reverso no REM** (causalidade bidirecional hipocampal)

```
Sensores → Tálamo → Regiões Cerebrais → Neuroquímica → Memória → WebSocket
```

---

## Estado Atual

### Testes
```
81 testes unitários — todos passando (0 falhas)
  - encoding:      11 testes (phoneme, spike_codec, helix_store, fft)
  - learning:      36 testes (binding, chunking, curriculo, hypothesis, templates)
  - sensors:        6 testes (audio)
  - templates:      5 testes (ciclo_vida, preenchimento_efemero, consolidacao, filhos)
  - thalamus:       8 testes (lgn/mgn relay, burst/tonic mode, NRT, feedback_cortical)
  - basal_ganglia:  9 testes (D1/D2 pathway, Go/NoGo gate, RPE habit update)
  - multimodal:     6 testes (cross-modal prediction, binding score, AV amplification)
```

> **Nota**: 6 doctests desatualizados em `rl.rs` e `sensors/hardware.rs` — código funcional, apenas exemplos de doc quebrados.

### O que funciona hoje

| Componente | Status |
|---|---|
| Tick neural (~200Hz adaptivo) | ✅ estável |
| STDP assimétrico (LTP causal + LTD anti-causal) | ✅ ativo |
| Plasticidade homeostática (synaptic scaling) | ✅ ativo |
| Sparse coding L1 (~20% esparsidade) | ✅ ativo |
| Chat via WebSocket | ✅ funcionando |
| `gerar_resposta_emergente` (graph-walk) | ✅ funcional |
| Cache do grafo semântico (`grafo_dirty`) | ✅ O(1) amortizado |
| Cache de trigramas (`trigrama_cache`) | ✅ pré-computado em BrainState |
| LRU sinapses (≤500k) + spike_vocab (≤50k) | ✅ crescimento limitado |
| Sistema de templates cognitivos (19 base) | ✅ loop completo: usar() + tick_decay() |
| PatternEngine integrado ao loop neural | ✅ episódios visuais/auditivos/pensamento |
| Motor de hipóteses (HypothesisEngine) | ✅ formular() + testar() + STDP |
| Grounding semântico dinâmico | ✅ acumula por co-ocorrência sensorial |
| Narrativa nas respostas | ✅ estado emocional colore o vocabulário |
| Ciclo sono N1–N4 com replay reverso | ✅ consolidação + REM causal |
| Sono forçado via interface (30 min) | ✅ botão SONO 30MIN no header |
| n_neurons dinâmico por RAM disponível | ✅ clamped [1024, 8192] |
| Filtro Go/NoGo de fala | ✅ Selene decide quando falar |
| GPU (wgpu, feature "gpu") | ✅ opcional, shader Izhikevich |
| Tálamo como gate sensorial (LGN/MGN/VPM/NRT) | ✅ tônico + burst |
| Gânglios da Base D1/D2 Go/NoGo | ✅ GPe/STN/GPi pathway completo |
| Integração multimodal AV | ✅ predição cruzada + Hebbian AV |
| Amígdala BLA+CeA | ✅ one-shot learning, condicionamento |
| Script de treinamento de templates | ✅ `treinar_templates.py` |
| 17 tipos neuronais (Izhikevich + biológicos) | ✅ PS, PB, AC, BI, DAP, IIS, PV, SST, VIP, DA_N |
| 11 neurotransmissores dinâmicos | ✅ + histamina, adenosina, endocanabinoide |
| STP calibrado para todos 17 tipos | ✅ Tsodyks-Markram por tipo |
| W_MAX alinhado entre módulos (2.5) | ✅ inter_lobe.rs = swap_manager.rs |

### Limitações conhecidas
- Respostas podem soar como "frases empilhadas" — o graph-walk semântico não tem gramática; templates mitigam mas ainda são jovens
- Doctests em `rl.rs` e `sensors/hardware.rs` precisam atualização
- `#[allow(dead_code)]` global em 31 arquivos suprime detecção automática de código morto

---

## Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SELENE BRAIN V2                                │
│                                                                         │
│  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────────┐ │
│  │   WebSocket     │   │   BrainState     │   │  main.rs tick loop   │ │
│  │   server.rs     │◄──│   bridge.rs      │◄──│  (200Hz / 5Hz idle)  │ │
│  └─────────────────┘   └──────────────────┘   └──────────────────────┘ │
│           │                                              │               │
│  ┌────────┴─────────────────────────────────────────────┴─────────────┐ │
│  │                     REGIÕES CEREBRAIS (14)                         │ │
│  │  Frontal │ Parietal │ Temporal │ Occipital │ Limbic                │ │
│  │  Hippoc. │ Cerebelo │ Caloso   │ ACC       │ OFC                   │ │
│  │  Language│ Mirror   │ Depth    │ Amígdala  │                       │ │
│  └──────────────────────────────┬──────────────────────────────────────┘ │
│                                 │                                         │
│  ┌──────────────────────────────┴──────────────────────────────────────┐ │
│  │         SYNAPTIC CORE — NeuronioHibrido (7 camadas, 17 tipos)      │ │
│  │   Izhikevich │ HhV3 │ Canais Iônicos │ STP │ Ca²⁺ │ STDP-3f │ ACh │ │
│  └──────────────────────────────┬──────────────────────────────────────┘ │
│                                 │                                         │
│  ┌──────────────────────────────┴──────────────────────────────────────┐ │
│  │       APRENDIZADO — SwapManager + PatternEngine + HypothesisEngine │ │
│  │  STDP assimétrico │ Homeostase │ Sparse L1 │ LRU │ TemplateStore   │ │
│  └──────────────────────────────┬──────────────────────────────────────┘ │
│                                 │                                         │
│  ┌──────────────────────────────┴──────────────────────────────────────┐ │
│  │          STORAGE — SurrealDB + Checkpoints + Sleep Cycle           │ │
│  │  L1: RAM  │  L2: NVMe  │  L3: SurrealDB  │  L4: Checkpoint        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Núcleo Neural — synaptic_core

Cada neurônio em Selene é um `NeuronioHibrido` com 7 camadas biológicas:

| Camada | Mecanismo | Referência |
|--------|-----------|-----------|
| 1 | Izhikevich (todos os 17 tipos) | Izhikevich 2003 |
| 2 | Hodgkin-Huxley (HhV3) + I_T Ca²⁺ (TC e RZ) | Destexhe et al. 1994 |
| 3 | Canais iônicos: I_NaP, I_M, I_A, I_T, I_BK | Adams 1982, Connor 1971 |
| 4 | Short-Term Plasticity (Tsodyks-Markram, calibrado por tipo) | Tsodyks & Markram 1997 |
| 5 | Ca²⁺ dual: AHP (SK) + NMDA (LTP trigger) | — |
| 6 | STDP 3 fatores (dopamina como gate) | Frémaux & Gerstner 2016 |
| 7 | ACh como 4º neuromodulador | — |

### STDP Assimétrico

- **LTP**: pré dispara **antes** do pós (Δt < 0) → reforço sináptico (causal)
- **LTD**: pós dispara **antes** do pré (Δt > 0) → enfraquecimento sináptico (anti-causal)
- `LTD_CONCEITO = LTP_CONCEITO × 0.7` — assimetria biológica preservada
- Janela temporal: ±20ms, τ = 10ms

### Plasticidade Homeostática

- Alvo de ativação: 20% da população neuronal (`HOMEOSTASE_ALVO = 0.20`)
- Synaptic scaling: se ativação < 15% → pesos sobem; se > 25% → pesos descem
- Previne silêncio e runaway excitation (Turrigiano 2008)

### Sparse Coding

- Regularização L1 suave (`SPARSE_REG = 0.003`) — suprime neurônios hiperativos
- Mantém ~20% de esparsidade na representação distribuída (Yamins 2021)

---

## Tipos de Neurônio (17 implementados)

Definidos em `src/synaptic_core.rs` com parâmetros Izhikevich (a, b, c, d), condutâncias iônicas (g_nap, g_m, g_a, g_t, g_bk) e STP específicos por tipo.

### Tipos Originais (7)

| Tipo | Nome Completo | Comportamento | Onde Predomina | HH |
|------|--------------|---------------|----------------|-----|
| **RS** | Regular Spiking | Disparo tônico sustentado, adaptação lenta | Córtex, frontal, hipocampo | ❌ |
| **IB** | Intrinsic Bursting | Burst inicial seguido de regular | ACC conflict, amígdala BLA | ❌ |
| **CH** | Chattering | Bursts rápidos repetitivos | Wernicke, temporal, visual V2/V3 | ❌ |
| **FS** | Fast Spiking | Interneurônio GABAérgico inibitório rápido | Broca, límbico, ACC regulation | ❌ |
| **LT** | Low-Threshold | Interneurônio limiar baixo, disparo suave | Tálamo sensorial, inibidor | ❌ |
| **TC** | Thalamo-Cortical | Burst (sono) ↔ tônico (vigília) | LGN, MGN, VPM, NRT | ✅ |
| **RZ** | Resonator/Purkinje | Ressoa em frequências, rebound bursting | Cerebelo, giro dentado | ✅ |

### Tipos Izhikevich Adicionais (6)

| Tipo | Nome Completo | Comportamento | Papel Funcional |
|------|--------------|---------------|-----------------|
| **PS** | Phasic Spiking | Dispara APENAS no onset do estímulo | Detecção de mudança, córtex sensorial L4 |
| **PB** | Phasic Bursting | Burst único na borda de subida | Sinalização de novidade sensorial |
| **AC** | Accommodating | Adapta progressivamente até silêncio total | Habituação local (barril cortical, colículo) |
| **BI** | Bistable | Dois estados estáveis (ON/OFF) com histerese | Working memory de curto prazo |
| **DAP** | Depolarizing Afterpotential | Rebound despolarizante após spike | Trato olfatório, interneurônios hipocampais |
| **IIS** | Inhibition-Induced Spiking | Dispara quando a inibição é REMOVIDA | Rebound burst talâmico, desinibição basal |

### Subtipos Biológicos (4)

| Tipo | Nome Completo | Comportamento | Papel Funcional |
|------|--------------|---------------|-----------------|
| **PV** | Parvalbumin | FS de alta precisão, Ca²⁺-buffered | Ritmo gamma (40 Hz), inibição perisomal |
| **SST** | Somatostatin (Martinotti) | Adapting, facilitante, inibição dendrítica | Controla janela de plasticidade apical |
| **VIP** | VIP interneuron | Desinibidor — inibe SST e PV | Gating atencional top-down, modulação ACh |
| **DA_N** | Dopaminergic neuron | Pacemaker lento ~4 Hz, AHP prolongado | VTA e SNc — fonte real do sinal dopaminérgico |

> TC e RZ usam `ModeloDinamico::IzhikevichHH` (HhV3 completo com I_T Ca²⁺). Os demais usam `ModeloDinamico::Izhikevich`.

---

## Tipos de Neurônio Ainda Faltando

### Tipos Izhikevich Restantes (~7 tipos)

| Tipo | Comportamento | Importância |
|------|--------------|-------------|
| **Inhibition-Induced Bursting** | Burst por remoção de inibição | Tálamo wake-up |
| **Subthreshold Oscillations** | Oscila sem disparar | Ritmo theta/gamma sub-limiar |
| **Resonator puro (Class 2)** | Responde seletivo a frequência | Seletividade espectral |
| **Integrator (Class 1)** | Acumula até threshold | Decisão lenta |
| **Mixed Mode** | Alterna tônico e burst | Plasticidade dinâmica |
| **Spike Latency** | Atraso proporcional no 1º spike | Codificação temporal |
| **Class 2 Excitability** | Curva f-I com descontinuidade | Bifurcação Hopf |

### Tipos Biológicos Estruturalmente Distintos Faltando

| Tipo | Localização | O que adicionaria |
|------|------------|-------------------|
| **Neurônios serotonérgicos (Raphe)** | Tronco encefálico | Fonte real do 5HT |
| **Neurônios noradrenérgicos (LC)** | Locus Coeruleus | Fonte real do NA/arousal |
| **Neurônios colinérgicos (BF)** | Prosencéfalo basal | Fonte real da ACh/plasticidade |
| **Células granulares (cerebelo)** | Cerebelo | 50% de todos neurônios do cérebro |
| **Células de Purkinje completas** | Cerebelo | Dendrito extenso com 200k sinapses |
| **Células de lugar (Place cells)** | Hipocampo CA1 | Navegação espacial e episódica |
| **Células de grade (Grid cells)** | Entorhinal cortex | Mapa cognitivo do espaço |
| **Neurônios de Von Economo** | ACC, ínsula | Cognição social, autoconsciência |
| **Células estreladas espinhosas** | Córtex sensorial L4 | Input talâmico cortical |
| **Pirâmides L5 (corticospinais)** | Córtex motor | Axônio longo, saída motora |

**Resumo:** Selene implementa 17/~27 tipos funcionais catalogados e ~9/~30 classes biológicas relevantes.

---

## Precisão Mista

| Precisão | Bits | Uso típico | Memória relativa |
|----------|------|------------|-----------------|
| FP32 | 32 | Neurônios críticos de decisão | 100% |
| FP16 | 16 | Camadas intermediárias | 50% |
| INT8 | 8 | Processamento de volume | 25% |
| INT4 | 4 | Compressão máxima | 12.5% |

Distribuição por região otimizada para economizar ~60% de memória vs FP32 puro.

---

## Regiões Cerebrais

### 1. Lobo Frontal (`frontal.rs`)
- **Composição**: RS 60% + IB 20% + FS 20%
- **Função**: Planejamento, tomada de decisão, working memory
- **Conexões**: Cerebelo→PFC (5% saída cerebelar a cada 5 ticks), D1 boost dopaminérgico

### 2. Lobo Parietal (`parietal.rs`)
- **Composição**: RS 70% + CH 30%
- **Função**: Integração sensorial, atenção espacial
- **Conexão**: `attention_weight` modula entrada do graph-walk

### 3. Lobo Temporal (`temporal.rs`)
- **Composição**: RS 50% + CH 30% + FS 20%
- **Função**: Processamento auditivo, semântica, memória de trabalho
- **Conexão**: `apply_rpe()` recebe erro de predição de recompensa

### 4. Lobo Occipital (`occipital.rs`)
- **Composição**: RS 50% + RZ 30% + LT 20%
- **Função**: Processamento visual V1→V2→reconhecimento

### 5. Sistema Límbico (`limbic.rs`)
- **Composição**: RS 40% + FS 40% + IB 20%
- **Função**: Emoção, motivação, valência afetiva, habituação

### 6. Hipocampo (`hippocampus.rs`)
- **Composição**: RS 60% + CH 20% + LT 20%
- **Função**: Consolidação episódica, one-shot learning, motor de hipóteses

### 7. Cerebelo (`cerebellum.rs`)
- **Composição**: RS 70% + FS 20% + LT 10%
- **Função**: Coordenação, predição de erro temporal, projeção cerebelo→PFC

### 8. Corpo Caloso (`corpus_callosum.rs`)
- **Composição**: RS 80% + CH 20%
- **Função**: Transferência inter-hemisférica, latência dinâmica 4–20ms por arousal

### 9. ACC (`cingulate.rs`)
- **conflict_layer**: IB 40% + RS 60% — burst em conflito > dACC
- **regulation_layer**: RS 70% + FS 30% — inibição emocional rACC
- **Saídas**: `noradrenaline_drive()`, `amygdala_inhibition()`, `adjustment_factor`

### 10. OFC (`orbitofrontal.rs`)
- **value_layer**: RS + IB — encoding de valor por contexto (até 512 entradas)
- **extinction_layer**: RS + FS — reversal learning (extinção 3× mais rápida)

### 11. Áreas de Linguagem (`language.rs`)
- **wernicke_layer**: RS 60% + CH 40% — compreensão, familiarity scoring
- **broca_layer**: RS 70% + FS 30% — produção, fluência, syntax_template

### 12. Neurônios Espelho (`mirror_neurons.rs`)
- **Composição**: RS + IB
- **Função**: Reconhecimento de intenção, ressonância empática a cada 50 ticks

### 13. Profundidade de Processamento (`depth_stack.rs`)
- **Função**: Pilha de contexto com múltiplas camadas cognitivas

### 14. Amígdala (`amygdala.rs`)
- **BLA**: Condicionamento, one-shot learning emocional
- **CeA**: Saída de medo, modulação autonômica

---

## Neuroquímica (11 moléculas)

| Neurotransmissor | Função | Range | Dinâmica |
|------------------|--------|-------|----------|
| **Dopamina** | Recompensa, RPE, motivação | 0.0–2.0 | RAM usage → target |
| **Serotonina** | Humor, regulação social | 0.0–1.5 | Jitter + context switches |
| **Noradrenalina** | Atenção, arousal | 0.0–1.6 | CPU temp → target |
| **Cortisol** | Estresse, threshold Na⁺ | 0.0–1.0 | Delta temp |
| **Acetilcolina** | Aprendizado, atenção, bloqueia I_M | 0.0–1.2 | Arousal − adenosina × 0.3 |
| **Ocitocina** | Vínculo social, trust | 0.0–1.5 | Cresce com RPE > 0 |
| **Histamina** | Arousal, vigília, anti-sono | 0.1–1.2 | Inversamente à adenosina |
| **Adenosina** | Pressão de sono acumulada | 0.0–1.0 | Sobe com carga (jitter + RAM) |
| **Endocanabinoide** | Homeostase sináptica, supressão retrógrada | 0.0–1.0 | Dopamina × 0.4 + cortisol × 0.3 |
| **D1 (receptor)** | Alta dopamina → excitação PFC | 0.0–1.0 | Sigmoide: satura acima dopa ≈ 1.0 |
| **D2 (receptor)** | Baixa dopamina → filtragem estriatal | 0.0–1.0 | Alta afinidade, ativa com dopa baixa |

```
RPE > 0.2   → dopamina↑ → D1↑ → PFC boost + oxytocina↑
RPE < −0.2  → cortisol↑ → ACC.registrar_rejeicao() + social_pain↑
conflito > 0.45 → ACC → noradrenaline_drive() → NA↑ → atenção↑
adenosina alta → histamina↓ + ACh↓ → sonolência
dopamina alta → endocanabinoide↑ → supressão de excitação excessiva
```

---

## Sistema de Templates Cognitivos

Templates são **topologias sinápticas persistentes com slots em branco**. O conteúdo dos slots é efêmero — entra durante o uso, é apagado depois. A estrutura persiste e evolui.

### Ciclo de Vida

| Estado | Validações | Plasticidade | Comportamento |
|--------|-----------|--------------|---------------|
| Nascente | 0–2 | 1.0 | Totalmente maleável |
| Desenvolvendo | 3–19 | 0.7 | Restrições emergindo |
| Consolidado | 20–99 | 0.3 | Estrutura estável, **gera filhos** |
| Automático | ≥100 | 0.1 | Ativa sem esforço |
| Arquivado | força < 0.05 | 0.5 | Dormente, reativa com uso |

### Loop de Treinamento (Completo)

```
reconhecer(tokens_input) → (scaffold, uuid)
         ↓
gerar_resposta_emergente(scaffold)
         ↓
usar(uuid, reply_tokens, validado=true)   ← alimenta histórico de slots
         ↓
restricao_emergente() fica mais precisa   ← ciclo fecha
         ↓
tick_decay() a cada 500 respostas         ← templates inativos decaem
```

### Templates Base (19 carregados automaticamente)

| Domínio | Templates |
|---------|-----------|
| **Linguagem** | `observacao_atributiva`, `relacao_causal`, `associacao_dupla`, `reflexao_expandida`, `pergunta_direta`, `afirmacao_modal`, `negacao_contrastiva` |
| **Causal** | `cadeia_causal`, `condicional_simples`, `condicional_composta` |
| **Lógica** | `se_entao`, `transitividade`, `contraexemplo`, `silogismo` |
| **Matemática** | `lei_produto_linear`, `lei_razao`, `lei_potencia`, `proporcao_direta` |
| **Fala Conversacional** | `saudacao_resposta` |

### Treinamento Offline

```bash
pip install websockets
python treinar_templates.py                    # corpus embutido (110 frases)
python treinar_templates.py meu_corpus.txt     # corpus próprio
python treinar_templates.py --verbose          # ver cada frase e resposta
```

---

## Aprendizado Coerente (CLS)

Selene implementa a teoria CLS (Complementary Learning Systems, McClelland et al. 1995):

| Sistema | Biológico | Selene |
|---------|-----------|--------|
| **Hipocampo** (rápido, episódico) | Aprende em 1 exposição | `memorize_with_connections()` + one-shot |
| **Neocórtex** (lento, estatístico) | Consolida padrões no sono | `PatternEngine` — episódios visuais/auditivos/pensamento |
| **Conexão entre eles** | Replay noturno | REM semântico + replay reverso N3 |

### Fontes de Episódio para PatternEngine

- `FonteEpisodio::Visual` — após processamento occipital (step % 100, emotion > 0.1)
- `FonteEpisodio::Ambiente` — modo ambiente com áudio ativo (emotion > 0.05)
- `FonteEpisodio::Pensamento` — após pensamento espontâneo (emotion > 0.15)
- `FonteEpisodio::Aprendizado` — handler WebSocket `learn`
- `FonteEpisodio::Chat` — handler WebSocket `chat`

### Replay Reverso no REM (N3)

Episódios emocionalmente salientes (emoção > 0.5) são replayed em ordem reversa durante o sono N3, criando arcos causais invertidos (recompensa → causa) — base do aprendizado de causalidade bidirecional (Wilson & McNaughton 1994, Pfeiffer 2020).

---

## Memória e Storage

### Hierarquia L1–L4

```
L1: NeuronioHibrido.historico_spikes         (RAM, ~1ms)
L2: working_memory_trace frontal              (RAM, deque circular)
L3: SwapManager — grafo causal semântico      (NVMe, ~10ms)
    ├── TemplateStore (19 templates base)
    ├── Cache do grafo semântico (grafo_dirty)
    └── Cache de trigramas (trigrama_cache em BrainState)
L4: SurrealDB checkpoint                      (disco, persistência)
```

### SwapManager — Performance

| Estrutura | Cap (LRU) | Custo |
|-----------|-----------|-------|
| `sinapses_conceito` | ≤ 500.000 | Remove 5% mais fracos ao ultrapassar |
| `spike_vocab` (BrainState) | ≤ 50.000 | Remove aleatórios ao ultrapassar |
| `grafo_palavras()` | Cache incremental | Reconstrói só quando `grafo_dirty = true` |
| `trigrama_cache` | Pré-computado | Recalcula só quando `frases_padrao` muda |

---

## Motor de Hipóteses

`HypothesisEngine` implementa Predictive Coding (Friston 2022) — o cérebro minimiza erro de predição:

- `formular()` — antes de gerar resposta: prevê próximas palavras/intenções
- `testar()` — ao receber input do usuário: confronta predição com realidade → RPE episódico
- `observar_comportamento()` — monitora padrões das próprias respostas
- `hipoteses_confiaveis()` (≥10 testes, taxa >65%) → STDP automático no swap
- `gaps_conhecimento()` → injetados no neural_context → perguntas autônomas
- `proximo_topico_previsto()` → push_front no contexto → bias preditivo

---

## Interface WebSocket

### Conexão
```
ws://127.0.0.1:3030/selene
```

Interface desktop: `http://127.0.0.1:3030/`
Interface mobile: `http://127.0.0.1:3030/mobile`

### Mensagens de Entrada

| Tipo | Payload | Descrição |
|------|---------|-----------|
| `chat` | `{"type":"chat","message":"texto"}` | Chat principal |
| `learn` | `{"type":"learn","text":"...","context":"...","valence":0.5}` | Aprendizado direto |
| `learn_frase` | `{"action":"learn_frase","words":["eu","sinto"]}` | Padrão de frase |
| `train_template` | `{"type":"train_template","nome":"...","slots":{...},"validado":true}` | Treino manual de template |
| `reward` | `{"type":"reward","value":0.5}` | Sinal de recompensa |
| `force_sleep` | `{"action":"force_sleep","duration_min":30}` | Força ciclo de sono por N min |
| `toggle_sensor` | `{"action":"toggle_sensor","sensor":"video","active":true}` | Liga/desliga câmera ou microfone |
| `ping` | `{"type":"ping"}` | Heartbeat |

### Mensagens de Saída

| Evento | Descrição |
|--------|-----------|
| `chat_reply` | Resposta emergente com emoção e arousal |
| `neural_status` | Estado completo a cada tick (200Hz) |
| `pensamento_espontaneo` | Pensamento autônomo |
| `curiosidade_espontanea` | Pergunta autônoma por lacuna |
| `sono` / `despertar` | Início e fim do ciclo de sono |
| `template_trained` | Confirmação de treinamento de template |
| `voz_params` | Parâmetros de síntese de voz (formantes) |
| `sensor_ack` | Confirmação de toggle de sensor |

---

## Interface Neural (Desktop/Mobile)

### Desktop (`neural_interface.html`)

Acessar em `http://127.0.0.1:3030/`

| Elemento | Função |
|----------|--------|
| **Grafo Mental** | D3 force-directed — nós = conceitos, arestas = associações. Nós pequenos (~2–4px) para melhor visualização de redes densas |
| **Constelação Neural** | Canvas 2D — 55 neurônios primitivos (35 fonemas + 20 visuais) com pulsos em tempo real |
| **Métricas** | Neuroquímica, ondas cerebrais, espectro visual, personalidade, step |
| **Chat** | Área de diálogo com feedback positivo/negativo por mensagem |
| **Painel de Visão** | Feed ao vivo da webcam (funciona sem WebSocket) + overlay 64×32px do que a Selene processa |
| **Botão SONO 30MIN** | Força ciclo de consolidação N1–N4 por 30 minutos; despertar automático ao fim |
| **Botão CÂMERA** | Liga webcam sem precisar estar conectado |
| **Modo Ambiente** | Selene ouve tudo e foca quando relevante (score ≥ 0.40) |

### Mobile (`selene_mobile_ui.html`)

Acessar em `http://127.0.0.1:3030/mobile`

---

## Como Compilar e Rodar

### Pré-requisitos
- Rust 1.75+ (`rustup update stable`)
- Cargo

### Compilar e Rodar
```bash
cd F:/Selene_brain_2.0
cargo run --release
```

### Compilar com GPU (wgpu)
```bash
cargo build --release --features gpu
```

### Abrir Interface
```
http://127.0.0.1:3030/
```

### Treinar Templates (requer Selene rodando)
```bash
pip install websockets
python treinar_templates.py
python treinar_templates.py meu_corpus.txt --verbose
```

### Testes e Benchmarks
```bash
cargo test --lib
cargo test --lib templates
cargo run --bin intensive_benchmark --release
cargo run --bin system_test --release
```

---

## Estrutura de Arquivos

```
Selene_Brain_2.0/
├── src/
│   ├── main.rs                          Loop principal (~200Hz adaptivo)
│   ├── neurochem.rs                     Neuroquímica (11 moléculas)
│   ├── config.rs                        Configuração global
│   ├── sleep_cycle.rs                   Ciclo de sono N1–N4 + replay reverso
│   ├── synaptic_core.rs                 NeuronioHibrido (7 camadas), 17 tipos
│   ├── brain_zones/
│   │   ├── frontal.rs                   Working memory, decisão
│   │   ├── parietal.rs                  Atenção, integração sensorial
│   │   ├── temporal.rs                  Auditivo, semântica
│   │   ├── occipital.rs                 Visual V1→V2→reconhecimento
│   │   ├── limbic.rs                    Emoção, habituação
│   │   ├── hippocampus.rs               Episódico, one-shot
│   │   ├── cerebellum.rs                Erro temporal, cerebelo→PFC
│   │   ├── corpus_callosum.rs           Inter-hemisférico
│   │   ├── cingulate.rs                 ACC — conflito, dor social
│   │   ├── orbitofrontal.rs             OFC — valor, reversal
│   │   ├── language.rs                  Broca + Wernicke
│   │   ├── mirror_neurons.rs            Empatia, intenção
│   │   ├── depth_stack.rs               Profundidade cognitiva
│   │   └── amygdala.rs                  BLA + CeA — medo, condicionamento
│   ├── learning/
│   │   ├── templates.rs                 19 templates base, ciclo de vida completo
│   │   ├── pattern_engine.rs            PatternEngine (neocórtex CLS)
│   │   ├── hypothesis.rs                HypothesisEngine (predictive coding)
│   │   ├── pensamento.rs                Pensamento emergente autônomo
│   │   ├── narrativa.rs                 Estado emocional → vocabulário
│   │   ├── chunking.rs                  ChunkingEngine (threshold=5)
│   │   ├── binding.rs                   Binding temporal gamma
│   │   ├── rl.rs                        Reinforcement learning (Q-table)
│   │   ├── attention.rs                 Atenção seletiva
│   │   └── curriculo.rs                 Currículo fonético PT-BR
│   ├── storage/
│   │   ├── swap_manager.rs              Grafo causal + TemplateStore + LRU (W_MAX=2.5)
│   │   ├── memory_graph.rs              Grafo sináptico persistente
│   │   ├── checkpoint.rs                Checkpoints periódicos
│   │   └── episodic.rs                  Memória episódica
│   ├── sensors/
│   │   ├── audio.rs                     Processamento de áudio
│   │   ├── camera.rs                    Visão
│   │   └── hardware.rs                  Sensores de hardware
│   ├── encoding/
│   │   ├── spike_codec.rs               Codec de spikes
│   │   ├── phoneme.rs                   Codificação fonética PT-BR
│   │   └── helix_store.rs               Armazenamento helix
│   ├── interoception/                   Estados internos homeostáticos
│   ├── gpu/                             Feature "gpu" (wgpu 0.19)
│   └── websocket/
│       ├── server.rs                    Handler WebSocket, chat, templates, sono
│       └── bridge.rs                    BrainState, trigrama_cache, LRU
├── treinar_templates.py                 Script de treinamento offline
├── neural_interface.html                Interface desktop
├── selene_mobile_ui.html                Interface mobile
├── Cargo.toml
└── selene_memories.db/                  SurrealDB local
```

---

## Roadmap

### V2.x (implementado)
- [x] 17 tipos neuronais (7 originais + 6 Izhikevich adicionais + 4 subtipos biológicos)
- [x] Precisão mista FP32/FP16/INT8/INT4
- [x] STDP assimétrico (LTP causal + LTD anti-causal)
- [x] Plasticidade homeostática (synaptic scaling)
- [x] Sparse coding L1 (~20% esparsidade)
- [x] 14 regiões cerebrais
- [x] 11 neurotransmissores dinâmicos (+ histamina, adenosina, endocanabinoide)
- [x] STP Tsodyks-Markram calibrado por tipo neuronal
- [x] ACC, OFC, Broca/Wernicke, Amígdala BLA+CeA
- [x] One-shot learning, graph versioning, embeddings 32d
- [x] Tálamo (LGN/MGN/VPM/NRT), Gânglios da Base (D1/D2 Go/NoGo)
- [x] Integração multimodal AV
- [x] Motor de hipóteses preditivo (Friston 2022)
- [x] PatternEngine integrado ao loop neural (CLS neocortical)
- [x] 19 templates cognitivos com loop de treinamento completo
- [x] Replay reverso no REM (Wilson & McNaughton 1994)
- [x] Cache incremental do grafo + trigramas pré-computados
- [x] LRU para sinapses (≤500k) e spike_vocab (≤50k)
- [x] n_neurons dinâmico por RAM disponível
- [x] Script de treinamento de templates (`treinar_templates.py`)
- [x] GPU opcional (wgpu 0.19)
- [x] Botão SONO 30MIN na interface neural com despertar automático
- [x] Webcam funciona sem WebSocket (modo offline)
- [x] W_MAX alinhado entre todos os módulos (2.5)
- [x] Duplicação HH/HhV3 eliminada — apenas HhV3 ativo

### V3 (próximo)
- [ ] Migração completa brain_zones para neurônio V3 (usar novos tipos PV/SST/VIP/DA_N nas regiões)
- [ ] Neurônios serotonérgicos/noradrenérgicos como fonte real de 5HT/NA
- [ ] Tipos Izhikevich restantes: Mixed Mode, Subthreshold Oscillations, Integrator
- [ ] Correção dos doctests desatualizados (rl.rs, hardware.rs)
- [ ] Centralizar W_MAX / PESO_MAX_CONCEITO em `config.rs`
- [ ] Resolver `#[allow(dead_code)]` global — auditar código morto real

---

*Selene Brain V2 — Criado por Rodrigo Luz*
