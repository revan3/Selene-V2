# Selene Brain V2 — Sistema Neural Bio-Inspirado

> Simulacao de cerebro artificial em Rust com neuronios Izhikevich+HH, precisao mista, STDP, 7 tipos neuronais biologicos, visao, audio, sono e linguagem emergente.

---

## Indice

1. [Visao Geral](#visao-geral)
2. [Arquitetura do Sistema](#arquitetura-do-sistema)
3. [Nucleo Neural — synaptic_core](#nucleo-neural--synaptic_core)
4. [Tipos de Neuronio](#tipos-de-neuronio)
5. [Precisao Mista](#precisao-mista)
6. [Regioes Cerebrais](#regioes-cerebrais)
7. [Neuroquimica](#neuroquimica)
8. [Linguagem Emergente](#linguagem-emergente)
9. [Sensores e Percepcao](#sensores-e-percepcao)
10. [Ciclo de Sono](#ciclo-de-sono)
11. [Memoria e Storage](#memoria-e-storage)
12. [Interface WebSocket](#interface-websocket)
13. [Benchmark](#benchmark)
14. [Como Compilar e Rodar](#como-compilar-e-rodar)
15. [Estrutura de Arquivos](#estrutura-de-arquivos)
16. [Historico de Versoes](#historico-de-versoes)
17. [Roadmap](#roadmap)

---

## Visao Geral

Selene e uma simulacao de sistema nervoso artificial que replica aspectos centrais da neurobiologia computacional:

- **Neuronios Izhikevich** com 7 tipos funcionais distintos (RS, IB, CH, FS, LT, TC, RZ)
- **Hodgkin-Huxley** completo para tipos TC e RZ — canais ionicos Na+/K+/leak reais
- **Precisao mista** (FP32/FP16/INT8/INT4) por neuronio — economia de ~60% de memoria
- **STDP bidirecional** com LTP causal e LTD anti-Hebbiano + threshold adaptivo
- **Neuromodulacao HH**: dopamina/serotonina/cortisol modulam condutancias ionicas
- **9 regioes cerebrais** com composicao neuronal especifica por area
- **Neuroquimica dinamica** (serotonina, dopamina, cortisol, noradrenalina) + Plutchik
- **Linguagem emergente** via grafo de associacoes spike-based com bigrams sequenciais
- **Visao**: webcam -> OccipitalLobe V1/V2 -> grafo linguistico
- **Audio**: microfone -> TemporalLobe -> spike_vocab -> aprendizado
- **Ciclo de sono** biologico: N1 -> N2 -> N3/REM -> consolidacao/poda/backup
- **Memorias episodicas** com hipocampo CA1/CA3 e onda theta (~8Hz)
- **Interface WebSocket** em tempo real com TTS e interface visual

```
Webcam/Mic -> Thalamo -> Occipital/Temporal -> Parietal -> Limbico -> Hipocampo -> Frontal
                                                     |
                                          Linguagem Emergente (grafo)
                                                     |
                                          WebSocket -> TTS -> Usuario
```

---

## Arquitetura do Sistema

```
+-------------------------------------------------------------------+
|                        SELENE BRAIN V2                            |
|                                                                   |
|  [Webcam]     [Microfone]    [Hardware Sensor]                    |
|      |              |               |                             |
|  +-----------TALAMO (filtra + roteia)----------------------------+|
|  |                                                               ||
|  |  [Occipital V1/V2]  [Temporal]  [Parietal]                   ||
|  |       |                  |           |                         ||
|  |  [Limbico/Amigdala] [Hipocampo CA1/CA3]  [Frontal PFC]       ||
|  |       |                  |           |                         ||
|  |  [BasalGanglia]  [CorpusCallosum]  [Brainstem]                ||
|  |                                                               ||
|  +---[NeuroChem: dopamina/serotonina/cortisol/noradrenalina]---+ ||
|                                                                   |
|  [Grafo de Linguagem]  [STDP/ChunkingEngine]  [RL/Q-table]       |
|                                                                   |
|  [WebSocket Server] -> [TTS] -> [Interface HTML]                  |
+-------------------------------------------------------------------+
```

### Fluxo Principal (main loop a 200Hz)

```
A. Interoception + Brainstem (adenosina, temperatura)
B. NeuroChem.update() — neuroquimica dinamica
C. rx_vision + rx_audio — sensores
D. Thalamus.relay() + Brainstem.modulate()
E. OccipitalLobe.visual_sweep() — V1/V2
F. ParietalLobe.integrate() — atencao espacial
G. TemporalLobe.process() — reconhecimento
H. ChunkingEngine.registrar_spikes() — emergencia de chunks
I. LimbicSystem.evaluate() — emocao + arousal
J. Plutchik EmotionalState — alegria/medo/tristeza/etc
K. HippocampusV2.memorize_with_connections() — memoria episodica
L. FrontalLobe.decide() — acao executiva
M. BasalGanglia.update_habits() — habitos e gatekeeping
N. CorpusCallosum.sincronizar() — sync hemisferios
O. NeuralEnactiveMemory (feedback de memoria -> imaginacao)
P. WebSocket broadcast — telemetria
```

---

## Nucleo Neural — synaptic_core

O modelo neuronal e hibrido de 4 camadas:

### Camada 1 — Izhikevich (todos os tipos)
```
dv/dt = 0.04v^2 + 5v + 140 - u + I_eff
du/dt = a(bv - u)
```
Captura 20+ padroes de disparo biologicos com custo O(1) por tick.

### Camada 2 — Refratario / LIF
Periodo refratario absoluto de 2ms + hiperpolarizacao pos-spike.

### Camada 3 — Hodgkin-Huxley (TC e RZ apenas)
```
I_Na = g_Na * m^3 * h * (V - E_Na)   <- canal sodio
I_K  = g_K  * n^4  * (V - E_K)       <- canal potassio
I_L  = g_L         * (V - E_L)       <- corrente de vazamento
```
Variaveis de portao m, h, n com cinetica Alpha/Beta completa (Hodgkin-Huxley 1952).

TC: modo burst (sono, h->0) <-> tonico (vigilia, h->1) via inativacao do canal Na+.
RZ (Purkinje): g_Na alto, bursts de alta frequencia, timing preciso.

### Camada 4 — STDP bidirecional
```
trace_pre -> LTP: pre antes de pos -> potencia peso
trace_pos -> LTD: pos sem pre     -> deprime peso
+ Threshold adaptivo (spike-frequency adaptation)
```

### Neuromodulacao HH
```
dopamina  alto -> g_K_mod  baixo -> repolarizacao lenta -> mais disparo
serotonina alto -> g_L_mod  baixo -> menos vazamento     -> mais excitavel
cortisol  alto -> g_Na_mod baixo -> Na+ reduzido        -> limiar mais alto
```

---

## Tipos de Neuronio

| Tipo | Nome | Comportamento | Regiao |
|------|------|---------------|--------|
| RS | Regular Spiking | Disparo regular, ~80% dos neuronios corticais | Frontal, Parietal, Temporal |
| IB | Intrinsic Bursting | Burst inicial + regular; amigdala | Limbico |
| CH | Chattering | Bursts rapidos repetitivos | Occipital V2, Temporal |
| FS | Fast Spiking | Interneuronio GABAergico, sem adaptacao | Frontal (inibicao lateral) |
| LT | Low-Threshold Spiking | Interneuronio de limiar baixo | Parietal, Hipocampo CA1 |
| TC | Thalamo-Cortical | Burst/tonico via inativacao h (HH) | Talamo |
| RZ | Resonator/Purkinje | Burst alta frequencia, timing motor (HH) | Cerebelo, Hipocampo CA3 |

---

## Precisao Mista

Cada neuronio recebe precisao conforme seu papel funcional:

| Precisao | Bytes | Uso |
|----------|-------|-----|
| FP32 | 4 | Neuronios criticos (~5%) |
| FP16 | 2 | Working memory, encoding (~50%) |
| INT8 | 1 | Processamento em massa (~35%) |
| INT4 | 0.5 | Alta densidade, baixa precisao (~10%) |

Economia media: ~60% vs FP32 puro para 1024 neuronios.

---

## Regioes Cerebrais

| Regiao | Arquivo | Composicao | Conectada ao Loop |
|--------|---------|------------|-------------------|
| OccipitalLobe | brain_zones/occipital.rs | 70% RS + 40% CH (V1), 70% CH + 30% RS (V2) | Sim — visual_sweep() |
| ParietalLobe | brain_zones/parietal.rs | RS + 20% LT | Sim — integrate() |
| TemporalLobe | brain_zones/temporal.rs | 70% RS + 30% CH | Sim — process() |
| LimbicSystem | brain_zones/limbic.rs | 50% IB/RS (amigdala) + RS (accumbens) | Sim — evaluate() |
| HippocampusV2 | brain_zones/hippocampus.rs | 80% RS + 20% LT (CA1), 70% RS + 30% RZ (CA3) | Sim — memorize_with_connections() |
| FrontalLobe | brain_zones/frontal.rs | 80% RS + 20% FS (exec), 100% FS (inhib) | Sim — decide() |
| CorpusCallosum | brain_zones/corpus_callosum.rs | Sincronizador de hemisferios | Sim — sincronizar() |
| BasalGanglia | basal_ganglia/mod.rs | Gating + habitos | Sim — update_habits() |
| Cerebellum | brain_zones/cerebellum.rs | 80% RS + 20% RZ (Purkinje), 100% RZ (granular) | Nao — instanciado, nao chamado |

---

## Neuroquimica

```rust
pub struct NeuroChem {
    pub serotonin: f32,      // 0.0..2.0 — estabilidade emocional
    pub dopamine: f32,       // 0.0..2.0 — recompensa, motivacao
    pub cortisol: f32,       // 0.0..1.0 — estresse, limiar alto
    pub noradrenaline: f32,  // 0.0..2.0 — arousal, atencao
}
```

Atualizado a cada tick com base em: jitter de CPU, context switches, RAM usage, temperatura.

**Roda de Plutchik** (EmotionalState): joy, trust, fear, surprise, sadness, disgust, anger, anticipation — derivados das 4 substancias quimicas. Usado para colorir a linguagem emergente.

---

## Linguagem Emergente

O sistema de linguagem roda no WebSocket server (server.rs) em paralelo ao loop neural:

### Componentes
- **spike_vocab**: HashMap<palavra, SpikePattern(512 bits)> — representacao esparsa de palavras
- **grafo**: HashMap<palavra, HashMap<vizinho, peso>> — associacoes aprendidas
- **frases_padrao**: Vec<Vec<String>> — prefixos de frases para inicializacao do walk
- **helix_store**: armazenamento de padroes spike em arquivo .hlx

### Aprendizado
- `audio_learn`: STT -> spike pattern -> spike_vocab + associacoes no grafo
- `visual_learn`: webcam pixels -> OccipitalLobe V1/V2 -> spike pattern -> `visual:{palavra}`
- `learn_frase`: ensina frase completa com bigrams sequenciais (peso 0.90-0.95)
- `associate`: associa duas palavras diretamente com peso customizado
- `train`: STDP sobre o grafo (N epocas) — LTP nas arestas mais ativas

### Geracao de Linguagem
- `gerar_resposta_emergente()`: graph-walk guiado por valencia emocional
- Profundidade do walk varia com estado neuronal: delta=6, theta=9, beta=10, gamma=13 passos
- Bigrams sequenciais dominam o walk (0.90) -> frases ordenadas
- Conectivo emocional inserido em posicao >=60% da frase (so para frases >=7 palavras)

### Fases do Sono Linguistico
- **N1 Consolidacao**: reforca top-30 arestas mais ativas (+0.07)
- **N2 Poda**: remove arestas com peso < 0.12
- **N3/REM**: fechamento transitivo — cria novas associacoes entre palavras com vizinho comum
- **N4 Backup**: serializa grafo completo para selene_linguagem.json

---

## Sensores e Percepcao

### Visao
- `VisualTransducer` (sensors/camera.rs): captura webcam via nokhwa, resampling para n_neurons pontos
- `OccipitalLobe.visual_sweep()`: V1 (deteccao de bordas + contraste) -> V2 (integracao de features)
- Pipeline completo: webcam -> thalamo -> occipital -> parietal -> temporal
- `visual_learn` WebSocket: recebe pixels 32x16 -> `pixels_to_spike_pattern()` -> grafo linguistico

### Audio
- `sensors/audio.rs`: microfone -> FFT -> bandas de frequencia + energia + pitch
- `audio_learn` WebSocket: STT (Web Speech API) + FFT bands -> spike_vocab + grafo
- Modo escuta continua: reconhecimento continuo com auto-restart

### Compartilhamento de Tela
- `getDisplayMedia` no frontend: captura tela/aba + audio (ex: YouTube)
- Frames visuais -> `visual_learn`, audio bands -> `audio_learn`

---

## Ciclo de Sono

Sono baseado em horario: 00:00 - 05:00 a Selene dorme.

```
Fases:
  N1 Consolidacao  40 min  — reforca sinapses ativas
  N2 Poda          30 min  — remove conexoes fracas
  N3 REM          100 min  — criatividade: novas associacoes transitivas
  N2 Poda          30 min
  N3 REM           90 min  — (maior fase REM)
  N1 Consolidacao  20 min
  N4 Backup        10 min  — salva estado completo
```

Desperta com qualquer interacao (chat, audio, video).
Interface exibe overlay de sono com fase atual.

---

## Memoria e Storage

### Hierarquia
```
L1: RAM (vetores neurais ativos, working memory)
L2: NVMe buffer (nvme_buffer.bin — spikes recentes)
L3: SurrealDB / RocksDB (episodios, sinapses de longo prazo)
L4: JSON files (selene_linguagem.json, selene_hippo_ltp.json)
```

### Tipos de Memoria
- **NeuralEnactiveMemory**: episodio sensorial-motor com timestamp e valencia emocional
- **MemoryTier**: gerencia L1/L2/L3 com swap automatico por tempo de acesso
- **SwapManager**: neurogênese dinamica — novos neuronios conforme RAM disponivel
- **HippocampusV2.ltp_matrix**: pesos sinapticos de longo prazo persistidos em JSON
- **HelixStore**: arquivo binario compacto para spike patterns (.hlx)

---

## Interface WebSocket

### Acoes disponiveis (cliente -> servidor)

| Acao | Descricao |
|------|-----------|
| `chat` | Envia mensagem, recebe resposta por linguagem emergente |
| `audio_learn` | Envia FFT bands + transcript para aprendizado auditivo |
| `visual_learn` | Envia pixels 32x16 + transcript para aprendizado visual |
| `learn` | Aprende uma palavra isolada |
| `learn_frase` | Aprende frase com bigrams sequenciais |
| `associate` | Associa duas palavras com peso customizado |
| `train` | Executa N epocas STDP no grafo |
| `curiosidade` | Selene faz uma pergunta sobre lacuna no grafo |
| `diagnostico` | Retorna estat­isticas do grafo e vocabulario |
| `export_linguagem` | Salva estado do grafo em JSON |
| `shutdown` | Encerra o processo com seguranca |

### Eventos (servidor -> cliente)

| Evento | Descricao |
|--------|-----------|
| `chat_reply` | Resposta de linguagem emergente |
| `neural_status` | Telemetria neurologica (spikes, ondas, neuro) |
| `curiosidade` | Selene questiona uma lacuna de conhecimento |
| `sono` | Selene entra em fase de sono (overlay na UI) |
| `despertar` | Selene acorda |

---

## Benchmark

Resultados em hardware: Windows 11, release build.

```
Velocidade Neural (1024 neuronios):
  Ticks/segundo : 19.192
  Tempo/tick    : 52 µs
  Headroom      : 99% (96x mais rapido que 200 Hz)

Escala de Neuronios:
  1.024    ->  20.211 ticks/s  (49 µs/tick)
  4.096    ->   5.102 ticks/s  (196 µs/tick)
  16.384   ->   1.277 ticks/s  (783 µs/tick)
  65.536   ->     320 ticks/s  (3.1 ms/tick)

Testes Unitarios: 25/25 passando
  - encoding/phoneme: 6 testes
  - encoding/spike_codec: 8 testes
  - encoding/helix_store: 5 testes
  - learning/chunking: 6 testes
```

---

## Como Compilar e Rodar

### Pre-requisitos
- Rust 1.75+ (nightly recomendado)
- Python 3.10+ (para scripts de treinamento)
- Windows 10/11 (sensor de hardware usa API Win32)

### Compilar
```bash
cd selene_kernel
cargo build --release
```

### Rodar
```bash
# Inicia o servidor (porta 3030)
./target/release/selene_brain.exe

# Acessa a interface
# Abra neural_interface.html ou selene_mobile_ui.html no browser
```

### Treinamento inicial (sequencia recomendada)
```bash
cd selene_kernel
python scripts/selene_genesis.py       # vocabulario base
python scripts/selene_identidade.py    # identidade da Selene
python scripts/selene_abc.py           # alfabeto e silabas
python scripts/selene_frases_estruturais.py  # 127 frases estruturais + bigrams
```

---

## Estrutura de Arquivos

```
selene_kernel/
├── src/
│   ├── main.rs                     # Loop principal + instanciacao dos lobos
│   ├── synaptic_core.rs            # Neuronio hibrido Izhikevich+HH+STDP
│   ├── neurochem.rs                # Neuroquimica + Plutchik
│   ├── config.rs                   # Modos de operacao e parametros
│   ├── brain_zones/
│   │   ├── occipital.rs            # Visao V1/V2 (CH dominante)
│   │   ├── temporal.rs             # Audio + linguagem
│   │   ├── parietal.rs             # Integracao sensorial + atencao
│   │   ├── frontal.rs              # Decisao executiva + working memory
│   │   ├── limbic.rs               # Amigdala (IB) + accumbens
│   │   ├── hippocampus.rs          # Memoria episodica CA1/CA3 + theta
│   │   ├── cerebellum.rs           # Timing motor (Purkinje + granular)
│   │   └── corpus_callosum.rs      # Sincronizador de hemisferios
│   ├── sensors/
│   │   ├── camera.rs               # VisualTransducer (nokhwa)
│   │   ├── audio.rs                # Microfone + FFT
│   │   └── hardware.rs             # CPU temp, jitter, RAM
│   ├── learning/
│   │   ├── chunking.rs             # Emergencia de chunks via STDP
│   │   └── rl.rs                   # Q-Learning TD-lambda
│   ├── encoding/
│   │   ├── spike_codec.rs          # Codificacao de palavras em spike patterns
│   │   ├── helix_store.rs          # Armazenamento compacto .hlx
│   │   └── phoneme.rs              # Formantes foneticos pt-BR
│   ├── storage/
│   │   ├── episodic.rs             # Memoria episodica
│   │   ├── memory_graph.rs         # Grafo de memoria sinaptica
│   │   ├── memory_tier.rs          # Hierarquia L1-L3
│   │   └── swap_manager.rs        # Neurogênese dinamica
│   ├── websocket/
│   │   ├── server.rs               # Logica principal + linguagem emergente
│   │   └── bridge.rs               # BrainState compartilhado
│   ├── thalamus/mod.rs             # Filtragem e roteamento sensorial
│   ├── basal_ganglia/mod.rs        # Habitos + gating de acoes
│   ├── brainstem/mod.rs            # Modulacao auditiva + adenosina
│   ├── ego/mod.rs                  # Narrativa de self
│   ├── interoception/mod.rs        # Sinais corporais internos
│   ├── meta/consciousness.rs       # Metacognicao
│   └── sleep_cycle.rs              # Ciclo sono biologico
├── scripts/
│   ├── selene_genesis.py           # Vocabulario base
│   ├── selene_identidade.py        # Identidade e relacoes
│   ├── selene_abc.py               # Alfabeto e silabas
│   ├── selene_frases_estruturais.py # 127 frases com bigrams sequenciais
│   └── selene_diagnostico.py       # Diagnostico do grafo
├── neural_interface.html           # Interface principal (desktop)
└── selene_mobile_ui.html           # Interface mobile
```

---

## Historico de Versoes

### v2.5 (atual)
- OccipitalLobe conectado ao pipeline visual completo (webcam -> V1/V2 -> grafo)
- `visual_learn` — pixels da webcam processados pelo OccipitalLobe antes de entrar no grafo
- Ciclo de sono biologico baseado em horario (00:00-05:00) com fases N1/N2/N3/REM/N4
- TTS (Web Speech API) com voz feminina jovem pt-BR
- Modo escuta continua — microfone sempre aberto com auto-restart
- Compartilhamento de tela (getDisplayMedia) — Selene pode "assistir" YouTube
- Frases estruturais: 127 frases com bigrams sequenciais corrigindo ordem das palavras
- Fix: conectivo emocional nao mais inserido na posicao 1 (causava frases embaralhadas)

### v2.4
- DB corrigido, persistencia sinaptica completa
- HelixStore para spike patterns

### v2.3
- Spike storage, sensores controlados, ciclo de sono real

### v2.2
- Modelo HH para TC e RZ, neuromodulacao de condutancias
- ChunkingEngine para emergencia de linguagem

### v2.1
- Modelo Izhikevich completo com 7 tipos
- Precisao mista FP32/FP16/INT8/INT4

---

## Roadmap

### Prioritario
- [ ] Conectar Cerebelo ao loop principal (timing motor, aprendizado por erro)
- [ ] Integrar RL (Q-Learning) ao loop neural — RPE -> dopamina em tempo real
- [ ] Aprendizado visual por associacao: ver objeto + ouvir palavra -> par visual-linguistico
- [ ] Persistencia da Q-table (bincode checkpoint)

### Medio Prazo
- [ ] Neurogenese dinamica via SwapManager (novos neuronios conforme experiencia)
- [ ] Sinapses espinhosas (spine density) — densidade sinaptica por regiao
- [ ] Correntes de calcio (Ca2+) para plasticidade de longo prazo no hipocampo
- [ ] GABA shunting inhibition real (nao so fator de escala)
- [ ] Oscilacoes gamma (30-100Hz) via interneuronios FS acoplados

### Longo Prazo
- [ ] Rede de modo padrao (DMN) — atividade em repouso
- [ ] Teoria da mente basica (modelo de outros agentes)
- [ ] Projecao de futuro (episodic future thinking via hipocampo)
- [ ] Consciencia de fluxo temporal (senso de passagem do tempo)
