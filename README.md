# Selene Brain 2.0 — Sistema Neural Bio-Inspirado

> Simulacao de cerebro artificial em Rust com neuronios Izhikevich+HH, precisao mista, STDP,
> 7 tipos neuronais biologicos, visao, audio, sono, linguagem emergente e grounding fonetico.

---

## Indice

1. [Visao Geral](#visao-geral)
2. [Arquitetura do Sistema](#arquitetura-do-sistema)
3. [Nucleo Neural](#nucleo-neural--synaptic_core)
4. [Tipos de Neuronio](#tipos-de-neuronio)
5. [Regioes Cerebrais](#regioes-cerebrais)
6. [Neuroquimica](#neuroquimica)
7. [Linguagem Emergente e Grounding](#linguagem-emergente-e-grounding)
8. [Aprendizado Fonetico](#aprendizado-fonetico)
9. [Sensores e Percepcao](#sensores-e-percepcao)
10. [Ciclo de Sono](#ciclo-de-sono)
11. [Memoria e Storage](#memoria-e-storage)
12. [Interface WebSocket](#interface-websocket)
13. [Scripts de Treinamento](#scripts-de-treinamento)
14. [Benchmark](#benchmark)
15. [Como Compilar e Rodar](#como-compilar-e-rodar)
16. [Estrutura de Arquivos](#estrutura-de-arquivos)
17. [Historico de Versoes](#historico-de-versoes)
18. [Roadmap](#roadmap)

---

## Visao Geral

Selene e uma simulacao de sistema nervoso artificial que replica aspectos centrais da neurobiologia computacional:

- **Neuronios Izhikevich V3** com 7 tipos funcionais (RS, IB, CH, FS, LT, TC, RZ)
- **Hodgkin-Huxley** completo para TC e RZ — canais ionicos Na+/K+/leak reais
- **STP (Tsodyks-Markram)** — facilitacao e depressao sinaptica de curto prazo
- **Correntes extras biologicas**: I_T (Ca2+ talamico), g_Nap (persistent Na+), g_M (K+ muscarinico), g_A (K+ transiente), g_BK (K+ ativado por Ca2+)
- **Precisao mista** (FP32/FP16/INT8/INT4) por neuronio — economia de ~60% de memoria
- **STDP bidirecional** com LTP causal, LTD anti-Hebbiano e threshold adaptivo BCM
- **Neuromodulacao**: dopamina/serotonina/cortisol modulam condutancias ionicas
- **9 regioes cerebrais** com composicao neuronal especifica por area
- **Ciclo consciente (Eternal Hole)**: loop interno de pensamento autonomo a 50Hz/10Hz
- **Ego e Introspecção**: narrativa de self escrita a cada ciclo consciente
- **Grounding fonetico**: SpikePattern(audio) ↔ grafema ↔ letras individuais
- **Linguagem emergente** via grafo de associacoes spike-based com bigrams sequenciais
- **Ciclo de sono** biologico: N1 -> N2 -> N3/REM -> consolidacao/poda/backup
- **Interface WebSocket** em tempo real com TTS e grafo visual D3

```
Webcam/Mic -> Thalamo -> Occipital/Temporal -> Parietal -> Limbico -> Hipocampo -> Frontal
                                                     |
                                          Linguagem Emergente (grafo)
                              Grounding Fonetico (som <-> letra <-> spike)
                                                     |
                                     WebSocket -> TTS -> Interface HTML
```

---

## Arquitetura do Sistema

```
+-------------------------------------------------------------------+
|                      SELENE BRAIN 2.0                             |
|                                                                   |
|  [Webcam]     [Microfone]    [Hardware Sensor]                    |
|      |              |               |                             |
|  +-----------TALAMO (filtra + roteia)---------------------------+ |
|  |                                                              | |
|  |  [Occipital V1/V2]  [Temporal]  [Parietal]                  | |
|  |       |                  |           |                       | |
|  |  [Limbico/Amigdala] [Hipocampo CA1/CA3]  [Frontal PFC]      | |
|  |       |                  |           |                       | |
|  |  [BasalGanglia]  [CorpusCallosum]  [Brainstem]               | |
|  |                                                              | |
|  +---[NeuroChem: dopamina/serotonina/cortisol/noradrenalina]--+ | |
|                                                                   |
|  [Eternal Hole: ciclo consciente 50Hz]  [Ego/Introspecção]       |
|  [Grafo de Linguagem]  [ChunkingEngine]  [RL/Q-table]            |
|  [Grounding Fonetico: audio_spike <-> grafema <-> letras]        |
|                                                                   |
|  [WebSocket Server] -> [TTS] -> [Interface HTML + Grafo D3]      |
+-------------------------------------------------------------------+
```

### Fluxo Principal (200Hz)

```
A. Interoception + Brainstem (adenosina, temperatura)
B. NeuroChem.update() — neuroquimica dinamica
C. rx_vision + rx_audio — sensores
D. Thalamus.relay() + Brainstem.modulate()
E. OccipitalLobe.visual_sweep() — V1/V2
F. ParietalLobe.integrate() — atencao espacial
G. TemporalLobe.process() — reconhecimento
H. ChunkingEngine.registrar_spikes() — emergencia de chunks
I. LimbicSystem.evaluate() — emocao + arousal (Plutchik)
J. HippocampusV2.memorize_with_connections() — memoria episodica + theta
K. FrontalLobe.decide() — acao executiva
L. BasalGanglia.update_habits() — habitos e gating
M. CorpusCallosum.sincronizar() — sync hemisferios
N. NeuralEnactiveMemory (feedback memoria -> imaginacao)
O. Eternal Hole tick (pensamento interno autonomo)
P. Ego.registrar() — narrativa de self
Q. WebSocket broadcast — telemetria + linguagem
```

---

## Nucleo Neural — synaptic_core

O neuronio e hibrido V3 com 14 passos de update biologico:

### Modelo base: Izhikevich (todos os tipos)
```
dv/dt = 0.04v^2 + 5v + 140 - u + I_eff
du/dt = a(bv - u)
```
Captura 20+ padroes de disparo biologicos com custo O(1) por tick.

### Hodgkin-Huxley para TC e RZ
```
I_Na = g_Na * m^3 * h * (V - E_Na)
I_K  = g_K  * n^4  * (V - E_K)
I_L  = g_L         * (V - E_L)
I_T  = g_T  * m2h  * (V - E_Ca)    <- canal Ca2+ talamico (burst/tonico)
```

### Correntes extras (V3)
| Corrente | Tipo | Papel |
|----------|------|-------|
| I_Nap | Persistent Na+ | Amplificacao subthreshold |
| I_M | K+ muscarinico | Adaptacao de frequencia (RS) |
| I_A | K+ transiente | Atraso no primeiro spike (FS, LT) |
| I_BK | K+ ativado por Ca2+ | AHP apos spike |
| I_T | Ca2+ talamico | Burst/tonico no TC |

### STDP + STP
```
STDP: trace_pre -> LTP (pre antes de pos)
      trace_pos -> LTD (pos sem pre)
      + threshold BCM adaptivo

STP (Tsodyks-Markram):
  u_fac: facilitacao — uso sinaptico aumenta com atividade
  x_dep: depressao — recursos sinapticos se esgotam
```

### Neuromodulacao
```
dopamina  alta -> g_K_mod  baixo -> mais disparo
serotonina alta -> g_L_mod  baixo -> mais excitavel
cortisol  alto -> g_Na_mod baixo -> limiar mais alto
```

---

## Tipos de Neuronio

| Tipo | Nome | Comportamento | Regiao |
|------|------|---------------|--------|
| RS | Regular Spiking | Disparo regular, ~80% dos neuronios corticais | Frontal, Parietal, Temporal |
| IB | Intrinsic Bursting | Burst inicial + regular | Limbico |
| CH | Chattering | Bursts rapidos repetitivos | Occipital V2, Temporal |
| FS | Fast Spiking | Interneuronio GABAergico, sem adaptacao | Frontal (inibicao lateral) |
| LT | Low-Threshold Spiking | Interneuronio de limiar baixo | Parietal, Hipocampo CA1 |
| TC | Thalamo-Cortical | Burst/tonico via inativacao h (HH) | Talamo |
| RZ | Resonator/Purkinje | Burst alta frequencia, timing motor (HH) | Cerebelo, Hipocampo CA3 |

---

## Regioes Cerebrais

| Regiao | Arquivo | Composicao | Loop |
|--------|---------|------------|------|
| OccipitalLobe | brain_zones/occipital.rs | 70% RS + 40% CH (V1), 70% CH + 30% RS (V2) | visual_sweep() |
| ParietalLobe | brain_zones/parietal.rs | RS + 20% LT | integrate() |
| TemporalLobe | brain_zones/temporal.rs | 70% RS + 30% CH | process() |
| LimbicSystem | brain_zones/limbic.rs | 50% IB/RS (amigdala) + RS (accumbens) | evaluate() |
| HippocampusV2 | brain_zones/hippocampus.rs | 80% RS + 20% LT (CA1), 70% RS + 30% RZ (CA3) | memorize_with_connections() |
| FrontalLobe | brain_zones/frontal.rs | 80% RS + 20% FS (exec), 100% FS (inhib) | decide() |
| CorpusCallosum | brain_zones/corpus_callosum.rs | Sincronizador de hemisferios | sincronizar() |
| BasalGanglia | basal_ganglia/mod.rs | Gating + habitos | update_habits() |
| Cerebellum | brain_zones/cerebellum.rs | 80% RS + 20% RZ (Purkinje) | instanciado (futuro) |

---

## Neuroquimica

```rust
pub struct NeuroChem {
    pub serotonin:    f32,  // 0.0..2.0 — estabilidade emocional
    pub dopamine:     f32,  // 0.0..2.0 — recompensa, motivacao
    pub cortisol:     f32,  // 0.0..1.0 — estresse, limiar alto
    pub noradrenaline: f32, // 0.0..2.0 — arousal, atencao
}
```

Atualizado a cada tick com base em jitter de CPU, context switches, RAM e temperatura.

**Roda de Plutchik** (8 emocoes): joy, trust, fear, surprise, sadness, disgust, anger, anticipation.
Deriva das 4 substancias quimicas e colore a linguagem emergente.

---

## Linguagem Emergente e Grounding

### Componentes principais
- **spike_vocab**: `HashMap<palavra, SpikePattern(512 bits)>` — representacao esparsa
- **grafo**: `HashMap<palavra, Vec<(vizinho, peso)>>` — associacoes aprendidas
- **grounding**: `HashMap<palavra, f32>` — nivel de ancoragem perceptual (0=linguistico, 1=grounded)
- **historico_episodico**: fila de eventos sensoriais com emocao, arousal e padroes de spike

### Mecanismo de grounding semantico
Grounding e a conexao entre linguagem simbolica e percepcao real.
Quando o usuario diz uma palavra enquanto Selene percebe algo (visual ou sonoro),
`grounding_bind()` cria a associacao `spike_pattern ↔ palavra`.

```
Niveis de grounding por fonte:
  visual_learn  -> +0.25 (percepcao visual direta)
  audio_learn   -> +0.15 (percepcao auditiva)
  interoceptivo -> +0.08 (estado interno)
  RPE positivo  -> +0.05 (previsao correta)
  fonetico      -> +0.15 + 0.12 extra (associacao supervisionada som↔letra)
```

### Geracao de linguagem
- `gerar_resposta_emergente()`: graph-walk guiado por valencia emocional
- Profundidade varia com onda dominante: delta=6, theta=9, beta=10, gamma=13 passos
- Bigrams sequenciais (peso 0.90) dominam -> frases com ordem natural
- Perturbacao deterministica por step -> respostas diferentes mesmo para mesmo topico

### Fases do sono linguistico
- **N1**: reforca top-30 arestas mais ativas (+0.07)
- **N2**: remove arestas com peso < 0.12 (poda sinaptica)
- **N3/REM**: fechamento transitivo — cria associacoes entre nos com vizinho comum
- **N4**: serializa grafo completo para `selene_linguagem.json`

---

## Aprendizado Fonetico

O loop de grounding fonetico conecta o espectro fisico do som ao grafema escrito:

```
espeak-ng sintetiza "ba"
  -> WAV -> frames FFT de 25ms -> learn_audio_fft
       └-> Rust: ultimo_padrao_audio = SpikePattern das frequencias

  -> grounding_fonetico {"grafema":"ba", "letras":["b","a"]}
       └-> grounding_bind(["ba","b","a"], audio_spike)
            └-> associacao permanente: aquele padrao de onda = "ba" = "b" + "a"
```

Com repeticao, Selene aprende que certos padroes de frequencia correspondem
a certas letras — mecanismo analogo ao que humanos usam para aprender a ler.

### Curriculo baba_selene.py (8 fases)
| Fase | Conteudo | Exemplos |
|------|----------|---------|
| 1 | Vogais puras | a, e, i, o, u |
| 2 | Vogais nasais | an, en, in, on, un, a~, e~ |
| 3 | Silabas CV | ba, pa, fa, va, da, ta, na, sa |
| 4 | Silabas CVC | bal, par, dor, sol, mar |
| 5 | Pares minimos | pato/bato, calo/galo, faca/vaca |
| 6 | Digrafos e ditongos | lha, nha, cha, ai, ei, oi, au, ao~ |
| 7 | Encontros consonantais | bra, cla, fla, tra, pri, fro |
| 8 | Alfabeto completo | a, be~, ce~, de~, efe, ge~... |

---

## Sensores e Percepcao

### Visao
- `VisualTransducer` (sensors/camera.rs): captura webcam via nokhwa
- `OccipitalLobe.visual_sweep()`: V1 (bordas + contraste) -> V2 (integracao de features)
- `visual_learn` WebSocket: pixels 32x16 -> `pixels_to_spike_pattern()` -> grafo

### Audio
- `sensors/audio.rs`: microfone -> FFT -> bandas + energia + pitch
- `learn_audio_fft` WebSocket: FFT bins -> primitiva de onda -> RocksDB + spike pattern
- `audio_learn` WebSocket: STT + FFT bands -> spike_vocab + grafo

### Hardware
- CPU temp, jitter, RAM usage -> NeuroChem -> modulacao neuroquimica

---

## Ciclo de Sono

Sono baseado em horario: 00:00 - 05:00 a Selene dorme.

```
N1 Consolidacao  40 min  — reforca sinapses ativas
N2 Poda          30 min  — remove conexoes fracas (<0.12)
N3 REM          100 min  — novas associacoes transitivas
N2 Poda          30 min
N3 REM           90 min  — maior fase REM
N1 Consolidacao  20 min
N4 Backup        10 min  — serializa grafo completo
```

Desperta com qualquer interacao (chat, audio, video).

---

## Memoria e Storage

### Hierarquia
```
L1: RAM       — vetores neurais ativos, working memory
L2: NVMe      — spikes recentes (nvme_buffer.bin)
L3: RocksDB   — episodios e primitivas de onda (selene_memories.db)
L4: JSON      — selene_linguagem.json, selene_hippo_ltp.json
```

### Tipos
- **NeuralEnactiveMemory**: episodio sensorial-motor com timestamp e valencia emocional
- **MemoryTier**: gerencia L1/L2/L3 com swap automatico
- **SwapManager**: neurogênese dinamica — novos neuronios conforme RAM disponivel
- **HippocampusV2.ltp_matrix**: pesos sinapticos de longo prazo em JSON
- **HelixStore**: arquivo binario compacto para spike patterns (.hlx)
- **storage/ondas.rs**: primitivas de onda (F0, F1, F2, F3, onset, VOT) no RocksDB

### Ferramentas de manutencao
```bash
# Diagnostico e benchmark do DB
python benchmark_db.py

# Limpeza: remove orphans do grafo, assocs quebradas, LOG.old RocksDB
python limpar_db.py
python limpar_db.py --dry-run        # simula sem alterar
python limpar_db.py --threshold 0.01 # remove links mais fracos
```

---

## Interface WebSocket

### Acoes (cliente -> servidor)

| Acao | Descricao |
|------|-----------|
| `chat` | Envia mensagem, recebe resposta por linguagem emergente |
| `audio_learn` | FFT bands + transcript -> spike_vocab + grafo + grounding |
| `learn_audio_fft` | FFT bins de frame 25ms -> primitiva de onda + spike pattern |
| `grounding_fonetico` | Associa ultimo_padrao_audio ao grafema e suas letras |
| `visual_learn` | Pixels 32x16 + transcript -> aprendizado visual |
| `learn` | Aprende uma palavra isolada |
| `learn_frase` | Frase com bigrams sequenciais |
| `associate` | Associa duas palavras com peso customizado |
| `train` | N epocas STDP no grafo |
| `curiosidade` | Selene faz pergunta sobre lacuna no grafo |
| `diagnostico` | Retorna estatisticas do grafo e vocabulario |
| `export_linguagem` | Salva estado do grafo em JSON |
| `shutdown` | Encerra o processo com seguranca |

### Eventos (servidor -> cliente)

| Evento | Descricao |
|--------|-----------|
| `chat_reply` | Resposta de linguagem emergente |
| `neural_status` | Telemetria neurologica (spikes, ondas, neuro, ego) |
| `audio_ack` | Confirmacao de frame FFT recebido + hash da primitiva |
| `grounding_ack` | Confirmacao de grounding fonetico aplicado |
| `curiosidade` | Selene questiona uma lacuna de conhecimento |
| `sono` | Selene entra em fase de sono |
| `despertar` | Selene acorda |

---

## Scripts de Treinamento

Todos os scripts comunicam com Selene via WebSocket (porta 3030).
**Nenhum envia texto diretamente** — usam FFT fisico ou protocolo linguistico.

| Script | Descricao | Requisitos |
|--------|-----------|-----------|
| `baba_selene.py` | 8 fases foneticas + grounding som↔letras | espeak-ng, scipy, numpy |
| `treinar_fonemas.py` | Curriculo TTS fonetico detalhado | espeak-ng, scipy, numpy |
| `audiolivro_selene.py` | Aprende com WAV/MP3/OGG/FLAC via FFT | scipy, numpy, soundfile |
| `pdf_para_audio_selene.py` | Le PDF via PyMuPDF + espeak-ng -> FFT | pymupdf, scipy, numpy, espeak-ng |
| `benchmark_db.py` | Benchmark DB (leitura, lookup, throughput FFT) | numpy |
| `limpar_db.py` | Compactacao RocksDB + remocao de orphans | — |
| Scripts/ | Genesis, identidade, frases estruturais, diagnostico | websockets |

### Sequencia de treinamento recomendada

```bash
# 1. Vocabulario base e identidade
python Scripts/selene_genesis.py
python Scripts/selene_identidade.py
python Scripts/selene_frases_estruturais.py

# 2. Alfabeto fonetico (requer espeak-ng no PATH)
python baba_selene.py --fase 1   # vogais
python baba_selene.py --fase 2   # nasais
python baba_selene.py --todas    # todas as 8 fases

# 3. Treinamento com material rico
python audiolivro_selene.py livros/aprendizado/LEXICO/TREINO_FINAL_SELENE.txt
python pdf_para_audio_selene.py livros/aprendizado/LINGUISTICA_COGNITIVA/linguistica.pdf

# 4. Manutencao periodica
python limpar_db.py
python benchmark_db.py
```

---

## Benchmark

Resultados em Windows 11, release build, Rust 1.80+.

### Velocidade neural (1024 neuronios)
```
Ticks/segundo : 19.192
Tempo/tick    : 52 µs
Headroom      : 99% (96x mais rapido que 200 Hz)
```

### Escala
```
  1.024    ->  20.211 ticks/s  (49  µs/tick)
  4.096    ->   5.102 ticks/s  (196 µs/tick)
  16.384   ->   1.277 ticks/s  (783 µs/tick)
  65.536   ->     320 ticks/s  (3.1 ms/tick)
```

### Throughput de aprendizado (benchmark_db.py)
```
FFT (25ms frames, 128 bins):
  Processamento : 13.154 frames/s = 329x tempo real
  Serialização  : 0.09 ms/frame   = 11.600 ops/s

Lookup no grafo (dict com indice de adjacencia):
  Nos->vizinhos : 4.454.244 ops/s
  Palavra->peso : 7.953.614 ops/s

RocksDB (pos-limpeza):
  Tamanho       : 3.2 MB (1 SST file)
  Capacidade    : ~1 min audio/2.34 MB
```

### Testes unitarios
```
57 testes passando (cargo test --lib)
  - encoding/phoneme    : 6
  - encoding/spike_codec: 8
  - encoding/helix_store: 5
  - learning/chunking   : 6
  - sensors/audio       : 8
  - storage             : 4
  - ...outros           : 20
```

---

## Como Compilar e Rodar

### Pre-requisitos
- Rust 1.75+ (stable)
- Python 3.10+ com pip
- Windows 10/11 (sensor hardware usa API Win32)
- espeak-ng no PATH (para scripts foneticos)

### Compilar e rodar
```bash
cd selene_kernel

# Compilar
cargo build --release

# Iniciar servidor (porta 3030)
./target/release/selene_brain.exe

# Interface: abrir no browser
# neural_interface.html     <- interface principal + grafo
# selene_mobile_ui.html     <- interface mobile
```

### Dependencias Python
```bash
pip install websockets numpy scipy soundfile pymupdf vosk
# vosk opcional — STT offline para verificacao de sintese
```

### espeak-ng (Windows)
Baixar e instalar: https://github.com/espeak-ng/espeak-ng/releases
Adicionar ao PATH apos instalar.

---

## Estrutura de Arquivos

```
selene_kernel/
├── src/
│   ├── main.rs                      # Loop principal 200Hz
│   ├── lib.rs                       # Modulo raiz
│   ├── synaptic_core.rs             # Neuronio hibrido V3 Izhikevich+HH+STDP+STP
│   ├── neurochem.rs                 # Neuroquimica + Plutchik
│   ├── config.rs                    # Modos de operacao e parametros
│   ├── sleep_cycle.rs               # Ciclo sono biologico
│   ├── sleep_manager.rs             # Gerenciador de sono
│   ├── ego/mod.rs                   # Narrativa de self + introspecção
│   ├── meta/
│   │   └── consciousness.rs         # Metacognicao + Eternal Hole
│   ├── brain_zones/
│   │   ├── occipital.rs             # Visao V1/V2
│   │   ├── temporal.rs              # Audio + linguagem
│   │   ├── parietal.rs              # Atencao espacial
│   │   ├── frontal.rs               # Decisao executiva
│   │   ├── limbic.rs                # Amigdala + accumbens
│   │   ├── hippocampus.rs           # Memoria episodica CA1/CA3
│   │   ├── cerebellum.rs            # Timing motor (Purkinje)
│   │   ├── corpus_callosum.rs       # Sync hemisferios
│   │   ├── mirror_neurons.rs        # Empatia e imitacao
│   │   └── depth_stack.rs           # Pilha de processamento profundo
│   ├── sensors/
│   │   ├── audio.rs                 # Microfone + FFT + WordAccumulator
│   │   ├── camera.rs                # VisualTransducer (nokhwa)
│   │   ├── hardware.rs              # CPU temp, jitter, RAM
│   │   └── sensor_control.rs        # Controle dos sensores
│   ├── learning/
│   │   ├── inter_lobe.rs            # STDP entre lobos
│   │   ├── chunking.rs              # Emergencia de chunks
│   │   ├── rl.rs                    # Q-Learning TD-lambda
│   │   ├── hypothesis.rs            # Motor de hipoteses
│   │   ├── attention.rs             # Atencao seletiva
│   │   ├── binding.rs               # Binding multimodal
│   │   ├── curriculo.rs             # Curriculo de aprendizado
│   │   ├── lobe_router.rs           # Roteamento dinamico
│   │   └── pensamento.rs            # Ciclo de pensamento interno
│   ├── encoding/
│   │   ├── spike_codec.rs           # Codificacao palavra -> SpikePattern 512 bits
│   │   ├── helix_store.rs           # Armazenamento compacto .hlx
│   │   ├── phoneme.rs               # Formantes foneticos pt-BR
│   │   ├── fft_encoder.rs           # FFT -> PrimitivaOnda (F0,F1,F2,F3,onset,VOT)
│   │   ├── fonetico.rs              # Classificacao fonetica
│   │   └── espectro_visual.rs       # Codificacao visual
│   ├── storage/
│   │   ├── mod.rs                   # NeuralEnactiveMemory, exports
│   │   ├── episodic.rs              # Memoria episodica
│   │   ├── memory_graph.rs          # Grafo de memoria sinaptica
│   │   ├── memory_tier.rs           # Hierarquia L1-L3
│   │   ├── swap_manager.rs          # Neurogênese dinamica
│   │   ├── ondas.rs                 # Primitivas de onda no RocksDB
│   │   ├── spike_store.rs           # Armazenamento de spike patterns
│   │   └── tipos.rs                 # Tipos compartilhados de storage
│   ├── websocket/
│   │   ├── server.rs                # Actions + linguagem emergente
│   │   ├── bridge.rs                # BrainState compartilhado + grounding
│   │   ├── converter.rs             # Conversao de mensagens
│   │   └── mod.rs
│   ├── thalamus/mod.rs              # Filtragem e roteamento sensorial
│   ├── basal_ganglia/mod.rs         # Habitos + gating de acoes
│   ├── brainstem/mod.rs             # Modulacao auditiva + adenosina
│   ├── interoception/mod.rs         # Sinais corporais internos
│   ├── compressor/
│   │   └── salient.rs               # Compressao de padroes salientes
│   └── bin/                         # Binarios de teste e benchmark
│       ├── test_neuron_v3.rs
│       ├── benchmark.rs
│       ├── stability_test.rs
│       └── system_test.rs
│
├── Scripts/                         # Scripts de treinamento (protocolo linguistico)
│   ├── selene_genesis.py            # Vocabulario base
│   ├── selene_identidade.py         # Identidade e relacoes
│   ├── selene_abc.py                # Alfabeto e silabas
│   ├── selene_frases_estruturais.py # Frases com bigrams sequenciais
│   └── selene_diagnostico.py        # Diagnostico do grafo
│
├── baba_selene.py                   # Treinamento fonetico 8 fases + grounding
├── treinar_fonemas.py               # Curriculo TTS fonetico
├── audiolivro_selene.py             # Aprende com arquivos de audio
├── pdf_para_audio_selene.py         # Le PDF e envia como audio
├── benchmark_db.py                  # Benchmark do banco de dados
├── limpar_db.py                     # Limpeza e compactacao do DB
│
├── neural_interface.html            # Interface principal + grafo D3
├── selene_mobile_ui.html            # Interface mobile
├── grafo_selene.html                # Visualizacao do grafo de linguagem
├── grafo_selene.json                # Estado do grafo (gerado em runtime)
├── selene_linguagem.json            # Vocabulario e associacoes (gerado)
├── d3.v7.min.js                     # Biblioteca D3 (servida pelo Warp)
│
├── Cargo.toml
└── README.md
```

---

## Historico de Versoes

### v2.8 (atual)
- **Grounding fonetico**: loop completo som -> FFT -> SpikePattern -> grafema -> letras
- **baba_selene.py**: curriculo de 8 fases foneticas do Portugues Brasileiro
- **pdf_para_audio_selene.py**: leitura de PDF via PyMuPDF + espeak-ng -> FFT -> Selene
- **audiolivro_selene.py**: aprende com qualquer arquivo de audio (WAV/MP3/OGG/FLAC)
- **benchmark_db.py + limpar_db.py**: manutencao e metricas do banco de dados
- Remocao do pyo3/cdylib — kernel puro Rust, scripts Python via WebSocket
- Limpeza geral: -102.381 linhas de codigo legado, 57 testes passando
- RocksDB compactado: 3.2 MB (1 SST file), -15.019 associacoes quebradas

### v2.7
- Eternal Hole: ciclo consciente autonomo a 50Hz
- Introspecao: ciclo consciente escreve no ego
- Deteccao de pensamento vs fala

### v2.6
- Audio grounding, WordAccumulator, hypothesis engine
- Neuronio V3 unificado (elimina split V2/V3)
- LobeRouter: roteamento dinamico + especializacao emergente

### v2.5
- OccipitalLobe conectado ao pipeline visual completo
- Ciclo de sono biologico baseado em horario
- TTS com voz feminina jovem pt-BR
- Compartilhamento de tela (getDisplayMedia)

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
- [ ] Cerebelo no loop principal (timing motor, aprendizado por erro)
- [ ] RL em tempo real: RPE -> dopamina -> modula STDP
- [ ] STT offline (vosk) integrado ao loop de audio continuo
- [ ] Grafo de neuronios D3: visualizar disparos em tempo real

### Medio Prazo
- [ ] One-shot learning: fast-weights para aprender com 1 exemplo
- [ ] Graph versioning: proteger bom aprendizado contra sessoes ruins
- [ ] Embeddings vetoriais no vocabulario (palavra -> [f32;128])
- [ ] GABA shunting inhibition real
- [ ] Oscilacoes gamma via interneuronios FS acoplados

### Longo Prazo
- [ ] Rede de modo padrao (DMN) — atividade em repouso
- [ ] Teoria da mente basica (modelo de outros agentes)
- [ ] Projecao de futuro (episodic future thinking via hipocampo)
- [ ] Consciencia de fluxo temporal
