# 🧠 Selene Brain V2 — Sistema Neural Bio-Inspirado

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
9. [Modos de Operação](#modos-de-operação)
10. [Como Compilar e Rodar](#como-compilar-e-rodar)
11. [Estrutura de Arquivos](#estrutura-de-arquivos)
12. [Bugs Corrigidos na V2.1](#bugs-corrigidos-na-v21)
13. [Roadmap](#roadmap)

---

## Visão Geral

Selene é uma simulação de sistema nervoso artificial que replica aspectos centrais da neurobiologia computacional:

- **Neurônios Izhikevich** com 7 tipos funcionais distintos (RS, IB, CH, FS, LT, TC, RZ)
- **Precisão mista** (FP32/FP16/INT8/INT4) por neurônio, economia de ~60% de memória
- **STDP** (Spike-Timing Dependent Plasticity) com LTP e LTD anti-Hebbiana
- **9 regiões cerebrais** com composição neuronal específica por área
- **Neuroquímica dinâmica** (serotonina, dopamina, cortisol, noradrenalina)
- **Memória hierárquica** L1→L4 (RAM → NVMe → SurrealDB/RocksDB)
- **Interface WebSocket** em tempo real para monitoramento neural
- **Ciclo de sono/vigília** com consolidação de memórias em REM

```
Sensores → Tálamo → Regiões Cerebrais → Neuroquímica → Memória → WebSocket
```

---

## Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                     SELENE BRAIN V2                          │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │  Câmera  │    │   Mic    │    │   Hardware Sensor    │  │
│  └────┬─────┘    └────┬─────┘    └──────────┬───────────┘  │
│       │               │                      │               │
│  ┌────▼───────────────▼──────────────────────▼───────────┐  │
│  │                    TÁLAMO                             │  │
│  │  (filtra, amplifica e roteia sinais sensoriais)       │  │
│  └────────────────────┬──────────────────────────────────┘  │
│                        │                                      │
│  ┌─────────────────────▼──────────────────────────────────┐ │
│  │                REGIÕES CEREBRAIS                       │ │
│  │                                                        │ │
│  │  Occipital ──► Parietal ──► Temporal ──► Frontal      │ │
│  │      ↓                                     ↑          │ │
│  │   Limbic ◄──────── Hipocampo ──────────────┘          │ │
│  │      ↓                                                 │ │
│  │  Cerebelo    Corpus Callosum    Tronco Encefálico      │ │
│  └────────────────────┬───────────────────────────────────┘ │
│                        │                                      │
│  ┌─────────────────────▼──────────────────────────────────┐ │
│  │            NEUROQUÍMICA + EGO                          │ │
│  │  Serotonina · Dopamina · Cortisol · Noradrenalina      │ │
│  └────────────────────┬───────────────────────────────────┘ │
│                        │                                      │
│  ┌─────────────────────▼──────────────────────────────────┐ │
│  │              MEMÓRIA HIERÁRQUICA                       │ │
│  │  L1: RAM (ativo) → L2: NVMe → L3: RocksDB → L4: Cold  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Núcleo Neural — synaptic_core

`src/synaptic_core.rs` é o coração do projeto. Toda atividade neural passa por aqui.

### Modelo Izhikevich

O modelo escolhido combina realismo biológico com eficiência computacional:

```
dv/dt = 0.04v² + 5v + 140 − u + I
du/dt = a(bv − u)
if v ≥ 30mV → v = c, u += d   (reset após spike)
```

Onde `dt` deve estar **sempre em milissegundos**. O código resolve internamente a conversão `dt_segundos → substeps de 1ms`.

### Ciclo de update por neurônio

```
1. Período refratário?  → v = -70mV, decai traços STDP, retorna
2. Adapta input à precisão (INT8/INT4: quantização com escala dinâmica)
3. Divide dt em substeps de ~1ms
4. Para cada substep: integra Izhikevich, verifica spike
5. Se spiked: aplica LTP/LTD, atualiza threshold adaptivo, seta refr_count
6. Decai traços STDP com τ = 20ms
7. Retorna threshold ao padrão gradualmente
```

### Plasticidade STDP

```
LTP (potenciação):  spike pós quando trace_pre > 0.1 → +0.012 × trace_pre
LTD (depressão):   spike pós sem trace_pre        → −0.006 × (1 − trace_pre)
Decaimento traço:  trace *= exp(−dt_ms / 20ms)     → meia-vida ~14ms
```

---

## Tipos de Neurônio

A Selene implementa os 7 tipos funcionais do modelo Izhikevich 2003:

| Tipo | a     | b    | c     | d    | Uso biológico | Regiões da Selene |
|------|-------|------|-------|------|---------------|-------------------|
| **RS** — Regular Spiking   | 0.02 | 0.20 | -65 | 8.0 | Córtex piramidal (80% dos neurônios) | Todas |
| **IB** — Intrinsic Bursting | 0.02 | 0.20 | -55 | 4.0 | Camada 5, resposta de medo em burst | Amígdala (50%) |
| **CH** — Chattering         | 0.02 | 0.20 | -50 | 2.0 | Visual V2/V3, padrões rápidos | Occipital V2 (70%), Temporal (30%) |
| **FS** — Fast Spiking       | 0.10 | 0.20 | -65 | 2.0 | Interneurônios GABAérgicos inibitórios | Frontal (20%), Camada Inibitória (100%) |
| **LT** — Low-Threshold      | 0.02 | 0.25 | -65 | 2.0 | Interneurônios de threshold baixo | Parietal (20%), CA1 (20%) |
| **TC** — Thalamo-Cortical   | 0.02 | 0.25 | -65 | 0.05| Tálamo — dois modos (tônico/burst) | Tálamo |
| **RZ** — Resonator          | 0.10 | 0.26 | -65 | 2.0 | Giro dentado, detecção rítmica | Cerebelo granular (100%), CA3 (30%) |

### Como funcionam na prática

**RS (Regular Spiking)** — dispara regularmente sob corrente constante. Taxa proporcional à intensidade. Neurônio "padrão" do córtex.

**IB (Intrinsic Bursting)** — dispara em burst intenso inicial, depois regular. Na amígdala, isso significa que eventos emocionais disparam uma salva intensa de spikes antes de se estabilizar — biologicamente correto.

**CH (Chattering)** — bursts rápidos repetitivos. Ideal para reconhecimento visual: responde a bordas, faces e padrões com alta frequência de disparo.

**FS (Fast Spiking)** — dispara muito rápido sem adaptação. Os interneurônios FS do Frontal implementam inibição lateral real: quando muitos executivos disparam juntos, os FS são ativados e reduzem o potencial dos vizinhos — prevenindo epilepsia cortical.

**TC (Thalamo-Cortical)** — em modo tônico (acordado): dispara regularmente para retransmitir sinais. Em modo burst (sono/desatenção): silêncio seguido de burst.

---

## Precisão Mista

Cada neurônio tem um nível de precisão para seu peso sináptico:

| Precisão | Bytes/peso | Range | Quando usar |
|----------|-----------|-------|-------------|
| **FP32** | 4 bytes | ±3.4×10³⁸ | Neurônios críticos (decisão, emoção) — 5% |
| **FP16** | 2 bytes | ±65504 | Working memory, reconhecimento — 35% |
| **INT8** | 1 byte  | depende da escala | Processamento em massa — 50% |
| **INT4** | 0.5 bytes* | -8 a +7 × escala | Background, reservatório — 10% |

\* INT4 empacota dois valores em um byte (`Int4Par(u8)` com nibble alto e baixo).

### Escala dinâmica para INT8/INT4

**Correção crítica:** a escala para quantização é calculada como `I_max / 127.0` para INT8 e `I_max / 7.0` para INT4, com base na corrente máxima esperada para cada região — não mais fixada em 0.125 (que causava overflow de 97.7% para correntes típicas de 38pA).

Exemplos por região:
- Frontal: `escala = 50.0 / 127.0 ≈ 0.394`
- Amígdala: `escala = 35.0 / 127.0 ≈ 0.276`
- Cerebelo granular: `escala = 20.0 / 127.0 ≈ 0.157`

---

## Regiões Cerebrais

### Fluxo principal no loop neural

```
retina → Tálamo → Occipital (V1→V2) → Parietal → Temporal → Frontal
                                                ↓
cochlea → Brainstem → Límbico (Amígdala + Accumbens)
                              ↓
                    Hipocampo (CA1→CA3) → Memória
                              ↓
                    Cerebelo (Granular→Purkinje) → Motor
```

### Composição por região

| Região | Tipos neuronais | Foco |
|--------|----------------|------|
| **Occipital V1** | 60% RS + 40% CH | Detecção de bordas, contraste, movimento |
| **Occipital V2** | 70% CH + 30% RS | Integração de características visuais |
| **Parietal** | 80% RS + 20% LT | Atenção espacial, integração multissensorial |
| **Temporal** | 70% RS + 30% CH | Reconhecimento auditivo, semântica |
| **Frontal exec.** | 80% RS + 20% FS | Decisão, working memory, planejamento |
| **Frontal inhib.** | 100% FS | Inibição lateral, controle de ganho |
| **Amígdala** | 50% IB + 50% RS | Resposta de medo em burst, emoção |
| **Accumbens** | 100% RS | Recompensa, prazer, motivação |
| **CA1 (Hipocampo)** | 80% RS + 20% LT | Encoding de memórias episódicas |
| **CA3 (Hipocampo)** | 70% RS + 30% RZ | Recorrência com ondas theta |
| **Purkinje (Cerebelo)** | 80% RS + 20% RZ | Controle motor, output inibitório |
| **Granular (Cerebelo)** | 100% RZ | Detecção de padrões rítmicos temporais |

---

## Neuroquímica

O sistema neuroquímico modula a dinâmica de todas as regiões:

| Neurotransmissor | Fonte de input | Efeito |
|-----------------|----------------|--------|
| **Serotonina** | Jitter de CPU + context switches | Estabilidade emocional, humor |
| **Dopamina** | Uso de RAM | Motivação, ganho frontal |
| **Cortisol** | Delta de temperatura | Stress, consolidação de medo |
| **Noradrenalina** | Temperatura da CPU | Alerta, atenção |

### Ciclo neuroquímico

```rust
// Serotonina: penalidade por jitter e context switches
let target_sero = 1.0 - 0.5 * (jitter_penalty + switches_penalty);
self.serotonin += (target_sero - self.serotonin) * decay_rate;

// Dopamina: proporcional ao uso de RAM (recurso de processamento)
let target_dopa = ram_usage / i_max_por_modo;
self.dopamine += (target_dopa - self.dopamine) * decay_rate;

// Cortisol: resposta a picos de temperatura (stress térmico)
self.cortisol = (delta_temp / 5.0).clamp(0.0, 1.0);
```

---

## Memória e Storage

### Hierarquia de 4 camadas

```
L1: RAM  (ativo)    → neurônios ativos, acesso < 1μs
L2: NVMe (buffer)   → neurônios recentes, acesso < 1ms
L3: RocksDB         → memória de longo prazo, acesso < 10ms
L4: SurrealDB (cold)→ arquivo histórico, acesso < 100ms
```

### Tipos de memória

- **NeuralEnactiveMemory**: snapshot emocional (padrão visual + auditivo + estado emocional)
- **ConexaoSinaptica**: peso + contexto de criação, consolidada pelo hipocampo
- **Memória autobiográfica**: registrada pelo Ego quando emoção > 0.8

### Ciclo de sono

Durante o sono, o `CicloSono` executa:
1. Transfere neurônios de L1 → L2 (liberando RAM)
2. Consolida ConexaoSinaptica de CA3 para RocksDB
3. Executa fase REM: reativa padrões de alta emoção para reforço
4. Poda conexões fracas (peso < 0.1 após 24h sem uso)

---

## Modos de Operação

| Modo | Hz | dt | Energia | Uso |
|------|----|----|---------|-----|
| **Humano** | 100 Hz | 10ms | 15W | Eficiência máxima, uso geral |
| **Boost200** | 200 Hz | 5ms | 25W | Equilíbrio — modo padrão recomendado |
| **Boost800** | 800 Hz | 1.25ms | 45W | Performance elevada |
| **Ultra** | 3200 Hz | 0.31ms | 80W | Alta performance, GPU recomendada |
| **Insano** | 6400 Hz | 0.16ms | 120W | Máximo, requer GPU dedicada |

> **Nota sobre dt:** independente do modo, o núcleo sempre usa substeps de ~1ms internamente para garantir estabilidade do modelo Izhikevich. O dt externo apenas controla a frequência do loop principal.

---

## Como Compilar e Rodar

### Pré-requisitos

- Rust 1.75+ (`rustup update stable`)
- Windows 10/11 (para `timeBeginPeriod` de alta resolução)
- 8GB RAM mínimo (16GB recomendado para Boost800+)
- Opcional: câmera USB e microfone para input real

### Compilação

```bash
# Debug (mais lento, logs completos)
cargo build

# Release (otimizado)
cargo build --release

# Com verificação do sistema
cargo build --features test-bin
cargo run --bin check_selene --features test-bin
```

### Executar

```bash
# Modo padrão (Boost200)
cargo run --release

# O modo é configurado em main.rs:
# let config = Config::new(ModoOperacao::Boost200);
```

### Interface WebSocket

Com o sistema rodando, acesse `http://localhost:3030` no navegador para abrir a interface neural.

O WebSocket é exposto em `ws://localhost:3030/selene`.

Formato da mensagem enviada pelo Rust (`NeuralStatus`):
```json
{
  "neurotransmissores": {
    "dopamina": 0.72,
    "serotonina": 0.85,
    "noradrenalina": 0.60
  },
  "hardware": {
    "cpu_temp": 62.5,
    "ram_usage_gb": 8.3
  },
  "ego": {
    "pensamentos": ["Processando estímulo visual..."],
    "sentimento_atual": 0.15
  },
  "atividade": {
    "step": 12450,
    "alerta": 0.95,
    "emocao": -0.03
  },
  "swap": {
    "neuronios_ativos": 5000,
    "capacidade_max": 1000000
  }
}
```

---

## Estrutura de Arquivos

```
src/
├── main.rs                    # Loop neural principal + integração
├── config.rs                  # Modos de operação e parâmetros globais
├── synaptic_core.rs           # ⭐ Núcleo: Izhikevich, STDP, precisão mista
├── neurochem.rs               # Dinâmica de neurotransmissores
├── ego.rs                     # Identidade, estado interno, autobiografia
├── thalamus.rs                # Filtragem e roteamento sensorial
├── brainstem.rs               # Tronco encefálico, arousal, adenosina
├── interoception.rs           # Propriocepção, fadiga, fome de CPU
│
├── brain_zones/
│   ├── mod.rs                 # Re-exportações e RegionType
│   ├── frontal.rs             # Córtex pré-frontal: decisão + FS inibitório
│   ├── occipital.rs           # Córtex visual: V1 (RS+CH) + V2 (CH)
│   ├── parietal.rs            # Integração espacial: RS + LT
│   ├── temporal.rs            # Reconhecimento: RS + CH
│   ├── limbic.rs              # Emoção: IB (amígdala) + RS (accumbens)
│   ├── hippocampus.rs         # Memória: CA1 (RS+LT) + CA3 (RS+RZ)
│   ├── cerebellum.rs          # Motor: Purkinje (RS+RZ) + granular (RZ)
│   └── corpus_callosum.rs     # Comunicação inter-hemisférica
│
├── sensors/
│   ├── camera.rs              # Câmera → array de luminância para Occipital
│   ├── audio.rs               # Microfone → FFT 32 bandas para Temporal
│   └── hardware.rs            # CPU temp, RAM, jitter para NeuroChem
│
├── storage/
│   ├── mod.rs                 # BrainStorage, NeuralEnactiveMemory
│   ├── memory_tier.rs         # Hierarquia L1→L4
│   ├── checkpoint.rs          # Serialização de estado neural
│   ├── episodic.rs            # Memória episódica
│   └── swap_manager.rs        # Neurogênese: RAM ↔ NVMe
│
├── compressor/
│   └── salient.rs             # Compressor de spikes (75% redução)
│
├── learning/
│   └── rl.rs                  # Q-Learning com TD(λ) para reforço
│
├── io/
│   └── pipeline.rs            # Canal de eventos: câmera/áudio/texto/motor
│
├── websocket/
│   ├── mod.rs
│   ├── bridge.rs              # BrainState → JSON
│   ├── server.rs              # Servidor warp WebSocket
│   └── converter.rs           # Conversão de tipos
│
├── sleep_manager.rs           # Gerenciamento de sono e consolidação
├── sleep_cycle.rs             # Fases: N1/N2/N3/REM
├── basal_ganglia.rs           # Hábitos e seleção de ação
├── pid_controller.rs          # Controle de homeostase
│
├── meta/
│   └── consciousness.rs       # Metacognição (stub)
│
├── gpu/
│   ├── mod.rs                 # Backend GPU (stub)
│   └── wgpu.rs                # wgpu para aceleração (stub)
│
├── drivers/
│   └── opencl_gen.rs          # OpenCL (stub)
│
├── telemetry/
│   ├── mod.rs
│   └── broadcaster.rs         # BrainSnapshot para telemetria
│
└── bin/
    └── check_selene.rs        # Diagnóstico do sistema
```

---

## Bugs Corrigidos na V2.2

### 🔴 Erros de compilação (impediam `cargo build`)

**1. `tipo_para_regiao` sem chave de fechamento**
- `arquivar_para_hdd` acidentalmente colocado dentro da função livre — movido para `impl SwapManager`

**2. Import `surrealdb::engine::local::Mem` inválido**
- Feature `kv-mem` não habilitada no Cargo.toml — import removido

**3. `tx` e `brain_state` movidos para closure antes do `tokio::spawn`**
- Clones antecipados antes do closure `ws_route` resolvem o `E0382`

**4. `limbic` sem `mut`**
- `let limbic` → `let mut limbic` em main.rs

**5. `panic_info.location()` com lifetime inválido (`E0521`)**
- Substituído por `.map(|l| (l.file().to_string(), l.line()))` para converter antes de escapar o closure

**6. `ModeloDinamico` e `NeuronioHibrido` sem `Serialize/Deserialize`**
- Necessário para `arquivar_para_hdd` serializar neurônios com `serde_json`

---

## Bugs Corrigidos na V2.1

### 🔴 Críticos (impediam qualquer disparo)

**1. dt em unidades erradas**
- **Antes:** `self.v += dt * (...)` onde `dt = 0.005` segundos
- **Depois:** Conversão interna para ms + substeps de ~1ms
- **Impacto:** Neurônios passaram de 0 disparos para ~79 Hz (I=38pA, Boost200)

**2. Período refratário de 2000ms**
- **Antes:** `refr_count = (2.0 / dt_segundos)` → 400 steps → 2000ms
- **Depois:** `refr_count = (2.0ms / dt_interno_ms)` → ~2 steps → 2ms
- **Impacto:** Neurônio ficava travado por 667× mais tempo que o biológico

**3. STDP com meia-vida de 28 segundos**
- **Antes:** `trace *= exp(-dt_s / 20.0)` → meia-vida 28 segundos
- **Depois:** `trace *= exp(-dt_ms / 20.0)` → meia-vida 14ms
- **Impacto:** Plasticidade sináptica completamente ineficaz antes da correção

### 🟠 Altos

**4. INT4 com overflow de sinal**
- **Antes:** Scale fixo em 0.125 → I=38pA quantizado para 0.875pA (perde 97.7%)
- **Depois:** Scale dinâmica = `I_max / 7.0` por região
- **Impacto:** 10% dos neurônios voltam a receber sinal correto

### 🟡 Médios

**5. UUID por neurônio (16 bytes)**
- **Antes:** `id: Uuid` em cada neurônio
- **Depois:** `id: u32` — suficiente para identificação dentro da camada
- **Economia:** ~36KB para 2304 neurônios

**6. Parâmetros a,b,c,d por instância**
- **Antes:** 4 × f32 = 16 bytes armazenados em cada neurônio
- **Depois:** `TipoNeuronal::parametros()` retorna constantes por tipo
- **Economia:** ~36KB para 2304 neurônios

**7. `escala_compartilhada` declarada mas nunca usada**
- **Antes:** declarada em `CamadaHibrida` mas ignorada no `update()`
- **Depois:** `escala_camada` é passada a cada `neuronio.update()`

**8. Tipo neuronal único (todos RS)**
- **Antes:** todos os neurônios com a=0.02, b=0.2, c=-65, d=8
- **Depois:** 7 tipos com parâmetros e comportamentos distintos por região

---

## Roadmap

### Sprint atual (v2.2) ✅
- [x] Correção do dt (substeps de 1ms)
- [x] Período refratário correto
- [x] STDP com tau=20ms
- [x] Escala dinâmica para INT4/INT8
- [x] 7 tipos neuronais (RS, IB, CH, FS, LT, TC, RZ)
- [x] ID compacto u32 em vez de UUID
- [x] LTD anti-Hebbiana no STDP
- [x] Threshold adaptivo
- [x] 6 erros de compilação corrigidos (E0308, E0382, E0432, E0521, E0596, E0277)
- [x] `NeuronioHibrido` + `ModeloDinamico` com Serialize/Deserialize
- [x] Interface WebSocket corrigida (URL, parsing NeuralStatus, handlers deduplicados)
- [x] `crate-type = ["cdylib", "rlib"]` e `nokhwa` com features corretas

### Sprint 2 (v2.3)
- [ ] `src/gpu/wgpu.rs` — aceleração GPU para camadas densas
- [ ] `src/meta/consciousness.rs` — metacognição e auto-monitoramento
- [ ] `src/pid_controller.rs` — homeostase de temperatura neural
- [ ] Temperatura real de CPU (WMI Windows / sysfs Linux)

### Sprint 2 (v2.3)
- [ ] Onset detection no `audio.rs` para Temporal
- [ ] Dashboard WebSocket com gráficos de spike em tempo real
- [ ] Vocabulário com reforço via `learning/rl.rs`
- [ ] TTS (Text-To-Speech) no `actuators/speech.rs`

### Sprint 4 (v3.0)
- [ ] Port para Linux (remover dependências Win32)
- [ ] Binding PyO3 funcional para experimentos Python
- [ ] Neurogênese dinâmica controlada por `swap_manager.rs`
- [ ] Multi-instância: Selene conversando com outra Selene

---

## Referências

- Izhikevich, E.M. (2003). *Simple Model of Spiking Neurons*. IEEE Transactions on Neural Networks, 14(6), 1569-1572.
- Izhikevich, E.M. (2007). *Dynamical Systems in Neuroscience*. MIT Press.
- Bi, G.Q. & Poo, M.M. (1998). *Synaptic modifications in cultured hippocampal neurons*. Journal of Neuroscience.
- Dayan, P. & Abbott, L.F. (2001). *Theoretical Neuroscience*. MIT Press.

---

*Selene V2 — Sistema Neural Bio-Inspirado em Rust*  
*Licença: proprietária — todos os direitos reservados*
