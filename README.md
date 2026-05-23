# Selene Brain V4.1 — Resiliência + DSU sobre V4 Multicompartimental

> **Simulação de cérebro artificial em Rust com neurônio V4 multicompartimental (5 compartimentos + metabolismo ATP + [K⁺]o dinâmico + acoplamento ephaptic), 17 tipos Izhikevich, pool neural 4096-bloco FP4–FP32, codificação localista, STDP 3-fatores, 14 regiões cerebrais, 11 neurotransmissores dinâmicos, processamento interno 100% em frequência/u32 (sem texto no núcleo neural), motor de hipóteses preditivo, watchdog + invariants do loop 200Hz, e estrutura Union-Find (DSU) no `ChunkingEngine` e no diagnóstico de grafo do `SwapManager`.**

---

## Índice

1. [Visão Geral](#visão-geral)
2. [Estado Atual — V4.1](#estado-atual--v41)
3. [V4.1 — Resiliência + DSU](#v41--resiliência--dsu)
4. [V4 — Neurônio Híbrido Multicompartimental](#v4--neurônio-híbrido-multicompartimental)
4. [Migração Texto→Frequência (Sprints 1–4)](#migração-textofrequência-sprints-14)
5. [V3.5 — Melhorias Biológicas](#v35--melhorias-biológicas)
6. [Arquitetura do Sistema](#arquitetura-do-sistema)
7. [Pool Neural & Codificação Localista](#pool-neural--codificação-localista)
8. [Núcleo Neural — synaptic_core](#núcleo-neural--synaptic_core)
9. [Tipos de Neurônio (17 implementados)](#tipos-de-neurônio-17-implementados)
10. [Precisão Mista & Metaplasticidade](#precisão-mista--metaplasticidade)
11. [Regiões Cerebrais (14)](#regiões-cerebrais)
12. [Neuroquímica (11 moléculas)](#neuroquímica-11-moléculas)
13. [Sistema de Templates Cognitivos](#sistema-de-templates-cognitivos)
14. [Aprendizado Coerente (CLS)](#aprendizado-coerente-cls)
15. [Memória e Storage](#memória-e-storage)
16. [Motor de Hipóteses](#motor-de-hipóteses)
17. [Interface WebSocket](#interface-websocket)
18. [Como Compilar e Rodar](#como-compilar-e-rodar)
19. [Estrutura de Arquivos](#estrutura-de-arquivos)
20. [Roadmap](#roadmap)

---

## Visão Geral

Selene é uma simulação de sistema nervoso artificial que replica aspectos centrais da neurobiologia computacional moderna:

- **Neurônio V4 Multicompartimental**: AIS + Soma + Tronco Apical + Tufo Apical + Extracelular — 5 compartimentos acoplados via teoria de cabo de Rall (1967)
- **Metabolismo ATP real**: bomba Na⁺K⁺-ATPase eletrogenica, [K⁺]o dinâmico via Nernst, penalidade por depleção de ATP
- **Acoplamento ephaptic**: campo extracelular bidirecional modula timing de spike (Anastassiou 2011)
- **Coincidência dendrítica (BAC firing)**: BAP retrógrado + NMDA spike apical → burst amplificado (Larkum 1999)
- **17 tipos neuronais** (7 Izhikevich originais + 6 adicionais + 4 subtipos biológicos)
- **Processamento 100% audio/frequência**: texto entra via TTS→FFT→spike_vocab; internamente tudo é u32/SpikePattern
- **BDNF** como mediador early→late LTP, **BCM rule** dinâmica (theta_m por neurônio), **oxitocina→BLA gate**
- **11 neurotransmissores dinâmicos**: dopamina, serotonina, noradrenalina, cortisol, acetilcolina, oxitocina, histamina, adenosina, endocanabinoide, D1, D2
- **Memória hierárquica** L1→L4 (RAM → NVMe → SwapManager → SurrealDB)
- **Interface WebSocket** em `ws://127.0.0.1:3030/selene`
- **Motor de hipóteses preditivo** com feedback STDP e aprendizado causal

```
Texto (UI) → TTS→FFT→spike_vocab → u32 concept_ids → núcleo neural → resposta (UI)
Microfone  → FFT coclear → SpikePattern → u32 concept_ids → núcleo neural
```

---

## Estado Atual — V4.1

### Testes

```
Sistema completamente validado:
  - testes_v4 (synaptic_core):  9/9   ✅  (neurônio multicompartimental V4)
  - system_test:                22/22 ✅  (grounding, neurochem, pipeline sensório-motor)
  - test_neuron_v3:             12/12 ✅  (firing rates, I_NaP/M/A/T, AHP, STP, STDP)
  - learning_test:               5/5  ✅  (grafo sináptico, backup, consolidação)
  - stability_test:              5/5  ✅  (gates, Hebbian, stress, determinismo)
  - lib unit tests:            103/103 ✅  (inclui voices.rs Multi-Self — 2 falhas corrigidas)
```

> **Auditoria V4.1 (2026-05-22):** watchdog + invariants no loop 200Hz, DSU
> (Union-Find) no `ChunkingEngine` para clusters O(α(n)) + `chaves_set` O(1),
> diagnostic DSU de componentes conectados no startup do `SwapManager`, I/O
> assíncrono completo no cold storage (`arquivar_para_hdd`/`restaurar_do_hdd`
> 100% `tokio::fs`).
>
> **Auditoria texto→u32 (2026-05-16):** todo caminho de processamento neural —
> hot path 200Hz, ciclos de pensamento (Eternal Hole), consolidação onírica/sono
> N3, `treinar_semantico`, hipóteses e `frontal_goal_words` — opera 100% em
> `u32`/`SpikePattern`. Texto existe apenas na fronteira de display, persistência
> e memória episódica. Helpers canônicos: `word_to_concept_id` (FNV-1a 32-bit) e
> `conectar_conceitos_ids` (sinapse u32-nativa, sem round-trip de texto).

### O que funciona hoje

| Componente | Status |
|---|---|
| **V4: Neurônio 5-compartimentos (AIS+Soma+Trunk+Apical+Extracell.)** | ✅ RS/IB |
| **V4: Metabolismo ATP + bomba Na⁺K⁺-ATPase eletrogenica** | ✅ |
| **V4: [K⁺]o dinâmico + E_K(t) Nernst** | ✅ |
| **V4: Acoplamento ephaptic (CamadaHibrida.ephaptic_pool)** | ✅ |
| **V4: BAC firing — coincidência BAP + NMDA spike apical** | ✅ |
| **V4: Brain states (Vigilia/NremProfundo/Rem)** | ✅ fator_apical por estado |
| **Sprint 1–4: Processamento interno 100% u32/SpikePattern** | ✅ sem texto no núcleo |
| **TTS→FFT→spike_vocab cabeado por palavra no chat handler** | ✅ |
| **Templates, hipóteses, grounding: chaves u32** | ✅ |
| **BDNF early→late LTP (tau=30s)** | ✅ V3.5 |
| **BCM rule dinâmica (theta_m por neurônio, tau=30s)** | ✅ V3.5 |
| **Oxitocina→BLA gate (0.3–1.0)** | ✅ V3.5 |
| **Adenosina→D2 antagonismo (Ferré 2022)** | ✅ V3.5 |
| **WM Capacity Limit 4±1 chunks (Cowan 2001)** | ✅ V3.5 |
| **Episodic Buffer Baddeley 2000** | ✅ V3.5 |
| **Memória Prospectiva + set_intention WS** | ✅ V3.5 |
| Tick neural (~200Hz adaptivo) | ✅ estável |
| STDP 3-fatores (dopamina como gate) | ✅ ativo |
| Plasticidade homeostática (synaptic scaling) | ✅ ativo |
| Chat via WebSocket | ✅ funcionando |
| Ciclo sono N1–N4 com replay reverso | ✅ consolidação + REM causal |
| Multi-Self Kernel: 4 vozes (V3.4) | ✅ |
| Pool neural 4096-bloco FP4–FP32 (V3.2) | ✅ |
| WebSocket heartbeat + Message ID + Thinking event | ✅ V3.2 |
| GPU (wgpu, feature "gpu") | ✅ opcional |
| **Watchdog `AtomicU64` no loop 200Hz** | ✅ V4.1 |
| **Invariants no save cycle (vocab/edges)** | ✅ V4.1 |
| **`ChunkDsu` — Union-Find no `ChunkingEngine`** | ✅ V4.1 |
| **`chaves_set` — `ja_existe()` em O(1)** | ✅ V4.1 |
| **DSU startup diagnostic — componentes conectados** | ✅ V4.1 |
| **`arquivar_para_hdd`/`restaurar_do_hdd` 100% `tokio::fs`** | ✅ V4.1 |
| Fase 3 (brain_zones V3, HNSW, ToM) | ⏳ pendente |

---

## V4.1 — Resiliência + DSU

Auditoria 2026-05-22 adicionou camada de resiliência sobre o V4 e introduziu
Union-Find (DSU / Disjoint Set Union) em dois pontos de impacto real.

### Watchdog do loop 200Hz (`main.rs`)

```rust
static LOOP_HEARTBEAT: AtomicU64 = AtomicU64::new(0);

// No bloco de telemetria (step % 500):
LOOP_HEARTBEAT.store(step, Ordering::Relaxed);

// Task paralela (5s):
let cur = LOOP_HEARTBEAT.load(Ordering::Relaxed);
if cur == last && last > 0 {
    log::error!("[WATCHDOG] loop neural parado há >5s (step={cur})");
}
```

Detecta deadlock, panic não capturado, ou stall por I/O síncrono acidental.
Custo: 1 store atômico a cada 500 steps + 1 load a cada 5s.

### Invariants no save cycle (`main.rs`)

Antes de chamar `ontogeny.tick(vocab_n, edges_n, ...)`:

| Condição (`step > N`) | Mensagem |
|----------------------|---------|
| `vocab_n == 0 && step > 10_000` | `[INVARIANTE] possível lock starvation` |
| `edges_n == 0 && step > 20_000` | `[INVARIANTE] grafo pode estar desconexo` |
| `vocab_n > 0 && edges_n == 0 && step > 15_000` | `[INVARIANTE] sinapses não estão sendo criadas` |

Detecta a exata classe de bug que custou horas de debug no incidente
2026-05-17 ("Neonatal travada para sempre" — `ontogeny.tick(0,0,...)`
zerando métricas quando o swap estava locked).

### `ChunkDsu` — Union-Find em `chunking.rs`

```rust
pub struct ChunkDsu {
    parent: HashMap<usize, usize>,  // path compression
    rank:   HashMap<usize, usize>,  // union by rank
}

// API pública do ChunkingEngine:
pub fn cluster_of(&mut self, neuronio_idx: usize) -> usize;
pub fn n_clusters(&mut self) -> usize;
pub fn mesmo_cluster(&mut self, a: usize, b: usize) -> bool;
```

A cada chunk emergido, todos os neurônios componentes são unidos no DSU.
Query "estes dois neurônios participam de chunks que se sobrepõem
transitivamente?" responde em **O(α(n))** ≈ O(1) amortizado (Ackermann
inversa).

Adicionalmente, `chaves_set: HashSet<String>` substitui a iteração linear
sobre `Vec<Chunk>` em `ja_existe()` — agora O(1).

**Por que NÃO usar DSU para causalidade STDP:** causalidade é direcional
(`t_pre < t_post`). Union-Find é não-direcional — `union(A,B)` apaga a
distinção entre `A→B` e `B→A`. Causalidade direcional permanece no grafo
dirigido `sinapses_conceito: HashMap<(Uuid,Uuid),f32>`. DSU só atua sobre
relações de **co-pertencimento** (clusters), não de causa.

### DSU diagnostic no startup (`swap_manager.rs`)

Em `carregar_estado()`, após reconstruir `sinapses_conceito`:

```
✓ [DSU-STARTUP] grafo íntegro: N componente(s) / M nós / K conceitos
  (ou)
⚠ [DSU-STARTUP] grafo fragmentado: N componentes para M nós — possível corrupção
```

Operação O(V+E) one-shot. Alerta quando `n_componentes > max(n_conceitos/10, 5)`
— detecta corrupção de schema, edição manual do JSON ou save cycle interrompido
no meio.

### Sprint 0: `tokio::fs` no cold storage

`arquivar_para_hdd` e `restaurar_do_hdd` agora são `async fn` com
`tokio::fs::*` — fecha a auditoria I/O assíncrono do BUG 1 original
(sincronia bloqueando o executor Tokio).

### Itens deliberadamente descartados

- **Checksums JSON**: `salvar_estado` já usa rename atômico (`.tmp` → final);
  checksum só protegeria contra corrupção física de disco (raro demais)
- **DSU safe pruning**: nenhuma evidência de fragmentação em produção;
  bridge detection incremental tem complexidade alta sem ganho comprovado

---

## V4 — Neurônio Híbrido Multicompartimental

### Arquitetura

O neurônio V4 implementa teoria de cabo de Rall (1967) com 5 compartimentos acoplados:

```
┌─────────────────────────────────────────────────────────────┐
│                   NeuronioHibrido V4                        │
│                                                             │
│  [Tufo Apical]  ←── G_C_APICAL (0.08) ──→  [Tronco]       │
│      NMDA spike         Larkum 1999         Ca²⁺ hotzone    │
│      nmda_gate          BAC firing          ca_trunk        │
│                                    │                        │
│                            G_C_TRUNK (0.15)                 │
│                                    │                        │
│  [AIS] ←── G_C_AIS (0.25) ──→  [Soma]                      │
│  iniciação                  Izhikevich + HH                  │
│  HH gates m,h,n                    │                        │
│  G_NA=200, G_K=60                  │                        │
│                            [Extracelular]                    │
│                            V_e = κ × ΣI_trans               │
│                            ephaptic_pool (CamadaHibrida)    │
└─────────────────────────────────────────────────────────────┘
```

### Compartimentos (EstadoCompartimentos)

| Campo | Descrição | Referência |
|---|---|---|
| `v_ais, m_ais, h_ais, n_ais` | AIS HH gates — G_Na=200, G_K=60 | Kole & Stuart 2012 |
| `ais_spiked` | AIS dispara ANTES ou com o soma | — |
| `v_trunk, ca_trunk, g_ca_trunk` | Tronco apical — Ca²⁺ hotzone | Larkum 1999 |
| `v_apical, ca_apical, nmda_gate` | Tufo apical — NMDA spike | Schiller 2000 |
| `bap_active, bap_timer_ms` | Back-Propagating AP (6ms duração) | Stuart 1994 |
| `coincidencia_ativa` | BAP + NMDA gate > 0.4 (em Vigília) | Larkum 1999 |

### Metabolismo (EstadoMetabolico)

| Campo | Descrição | Referência |
|---|---|---|
| `atp` | ATP [0.05, 2.5] — cai com spikes | PLOS CompBiol 2020 |
| `na_intra` | [Na⁺]i — sobe com spike, bomba extrui | — |
| `k_o` | [K⁺]o extracelular — clearance glial | Kager 2000 |
| `e_k_dyn` | E_K(t) via Nernst — dinâmico | Hodgkin & Katz 1949 |
| `i_pump` | Corrente eletrogenica da bomba | — |

**Equações principais:**
```
E_K(t) = (RT/F) × ln([K⁺]o / [K⁺]i)   [Nernst; RT/F = 26.7mV a 37°C]
I_pump = ρ × f_ATP × f_Na × f_Ko        [bomba Na/K ATPase]
ATP(t) = ATP + prod × dt - custo × dt   [Michaelis-Menten mitocondrial]
```

### Acoplamento Ephaptic (CamadaHibrida)

```rust
// Anastassiou et al. 2011: V_e_local = κ × ΣI_transmembrana(vizinhos)
let i_trans_approx = n_spikes_prev * 50.0 + (v_avg_prev + 70.0).max(0.0) * 0.3;
self.ephaptic_pool = self.ephaptic_pool * (-dt_ms / TAU_EPH_MS).exp()
                   + KAPPA_EPHAPTIC * i_trans_approx;
// Injeta em cada neurônio: i_eph = 0.12 × (eph_pool - v_soma)
```

### Brain States

| Estado | fator_apical | Coincidência | Biológico |
|---|---|---|---|
| `Vigilia` | 1.0 | ativa (fator ≥ 0.8) | ACh + NE → amplificação apical |
| `NremProfundo` | 0.3 | suprimida | isolamento tufo, downscaling |
| `Rem` | 1.2 | amplificada | theta rhythm, consolidação |

### Testes V4 (9/9 ✅)

| Teste | O que verifica |
|---|---|
| `rs_dispara_com_compartimentos_ativos` | RS com compartimentos=Some dispara |
| `fs_dispara_sem_compartimentos_sem_regressao` | FS com compartimentos=None não regride |
| `atp_cai_apos_burst_e_recupera_em_500ms` | ATP depleta e recupera |
| `ko_sobe_apos_burst_e_volta_ao_repouso` | [K⁺]o sobe e clearance glial funciona |
| `coincidencia_dendritica_produz_boost` | BAP + nmda_gate → coincidencia_ativa |
| `ais_spike_precede_ou_coincide_nunca_depois` | AIS dispara antes ou junto do soma |
| `brain_state_modula_acoplamento_apical` | NREM suprime coincidência; Vigília permite |
| `ek_dinamico_responde_a_ko` | E_K Nernst ~-102mV com [K+]o=3mM |
| `camada_ephaptic_acumula_e_decai` | pool ephaptic acumula e decai a zero |

---

## Migração Texto→Frequência (Sprints 1–4)

O núcleo neural de Selene não processa texto. **Toda entrada passa por TTS→FFT antes de tocar a camada neural.** Internamente, conceitos são IDs numéricos (u32), não Strings.

### Pipeline de Entrada

```
Usuário digita "como você está?" (UI)
         ↓
server.rs: tts_para_bandas(palavra, dopa, sero, nor) → [f32; 32]
         ↓ (por palavra: TTS formantes PT-BR → 32 bandas FFT)
bands_to_spike_pattern(bands) → SpikePattern [u64; 8]
         ↓
spike_vocab: HashMap<u64, SpikePattern> + spike_labels: HashMap<u64, String>
         ↓
word_to_concept_id(palavra) → u32 (FNV-1a 32-bit — estável entre sessões)
         ↓
gerar_resposta_emergente(contexto: &[u32]) → String (só na saída, para display)
```

### O que é u32 (Sprint 1–4 completos)

| Componente | Tipo interno | Referência |
|---|---|---|
| `conceito_para_id` | `HashMap<u32, Vec<Uuid>>` | Sprint 2 |
| `fast_weights`, `embeddings` | `HashMap<u32, _>` | Sprint 2 |
| `grafo_cache` | `HashMap<u32, Vec<(u32, f32)>>` | Sprint 2 |
| `spike_vocab` | `HashMap<u64, SpikePattern>` | Sprint 3 |
| `neural_context` | `VecDeque<u32>` | Sprint 3 |
| `grounding` | `HashMap<u32, f32>` | Sprint 3 |
| `emocao_palavras`, `palavra_qvalores` | `HashMap<u32, f32>` | Sprint 3 |
| `aresta_contagem` | `HashMap<(u32,u32), u32>` | Sprint 3 |
| `Hipotese::premissas`, `conclusao` | `Vec<u32>`, `u32` | Sprint 3d |
| `Slot::historico`, `conteudo_atual` | `Vec<(u32,f32)>`, `Option<u32>` | Sprint 4b |
| `Dominio::Composto` | `Vec<u32>` | Sprint 4b |
| `frontal_goal_words` | `Vec<u32>` | Auditoria 2026-05-16 |
| Pensamento (Eternal Hole: consciente/inconsciente/curiosidade) | walk em `grafo_conceitos()` u32 | Auditoria 2026-05-16 |
| `treinar_semantico(valencias)` | `&HashMap<u32, f32>` | Auditoria 2026-05-16 |
| Consolidação onírica/sono N3 (`rem_semantico`) | walk + atalhos + hipóteses u32 | Auditoria 2026-05-16 |

### O que permanece em String (por design — display/saída)

- `reply`, `ultimo_reply` (UI output)
- `id_to_word: HashMap<u32, String>` (reverse lookup para display)
- `conversa_ctx: Vec<String>` (histórico de chat para UI)
- `gerar_resposta_emergente()` — função de OUTPUT (renderiza texto para a UI)
- `EventoEpisodico::palavras: Vec<String>` (memória episódica — registro, não computação)
- Logs `.jsonl`

### Funções canônicas de conversão

```rust
// FNV-1a 32-bit — canônico para String→concept_id (estável entre sessões)
pub fn word_to_concept_id(palavra: &str) -> u32  // em neural_pool.rs

// FNV-1a 64-bit — canônico para chaves de spike_vocab
pub fn spike_label_hash(s: &str) -> u64  // em bridge.rs

// Sinapse u32-nativa — conecta concept_ids existentes sem round-trip de texto
pub fn conectar_conceitos_ids(&mut self, a: u32, b: u32, peso: f32)  // SwapManager
```

---

## V3.5 — Melhorias Biológicas

Implementadas em 2026-05-13:

| Feature | Arquivo | Base científica |
|---|---|---|
| **BDNF** como mediador early→late LTP | `synaptic_core.rs` | Turrigiano 2022 |
| **BCM rule dinâmica** — theta_m por neurônio | `synaptic_core.rs` | BCM 1982 |
| **Adenosina→D2** antagonismo | `neurochem.rs` | Ferré 2022 |
| **Oxitocina→BLA gate** | `neurochem.rs` + `amygdala.rs` | Kirsch 2005 |
| **WM Capacity Limit** 4±1 chunks | `brain_zones/frontal.rs` | Cowan 2001 |
| **Episodic Buffer** Baddeley | `frontal.rs` + `bridge.rs` | Baddeley 2000 |
| **Memória Prospectiva** — fila de intenções | `bridge.rs` + `main.rs` | Pfeiffer 2020 |
| **RegionType enum** completo (14 regiões) | `brain_zones/mod.rs` | — |

### BDNF

Campo `bdnf: f32 [0,2]` em `NeuronioHibrido`. Acumula ∝ delta_ltp, decai com τ=30s.
Modula magnitude do LTP tardio (late LTP):
```rust
let bdnf_gate = 1.0 + self.bdnf * 0.4;
let delta_peso = ltp_magnitude * bdnf_gate * dopamina_gate;
```

### BCM Dinâmica

Campo `theta_m: f32 [0.001, 0.5]` — limiar deslizante por neurônio (não global):
```rust
self.theta_m += (self.activity_avg.powi(2) - self.theta_m) * dt_ms / TAU_BCM_THETA_MS;
```
Alta atividade → theta_m sobe → mais difícil de potenciar (homeostase local).

---

## Arquitetura do Sistema

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        SELENE BRAIN V4.0                                 │
│                                                                          │
│  ┌─────────────────┐   ┌──────────────────┐   ┌───────────────────────┐ │
│  │  WebSocket 2-F  │   │   BrainState     │   │  main.rs 200Hz loop  │ │
│  │  Heartbeat 30s  │◄──│   NeuralPool     │◄──│  try_lock() + async  │ │
│  │  TTS→FFT→spike  │   │   u32 keys       │   │  swap 5000 ticks     │ │
│  └─────────────────┘   └──────────────────┘   └───────────────────────┘ │
│           │                      │                        │              │
│  ┌────────┴──────────────────────┴────────────────────────┴───────────┐  │
│  │              POOL NEURAL — Codificação Localista                   │  │
│  │  NeuralBlock[4096]: FP4→FP32, Concept ID u32, LTP count          │  │
│  └────────────────────────┬───────────────────────────────────────────┘  │
│                           │                                              │
│  ┌────────────────────────┴───────────────────────────────────────────┐  │
│  │              REGIÕES CEREBRAIS (14) — Hierarquia C0–C4            │  │
│  └────────────────────────┬───────────────────────────────────────────┘  │
│                           │                                              │
│  ┌────────────────────────┴───────────────────────────────────────────┐  │
│  │  SYNAPTIC CORE V4 — NeuronioHibrido Multicompartimental           │  │
│  │  AIS │ Soma │ Tronco │ Tufo Apical │ Extracelular                 │  │
│  │  ATP + Na/K ATPase │ [K⁺]o Nernst │ Ephaptic │ BAC               │  │
│  │  Izhikevich │ HhV3 │ STDP-3f │ BDNF │ BCM θ_m                   │  │
│  └────────────────────────┬───────────────────────────────────────────┘  │
│                           │                                              │
│  ┌────────────────────────┴───────────────────────────────────────────┐  │
│  │  APRENDIZADO — SwapManager + HypothesisEngine + TemplateStore     │  │
│  │  u32 keys │ STDP │ Homeostase │ LRU │ spike_vocab HashMap<u64,_> │  │
│  └────────────────────────┬───────────────────────────────────────────┘  │
│                           │                                              │
│  ┌────────────────────────┴───────────────────────────────────────────┐  │
│  │     STORAGE — SurrealDB + Checkpoints + Sleep Cycle               │  │
│  │  L1: RAM │ L2: NVMe │ L3: SurrealDB │ L4: Checkpoint            │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Pool Neural & Codificação Localista

Implementado em `src/neural_pool.rs` (~750 linhas).

### NeuralBlock

```rust
pub struct NeuralBlock {
    raw:            u32,           // valor bruto armazenado
    precision:      PrecisionLevel, // FP4, FP8, FP16, FP32, INT32
    level:          CorticalLevel,  // C0–C4 (sensorial→abstrato)
    ltp_count:      u32,           // eventos de LTP recentes (100ms janela)
    concept_id:     u32,           // ID único do conceito (FNV-1a hash)
    valence:        f32,           // -1.0 (negativo) a +1.0 (positivo)
    last_active_ms: u64,           // timestamp último acesso
    in_use:         bool,
}
```

### Precisão Dinâmica

| PrecisionLevel | Bits | Economia vs FP32 |
|---|---|---|
| **FP4** | 4 bits | 87.5% |
| **FP8** | 8 bits | 75% |
| **FP16** | 16 bits | 50% |
| **FP32** | 32 bits | — |

### Metaplasticidade: Promoção de Precisão

```
FP4 (ltp_count ≥ 8)  → FP8
FP8 (ltp_count ≥ 32) → FP16
FP16 (ltp_count ≥ 128) → FP32
```

Neurônios críticos ganham precisão automaticamente. Inativos degradam para FP4 (87.5% economia).

### Hierarquia Cortical C0–C4

| Nível | Descrição | Teto de Precisão |
|---|---|---|
| **C0 Sensorial** | FFT bruto, pixel RGB | FP4 |
| **C1 Perceptual** | Fonemas, contornos | FP8 |
| **C2 Lexical** | Palavras, concept_id | FP16 |
| **C3 Contextual** | Frases, causalidade | FP32 |
| **C4 Abstrato** | Metacognição, self | FP32 (sempre) |

---

## Núcleo Neural — synaptic_core

Cada neurônio em Selene é um `NeuronioHibrido` com múltiplas camadas biológicas:

| Camada | Mecanismo | Referência |
|---|---|---|
| 1 | Izhikevich (todos os 17 tipos) | Izhikevich 2003 |
| 2 | HH V3 + I_T Ca²⁺ (TC e RZ) | Destexhe 1994 |
| 3 | Canais iônicos: I_NaP, I_M, I_A, I_T, I_BK | Adams 1982 |
| 4 | STP Tsodyks-Markram calibrado por tipo | Markram 1997 |
| 5 | Ca²⁺ dual: AHP (SK) + NMDA (LTP trigger) | — |
| 6 | STDP 3-fatores (dopamina como gate) | Frémaux 2016 |
| 7 | ACh como 4º neuromodulador | — |
| **V4** | **5 compartimentos: AIS+Soma+Trunk+Apical+Extracell.** | **Rall 1967, Larkum 1999** |
| **V4** | **Metabolismo ATP + Na⁺K⁺-ATPase + [K⁺]o Nernst** | **PLOS CompBiol 2020** |
| **V4** | **Ephaptic coupling em CamadaHibrida** | **Anastassiou 2011** |
| **V3.5** | **BDNF early→late LTP (τ=30s)** | **Turrigiano 2022** |
| **V3.5** | **BCM rule — theta_m por neurônio (τ=30s)** | **BCM 1982** |

### STDP Assimétrico

- **LTP**: pré dispara **antes** do pós (Δt < 0) → reforço sináptico (causal)
- **LTD**: pós dispara **antes** do pré (Δt > 0) → enfraquecimento
- `LTD_CONCEITO = LTP_CONCEITO × 0.7` — assimetria biológica
- Janela temporal: ±20ms, τ = 10ms

---

## Tipos de Neurônio (17 implementados)

### Tipos Originais (7)

| Tipo | Nome | Comportamento | HH | Compartimentos V4 |
|---|---|---|---|---|
| **RS** | Regular Spiking | Tônico, adaptação lenta | ❌ | ✅ |
| **IB** | Intrinsic Bursting | Burst inicial + regular | ❌ | ✅ |
| **CH** | Chattering | Bursts rápidos repetitivos | ❌ | ❌ |
| **FS** | Fast Spiking | GABAérgico rápido | ❌ | ❌ |
| **LT** | Low-Threshold | Limiar baixo, suave | ❌ | ❌ |
| **TC** | Thalamo-Cortical | Burst (sono) ↔ tônico (vigília) | ✅ | ❌ |
| **RZ** | Resonator/Purkinje | Ressoa em frequências | ✅ | ❌ |

> RS e IB são os únicos tipos com compartimentos V4 (`Some(EstadoCompartimentos)`) — os demais têm `compartimentos = None` e não regridem.

### Tipos Adicionais (6)

| Tipo | Nome | Papel |
|---|---|---|
| **PS** | Phasic Spiking | Detecção de mudança (onset only) |
| **PB** | Phasic Bursting | Novidade sensorial |
| **AC** | Accommodating | Habituação progressiva |
| **BI** | Bistable | Working memory de curto prazo |
| **DAP** | Depolarizing Afterpotential | Rebound despolarizante |
| **IIS** | Inhibition-Induced Spiking | Desinibição basal |

### Subtipos Biológicos (4)

| Tipo | Nome | Papel |
|---|---|---|
| **PV** | Parvalbumin | Ritmo gamma, inibição perisomal |
| **SST** | Somatostatin | Inibição dendrítica, janela de plasticidade |
| **VIP** | VIP interneuron | Desinibidor, gating atencional |
| **DA_N** | Dopaminergic | VTA/SNc pacemaker ~4 Hz |

---

## Precisão Mista & Metaplasticidade

### Estratégia: Dinâmica por Atividade

- **Repouso**: neurônio silencioso degrada para FP4 (87.5% economia)
- **Atividade leve**: FP8 (perceptual)
- **LTP**: 8+ eventos → FP8; 32+ → FP16; 128+ → FP32
- **Consolidação**: FP32 permanente (LTM)

### Economia Estimada

| Cenário | Bits efetivo | Economia |
|---|---|---|
| 90% FP4 + 10% FP32 | ~6.8 bits | **93%** |
| 20% cada nível | 15 bits | 53% |
| Tudo FP32 | 32 bits | 0% |

---

## Regiões Cerebrais

14 regiões com composição neuronal específica:

| Região | Tipos | Função |
|---|---|---|
| **Frontal** | RS 60%, IB 20%, FS 20% | WM (4±1 chunks), planejamento, episodic buffer |
| **Parietal** | RS 70%, CH 30% | Atenção espacial, integração sensorial |
| **Temporal** | RS 50%, CH 30%, FS 20% | Auditivo, semântica |
| **Occipital** | RS 50%, RZ 30%, LT 20% | Visual V1→V2 |
| **Límbico** | RS 40%, FS 40%, IB 20% | Emoção, valência afetiva |
| **Hipocampo** | RS 60%, CH 20%, LT 20% | Episódico, one-shot |
| **Cerebelo** | RS 70%, FS 20%, LT 10% | Predição de erro, cerebelo→PFC |
| **Corpo Caloso** | RS 80%, CH 20% | Inter-hemisférico, 4–20ms latência |
| **ACC** | IB 40% + RS 60% | Conflito, dor social |
| **OFC** | RS + IB | Valor contextual, reversal learning |
| **Linguagem** | Wernicke RS/CH + Broca RS/FS | Broca+Wernicke — u32 concept_ids |
| **Neurônios Espelho** | RS + IB | Empatia, intenção |
| **DepthStack** | — | Profundidade cognitiva |
| **Amígdala** | BLA + CeA | One-shot emocional, oxitocina gate |

---

## Neuroquímica (11 moléculas)

| Neurotransmissor | Função | Dinâmica |
|---|---|---|
| **Dopamina** | Recompensa, RPE, motivação | RAM usage → target |
| **Serotonina** | Humor, regulação social | Jitter + context switches |
| **Noradrenalina** | Atenção, arousal | CPU temp → target |
| **Cortisol** | Estresse, threshold Na⁺ | Delta temp; suprime oxitocina |
| **Acetilcolina** | Aprendizado, atenção, bloqueia I_M | Arousal − adenosina × 0.3 |
| **Oxitocina** | Vínculo social; inibe BLA (gate 0.3–1.0) | Cresce com RPE > 0 |
| **Histamina** | Arousal, vigília, anti-sono | Inversamente à adenosina |
| **Adenosina** | Pressão de sono; inibe D2 (Ferré 2022) | Sobe com carga |
| **Endocanabinoide** | Homeostase sináptica | Dopamina × 0.4 + cortisol × 0.3 |
| **D1 (receptor)** | Alta dopamina → excitação PFC | Sigmoide acima dopa ≈ 1.0 |
| **D2 (receptor)** | Filtragem estriatal; inibido por adenosina | Alta afinidade |

```
RPE > 0.2   → dopamina↑ → D1↑ → PFC boost + oxitocina↑
RPE < −0.2  → cortisol↑ → ACC.registrar_rejeicao() + social_pain↑
adenosina alta → D2↓ (antagonismo Ferré 2022) + histamina↓ + ACh↓
oxitocina alta → BLA gate [0.3, 1.0] → medo atenuado
```

---

## Sistema de Templates Cognitivos

Templates são topologias sinápticas persistentes com slots em branco (u32 concept_ids).

### Ciclo de Vida

| Estado | Validações | Plasticidade |
|---|---|---|
| Nascente | 0–2 | 1.0 |
| Desenvolvendo | 3–19 | 0.7 |
| Consolidado | 20–99 | 0.3 (gera filhos) |
| Automático | ≥100 | 0.1 |
| Arquivado | força < 0.05 | 0.5 (dormente) |

### Templates Base (19)

| Domínio | Templates |
|---|---|
| **Linguagem** | `observacao_atributiva`, `relacao_causal`, `associacao_dupla`, `reflexao_expandida`, `pergunta_direta`, `afirmacao_modal`, `negacao_contrastiva` |
| **Causal** | `cadeia_causal`, `condicional_simples`, `condicional_composta` |
| **Lógica** | `se_entao`, `transitividade`, `contraexemplo`, `silogismo` |
| **Matemática** | `lei_produto_linear`, `lei_razao`, `lei_potencia`, `proporcao_direta` |
| **Conversacional** | `saudacao_resposta` |

---

## Aprendizado Coerente (CLS)

| Sistema | Biológico | Selene |
|---|---|---|
| **Hipocampo** (rápido) | Aprende em 1 exposição | `memorize_with_connections()` |
| **Neocórtex** (lento) | Consolida no sono | `PatternEngine` |
| **Conexão** | Replay noturno | REM semântico + replay reverso N3 |

---

## Memória e Storage

```
L1: NeuronioHibrido.historico_spikes         (RAM, ~1ms)
L2: working_memory_trace frontal              (RAM, deque 4±1 chunks)
L3: SwapManager — grafo causal u32            (NVMe, ~10ms)
    ├── conceito_para_id: HashMap<u32, Vec<Uuid>>
    ├── spike_vocab: HashMap<u64, SpikePattern>
    ├── TemplateStore (19 templates, slots u32)
    └── id_to_word: HashMap<u32, String>  (reverse lookup display)
L4: SurrealDB checkpoint                      (disco, persistência)
```

### SwapManager — Performance

| Estrutura | Cap (LRU) | Custo |
|---|---|---|
| `sinapses_conceito` | ≤ 500.000 | Remove 5% mais fracos |
| `spike_vocab` | ≤ 50.000 | Remove aleatórios |
| `grafo_cache` | Cache incremental | Reconstrói com `grafo_dirty` |

---

## Motor de Hipóteses

`HypothesisEngine` implementa Predictive Coding (Friston 2022):

- `formular(contexto: &[u32])` — prevê próximas intenções
- `testar(input: &[u32])` — confronta predição → RPE episódico
- `observar_comportamento(premissa: u32, conclusao: u32)` — padrões próprios
- `hipoteses_confiaveis()` (≥10 testes, taxa >65%) → STDP automático
- `gaps_conhecimento() → Vec<u32>` → injetados no neural_context
- `proximo_topico_previsto() → Option<u32>` → bias preditivo

---

## Interface WebSocket

### Conexão

```
ws://127.0.0.1:3030/selene
Interface desktop: http://127.0.0.1:3030/
Interface mobile:  http://127.0.0.1:3030/mobile
```

### Mensagens de Entrada (principais)

| Tipo | Payload | Descrição |
|---|---|---|
| `chat` | `{"type":"chat","message":"texto"}` | TTS→FFT→spike pipeline |
| `audio_raw` | `{"action":"audio_raw","bands":[...32f...]}` | Bandas FFT diretas (mobile) |
| `learn` | `{"type":"learn","bands":[...32f...]}` | Aprendizado via frequência |
| `feedback` | `{"type":"reward","value":0.5}` | 👍/👎 → grounding RPE |
| `force_sleep` | `{"action":"force_sleep","duration_min":30}` | Ciclo de sono forçado |
| `set_intention` | `{"action":"set_intention","concept_ids":[u32,...]}` | Memória prospectiva |
| `set_stage` | `{"action":"set_stage","mode":"Boost200"}` | Modo de operação |

### Mensagens de Saída

| Evento | Descrição |
|---|---|
| `thinking` | UI mostra "Pensando..." (2-fase response) |
| `chat_reply` | Resposta emergente com emoção e arousal |
| `neural_status` | Estado completo a cada tick |
| `pensamento_espontaneo` | Pensamento autônomo |
| `sono` / `despertar` | Ciclo de consolidação |
| `voz_params` | Parâmetros de síntese de voz (Klatt formantes) |

---

## Como Compilar e Rodar

### Pré-requisitos

- Rust 1.75+ (`rustup update stable`)

### Compilar e Rodar

```bash
cd F:/Selene_brain_2.0
cargo run --release
```

### Compilar com GPU (wgpu)

```bash
cargo build --release --features gpu
```

### Testes

```bash
# Suite completa de validação (22 testes)
cargo run --bin system_test --release

# Testes V4 do neurônio multicompartimental
cargo test --release --lib testes_v4

# Todos os testes unitários
cargo test --lib

# 17 tipos neuronais (12 testes)
cargo run --bin test_neuron_v3 --release

# Benchmark de performance
cargo run --bin intensive_benchmark --release
```

### Treinar Templates (requer Selene rodando)

```bash
pip install websockets
python treinar_templates.py
python treinar_templates.py meu_corpus.txt --verbose --epochs 5
```

---

## Estrutura de Arquivos

```
Selene_Brain_2.0/
├── src/
│   ├── main.rs                    Loop 200Hz, save cycle (5000 ticks), hipocampo/frontal/amígdala
│   ├── neural_pool.rs             Pool 4096-bloco, Localist Coding, metaplasticidade, word_to_concept_id
│   ├── neurochem.rs               11 neurotransmissores + oxytocin_bla_gate + adenosina→D2
│   ├── config.rs                  Configuração global
│   ├── sleep_cycle.rs             N1–N4 + replay reverso + reciclagem neural pool
│   ├── synaptic_core.rs           NeuronioHibrido V4 multicompartimental (5 compartimentos)
│   │                              STDP 3-fatores, BDNF, BCM θ_m, ATP metabolismo, ephaptic
│   ├── brain_zones/
│   │   ├── frontal.rs             WM (4±1 chunks Cowan), Episodic Buffer, Goal queue
│   │   ├── parietal.rs            Atenção espacial
│   │   ├── temporal.rs            Auditivo, semântica
│   │   ├── occipital.rs           Visual V1→V2
│   │   ├── limbic.rs              Emoção, habituação
│   │   ├── hippocampus.rs         CA1/CA3, one-shot, LTP persistido
│   │   ├── cerebellum.rs          Erro temporal, cerebelo→PFC
│   │   ├── corpus_callosum.rs     Inter-hemisférico
│   │   ├── cingulate.rs           ACC — conflito, dor social
│   │   ├── orbitofrontal.rs       OFC — valor, reversal
│   │   ├── language.rs            Broca+Wernicke, familiarity_map u32
│   │   ├── mirror_neurons.rs      Empatia, intenção
│   │   ├── depth_stack.rs         Profundidade cognitiva
│   │   └── amygdala.rs            BLA+CeA, oxitocina gate, extinção
│   ├── learning/
│   │   ├── templates.rs           TemplateStore, Slot/Dominio u32, por_dominio HashMap<u32,_>
│   │   ├── hypothesis.rs          HypothesisEngine, premissas/conclusao u32
│   │   ├── pensamento.rs          Eternal Hole — ciclo consciente/inconsciente
│   │   ├── voices.rs              VoiceArbiter — 4 vozes Multi-Self (V3.4)
│   │   ├── ontogeny.rs            DevStage: Neonatal→Discurso
│   │   ├── pattern_engine.rs      PatternEngine (neocórtex CLS)
│   │   ├── chunking.rs            ChunkingEngine, detecção de chunks STDP
│   │   ├── go_nogo.rs             GoNoGoFilter + ForceInterrupt AtomicBool
│   │   ├── active_context.rs      ActiveContext lock-free (Arc, AtomicU64)
│   │   └── rl.rs                  Q-table RL
│   ├── storage/
│   │   ├── swap_manager.rs        Grafo causal u32, spike_vocab u64, LRU, template_scaffold
│   │   ├── reconsolidacao.rs      Janela de labilidade — sono N3
│   │   └── helix_store.rs         HelixStore mmap (spike patterns)
│   ├── sensors/
│   │   ├── audio.rs               FFT coclear → SpikePattern; mic cpal nativo
│   │   └── vision_stream.rs       Visão
│   ├── synthesis/
│   │   ├── formant_synth.rs       Klatt simplificado + vocoder neural
│   │   └── cpal_output.rs         AudioOutput: SyncSender → thread cpal → speaker
│   ├── encoding/
│   │   ├── phoneme.rs             Codificação fonética PT-BR, tts_para_bandas
│   │   └── helix_store.rs         Spike store
│   └── websocket/
│       ├── server.rs              Handlers WS, TTS→FFT→spike por palavra, try_lock()
│       └── bridge.rs              BrainState, spike_vocab u64, neural_context u32, id_to_word
├── treinar_templates.py           Treinamento offline de templates
├── neural_interface.html          Interface desktop
├── selene_mobile_ui.html          Interface mobile
├── Cargo.toml
└── selene_memories.db/            SurrealDB local
```

---

## Roadmap

### Implementado ✅

#### V2.x
- [x] 17 tipos neuronais (7+6+4)
- [x] Precisão mista FP4–FP32
- [x] STDP assimétrico + homeostase
- [x] 14 regiões cerebrais + 11 neurotransmissores
- [x] Motor de hipóteses (Friston 2022)
- [x] PatternEngine CLS + 19 templates cognitivos
- [x] Replay reverso REM (Wilson & McNaughton 1994)
- [x] Tálamo, Gânglios da Base, Amígdala

#### V3.2
- [x] Pool neural 4096-bloco com Localist Coding
- [x] Metaplasticidade: LTP → promoção FP4→FP32
- [x] WebSocket heartbeat 30s + Message ID + Thinking event
- [x] Passive_hear non-blocking (try_lock + dedup FNV-1a)

#### V3.4 — Multi-Self Kernel
- [x] 4 vozes paralelas: Analítica, Censor, Dopamina, Criativa
- [x] Lock-free arbitration (AtomicU32/AtomicBool)
- [x] ForceInterrupt cooperativo
- [x] ACh/STDP/Grounding bugs corrigidos

#### V3.5 — Biologia Avançada
- [x] BDNF early→late LTP (τ=30s)
- [x] BCM rule dinâmica (theta_m por neurônio)
- [x] Adenosina→D2 antagonismo
- [x] Oxitocina→BLA gate
- [x] WM Capacity 4±1 (Cowan 2001)
- [x] Episodic Buffer (Baddeley 2000)
- [x] Memória Prospectiva

#### Sprints 1–4 — Migração 100% Audio/Frequência
- [x] Sprint 1: WebSocket rejeita texto puro; TTS→FFT cabeado
- [x] Sprint 2: SwapManager keys String→u32; I/O async
- [x] Sprint 3: neural_context, grounding, spike_vocab, hypothesis → u32
- [x] Sprint 4a: Chat handler cabeia TTS→FFT→spike_vocab por palavra
- [x] Sprint 4b: Templates completamente u32 (Slot, Dominio, por_dominio)

#### V4.0 — Neurônio Híbrido Multicompartimental
- [x] 5 compartimentos: AIS + Soma + Tronco + Tufo Apical + Extracelular
- [x] Metabolismo ATP real (Michaelis-Menten + bomba Na⁺K⁺-ATPase)
- [x] [K⁺]o dinâmico + E_K(t) via Nernst
- [x] Acoplamento ephaptic bidirecional (CamadaHibrida.ephaptic_pool)
- [x] BAC firing: coincidência BAP + NMDA spike apical (Larkum 1999)
- [x] Brain states: Vigilia/NremProfundo/Rem (fator_apical)
- [x] 9/9 testes V4 passando

#### Auditoria 2026-05-16 — Texto→u32 final + Multi-Self
- [x] Pensamento (Eternal Hole): walk consciente/inconsciente/curiosidade em u32
- [x] `frontal_goal_words: Vec<String>` → `Vec<u32>`
- [x] `treinar_semantico` recebe `&HashMap<u32, f32>` (sem round-trip de texto)
- [x] Consolidação onírica/sono N3: walk + atalhos + hipóteses u32-nativos
- [x] `SwapManager::conectar_conceitos_ids` — sinapse u32 sem reconversão
- [x] voices.rs: 2 falhas corrigidas (AnaliticaVoice structural, DopaminaVoice guard) — 103/103 lib tests

### Pendente ⏳

#### Fase 3
- [ ] Migração brain_zones para composição neuronal V3 (PV/SST/VIP/DA_N por região)
- [ ] HelixStore: busca linear → HNSW quando vocab > 10.000
- [ ] Theory of Mind básico (`src/learning/tom.rs`)
- [ ] Neurônios serotonérgicos (Raphe), noradrenérgicos (LC) como módulos próprios
- [ ] Tipos Izhikevich restantes: Mixed Mode, Subthreshold Oscillations, Integrator

---

*Selene Brain V4.0 — Criado por Rodrigo Luz ("Pai")* — Neurônio Multicompartimental + 100% Audio/Frequência + BDNF/BCM/Oxitocina
