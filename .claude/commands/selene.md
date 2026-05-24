Você está trabalhando no projeto **Selene Brain 2.0** — uma IA com cérebro neural simulado biologicamente em Rust. Leia este contexto antes de qualquer ação.

## Princípios arquiteturais inegociáveis

1. **Spikes > texto**: Selene processa em SpikePattern `[u64;8]` e bandas FFT `[f32;32]`. Texto de entrada SEMPRE passa por `texto_para_bandas_fft()` antes de aprender.
2. **try_lock > lock().await** no loop neural (200Hz). Nunca usar `.lock().await` dentro do loop principal — causa deadlock com o chat handler.
3. **tokio::fs::write** para I/O assíncrono. `std::fs::write` bloqueia o executor Tokio.
4. **Decaimento por `fator_boost()`**: todo `decay_rate` deve ser dividido por `config.fator_boost()` para escalar corretamente com o modo de operação.
5. **cpal::Stream é !Send no Windows (WASAPI)**: nunca colocar `cpal::Stream` em structs que precisam ser `Send`. Usar thread `std` dedicado que owna a stream; comunicar via `SyncSender<Vec<f32>>`.

---

## Estado atual — V4.5 (2026-05-24)

Versões mais recentes:
- **V4.5** — Persistência HIT (engrams + CA3 binário) + Export/Import
  knowledge para clonagem entre agentes + script `clonar_conhecimento.py` +
  Manual completo `MANUAL_ENGRAMS.md`
- **V4.4** — Implante Tonegawa (false memory style): `EngramOrigem` (Organico/
  Implantado/Restaurado), `implantar_conhecimento()`, 3 handlers WS
  (implant_memory/list_implants/purge_implants), script Python com 9 domínios
  pré-prontos (matemática, semântica, oratória, física, química, biologia,
  programação, informática, jogos)
- **V4.3** — Memória episódica avançada: SR (mapa preditivo de Stachenfeld),
  Priority Replay (EVB de Mattar & Daw), Hippocampal Indexing (HIT),
  Memory Engrams (Tonegawa), DG sparse encoder + CA3 attractor (separation/completion)
- **V4.2** — Hardware real (WMI), BG↔RL nigrostriatal, interocepção em multimodal, curriculo PT-BR fases 8-11
- **V4.1** — Resiliência (watchdog/invariants) + DSU em chunking/swap
- **V4.0** — Neurônio Híbrido Multicompartimental (5 comp. + ATP + Nernst + ephaptic)
- Histórico: V3.5 (BDNF/BCM/Adenosina-D2/Oxitocina-BLA/WM+Cowan/Baddeley/Prospectiva),
  V3.4 (Multi-Self), V3.2 (Pool localista)

### Melhorias V4.5 (2026-05-24)

| Feature | Arquivo | Função |
|---------|---------|--------|
| CA3 binário (`salvar_async`/`carregar_async`) | `brain_zones/ca3_attractor.rs` | 12 bytes/sinapse |
| `HippocampalIndex.salvar/carregar_estado` | `brain_zones/hippocampal_index.rs` | Orquestra engrams + CA3 |
| `export_knowledge_json` / `import_knowledge_json` | idem | Portabilidade entre agentes |
| `EngramStore.clone()` / `CA3.clone()` | `*_engrams.rs`, `ca3_attractor.rs` | Necessário p/ snapshot save |
| Hook save em step%5000 | `main.rs` | tokio::spawn — não bloqueia 200Hz |
| Hook load no startup | `main.rs` | `carregar_estado("selene_hit").await` |
| 2 WS handlers: `export_knowledge`, `import_knowledge` | `websocket/server.rs` | — |
| `clonar_conhecimento.py` (CLI clone) | (raiz) | --exportar/--importar/--inspecionar |
| `MANUAL_ENGRAMS.md` (10 seções) | (raiz) | Documentação completa de uso |

### Melhorias V4.4 (2026-05-24)

| Feature | Arquivo | Base científica |
|---------|---------|----------------|
| `EngramOrigem` enum (Organico/Implantado/Restaurado) | `brain_zones/memory_engrams.rs` | — |
| `implantar_conhecimento()` + log warn | `brain_zones/hippocampal_index.rs` | Tonegawa 2013 (false memory) |
| Handler WS `implant_memory` (resolve words→cells, boost STDP) | `websocket/server.rs` | — |
| Handlers WS `list_implants` / `purge_implants` (salvaguardas) | `websocket/server.rs` | — |
| `BrainState.hippocampal_index` | `websocket/bridge.rs` | Teyler & DiScenna 1986 |
| Script Python `implantar_conhecimento.py` (9 domínios, 130+ memórias) | (raiz do projeto) | — |

### Melhorias V4.3 (2026-05-23)

| Feature | Arquivo | Base científica |
|---------|---------|----------------|
| **Successor Representation** (TD-SR sparse) | `learning/successor.rs` | Dayan 1993, Stachenfeld 2017 |
| **Priority Replay** (EVB BinaryHeap) | `learning/priority_replay.rs` | Mattar & Daw 2018 |
| **Memory Engrams** (cell ensembles + idx reverso) | `brain_zones/memory_engrams.rs` | Tonegawa 2012-2016 |
| **Dentate Gyrus** (sparse encoder top-k WTA) | `brain_zones/dentate_gyrus.rs` | Marr 1971 (pattern separation) |
| **CA3 Attractor** (Hopfield Hebbian) | `brain_zones/ca3_attractor.rs` | Hopfield 1982, Rolls 2008 |
| **Hippocampal Index** (HIT orquestrador) | `brain_zones/hippocampal_index.rs` | Teyler & DiScenna 1986, Rudy 2024 |

### Melhorias V4.2 (2026-05-23)

| Feature | Arquivo | Base científica |
|---------|---------|----------------|
| **CPU temp real via WMI** | `sensors/hardware.rs` + Cargo `[target.cfg(windows)]` | — |
| **BG ← RPE (nigrostriatal)** `aplicar_rpe` + `dopamine_mod` | `basal_ganglia/mod.rs` + `main.rs` | Schultz 1998, SNpc→striatum |
| **Interocepção modula binding AV** `engajamento_corporal` | `learning/multimodal.rs` | Critchley 2004, Craig 2009 |
| **Curriculo PT-BR fases 8-11** (CVC, CVCV, CCV, alta freq) | `learning/curriculo.rs` | Bortolini PT-BR fonologia |

### Pendente

- Migração brain_zones para composição neuronal V3 (PV/SST/VIP/DA_N por região)
- HelixStore O(n) → HNSW quando vocab > 10.000
- Núcleos neuromoduladores reais: Raphe (5-HT), LC (NA), VTA (DA) como módulos próprios
- Theory of Mind básico (`src/learning/tom.rs`)
- Whisper ASR para validação de produção (audio.rs TODO)
- Busca externa via WS para `GapConhecimento` (hypothesis.rs)

---

## Mapa de arquivos críticos

| Arquivo | Responsabilidade |
|---|---|
| `src/main.rs` (2302 linhas) | Loop neural 200Hz, inicialização, save cycle (step%5000), hipocampo, frontal, amígdala |
| `src/websocket/server.rs` (4294 linhas) | Handlers WS: chat, audio_raw, learn, feedback, set_intention, force_sleep, start_mic, stop_mic, reset_memory, set_stage, ontogeny_status |
| `src/websocket/bridge.rs` (1252 linhas) | BrainState, EgoVoiceState, startup load/init |
| `src/websocket/mod.rs` | start_websocket_server — roteamento warp |
| `src/neurochem.rs` (237 linhas) | 11 neurotransmissores + cascata biológica + oxytocin_bla_gate() |
| `src/synaptic_core.rs` (2042 linhas) | 17+3 tipos neurais Izhikevich, STP, STDP 3-fatores, BDNF, BCM dinâmica, theta_m |
| `src/storage/swap_manager.rs` (78.7K) | Grafo semântico, STDP, spike vocab, reconsolidacao |
| `src/storage/reconsolidacao.rs` | Janela de labilidade — sono N3 |
| `src/learning/hypothesis.rs` (29.1K) | HypothesisEngine — predições + confiança |
| `src/learning/ontogeny.rs` | DevStage: Neonatal→Discurso — gate verbal |
| `src/learning/voices.rs` (24.7K) | VoiceArbiter — 4 vozes Multi-Self (V3.4) |
| `src/learning/pensamento.rs` (22.8K) | Eternal Hole — ciclo consciente/inconsciente |
| `src/learning/templates.rs` (47.1K) | Motor de templates de linguagem |
| `src/learning/go_nogo.rs` | GoNoGoFilter + ForceInterrupt AtomicBool |
| `src/learning/active_context.rs` | ActiveContext lock-free (Arc, AtomicU64) |
| `src/learning/inter_lobe.rs` | InterlLobeCurrents — comunicação entre regiões |
| `src/learning/chunking.rs` (20.1K) | Detecção de chunks via co-ativação STDP |
| `src/sensors/audio.rs` | FFT coclear → SpikePattern; mic nativo cpal input |
| `src/brain_zones/occipital.rs` | V1→V2 visual → SpikePattern |
| `src/brain_zones/frontal.rs` (18.4K) | PFC: WM (4±1 chunks), Episodic Buffer, Goal queue |
| `src/brain_zones/amygdala.rs` | BLA+CeA: fear signal, oxytocin gate, extinção |
| `src/brain_zones/hippocampus.rs` | CA1/CA3: memorize_with_connections, LTP persistido |
| `src/brain_zones/language.rs` (13.2K) | Áreas de Broca/Wernicke |
| `src/synthesis/formant_synth.rs` | Klatt simplificado + vocoder neural |
| `src/synthesis/cpal_output.rs` | AudioOutput: SyncSender → thread cpal → speaker |

---

## Arquivos de estado persistido

| Arquivo | Conteúdo |
|---|---|
| `selene_ego.json` | `Vec<(String, f32)>` — traços de personalidade |
| `selene_autobiografia.json` | `{sentimento, memorias: [(desc, valência)]}` |
| `selene_hypotheses.json` | HypothesisEngine serializado |
| `selene_linguagem.json` | Vocabulário, grafo, grounding, neural_context |
| `selene_swap_state.json` | Estado completo do swap_manager |
| `selene_qtable.bin` | Q-table do RL |
| `selene_ontogeny.json` | OntogenyState — estágio atual + métricas |
| `selene_hippo_ltp.json` | Pesos LTP do hipocampo (CA1/CA3) |
| `selene_spikes.hlx` | HelixStore — spike vocab mmap |
| `selene_response_log.jsonl` | Log de respostas geradas |

---

## Arquitetura de áudio (nativa)

```
INPUT:  OS mic → cpal (audio.rs) → AudioSignal → rx_audio → main.rs → aprender_conceito()
        Interface: {"action":"start_mic"} / {"action":"stop_mic"}
        Fallback mobile: {"action":"audio_raw", "bands":[...32 floats...]}

OUTPUT: chat handler → lookup audio_frames → sintetizar_neural() OU sintetizar() (Klatt)
                     → AudioOutput.enqueue() → cpal thread → speaker
        Porta mobile: voz_params JSON via WS (síntese no browser)
```

---

## Ontogenia — estágios de desenvolvimento

| Estágio | Saída verbal | Thresholds para avançar |
|---|---|---|
| Neonatal | nenhuma | vocab≥30, edges≥10, 0.5h escuta |
| PreVerbal | só reações emocionais | vocab≥100, edges≥20, reward≥0.05, 2h |
| PalavraUnica | máx 2 palavras | vocab≥300, edges≥100, reward≥0.10, 5h |
| Frase | máx 5 palavras | vocab≥800, edges≥500, reward≥0.15, 15h |
| Discurso | livre | — |

Handlers WS: `reset_memory`, `set_stage`, `ontogeny_status`.

---

## Campos-chave de BrainState (bridge.rs)

| Campo | Tipo | Uso |
|---|---|---|
| `neural_context` | `VecDeque<String>` (máx 20) | Palavras ativas agora — semente do walk |
| `episodic_buffer_words` | `VecDeque<String>` (cap 4) | Buffer episódico Baddeley — boost no walk |
| `prospective_queue` | `VecDeque<(String, u64, f32)>` | Memória prospectiva — intenções agendadas |
| `frontal_goal_words` | `Vec<String>` | Tokens do goal frontal → semente do walk |
| `audio_frames` | `HashMap<String, Vec<[f32;32]>>` | Memória acústica por palavra |
| `audio_output` | `Option<Arc<AudioOutput>>` | Saída nativa; None em headless |
| `ontogeny` | `OntogenyState` | Gate verbal; salvo em selene_ontogeny.json |
| `convergencia_multimodal` | `ConvergenciaMultimodal` | Predição AV cruzada |
| `ultimo_audio_hash` | `u64` | Dedup FNV-1a (janela 300ms) |
| `hypothesis_engine` | `HypothesisEngine` | Predições + RPE semântico |
| `go_nogo` | `Arc<GoNoGoFilter>` | Filtro executivo + ForceInterrupt |
| `active_context` | `Arc<ActiveContext>` | Contexto lock-free entre vozes (V3.4) |
| `pattern_engine` | `PatternEngine` | Episódica → extração → consolidação N3 |
| `oxytocin_level` | `f32` | Nível atual — afeta gate da amígdala |
| `amygdala_fear` | `f32` | Fear signal BLA → cautela na resposta |
| `acc_conflict` | `f32` | Conflito cingulado → walk mais cauteloso |
| `ofc_value_bias` | `f32` | Valência contextual OFC [-1, 1] |
| `neural_pool` | `NeuralPool` | 4096 blocos u32 — localist coding V3.2 |

---

## NeuronioHibrido — campos V3.5

Campos novos adicionados em V3.5 (além dos existentes):

| Campo | Tipo | Descrição |
|---|---|---|
| `input_apical` | `f32` | Entrada BAC compartimento apical (RS) |
| `bdnf` | `f32 [0,2]` | BDNF — mediador early→late LTP, τ=30s |
| `theta_m` | `f32 [0.001,0.5]` | Limiar BCM por neurônio — desliza com activity² (τ=30s) |

---

## Neuroquímica — 11 moduladores (neurochem.rs)

| Campo | Biológico | Efeito principal |
|---|---|---|
| `serotonin` | 5-HT | Decay WM, mood, ritmo |
| `dopamine` | DA | LTP gate, reward, WM encoding |
| `cortisol` | HPA | Estresse, suprime oxitocina |
| `noradrenaline` | NA/NE | Arousal, atenção, histamina↑ |
| `acetylcholine` | ACh | Codificação hipocampal, LTP boost |
| `oxytocin` | OXT | Inibe BLA (gate 0.3–1.0), confiança social |
| `histamine` | HA | Vigília, anti-adenosina |
| `adenosine` | ADO | Pressão de sono, inibe D2 (Ferré 2022) |
| `endocannabinoid` | eCB | Homeostase sináptica, suprime excesso |
| `d1_signal` | D1R | PFC excitação — WM e planejamento |
| `d2_signal` | D2R | Estriado filtro — inibido por adenosina |

---

## Convenções de código

- Novos handlers WS: `server.rs` no bloco `match action.as_deref()`
- Novos neurotransmissores: campo em `NeuroChem` + `update()` com cascata + `new()` com baseline
- Novo tipo neuronal: enum `TipoNeuronal` + `impl TipoNeuronalV3` (g_nap/g_m/etc) + `SinapseSTP::para_tipo()`
- Save cycle: coletar dados **dentro** do lock → escrever **fora** com `tokio::fs::write`
- Audio nativo: todo PCM passa por `AudioOutput.enqueue(pcm, SAMPLE_RATE)` — nunca cpal direto
- Episodic buffer: alimentado por `frontal.push_episodio(palavra, spike, arousal)` em main.rs quando arousal > 0.4
- Memória prospectiva: `brain_state.agendar_intencao(texto, step, delay_ticks, prio)` — verificada a cada 1000 ticks

---

## Status funcional atual

| Módulo | Status |
|---|---|
| Memória entre sessões | ✅ funcional |
| Pipeline áudio/visual (texto → FFT → spike) | ✅ correto |
| Feedback 👍👎 | ✅ conectado ao RL |
| Locks não-bloqueantes (loop 200Hz) | ✅ sem deadlock |
| Reconsolidação de memória (sono N3) | ✅ janela de labilidade |
| Integração multimodal AV (com interocepção) | ✅ V4.2 |
| Ontogenia (5 estágios + handlers WS) | ✅ |
| Áudio nativo input (cpal mic) | ✅ |
| Áudio nativo output (Klatt/neural → cpal) | ✅ |
| Porta mobile (audio_raw + voz_params WS) | ✅ |
| Multi-Self Kernel (4 vozes V3.4) | ✅ |
| Neurônio V4 multicompartimental (ATP + Nernst + ephaptic) | ✅ V4.0 |
| Resiliência: watchdog + invariants + DSU | ✅ V4.1 |
| **CPU temp real via WMI** | ✅ V4.2 |
| **BG nigrostriatal (RPE→dopamine_mod)** | ✅ V4.2 |
| **Interocepção→binding AV (insula gate)** | ✅ V4.2 |
| **Curriculo fonético PT-BR completo (Fase 1-11)** | ✅ V4.2 |
| BDNF/BCM/Adenosina-D2/Oxitocina-BLA/WM/Episodic/Prospectiva | ✅ V3.5 |
| Fase 3 (brain_zones V3, HNSW, ToM) | ⏳ pendente |

---

## Ao receber uma tarefa

1. Se envolve novo arquivo ou área desconhecida: use o agente Explore antes de editar
2. Sempre `cargo check --release` após modificações em Rust
3. Commit apenas quando solicitado explicitamente
4. Nunca adicionar tratamento de erro para cenários impossíveis internamente
5. Consultar memória em `C:\Users\alx_r\.claude\projects\f--Selene-brain-2-0\memory\MEMORY.md` quando relevante
