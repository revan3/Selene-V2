# Selene Brain V3.2 — Arquitetura com Codificação Localista e Metaplasticidade

> Documento complementar ao README.md — detalhes técnicos profundos da arquitetura V3.2 implementada entre 2026-04-15 e 2026-04-25.

---

## Sumário Executivo

Selene V3.2 implementa **6 directives arquiteturais críticos** que transformam o sistema de V2 (distribuído, geral) para V3.2 (localista, especializado):

1. **Localist Coding** — 1 conceito = 1 neurônio (`NeuralBlock`) — elimina superposição semântica
2. **Metaplasticidade** — Plasticidade adapta sua própria força via LTP-mediated precision promotion
3. **Grounding Sensório-Motor C0–C4** — Hierarquia cortical com tetos de precisão por nível
4. **On-Demand Allocation** — Pool de 4096 blocos com free_list e reciclagem automática
5. **Resiliência WebSocket 2-Fases** — Heartbeat 30s, Message ID, Thinking Event, Non-Blocking passive_hear
6. **Non-Blocking Loop 200Hz** — try_lock() em lugar de .lock().await, Sink-Source paradigm

---

## 1. Localist Coding / Grandmother Cell (Quiroga 2005)

### O que?
Cada **conceito léxico ou visual** é codificado por **exatamente 1 NeuralBlock físico** na `NeuralPool`. Não há superposição; não há distributed representation para um conceito.

### Biologia
Quiroga (2005) registrou células individuais em hipocampo humano (Jennifer Aniston cell, etc.) que disparam APENAS quando a pessoa vê aquela célula específica. Selene replica esse fenômeno.

### Implementação
```rust
// Em src/neural_pool.rs
pub struct NeuralBlock {
    concept_id: u32,  // Único identificador do conceito
    // ... outros campos ...
}

pub struct NeuralPool {
    concept_index: HashMap<u32, usize>,  // concept_id → índice do bloco
}

// Busca O(1)
fn buscar_conceito(&self, id: u32) -> Option<&NeuralBlock> {
    self.concept_index.get(&id)
        .and_then(|&idx| self.blocks.get(idx))
}
```

### Vantagens
- **Decodificação instantânea**: Leitura do conceito em O(1), sem ambiguidade
- **Binding natural**: Múltiplos blocos disparando juntos = associação direta
- **Epistemologia clara**: "O que é 'gato'?" responde: bloco 1001
- **Evita composicionalidade falsa**: Palavra não é soma de features; é bloco atômico

### Limitações
- Pool limitado (4096 blocos) — criança humana: ~3 bilhões de neurônios
- Novo conceito = novo bloco; não há generalização interna

---

## 2. Metaplasticidade (Abraham & Bear 1996)

### O que?
A **plasticidade sináptica (STDP) é ela mesma modulada por eventos recentes**. LTP cria uma "marca" no neurônio que facilita mais LTP (sliding modification threshold, SMT).

Selene implementa isso como **precisão binária adaptativa**:

### Mecanismo
```
Repouso (sem LTP)  →  FP4  (4 bits úteis, ~12% RAM vs FP32)
                        ↓
Atividade leve     →  FP8  (8 bits úteis, ~25% RAM)
                        ↓
Aprendizado        →  FP16 (16 bits úteis, ~50% RAM)
  (LTP eventos)        ↓
                       FP32 (32 bits, consolidado)
```

### Equação de Promoção
```
ltp_count >= TETO[precision_atual] ?
  └─ Sim → precision ← próximo nível
  └─ Não → mantém
```

Tetos por nível (LIFO — last input, first output):
- FP4 teto = 8 eventos
- FP8 teto = 32 eventos
- FP16 teto = 128 eventos
- FP32 teto = ∞ (máximo)

### Biologia
Neurônios com spines que recebem LTP repetido aumentam seu diâmetro, capturando mais Ca2+ (maior `g_i`), efetivamente ganhando "resoluçã o". Selene mapeia isso para bits.

### Exemplo Prático
```
[t=0] Conceito "maçã" carregado, FP4 (baixo custo, novo)
[t=100ms] Terceira exposição, LTP event → ltp_count += 5
[t=500ms] Consolidação REM, 30+ eventos LTP → ltp_count = 35
[t=500ms+] ltp_count ≥ 32? Sim → **Promove para FP8**
           Memória "maçã" sobe de 12.5% para 25% RAM
           Precisão de cálculo dobra

[t=2000ms] Muito uso, ltp_count agora 150
           ltp_count ≥ 128? Sim → **Promove para FP16**
           Memória "maçã" sobe para 50% RAM
           Conceito agora "consolidado em memória de longo prazo"
```

### Implicação para RL/Aprendizado
Objetos visitados frequentemente (alta recompensa, alta relevância) naturalmente ganham mais precisão sem decisão explícita. Esparsidade emergente (90% em FP4, 10% em FP32) sem regularização L1.

---

## 3. Grounding Sensório-Motor: Hierarquia C0–C4

### Teorema
Toda representação neuronal é ancorada em uma **hierarquia contínua do estímulo bruto até conceito abstrato**. Sem esse grounding, símbolos "flutuam" (símbolo solto problem, Searle 1980).

### Níveis e Tetos de Precisão

| Nível | Descrição | Input Típico | Max Precision | Exemplo Biológico |
|---|---|---|---|---|
| **C0 Sensorial** | Raw sensory → feature detector | FFT 32-band, pixel RGB, oscillogram | FP4 | V1 Layer 4, Nucleus Cochlearis, Dorsal Cochlea |
| **C1 Perceptual** | Features, edges, onset detection | Contornos de imagem, formantes F1-F4, pitch | FP8 | V2 Intermediate, SC superior, Inferior Colliculus |
| **C2 Lexical** | Símbolos linguísticos, conceitos discretos | Tokens, palavra_id, embedding simples | FP16 | Wernicke, Temporal Superior, Angular Gyrus |
| **C3 Contextual** | Sintaxe, causalidade local, frases | Graph walk, binding temporal, sequência | FP32 | Broca, PFC ventrolateral, Hippocampus CA1 |
| **C4 Abstrato** | Lógica, metacognição, identidade | Self-model recursivo, counterfactual, teoria da mente | FP32 | DLPFC, mPFC, ACC dorsal, Precuneus |

### Regra de Não-Violação
```
Um NeuralBlock NUNCA pode ter precisão > teto[level_cortical]
```

Exemplo: Bloco C0 Sensorial teto é **FP4 máximo**, mesmo com 1000 LTP eventos. Requer migração explícita para C1 para ganhar FP8.

### Integração com Passive_hear
Quando `passive_hear` processa tokens via WebSocket:
1. Tokens iniciam em C0 Sensorial (FP4, ruído)
2. `localist_observar()` observa tokens, busca matches em C2 Lexical
3. Se match encontrado → ativa bloco C2 (FP16 teto)
4. Se novo → aloca bloco C2 novo, `ltp_count = 1`

---

## 4. On-Demand Allocation & Reset Neural

### Estrutura
```rust
pub struct NeuralPool {
    blocks: Vec<NeuralBlock>,          // 4096 slots
    free_list: VecDeque<usize>,        // índices livres
    concept_index: HashMap<u32, usize>, // busca rápida
}
```

### Fluxo de Alocação
```
novo_conceito(concept_id=2541)
  ├─ buscar_conceito() → já existe?
  │   └─ Sim: retorna índice
  │   └─ Não: continua
  ├─ free_list.pop_front()?
  │   └─ Índice disponível: realoca esse bloco
  │   └─ free_list vazio: LRU eviction (last_active_ms mais antigo)
  ├─ Novo bloco com concept_id, in_use=true
  └─ concept_index.insert(id, idx)
```

### Reciclagem (Sono N2)
A cada N2, blocos com `last_active_ms > 60s` são:
1. Marcados como `in_use = false`
2. Devolvidos a `free_list` (push_back)
3. Removidos de `concept_index`
4. Metrics coletadas

Saída no sleep handler:
```
[SONO N2] RECICLAGEM:
  ├─ Blocos devolvidos: 247
  ├─ Taxa ocupação: 62% (2534 de 4096)
  ├─ Dist. precisão: [FP4: 15%, FP8: 28%, FP16: 35%, FP32: 22%]
  └─ Dist. cortical: [C0: 8%, C1: 12%, C2: 40%, C3: 30%, C4: 10%]
```

### Benefício
- Memória não cresce linearmente com descobertas
- Máximo overhead = 4096 × sizeof(NeuralBlock) ≈ 4KB + HashMap
- Nunca realoca; reutiliza blocos

---

## 5. Resiliência WebSocket 2-Fases (V3.2)

### Problema Anterior
- YouTube chat loop: `pending_wernicke_tokens: Option<Vec<String>>` era single-slot, sobrescrevia antes do loop consumir
- passive_hear starving: `.lock().await` bloqueava loop 200Hz
- Sem heartbeat: proxies desconectavam silenciosamente
- Sem rastreamento: chat timeout após 5s sem visibilidade

### Solução V3.2

#### 5.1 Heartbeat 30s (Nativo WebSocket)
```rust
// src/websocket/server.rs
let mut interval = interval_at(start + Duration::from_secs(30), Duration::from_secs(30));

select! {
    _ = interval.tick() => {
        let _ = socket.send(Message::Ping(vec![])).await;
    }
    // ... outros handlers
}
```

Benefício: Proxy não desconecta; morte de rede detectável em <30s

#### 5.2 Thinking Event (2-Fase Response)
Antes:
```
[Cliente] chat request
           ↓ (5s sem resposta)
[Cliente] timeout
```

Agora:
```
[Cliente] {"type": "chat", "id": "UUID123", "message": "..."}
           ↓
[Servidor] {"type": "thinking", "id": "UUID123"}
           ↓ (UI: "Pensando...", cliente sabe que está processando)
[Servidor] (500ms–2000ms depois) {"type": "chat_reply", "id": "UUID123", "message": "..."}
           ↓
[UI] Exibe resposta com confiança
```

Implementação:
```rust
// Fase 1: imediato
socket.send(json!({"type": "thinking", "id": msg_id})).await?;

// Fase 2: processamento longo (dentro de lock, depois envio)
let reply = /* processing */;
socket.send(json!({"type": "chat_reply", "id": msg_id, "message": reply})).await?;
```

#### 5.3 Message ID + ACK
```json
{
  "type": "chat",
  "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "message": "Qual é meu nome?"
}
```

Servidor rastreia em `pending_messages: HashMap<UUID, Instant>`:
- Se ACK não chegar em 10s → cliente desconectou, retry automático
- Se ACK chegar → remove de pending

#### 5.4 Passive_hear Non-Blocking (try_lock)
Antes:
```rust
let brain = brain_state.lock().await;  // ⚠️ bloqueia loop 200Hz!
brain.hipocampo.aprender_conceito(token);
```

Agora:
```rust
if let Ok(mut brain) = brain_state.try_lock() {  // ← retorna None se locked
    brain.hipocampo.aprender_conceito(token);
} else {
    // graceful skip, próximo tick tenta novamente
}
```

Sem `.await`, apenas ~1µs overhead se lock indisponível.

##### Dedup Semântico
Tokens são hasheados (FNV-1a) em ordem:
```rust
let sorted = {
    let mut t = tokens.clone();
    t.sort();
    t
};
let hash = fnv_hash(&sorted);

if hash == ultimo_hash && agora - ultimo_ts < 1000ms {
    // Duplicate, skip
    return;
}
```

Elimina "gato, gato, gato" de áudio contínuo.

##### Rate Limiting 400ms
```rust
if agora - ultimo_passive_hear_ts < 400 {
    return;  // Aguarda 400ms mínimo
}
```

Balanceia reatividade (não ignorar) vs. ruído (não descarregar em 200ms).

---

## 6. Diagnóstico de Problemas Fixados (V3.2)

### Problema 1: YouTube Chat Loop (Wernicke Tokens Perdidos)
**Root Cause**: `pending_wernicke_tokens: Option<Vec<String>>` era single-slot channel.

Timeline:
```
[t=0] handler recebe "olá olá olá" → pending = Some(["olá"])
[t=50ms] handler recebe "mundo mundo" → pending = Some(["mundo"]) ❌ SOBRESCREVE
[t=100ms] loop neural consome pending → pega apenas ["mundo"]
          primeira "olá" perdida

→ Selene responde "mundo!" mas "olá" fica no esquecimento
→ usuário repete "olá" → Selene responde "olá olá!" → loop
```

**Solução**: `VecDeque<Vec<String>>` FIFO channel.
```
[t=0] handler → pending.push_back(["olá"])
[t=50ms] handler → pending.push_back(["mundo"])
[t=100ms] loop → pending.pop_front() = ["olá"] ✅
[t=150ms] loop → pending.pop_front() = ["mundo"] ✅

Ambos processados em sequência.
```

### Problema 2: Passive_hear Starving Loop 200Hz
**Root Cause**: `.lock().await` em handler bloqueava Tokio executor

Timeline:
```
[loop 200Hz] loop.tick() começa
[loop 200Hz] try_next passive_hear...
[handler] Chega passivo_hear, faz lock().await
[lock aqui] Espera pelo handler terminar...
[loop 200Hz] STALLED — não pode rodar próximo tick!

Resultado: loop cai de 200Hz para <10Hz quando passive_hear ativo
           "Selene congela"
```

**Solução**: `try_lock()` em passive_hear, graceful skip se locked.

### Problema 3: Zero Chunks Emergindo (Benchmark D3)
**Root Cause**: `escala = 40.0` quantizava inputs < 1.0 para zero

Timeline:
```
[input] audio_frames → 0.5V RMS (normal)
[D3 escala=40.0] 0.5 × 40 = 20.0 (acima de INT8 max 127)
[quantize] clamp(20, 0, 127) = 127 ✓ Bom
[mas] luego 0.01V de ruído → 0.01 × 40 = 0.4
[quantize] floor(0.4) = 0 ❌ Perdeu!
[trace_pre] permanece 0 por 50ms
[chunking] conta_spikes = 0 → chunk nunca emerge
```

**Solução**: Escala = 1.0 (sem amplificação abusiva), inputs naturais 0-127.

### Problema 4: Tonic = 0.04, Contrast Threshold = 0.15 (Benchmark A2)
**Root Cause**: Vision V1 never fires, vision_full stays zero.

```
[V1 tonic] I_tonic = 0.04
[V1 activation threshold] ≈ 0.15 (pra um V1 neuron depolarisar)
[problema] 0.04 < 0.15 → V1 fica silencioso
[resultado] V1 never produces vision_full output
             → Occipital não vê nada → resposta cega

[Fix] tonic ≥ 0.20 (traz V1 acima do threshold)
      + sin() modulation para variação
```

### Problema 5: Frases_padrao Vazio
**Root Cause**: Sem seed, apenas crescimento runtime via WebSocket/persistence

```
[inicializar] BrainState::new() → frases_padrao = HashMap::new()
[sem input] frases nunca são alimentadas
[chunking] precisa de frases_padrao para score semântico
[resultado] output = empty list

[Fix] BrainState::new() → frases_padrao com 13 frases base:
      "oi", "olá", "como você?", "qual é meu nome?", ...
```

---

## 7. Integrações Arquiteturais

### 7.1 Passive_hear → Neural Pool
```
passive_hear(tokens)
  ├─ Dedup (FNV-1a)
  ├─ Rate limit (400ms)
  └─ Observar (localist_observar)
      ├─ Para cada token: buscar_conceito()
      │   ├─ Encontrado: ativa bloco C2 existente
      │   │   └─ ltp_count += 1 (aprendizado incremental)
      │   └─ Não encontrado: aloca novo bloco C2
      │       └─ aloca_para_tarefa(C2Lexical) → novo_bloco
      └─ Integra ao swap_manager para próximo STDP
```

### 7.2 Sleep N2 → Neural Pool Reciclagem
```
sleep_cycle N2
  ├─ Consolidation passes (padrão)
  ├─ Replay reverso (episódios emocionais)
  └─ **Recycling:**
      ├─ reciclar_inativos() chamado
      │   └─ blocos com last_active > 60s → free_list
      ├─ Métricas: taxa_ocupacao, dist_precisao, dist_cortical
      └─ Output: log de reciclagem
```

### 7.3 WebSocket Chat Handler → Thinking + Chat Reply
```
chat handler
  ├─ Parse JSON, extrair message_id
  ├─ Enviar {"type": "thinking", "id": message_id} (imediato)
  ├─ Adquirir lock, processar resposta
  │   ├─ graph walk
  │   ├─ STDP update
  │   └─ template scoring
  ├─ Soltar lock
  └─ Enviar {"type": "chat_reply", "id": message_id, "message": reply}
```

---

## 8. Checklist de Validação V3.2

- [x] `src/neural_pool.rs` — 750 linhas, 4 testes
  - [x] Alocação localist
  - [x] Metaplasticidade promoção
  - [x] Reset neural
  - [x] Teto cortical C0–C4
- [x] `src/main.rs` — try_lock() passive_hear
- [x] `src/websocket/server.rs` — heartbeat, message ID, thinking event, dedup, rate limit
- [x] `src/websocket/bridge.rs` — VecDeque pending_wernicke, NeuralPool field
- [x] `src/synaptic_core.rs` — 4 bug fixes
  - [x] DA_N RPE negativodisponível
  - [x] LC_N SNR scaling
  - [x] BAC firing input_apical
  - [x] VIP→SST lateral connections
- [x] `src/bin/intensive_benchmark.rs` — 3 fixes D3/A2/seed
- [x] Testes: 85/85 passando
- [x] Cargo check: zero erros
- [x] Commit: "V3.2: Localist + Metaplasticity + WebSocket resilience"

---

## 9. Futuro: V3 (Full Brain_zones Migration)

Após V3.2 consolidado, próximas fases:

1. **Migração brain_zones** — usar novos tipos PV/SST/VIP/DA_N em regiões
2. **Neurônios monoaminérgicos reais** — LC_N (noradrenalina), DR (serotonina), VTA (dopamina)
3. **Tipos Izhikevich completos** — 27 tipos em lugar de 17
4. **Doctests fixados** — rl.rs, hardware.rs
5. **Mirror neurons** — simetria temporal, replicação de intenção

---

## Referências

- Quiroga, Q. K. (2005). "Invariant visual representation by single neurons in the human brain." *Nature*.
- Abraham, W. C., & Bear, M. F. (1996). "Metaplasticity: the plasticity of synaptic plasticity." *Trends in Neurosciences*.
- Izhikevich, E. M. (2003). "Simple model of spiking neurons." *IEEE Transactions on Neural Networks*.
- Friston, K. (2022). "Variational Bayes and beyond." *Nature Neuroscience*.
- Tsodyks, M., & Markram, H. (1997). "The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability." *PNAS*.

---

**Criado**: 2026-04-25  
**Status**: Implementação Completa ✅  
**Próximo**: V3 — Full Migration + Monoamines
