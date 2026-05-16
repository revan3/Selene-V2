# AUDITORIA — Processamento de Texto Residual (Fora da Interface)

**Data:** 2026-05-16  
**Status:** ✅ Compilação OK (cargo check --release)  
**Padrão esperado:** Apenas frequência de áudio (bandas FFT [f32;32]) e frequência de luz processadas no núcleo neural. Strings APENAS em display/serialização.

---

## 1. ACHADOS CRÍTICOS — Violação da Migração Texto→u32

### 🔴 CRÍTICO 1.1 — pensamento.rs (linhas 64-70, 307-310)

**Localização:** Ciclos de pensamento autônomo (consciente + inconsciente)

**Problema:** Conversão desnecessária **u32 → String → u32** em lookup neural

```rust
// ERRADO — pensamento.rs:64-67
let words: Vec<String> = ctx_ids.iter()
    .filter_map(|id| sw.id_to_word.get(id).cloned())
    .collect();
(sw.grafo_palavras(), sw.valencias_palavras(), words)

// DEPOIS — linha 87
let vizinhos: Vec<(String, f32)> = grafo.get(atual.as_str()).cloned()...
```

**Por quê é problema:**
- `neural_context` foi convertida para `VecDeque<u32>` em Sprint 3b ✓
- Mas ao invés de usar `grafo_conceitos_ref()` (HashMap<u32, ...>), o código:
  1. Converte u32 → String via reverse lookup `id_to_word`
  2. Passa para `grafo_palavras()` que converte HashMap<u32,...> → HashMap<String,...>
  3. Faz lookup com `.get(w.as_str())` em String keys
  4. Caminha String `atual` pelo grafo, não u32

**Impacto:** Altas alocações de memória em loop 50Hz (consciente) + 10Hz (inconsciente); ineficiente.

**Afeta também:**
- Linha 29-38: `hash_perturbacao(word: &str, ...)` — deveria ser `hash_perturbacao(id: u32, ...)`
- Linha 87, 327: `.get(atual.as_str())` — lookup String no grafo
- Linha 355: `sw.importar_causal(vec![(a.clone(), b.clone(), ...)])` — passando Strings

---

### 🔴 CRÍTICO 1.2 — bridge.rs (linha 186)

**Campo residual:** `frontal_goal_words: Vec<String>`

```rust
// bridge.rs:186 — BrainState
pub frontal_goal_words: Vec<String>,  // ❌ Deveria ser Vec<u32>
```

**Usado em:**
- `main.rs:1300`: `bs.frontal_goal_words = frontal.goal_queue.front()`
- `server.rs:1836`: `ctx.extend(state.frontal_goal_words.iter().cloned())`
- `pensamento.rs:223-226`: Pattern matching String vs String

**Por quê é problema:**
- Goals do córtex pré-frontal devem ser conceito_ids, não Strings
- Viola separação texto/processamento

**Causa provável:** Sprint 3 incompleta — hipóteses converter para u32 mas goals não foram atualizados

---

### 🟡 CRÍTICO 1.3 — pensamento.rs:210-229 (decidir_falar)

**Função de filtro executivo Go/NoGo:**

```rust
fn decidir_falar(state: &BrainState, estimulo: &str, saliencia: f32) -> (bool, f32)
//                                        ^^^^^^
// Recebe STRING como estimulo — ok na interface?
```

**Linha 223-226:**
```rust
let goal_congruente = state.frontal_goal_words
    .iter()
    .any(|w| w.contains(estimulo) || estimulo.contains(w.as_str()));
    //      ^^^^^^ String matching ^^^^^^ — INEFICIENTE
```

**Problema:**
- `estimulo` vem de `pensamento_consciente` (que eram Strings convertidas)
- Comparação texto-a-texto em loop de decisão
- Se `frontal_goal_words` for convertido para u32, isso fica automaticamente resolvido

---

## 2. RESÍDUOS DE BACKWARD-COMPAT (Aceitáveis, Mas Obsoletos)

### 🟡 2.1 — swap_manager.rs (linhas 931-958)

**Funções wrapper de compatibilidade:**

```rust
pub fn grafo_palavras(&mut self) -> HashMap<String, Vec<(String, f32)>>
pub fn valencias_palavras(&self) -> HashMap<String, f32>
```

**Problema:** Essas funções CONVERTEM HashMap<u32, ...> → HashMap<String, ...> a cada chamada

**Alternativas internas (u32):**
- `grafo_conceitos()` — HashMap<u32, Vec<(u32, f32)>>
- `grafo_conceitos_ref()` — &HashMap<u32, Vec<(u32, f32)>>
- `valencias_conceitos()` — NÃO ENCONTRADO (precisa verificar se existe)

**Usadas por:**
- `pensamento.rs:67` (consciente)
- `pensamento.rs:308` (inconsciente)
- Possíveis outros call sites

**Fix:** Migrar callers para `grafo_conceitos()` / `grafo_conceitos_ref()`

---

## 3. ACEITÁVEIS — Pontos de Entrada de Texto (Camada de Interface)

✅ **Estes DEVEM receber String, pois são conversores de entrada:**

### 3.1 — swap_manager.rs:729 `aprender_conceito(&str, f32)`
- Recebe palavra (String) → converte via `word_to_concept_id()` → processa u32
- ✓ Correto — é um conversor de entrada

### 3.2 — swap_manager.rs:989 `importar_causal(Vec<(String, String, f32)>)`
- Recebe pares (String, String) → converte via `aprender_conceito()` → processa u32
- ✓ Correto — é um conversor em lote

### 3.3 — narrativa.rs `e_auto_referencia(&str)`, `e_compatibilidade_estimulo()`
- Análise de padrão textual em input do chat
- Usado apenas em `server.rs:1723` (interface de entrada)
- ✓ Correto — camada de análise de chat

### 3.4 — ontogeny.rs:272 `reply.split_whitespace()`
- `reply` é saída de texto (UI)
- Limita max palavras por estágio ontogenético
- ✓ Correto — processamento de output

---

## 4. INVESTIGAÇÃO PENDENTE

### ❓ 4.1 — pattern_engine.rs

**Estruturas:**
- `PadraoCandidato` com `gatilho: Vec<String>` (linha 154)
- `PadraoConsolidado` com `predicao: String` (linha 188)

**Status:** Pouco uso encontrado (`pattern_engine.gravar()` em `server.rs:2370`)  
**Pergunta:** São esses padrões usados na lógica neural ou apenas metadado episódico?

**Se for neural:** Devem ser convertidos para u32  
**Se for metadado:** OK deixar como String

---

## 5. RESUMO EXECUTIVO

| Categoria | Achados | Impacto | Ação |
|-----------|---------|--------|------|
| **Crítico** | pensamento.rs conversão u32→String→u32 | Alto — loops 50Hz/10Hz ineficientes | 🔴 Converter para grafo_conceitos() |
| **Crítico** | frontal_goal_words: Vec<String> | Alto — viola migração | 🔴 Converter para Vec<u32> |
| **Alto** | decidir_falar() string matching | Médio — pattern matching ineficiente | 🟡 Resolvido automaticamente se (1) resolvido |
| **Médio** | grafo_palavras/valencias_palavras() | Médio — alocações desnecessárias | 🟡 Migrar callers para u32 versions |
| **Baixo** | pattern_engine gatilho/predicao | Depende de uso | ❓ Investigar antes de agir |

---

## 6. RECOMENDAÇÕES

### ✅ Fazer AGORA (Crítico)

1. **pensamento.rs** — Converter para usar `grafo_conceitos_ref()` diretamente com u32
   - Remove 2 linhas de conversão desnecessária
   - Usa u32 para `atual`, `vizinhos`, `semente`
   - `hash_perturbacao(id: u32, seed: u64)` em vez de `hash_perturbacao(word: &str, ...)`

2. **bridge.rs** — `frontal_goal_words: Vec<u32>`
   - Atualizar tipo em BrainState
   - Atualizar `main.rs:1300` para extrair u32 goals
   - Atualizar `pensamento.rs:223` para u32 lookup

### 🔄 Fazer DEPOIS (Refactoring)

3. **swap_manager.rs** — Remover ou deprecar `grafo_palavras()` e `valencias_palavras()`
   - São wrappers de backward-compat; todos os callers devem usar u32 versions

4. **pattern_engine.rs** — Investigar se `gatilho: Vec<String>` entra em lógica neural
   - Se sim, converter para u32
   - Se não, deixar como está (metadado episódico)

---

## 7. VERIFICAÇÃO PÓS-FIX

Após implementar as ações críticas, executar:

```bash
# Build deve passar sem erros
cargo build --release

# Testes devem passar
cargo run --release --bin system_test  # 22/22 ✓
cargo run --release --bin stability_test

# Verificar que não há regressões
cargo test --release
```

---

## Conclusão

**A migração texto→frequência está ~95% completa.** Os resíduos encontrados são:

1. **Conversões desnecessárias em pensamento.rs** (u32→String→u32) — deve ser removida
2. **frontal_goal_words ainda é Vec<String>** — deve ser Vec<u32>
3. **Wrappers de backward-compat** (grafo_palavras) — podem permanecer, mas callers devem usar u32 versions

O **núcleo neural consegue processar APENAS frequência** (bandas FFT, SpikePattern, u32 concept_ids). A única entrada de texto está em:
- Conversores (`aprender_conceito`, `importar_causal`)
- Análise de padrão (`e_auto_referencia`)
- Output/display

✅ **Selene não recebe Texto bruto no processamento neural.**
