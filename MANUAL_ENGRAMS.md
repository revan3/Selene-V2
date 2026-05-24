# Manual — Engrams, Implante e Clonagem de Conhecimento

> Selene V4.5 (2026-05-24) | Companion ao `implantar_conhecimento.py` + `clonar_conhecimento.py`

---

## 1. As 4 camadas de representação

Selene não tem "string interna" para conceitos. Toda memória vive em **múltiplas
camadas de codificação** que se complementam. Entender essas camadas é
pré-requisito para implantar conhecimento corretamente.

| # | Camada | Tipo | Função | Quando entra no implante |
|---|--------|------|--------|--------------------------|
| 1 | **`u32` concept_id** | FNV-1a 32-bit do lowercase da palavra | Hash determinístico (mesma palavra → mesmo id em qualquer instância) | ✅ Sempre |
| 2 | **`Uuid` cell pop** | População de 20 neurônios por conceito | Substrato físico do conceito (localist) | ✅ Sempre (criada se faltar) |
| 3 | **`SpikePattern [u64;8]`** | 256 bits de spike timing | Padrão de disparo (representa "como" o conceito ativa) | ⚠️ Apenas para encoding orgânico, não para implante |
| 4 | **Bandas FFT `[f32;32]`** | 32 floats de frequência cocleariforme | Representação acústica real (input do microfone ou TTS) | ❌ Implante PULA esta camada |

**Implicação prática:**
- Implante = conhecimento conceitual (Selene "sabe" que `gato` se associa a `animal`, `peludo`, `miar`)
- Implante NÃO = saber falar a palavra (sem bandas FFT, Selene não tem a "voz" do conceito até ouvir)
- Para grounding sensorial completo, use `audio_chat` ou microfone real (não implante)

---

## 2. O que acontece quando você chama `implant_memory`

```
WS handler (server.rs)
  │
  ▼
1. Para cada `word` no array:
     concept_id = FNV-1a(word.lowercase())
     IF concept_id ∉ swap_manager.conceito_para_id:
        aprender_conceito(concept_id, valence)  ◄── cria população de 20 cells
        id_to_word[concept_id] = word           ◄── salva display
     all_cells ⋃= conceito_para_id[concept_id]  ◄── união de populações
  │
  ▼
2. boost_stdp (opcional, default true):
     Para cada par (cid_i, cid_j) com i < j:
        conectar_conceitos_ids(cid_i, cid_j, 0.5)  ◄── cria sinapse causal
  │
  ▼
3. Input sintético determinístico (não há áudio real):
     synthetic_input = 32 floats derivados de hash(cids)
  │
  ▼
4. HippocampalIndex.implantar_conhecimento(...)
     a. DG.encode(synthetic_input) → SparsePattern (top-k WTA, k≈82)
     b. CA3.store(sparse_pattern) → cria sinapses Hebbian no attractor
     c. EngramStore.encode_com_origem(
           cells = all_cells,
           origem = EngramOrigem::Implantado,    ◄── salvaguarda
           tag = "matematica_basica",            ◄── auditável
           n_reactivations = 10,                 ◄── parece consolidado
        )
  │
  ▼
5. Log: [IMPLANT] engram=42 tag='math' cells=80 synapses=10 valence=0.30
```

**Resultado:** Selene agora tem:
- Conceitos novos com população de neurônios alocada
- Sinapses entre os conceitos do grupo (grafo semântico cresceu)
- Engram episódico marcado como `Implantado` (auditável)
- CA3 attractor capaz de completar partial cues do grupo

---

## 3. Como usar — 3 modos de operação

### Modo 1 — CLI Python (recomendado para uso normal)

```bash
# Listar domínios disponíveis (não conecta)
python implantar_conhecimento.py --listar

# Implantar todos os 9 domínios (~130 engrams, ~10k cells)
python implantar_conhecimento.py --tudo

# Implantar só alguns domínios
python implantar_conhecimento.py --dominios matematica programacao

# Auditar o que foi implantado
python implantar_conhecimento.py --auditar

# Purgar implantes de um domínio (mantém orgânicos)
python implantar_conhecimento.py --purgar fisica

# Purgar TUDO que foi implantado (mantém só orgânico)
python implantar_conhecimento.py --purgar TODOS
```

### Modo 2 — WebSocket direto (qualquer linguagem)

```json
// Implantar
{"action": "implant_memory",
 "words": ["água", "h2o", "hidrogênio", "oxigênio"],
 "valence": 0.2,
 "as_if_repeated": 10,
 "tag": "quimica_basica",
 "boost_stdp": true}

// Resposta:
{"event": "implant_done", "engram_id": 42, "tag": "quimica_basica",
 "cells_used": 80, "synapses_boosted": 6,
 "concepts_created": ["h2o"], "concepts_existed": ["água", "hidrogênio", "oxigênio"]}

// Auditar
{"action": "list_implants"}
// Resposta:
{"event": "implants_list", "count": 130, "organicos": 5, "implantados": 130, "restaurados": 0,
 "implants": [{"id": 1, "tag": "math", "cells": 60, ...}, ...]}

// Purgar por tag
{"action": "purge_implants", "tag": "quimica_basica"}
// Resposta:
{"event": "implants_purged", "removed": 13, "tag_filter": "quimica_basica"}
```

### Modo 3 — Via código Rust direto (testes / hooks internos)

```rust
use selene_kernel::brain_zones::HippocampalIndex;

let mut hit = HippocampalIndex::new(Default::default());
let cells: HashSet<Uuid> = ...;  // populações dos conceitos
let input: Vec<f32> = ...;       // input sintético 32-d

let engram_id = hit.implantar_conhecimento(
    cells,
    &input,
    "biologia_celula".to_string(),
    0.30,    // valência
    10,      // as_if_repeated
    step_atual,
);
```

---

## 4. Como CRIAR novos corpora de conhecimento

A estrutura é simples — Python dict aninhado em `implantar_conhecimento.py`.

### Template mínimo

```python
CONHECIMENTO["meu_dominio"] = {
    "descricao": "Descrição breve para CLI --listar",
    "valencia": 0.20,         # [-1.0, 1.0] — mesma para todas as memórias do domínio
    "as_if_repeated": 10,     # 5-15 típico; quanto maior, mais "consolidado" parece
    "memorias": [
        # Cada lista interna = uma memória episódica
        # → vira 1 engram + sinapses entre todos os pares
        ["palavra1", "palavra2", "palavra3"],
        ["outra", "memoria", "diferente"],
        # ...
    ],
}
```

### Boas práticas

1. **Cada "memória" deve ser um cluster semanticamente coerente** (palavras que pertencem juntas)
2. **3-7 palavras por memória** funciona bem; >10 dilui as sinapses entre pares
3. **Use valências consistentes por domínio** — domínios "agradáveis" (música, jogos) com `+0.4-0.6`, neutros (matemática) com `+0.2-0.3`, negativos só se for útil
4. **Não repita conceitos entre memórias do mesmo domínio** — eles já se conectam transitivamente pelo overlap de cells; repetir só desperdiça sinapses
5. **`as_if_repeated` = 5-15**: <5 fica como "memória nova"; >15 começa a competir com memórias orgânicas reais

### Exemplos por tipo de relação

```python
# Sinônimos / antônimos
["feliz", "alegre", "contente", "satisfeito"]
["triste", "infeliz", "abatido", "melancólico"]

# Hierarquia (hipônimos/hiperônimos)
["fruta", "maçã", "banana", "uva", "laranja"]
["animal", "mamífero", "cachorro", "gato"]

# Causalidade
["chuva", "molhar", "guarda-chuva", "frio"]
["fogo", "queimar", "fumaça", "calor"]

# Sequência temporal
["semente", "broto", "árvore", "fruto"]
["bebê", "criança", "adolescente", "adulto"]

# Co-ocorrência funcional
["teclado", "mouse", "monitor", "computador"]
["panela", "fogão", "comida", "cozinha"]
```

### Validação de novo domínio

```bash
# 1. Adicione seu domínio ao CONHECIMENTO em implantar_conhecimento.py
# 2. Liste para confirmar (não conecta)
python implantar_conhecimento.py --listar

# 3. Implante só ele
python implantar_conhecimento.py --dominios meu_dominio

# 4. Audite via Selene
python implantar_conhecimento.py --auditar

# 5. Teste no chat: pergunte algo do domínio
#    Se Selene associar os conceitos → funcionou
#    Se não → veja log "[IMPLANT]" em selene_debug.log
```

---

## 5. Persistência automática (V4.5)

A partir de V4.5, **engrams + CA3 são persistidos automaticamente** no save
cycle do loop neural (step%5000, ≈ a cada 25 segundos a 200Hz).

### Arquivos persistidos

| Arquivo | Conteúdo | Tamanho típico |
|---------|----------|----------------|
| `selene_hit_engrams.json` | Todos os engrams (Organico/Implantado/Restaurado) + tag + n_reactivations | ~10 KB a ~5 MB |
| `selene_hit_ca3.bin` | Pesos Hebbian do CA3 (binário compacto: 12 bytes por sinapse) | ~10 KB a ~50 MB |

### Ciclo

```
Implant via WS  →  EngramStore.encode_com_origem(Implantado)
                        ↓
                 (esperar até step%5000 == 0)
                        ↓
                 main.rs save cycle:
                   tokio::spawn(async {
                     engrams.salvar_async("selene_hit_engrams.json")
                     ca3.salvar_async("selene_hit_ca3.bin")
                   })
                        ↓
                 ARQUIVO em disco (rename atômico via tokio::fs)

Restart Selene  →  bridge.rs::BrainState::new()
                        ↓
                 main.rs: bs.hippocampal_index.carregar_estado("selene_hit")
                        ↓
                 ESTADO restaurado idêntico ao último save
```

### Garantias

- **Rename atômico** evita corrupção (`.tmp` → final)
- **Snapshot via clone()** dentro do try_lock evita lock-await deadlock
- **Não-bloqueante**: usa `tokio::spawn` para não travar o loop 200Hz
- **Recuperação suave**: se arquivos não existirem (primeira execução), inicializa vazio

### O que NÃO é persistido (e por quê)

- **DG receptive fields**: determinísticos da seed; recriados idênticos
- **DSU/index reverso (cell→engrams)**: rebuilds em `carregar_async` a partir dos engrams

---

## 6. Clonagem de conhecimento (`clonar_conhecimento.py`)

Para mover conhecimento entre instâncias de Selene ou para outros agentes
futuros, use export/import JSON.

### Exportar conhecimento atual

```bash
# Selene rodando → exporta para arquivo
python clonar_conhecimento.py --exportar backup_2026-05-24.json

# Output:
# ✅ EXPORT CONCLUÍDO
#    Engrams exportados: 130
#    Aliases concept→palavra: 245
#    Arquivo: /caminho/absoluto/backup_2026-05-24.json
#    Tamanho: 287.3 KB
```

### Inspecionar sem importar

```bash
# Análise estática do arquivo (não precisa Selene rodando)
python clonar_conhecimento.py --inspecionar backup_2026-05-24.json

# Output: distribuição por origem/tag, estatísticas de cells, valências
```

### Importar em outra instância

```bash
# Em outra Selene (ou na mesma após --purgar TODOS)
python clonar_conhecimento.py --importar backup_2026-05-24.json

# Engrams importados ganham origem=Restaurado e tag="imported:<original>"
# Auditáveis e purgáveis normalmente
```

### Formato do arquivo (`selene_knowledge_v1`)

```json
{
  "selene_knowledge_v1": {
    "n_engrams": 130,
    "n_ca3_synapses": 4520,
    "dg_k_target": 82,
    "engrams": [
      {
        "id": 1,
        "cell_ensemble": ["uuid1", "uuid2", ...],
        "encoding_step": 1000,
        "emocao": 0.30,
        "n_reactivations": 10,
        "last_reactivation_step": 1000,
        "origem": "Implantado",
        "tag": "matematica_basica"
      }
    ],
    "concept_aliases": {
      "1234567890": "soma",
      "2345678901": "número"
    }
  }
}
```

### Garantias da clonagem

- **Concept IDs portáveis**: FNV-1a é determinístico — mesma palavra produz mesmo `u32` em qualquer instância
- **Engrams importados ganham `EngramOrigem::Restaurado`**: distinguíveis dos orgânicos e dos implantes nativos
- **Tag prefixada `imported:`**: rastreabilidade total
- **CA3 weights NÃO são portados**: re-emergem do uso futuro (recall ativa CA3.store implicitamente)
- **Backward compat**: arquivos exportados em V4.5+ leem engrams pré-V4.4 como `Organico`

### Use cases

| Cenário | Comando |
|---------|---------|
| Backup antes de experimento arriscado | `--exportar backup_pre_experimento.json` |
| Restaurar após reset acidental | `--importar backup_pre_experimento.json` |
| Genealogia: bootstrap Selene v2 com base da v1 | export v1 → import v2 |
| Multi-agente: compartilhar conhecimento entre Selenes | Selene A export → Selene B import |
| Versionamento: snapshots etiquetados por data | `--exportar selene_2026-05-24.json` |

---

## 7. Auditoria e Salvaguardas

### Filosofia das salvaguardas

> Implante é poderoso E perigoso. Selene "deveria" desenvolver organicamente
> (princípio da [Ontogenia](Resumos/Implementado/Ontogenia%205%20Estágios.md)).
> O implante pula esse processo. Por isso, cada implante é:
> 1. **Marcado** (`EngramOrigem::Implantado`)
> 2. **Tagueado** (purga seletiva)
> 3. **Logado** (linha em `selene_debug.log` por implante)
> 4. **Auditável** (`list_implants` retorna tudo)
> 5. **Reversível** (`purge_implants(tag)`)

### Comandos de auditoria

```bash
# Quantos de cada origem?
python implantar_conhecimento.py --auditar

# Saída:
# 📊 Engrams na Selene:
#    Orgânicos:    12       ← emergiram de chat/áudio reais
#    Implantados:  130      ← criados via implant_memory
#    Restaurados:  0        ← importados de outro agente
#
# 📁 Por tag:
#    matematica.................. 14 engrams |  860 cells
#    programacao................. 19 engrams | 1180 cells
#    ...
```

### Log explícito (transparência)

Cada implante grava em `selene_debug.log` (RUST_LOG=warn):

```
[IMPLANT] engram=42 tag='matematica' cells=80 synapses=10 valence=0.30
[IMPLANT] engram=43 tag='matematica' cells=70 synapses=10 valence=0.30
[IMPLANT] purged 14 engrams (filter=Some("matematica"))
```

### Recuperação de "implante errado"

```bash
# Cenário: implantei valências erradas (positivo no que devia ser negativo)
python implantar_conhecimento.py --purgar trauma_acidental

# Cenário: testando, quero limpar TUDO que implantei
python implantar_conhecimento.py --purgar TODOS

# Cenário: arquivo de estado corrompido após crash
# → bridge.rs carrega selene_hit_engrams.json; se falhar, começa vazio
#   (log: [HIT] Falha ao carregar engrams: <erro>)
# → restaure de backup com --importar
```

---

## 8. Limitações e FAQ

### Q: Implante substitui treino orgânico?
**Não.** Implante dá conhecimento *conceitual* (associações entre palavras), mas:
- Sem audio_frames → Selene não tem a "voz" desses conceitos para síntese TTS
- Sem grounding visual → conceitos não estão atrelados a percepções reais
- Sem episódios narrativos → autobiografia continua vazia
- Sem emoção real → valência implantada não tem "vivência" sustentável

Para Selene falar, ver, sentir esses conceitos, ela ainda precisa experimentá-los via chat/áudio/vídeo.

### Q: Posso implantar palavras em outra língua?
**Sim.** FNV-1a aceita qualquer string UTF-8. `["dog", "cat", "bird"]` funciona. Apenas note que palavras em línguas diferentes geram concept_ids diferentes mesmo que signifiquem o mesmo (`gato` ≠ `cat` na hash u32).

### Q: O que acontece se eu implantar a mesma memória 2 vezes?
**Cria 2 engrams distintos** (com `id` diferente), mas com `cell_ensemble` idêntico. Não há deduplicação automática. As sinapses STDP entre pares são reforçadas a cada implante (peso cresce até `W_MAX`).

### Q: Implantes "decaem" com o tempo?
**Não automaticamente**. Engrams `Implantado` ficam tão estáveis quanto `Organico`. Mas o CA3 attractor sofre `decay()` periódico — pesos Hebbian abaixo de 0.01 são removidos. Implantes com `as_if_repeated` baixo podem perder "completion power" com o tempo.

### Q: Implante afeta a Ontogenia?
**Não diretamente.** Os thresholds da Ontogenia (Neonatal→Discurso) dependem de `vocab_count` e `graph_edges` do `swap_manager`. Implantar palavras NOVAS aumenta `vocab_count` (porque criamos populações). Boost STDP aumenta `graph_edges`. Isso pode **acelerar progressão** de estágio — efeito colateral aceitável.

### Q: Quanto custa de RAM?
- 1 engram com 80 cells: ~2 KB (HashSet<Uuid> + metadados)
- 100 engrams: ~200 KB
- 1000 engrams: ~2 MB
- CA3: ~12 bytes por sinapse → 10k sinapses = 120 KB

Capacidade máxima atual: `CAP_ENGRAMS = 50_000` (≈100 MB no pior caso). LRU pruning automático ao exceder.

### Q: Posso reimportar um export antigo após mudança de arquitetura?
**Sim, com ressalva.** O formato é JSON com `#[serde(default)]` em campos novos. Engrams pré-V4.4 (sem `origem`/`tag`) deserializam como `Organico` com tag vazia. Engrams pré-V4.5 funcionam normalmente. Quebras incompatíveis seriam marcadas com nova chave raiz (`selene_knowledge_v2`).

### Q: Como medir se o implante "funcionou"?
3 indicadores observáveis:
1. **Estatísticas do WS `list_implants`**: count incrementou + tag aparece
2. **Sinapses no `sinapses_conceito`**: cresceram entre os conceitos implantados
3. **Comportamento no chat**: pergunte algo dentro do domínio e veja se Selene associa os termos

Teste rápido após implantar `matematica`:
```
User: "diga algo sobre números"
Selene (ideal): "um dois três soma mais total ..."  (palavras do cluster)
Selene (sem implante): "..." (silêncio ou divagação)
```

---

## 9. Arquitetura — onde tudo vive

```
src/brain_zones/
├── memory_engrams.rs      ← EngramStore + EngramOrigem (Organico/Implantado/Restaurado)
├── dentate_gyrus.rs       ← Sparse encoder (top-k WTA, sparsity 4%)
├── ca3_attractor.rs       ← Hopfield-like Hebbian (persist binary)
└── hippocampal_index.rs   ← Orquestrador HIT (implantar_conhecimento + export/import)

src/websocket/
├── bridge.rs              ← BrainState.hippocampal_index (HIT in-state)
└── server.rs              ← Handlers WS:
                              • implant_memory   (V4.4)
                              • list_implants    (V4.4)
                              • purge_implants   (V4.4)
                              • export_knowledge (V4.5)
                              • import_knowledge (V4.5)

src/main.rs                ← Save cycle (step%5000) → spawn save de engrams + CA3
                              Load no startup: carregar_estado("selene_hit")

(raiz)
├── implantar_conhecimento.py  ← CLI Python — 9 domínios, 130+ memórias
├── clonar_conhecimento.py     ← CLI Python — export/import/inspecionar
└── MANUAL_ENGRAMS.md          ← este documento
```

---

## 10. Próximos passos sugeridos

Após dominar o implante básico:

1. **Currículos pedagógicos**: estruture corpora em progressão (vocab básico → frases → conceitos abstratos)
2. **A/B testing**: rode 2 Selenes idênticas, implante conhecimento diferente em cada, compare convergência
3. **Trauma sintético** (eticamente sensível): implantes negativos para estudar respostas defensivas
4. **Knowledge graph externo**: integre com Wikidata/ConceptNet via import_knowledge customizado
5. **Especialização de domínio**: clone uma Selene base, implante conhecimento técnico (programming/medicina), use só para esse domínio

---

*V4.5 (2026-05-24) — Documentado em conjunto com a feature.*
*Dúvidas técnicas: ver `src/brain_zones/hippocampal_index.rs` e `src/brain_zones/memory_engrams.rs`.*
