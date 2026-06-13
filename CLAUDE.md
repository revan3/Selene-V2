# CLAUDE.md — Selene Brain V4.6.1

Instruções persistentes para o Claude Code neste projeto.

> **Versão atual: V4.6.1** — corpo digital (`motor_cortex.rs` + `selene_agent.py`),
> leitura de documentos (`ingest` + `ler_documento.py`), visualização 3D em `/neural`
> (`selene_neural_viz.html`), 23/23 tipos neuronais conectados às zonas, telemetria
> real (`NeuralStatus` estendido), célula-tronco (`stem_cell.rs`). Segurança: bind
> `127.0.0.1` + `SELENE_TOKEN`. Build Avell: `--profile release-avell` + `SELENE_HW=avell`.

---

## 🧠 Obsidian Vault — Knowledge Base REAL

**O vault do Obsidian é um diretório REAL no sistema de arquivos**, não uma metáfora.

**Caminho absoluto:**
```
C:\Users\alx_r\OneDrive\imgs\Documentos\Obsidian Vault\Selene\
```

### Notas existentes no vault

| Arquivo | Conteúdo |
|---------|---------|
| `🧠 Selene — Índice do Projeto.md` | MOC principal — atualizar sempre que houver mudança estrutural |
| `Selene — Arquitetura Geral.md` | Diagramas e directives arquiteturais |
| `Selene — Neural Pool & Codificação Localista.md` | NeuralPool, FP4→FP32, metaplasticidade |
| `Selene — Tipos de Neurônio (17).md` | Todos os tipos neuronais implementados |
| `Selene — Regiões Cerebrais (14).md` | brain_zones/ detalhado |
| `Selene — Neuroquímica (11 moléculas).md` | neurochem.rs documentado |
| `Selene — Sistema de Aprendizado (CLS).md` | STDP, PatternEngine, replay REM |
| `Selene — Templates Cognitivos (19).md` | templates.rs documentado |
| `Selene — Motor de Hipóteses.md` | hypothesis.rs documentado |
| `Selene — Memória e Storage.md` | swap_manager.rs, hierarquia L1–L4 |
| `Selene — Interface WebSocket V3.2.md` | server.rs, todas as mensagens |
| `Selene — Como Compilar e Rodar.md` | Comandos, testes, troubleshooting |
| `Selene — Roadmap & Status.md` | Versões, bugs corrigidos, próximos passos |

### Como atualizar o vault

Ao implementar novas features, corrigir bugs ou alterar a arquitetura:

1. **Identificar qual nota do vault é afetada** (ver tabela acima)
2. **Editar o arquivo diretamente** usando o caminho absoluto acima
3. **Atualizar o índice** (`🧠 Selene — Índice do Projeto.md`) se adicionar nova nota
4. **Usar sintaxe Obsidian** para links internos: `[[Nome da Nota]]`

**Exemplo de edição de nota:**
```
Arquivo: C:\Users\alx_r\OneDrive\imgs\Documentos\Obsidian Vault\Selene\Selene — Roadmap & Status.md
Ação: Adicionar item ao V3.5 ou marcar item como concluído
```

---

## 📋 Memória do Projeto (MEMORY.md interno)

Para memória de curto prazo entre sessões do Claude Code, use também:
- `AUDIT_STATUS_FINAL.md` — estado atual de auditoria/testes
- Criar `MEMORY.md` na raiz se ainda não existir para contexto rápido

---

## 🏗️ Arquitetura Resumida

- **Linguagem:** Rust (+ Python para treinamento)
- **Loop principal:** ~200Hz adaptivo (`src/main.rs`)
- **WebSocket:** `ws://127.0.0.1:3030/selene`
- **Interface:** `http://127.0.0.1:3030/`
- **Testes:** `cargo test --lib` | `cargo run --bin system_test --release`
- **Versão atual:** V3.4 — 36 testes passando (100% ✓)

## 📁 Módulos Principais

```
src/
├── main.rs              Loop 200Hz, try_lock(), passive_hear
├── neural_pool.rs       Pool 4096-bloco, Localist Coding, FP4→FP32
├── synaptic_core.rs     NeuronioHibrido (7 camadas, 17 tipos)
├── neurochem.rs         11 neurotransmissores dinâmicos
├── sleep_cycle.rs       N1–N4 + replay reverso REM
├── brain_zones/         14 regiões cerebrais
├── learning/            templates, CLS, hipóteses, RL
├── storage/             swap_manager, SurrealDB, checkpoints
├── sensors/             áudio, câmera, hardware
└── websocket/           server.rs + bridge.rs
```

## ⚠️ Bugs Corrigidos (V3.4) — Não Reverter

1. **ACh Pipeline** (`main.rs`): hipocampo usa ACh real via `modular_neuro_v3()`
2. **STDP 3-Fator** (`synaptic_core.rs`): `chin_window_open` = `true` por padrão
3. **Grounding RPE** (`bridge.rs`): usa as **8 mais recentes** do `neural_context`
4. **Feedback handler** (`server.rs`): 👍/👎 conectado ao `grounding_rpe()`

## 🎯 Próximos Passos (V3.5)

- Migrar `brain_zones/` para usar PV/SST/VIP/DA_N
- Neurônios serotonérgicos (Raphe) e noradrenérgicos (LC) como fonte real
- Centralizar `W_MAX` / `PESO_MAX_CONCEITO` em `config.rs`
- Resolver `#[allow(dead_code)]` global nos 31 arquivos

---

*Atualizado em: 2026-05-16 | Vault: C:\Users\alx_r\OneDrive\imgs\Documentos\Obsidian Vault\Selene\*
