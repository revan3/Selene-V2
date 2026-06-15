# 🔐 Auditoria de Segurança — Selene Brain V4.6.1 (RE-VERIFICADA)

**Data:** 2026-06-15  
**Revisão:** 2 (re-verificação linha-a-linha contra o código real)

> ⚠️ **Nota importante sobre a revisão 1:** A primeira versão deste documento foi
> gerada a partir de uma varredura automatizada que **citou linhas imprecisas e
> reportou vulnerabilidades que não existem no código**. Esta revisão 2 corrige
> isso: cada item abaixo foi verificado lendo o arquivo-fonte real. Mantive os
> falsos positivos listados (marcados ❌) para registro — é importante saber o que
> foi descartado e por quê.

---

## Resumo após verificação

| # | Item reportado | Veredito | Ação |
|---|----------------|----------|------|
| V1 | Path injection no WebSocket | ❌ **Falso positivo** | Nenhuma |
| V2 | Sem limite de conexões WS | ✅ **Confirmado** | **Corrigido** |
| V3 | Fallback silencioso de sensor | ⚠️ **Trade-off de design** | Opcional |
| V4 | NaN nas portas Hodgkin-Huxley | ❌ **Falso positivo** | Nenhuma (já protegido) |
| V5 | Overflow de peso no STDP | ❌ **Falso positivo** | Nenhuma (já clamped) |
| V6 | Panic em tasks async | ⚠️ **Menor** | Opcional |

**Conclusão:** Das 6 "vulnerabilidades", **apenas 1 era real** (V2). O código já
possui os controles de segurança que a varredura inicial ignorou: bind em
`127.0.0.1` por padrão, autenticação por `SELENE_TOKEN`, LAN apenas com opt-in
explícito (`SELENE_LAN=1`).

---

## ✅ V2 — Sem limite de conexões WebSocket (CONFIRMADO E CORRIGIDO)

**Localização real:** `src/websocket/mod.rs` — `ws.on_upgrade(...)` (era ~linha 68)

**Problema real:** cada conexão aceita dispara um `handle_connection` sob o
executor tokio, sem teto. Em LAN (`SELENE_LAN=1`) um flood de conexões poderia
multiplicar handlers indefinidamente.

**Contexto que reduz a severidade (ignorado na rev. 1):**
- Bind padrão é `127.0.0.1` → só processos locais conectam.
- Com `SELENE_TOKEN`, conexões sem token são rejeitadas **antes** do upgrade.
- LAN exige opt-in explícito.

Portanto: defesa em profundidade, não um buraco crítico. Mesmo assim vale corrigir.

### Correção aplicada

```rust
const MAX_CONEXOES_WS: usize = 64;
static CONEXOES_WS_ATIVAS: AtomicUsize = AtomicUsize::new(0);

// dentro de on_upgrade:
let n = CONEXOES_WS_ATIVAS.fetch_add(1, Ordering::SeqCst);
if n >= MAX_CONEXOES_WS {
    CONEXOES_WS_ATIVAS.fetch_sub(1, Ordering::SeqCst);  // devolve o slot
    eprintln!("⚠️  [WS] Limite de {} conexões atingido — recusando", MAX_CONEXOES_WS);
    return;
}
server::handle_connection(socket, rx, brain).await;
CONEXOES_WS_ATIVAS.fetch_sub(1, Ordering::SeqCst);       // libera ao fechar
```

O `fetch_add`/`fetch_sub` é atomicamente correto: o contador nunca vaza slots,
mesmo sob conexões concorrentes. **Status: aplicado.**

---

## ❌ V1 — Path injection (FALSO POSITIVO)

**Reportado:** "escrita de arquivo em caminho arbitrário controlado pelo usuário".

**Verificação real:** `src/websocket/server.rs:101-102`
```rust
std::fs::OpenOptions::new()
    .create(true).append(true).open("selene_response_log.jsonl")  // nome FIXO
```
Todos os `fs::write` em server.rs usam nomes **hardcoded** (`selene_linguagem.json`,
`selene_ego.json`, etc.). Não há nome de arquivo derivado de input do cliente.
**Não existe path traversal.** Nenhuma ação necessária.

---

## ❌ V4 — NaN nas portas Hodgkin-Huxley (FALSO POSITIVO)

**Reportado:** "exp() e divisão sem guards → NaN".

**Verificação real:** `src/synaptic_core.rs:1141-1152` — as funções já tratam a
singularidade removível clássica do HH:
```rust
fn alpha_m(v: f32) -> f32 {
    let dv = v + 40.0;
    if dv.abs() < 1e-4 { 1.0 } else { 0.1 * dv / (1.0 - (-dv / 10.0).exp()) }  // guard ✓
}
fn alpha_n(v: f32) -> f32 {
    let dv = v + 55.0;
    if dv.abs() < 1e-4 { 0.1 } else { 0.01 * dv / (1.0 - (-dv / 10.0).exp()) } // guard ✓
}
```
Além disso, `m/h/n` são `clamp(0.0, 1.0)` a cada sub-passo e o `i_h` usa `.min(10.0)`.
**Já protegido.** Validado também pelo teste `l1_todos_tipos_estaveis_*` (TC usa HH e
passa 500 ticks sem NaN). Nenhuma ação necessária.

---

## ❌ V5 — Overflow de peso no STDP (FALSO POSITIVO)

**Reportado:** "dopamina alta → overflow em weight updates".

**Verificação real:** todas as atualizações de peso terminam em `.clamp(-2.5, 2.5)`
(ou bounds equivalentes). Não há caminho de acúmulo sem saturação. O teste
`l1_todos_tipos_estaveis_sob_corrente_forte` confirma traces STDP finitos ao longo
de 500 ticks. Nenhuma ação necessária.

---

## ⚠️ V3 — Fallback silencioso de sensor (TRADE-OFF DE DESIGN)

**Localização:** `src/main.rs` — `HardwareSensor` cai para `dummy()` se o hardware falhar.

Isto é uma **decisão de design** (degradar graciosamente em vez de abortar), não um
bug. O ponto válido: a degradação poderia ser **mais visível**. Sugestão opcional —
logar em nível `error` quando o sensor real falha, para o operador notar. Baixa
prioridade; não altera comportamento neural.

---

## ⚠️ V6 — Panic em tasks async (MENOR)

Um panic dentro de um `tokio::spawn` morre isolado naquela task. Para o servidor
WS isso significa que uma conexão problemática derruba só a si mesma — o que é
aceitável. Um supervisor com restart seria um luxo, não uma necessidade. Baixa
prioridade.

---

## Recomendações de hardening (independentes da auditoria)

Estas valem sempre, e o código já suporta:
1. **Sempre definir `SELENE_TOKEN`** ao usar `SELENE_LAN=1`. O próprio servidor já
   avisa com `‼️` quando LAN está ligada sem token.
2. **Manter o bind padrão** `127.0.0.1` salvo necessidade real de rede.
3. (Aplicado) Teto de conexões WS = 64.

---

## Veredito final

🟢 **Postura de segurança boa.** A única correção real (V2) foi aplicada. Os demais
itens eram falsos positivos da varredura inicial ou trade-offs de design de baixa
prioridade. Lição registrada: varreduras automatizadas devem ser **verificadas
contra o código** antes de virar trabalho.
