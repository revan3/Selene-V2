# AUDITORIA FASE 1: BUGS E VULNERABILIDADES
**Data:** 2026-05-02  
**Status:** ✅ COMPLETO  
**Total:** 22 bugs encontrados

---

## 🔴 CRÍTICOS (3)

### Bug #1: Panic na Inicialização do Debug Log
- **Arquivo**: `src/main.rs:175`
- **Problema**: `File::create("selene_debug.log").unwrap()` — se falhar (permissões, disco), aplicação inteira cai
- **Fix**: Usar `.unwrap_or_else(|e| { eprintln!(...); stderr })`

### Bug #2: Panic Duplo no Hook de Pânico
- **Arquivo**: `src/main.rs:191`
- **Problema**: Tentar escrever crash report dentro do panic hook. Se falhar, panic duplo = abort
- **Fix**: Criar diretório antes, usar stderr como fallback (nunca panic em panic hook)

### Bug #3: Unsafe Windows API sem Validação
- **Arquivo**: `src/main.rs:181`
- **Problema**: `timeEndPeriod()` chamado sem confirmar se `timeBeginPeriod()` foi chamado antes
- **Fix**: Rastrear com `static AtomicBool TIME_PERIOD_SET`

---

## 🟠 ALTOS (11)

### Bug #4: Tokio Runtime Panic
- **Arquivo**: `src/main.rs:219`
- **Problema**: `tokio::runtime::Runtime::new().unwrap()` — pode falhar sob pressão de recursos
- **Fix**: Usar `.expect()` com mensagem ou tratamento de erro

### Bug #5: NaN Propagation em STDP
- **Arquivo**: `src/synaptic_core.rs:1829`
- **Problema**: Divisão `dan_sum / dan_count` quando `dan_count == 0` → NaN → pesos corrompidos
- **Fix**: Verificar `if dan_count > 0` antes

### Bug #6: Múltiplas Divisões por Zero
- **Arquivo**: `src/synaptic_core.rs:1842`
- **Problema**: Nem todas as divisões protegidas com `.max(1)`. Gera `Inf` em cálculos neurais
- **Fix**: Auditar TODAS divisões, proteger com `.max(1)`

### Bug #7: Panic em Sort de Floats
- **Arquivo**: `src/compressor/salient.rs:106`, `src/encoding/fft_encoder.rs:122`, etc.
- **Problema**: `.unwrap()` em `partial_cmp()` — se NaN, panic
- **Fix**: Usar `.unwrap_or(std::cmp::Ordering::Equal)`

### Bug #8: Hard-coded Paths Windows
- **Arquivo**: `src/main.rs:183-185`, `src/storage/swap_manager.rs:122`
- **Problema**: `"F:/Selene/..."`, `"D:/Selene_Archive"` — quebra em Linux/macOS, falha se F:/ removido
- **Fix**: Usar `dirs` crate ou `std::env::var("HOME")`

### Bug #9: Arquivo Não Flushed em Crash
- **Arquivo**: `src/main.rs:92-96`
- **Problema**: `selene_response_log.jsonl` aberto mas não closed explicitamente — escrita incompleta
- **Fix**: Usar block scope ou `drop(f)` para forçar flush

### Bug #10: JSON Injection
- **Arquivo**: `src/websocket/server.rs:1191-1320`
- **Problema**: Campos JSON não escapados → injeção em logs, quebra parsing JSONL
- **Fix**: Re-serialize JSON via serde para validar

### Bug #11: Unwrap em Cache que Pode ser None
- **Arquivo**: `src/storage/swap_manager.rs:890`
- **Problema**: `grafo_cache.as_ref().unwrap()` após rebuild que pode falhar
- **Fix**: Retornar `Result` ou `.expect()` com mensagem

### Bug #12: Divisão por Zero em Telemetria
- **Arquivo**: `src/synaptic_core.rs:1983`
- **Problema**: `self.bytes_total as f32 / self.total as f32` quando `total == 0` → Inf
- **Fix**: Verificar `total > 0` antes

### Bug #13: Arquivo Binário Sem Atomic Write
- **Arquivo**: `src/storage/swap_manager.rs:1752`
- **Problema**: `std::fs::write()` não atômico — se crash durante escrita, arquivo corrompido
- **Fix**: Usar write-tmp-then-rename pattern (como `escrever_nvme()` já faz)

### Bug #14: Broadcast Listener Não Dropado
- **Arquivo**: `src/main.rs:422-426`
- **Problema**: Listeners nunca desconectam → RAM cresce indefinido com cada WebSocket
- **Fix**: Explicit cleanup no drop do broadcast

---

## 🟡 MÉDIOS (8)

### Bug #15: Race Condition try_lock()
- **Arquivo**: `src/main.rs` (múltiplas linhas)
- **Problema**: `try_lock()` falha silenciosamente ~100x no código. Estado inconsistente
- **Fix**: Usar `.await lock()` em inicialização, sincronizar com channels

### Bug #16: Mutex em Panic Hook
- **Arquivo**: `src/sensors/hardware.rs:151`
- **Problema**: Lock durante panic hook → deadlock potencial
- **Fix**: Usar `try_lock()` com timeout ou skip em modo pânico

### Bug #17: Caminhos JSON Hardcoded
- **Arquivo**: `src/main.rs:340, 344, 357`
- **Problema**: Nomes fixos como `"selene_hippo_ltp.json"` — não funciona fora do diretório
- **Fix**: Parametrizar via variáveis de ambiente

### Bug #18: Unbounded HashMap Fast Weights
- **Arquivo**: `src/storage/swap_manager.rs:747`
- **Problema**: Fast weights crescem indefinido — TTL não é enforçado
- **Fix**: Cleanup automático: `self.fast_weights.retain(|_, fw| agora - fw.t_criacao < TTL)`

### Bug #19: Buffer Overread em HelixStore
- **Arquivo**: `src/encoding/helix_store.rs:59`
- **Problema**: `.try_into().unwrap()` lê 64 bytes sem validar tamanho
- **Fix**: Validar `buf.len() >= 64` antes

### Bug #20: JSON Field Sem Validação de Tipo
- **Arquivo**: `src/websocket/server.rs:1250`
- **Problema**: `duration_min` pode bypass limites se manipulado
- **Fix**: Validar range `.min(1440)` — máx 24h

### Bug #21: Divisão por Zero em Taxa de Frequência
**Problema**: Outras divisões descobertas durante varredura

### Bug #22: Validação Incompleta de Input
**Problema**: WS handlers confiam demais em `.unwrap_or()` sem validação real

---

## 📊 RESUMO

| Severidade | Qtd | Action |
|-----------|-----|--------|
| 🔴 CRÍTICO | 3 | **Fixar antes de usar em produção** |
| 🟠 ALTO | 11 | **Fixar antes do V3.7** |
| 🟡 MÉDIO | 8 | **Fixar em V3.7** |

---

## 🎯 Próximas Ações
- [ ] Fase 2: Pesquisa científica (gaps teóricos)
- [ ] Fase 3: Compilação + plano V3.7

**Última atualização:** 2026-05-02 02:45 UTC
