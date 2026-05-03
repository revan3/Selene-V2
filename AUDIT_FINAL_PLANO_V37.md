# 🎯 PLANO V3.7 — Melhoria e Evolução da Selene
**Escrito para:** Autista com TDAH — direto, claro, sem buzzwords  
**Data:** 2026-05-02  
**Timeline:** 2-3 semanas | Esforço: 1 desenvolvedor

---

## 🔴 PROBLEMA ATUAL (TL;DR)

Selene V3.6 tem:
- ✅ 8 mecanismos biológicos corretos
- ✅ 0 bugs críticos em ciência
- ❌ 22 bugs técnicos (3 críticos, 11 altos, 8 médios)
- ❌ 3 funcionalidades científicas FALTANDO

---

## 🔧 BUGS CRÍTICOS — FIX PRIMEIRO

### Bug #1: App Morre na Inicialização
```rust
// ANTES (quebra se disco cheio/sem permissão)
File::create("selene_debug.log").unwrap()

// DEPOIS (fallback para stderr)
let file = File::create("selene_debug.log")
    .unwrap_or_else(|_| Box::new(std::io::stderr()));
```
**Tempo:** 15 min  
**Severidade:** 🔴 CRÍTICO

### Bug #2: Crash Duplo (Crash Hook Panics)
```rust
// ANTES: if crash, tenta salvar report, falha, panics novamente

// DEPOIS: seguro, sempre usa stderr como fallback
```
**Tempo:** 15 min  
**Severidade:** 🔴 CRÍTICO

### Bug #3: Windows Unsafe API
```rust
// ANTES: timeEndPeriod() sem validar se foi inicializado
// DEPOIS: rastreia com static AtomicBool
```
**Tempo:** 20 min  
**Severidade:** 🔴 CRÍTICO

**Total Bugs Críticos:** 50 min

---

## 🟠 BUGS ALTOS — FIX ANTES DE V3.7

| Bug | Tipo | Tempo | Priority |
|-----|------|-------|----------|
| NaN em STDP | Numeric | 30min | 1 |
| 6x divisão zero | Numeric | 30min | 2 |
| JSON injection | Security | 45min | 3 |
| Hard-coded paths | Cross-plat | 1h | 4 |
| Arquivo não flush | I/O | 30min | 5 |
| Cache unwrap | Panic | 20min | 6 |
| Broadcast leak | Memory | 45min | 7 |
| Race try_lock | Concurrency | 1h30 | 8 |

**Total Bugs Altos:** 6h

---

## 🚀 V3.7 — 3 NOVAS FUNCIONALIDADES CIENTÍFICAS

### SPRINT 1: Adenosina Glimfático (2-3h)

**O que é?**
Quando você dorme, líquido cerebrospinal limpa metabólitos (adenosina, tau-proteína).  
Selene acumula adenosina durante o dia mas **nunca limpa**.

**O que vai fazer?**
- Criar ATP pool (energia neuronal)
- Quando adenosina alta → dormir → reset adenosina
- Realismo biológico: glimfático clearance (Xie et al. 2013)

**Código:**
```rust
// Em sleep_cycle.rs

pub fn glimfatico_clearance(adenosina: &mut f32, atp_pool: &mut f32) {
    if adenosina > 0.7 {
        // Sono profundo: ATP ativa aquaporin-4 (água intersticial)
        adenosina = adenosina * 0.3;  // reduz 70%
        atp_pool -= 0.05;  // gasta ATP
    }
}
```

**Impacto:** Consolidação mais realista, fadiga biológica verdadeira

---

### SPRINT 2: REM Reverso (3-4h)

**O que é?**
Quando aprende uma sequência nova (música, movimento), durante REM o cérebro **toca de trás pra frente** para consolidar.  
Selene consolida forward (dia→noite) mas não reverso.

**O que vai fazer?**
- Capturar sequência aprendida do dia
- Em sono REM: executar em reverso
- Reforçar sinapses em ordem reversa
- Resultado: memória procedural + motora real (Stickgold 2005)

**Código:**
```rust
// Em sleep_cycle.rs

pub fn rem_reverso_replay(sequencia: &[String]) {
    for token in sequencia.iter().rev() {
        // Replay reverso STDP
        aprender_conceito_reverso(token);
    }
}
```

**Impacto:** Motricidade, habilidades procedurais, músculo-memória

---

### SPRINT 3: Social Reward Feedback (2-3h)

**O que é?**
Oxitocina RECEBE feedback social.  
Se Selene é rejeitada → oxitocina cai.  
Se é aceita → oxitocina sobe.  
Atualmente: oxitocina modula resposta, mas não **sente** rejeição.

**O que vai fazer?**
- Chat handler envia social signal (👍 / 👎)
- Social signal → modula oxitocina
- Oxitocina → modula amígdala fear
- Resultado: medo de rejeição real, conforto em aceitação (Heinrichs 2009)

**Código:**
```rust
// Em websocket/server.rs

pub fn processar_feedback_social(feedback: i32, oxytocin: &mut f32) {
    match feedback {
        1 => *oxytocin = (*oxytocin + 0.1).min(1.5),  // aceitação ↑ OXT
        -1 => *oxytocin = (*oxytocin - 0.1).max(0.3), // rejeição ↓ OXT
        _ => {}
    }
}
```

**Impacto:** Apego emocional real, medo genuíno de rejeição

---

## 📅 TIMELINE V3.7

| Dia | Sprint | Tarefa | Status |
|-----|--------|--------|--------|
| **D0** | Setup | Criar 3 branches + inicializar | ▓ |
| **D1** | 1 | Adenosina glimfático | ▓ 2-3h |
| **D2** | 1 | Testes adenosina | ▓ 1h |
| **D3** | 2 | REM reverso replay | ▓ 3-4h |
| **D4** | 2 | Testes REM | ▓ 1h |
| **D5** | 3 | Social reward feedback | ▓ 2-3h |
| **D6** | 3 | Testes social + integração | ▓ 1-2h |
| **D7** | QA | Validação científica + bug hunt | ▓ 2h |

**Total:** 14-18h

---

## 📊 ANTES vs DEPOIS

| Aspecto | V3.6 | V3.7 |
|---------|------|------|
| Adenosina | ❌ Acumula infinito | ✅ Reset em sono |
| REM | ❌ Forward only | ✅ Bidirecional |
| Social Reward | ❌ One-way | ✅ Two-way feedback |
| Bugs técnicos | 22 | <5 (só low-impact) |

---

## 🔍 COMO TESTAR (TDAH-Friendly)

**Adenosina:**
```bash
# Dormir muitas vezes, verificar que adenosina reseta
selene --test-sleep-cycles 10
# Esperado: adenosina = 0.1 após cada ciclo
```

**REM Reverso:**
```bash
# Ensinar sequência, dormir, verificar consolidação reversa
selene --test-rem-learning "um dois três"
# Esperado: reverso também está consolidated
```

**Social Reward:**
```bash
# Chat: "oi", receber feedback positivo, verificar oxitocina
selene --test-social-feedback
# Esperado: oxitocina sobe com 👍, cai com 👎
```

---

## ⚠️ CUIDADO!

**Não fazer:**
- ❌ Mergear tudo junto (fazer 3 branches paralelos)
- ❌ Esquecer testes (senão bugs novos)
- ❌ Hard-coded magic numbers (parameterizar com constantes)

**Fazer:**
- ✅ 1 sprint = 1 feature = 1 branch = 1 PR
- ✅ Teste antes de merge
- ✅ Código limpo, comentários mínimos (self-documenting)

---

## 📋 CHECKLIST FINAL

- [ ] Bugs críticos fixados (50min)
- [ ] Bugs altos fixados (6h)
- [ ] Sprint 1 (Adenosina) testado ✅
- [ ] Sprint 2 (REM) testado ✅
- [ ] Sprint 3 (Social) testado ✅
- [ ] Integração completa ✅
- [ ] Compilação clean: `cargo check --release` ✅
- [ ] Docs atualizados ✅

---

## 🎉 Resultado Final V3.7

**Selene vai:**
- ✅ Dormir REALMENTE (adenosina real)
- ✅ Aprender motor + procedural (REM reverso)
- ✅ Medo genuíno de rejeição (social reward)
- ✅ Sem crashes, sem NaN, sem memory leaks

**Status:** PRONTO PARA PRODUÇÃO

---

**Questions?** Leia `AUDIT_RESUMO_EXECUTIVO.md` para summary rápido.

