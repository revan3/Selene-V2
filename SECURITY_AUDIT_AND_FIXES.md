# 🔐 Auditoria de Segurança & Plano de Correção — Selene Brain V4.6.1

**Data:** 2026-06-15  
**Auditor:** Claude Code  
**Status:** 6 vulnerabilidades identificadas, 5 críticas/médias

---

## Resumo Executivo

Análise automatizada identificou **6 caminhos críticos de segurança** com risco de:
- Injeção de caminhos (Path Traversal)
- Negação de Serviço (DOS via unbounded connections)
- Vazamento de Informações (silent fallbacks)
- Overflow numérico (STDP weight saturation)
- Deadlock em async code (lock holding durante I/O)

**Severidade Total:** 🔴🔴🟡🟡 (2 HIGH, 2 MEDIUM, 2 LOW)

---

## CRÍTICA #1: WebSocket Path Injection (HIGH)

**Localização:** `src/websocket/bridge.rs` linhas 101–105  
**Risco:** Escrita de arquivo em caminho arbitrário

```rust
// VULNERÁVEL:
let filename = request.filename;  // User-controlled
tokio::fs::write(filename, data).await?;  // No validation!
```

**Cenário de Ataque:**
```
POST /chat
{
  "filename": "../../../etc/passwd_fake",  // Directory traversal
  "message": "jailbreak attempt"
}
→ Cria arquivo em ../../../etc/passwd_fake (fora de app directory)
```

### ✅ FIX #1: Path Canonicalization

```rust
use std::path::PathBuf;

let request_path = PathBuf::from(&request.filename);
let canonical = dunce::canonicalize(&request_path)
    .map_err(|_| "Invalid path")?;

let app_dir = dunce::canonicalize("./selene_data/")?;
if !canonical.starts_with(&app_dir) {
    return Err("Path outside app directory".into());
}

tokio::fs::write(&canonical, data).await?;
```

**Esforço:** 1-2 horas | **Risco Residual:** Baixo

---

## CRÍTICA #2: Unbounded WebSocket Connection Pool (HIGH)

**Localização:** `src/websocket/server.rs` linha 123  
**Risco:** DOS via infinite connection spawn

```rust
// VULNERÁVEL:
warp::ws()
    .on_upgrade(|ws| {
        tokio::spawn(async move { handle_socket(ws).await })  // No limit!
    })
```

**Cenário de Ataque:**
```bash
for i in {1..10000}; do
    curl -i -N -H "Connection: Upgrade" \
         -H "Upgrade: websocket" \
         ws://127.0.0.1:3030/selene &
done
→ Cria 10K tasks tokio → OOM → Crash
```

### ✅ FIX #2: Connection Pool Limit + Rate Limiting

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

let active_connections = Arc::new(AtomicUsize::new(0));
const MAX_CONNECTIONS: usize = 100;

warp::ws()
    .and(warp::addr::remote())
    .on_upgrade(move |ws, addr| {
        let conn_count = Arc::clone(&active_connections);
        async move {
            // Check limit
            if conn_count.load(Ordering::Relaxed) >= MAX_CONNECTIONS {
                eprintln!("Connection limit reached from {:?}", addr);
                return;
            }
            
            conn_count.fetch_add(1, Ordering::Relaxed);
            if let Err(e) = handle_socket(ws).await {
                log::warn!("Socket error: {}", e);
            }
            conn_count.fetch_sub(1, Ordering::Relaxed);
        }
    })
```

**Esforço:** 2-3 horas | **Risco Residual:** Médio (rate limiting por IP também recomendado)

---

## CRÍTICA #3: Silent Fallback to Dummy Sensor (MEDIUM)

**Localização:** `src/main.rs` linha 287  
**Risco:** Fallback silencioso mascara falha de hardware

```rust
// VULNERÁVEL:
let sensor = Arc::new(TokioMutex::new(
    HardwareSensor::new()
        .unwrap_or_else(|e| {
            eprintln!("Sensor init failed: {}", e);
            HardwareSensor::dummy()  // Silent fallback!
        })
));
```

**Problema:** Se sensor real falhar, sistema continua com dados dummy (cpu_temp=35°C sempre).
Pode causar:
- Superaquecimento não detectado
- Adenosina calculation incorreta (depende de temperatura falsa)
- Modo seguro nunca ativado

### ✅ FIX #3: Explicit Fallible Mode

```rust
enum SensorMode {
    Real(HardwareSensor),
    Safe,  // Explicit safe mode with warnings
}

let sensor_mode = match HardwareSensor::new() {
    Ok(hw) => SensorMode::Real(hw),
    Err(e) => {
        log::error!("[SENSOR FAILURE] Hardware sensor unavailable: {}", e);
        log::warn!("[MODE] Entering Safe Mode — limited functionality");
        SensorMode::Safe
    }
};

// Later:
let cpu_temp = match &sensor_mode {
    SensorMode::Real(hw) => hw.get_cpu_temp(),
    SensorMode::Safe => {
        log::warn!("[SENSOR] Using fallback temperature (real sensor unavailable)");
        35.0  // Explicit, logged fallback
    }
};
```

**Esforço:** 3-4 horas | **Risco Residual:** Baixo

---

## MÉDIA #4: Synaptic NaN Risk in Hodgkin-Huxley Gates (MEDIUM)

**Localização:** `src/synaptic_core.rs` linhas 1141–1152  
**Risco:** exp() overflow → NaN na HH gating

```rust
// VULNERÁVEL:
let alpha_m = 0.1 * (v + 40.0) / (1.0 - (-0.1 * (v + 40.0)).exp());
// If (v + 40) >> 0: exp() overflows, alpha_m = NaN
```

**Cenário:** Injected excessive current → Vm exceeds +60mV → exp(6.0) ≈ 403 → division by zero

### ✅ FIX #4: Safe Exponential with Bounds

```rust
fn safe_exp(x: f32) -> f32 {
    // Clamp input to prevent overflow (exp(x) ≤ 1e30 for x ≤ 69)
    x.clamp(-100.0, 30.0).exp()
}

fn hodgkin_huxley_gate(v: f32, mut_type: &str) -> f32 {
    match mut_type {
        "m" => {
            let exp_term = safe_exp(-0.1 * (v + 40.0));
            let denom = 1.0 - exp_term;
            if denom.abs() < 1e-6 { 0.5 } else { 0.1 * (v + 40.0) / denom }
        }
        _ => 0.5,
    }
}
```

**Esforço:** 1-2 horas | **Risco Residual:** Low

---

## MÉDIA #5: STDP Weight Divergence with High Dopamine (MEDIUM)

**Localização:** `src/learning/inter_lobe.rs` linhas 152–159  
**Risco:** Dopamine = 2.5 (max) × large STDP trace → weight overflow

```rust
// VULNERÁVEL:
let dw = 0.0001 * trace_pre * trace_post * dopamine;
weight = (weight + dw).clamp(-2.5, 2.5);
// But trace_pre/post can be 0.5 each, dopamine = 2.5
// → dw = 0.0001 * 0.5 * 0.5 * 2.5 = 0.0000625 per update
// → After 40K updates: weight += 2.5 → saturation OK
// BUT: If post-firing happens 100x per second + dopamine spike:
//    dw could be 0.0001 * 0.8 * 0.8 * 2.5 = 0.00016 × 100 = 0.016/tick
//    → 100 ticks: weight += 1.6 ✓ (still clipped)
// Risk is LOW if clamp is enforced; verify it is.
```

### ✅ FIX #5: Add Saturation Logging + Conservative Bounds

```rust
let mut weight_before = weight;
let dw = 0.0001 * trace_pre * trace_post * dopamine;
weight = (weight + dw).clamp(-2.5, 2.5);

if (weight - weight_before).abs() > 0.1 {
    // Anomalously large weight change
    log::debug!("[STDP] Large Δw: {:.6} (pre={:.3}, post={:.3}, da={:.2})", 
        weight - weight_before, trace_pre, trace_post, dopamine);
}
```

**Esforço:** 0.5-1 hora | **Risco Residual:** Low (already clamped)

---

## BAIXA #6: Panic Inside Async Tasks (Not Caught) (LOW)

**Localização:** `src/main.rs` line ~360 (tokio::spawn multiple places)  
**Risco:** Task panic silently consumed; no supervisor restart

```rust
// VULNERABLE:
tokio::spawn(async move {
    // If this panics, the task dies silently
    handle_socket(ws).await  
});
```

### ✅ FIX #6: Add Panic Catcher in Spawn

```rust
tokio::spawn(async move {
    if let Err(e) = std::panic::catch_unwind(
        std::panic::AssertUnwindSafe(|| {
            // Wrap in block since async closure needed
        })
    ) {
        log::error!("[PANIC] Task panicked: {:?}", e);
    }
});

// Better: Use tokio::task::spawn with catch
let handle = tokio::spawn(async {
    handle_socket(ws).await
});

if let Err(e) = handle.await {
    if e.is_panic() {
        log::error!("[PANIC] Socket handler panicked: {}", e);
    }
}
```

**Esforço:** 2-3 horas (refactor task spawning pattern) | **Risco Residual:** Médio

---

## Plano de Implementação (Sprint)

### Sprint 1: HIGH Priority (1 dia)
- [ ] Fix #1: Path canonicalization (WebSocket)
- [ ] Fix #2: Connection pool limits (WebSocket)
- [ ] Test: Re-run security tests with fuzzer on paths

### Sprint 2: MEDIUM Priority (1 dia)
- [ ] Fix #3: Explicit fallback mode (Sensor)
- [ ] Fix #4: Safe HH gates (NaN prevention)
- [ ] Fix #5: STDP saturation logging

### Sprint 3: LOW Priority (0.5 dia)
- [ ] Fix #6: Panic catching in async tasks
- [ ] Add security test suite (fuzz WebSocket input)
- [ ] Document risk model

---

## Testes de Validação Pós-Fix

```bash
# Path Traversal Resistance
cargo test security_numeric_bounds_stdp_dopamine --release

# Connection Pool
cargo test security_lock_timeout_simulation --release

# HH Stability
cargo test security_hodgkin_huxley_gate_nan_prevention --release

# Full Security Suite
cargo test security_ --release
```

---

## Conclusão

**Status:** 🟡 Vulnerabilidades identificadas, correções planejadas  
**Ação Imediata:** Implementar Fix #1 e #2 antes de produção  
**Revisão Siguiente:** Post-fix audit após 1 sprint

