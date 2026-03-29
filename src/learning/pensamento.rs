// src/learning/pensamento.rs
// Eternal Hole — ciclos de pensamento interno autônomo.
//
// Inspirado no conceito do Eternal Hole de Damn Reincarnation:
// em vez de depender de input externo, Selene gera ciclos de mana (ativação)
// que se auto-sustentam e produzem novas associações continuamente.
//
// Dois níveis:
//   • Consciente  (50Hz max): focado no tópico atual (neural_context como semente).
//                             Caminha 1-3 passos a partir da âncora mais conectada.
//                             Influencia diretamente as respostas.
//   • Inconsciente (10Hz):    deriva livre pelo grafo, parte de palavras aleatórias.
//                             Cria associações transitivas ocasionais (sonho acordado).
//                             Pode emergir espontaneamente em respostas.
//
// Ambos usam try_lock — jamais bloqueiam o loop neural principal (200Hz).

use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;
use tokio::time::{interval, Duration};

use crate::websocket::bridge::BrainState;

// ── Utilitários determinísticos ────────────────────────────────────────────

/// Perturbação hash por palavra + seed — substitui RNG externo.
/// Garante diversidade de caminhada sem sorteio verdadeiro.
fn hash_perturbacao(word: &str, seed: u64) -> u64 {
    let mut h = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for b in word.bytes() {
        h = h
            .wrapping_mul(6364136223846793005)
            .wrapping_add(b as u64);
    }
    h & 0xFFFF
}

// ── Ciclo consciente ───────────────────────────────────────────────────────

/// Um tick do pensamento consciente:
/// - Lê neural_context como semente
/// - Escolhe âncora: palavra com mais vizinhos no grafo
/// - Caminha 1–3 passos seguindo arestas de maior peso (+ perturbação)
/// - Injeta palavras percorridas em `pensamento_consciente` (janela ≤10)
async fn ciclo_consciente_tick(brain: &Arc<TokioMutex<BrainState>>) {
    let Ok(mut state) = brain.try_lock() else { return };

    if state.dormindo { return; }

    let seed = state.pensamento_step;
    state.pensamento_step = state.pensamento_step.wrapping_add(1);

    // Clona o contexto agora para não manter borrow de state.neural_context
    let contexto: Vec<String> = state.neural_context.iter().cloned().collect();
    if contexto.is_empty() { return; }

    // Âncora: palavra do contexto com maior grau no grafo (mais conectada)
    let ancora: Option<String> = {
        contexto.iter()
            .filter_map(|w| {
                state.grafo_associacoes
                    .get(w.as_str())
                    .map(|v| (w.clone(), v.len()))
            })
            .max_by_key(|(_, n)| *n)
            .map(|(w, _)| w)
    };

    let ancora = match ancora {
        Some(a) => a,
        None => return,
    };

    // Caminhada consciente: 1–3 passos, prefere arestas mais fortes
    let n_passos = 1 + (seed % 3) as usize;
    let mut atual = ancora.clone();
    let mut visitados = std::collections::HashSet::new();
    visitados.insert(atual.clone());
    let mut pensados: Vec<String> = vec![atual.clone()];

    for i in 0..n_passos {
        // Clone dos vizinhos para liberar o borrow imutável de grafo_associacoes
        let vizinhos: Vec<(String, f32)> = state
            .grafo_associacoes
            .get(atual.as_str())
            .cloned()
            .unwrap_or_default();

        if vizinhos.is_empty() { break; }

        // Próximo passo: maximiza (peso + perturbação hash) entre não visitados
        let proximo = vizinhos
            .iter()
            .filter(|(w, _)| !visitados.contains(w.as_str()))
            .max_by(|(wa, pa), (wb, pb)| {
                let ha = hash_perturbacao(wa, seed.wrapping_add(i as u64));
                let hb = hash_perturbacao(wb, seed.wrapping_add(i as u64));
                let sa = pa + ha as f32 * 0.0001;
                let sb = pb + hb as f32 * 0.0001;
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(w, _)| w.clone());

        match proximo {
            Some(p) => {
                visitados.insert(p.clone());
                pensados.push(p.clone());
                atual = p;
            }
            None => break,
        }
    }

    // Injeta na janela deslizante de pensamentos conscientes
    for p in pensados.iter().cloned() {
        state.pensamento_consciente.push_back(p);
    }
    while state.pensamento_consciente.len() > 10 {
        state.pensamento_consciente.pop_front();
    }

    // ── Introspecção ───────────────────────────────────────────────────────
    // A cada ~50 ticks do ciclo consciente (1 segundo a 50Hz), registra
    // os pensamentos no ego — Selene passa a "saber" o que estava pensando.
    // Isso fecha o loop: hipótese → pensamento → ego → comportamento observado.
    // A auto-referência pode emergir quando o hypothesis_engine aprender que
    // os pensamentos do ego coincidem com o tópico das respostas.
    if seed % 50 == 0 && !pensados.is_empty() {
        let resumo = format!("pensei em: {}", pensados.join(", "));
        state.ego.pensamentos_recentes.push_back(resumo);
        if state.ego.pensamentos_recentes.len() > 10 {
            state.ego.pensamentos_recentes.pop_front();
        }
    }
}

// ── Ciclo inconsciente ─────────────────────────────────────────────────────

/// Um tick do pensamento inconsciente:
/// - Parte de uma palavra aleatória (determinística via hash do step)
/// - Caminha 2–4 passos, preferindo arestas MENOS percorridas (deriva)
/// - Com chance 1/15: cria associação transitiva fraca entre ponta inicial e final
/// - Injeta em `pensamento_inconsciente` (janela ≤20)
async fn ciclo_inconsciente_tick(brain: &Arc<TokioMutex<BrainState>>) {
    let Ok(mut state) = brain.try_lock() else { return };

    if state.dormindo { return; }

    let seed = state
        .pensamento_step
        .wrapping_mul(2654435761)
        .wrapping_add(1013904223);
    state.pensamento_step = state.pensamento_step.wrapping_add(1);

    let n_palavras = state.palavra_valencias.len();
    if n_palavras == 0 { return; }

    // Palavra-semente: índice determinístico no vocabulário
    let idx = (seed as usize) % n_palavras;
    let semente: Option<String> = state.palavra_valencias.keys().nth(idx).cloned();
    let semente = match semente {
        Some(s) => s,
        None => return,
    };

    // Caminhada inconsciente: 2–4 passos, prefere arestas mais fracas (exploração)
    let n_passos = 2 + (seed % 3) as usize;
    let mut atual = semente.clone();
    let mut visitados = std::collections::HashSet::new();
    visitados.insert(atual.clone());
    let mut pensados: Vec<String> = vec![atual.clone()];

    for i in 0..n_passos {
        let vizinhos: Vec<(String, f32)> = state
            .grafo_associacoes
            .get(atual.as_str())
            .cloned()
            .unwrap_or_default();

        if vizinhos.is_empty() { break; }

        // Inconsciente prefere caminhos menos percorridos (pesos menores = regiões desconhecidas)
        let proximo = vizinhos
            .iter()
            .filter(|(w, _)| !visitados.contains(w.as_str()))
            .min_by(|(wa, pa), (wb, pb)| {
                let ha = hash_perturbacao(wa, seed.wrapping_add(i as u64 * 13));
                let hb = hash_perturbacao(wb, seed.wrapping_add(i as u64 * 13));
                // Mistura peso pequeno com hash para romper empates
                let sa = pa * 0.5 + ha as f32 * 0.00001;
                let sb = pb * 0.5 + hb as f32 * 0.00001;
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(w, _)| w.clone());

        match proximo {
            Some(p) => {
                visitados.insert(p.clone());
                pensados.push(p.clone());
                atual = p;
            }
            None => break,
        }
    }

    // ── Associação transitiva onírica (1/15 chance) ────────────────────────
    // Se a caminhada cobriu pelo menos 2 palavras distintas e o step bate:
    // liga ponta inicial ↔ ponta final com peso fraco (0.10).
    // Simula o "sonho acordado" — o Eternal Hole criando novas conexões
    // sem input externo, expandindo o grafo gradualmente.
    if seed % 15 == 0 && pensados.len() >= 2 {
        let a = pensados[0].clone();
        let b = pensados[pensados.len() - 1].clone();

        if a != b {
            // Verifica se ligação direta já existe (borrow imutável temporário)
            let ja_conectados = state
                .grafo_associacoes
                .get(&a)
                .map(|v| v.iter().any(|(w, _)| w == &b))
                .unwrap_or(false);

            if !ja_conectados {
                // Borrow imutável liberado; agora pode modificar
                state
                    .grafo_associacoes
                    .entry(a.clone())
                    .or_default()
                    .push((b.clone(), 0.10));
                state
                    .grafo_associacoes
                    .entry(b.clone())
                    .or_default()
                    .push((a.clone(), 0.10));
                log::debug!(
                    "🌀 [INCONSCIENTE] Associação transitiva criada: {} ↔ {}",
                    a, b
                );
            }
        }
    }

    // Injeta na janela deslizante de pensamentos inconscientes
    for p in pensados {
        state.pensamento_inconsciente.push_back(p);
    }
    while state.pensamento_inconsciente.len() > 20 {
        state.pensamento_inconsciente.pop_front();
    }
}

// ── API pública ────────────────────────────────────────────────────────────

/// Inicia os dois ciclos de pensamento autônomo (Eternal Hole).
///
/// Ciclo consciente  — 50Hz (20ms): foca no neural_context atual.
///                     Influência direta em respostas (contexto enriquecido).
///
/// Ciclo inconsciente — 10Hz (100ms): deriva livre pelo grafo.
///                      Cria associações transitivas; emergência espontânea ocasional.
///
/// Ambos usam try_lock — não bloqueiam o loop neural principal (200Hz) nem
/// o chat handler. Se o brain estiver ocupado, o tick simplesmente é pulado.
pub fn iniciar_ciclos_pensamento(brain: Arc<TokioMutex<BrainState>>) {
    let brain_c = Arc::clone(&brain);
    let brain_u = Arc::clone(&brain);

    // Ciclo consciente — até 50Hz
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_millis(20));
        loop {
            ticker.tick().await;
            ciclo_consciente_tick(&brain_c).await;
        }
    });

    // Ciclo inconsciente — 10Hz
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_millis(100));
        loop {
            ticker.tick().await;
            ciclo_inconsciente_tick(&brain_u).await;
        }
    });

    println!("🌀 Eternal Hole: pensamento autônomo iniciado (50Hz consciente | 10Hz inconsciente)");
}
