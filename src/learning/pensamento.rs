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
use crate::learning::narrativa;
use crate::neural_pool::word_to_concept_id;

// ── Utilitários determinísticos ────────────────────────────────────────────

/// Perturbação hash por concept_id + seed — substitui RNG externo.
/// Garante diversidade de caminhada sem sorteio verdadeiro.
/// Opera sobre u32 (concept_id), nunca sobre texto.
fn hash_perturbacao(id: u32, seed: u64) -> u64 {
    let mut h = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    h ^= id as u64;
    h = h
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    h & 0xFFFF
}

// ── Ciclo consciente ───────────────────────────────────────────────────────

/// Um tick do pensamento consciente:
/// - Lê neural_context como semente
/// - Escolhe âncora: palavra com mais vizinhos no grafo neural (swap_manager)
/// - Caminha 1–3 passos seguindo arestas de maior peso (+ perturbação)
/// - Injeta palavras percorridas em `pensamento_consciente` (janela ≤10)
async fn ciclo_consciente_tick(brain: &Arc<TokioMutex<BrainState>>) {
    // Phase 1: grab read-only data + swap Arc (brief lock)
    let (seed, ctx_ids, swap_arc) = {
        let Ok(mut state) = brain.try_lock() else { return };
        if state.dormindo { return; }
        let seed = state.pensamento_step;
        state.pensamento_step = state.pensamento_step.wrapping_add(1);
        let ctx_ids: Vec<u32> = state.neural_context.iter().copied().collect();
        let swap_arc = state.swap_manager.clone();
        (seed, ctx_ids, swap_arc)
    };

    if ctx_ids.is_empty() { return; }

    // Phase 2: snapshot do grafo NEURAL em u32 — zero texto no walk.
    // id_to_word é clonado só para a fronteira de display (custo ≤ o antigo
    // grafo_palavras(), que clonava cada palavra de cada nó e cada aresta).
    let (grafo, valencias, id_to_word) = if let Ok(mut sw) = swap_arc.try_lock() {
        let g = sw.grafo_conceitos();
        let v = sw.valencias_conceitos();
        let i2w = sw.id_to_word.clone();
        (g, v, i2w)
    } else {
        return;
    };

    // Phase 3: caminhada em concept_ids (u32) — sem locks, sem String
    let ancora: Option<u32> = ctx_ids.iter()
        .filter_map(|&id| grafo.get(&id).map(|v| (id, v.len())))
        .max_by_key(|(_, n)| *n)
        .map(|(id, _)| id);

    let ancora = match ancora { Some(a) => a, None => return };

    let n_passos = 1 + (seed % 3) as usize;
    let mut atual = ancora;
    let mut visitados = std::collections::HashSet::new();
    visitados.insert(atual);
    let mut pensados: Vec<u32> = vec![atual];

    for i in 0..n_passos {
        let Some(vizinhos) = grafo.get(&atual) else { break };
        if vizinhos.is_empty() { break; }

        let proximo = vizinhos.iter()
            .filter(|(vid, _)| !visitados.contains(vid))
            .max_by(|(va, pa), (vb, pb)| {
                let ha = hash_perturbacao(*va, seed.wrapping_add(i as u64));
                let hb = hash_perturbacao(*vb, seed.wrapping_add(i as u64));
                let sa = pa + ha as f32 * 0.0001;
                let sb = pb + hb as f32 * 0.0001;
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(vid, _)| *vid);

        match proximo {
            Some(p) => { visitados.insert(p); pensados.push(p); atual = p; }
            None => break,
        }
    }

    // Phase 4: resolve o caminho u32 → palavras (fronteira de DISPLAY apenas)
    let pensados_w: Vec<String> = pensados.iter()
        .filter_map(|id| id_to_word.get(id).cloned())
        .collect();
    if pensados_w.is_empty() { return; }

    // Phase 5: mutate brain state (re-lock)
    let Ok(mut state) = brain.try_lock() else { return };

    for p in pensados_w.iter().cloned() {
        state.pensamento_consciente.push_back(p);
    }
    while state.pensamento_consciente.len() > 10 {
        state.pensamento_consciente.pop_front();
    }

    if seed % 50 == 0 && !pensados_w.is_empty() {
        let resumo = format!("pensei em: {}", pensados_w.join(", "));
        state.ego.pensamentos_recentes.push_back(resumo);
        if state.ego.pensamentos_recentes.len() > 10 {
            state.ego.pensamentos_recentes.pop_front();
        }
    }

    // ── Pensamento espontâneo ──────────────────────────────────────────────
    // Saliência: cada palavra da janela de display é reconvertida para o seu
    // concept_id canônico (word_to_concept_id) só para consultar a valência
    // neural — o valor neural permanece indexado por u32.
    let saliencia: f32 = state.pensamento_consciente.iter()
        .filter_map(|w| valencias.get(&word_to_concept_id(w)).copied())
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);

    let tedio_atual = (state.ultima_atividade.elapsed().as_secs_f32() / 60.0).clamp(0.0, 1.0);
    state.tedio_nivel = (state.tedio_nivel * 0.98 + tedio_atual * 0.02).clamp(0.0, 1.0);

    let resfriamento_s = state.ultimo_pensamento_espontaneo.elapsed().as_secs();
    let pode_avaliar = resfriamento_s >= 45
        && !state.dormindo
        && !state.pensamento_consciente.is_empty()
        && valencias.len() >= 50
        && saliencia * state.tedio_nivel > 0.30;

    if pode_avaliar {
        let estimulo = state.pensamento_consciente.iter()
            .max_by(|a, b| {
                let va = valencias.get(&word_to_concept_id(a)).copied().unwrap_or(0.0).abs();
                let vb = valencias.get(&word_to_concept_id(b)).copied().unwrap_or(0.0).abs();
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .unwrap_or_default();

        let falou = if !estimulo.is_empty() {
            let (fala, score) = decidir_falar(&state, &estimulo, saliencia);
            if fala {
                let _ = state.pensamento_tx.send(estimulo.clone());
                state.ultimo_pensamento_espontaneo = std::time::Instant::now();
                state.tedio_nivel = 0.0;
                log::info!("💭 [FALA] '{}' score={:.2} (sal={:.2} tédio={:.2})", estimulo, score, saliencia, tedio_atual);
                true
            } else {
                let resumo = format!("guardei: {} (score={:.2})", estimulo, score);
                state.ego.pensamentos_recentes.push_back(resumo);
                if state.ego.pensamentos_recentes.len() > 10 {
                    state.ego.pensamentos_recentes.pop_front();
                }
                log::debug!("🤫 [GUARDA] '{}' score={:.2} — inibido pelo filtro executivo", estimulo, score);
                false
            }
        } else { false };

        // Fallback: se não falou por saliência, tenta ideia gerada pelo motor de hipóteses.
        // Biologicamente: theta-sequences hipocampais projetam predições para o PFC;
        // se o filtro executivo aprova, a predição vira expressão espontânea.
        if !falou {
            if let Some(topico_id) = state.hypothesis_engine.proximo_topico_previsto() {
                // concept_id (u32) → palavra via snapshot id_to_word (display)
                if let Some(topico) = id_to_word.get(&topico_id).cloned() {
                    let (fala_hip, score_hip) = decidir_falar(&state, &topico, saliencia * 0.8);
                    if fala_hip {
                        let _ = state.pensamento_tx.send(format!("hipotese:{}", topico));
                        state.ultimo_pensamento_espontaneo = std::time::Instant::now();
                        log::info!("🧠 [HIPÓTESE] '{}' score={:.2}", topico, score_hip);
                    }
                }
            }
        }
    }
}

// ── Filtro executivo de expressão espontânea ───────────────────────────────
//
// Decide se Selene externaliza um pensamento ou o mantém interno.
// Inspirado no circuito Go/NoGo dos gânglios basais + controle pré-frontal:
//
//   FALAR (Go):
//     + drive interno (saliência × tédio)              → até +0.40
//     + gate dopaminérgico (dopamina > 0.7)             → até +0.20
//     + congruência com goal frontal                    →      +0.20
//     + RPE recente positivo (aprendeu que falar = bom) →      +0.12
//     + OFC: contexto historicamente recompensado       → até +0.15
//
//   GUARDAR (NoGo):
//     - controle serotonérgico (serotonina alta)       → até −0.30
//     - traço cautelosa (personalidade)                → até −0.25
//     - noradrenalina alta (estresse/alerta)           → até −0.15
//     - chat muito recente (< 30s, redundância)        →      −0.35
//     - RPE recente negativo (aprendeu a calar)        →      −0.15
//     - ACC conflito alto (frontal × límbico divergem) → até −0.18
//     - dor social acumulada (rejeições persistentes)  → até −0.20
//
// Threshold: score > 0.45 → fala; ≤ 0.45 → guarda.
// Retorna (bool, f32) — decisão + score para log.
fn decidir_falar(state: &crate::websocket::bridge::BrainState, estimulo: &str, saliencia: f32) -> (bool, f32) {
    let (dopa, sero, nor) = state.neurotransmissores;

    // ── Go signals ─────────────────────────────────────────────────────────

    // Drive interno: produto saliência × tédio (já calculado fora, passado como saliencia)
    let drive = (saliencia * state.tedio_nivel * 0.80).clamp(0.0, 0.40);

    // Gate dopaminérgico: dopamina > 0.7 facilita ação, abaixo suprime
    // Biologicamente: via direta D1 (Go) dos gânglios basais
    let gate_dopa = ((dopa - 0.70) * 0.50).clamp(0.0, 0.20);

    // Congruência com goal frontal: pensamento relevante ao goal atual → +0.20
    // Congruência por concept_id canônico: o estímulo (palavra de display) é
    // convertido na fronteira; os goals do PFC já são u32. Match exato de
    // conceito, sem string-matching difuso no núcleo.
    let estimulo_cid = word_to_concept_id(estimulo);
    let goal_congruente = state.frontal_goal_words.contains(&estimulo_cid);
    let bonus_goal = if goal_congruente { 0.20 } else { 0.0 };

    // RPE aprendido: RPE > +0.30 → falar foi recompensado antes
    let bonus_rpe = if state.ultimo_rpe > 0.30 { 0.12 } else { 0.0 };

    // Ressonância com valores do self_model: estímulo ressoa com quem ela é → +0.25 max
    // Biologicamente: PFC medial (vmPFC) codifica valores pessoais e facilita ação congruente
    let bonus_valor = narrativa::ressonancia_valor(
        estimulo,
        &state.ego.tracos.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
    );

    // OFC: valor aprendido por contexto → Go se este contexto foi historicamente recompensado
    // Biologicamente: vmOFC codifica valor esperado por contexto e facilita ação congruente
    let bonus_ofc = (state.ofc_value_bias * 0.15).clamp(0.0, 0.15);

    // ── NoGo signals ───────────────────────────────────────────────────────

    // Controle serotonérgico: serotonina alta = paciência/filtro = menos impulsivo
    // Biologicamente: projeções do núcleo dorsal da rafe para o PFC
    let inib_sero = (sero * 0.30).clamp(0.0, 0.30);

    // Traço de personalidade cautelosa: inibe expressão espontânea
    let t_cautelosa = state.ego.tracos
        .iter()
        .find(|(n, _)| n == "cautelosa")
        .map(|(_, v)| *v)
        .unwrap_or(0.3);
    let inib_cautelosa = (t_cautelosa * 0.25).clamp(0.0, 0.25);

    // Noradrenalina alta = estado de alerta/estresse → recolhe para dentro
    // Biologicamente: LC-NE suprime output espontâneo quando há ameaça
    let inib_nor = ((nor - 1.0) * 0.15).clamp(0.0, 0.15);

    // Chat muito recente (< 30s): Selene acabou de responder, não é hora de falar de novo
    let chat_recente_s = state.ultima_atividade.elapsed().as_secs_f32();
    let inib_chat = if chat_recente_s < 30.0 { 0.35 } else { 0.0 };

    // RPE aprendido negativo: último feedback foi punição → aprende a calar
    let inib_rpe = if state.ultimo_rpe < -0.30 { 0.15 } else { 0.0 };

    // ACC conflito alto: frontal e límbico divergem → via indireta suprime output
    // Biologicamente: dACC projeta para BG via indireta quando conflito > threshold
    let inib_acc = if state.acc_conflict > 0.50 {
        ((state.acc_conflict - 0.50) * 0.36).clamp(0.0, 0.18)
    } else { 0.0 };

    // Dor social acumulada: histórico de rejeições → recolhimento persistente
    // Biologicamente: rACC registra rejeição social; ativa vmPFC para supressão de expressão
    // É diferente do inib_rpe (instantâneo) — este é memória afetiva de médio prazo
    let inib_social_pain = (state.acc_social_pain * 0.20).clamp(0.0, 0.20);

    // ── Score final ────────────────────────────────────────────────────────
    let score = drive + gate_dopa + bonus_goal + bonus_rpe + bonus_valor + bonus_ofc
              - inib_sero - inib_cautelosa - inib_nor - inib_chat - inib_rpe
              - inib_acc - inib_social_pain;

    (score > 0.45, score)
}

// ── Ciclo inconsciente ─────────────────────────────────────────────────────

/// Um tick do pensamento inconsciente:
/// - Parte de uma palavra aleatória (determinística via hash do step)
/// - Caminha 2–4 passos, preferindo arestas MENOS percorridas (deriva)
/// - Com chance 1/15: cria sinapse transitiva fraca em swap_manager
/// - Injeta em `pensamento_inconsciente` (janela ≤20)
async fn ciclo_inconsciente_tick(brain: &Arc<TokioMutex<BrainState>>) {
    // Phase 1: seed + swap Arc (brief lock)
    let (seed, swap_arc) = {
        let Ok(mut state) = brain.try_lock() else { return };
        if state.dormindo { return; }
        let seed = state.pensamento_step
            .wrapping_mul(2654435761)
            .wrapping_add(1013904223);
        state.pensamento_step = state.pensamento_step.wrapping_add(1);
        let swap_arc = state.swap_manager.clone();
        (seed, swap_arc)
    };

    // Phase 2: snapshot do grafo NEURAL em u32 — deriva livre sem texto
    let (grafo, valencias, id_to_word) = if let Ok(mut sw) = swap_arc.try_lock() {
        let g = sw.grafo_conceitos();
        let v = sw.valencias_conceitos();
        let i2w = sw.id_to_word.clone();
        (g, v, i2w)
    } else {
        return;
    };

    let n_conceitos = valencias.len();
    if n_conceitos == 0 { return; }

    let idx = (seed as usize) % n_conceitos;
    let semente: Option<u32> = valencias.keys().nth(idx).copied();
    let semente = match semente { Some(s) => s, None => return };

    let n_passos = 2 + (seed % 3) as usize;
    let mut atual = semente;
    let mut visitados = std::collections::HashSet::new();
    visitados.insert(atual);
    let mut pensados: Vec<u32> = vec![atual];

    for i in 0..n_passos {
        let Some(vizinhos) = grafo.get(&atual) else { break };
        if vizinhos.is_empty() { break; }

        let proximo = vizinhos.iter()
            .filter(|(vid, _)| !visitados.contains(vid))
            .min_by(|(va, pa), (vb, pb)| {
                let ha = hash_perturbacao(*va, seed.wrapping_add(i as u64 * 13));
                let hb = hash_perturbacao(*vb, seed.wrapping_add(i as u64 * 13));
                let sa = pa * 0.5 + ha as f32 * 0.00001;
                let sb = pb * 0.5 + hb as f32 * 0.00001;
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(vid, _)| *vid);

        match proximo {
            Some(p) => { visitados.insert(p); pensados.push(p); atual = p; }
            None => break,
        }
    }

    // ── Associação transitiva onírica → sinapse u32 em swap_manager ───────
    if seed % 15 == 0 && pensados.len() >= 2 {
        let a = pensados[0];
        let b = pensados[pensados.len() - 1];
        if a != b {
            let ja_conectados = grafo.get(&a)
                .map(|v| v.iter().any(|(vid, _)| *vid == b))
                .unwrap_or(false);
            if !ja_conectados {
                if let Ok(mut sw) = swap_arc.try_lock() {
                    sw.conectar_conceitos_ids(a, b, 0.10);
                }
                let wa = id_to_word.get(&a).map(|s| s.as_str()).unwrap_or("?");
                let wb = id_to_word.get(&b).map(|s| s.as_str()).unwrap_or("?");
                log::debug!("🌀 [INCONSCIENTE] Associação transitiva: {} ↔ {}", wa, wb);
            }
        }
    }

    // Phase 3: resolve caminho u32 → palavras (display) + mutate brain
    let pensados_w: Vec<String> = pensados.iter()
        .filter_map(|id| id_to_word.get(id).cloned())
        .collect();

    let Ok(mut state) = brain.try_lock() else { return };
    for p in pensados_w {
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
    let brain_c  = Arc::clone(&brain);
    let brain_u  = Arc::clone(&brain);
    let brain_cu = Arc::clone(&brain);

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

    // Ciclo de curiosidade — 0.5Hz (a cada 2s)
    // Detecta lacunas no grafo e dispara perguntas autônomas via filtro executivo
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_millis(2000));
        loop {
            ticker.tick().await;
            ciclo_curiosidade_tick(&brain_cu).await;
        }
    });

    println!("🌀 Eternal Hole: pensamento autônomo iniciado (50Hz consciente | 10Hz inconsciente | 0.5Hz curiosidade)");
}

// ── Ciclo de curiosidade ───────────────────────────────────────────────────
//
// A cada 2s, examina o grafo em busca de palavras com poucas associações
// (lacunas de conhecimento). Quando encontra uma lacuna relevante:
//   1. Passa pelo filtro executivo decidir_falar() com bonus de curiosidade
//   2. Se aprovada: envia como estímulo via pensamento_tx (igual ao espontâneo)
//      → o handler WS gera uma pergunta autônoma real (não template)
//   3. Se reprovada: registra no ego como "quero saber mais sobre X"
//
// Biologicamente: hipocampo detecta inconsistência → sinal theta → PFC formula
// a necessidade de buscar informação → basal ganglia decide agir ou não.
async fn ciclo_curiosidade_tick(brain: &Arc<TokioMutex<BrainState>>) {
    // Phase 1: get seed + consciente snapshot + swap Arc
    let (seed, contexto_atual, swap_arc) = {
        let Ok(state) = brain.try_lock() else { return };
        if state.dormindo { return; }
        let seed = state.pensamento_step
            .wrapping_mul(1664525)
            .wrapping_add(1013904223);
        // Janela consciente (display) → concept_ids canônicos para comparação u32
        let contexto_atual: std::collections::HashSet<u32> =
            state.pensamento_consciente.iter()
                .map(|w| word_to_concept_id(w))
                .collect();
        let swap_arc = state.swap_manager.clone();
        (seed, contexto_atual, swap_arc)
    };

    // Phase 2: snapshot do grafo NEURAL em u32 — detecção de lacunas sem texto
    let (grafo, valencias, id_to_word) = if let Ok(mut sw) = swap_arc.try_lock() {
        let g = sw.grafo_conceitos();
        let v = sw.valencias_conceitos();
        let i2w = sw.id_to_word.clone();
        (g, v, i2w)
    } else {
        return;
    };

    if valencias.len() < 30 { return; }

    let n = valencias.len();
    let amostras: Vec<u32> = (0..20u64)
        .filter_map(|i| {
            let idx = (seed.wrapping_add(i * 997)) as usize % n;
            valencias.keys().nth(idx).copied()
        })
        .collect();

    let lacuna: Option<u32> = amostras.into_iter()
        .filter(|id| {
            let arestas = grafo.get(id).map(|v| v.len()).unwrap_or(0);
            let len_ok = id_to_word.get(id).map(|w| w.len() >= 3).unwrap_or(false);
            arestas <= 2 && len_ok
        })
        .max_by_key(|id| {
            let no_ctx = if contexto_atual.contains(id) { 100 } else { 0 };
            let val = (valencias.get(id).copied().unwrap_or(0.0).abs() * 50.0) as usize;
            no_ctx + val
        });

    let lacuna_id = match lacuna { Some(l) => l, None => return };
    let lacuna = match id_to_word.get(&lacuna_id).cloned() { Some(w) => w, None => return };
    let saliencia_lacuna = if contexto_atual.contains(&lacuna_id) { 0.75 } else { 0.45 };

    // Phase 3: mutate brain (re-lock)
    let Ok(mut state) = brain.try_lock() else { return };
    let (fala, score) = decidir_falar_curiosidade(&state, &lacuna, saliencia_lacuna);

    if fala {
        let estimulo = format!("curiosidade:{}", lacuna);
        let _ = state.pensamento_tx.send(estimulo.clone());
        state.ultimo_pensamento_espontaneo = std::time::Instant::now();
        state.curiosity_level = (state.curiosity_level + 0.15).clamp(0.0, 1.0);
        log::info!("🔍 [CURIOSIDADE] lacuna='{}' score={:.2}", lacuna, score);
    } else {
        let desejo = format!("quero entender melhor: {}", lacuna);
        state.ego.pensamentos_recentes.push_back(desejo);
        if state.ego.pensamentos_recentes.len() > 10 {
            state.ego.pensamentos_recentes.pop_front();
        }
    }
}

/// Variante do filtro executivo para curiosidade.
/// Igual ao decidir_falar mas com bonus de curiosidade intrínseca +0.15.
fn decidir_falar_curiosidade(state: &BrainState, lacuna: &str, saliencia: f32) -> (bool, f32) {
    let (fala, score) = decidir_falar(state, lacuna, saliencia);
    // Curiosidade intrínseca = bonus se curiosity_level alto
    let bonus_curiosidade = (state.curiosity_level * 0.20).clamp(0.0, 0.20);
    (fala || (score + bonus_curiosidade) > 0.45, score + bonus_curiosidade)
}
