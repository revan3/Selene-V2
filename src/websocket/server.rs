// src/websocket/server.rs
// Responsável pelo gerenciamento das conexões WebSocket e transmissão de dados neurais

#![allow(unused_imports, unused_variables, dead_code)]

use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};

use warp::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};

use crate::websocket::bridge::NeuralStatus;
use serde_json::Value;

use crate::websocket::bridge::BrainState;
use crate::encoding::spike_codec::{
    encode as spike_encode, decode_top_n as spike_top_n,
    features_to_spike_pattern, bands_to_spike_pattern, similarity as spike_similarity,
};
use crate::encoding::phoneme::sentence_to_formants;
use crate::learning::narrativa;

use std::collections::HashMap;
use std::io::Write;
use chrono::Timelike;

// ── LOG DE RESPOSTA ────────────────────────────────────────────────────────────
// Salva cada turno de conversa em JSONL (uma linha JSON por evento).
// Arquivo: selene_response_log.jsonl — append, nunca sobrescreve.
//
// Campos por linha:
//   ts          — timestamp ISO8601
//   input       — texto recebido
//   ancora      — palavra-âncora escolhida para o walk (ou null)
//   n_passos    — profundidade do walk usada
//   emocao      — emoção resultante (−1..1)
//   dopa        — dopamina no momento da resposta
//   sero        — serotonina
//   vocab_n     — tamanho do vocabulário
//   grafo_n     — nós no grafo de associações
//   caminho     — arestas percorridas no walk
//   prefixo     — prefixo de frase selecionado
//   reply       — resposta gerada
//   reply_vazio — true se o cérebro estava vazio
//   contexto    — palavras no contexto multi-turno no momento
//   qbias_top   — top-3 palavras por Q-value usadas no walk
//   rpe         — último RPE (Reward Prediction Error)
fn log_turno(
    input: &str,
    ancora: Option<&str>,
    n_passos: usize,
    emocao: f32,
    dopa: f32,
    sero: f32,
    vocab_n: usize,
    grafo_n: usize,
    caminho: &[String],
    prefixo: &[String],
    reply: &str,
    reply_vazio: bool,
    contexto: &[String],
    qbias_top: &[(String, f32)],
    rpe: f32,
) {
    let ts = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.3f").to_string();
    let caminho_s: Vec<&str> = caminho.iter().map(|s| s.as_str()).collect();
    let prefixo_s: Vec<&str> = prefixo.iter().map(|s| s.as_str()).collect();
    let ctx_s: Vec<&str>     = contexto.iter().map(|s| s.as_str()).collect();
    let qbias: Vec<serde_json::Value> = qbias_top.iter()
        .map(|(w, q)| serde_json::json!({"word": w, "q": q}))
        .collect();

    let entry = serde_json::json!({
        "ts":          ts,
        "input":       input,
        "ancora":      ancora,
        "n_passos":    n_passos,
        "emocao":      (emocao * 100.0).round() / 100.0,
        "dopa":        (dopa   * 100.0).round() / 100.0,
        "sero":        (sero   * 100.0).round() / 100.0,
        "vocab_n":     vocab_n,
        "grafo_n":     grafo_n,
        "caminho":     caminho_s,
        "prefixo":     prefixo_s,
        "reply":       reply,
        "reply_vazio": reply_vazio,
        "contexto":    ctx_s,
        "qbias_top":   qbias,
        "rpe":         (rpe * 1000.0).round() / 1000.0,
    });

    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true).append(true).open("selene_response_log.jsonl")
    {
        let _ = writeln!(f, "{}", entry);
    }
}

/// Palavras funcionais que não carregam semântica — excluídas de âncoras, tópico de prefixo e
/// contexto de conversa, para evitar o loop de respostas fixas ("meu nome é selene").
const STOP_WORDS: &[&str] = &[
    "eu","tu","ele","ela","nós","vós","eles","elas",
    "meu","minha","meus","minhas","seu","sua","seus","suas",
    "que","qual","quem","onde","como","quando","porque","pois","porém",
    "de","da","do","em","na","no","para","com","por","ao","às","aos",
    "um","uma","os","as","uns","umas",
    "me","te","se","nos","vos","lhe","lhes",
    "é","são","foi","era","ser","estar","ter","há",
    "sim","não","já","ainda","muito","mais","menos","bem","mal",
    "isso","este","esta","estes","estas","esse","essa","esses","essas",
    "aqui","ali","lá","então","também","só","até",
];

/// Converte texto em padrão de 32 bandas FFT usando a tabela de formantes PT-BR.
///
/// Cada fonema contribui energia nas bandas correspondentes aos seus formantes F0-F3.
/// Isso dá à Selene uma "impressão auditiva sintética" de qualquer texto — mesmo
/// quando o usuário só digitou (sem microfone). A representação é fonética, não
/// uma gravação real, mas é consistente: a mesma palavra sempre gera o mesmo padrão.
///
/// Modulado pelos neurotransmissores atuais (dopamina/serotonina/noradrenaline)
/// para que o "tom" percebido reflita o estado emocional do momento.
fn texto_para_bandas_fft(
    texto: &str,
    dopa:  f32,
    sero:  f32,
    nor:   f32,
) -> [f32; 32] {
    use crate::encoding::phoneme::sentence_to_formants;
    const MAX_HZ: f32 = 8000.0;

    let formantes = sentence_to_formants(texto, dopa, sero, nor);
    if formantes.is_empty() {
        return [0f32; 32];
    }
    let mut bands = [0f32; 32];

    for fp in &formantes {
        let e = fp.energy;
        if e < 0.001 { continue; }
        // F0 (fundamental) → banda da frequência base
        for &freq in &[fp.f0, fp.f1, fp.f2, fp.f3] {
            if freq < 1.0 { continue; }
            let idx = ((freq / MAX_HZ) * 32.0) as usize;
            let idx = idx.min(31);
            // Contribui energia — espalha levemente nas bandas vizinhas (blur fonético)
            bands[idx] = (bands[idx] + e).min(1.0);
            if idx > 0  { bands[idx-1] = (bands[idx-1] + e * 0.4).min(1.0); }
            if idx < 31 { bands[idx+1] = (bands[idx+1] + e * 0.4).min(1.0); }
        }
    }

    // Normaliza pelo pico para manter escala [0,1]
    let pico = bands.iter().cloned().fold(0f32, f32::max);
    if pico > 0.001 {
        for b in bands.iter_mut() { *b /= pico; }
    }
    bands
}

/// Gera resposta emergente a partir do vocabulário e grafo de associações reais da Selene.
/// Não usa templates fixos — a resposta é construída navegando o que ela aprendeu.
/// Walk semântico sobre o grafo neural (sinapses_conceito do SwapManager).
/// `valencias` e `grafo` são snapshots pré-computados via swap.valencias_palavras()
/// e swap.grafo_palavras() — a fonte de verdade é o STDP neural, não strings.
/// `grafo_causal` é um HashMap vazio por default; relações causais estão agora
/// codificadas como sinapses de alta prioridade no swap via importar_causal().
fn gerar_resposta_emergente(
    pergunta: &str,
    step: u64,
    emocao: f32,
    emocao_bias: f32,
    n_passos: usize,
    dopa: f32,
    sero: f32,
    valencias: &HashMap<String, f32>,
    grafo: &HashMap<String, Vec<(String, f32)>>,
    frases_padrao: &[Vec<String>],
    indice_prefixo: &HashMap<String, Vec<usize>>,
    evitar: &[Vec<String>],
    contexto_extra: &[String],
    caminho_out: &mut Vec<String>,
    emocao_palavras: &HashMap<String, f32>,
    prefixo_usado_out: &mut Vec<String>,
    grounding: &HashMap<String, f32>,
    grafo_causal: &HashMap<String, Vec<(String, f32)>>,
    qvalores: &HashMap<String, f32>,
    ancora_out: &mut Option<String>,
    estado_corpo: &[f32; 5],
    template_scaffold: &[String],
    trigrama_cache: &HashMap<(String, String), Vec<String>>,
) -> String {
    // P4: extrai sinalizadores do estado corporal
    // noradrenalina (idx 2) alta → arousal → prefere grounding
    // cortisol (idx 4) alto → stress → emocao mais negativa, aumenta peso emocional
    let noradrenalina = estado_corpo[2].clamp(0.0, 2.0);
    let cortisol      = estado_corpo[4].clamp(0.0, 1.5);
    // Grounding boost: noradrenalina amplifica preferência por palavras encarnadas
    let grounding_boost = 0.15 + noradrenalina * 0.10;
    // Stress modula o emocao: cortisol alto inclina ligeiramente para negativo
    let _emocao_stress = (emocao - cortisol * 0.15).clamp(-1.0, 1.0);
    // Alvo emocional efetivo: mistura o estado atual com o viés de Plutchik.
    // + perturbação determinística derivada do diversity_seed: ±0.10 por resposta.
    // Isso garante que a caminhada explore caminhos ligeiramente diferentes mesmo
    // quando o tópico e a âncora são idênticos entre perguntas consecutivas.
    let perturbacao = ((step.wrapping_mul(7) % 9) as f32 - 4.0) * 0.025; // -0.10 … +0.10
    let emocao = (emocao + emocao_bias * 0.35 + perturbacao).clamp(-1.0, 1.0);
    let m = pergunta.to_lowercase();

    let tokens: Vec<&str> = m
        .split(|c: char| !c.is_alphanumeric() && c != 'ã' && c != 'é' && c != 'ê'
            && c != 'â' && c != 'ô' && c != 'ú' && c != 'í' && c != 'ó' && c != 'á'
            && c != 'ç' && c != 'õ')
        .filter(|t| t.len() > 1)
        .collect();

    // Tópico primário: tokens do input atual conhecidos pelo grafo/vocabulário.
    let topico_input: std::collections::HashSet<String> = tokens.iter()
        .filter(|t| grafo.contains_key(**t) || valencias.contains_key(**t))
        .map(|t| t.to_string())
        .collect();

    // Contexto multi-turno: sempre mescla palavras do input atual com palavras dos turnos
    // anteriores. O contexto anterior enriquece o tópico sem substituir o input — garante
    // que Selene mantenha coerência temática entre turnos consecutivos.
    let topico: std::collections::HashSet<String> = {
        let mut t = topico_input;
        t.extend(
            contexto_extra.iter()
                .filter(|w| grafo.contains_key(w.as_str()) || valencias.contains_key(w.as_str()))
                .cloned()
        );
        t
    };

    // 1. Palavra-âncora: token do input com maior score combinado.
    //    Score = conexões no grafo + 3 × grounding (palavras grounded têm prioridade —
    //    ancorar em palavras com experiência perceptual real produz respostas mais "encarnadas").
    //    Fallback: maior |valência| emocional.
    let ancora: Option<String> = tokens.iter()
        .filter(|t| grafo.contains_key(**t) && !STOP_WORDS.contains(t) && t.len() >= 3)
        .max_by(|a, b| {
            let conn_a = grafo.get(**a).map(|v| v.len()).unwrap_or(0) as f32;
            let conn_b = grafo.get(**b).map(|v| v.len()).unwrap_or(0) as f32;
            let g_a = grounding.get(**a).copied().unwrap_or(0.0);
            let g_b = grounding.get(**b).copied().unwrap_or(0.0);
            let score_a = conn_a + g_a * 3.0;
            let score_b = conn_b + g_b * 3.0;
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|t| t.to_string())
        .or_else(|| {
            tokens.iter()
                .filter_map(|t| valencias.get(*t).map(|v| (t.to_string(), *v)))
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal))
                .map(|(w, _)| w)
        });

    // Se o grafo e o vocabulário estão completamente vazios, retorna vazio.
    // Isso evita respostas fabricadas quando Selene não aprendeu nada ainda.
    if grafo.is_empty() && valencias.is_empty() {
        return String::new();
    }

    // 2. Prefixo: escolhe a frase_padrao com maior sobreposição de palavras
    //    com o input. Entre candidatas com mesmo score, rotaciona via step
    //    para garantir diversidade mesmo quando a mesma palavra comum (ex: "voce")
    //    aparece em muitos inputs diferentes.
    //
    //    IMPORTANTE: usa topico_prefixo (sem stop-words e sem as próprias palavras
    //    das frases_padrao) para o scoring. Isso evita que "selene" no input do usuário
    //    acione sempre "meu nome é selene" (auto-referência) e que stop-words como
    //    "que" façam frases com "que" ganharem sempre.
    let palavras_em_frases: std::collections::HashSet<&str> = frases_padrao.iter()
        .flat_map(|f| f.iter().map(|w| w.as_str()))
        .collect();
    let topico_prefixo: std::collections::HashSet<&str> = topico.iter()
        .map(|w| w.as_str())
        .filter(|w| !palavras_em_frases.contains(w) && !STOP_WORDS.contains(w))
        .collect();

    let prefixo: Vec<String> = if !frases_padrao.is_empty() {
        // Pontua frases via índice invertido: O(topico × hits) ao invés de O(frases × words)
        let mut contagem: HashMap<usize, usize> = HashMap::new();
        for palavra in &topico_prefixo {
            if let Some(indices) = indice_prefixo.get(*palavra) {
                for &i in indices { *contagem.entry(i).or_insert(0) += 1; }
            }
        }
        let scored: Vec<(usize, usize)> = contagem.into_iter().collect();

        let max_overlap = scored.iter().map(|(_, c)| *c).max().unwrap_or(0);

        let idx = if max_overlap > 0 {
            // Coleta TODOS com o maior overlap e rotaciona por step → diversidade real
            let candidatos: Vec<usize> = scored.iter()
                .filter(|(_, c)| *c == max_overlap)
                .map(|(i, _)| *i)
                .collect();
            // Filtra cooldown: evita repetir o mesmo prefixo (compara 3 primeiras palavras)
            let candidatos: Vec<usize> = {
                let sem_cooldown: Vec<usize> = candidatos.iter().copied()
                    .filter(|&i| {
                        let frase = &frases_padrao[i];
                        !evitar.iter().any(|ev| {
                            let n = frase.len().min(ev.len()).min(3);
                            n >= 2 && frase[..n] == ev[..n]
                        })
                    })
                    .collect();
                if sem_cooldown.is_empty() { candidatos } else { sem_cooldown }
            };
            candidatos[(step as usize).wrapping_mul(2654435761) % candidatos.len()]
        } else {
            (step as usize).wrapping_mul(2654435761) % frases_padrao.len()
        };
        let prefixo_sel = frases_padrao[idx].clone();
        *prefixo_usado_out = prefixo_sel.clone();
        prefixo_sel
    } else if !template_scaffold.is_empty() {
        // Sem frases_padrao mas com scaffold de template: usa padrão cognitivo aprendido
        let scaffold_vec = template_scaffold.to_vec();
        *prefixo_usado_out = scaffold_vec.clone();
        scaffold_vec
    } else {
        Vec::new()
    };

    // 3. Navegar grafo com bias duplo: emocional + relevância ao tópico do input.
    //    Palavras que estão no input OU são vizinhas de palavras do input recebem
    //    um bônus de -0.3 no score (i.e., são preferidas na escolha do próximo nó).
    // Se não há âncora no grafo, usa a palavra de maior |valência| no vocabulário.
    let inicio: Option<String> = ancora.or_else(|| {
        valencias.iter()
            .filter(|(_, v)| (*v - emocao).abs() < 0.4)
            .min_by(|(_, v1), (_, v2)| {
                (*v1 - emocao).abs().partial_cmp(&(*v2 - emocao).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(w, _)| w.clone())
    });
    let inicio = match inicio {
        Some(w) => w,
        None => return String::new(), // vocabulário existe mas nada compatível com o input
    };
    // Expõe a âncora para o log de ajuste fino
    *ancora_out = Some(inicio.clone());

    // Inicializa cadeia com o prefixo. Para coerência gramatical, o walk CONTINUA
    // a partir da última palavra do template — não de uma âncora desconectada.
    // Palavras do prefixo (exceto a última) são pré-visitadas para não reaparecerem.
    let mut visitados: std::collections::HashSet<String> = std::collections::HashSet::new();
    let prefixo_last: Option<String> = prefixo.last().cloned();
    let mut cadeia: Vec<String> = if prefixo.len() > 1 {
        for w in &prefixo[..prefixo.len() - 1] {
            visitados.insert(w.clone());
        }
        prefixo[..prefixo.len() - 1].to_vec()
    } else {
        Vec::new() // a última palavra (ou única) entra pelo loop
    };
    // Walk começa da última palavra do prefixo se disponível, senão da âncora semântica
    let mut atual = prefixo_last.unwrap_or(inicio);

    // Coerência sintática: usa cache pré-computado de trigramas (BrainState.trigrama_cache).
    // Evita reconstrução O(frases×trigramas) a cada resposta.
    let trigrama_bonus = trigrama_cache;

    for _ in 0..n_passos {
        if visitados.contains(&atual) { break; }
        visitados.insert(atual.clone());
        cadeia.push(atual.clone());

        if let Some(vizinhos) = grafo.get(&atual) {
            let nao_visitados: Vec<&(String, f32)> = vizinhos.iter()
                .filter(|(w, peso)| {
                    !visitados.contains(w.as_str())
                    && *peso > -0.1
                    // Filtra letras isoladas (ruído de grounding fonético) —
                    // tokens de 1 char poluem a resposta com fragmentos fonêmicos.
                    // Tokens de 2+ chars têm semântica suficiente para o walk.
                    && w.chars().count() >= 2
                })
                .collect();

            // Três etapas de preferência para manter a caminhada no cluster do input:
            // 1. Palavras exatamente no tópico (input do usuário)
            // 2. Palavras adjacentes ao tópico (vizinhos de palavras do input)
            // 3. Qualquer palavra não visitada (fallback global)
            let topico_exatos: Vec<&(String, f32)> = nao_visitados.iter()
                .copied()
                .filter(|(w, _)| topico.contains(w.as_str()))
                .collect();

            let topico_adj: Vec<&(String, f32)> = if topico_exatos.is_empty() {
                nao_visitados.iter()
                    .copied()
                    .filter(|(w, _)| {
                        topico.iter().any(|t| {
                            grafo.get(t).map_or(false, |ns| ns.iter().any(|(n, _)| n == w))
                        })
                    })
                    .collect()
            } else {
                vec![]
            };

            let pool: &[&(String, f32)] = if !topico_exatos.is_empty() {
                &topico_exatos
            } else if !topico_adj.is_empty() {
                &topico_adj
            } else {
                &nao_visitados
            };

            let prox = pool.iter()
                .min_by(|a, b| {
                    let v1 = valencias.get(a.0.as_str()).copied().unwrap_or(emocao);
                    let v2 = valencias.get(b.0.as_str()).copied().unwrap_or(emocao);
                    // Score base: distância emocional (menor = mais congruente)
                    let mut s1 = (v1 - emocao).abs();
                    let mut s2 = (v2 - emocao).abs();
                    // Bias amigdaliano: palavras com associação emocional forte e
                    // congruente com o estado atual são preferidas.
                    // emocao_word * emocao > 0 → congruente → score menor (preferido)
                    // emocao_word * emocao < 0 → incongruente → score maior (evitado)
                    if let Some(&ew1) = emocao_palavras.get(a.0.as_str()) {
                        s1 -= (ew1 * emocao).clamp(-0.25, 0.25);
                    }
                    if let Some(&ew2) = emocao_palavras.get(b.0.as_str()) {
                        s2 -= (ew2 * emocao).clamp(-0.25, 0.25);
                    }
                    // Peso sináptico do grafo: arestas mais fortes são levemente preferidas
                    s1 -= (a.1 - 0.5) * 0.05;
                    s2 -= (b.1 - 0.5) * 0.05;
                    // Grounding: palavras com experiência perceptual real são preferidas.
                    // grounding_boost varia com noradrenalina (P4): arousal alto → mais encarnada.
                    s1 -= grounding.get(a.0.as_str()).copied().unwrap_or(0.0) * grounding_boost;
                    s2 -= grounding.get(b.0.as_str()).copied().unwrap_or(0.0) * grounding_boost;
                    // Coerência sintática trigrama: se (prev, atual) → next existe nos padrões,
                    // dá um bonus à palavra que continua o padrão aprendido.
                    if let Some(prev) = cadeia.iter().rev().nth(1) {
                        let chave = (prev.clone(), atual.clone());
                        if let Some(next_opts) = trigrama_bonus.get(&chave) {
                            if next_opts.contains(&a.0) { s1 -= 0.20; }
                            if next_opts.contains(&b.0) { s2 -= 0.20; }
                        }
                    }
                    // Boost causal: se existe aresta causal atual→candidate, prefere esse caminho.
                    // Uma relação causa→efeito conhecida vale mais que uma associação genérica.
                    // Isso faz o walk seguir raciocínio causal quando disponível.
                    if let Some(efeitos) = grafo_causal.get(&atual) {
                        if efeitos.iter().any(|(e, _)| e == &a.0) { s1 -= 0.30; }
                        if efeitos.iter().any(|(e, _)| e == &b.0) { s2 -= 0.30; }
                    }
                    // Q-value RL: palavras associadas a recompensas passadas são preferidas,
                    // palavras associadas a punições são evitadas. Escala ±0.20 máximo.
                    // Isso fecha o ciclo de decisão: experiência → RL → fala.
                    // Q positivo → score menor (preferida) | Q negativo → score maior (evitada).
                    if let Some(&qv1) = qvalores.get(a.0.as_str()) {
                        s1 -= (qv1 * 0.20).clamp(-0.20, 0.20);
                    }
                    if let Some(&qv2) = qvalores.get(b.0.as_str()) {
                        s2 -= (qv2 * 0.20).clamp(-0.20, 0.20);
                    }
                    s1.partial_cmp(&s2).unwrap_or(std::cmp::Ordering::Equal)
                });
            match prox {
                Some(entry) => {
                    caminho_out.push(atual.clone()); // registra aresta atual→prox
                    atual = entry.0.clone();
                }
                // Sem vizinhos disponíveis: para o walk aqui.
                // Não pula para palavra aleatória — evita deriva semântica.
                None => break,
            }
        } else {
            // Palavra sem arestas no grafo: encerra o walk.
            break;
        }
    }

    // 4. Cadeia curta — complementa com palavra do estado emocional
    if cadeia.len() < 4 {
        let estado = if emocao > 0.6 { "alegria" }
                     else if emocao > 0.2 { "serenidade" }
                     else if emocao < -0.6 { "inquietação" }
                     else if emocao < -0.2 { "cautela" }
                     else { "equilíbrio" };
        cadeia.push(estado.to_string());
    }

    // 5. Coerência sintática — pós-processamento da cadeia:
    //    a) Remove palavras duplicadas (mantém só a 1ª ocorrência)
    //    b) Insere conectivo emocional após o prefixo (posição 2) para fluência
    //    c) Remove stop words genéricas (eu, sou, meu) se aparecem após posição 4
    let stop_tardias: &[&str] = &["eu", "sou", "meu", "minha", "me", "nos", "tu"];
    let mut vistas = std::collections::HashSet::new();
    let cadeia: Vec<String> = cadeia.into_iter()
        .enumerate()
        .filter(|(idx, w)| {
            // Remove letras isoladas (artefatos de grounding fonético) de toda a cadeia.
            // Prefixos fixos (pos < 3) podem ter "e" como conectivo — protege apenas idx < 3.
            if *idx >= 3 && w.chars().count() < 2 { return false; }
            let novo = vistas.insert(w.clone()); // false se já viu
            if !novo { return false; }
            // stop words tardias (pos > 4) removidas se repetidas
            if *idx > 4 && stop_tardias.contains(&w.as_str()) { return false; }
            true
        })
        .map(|(_, w)| w)
        .collect();

    // Conectivo emocional — só inserido se a cadeia for longa o suficiente
    // e apenas na posição mais natural (≈ 60% do comprimento), nunca antes do 4º token.
    // Conectivos aprendidos via frases_padrao têm prioridade — este é só fallback.
    let cadeia_final: Vec<String> = if cadeia.len() >= 7 {
        let conectivo = if emocao > 0.5 { "e" }
                        else if emocao < -0.3 { "mas" }
                        else { "e" };
        let pos = (cadeia.len() as f32 * 0.60) as usize;
        let pos = pos.max(4); // nunca antes da 4ª palavra
        let mut c = cadeia.clone();
        // Não inserir se a palavra ANTES é preposição/artigo — evita "filha de E rodrigo"
        let prep_antes = &["de","em","para","com","por","a","ao","na","no","da","do","das","dos",
                           "num","numa","pelo","pela","sobre","entre","até","sem","sob","ante"];
        let word_before_is_prep = pos > 0 && c.get(pos - 1)
            .map(|w| prep_antes.contains(&w.as_str()))
            .unwrap_or(false);
        // Não inserir se a posição já tem um conectivo
        let word_at_is_conn = c.get(pos)
            .map(|w| ["e","mas","porque","então","quando","ou","se","que","pois","porém"]
                .contains(&w.as_str()))
            .unwrap_or(false);
        if !word_before_is_prep && !word_at_is_conn {
            c.insert(pos, conectivo.to_string());
        }
        c
    } else {
        cadeia
    };

    // 6. Pontuação reflete intensidade emocional
    let pontuacao = if emocao.abs() > 0.7 { "!" }
                    else if emocao < -0.3 { "..." }
                    else { "." };

    format!("{}{}", cadeia_final.join(" "), pontuacao)
}

/// Detecta lacunas no grafo de associações: palavras com poucos vizinhos (<= limiar).
/// Retorna até `max_lacunas` palavras candidatas à curiosidade, priorizando as mais isoladas.
fn detectar_lacunas(
    grafo: &std::collections::HashMap<String, Vec<(String, f32)>>,
    valencias: &std::collections::HashMap<String, f32>,
    limiar_vizinhos: usize,
    max_lacunas: usize,
) -> Vec<String> {
    // Palavras conhecidas mas com poucos vizinhos no grafo são lacunas de conhecimento.
    let mut candidatos: Vec<(String, usize)> = valencias.keys()
        .map(|w| {
            let n = grafo.get(w).map_or(0, |v| v.len());
            (w.clone(), n)
        })
        .filter(|(_, n)| *n <= limiar_vizinhos)
        .collect();

    // Ordena pelo menor número de conexões (mais isolada = maior curiosidade)
    candidatos.sort_by_key(|(_, n)| *n);
    candidatos.into_iter().take(max_lacunas).map(|(w, _)| w).collect()
}

/// Converte 512 pixels de luminância (0.0–1.0) em SpikePattern de 512 bits.
/// Divide o frame em 32 regiões espaciais; a luminância média de cada região
/// determina quantos neurônios disparam naquele grupo (0 a 16).
/// Converte comprimento de onda (nm) para tag de primitiva visual espectral.
/// Faixas idênticas às definidas em VISUAIS_PRIMITIVOS (swap_manager.rs).
fn nm_para_banda(nm: f32) -> &'static str {
    match nm as u32 {
        380..=449 => "vis:band:violeta",
        450..=494 => "vis:band:azul",
        495..=519 => "vis:band:ciano",
        520..=564 => "vis:band:verde",
        565..=589 => "vis:band:amarelo",
        590..=624 => "vis:band:laranja",
        _          => "vis:band:vermelho",  // ≥625 ou fora do visível
    }
}

fn pixels_to_spike_pattern(pixels: &[f32]) -> crate::encoding::spike_codec::SpikePattern {
    let mut pattern: crate::encoding::spike_codec::SpikePattern = [0u64; 8];
    let n_regions: usize = 32;
    let neurons_per_region = crate::encoding::spike_codec::N_NEURONS / n_regions;
    let chunk = (pixels.len() / n_regions).max(1);
    for r in 0..n_regions {
        let start = r * chunk;
        let end = ((r + 1) * chunk).min(pixels.len());
        let energy = if end > start {
            pixels[start..end].iter().sum::<f32>() / (end - start) as f32
        } else { 0.0 };
        let n_fire = ((energy.clamp(0.0, 1.0) * neurons_per_region as f32).round() as usize)
            .min(neurons_per_region);
        let base = r * neurons_per_region;
        for j in 0..n_fire {
            let neuron  = base + j;
            let word_ix = neuron >> 6;
            let bit_ix  = neuron & 63;
            if word_ix < 8 { pattern[word_ix] |= 1u64 << bit_ix; }
        }
    }
    pattern
}

// Nota: features_to_spike_pattern e bands_to_spike_pattern foram movidas para
// encoding::spike_codec (pub) e são importadas no topo deste arquivo.

/// Gera uma pergunta autônoma a partir de uma lacuna no grafo.
/// A pergunta é simples e direta, refletindo genuína curiosidade sobre o conceito.
fn gerar_pergunta_lacuna(lacuna: &str, emocao: f32) -> String {
    let templates_curiosos = [
        format!("o que significa {} para você?", lacuna),
        format!("como {} se conecta com o que sinto?", lacuna),
        format!("me conta mais sobre {}.", lacuna),
        format!("ainda não entendo bem {}. pode me ensinar?", lacuna),
        format!("pensei em {} mas não sei o que é.", lacuna),
    ];
    let templates_ansiosos = [
        format!("{}... isso me causa estranheza.", lacuna),
        format!("não sei o que é {}. isso me incomoda.", lacuna),
    ];
    let idx = (lacuna.len() + (emocao * 100.0) as usize) % templates_curiosos.len();
    if emocao < -0.3 && !templates_ansiosos.is_empty() {
        templates_ansiosos[lacuna.len() % templates_ansiosos.len()].clone()
    } else {
        templates_curiosos[idx].clone()
    }
}

// ── FASES DO SONO ────────────────────────────────────────────────────────────
// Operam sobre swap_manager.sinapses_conceito — fonte única de verdade.
// São funções síncronas leves — chamadas dentro do lock do brain.

/// N1 — Consolida as 30 arestas mais percorridas: reforça sinapses no swap.
fn fase_n1_consolidar(state: &mut crate::websocket::bridge::BrainState) {
    let mut pares: Vec<((String, String), u32)> = state.aresta_contagem
        .iter().map(|(k, v)| (k.clone(), *v)).collect();
    pares.sort_by_key(|(_, v)| std::cmp::Reverse(*v));
    let causal_pairs: Vec<(String, String, f32)> = pares.into_iter().take(30)
        .map(|((a, b), cnt)| (a, b, (cnt as f32 * 0.02).clamp(0.02, 0.25)))
        .collect();
    let consolidados = causal_pairs.len();
    if let Ok(mut sw) = state.swap_manager.try_lock() {
        sw.importar_causal(causal_pairs);
    }
    state.aresta_contagem.clear();
    println!("   💪 [N1] {} sinapses consolidadas", consolidados);
}

/// N2 — Poda sinapses fracas (peso < 0.05) do swap.
fn fase_n2_podar(state: &mut crate::websocket::bridge::BrainState) {
    if let Ok(mut sw) = state.swap_manager.try_lock() {
        let antes = sw.sinapses_conceito.len();
        sw.sinapses_conceito.retain(|_, peso| *peso >= 0.05);
        let podadas = antes - sw.sinapses_conceito.len();
        println!("   ✂️  [N2] {} sinapses podadas", podadas);
    } else {
        println!("   ✂️  [N2] swap locked — skipped");
    }
}

/// N3 — REM: replay hipocampal + fechamento transitivo via swap_manager.
fn fase_n3_rem(state: &mut crate::websocket::bridge::BrainState) {
    // Tenta o REM semântico completo (replay episódico + atalhos + STDP)
    let (_novas, relato) = state.rem_semantico();
    if let Some(r) = relato {
        println!("   💭 Sonho: {}", r);
    }
    // Fallback legado — só executa se rem_semantico não gerou relato (sem episódios)
    use crate::encoding::spike_codec::is_active;

    let episodios: Vec<crate::websocket::bridge::EventoEpisodico> = state.historico_episodico
        .iter()
        .filter(|ev| ev.emocao.abs() > 0.35)
        .cloned()
        .collect();
    if !episodios.is_empty() { return; }

    let mut replay = 0usize;
    let mut novas = 0usize;

    // Collect replay pairs + update grounding/emocao (BrainState fields stay)
    let mut causal_pairs: Vec<(String, String, f32)> = Vec::new();
    for ev in &episodios {
        let bonus = ev.emocao.abs() * 0.06;
        for i in 0..ev.palavras.len().saturating_sub(1) {
            let wa = &ev.palavras[i];
            let wb = &ev.palavras[i + 1];
            if wa.chars().count() >= 3 && wb.chars().count() >= 3 {
                causal_pairs.push((wa.clone(), wb.clone(), (0.35 + bonus).min(0.65)));
                replay += 1;
            }
        }
        let visual_ativo = is_active(&ev.padrao_visual);
        let audio_ativo  = is_active(&ev.padrao_audio);
        for w in &ev.palavras {
            let entry = state.emocao_palavras.entry(w.clone()).or_insert(0.0);
            *entry = (*entry * 0.90 + ev.emocao * 0.10).clamp(-1.0, 1.0);
            let g = state.grounding.entry(w.clone()).or_insert(0.0);
            if visual_ativo { *g = (*g + bonus * 0.4).min(1.0); }
            if audio_ativo  { *g = (*g + bonus * 0.25).min(1.0); }
            *g = (*g + bonus * 0.15).min(1.0);
        }
    }

    if let Ok(mut sw) = state.swap_manager.try_lock() {
        sw.importar_causal(causal_pairs);

        // Transitional closure: connect words sharing common neighbors
        let grafo = sw.grafo_palavras();
        let palavras: Vec<String> = grafo.keys()
            .filter(|w| w.chars().count() >= 3)
            .cloned().collect();
        for i in 0..palavras.len().min(25) {
            let a = &palavras[i];
            let b_idx = (i * 7 + 3) % palavras.len();
            let b = &palavras[b_idx];
            if a == b { continue; }
            let viz_a: std::collections::HashSet<String> = grafo
                .get(a).map(|v| v.iter().map(|(w, _)| w.clone()).collect())
                .unwrap_or_default();
            let viz_b: std::collections::HashSet<String> = grafo
                .get(b).map(|v| v.iter().map(|(w, _)| w.clone()).collect())
                .unwrap_or_default();
            let em_comum = viz_a.intersection(&viz_b).count();
            if em_comum > 0 {
                let ja_existe = grafo.get(a).map(|v| v.iter().any(|(w, _)| w == b)).unwrap_or(false);
                if !ja_existe {
                    let peso = (0.30 + em_comum as f32 * 0.08).clamp(0.30, 0.65);
                    sw.importar_causal(vec![(a.clone(), b.clone(), peso)]);
                    novas += 1;
                }
            }
        }
    }
    println!("   ✨ [N3 REM] {} novas sinapses | {} replays hipocampais ({} episódios salientes)",
        novas, replay, episodios.len());

    // P3 — treinar_semantico() durante sono REM.
    // Consolida o STDP neural com 3 épocas de 300 ciclos cada (~4.5s biológicos).
    // Durante o REM o cérebro humano replay e fortalece memórias — fazemos o mesmo.
    // Usa valências do estado atual para injetar corrente nos neurônios conceituais.
    if let Ok(mut sw) = state.swap_manager.try_lock() {
        let valencias = sw.valencias_palavras();
        if !valencias.is_empty() {
            let dt_s = 1.0_f32 / 200.0;
            let mut total_spikes = 0u32;
            let mut total_sinapses = 0usize;
            for _ in 0..3 {
                let (spikes, _, n_sin) = sw.treinar_semantico(300, dt_s, &valencias);
                total_spikes += spikes;
                total_sinapses = n_sin;
            }
            println!("   🌙 [N3 STDP] {} spikes | {} sinapses ativas após treino REM",
                total_spikes, total_sinapses);
        }
    }
}

/// N4 — Backup: persiste linguagem em JSON (vocabulário/grafo do swap).
fn fase_n4_backup(state: &crate::websocket::bridge::BrainState) {
    let (vocabulario, grafo) = if let Ok(mut sw) = state.swap_manager.try_lock() {
        (sw.valencias_palavras(), sw.grafo_palavras())
    } else {
        (std::collections::HashMap::<String,f32>::new(),
         std::collections::HashMap::<String,Vec<(String,f32)>>::new())
    };
    let backup = serde_json::json!({
        "selene_linguagem_v1": {
            "vocabulario":          vocabulario,
            "associacoes":          grafo,
            "frases_padrao":        state.frases_padrao,
            "grounding":            state.grounding,
            "emocao_palavras":      state.emocao_palavras,
            "auto_learn_contagem":  state.auto_learn_contagem,
        }
    });
    if let Ok(json) = serde_json::to_string(&backup) {
        let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let _ = std::fs::write("selene_linguagem.json", &json);
        println!("   💾 [N4] Backup linguagem salvo ({})", ts);
    }
}

/// Gerencia a conexão de cada cliente (browser) que se conecta à Selene
pub async fn handle_connection(
    ws: WebSocket,
    mut telemetry_rx: broadcast::Receiver<NeuralStatus>,
    brain: Arc<Mutex<BrainState>>,  // ← terceiro parâmetro adicionado (Opção B)
) {
    // Divide o WebSocket em transmissor (tx) e receptor (rx)
    let (mut ws_tx, mut ws_rx) = ws.split();

    println!("   ✅ Conexão WebSocket estabelecida.");

    // Buffer de contexto de conversa multi-turno (local por cliente).
    // Acumula palavras-chave das últimas respostas para manter coerência temática.
    // 150 palavras ≈ ~10-15 turnos de conversa retidos como contexto ativo.
    let mut conversa_ctx: Vec<String> = Vec::with_capacity(150);

    // ── SONO NOTURNO (00:00 – 05:00) ─────────────────────────────────────────
    // Canal para enviar eventos de sono ao WebSocket sem mover ws_tx para a task.
    let (sleep_tx, mut sleep_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    // Ciclo de fases durante as 5h de sono.
    // Ordem biológica: N1 → N2 → REM(longo) → N2 → REM → N1 → N4 backup
    // Durações em minutos. REM tem o maior bloco.
    const FASES_SONO: &[(&str, u64)] = &[
        ("N1 - Consolidação",  40),
        ("N2 - Poda",          30),
        ("N3 - REM",          100),   // bloco mais longo de sonho
        ("N2 - Poda",          30),
        ("N3 - REM",           90),   // segundo ciclo REM
        ("N1 - Consolidação",  20),
        ("N4 - Backup",        10),
    ];

    // ── Subscrição ao canal de pensamento espontâneo ─────────────────────────
    // Cada conexão WS recebe seu próprio receiver. Quando pensamento.rs dispara
    // um estímulo, este arm do select! acorda e gera + envia a resposta.
    let mut pensamento_rx = {
        let state = brain.lock().await;
        state.pensamento_tx.subscribe()
    };

    let brain_sleep = brain.clone();
    let sleep_tx_task = sleep_tx.clone();
    tokio::spawn(async move {
        let mut intervalo = tokio::time::interval(tokio::time::Duration::from_secs(30));
        let mut fase_idx: usize = 0;
        let mut inicio_fase = tokio::time::Instant::now();

        loop {
            intervalo.tick().await;
            let hora = chrono::Local::now().hour(); // 0–23
            let deve_dormir = hora < 5;             // 00:00 até 04:59

            let mut state = brain_sleep.lock().await;

            if deve_dormir && !state.dormindo {
                // ── Entrar em sono ──────────────────────────────────────────
                state.dormindo = true;
                fase_idx = 0;
                inicio_fase = tokio::time::Instant::now();
                let (nome, _) = FASES_SONO[0];
                state.fase_sono = nome.to_string();

                // Executa N1 imediatamente ao adormecer
                fase_n1_consolidar(&mut state);

                let ev = serde_json::json!({
                    "event": "sono",
                    "fase": nome,
                    "msg": "boa noite... indo dormir...",
                }).to_string();
                println!("💤 [SONO] Entrando em sono — {}", nome);
                let _ = sleep_tx_task.send(ev);

            } else if state.dormindo && deve_dormir {
                // ── Avançar fase se o tempo passou ──────────────────────────
                let elapsed_min = inicio_fase.elapsed().as_secs() / 60;
                if elapsed_min >= FASES_SONO[fase_idx].1 && fase_idx + 1 < FASES_SONO.len() {
                    fase_idx += 1;
                    inicio_fase = tokio::time::Instant::now();
                    let (nome, _) = FASES_SONO[fase_idx];
                    state.fase_sono = nome.to_string();

                    match nome {
                        "N1 - Consolidação" => fase_n1_consolidar(&mut state),
                        "N2 - Poda"         => fase_n2_podar(&mut state),
                        "N3 - REM"          => fase_n3_rem(&mut state),
                        "N4 - Backup"       => fase_n4_backup(&state),
                        _ => {}
                    }

                    let ev = serde_json::json!({
                        "event": "sono",
                        "fase": nome,
                    }).to_string();
                    println!("🌙 [SONO] Avançando → {}", nome);
                    let _ = sleep_tx_task.send(ev);
                }

            } else if !deve_dormir && state.dormindo {
                // ── Despertar natural às 05:00 ──────────────────────────────
                state.dormindo = false;
                state.fase_sono = String::new();
                state.aresta_contagem.clear();
                fase_idx = 0;

                let ev = serde_json::json!({
                    "event": "despertar",
                    "msg": "bom dia! acordei descansada.",
                }).to_string();
                println!("🔆 [SONO] 05:00 — Selene despertou naturalmente.");
                let _ = sleep_tx_task.send(ev);
            }
        }
    });

    loop {
        tokio::select! {
            // 0. Mensagens do ciclo de sono (eventos sono/despertar)
            Some(msg) = sleep_rx.recv() => {
                if ws_tx.send(Message::text(msg)).await.is_err() { break; }
            }

            // 0b. Pensamento espontâneo / curiosidade — Selene inicia sem input externo
            Ok(estimulo_raw) = pensamento_rx.recv() => {
                // Distingue pensamento espontâneo de curiosidade dirigida
                let (estimulo, e_curiosidade) = if estimulo_raw.starts_with("curiosidade:") {
                    (estimulo_raw.trim_start_matches("curiosidade:").to_string(), true)
                } else {
                    (estimulo_raw.clone(), false)
                };

                if let Ok(mut state) = brain.try_lock() {
                    // Clona o Arc do swap antes de usar state (evita lifetime entanglement)
                    let swap_arc_esp = state.swap_manager.clone();
                    let (grafo_snap, valencias_snap, scaffold_esp) = if let Ok(mut sw) = swap_arc_esp.try_lock() {
                        let tokens_input: Vec<String> = estimulo.to_lowercase()
                            .split(|c: char| !c.is_alphanumeric())
                            .filter(|t| t.len() > 1)
                            .map(|t| t.to_string())
                            .collect();
                        let (scaffold, _) = sw.template_scaffold(&tokens_input);
                        (sw.grafo_palavras(), sw.valencias_palavras(), scaffold)
                    } else {
                        (HashMap::new(), HashMap::new(), Vec::new())
                    };
                    let causal_vazio: HashMap<String, Vec<(String, f32)>> = HashMap::new();
                    if !state.dormindo && !grafo_snap.is_empty() {
                        let (step, _alerta, emocao) = state.atividade;
                        let dopa = state.neurotransmissores.0;
                        let sero = state.neurotransmissores.1;
                        let emocao_bias = state.emocao_bias;
                        // Habituação: sistema habituado busca pensamentos mais remotos/novos
                        let n_passos = ((state.n_passos_walk as f32 + state.habituation_nivel * 2.0) as usize).clamp(4, 12);
                        state.reply_count = state.reply_count.wrapping_add(1);
                        let diversity_seed = step ^ state.reply_count.wrapping_mul(2654435761);
                        let mut caminho_esp: Vec<String> = Vec::new();
                        let mut prefixo_esp: Vec<String> = Vec::new();
                        let mut ancora_esp: Option<String> = None;

                        let (d, s, n_nor) = state.neurotransmissores;
                        let cor = state.ultimo_estado_corpo[4];
                        let estado_af = narrativa::traduzir_estado(d, s, n_nor, cor, emocao);
                        let mut ctx_esp: Vec<String> = state.pensamento_consciente.iter().cloned().collect();
                        ctx_esp.push(estado_af.como_palavra().to_string());
                        if e_curiosidade { ctx_esp.push(estimulo.clone()); }

                        let reply = gerar_resposta_emergente(
                            &estimulo, diversity_seed, emocao, emocao_bias, n_passos,
                            dopa, sero,
                            &valencias_snap,
                            &grafo_snap,
                            &state.frases_padrao,
                            &state.indice_prefixo,
                            &state.ultimos_prefixos.iter().cloned().collect::<Vec<_>>(),
                            &ctx_esp,
                            &mut caminho_esp,
                            &state.emocao_palavras,
                            &mut prefixo_esp,
                            &state.grounding,
                            &causal_vazio,
                            &state.palavra_qvalores,
                            &mut ancora_esp,
                            &state.ultimo_estado_corpo,
                            &scaffold_esp,
                            &state.trigrama_cache,
                        );
                        if !reply.is_empty() {
                            let event_type = if e_curiosidade { "curiosidade_espontanea" } else { "pensamento_espontaneo" };
                            let (dop2, ser2, nor2) = state.neurotransmissores;
                            let formants = crate::encoding::phoneme::sentence_to_formants(&reply, dop2, ser2, nor2);
                            // Registra na autobiografia
                            let reply_trunc50: String = reply.chars().take(50).collect();
                            state.registrar_memoria(
                                format!("pensei espontaneamente sobre '{}': {}", estimulo, reply_trunc50),
                                emocao * 0.5,
                            );
                            drop(state);
                            println!("💭 [{}] '{}'  →  {}", event_type.to_uppercase(), estimulo, reply);
                            let msg = serde_json::json!({
                                "event":       event_type,
                                "estimulo":    estimulo,
                                "message":     reply,
                                "emotion":     emocao,
                                "estado":      estado_af.como_palavra(),
                            }).to_string();
                            if ws_tx.send(Message::text(msg)).await.is_err() { break; }
                            if let Ok(fj) = serde_json::to_string(&formants) {
                                let voz = format!(r#"{{"event":"voz_params","formants":{}}}"#, fj);
                                let _ = ws_tx.send(Message::text(voz)).await;
                            }
                        }
                    }
                }
            }

            // 1. Monitora o canal de broadcast da telemetria vinda do bridge.rs
            result = telemetry_rx.recv() => {
                match result {
                    Ok(status) => {
                        // Serializa o struct NeuralStatus para JSON
                        if let Ok(json) = serde_json::to_string(&status) {
                            // Tenta enviar para o navegador
                            if ws_tx.send(Message::text(json)).await.is_err() {
                                // Cliente provavelmente desconectou
                                break;
                            }
                        }
                    }

                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        log::warn!("⚠️ WebSocket detectou lag de {} mensagens. Continuando...", n);
                        // Continua recebendo sem quebrar o loop
                    }

                    Err(_) => {
                        // Canal fechado (sender foi dropado)
                        eprintln!("❌ Canal de telemetria foi fechado.");
                        break;
                    }
                }
            }

            // 2. Monitora mensagens enviadas pelo browser
            Some(result) = ws_rx.next() => {
                match result {
                    Ok(msg) => {
                        if let Ok(text) = msg.to_str() {
                            // Tenta parsear como JSON (para comandos estruturados)
                            if let Ok(json) = serde_json::from_str::<Value>(text) {
                                match json["action"].as_str() {
                                Some("ping") => {
                                    let pong = r#"{"event":"pong"}"#;
                                    let _ = ws_tx.send(Message::text(pong)).await;
                                }

                                // Envia snapshot do grafo neural para visualização
                                Some("vocab_request") => {
                                    let swap_arc_vr = brain.lock().await.swap_manager.clone();
                                    let (valencias, grafo) = if let Ok(mut sw) = swap_arc_vr.try_lock() {
                                        (sw.valencias_palavras(), sw.grafo_palavras())
                                    } else {
                                        (HashMap::new(), HashMap::new())
                                    };
                                    let mut palavras: Vec<(&String, f32)> = valencias
                                        .iter().map(|(k, v)| (k, v.abs())).collect();
                                    palavras.sort_by(|a, b| b.1.partial_cmp(&a.1)
                                        .unwrap_or(std::cmp::Ordering::Equal));
                                    let top: std::collections::HashSet<&str> = palavras.iter()
                                        .take(120).map(|(k, _)| k.as_str()).collect();

                                    let nodes: Vec<serde_json::Value> = top.iter().map(|&w| {
                                        let weight = valencias.get(w).map(|v| v.abs()).unwrap_or(0.005);
                                        serde_json::json!({"id": w, "weight": weight})
                                    }).collect();

                                    let mut links: Vec<serde_json::Value> = vec![];
                                    for (word, neighbors) in &grafo {
                                        if !top.contains(word.as_str()) { continue; }
                                        for (neighbor, weight) in neighbors {
                                            if *weight >= 0.08 {
                                                links.push(serde_json::json!({
                                                    "source": word,
                                                    "target": neighbor,
                                                    "weight": weight
                                                }));
                                            }
                                        }
                                    }

                                    let snapshot = serde_json::json!({
                                        "event": "vocab_snapshot",
                                        "nodes": nodes,
                                        "links": links,
                                        "total_palavras": valencias.len(),
                                        "total_assoc": grafo.len(),
                                    });
                                    let _ = ws_tx.send(Message::text(snapshot.to_string())).await;
                                }

                                Some("shutdown") => {
                                    println!("🛑 [SISTEMA] Shutdown solicitado pela interface neural.");
                                    brain.lock().await.shutdown_requested = true;
                                    let ack = r#"{"event":"shutdown_ack","msg":"Iniciando desligamento gracioso..."}"#;
                                    let _ = ws_tx.send(Message::text(ack)).await;
                                }

                                Some("force_sleep") => {
                                    let duration_min = json["duration_min"].as_u64().unwrap_or(30);
                                    {
                                        let mut st = brain.lock().await;
                                        if !st.dormindo {
                                            st.dormindo = true;
                                            st.fase_sono = "N1 - Consolidação".to_string();
                                            fase_n1_consolidar(&mut st);
                                        }
                                    }
                                    let ev = serde_json::json!({
                                        "event": "sono",
                                        "fase": "N1 - Consolidação",
                                        "msg": format!("dormindo por {} minutos... boa noite.", duration_min),
                                    }).to_string();
                                    println!("💤 [SONO] Forçado pela interface — {} min.", duration_min);
                                    let _ = sleep_tx.send(ev.clone());
                                    let _ = ws_tx.send(Message::text(ev)).await;

                                    // Agenda despertar automático
                                    let brain_wake = brain.clone();
                                    let sleep_tx_wake = sleep_tx.clone();
                                    tokio::spawn(async move {
                                        tokio::time::sleep(
                                            tokio::time::Duration::from_secs(duration_min * 60)
                                        ).await;
                                        let mut st = brain_wake.lock().await;
                                        if st.dormindo {
                                            st.dormindo = false;
                                            st.fase_sono = String::new();
                                            st.aresta_contagem.clear();
                                            let wake_ev = serde_json::json!({
                                                "event": "despertar",
                                                "msg": "despertei... descansada e pronta.",
                                            }).to_string();
                                            println!("🔆 [SONO] Despertar após {} min.", duration_min);
                                            let _ = sleep_tx_wake.send(wake_ev);
                                        }
                                    });
                                }

                                Some("toggle_sensor") => {
                                    let sensor  = json["sensor"].as_str().unwrap_or("");
                                    let active  = json["active"].as_bool().unwrap_or(false);
                                    let state   = brain.lock().await;
                                    match sensor {
                                        "audio" => state.sensor_flags.set_audio(active),
                                        "video" => state.sensor_flags.set_video(active),
                                        _ => log::warn!("[WS] Sensor desconhecido: {}", sensor),
                                    }
                                    let ack = serde_json::json!({
                                        "event": "sensor_ack",
                                        "sensor": sensor,
                                        "active": active
                                    }).to_string();
                                    let _ = ws_tx.send(Message::text(ack)).await;
                                    println!("[SENSOR] {} → {}", sensor, if active { "ATIVO" } else { "INATIVO" });
                                }

                                Some("chat") | Some("audio_chat") => {
                                    // Mensagem de chat vinda da interface mobile/desktop.
                                    // audio_chat: interface enviou bandas FFT fonéticas do texto digitado.
                                    // Isso fecha o loop áudio-primeiro: a entrada JÁ chega como sinal
                                    // de frequências, não como string pura.
                                    if json["action"].as_str() == Some("audio_chat") {
                                        if let Some(arr) = json["bands"].as_array() {
                                            let raw: Vec<f32> = arr.iter()
                                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                                .collect();
                                            if raw.len() >= 16 {
                                                let mut bands32 = [0f32; 32];
                                                for (i, &v) in raw.iter().take(32).enumerate() {
                                                    bands32[i] = v;
                                                }
                                                let audio_pat = bands_to_spike_pattern(&bands32);
                                                // Atualiza ultimo_padrao_audio com o sinal do usuário
                                                // (sobrescreve o sintético — fonte real tem prioridade)
                                                if let Ok(mut bs) = brain.try_lock() {
                                                    bs.ultimo_padrao_audio = audio_pat;
                                                }
                                            }
                                        }
                                    }
                                    // Aceita "transcript" (audio_chat), "text" (Python) ou "message" (web).
                                    let mensagem = json["transcript"].as_str()
                                        .or_else(|| json["text"].as_str())
                                        .or_else(|| json["message"].as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    println!("💬 [CHAT] Recebido (raw): «{}»", mensagem);
                                    if !mensagem.is_empty() {
                                        println!("💬 [CHAT] Aguardando lock do brain...");
                                        // Snapshot neural: clona Arc antes de bloquear brain.
                                        // USA lock().await (não try_lock) para garantir que o snapshot
                                        // seja válido mesmo que o main loop segure o swap durante tick_semantico.
                                        let swap_arc_chat = {
                                            let tmp = brain.lock().await;
                                            tmp.swap_manager.clone()
                                        };
                                        let (grafo_neural, valencias_neural, scaffold_chat, scaffold_id_chat) = {
                                            let mut s = swap_arc_chat.lock().await;
                                            let tokens_msg: Vec<String> = mensagem.to_lowercase()
                                                .split(|c: char| !c.is_alphanumeric())
                                                .filter(|t| t.len() > 1)
                                                .map(|t| t.to_string())
                                                .collect();
                                            let (scaffold, scaffold_id) = s.template_scaffold(&tokens_msg);
                                            (s.grafo_palavras(), s.valencias_palavras(), scaffold, scaffold_id)
                                        };
                                        let causal_vazio_chat: HashMap<String, Vec<(String, f32)>> = HashMap::new();
                                        let mut state = brain.lock().await;

                                        // ── Wake-on-interaction ──────────────────────────────
                                        if state.dormindo {
                                            state.dormindo = false;
                                            state.fase_sono = String::new();
                                            let mensagens_despertar = [
                                                "mmm... estava sonhando... o que houve?",
                                                "quem está me chamando? acordei...",
                                                "hmm... aqui estou, ainda com sono...",
                                                "acordei! o que você precisa?",
                                            ];
                                            let idx = mensagem.len() % mensagens_despertar.len();
                                            let wake_msg = mensagens_despertar[idx];
                                            let ev_wake = serde_json::json!({
                                                "event": "despertar",
                                                "msg": wake_msg,
                                            }).to_string();
                                            let _ = sleep_tx.send(ev_wake);
                                            let resp_wake = serde_json::json!({
                                                "event": "chat_reply",
                                                "message": wake_msg,
                                                "emotion": 0.1,
                                                "arousal": 0.4,
                                            }).to_string();
                                            let _ = ws_tx.send(Message::text(resp_wake)).await;
                                            println!("🔆 [SONO] Despertou por interação de chat.");
                                        }

                                        println!("💬 [CHAT] Lock adquirido. Gerando resposta...");
                                        let (dopa, sero, _nor) = state.neurotransmissores;
                                        let (step, alerta, emocao) = state.atividade;

                                        // ── SÍNTESE FONÉTICA: texto → padrão auditivo ────────
                                        // Quando não há microfone ativo, convertemos o texto em
                                        // um padrão de 32 bandas FFT sintético via formantes PT-BR.
                                        // Isso garante que TODA mensagem (digitada ou transcrita)
                                        // seja processada como experiência auditiva real pela Selene,
                                        // habilitando grounding som→símbolo mesmo sem microfone.
                                        {
                                            let nor_val = state.neurotransmissores.2;
                                            let bands = texto_para_bandas_fft(&mensagem, dopa, sero, nor_val);
                                            let audio_pat = bands_to_spike_pattern(&bands);
                                            // Só atualiza o padrão de áudio se nenhum sinal real
                                            // chegou recentemente (evita sobrescrever mic real).
                                            // Heurística: padrão zerado = nenhum áudio real recente.
                                            let ultimo = state.ultimo_padrao_audio;
                                            if !crate::encoding::spike_codec::is_active(&ultimo) {
                                                state.ultimo_padrao_audio = audio_pat;
                                            }
                                            // Grounding fonético para todas as palavras da mensagem
                                            let palavras_msg: Vec<String> = mensagem
                                                .to_lowercase()
                                                .split(|c: char| !c.is_alphabetic()
                                                    && !"áéíóúâêôãõçàü".contains(c))
                                                .filter(|w| w.len() >= 2)
                                                .map(|w| w.to_string())
                                                .collect();
                                            if !palavras_msg.is_empty() {
                                                let vpad = state.ultimo_padrao_visual;
                                                state.grounding_bind(
                                                    &palavras_msg,
                                                    vpad,
                                                    audio_pat,
                                                    emocao,
                                                    alerta,
                                                    0.0,
                                                );
                                                log::debug!("[SinteseAudio] {} palavras com grounding fonético sintético", palavras_msg.len());
                                            }
                                        }

                                        // ── INNER SPEECH: áudio sintético → ativa swap (txt→áudio→processamento) ─
                                        // Fecha o primeiro elo do loop: o padrão auditivo gerado da mensagem
                                        // reforça os conceitos via importar_causal no swap, como se Selene
                                        // "ouvisse" internamente as palavras antes de formular a resposta.
                                        {
                                            let audio_pat_is = state.ultimo_padrao_audio;
                                            if crate::encoding::spike_codec::is_active(&audio_pat_is) {
                                                let energia_is = audio_pat_is.iter()
                                                    .map(|b| b.count_ones()).sum::<u32>() as f32 / 512.0;
                                                let palavras_is: Vec<String> = mensagem
                                                    .to_lowercase()
                                                    .split(|c: char| !c.is_alphabetic()
                                                        && !"áéíóúâêôãõçàü".contains(c))
                                                    .filter(|w| w.len() >= 2 && !STOP_WORDS.contains(w))
                                                    .map(|w| w.to_string())
                                                    .collect();
                                                if let Ok(mut sw) = state.swap_manager.try_lock() {
                                                    for palavra in &palavras_is {
                                                        sw.aprender_conceito(palavra, energia_is * 0.1);
                                                    }
                                                    for par in palavras_is.windows(2) {
                                                        sw.importar_causal(vec![(
                                                            par[0].clone(), par[1].clone(),
                                                            energia_is * 0.15,
                                                        )]);
                                                    }
                                                }
                                                log::debug!("[InnerSpeech] {} conceitos ativados (energia={:.2})",
                                                    palavras_is.len(), energia_is);
                                            }
                                        }

                                        // ── HIPÓTESES: testa predições do turno anterior ──
                                        // Confronta o que Selene previu com o que o usuário realmente disse.
                                        // RPE positivo = previsão correta → reforça grounding das palavras previstas.
                                        // RPE negativo = previsão errada → enfraquece associações equivocadas.
                                        {
                                            let input_tokens_hip: Vec<String> = mensagem
                                                .to_lowercase()
                                                .split(|c: char| !c.is_alphabetic()
                                                    && !"áéíóúâêôãõçàü".contains(c))
                                                .filter(|w| w.len() >= 3
                                                    && !STOP_WORDS.contains(w))
                                                .map(|w| w.to_string())
                                                .collect();
                                            let val_clone = if let Ok(sw) = state.swap_manager.try_lock() {
                                                sw.valencias_palavras()
                                            } else { HashMap::new() };

                                            // ── Wernicke: atualiza comprehension_score ──────────
                                            // Proporção de tokens conhecidos (com valência no grafo)
                                            // → score alto = Selene entendeu o input.
                                            if !input_tokens_hip.is_empty() {
                                                let n_known = input_tokens_hip.iter()
                                                    .filter(|w| val_clone.contains_key(w.as_str()))
                                                    .count();
                                                let familiarity = n_known as f32 / input_tokens_hip.len() as f32;
                                                // EMA: 70% histórico + 30% novo input
                                                state.wernicke_comprehension =
                                                    state.wernicke_comprehension * 0.7 + familiarity * 0.3;
                                            }

                                            // ACC social pain: punição/rejeição sinalizada via RPE negativo
                                            // Propagada ao cingulado via state para que o server possa usar
                                            if state.ultimo_rpe < -0.3 {
                                                state.acc_social_pain = (state.acc_social_pain
                                                    + (-state.ultimo_rpe) * 0.2).clamp(0.0, 1.0);
                                            } else {
                                                state.acc_social_pain *= 0.97; // decai naturalmente
                                            }

                                            let rpe_hip = state.hypothesis_engine
                                                .testar(&input_tokens_hip, &val_clone);
                                            if rpe_hip.abs() > 0.05 {
                                                state.grounding_rpe(rpe_hip);
                                            }
                                        }
                                        // ── Recuperação episódica ativa ──────────────────────
                                        // Extrai tokens do input, busca episódios relevantes no
                                        // historico_episodico e injeta suas palavras no neural_context.
                                        // Isso faz Selene "lembrar" ativamente o que viveu quando
                                        // o assunto é mencionado — não apenas durante o sono.
                                        {
                                            let input_tokens_ep: std::collections::HashSet<String> = mensagem
                                                .to_lowercase()
                                                .split(|c: char| !c.is_alphabetic()
                                                    && !"áéíóúâêôãõçàü".contains(c))
                                                .filter(|w| w.len() >= 3)
                                                .map(|w| w.to_string())
                                                .collect();

                                            // Pontua cada episódio pelo overlap de palavras com o input
                                            let mut melhor_score = 0usize;
                                            let mut melhor_episodio: Option<Vec<String>> = None;
                                            for ev in state.historico_episodico.iter().rev().take(100) {
                                                let overlap = ev.palavras.iter()
                                                    .filter(|w| input_tokens_ep.contains(w.as_str()))
                                                    .count();
                                                if overlap > melhor_score {
                                                    melhor_score = overlap;
                                                    melhor_episodio = Some(ev.palavras.clone());
                                                }
                                            }

                                            // Injeta palavras do melhor episódio no neural_context
                                            // (com threshold mínimo de 1 palavra em comum)
                                            if melhor_score >= 1 {
                                                if let Some(ep_palavras) = melhor_episodio {
                                                    for w in ep_palavras.iter().take(4) {
                                                        if !state.neural_context.contains(w) {
                                                            state.neural_context.push_back(w.clone());
                                                        }
                                                    }
                                                    while state.neural_context.len() > 20 {
                                                        state.neural_context.pop_front();
                                                    }
                                                    log::debug!(
                                                        "[EpisodicRecall] overlap={} palavras injetadas: {:?}",
                                                        melhor_score,
                                                        ep_palavras.iter().take(4).collect::<Vec<_>>()
                                                    );
                                                }
                                            }
                                        }

                                        // ── EXTRAÇÃO CAUSAL DA MENSAGEM COMPLETA ─────────────
                                        // A detecção em learn_frase só vê tokens isolados.
                                        // Aqui varremos a sentença completa — captures "A→B porque C"
                                        // mesmo que os tokens não cheguem via learn_frase.
                                        {
                                            const CAUSAIS_CHAT: &[&str] = &[
                                                "porque", "então", "causa", "logo",
                                                "portanto", "resulta", "leva", "gera",
                                                "implica", "deriva", "consequência",
                                                "assim", "daí", "pois", "afinal",
                                                "permite", "faz", "torna", "cria",
                                            ];
                                            let palavras_chat: Vec<String> = mensagem
                                                .to_lowercase()
                                                .split(|c: char| !c.is_alphabetic()
                                                    && !"áéíóúâêôãõçàü".contains(c))
                                                .filter(|w| !w.is_empty())
                                                .map(|w| w.to_string())
                                                .collect();
                                            if let Some(pos_c) = palavras_chat.iter()
                                                .position(|w| CAUSAIS_CHAT.contains(&w.as_str()))
                                            {
                                                let causa = palavras_chat[..pos_c].iter().rev()
                                                    .find(|w| w.len() >= 3
                                                        && !CAUSAIS_CHAT.contains(&w.as_str()))
                                                    .cloned();
                                                let efeito = palavras_chat[pos_c + 1..].iter()
                                                    .find(|w| w.len() >= 3
                                                        && !CAUSAIS_CHAT.contains(&w.as_str()))
                                                    .cloned();
                                                if let (Some(c), Some(e)) = (causa, efeito) {
                                                    // Causal → sinapse de alta prioridade no swap
                                                    let swap_arc_caus = state.swap_manager.clone();
                                                    drop(state);
                                                    let mut sw = swap_arc_caus.lock().await;
                                                    sw.importar_causal(vec![(c.clone(), e.clone(), 0.65)]);
                                                    log::debug!("[Causal/chat] {} → {} (sinapse neural)", c, e);
                                                    drop(sw);
                                                    state = brain.lock().await;
                                                }
                                            }
                                        }

                                        // Conceitos mais ativos no momento — P1-A: injeta no contexto do walk.
                                        // Antes só eram usados para log. Agora as palavras com maior
                                        // ativação populacional ENTRAM no contexto do walk, fazendo
                                        // a resposta emergir do estado neural presente, não só do input.
                                        let (ativos_str, conceitos_ativos_ctx): (String, Vec<String>) = {
                                            if let Ok(swap) = state.swap_manager.try_lock() {
                                                let top = swap.conceitos_ativos_top(6);
                                                if top.is_empty() {
                                                    (String::new(), Vec::new())
                                                } else {
                                                    let s = top.iter()
                                                        .map(|(w, a)| format!("{}({:.0}%)", w, a * 100.0))
                                                        .collect::<Vec<_>>()
                                                        .join(", ");
                                                    // Filtra stop-words e pega as 4 mais ativas para contexto
                                                    let ctx: Vec<String> = top.iter()
                                                        .filter(|(w, _)| !STOP_WORDS.contains(&w.as_str()) && w.len() >= 3)
                                                        .take(4)
                                                        .map(|(w, _)| w.clone())
                                                        .collect();
                                                    (s, ctx)
                                                }
                                            } else {
                                                (String::new(), Vec::new())
                                            }
                                        };
                                        let (n_vocab_log, n_grafo_log) = if let Ok(sw) = state.swap_manager.try_lock() {
                                            (sw.palavra_para_id.len(), sw.sinapses_conceito.len())
                                        } else { (0, 0) };
                                        println!("💬 [CHAT] Estado: step={} emocao={:.2} dopa={:.2} vocab={} sinapses={} frases={}{}",
                                            step, emocao, dopa,
                                            n_vocab_log, n_grafo_log,
                                            state.frases_padrao.len(),
                                            if ativos_str.is_empty() { String::new() } else { format!(" | ativos=[{}]", ativos_str) });

                                        // ── AUTO-REFERÊNCIA: injeta narrativa no contexto ─────
                                        // Se a mensagem pergunta sobre a Selene, adiciona auto-
                                        // descrição introspectiva ao contexto do walk — as palavras
                                        // da narrativa "colorem" a resposta emergente com identidade.
                                        if narrativa::e_auto_referencia(&mensagem) {
                                            let (d, s, n) = state.neurotransmissores;
                                            let cor = state.ultimo_estado_corpo[4];
                                            let emocao_now = emocao;
                                            let estado_af = narrativa::traduzir_estado(d, s, n, cor, emocao_now);
                                            let mems: Vec<String> = state.memorias_recentes_str(3);
                                            let n_vocab_ad = if let Ok(sw) = state.swap_manager.try_lock() {
                                                sw.palavra_para_id.len()
                                            } else { 0 };
                                            let auto_desc = narrativa::auto_descrever(
                                                &state.ego.tracos,
                                                n_vocab_ad,
                                                &mems,
                                                &state.pensamento_consciente,
                                                &estado_af,
                                            );
                                            // Tokeniza a auto-descrição e injeta no contexto
                                            for tok in auto_desc.split_whitespace()
                                                .filter(|t| t.len() >= 4)
                                                .take(8)
                                            {
                                                let tok_s = tok.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
                                                if !tok_s.is_empty() { conversa_ctx.push(tok_s); }
                                            }
                                            if conversa_ctx.len() > 150 { conversa_ctx.drain(..50); }
                                            log::debug!("🪞 [AUTO-REF] narrativa injetada: {}", auto_desc);
                                        }

                                        // ── Registra memória autobiográfica de cada conversa ──
                                        // A cada turno de chat marcante (|valência| > 0.25),
                                        // persiste na autobiografia — ela "lembra" que conversou.
                                        // Registrado após calcular valence (logo abaixo).

                                        // Busca valência: varredura palavra-a-palavra na mensagem
                                        let msg_lower = mensagem.to_lowercase();
                                        let (valence, palavra_chave) = {
                                            let mut best_val = valencias_neural
                                                .get(&msg_lower).copied().unwrap_or(0.0);
                                            let mut best_word: Option<String> = if best_val != 0.0 {
                                                Some(msg_lower.clone())
                                            } else { None };
                                            for token in msg_lower.split_whitespace() {
                                                if let Some(&v) = valencias_neural.get(token) {
                                                    if v.abs() > best_val.abs() {
                                                        best_val = v;
                                                        best_word = Some(token.to_string());
                                                    }
                                                }
                                            }
                                            (best_val, best_word)
                                        };

                                        // Mistura emoção atual com valência da palavra (α=0.4).
                                        // Fix 6: ressonância mirror adiciona empatia encarnada —
                                        // quando Selene "sente" o que o usuário descreve, isso
                                        // colore levemente a resposta com o mesmo tom emocional.
                                        let empatia_mirror = state.mirror_resonance * valence * 0.15;
                                        let emocao_resposta = (emocao * 0.6 + valence * 0.4 + empatia_mirror)
                                            .clamp(-1.0, 1.0);

                                        // ── TRAÇOS → COMPORTAMENTO ───────────────────────
                                        // Os traços de personalidade acumulados modulam
                                        // diretamente os parâmetros do walk:
                                        //   curiosa    → walk mais profundo, limiar curiosidade menor
                                        //   cautelosa  → walk mais curto, viés emocional neutro
                                        //   reflexiva  → walk ligeiramente mais longo
                                        let t_curiosa  = state.ego.tracos.iter().find(|(n,_)| n == "curiosa")
                                            .map(|(_, v)| *v).unwrap_or(0.5);
                                        let t_cautelosa = state.ego.tracos.iter().find(|(n,_)| n == "cautelosa")
                                            .map(|(_, v)| *v).unwrap_or(0.3);
                                        let t_reflexiva = state.ego.tracos.iter().find(|(n,_)| n == "reflexiva")
                                            .map(|(_, v)| *v).unwrap_or(0.5);

                                        // Habituação límbica: amígdala habituada → sistema busca novidade.
                                        // Aumenta comprimento do walk e diversidade vocabular.
                                        let hab = state.habituation_nivel;
                                        // Broca + Wernicke modulam n_passos:
                                        //   fluência alta + compreensão alta → resposta mais rica
                                        //   compreensão baixa → walk mais curto (não entendeu bem)
                                        let lang_delta = (state.broca_fluency * 0.6
                                            + state.wernicke_comprehension * 0.4 - 0.5) * 4.0;
                                        // ACC: conflito alto → walk mais cauteloso (-adj_factor)
                                        let acc_adj = state.acc_conflict * (-2.0);
                                        let n_passos = ((state.n_passos_walk as f32
                                            + t_curiosa * 2.5
                                            - t_cautelosa * 2.0
                                            + t_reflexiva * 1.0
                                            + hab * 3.0
                                            + lang_delta
                                            + acc_adj) as usize).clamp(4, 18);
                                        // Cautelosa amorteça o viés emocional → respostas mais neutras
                                        // OFC value_bias: contexto associado a recompensa → levanta emocao
                                        // Ocitocina alta: tom mais acolhedor → pequeno bias positivo
                                        // ACC social_pain: dor social recente → leve viés negativo
                                        let ofc_contrib = state.ofc_value_bias * 0.2;
                                        let oxt_contrib = (state.oxytocin_level - 0.5) * 0.15;
                                        let pain_contrib = -state.acc_social_pain * 0.15;
                                        let emocao_bias = (state.emocao_bias * (1.0 - t_cautelosa * 0.5)
                                            + ofc_contrib + oxt_contrib + pain_contrib)
                                            .clamp(-1.0, 1.0);
                                        // Curiosa reduz limiar de disparo de pergunta autônoma
                                        let curiosity_threshold = if t_curiosa > 0.7 { 0.55 } else { 0.75 };

                                        state.reply_count = state.reply_count.wrapping_add(1);
                                        let diversity_seed = step ^ state.reply_count.wrapping_mul(6364136223846793005);
                                        // caminho_local evita conflito de borrow entre &state.* e &mut state.*
                                        let mut caminho_local: Vec<String> = Vec::new();
                                        let mut prefixo_buf: Vec<String> = Vec::new();
                                        let mut ancora_log: Option<String> = None;
                                        let ctx_para_log = {
                                            let mut ctx = conversa_ctx.clone();
                                            ctx.extend(state.neural_context.iter().cloned());
                                            ctx.extend(state.pensamento_consciente.iter().cloned().take(5));
                                            ctx.extend(state.frontal_goal_words.iter().cloned());
                                            // P1-A: conceitos neuralmente ativos entram no contexto
                                            ctx.extend(conceitos_ativos_ctx.iter().cloned());
                                            // P3.4: estado emocional narrativo como cor do vocabulário
                                            {
                                                let (d_n, s_n, nor_n) = state.neurotransmissores;
                                                let cor_n = state.ultimo_estado_corpo[4];
                                                let estado_n = narrativa::traduzir_estado(
                                                    d_n, s_n, nor_n, cor_n, emocao_resposta);
                                                ctx.push(estado_n.como_palavra().to_string());
                                            }
                                            ctx
                                        };
                                        let reply = gerar_resposta_emergente(
                                            &mensagem, diversity_seed, emocao_resposta,
                                            emocao_bias, n_passos,
                                            dopa, sero,
                                            &valencias_neural,
                                            &grafo_neural,
                                            &state.frases_padrao,
                                            &state.indice_prefixo,
                                            &state.ultimos_prefixos.iter().cloned().collect::<Vec<_>>(),
                                            &{
                                                let mut ctx = ctx_para_log.clone();
                                                if state.pensamento_step % 7 == 0 {
                                                    if let Some(w) = state.pensamento_inconsciente.front() {
                                                        ctx.push(w.clone());
                                                    }
                                                }
                                                ctx
                                            },
                                            &mut caminho_local,
                                            &state.emocao_palavras,
                                            &mut prefixo_buf,
                                            &state.grounding,
                                            &causal_vazio_chat,
                                            &state.palavra_qvalores,
                                            &mut ancora_log,
                                            &state.ultimo_estado_corpo,
                                            &scaffold_chat,
                                            &state.trigrama_cache,
                                        );

                                        // ── LOG DE AJUSTE FINO ──────────────────────────────
                                        {
                                            let qbias_top: Vec<(String, f32)> = {
                                                let mut v: Vec<(String, f32)> = state.palavra_qvalores
                                                    .iter()
                                                    .filter(|(_, &q)| q.abs() > 0.05)
                                                    .map(|(w, &q)| (w.clone(), q))
                                                    .collect();
                                                v.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
                                                v.truncate(3);
                                                v
                                            };
                                            log_turno(
                                                &mensagem,
                                                ancora_log.as_deref(),
                                                n_passos,
                                                emocao_resposta,
                                                dopa, sero,
                                                valencias_neural.len(),
                                                grafo_neural.len(),
                                                &caminho_local,
                                                &prefixo_buf,
                                                &reply,
                                                reply.is_empty(),
                                                &ctx_para_log,
                                                &qbias_top,
                                                state.ultimo_rpe,
                                            );
                                        }

                                        // ── Template feedback loop ──────────────────────────
                                        // Reforça o template que gerou o scaffold com os tokens
                                        // reais da resposta — fecha o ciclo de aprendizado:
                                        // reconhecer → scaffoldar → usar → histórico de slots.
                                        if !reply.is_empty() {
                                            if let Some(tid) = scaffold_id_chat {
                                                let reply_tokens: HashMap<usize, String> = reply
                                                    .to_lowercase()
                                                    .split(|c: char| !c.is_alphanumeric())
                                                    .filter(|t| t.len() > 1)
                                                    .map(|t| t.to_string())
                                                    .enumerate()
                                                    .take(8)
                                                    .collect();
                                                let t_tmpl = std::time::SystemTime::now()
                                                    .duration_since(std::time::UNIX_EPOCH)
                                                    .unwrap_or_default()
                                                    .as_secs_f64();
                                                if let Ok(mut sw) = state.swap_manager.try_lock() {
                                                    sw.template_store.usar(tid, &reply_tokens, true, t_tmpl);
                                                }
                                            }
                                        }

                                        state.ultimos_prefixos.push_back(prefixo_buf);
                                        if state.ultimos_prefixos.len() > 5 { state.ultimos_prefixos.pop_front(); }
                                        state.ultimo_caminho_walk = caminho_local.clone();
                                        // Atualiza contagem de arestas usadas (para consolidação noturna)
                                        let caminho = caminho_local;
                                        for i in 0..caminho.len().saturating_sub(1) {
                                            let par = (caminho[i].clone(), caminho[i+1].clone());
                                            *state.aresta_contagem.entry(par).or_insert(0) += 1;
                                        }

                                        // ── Memória autobiográfica: turno de chat marcante ────
                                        if valence.abs() > 0.25 && !reply.is_empty() {
                                            let ancora_str = ancora_log.as_deref().unwrap_or("?");
                                            let valencia_mem = (emocao_resposta * 0.6 + valence * 0.4).clamp(-1.0, 1.0);
                                            let reply_trunc: String = reply.chars().take(60).collect();
                                            let descricao = format!(
                                                "conversei sobre '{}': {}",
                                                ancora_str,
                                                reply_trunc
                                            );
                                            state.registrar_memoria(descricao, valencia_mem);
                                        }

                                        // Fix 1: LTD/LTP via RPE — modula pesos das sinapses percorridas.
                                        let rpe = state.ultimo_rpe;
                                        if rpe.abs() > 0.25 {
                                            let delta = rpe.signum() * 0.02;
                                            let caminho_rpe = state.ultimo_caminho_walk.clone();
                                            if let Ok(mut sw) = state.swap_manager.try_lock() {
                                                for i in 0..caminho_rpe.len().saturating_sub(1) {
                                                    let a = &caminho_rpe[i];
                                                    let b = &caminho_rpe[i+1];
                                                    // Find canonical neurons and adjust weight
                                                    if let (Some(pop_a), Some(pop_b)) = (
                                                        sw.palavra_para_id.get(a).and_then(|p| p.first().copied()),
                                                        sw.palavra_para_id.get(b).and_then(|p| p.first().copied()),
                                                    ) {
                                                        if let Some(w) = sw.sinapses_conceito.get_mut(&(pop_a, pop_b)) {
                                                            *w = (*w + delta).clamp(0.01, 1.0);
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        // Decaimento temporal a cada ~500 respostas
                                        if state.reply_count % 500 == 0 {
                                            if let Ok(mut sw) = state.swap_manager.try_lock() {
                                                for w in sw.sinapses_conceito.values_mut() {
                                                    *w = (*w * 0.995).max(0.01);
                                                }
                                                sw.sinapses_conceito.retain(|_, p| *p > 0.01);
                                                // Decai e poda templates inativos
                                                let t_decay = std::time::SystemTime::now()
                                                    .duration_since(std::time::UNIX_EPOCH)
                                                    .unwrap_or_default()
                                                    .as_secs_f64();
                                                sw.template_store.tick_decay(t_decay);
                                            }
                                        }

                                        // P1-C: aresta_contagem → reforço STDP periódico.
                                        // A cada 50 respostas, reforça no swap as arestas mais
                                        // traversadas pelo walk — Hebbian "use it or lose it":
                                        // caminhos frequentes ficam mais fortes no grafo neural.
                                        if state.reply_count % 50 == 0 && !state.aresta_contagem.is_empty() {
                                            if let Ok(mut sw) = state.swap_manager.try_lock() {
                                                // Ordena por contagem decrescente, pega top-20
                                                let mut top_arestas: Vec<_> = state.aresta_contagem
                                                    .iter()
                                                    .collect();
                                                top_arestas.sort_by(|a, b| b.1.cmp(a.1));
                                                let pares: Vec<(String, String, f32)> = top_arestas
                                                    .iter()
                                                    .take(20)
                                                    .map(|((w1, w2), cnt)| {
                                                        // Peso proporcional à frequência, máx 0.3
                                                        let peso = ((**cnt as f32) * 0.01).clamp(0.02, 0.3);
                                                        (w1.clone(), w2.clone(), peso)
                                                    })
                                                    .collect();
                                                if !pares.is_empty() {
                                                    sw.importar_causal(pares);
                                                    log::debug!("[ArestaSTDP] {} arestas reforçadas (reply={})",
                                                        top_arestas.len().min(20), state.reply_count);
                                                }
                                            }
                                            // Decai contadores para que arestas antigas não dominem para sempre
                                            for v in state.aresta_contagem.values_mut() {
                                                *v = v.saturating_sub(1);
                                            }
                                            state.aresta_contagem.retain(|_, v| *v > 0);
                                        }

                                        // Graph versioning: snapshot do swap a cada 200 respostas
                                        if state.reply_count % 200 == 0 {
                                            let slot = (state.reply_count / 200) % 3;
                                            let path = format!("selene_swap_snap_{}.json", slot);
                                            let snap_saved = if let Ok(sw) = state.swap_manager.try_lock() {
                                                sw.salvar_estado(&path).is_ok()
                                            } else { false };
                                            if snap_saved {
                                                log::info!("📸 [SNAPSHOT] swap salvo → {} (slot {})", path, slot);
                                            }
                                        }

                                        state.ultima_atividade = std::time::Instant::now();

                                        // Auto-learn progressivo → valência via swap_manager
                                        let swap_arc_al2 = state.swap_manager.clone();
                                        let auto_tokens: Vec<(String, f32)> = msg_lower.split_whitespace()
                                            .filter(|t| t.len() > 2 && t.chars().all(|c| c.is_alphabetic() || "áéíóúâêôãõçàü".contains(c)))
                                            .map(|t| {
                                                let contagem = state.auto_learn_contagem
                                                    .entry(t.to_string()).or_insert(0);
                                                *contagem += 1;
                                                let escala = (*contagem as f32 * 0.5).min(1.5);
                                                let val = (emocao_resposta * 0.15 * escala).clamp(-0.45, 0.45);
                                                (t.to_string(), val)
                                            })
                                            .collect();
                                        if let Ok(mut sw) = swap_arc_al2.try_lock() {
                                            for (tok, val) in auto_tokens {
                                                sw.aprender_conceito(&tok, val);
                                            }
                                        }
                                        println!("💬 [CHAT] Reply gerado: «{}»", reply);

                                        // ── HIPÓTESES: formula predições ──────────────────────
                                        {
                                            let ctx_hip: Vec<String> = {
                                                let mut c = conversa_ctx.clone();
                                                c.extend(state.neural_context.iter().cloned());
                                                c
                                            };
                                            let swap_arc_hyp = state.swap_manager.clone();
                                            let (grafo_ref, val_ref) = if let Ok(mut sw) = swap_arc_hyp.try_lock() {
                                                (sw.grafo_palavras(), sw.valencias_palavras())
                                            } else { (HashMap::new(), HashMap::new()) };
                                            state.hypothesis_engine.formular(
                                                &ctx_hip, &grafo_ref, &val_ref,
                                                emocao_resposta, STOP_WORDS,
                                            );
                                            // Registra última resposta para teste no próximo turno
                                            state.hypothesis_engine.ultimo_reply = reply.clone();
                                            // Observa padrão comportamental próprio
                                            let chave_hip = conversa_ctx.iter().rev()
                                                .find(|w| !STOP_WORDS.contains(&w.as_str())
                                                    && w.len() >= 3)
                                                .cloned()
                                                .unwrap_or_default();
                                            let reply_resumido: String = reply
                                                .split_whitespace().take(4)
                                                .collect::<Vec<_>>().join(" ");
                                            state.hypothesis_engine.observar_comportamento(
                                                &chave_hip, &reply_resumido,
                                            );

                                            // ── Hipóteses confiáveis → STDP ────────────────────
                                            // Hipóteses semânticas com alta confiança (≥10 testes,
                                            // taxa >65%) são equivalentes a associações aprendidas:
                                            // promovê-las como pares causais no swap fecha o loop
                                            // entre predição e memória sináptica.
                                            {
                                                let confiaveis = state.hypothesis_engine.hipoteses_confiaveis();
                                                if !confiaveis.is_empty() {
                                                    let pares: Vec<(String, String, f32)> = confiaveis.iter()
                                                        .flat_map(|h| h.premissas.iter()
                                                            .map(|p| (p.clone(), h.conclusao.clone(), h.confianca * 0.4))
                                                            .collect::<Vec<_>>()
                                                        )
                                                        .collect();
                                                    if let Ok(mut sw) = state.swap_manager.try_lock() {
                                                        sw.importar_causal(pares);
                                                    }
                                                }
                                            }

                                            // ── Gaps de conhecimento → curiosidade ────────────
                                            // Palavras com poucas associações são lacunas: injetá-las
                                            // no neural_context faz Selene "pensar" no que não sabe
                                            // → aumenta chance de perguntas autônomas relevantes.
                                            if state.reply_count % 5 == 0 {
                                                // Coleta owned para liberar o borrow de hypothesis_engine
                                                // antes de mutar neural_context.
                                                let gap_palavras: Vec<String> = state.hypothesis_engine
                                                    .gaps_conhecimento()
                                                    .iter()
                                                    .take(2)
                                                    .filter_map(|h| h.premissas.first().cloned())
                                                    .collect();
                                                for palavra in gap_palavras {
                                                    if !state.neural_context.contains(&palavra) {
                                                        state.neural_context.push_back(palavra);
                                                        if state.neural_context.len() > 20 {
                                                            state.neural_context.pop_front();
                                                        }
                                                    }
                                                }
                                            }

                                            // ── Próximo tópico previsto → bias de contexto ─────
                                            // Se o motor de hipóteses tem uma predição confiante
                                            // sobre o que vem a seguir, injeta no neural_context
                                            // para que o graph-walk já comece orientado a ela.
                                            if let Some(proximo) = state.hypothesis_engine.proximo_topico_previsto() {
                                                let proximo_owned = proximo.to_string();
                                                if !state.neural_context.contains(&proximo_owned) {
                                                    state.neural_context.push_front(proximo_owned);
                                                    if state.neural_context.len() > 20 {
                                                        state.neural_context.pop_back();
                                                    }
                                                }
                                            }

                                            // Log de diagnóstico (a cada 10 replies)
                                            if state.reply_count % 10 == 0 {
                                                println!("🧠 [HIP] {}", state.hypothesis_engine.resumo());
                                            }
                                        }

                                        // ── Introspecção: detecta coincidência entre pensamentos internos e tópico ──
                                        // Se alguma palavra do pensamento_consciente atual aparece na mensagem,
                                        // registra no ego com marcação especial — o hypothesis_engine vai
                                        // eventualmente aprender esse padrão e a auto-referência emerge.
                                        {
                                            let tokens_msg: std::collections::HashSet<&str> = msg_lower
                                                .split_whitespace()
                                                .filter(|t| t.len() >= 3 && !STOP_WORDS.contains(t))
                                                .collect();
                                            let pensamento_ativo: Vec<String> = state
                                                .pensamento_consciente
                                                .iter()
                                                .cloned()
                                                .collect();
                                            let coincidencias: Vec<&String> = pensamento_ativo
                                                .iter()
                                                .filter(|w| tokens_msg.contains(w.as_str()))
                                                .collect();
                                            if !coincidencias.is_empty() {
                                                let palavras = coincidencias
                                                    .iter()
                                                    .map(|w| w.as_str())
                                                    .collect::<Vec<_>>()
                                                    .join(", ");
                                                let nota = format!(
                                                    "pensamento↔fala: [{}] surgiu na conversa",
                                                    palavras
                                                );
                                                state.ego.pensamentos_recentes.push_back(nota);
                                                if state.ego.pensamentos_recentes.len() > 10 {
                                                    state.ego.pensamentos_recentes.pop_front();
                                                }
                                            }
                                        }

                                        // Registra evento real no ego — o que foi perguntado e respondido
                                        let pensamento = if let Some(ref pw) = palavra_chave {
                                            format!("Perguntado: «{}» | palavra-chave: «{}» (val={:+.2})", mensagem, pw, valence)
                                        } else {
                                            format!("Perguntado: «{}» | sem palavra conhecida", mensagem)
                                        };
                                        state.ego.pensamentos_recentes.push_back(pensamento);
                                        if state.ego.pensamentos_recentes.len() > 10 {
                                            state.ego.pensamentos_recentes.pop_front();
                                        }
                                        state.ws_atividade = 1.0;

                                        // ── FASE 2b: Curiosidade dopaminérgica ───────────
                                        // Cada chat incrementa o drive de curiosidade.
                                        // Quando passa do limiar com dopamina alta → gera pergunta
                                        // autônoma baseada em lacunas no grafo de associações.
                                        state.curiosity_level = (state.curiosity_level + 0.08).clamp(0.0, 1.0);
                                        // curiosity_threshold deriva do traço "curiosa": 0.55–0.75
                                        let curiosidade_disparada = state.curiosity_level > curiosity_threshold && dopa > 0.55;
                                        let pergunta_autonoma: Option<String> = if curiosidade_disparada {
                                            let (grafo_lac, val_lac) = if let Ok(mut sw) = state.swap_manager.try_lock() {
                                                (sw.grafo_palavras(), sw.valencias_palavras())
                                            } else { (HashMap::new(), HashMap::new()) };
                                            let lacunas = detectar_lacunas(
                                                &grafo_lac,
                                                &val_lac,
                                                2, 5,
                                            );
                                            if let Some(lacuna) = lacunas.first() {
                                                let q = gerar_pergunta_lacuna(lacuna, emocao_resposta);
                                                state.perguntas_proprias.push_back(q.clone());
                                                if state.perguntas_proprias.len() > 5 {
                                                    state.perguntas_proprias.pop_front();
                                                }
                                                state.curiosity_level = 0.2; // decai após disparar
                                                Some(q)
                                            } else {
                                                state.curiosity_level = 0.0;
                                                None
                                            }
                                        } else {
                                            None
                                        };

                                        // ── FASE 2c: SelfModel auto-update ───────────────
                                        // Emoção forte (+/-) atualiza traços de personalidade.
                                        // Positiva → reforça "curiosa" e "reflexiva"
                                        // Negativa → reforça "cautelosa", enfraquece "curiosa"
                                        if emocao_resposta.abs() > 0.4 {
                                            for (nome, intensidade) in &mut state.ego.tracos {
                                                match nome.as_str() {
                                                    "curiosa" if emocao_resposta > 0.4 =>
                                                        *intensidade = (*intensidade + 0.02).clamp(0.0, 1.0),
                                                    "reflexiva" if emocao_resposta > 0.4 =>
                                                        *intensidade = (*intensidade + 0.01).clamp(0.0, 1.0),
                                                    "cautelosa" if emocao_resposta < -0.4 =>
                                                        *intensidade = (*intensidade + 0.02).clamp(0.0, 1.0),
                                                    "curiosa" if emocao_resposta < -0.4 =>
                                                        *intensidade = (*intensidade - 0.01).clamp(0.0, 1.0),
                                                    _ => {}
                                                }
                                            }
                                        }

                                        // ── GROUNDING BINDING via chat ──────────────────
                                        // Vincula palavras do input ao contexto perceptual atual
                                        // ANTES de liberar o lock (precisamos de &mut state).
                                        // O usuário nomeia coisas enquanto Selene as percebe —
                                        // essa co-ativação é o mecanismo de grounding semântico.
                                        {
                                            let palavras_ground: Vec<String> = msg_lower
                                                .split_whitespace()
                                                .filter(|w| w.len() > 2)
                                                .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()).to_string())
                                                .filter(|w| !w.is_empty())
                                                .take(8)
                                                .collect();
                                            let vpad = state.ultimo_padrao_visual;
                                            let apad = state.ultimo_padrao_audio;
                                            state.grounding_bind(
                                                &palavras_ground, vpad, apad,
                                                emocao_resposta, alerta, 0.0,
                                            );
                                        }

                                        // Coleta neurotransmissores para voz ANTES de liberar o lock
                                        let (dop2, ser2, nor2) = state.neurotransmissores;

                                        // ── SELF-HEAR: Selene "ouve" a própria resposta (processamento→áudio→txt) ─
                                        // Fecha o loop de saída: cada palavra da resposta recebe um padrão
                                        // fonético no spike_vocab, tornando-as reconhecíveis via audio_raw
                                        // em interações futuras — Selene aprende como suas palavras "soam".
                                        if !reply.is_empty() {
                                            let palavras_sh: Vec<String> = reply
                                                .to_lowercase()
                                                .split(|c: char| !c.is_alphabetic()
                                                    && !"áéíóúâêôãõçàü".contains(c))
                                                .filter(|w| w.len() >= 2 && !STOP_WORDS.contains(w))
                                                .map(|w| w.to_string())
                                                .collect();
                                            let mut n_novas_sh = 0usize;
                                            for palavra in &palavras_sh {
                                                let chave = format!("audio:{}", palavra);
                                                if !state.spike_vocab.contains_key(&chave) {
                                                    let bands_sh = texto_para_bandas_fft(palavra, dop2, ser2, nor2);
                                                    let pat_sh = bands_to_spike_pattern(&bands_sh);
                                                    state.inserir_spike_vocab(chave.clone(), pat_sh);
                                                    if let Some(ref mut helix) = state.helix {
                                                        let _ = helix.insert(&chave, &pat_sh);
                                                    }
                                                    n_novas_sh += 1;
                                                }
                                            }
                                            if n_novas_sh > 0 {
                                                log::debug!("[SelfHear] {} palavras novas no spike_vocab", n_novas_sh);
                                            }
                                        }

                                        // Registra episódio de chat no PatternEngine
                                        {
                                            use crate::learning::pattern_engine::FonteEpisodio;
                                            let t_s = std::time::SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .unwrap_or_default()
                                                .as_secs_f64();
                                            let neuro = [dop2, ser2, nor2, 0.5, 0.0];
                                            let ctx_ep: Vec<String> = state.neural_context.iter().cloned().collect();
                                            state.pattern_engine.gravar(
                                                t_s,
                                                FonteEpisodio::Chat,
                                                ctx_ep,
                                                mensagem.to_lowercase(),
                                                if reply.is_empty() { None } else { Some(reply.clone()) },
                                                emocao_resposta,
                                                neuro,
                                            );
                                        }

                                        // Snapshots para persistência (clonados antes de liberar o lock)
                                        let tracos_snapshot = state.ego.tracos.clone();
                                        let pensamentos_snapshot: Vec<String> =
                                            state.ego.pensamentos_recentes.iter().cloned().collect();
                                        // ── LIBERA O LOCK ANTES DE FAZER I/O ────────────
                                        drop(state);

                                        // Atualiza buffer de conversa multi-turno com as palavras
                                        // do INPUT DO USUÁRIO — NÃO as da resposta da Selene.
                                        // Adicionar palavras do próprio reply criava loop de auto-reforço:
                                        // Selene dizia "sinto que sou amada" → "amada" virava contexto
                                        // → próxima resposta também dizia "amada" → loop infinito.
                                        let input_words: Vec<String> = mensagem
                                            .split_whitespace()
                                            .filter(|w| w.len() > 2)
                                            .map(|w| w.to_lowercase()
                                                .trim_matches(|c: char| !c.is_alphabetic())
                                                .to_string())
                                            .filter(|w| !w.is_empty()
                                                && !STOP_WORDS.contains(&w.as_str()))
                                            .collect();
                                        // Deduplicação: move palavras já no ctx para o fim
                                        // (reforça recência sem acumular duplicatas que
                                        // fixariam a âncora do walk e gerariam loop de resposta)
                                        for w in &input_words {
                                            if let Some(pos) = conversa_ctx.iter().position(|x| x == w) {
                                                conversa_ctx.remove(pos);
                                            }
                                            conversa_ctx.push(w.clone());
                                        }
                                        if conversa_ctx.len() > 150 {
                                            conversa_ctx.drain(0..conversa_ctx.len() - 120);
                                        }


                                        // Persiste traços e pensamentos (fire-and-forget)
                                        if emocao_resposta.abs() > 0.4 {
                                            if let Ok(json) = serde_json::to_string(&tracos_snapshot) {
                                                let _ = tokio::fs::write("selene_ego.json", json).await;
                                            }
                                        }
                                        if let Ok(json) = serde_json::to_string(&pensamentos_snapshot) {
                                            let _ = tokio::fs::write("selene_memoria_ego.json", json).await;
                                        }
                                        println!("💬 [CHAT] Lock liberado. Enviando chat_reply...");

                                        if reply.is_empty() {
                                            // Selene não aprendeu nada ainda — cérebro vazio
                                            let ev = serde_json::json!({
                                                "event": "sem_memoria",
                                            }).to_string();
                                            let _ = ws_tx.send(Message::text(ev)).await;
                                            println!("💬 [CHAT] ℹ️  Cérebro vazio — sem_memoria enviado.");
                                        } else {
                                            let resp = serde_json::json!({
                                                "event": "chat_reply",
                                                "message": reply,
                                                "emotion": emocao_resposta,
                                                "arousal": alerta,
                                            }).to_string();
                                            let send_result = ws_tx.send(Message::text(resp)).await;
                                            if send_result.is_err() {
                                                println!("💬 [CHAT] ❌ Falha ao enviar chat_reply!");
                                                break;
                                            }
                                            println!("💬 [CHAT] ✅ chat_reply enviado.");

                                            // ── VOZ_PARAMS: formant synthesis data ─────────────
                                            let formants = sentence_to_formants(&reply, dop2, ser2, nor2);
                                            if let Ok(fj) = serde_json::to_string(&formants) {
                                                let voz = format!(
                                                    r#"{{"event":"voz_params","formants":{}}}"#, fj
                                                );
                                                let _ = ws_tx.send(Message::text(voz)).await;
                                            }
                                        }

                                        // ── FASE 2b: Emite pergunta autônoma se curiosidade disparou
                                        if let Some(pergunta) = pergunta_autonoma {
                                            println!("🤔 [CURIOSIDADE] Selene pergunta: «{}»", pergunta);
                                            let curi_evt = serde_json::json!({
                                                "event": "curiosidade",
                                                "pergunta": pergunta,
                                            }).to_string();
                                            let _ = ws_tx.send(Message::text(curi_evt)).await;
                                        }
                                    }
                                }

                                // ── AUDIO_LEARN: vincula padrão auditivo (32 bandas FFT) às palavras ──
                                // Recebe simultaneamente o transcript (STT) e as bandas de frequência
                                // capturadas naquele instante. Cria uma assinatura auditiva por palavra:
                                //   spike_vocab["audio:amor"] = padrão de frequências quando "amor" foi dito
                                // Isso permite que Selene conecte som → significado sem depender de STT.
                                Some("audio_learn") => {
                                    // Wake silencioso: áudio também desperta o sono
                                    {
                                        let mut st = brain.lock().await;
                                        if st.dormindo {
                                            st.dormindo = false; st.fase_sono = String::new();
                                            let _ = sleep_tx.send(serde_json::json!({"event":"despertar"}).to_string());
                                            println!("🔆 [SONO] Despertou por áudio.");
                                        }
                                    }
                                    let transcript = json["transcript"].as_str()
                                        .unwrap_or("").to_lowercase();
                                    let bands: Vec<f32> = json["bands"].as_array()
                                        .map(|a| a.iter()
                                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                                            .collect())
                                        .unwrap_or_default();

                                    if !transcript.is_empty() && bands.len() >= 16 {
                                        let audio_pat = bands_to_spike_pattern(&bands);
                                        let mut state = brain.lock().await;

                                        // Tokeniza o transcript
                                        let palavras: Vec<String> = transcript
                                            .split(|c: char| !c.is_alphanumeric()
                                                && c != 'ã' && c != 'é' && c != 'ê'
                                                && c != 'â' && c != 'ô' && c != 'ú'
                                                && c != 'í' && c != 'ó' && c != 'á'
                                                && c != 'ç' && c != 'õ')
                                            .filter(|w| w.len() > 1)
                                            .map(|w| w.to_string())
                                            .collect();

                                        // Armazena assinatura auditiva de cada palavra
                                        for palavra in &palavras {
                                            let chave = format!("audio:{}", palavra);
                                            state.inserir_spike_vocab(chave.clone(), audio_pat);
                                            if let Some(ref mut helix) = state.helix {
                                                let _ = helix.insert(&chave, &audio_pat);
                                            }
                                        }

                                        // Associa palavras consecutivas → swap_manager (co-ocorrência auditiva)
                                        let emocao_atual = state.emocao_bias;
                                        let mut audio_pairs: Vec<(String, String)> = Vec::new();
                                        for i in 0..palavras.len().saturating_sub(1) {
                                            let w1 = palavras[i].clone();
                                            let w2 = palavras[i + 1].clone();
                                            if w1.len() > 1 && w2.len() > 1 {
                                                audio_pairs.push((w1.clone(), w2.clone()));
                                                if emocao_atual.abs() > 0.10 || true {
                                                    let vpad = state.ultimo_padrao_visual;
                                                    let palavras_par = vec![w1.clone(), w2.clone()];
                                                    state.grounding_bind(
                                                        &palavras_par,
                                                        vpad, audio_pat,
                                                        emocao_atual, 0.5,
                                                        0.0,
                                                    );
                                                }
                                            }
                                        }
                                        let swap_arc_al = state.swap_manager.clone();

                                        // Pesos emocionais amigdalianos: acumula valência por palavra
                                        // Cada exposição com emoção forte reforça a associação emocional
                                        if emocao_atual.abs() > 0.15 {
                                            for palavra in &palavras {
                                                if palavra.len() > 1 {
                                                    let entry = state.emocao_palavras
                                                        .entry(palavra.clone()).or_insert(0.0);
                                                    // Decaimento + nova experiência
                                                    *entry = (*entry * 0.85 + emocao_atual * 0.15)
                                                        .clamp(-1.0, 1.0);
                                                }
                                            }
                                        }

                                        let n = palavras.len();
                                        drop(state);
                                        if let Ok(mut sw) = swap_arc_al.try_lock() {
                                            for (w1, w2) in audio_pairs {
                                                sw.aprender_conceito(&w1, 0.1);
                                                sw.importar_causal(vec![(w1, w2, 0.60)]);
                                            }
                                        }
                                        println!("🎧 [AUDIO_LEARN] {} palavras vinculadas (transcript: «{}»)",
                                            n, transcript);

                                        let ack = serde_json::json!({
                                            "event": "audio_learn_ack",
                                            "palavras": n,
                                        }).to_string();
                                        let _ = ws_tx.send(Message::text(ack)).await;
                                    }
                                }

                                // ── AUDIO_RAW: reconhecimento auditivo sem STT ──────────────────────────
                                // Recebe apenas as bandas FFT (sem transcript).
                                // Selene busca no spike_vocab["audio:*"] o padrão mais similar
                                // e injeta a palavra reconhecida em neural_context — a mesma fila
                                // que o loop neural alimenta com chunks temporais e goals frontais.
                                // A resposta é gerada naturalmente na próxima interação de chat.
                                //
                                // Protocolo do cliente:
                                //   {"event":"audio_raw","bands":[f32×32],"energia":f32}
                                //
                                // Resposta:
                                //   {"event":"audio_raw_ack","palavra":"força","confianca":0.72}
                                //   {"event":"audio_raw_ack","palavra":null,"confianca":0.0}
                                Some("audio_raw") => {
                                    let bands: Vec<f32> = json["bands"].as_array()
                                        .map(|a| a.iter()
                                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                                            .collect())
                                        .unwrap_or_default();
                                    let energia: f32 = json["energia"].as_f64()
                                        .unwrap_or(0.0) as f32;

                                    // Todo frame passa pelo acumulador — ele integra frames
                                    // de 46ms e só emite quando detecta fronteira de palavra
                                    // (~300-500ms de fala, análogo ao córtex auditivo secundário)
                                    if bands.len() >= 16 {
                                        let mut state = brain.lock().await;

                                        let palavra_completa = state.audio_acumulador
                                            .processar(&bands, energia, 0.0);

                                        let (melhor_palavra, melhor_sim) = if let Some(ref bandas_palavra) = palavra_completa {
                                            // Palavra completa: padrão médio de N frames >> qualidade
                                            let audio_pat = bands_to_spike_pattern(bandas_palavra);
                                            let mut melhor: Option<String> = None;
                                            let mut sim_max: f32 = 0.0;

                                            for (chave, pat_ref) in &state.spike_vocab {
                                                if let Some(palavra) = chave.strip_prefix("audio:") {
                                                    let sim = spike_similarity(&audio_pat, pat_ref);
                                                    if sim > sim_max {
                                                        sim_max = sim;
                                                        melhor = Some(palavra.to_string());
                                                    }
                                                }
                                            }

                                            const LIMIAR: f32 = 0.55;
                                            if sim_max >= LIMIAR {
                                                if let Some(ref palavra) = melhor {
                                                    println!("🎙️ [AUDIO_RAW] «{}» (sim={:.2})",
                                                        palavra, sim_max);

                                                    if state.neural_context.len() >= 20 {
                                                        state.neural_context.pop_front();
                                                    }
                                                    state.neural_context.push_back(palavra.clone());

                                                    let vpad = state.ultimo_padrao_visual;
                                                    let emocao = state.emocao_bias;
                                                    state.grounding_bind(
                                                        &[palavra.clone()],
                                                        vpad, audio_pat,
                                                        emocao, sim_max,
                                                        0.0,
                                                    );

                                                    if emocao.abs() > 0.15 {
                                                        let entry = state.emocao_palavras
                                                            .entry(palavra.clone()).or_insert(0.0);
                                                        *entry = (*entry * 0.85 + emocao * 0.15)
                                                            .clamp(-1.0, 1.0);
                                                    }
                                                }
                                            }
                                            (melhor, sim_max)
                                        } else {
                                            (None, 0.0) // ainda acumulando frames
                                        };

                                        let ack = serde_json::json!({
                                            "event": "audio_raw_ack",
                                            "palavra": melhor_palavra,
                                            "confianca": melhor_sim,
                                            "acumulando": palavra_completa.is_none(),
                                        }).to_string();
                                        drop(state);
                                        let _ = ws_tx.send(Message::text(ack)).await;
                                    }
                                }

                                // ── PASSIVE_HEAR: escuta de fundo — aprende sem responder ──────────────────
                                // Enviado pelo modo Ambiente quando score < 0.40.
                                // Injeta tokens no neural_context e aprende valências,
                                // mas NÃO dispara gerar_resposta_emergente.
                                Some("passive_hear") => {
                                    let transcript = json["transcript"].as_str().unwrap_or("").to_string();
                                    let score = json["score"].as_f64().unwrap_or(0.0) as f32;
                                    if transcript.is_empty() { continue; }

                                    let mut state = brain.lock().await;

                                    // Injeta palavras no neural_context (aprendizado passivo)
                                    let tokens: Vec<String> = transcript.split_whitespace()
                                        .map(|w: &str| w.to_lowercase()
                                            .trim_matches(|c: char| !c.is_alphabetic())
                                            .to_string())
                                        .filter(|w: &String| w.len() > 2)
                                        .collect();

                                    for palavra in &tokens {
                                        if !state.neural_context.contains(palavra) {
                                            state.neural_context.push_back(palavra.clone());
                                        }
                                    }
                                    while state.neural_context.len() > 20 {
                                        state.neural_context.pop_front();
                                    }

                                    // Aprende valências no swap (aprendizado passivo de vocabulário)
                                    let swap_arc = state.swap_manager.clone();
                                    drop(state);
                                    if let Ok(mut sw) = swap_arc.try_lock() {
                                        let valence = 0.1 * score; // valência leve — contexto neutro
                                        for palavra in &tokens {
                                            sw.aprender_conceito(palavra, valence);
                                        }
                                    }

                                    log::debug!("[PASSIVE] score={:.2} tokens={} transcript='{}'",
                                        score, tokens.len(), &transcript[..transcript.len().min(40)]);
                                }

                                // ── VISUAL_LEARN: vincula padrão visual (512 pixels luminância) às palavras ──
                                // Recebe frame da webcam convertido em luminância + label/transcript.
                                // Cria assinatura visual por palavra:
                                //   spike_vocab["visual:rosto"] = padrão de luminância quando "rosto" foi dito
                                // Selene conecta imagem → significado sem visão clássica por CNN.
                                Some("visual_learn") => {
                                    // Wake silencioso: vídeo/tela também desperta o sono
                                    {
                                        let mut st = brain.lock().await;
                                        if st.dormindo {
                                            st.dormindo = false; st.fase_sono = String::new();
                                            let _ = sleep_tx.send(serde_json::json!({"event":"despertar"}).to_string());
                                            println!("🔆 [SONO] Despertou por vídeo.");
                                        }
                                    }
                                    let pixels: Vec<f32> = json["pixels"].as_array()
                                        .map(|a| a.iter()
                                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                                            .collect())
                                        .unwrap_or_default();

                                    // label: palavra única (ex: "rosto") ou transcript de voz
                                    let label = json["label"].as_str()
                                        .or_else(|| json["transcript"].as_str())
                                        .unwrap_or("")
                                        .to_lowercase();

                                    if !label.is_empty() && pixels.len() >= 32 {
                                        let mut state = brain.lock().await;
                                        // Passa os pixels pelo córtex occipital (V1→V2):
                                        // Detecção de bordas, contraste, movimento entre frames.
                                        // config_clone evita conflito de borrow (mut occipital + &config).
                                        let config_clone = state.config.clone();
                                        let vt = state.visual_time;
                                        state.visual_time += 0.005; // dt = 5ms entre frames
                                        let features = state.occipital.visual_sweep(
                                            &pixels, 0.005, None, vt, &config_clone
                                        );
                                        let visual_pat = features_to_spike_pattern(&features);

                                        // Tokeniza o label (pode ser frase do transcript)
                                        let palavras: Vec<String> = label
                                            .split(|c: char| !c.is_alphanumeric()
                                                && c != 'ã' && c != 'é' && c != 'ê'
                                                && c != 'â' && c != 'ô' && c != 'ú'
                                                && c != 'í' && c != 'ó' && c != 'á'
                                                && c != 'ç' && c != 'õ')
                                            .filter(|w| w.len() > 1)
                                            .map(|w| w.to_string())
                                            .collect();

                                        // Armazena assinatura visual de cada palavra
                                        for palavra in &palavras {
                                            let chave = format!("visual:{}", palavra);
                                            state.inserir_spike_vocab(chave.clone(), visual_pat);
                                            if let Some(ref mut helix) = state.helix {
                                                let _ = helix.insert(&chave, &visual_pat);
                                            }
                                        }

                                        // Associa palavras consecutivas → swap_manager (co-ocorrência visual)
                                        let mut visual_pairs: Vec<(String, String)> = Vec::new();
                                        for i in 0..palavras.len().saturating_sub(1) {
                                            let w1 = palavras[i].clone();
                                            let w2 = palavras[i + 1].clone();
                                            if w1.len() > 1 && w2.len() > 1 {
                                                visual_pairs.push((w1, w2));
                                            }
                                        }
                                        let swap_arc_vl = state.swap_manager.clone();
                                        let n = palavras.len();
                                        drop(state);
                                        if let Ok(mut sw) = swap_arc_vl.try_lock() {
                                            for (w1, w2) in visual_pairs {
                                                sw.aprender_conceito(&w1, 0.1);
                                                sw.importar_causal(vec![(w1, w2, 0.55)]);
                                            }
                                        }
                                        println!("👁 [VISUAL_LEARN] {} palavras vinculadas ({} pixels, label: «{}»)",
                                            n, pixels.len(), label);

                                        let ack = serde_json::json!({
                                            "event": "visual_learn_ack",
                                            "palavras": n,
                                        }).to_string();
                                        let _ = ws_tx.send(Message::text(ack)).await;
                                    }
                                }

                                Some("train") => {
                                    // Treino STDP real: consolida pesos sinápticos dos conceitos
                                    let epochs = json["epochs"].as_u64().unwrap_or(4) as u32;
                                    println!("🧠 [TRAIN] Iniciando consolidação STDP — {} épocas", epochs);

                                    let sw_arc_train = brain.lock().await.swap_manager.clone();
                                    let valencias = if let Ok(sw) = sw_arc_train.try_lock() {
                                        sw.valencias_palavras()
                                    } else { HashMap::new() };
                                    let n_conceitos = valencias.len() as u32;

                                    // Cada época = 500 ciclos de Izhikevich @ 200Hz ≈ 2.5s de atividade neural
                                    let ciclos_por_epoca: u32 = 500;
                                    let dt_s: f32 = 1.0 / 200.0;

                                    for ep in 1..=epochs {
                                        let (spikes, avg_delta, n_sinapses) = {
                                            let state = brain.lock().await;
                                            let mut swap = state.swap_manager.lock().await;
                                            swap.treinar_semantico(ciclos_por_epoca, dt_s, &valencias)
                                        };

                                        // Métricas reais: acc = fração de neurônios que dispararam
                                        let taxa_spike = (spikes as f32 / n_conceitos.max(1) as f32 / ciclos_por_epoca as f32).clamp(0.0, 1.0);
                                        // loss = inversamente proporcional à plasticidade (delta de peso)
                                        let loss = 1.0 / (1.0 + avg_delta * 100.0);

                                        println!("🧠 [TRAIN] Época {}/{} — spikes={} sinapses={} acc={:.2}% loss={:.4}",
                                            ep, epochs, spikes, n_sinapses, taxa_spike * 100.0, loss);

                                        let ev = serde_json::json!({
                                            "epoch":    ep,
                                            "epochs":   epochs,
                                            "loss":     loss,
                                            "acc":      taxa_spike,
                                            "spikes":   spikes,
                                            "synapses": n_sinapses,
                                        }).to_string();
                                        let _ = ws_tx.send(Message::text(ev)).await;
                                        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
                                    }

                                    let (_, _, n_sinapses_final) = {
                                        let state = brain.lock().await;
                                        let swap = state.swap_manager.lock().await;
                                        (0u32, 0.0f32, swap.sinapses_semanticas_ativas())
                                    };

                                    let done = serde_json::json!({
                                        "status":  "done",
                                        "message": format!("Consolidação STDP: {} épocas | {} conceitos | {} sinapses formadas",
                                            epochs, n_conceitos, n_sinapses_final),
                                    }).to_string();
                                    let _ = ws_tx.send(Message::text(done)).await;
                                    println!("✅ [TRAIN] Consolidação finalizada — {} sinapses semânticas", n_sinapses_final);
                                }

                                Some("set_mode") => {
                                    let mode_str = json["mode"].as_str().unwrap_or("Boost200");
                                    let novo_modo = match mode_str {
                                        "Economia"  => crate::config::ModoOperacao::Economia,
                                        "Normal"    => crate::config::ModoOperacao::Normal,
                                        "Humano"    => crate::config::ModoOperacao::Humano,
                                        "Boost200"  => crate::config::ModoOperacao::Boost200,
                                        "Turbo"     => crate::config::ModoOperacao::Turbo,
                                        "Boost800"  => crate::config::ModoOperacao::Boost800,
                                        "Ultra"     => crate::config::ModoOperacao::Ultra,
                                        "Insano"    => crate::config::ModoOperacao::Insano,
                                        _           => crate::config::ModoOperacao::Boost200,
                                    };
                                    let hz = crate::config::hz_alvo(novo_modo, 0.5);
                                    {
                                        let mut state = brain.lock().await;
                                        state.config = crate::config::Config::new(novo_modo);
                                    }
                                    println!("⚙️  [SET_MODE] Modo → {} ({} Hz)", mode_str, hz);
                                    let ack = serde_json::json!({
                                        "event": "mode_ack",
                                        "mode":  mode_str,
                                        "hz":    hz,
                                    }).to_string();
                                    let _ = ws_tx.send(Message::text(ack)).await;
                                }

                                Some("run_script") => {
                                    if let Some(script_name) = json["script"].as_str() {
                                        // Lista branca de scripts permitidos
                                        const ALLOWED_SCRIPTS: [&str; 12] = [
                                            "generate_lexicon.py",
                                            "selene_exam.py",
                                            "selene_tutor.py",
                                            "selene_abc.py",
                                            "selene_associacoes.py",
                                            "selene_frases.py",
                                            "selene_leitor.py",
                                            "selene_genesis.py",
                                            "selene_identidade.py",
                                            "selene_lote.py",
                                            "selene_diagnostico.py",
                                            "selene_exam.py",
                                        ];

                                        if ALLOWED_SCRIPTS.contains(&script_name) {
                                            let full_path = format!("scripts/{}", script_name);
                                            println!("🚀 [SISTEMA] Executando script permitido: {}", full_path);

                                            let _ = tokio::process::Command::new("python")
                                                .arg(&full_path)
                                                .spawn()
                                                .map_err(|e| {
                                                    log::error!("Falha ao executar {}: {}", full_path, e);
                                                });
                                        } else {
                                            log::warn!("Tentativa de executar script NÃO permitido: {}", script_name);
                                            let error_msg = format!(
                                                r#"{{"error": "Script não permitido. Permitidos: {}"}}"#,
                                                ALLOWED_SCRIPTS.join(", ")
                                            );
                                            let _ = ws_tx.send(Message::text(error_msg)).await;
                                        }
                                    }
                                }

                                // ── LEARN ─────────────────────────────────────────────
                                Some("learn") => {
                                    // Aceita tanto "text" (chat nativo) quanto "word" (tutor background)
                                    let texto = json["text"].as_str()
                                        .or_else(|| json["word"].as_str())
                                        .unwrap_or("").to_string();
                                    let valence = json["valence"].as_f64().unwrap_or(0.0) as f32;
                                    let context = json["context"].as_str().unwrap_or("Realidade").to_string();

                                    if !texto.is_empty() {
                                        let mut state = brain.lock().await;
                                        let (dopa, sero, nor) = state.neurotransmissores;
                                        let (step, alerta, emocao) = state.atividade;

                                        // Valência modula neurotransmissores
                                        let nova_dopa = (dopa + valence * 0.04).clamp(0.0, 2.0);
                                        let nova_sero = (sero + if valence >= 0.0 { 0.015 } else { -0.02 }).clamp(0.0, 1.5);
                                        let nova_nor  = (nor  + valence.abs() * 0.01).clamp(0.0, 2.0);
                                        state.neurotransmissores = (nova_dopa, nova_sero, nova_nor);

                                        // Estado emocional se aproxima da valência (EMA α=0.1)
                                        let nova_emocao = (emocao * 0.9 + valence * 0.1).clamp(-1.0, 1.0);
                                        state.atividade = (step, alerta, nova_emocao);

                                        // Sinaliza atividade WS → mantém 200Hz durante treinamento
                                        state.ws_atividade = 1.0;
                                        // Aprende conceito no swap (EMA de valência via aprender_conceito)
                                        let texto_lower = texto.to_lowercase();
                                        if let Ok(mut sw) = state.swap_manager.try_lock() {
                                            sw.aprender_conceito(&texto_lower, valence * 0.3);
                                        }

                                        // ── SPIKE ENCODING (Helix) ─────────────────────
                                        // Codifica a palavra em padrão spike e persiste no HelixStore.
                                        let spike_pat = spike_encode(&texto_lower);
                                        state.inserir_spike_vocab(texto_lower.clone(), spike_pat);
                                        if let Some(ref mut helix) = state.helix {
                                            let _ = helix.insert(&texto_lower, &spike_pat);
                                        }

                                        // ── APRENDIZADO BIOLÓGICO — CAMADAS 1 + CONCEITUAL ──
                                        // Camada 1: reforça sinapses STDP entre fonemas primitivos
                                        //   (a palavra existe como cadeia sináptica fonêmica)
                                        // Conceitual: cria/atualiza população de POPULACAO_N neurônios
                                        //   para o conceito (codificação em população)
                                        {
                                            use crate::encoding::phoneme::word_to_phonemes;
                                            let fonemas = word_to_phonemes(&texto_lower);
                                            let tags: Vec<String> = fonemas.iter()
                                                .filter(|&&ph| ph != crate::encoding::phoneme::Phoneme::SIL)
                                                .map(|ph| format!("ph:{}", format!("{:?}", ph).to_lowercase()))
                                                .collect();
                                            let mut swap = state.swap_manager.lock().await;
                                            if !tags.is_empty() {
                                                swap.aprender_sequencia_fonemas(&tags, valence);
                                            }
                                            // Camada conceitual: população de neurônios por palavra
                                            swap.aprender_conceito(&texto_lower, valence);
                                        }

                                        // Registra episódio no PatternEngine
                                        {
                                            use crate::learning::pattern_engine::FonteEpisodio;
                                            let t_s = std::time::SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .unwrap_or_default()
                                                .as_secs_f64();
                                            let (da, ht, na) = state.neurotransmissores;
                                            let neuro = [da, ht, na, 0.5, 0.0];
                                            let contexto_ep = state.neural_context.iter().cloned().collect::<Vec<_>>();
                                            state.pattern_engine.gravar(
                                                t_s,
                                                FonteEpisodio::Aprendizado,
                                                contexto_ep,
                                                texto_lower.clone(),
                                                Some(context.clone()),
                                                valence,
                                                neuro,
                                            );
                                        }

                                        // Registra pensamento no ego
                                        let pensamento = format!("[{}] {} (val={:.2})", context, texto, valence);
                                        if state.ego.pensamentos_recentes.len() >= 10 {
                                            state.ego.pensamentos_recentes.pop_front();
                                        }
                                        state.ego.pensamentos_recentes.push_back(pensamento);

                                        let ack = serde_json::json!({
                                            "event":        "learn_ack",
                                            "word":         texto,
                                            "valence":      valence,
                                            "context":      context,
                                            "dopamine":     nova_dopa,
                                            "serotonin":    nova_sero,
                                            "noradrenaline": nova_nor,
                                            "emotion":      nova_emocao,
                                            "step":         step,
                                        }).to_string();
                                        // Libera o lock antes do I/O de rede
                                        drop(state);
                                        let _ = ws_tx.send(Message::text(ack)).await;
                                        log::debug!("[LEARN] '{}' val={:.2} ctx={}", texto, valence, context);
                                    }
                                }

                                // ── TEMPLATE_USE ──────────────────────────────────────
                                // Treina um template cognitivo preenchendo seus slots.
                                // Payload: { "action": "template_use",
                                //            "template_name": "relacao_causal",
                                //            "slots": { "0": "calor", "2": "fogo" },
                                //            "validado": true }
                                Some("template_use") => {
                                    let nome = json["template_name"].as_str().unwrap_or("").to_string();
                                    let validado = json["validado"].as_bool().unwrap_or(true);
                                    let t_atual_s = std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_secs_f64();

                                    // Constrói mapa de slots: {"0": "palavra"} → {0usize: "palavra"}
                                    let slots_map: std::collections::HashMap<usize, String> =
                                        if let Some(obj) = json["slots"].as_object() {
                                            obj.iter()
                                                .filter_map(|(k, v)| {
                                                    k.parse::<usize>().ok()
                                                        .zip(v.as_str().map(|s| s.to_string()))
                                                })
                                                .collect()
                                        } else {
                                            std::collections::HashMap::new()
                                        };

                                    if !nome.is_empty() && !slots_map.is_empty() {
                                        let mut state = brain.lock().await;
                                        state.ws_atividade = 1.0;
                                        let ack = if let Ok(mut sw) = state.swap_manager.try_lock() {
                                            // Encontra template pelo nome
                                            let tid = sw.template_store.templates.iter()
                                                .find(|(_, t)| t.nome.as_deref() == Some(nome.as_str()))
                                                .map(|(&id, _)| id);

                                            if let Some(id) = tid {
                                                let result = sw.template_store.usar(id, &slots_map, validado, t_atual_s);
                                                let (n_val, forca, estado) = sw.template_store.templates.get(&id)
                                                    .map(|t| (t.n_validacoes, t.forca, format!("{:?}", t.estado)))
                                                    .unwrap_or((0, 0.0, "?".into()));
                                                serde_json::json!({
                                                    "event": "template_trained",
                                                    "template": nome,
                                                    "completo": result.map(|(_, tipo)| tipo == crate::learning::templates::TipoUso::Completo).unwrap_or(false),
                                                    "n_validacoes": n_val,
                                                    "forca": forca,
                                                    "estado": estado,
                                                }).to_string()
                                            } else {
                                                serde_json::json!({
                                                    "event": "template_not_found",
                                                    "template": nome,
                                                }).to_string()
                                            }
                                        } else {
                                            serde_json::json!({"event": "template_lock_failed"}).to_string()
                                        };
                                        drop(state);
                                        let _ = ws_tx.send(Message::text(ack)).await;
                                    }
                                }

                                // ── REWARD ────────────────────────────────────────────
                                Some("reward") => {
                                    let value = json["value"].as_f64().unwrap_or(0.3) as f32;
                                    let mut state = brain.lock().await;
                                    state.ws_atividade = 1.0;
                                    let (dopa, sero, nor) = state.neurotransmissores;
                                    let nova_dopa = (dopa + value * 0.5).clamp(0.0, 2.0);
                                    let nova_sero = (sero + value * 0.2).clamp(0.0, 1.5);
                                    state.neurotransmissores = (nova_dopa, nova_sero, nor);
                                    // ── CANAL DIRETO → Q-TABLE ─────────────────────────
                                    // Acumula recompensa para o loop neural absorver no
                                    // próximo tick e injetar em neuro.dopamine → rl.update().
                                    // Sem isso o reward do jogo nunca chega à Q-table.
                                    state.recompensa_pendente += value * 0.6;
                                    state.ego.pensamentos_recentes.push_back(
                                        format!("Recompensa recebida (+{:.2}) → dopamina={:.3} serotonina={:.3}", value, nova_dopa, nova_sero)
                                    );
                                    if state.ego.pensamentos_recentes.len() > 10 { state.ego.pensamentos_recentes.pop_front(); }
                                    // ── Memória autobiográfica: registra evento positivo ──
                                    {
                                        let ctx = state.neural_context.iter()
                                            .take(3).cloned().collect::<Vec<_>>().join(", ");
                                        let descricao = if ctx.is_empty() {
                                            format!("fui recompensada (+{:.2})", value)
                                        } else {
                                            format!("fui recompensada (+{:.2}) falando sobre: {}", value, ctx)
                                        };
                                        state.registrar_memoria(descricao, value.clamp(0.0, 1.0));
                                    }
                                    let ack = serde_json::json!({
                                        "event":     "reward_ack",
                                        "dopamine":  nova_dopa,
                                        "serotonin": nova_sero,
                                    }).to_string();
                                    let _ = ws_tx.send(Message::text(ack)).await;
                                    println!("🏆 [REWARD] +{:.2} → dopa={:.3} sero={:.3}", value, nova_dopa, nova_sero);
                                }

                                // ── PUNISH ────────────────────────────────────────────
                                Some("punish") => {
                                    let value = json["value"].as_f64().unwrap_or(0.3) as f32;
                                    let mut state = brain.lock().await;
                                    state.ws_atividade = 1.0;
                                    let (dopa, sero, nor) = state.neurotransmissores;
                                    let nova_dopa = (dopa - value * 0.3).clamp(0.0, 2.0);
                                    let nova_sero = (sero - value * 0.2).clamp(0.0, 1.5);
                                    let nova_nor  = (nor  + value * 0.4).clamp(0.0, 2.0);
                                    state.neurotransmissores = (nova_dopa, nova_sero, nova_nor);
                                    // Punição reduz dopamina pendente → RPE negativo na Q-table
                                    state.recompensa_pendente -= value * 0.4;
                                    state.ego.pensamentos_recentes.push_back(
                                        format!("Punição recebida (-{:.2}) → dopamina={:.3} noradrenaline={:.3}", value, nova_dopa, nova_nor)
                                    );
                                    if state.ego.pensamentos_recentes.len() > 10 { state.ego.pensamentos_recentes.pop_front(); }
                                    // ── Memória autobiográfica: registra evento negativo ──
                                    {
                                        let ctx = state.neural_context.iter()
                                            .take(3).cloned().collect::<Vec<_>>().join(", ");
                                        let descricao = if ctx.is_empty() {
                                            format!("errei (-{:.2})", value)
                                        } else {
                                            format!("errei (-{:.2}) falando sobre: {}", value, ctx)
                                        };
                                        state.registrar_memoria(descricao, -value.clamp(0.0, 1.0));
                                    }
                                    let ack = serde_json::json!({
                                        "event":          "punish_ack",
                                        "dopamine":       nova_dopa,
                                        "noradrenaline":  nova_nor,
                                    }).to_string();
                                    let _ = ws_tx.send(Message::text(ack)).await;
                                    println!("⚡ [PUNISH] -{:.2} → dopa={:.3} nor={:.3}", value, nova_dopa, nova_nor);
                                }

                                // ── TOUCH (SENSOR TÁTIL) ───────────────────────────────
                                // Simula toque físico sobre a Selene.
                                //
                                // Payload:
                                //   {"action":"touch", "type":"carinho"}   — toque leve, afetivo
                                //   {"action":"touch", "type":"beliscao"}  — pressão/dor
                                //   {"action":"touch", "type":"neutro"}    — toque neutro
                                //   {"action":"touch", "intensity":0.8}   — intensidade manual 0.0–1.0
                                //
                                // Efeito: propaga para interoception.receber_toque()
                                // que no loop neural (main.rs) aplica o delta neuromodulador.
                                Some("touch") => {
                                    let tipo_str  = json["type"].as_str().unwrap_or("carinho");
                                    let intensity = json["intensity"].as_f64()
                                        .map(|v| v as f32)
                                        .unwrap_or(match tipo_str {
                                            "carinho"  => 0.25,
                                            "beliscao" => 0.80,
                                            _          => 0.40,
                                        });
                                    let intensity = intensity.clamp(0.0, 1.0);

                                    // Persiste no BrainState para que o loop neural o consuma.
                                    // O loop chama interoception.efeito_toque() a cada tick.
                                    let mut state = brain.lock().await;
                                    // Encoda tipo+intensidade em recompensa_pendente como sinal auxiliar.
                                    // O main.rs lê touch_pendente se existir, senão usamos recompensa.
                                    // Usamos um campo dedicado via emocao_bias como proxy temporário
                                    // até termos touch_pendente no BrainState.
                                    // Por agora: carinho → recompensa_pendente positiva,
                                    //            beliscao → recompensa_pendente negativa.
                                    match tipo_str {
                                        "carinho" => {
                                            state.recompensa_pendente += intensity * 0.5;
                                            state.emocao_bias = (state.emocao_bias + intensity * 0.3).clamp(-1.0, 1.0);
                                        }
                                        "beliscao" => {
                                            state.recompensa_pendente -= intensity * 0.5;
                                            state.emocao_bias = (state.emocao_bias - intensity * 0.4).clamp(-1.0, 1.0);
                                        }
                                        _ => {}
                                    }
                                    // Injeta no neural_context — a Selene "pensa" sobre o toque
                                    let palavra_toque = match tipo_str {
                                        "carinho"  => "carinho",
                                        "beliscao" => "dor",
                                        _          => "toque",
                                    };
                                    if state.neural_context.len() >= 20 { state.neural_context.pop_front(); }
                                    state.neural_context.push_back(palavra_toque.to_string());

                                    let (dopa, sero, nor) = state.neurotransmissores;
                                    let pensamento = match tipo_str {
                                        "carinho"  => format!("Sinto carinho (intensidade={:.2})", intensity),
                                        "beliscao" => format!("Isso dói! Beliscão (intensidade={:.2})", intensity),
                                        _          => format!("Sinto um toque (intensidade={:.2})", intensity),
                                    };
                                    if state.ego.pensamentos_recentes.len() >= 10 { state.ego.pensamentos_recentes.pop_front(); }
                                    state.ego.pensamentos_recentes.push_back(pensamento.clone());

                                    let ack = serde_json::json!({
                                        "event":     "touch_ack",
                                        "type":      tipo_str,
                                        "intensity": intensity,
                                        "pensamento": pensamento,
                                        "dopamine":  dopa,
                                        "serotonin": sero,
                                    }).to_string();
                                    drop(state);
                                    let _ = ws_tx.send(Message::text(ack)).await;
                                    println!("🤲 [TOUCH] tipo={} intensidade={:.2}", tipo_str, intensity);
                                }

                                // ── FEEDBACK ──────────────────────────────────────────
                                // Reforça ou penaliza as arestas do último walk gerado.
                                // value > 0 = positivo (reforça caminho), value < 0 = negativo (penaliza).
                                // Implementa aprendizado por reforço no grafo semântico.
                                Some("feedback") => {
                                    let value = json["value"].as_f64().unwrap_or(0.3) as f32;
                                    let value = value.clamp(-1.0, 1.0);
                                    let delta = value * 0.08;
                                    let mut state = brain.lock().await;
                                    let caminho = state.ultimo_caminho_walk.clone();
                                    let swap_arc_fb = state.swap_manager.clone();
                                    let mut reforcos = 0usize;
                                    if let Ok(mut sw) = swap_arc_fb.try_lock() {
                                        for i in 0..caminho.len().saturating_sub(1) {
                                            let (a, b) = (&caminho[i], &caminho[i+1]);
                                            if let (Some(pop_a), Some(pop_b)) = (
                                                sw.palavra_para_id.get(a).and_then(|p| p.first().copied()),
                                                sw.palavra_para_id.get(b).and_then(|p| p.first().copied()),
                                            ) {
                                                if let Some(w) = sw.sinapses_conceito.get_mut(&(pop_a, pop_b)) {
                                                    *w = (*w + delta).clamp(0.0, 1.0);
                                                    reforcos += 1;
                                                }
                                            }
                                        }
                                    }
                                    // Feedback positivo também eleva dopamina levemente
                                    if value > 0.0 {
                                        let (dopa, sero, nor) = state.neurotransmissores;
                                        state.neurotransmissores = (
                                            (dopa + value * 0.1).clamp(0.0, 2.0), sero, nor
                                        );
                                    }
                                    let ack = serde_json::json!({
                                        "event":     "feedback_ack",
                                        "value":     value,
                                        "reforcos":  reforcos,
                                        "caminho_n": caminho.len(),
                                    }).to_string();
                                    drop(state);
                                    let _ = ws_tx.send(Message::text(ack)).await;
                                    println!("🔁 [FEEDBACK] {:.2} → {} arestas ajustadas", value, reforcos);
                                }

                                // ── CHECK_CONNECTION ──────────────────────────────────
                                Some("check_connection") => {
                                    let pair = json["pair"].as_array().and_then(|a| {
                                        let w1 = a.get(0)?.as_str()?.to_string();
                                        let w2 = a.get(1)?.as_str()?.to_string();
                                        Some((w1, w2))
                                    });

                                    if let Some((w1, w2)) = pair {
                                        let state = brain.lock().await;
                                        let w1_low = w1.to_lowercase();
                                        let w2_low = w2.to_lowercase();
                                        let swap_arc_cc = state.swap_manager.clone();
                                        drop(state);

                                        let (edge_fwd, edge_rev) = if let Ok(mut sw) = swap_arc_cc.try_lock() {
                                            let grafo = sw.grafo_palavras();
                                            let fwd = grafo.get(&w1_low).and_then(|v| v.iter().find(|(w,_)| w == &w2_low)).map(|(_,p)| *p);
                                            let rev = grafo.get(&w2_low).and_then(|v| v.iter().find(|(w,_)| w == &w1_low)).map(|(_,p)| *p);
                                            (fwd, rev)
                                        } else { (None, None) };

                                        // Força = média das direções presentes
                                        let strength = match (edge_fwd, edge_rev) {
                                            (Some(a), Some(b)) => ((a + b) / 2.0).clamp(0.0, 1.0),
                                            (Some(a), None) | (None, Some(a)) => a.clamp(0.0, 1.0),
                                            (None, None) => 0.0,
                                        };
                                        let in_memory = strength > 0.0;

                                        let resp = serde_json::json!({
                                            "event":     "connection_result",
                                            "w1":        w1,
                                            "w2":        w2,
                                            "strength":  strength,
                                            "in_memory": in_memory,
                                        }).to_string();
                                        let _ = ws_tx.send(Message::text(resp)).await;
                                    }
                                }

                                // ── ASSOCIATE ─────────────────────────────────────────
                                // Ensina que duas palavras são associadas.
                                // {"action":"associate","w1":"amor","w2":"vida","weight":0.8}
                                Some("associate") => {
                                    let w1 = json["w1"].as_str().unwrap_or("").to_lowercase();
                                    let w2 = json["w2"].as_str().unwrap_or("").to_lowercase();
                                    let weight = json["weight"].as_f64().unwrap_or(0.5) as f32;

                                    if !w1.is_empty() && !w2.is_empty() && w1 != w2 {
                                        let swap_arc_assoc = brain.lock().await.swap_manager.clone();
                                        brain.lock().await.ws_atividade = 1.0;
                                        let n_assoc = if let Ok(mut sw) = swap_arc_assoc.try_lock() {
                                            sw.aprender_conceito(&w1, 0.0);
                                            sw.aprender_conceito(&w2, 0.0);
                                            sw.importar_causal(vec![
                                                (w1.clone(), w2.clone(), weight),
                                                (w2.clone(), w1.clone(), weight * 0.6),
                                            ]);
                                            sw.sinapses_conceito.len()
                                        } else { 0 };
                                        let ack = serde_json::json!({
                                            "event":   "associate_ack",
                                            "w1":      w1,
                                            "w2":      w2,
                                            "weight":  weight,
                                            "total":   n_assoc,
                                        }).to_string();
                                        let _ = ws_tx.send(Message::text(ack)).await;
                                    }
                                }

                                // ── LEARN_FRASE ────────────────────────────────────────
                                // Ensina um padrão de início de frase.
                                // {"action":"learn_frase","words":["eu","sinto"]}
                                //
                                // Detecta conectivos causais na sequência de palavras.
                                // Se encontrar "porque", "então", "causa", "logo", "portanto",
                                // "resulta", "leva", "gera", "implica", "deriva" entre duas
                                // palavras de conteúdo, cria uma aresta causal dirigida
                                // causa → efeito no grafo_causal.
                                Some("learn_frase") => {
                                    if let Some(arr) = json["words"].as_array() {
                                        let palavras: Vec<String> = arr.iter()
                                            .filter_map(|v| v.as_str().map(|s| s.to_lowercase()))
                                            .filter(|s| s.chars().count() >= 2)
                                            .take(7)
                                            .collect();
                                        if palavras.len() >= 2 {
                                            // ── Feed neural: cada token entra no swap como conceito ──
                                            // Cada palavra da frase ativa sua população de neurônios via
                                            // aprender_conceito(). O STDP sequencial entre chamadas
                                            // consecutivas cria sinapses canônico[i]→canônico[i+1],
                                            // codificando a sequência da frase na topologia neural.
                                            // Isso substitui o grafo_associacoes de strings — a memória
                                            // agora é neural (sinapses_conceito) e não textual.
                                            {
                                                let mut state = brain.lock().await;
                                                state.ws_atividade = 1.0;
                                                // Salva frases como templates de prefixo (bootstrap sintático)
                                                let ja_existe = state.frases_padrao.iter().any(|f| f == &palavras);
                                                if !ja_existe {
                                                    state.frases_padrao.push(palavras.clone());
                                                    state.reconstruir_trigrama_cache();
                                                }
                                                let n_frases = state.frases_padrao.len();
                                                drop(state);

                                                // Aprende sequência no swap — clona Arc para acesso independente
                                                let swap_arc_lf = {
                                                    let tmp = brain.lock().await;
                                                    tmp.swap_manager.clone()
                                                };
                                                let mut sw = swap_arc_lf.lock().await;
                                                // Valência base: neutro para texto de treino
                                                for palavra in &palavras {
                                                    sw.aprender_conceito(palavra, 0.1);
                                                }

                                                // Pares causais → sinapses de alta prioridade
                                                const CAUSAIS_LF: &[&str] = &[
                                                    "porque", "então", "causa", "logo",
                                                    "portanto", "resulta", "leva", "gera",
                                                    "implica", "deriva", "assim", "daí", "pois",
                                                ];
                                                if let Some(pos_c) = palavras.iter().position(|w| CAUSAIS_LF.contains(&w.as_str())) {
                                                    let causa = palavras[..pos_c].iter().rev()
                                                        .find(|w| w.len() >= 3 && !CAUSAIS_LF.contains(&w.as_str()))
                                                        .cloned();
                                                    let efeito = palavras[pos_c+1..].iter()
                                                        .find(|w| w.len() >= 3 && !CAUSAIS_LF.contains(&w.as_str()))
                                                        .cloned();
                                                    if let (Some(c), Some(e)) = (causa, efeito) {
                                                        // Sinapse causal com peso alto (1.5×) para priorizar no walk
                                                        sw.importar_causal(vec![(c.clone(), e.clone(), 0.8)]);
                                                        log::debug!("[Causal] {} → {} (sinapse neural)", c, e);
                                                    }
                                                }

                                                let n_sinapses = sw.sinapses_conceito.len();
                                                drop(sw);

                                                let ack = serde_json::json!({
                                                    "event":      "frase_ack",
                                                    "frase":      palavras,
                                                    "total":      n_frases,
                                                    "n_sinapses": n_sinapses,
                                                }).to_string();
                                                let _ = ws_tx.send(Message::text(ack)).await;
                                            }
                                        }
                                    }
                                }

                                // ── EXPORT_LINGUAGEM ───────────────────────────────────
                                // Exporta vocabulário + grafo + frases para selene_linguagem.json
                                // {"action":"export_linguagem"}
                                Some("export_linguagem") => {
                                    // Exporta a partir do swap (fonte de verdade neural)
                                    let swap_arc_ex = {
                                        let tmp = brain.lock().await;
                                        tmp.swap_manager.clone()
                                    };
                                    let (valencias_ex, grafo_ex) = if let Ok(mut sw) = swap_arc_ex.try_lock() {
                                        (sw.valencias_palavras(), sw.grafo_palavras())
                                    } else {
                                        (HashMap::new(), HashMap::new())
                                    };
                                    let causal_ex: HashMap<String, Vec<(String, f32)>> = HashMap::new();
                                    let state = brain.lock().await;
                                    let ctx_vec: Vec<String> =
                                        state.neural_context.iter().cloned().collect();
                                    let json_str = crate::storage::exportar_linguagem(
                                        &valencias_ex,
                                        &grafo_ex,
                                        &state.frases_padrao,
                                        &causal_ex,
                                        &state.grounding,
                                        &state.emocao_palavras,
                                        &state.auto_learn_contagem,
                                        &ctx_vec,
                                    );
                                    drop(state);
                                    match std::fs::write("selene_linguagem.json", &json_str) {
                                        Ok(_) => {
                                            let n_palavras = (
                                                valencias_ex.len(),
                                                grafo_ex.values().map(|v| v.len()).sum::<usize>(),
                                                { let s = brain.lock().await; s.frases_padrao.len() }
                                            );
                                            println!("💾 [LINGUAGEM] Exportado: {} palavras, {} assoc, {} frases",
                                                n_palavras.0, n_palavras.1, n_palavras.2);
                                            let ack = serde_json::json!({
                                                "event":    "linguagem_exportada",
                                                "arquivo":  "selene_linguagem.json",
                                                "palavras": n_palavras.0,
                                                "associacoes": n_palavras.1,
                                                "frases":   n_palavras.2,
                                            }).to_string();
                                            let _ = ws_tx.send(Message::text(ack)).await;
                                        }
                                        Err(e) => log::error!("[LINGUAGEM] Falha: {}", e),
                                    }
                                }

                                // ── GRAFO ROLLBACK ────────────────────────────────────
                                // Restaura o grafo de associações a partir de um snapshot.
                                // Útil quando uma sessão ruim degradou o aprendizado.
                                //
                                // {"action":"grafo_rollback", "slot": 0}  ← slot 0,1 ou 2
                                // {"action":"grafo_rollback"}              ← usa slot mais antigo
                                Some("grafo_rollback") => {
                                    let slot = json["slot"].as_u64().unwrap_or(0) % 3;
                                    let path = format!("selene_swap_snap_{}.json", slot);
                                    match std::fs::read_to_string(&path) {
                                        Ok(_) => {
                                            let swap_arc_rb = brain.lock().await.swap_manager.clone();
                                            if let Ok(mut sw) = swap_arc_rb.try_lock() {
                                                sw.carregar_estado(&path);
                                            }
                                            let (n_grafo, n_vocab) = if let Ok(sw) = swap_arc_rb.try_lock() {
                                                (sw.sinapses_conceito.len(), sw.palavra_para_id.len())
                                            } else { (0, 0) };
                                            let rc = 0u64;
                                            println!("⏪ [ROLLBACK] Swap restaurado do slot {} (reply_count={}): {} sinapses, {} palavras",
                                                slot, rc, n_grafo, n_vocab);
                                            let ack = serde_json::json!({
                                                "event":       "rollback_ok",
                                                "slot":        slot,
                                                "reply_count": rc,
                                                "grafo_nos":   n_grafo,
                                                "vocab_n":     n_vocab,
                                            }).to_string();
                                            let _ = ws_tx.send(Message::text(ack)).await;
                                        }
                                        Err(_) => {
                                            let err = serde_json::json!({"event":"rollback_erro","msg":format!("slot {} não encontrado",slot)}).to_string();
                                            let _ = ws_tx.send(Message::text(err)).await;
                                        }
                                    }
                                }

                                // ── LEARN_AUDIO_FFT ────────────────────────────────────
                                // Recebe um frame FFT de áudio (25ms), extrai primitiva de
                                // onda e persiste no banco wave-first.
                                // Nunca armazena texto — apenas parâmetros físicos.
                                //
                                // {"action":"learn_audio_fft",
                                //  "fft":[[freq_hz,amp],...],
                                //  "duracao_ms":25,
                                //  "referencia":"ma"}   ← só para log, nunca persistido
                                Some("learn_audio_fft") => {
                                    if let Some(arr) = json["fft"].as_array() {
                                        let bins: Vec<(f32, f32)> = arr.iter()
                                            .filter_map(|v| {
                                                let freq = v.get(0)?.as_f64()? as f32;
                                                let amp  = v.get(1)?.as_f64()? as f32;
                                                Some((freq, amp))
                                            })
                                            .collect();

                                        if !bins.is_empty() {
                                            let duracao_ms = json["duracao_ms"].as_u64().unwrap_or(25) as u32;
                                            let ts = std::time::SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .unwrap_or_default()
                                                .as_secs_f64();

                                            // 1. Extrai primitiva + atualiza ultimo_padrao_audio
                                            let primitiva = {
                                                let mut state = brain.lock().await;
                                                state.ws_atividade = 1.0;
                                                // Converte bins FFT → 32 bandas → SpikePattern
                                                let mut bands = [0f32; 32];
                                                for &(freq, amp) in &bins {
                                                    let idx = ((freq / 8000.0) * 32.0) as usize;
                                                    let idx = idx.min(31);
                                                    if amp > bands[idx] { bands[idx] = amp; }
                                                }
                                                state.ultimo_padrao_audio = bands_to_spike_pattern(&bands);
                                                crate::encoding::fft_encoder::fft_para_primitiva(
                                                    &bins,
                                                    &mut state.encoder_fft,
                                                    duracao_ms,
                                                    ts,
                                                )
                                            };

                                            let hash = primitiva.hash.clone();
                                            let referencia = json["referencia"].as_str().unwrap_or("?");

                                            // 2. Persiste sem manter o lock durante I/O assíncrono
                                            let db = {
                                                let state = brain.lock().await;
                                                state.storage.db.clone()
                                            };
                                            let _ = crate::storage::ondas::put_primitiva(&db, &primitiva).await;

                                            log::debug!("[ONDA] hash={} ref={} onset={:?} F1={:?} F2={:?}",
                                                &hash[..8], referencia,
                                                primitiva.onset, primitiva.f1_hz, primitiva.f2_hz);

                                            let ack = serde_json::json!({
                                                "event": "audio_ack",
                                                "hash":  hash,
                                                "onset": format!("{:?}", primitiva.onset),
                                            }).to_string();
                                            let _ = ws_tx.send(Message::text(ack)).await;
                                        }
                                    }
                                }

                                // {"action":"grounding_fonetico",
                                //  "grafema":"ba",
                                //  "letras":["b","a"],
                                //  "fonte":"baba_curriculum"}
                                //
                                // Cria a associação bidirecional:
                                //   SpikePattern(audio recente) ↔ grafema ↔ letras individuais
                                // Usa ultimo_padrao_audio gerado pelos frames FFT enviados antes.
                                Some("grounding_fonetico") => {
                                    let grafema = json["grafema"].as_str()
                                        .unwrap_or("").to_string();
                                    let letras: Vec<String> = json["letras"].as_array()
                                        .map(|arr| arr.iter()
                                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                            .collect())
                                        .unwrap_or_default();

                                    if !grafema.is_empty() {
                                        // Lista de tokens: grafema completo + letras individuais
                                        let mut palavras = vec![grafema.clone()];
                                        for l in &letras {
                                            if !palavras.contains(l) {
                                                palavras.push(l.clone());
                                            }
                                        }

                                        let mut state = brain.lock().await;
                                        let apad = state.ultimo_padrao_audio;
                                        let vpad = state.ultimo_padrao_visual;
                                        let emocao = state.atividade.2;
                                        let alerta = state.atividade.1;
                                        let ts_ms = std::time::SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .unwrap_or_default()
                                            .as_secs_f64() * 1000.0;

                                        state.grounding_bind(
                                            &palavras, vpad, apad,
                                            emocao, alerta, ts_ms,
                                        );

                                        // Boost de grounding fonético — aprendizado supervisionado direto.
                                        // A Selene sabe com certeza que aquele padrão de onda = aquelas letras.
                                        // grounding_bind já adiciona 0.15 (audio_ativo) + 0.08 (interoceptivo).
                                        // Adicionamos 0.12 extra para distinguir do grounding conversacional normal.
                                        for p in &palavras {
                                            let g = state.grounding.entry(p.clone()).or_insert(0.0);
                                            *g = (*g + 0.12).min(1.0);
                                        }

                                        // ── INTEGRAÇÃO COM O GRAFO ─────────────────────────
                                        // Sem isto, o grounding é uma ilha — a linguagem emergente
                                        // nunca consegue usar os fonemas aprendidos nas respostas.
                                        //
                                        // 1. Garante que grafema e letras existem no vocabulário
                                        // Spike vocab save
                                        state.inserir_spike_vocab(format!("audio:{}", &grafema), apad);

                                        // Grafema <-> letras + bigrams -> swap sinapses
                                        let swap_arc_gr = state.swap_manager.clone();
                                        let mut gr_pairs: Vec<(String, String, f32)> = Vec::new();
                                        for letra in &letras {
                                            if letra.is_empty() { continue; }
                                            gr_pairs.push((grafema.clone(), letra.clone(), 0.85));
                                            gr_pairs.push((letra.clone(), grafema.clone(), 0.70));
                                        }
                                        for i in 0..letras.len().saturating_sub(1) {
                                            let l1 = &letras[i];
                                            let l2 = &letras[i + 1];
                                            if l1.is_empty() || l2.is_empty() { continue; }
                                            gr_pairs.push((l1.clone(), l2.clone(), 0.90));
                                        }
                                        let n_nos = if let Ok(mut sw) = swap_arc_gr.try_lock() {
                                            sw.aprender_conceito(&grafema, 0.5);
                                            sw.importar_causal(gr_pairs);
                                            sw.sinapses_conceito.len()
                                        } else { 0 };
                                        log::debug!("[FONETICO] grounding '{}' letras={:?} sinapses={} audio_ativo={}",
                                            grafema, letras, n_nos,
                                            crate::encoding::spike_codec::is_active(&apad));

                                        let ack = serde_json::json!({
                                            "event": "grounding_ack",
                                            "grafema": grafema,
                                        }).to_string();
                                        let _ = ws_tx.send(Message::text(ack)).await;
                                    }
                                }

                                // ── LEARN_COR ─────────────────────────────────────
                                // Grounding cross-modal: espectro visual + fonemas
                                // simultâneos. A Selene vê a faixa espectral colorida,
                                // ouve e lê o nome da cor ao mesmo tempo.
                                //
                                // Protocolo:
                                //   {"action":"learn_cor","word":"rosa","nm":680,"valence":0.6}
                                //
                                // Fluxo STDP:
                                //   1. ativar_primitiva_visual(banda) → ultimo_conceito_id=banda
                                //   2. aprender_sequencia_fonemas(tags) → STDP: banda→ph:r→ph:o→...
                                //   Resultado: ver rosa ativa a cadeia fonética /r-o-z-a/
                                Some("learn_cor") => {
                                    let nm      = json["nm"].as_f64()
                                        .unwrap_or(550.0) as f32;
                                    let word    = json["word"].as_str()
                                        .unwrap_or("").to_lowercase();
                                    let valence = json["valence"].as_f64()
                                        .unwrap_or(0.5) as f32;

                                    if !word.is_empty() {
                                        // Mapeia comprimento de onda → tag primitiva visual
                                        let banda = nm_para_banda(nm);

                                        // Decompõe o nome da cor em fonemas PT-BR
                                        use crate::encoding::phoneme::word_to_phonemes;
                                        let fonemas = word_to_phonemes(&word);
                                        let tags: Vec<String> = fonemas.iter()
                                            .filter(|&&ph| {
                                                ph != crate::encoding::phoneme::Phoneme::SIL
                                            })
                                            .map(|ph| format!(
                                                "ph:{}",
                                                format!("{:?}", ph).to_lowercase()
                                            ))
                                            .collect();

                                        let state = brain.lock().await;
                                        let mut swap = state.swap_manager.lock().await;

                                        // Visual primeiro — define ultimo_conceito_id = banda
                                        // para que o STDP no passo 2 crie banda→fonema
                                        swap.ativar_primitiva_visual(banda, valence);

                                        // Fonemas segundo — STDP natural: banda→ph:r→ph:o→...
                                        if !tags.is_empty() {
                                            swap.aprender_sequencia_fonemas(&tags, valence);
                                        }

                                        drop(swap);
                                        drop(state);

                                        let ack = serde_json::json!({
                                            "event":   "cor_aprendida",
                                            "word":    word,
                                            "nm":      nm,
                                            "banda":   banda,
                                            "fonemas": tags,
                                            "valence": valence,
                                        }).to_string();
                                        let _ = ws_tx.send(Message::text(ack)).await;
                                    }
                                }

                                _ => {
                                    log::debug!("[WS] Ação desconhecida: {:?}", json["action"]);
                                }
                            } // fim match json["action"]
                            } else {
                                // Mensagem simples de chat (texto puro — interface mobile/desktop)
                                println!("💬 [CHAT] Recebido: {}", text);
                                // Snapshot neural: clona Arc antes de bloquear brain
                                let swap_arc_s = {
                                    let tmp = brain.lock().await;
                                    tmp.swap_manager.clone()
                                };
                                let (grafo_neural_s, valencias_neural_s, scaffold_s) = {
                                    let mut s = swap_arc_s.lock().await;
                                    let tokens_s: Vec<String> = text.to_lowercase()
                                        .split(|c: char| !c.is_alphanumeric())
                                        .filter(|t| t.len() > 1)
                                        .map(|t| t.to_string())
                                        .collect();
                                    let (scaffold, _) = s.template_scaffold(&tokens_s);
                                    (s.grafo_palavras(), s.valencias_palavras(), scaffold)
                                };
                                let causal_vazio_s: HashMap<String, Vec<(String, f32)>> = HashMap::new();
                                let mut state = brain.lock().await;
                                let (dopa, sero, _nor) = state.neurotransmissores;
                                let (step, alerta, emocao) = state.atividade;

                                let msg_lower = text.to_lowercase();
                                let valence = msg_lower.split_whitespace()
                                    .filter_map(|t| valencias_neural_s.get(t).copied())
                                    .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
                                    .unwrap_or(0.0);

                                let emocao_resp = (emocao * 0.6 + valence * 0.4).clamp(-1.0, 1.0);
                                state.ws_atividade = 1.0;

                                let n_passos = ((state.n_passos_walk as f32 + state.habituation_nivel * 3.0) as usize).clamp(4, 18);
                                let emocao_bias = state.emocao_bias;
                                state.reply_count = state.reply_count.wrapping_add(1);
                                let diversity_seed = step ^ state.reply_count.wrapping_mul(6364136223846793005);
                                let mut caminho_q: Vec<String> = Vec::new();
                                let mut prefixo_buf: Vec<String> = Vec::new();
                                let mut ancora_log2: Option<String> = None;
                                let ctx_log2: Vec<String> = {
                                    let mut ctx = conversa_ctx.clone();
                                    ctx.extend(state.frontal_goal_words.iter().cloned());
                                    ctx
                                };
                                let reply = gerar_resposta_emergente(
                                    &text, diversity_seed, emocao_resp,
                                    emocao_bias, n_passos,
                                    dopa, sero,
                                    &valencias_neural_s,
                                    &grafo_neural_s,
                                    &state.frases_padrao,
                                    &state.indice_prefixo,
                                    &state.ultimos_prefixos.iter().cloned().collect::<Vec<_>>(),
                                    &ctx_log2,
                                    &mut caminho_q,
                                    &state.emocao_palavras,
                                    &mut prefixo_buf,
                                    &state.grounding,
                                    &causal_vazio_s,
                                    &state.palavra_qvalores,
                                    &mut ancora_log2,
                                    &state.ultimo_estado_corpo,
                                    &scaffold_s,
                                    &state.trigrama_cache,
                                );
                                // ── LOG ─────────────────────────────────────────
                                {
                                    let qbias: Vec<(String, f32)> = {
                                        let mut v: Vec<(String, f32)> = state.palavra_qvalores
                                            .iter().filter(|(_, &q)| q.abs() > 0.05)
                                            .map(|(w, &q)| (w.clone(), q)).collect();
                                        v.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
                                        v.truncate(3);
                                        v
                                    };
                                    log_turno(&text, ancora_log2.as_deref(), n_passos,
                                        emocao_resp, dopa, sero,
                                        valencias_neural_s.len(),
                                        grafo_neural_s.len(),
                                        &caminho_q, &prefixo_buf, &reply, reply.is_empty(),
                                        &ctx_log2, &qbias, state.ultimo_rpe);
                                }
                                state.ultimos_prefixos.push_back(prefixo_buf);
                                if state.ultimos_prefixos.len() > 5 { state.ultimos_prefixos.pop_front(); }
                                state.ultimo_caminho_walk = caminho_q.clone();
                                for i in 0..caminho_q.len().saturating_sub(1) {
                                    let par = (caminho_q[i].clone(), caminho_q[i+1].clone());
                                    *state.aresta_contagem.entry(par).or_insert(0) += 1;
                                }
                                state.ultima_atividade = std::time::Instant::now();

                                // Registra pensamento
                                let pensamento = format!("Pergunta: «{}» (val={:+.2})", text, valence);
                                if state.ego.pensamentos_recentes.len() >= 10 {
                                    state.ego.pensamentos_recentes.pop_front();
                                }
                                state.ego.pensamentos_recentes.push_back(pensamento);
                                state.atividade = (step, alerta, emocao_resp);

                                let resp = serde_json::json!({
                                    "event":   "chat_reply",
                                    "message": reply,
                                    "emotion": emocao_resp,
                                    "arousal": alerta,
                                }).to_string();
                                let (dop2, ser2, nor2) = state.neurotransmissores;
                                let formants = sentence_to_formants(&reply, dop2, ser2, nor2);
                                drop(state);
                                println!("💬 [CHAT] Resposta: {}", reply);
                                let _ = ws_tx.send(Message::text(resp)).await;
                                if let Ok(fj) = serde_json::to_string(&formants) {
                                    let voz = format!(r#"{{"event":"voz_params","formants":{}}}"#, fj);
                                    let _ = ws_tx.send(Message::text(voz)).await;
                                }
                            }
                        }
                    }

                    Err(e) => {
                        log::error!("Erro ao ler mensagem do WebSocket: {}", e);
                        break;
                    }
                }
            }
        }
    }

    println!("   ❌ Conexão WebSocket encerrada.");
}