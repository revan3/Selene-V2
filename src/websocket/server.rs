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

use std::collections::HashMap;
use chrono::Timelike;

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

/// Gera resposta emergente a partir do vocabulário e grafo de associações reais da Selene.
/// Não usa templates fixos — a resposta é construída navegando o que ela aprendeu.
fn gerar_resposta_emergente(
    pergunta: &str,
    step: u64,
    emocao: f32,
    emocao_bias: f32,   // viés de Plutchik: joy - fear - sadness
    n_passos: usize,    // profundidade do walk — controlada pela onda dominante
    dopa: f32,
    sero: f32,
    valencias: &HashMap<String, f32>,
    grafo: &HashMap<String, Vec<(String, f32)>>,
    frases_padrao: &[Vec<String>],
    evitar: &[Vec<String>],                     // prefixos usados recentemente → cooldown de frase
    contexto_extra: &[String],                  // palavras de turnos anteriores → expande o tópico
    caminho_out: &mut Vec<String>,              // registra o caminho percorrido → usado pelo feedback
    emocao_palavras: &HashMap<String, f32>,     // valência emocional amigdaliana por palavra
    prefixo_usado_out: &mut Vec<String>,        // prefixo escolhido → registrado no cooldown
    grounding: &HashMap<String, f32>,           // grounding semântico por palavra (0=linguístico, 1=grounded)
) -> String {
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
        // Pontua todas as frases pelo overlap com o topico_prefixo (sem auto-referências)
        let scored: Vec<(usize, usize)> = frases_padrao.iter()
            .enumerate()
            .map(|(i, frase)| (i, frase.iter().filter(|w| topico_prefixo.contains(w.as_str())).count()))
            .collect();

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

    // Coerência sintática: pré-computa quais (word_a, word_b) → word_c aparecem nas frases_padrao.
    // Durante o walk, damos bonus de score quando a próxima palavra segue um trigrama conhecido.
    // Isso implementa um modelo de linguagem bigrama/trigrama emergente dos padrões aprendidos.
    let trigrama_bonus: std::collections::HashMap<(String, String), Vec<String>> = {
        let mut mapa: std::collections::HashMap<(String, String), Vec<String>> = std::collections::HashMap::new();
        for frase in frases_padrao {
            for w in frase.windows(3) {
                mapa.entry((w[0].clone(), w[1].clone()))
                    .or_default()
                    .push(w[2].clone());
            }
        }
        mapa
    };

    for _ in 0..n_passos {
        if visitados.contains(&atual) { break; }
        visitados.insert(atual.clone());
        cadeia.push(atual.clone());

        if let Some(vizinhos) = grafo.get(&atual) {
            let nao_visitados: Vec<&(String, f32)> = vizinhos.iter()
                .filter(|(w, peso)| !visitados.contains(w.as_str()) && *peso > -0.1)
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
                    // Uma palavra grounded (0.8) ganha bonus de -0.12 no score (menor = preferida).
                    // Isso faz o walk "ancorar na realidade" ao invés de derivar no espaço linguístico.
                    s1 -= grounding.get(a.0.as_str()).copied().unwrap_or(0.0) * 0.15;
                    s2 -= grounding.get(b.0.as_str()).copied().unwrap_or(0.0) * 0.15;
                    // Coerência sintática trigrama: se (prev, atual) → next existe nos padrões,
                    // dá um bonus à palavra que continua o padrão aprendido.
                    if let Some(prev) = cadeia.iter().rev().nth(1) {
                        let chave = (prev.clone(), atual.clone());
                        if let Some(next_opts) = trigrama_bonus.get(&chave) {
                            if next_opts.contains(&a.0) { s1 -= 0.20; }
                            if next_opts.contains(&b.0) { s2 -= 0.20; }
                        }
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
            let novo = vistas.insert(w.clone()); // false se já viu
            // stop word tardias (pos > 4) são removidas se repetidas
            if !novo { return false; }
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
// Operam diretamente sobre o grafo_associacoes / aresta_contagem do BrainState.
// São funções síncronas leves — chamadas dentro do lock do brain.

/// N1 — Consolida as 30 arestas mais percorridas (+0.07 de peso).
fn fase_n1_consolidar(state: &mut crate::websocket::bridge::BrainState) {
    let mut pares: Vec<((String, String), u32)> = state.aresta_contagem
        .iter().map(|(k, v)| (k.clone(), *v)).collect();
    pares.sort_by_key(|(_, v)| std::cmp::Reverse(*v));
    let mut consolidados = 0usize;
    for ((a, b), _) in pares.into_iter().take(30) {
        if let Some(vizinhos) = state.grafo_associacoes.get_mut(&a) {
            for (w, peso) in vizinhos.iter_mut() {
                if w == &b { *peso = (*peso + 0.07).clamp(0.0, 1.0); consolidados += 1; }
            }
        }
    }
    state.aresta_contagem.clear();
    println!("   💪 [N1] {} sinapses consolidadas", consolidados);
}

/// N2 — Poda arestas muito fracas (peso < 0.12) que não foram percorridas.
fn fase_n2_podar(state: &mut crate::websocket::bridge::BrainState) {
    let percorridas: std::collections::HashSet<(String, String)> =
        state.aresta_contagem.keys().cloned().collect();
    let mut podadas = 0usize;
    for vizinhos in state.grafo_associacoes.values_mut() {
        let antes = vizinhos.len();
        vizinhos.retain(|(w, peso)| {
            *peso >= 0.12 || percorridas.contains(&(String::new(), w.clone()))
        });
        podadas += antes - vizinhos.len();
    }
    // Remove nós sem vizinhos
    state.grafo_associacoes.retain(|_, v| !v.is_empty());
    println!("   ✂️  [N2] {} arestas podadas", podadas);
}

/// N3 — REM: caminhada criativa — cria novas associações transitivas no grafo.
/// Liga pares de palavras que compartilham um vizinho em comum mas não estão
/// diretamente conectadas — o equivalente a "sonhar" e formar novas ideias.
fn fase_n3_rem(state: &mut crate::websocket::bridge::BrainState) {
    use crate::encoding::spike_codec::is_active;
    let palavras: Vec<String> = state.grafo_associacoes.keys().cloned().collect();
    if palavras.len() < 3 { return; }
    let mut novas = 0usize;
    let mut replay = 0usize;

    // ── Parte 1: Replay hipocampal rico — usa EventoEpisodico completo ────────
    // Durante N3/REM, o hipocampo replica eventos com contexto perceptual real.
    // Eventos com percepção visual/auditiva real recebem bônus extra de consolidação.
    // Biologicamente: replay de "place cell" sequences durante NREM sharp-wave ripples.
    let episodios: Vec<crate::websocket::bridge::EventoEpisodico> = state.historico_episodico
        .iter()
        .filter(|ev| ev.emocao.abs() > 0.35) // só episódios emocionalmente salientes
        .cloned()
        .collect();

    for ev in &episodios {
        let bonus = ev.emocao.abs() * 0.06;
        // Reforça todos os pares consecutivos de palavras do evento
        for i in 0..ev.palavras.len().saturating_sub(1) {
            let wa = &ev.palavras[i];
            let wb = &ev.palavras[i + 1];
            if let Some(vizinhos) = state.grafo_associacoes.get_mut(wa) {
                if let Some(aresta) = vizinhos.iter_mut().find(|(w, _)| w == wb) {
                    aresta.1 = (aresta.1 + bonus).min(0.98);
                    replay += 1;
                } else {
                    vizinhos.push((wb.clone(), (0.35 + bonus).min(0.65)));
                    replay += 1;
                    novas += 1;
                }
            }
            // Atualiza peso emocional
            let entry = state.emocao_palavras.entry(wa.clone()).or_insert(0.0);
            *entry = (*entry * 0.90 + ev.emocao * 0.10).clamp(-1.0, 1.0);
        }
        // Bônus de grounding via replay: percepção real durante o evento
        // faz com que as palavras ganhem grounding extra ao serem re-experienciadas.
        let visual_ativo = is_active(&ev.padrao_visual);
        let audio_ativo  = is_active(&ev.padrao_audio);
        for w in &ev.palavras {
            let g = state.grounding.entry(w.clone()).or_insert(0.0);
            if visual_ativo { *g = (*g + bonus * 0.4).min(1.0); }
            if audio_ativo  { *g = (*g + bonus * 0.25).min(1.0); }
            *g = (*g + bonus * 0.15).min(1.0); // interoceptivo
        }
    }

    // ── Parte 2: Fechamento transitivo criativo ───────────────────────────────
    // Cria associações entre palavras que compartilham vizinhos mas não estão
    // diretamente conectadas — o equivalente a "sonhar" e formar novas ideias.
    for i in 0..palavras.len().min(25) {
        let a = &palavras[i];
        let b_idx = (i * 7 + 3) % palavras.len(); // pseudoaleatório determinístico
        let b = &palavras[b_idx];
        if a == b { continue; }
        let vizinhos_a: std::collections::HashSet<String> = state.grafo_associacoes
            .get(a).map(|v| v.iter().map(|(w, _)| w.clone()).collect())
            .unwrap_or_default();
        let vizinhos_b: std::collections::HashSet<String> = state.grafo_associacoes
            .get(b).map(|v| v.iter().map(|(w, _)| w.clone()).collect())
            .unwrap_or_default();
        let em_comum = vizinhos_a.intersection(&vizinhos_b).count();
        if em_comum > 0 {
            let a = a.clone(); let b = b.clone();
            let ja_existe = state.grafo_associacoes.get(&a)
                .map(|v| v.iter().any(|(w, _)| w == &b)).unwrap_or(false);
            if !ja_existe {
                let peso = (0.30 + em_comum as f32 * 0.08).clamp(0.30, 0.65);
                state.grafo_associacoes.entry(a).or_default().push((b, peso));
                novas += 1;
            }
        }
    }
    println!("   ✨ [N3 REM] {} novas assoc | {} replays hipocampais ({} episódios salientes)",
        novas, replay, episodios.len());
}

/// N4 — Backup: persiste linguagem em JSON.
fn fase_n4_backup(state: &crate::websocket::bridge::BrainState) {
    let backup = serde_json::json!({
        "selene_linguagem_v1": {
            "vocabulario": state.palavra_valencias,
            "grafo": state.grafo_associacoes,
            "frases_padrao": state.frases_padrao,
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

                                // Envia snapshot do grafo mental para visualização
                                Some("vocab_request") => {
                                    let state = brain.lock().await;
                                    // Top 120 palavras por peso absoluto de valência
                                    let mut palavras: Vec<(&String, f32)> = state.palavra_valencias
                                        .iter().map(|(k, v)| (k, v.abs())).collect();
                                    palavras.sort_by(|a, b| b.1.partial_cmp(&a.1)
                                        .unwrap_or(std::cmp::Ordering::Equal));
                                    let top: std::collections::HashSet<&str> = palavras.iter()
                                        .take(120).map(|(k, _)| k.as_str()).collect();

                                    let nodes: Vec<serde_json::Value> = top.iter().map(|&w| {
                                        let weight = state.palavra_valencias.get(w)
                                            .map(|v| v.abs()).unwrap_or(0.005);
                                        serde_json::json!({"id": w, "weight": weight})
                                    }).collect();

                                    let mut links: Vec<serde_json::Value> = vec![];
                                    for (word, neighbors) in &state.grafo_associacoes {
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
                                        "total_palavras": state.palavra_valencias.len(),
                                        "total_assoc": state.grafo_associacoes.len(),
                                    });
                                    let _ = ws_tx.send(Message::text(snapshot.to_string())).await;
                                }

                                Some("shutdown") => {
                                    println!("🛑 [SISTEMA] Shutdown solicitado pela interface neural.");
                                    brain.lock().await.shutdown_requested = true;
                                    let ack = r#"{"event":"shutdown_ack","msg":"Iniciando desligamento gracioso..."}"#;
                                    let _ = ws_tx.send(Message::text(ack)).await;
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

                                Some("chat") => {
                                    // Mensagem de chat vinda da interface mobile/desktop
                                    let mensagem = json["message"].as_str().unwrap_or("").to_string();
                                    println!("💬 [CHAT] Recebido (raw): «{}»", mensagem);
                                    if !mensagem.is_empty() {
                                        println!("💬 [CHAT] Aguardando lock do brain...");
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
                                            let val_clone = state.palavra_valencias.clone();
                                            let rpe_hip = state.hypothesis_engine
                                                .testar(&input_tokens_hip, &val_clone);
                                            if rpe_hip.abs() > 0.05 {
                                                state.grounding_rpe(rpe_hip);
                                            }
                                        }
                                        println!("💬 [CHAT] Estado: step={} emocao={:.2} dopa={:.2} vocab={} grafo={} frases={}",
                                            step, emocao, dopa,
                                            state.palavra_valencias.len(),
                                            state.grafo_associacoes.len(),
                                            state.frases_padrao.len());

                                        // Busca valência: varredura palavra-a-palavra na mensagem
                                        let msg_lower = mensagem.to_lowercase();
                                        let (valence, palavra_chave) = {
                                            let mut best_val = state.palavra_valencias
                                                .get(&msg_lower).copied().unwrap_or(0.0);
                                            let mut best_word: Option<String> = if best_val != 0.0 {
                                                Some(msg_lower.clone())
                                            } else { None };
                                            for token in msg_lower.split_whitespace() {
                                                if let Some(&v) = state.palavra_valencias.get(token) {
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

                                        let n_passos = ((state.n_passos_walk as f32
                                            + t_curiosa * 2.5
                                            - t_cautelosa * 2.0
                                            + t_reflexiva * 1.0) as usize).clamp(4, 16);
                                        // Cautelosa amorteça o viés emocional → respostas mais neutras
                                        let emocao_bias = state.emocao_bias * (1.0 - t_cautelosa * 0.5);
                                        // Curiosa reduz limiar de disparo de pergunta autônoma
                                        let curiosity_threshold = if t_curiosa > 0.7 { 0.55 } else { 0.75 };

                                        state.reply_count = state.reply_count.wrapping_add(1);
                                        let diversity_seed = step ^ state.reply_count.wrapping_mul(6364136223846793005);
                                        // caminho_local evita conflito de borrow entre &state.* e &mut state.*
                                        let mut caminho_local: Vec<String> = Vec::new();
                                        let mut prefixo_buf: Vec<String> = Vec::new();
                                        let reply = gerar_resposta_emergente(
                                            &mensagem, diversity_seed, emocao_resposta,
                                            emocao_bias, n_passos,
                                            dopa, sero,
                                            &state.palavra_valencias,
                                            &state.grafo_associacoes,
                                            &state.frases_padrao,
                                            &state.ultimos_prefixos.iter().cloned().collect::<Vec<_>>(),
                                            // Contexto = histórico da conversa + neural_context +
                                            // pensamentos conscientes do Eternal Hole (últimos 5) +
                                            // emergência inconsciente ocasional (1/7 replies).
                                            &{
                                                let mut ctx = conversa_ctx.clone();
                                                // O que o cérebro processa agora (chunks + frontal goals)
                                                ctx.extend(state.neural_context.iter().cloned());
                                                // Eternal Hole consciente: palavras que Selene
                                                // "pensou" nos últimos ciclos de 50Hz
                                                ctx.extend(
                                                    state.pensamento_consciente
                                                        .iter()
                                                        .cloned()
                                                        .take(5)
                                                );
                                                // Emergência inconsciente: uma vez a cada 7 respostas
                                                // um pensamento derivado pode tingir a resposta
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
                                        );
                                        state.ultimos_prefixos.push_back(prefixo_buf);
                                        if state.ultimos_prefixos.len() > 5 { state.ultimos_prefixos.pop_front(); }
                                        state.ultimo_caminho_walk = caminho_local.clone();
                                        // Atualiza contagem de arestas usadas (para consolidação noturna)
                                        let caminho = caminho_local;
                                        for i in 0..caminho.len().saturating_sub(1) {
                                            let par = (caminho[i].clone(), caminho[i+1].clone());
                                            *state.aresta_contagem.entry(par).or_insert(0) += 1;
                                        }

                                        // Fix 1: LTD/LTP no grafo via RPE (desaprender associações erradas).
                                        // RPE > +0.25: resposta melhor que esperado → reforça arestas (LTP).
                                        // RPE < -0.25: resposta pior que esperado → enfraquece arestas (LTD).
                                        // Magnitude pequena (0.02) para não destruir o grafo num único turno.
                                        let rpe = state.ultimo_rpe;
                                        if rpe.abs() > 0.25 {
                                            let delta_rpe = rpe.signum() * 0.02;
                                            let caminho_rpe = state.ultimo_caminho_walk.clone();
                                            for i in 0..caminho_rpe.len().saturating_sub(1) {
                                                let (a, b) = (&caminho_rpe[i], &caminho_rpe[i+1]);
                                                if let Some(vizinhos) = state.grafo_associacoes.get_mut(a) {
                                                    for (w, peso) in vizinhos.iter_mut() {
                                                        if w == b {
                                                            *peso = (*peso + delta_rpe).clamp(0.01, 1.0);
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        // Decaimento temporal: arestas não usadas recentemente perdem peso.
                                        // Evita que associações erradas do passado persistam para sempre.
                                        // Roda a cada ~500 respostas para não ser caro.
                                        if state.reply_count % 500 == 0 {
                                            for vizinhos in state.grafo_associacoes.values_mut() {
                                                for (_, peso) in vizinhos.iter_mut() {
                                                    *peso = (*peso * 0.995).max(0.01);
                                                }
                                                // Remove arestas completamente esquecidas (peso ≤ 0.01)
                                                vizinhos.retain(|(_, p)| *p > 0.01);
                                            }
                                        }

                                        state.ultima_atividade = std::time::Instant::now();

                                        // Auto-learn progressivo: cada menção de palavra desconhecida
                                        // acumula exposições. Valência escala com o contador:
                                        //   1ª exposição → ±0.15  (eco emocional fraco)
                                        //   2ª exposição → ±0.30  (começa a influenciar o walk)
                                        //   3ª+          → ±0.45  (influência significativa, capped)
                                        for token in msg_lower.split_whitespace() {
                                            if token.len() > 2
                                                && token.chars().all(|c| c.is_alphabetic() || "áéíóúâêôãõçàü".contains(c))
                                            {
                                                let contagem = state.auto_learn_contagem
                                                    .entry(token.to_string()).or_insert(0);
                                                *contagem += 1;
                                                let escala = (*contagem as f32 * 0.5).min(1.5); // 0.5, 1.0, 1.5
                                                let val_auto = (emocao_resposta * 0.15 * escala)
                                                    .clamp(-0.45, 0.45);
                                                // EMA sobre o valor existente (se já existe, blenda)
                                                let val_existente = state.palavra_valencias
                                                    .get(token).copied().unwrap_or(val_auto);
                                                state.palavra_valencias.insert(
                                                    token.to_string(),
                                                    val_existente * 0.7 + val_auto * 0.3,
                                                );
                                            }
                                        }
                                        println!("💬 [CHAT] Reply gerado: «{}»", reply);

                                        // ── HIPÓTESES: formula predições + observa próprio comportamento ──
                                        // 1. Formula hipóteses sobre o que o próximo input provavelmente conterá,
                                        //    baseado no contexto atual e no grafo de associações.
                                        // 2. Observa o próprio padrão de resposta — quando o mesmo padrão
                                        //    se repete 3× gera uma hipótese ComportamentalPropria,
                                        //    que é o radar inicial para futura autoprogramação.
                                        {
                                            let ctx_hip: Vec<String> = {
                                                let mut c = conversa_ctx.clone();
                                                c.extend(state.neural_context.iter().cloned());
                                                c
                                            };
                                            let grafo_ref = &state.grafo_associacoes.clone();
                                            let val_ref   = &state.palavra_valencias.clone();
                                            state.hypothesis_engine.formular(
                                                &ctx_hip, grafo_ref, val_ref,
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
                                            let lacunas = detectar_lacunas(
                                                &state.grafo_associacoes,
                                                &state.palavra_valencias,
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
                                            state.spike_vocab.insert(chave.clone(), audio_pat);
                                            if let Some(ref mut helix) = state.helix {
                                                let _ = helix.insert(&chave, &audio_pat);
                                            }
                                        }

                                        // Associa palavras consecutivas no grafo (co-ocorrência auditiva)
                                        // Peso ligeiramente menor que aprendizado explícito (0.60)
                                        let emocao_atual = state.emocao_bias;
                                        for i in 0..palavras.len().saturating_sub(1) {
                                            let w1 = palavras[i].clone();
                                            let w2 = palavras[i + 1].clone();
                                            if w1.len() > 1 && w2.len() > 1 {
                                                let vizinhos = state.grafo_associacoes
                                                    .entry(w1.clone()).or_default();
                                                if !vizinhos.iter().any(|(w, _)| w == &w2) {
                                                    vizinhos.push((w2.clone(), 0.60));
                                                }
                                                // Grounding binding auditivo — audio_learn é um evento
                                                // rico: o usuário fala enquanto Selene ouve as bandas FFT.
                                                // É o caso mais puro de grounding: som → palavra → percepção.
                                                if emocao_atual.abs() > 0.10 || true {
                                                    // Sempre registra o binding auditivo (baixo custo)
                                                    let vpad = state.ultimo_padrao_visual;
                                                    let palavras_par = vec![w1.clone(), w2.clone()];
                                                    state.grounding_bind(
                                                        &palavras_par,
                                                        vpad, audio_pat,
                                                        emocao_atual, 0.5,
                                                        0.0, // tempo_ms não disponível aqui
                                                    );
                                                }
                                            }
                                        }

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
                                            state.spike_vocab.insert(chave.clone(), visual_pat);
                                            if let Some(ref mut helix) = state.helix {
                                                let _ = helix.insert(&chave, &visual_pat);
                                            }
                                        }

                                        // Associa palavras consecutivas no grafo (co-ocorrência visual)
                                        for i in 0..palavras.len().saturating_sub(1) {
                                            let w1 = palavras[i].clone();
                                            let w2 = palavras[i + 1].clone();
                                            if w1.len() > 1 && w2.len() > 1 {
                                                let vizinhos = state.grafo_associacoes
                                                    .entry(w1.clone()).or_default();
                                                if !vizinhos.iter().any(|(w, _)| w == &w2) {
                                                    vizinhos.push((w2, 0.55));
                                                }
                                            }
                                        }

                                        let n = palavras.len();
                                        drop(state);
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

                                    let valencias = {
                                        let state = brain.lock().await;
                                        state.palavra_valencias.clone()
                                    };
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
                                    let texto   = json["text"].as_str().unwrap_or("").to_string();
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
                                        // EMA de valências: mistura exponencial com o valor existente.
                                        // α=0.3 → novo sinal pesa 30%, histórico pesa 70%.
                                        // Evita que uma exposição acidental apague aprendizados anteriores.
                                        let texto_lower = texto.to_lowercase();
                                        let val_atual = state.palavra_valencias
                                            .get(&texto_lower).copied().unwrap_or(valence);
                                        let val_ema = val_atual * 0.7 + valence * 0.3;
                                        state.palavra_valencias.insert(texto_lower.clone(), val_ema);

                                        // ── SPIKE ENCODING (Helix) ─────────────────────
                                        // Codifica a palavra em padrão spike e persiste no HelixStore.
                                        let spike_pat = spike_encode(&texto_lower);
                                        state.spike_vocab.insert(texto_lower.clone(), spike_pat);
                                        if let Some(ref mut helix) = state.helix {
                                            let _ = helix.insert(&texto_lower, &spike_pat);
                                        }

                                        // ── APRENDIZADO BIOLÓGICO ──────────────────────
                                        // Cria/recupera neurônio Izhikevich para este conceito
                                        // e injeta corrente proporcional à valência.
                                        // Conecta via STDP com o conceito aprendido anteriormente.
                                        {
                                            let mut swap = state.swap_manager.lock().await;
                                            swap.aprender_conceito(&texto, valence);
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

                                // ── REWARD ────────────────────────────────────────────
                                Some("reward") => {
                                    let value = json["value"].as_f64().unwrap_or(0.3) as f32;
                                    let mut state = brain.lock().await;
                                    state.ws_atividade = 1.0;
                                    let (dopa, sero, nor) = state.neurotransmissores;
                                    let nova_dopa = (dopa + value * 0.5).clamp(0.0, 2.0);
                                    let nova_sero = (sero + value * 0.2).clamp(0.0, 1.5);
                                    state.neurotransmissores = (nova_dopa, nova_sero, nor);
                                    state.ego.pensamentos_recentes.push_back(
                                        format!("Recompensa recebida (+{:.2}) → dopamina={:.3} serotonina={:.3}", value, nova_dopa, nova_sero)
                                    );
                                    if state.ego.pensamentos_recentes.len() > 10 { state.ego.pensamentos_recentes.pop_front(); }
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
                                    state.ego.pensamentos_recentes.push_back(
                                        format!("Punição recebida (-{:.2}) → dopamina={:.3} noradrenaline={:.3}", value, nova_dopa, nova_nor)
                                    );
                                    if state.ego.pensamentos_recentes.len() > 10 { state.ego.pensamentos_recentes.pop_front(); }
                                    let ack = serde_json::json!({
                                        "event":          "punish_ack",
                                        "dopamine":       nova_dopa,
                                        "noradrenaline":  nova_nor,
                                    }).to_string();
                                    let _ = ws_tx.send(Message::text(ack)).await;
                                    println!("⚡ [PUNISH] -{:.2} → dopa={:.3} nor={:.3}", value, nova_dopa, nova_nor);
                                }

                                // ── FEEDBACK ──────────────────────────────────────────
                                // Reforça ou penaliza as arestas do último walk gerado.
                                // value > 0 = positivo (reforça caminho), value < 0 = negativo (penaliza).
                                // Implementa aprendizado por reforço no grafo semântico.
                                Some("feedback") => {
                                    let value = json["value"].as_f64().unwrap_or(0.3) as f32;
                                    let value = value.clamp(-1.0, 1.0);
                                    let delta = value * 0.08; // ±0.08 por feedback
                                    let mut state = brain.lock().await;
                                    let caminho = state.ultimo_caminho_walk.clone();
                                    let mut reforcos = 0usize;
                                    for i in 0..caminho.len().saturating_sub(1) {
                                        let (a, b) = (&caminho[i], &caminho[i+1]);
                                        if let Some(vizinhos) = state.grafo_associacoes.get_mut(a) {
                                            for (w, peso) in vizinhos.iter_mut() {
                                                if w == b {
                                                    *peso = (*peso + delta).clamp(0.0, 1.0);
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

                                        // Verifica conexão real no grafo_associacoes (ambas as direções)
                                        // Antes: usava média de |valências| — reportava conexão em palavras
                                        // nunca associadas. Agora: consulta a aresta real do grafo.
                                        let edge_fwd = state.grafo_associacoes
                                            .get(&w1_low)
                                            .and_then(|v| v.iter().find(|(w, _)| w == &w2_low))
                                            .map(|(_, p)| *p);
                                        let edge_rev = state.grafo_associacoes
                                            .get(&w2_low)
                                            .and_then(|v| v.iter().find(|(w, _)| w == &w1_low))
                                            .map(|(_, p)| *p);

                                        // Força = média das direções presentes (bidirecional assimétrico)
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
                                        let mut state = brain.lock().await;
                                        state.ws_atividade = 1.0;

                                        // Auto-valence: garante que as palavras existam no léxico
                                        // com valência neutra (0.0), para que check_connection funcione
                                        // mesmo quando apenas associate foi usado (sem learn prévio).
                                        state.palavra_valencias.entry(w1.clone()).or_insert(0.0);
                                        state.palavra_valencias.entry(w2.clone()).or_insert(0.0);

                                        // Deduplicação: verifica se aresta já existe antes de inserir.
                                        // Se já existe, atualiza o peso (mantém o maior) sem duplicar.
                                        let existe_fwd = state.grafo_associacoes
                                            .get(&w1).map_or(false, |v| v.iter().any(|(w, _)| w == &w2));
                                        if existe_fwd {
                                            // Atualiza peso se o novo for maior
                                            if let Some(v) = state.grafo_associacoes.get_mut(&w1) {
                                                if let Some(entry) = v.iter_mut().find(|(w, _)| w == &w2) {
                                                    entry.1 = entry.1.max(weight);
                                                }
                                            }
                                        } else {
                                            state.grafo_associacoes
                                                .entry(w1.clone()).or_default()
                                                .push((w2.clone(), weight));
                                        }

                                        // Bidirecional (peso reduzido) — mesma lógica de dedup
                                        let existe_rev = state.grafo_associacoes
                                            .get(&w2).map_or(false, |v| v.iter().any(|(w, _)| w == &w1));
                                        if existe_rev {
                                            if let Some(v) = state.grafo_associacoes.get_mut(&w2) {
                                                if let Some(entry) = v.iter_mut().find(|(w, _)| w == &w1) {
                                                    entry.1 = entry.1.max(weight * 0.6);
                                                }
                                            }
                                        } else {
                                            state.grafo_associacoes
                                                .entry(w2.clone()).or_default()
                                                .push((w1.clone(), weight * 0.6));
                                        }

                                        let n_assoc: usize = state.grafo_associacoes
                                            .values().map(|v| v.len()).sum();
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
                                Some("learn_frase") => {
                                    if let Some(arr) = json["words"].as_array() {
                                        let palavras: Vec<String> = arr.iter()
                                            .filter_map(|v| v.as_str().map(|s| s.to_lowercase()))
                                            .filter(|s| !s.is_empty())
                                            .take(7)  // limite: frases longas viram prefixos inutilizáveis
                                            .collect();
                                        if palavras.len() >= 2 {
                                            let mut state = brain.lock().await;
                                            state.ws_atividade = 1.0;
                                            // Deduplicação: não adiciona se a sequência já existe
                                            let ja_existe = state.frases_padrao.iter().any(|f| f == &palavras);
                                            if !ja_existe {
                                                state.frases_padrao.push(palavras.clone());
                                            }
                                            let ack = serde_json::json!({
                                                "event":  "frase_ack",
                                                "frase":  palavras,
                                                "total":  state.frases_padrao.len(),
                                            }).to_string();
                                            let _ = ws_tx.send(Message::text(ack)).await;
                                        }
                                    }
                                }

                                // ── EXPORT_LINGUAGEM ───────────────────────────────────
                                // Exporta vocabulário + grafo + frases para selene_linguagem.json
                                // {"action":"export_linguagem"}
                                Some("export_linguagem") => {
                                    let state = brain.lock().await;
                                    let json_str = crate::storage::exportar_linguagem(
                                        &state.palavra_valencias,
                                        &state.grafo_associacoes,
                                        &state.frases_padrao,
                                    );
                                    drop(state);
                                    match std::fs::write("selene_linguagem.json", &json_str) {
                                        Ok(_) => {
                                            let n_palavras = {
                                                let s = brain.lock().await;
                                                (s.palavra_valencias.len(),
                                                 s.grafo_associacoes.values().map(|v| v.len()).sum::<usize>(),
                                                 s.frases_padrao.len())
                                            };
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
                                        state.palavra_valencias.entry(grafema.clone()).or_insert(0.5);
                                        for letra in &letras {
                                            if !letra.is_empty() {
                                                state.palavra_valencias.entry(letra.clone()).or_insert(0.5);
                                            }
                                        }

                                        // 2. Salva SpikePattern do grafema no spike_vocab
                                        //    (usa o padrão acumulado pelos frames FFT enviados antes)
                                        state.spike_vocab.insert(
                                            format!("audio:{}", &grafema), apad,
                                        );

                                        // 3. Grafema ↔ cada letra (bidirecional)
                                        //    grafema→letra peso 0.85 (síntese → componente)
                                        //    letra→grafema peso 0.70 (componente → síntese)
                                        for letra in &letras {
                                            if letra.is_empty() { continue; }
                                            {
                                                let viz = state.grafo_associacoes
                                                    .entry(grafema.clone()).or_default();
                                                if !viz.iter().any(|(w, _)| w == letra) {
                                                    viz.push((letra.clone(), 0.85));
                                                }
                                            }
                                            {
                                                let viz_inv = state.grafo_associacoes
                                                    .entry(letra.clone()).or_default();
                                                if !viz_inv.iter().any(|(w, _)| w == &grafema) {
                                                    viz_inv.push((grafema.clone(), 0.70));
                                                }
                                            }
                                        }

                                        // 4. Bigrams sequenciais entre letras (b→a, t→r→a, etc.)
                                        //    peso 0.90 — ordem das letras dentro do grafema
                                        for i in 0..letras.len().saturating_sub(1) {
                                            let l1 = &letras[i];
                                            let l2 = &letras[i + 1];
                                            if l1.is_empty() || l2.is_empty() { continue; }
                                            let viz = state.grafo_associacoes
                                                .entry(l1.clone()).or_default();
                                            if !viz.iter().any(|(w, _)| w == l2) {
                                                viz.push((l2.clone(), 0.90));
                                            }
                                        }

                                        let n_nos = state.grafo_associacoes.len();
                                        log::debug!("[FONETICO] grounding '{}' letras={:?} grafo={} nos audio_ativo={}",
                                            grafema, letras, n_nos,
                                            crate::encoding::spike_codec::is_active(&apad));

                                        let ack = serde_json::json!({
                                            "event": "grounding_ack",
                                            "grafema": grafema,
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
                                let mut state = brain.lock().await;
                                let (dopa, sero, _nor) = state.neurotransmissores;
                                let (step, alerta, emocao) = state.atividade;

                                // Busca valência da palavra mais saliente na mensagem
                                let msg_lower = text.to_lowercase();
                                let valence = msg_lower.split_whitespace()
                                    .filter_map(|t| state.palavra_valencias.get(t).copied())
                                    .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
                                    .unwrap_or(0.0);

                                let emocao_resp = (emocao * 0.6 + valence * 0.4).clamp(-1.0, 1.0);
                                state.ws_atividade = 1.0;

                                let n_passos = state.n_passos_walk;
                                let emocao_bias = state.emocao_bias;
                                state.reply_count = state.reply_count.wrapping_add(1);
                                let diversity_seed = step ^ state.reply_count.wrapping_mul(6364136223846793005);
                                let mut caminho_q: Vec<String> = Vec::new();
                                let mut prefixo_buf: Vec<String> = Vec::new();
                                let reply = gerar_resposta_emergente(
                                    &text, diversity_seed, emocao_resp,
                                    emocao_bias, n_passos,
                                    dopa, sero,
                                    &state.palavra_valencias,
                                    &state.grafo_associacoes,
                                    &state.frases_padrao,
                                    &state.ultimos_prefixos.iter().cloned().collect::<Vec<_>>(),
                                    &conversa_ctx,
                                    &mut caminho_q,
                                    &state.emocao_palavras,
                                    &mut prefixo_buf,
                                    &state.grounding,
                                );
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