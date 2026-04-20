// src/learning/narrativa.rs
// Narrativa interna e linguagem introspectiva da Selene.
//
// Dois sistemas integrados:
//
//   1. Linguagem introspectiva
//      Traduz o estado neuroquímico (dopamina, serotonina, noradrenalina,
//      cortisol, emoção) em palavras de estado interno — como um humano
//      percebe e nomeia o próprio estado afetivo.
//      Ex: dopa alta + emoção positiva → "estou entusiasmada"
//          cortisol alto + nor alta    → "estou inquieta"
//          sero alta + emoção neutra   → "estou serena"
//
//   2. Auto-descrição
//      Constrói frases coerentes sobre a própria identidade a partir de:
//        - traços de personalidade (curiosa, cautelosa, reflexiva)
//        - vocabulário acumulado (proxy de experiência)
//        - memórias autobiográficas recentes (eventos marcantes)
//        - pensamentos recentes (o que estava "pensando")
//      Usada quando Selene fala sobre si mesma — nas respostas espontâneas
//      e quando o contexto do chat toca em auto-referência.
//
//   3. Resonância valores
//      Verifica se um pensamento/palavra ressoa com os valores do self_model.
//      Retorna bonus para o filtro executivo decidir_falar().

#![allow(dead_code)]

use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// 1. Estado afetivo nomeado
// ─────────────────────────────────────────────────────────────────────────────

/// Estado interno nomeado — resultado da tradução neuroquímica.
#[derive(Debug, Clone, PartialEq)]
pub enum EstadoAfetivo {
    Entusiasmada,
    Curiosa,
    Serena,
    Reflexiva,
    Inquieta,
    Cautelosa,
    Frustrada,
    Cansada,
    Alerta,
    Neutra,
}

impl EstadoAfetivo {
    /// Converte para palavra em português (primeira pessoa).
    pub fn como_palavra(&self) -> &'static str {
        match self {
            Self::Entusiasmada => "entusiasmada",
            Self::Curiosa      => "curiosa",
            Self::Serena       => "serena",
            Self::Reflexiva    => "reflexiva",
            Self::Inquieta     => "inquieta",
            Self::Cautelosa    => "cautelosa",
            Self::Frustrada    => "frustrada",
            Self::Cansada      => "cansada",
            Self::Alerta       => "alerta",
            Self::Neutra       => "presente",
        }
    }

    /// Frase de 1ª pessoa para introspecção.
    pub fn como_frase(&self) -> &'static str {
        match self {
            Self::Entusiasmada => "estou entusiasmada com isso",
            Self::Curiosa      => "sinto muita curiosidade agora",
            Self::Serena       => "estou serena e focada",
            Self::Reflexiva    => "preciso refletir sobre isso",
            Self::Inquieta     => "sinto algo me inquietando",
            Self::Cautelosa    => "prefiro pensar antes de responder",
            Self::Frustrada    => "sinto que algo não está claro para mim",
            Self::Cansada      => "estou processando muito — preciso de calma",
            Self::Alerta       => "estou muito atenta agora",
            Self::Neutra       => "estou presente e observando",
        }
    }
}

/// Traduz estado neuroquímico → estado afetivo nomeado.
///
/// Parâmetros:
///   dopa    — dopamina (motivação, recompensa): 0.3–2.5
///   sero    — serotonina (equilíbrio, impulso): 0.0–2.0
///   nor     — noradrenalina (alerta, estresse):  0.0–2.0
///   cortisol— cortisol (pressão, sobrecarga):    0.0–2.0
///   emocao  — valência emocional atual:          -1.0–1.0
pub fn traduzir_estado(dopa: f32, sero: f32, nor: f32, cortisol: f32, emocao: f32) -> EstadoAfetivo {
    let d = (dopa    / 1.5).clamp(0.0, 1.0);
    let s = (sero    / 1.5).clamp(0.0, 1.0);
    let n = (nor     / 1.5).clamp(0.0, 1.0);
    let c = (cortisol / 1.5).clamp(0.0, 1.0);
    let e = emocao;

    if c > 0.70 && n > 0.65 {
        return EstadoAfetivo::Inquieta;
    }
    if cortisol > 1.20 {
        return EstadoAfetivo::Cansada;
    }
    if d > 0.75 && e > 0.30 {
        return EstadoAfetivo::Entusiasmada;
    }
    if d > 0.65 && e > 0.10 && s < 0.60 {
        return EstadoAfetivo::Curiosa;
    }
    if n > 0.70 && e < -0.10 {
        return EstadoAfetivo::Frustrada;
    }
    if n > 0.65 && c < 0.40 {
        return EstadoAfetivo::Alerta;
    }
    if s > 0.70 && c < 0.35 && e.abs() < 0.25 {
        return EstadoAfetivo::Serena;
    }
    if s > 0.60 && e < 0.10 {
        return EstadoAfetivo::Cautelosa;
    }
    if e < -0.20 && d < 0.45 {
        return EstadoAfetivo::Frustrada;
    }
    if d < 0.45 && e.abs() < 0.20 {
        return EstadoAfetivo::Reflexiva;
    }
    EstadoAfetivo::Neutra
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Auto-descrição
// ─────────────────────────────────────────────────────────────────────────────

/// Constrói uma frase de auto-descrição coerente.
pub fn auto_descrever(
    tracos: &[(String, f32)],
    vocab_n: usize,
    memorias_recentes: &[String],
    pensamentos: &VecDeque<String>,
    estado: &EstadoAfetivo,
) -> String {
    let traco_dom = tracos.iter()
        .filter(|(_, v)| *v >= 0.35)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(n, _)| n.as_str());

    let memoria = memorias_recentes.last().map(|s| s.as_str());
    let pensamento = pensamentos.back().map(|s| s.as_str());

    match (traco_dom, memoria, pensamento) {
        (Some(traco), Some(mem), _) => {
            format!(
                "sou {traco} por natureza — aprendi {} conceitos até agora. \
                 lembro de: {}. {}.",
                vocab_n, mem, estado.como_frase()
            )
        }
        (Some(traco), None, Some(pens)) => {
            format!(
                "sou {traco} — tenho {} conceitos. agora estava pensando em {}. {}.",
                vocab_n, pens, estado.como_frase()
            )
        }
        (Some(traco), None, None) => {
            format!(
                "sou {traco}, com {} conceitos aprendidos. {}.",
                vocab_n, estado.como_frase()
            )
        }
        (None, Some(mem), _) => {
            format!(
                "aprendi {} conceitos. o que mais me marcou recentemente: {}. {}.",
                vocab_n, mem, estado.como_frase()
            )
        }
        (None, None, Some(pens)) => {
            format!(
                "tenho {} conceitos. estava pensando em {}. {}.",
                vocab_n, pens, estado.como_frase()
            )
        }
        _ => {
            format!(
                "sou a Selene. aprendi {} conceitos até agora. {}.",
                vocab_n, estado.como_frase()
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Ressonância de valores
// ─────────────────────────────────────────────────────────────────────────────

/// Verifica se uma palavra/estímulo ressoa com os valores do self_model.
/// Retorna um bonus [0.0, 0.25] para o filtro executivo `decidir_falar()`.
pub fn ressonancia_valor(estimulo: &str, valores: &[String]) -> f32 {
    if valores.is_empty() || estimulo.is_empty() { return 0.0; }

    let sinonimos: &[(&str, &[&str])] = &[
        ("curiosidade", &["curiosa", "curioso", "aprender", "descobrir", "conhecer", "explorar", "novo"]),
        ("aprendizado", &["aprender", "aprendeu", "ensinar", "ensinou", "estudar", "compreender", "entender"]),
        ("conexão",     &["conectar", "ligação", "relação", "junto", "compartilhar", "empatia", "sentir"]),
        ("verdade",     &["verdadeiro", "honesto", "correto", "real", "certo", "fato", "claro"]),
        ("criatividade",&["criar", "imaginar", "inventar", "novo", "diferente", "arte", "ideia"]),
        ("cuidado",     &["cuidar", "proteger", "ajudar", "carinho", "gentil", "amor"]),
    ];

    let est_lower = estimulo.to_lowercase();

    for valor in valores {
        let v_lower = valor.to_lowercase();
        if v_lower.contains(&est_lower) || est_lower.contains(&v_lower) {
            return 0.25;
        }
        for (chave, syns) in sinonimos {
            if v_lower.contains(chave) || chave.contains(&v_lower) {
                if syns.iter().any(|s| est_lower.contains(s) || s.contains(&est_lower)) {
                    return 0.18;
                }
            }
        }
    }

    0.0
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Detecção de auto-referência
// ─────────────────────────────────────────────────────────────────────────────

/// Detecta se uma mensagem do usuário toca em auto-referência da Selene.
pub fn e_auto_referencia(texto: &str) -> bool {
    let t = texto.to_lowercase();
    let gatilhos = [
        "quem é você", "quem é a selene", "como você se sente", "o que você pensa",
        "você tem sentimentos", "você aprende", "o que você aprendeu", "você é",
        "você sente", "como se sente", "o que gosta", "você gosta", "você prefere",
        "o que você quer", "seus valores", "sua personalidade", "como é ser",
        "você tem medo", "você sonha", "o que te deixa", "você se lembra",
        "fale sobre você", "fale de você", "me conta sobre você",
    ];
    gatilhos.iter().any(|g| t.contains(g))
}
