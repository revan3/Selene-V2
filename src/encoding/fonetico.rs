// src/encoding/fonetico.rs
// Codificador fonético hierárquico para Selene Brain 2.0.
//
// Hierarquia: Fonema → Sílaba → Palavra → Significado → Contexto
//
// Cada nível usa o spike_codec::encode() existente (FNV-1a → xorshift64 LFSR)
// para gerar um SpikePattern ([u64;8]), depois aplica hash_pattern() para obter
// o SpikeHash (string hex 16 dígitos) usado como chave no armazenamento.
#![allow(dead_code)]

use crate::encoding::phoneme::Phoneme;
use crate::encoding::spike_codec;
use crate::storage::tipos::{hash_pattern, SpikeHash};

// ─── Nível Fonema ─────────────────────────────────────────────────────────────

/// Codifica um Phoneme para SpikeHash.
/// Usa o nome debug em minúsculas como token (ex: Phoneme::A → "a").
pub fn encode_fonema(phoneme: &Phoneme) -> SpikeHash {
    let tag = format!("ph:{}", format!("{:?}", phoneme).to_lowercase());
    let pattern = spike_codec::encode(&tag);
    hash_pattern(&pattern)
}

// ─── Nível Sílaba ─────────────────────────────────────────────────────────────

/// Codifica uma sílaba para SpikeHash.
/// Prefixa com "sil:" para separar o espaço de hashes das palavras completas.
pub fn encode_silaba(silaba: &str) -> SpikeHash {
    let token = format!("sil:{}", silaba.to_lowercase());
    let pattern = spike_codec::encode(&token);
    hash_pattern(&pattern)
}

// ─── Nível Palavra ────────────────────────────────────────────────────────────

/// Codifica uma palavra para SpikeHash usando o spike_codec diretamente.
pub fn encode_palavra(palavra: &str) -> SpikeHash {
    let pattern = spike_codec::encode(&palavra.to_lowercase());
    hash_pattern(&pattern)
}

// ─── Nível Texto ──────────────────────────────────────────────────────────────

/// Codifica cada token de um texto para um SpikeHash separado.
/// Tokens são separados por espaços e pontuação comum.
pub fn encode_texto(texto: &str) -> Vec<SpikeHash> {
    texto
        .split(|c: char| c.is_whitespace() || matches!(c, '.' | ',' | '!' | '?' | ';' | ':'))
        .filter(|t| !t.is_empty())
        .map(encode_palavra)
        .collect()
}

// ─── Batch: texto → (token, hash) ────────────────────────────────────────────

/// Retorna pares (palavra_original, SpikeHash) para cada token do texto.
/// Útil para popular a tabela `fonemas` no spike_store.
pub fn encode_texto_com_labels(texto: &str) -> Vec<(String, SpikeHash)> {
    texto
        .split(|c: char| c.is_whitespace() || matches!(c, '.' | ',' | '!' | '?' | ';' | ':'))
        .filter(|t| !t.is_empty())
        .map(|t| {
            let lower = t.to_lowercase();
            let hash = encode_palavra(&lower);
            (lower, hash)
        })
        .collect()
}

// ─── Testes unitários ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesma_palavra_mesmo_hash() {
        assert_eq!(encode_palavra("amor"), encode_palavra("amor"));
    }

    #[test]
    fn palavras_diferentes_hashes_diferentes() {
        assert_ne!(encode_palavra("amor"), encode_palavra("medo"));
    }

    #[test]
    fn fonema_e_palavra_hashes_diferentes() {
        // "a" como fonema (prefixo ph:) ≠ "a" como palavra
        assert_ne!(encode_fonema(&Phoneme::A), encode_palavra("a"));
    }

    #[test]
    fn silaba_e_palavra_hashes_diferentes() {
        assert_ne!(encode_silaba("ma"), encode_palavra("ma"));
    }

    #[test]
    fn encode_texto_retorna_um_hash_por_token() {
        let hashes = encode_texto("olá mundo cruel");
        assert_eq!(hashes.len(), 3);
    }
}
