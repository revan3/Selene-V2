// src/encoding/spike_codec.rs
//
// Sparse population code for text tokens.
// Each word is encoded as a 512-bit pattern with exactly K=26 active bits (~5% sparsity,
// matching cortical firing rates). Encoding is deterministic: FNV-1a hash seed → xorshift64
// LFSR to scatter K bits without collision.
//
// Similarity metric: Jaccard(A, B) = popcount(A AND B) / popcount(A OR B)
// This gives 1.0 for identical words and ~0.05 for random pairs.

/// Total neuron pool for the word representation layer.
pub const N_NEURONS: usize = 512;

/// Number of active neurons per word (~5% of pool).
pub const K_ACTIVE: usize = 26;

/// 512-bit sparse pattern: [u64; 8]  (8 × 64 = 512 bits).
pub type SpikePattern = [u64; 8];

// ── Encoding ────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash over the UTF-8 bytes of `word`.
#[inline]
fn fnv1a(word: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64  = 0x0000_0100_0000_01b3;
    word.bytes().fold(OFFSET, |h, b| (h ^ b as u64).wrapping_mul(PRIME))
}

/// xorshift64 — steps the PRNG one tick.
#[inline]
fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Encode `word` to a deterministic sparse spike pattern.
///
/// Algorithm:
/// 1. Seed = FNV-1a(word)
/// 2. Iteratively call xorshift64 to get candidate bit positions in [0, N_NEURONS)
/// 3. Accept position if not already set; stop once K_ACTIVE bits are placed.
pub fn encode(word: &str) -> SpikePattern {
    let mut pattern: SpikePattern = [0u64; 8];
    let mut placed = 0usize;
    let mut state = fnv1a(word);
    // Guard against infinite loop on degenerate seeds
    let mut iters = 0u64;

    while placed < K_ACTIVE {
        state = xorshift64(state);
        iters += 1;
        // Map to [0, 512) by taking mod 512 (512 = 2^9, so bitwise AND works)
        let pos = (state & 0x1FF) as usize; // mod 512
        let word_idx = pos >> 6;            // which u64
        let bit_idx  = pos & 63;            // which bit within the u64
        let mask     = 1u64 << bit_idx;
        if pattern[word_idx] & mask == 0 {
            pattern[word_idx] |= mask;
            placed += 1;
        }
        // Safety valve: if the PRNG cycles too long (shouldn't happen with 512 slots and 26 bits)
        if iters > 100_000 { break; }
    }

    pattern
}

// ── Similarity ───────────────────────────────────────────────────────────────

/// Jaccard similarity between two spike patterns.
///
/// Returns 1.0 for identical patterns, ~0.0 for unrelated words.
pub fn similarity(a: &SpikePattern, b: &SpikePattern) -> f32 {
    let mut intersection = 0u32;
    let mut union_count  = 0u32;

    for i in 0..8 {
        intersection += (a[i] & b[i]).count_ones();
        union_count  += (a[i] | b[i]).count_ones();
    }

    if union_count == 0 {
        return 0.0;
    }
    intersection as f32 / union_count as f32
}

/// Count active bits in a pattern.
#[inline]
pub fn popcount(p: &SpikePattern) -> u32 {
    p.iter().map(|w| w.count_ones()).sum()
}

// ── Decoding ─────────────────────────────────────────────────────────────────

/// Nearest-neighbour decode: returns the word from `vocab` with highest Jaccard to `pattern`,
/// if above `threshold`.
///
/// `vocab`: iterator of `(word, spike_pattern)` pairs (borrowed).
pub fn decode<'a>(
    pattern:   &SpikePattern,
    vocab:     impl Iterator<Item = (&'a str, &'a SpikePattern)>,
    threshold: f32,
) -> Option<&'a str> {
    let mut best_word  = None;
    let mut best_score = threshold;

    for (word, sp) in vocab {
        let s = similarity(pattern, sp);
        if s > best_score {
            best_score = s;
            best_word  = Some(word);
        }
    }

    best_word
}

/// Returns up to `n` closest words by Jaccard similarity, sorted descending.
pub fn decode_top_n<'a>(
    pattern:   &SpikePattern,
    vocab:     impl Iterator<Item = (&'a str, &'a SpikePattern)>,
    threshold: f32,
    n:         usize,
) -> Vec<(&'a str, f32)> {
    let mut hits: Vec<(&'a str, f32)> = vocab
        .filter_map(|(word, sp)| {
            let s = similarity(pattern, sp);
            if s >= threshold { Some((word, s)) } else { None }
        })
        .collect();

    hits.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    hits.truncate(n);
    hits
}

// ── Pattern arithmetic ────────────────────────────────────────────────────────

/// OR two patterns together (union — additive superposition).
pub fn superimpose(a: &SpikePattern, b: &SpikePattern) -> SpikePattern {
    let mut out = [0u64; 8];
    for i in 0..8 { out[i] = a[i] | b[i]; }
    out
}

/// AND two patterns (intersection — common features).
pub fn intersect(a: &SpikePattern, b: &SpikePattern) -> SpikePattern {
    let mut out = [0u64; 8];
    for i in 0..8 { out[i] = a[i] & b[i]; }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn encode_is_deterministic() {
        let p1 = encode("selene");
        let p2 = encode("selene");
        assert_eq!(p1, p2, "encoding must be deterministic");
    }

    #[test]
    fn encode_produces_k_active_bits() {
        for word in &["selene", "amor", "consciência", "hello", "rust", "a", "zzz"] {
            let p = encode(word);
            let k = popcount(&p);
            assert_eq!(k, K_ACTIVE as u32,
                "word '{}' has {} bits, expected {}", word, k, K_ACTIVE);
        }
    }

    #[test]
    fn self_similarity_is_one() {
        let p = encode("selene");
        let s = similarity(&p, &p);
        assert!((s - 1.0).abs() < 1e-6,
            "self-similarity should be 1.0, got {}", s);
    }

    #[test]
    fn different_words_low_similarity() {
        let p1 = encode("amor");
        let p2 = encode("computador");
        let s = similarity(&p1, &p2);
        // Expected overlap ≈ K²/N = 26²/512 ≈ 1.3 bits → Jaccard ≈ 2.5%
        // Allow up to 30% to account for hash collisions
        assert!(s < 0.30,
            "unrelated words 'amor'/'computador' have suspiciously high similarity: {}", s);
    }

    #[test]
    fn decode_returns_best_match() {
        let pairs: Vec<(&str, SpikePattern)> = vec![
            ("gato",    encode("gato")),
            ("cachorro", encode("cachorro")),
            ("carro",   encode("carro")),
        ];
        let query = encode("cachorro");
        let result = decode(
            &query,
            pairs.iter().map(|(w, p)| (*w, p)),
            0.5,
        );
        assert_eq!(result, Some("cachorro"));
    }

    #[test]
    fn decode_top_n_ordering() {
        let words = ["rio", "mar", "lago", "oceano", "chuva"];
        let pairs: Vec<(&str, SpikePattern)> = words.iter()
            .map(|&w| (w, encode(w)))
            .collect();

        let query = encode("mar");
        let top = decode_top_n(
            &query,
            pairs.iter().map(|(w, p)| (*w, p)),
            0.0,
            3,
        );
        assert!(!top.is_empty());
        // Best match must be "mar" itself
        assert_eq!(top[0].0, "mar");
        // Scores must be descending
        for i in 1..top.len() {
            assert!(top[i-1].1 >= top[i].1);
        }
    }

    #[test]
    fn superimpose_has_more_bits() {
        let a = encode("fogo");
        let b = encode("agua");
        let sup = superimpose(&a, &b);
        let k = popcount(&sup);
        // Union cannot be less than K_ACTIVE
        assert!(k >= K_ACTIVE as u32);
    }

    #[test]
    fn empty_string_encodes_without_panic() {
        let p = encode("");
        assert_eq!(popcount(&p), K_ACTIVE as u32);
    }
}
