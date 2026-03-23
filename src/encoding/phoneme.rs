// src/encoding/phoneme.rs
//
// PT-BR Grapheme-to-Phoneme (G2P) rules + formant parameter table.
//
// Formant parameters per phoneme:
//   F0  — fundamental (voiced pitch), Hz
//   F1  — first formant (jaw opening), Hz
//   F2  — second formant (tongue front-back), Hz
//   F3  — third formant (lip rounding), Hz
//   energy    — relative RMS amplitude [0,1]
//   dur_ms    — base duration in milliseconds
//   voiced    — true if glottis vibrates
//
// These values are used by voz_selene.py to drive the formant synthesizer.
// The Rust side exports phoneme sequences as JSON so Python can read them
// without re-implementing G2P.

use serde::{Serialize, Deserialize};

// ── Phoneme enum ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phoneme {
    // Vowels
    A, E, I, O, U,
    // Nasal vowels
    AN, EN, IN, ON, UN,
    // Consonants — stops
    P, B, T, D, K, G,
    // Fricatives
    F, V, S, Z, SH, ZH,
    // Affricates
    CH, DJ,
    // Nasals
    M, N, NH,
    // Liquids
    L, LH, R, RR,
    // Semivowels / glides
    W, Y,
    // Silence / pause
    SIL,
}

// ── Formant parameters ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantParams {
    pub phoneme:  String, // human-readable tag
    pub f0:       f32,    // Hz — 0 = unvoiced
    pub f1:       f32,    // Hz
    pub f2:       f32,    // Hz
    pub f3:       f32,    // Hz
    pub energy:   f32,    // [0,1]
    pub dur_ms:   f32,    // base duration ms
    pub voiced:   bool,
}

impl FormantParams {
    fn new(ph: &str, f0: f32, f1: f32, f2: f32, f3: f32, energy: f32, dur_ms: f32, voiced: bool) -> Self {
        Self { phoneme: ph.into(), f0, f1, f2, f3, energy, dur_ms, voiced }
    }

    /// Apply emotional modulation.
    /// dopamine  → pitch lift (+15%)
    /// serotonin → slower rate (dur +20%)
    /// noradrenaline → louder (+10%) + faster (dur -15%)
    pub fn with_emotion(&self, dopamine: f32, serotonin: f32, noradrenaline: f32) -> Self {
        let pitch_mod = 1.0 + (dopamine - 0.5) * 0.30;    // ±15%
        let speed_mod = 1.0 - (serotonin - 0.5) * 0.40    // serotonin slows
                          + (noradrenaline - 0.5) * 0.30;  // NE speeds up
        let amp_mod   = 1.0 + (noradrenaline - 0.5) * 0.20;

        Self {
            phoneme:  self.phoneme.clone(),
            f0:       (self.f0 * pitch_mod).max(0.0),
            f1:       self.f1,
            f2:       self.f2,
            f3:       self.f3,
            energy:   (self.energy * amp_mod).clamp(0.0, 1.0),
            dur_ms:   (self.dur_ms / speed_mod.max(0.3)).clamp(20.0, 400.0),
            voiced:   self.voiced,
        }
    }
}

// ── Formant table (PT-BR female voice, ~200 Hz F0) ───────────────────────────
//
// References: Barbosa (2002), Madureira (2018), and IPA-based averages.

pub fn formant_table(ph: Phoneme) -> FormantParams {
    use Phoneme::*;
    match ph {
        // ── Vowels ─────────────────────────────────────────────────────────
        A  => FormantParams::new("/a/",  200.0,  800.0, 1200.0, 2500.0, 0.90, 80.0, true),
        E  => FormantParams::new("/e/",  200.0,  450.0, 2000.0, 2700.0, 0.85, 75.0, true),
        I  => FormantParams::new("/i/",  200.0,  300.0, 2300.0, 3000.0, 0.80, 65.0, true),
        O  => FormantParams::new("/o/",  200.0,  500.0,  900.0, 2500.0, 0.85, 75.0, true),
        U  => FormantParams::new("/u/",  200.0,  300.0,  600.0, 2400.0, 0.75, 65.0, true),

        // Nasal vowels (lower F1, extra nasal resonance at ~250 Hz added by synthesizer)
        AN => FormantParams::new("/ã/",  200.0,  650.0, 1100.0, 2400.0, 0.80, 90.0, true),
        EN => FormantParams::new("/ẽ/",  200.0,  400.0, 1800.0, 2600.0, 0.75, 85.0, true),
        IN => FormantParams::new("/ĩ/",  200.0,  280.0, 2100.0, 2900.0, 0.70, 80.0, true),
        ON => FormantParams::new("/õ/",  200.0,  450.0,  850.0, 2400.0, 0.75, 85.0, true),
        UN => FormantParams::new("/ũ/",  200.0,  270.0,  550.0, 2350.0, 0.70, 80.0, true),

        // ── Stops ──────────────────────────────────────────────────────────
        P  => FormantParams::new("/p/",    0.0,    0.0,    0.0,    0.0, 0.10, 60.0, false),
        B  => FormantParams::new("/b/",  100.0,    0.0,    0.0,    0.0, 0.15, 65.0, true),
        T  => FormantParams::new("/t/",    0.0,    0.0,    0.0,    0.0, 0.10, 55.0, false),
        D  => FormantParams::new("/d/",  100.0,    0.0,    0.0,    0.0, 0.15, 60.0, true),
        K  => FormantParams::new("/k/",    0.0,    0.0,    0.0,    0.0, 0.12, 65.0, false),
        G  => FormantParams::new("/g/",  100.0,    0.0,    0.0,    0.0, 0.15, 70.0, true),

        // ── Fricatives ─────────────────────────────────────────────────────
        F  => FormantParams::new("/f/",    0.0,    0.0,    0.0, 7000.0, 0.30, 90.0, false),
        V  => FormantParams::new("/v/",  100.0,    0.0,    0.0, 6500.0, 0.35, 85.0, true),
        S  => FormantParams::new("/s/",    0.0,    0.0,    0.0, 8000.0, 0.40, 85.0, false),
        Z  => FormantParams::new("/z/",  100.0,    0.0,    0.0, 7500.0, 0.40, 80.0, true),
        SH => FormantParams::new("/ʃ/",    0.0,    0.0,    0.0, 4500.0, 0.45, 90.0, false),
        ZH => FormantParams::new("/ʒ/",  100.0,    0.0,    0.0, 4000.0, 0.45, 85.0, true),

        // ── Affricates ─────────────────────────────────────────────────────
        CH => FormantParams::new("/tʃ/",   0.0,    0.0,    0.0, 4000.0, 0.35, 95.0, false),
        DJ => FormantParams::new("/dʒ/", 100.0,    0.0,    0.0, 3800.0, 0.38, 95.0, true),

        // ── Nasals ─────────────────────────────────────────────────────────
        M  => FormantParams::new("/m/",  200.0,  250.0,  950.0, 2200.0, 0.50, 80.0, true),
        N  => FormantParams::new("/n/",  200.0,  250.0, 1700.0, 2600.0, 0.50, 75.0, true),
        NH => FormantParams::new("/ɲ/",  200.0,  250.0, 2000.0, 2700.0, 0.50, 80.0, true),

        // ── Liquids ────────────────────────────────────────────────────────
        L  => FormantParams::new("/l/",  200.0,  350.0, 1100.0, 2500.0, 0.60, 70.0, true),
        LH => FormantParams::new("/ʎ/",  200.0,  350.0, 1800.0, 2700.0, 0.55, 75.0, true),
        R  => FormantParams::new("/ɾ/",  200.0,  450.0, 1100.0, 2500.0, 0.55, 50.0, true),
        RR => FormantParams::new("/x/",    0.0,    0.0,    0.0, 3000.0, 0.40, 80.0, false),

        // ── Glides ─────────────────────────────────────────────────────────
        W  => FormantParams::new("/w/",  200.0,  300.0,  700.0, 2200.0, 0.65, 60.0, true),
        Y  => FormantParams::new("/j/",  200.0,  300.0, 2200.0, 2800.0, 0.65, 60.0, true),

        // ── Silence ────────────────────────────────────────────────────────
        SIL => FormantParams::new("sil",   0.0,    0.0,    0.0,    0.0, 0.00, 80.0, false),
    }
}

// ── G2P rules (PT-BR) ────────────────────────────────────────────────────────
//
// Very simplified rule-based G2P. Coverage ~85% for common vocabulary.
// For full accuracy, a trained model (e.g. espeak-ng) should be used.

pub fn word_to_phonemes(word: &str) -> Vec<Phoneme> {
    use Phoneme::*;

    let w = word.to_lowercase();
    let chars: Vec<char> = w.chars().collect();
    let n = chars.len();
    let mut out: Vec<Phoneme> = Vec::with_capacity(n * 2);
    let mut i = 0usize;

    while i < n {
        let c = chars[i];
        let next = if i + 1 < n { Some(chars[i+1]) } else { None };
        let prev = if i > 0 { Some(chars[i-1]) } else { None };

        match c {
            // ── Vowels ────────────────────────────────────────────────────
            'a' => { push_nasal_or_vowel(&chars, i, A, AN, &mut out); i += 1; }
            'e' => { push_nasal_or_vowel(&chars, i, E, EN, &mut out); i += 1; }
            'i' | 'í' | 'ï' => { out.push(I); i += 1; }
            'o' => { push_nasal_or_vowel(&chars, i, O, ON, &mut out); i += 1; }
            'u' | 'ú' | 'ü' => { out.push(U); i += 1; }
            'á' => { out.push(A); i += 1; }
            'â' => { out.push(AN); i += 1; }
            'ã' => { out.push(AN); i += 1; }
            'é' => { out.push(E); i += 1; }
            'ê' => { out.push(EN); i += 1; }
            'ó' => { out.push(O); i += 1; }
            'ô' => { out.push(ON); i += 1; }

            // ── Digraphs first ────────────────────────────────────────────
            'l' if next == Some('h') => { out.push(LH); i += 2; }
            'n' if next == Some('h') => { out.push(NH); i += 2; }
            'c' if next == Some('h') => { out.push(SH); i += 2; }
            'r' if prev == Some('r') || (i == 0) => { out.push(RR); i += 1; }
            'r' if next == Some('r') => { out.push(RR); i += 2; }
            's' if next == Some('s') => { out.push(S);  i += 2; }
            'x' if next == Some('c') => { out.push(S);  i += 2; }
            'q' if next == Some('u') => {
                out.push(K);
                i += 2;
                // 'que'/'qui' → k, 'qua'/'quo' → kw
                if let Some(vow) = chars.get(i) {
                    if *vow == 'a' || *vow == 'o' { out.push(W); }
                }
            }
            'g' if next == Some('u') => {
                // gue/gui → g (silent u); gua/guo → gw
                i += 2;
                if let Some(vow) = chars.get(i) {
                    if *vow == 'a' || *vow == 'o' { out.push(G); out.push(W); }
                    else { out.push(G); }
                } else { out.push(G); }
            }

            // ── Simple consonants ─────────────────────────────────────────
            'p' => { out.push(P); i += 1; }
            'b' | 'v' if is_after_nasal(prev) => { out.push(B); i += 1; }
            'b' => { out.push(B); i += 1; }
            't' => {
                // 'ti' before front vowel → CH in Brazilian
                if matches!(next, Some('i') | Some('e') | Some('é') | Some('ê')) {
                    out.push(CH);
                } else { out.push(T); }
                i += 1;
            }
            'd' => {
                if matches!(next, Some('i') | Some('e') | Some('é') | Some('ê')) {
                    out.push(DJ);
                } else { out.push(D); }
                i += 1;
            }
            'c' if matches!(next, Some('e') | Some('i') | Some('é') | Some('ê') | Some('í'))
                => { out.push(S); i += 1; }
            'c' | 'k' => { out.push(K); i += 1; }
            'g' if matches!(next, Some('e') | Some('i') | Some('é') | Some('ê') | Some('í'))
                => { out.push(ZH); i += 1; }
            'g' => { out.push(G); i += 1; }
            'f' => { out.push(F); i += 1; }
            'v' => { out.push(V); i += 1; }
            's' | 'ß' => {
                // intervocalic 's' → Z
                if is_vowel(prev) && is_vowel(next) { out.push(Z); }
                else { out.push(S); }
                i += 1;
            }
            'z' => { out.push(Z); i += 1; }
            'x' => {
                match next {
                    Some('a') | Some('o') | Some('u') => { out.push(SH); }
                    _ => { out.push(S); }
                }
                i += 1;
            }
            'j' => { out.push(ZH); i += 1; }
            'm' => { out.push(M); i += 1; }
            'n' => { out.push(N); i += 1; }
            'l' => { out.push(L); i += 1; }
            'r' => { out.push(R); i += 1; }
            'w' => { out.push(W); i += 1; }
            'y' => { out.push(Y); i += 1; }
            'ç' => { out.push(S); i += 1; }
            'h' => { /* silent */ i += 1; }

            // ── Punctuation / spaces → silence ────────────────────────────
            ' ' | ',' | '.' | ';' | '!' | '?' | '\n' | '\t' => {
                if out.last() != Some(&SIL) { out.push(SIL); }
                i += 1;
            }

            // ── Unknown → skip ────────────────────────────────────────────
            _ => { i += 1; }
        }
    }

    // Strip leading/trailing silence
    while out.first() == Some(&SIL) { out.remove(0); }
    while out.last()  == Some(&SIL) { out.pop(); }

    out
}

fn is_vowel(c: Option<char>) -> bool {
    matches!(c, Some('a'|'e'|'i'|'o'|'u'|'á'|'é'|'í'|'ó'|'ú'|'â'|'ê'|'ô'|'ã'|'õ'))
}

fn is_nasal_char(c: char) -> bool {
    matches!(c, 'm' | 'n')
}

fn is_after_nasal(prev: Option<char>) -> bool {
    prev.map(is_nasal_char).unwrap_or(false)
}

/// Decide if vowel at position `i` is nasalized by a following nasal consonant.
fn push_nasal_or_vowel(
    chars: &[char],
    i: usize,
    plain: Phoneme,
    nasal: Phoneme,
    out: &mut Vec<Phoneme>,
) {
    let next = chars.get(i + 1).copied();
    let after_next = chars.get(i + 2).copied();
    // Nasalization: vowel followed by m/n at coda (before consonant or end)
    let nasalized = match next {
        Some('m') | Some('n') => {
            match after_next {
                None | Some(' ') | Some('.') | Some(',') => true,
                Some(c) if !is_vowel(Some(c)) => true,
                _ => false,
            }
        }
        _ => false,
    };
    out.push(if nasalized { nasal } else { plain });
}

/// Convert a full sentence to a sequence of FormantParams, applying emotion.
pub fn sentence_to_formants(
    text:         &str,
    dopamine:     f32,
    serotonin:    f32,
    noradrenaline: f32,
) -> Vec<FormantParams> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut result = Vec::new();

    for (wi, word) in words.iter().enumerate() {
        let phonemes = word_to_phonemes(word);
        for ph in phonemes {
            let fp = formant_table(ph).with_emotion(dopamine, serotonin, noradrenaline);
            result.push(fp);
        }
        // Inter-word pause (shorter than sentence boundary)
        if wi + 1 < words.len() {
            result.push(FormantParams::new("sil", 0.0, 0.0, 0.0, 0.0, 0.0, 40.0, false));
        }
    }

    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use Phoneme::*;

    #[test]
    fn selene_has_phonemes() {
        let ph = word_to_phonemes("selene");
        assert!(!ph.is_empty());
        // should contain at least the vowels E, E
        let vowels: Vec<_> = ph.iter().filter(|&&p| matches!(p, E|I|A|O|U)).collect();
        assert!(vowels.len() >= 2, "expected vowels in 'selene', got {:?}", ph);
    }

    #[test]
    fn amor_phonemes() {
        let ph = word_to_phonemes("amor");
        assert!(ph.contains(&A), "'amor' should start with /a/");
        assert!(ph.contains(&M), "'amor' should contain /m/");
        assert!(ph.contains(&R), "'amor' should end with /r/");
    }

    #[test]
    fn emotion_modulates_pitch() {
        let base = formant_table(Phoneme::A);
        let happy = base.with_emotion(0.9, 0.5, 0.5); // high dopamine
        let sad   = base.with_emotion(0.1, 0.5, 0.5); // low dopamine
        assert!(happy.f0 > base.f0, "high dopamine should raise pitch");
        assert!(sad.f0   < base.f0, "low dopamine should lower pitch");
    }

    #[test]
    fn sentence_to_formants_nonempty() {
        let f = sentence_to_formants("eu sou selene", 0.5, 0.5, 0.5);
        assert!(!f.is_empty());
    }

    #[test]
    fn formant_table_covers_all_phonemes() {
        // Ensure no panic for every variant
        use Phoneme::*;
        let all = [A,E,I,O,U,AN,EN,IN,ON,UN,
                   P,B,T,D,K,G,F,V,S,Z,SH,ZH,
                   CH,DJ,M,N,NH,L,LH,R,RR,W,Y,SIL];
        for p in all {
            let _ = formant_table(p);
        }
    }

    #[test]
    fn silence_stripped_from_boundaries() {
        let ph = word_to_phonemes("olá");
        assert_ne!(ph.first(), Some(&Phoneme::SIL), "leading SIL should be stripped");
        assert_ne!(ph.last(),  Some(&Phoneme::SIL), "trailing SIL should be stripped");
    }
}
