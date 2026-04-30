// src/learning/go_nogo.rs
// V3.4 — Filtro Executivo Go/NoGo + ForceInterrupt.
//
// Recebe o Verdict do VoiceArbiter e decide:
//   - Go            → liberar fala (output normal).
//   - NoGo          → silenciar (registrar no log de pensamentos internos).
//   - ForceInterrupt→ interromper output em vôo (Censor detectou erro crítico
//                     ou Voz Criativa encontrou conexão de alta urgência).
//
// Comunicação com gerar_resposta_emergente via AtomicBool — o walk checa o
// flag a cada N palavras e encerra cooperativamente. Sem Mutex no caminho
// quente; ideal para os 4 cores físicos do Ryzen 3500U.

#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::voices::{Verdict, VoteAction};

/// Decisão do filtro executivo.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoNoGoDecision {
    /// Liberar fala normalmente.
    Go,
    /// Silenciar — registrar como pensamento interno.
    NoGo { reason: NoGoReason },
    /// Interromper output em vôo (Censor crítico ou salto criativo urgente).
    ForceInterrupt { kind: InterruptKind },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoGoReason {
    /// Censor venceu com confiança alta — risco detectado.
    CensorRisk,
    /// Confiança total insuficiente — sem o que dizer.
    LowConfidence,
    /// Voz Criativa dominou mas sem ancoragem.
    CreativeUngrounded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterruptKind {
    /// Censor detectou inconsistência crítica enquanto Selene falava.
    CensorAlarm,
    /// Voz Criativa encontrou conexão de altíssima urgência (insight).
    CreativeInsight,
}

/// Estado compartilhado do filtro executivo. Lock-free para o walk em vôo.
pub struct GoNoGoFilter {
    /// Threshold de urgência para disparar ForceInterrupt (0..1). Default 0.7.
    pub urgency_threshold: f32,
    /// Threshold mínimo de confiança para Go. Abaixo disso, NoGo. Default 0.30.
    pub confidence_floor: f32,
    /// Flag atômico — setado quando ForceInterrupt deve disparar. O walk em
    /// gerar_resposta_emergente checa isto a cada N palavras e encerra cooperativamente.
    pub force_interrupt: Arc<AtomicBool>,
    /// Tipo do último ForceInterrupt (codificado: 0=CensorAlarm, 1=CreativeInsight).
    pub last_interrupt_kind: Arc<AtomicU64>,
    /// Contador total de NoGo (telemetria).
    pub nogo_count: Arc<AtomicU64>,
    /// Contador total de ForceInterrupt (telemetria).
    pub interrupt_count: Arc<AtomicU64>,
}

impl GoNoGoFilter {
    pub fn new() -> Self {
        Self {
            urgency_threshold:   0.70,
            confidence_floor:    0.30,
            force_interrupt:     Arc::new(AtomicBool::new(false)),
            last_interrupt_kind: Arc::new(AtomicU64::new(0)),
            nogo_count:          Arc::new(AtomicU64::new(0)),
            interrupt_count:     Arc::new(AtomicU64::new(0)),
        }
    }

    /// Avalia o Verdict do VoiceArbiter e retorna a decisão executiva.
    /// Side effect: pode setar `force_interrupt` se a urgência ultrapassar o threshold
    /// (gerar_resposta_emergente lê o flag e encerra o walk cooperativamente).
    pub fn evaluate(&self, v: &Verdict) -> GoNoGoDecision {
        match v.action {
            // Inhibit com urgência alta = Censor crítico → ForceInterrupt.
            VoteAction::Inhibit if v.urgency >= self.urgency_threshold => {
                self.force_interrupt.store(true, Ordering::Release);
                self.last_interrupt_kind.store(0, Ordering::Release);
                self.interrupt_count.fetch_add(1, Ordering::Relaxed);
                GoNoGoDecision::ForceInterrupt { kind: InterruptKind::CensorAlarm }
            }
            // Inhibit normal → NoGo (silenciar e guardar como pensamento).
            VoteAction::Inhibit => {
                self.nogo_count.fetch_add(1, Ordering::Relaxed);
                GoNoGoDecision::NoGo { reason: NoGoReason::CensorRisk }
            }
            // LeapCreative com urgência altíssima → ForceInterrupt (insight).
            VoteAction::LeapCreative if v.urgency >= self.urgency_threshold + 0.10 => {
                self.force_interrupt.store(true, Ordering::Release);
                self.last_interrupt_kind.store(1, Ordering::Release);
                self.interrupt_count.fetch_add(1, Ordering::Relaxed);
                GoNoGoDecision::ForceInterrupt { kind: InterruptKind::CreativeInsight }
            }
            // LeapCreative sem ancoragem → NoGo.
            VoteAction::LeapCreative if v.dissonance > 0.6 => {
                self.nogo_count.fetch_add(1, Ordering::Relaxed);
                GoNoGoDecision::NoGo { reason: NoGoReason::CreativeUngrounded }
            }
            // Confiança muito baixa → NoGo.
            _ if v.confidence < self.confidence_floor => {
                self.nogo_count.fetch_add(1, Ordering::Relaxed);
                GoNoGoDecision::NoGo { reason: NoGoReason::LowConfidence }
            }
            _ => GoNoGoDecision::Go,
        }
    }

    /// Limpa o flag de ForceInterrupt (chamado pelo handler de chat após
    /// processar uma interrupção e enviar o evento WS ao cliente).
    pub fn clear_interrupt(&self) {
        self.force_interrupt.store(false, Ordering::Release);
    }

    /// True se o walk em vôo deve abortar imediatamente.
    pub fn should_abort(&self) -> bool {
        self.force_interrupt.load(Ordering::Acquire)
    }

    /// Tipo do último ForceInterrupt para o handler decodificar.
    pub fn last_kind(&self) -> InterruptKind {
        match self.last_interrupt_kind.load(Ordering::Acquire) {
            0 => InterruptKind::CensorAlarm,
            _ => InterruptKind::CreativeInsight,
        }
    }
}

impl Default for GoNoGoFilter {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::voices::{Verdict, VoteAction};

    fn vd(action: VoteAction, conf: f32, urg: f32, diss: f32) -> Verdict {
        Verdict { action, confidence: conf, urgency: urg, concept_id: 1, dissonance: diss }
    }

    #[test]
    fn go_em_speak_normal() {
        let f = GoNoGoFilter::new();
        let d = f.evaluate(&vd(VoteAction::Speak, 0.7, 0.4, 0.2));
        assert_eq!(d, GoNoGoDecision::Go);
    }

    #[test]
    fn nogo_em_inhibit_baixo() {
        let f = GoNoGoFilter::new();
        let d = f.evaluate(&vd(VoteAction::Inhibit, 0.5, 0.4, 0.3));
        assert!(matches!(d, GoNoGoDecision::NoGo { .. }));
        assert!(!f.should_abort());
    }

    #[test]
    fn force_interrupt_em_censor_critico() {
        let f = GoNoGoFilter::new();
        let d = f.evaluate(&vd(VoteAction::Inhibit, 0.9, 0.85, 0.1));
        assert_eq!(d, GoNoGoDecision::ForceInterrupt { kind: InterruptKind::CensorAlarm });
        assert!(f.should_abort());
    }

    #[test]
    fn force_interrupt_em_insight_criativo() {
        let f = GoNoGoFilter::new();
        let d = f.evaluate(&vd(VoteAction::LeapCreative, 0.7, 0.85, 0.2));
        assert_eq!(d, GoNoGoDecision::ForceInterrupt { kind: InterruptKind::CreativeInsight });
    }

    #[test]
    fn nogo_em_confianca_baixa() {
        let f = GoNoGoFilter::new();
        let d = f.evaluate(&vd(VoteAction::Speak, 0.10, 0.10, 0.5));
        assert!(matches!(d, GoNoGoDecision::NoGo { reason: NoGoReason::LowConfidence }));
    }
}
