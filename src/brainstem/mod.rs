// src/brainstem/mod.rs
//
// CORREÇÕES (E0609 + E0308):
//
// Erro original:
//   self.modulate(&input.data)  →  AudioSignal não tem campo .data
//   Os campos disponíveis são: bandas, onset, energia, pitch_dominante
//
// Causa raiz: o canal rx_audio envia Vec<f32> (ver sensors/audio.rs).
// O main.rs chama: brainstem.modulate(&raw_cochlea)  onde raw_cochlea: Vec<f32>
// Portanto modulate() deve aceitar &[f32] diretamente — sem .data.
//
// CORREÇÃO E0308 no main.rs linha 272:
//   unwrap_or_else(|_| vec![0.0f32; n_neurons]) já retorna Vec<f32> → correto.
//   O erro E0308 desaparece quando este mod.rs aceita &[f32] corretamente.

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

/// Estatísticas do Brainstem para o dashboard WebSocket.
/// `alertness` é lido por bridge.rs e main.rs.
pub struct BrainstemStats {
    pub alertness: f32,       // 0.0 (coma) .. 1.0 (alerta máximo)
    pub arousal: f32,         // nível de ativação geral
    pub adenosina: f32,       // pressão homeostática de sono
    pub reticular_rate: f32,  // taxa de disparo da formação reticular
}

/// Brainstem — tronco cerebral.
///
/// Funções modeladas:
///  1. `update(adenosina, dt)` — atualiza pressão de sono e arousal
///  2. `modulate(sinal)` — filtra sinal auditivo/visceral pela formação reticular
///  3. `stats()` — retorna métricas para o dashboard
pub struct Brainstem {
    alertness: f32,
    arousal: f32,
    adenosina: f32,
    reticular_rate: f32,
}

impl Brainstem {
    pub fn new() -> Self {
        Self {
            alertness: 0.8,
            arousal: 1.0,
            adenosina: 0.0,
            reticular_rate: 0.5,
        }
    }

    /// Atualiza a dinâmica homeostática do tronco cerebral.
    ///
    /// `adenosina`: pressão de sono acumulada (0.0 = acordado, 1.0 = exausto)
    /// `dt`: passo de tempo em segundos
    pub fn update(&mut self, adenosina: f32, dt: f32) {
        self.adenosina = adenosina.clamp(0.0, 1.0);

        // Alertness cai com adenosina acumulada (pressão de sono)
        let target_alertness = (1.0 - self.adenosina * 0.8).clamp(0.1, 1.0);
        self.alertness += (target_alertness - self.alertness) * dt * 0.5;
        self.alertness = self.alertness.clamp(0.0, 1.0);

        // Arousal acompanha alertness com inércia
        self.arousal += (self.alertness - self.arousal) * dt * 0.3;
        self.arousal = self.arousal.clamp(0.0, 1.5);

        // Taxa da formação reticular ascendente (ARAS)
        self.reticular_rate = self.alertness * 0.6 + self.arousal * 0.4;
    }

    /// Modula sinal sensorial (auditivo ou visceral) pelo filtro do tronco cerebral.
    ///
    /// CORREÇÃO E0609: aceita &[f32] diretamente — o canal rx_audio envia Vec<f32>,
    /// não AudioSignal. O campo `.data` que causava o erro não existe.
    ///
    /// Comportamento biológico:
    ///   - Alta adenosina (cansaço): sinal atenuado (o cérebro ignora inputs)
    ///   - Alto alertness: sinal amplificado (maior sensibilidade sensorial)
    pub fn modulate(&self, sinal: &[f32]) -> Vec<f32> {
        // Ganho da formação reticular: filtra o sinal pelo estado de alerta
        let ganho = self.reticular_rate.clamp(0.1, 1.2);

        sinal.iter()
            .map(|&v| (v * ganho).clamp(0.0, 1.0))
            .collect()
    }

    /// Retorna estatísticas para bridge.rs e main.rs.
    pub fn stats(&self) -> BrainstemStats {
        BrainstemStats {
            alertness: self.alertness,
            arousal: self.arousal,
            adenosina: self.adenosina,
            reticular_rate: self.reticular_rate,
        }
    }
}

impl Default for Brainstem {
    fn default() -> Self {
        Self::new()
    }
}