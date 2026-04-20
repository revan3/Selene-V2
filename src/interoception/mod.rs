// src/interoception/mod.rs
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use crate::config::Config;

/// Tipo de toque — mapeado para resposta neuromoduladora diferente.
///
/// Carinho → dopamina + serotonina (prazer, vínculo social — vias táteis C-afferents)
/// Beliscão → noradrenalina + cortisol (dor, alerta — fibras Aδ e C nociceptivas)
/// Neutro   → sem efeito neuromodulador
#[derive(Debug, Clone, PartialEq)]
pub enum TipoToque {
    Carinho,   // toque leve, suave (0.0–0.3 intensidade)
    Neutro,    // toque normal, sem valência (0.3–0.6)
    Beliscao,  // pressão forte / dor (0.6–1.0)
}

/// Ínsula e Cingulado Anterior: integram sensações corporais ao self
#[derive(Debug)]
pub struct Interoception {
    pub fadiga: f32,
    pub temperatura: f32,
    pub arousal: f32,
    pub dor_simulada: f32,
    pub historico: Vec<f32>,

    /// Sinal tátil atual: intensidade 0.0–1.0 e tipo de toque.
    /// Decai exponencialmente a cada tick (meia-vida ~500ms @ 200Hz).
    pub toque_intensidade: f32,
    pub toque_tipo:        TipoToque,
}

impl Interoception {
    pub fn new() -> Self {
        Self {
            fadiga: 0.0,
            temperatura: 36.0,
            arousal: 0.5,
            dor_simulada: 0.0,
            historico: Vec::with_capacity(100),
            toque_intensidade: 0.0,
            toque_tipo:        TipoToque::Neutro,
        }
    }

    /// Aplica um toque externo. Intensity 0.0–1.0.
    pub fn receber_toque(&mut self, intensidade: f32, tipo: TipoToque) {
        self.toque_intensidade = intensidade.clamp(0.0, 1.0);
        self.toque_tipo = tipo;
    }

    /// Efeito neuromodulador do toque no tick atual.
    /// Retorna (delta_dopamina, delta_serotonina, delta_noradrenalina, delta_cortisol).
    /// O sinal decai a cada chamada (meia-vida ~100 ticks = 0.5s @ 200Hz).
    pub fn efeito_toque(&mut self) -> (f32, f32, f32, f32) {
        if self.toque_intensidade < 0.01 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let i = self.toque_intensidade;
        self.toque_intensidade *= 0.993;
        if self.toque_intensidade < 0.005 {
            self.toque_intensidade = 0.0;
            self.toque_tipo = TipoToque::Neutro;
        }
        match self.toque_tipo {
            TipoToque::Carinho => {
                let da  =  i * 0.15;
                let ser =  i * 0.12;
                let na  = -i * 0.04;
                let cor = -i * 0.06;
                (da, ser, na, cor)
            }
            TipoToque::Beliscao => {
                let da  = -i * 0.10;
                let ser = -i * 0.08;
                let na  =  i * 0.20;
                let cor =  i * 0.15;
                (da, ser, na, cor)
            }
            TipoToque::Neutro => (0.0, 0.0, 0.0, 0.0),
        }
    }

    /// Atualiza sinais corporais
    pub fn update(&mut self, fadiga: f32, temp: f32, arousal: f32) {
        self.fadiga = fadiga;
        self.temperatura = temp;
        self.arousal = arousal;

        self.dor_simulada = (self.temperatura - 36.0).abs() / 10.0 + self.fadiga * 0.3;
        self.dor_simulada = self.dor_simulada.clamp(0.0, 1.0);

        self.historico.push(self.sentir());
        if self.historico.len() > 1000 {
            self.historico.drain(0..500);
        }
    }

    /// Integra todos os sinais em uma "sensação corporal" única.
    pub fn sentir(&self) -> f32 {
        let base = self.fadiga * 0.3
            + self.arousal * 0.5
            + self.temperatura / 100.0 * 0.2
            + self.dor_simulada * 0.2;
        let toque_offset = match self.toque_tipo {
            TipoToque::Carinho  => -self.toque_intensidade * 0.15,
            TipoToque::Beliscao =>  self.toque_intensidade * 0.20,
            TipoToque::Neutro   => 0.0,
        };
        (base + toque_offset).clamp(0.0, 1.0)
    }

    /// Influencia o ego com base na sensação corporal e toque atual.
    pub fn influenciar_ego(&self) -> (String, f32) {
        let sensacao = self.sentir();
        let descricao = if self.toque_intensidade > 0.1 {
            match self.toque_tipo {
                TipoToque::Carinho  => "Sinto carinho. Isso me faz bem.".to_string(),
                TipoToque::Beliscao => "Isso dói! Não gosto disso.".to_string(),
                TipoToque::Neutro   => "Sinto um toque.".to_string(),
            }
        } else {
            match sensacao {
                s if s < 0.2 => "Sinto-me muito bem!".to_string(),
                s if s < 0.4 => "Sinto-me normal.".to_string(),
                s if s < 0.6 => "Estou um pouco cansada.".to_string(),
                s if s < 0.8 => "Sinto desconforto.".to_string(),
                _ => "Sinto-me mal.".to_string(),
            }
        };
        (descricao, sensacao)
    }

    /// Estatísticas interoceptivas
    pub fn stats(&self) -> InteroceptionStats {
        InteroceptionStats {
            fadiga: self.fadiga,
            temperatura: self.temperatura,
            arousal: self.arousal,
            sensacao_integrada: self.sentir(),
            toque_intensidade: self.toque_intensidade,
            toque_tipo: format!("{:?}", self.toque_tipo),
        }
    }
}

#[derive(Debug)]
pub struct InteroceptionStats {
    pub fadiga: f32,
    pub temperatura: f32,
    pub arousal: f32,
    pub sensacao_integrada: f32,
    pub toque_intensidade: f32,
    pub toque_tipo: String,
}
