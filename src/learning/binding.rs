// src/learning/binding.rs
// Binding temporal de primitivas de onda para Selene Brain 2.0.
//
// Resolve o "binding problem": como diferentes percepções simultâneas
// (som + luz + estado interno) se tornam um único episódio de experiência?
//
// Solução biológica: oscilações gama (~40Hz) sincronizam regiões cortical.
// Tudo que dispara dentro da mesma janela gamma (~25ms) é "ligado".
//
// Implementação:
//   - Janela principal de 25ms (gamma) — une primitivas sincrônicas
//   - Janela de sílaba de 200-300ms — une bigramas em padrão temporal
//   - Janela de frase de 2-5s — une padrões em episódio
//
// Saídas:
//   - Vec<PrimitivaOnda>  → janela gamma fechada (sinapse multi-modal)
//   - PadraoTemporal      → janela de sílaba fechada (fonema emergente)
//   - Episodio            → janela de frase fechada (memória episódica)
//
// Melhoria: sistema de 3 janelas aninhadas (gamma/sílaba/frase).
// Melhoria: detector de silêncio para forçar fechamento de janela de sílaba.
// Melhoria: valência emocional propagada do estado interno para o padrão.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::storage::ondas::{BigramaFonetico, PadraoTemporal, PrimitivaOnda, TipoOnda};
use crate::storage::tipos::CamadaFonetica;
use crate::storage::episodic::Episodio;

// ─── Constantes de janela ─────────────────────────────────────────────────────

/// Janela gamma: primitivas simultâneas são "ligadas" (ms).
pub const JANELA_GAMMA_MS: f64 = 25.0;

/// Janela de sílaba: duração máxima de um padrão fonético (ms).
pub const JANELA_SILABA_MS: f64 = 300.0;

/// Janela de frase: duração máxima de um episódio (ms).
pub const JANELA_FRASE_MS: f64 = 5_000.0;

/// Energia mínima para considerar primitiva "ativa" (silêncio abaixo disso).
pub const LIMIAR_SILENCIO: f32 = 0.02;

/// Silêncio por mais de N ms fecha a janela de sílaba.
pub const SILENCIO_FECHA_SILABA_MS: f64 = 80.0;

// ─── Janela Gamma (25ms) ─────────────────────────────────────────────────────

/// Grupo de primitivas co-ativas dentro de uma janela gamma.
/// Representa um "instante de experiência" multi-modal.
#[derive(Debug, Clone)]
pub struct JanelaGamma {
    pub primitivas: Vec<PrimitivaOnda>,
    pub t_inicio:   f64,
    pub t_fim:      f64,
    /// Valência emocional do frame (média ponderada das primitivas internas).
    pub valencia:   f32,
}

impl JanelaGamma {
    pub fn nova(t: f64) -> Self {
        Self { primitivas: Vec::new(), t_inicio: t, t_fim: t + JANELA_GAMMA_MS / 1000.0, valencia: 0.0 }
    }

    pub fn aceita(&self, timestamp: f64) -> bool {
        timestamp <= self.t_fim
    }

    pub fn adicionar(&mut self, p: PrimitivaOnda) {
        // Extrai valência de primitivas internas (estado corporal)
        if p.tipo == TipoOnda::Interna {
            // Batimento cardíaco acelerado (>1.5Hz) = ativação = valência sinaliza alerta
            if let Some(f) = p.freq_interna_hz {
                if f > 1.5 { self.valencia += 0.1 * p.amplitude; }
            }
        }
        self.primitivas.push(p);
    }

    pub fn esta_vazia(&self) -> bool {
        self.primitivas.iter().all(|p| p.amplitude < LIMIAR_SILENCIO)
    }
}

// ─── Janela de Sílaba (≤300ms) ───────────────────────────────────────────────

/// Acumula janelas gamma até completar um padrão fonético.
/// Fecha quando: silêncio > 80ms, ou duração > 300ms.
#[derive(Debug, Clone)]
pub struct JanelaSilaba {
    pub janelas:    Vec<JanelaGamma>,
    pub t_inicio:   f64,
    pub ultimo_som: f64,   // timestamp do último frame não-silencioso
}

impl JanelaSilaba {
    pub fn nova(t: f64) -> Self {
        Self { janelas: Vec::new(), t_inicio: t, ultimo_som: t }
    }

    pub fn adicionar_gamma(&mut self, jg: JanelaGamma) {
        if !jg.esta_vazia() {
            self.ultimo_som = jg.t_fim;
        }
        self.janelas.push(jg);
    }

    /// Verifica se a janela deve ser fechada.
    pub fn deve_fechar(&self, agora: f64) -> bool {
        let duracao_ms = (agora - self.t_inicio) * 1000.0;
        let silencio_ms = (agora - self.ultimo_som) * 1000.0;
        duracao_ms >= JANELA_SILABA_MS || silencio_ms >= SILENCIO_FECHA_SILABA_MS
    }

    /// Converte a janela de sílaba em PadraoTemporal.
    pub fn para_padrao(&self) -> Option<PadraoTemporal> {
        let primitivas: Vec<&PrimitivaOnda> = self.janelas.iter()
            .flat_map(|jg| jg.primitivas.iter())
            .filter(|p| p.amplitude >= LIMIAR_SILENCIO)
            .collect();

        if primitivas.is_empty() { return None; }

        // Valência média do episódio
        let valencia = if self.janelas.is_empty() { 0.0 } else {
            self.janelas.iter().map(|j| j.valencia).sum::<f32>() / self.janelas.len() as f32
        };

        let mut padrao = PadraoTemporal::de_primitivas(&primitivas, CamadaFonetica::Silaba);
        padrao.valencia = valencia;
        Some(padrao)
    }

    /// Extrai bigramas fonéticos ordenados da sequência de primitivas sonoras.
    pub fn extrair_bigramas(&self) -> Vec<BigramaFonetico> {
        let sonoras: Vec<&PrimitivaOnda> = self.janelas.iter()
            .flat_map(|jg| jg.primitivas.iter())
            .filter(|p| p.tipo == TipoOnda::Sonora && p.amplitude >= LIMIAR_SILENCIO)
            .collect();

        sonoras.windows(2).enumerate().map(|(i, par)| {
            BigramaFonetico::novo(par[0].hash.clone(), par[1].hash.clone(), i as u8)
        }).collect()
    }
}

// ─── Janela de Frase (≤5s) ───────────────────────────────────────────────────

/// Acumula padrões de sílaba até formar um episódio completo.
/// Fecha quando: pausa > 500ms, ou duração > 5s.
#[derive(Debug, Clone)]
pub struct JanelaFrase {
    pub padroes:    Vec<PadraoTemporal>,
    pub t_inicio:   f64,
    pub ultimo_padrao: f64,
    pub valencia_acumulada: f32,
}

impl JanelaFrase {
    pub fn nova(t: f64) -> Self {
        Self { padroes: Vec::new(), t_inicio: t, ultimo_padrao: t, valencia_acumulada: 0.0 }
    }

    pub fn adicionar_padrao(&mut self, p: PadraoTemporal) {
        self.ultimo_padrao = agora_f64();
        self.valencia_acumulada += p.valencia;
        self.padroes.push(p);
    }

    pub fn deve_fechar(&self, agora: f64) -> bool {
        let duracao_ms = (agora - self.t_inicio) * 1000.0;
        let pausa_ms   = (agora - self.ultimo_padrao) * 1000.0;
        duracao_ms >= JANELA_FRASE_MS || pausa_ms >= 500.0
    }

    /// Converte a janela de frase em Episodio persistível.
    pub fn para_episodio(&self, contexto: Option<String>) -> Option<Episodio> {
        if self.padroes.is_empty() { return None; }

        let spike_hashes: Vec<String> = self.padroes.iter()
            .map(|p| p.hash.clone())
            .collect();

        let valencia = if self.padroes.is_empty() { 0.0 } else {
            self.valencia_acumulada / self.padroes.len() as f32
        };

        let descricao = format!(
            "episodio_{}_padroes_v{:.2}",
            self.padroes.len(),
            valencia
        );

        Some(Episodio::novo(descricao, spike_hashes, valencia, contexto))
    }
}

// ─── BindingBuffer ────────────────────────────────────────────────────────────

/// Buffer central de binding que gerencia as 3 janelas aninhadas.
///
/// Uso:
/// ```ignore
/// let mut buf = BindingBuffer::novo();
/// // A cada frame de áudio/vídeo:
/// buf.alimentar(primitiva, agora);
/// // Coleta resultados:
/// for bigrama in buf.drena_bigramas() { ... }
/// for padrao  in buf.drena_padroes()  { ... }
/// for episodio in buf.drena_episodios() { ... }
/// ```
#[derive(Debug)]
pub struct BindingBuffer {
    /// Janela gamma atual (25ms).
    gamma_atual:    Option<JanelaGamma>,
    /// Janela de sílaba atual (≤300ms).
    silaba_atual:   Option<JanelaSilaba>,
    /// Janela de frase atual (≤5s).
    frase_atual:    Option<JanelaFrase>,
    /// Bigramas prontos para serem persistidos.
    bigramas_prontos: VecDeque<BigramaFonetico>,
    /// Padrões prontos para serem persistidos.
    padroes_prontos: VecDeque<PadraoTemporal>,
    /// Episódios prontos para serem persistidos.
    episodios_prontos: VecDeque<Episodio>,
}

impl BindingBuffer {
    pub fn novo() -> Self {
        Self {
            gamma_atual: None,
            silaba_atual: None,
            frase_atual: None,
            bigramas_prontos: VecDeque::new(),
            padroes_prontos: VecDeque::new(),
            episodios_prontos: VecDeque::new(),
        }
    }

    /// Alimenta o buffer com uma nova primitiva de onda.
    /// Deve ser chamado para cada frame de áudio (~25ms) e visual.
    pub fn alimentar(&mut self, primitiva: PrimitivaOnda) {
        let t = primitiva.timestamp;

        // ── Nível 1: Gamma ────────────────────────────────────────────────
        let fechar_gamma = self.gamma_atual.as_ref()
            .map(|j| !j.aceita(t))
            .unwrap_or(false);

        if fechar_gamma {
            self.fechar_gamma(t);
        }

        if self.gamma_atual.is_none() {
            self.gamma_atual = Some(JanelaGamma::nova(t));
        }
        self.gamma_atual.as_mut().unwrap().adicionar(primitiva);

        // ── Nível 2: Sílaba ───────────────────────────────────────────────
        let fechar_silaba = self.silaba_atual.as_ref()
            .map(|j| j.deve_fechar(t))
            .unwrap_or(false);

        if fechar_silaba {
            self.fechar_silaba(t);
        }

        // ── Nível 3: Frase ────────────────────────────────────────────────
        let fechar_frase = self.frase_atual.as_ref()
            .map(|j| j.deve_fechar(t))
            .unwrap_or(false);

        if fechar_frase {
            self.fechar_frase(None);
        }
    }

    /// Força o fechamento do frame atual (ex: fim de fala detectado).
    pub fn forcar_fechamento(&mut self, contexto: Option<String>) {
        let t = agora_f64();
        self.fechar_gamma(t);
        self.fechar_silaba(t);
        self.fechar_frase(contexto);
    }

    fn fechar_gamma(&mut self, t: f64) {
        if let Some(jg) = self.gamma_atual.take() {
            if self.silaba_atual.is_none() {
                self.silaba_atual = Some(JanelaSilaba::nova(t));
            }
            self.silaba_atual.as_mut().unwrap().adicionar_gamma(jg);
        }
    }

    fn fechar_silaba(&mut self, t: f64) {
        if let Some(js) = self.silaba_atual.take() {
            // Extrai bigramas
            for bg in js.extrair_bigramas() {
                self.bigramas_prontos.push_back(bg);
            }
            // Extrai padrão temporal
            if let Some(padrao) = js.para_padrao() {
                if self.frase_atual.is_none() {
                    self.frase_atual = Some(JanelaFrase::nova(t));
                }
                self.frase_atual.as_mut().unwrap().adicionar_padrao(padrao.clone());
                self.padroes_prontos.push_back(padrao);
            }
        }
    }

    fn fechar_frase(&mut self, contexto: Option<String>) {
        if let Some(jf) = self.frase_atual.take() {
            if let Some(episodio) = jf.para_episodio(contexto) {
                self.episodios_prontos.push_back(episodio);
            }
        }
    }

    /// Drena todos os bigramas prontos.
    pub fn drena_bigramas(&mut self) -> Vec<BigramaFonetico> {
        self.bigramas_prontos.drain(..).collect()
    }

    /// Drena todos os padrões prontos.
    pub fn drena_padroes(&mut self) -> Vec<PadraoTemporal> {
        self.padroes_prontos.drain(..).collect()
    }

    /// Drena todos os episódios prontos.
    pub fn drena_episodios(&mut self) -> Vec<Episodio> {
        self.episodios_prontos.drain(..).collect()
    }

    /// Retorna estatísticas do estado atual do buffer.
    pub fn stats(&self) -> BindingStats {
        BindingStats {
            gamma_ativa:     self.gamma_atual.is_some(),
            silaba_ativa:    self.silaba_atual.is_some(),
            frase_ativa:     self.frase_atual.is_some(),
            bigramas_fila:   self.bigramas_prontos.len(),
            padroes_fila:    self.padroes_prontos.len(),
            episodios_fila:  self.episodios_prontos.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BindingStats {
    pub gamma_ativa:    bool,
    pub silaba_ativa:   bool,
    pub frase_ativa:    bool,
    pub bigramas_fila:  usize,
    pub padroes_fila:   usize,
    pub episodios_fila: usize,
}

// ─── Auxiliar ─────────────────────────────────────────────────────────────────

fn agora_f64() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64()
}

// ─── Testes ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn primitiva_teste(t: f64, amplitude: f32) -> PrimitivaOnda {
        PrimitivaOnda::sonora(
            Some(170.0), Some(800.0), Some(1200.0), None,
            0.0, 0.0, 0.0, crate::storage::ondas::TipoOnset::Vogal,
            amplitude, 0.1,
            25, t,
        )
    }

    #[test]
    fn buffer_inicia_vazio() {
        let mut buf = BindingBuffer::novo();
        assert!(buf.drena_bigramas().is_empty());
        assert!(buf.drena_padroes().is_empty());
        assert!(buf.drena_episodios().is_empty());
    }

    #[test]
    fn janela_gamma_aceita_dentro_do_intervalo() {
        let jg = JanelaGamma::nova(0.0);
        assert!(jg.aceita(0.020));   // 20ms < 25ms ✓
        assert!(!jg.aceita(0.030)); // 30ms > 25ms ✗
    }

    #[test]
    fn forcar_fechamento_produz_padrao() {
        let mut buf = BindingBuffer::novo();
        // Alimenta 3 primitivas sequenciais (simulando /a/ /m/ /a/)
        for i in 0..3usize {
            buf.alimentar(primitiva_teste(i as f64 * 0.010, 0.8));
        }
        buf.forcar_fechamento(None);
        // Deve ter gerado pelo menos um padrão
        assert!(!buf.drena_padroes().is_empty() || !buf.drena_bigramas().is_empty());
    }
}
