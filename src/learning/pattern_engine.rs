// src/learning/pattern_engine.rs
// Motor de reconhecimento e predição de padrões da Selene.
//
// ARQUITETURA — 3 camadas inspiradas no cérebro humano:
//
//  Camada 1 — EPISÓDICA (Hipocampo)
//    Grava toda interação significativa como episódio esparso com contexto.
//    Alta fidelidade, curta duração. Máx 1000 episódios.
//
//  Camada 2 — EXTRAÇÃO (Neocórtex / REM)
//    Durante o sono/replay, varre episódios recentes e extrai sub-padrões
//    comuns. Se N episódios compartilham contexto X → resultado Y, promove
//    para padrão candidato.
//
//  Camada 3 — CONSOLIDAÇÃO (Schema)
//    Padrão candidato que se repete com consistência alta → vira fato
//    consolidado. Passa a ser usado para PREDIÇÃO: dado contexto X, o
//    que vem a seguir? Erro de predição → ajusta peso do padrão.
//
// FONTES DE EPISÓDIO:
//   - Chat direto (mensagem + resposta + feedback)
//   - Aprendizado semântico (conceito + valência + contexto)
//   - Sensorial (visual, auditivo — quando modo ambiente ativo)
//   - Interno (pensamento espontâneo + estado emocional)

#![allow(dead_code)]

use std::collections::HashMap;
use uuid::Uuid;

// ── Constantes ────────────────────────────────────────────────────────────────

/// Máximo de episódios na memória episódica (janela deslizante).
const MAX_EPISODIOS: usize = 1000;

/// Mínimo de episódios com mesmo padrão para promover a candidato.
const MIN_EPISODIOS_CANDIDATO: usize = 3;

/// Mínimo de confirmações de candidato para consolidar como fato.
const MIN_CONFIRMACOES_CONSOLIDAR: usize = 5;

/// Consistência mínima (proporção acertos/total) para consolidação.
const CONSISTENCIA_MIN: f32 = 0.70;

/// Decaimento de força por tick sem uso.
const DECAIMENTO_PADRAO: f32 = 0.0002;

/// Janela de contexto: quantas palavras/tokens do contexto são usadas
/// para identificar o padrão.
const JANELA_CONTEXTO: usize = 8;

// ── Tipos de fonte de episódio ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum FonteEpisodio {
    Chat,           // Interação direta de texto
    Aprendizado,    // learn/treino semântico
    Visual,         // Frame de câmera processado
    Auditivo,       // Entrada de áudio
    Pensamento,     // Pensamento espontâneo
    Ambiente,       // Modo ambiente — observação passiva
    Template,       // Uso de template cognitivo
    Sono,           // Gerado durante REM
}

impl FonteEpisodio {
    pub fn como_str(&self) -> &'static str {
        match self {
            FonteEpisodio::Chat        => "chat",
            FonteEpisodio::Aprendizado => "aprendizado",
            FonteEpisodio::Visual      => "visual",
            FonteEpisodio::Auditivo    => "auditivo",
            FonteEpisodio::Pensamento  => "pensamento",
            FonteEpisodio::Ambiente    => "ambiente",
            FonteEpisodio::Template    => "template",
            FonteEpisodio::Sono        => "sono",
        }
    }
}

// ── Episódio ──────────────────────────────────────────────────────────────────

/// Uma experiência singular gravada com alta fidelidade.
/// Corresponde ao hipocampo: contexto rico, memória esparsa.
#[derive(Debug, Clone)]
pub struct Episodio {
    pub id:          Uuid,
    pub timestamp_s: f64,
    pub fonte:       FonteEpisodio,

    /// Tokens de contexto que estavam ativos no momento.
    /// Ex: últimas palavras da conversa, conceitos neuralmente ativos.
    pub contexto:    Vec<String>,

    /// O que aconteceu / o que foi dito / o que foi percebido.
    pub acao:        String,

    /// Resultado observado (resposta gerada, reação, mudança de estado).
    pub resultado:   Option<String>,

    /// Valência emocional no momento (-1.0 = aversivo, +1.0 = positivo).
    pub valence:     f32,

    /// Estado neuroquímico no momento [DA, 5HT, NA, ACh, cortisol].
    pub neuro:       [f32; 5],

    /// Já foi processado pelo extrator de padrões?
    pub processado:  bool,
}

impl Episodio {
    pub fn novo(
        timestamp_s: f64,
        fonte: FonteEpisodio,
        contexto: Vec<String>,
        acao: String,
        resultado: Option<String>,
        valence: f32,
        neuro: [f32; 5],
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp_s,
            fonte,
            contexto,
            acao,
            resultado,
            valence,
            neuro,
            processado: false,
        }
    }

    /// Chave de padrão: os primeiros JANELA_CONTEXTO tokens do contexto.
    /// Usada para agrupar episódios similares durante a extração.
    pub fn chave_contexto(&self) -> Vec<String> {
        self.contexto.iter()
            .filter(|w| w.len() >= 2)
            .take(JANELA_CONTEXTO)
            .cloned()
            .collect()
    }
}

// ── Padrão Candidato ──────────────────────────────────────────────────────────

/// Padrão identificado mas ainda não consolidado.
/// Nasce quando MIN_EPISODIOS_CANDIDATO episódios compartilham
/// contexto similar → resultado similar.
#[derive(Debug, Clone)]
pub struct PadraoCandidato {
    pub id:             Uuid,
    /// Tokens de contexto que caracterizam este padrão.
    pub gatilho:        Vec<String>,
    /// O que tende a acontecer dado este contexto.
    pub resultado_esperado: String,
    /// Episódios que confirmam este padrão.
    pub episodios_confirmados: Vec<Uuid>,
    /// Episódios que contradizem este padrão.
    pub episodios_contradizem: Vec<Uuid>,
    pub fonte:          FonteEpisodio,
    pub valence_media:  f32,
    pub criado_em:      f64,
    pub ultimo_teste_s: f64,
}

impl PadraoCandidato {
    pub fn consistencia(&self) -> f32 {
        let total = self.episodios_confirmados.len() + self.episodios_contradizem.len();
        if total == 0 { return 0.0; }
        self.episodios_confirmados.len() as f32 / total as f32
    }

    pub fn pronto_para_consolidar(&self) -> bool {
        self.episodios_confirmados.len() >= MIN_CONFIRMACOES_CONSOLIDAR
            && self.consistencia() >= CONSISTENCIA_MIN
    }
}

// ── Padrão Consolidado ────────────────────────────────────────────────────────

/// Fato consolidado: padrão com alta consistência que a Selene
/// usa ativamente para PREDIÇÃO e antecipação.
#[derive(Debug, Clone)]
pub struct PadraoConsolidado {
    pub id:              Uuid,
    pub gatilho:         Vec<String>,
    pub predicao:        String,
    pub forca:           f32,       // 0.0–1.0, decai sem uso
    pub n_acertos:       u32,
    pub n_erros:         u32,
    pub fonte:           FonteEpisodio,
    pub valence:         f32,
    pub consolidado_em:  f64,
    pub ultimo_uso_s:    f64,
    /// Filho de qual candidato?
    pub candidato_origem: Uuid,
}

impl PadraoConsolidado {
    pub fn precisao(&self) -> f32 {
        let total = self.n_acertos + self.n_erros;
        if total == 0 { return 0.5; }
        self.n_acertos as f32 / total as f32
    }

    /// Registra resultado de uma predição.
    /// `acertou`: o que a Selene predisse aconteceu?
    /// Dopamina-like: acerto reforça, erro enfraquece.
    pub fn registrar_predicao(&mut self, acertou: bool, t_s: f64) {
        if acertou {
            self.n_acertos += 1;
            self.forca = (self.forca + 0.05).min(1.0);
        } else {
            self.n_erros += 1;
            self.forca = (self.forca - 0.03).max(0.0);
        }
        self.ultimo_uso_s = t_s;
    }

    pub fn decair(&mut self, t_s: f64) {
        if self.ultimo_uso_s <= 0.0 { return; }
        let dt = (t_s - self.ultimo_uso_s).max(0.0) as f32;
        self.forca = (self.forca - DECAIMENTO_PADRAO * dt).max(0.0);
    }
}

// ── PatternEngine ─────────────────────────────────────────────────────────────

pub struct PatternEngine {
    // Camada 1 — Episódica
    pub episodios: std::collections::VecDeque<Episodio>,

    // Camada 2 — Candidatos
    pub candidatos: HashMap<Uuid, PadraoCandidato>,

    // Camada 3 — Consolidados
    pub consolidados: HashMap<Uuid, PadraoConsolidado>,

    /// Índice: hash de gatilho → IDs de consolidados matching.
    indice_gatilho: HashMap<String, Vec<Uuid>>,

    /// Estatísticas de uso.
    pub total_episodios_gravados: u64,
    pub total_padroes_extraidos:  u64,
    pub total_consolidados:       u64,
    pub total_predicoes:          u64,
    pub total_acertos:            u64,
}

impl PatternEngine {
    pub fn novo() -> Self {
        Self {
            episodios:                std::collections::VecDeque::with_capacity(MAX_EPISODIOS),
            candidatos:               HashMap::new(),
            consolidados:             HashMap::new(),
            indice_gatilho:           HashMap::new(),
            total_episodios_gravados: 0,
            total_padroes_extraidos:  0,
            total_consolidados:       0,
            total_predicoes:          0,
            total_acertos:            0,
        }
    }

    // ── Camada 1: Gravar Episódio ─────────────────────────────────────────────

    /// Grava uma nova experiência na memória episódica.
    /// Chamado por qualquer fonte: chat, learn, visual, ambiente, etc.
    pub fn gravar(
        &mut self,
        timestamp_s:  f64,
        fonte:        FonteEpisodio,
        contexto:     Vec<String>,
        acao:         String,
        resultado:    Option<String>,
        valence:      f32,
        neuro:        [f32; 5],
    ) {
        // Janela deslizante: remove o mais antigo se cheio
        if self.episodios.len() >= MAX_EPISODIOS {
            self.episodios.pop_front();
        }
        let ep = Episodio::novo(timestamp_s, fonte, contexto, acao, resultado, valence, neuro);
        self.episodios.push_back(ep);
        self.total_episodios_gravados += 1;
    }

    // ── Camada 2: Extração de Padrões (chamado no REM/sono) ───────────────────

    /// Varre episódios não processados e extrai padrões candidatos.
    /// Deve ser chamado durante a fase N3 do sono.
    /// Retorna o número de novos candidatos criados.
    pub fn extrair_padroes(&mut self, t_atual_s: f64) -> usize {
        // Coleta episódios não processados com resultado
        let nao_processados: Vec<usize> = self.episodios.iter()
            .enumerate()
            .filter(|(_, e)| !e.processado && e.resultado.is_some())
            .map(|(i, _)| i)
            .collect();

        if nao_processados.is_empty() { return 0; }

        // Agrupa por chave de contexto similar
        let mut grupos: HashMap<String, Vec<usize>> = HashMap::new();
        for &i in &nao_processados {
            let ep = &self.episodios[i];
            let chave = ep.chave_contexto().join("|");
            if !chave.is_empty() {
                grupos.entry(chave).or_default().push(i);
            }
        }

        let mut novos = 0usize;

        for (chave, indices) in &grupos {
            if indices.len() < MIN_EPISODIOS_CANDIDATO { continue; }

            // Resultado mais frequente neste grupo
            let mut freq_resultado: HashMap<String, (usize, f32)> = HashMap::new();
            for &i in indices {
                let ep = &self.episodios[i];
                if let Some(res) = &ep.resultado {
                    let entry = freq_resultado.entry(res.clone()).or_insert((0, 0.0));
                    entry.0 += 1;
                    entry.1 += ep.valence;
                }
            }

            let resultado_dominante = freq_resultado.iter()
                .max_by_key(|(_, (n, _))| *n)
                .map(|(r, (n, v))| (r.clone(), *n, *v / *n as f32));

            let (resultado, n_confirm, valence_media) = match resultado_dominante {
                Some(x) if x.1 >= MIN_EPISODIOS_CANDIDATO => x,
                _ => continue,
            };

            // Verifica se já existe candidato com este gatilho
            let gatilho: Vec<String> = chave.split('|').map(|s| s.to_string()).collect();
            let ja_existe = self.candidatos.values()
                .any(|c| c.gatilho == gatilho && c.resultado_esperado == resultado);
            if ja_existe { continue; }

            // Determina fonte dominante do grupo
            let fonte = self.episodios[indices[0]].fonte.clone();

            let mut candidato = PadraoCandidato {
                id:                   Uuid::new_v4(),
                gatilho:              gatilho,
                resultado_esperado:   resultado,
                episodios_confirmados: indices.iter()
                    .take(n_confirm)
                    .map(|&i| self.episodios[i].id)
                    .collect(),
                episodios_contradizem: Vec::new(),
                fonte,
                valence_media,
                criado_em:            t_atual_s,
                ultimo_teste_s:       t_atual_s,
            };

            // Episódios do mesmo grupo com resultado diferente = contradições
            for &i in indices {
                let ep = &self.episodios[i];
                if let Some(res) = &ep.resultado {
                    if *res != candidato.resultado_esperado {
                        candidato.episodios_contradizem.push(ep.id);
                    }
                }
            }

            self.candidatos.insert(candidato.id, candidato);
            novos += 1;
            self.total_padroes_extraidos += 1;
        }

        // Marca episódios como processados
        for &i in &nao_processados {
            if let Some(ep) = self.episodios.get_mut(i) {
                ep.processado = true;
            }
        }

        novos
    }

    // ── Camada 3: Consolidação ────────────────────────────────────────────────

    /// Verifica candidatos prontos para consolidação e os promove.
    /// Chamado após a extração, ainda durante o sono.
    /// Retorna o número de padrões consolidados nesta rodada.
    pub fn consolidar(&mut self, t_atual_s: f64) -> usize {
        let prontos: Vec<Uuid> = self.candidatos.values()
            .filter(|c| c.pronto_para_consolidar())
            .map(|c| c.id)
            .collect();

        let mut consolidados = 0usize;

        for id in prontos {
            let Some(candidato) = self.candidatos.remove(&id) else { continue };

            let consolidado = PadraoConsolidado {
                id:               Uuid::new_v4(),
                gatilho:          candidato.gatilho.clone(),
                predicao:         candidato.resultado_esperado.clone(),
                forca:            0.6 + candidato.consistencia() * 0.4,
                n_acertos:        candidato.episodios_confirmados.len() as u32,
                n_erros:          candidato.episodios_contradizem.len() as u32,
                fonte:            candidato.fonte,
                valence:          candidato.valence_media,
                consolidado_em:   t_atual_s,
                ultimo_uso_s:     t_atual_s,
                candidato_origem: candidato.id,
            };

            // Indexa pelo gatilho para lookup O(1)
            let chave = candidato.gatilho.join("|");
            self.indice_gatilho.entry(chave).or_default().push(consolidado.id);

            self.consolidados.insert(consolidado.id, consolidado);
            consolidados += 1;
            self.total_consolidados += 1;
        }

        consolidados
    }

    // ── Predição ──────────────────────────────────────────────────────────────

    /// Dado o contexto atual, retorna as predições mais prováveis.
    /// Cada predição: (texto_predito, força, valência_esperada).
    /// Chamado a cada tick do loop principal.
    pub fn predizer(&self, contexto: &[String]) -> Vec<(String, f32, f32)> {
        if self.consolidados.is_empty() || contexto.is_empty() { return Vec::new(); }

        let ctx_set: std::collections::HashSet<&str> =
            contexto.iter().map(|s| s.as_str()).collect();

        let mut resultados: Vec<(String, f32, f32)> = self.consolidados.values()
            .filter(|p| p.forca > 0.1)
            .filter_map(|p| {
                // Score = sobreposição do gatilho com o contexto atual × força
                let overlap = p.gatilho.iter()
                    .filter(|g| ctx_set.contains(g.as_str()))
                    .count() as f32;
                if overlap == 0.0 { return None; }
                let score = (overlap / p.gatilho.len() as f32) * p.forca;
                if score < 0.2 { return None; }
                Some((p.predicao.clone(), score, p.valence))
            })
            .collect();

        resultados.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        resultados.truncate(5); // top 5 predições
        resultados
    }

    /// Registra o resultado real após uma predição.
    /// `predicao_feita`: o que a Selene predisse.
    /// `resultado_real`: o que de fato aconteceu.
    /// Atualiza força dos padrões envolvidos (erro de predição).
    pub fn registrar_resultado(
        &mut self,
        contexto:       &[String],
        predicao_feita: &str,
        resultado_real: &str,
        t_s:            f64,
    ) {
        self.total_predicoes += 1;
        let acertou = predicao_feita == resultado_real;
        if acertou { self.total_acertos += 1; }

        let ctx_set: std::collections::HashSet<&str> =
            contexto.iter().map(|s| s.as_str()).collect();

        for p in self.consolidados.values_mut() {
            let overlap = p.gatilho.iter()
                .filter(|g| ctx_set.contains(g.as_str()))
                .count();
            if overlap > 0 && p.predicao == predicao_feita {
                p.registrar_predicao(acertou, t_s);
            }
        }
    }

    // ── Decaimento ────────────────────────────────────────────────────────────

    /// Decai força de padrões consolidados não usados.
    /// Deve ser chamado periodicamente (ex: a cada ciclo de sono N1).
    pub fn tick_decay(&mut self, t_s: f64) {
        for p in self.consolidados.values_mut() {
            p.decair(t_s);
        }
        // Remove padrões muito fracos (força < 0.02)
        self.consolidados.retain(|_, p| p.forca >= 0.02);
        // Reconstrói índice
        self.reconstruir_indice();
    }

    // ── Snapshot para UI / WebSocket ──────────────────────────────────────────

    /// Retorna resumo dos padrões mais fortes para o frontend.
    pub fn snapshot_top(&self, n: usize) -> Vec<(String, String, f32, &str)> {
        let mut v: Vec<_> = self.consolidados.values()
            .map(|p| (
                p.gatilho.join(" "),
                p.predicao.clone(),
                p.forca,
                p.fonte.como_str(),
            ))
            .collect();
        v.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        v.truncate(n);
        v
    }

    pub fn stats(&self) -> PatternStats {
        PatternStats {
            episodios_na_fila:   self.episodios.len(),
            candidatos:          self.candidatos.len(),
            consolidados:        self.consolidados.len(),
            total_gravados:      self.total_episodios_gravados,
            total_extraidos:     self.total_padroes_extraidos,
            total_consolidados:  self.total_consolidados,
            precisao_media:      if self.total_predicoes == 0 { 0.0 }
                                 else { self.total_acertos as f32 / self.total_predicoes as f32 },
        }
    }

    // ── Helpers privados ──────────────────────────────────────────────────────

    fn reconstruir_indice(&mut self) {
        self.indice_gatilho.clear();
        for (&id, p) in &self.consolidados {
            let chave = p.gatilho.join("|");
            self.indice_gatilho.entry(chave).or_default().push(id);
        }
    }
}

// ── Stats ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PatternStats {
    pub episodios_na_fila:  usize,
    pub candidatos:         usize,
    pub consolidados:       usize,
    pub total_gravados:     u64,
    pub total_extraidos:    u64,
    pub total_consolidados: u64,
    pub precisao_media:     f32,
}

// ── Testes ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(engine: &mut PatternEngine, ctx: &[&str], acao: &str, resultado: &str, v: f32) {
        engine.gravar(
            1000.0,
            FonteEpisodio::Chat,
            ctx.iter().map(|s| s.to_string()).collect(),
            acao.to_string(),
            Some(resultado.to_string()),
            v,
            [1.0, 0.5, 0.3, 0.8, 0.2],
        );
    }

    #[test]
    fn extrai_padrao_com_3_episodios() {
        let mut engine = PatternEngine::novo();
        // 4 episódios com mesmo contexto e resultado
        for _ in 0..4 {
            ep(&mut engine, &["calor", "verão", "sol"], "temperatura alta", "desconforto", 0.3);
        }
        let novos = engine.extrair_padroes(2000.0);
        assert!(novos >= 1, "deveria ter extraído ao menos 1 candidato");
        assert!(engine.candidatos.len() >= 1);
    }

    #[test]
    fn consolida_apos_confirmacoes() {
        let mut engine = PatternEngine::novo();
        // MIN_CONFIRMACOES_CONSOLIDAR = 5
        for _ in 0..6 {
            ep(&mut engine, &["fome", "vazio", "dor"], "fome intensa", "comer", 0.8);
        }
        engine.extrair_padroes(2000.0);
        // Força consolidação manual dos candidatos
        for c in engine.candidatos.values_mut() {
            for _ in 0..6 {
                c.episodios_confirmados.push(Uuid::new_v4());
            }
        }
        let n = engine.consolidar(3000.0);
        assert!(n >= 1, "deveria ter consolidado ao menos 1 padrão");
    }

    #[test]
    fn prediz_a_partir_de_contexto() {
        let mut engine = PatternEngine::novo();
        // Insere padrão consolidado diretamente para testar predição
        let id = Uuid::new_v4();
        engine.consolidados.insert(id, PadraoConsolidado {
            id,
            gatilho:          vec!["frio".into(), "inverno".into()],
            predicao:         "agasalho".into(),
            forca:            0.8,
            n_acertos:        10,
            n_erros:          1,
            fonte:            FonteEpisodio::Chat,
            valence:          0.5,
            consolidado_em:   1000.0,
            ultimo_uso_s:     1000.0,
            candidato_origem: Uuid::new_v4(),
        });

        let preds = engine.predizer(&["frio".to_string(), "inverno".to_string(), "neve".to_string()]);
        assert!(!preds.is_empty(), "deveria predizer algo");
        assert_eq!(preds[0].0, "agasalho");
    }

    #[test]
    fn erro_predicao_reduz_forca() {
        let mut engine = PatternEngine::novo();
        let id = Uuid::new_v4();
        engine.consolidados.insert(id, PadraoConsolidado {
            id,
            gatilho:          vec!["teste".into()],
            predicao:         "resultado_a".into(),
            forca:            0.8,
            n_acertos:        5,
            n_erros:          0,
            fonte:            FonteEpisodio::Chat,
            valence:          0.5,
            consolidado_em:   0.0,
            ultimo_uso_s:     0.0,
            candidato_origem: Uuid::new_v4(),
        });

        engine.registrar_resultado(&["teste".to_string()], "resultado_a", "resultado_b", 2000.0);
        let forca_depois = engine.consolidados[&id].forca;
        assert!(forca_depois < 0.8, "força deveria ter caído após erro");
    }
}
