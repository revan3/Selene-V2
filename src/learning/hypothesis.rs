// src/learning/hypothesis.rs
//
// Motor de hipóteses — Selene formula, testa e aprende com predições sobre o mundo.
//
// Uma hipótese é uma predição sobre o que vai acontecer dado o contexto atual.
// Biologicamente: o PFC (córtex pré-frontal) gera predições; o hipocampo testa via
// theta-sequences; a amígdala modula a salência pelo resultado (RPE).
//
// Tipos de hipóteses:
//
//   SemanticaContigua    — "dado contexto [A, B], o conceito C deve aparecer logo"
//                          → aprendizado associativo guiado por predição
//
//   EmocionalCongruente  — "no meu estado emocional atual, palavras com valência V
//                           são mais prováveis no próximo input"
//                          → coloração afetiva da percepção (congruência de humor)
//
//   GapConhecimento      — "sobre o tópico T tenho ≤2 associações — é uma lacuna"
//                          → direciona curiosidade e guia perguntas autônomas
//
//   ComportamentalPropria — "quando o contexto tem [X], minha resposta tem padrão Y"
//                           → autoobservação; radar para futura auto-correção de código
//                           → fundação para Selene aprender a programar e corrigir
//                              o próprio comportamento
//
// Fluxo de uso por turno de chat:
//   1. Mensagem chega → testar() — confronta hipóteses do turno anterior, gera RPE
//   2. RPE propaga para grounding_rpe() → reforça/enfraquece associações perceptuais
//   3. Resposta gerada → formular() — cria novas hipóteses para o próximo turno
//   4. observar_comportamento() — acumula padrões da própria resposta
//   5. Hipóteses fracas são purgadas automaticamente; fortes persistem

#![allow(dead_code)]

use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

fn instant_now() -> Instant { Instant::now() }

// Sentinels para EmocionalCongruente — valores impossíveis para FNV-1a hash
const CONCLUSAO_EMO_POS: u32 = 1;
const CONCLUSAO_EMO_NEG: u32 = 2;

// ─────────────────────────────────────────────────────────────────────────────
// Tipos
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TipoHipotese {
    /// "Dado contexto [A, B], o próximo conceito relevante será C"
    SemanticaContigua,
    /// "No meu estado emocional atual, palavras com valência V são esperadas"
    EmocionalCongruente,
    /// "Sobre o tópico T tenho poucas associações — é uma lacuna no meu conhecimento"
    GapConhecimento,
    /// "Quando o contexto tem palavras [X], minha resposta tem padrão Y"
    /// Fundação para Selene observar e futuramente corrigir o próprio código.
    ComportamentalPropria,
}

/// Uma predição que Selene mantém em memória de trabalho até ser testada.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hipotese {
    /// Identificador único da hipótese nesta sessão.
    pub id: u64,
    pub tipo: TipoHipotese,
    /// Evidências que geraram a hipótese (concept_ids do contexto).
    pub premissas: Vec<u32>,
    /// Concept_id previsto. Sentinels: 1=emo_pos, 2=emo_neg.
    pub conclusao: u32,
    /// Confiança atual [0.0, 1.0] — sobe com confirmações, cai com refutações.
    pub confianca: f32,
    /// Número de vezes confirmada.
    pub confirmacoes: u32,
    /// Número de vezes refutada.
    pub refutacoes: u32,
    /// Número total de testes.
    pub n_testes: u32,
    /// Timestamp de criação (não persiste — reiniciado ao carregar).
    #[serde(skip, default = "instant_now")]
    pub criada_em: Instant,
}

impl Hipotese {
    /// Taxa de acerto desta hipótese [0.0, 1.0].
    /// Prior neutro de 0.5 antes do primeiro teste.
    pub fn taxa_acerto(&self) -> f32 {
        let total = self.confirmacoes + self.refutacoes;
        if total == 0 { return 0.5; }
        self.confirmacoes as f32 / total as f32
    }

    /// Hipótese confiável: ≥10 testes e taxa > 65%.
    pub fn e_confiavel(&self) -> bool {
        self.n_testes >= 5 && self.taxa_acerto() > 0.65
    }

    /// Hipótese descartável: ≥5 testes e taxa < 20%.
    pub fn deve_descartar(&self) -> bool {
        self.n_testes >= 5 && self.taxa_acerto() < 0.20
    }

    /// Hipótese nova: ainda sem testes suficientes para julgar.
    pub fn e_nova(&self) -> bool {
        self.n_testes < 3
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Motor principal
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct HypothesisEngine {
    /// Hipóteses ativas em memória de trabalho (máx 60).
    pub hipoteses: Vec<Hipotese>,
    /// Histórico de resultados: (id_hipotese, foi_confirmada, rpe_gerado).
    pub historico: VecDeque<(u64, bool, f32)>,
    /// Taxa de acerto recente (janela deslizante de 50 testes).
    pub taxa_acerto_recente: f32,
    /// Total de hipóteses formuladas na sessão.
    pub total_formuladas: u64,
    /// Total de hipóteses testadas na sessão.
    pub total_testadas: u64,
    /// Padrões comportamentais: chave_ctx_id → (reply_id, contagem).
    padroes_proprios: HashMap<u32, (u32, u32)>,
    /// Próximo ID a ser atribuído.
    proximo_id: u64,
    /// Última resposta gerada — necessária para testar hipóteses no próximo turno.
    pub ultimo_reply: String,
}

impl Default for HypothesisEngine {
    fn default() -> Self { Self::new() }
}

impl HypothesisEngine {
    pub fn new() -> Self {
        Self {
            hipoteses: Vec::with_capacity(60),
            historico: VecDeque::with_capacity(200),
            taxa_acerto_recente: 0.5,
            total_formuladas: 0,
            total_testadas: 0,
            padroes_proprios: HashMap::new(),
            proximo_id: 1,
            ultimo_reply: String::new(),
        }
    }

    // ── Formulação ────────────────────────────────────────────────────────────

    /// Formula hipóteses a partir do contexto atual.
    ///
    /// Gera dois tipos automaticamente:
    ///   1. SemanticaContigua  — para cada palavra significativa, prevê o melhor vizinho no grafo
    ///   2. GapConhecimento    — palavras com ≤2 arestas e sem valência → lacuna
    ///
    /// Retorna os IDs das hipóteses recém-criadas.
    pub fn formular(
        &mut self,
        contexto: &[u32],
        grafo: &HashMap<u32, Vec<(u32, f32)>>,
        valencias: &HashMap<u32, f32>,
        emocao: f32,
        stop_ids: &[u32],
    ) -> Vec<u64> {
        let mut novos_ids = Vec::new();

        for &palavra in contexto.iter()
            .filter(|id| !stop_ids.contains(id))
        {
            // ── Hipótese semântica ──────────────────────────────────────────
            if let Some(vizinhos) = grafo.get(&palavra) {
                if !vizinhos.is_empty() {
                    let melhor = vizinhos.iter()
                        .filter(|(w, _)| !stop_ids.contains(w))
                        .max_by(|a, b| a.1.partial_cmp(&b.1)
                            .unwrap_or(std::cmp::Ordering::Equal));

                    if let Some(&(proximo, peso)) = melhor {
                        let ja_existe = self.hipoteses.iter().any(|h|
                            h.tipo == TipoHipotese::SemanticaContigua
                            && h.premissas.first().copied() == Some(palavra)
                            && h.conclusao == proximo
                        );
                        if !ja_existe {
                            let id = self.novo_id();
                            let confianca = (peso * 0.5 + 0.20).clamp(0.15, 0.80);
                            self.hipoteses.push(Hipotese {
                                id,
                                tipo: TipoHipotese::SemanticaContigua,
                                premissas: vec![palavra],
                                conclusao: proximo,
                                confianca,
                                confirmacoes: 0,
                                refutacoes: 0,
                                n_testes: 0,
                                criada_em: Instant::now(),
                            });
                            self.total_formuladas += 1;
                            novos_ids.push(id);
                        }
                    }
                }
            }

            // ── Hipótese de gap ─────────────────────────────────────────────
            {
                let n_arestas = grafo.get(&palavra).map(|v| v.len()).unwrap_or(0);
                let tem_valencia = valencias.contains_key(&palavra);
                if n_arestas <= 2 && !tem_valencia {
                    let ja_existe = self.hipoteses.iter().any(|h|
                        h.tipo == TipoHipotese::GapConhecimento
                        && h.premissas.first().copied() == Some(palavra)
                    );
                    if !ja_existe {
                        let id = self.novo_id();
                        self.hipoteses.push(Hipotese {
                            id,
                            tipo: TipoHipotese::GapConhecimento,
                            premissas: vec![palavra],
                            conclusao: palavra,
                            confianca: 0.80,
                            confirmacoes: 0,
                            refutacoes: 0,
                            n_testes: 0,
                            criada_em: Instant::now(),
                        });
                        self.total_formuladas += 1;
                        novos_ids.push(id);
                    }
                }
            }

            // ── Hipótese emocional ──────────────────────────────────────────
            if let Some(&valencia) = valencias.get(&palavra) {
                let congruente = emocao * valencia > 0.0;
                if congruente && emocao.abs() > 0.3 {
                    let conclusao = if valencia > 0.0 { CONCLUSAO_EMO_POS } else { CONCLUSAO_EMO_NEG };
                    let ja_existe = self.hipoteses.iter().any(|h|
                        h.tipo == TipoHipotese::EmocionalCongruente
                        && h.conclusao == conclusao
                    );
                    if !ja_existe {
                        let id = self.novo_id();
                        self.hipoteses.push(Hipotese {
                            id,
                            tipo: TipoHipotese::EmocionalCongruente,
                            premissas: vec![palavra],
                            conclusao,
                            confianca: 0.55,
                            confirmacoes: 0,
                            refutacoes: 0,
                            n_testes: 0,
                            criada_em: Instant::now(),
                        });
                        self.total_formuladas += 1;
                        novos_ids.push(id);
                    }
                }
            }
        }

        // Limite de hipóteses ativas: 60 — retém as mais confiantes
        if self.hipoteses.len() > 60 {
            self.hipoteses.sort_by(|a, b|
                b.confianca.partial_cmp(&a.confianca)
                    .unwrap_or(std::cmp::Ordering::Equal)
            );
            self.hipoteses.truncate(60);
        }

        novos_ids
    }

    // ── Teste ─────────────────────────────────────────────────────────────────

    /// Testa hipóteses ativas contra o input observado e a resposta anterior.
    ///
    /// Confronta as predições do turno anterior com o que o usuário realmente disse.
    /// Retorna RPE agregado: > 0 = predições melhores que esperado, < 0 = pior.
    ///
    /// O RPE deve ser propagado para `BrainState::grounding_rpe()` para que as
    /// associações perceptuais confirmadas sejam reforçadas.
    pub fn testar(
        &mut self,
        input_tokens: &[u32],
        valencias: &HashMap<u32, f32>,
        reply_tokens: &[u32],
    ) -> f32 {
        let tokens_set: std::collections::HashSet<u32> = input_tokens.iter().copied().collect();
        let reply_set: std::collections::HashSet<u32> = reply_tokens.iter().copied().collect();

        let mut rpe_soma = 0.0f32;
        let mut n_testadas = 0usize;

        for h in self.hipoteses.iter_mut() {
            match h.tipo {
                TipoHipotese::SemanticaContigua => {
                    let premissa_presente = h.premissas.iter()
                        .any(|p| tokens_set.contains(p));
                    if !premissa_presente { continue; }

                    let confirmada = tokens_set.contains(&h.conclusao)
                        || reply_set.contains(&h.conclusao);

                    h.n_testes += 1;
                    n_testadas += 1;
                    self.total_testadas += 1;

                    if confirmada {
                        h.confirmacoes += 1;
                        h.confianca = (h.confianca + 0.05).min(1.0);
                        let rpe = 0.25 * h.confianca; // mais confiante → RPE mais forte
                        rpe_soma += rpe;
                        self.historico.push_back((h.id, true, rpe));
                    } else {
                        h.refutacoes += 1;
                        h.confianca = (h.confianca - 0.08).max(0.0);
                        let rpe = -0.12 * (1.0 - h.confianca);
                        rpe_soma += rpe;
                        self.historico.push_back((h.id, false, rpe));
                    }
                }

                TipoHipotese::EmocionalCongruente => {
                    let valencias_input: Vec<f32> = input_tokens.iter()
                        .filter_map(|t| valencias.get(t).copied())
                        .collect();
                    if valencias_input.is_empty() { continue; }

                    let media_val: f32 = valencias_input.iter().sum::<f32>()
                        / valencias_input.len() as f32;
                    let confirmada = if h.conclusao == CONCLUSAO_EMO_POS {
                        media_val > 0.05
                    } else {
                        media_val < -0.05
                    };

                    h.n_testes += 1;
                    n_testadas += 1;
                    self.total_testadas += 1;

                    if confirmada {
                        h.confirmacoes += 1;
                        h.confianca = (h.confianca + 0.04).min(1.0);
                        rpe_soma += 0.15;
                        self.historico.push_back((h.id, true, 0.15));
                    } else {
                        h.refutacoes += 1;
                        h.confianca = (h.confianca - 0.06).max(0.0);
                        rpe_soma -= 0.08;
                        self.historico.push_back((h.id, false, -0.08));
                    }
                }

                // GapConhecimento e ComportamentalPropria são observacionais —
                // não geram RPE diretamente
                _ => {}
            }
        }

        // Purga hipóteses descartáveis
        self.hipoteses.retain(|h| !h.deve_descartar());

        // Mantém histórico em 200
        while self.historico.len() > 200 {
            self.historico.pop_front();
        }

        // Atualiza taxa de acerto recente (janela de 50)
        if !self.historico.is_empty() {
            let confirmadas = self.historico.iter().rev().take(50)
                .filter(|(_, ok, _)| *ok)
                .count();
            let total_janela = self.historico.len().min(50);
            self.taxa_acerto_recente = confirmadas as f32 / total_janela as f32;
        }

        if n_testadas > 0 { rpe_soma / n_testadas as f32 } else { 0.0 }
    }

    // ── Autoobservação comportamental ──────────────────────────────────────────

    /// Registra uma observação do próprio comportamento.
    ///
    /// Quando Selene recebe contexto com chave `chave_ctx` e sua resposta começa
    /// com `reply_resumido`, acumula esta observação.
    /// Após 3 repetições do mesmo padrão, formula uma hipótese ComportamentalPropria.
    ///
    /// Esta é a fundação para a futura capacidade de autoprogramação:
    /// Selene observa que "quando ouço X, sempre digo Y" → pode questionar se Y
    /// é a resposta ideal → pode propor mudança no próprio comportamento.
    pub fn observar_comportamento(&mut self, chave_id: u32, reply_id: u32) {
        if chave_id == 0 || reply_id == 0 { return; }

        let entry = self.padroes_proprios
            .entry(chave_id)
            .or_insert((reply_id, 0));

        if entry.0 == reply_id {
            entry.1 += 1;
        } else {
            *entry = (reply_id, 1);
            return;
        }

        if entry.1 == 3 {
            let ja_existe = self.hipoteses.iter().any(|h|
                h.tipo == TipoHipotese::ComportamentalPropria
                && h.premissas.first().copied() == Some(chave_id)
            );
            if !ja_existe {
                let id = self.novo_id();
                self.hipoteses.push(Hipotese {
                    id,
                    tipo: TipoHipotese::ComportamentalPropria,
                    premissas: vec![chave_id],
                    conclusao: reply_id,
                    confianca: 0.70,
                    confirmacoes: 3,
                    refutacoes: 0,
                    n_testes: 3,
                    criada_em: Instant::now(),
                });
                self.total_formuladas += 1;
            }
        }
    }

    // ── Consultas ─────────────────────────────────────────────────────────────

    /// Hipóteses de gap de conhecimento — direcionam curiosidade e perguntas autônomas.
    pub fn gaps_conhecimento(&self) -> Vec<&Hipotese> {
        self.hipoteses.iter()
            .filter(|h| h.tipo == TipoHipotese::GapConhecimento)
            .collect()
    }

    /// Hipóteses comportamentais próprias — radar de autoprogramação.
    pub fn padroes_proprios_ativos(&self) -> Vec<&Hipotese> {
        self.hipoteses.iter()
            .filter(|h| h.tipo == TipoHipotese::ComportamentalPropria)
            .collect()
    }

    /// Hipóteses semânticas confiáveis (≥10 testes, taxa >65%).
    /// Podem ser usadas para enriquecer o grafo com arestas de alta confiança.
    pub fn hipoteses_confiaveis(&self) -> Vec<&Hipotese> {
        self.hipoteses.iter()
            .filter(|h| h.tipo == TipoHipotese::SemanticaContigua && h.e_confiavel())
            .collect()
    }

    /// Retorna o tópico mais provável do próximo input do usuário,
    /// baseado nas hipóteses semânticas com maior confiança.
    pub fn proximo_topico_previsto(&self) -> Option<u32> {
        self.hipoteses.iter()
            .filter(|h| h.tipo == TipoHipotese::SemanticaContigua && !h.e_nova())
            .max_by(|a, b| a.confianca.partial_cmp(&b.confianca)
                .unwrap_or(std::cmp::Ordering::Equal))
            .map(|h| h.conclusao)
    }

    /// Resumo compacto para logging.
    pub fn resumo(&self) -> String {
        let semanticas = self.hipoteses.iter()
            .filter(|h| h.tipo == TipoHipotese::SemanticaContigua).count();
        let gaps = self.hipoteses.iter()
            .filter(|h| h.tipo == TipoHipotese::GapConhecimento).count();
        let proprias = self.hipoteses.iter()
            .filter(|h| h.tipo == TipoHipotese::ComportamentalPropria).count();
        format!(
            "hipóteses[sem={} gap={} prop={} | form={} test={} acerto={:.0}%]",
            semanticas, gaps, proprias,
            self.total_formuladas, self.total_testadas,
            self.taxa_acerto_recente * 100.0
        )
    }

    // ── Persistência ─────────────────────────────────────────────────────────

    pub fn salvar(&self, path: &str) {
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, json);
        }
    }

    pub fn carregar(path: &str) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(Self::new)
    }

    // ── Privado ───────────────────────────────────────────────────────────────

    fn novo_id(&mut self) -> u64 {
        let id = self.proximo_id;
        self.proximo_id += 1;
        id
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Testes
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_pool::word_to_concept_id as wid;

    fn grafo_teste() -> HashMap<u32, Vec<(u32, f32)>> {
        let mut g = HashMap::new();
        g.insert(wid("amor"), vec![
            (wid("coração"), 0.9),
            (wid("paixão"), 0.7),
        ]);
        g.insert(wid("fogo"), vec![
            (wid("calor"), 0.8),
            (wid("luz"), 0.6),
        ]);
        g
    }

    fn valencias_teste() -> HashMap<u32, f32> {
        let mut v = HashMap::new();
        v.insert(wid("amor"), 0.9f32);
        v.insert(wid("fogo"), 0.6f32);
        v.insert(wid("medo"), -0.8f32);
        v
    }

    #[test]
    fn formula_hipotese_semantica() {
        let mut engine = HypothesisEngine::new();
        let grafo = grafo_teste();
        let val = valencias_teste();
        let stop: &[u32] = &[];
        let ctx = vec![wid("amor")];

        let ids = engine.formular(&ctx, &grafo, &val, 0.5, stop);
        assert!(!ids.is_empty(), "deve formular pelo menos uma hipótese");

        let sem: Vec<_> = engine.hipoteses.iter()
            .filter(|h| h.tipo == TipoHipotese::SemanticaContigua)
            .collect();
        assert!(!sem.is_empty());
        assert_eq!(sem[0].premissas[0], wid("amor"));
        assert_eq!(sem[0].conclusao, wid("coração"));
    }

    #[test]
    fn formula_gap_para_palavra_sem_valencia() {
        let mut engine = HypothesisEngine::new();
        let mut grafo = HashMap::new();
        grafo.insert(wid("zumbi"), vec![(wid("susto"), 0.3)]);
        let val: HashMap<u32, f32> = HashMap::new();
        let stop: &[u32] = &[];
        let ctx = vec![wid("zumbi")];

        engine.formular(&ctx, &grafo, &val, 0.0, stop);
        let gaps: Vec<_> = engine.hipoteses.iter()
            .filter(|h| h.tipo == TipoHipotese::GapConhecimento)
            .collect();
        assert!(!gaps.is_empty(), "deve detectar gap para 'zumbi'");
    }

    #[test]
    fn testa_confirmacao_aumenta_confianca() {
        let mut engine = HypothesisEngine::new();
        let grafo = grafo_teste();
        let val = valencias_teste();
        let stop: &[u32] = &[];
        let ctx = vec![wid("amor")];
        engine.formular(&ctx, &grafo, &val, 0.5, stop);

        let confianca_antes = engine.hipoteses.iter()
            .find(|h| h.tipo == TipoHipotese::SemanticaContigua)
            .map(|h| h.confianca)
            .unwrap_or(0.0);

        let input = vec![wid("amor"), wid("coração")];
        let rpe = engine.testar(&input, &val, &[]);

        let confianca_depois = engine.hipoteses.iter()
            .find(|h| h.tipo == TipoHipotese::SemanticaContigua)
            .map(|h| h.confianca)
            .unwrap_or(0.0);

        assert!(rpe > 0.0, "RPE deve ser positivo quando confirmado: rpe={}", rpe);
        assert!(confianca_depois > confianca_antes,
            "confiança deve crescer: antes={} depois={}", confianca_antes, confianca_depois);
    }

    #[test]
    fn testa_refutacao_reduz_confianca() {
        let mut engine = HypothesisEngine::new();
        let grafo = grafo_teste();
        let val = valencias_teste();
        let stop: &[u32] = &[];
        let ctx = vec![wid("amor")];
        engine.formular(&ctx, &grafo, &val, 0.0, stop);

        let confianca_antes = engine.hipoteses.iter()
            .find(|h| h.tipo == TipoHipotese::SemanticaContigua)
            .map(|h| h.confianca)
            .unwrap_or(0.0);

        let input = vec![wid("amor"), wid("montanha"), wid("neve")];
        let rpe = engine.testar(&input, &val, &[]);

        let confianca_depois = engine.hipoteses.iter()
            .find(|h| h.tipo == TipoHipotese::SemanticaContigua)
            .map(|h| h.confianca)
            .unwrap_or(0.0);

        assert!(rpe < 0.0, "RPE deve ser negativo quando refutado: rpe={}", rpe);
        assert!(confianca_depois < confianca_antes,
            "confiança deve cair: antes={} depois={}", confianca_antes, confianca_depois);
    }

    #[test]
    fn observar_comportamento_gera_hipotese_propria() {
        let mut engine = HypothesisEngine::new();
        let chave = wid("selene");
        let reply = wid("meu nome");
        for _ in 0..3 {
            engine.observar_comportamento(chave, reply);
        }
        let proprias = engine.padroes_proprios_ativos();
        assert!(!proprias.is_empty(), "deve gerar hipótese comportamental após 3 repetições");
        assert_eq!(proprias[0].conclusao, reply);
    }

    #[test]
    fn nao_duplica_hipotese_identica() {
        let mut engine = HypothesisEngine::new();
        let grafo = grafo_teste();
        let val = valencias_teste();
        let stop: &[u32] = &[];
        let ctx = vec![wid("amor")];

        engine.formular(&ctx, &grafo, &val, 0.5, stop);
        let n_antes = engine.hipoteses.len();
        engine.formular(&ctx, &grafo, &val, 0.5, stop);
        assert_eq!(engine.hipoteses.len(), n_antes, "não deve duplicar");
    }
}
