// src/learning/templates.rs
// Sistema de Templates Cognitivos — padrões relacionais com slots efêmeros.
//
// CONCEITO CENTRAL:
//   Um template é uma topologia sináptica persistente com lacunas (slots) em branco.
//   O conteúdo dos slots é efêmero: entra durante o uso, é apagado após.
//   A estrutura (topologia + relações) persiste e evolui com uso validado.
//
//   Analogia: "saber desenhar" = template motor ainda presente mas com slots
//   de calibração fina decaídos por desuso — a estrutura sobrevive, o
//   preenchimento preciso não.
//
// CICLO DE VIDA:
//   Nascente (0–2 validações)  → muito maleável, slots sem restrições
//   Desenvolvendo (3–19)       → plástico, restrições emergindo
//   Consolidado (20–99)        → estrutura estável, gera filhos
//   Automático (≥100)          → ativa sem esforço, plasticidade mínima
//   Arquivado (força < 0.05)   → dormente, reativa com plasticidade 0.5
//
// OPERAÇÕES:
//   Uso parcial    — slots obrigatórios preenchidos, opcionais vazios
//   Uso completo   — todos os slots preenchidos
//   Empilhamento   — output de A alimenta slot de B (PilhaTemplates)
//   Combinação     — A+B usados juntos ≥5x → novo template composto
//   Decomposição   — template complexo → sub-templates constituintes
//   Filho          — template consolidado gera variante especializada
//
// DOMÍNIOS:
//   Linguagem, Matemática, Lógica, Motor, Sensorial, Causal, Composto
//
// INTEGRAÇÃO COM SELENE:
//   SwapManager.template_store: TemplateStore
//   gerar_resposta_emergente usa templates antes do walk livre como fallback

#![allow(dead_code)]

use std::collections::HashMap;
use std::collections::VecDeque;
use uuid::Uuid;

// ── Constantes ────────────────────────────────────────────────────────────────

/// Validações necessárias para cada transição de estado.
const VALIDACOES_DESENVOLVENDO: u32 = 3;
const VALIDACOES_CONSOLIDADO: u32   = 20;
const VALIDACOES_AUTOMATICO: u32    = 100;

/// Força mínima antes de arquivar.
const FORCA_MINIMA_ARQUIVO: f32 = 0.05;

/// Quantas vezes A+B precisam ser usados juntos para gerar template composto.
const MIN_COMBINACOES_CONSOLIDA: u32 = 5;

/// Máximo de entradas no histórico de cada slot.
const MAX_HISTORICO_SLOT: usize = 100;

/// Máximo de pilhas recentes rastreadas.
const MAX_PILHAS_HIST: usize = 50;

/// Máximo de templates no repositório antes de podar os mais fracos.
const MAX_TEMPLATES: usize = 2048;

// ── Enums ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum EstadoTemplate {
    Nascente,
    Desenvolvendo,
    Consolidado,
    Automatico,
    Arquivado,
}

/// Domínio primário de um template.
#[derive(Debug, Clone, PartialEq)]
pub enum Dominio {
    Linguagem,
    Matematica,
    Logica,
    Motor,
    Sensorial,
    Causal,
    Composto(Vec<String>), // domínios que contribuíram — emerge do uso
}

impl Dominio {
    pub fn como_str(&self) -> String {
        match self {
            Dominio::Linguagem         => "linguagem".into(),
            Dominio::Matematica        => "matematica".into(),
            Dominio::Logica            => "logica".into(),
            Dominio::Motor             => "motor".into(),
            Dominio::Sensorial         => "sensorial".into(),
            Dominio::Causal            => "causal".into(),
            Dominio::Composto(ds)      => ds.join("+"),
        }
    }
}

/// Tipo semântico de um slot.
#[derive(Debug, Clone, PartialEq)]
pub enum TipoSlot {
    Conceito,    // substantivo / entidade
    Acao,        // verbo / processo
    Relacao,     // conector relacional (causa, implica, =, >)
    Operador,    // operador matemático (+, ×, /, ^)
    Quantidade,  // grandeza mensurável / numérica
    Atributo,    // qualidade / propriedade
    Livre,       // sem restrição de tipo
}

/// Tipo de relação entre slots.
#[derive(Debug, Clone, PartialEq)]
pub enum TipoRelacao {
    Sequencial,    // A precede B (linguagem, motor)
    Causal,        // A implica / causa B
    Matematica,    // A = f(B, C) — relação funcional
    Atributiva,    // A tem propriedade B
    Composicional, // A é parte de B
}

/// Como o template foi usado num dado instante.
#[derive(Debug, Clone, PartialEq)]
pub enum TipoUso {
    Completo,  // todos os slots obrigatórios preenchidos
    Parcial,   // ao menos um slot obrigatório vazio
    Empilhado, // usado como parte de uma PilhaTemplates
}

// ── Slot ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Slot {
    pub indice:      usize,
    pub tipo:        TipoSlot,
    pub obrigatorio: bool,

    /// Histórico de conteúdos que preencheram este slot com sucesso.
    /// Permite inferir restrições semânticas emergentes.
    /// Formato: (conceito, score_de_validacao)
    pub historico: Vec<(String, f32)>,

    /// Conteúdo atual — EFÊMERO.
    /// Preenchido durante o uso via `Template::preencher()`.
    /// Limpo via `Template::limpar_slots()` após uso.
    pub conteudo_atual: Option<String>,
}

impl Slot {
    pub fn novo(indice: usize, tipo: TipoSlot, obrigatorio: bool) -> Self {
        Self {
            indice,
            tipo,
            obrigatorio,
            historico: Vec::new(),
            conteudo_atual: None,
        }
    }

    pub fn limpar(&mut self) {
        self.conteudo_atual = None;
    }

    /// Top-N conceitos mais frequentes no histórico — restrição semântica emergente.
    pub fn restricao_emergente(&self, top_n: usize) -> Vec<String> {
        let mut freq: HashMap<&str, u32> = HashMap::new();
        for (c, _) in &self.historico {
            *freq.entry(c.as_str()).or_insert(0) += 1;
        }
        let mut sorted: Vec<(&str, u32)> = freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.into_iter().take(top_n).map(|(s, _)| s.to_string()).collect()
    }

    /// Score médio de validação do histórico deste slot.
    pub fn score_medio(&self) -> f32 {
        if self.historico.is_empty() { return 0.0; }
        self.historico.iter().map(|(_, s)| s).sum::<f32>() / self.historico.len() as f32
    }
}

// ── Relacao ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Relacao {
    pub de:    usize,
    pub para:  usize,
    pub tipo:  TipoRelacao,
    pub forca: f32,
}

// ── Template ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Template {
    pub id:      Uuid,
    pub nome:    Option<String>,
    pub dominio: Dominio,
    pub estado:  EstadoTemplate,

    pub slots:    Vec<Slot>,
    pub relacoes: Vec<Relacao>,

    // ── Memória ───────────────────────────────────────────────────────────
    pub forca:          f32,
    pub ultimo_uso_s:   f64,  // timestamp em segundos (monotonic ou unix)
    pub n_validacoes:   u32,
    pub n_uso_completo: u32,
    pub n_uso_parcial:  u32,
    pub n_uso_empilhado: u32,

    // ── Genealogia ────────────────────────────────────────────────────────
    /// Template mais simples do qual este derivou (None = raiz).
    pub pai: Option<Uuid>,
    /// Sub-templates que compõem este (se for composto).
    pub sub_templates: Vec<Uuid>,
    /// Combinações frequentes com outros templates → candidatos a composição.
    pub combina_com: Vec<(Uuid, u32)>,

    // ── Categorização emergente ───────────────────────────────────────────
    pub tags: Vec<String>,

    pub complexidade: u32,
}

impl Template {
    pub fn novo(dominio: Dominio, slots: Vec<Slot>, relacoes: Vec<Relacao>) -> Self {
        let complexidade = slots.len() as u32 + relacoes.len() as u32;
        Self {
            id:              Uuid::new_v4(),
            nome:            None,
            dominio,
            estado:          EstadoTemplate::Nascente,
            slots,
            relacoes,
            forca:           0.5,
            ultimo_uso_s:    0.0,
            n_validacoes:    0,
            n_uso_completo:  0,
            n_uso_parcial:   0,
            n_uso_empilhado: 0,
            pai:             None,
            sub_templates:   Vec::new(),
            combina_com:     Vec::new(),
            tags:            Vec::new(),
            complexidade,
        }
    }

    /// Plasticidade atual: quão maleável é a estrutura deste template.
    /// Alta = aceita novos conteúdos nos slots, pode crescer.
    /// Baixa = resistente a modificação, gera filhos em vez de mudar.
    pub fn plasticidade(&self) -> f32 {
        match self.estado {
            EstadoTemplate::Nascente      => 1.0,
            EstadoTemplate::Desenvolvendo => 0.7,
            EstadoTemplate::Consolidado   => 0.3,
            EstadoTemplate::Automatico    => 0.1,
            EstadoTemplate::Arquivado     => 0.5,
        }
    }

    /// Registra um uso e atualiza força + estado.
    pub fn reforcar(&mut self, tipo: TipoUso, validado: bool, t_atual_s: f64) {
        let delta: f32 = match (&tipo, validado) {
            (TipoUso::Completo,  true)  => 0.08,
            (TipoUso::Empilhado, true)  => 0.05,
            (TipoUso::Parcial,   true)  => 0.03,
            (_,                  false) => -0.02,
        };
        self.forca = (self.forca + delta).clamp(0.0, 1.0);
        self.ultimo_uso_s = t_atual_s;

        match tipo {
            TipoUso::Completo   => self.n_uso_completo  += 1,
            TipoUso::Parcial    => self.n_uso_parcial    += 1,
            TipoUso::Empilhado  => self.n_uso_empilhado += 1,
        }
        if validado { self.n_validacoes += 1; }

        self.atualizar_estado();
    }

    /// Decaimento temporal — deve ser chamado periodicamente.
    pub fn decair(&mut self, t_atual_s: f64) {
        if self.ultimo_uso_s <= 0.0 { return; }
        let dt = t_atual_s - self.ultimo_uso_s;
        if dt <= 0.0 { return; }

        // Meia-vida cresce com o uso: template bem praticado decai muito mais devagar.
        let meia_vida_s: f64 = match self.n_validacoes {
            0..=5    =>     86_400.0,  // 1 dia
            6..=20   =>   7.0*86_400.0,  // 1 semana
            21..=100 =>  30.0*86_400.0,  // 1 mês
            _        => 180.0*86_400.0,  // 6 meses
        };
        let fator = (-dt / meia_vida_s).exp() as f32;
        self.forca *= fator;

        if self.forca < FORCA_MINIMA_ARQUIVO {
            self.estado = EstadoTemplate::Arquivado;
        }
    }

    /// Preenche slots com o mapa fornecido `{ indice → conteúdo }`.
    /// Retorna `true` se todos os slots obrigatórios foram preenchidos (uso completo).
    pub fn preencher(&mut self, valores: &HashMap<usize, String>) -> bool {
        for (idx, conteudo) in valores {
            if let Some(slot) = self.slots.get_mut(*idx) {
                slot.conteudo_atual = Some(conteudo.clone());
            }
        }
        self.slots.iter()
            .filter(|s| s.obrigatorio)
            .all(|s| s.conteudo_atual.is_some())
    }

    /// Renderiza o template preenchido como sequência de tokens.
    /// Segue a ordem das relações sequenciais; fallback: ordem dos slots.
    pub fn renderizar(&self) -> Vec<String> {
        // Tenta ordenar slots pela topologia sequencial
        let mut ordem: Vec<usize> = Vec::new();
        let seq: Vec<(usize, usize)> = self.relacoes.iter()
            .filter(|r| r.tipo == TipoRelacao::Sequencial)
            .map(|r| (r.de, r.para))
            .collect();

        if seq.is_empty() {
            // Sem relações sequenciais: usa ordem natural dos slots
            ordem = (0..self.slots.len()).collect();
        } else {
            // Topo da cadeia sequencial = nó sem entrada
            let destinos: std::collections::HashSet<usize> = seq.iter().map(|(_, d)| *d).collect();
            let raizes: Vec<usize> = seq.iter()
                .map(|(o, _)| *o)
                .filter(|o| !destinos.contains(o))
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            let inicio = raizes.first().copied().unwrap_or(0);
            ordem.push(inicio);
            let mut atual = inicio;
            for _ in 0..self.slots.len() {
                if let Some((_, prox)) = seq.iter().find(|(de, _)| *de == atual) {
                    if ordem.contains(prox) { break; }
                    ordem.push(*prox);
                    atual = *prox;
                } else {
                    break;
                }
            }
            // Slots não na cadeia (ex: paralelos) no final
            for i in 0..self.slots.len() {
                if !ordem.contains(&i) { ordem.push(i); }
            }
        }

        ordem.iter()
            .filter_map(|&i| self.slots.get(i))
            .filter_map(|s| s.conteudo_atual.clone())
            .filter(|c| !c.is_empty())
            .collect()
    }

    /// Limpa todos os slots (chamado após o uso — conteúdo é efêmero).
    pub fn limpar_slots(&mut self) {
        for slot in &mut self.slots { slot.limpar(); }
    }

    /// Registra no histórico de cada slot o conteúdo usado.
    /// Só atualiza se a plasticidade permitir.
    pub fn registrar_historico(&mut self, score: f32) {
        if self.plasticidade() < 0.05 { return; }
        for slot in &mut self.slots {
            if let Some(c) = slot.conteudo_atual.clone() {
                if slot.historico.len() >= MAX_HISTORICO_SLOT {
                    slot.historico.remove(0);
                }
                slot.historico.push((c, score));
            }
        }
    }

    /// Gera um template filho especializado com uma tag adicional.
    /// Só produz filho se o template estiver suficientemente consolidado
    /// (plasticidade baixa = estrutura estável o suficiente para derivar).
    pub fn gerar_filho(&self, tag_especializacao: String) -> Option<Template> {
        if self.plasticidade() > 0.5 {
            return None; // muito plástico — modifica-se diretamente, não gera filho
        }
        let mut filho = self.clone();
        filho.id            = Uuid::new_v4();
        filho.pai           = Some(self.id);
        filho.estado        = EstadoTemplate::Nascente;
        filho.n_validacoes  = 0;
        filho.n_uso_completo = 0;
        filho.n_uso_parcial  = 0;
        filho.n_uso_empilhado = 0;
        filho.forca         = 0.5;
        filho.sub_templates = Vec::new();
        filho.tags.push(tag_especializacao);
        // Filho começa com histórico dos slots limpo — aprende do zero
        for slot in &mut filho.slots { slot.historico.clear(); }
        Some(filho)
    }

    /// Quantos slots obrigatórios ainda estão vazios.
    pub fn slots_vazios_obrigatorios(&self) -> usize {
        self.slots.iter()
            .filter(|s| s.obrigatorio && s.conteudo_atual.is_none())
            .count()
    }

    pub fn atualizar_estado(&mut self) {
        if self.forca < FORCA_MINIMA_ARQUIVO {
            self.estado = EstadoTemplate::Arquivado;
            return;
        }
        self.estado = match self.n_validacoes {
            n if n < VALIDACOES_DESENVOLVENDO => EstadoTemplate::Nascente,
            n if n < VALIDACOES_CONSOLIDADO   => EstadoTemplate::Desenvolvendo,
            n if n < VALIDACOES_AUTOMATICO    => EstadoTemplate::Consolidado,
            _                                  => EstadoTemplate::Automatico,
        };
    }
}

// ── TemplateStore ─────────────────────────────────────────────────────────────

/// Repositório central de templates cognitivos da Selene.
/// Gerencia ciclo de vida, composição, decaimento e reconhecimento de padrões.
pub struct TemplateStore {
    pub templates: HashMap<Uuid, Template>,

    /// Índice por domínio para lookup rápido.
    por_dominio: HashMap<String, Vec<Uuid>>,

    /// Histórico de pilhas recentes: detecta combinações recorrentes.
    pilhas_recentes: VecDeque<Vec<Uuid>>,

    /// Contador de combinações pares: (A, B) → quantas vezes usados juntos.
    combinacoes: HashMap<(Uuid, Uuid), u32>,

    /// Timestamp do último decay global (para não decair em todo tick).
    ultimo_decay_s: f64,
}

impl TemplateStore {
    pub fn novo() -> Self {
        let mut store = Self {
            templates:      HashMap::new(),
            por_dominio:    HashMap::new(),
            pilhas_recentes: VecDeque::with_capacity(MAX_PILHAS_HIST),
            combinacoes:    HashMap::new(),
            ultimo_decay_s: 0.0,
        };
        // Carrega templates base ao inicializar
        for t in templates_base() {
            store.registrar(t);
        }
        store
    }

    /// Registra um template no repositório.
    pub fn registrar(&mut self, template: Template) -> Uuid {
        let id           = template.id;
        let dominio_str  = template.dominio.como_str();
        self.por_dominio.entry(dominio_str).or_default().push(id);
        self.templates.insert(id, template);
        id
    }

    /// Usa um template simples (não empilhado).
    /// Preenche slots, renderiza, registra histórico e reforça.
    /// Retorna os tokens renderizados + se foi uso completo.
    pub fn usar(
        &mut self,
        id: Uuid,
        valores: &HashMap<usize, String>,
        validado: bool,
        t_atual_s: f64,
    ) -> Option<(Vec<String>, TipoUso)> {
        let template = self.templates.get_mut(&id)?;
        let completo = template.preencher(valores);
        let tipo = if completo { TipoUso::Completo } else { TipoUso::Parcial };
        let tokens = template.renderizar();
        let score  = if validado { 1.0 } else { 0.0 };
        template.registrar_historico(score);
        template.reforcar(tipo.clone(), validado, t_atual_s);
        template.limpar_slots();
        Some((tokens, tipo))
    }

    /// Registra uso de uma sequência (pilha) de templates.
    /// Se A+B aparecerem juntos ≥ MIN_COMBINACOES_CONSOLIDA vezes,
    /// gera automaticamente um template composto.
    /// Retorna o UUID do composto gerado (se acontecer).
    pub fn registrar_pilha(
        &mut self,
        ids: Vec<Uuid>,
        validado: bool,
        t_atual_s: f64,
    ) -> Option<Uuid> {
        if ids.len() < 2 { return None; }

        // Reforça cada template individualmente
        for &id in &ids {
            if let Some(t) = self.templates.get_mut(&id) {
                t.reforcar(TipoUso::Empilhado, validado, t_atual_s);
            }
        }

        if !validado { return None; }

        // Detecta e contabiliza pares consecutivos
        let mut novo_composto: Option<Uuid> = None;
        for w in ids.windows(2) {
            let chave = (w[0], w[1]);
            let cnt   = self.combinacoes.entry(chave).or_insert(0);
            *cnt += 1;

            if *cnt == MIN_COMBINACOES_CONSOLIDA {
                if let (Some(a), Some(b)) = (
                    self.templates.get(&w[0]).cloned(),
                    self.templates.get(&w[1]).cloned(),
                ) {
                    let composto = Self::compor_dois(&a, &b);
                    let cid = composto.id;
                    self.registrar(composto);
                    novo_composto = Some(cid);
                    println!(
                        "[TEMPLATE] Novo template composto gerado \
                         (domínio: {}) a partir de {} usos combinados.",
                        self.templates[&cid].dominio.como_str(),
                        MIN_COMBINACOES_CONSOLIDA
                    );
                }
            }
        }

        // Guarda a pilha no histórico
        if self.pilhas_recentes.len() >= MAX_PILHAS_HIST {
            self.pilhas_recentes.pop_front();
        }
        self.pilhas_recentes.push_back(ids);

        novo_composto
    }

    /// Reconhece quais templates melhor se encaixam com um conjunto de conceitos.
    /// Retorna lista ordenada por score (sobreposição histórica × força).
    pub fn reconhecer(&self, conceitos: &[String]) -> Vec<(Uuid, f32)> {
        let cset: std::collections::HashSet<&str> = conceitos.iter()
            .map(|s| s.as_str())
            .collect();

        let mut scores: Vec<(Uuid, f32)> = self.templates.iter()
            .filter(|(_, t)| t.estado != EstadoTemplate::Arquivado)
            .map(|(&id, t)| {
                let overlap = t.slots.iter()
                    .flat_map(|s| &s.historico)
                    .filter(|(c, _)| cset.contains(c.as_str()))
                    .count() as f32;
                let score = overlap * 0.6 + t.forca * 0.4;
                (id, score)
            })
            .filter(|(_, s)| *s > 0.0)
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Tenta gerar um filho especializado de um template consolidado.
    pub fn especializar(
        &mut self,
        id: Uuid,
        tag: String,
    ) -> Option<Uuid> {
        let filho = self.templates.get(&id)?.gerar_filho(tag)?;
        let fid   = filho.id;
        self.registrar(filho);
        Some(fid)
    }

    /// Decaimento global — chamar a cada ~3600 ticks (poucos Hz, escala de segundos).
    pub fn tick_decay(&mut self, t_atual_s: f64) {
        // Evita decair mais de uma vez por segundo
        if t_atual_s - self.ultimo_decay_s < 1.0 { return; }
        self.ultimo_decay_s = t_atual_s;

        for t in self.templates.values_mut() {
            t.decair(t_atual_s);
        }

        // Poda: remove templates arquivados com validações insuficientes
        self.templates.retain(|_, t| {
            t.estado != EstadoTemplate::Arquivado || t.n_validacoes >= VALIDACOES_CONSOLIDADO
        });

        // Poda extra: se acima do limite, remove os mais fracos (exceto consolidados+)
        if self.templates.len() > MAX_TEMPLATES {
            let mut candidatos: Vec<(Uuid, f32)> = self.templates.iter()
                .filter(|(_, t)| (t.estado == EstadoTemplate::Nascente
                               || t.estado == EstadoTemplate::Desenvolvendo)
                              && t.n_validacoes < VALIDACOES_CONSOLIDADO)
                .map(|(&id, t)| (id, t.forca))
                .collect();
            candidatos.sort_by(|a, b| a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal));
            let n_remover = self.templates.len() - MAX_TEMPLATES;
            for (id, _) in candidatos.into_iter().take(n_remover) {
                self.templates.remove(&id);
            }
        }

        // Reconstrói índice de domínio
        self.reconstruir_indice();
    }

    // ── Estatísticas ──────────────────────────────────────────────────────────

    pub fn total(&self) -> usize { self.templates.len() }

    pub fn por_estado(&self) -> (usize, usize, usize, usize, usize) {
        let mut n = (0usize, 0, 0, 0, 0);
        for t in self.templates.values() {
            match t.estado {
                EstadoTemplate::Nascente      => n.0 += 1,
                EstadoTemplate::Desenvolvendo => n.1 += 1,
                EstadoTemplate::Consolidado   => n.2 += 1,
                EstadoTemplate::Automatico    => n.3 += 1,
                EstadoTemplate::Arquivado     => n.4 += 1,
            }
        }
        n
    }

    pub fn por_dominio_str(&self, dominio: &str) -> Vec<&Template> {
        self.por_dominio.get(dominio)
            .map(|ids| ids.iter()
                .filter_map(|id| self.templates.get(id))
                .collect())
            .unwrap_or_default()
    }

    // ── Helpers privados ──────────────────────────────────────────────────────

    /// Compõe dois templates em um novo template composto.
    /// Os slots externos tornam-se os slots do composto.
    /// Uma relação sequencial liga o último slot de A ao primeiro de B.
    fn compor_dois(a: &Template, b: &Template) -> Template {
        let offset = a.slots.len();

        let mut slots: Vec<Slot> = a.slots.clone();
        for s in &b.slots {
            let mut s2    = s.clone();
            s2.indice    += offset;
            slots.push(s2);
        }

        let mut relacoes: Vec<Relacao> = a.relacoes.clone();
        for r in &b.relacoes {
            relacoes.push(Relacao {
                de:    r.de + offset,
                para:  r.para + offset,
                tipo:  r.tipo.clone(),
                forca: r.forca,
            });
        }
        // Ligação entre A e B
        if !a.slots.is_empty() && !b.slots.is_empty() {
            relacoes.push(Relacao {
                de:    a.slots.len() - 1,
                para:  offset,
                tipo:  TipoRelacao::Sequencial,
                forca: 0.8,
            });
        }

        let dominio = Dominio::Composto(vec![
            a.dominio.como_str(),
            b.dominio.como_str(),
        ]);
        let mut t = Template::novo(dominio, slots, relacoes);
        t.sub_templates = vec![a.id, b.id];
        t.complexidade  = a.complexidade + b.complexidade + 1;
        t
    }

    fn reconstruir_indice(&mut self) {
        self.por_dominio.clear();
        for (&id, t) in &self.templates {
            self.por_dominio
                .entry(t.dominio.como_str())
                .or_default()
                .push(id);
        }
    }
}

// ── Templates base ────────────────────────────────────────────────────────────
// Carregados automaticamente no TemplateStore::novo().
// Começam já no estado Consolidado/Automático para que a Selene possa
// usá-los imediatamente sem precisar de treino inicial.

pub fn templates_base() -> Vec<Template> {
    vec![
        // ── Linguagem ─────────────────────────────────────────────────────
        {
            // "X é/parece Y" — observação atributiva simples
            let slots = vec![
                Slot::novo(0, TipoSlot::Conceito,  true),
                Slot::novo(1, TipoSlot::Atributo,  true),
            ];
            let relacoes = vec![
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Atributiva, forca: 0.9 },
            ];
            let mut t = Template::novo(Dominio::Linguagem, slots, relacoes);
            t.nome         = Some("observacao_atributiva".into());
            t.n_validacoes = 25; t.forca = 0.90; t.atualizar_estado(); t
        },
        {
            // "X causa/leva a Y" — relação causal bidirecional
            let slots = vec![
                Slot::novo(0, TipoSlot::Conceito, true),
                Slot::novo(1, TipoSlot::Acao,     false),
                Slot::novo(2, TipoSlot::Conceito, true),
            ];
            let relacoes = vec![
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Sequencial, forca: 0.8 },
                Relacao { de: 1, para: 2, tipo: TipoRelacao::Sequencial, forca: 0.8 },
                Relacao { de: 0, para: 2, tipo: TipoRelacao::Causal,     forca: 0.9 },
            ];
            let mut t = Template::novo(Dominio::Linguagem, slots, relacoes);
            t.nome         = Some("relacao_causal".into());
            t.n_validacoes = 30; t.forca = 0.92; t.atualizar_estado(); t
        },
        {
            // "X me faz pensar em Y e Z" — associação com extensão
            let slots = vec![
                Slot::novo(0, TipoSlot::Conceito, true),
                Slot::novo(1, TipoSlot::Conceito, true),
                Slot::novo(2, TipoSlot::Conceito, false),
            ];
            let relacoes = vec![
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Atributiva,  forca: 0.8 },
                Relacao { de: 1, para: 2, tipo: TipoRelacao::Sequencial,  forca: 0.6 },
            ];
            let mut t = Template::novo(Dominio::Linguagem, slots, relacoes);
            t.nome         = Some("associacao_dupla".into());
            t.n_validacoes = 20; t.forca = 0.85; t.atualizar_estado(); t
        },
        {
            // "Interessante que X. Y parece Y2." — reflexão com extensão
            let slots = vec![
                Slot::novo(0, TipoSlot::Conceito, true),
                Slot::novo(1, TipoSlot::Conceito, false),
                Slot::novo(2, TipoSlot::Atributo, false),
            ];
            let relacoes = vec![
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Causal,    forca: 0.7 },
                Relacao { de: 1, para: 2, tipo: TipoRelacao::Atributiva, forca: 0.7 },
            ];
            let mut t = Template::novo(Dominio::Linguagem, slots, relacoes);
            t.nome         = Some("reflexao_expandida".into());
            t.n_validacoes = 15; t.forca = 0.80; t.atualizar_estado(); t
        },

        // ── Causal ────────────────────────────────────────────────────────
        {
            // A → B → C (cadeia causal de 3 elos)
            let slots = vec![
                Slot::novo(0, TipoSlot::Conceito, true),
                Slot::novo(1, TipoSlot::Conceito, true),
                Slot::novo(2, TipoSlot::Conceito, false),
            ];
            let relacoes = vec![
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Causal,    forca: 0.9 },
                Relacao { de: 1, para: 2, tipo: TipoRelacao::Causal,    forca: 0.9 },
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Sequencial, forca: 0.8 },
                Relacao { de: 1, para: 2, tipo: TipoRelacao::Sequencial, forca: 0.8 },
            ];
            let mut t = Template::novo(Dominio::Causal, slots, relacoes);
            t.nome         = Some("cadeia_causal".into());
            t.n_validacoes = 10; t.forca = 0.78; t.atualizar_estado(); t
        },

        // ── Lógica ────────────────────────────────────────────────────────
        {
            // Se A então B (condicional básico)
            let slots = vec![
                Slot::novo(0, TipoSlot::Conceito, true),
                Slot::novo(1, TipoSlot::Conceito, true),
            ];
            let relacoes = vec![
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Causal, forca: 0.95 },
            ];
            let mut t = Template::novo(Dominio::Logica, slots, relacoes);
            t.nome         = Some("se_entao".into());
            t.n_validacoes = 10; t.forca = 0.82; t.atualizar_estado(); t
        },
        {
            // A implica B; B implica C → A implica C (transitividade)
            let slots = vec![
                Slot::novo(0, TipoSlot::Conceito, true),
                Slot::novo(1, TipoSlot::Conceito, true),
                Slot::novo(2, TipoSlot::Conceito, true),
            ];
            let relacoes = vec![
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Causal, forca: 1.0 },
                Relacao { de: 1, para: 2, tipo: TipoRelacao::Causal, forca: 1.0 },
                Relacao { de: 0, para: 2, tipo: TipoRelacao::Causal, forca: 0.9 },
            ];
            let mut t = Template::novo(Dominio::Logica, slots, relacoes);
            t.nome         = Some("transitividade".into());
            t.n_validacoes = 8; t.forca = 0.75; t.atualizar_estado(); t
        },

        // ── Matemática ────────────────────────────────────────────────────
        {
            // R = A × B (lei linear de produto: F=ma, V=IR, P=Fv ...)
            let slots = vec![
                Slot::novo(0, TipoSlot::Quantidade, true),   // resultado
                Slot::novo(1, TipoSlot::Quantidade, true),   // fator A
                Slot::novo(2, TipoSlot::Quantidade, true),   // fator B
            ];
            let relacoes = vec![
                Relacao { de: 1, para: 0, tipo: TipoRelacao::Matematica, forca: 1.0 },
                Relacao { de: 2, para: 0, tipo: TipoRelacao::Matematica, forca: 1.0 },
            ];
            let mut t = Template::novo(Dominio::Matematica, slots, relacoes);
            t.nome         = Some("lei_produto_linear".into());
            t.n_validacoes = 5; t.forca = 0.72; t.atualizar_estado(); t
        },
        {
            // R = A / B (razão: velocidade=d/t, densidade=m/v ...)
            let slots = vec![
                Slot::novo(0, TipoSlot::Quantidade, true),
                Slot::novo(1, TipoSlot::Quantidade, true),
                Slot::novo(2, TipoSlot::Quantidade, true),
            ];
            let relacoes = vec![
                Relacao { de: 1, para: 0, tipo: TipoRelacao::Matematica, forca: 1.0 },
                Relacao { de: 0, para: 2, tipo: TipoRelacao::Matematica, forca: 1.0 },
            ];
            let mut t = Template::novo(Dominio::Matematica, slots, relacoes);
            t.nome         = Some("lei_razao".into());
            t.n_validacoes = 5; t.forca = 0.70; t.atualizar_estado(); t
        },
        {
            // R = A × B^n (lei potência: E=mc², F=kx² ...)
            let slots = vec![
                Slot::novo(0, TipoSlot::Quantidade, true),
                Slot::novo(1, TipoSlot::Quantidade, true),
                Slot::novo(2, TipoSlot::Quantidade, true),
                Slot::novo(3, TipoSlot::Quantidade, false),  // expoente (opcional)
            ];
            let relacoes = vec![
                Relacao { de: 1, para: 0, tipo: TipoRelacao::Matematica, forca: 1.0 },
                Relacao { de: 2, para: 0, tipo: TipoRelacao::Matematica, forca: 1.0 },
                Relacao { de: 3, para: 2, tipo: TipoRelacao::Matematica, forca: 0.8 },
            ];
            let mut t = Template::novo(Dominio::Matematica, slots, relacoes);
            t.nome         = Some("lei_potencia".into());
            t.n_validacoes = 3; t.forca = 0.65; t.atualizar_estado(); t
        },
        {
            // A aumenta → B aumenta (proporção direta qualitativa)
            let slots = vec![
                Slot::novo(0, TipoSlot::Quantidade, true),
                Slot::novo(1, TipoSlot::Quantidade, true),
            ];
            let relacoes = vec![
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Causal, forca: 0.85 },
            ];
            let mut t = Template::novo(Dominio::Matematica, slots, relacoes);
            t.nome         = Some("proporcao_direta".into());
            t.n_validacoes = 5; t.forca = 0.70; t.atualizar_estado(); t
        },

        // ── Fala Conversacional ───────────────────────────────────────────────
        {
            // "Olá / oi / bom dia" — abertura de canal social
            let slots = vec![
                Slot::novo(0, TipoSlot::Acao,     true),  // ato de saudar
                Slot::novo(1, TipoSlot::Conceito, false), // interlocutor (opcional)
            ];
            let relacoes = vec![
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Atributiva, forca: 0.9 },
            ];
            let mut t = Template::novo(Dominio::Linguagem, slots, relacoes);
            t.nome         = Some("saudacao".into());
            t.n_validacoes = 50; t.forca = 0.95; t.atualizar_estado(); t
        },
        {
            // "Como você está? / Tudo bem?" — sondagem de estado do outro
            let slots = vec![
                Slot::novo(0, TipoSlot::Acao,     true),  // ato de perguntar
                Slot::novo(1, TipoSlot::Atributo, true),  // estado sondado
                Slot::novo(2, TipoSlot::Conceito, false), // interlocutor
            ];
            let relacoes = vec![
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Causal,    forca: 0.8 },
                Relacao { de: 0, para: 2, tipo: TipoRelacao::Atributiva, forca: 0.7 },
            ];
            let mut t = Template::novo(Dominio::Linguagem, slots, relacoes);
            t.nome         = Some("pergunta_estado".into());
            t.n_validacoes = 40; t.forca = 0.90; t.atualizar_estado(); t
        },
        {
            // "Sim / entendo / concordo" — confirmação e alinhamento
            let slots = vec![
                Slot::novo(0, TipoSlot::Acao,     true),  // ato de confirmar
                Slot::novo(1, TipoSlot::Conceito, true),  // o que é confirmado
            ];
            let relacoes = vec![
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Atributiva, forca: 0.95 },
            ];
            let mut t = Template::novo(Dominio::Linguagem, slots, relacoes);
            t.nome         = Some("confirmacao".into());
            t.n_validacoes = 60; t.forca = 0.96; t.atualizar_estado(); t
        },
        {
            // "Não tenho certeza / talvez / pode ser" — expressão de incerteza
            let slots = vec![
                Slot::novo(0, TipoSlot::Atributo, true),  // grau de incerteza
                Slot::novo(1, TipoSlot::Conceito, true),  // o que é incerto
                Slot::novo(2, TipoSlot::Conceito, false), // alternativa (opcional)
            ];
            let relacoes = vec![
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Atributiva, forca: 0.85 },
                Relacao { de: 1, para: 2, tipo: TipoRelacao::Causal,     forca: 0.5 },
            ];
            let mut t = Template::novo(Dominio::Linguagem, slots, relacoes);
            t.nome         = Some("expressao_incerteza".into());
            t.n_validacoes = 30; t.forca = 0.88; t.atualizar_estado(); t
        },
        {
            // "Que interessante! / Não sabia disso!" — expressão de surpresa/curiosidade
            let slots = vec![
                Slot::novo(0, TipoSlot::Atributo, true),  // qualidade emocional (surpresa)
                Slot::novo(1, TipoSlot::Conceito, true),  // o que surpreende
            ];
            let relacoes = vec![
                Relacao { de: 1, para: 0, tipo: TipoRelacao::Causal, forca: 0.9 },
            ];
            let mut t = Template::novo(Dominio::Linguagem, slots, relacoes);
            t.nome         = Some("reacao_curiosidade".into());
            t.n_validacoes = 25; t.forca = 0.87; t.atualizar_estado(); t
        },
        {
            // "Eu acho que X é Y" — opinião pessoal modulada por emoção
            let slots = vec![
                Slot::novo(0, TipoSlot::Conceito, true),  // perspectiva (eu/minha visão)
                Slot::novo(1, TipoSlot::Conceito, true),  // o sujeito da opinião
                Slot::novo(2, TipoSlot::Atributo, true),  // a qualidade atribuída
            ];
            let relacoes = vec![
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Atributiva, forca: 0.8 },
                Relacao { de: 1, para: 2, tipo: TipoRelacao::Atributiva, forca: 0.9 },
                Relacao { de: 0, para: 2, tipo: TipoRelacao::Causal,     forca: 0.6 },
            ];
            let mut t = Template::novo(Dominio::Linguagem, slots, relacoes);
            t.nome         = Some("opiniao_propria".into());
            t.n_validacoes = 20; t.forca = 0.83; t.atualizar_estado(); t
        },
        {
            // "Você sabe o que é X?" — pergunta de esclarecimento
            let slots = vec![
                Slot::novo(0, TipoSlot::Acao,     true),  // ato de perguntar
                Slot::novo(1, TipoSlot::Conceito, true),  // conceito que falta
            ];
            let relacoes = vec![
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Causal, forca: 0.85 },
            ];
            let mut t = Template::novo(Dominio::Linguagem, slots, relacoes);
            t.nome         = Some("pergunta_esclarecimento".into());
            t.n_validacoes = 15; t.forca = 0.80; t.atualizar_estado(); t
        },
        {
            // "Entendo o que você quer dizer" — espelhamento empático
            let slots = vec![
                Slot::novo(0, TipoSlot::Acao,     true),  // ato de compreender
                Slot::novo(1, TipoSlot::Conceito, true),  // intenção percebida
                Slot::novo(2, TipoSlot::Atributo, false), // validação emocional (opcional)
            ];
            let relacoes = vec![
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Atributiva,  forca: 0.9 },
                Relacao { de: 1, para: 2, tipo: TipoRelacao::Atributiva,  forca: 0.7 },
                Relacao { de: 0, para: 2, tipo: TipoRelacao::Composicional, forca: 0.6 },
            ];
            let mut t = Template::novo(Dominio::Linguagem, slots, relacoes);
            t.nome         = Some("espelhamento_empatico".into());
            t.n_validacoes = 20; t.forca = 0.85; t.atualizar_estado(); t
        },
        {
            // "Até logo / tchau / até mais" — fechamento de canal social
            let slots = vec![
                Slot::novo(0, TipoSlot::Acao,     true),  // ato de despedir
                Slot::novo(1, TipoSlot::Conceito, false), // interlocutor (opcional)
            ];
            let relacoes = vec![
                Relacao { de: 0, para: 1, tipo: TipoRelacao::Atributiva, forca: 0.85 },
            ];
            let mut t = Template::novo(Dominio::Linguagem, slots, relacoes);
            t.nome         = Some("despedida".into());
            t.n_validacoes = 45; t.forca = 0.93; t.atualizar_estado(); t
        },
    ]
}

// ── Testes ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ciclo_vida_template() {
        let slots = vec![
            Slot::novo(0, TipoSlot::Conceito, true),
            Slot::novo(1, TipoSlot::Atributo, true),
        ];
        let mut t = Template::novo(Dominio::Linguagem, slots, vec![]);
        assert_eq!(t.estado, EstadoTemplate::Nascente);
        assert!((t.plasticidade() - 1.0).abs() < 0.01);

        for _ in 0..VALIDACOES_DESENVOLVENDO {
            t.reforcar(TipoUso::Completo, true, 1000.0);
        }
        assert_eq!(t.estado, EstadoTemplate::Desenvolvendo);

        for _ in 0..(VALIDACOES_CONSOLIDADO - VALIDACOES_DESENVOLVENDO) {
            t.reforcar(TipoUso::Completo, true, 1000.0);
        }
        assert_eq!(t.estado, EstadoTemplate::Consolidado);
        assert!(t.plasticidade() < 0.5);
    }

    #[test]
    fn preenchimento_efemero() {
        let slots = vec![
            Slot::novo(0, TipoSlot::Conceito, true),
            Slot::novo(1, TipoSlot::Conceito, true),
        ];
        let mut t = Template::novo(Dominio::Linguagem, slots, vec![]);

        let mut vals = HashMap::new();
        vals.insert(0, "calor".to_string());
        vals.insert(1, "fogo".to_string());

        assert!(t.preencher(&vals));
        let tokens = t.renderizar();
        assert_eq!(tokens, vec!["calor", "fogo"]);

        t.limpar_slots();
        assert!(t.slots[0].conteudo_atual.is_none());
        assert!(t.slots[1].conteudo_atual.is_none());
    }

    #[test]
    fn gerar_filho_consolidado() {
        let slots = vec![Slot::novo(0, TipoSlot::Conceito, true)];
        let mut t = Template::novo(Dominio::Linguagem, slots, vec![]);
        // Antes de consolidar: não gera filho
        assert!(t.gerar_filho("fisica".into()).is_none());
        // Consolida
        for _ in 0..VALIDACOES_CONSOLIDADO {
            t.reforcar(TipoUso::Completo, true, 1.0);
        }
        let filho = t.gerar_filho("fisica".into());
        assert!(filho.is_some());
        assert_eq!(filho.unwrap().estado, EstadoTemplate::Nascente);
    }

    #[test]
    fn store_consolidacao_pilha() {
        let mut store = TemplateStore::novo();
        let t1 = Template::novo(
            Dominio::Linguagem,
            vec![Slot::novo(0, TipoSlot::Conceito, true)],
            vec![],
        );
        let t2 = Template::novo(
            Dominio::Causal,
            vec![Slot::novo(0, TipoSlot::Conceito, true)],
            vec![],
        );
        let id1 = store.registrar(t1);
        let id2 = store.registrar(t2);

        let total_antes = store.total();

        // Usa a pilha MIN_COMBINACOES_CONSOLIDA vezes
        let mut novo_id: Option<Uuid> = None;
        for i in 0..MIN_COMBINACOES_CONSOLIDA {
            novo_id = store.registrar_pilha(
                vec![id1, id2], true, (i * 10) as f64
            );
        }
        // Na última iteração deve ter gerado um composto
        assert!(novo_id.is_some(), "deveria ter gerado template composto");
        assert!(store.total() > total_antes);
    }

    #[test]
    fn templates_base_carregados() {
        let store = TemplateStore::novo();
        assert!(store.total() >= 20, "deve haver pelo menos 20 templates base (cognitivos + fala)");
        let linguagem = store.por_dominio_str("linguagem");
        assert!(linguagem.len() >= 13,
            "esperava pelo menos 13 templates de linguagem, got {}", linguagem.len());
    }

    #[test]
    fn templates_fala_conversacional_presentes() {
        let base = templates_base();
        let nomes: Vec<&str> = base.iter()
            .filter_map(|t| t.nome.as_deref())
            .collect();

        let esperados = [
            "saudacao",
            "pergunta_estado",
            "confirmacao",
            "expressao_incerteza",
            "reacao_curiosidade",
            "opiniao_propria",
            "pergunta_esclarecimento",
            "espelhamento_empatico",
            "despedida",
        ];
        for nome in &esperados {
            assert!(nomes.contains(nome), "template '{}' não encontrado", nome);
        }
    }

    #[test]
    fn templates_fala_comecam_consolidados() {
        let base = templates_base();
        let fala = ["saudacao", "confirmacao", "despedida"];
        for t in &base {
            if let Some(nome) = &t.nome {
                if fala.contains(&nome.as_str()) {
                    assert!(
                        matches!(t.estado, EstadoTemplate::Consolidado | EstadoTemplate::Automatico),
                        "template '{}' deveria começar Consolidado ou Automático, está {:?}",
                        nome, t.estado
                    );
                    assert!(t.forca >= 0.85,
                        "template '{}' deveria ter força >= 0.85, tem {:.2}", nome, t.forca);
                }
            }
        }
    }
}
