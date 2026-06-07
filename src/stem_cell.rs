// =============================================================================
// src/stem_cell.rs — Selene V4.6 — Célula-Tronco Neural Digital
// =============================================================================
//
// NEUROGÊNESE + NEUROEVOLUÇÃO AUTÔNOMA.
//
// Em vez de mapear os milhares de tipos/subtipos neuronais reais (impossível
// hoje), o sistema DESCOBRE os tipos de que precisa: uma célula-tronco nasce
// indiferenciada, sente a necessidade local da zona cerebral, diferencia o seu
// genoma (DnaNeuronal) nessa direção, é implantada em PERÍODO DE PROVA e então
// julga-se sozinha — é ACEITE (vira espécie registrada) ou REJEITADA (apoptose).
//
// Tudo acontece APENAS DURANTE O SONO (alinhado com a neurogênese hipocampal
// real, que ocorre no sono): a vigília é observação; o sono é seleção/poda
// offline, sem custo no loop 200 Hz.
//
// CICLO DE VIDA:
//   Indiferenciada → Diferenciada → EmProva → {Aceite | Rejeitada}
//
// SEGURANÇA (orçamento + estabilidade):
//   • Cap populacional por zona (não cresce sem limite → protege a RAM).
//   • Implante com peso baixo (não domina a rede durante a prova).
//   • Rejeição reusa o ciclo Dormant→Swapped já existente (apoptose barata).
//
// REFERÊNCIAS:
//   Eriksson et al. (1998)  — neurogênese no hipocampo adulto humano
//   Aimone et al. (2014)    — função computacional de neurônios recém-nascidos
//   Stanley & Miikkulainen (2002) — NEAT: evolução de topologias neurais
// =============================================================================

use serde::{Deserialize, Serialize};
use rand::Rng;

use crate::synaptic_core::{
    CamadaHibrida, DnaNeuronal, NeuronalStatus, NeuronioHibrido, PesoNeuronio,
    PrecisionType, TipoNeuronal, gerar_especie_hibrida,
};

// ─────────────────────────────────────────────────────────────────────────────
// PARÂMETROS DE NEUROGÊNESE
// ─────────────────────────────────────────────────────────────────────────────

/// Banda de atividade média "saudável" de um neurônio [spike-prob por tick].
/// Abaixo → silencioso (inútil); acima → runaway (desestabiliza).
/// 0.005 ≈ 5 Hz (mínimo útil) · 0.55 ≈ disparo quase contínuo (patológico).
const ATIVIDADE_VIAVEL_MIN: f32 = 0.005;
const ATIVIDADE_VIAVEL_MAX: f32 = 0.55;

/// Atividade média da CAMADA acima da qual há excesso de excitação (falta inibição).
const RUNAWAY_CAMADA: f32 = 0.22;

/// Fração de neurônios Dormant acima da qual a zona precisa de novos drivers.
const SILENCIO_CAMADA: f32 = 0.40;

/// Peso inicial baixo de um neurônio em prova (não domina a rede).
const PESO_PROVA: f32 = 0.15;

/// Peso de um neurônio aceito (entra em pé de igualdade).
const PESO_ACEITE: f32 = 1.0;

// ─────────────────────────────────────────────────────────────────────────────
// NECESSIDADE DA ZONA
// ─────────────────────────────────────────────────────────────────────────────

/// Diagnóstico do que falta numa zona cerebral — guia a diferenciação.
#[derive(Debug, Clone, Copy)]
pub struct NecessidadeZona {
    /// Atividade média da camada (spike-prob por tick).
    pub atividade_media: f32,
    /// Fração de neurônios Dormant (silenciosos).
    pub silenciosos_frac: f32,
    /// Excesso de excitação → precisa de mais inibição (FS/SST/MSN-like).
    pub falta_inibicao: bool,
    /// Zona subutilizada → precisa de mais drivers excitatórios (RS/Mirror-like).
    pub falta_drivers: bool,
}

impl NecessidadeZona {
    /// Avalia a necessidade de uma camada a partir do seu estado atual.
    pub fn avaliar(camada: &CamadaHibrida) -> Self {
        let n = camada.neuronios.len().max(1) as f32;
        let atividade_media =
            camada.neuronios.iter().map(|x| x.activity_avg).sum::<f32>() / n;
        let dormants = camada.neuronios.iter()
            .filter(|x| x.status == NeuronalStatus::Dormant)
            .count() as f32;
        let silenciosos_frac = dormants / n;

        NecessidadeZona {
            atividade_media,
            silenciosos_frac,
            falta_inibicao: atividade_media > RUNAWAY_CAMADA,
            falta_drivers: silenciosos_frac > SILENCIO_CAMADA && atividade_media < 0.05,
        }
    }

    /// Existe alguma necessidade que justifique gerar uma célula-tronco?
    pub fn precisa_intervir(&self) -> bool {
        self.falta_inibicao || self.falta_drivers
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FASE DO CICLO DE VIDA
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FaseTronco {
    /// Recém-nascida: DNA maleável (média da zona + ruído alto).
    Indiferenciada,
    /// DNA ajustado à necessidade; pronta para implante.
    Diferenciada,
    /// Implantada na camada; aguarda julgamento ao fim da prova.
    EmProva,
    /// Sobreviveu à prova → espécie registrada.
    Aceite,
    /// Falhou → apoptose (Dormant + peso zero).
    Rejeitada,
}

// ─────────────────────────────────────────────────────────────────────────────
// REGISTRO DE ESPÉCIES (mapeia a evolução)
// ─────────────────────────────────────────────────────────────────────────────

/// Uma espécie híbrida que FUNCIONOU no sistema — assinatura única + linhagem.
/// É a "assinatura diferente" que permite mapear a evolução e registrar os
/// novos tipos que vingaram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EspecieRegistrada {
    /// Assinatura única, ex.: "HYB-g3-0007".
    pub assinatura: String,
    /// Genoma que define o fenótipo.
    pub dna: DnaNeuronal,
    /// Geração evolutiva em que surgiu.
    pub geracao: u32,
    /// Zona onde nasceu e provou-se útil.
    pub zona_origem: String,
    /// Fitness final no julgamento.
    pub fitness: f32,
    /// Tick (de sono) em que foi aceita.
    pub aceite_em_tick: u64,
    /// Assinaturas dos pais (linhagem) — "puro:RS" se veio de um tipo puro.
    pub pais: (String, String),
}

/// Catálogo persistível de todas as espécies aceitas. Permite reinstanciar
/// fenótipos que já provaram valor (memória evolutiva entre sessões).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct RegistroEspecies {
    pub especies: Vec<EspecieRegistrada>,
    pub geracao_atual: u32,
    proximo_n: u32,
}

impl RegistroEspecies {
    pub fn novo() -> Self { Self::default() }

    /// Gera a próxima assinatura única.
    fn proxima_assinatura(&mut self) -> String {
        let s = format!("HYB-g{}-{:04}", self.geracao_atual, self.proximo_n);
        self.proximo_n += 1;
        s
    }

    /// Avança uma geração evolutiva.
    pub fn nova_geracao(&mut self) {
        self.geracao_atual += 1;
    }

    /// Registra uma espécie aceita.
    pub fn registrar(
        &mut self,
        dna: DnaNeuronal,
        zona: &str,
        fitness: f32,
        tick: u64,
        pais: (String, String),
    ) -> String {
        let assinatura = self.proxima_assinatura();
        self.especies.push(EspecieRegistrada {
            assinatura: assinatura.clone(),
            dna,
            geracao: self.geracao_atual,
            zona_origem: zona.to_string(),
            fitness,
            aceite_em_tick: tick,
            pais,
        });
        assinatura
    }

    pub fn total(&self) -> usize { self.especies.len() }
}

// ─────────────────────────────────────────────────────────────────────────────
// CÉLULA-TRONCO
// ─────────────────────────────────────────────────────────────────────────────

/// Uma célula-tronco neural digital em algum ponto do seu ciclo de vida.
#[derive(Debug, Clone)]
pub struct CelulaTronco {
    pub dna: DnaNeuronal,
    pub fase: FaseTronco,
    pub zona: String,
    pub assinatura: String,
    pub pais: (String, String),
    /// Índice na camada após implante (válido em EmProva/Aceite).
    pub idx_implante: Option<usize>,
    /// Saúde da camada no momento do implante (baseline de fitness).
    pub saude_base: f32,
}

impl CelulaTronco {
    /// FASE 1 — NASCE indiferenciada: genoma = cruzamento de dois tipos presentes
    /// na zona + mutação alta (alta plasticidade inicial).
    pub fn nascer(camada: &CamadaHibrida, zona: &str, assinatura: String) -> Self {
        let mut rng = rand::thread_rng();
        // Amostra dois tipos presentes (ou RS como fallback) para o cruzamento.
        let tipos: Vec<TipoNeuronal> = camada.neuronios.iter().map(|n| n.tipo).collect();
        let pega = |rng: &mut rand::rngs::ThreadRng| -> TipoNeuronal {
            if tipos.is_empty() { TipoNeuronal::RS }
            else { tipos[rng.gen_range(0..tipos.len())] }
        };
        let pa = pega(&mut rng);
        let pb = pega(&mut rng);
        // Mutação alta (0.15) → exploração ampla de fenótipos.
        let dna = gerar_especie_hibrida(&pa, &pb, 0.15);
        CelulaTronco {
            dna,
            fase: FaseTronco::Indiferenciada,
            zona: zona.to_string(),
            assinatura,
            pais: (format!("puro:{pa:?}"), format!("puro:{pb:?}")),
            idx_implante: None,
            saude_base: 0.0,
        }
    }

    /// FASE 2 — DIFERENCIA: empurra o genoma na direção da necessidade da zona.
    /// Não copia um tipo fixo — ajusta genes, gerando um fenótipo sob medida.
    pub fn diferenciar(&mut self, necessidade: &NecessidadeZona) {
        if necessidade.falta_inibicao {
            // Precisa frear a rede: vira GABAérgico com forte adaptação M (MSN/SST-like).
            self.dna.e_inibitorico = true;
            self.dna.g_m = (self.dna.g_m * 1.5 + 3.0).min(10.0);  // adaptação forte
            self.dna.threshold = (self.dna.threshold - 3.0).max(20.0); // dispara fácil
            self.dna.a = (self.dna.a * 1.5).clamp(0.05, 1.0);     // recuperação rápida (FS-like)
        } else if necessidade.falta_drivers {
            // Precisa de drivers: vira excitatório fácil de disparar (RS/Mirror-like).
            self.dna.e_inibitorico = false;
            self.dna.g_nap = (self.dna.g_nap + 0.5).min(4.0);     // excitabilidade persistente
            self.dna.g_m = (self.dna.g_m * 0.6).max(0.0);         // pouca adaptação
            self.dna.threshold = (self.dna.threshold - 2.0).max(20.0);
        }
        self.dna.clampar();
        self.fase = FaseTronco::Diferenciada;
    }

    /// FASE 3 — IMPLANTA na camada como neurônio Hybrid em prova (peso baixo).
    pub fn implantar(&mut self, camada: &mut CamadaHibrida) {
        let saude = saude_camada(camada);
        let id = camada.neuronios.len() as u32;
        let mut neur = NeuronioHibrido::novo_hibrido(id, self.dna.clone(), PrecisionType::FP32);
        // Peso baixo durante a prova → não domina a dinâmica.
        neur.peso = PesoNeuronio::FP32(PESO_PROVA);
        let idx = camada.adicionar_neuronio(neur);
        self.idx_implante = Some(idx);
        self.saude_base = saude;
        self.fase = FaseTronco::EmProva;
    }

    /// FASE 5 — JULGA: calcula fitness e decide aceitar ou rejeitar (apoptose).
    /// `registro` e `tick` são usados para registrar a espécie se aceita.
    /// Devolve `true` se aceita.
    pub fn julgar(
        &mut self,
        camada: &mut CamadaHibrida,
        registro: &mut RegistroEspecies,
        tick: u64,
    ) -> bool {
        let idx = match self.idx_implante {
            Some(i) if i < camada.neuronios.len() => i,
            _ => { self.fase = FaseTronco::Rejeitada; return false; }
        };

        // Fitness = viabilidade da célula + variação de saúde da camada.
        let viab = viabilidade_celula(&camada.neuronios[idx]);
        let saude_agora = saude_camada(camada);
        let delta_saude = saude_agora - self.saude_base;
        let fitness = viab + delta_saude;

        if fitness > 0.0 {
            // ACEITE: peso normal + registra espécie.
            camada.neuronios[idx].peso = PesoNeuronio::FP32(PESO_ACEITE);
            let assinatura = registro.registrar(
                self.dna.clone(), &self.zona, fitness, tick, self.pais.clone(),
            );
            self.assinatura = assinatura;
            self.fase = FaseTronco::Aceite;
            true
        } else {
            // APOPTOSE: reusa o ciclo Dormant→Swapped (evicção existente).
            camada.neuronios[idx].status = NeuronalStatus::Dormant;
            camada.neuronios[idx].peso = PesoNeuronio::FP32(0.0);
            self.fase = FaseTronco::Rejeitada;
            false
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MÉTRICAS DE FITNESS
// ─────────────────────────────────────────────────────────────────────────────

/// Saúde de uma camada ∈ [0,1]: combina fração de neurônios ativos com a
/// proximidade da atividade média a uma banda alvo saudável (~0.10).
pub fn saude_camada(camada: &CamadaHibrida) -> f32 {
    let n = camada.neuronios.len().max(1) as f32;
    let ativos = camada.neuronios.iter()
        .filter(|x| x.status == NeuronalStatus::Active)
        .count() as f32;
    let frac_ativa = ativos / n;
    let media_act = camada.neuronios.iter().map(|x| x.activity_avg).sum::<f32>() / n;
    // Banda alvo centrada em 0.10; cai linearmente ao afastar.
    let banda = (1.0 - ((media_act - 0.10).abs() / 0.10)).clamp(0.0, 1.0);
    0.5 * frac_ativa + 0.5 * banda
}

/// Viabilidade de uma célula individual: +1 se participa numa banda saudável,
/// −1 se ficou silenciosa (inútil) ou em runaway (desestabilizadora).
pub fn viabilidade_celula(n: &NeuronioHibrido) -> f32 {
    if n.activity_avg >= ATIVIDADE_VIAVEL_MIN && n.activity_avg <= ATIVIDADE_VIAVEL_MAX {
        1.0
    } else {
        -1.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GESTOR DE NEUROGÊNESE (orquestra tudo — APENAS no sono)
// ─────────────────────────────────────────────────────────────────────────────

/// Gere a neurogênese de uma ou mais zonas. O caller DEVE invocar `tick_sono`
/// somente durante fases de sono (NREM/REM) — nunca no loop de vigília 200 Hz.
#[derive(Debug)]
pub struct GestorNeurogenese {
    pub registro: RegistroEspecies,
    /// Máximo de células-tronco em prova simultâneas por zona (orçamento → RAM).
    pub cap_por_zona: usize,
    /// Mutação base ao gerar genomas.
    pub taxa_mutacao: f32,
    /// Células implantadas aguardando julgamento no próximo ciclo de sono.
    em_prova: Vec<CelulaTronco>,
    /// Contador de ciclos de sono (usado como "tick" evolutivo).
    pub ciclos_sono: u64,
}

impl GestorNeurogenese {
    pub fn novo(cap_por_zona: usize) -> Self {
        Self {
            registro: RegistroEspecies::novo(),
            cap_por_zona,
            taxa_mutacao: 0.12,
            em_prova: Vec::new(),
            ciclos_sono: 0,
        }
    }

    /// Um ciclo de neurogênese sobre UMA zona. Chame UMA vez por fase de sono.
    ///
    /// 1. JULGA as células que estavam em prova (sobreviveram à vigília anterior).
    /// 2. AVALIA a necessidade atual da zona.
    /// 3. Se há necessidade e orçamento → NASCE, DIFERENCIA e IMPLANTA uma nova.
    ///
    /// Devolve (aceitas, rejeitadas, nascidas) neste ciclo.
    pub fn tick_sono(&mut self, zona: &mut CamadaHibrida) -> (usize, usize, usize) {
        self.ciclos_sono += 1;
        self.registro.nova_geracao();

        // ── FASE 5: julga as células em prova desta zona ──────────────────────
        let mut aceitas = 0;
        let mut rejeitadas = 0;
        let mut restantes = Vec::new();
        let pendentes = std::mem::take(&mut self.em_prova);
        for mut celula in pendentes {
            if celula.zona != zona.nome {
                restantes.push(celula); // pertence a outra zona — preserva
                continue;
            }
            let aceite = celula.julgar(zona, &mut self.registro, self.ciclos_sono);
            if aceite { aceitas += 1; } else { rejeitadas += 1; }
            // Aceitas/rejeitadas saem da fila de prova.
        }
        self.em_prova = restantes;

        // ── FASES 1-3: nasce/diferencia/implanta se a zona precisar ───────────
        let mut nascidas = 0;
        let necessidade = NecessidadeZona::avaliar(zona);
        let em_prova_nesta = self.em_prova.iter().filter(|c| c.zona == zona.nome).count();
        if necessidade.precisa_intervir() && em_prova_nesta < self.cap_por_zona {
            let assinatura = self.registro.proxima_assinatura();
            let mut celula = CelulaTronco::nascer(zona, &zona.nome, assinatura);
            celula.diferenciar(&necessidade);
            celula.implantar(zona);
            self.em_prova.push(celula);
            nascidas += 1;
        }

        (aceitas, rejeitadas, nascidas)
    }

    /// Número de células atualmente em prova.
    pub fn em_prova(&self) -> usize { self.em_prova.len() }
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTES
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod testes {
    use super::*;

    const DT: f32 = 0.001;

    #[test]
    fn registro_gera_assinaturas_unicas() {
        let mut r = RegistroEspecies::novo();
        let dna = TipoNeuronal::RS.extrair_dna();
        let a = r.registrar(dna.clone(), "z", 1.0, 0, ("puro:RS".into(), "puro:FS".into()));
        let b = r.registrar(dna, "z", 1.0, 0, ("puro:RS".into(), "puro:FS".into()));
        assert_ne!(a, b, "assinaturas devem ser únicas");
        assert_eq!(r.total(), 2);
    }

    #[test]
    fn celula_nasce_diferencia_e_implanta() {
        let mut camada = CamadaHibrida::new(
            32, "z_test", TipoNeuronal::RS, Some((TipoNeuronal::FS, 0.2)), None, 1.0,
        );
        let n0 = camada.neuronios.len();
        let mut c = CelulaTronco::nascer(&camada, "z_test", "HYB-g0-0000".into());
        assert_eq!(c.fase, FaseTronco::Indiferenciada);

        let necessidade = NecessidadeZona {
            atividade_media: 0.0, silenciosos_frac: 0.5,
            falta_inibicao: false, falta_drivers: true,
        };
        c.diferenciar(&necessidade);
        assert_eq!(c.fase, FaseTronco::Diferenciada);
        assert!(!c.dna.e_inibitorico, "diferenciação p/ drivers → excitatório");

        c.implantar(&mut camada);
        assert_eq!(c.fase, FaseTronco::EmProva);
        assert_eq!(camada.neuronios.len(), n0 + 1, "implante adiciona 1 neurônio");
        assert_eq!(camada.neuronios.last().unwrap().tipo, TipoNeuronal::Hybrid);
    }

    #[test]
    fn celula_viavel_e_aceita_celula_silenciosa_e_rejeitada() {
        let mut registro = RegistroEspecies::novo();

        // ── Caso ACEITE: DNA excitável (RS) com drive forte → dispara na banda ──
        {
            let mut camada = CamadaHibrida::new(16, "aceite", TipoNeuronal::RS, None, None, 1.0);
            let mut c = CelulaTronco::nascer(&camada, "aceite", "HYB-g0-0001".into());
            c.dna = TipoNeuronal::RS.extrair_dna(); // fenótipo viável conhecido
            c.fase = FaseTronco::Diferenciada;
            c.implantar(&mut camada);
            let idx = c.idx_implante.unwrap();
            // Drive direto no neurônio em prova para garantir atividade na banda.
            for t in 0..3000 {
                let mut inputs = vec![0.0f32; camada.neuronios.len()];
                inputs[idx] = 40.0;
                camada.update(&inputs, DT, t as f32);
            }
            let aceite = c.julgar(&mut camada, &mut registro, 1);
            assert!(aceite, "célula viável (dispara na banda) deve ser ACEITA");
            assert_eq!(c.fase, FaseTronco::Aceite);
            assert_eq!(registro.total(), 1, "espécie aceita deve ser registrada");
        }

        // ── Caso REJEIÇÃO: DNA que nunca dispara (threshold no teto) → silencioso ─
        {
            let mut camada = CamadaHibrida::new(16, "rej", TipoNeuronal::RS, None, None, 1.0);
            let mut c = CelulaTronco::nascer(&camada, "rej", "HYB-g0-0002".into());
            let mut dna = TipoNeuronal::RS.extrair_dna();
            dna.threshold = 40.0; // limiar máximo + sem drive → nunca dispara
            c.dna = dna;
            c.fase = FaseTronco::Diferenciada;
            c.implantar(&mut camada);
            let idx = c.idx_implante.unwrap();
            // SEM drive no neurônio em prova → fica silencioso.
            for t in 0..2000 {
                let inputs = vec![0.0f32; camada.neuronios.len()];
                camada.update(&inputs, DT, t as f32);
            }
            let _ = idx;
            let aceite = c.julgar(&mut camada, &mut registro, 2);
            assert!(!aceite, "célula silenciosa deve ser REJEITADA (apoptose)");
            assert_eq!(c.fase, FaseTronco::Rejeitada);
        }
    }

    #[test]
    fn gestor_so_intervem_quando_ha_necessidade() {
        // Zona saudável (RS+FS equilibrada, sem runaway) → nenhuma intervenção.
        let mut zona = CamadaHibrida::new(
            64, "z_ok", TipoNeuronal::RS, Some((TipoNeuronal::FS, 0.2)), None, 1.0,
        );
        let mut gestor = GestorNeurogenese::novo(3);
        let (_, _, nascidas) = gestor.tick_sono(&mut zona);
        assert_eq!(nascidas, 0, "zona saudável não deve gerar células-tronco");
    }
}
