// src/learning/inter_lobe.rs
// Projeções sinápticas inter-lobais com STDP
//
// Modela as vias de longa distância que conectam os lobos cerebrais.
// Cada projeção é esparsa (cada neurônio fonte conecta a K destinos) e
// aprende por STDP baseado em firing rates (proxy de spikes populacionais).
//
// Conexões principais criadas em main.rs:
//   V1 → Temporal  (via ventral — "o quê")
//   V1 → Parietal  (via dorsal  — "onde")
//   Temporal → Frontal  (input para decisão executiva)
//   Frontal  → Temporal (top-down suppression / atenção)
//   Temporal ↔ Hippocampus (codificação episódica bidirecional)
//   Limbic   → Frontal (bias emocional bottom-up)
//   Frontal  → Limbic  (regulação emocional top-down)
//   Parietal → Frontal (integração espacial → ação)

#![allow(dead_code)]
#![allow(unused_variables)]

use rand::{Rng, thread_rng};

// ─────────────────────────────────────────────────────────────────────────────
// Constantes STDP inter-lobe
// ─────────────────────────────────────────────────────────────────────────────

/// Janela de tempo STDP para projeções de longa distância (ms).
/// Maior que o STDP local (20ms) para capturar latências de transmissão axonal.
const TAU_INTER_MS:    f32 = 35.0;
const LTP_INTER:       f32 = 0.006;  // potenciação inter-lobe (mais lenta que local)
const LTD_INTER:       f32 = 0.003;  // depressão inter-lobe
const W_MAX:           f32 = 3.0;
const W_MIN:           f32 = -2.0;   // projeções inibitórias têm peso negativo
const SPIKE_THRESHOLD: f32 = 0.45;   // firing rate mínimo para ser tratado como spike

// ─────────────────────────────────────────────────────────────────────────────
// Estrutura principal
// ─────────────────────────────────────────────────────────────────────────────

/// Projeção sináptica entre dois lobos cerebrais.
///
/// Opera com firing rates (Vec<f32>, 0..1) ao invés de spikes booleanos,
/// tornando-a compatível com as APIs de alto nível dos lobos.
pub struct InterLobeProjection {
    /// weights[i] = lista de (índice_destino, peso) para o neurônio fonte i.
    /// Esparso: cada neurônio fonte conecta a `k_connections` destinos.
    weights: Vec<Vec<(usize, f32)>>,
    n_source: usize,
    n_target: usize,

    /// Traços pré-sinápticos (para LTP: pré antes de pós).
    pre_trace:  Vec<f32>,
    /// Traços pós-sinápticos (para LTD: pós antes de pré).
    post_trace: Vec<f32>,

    /// Sinal positivo = excitatório, negativo = inibitório.
    sign: f32,

    /// Nome desta projeção (para telemetria/debug).
    pub nome: String,

    /// Peso médio atual (telemetria).
    pub peso_medio: f32,
}

impl InterLobeProjection {
    /// Cria uma nova projeção.
    ///
    /// - `n_source`: número de neurônios fonte
    /// - `n_target`: número de neurônios destino
    /// - `k_connections`: conexões por neurônio fonte (esparsidade)
    /// - `excitatorio`: true = pesos positivos (glutamatérgico), false = negativos (GABAérgico)
    /// - `nome`: identificador para logs
    pub fn new(
        n_source:      usize,
        n_target:      usize,
        k_connections: usize,
        excitatorio:   bool,
        nome:          &str,
    ) -> Self {
        let mut rng = thread_rng();
        let k = k_connections.min(n_target);
        let sign: f32 = if excitatorio { 1.0 } else { -1.0 };

        // Inicializa pesos pequenos e aleatórios (distribuição uniforme)
        let mut weights = Vec::with_capacity(n_source);
        for _ in 0..n_source {
            let mut conns = Vec::with_capacity(k);
            let mut destinos: Vec<usize> = (0..n_target).collect();
            // Amostragem sem reposição — evita conexões duplicadas
            for slot in 0..k {
                let idx = rng.gen_range(slot..n_target);
                destinos.swap(slot, idx);
                let w = sign * rng.gen_range(0.05f32..0.25f32);
                conns.push((destinos[slot], w));
            }
            weights.push(conns);
        }

        Self {
            weights,
            n_source,
            n_target,
            pre_trace:  vec![0.0; n_source],
            post_trace: vec![0.0; n_target],
            sign,
            nome: nome.to_string(),
            peso_medio: 0.15 * sign,
        }
    }

    /// Projeta firing rates da fonte para correntes aditivas no destino.
    ///
    /// Retorna vetor de tamanho `n_target` com correntes somadas.
    /// Cada elemento é a soma ponderada das ativações da fonte.
    pub fn project(&self, source_rates: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_target];
        let n = self.n_source.min(source_rates.len());
        for i in 0..n {
            let rate = source_rates[i];
            if rate < 0.01 { continue; }  // sparse: ignora silenciosos
            for &(dst, w) in &self.weights[i] {
                out[dst] += rate * w;
            }
        }
        out
    }

    /// Atualiza pesos via STDP usando firing rates como proxy de spike.
    ///
    /// Um neurônio "disparou" se sua taxa > SPIKE_THRESHOLD.
    /// Regra:
    ///   pré antes pós → LTP: pré_trace alto quando pós dispara → peso sobe
    ///   pós antes pré → LTD: pós_trace alto quando pré dispara → peso cai
    pub fn stdp_update(
        &mut self,
        source_rates: &[f32],
        target_rates: &[f32],
        dt_ms: f32,
    ) {
        let decay = (-dt_ms / TAU_INTER_MS).exp();

        // Decai todos os traços
        for t in &mut self.pre_trace  { *t *= decay; }
        for t in &mut self.post_trace { *t *= decay; }

        // Bump de traços nos neurônios que "dispararam"
        let n_src = self.n_source.min(source_rates.len());
        let n_tgt = self.n_target.min(target_rates.len());
        for i in 0..n_src {
            if source_rates[i] > SPIKE_THRESHOLD {
                self.pre_trace[i] += source_rates[i];
            }
        }
        for j in 0..n_tgt {
            if target_rates[j] > SPIKE_THRESHOLD {
                self.post_trace[j] += target_rates[j];
            }
        }

        // Atualiza pesos
        let mut soma_w = 0.0f32;
        let mut n_w = 0usize;
        for i in 0..n_src {
            let pre_active = source_rates[i] > SPIKE_THRESHOLD;
            for (dst, w) in &mut self.weights[i] {
                if *dst >= n_tgt { continue; }
                let post_active = target_rates[*dst] > SPIKE_THRESHOLD;

                if pre_active && post_active {
                    // Correlação simultânea → LTP proporcional ao traço pré
                    *w += self.sign * LTP_INTER * self.pre_trace[i];
                } else if post_active {
                    // Pós dispara sem pré → LTD (pós_trace do destino)
                    *w -= self.sign * LTD_INTER * self.post_trace[*dst];
                } else if pre_active {
                    // Pré dispara sem pós → LTD leve (pré sem resposta)
                    *w -= self.sign * LTD_INTER * 0.5 * self.pre_trace[i];
                }

                *w = w.clamp(W_MIN, W_MAX);
                soma_w += *w;
                n_w += 1;
            }
        }
        if n_w > 0 {
            self.peso_medio = soma_w / n_w as f32;
        }
    }

    /// Escala todo o conjunto de pesos por um fator (neuromodulação global).
    /// dopamina > 1 → escala aumenta LTP; cortisol > 0 → atenua plasticidade.
    pub fn modular(&mut self, dopamina: f32, cortisol: f32) {
        let fator = (dopamina * 0.8 - cortisol * 0.3).clamp(0.5, 1.8);
        for conns in &mut self.weights {
            for (_, w) in conns {
                *w = (*w * fator).clamp(W_MIN, W_MAX);
            }
        }
    }

    /// Telemetria: peso médio, número de conexões, traço médio.
    pub fn stats(&self) -> InterLobeStats {
        let total: usize = self.weights.iter().map(|c| c.len()).sum();
        let pre_med = self.pre_trace.iter().sum::<f32>() / self.pre_trace.len().max(1) as f32;
        let post_med = self.post_trace.iter().sum::<f32>() / self.post_trace.len().max(1) as f32;
        InterLobeStats {
            nome: self.nome.clone(),
            n_conexoes: total,
            peso_medio: self.peso_medio,
            pre_trace_medio: pre_med,
            post_trace_medio: post_med,
        }
    }
}

pub struct InterLobeStats {
    pub nome: String,
    pub n_conexoes: usize,
    pub peso_medio: f32,
    pub pre_trace_medio: f32,
    pub post_trace_medio: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Conjunto de projeções — instanciado uma vez em main.rs
// ─────────────────────────────────────────────────────────────────────────────

/// Todas as vias de longa distância do cérebro da Selene.
///
/// Criado com `BrainConnections::new(n_neurons)` e atualizado a cada tick
/// chamando `update()` com os firing rates de cada lobe.
pub struct BrainConnections {
    // Via ventral: V1 → Temporal (reconhecimento de objetos)
    pub v1_temporal:        InterLobeProjection,
    // Via dorsal: V1 → Parietal (onde está o objeto)
    pub v1_parietal:        InterLobeProjection,
    // Temporal → Frontal (padrão reconhecido → decisão)
    pub temporal_frontal:   InterLobeProjection,
    // Frontal → Temporal (top-down: atenção direciona reconhecimento)
    pub frontal_temporal:   InterLobeProjection,
    // Temporal ↔ Hippocampus (codificação e recall episódico)
    pub temporal_hippo:     InterLobeProjection,
    pub hippo_temporal:     InterLobeProjection,
    // Limbic → Frontal (bias emocional nas decisões)
    pub limbic_frontal:     InterLobeProjection,
    // Frontal → Limbic (regulação emocional consciente)
    pub frontal_limbic:     InterLobeProjection,
    // Parietal → Frontal (onde → o quê fazer)
    pub parietal_frontal:   InterLobeProjection,
    // NOVA: Hippocampus → Frontal (pattern completion → WM gate / goal setting).
    // O hipocampo completa padrões episódicos e os envia ao frontal para influenciar
    // a working memory e a seleção de goals — memória de longo prazo alimenta decisão.
    // Biologicamente: subículo CA1 → PFC via fibras de projeção direta.
    pub hippo_frontal:      InterLobeProjection,
    // NOVA: Parietal → Hippocampus (spatial context → episodic encoding).
    // Onde algo aconteceu importa para a memória episódica. O mapa espacial parietal
    // enriquece a codificação hipocampal com contexto de "onde" — como células de lugar.
    // Biologicamente: córtex parietal posterior → córtex entorrinal → CA1/CA3.
    pub parietal_hippo:     InterLobeProjection,
}

impl BrainConnections {
    /// Cria todas as projeções inter-lobais.
    /// `n`: número base de neurônios por lobe.
    pub fn new(n: usize) -> Self {
        // K conexões por neurônio: equilíbrio entre esparsidade e cobertura.
        // Biologicamente: ~10.000 sinapses por neurônio cortical, mas na Selene
        // a camada de lobe tem N neurônios, então K=sqrt(N) é razoável.
        let k = ((n as f32).sqrt() as usize).max(4).min(64);

        let k_half = k / 2; // projeções novas com menos conexões (n/2 destino)
        Self {
            v1_temporal:      InterLobeProjection::new(n, n, k, true,  "v1→temporal"),
            v1_parietal:      InterLobeProjection::new(n, n, k, true,  "v1→parietal"),
            temporal_frontal: InterLobeProjection::new(n, n, k, true,  "temporal→frontal"),
            frontal_temporal: InterLobeProjection::new(n, n, k, false, "frontal→temporal"),
            temporal_hippo:   InterLobeProjection::new(n, n / 2, k_half, true, "temporal→hippo"),
            hippo_temporal:   InterLobeProjection::new(n / 2, n, k_half, true, "hippo→temporal"),
            limbic_frontal:   InterLobeProjection::new(n / 2, n, k_half, true, "limbic→frontal"),
            frontal_limbic:   InterLobeProjection::new(n, n / 2, k_half, false,"frontal→limbic"),
            parietal_frontal: InterLobeProjection::new(n, n, k, true,  "parietal→frontal"),
            hippo_frontal:    InterLobeProjection::new(n / 2, n, k_half, true, "hippo→frontal"),
            parietal_hippo:   InterLobeProjection::new(n, n / 2, k_half, true, "parietal→hippo"),
        }
    }

    /// Aplica todas as projeções e retorna correntes aditivas para cada lobe.
    ///
    /// `source_*` são os firing rates de saída do tick anterior de cada lobe.
    /// Retorna correntes que devem ser SOMADAS aos inputs sensoriais de cada lobe
    /// antes do update() do tick atual.
    pub fn project_all(
        &self,
        v1_rates:       &[f32],
        temporal_rates: &[f32],
        parietal_rates: &[f32],
        frontal_rates:  &[f32],
        limbic_rates:   &[f32],
        hippo_rates:    &[f32],
    ) -> InterLobeCurrents {
        InterLobeCurrents {
            // Temporal: V1 (bottom-up) + hippo (recall episódico) + frontal (top-down supressão)
            para_temporal: add_vecs(
                &add_vecs(&self.v1_temporal.project(v1_rates), &self.hippo_temporal.project(hippo_rates)),
                &self.frontal_temporal.project(frontal_rates),
            ),
            // Parietal: V1 (onde o objeto está)
            para_parietal: self.v1_parietal.project(v1_rates),
            // Frontal: temporal (o quê) + limbic (emoção) + parietal (onde) + hippo (memória episódica)
            para_frontal: add_vecs(
                &add_vecs(
                    &add_vecs(&self.temporal_frontal.project(temporal_rates), &self.limbic_frontal.project(limbic_rates)),
                    &self.parietal_frontal.project(parietal_rates),
                ),
                &self.hippo_frontal.project(hippo_rates),
            ),
            // Limbic: frontal (regulação top-down)
            para_limbic: self.frontal_limbic.project(frontal_rates),
            // Hippo: temporal (codificação episódica) + parietal (contexto espacial)
            para_hippo: add_vecs(
                &self.temporal_hippo.project(temporal_rates),
                &self.parietal_hippo.project(parietal_rates),
            ),
        }
    }

    /// Atualiza pesos STDP de todas as projeções.
    pub fn stdp_update_all(
        &mut self,
        v1_rates:       &[f32],
        temporal_rates: &[f32],
        parietal_rates: &[f32],
        frontal_rates:  &[f32],
        limbic_rates:   &[f32],
        hippo_rates:    &[f32],
        dt_ms: f32,
    ) {
        self.v1_temporal.stdp_update(v1_rates, temporal_rates, dt_ms);
        self.v1_parietal.stdp_update(v1_rates, parietal_rates, dt_ms);
        self.temporal_frontal.stdp_update(temporal_rates, frontal_rates, dt_ms);
        self.frontal_temporal.stdp_update(frontal_rates, temporal_rates, dt_ms);
        self.temporal_hippo.stdp_update(temporal_rates, hippo_rates, dt_ms);
        self.hippo_temporal.stdp_update(hippo_rates, temporal_rates, dt_ms);
        self.limbic_frontal.stdp_update(limbic_rates, frontal_rates, dt_ms);
        self.frontal_limbic.stdp_update(frontal_rates, limbic_rates, dt_ms);
        self.parietal_frontal.stdp_update(parietal_rates, frontal_rates, dt_ms);
        self.hippo_frontal.stdp_update(hippo_rates, frontal_rates, dt_ms);
        self.parietal_hippo.stdp_update(parietal_rates, hippo_rates, dt_ms);
    }

    /// Neuromodulação de todas as projeções.
    pub fn modular_all(&mut self, dopamina: f32, cortisol: f32) {
        self.v1_temporal.modular(dopamina, cortisol);
        self.v1_parietal.modular(dopamina, cortisol);
        self.temporal_frontal.modular(dopamina, cortisol);
        self.frontal_temporal.modular(dopamina, cortisol);
        self.temporal_hippo.modular(dopamina, cortisol);
        self.hippo_temporal.modular(dopamina, cortisol);
        self.limbic_frontal.modular(dopamina, cortisol);
        self.frontal_limbic.modular(dopamina, cortisol);
        self.parietal_frontal.modular(dopamina, cortisol);
        self.hippo_frontal.modular(dopamina, cortisol);
        self.parietal_hippo.modular(dopamina, cortisol);
    }
}

/// Correntes aditivas calculadas pelas projeções inter-lobais para este tick.
pub struct InterLobeCurrents {
    pub para_temporal: Vec<f32>,
    pub para_parietal: Vec<f32>,
    pub para_frontal:  Vec<f32>,
    pub para_limbic:   Vec<f32>,
    pub para_hippo:    Vec<f32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Soma elemento a elemento, ajustando tamanho pelo menor dos dois.
fn add_vecs(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len().min(b.len());
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a[i] + if i < n { b[i] } else { 0.0 });
    }
    out
}
