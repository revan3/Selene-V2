// =============================================================================
// src/learning/rl.rs
// =============================================================================
//
// APRENDIZADO POR REFORÇO DA SELENE — Q-Learning com TD(λ)
//
// Analogia biológica:
//   Gânglios da base (Striatum + SNpc) + Dopamina como sinal de reforço
//
// O que este módulo faz:
//   1. Codifica padrões de ativação neural como chaves de estado compactas
//   2. Mantém uma Q-table: Q(estado) → valor esperado de recompensa futura
//   3. A cada tick, calcula o Erro de Predição de Recompensa (RPE)
//   4. Propaga o RPE para estados anteriores via traços de elegibilidade
//   5. Expõe `value_of(padrão)` para que o FrontalLobe tome decisões
//
// Por que isso importa para aprender palavras?
//   Sem RL, a Selene reage à INTENSIDADE de um padrão, mas não ao SIGNIFICADO.
//   Com RL + câmera + microfone:
//     - "medo" co-ocorre com barulho alto (amígdala dispara, dopamina cai)
//     - Q("medo") ← negativo após N repetições
//     - FrontalLobe lê Q("medo") < 0 → resposta de evitação
//   Requer câmera e microfone funcionando para ter experiências reais para aprender.
//
// Algoritmo: Q-Learning com Traços de Elegibilidade (TD-lambda)
//
//   TD padrão atualiza apenas o estado ATUAL. TD-lambda propaga o erro
//   para todos os estados visitados recentemente, ponderados pelo traço.
//   Isso acelera muito o aprendizado em sequências longas.
//
//   δ (TD error / RPE) = r + γ × Q(s') − Q(s)
//   Q(s) ← Q(s) + α × δ × e(s)     para todo s com e(s) > 0
//   e(s) ← γ × λ × e(s)             decai a cada tick
//   e(s_atual) ← e(s_atual) + 1     bump no estado atual
//
// =============================================================================

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use crate::brain_zones::RegionType;
use crate::config::Config;

// -----------------------------------------------------------------------------
// Constantes do algoritmo
// -----------------------------------------------------------------------------

/// Fator de desconto do futuro (γ — gamma).
/// 0.95 = recompensas daqui a 10 passos valem 0.95^10 ≈ 60% de uma imediata.
/// Valores mais altos (0.99) = Selene considera futuro distante mais.
/// Valores mais baixos (0.8) = Selene é mais "imediatista".
const GAMMA: f32 = 0.95;

/// Parâmetro lambda dos traços de elegibilidade (λ).
/// 0.9 = traços decaem a 90% a cada tick.
/// λ=0 = TD(0) — só atualiza o estado atual (mais rápido, menos preciso).
/// λ=1 = Monte Carlo — propaga para todos os estados igualmente (lento).
/// λ=0.9 é um bom equilíbrio empírico.
const LAMBDA: f32 = 0.9;

/// Dopamina de referência para calcular recompensa relativa.
/// Recompensa = dopamina_atual − BASELINE
/// > 0 → experiência melhor que o esperado (surpresa boa)
/// < 0 → experiência pior que o esperado (decepção)
/// Analogia: neurônios dopaminérgicos da SNpc têm taxa de disparo basal.
const BASELINE_DOPAMINA: f32 = 0.5;

/// Número máximo de estados na Q-table.
/// Com padrões de 512 neurônios quantizados em 256 níveis, o espaço de estados
/// é astronomicamente grande. Limitamos para evitar uso infinito de RAM.
/// Quando atingido, estados com traços < 0.01 são removidos (os "esquecidos").
const MAX_Q_ESTADOS: usize = 50_000;

/// Limiar mínimo de traço de elegibilidade para manter um estado ativo.
/// Estados com e < 0.01 são considerados "esquecidos" e podem ser removidos.
const ELIGIBILITY_THRESHOLD: f32 = 0.01;

// -----------------------------------------------------------------------------
// Tipo de chave de estado
// -----------------------------------------------------------------------------

/// Representação compacta de um padrão de ativação neural (16 bytes).
///
/// Por que 16 bytes e não guardar o vetor completo (512 floats = 2KB)?
///   1. Eficiência: inserção/busca em HashMap com chave de 16 bytes é rápida
///   2. Generalização: estados ligeiramente diferentes mapeiam para a mesma chave
///      (isso é desejável — evita overfitting a padrões exatos)
///   3. Memória: 50.000 estados × 16 bytes = 800KB vs 50.000 × 2KB = 100MB
type ChaveEstado = [u8; 16];

// -----------------------------------------------------------------------------
// Estrutura principal
// -----------------------------------------------------------------------------

/// Motor de Aprendizado por Reforço da Selene.
///
/// Deve ser instanciado uma vez no main.rs e ter `update()` chamado
/// a cada tick do loop principal, após o processamento do FrontalLobe.
pub struct ReinforcementLearning {
    /// Tabela de valores Q: Q(estado) → recompensa futura esperada.
    /// Positivo = estado associado a experiências boas.
    /// Negativo = estado associado a experiências ruins.
    q_table: HashMap<ChaveEstado, f32>,

    /// Traços de elegibilidade: e(estado) → quão recentemente visitado.
    /// Decaem exponencialmente com o tempo.
    /// Permitem propagar o TD error para estados anteriores.
    eligibility: HashMap<ChaveEstado, f32>,

    /// Histórico de recompensas recentes (últimas 1000).
    /// Usado para telemetria e para calcular baseline adaptativo no futuro.
    historico_recompensa: VecDeque<f32>,

    /// Último estado visitado (para calcular TD error no próximo tick).
    ultimo_estado: Option<ChaveEstado>,

    /// Última ação tomada pelo FrontalLobe (0..1).
    /// Guardado para análise e debugging.
    ultima_acao: f32,

    /// Erro de Predição de Recompensa do último tick.
    ///
    /// Equivale biologicamente ao sinal dopaminérgico dos neurônios da SNpc.
    /// Positivo = situação melhor que o esperado → dopamina sobe.
    /// Negativo = situação pior que o esperado  → dopamina cai.
    /// Zero     = situação exatamente como esperado → nada muda.
    rpe_atual: f32,

    /// Contador total de atualizações desde o início.
    total_atualizacoes: u64,
}

// -----------------------------------------------------------------------------
// Implementação
// -----------------------------------------------------------------------------

impl ReinforcementLearning {
    /// Cria uma nova instância do módulo RL, sem histórico anterior.
    ///
    /// No futuro: `new_from_checkpoint(path)` para restaurar Q-table salva.
    pub fn new() -> Self {
        Self {
            q_table: HashMap::with_capacity(10_000),
            eligibility: HashMap::with_capacity(1_000),
            historico_recompensa: VecDeque::with_capacity(1_000),
            ultimo_estado: None,
            ultima_acao: 0.0,
            rpe_atual: 0.0,
            total_atualizacoes: 0,
        }
    }

    // -------------------------------------------------------------------------
    // Update principal — chamar a cada tick do loop neural
    // -------------------------------------------------------------------------

    /// Atualiza a Q-table com base no novo estado e recompensa observada.
    ///
    /// Deve ser chamado APÓS o processamento do FrontalLobe, pois usa
    /// a ação do Frontal como referência para o aprendizado.
    ///
    /// # Parâmetros
    /// - `padrao_temporal`: saída do TemporalLobe (padrão reconhecido)
    /// - `dopamina`: nível atual de dopamina no NeuroChem (0.0..2.0)
    /// - `acao_frontal`: output do FrontalLobe normalizado (0.0..1.0)
    /// - `config`: configuração global (taxa_aprendizado, etc.)
    ///
    /// # Retorno
    /// O RPE (Reward Prediction Error) deste tick.
    /// Use para ajustar a dopamina no NeuroChem:
    /// ```rust
    /// let rpe = rl.update(&padrao, neuro.dopamine, acao, &config);
    /// neuro.dopamine = (neuro.dopamine + rpe * 0.1).clamp(0.0, 2.0);
    /// ```
    pub fn update(
        &mut self,
        padrao_temporal: &[f32],
        dopamina: f32,
        acao_frontal: f32,
        config: &Config,
    ) -> f32 {
        // Codifica o padrão de ativação como chave de estado compacta
        let estado_atual = Self::codificar_estado(padrao_temporal);

        // Recompensa = quanto a dopamina atual difere da baseline esperada.
        // Negativa se dopamina caiu → punição implícita.
        // Positiva se dopamina subiu → recompensa implícita.
        let recompensa = dopamina - BASELINE_DOPAMINA;

        // ── Atualização TD-lambda ─────────────────────────────────────────────
        if let Some(estado_anterior) = self.ultimo_estado {
            // Q-value do estado anterior (0.0 se nunca visto)
            let q_anterior = *self.q_table.get(&estado_anterior).unwrap_or(&0.0);

            // Q-value do estado atual (estimativa do futuro)
            let q_atual = *self.q_table.get(&estado_atual).unwrap_or(&0.0);

            // ── TD Error (= Reward Prediction Error = sinal dopaminérgico) ────
            //
            // δ = r + γ × V(s') − V(s)
            //
            // Se δ > 0: situação melhor que o previsto (dopamina sobe)
            // Se δ < 0: situação pior que o previsto  (dopamina cai)
            // Se δ = 0: exatamente como previsto      (dopamina estável)
            let td_error = recompensa + GAMMA * q_atual - q_anterior;
            self.rpe_atual = td_error;

            // ── Atualiza traços de elegibilidade ─────────────────────────────
            //
            // Todos os traços existentes decaem (γ × λ por tick).
            // Isso implementa a "memória de curto prazo" do RL.
            // Estados visitados há mais tempo têm menos responsabilidade
            // pelo erro atual do que estados visitados recentemente.
            for e in self.eligibility.values_mut() {
                *e *= GAMMA * LAMBDA;
            }

            // O estado anterior recebe um "bump" de elegibilidade.
            // Indica que ele acabou de ser visitado e tem alta responsabilidade.
            *self.eligibility.entry(estado_anterior).or_insert(0.0) += 1.0;

            // ── Propaga TD error para todos os estados elegíveis ──────────────
            //
            // Q(s) ← Q(s) + α × δ × e(s)
            //
            // Estados com traço alto (visitados recentemente) recebem
            // atualização maior que estados com traço baixo (visitados antes).
            let alpha = config.taxa_aprendizado;

            // Coleta atualizações primeiro para evitar borrow mutável duplo
            let atualizacoes: Vec<(ChaveEstado, f32)> = self.eligibility
                .iter()
                .map(|(&chave, &e)| {
                    let q_atual_estado = *self.q_table.get(&chave).unwrap_or(&0.0);
                    let novo_q = q_atual_estado + alpha * td_error * e;
                    (chave, novo_q)
                })
                .collect();

            // Aplica as atualizações
            for (chave, novo_q) in atualizacoes {
                self.q_table.insert(chave, novo_q);
            }

            // ── Poda da Q-table (evita crescimento infinito) ─────────────────
            if self.q_table.len() > MAX_Q_ESTADOS {
                // Remove estados com traço muito baixo (praticamente esquecidos)
                let para_remover: Vec<ChaveEstado> = self.eligibility
                    .iter()
                    .filter(|(_, &e)| e < ELIGIBILITY_THRESHOLD)
                    .map(|(&chave, _)| chave)
                    .collect();

                for chave in para_remover {
                    self.q_table.remove(&chave);
                    self.eligibility.remove(&chave);
                }
            }

            // Registra recompensa no histórico para telemetria
            self.historico_recompensa.push_back(recompensa);
            if self.historico_recompensa.len() > 1000 {
                self.historico_recompensa.pop_front();
            }

            self.total_atualizacoes += 1;
        }

        // Atualiza estado e ação para o próximo tick
        self.ultimo_estado = Some(estado_atual);
        self.ultima_acao   = acao_frontal;

        // Retorna o RPE para que o main.rs possa modular a dopamina
        self.rpe_atual
    }

    // -------------------------------------------------------------------------
    // Consulta de valor aprendido
    // -------------------------------------------------------------------------

    /// Retorna o valor Q aprendido para um padrão de ativação.
    ///
    /// Usado pelo FrontalLobe para tomar decisões baseadas em experiência:
    /// - Valor positivo → padrão associado a recompensas anteriores → aproximar
    /// - Valor negativo → padrão associado a punições anteriores  → evitar
    /// - Valor zero     → padrão desconhecido → explorar com cautela
    ///
    /// # Exemplo de uso no FrontalLobe
    /// ```rust
    /// let valor = rl.valor_de(&padrao_temporal);
    /// let bias_emocional = valor.clamp(-1.0, 1.0);
    /// // Adiciona bias ao output do Frontal antes de decidir a ação
    /// acao = (acao + bias_emocional * 0.3).clamp(0.0, 1.0);
    /// ```
    pub fn valor_de(&self, padrao: &[f32]) -> f32 {
        let chave = Self::codificar_estado(padrao);
        *self.q_table.get(&chave).unwrap_or(&0.0)
    }

    // -------------------------------------------------------------------------
    // Codificação de estado (hashing)
    // -------------------------------------------------------------------------

    /// Codifica um vetor de ativações em uma chave de 16 bytes.
    ///
    /// # Algoritmo
    /// 1. Para cada neurônio, quantiza o valor float em 256 níveis (0-255)
    /// 2. Distribui os bytes quantizados nos 16 slots com XOR acumulativo
    ///
    /// # Por que quantizar?
    ///   Sem quantização, dois padrões que diferem em 0.001 seriam estados
    ///   completamente diferentes. Com quantização, padrões similares
    ///   (dentro de ±0.004) mapeiam para a mesma chave → generalização.
    ///
    /// # Por que XOR acumulativo?
    ///   XOR é uma operação hash simples, rápida e sem overhead.
    ///   `key[i % 16] ^= valor` distribui os bytes pelos 16 slots
    ///   de forma que todos os neurônios contribuem para todos os slots.
    ///
    /// # Limitação
    ///   Colisões são possíveis (dois padrões diferentes → mesma chave).
    ///   Para melhor qualidade de hash, substitua por FNV-1a ou SipHash.
    fn codificar_estado(padrao: &[f32]) -> ChaveEstado {
        let mut chave = [0u8; 16];

        for (i, &valor) in padrao.iter().enumerate() {
            // Clamp para [0,1], quantiza em 256 níveis inteiros
            let quantizado = (valor.clamp(0.0, 1.0) * 255.0) as u8;

            // XOR acumulativo — garante que todos os neurônios contribuem
            // para todos os 16 bytes da chave
            chave[i % 16] ^= quantizado;
        }

        chave
    }

    // -------------------------------------------------------------------------
    // Telemetria e diagnóstico
    // -------------------------------------------------------------------------

    /// RPE (Reward Prediction Error) do último tick.
    /// Equivale ao sinal dopaminérgico — positivo = surpresa boa.
    pub fn rpe(&self) -> f32 {
        self.rpe_atual
    }

    /// Total de atualizações desde o início (proxy de "experiência acumulada").
    pub fn total_atualizacoes(&self) -> u64 {
        self.total_atualizacoes
    }

    /// Número de estados únicos aprendidos até agora.
    pub fn n_estados(&self) -> usize {
        self.q_table.len()
    }

    /// Recompensa média dos últimos `n` ticks.
    /// Útil para detectar se a Selene está em período de aprendizado positivo.
    pub fn recompensa_media_recente(&self, n: usize) -> f32 {
        let janela: Vec<f32> = self.historico_recompensa
            .iter()
            .rev()
            .take(n)
            .copied()
            .collect();

        if janela.is_empty() {
            return 0.0;
        }

        janela.iter().sum::<f32>() / janela.len() as f32
    }

    /// Retorna os N estados com maior Q-value positivo (os mais "apreciados").
    ///
    /// Útil para o Ego construir narrativa sobre o que a Selene gosta.
    pub fn estados_mais_positivos(&self, n: usize) -> Vec<f32> {
        let mut valores: Vec<f32> = self.q_table.values().copied().collect();
        valores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        valores.into_iter().take(n).collect()
    }

    /// Retorna os N estados com menor Q-value negativo (os mais "temidos").
    pub fn estados_mais_negativos(&self, n: usize) -> Vec<f32> {
        let mut valores: Vec<f32> = self.q_table.values().copied().collect();
        valores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        valores.into_iter().take(n).collect()
    }

    // -------------------------------------------------------------------------
    // Persistência (base para CheckpointSystem)
    // -------------------------------------------------------------------------

    /// Serializa a Q-table para bytes (para salvar em arquivo ou banco).
    ///
    /// Use bincode para serialização eficiente:
    /// ```rust
    /// let bytes = rl.serializar();
    /// std::fs::write("rl_checkpoint.bin", bytes)?;
    /// ```
    pub fn serializar(&self) -> Vec<u8> {
        // Converte HashMap<[u8;16], f32> para Vec<([u8;16], f32)> para serializar
        let pares: Vec<([u8; 16], f32)> = self.q_table
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();

        // bincode::serialize(&pares).unwrap_or_default()
        // TODO: implementar quando bincode estiver disponível no contexto
        Vec::new()
    }

    /// Reseta completamente o aprendizado (esquece tudo).
    ///
    /// Use com cuidado — equivale a "apagar a memória emocional" da Selene.
    /// Útil apenas para testes ou quando o ambiente mudou drasticamente.
    pub fn resetar(&mut self) {
        self.q_table.clear();
        self.eligibility.clear();
        self.historico_recompensa.clear();
        self.ultimo_estado = None;
        self.rpe_atual = 0.0;
        self.total_atualizacoes = 0;
        println!("[RL] ⚠️  Q-table resetada. Todo o aprendizado foi perdido.");
    }
}

// -----------------------------------------------------------------------------
// Implementação de Display para telemetria
// -----------------------------------------------------------------------------

impl std::fmt::Display for ReinforcementLearning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RL[estados={} | RPE={:+.3} | updates={} | recomp_média={:+.3}]",
            self.n_estados(),
            self.rpe(),
            self.total_atualizacoes(),
            self.recompensa_media_recente(100)
        )
    }
}

// =============================================================================
// NOTAS PARA IMPLEMENTAÇÃO FUTURA
// =============================================================================
//
// 1. POLÍTICA DE EXPLORAÇÃO (ε-greedy)
//    Atualmente o RL é puramente "explotação" — sempre segue o Q-value.
//    Para aprender de verdade, a Selene precisa explorar ações desconhecidas.
//    Adicione: `if rand::random::<f32>() < epsilon { acao_aleatoria() }`
//    Com epsilon decaindo de 1.0 para 0.1 ao longo do tempo de aprendizado.
//
// 2. HASH DE ESTADO MELHORADO
//    O XOR atual pode ter muitas colisões. Substitua por:
//    - FNV-1a: simples, rápido, menos colisões
//    - SimHash: preserva similaridade (estados parecidos → hashes parecidos)
//    - LSH (Locality Sensitive Hashing): ideal para generalização sensorial
//
// 3. REPLAY BUFFER (Experience Replay)
//    Guarde as últimas N experiências (estado, ação, recompensa, próximo estado).
//    A cada tick, re-treine com amostras aleatórias do buffer.
//    Isso estabiliza muito o aprendizado (técnica do DQN do DeepMind).
//
// 4. REWARD SHAPING
//    Além da dopamina, adicione recompensas intrínsecas:
//    - Novelty reward: Q-value de estado desconhecido → pequena recompensa positiva
//      (curiosidade — a Selene é motivada a explorar o desconhecido)
//    - Entropy reward: diversidade de estados recentes → recompensa
//      (evita loops de comportamento repetitivo)
//
// 5. INTEGRAÇÃO COM BASAL GANGLIA
//    O BasalGanglia já existe (brain_zones). Conecte:
//    - BasalGanglia recebe RPE do RL
//    - BasalGanglia modula qual região cortical tem "gate aberto" (ação permitida)
//    - Isso implementa o ciclo cortico-estriatal-talâmico completo
//
// =============================================================================
