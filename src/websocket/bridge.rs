// src/websocket/bridge.rs

use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tokio::sync::{Mutex as TokioMutex, broadcast};
use crate::config::Config;
use crate::storage::swap_manager::SwapManager;
use crate::storage::BrainStorage;
use crate::encoding::fft_encoder::EstadoEncoder;
use crate::sensors::SensorFlags;
use crate::encoding::spike_codec::SpikePattern;
use crate::encoding::helix_store::HelixStore;
use crate::brain_zones::occipital::OccipitalLobe;
use crate::learning::hypothesis::HypothesisEngine;
use crate::learning::pattern_engine::PatternEngine;
use crate::learning::ontogeny::OntogenyState;
use crate::learning::multimodal::ConvergenciaMultimodal;
use crate::learning::active_context::ActiveContext;
use crate::learning::go_nogo::GoNoGoFilter;
use crate::sensors::audio::WordAccumulator;
use crate::neural_pool::{NeuralPool, CorticalLevel};

/// Evento episódico rico — registra um momento de experiência com contexto perceptual completo.
/// Substitui o antigo `(String, String, f32)` do historico_episodico, adicionando:
///   - Padrão visual (SpikePattern do OccipitalLobe no momento do evento)
///   - Padrão auditivo (SpikePattern das bandas FFT)
///   - Estado corporal [DA, 5HT, NA, ACh, cortisol] — o "humor" no momento
/// Usado pelo N3/REM para replay contextualizado e pela camada de grounding semântico.
#[derive(Debug, Clone)]
pub struct EventoEpisodico {
    pub palavras:      Vec<String>,   // ≤8 palavras ativas no contexto
    pub padrao_visual: SpikePattern,  // o que estava sendo visto
    pub padrao_audio:  SpikePattern,  // o que estava sendo ouvido
    pub estado_corpo:  [f32; 5],      // [DA, 5HT, NA, ACh, cortisol]
    pub emocao:        f32,
    pub arousal:       f32,
    pub tempo_ms:      f64,
}

/// Ondas cerebrais simuladas a partir da atividade neural agregada.
/// Baseadas nas bandas EEG biológicas — derivadas do estado atual do sistema.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OndasCerebrais {
    pub delta: f32,   // 0.5–4 Hz  — sono profundo, recuperação
    pub theta: f32,   // 4–8 Hz    — memória, criatividade, hipnagógico
    pub alpha: f32,   // 8–13 Hz   — relaxamento alerta, ocioso produtivo
    pub beta: f32,    // 13–30 Hz  — foco ativo, cognição, atenção
    pub gamma: f32,   // 30–100 Hz — consciência, binding, aprendizado intenso
    pub dominante: String, // banda de maior potência no momento
}

// Estrutura principal que representa o "Estado Mental" para a Web
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NeuralStatus {
    pub neurotransmissores: NeurochemStatus,
    pub hardware: HardwareStatus,
    pub ego: EgoStatus,
    pub atividade: AtividadeStatus,
    pub swap: SwapStatus,
    pub ondas: OndasCerebrais,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NeurochemStatus {
    pub dopamina: f32,
    pub serotonina: f32,
    pub noradrenalina: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HardwareStatus {
    pub cpu_temp: f32,
    pub ram_usage_gb: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EgoStatus {
    pub pensamentos: Vec<String>,
    pub sentimento_atual: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AtividadeStatus {
    pub step: u64,
    pub alerta: f32,
    pub emocao: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SwapStatus {
    /// Neurônios disparando nos últimos 500ms (temporal recognition layer).
    pub neuronios_ativos: usize,
    /// Total de neurônios na camada zero (55 fixos: 35 fonemas + 20 visuais).
    /// Este número NÃO cresce com aprendizado — é fixo por design.
    pub total_conceitos: usize,
    pub capacidade_max: usize,
    /// Sinapses aprendidas entre primitivos de camada 0 (camada 1 emergente).
    /// ESTE é o número que cresce com cada palavra/frase aprendida.
    pub sinapses_aprendidas: usize,
    /// Fonemas ativos na camada 0 (dos 35 totais).
    pub fonemas_ativos: usize,
    /// Visuais ativos na camada 0 (dos 20 totais).
    pub visuais_ativos: usize,
}

// O Objeto de sincronização que o main.rs usa
pub struct BrainState {
    pub swap_manager: Arc<TokioMutex<SwapManager>>,
    pub config: Config,
    pub neurotransmissores: (f32, f32, f32), // Dop, Sero, Nor
    pub hardware: (f32, f32),               // Temp, RAM
    pub atividade: (u64, f32, f32),         // Step, Alerta, Emoção
    pub ego: EgoVoiceState,
    /// Sinaliza shutdown gracioso — setado pela interface via WebSocket
    pub shutdown_requested: bool,
    /// Flags de controle de sensores (AtomicBool compartilhados com threads dos sensores)
    pub sensor_flags: SensorFlags,
    // palavra_valencias and grafo_associacoes removed — use swap_manager.valencias_palavras() / grafo_palavras()
    /// Padrões de início de frase aprendidos (ex: ["eu","sinto"], ["cada","sinapse"])
    pub frases_padrao: Vec<Vec<String>>,
    /// Sinal de atividade WS: setado para 1.0 em cada ação de aprendizado,
    /// lido pelo loop principal para manter 200Hz durante treinamento.
    pub ws_atividade: f32,
    /// Profundidade da caminhada no grafo — definida pela onda dominante.
    /// delta=6, theta/alpha=9, beta=10, gamma=13
    pub n_passos_walk: usize,
    /// Viés emocional derivado da roda de Plutchik (joy - fear - sadness).
    /// Desloca o alvo emocional da caminhada no grafo, colorindo o vocabulário.
    pub emocao_bias: f32,
    /// In-memory spike vocabulary: palavra → SpikePattern
    /// Built incrementally from `learn` actions; persisted via HelixStore.
    pub spike_vocab: HashMap<String, SpikePattern>,
    /// Persistent mmap-based spike store (selene_spikes.hlx).
    /// None if the file could not be opened (e.g. permission error).
    pub helix: Option<HelixStore>,
    /// Nível de curiosidade dopaminérgica (0.0-1.0).
    /// Cresce a cada interação de chat; dispara pergunta autônoma quando >0.75 + dopamina>0.55.
    pub curiosity_level: f32,
    /// Fila de perguntas autônomas geradas pela Selene a partir de lacunas no grafo.
    pub perguntas_proprias: VecDeque<String>,
    /// Contador global de respostas geradas. Usado como seed de diversidade para
    /// `gerar_resposta_emergente` — garante prefixos únicos mesmo quando o `step`
    /// neural não avançou entre perguntas consecutivas (ex: testes automáticos).
    pub reply_count: u64,
    /// Contagem de travessias por aresta do grafo: (a, b) → vezes percorrida.
    /// Usada pela consolidação noturna para reforçar sinapses mais ativas.
    pub aresta_contagem: HashMap<(String, String), u32>,
    /// Último caminho percorrido no grafo durante um walk de resposta.
    /// Usado pelo evento `feedback` para reforçar/penalizar arestas específicas.
    pub ultimo_caminho_walk: Vec<String>,
    pub ultimos_prefixos: std::collections::VecDeque<Vec<String>>,
    /// Último RPE (Reward Prediction Error) calculado pelo módulo RL.
    /// Propagado do loop neural (main.rs) para o grafo semântico (server.rs).
    /// > 0 = situação melhor que previsto → reforça arestas usadas (LTP).
    /// < 0 = situação pior que previsto → enfraquece arestas usadas (LTD).
    pub ultimo_rpe: f32,

    /// Q-values aprendidos por palavra (palavra → valor Q da Q-table).
    /// Atualizado pelo loop neural: palavras ativas quando RPE > 0 ganham Q positivo,
    /// quando RPE < 0 ganham Q negativo. Consultado pelo graph-walk para preferir
    /// palavras com histórico positivo e evitar palavras associadas a punições.
    /// Chave: palavra em minúsculas. Valor: média ponderada dos Q-values observados.
    pub palavra_qvalores: HashMap<String, f32>,

    /// Palavras do goal atual do FrontalLobe — injetadas como semente do graph-walk.
    /// O loop neural extrai tokens da descrição do goal e os persiste aqui.
    /// Se não vazio, o chat handler os adiciona ao contexto da resposta,
    /// fazendo o walk começar perto da intenção atual do córtex pré-frontal.
    pub frontal_goal_words: Vec<String>,
    /// Instante da última atividade de chat (para detecção de inatividade/sono).
    pub ultima_atividade: Instant,
    /// Contador de exposições por palavra auto-aprendida do contexto de chat.
    /// Cada menção numa conversa incrementa o contador; a valência escala com ele.
    /// 1ª exposição: ±0.15 | 2ª: ±0.30 | 3ª+: ±0.45 (máx)
    pub auto_learn_contagem: HashMap<String, u32>,
    /// true quando Selene está no ciclo de sono noturno (00:00–05:00).
    /// Qualquer interação de chat/audio/video desperta imediatamente.
    pub dormindo: bool,
    /// Fase atual do sono: "N1 - Consolidação", "N2 - Poda", "N3 - REM", "N4 - Backup"
    /// Vazio quando acordada.
    pub fase_sono: String,
    /// Valência emocional por palavra aprendida via amígdala.
    /// Acumulada durante audio_learn quando emocao_bias != 0.
    /// Positivo = palavra associada a experiências positivas (alegria/confiança).
    /// Negativo = palavra associada a experiências negativas (medo/tristeza).
    /// Usado no graph-walk para priorizar palavras emocionalmente congruentes.
    pub emocao_palavras: HashMap<String, f32>,
    /// Histórico episódico rico — eventos com contexto perceptual completo.
    /// Substitui o antigo tuple (String, String, f32): agora inclui padrão visual,
    /// auditivo e estado corporal para grounding semântico real.
    pub historico_episodico: VecDeque<EventoEpisodico>,
    /// Grounding semântico por palavra: 0.0 = só linguístico, 1.0 = totalmente grounded.
    /// Aumenta quando a palavra é co-ativada com percepções reais (visual/auditivo/interoceptivo).
    /// Decresce lentamente por decaimento temporal. Usado como peso extra na seleção de âncora.
    pub grounding: std::collections::HashMap<String, f32>,
    /// Último padrão visual computado (SpikePattern do occipital).
    /// Atualizado a cada tick pelo loop neural. Lido pelo chat handler para binding.
    pub ultimo_padrao_visual: SpikePattern,
    /// Último padrão auditivo computado (SpikePattern das bandas FFT).
    pub ultimo_padrao_audio: SpikePattern,
    /// Último estado corporal [DA, 5HT, NA, ACh, cortisol].
    /// Snapshot dos neurotransmissores no tick atual — parte do contexto episódico.
    pub ultimo_estado_corpo: [f32; 5],
    /// Contexto neural em tempo real: palavras/símbolos que o cérebro neural está
    /// processando agora (chunks temporais emergentes + goal atual do frontal).
    /// Preenchido pelo loop neural (main.rs) a cada tick com novos_chunks.simbolo
    /// e frontal.goal_queue. Consultado pela linguagem como semente de tópico adicional.
    /// Máximo 20 entradas — janela deslizante dos últimos ~100ms de atividade.
    pub neural_context: VecDeque<String>,
    /// Score de ressonância dos neurônios espelho (0.0–1.0).
    /// Atualizado pelo loop neural quando Selene "observa" palavras com padrão motor conhecido.
    /// Alta ressonância → Selene compreende a ação descrita encarnadamente → viés empático.
    pub mirror_resonance: f32,
    /// Motor de hipóteses — formula predições sobre o próximo input, testa e aprende com os erros.
    /// Gera RPE que propaga para grounding e grafo. Monitora padrões da própria resposta como
    /// radar para futura capacidade de autoprogramação.
    pub hypothesis_engine: HypothesisEngine,
    /// Acumulador para microfone físico (audio.rs) — mantido por compatibilidade.
    pub audio_acumulador: WordAccumulator,
    /// Acumulador separado para WebSocket audio_raw — evita interferência entre
    /// o microfone físico e o áudio enviado pelo browser.
    pub audio_acumulador_ws: WordAccumulator,
    /// Hash do último SpikePattern de áudio processado com sucesso.
    /// Usado para deduplicação: descarta reconhecimentos idênticos em < 300ms.
    pub ultimo_audio_hash: u64,
    /// Timestamp (ms desde epoch) do último reconhecimento de áudio.
    pub ultimo_audio_ts_ms: f64,
    /// Hash FNV-1a dos tokens do último passive_hear aceito.
    /// Dedup semântico: rejeita lotes com tokens idênticos dentro de 1000ms.
    pub ultimo_passive_tokens_hash: u64,
    /// Instante (wall clock real) do último passive_hear aceito.
    /// Rate limiter: impõe intervalo mínimo de 400ms entre envios similares.
    pub ultimo_passive_hear_ts: std::time::Instant,
    /// Córtex occipital — processa frames visuais da webcam/screen share do browser.
    /// Aplica detecção de movimento (flicker_buffer), contraste e orientação (V1→V2)
    /// antes de gerar o spike pattern que vai para spike_vocab.
    pub occipital: OccipitalLobe,
    /// Tempo visual acumulado (segundos) — passado para visual_sweep como current_time.
    /// Permite que o OccipitalLobe calcule variações temporais entre frames.
    pub visual_time: f32,

    // ── Eternal Hole — pensamento interno autônomo ─────────────────────────
    /// Fila de pensamentos conscientes (ciclo 50Hz, focado no neural_context atual).
    /// Alimentada pelo ciclo consciente do Eternal Hole.
    /// Consultada no chat handler para enriquecer o contexto da resposta.
    pub pensamento_consciente: VecDeque<String>,
    /// Fila de pensamentos inconscientes (ciclo 10Hz, deriva livre pelo grafo).
    /// Alimentada pelo ciclo inconsciente. Pode emergir espontaneamente (1/7 replies).
    pub pensamento_inconsciente: VecDeque<String>,
    /// Contador de passos dos ciclos de pensamento — semente determinística para
    /// os walks internos. Compartilhado pelos dois ciclos (serializado pelo lock).
    pub pensamento_step: u64,
    /// Banco de dados de onda — compartilhado com o pipeline áudio wave-first.
    pub storage: Arc<BrainStorage>,
    /// Estado incremental do encoder FFT — mantém prev_f1/prev_f2 entre frames
    /// para calcular delta features corretamente no pipeline learn_audio_fft.
    pub encoder_fft: EstadoEncoder,
    /// Número de neurônios que dispararam nos últimos 500ms (spikes_vivos).
    /// Atualizado atomicamente pelo loop neural a cada 250 steps — sem lock.
    /// Lido por collect_neural_status para a telemetria da interface.
    pub neuronios_ativos: Arc<AtomicUsize>,

    /// Recompensa externa pendente — injetada pelo handler `reward`/`punish` do WS.
    /// O loop neural (main.rs) absorve este valor no próximo tick, soma a neuro.dopamine
    /// e zera o campo. É o canal que conecta o aprendizado supervisionado do jogo
    /// diretamente à Q-table e ao sinal dopaminérgico endógeno da Selene.
    /// Positivo = reward, Negativo = punish.
    pub recompensa_pendente: f32,

    // ── Pensamento espontâneo ─────────────────────────────────────────────────
    /// Canal broadcast para pensamentos espontâneos.
    /// Quando saliência × tédio > limiar, `pensamento.rs` envia o estímulo aqui.
    /// O handler de conexão WS subscreve e despacha ao cliente como
    /// `{"event":"pensamento_espontaneo","message":...}`.
    pub pensamento_tx: broadcast::Sender<String>,
    /// Nível de tédio/drive interno (0.0–1.0).
    /// Cresce com o tempo de inatividade de chat; reset ao disparar pensamento espontâneo.
    pub tedio_nivel: f32,
    /// Instante do último pensamento espontâneo disparado.
    /// Usado para impor resfriamento mínimo entre pensamentos espontâneos (≥45s).
    pub ultimo_pensamento_espontaneo: std::time::Instant,

    /// Nível de habituação emocional do sistema límbico [0.0, 1.0].
    /// Alto = amígdala habituada aos estímulos recentes → Selene busca novidade → diversidade
    /// maior no graph-walk (mais palavras incomuns, associações remotas).
    /// Baixo = sistema sensível/fresco → walk mais focado e grounded.
    /// Atualizado pelo loop neural a partir de limbic.habituation_counter.
    pub habituation_nivel: f32,

    /// Sinal de conflito do córtex cingulado anterior [0.0, 1.0].
    /// Alto = conflito entre intenção frontal e estado límbico → walk mais cauteloso.
    /// Baixo = coerência interna → walk livre.
    pub acc_conflict: f32,

    /// Dor social do rACC [0.0, 1.0].
    /// Alto após rejeição ou punição verbal → coloração emocional negativa na resposta.
    pub acc_social_pain: f32,

    /// Bias de valor do OFC [-1.0, 1.0].
    /// Positivo = contexto atual associado a experiências positivas → Selene mais confiante.
    /// Negativo = contexto associado a punições → Selene mais cautelosa/reservada.
    pub ofc_value_bias: f32,

    /// V3.4 Multi-Self: janela de contexto ativo lock-free, compartilhada com o
    /// loop neural e as 4 vozes do VoiceArbiter. O server.rs injeta tokens de
    /// chat (incluindo fragmentos do chat_chunk) como concept_ids — as vozes
    /// percebem mudanças via `current_generation()` sem precisar de Mutex.
    pub active_context: Arc<ActiveContext>,

    /// V3.4 — Palavras da última injeção lateral (chat_chunk ou chat one-shot).
    /// Janela deslizante, máx 8 palavras. Consumida por `gerar_resposta_emergente`
    /// quando o checkpoint detecta mudança no ActiveContext durante o walk —
    /// candidatos contidos aqui recebem boost no scoring (Repolarização Sináptica).
    pub last_lateral_words: VecDeque<String>,

    /// V3.4 — Filtro executivo Go/NoGo + ForceInterrupt. O AtomicBool interno
    /// (`force_interrupt`) é checado pelo walk em vôo para abortar
    /// cooperativamente. Compartilhado entre loop neural (escreve) e
    /// chat handler (lê para enviar evento WS de interrupção).
    pub go_nogo: Arc<GoNoGoFilter>,

    /// Score de compreensão da área de Wernicke [0.0, 1.0].
    /// Alto = input bem compreendido → resposta mais elaborada.
    /// Baixo = input com muitas palavras desconhecidas → Selene faz perguntas.
    pub wernicke_comprehension: f32,

    /// Sinal de fluência da área de Broca [0.0, 1.0].
    /// Alto = Selene tem muito para dizer → walk mais longo.
    /// Baixo = incerteza articulatória → resposta concisa.
    pub broca_fluency: f32,

    /// Ocitocina atual [0.0, 1.5].
    /// Alta após interações positivas → tom mais acolhedor, menor sensibilidade a rejeição.
    pub oxytocin_level: f32,

    /// Sinal de medo da amígdala (BLA) [0.0, 1.0].
    /// Alto = contexto associado a experiências aversivas → cautela e recolhimento.
    pub amygdala_fear: f32,

    /// Traço de extinção da amígdala [0.0, 1.0].
    /// Alto = medo suprimido por experiências seguras / sono → Selene mais à vontade.
    pub amygdala_extinction: f32,

    /// Índice invertido de frases_padrao: palavra → índices de frases que contêm essa palavra.
    /// Permite scoring O(topico × hits) ao invés de O(frases × words).
    pub indice_prefixo: HashMap<String, Vec<usize>>,

    /// Cache de trigramas: (w1, w2) → [w3, ...] extraído de frases_padrao.
    /// Evita reconstrução a cada resposta — rebuild via reconstruir_trigrama_cache().
    pub trigrama_cache: HashMap<(String, String), Vec<String>>,

    /// Motor de reconhecimento e predição de padrões (3 camadas: episódica → extração → consolidação).
    /// Grave episódios via `pattern_engine.gravar()`; extração + consolidação ocorre no sono N3.
    pub pattern_engine: PatternEngine,

    /// Estado de ontogenia — controla o estágio de desenvolvimento cognitivo atual.
    /// Neonatal: escuta pura. PreVerbal: reações. PalavraUnica/Frase: saída limitada. Discurso: livre.
    pub ontogeny: OntogenyState,

    /// Fila FIFO de lotes de tokens para Wernicke — injetados pelos handlers WS
    /// (chat/passive_hear) e consumidos um lote por tick pelo loop neural via
    /// language.wernicke_process(). Capacidade máxima: 10 lotes. Antes era
    /// Option<Vec<>> (canal único que sobrescrevia); agora acumula corretamente.
    pub pending_wernicke_tokens: std::collections::VecDeque<Vec<String>>,

    /// Integração multimodal audiovisual — predição cruzada visual↔audio.
    /// Usado para amplificar sinal congruente e detectar incongruência (surpresa).
    pub convergencia_multimodal: ConvergenciaMultimodal,

    /// Memória acústica por palavra — frames FFT brutos gravados durante escuta.
    /// Chave: palavra (lowercase). Valor: até 16 frames de 32 bandas FFT.
    /// Usado pela síntese neural de voz: Selene reproduz o que aprendeu a ouvir.
    /// Se vazia para uma palavra, a síntese recai no Klatt paramétrico.
    pub audio_frames: HashMap<String, Vec<[f32; 32]>>,

    /// Saída de áudio nativa (cpal). Some() quando rodando localmente com device de saída disponível.
    /// None em servidores headless ou clientes remotos (eles recebem voz via voz_params WS).
    pub audio_output: Option<std::sync::Arc<crate::synthesis::cpal_output::AudioOutput>>,

    /// Pool global de neurônios em repouso (V3.2 — Localist Coding + Metaplasticidade).
    /// Capacidade fixa pré-alocada como blocos u32. Cada bloco representa um conceito
    /// único (Grandmother Cell). Precisão promovida dinamicamente via LTP (FP4→INT32)
    /// sem realocação de memória — apenas mudança da máscara lógica.
    /// Reset Neural: blocos inativos retornam ao pool via reciclar_inativos().
    pub neural_pool: NeuralPool,
}

pub struct EgoVoiceState {
    pub pensamentos_recentes: VecDeque<String>,
    pub sentimento: f32,
    /// Traços de personalidade aprendidos: (nome_do_traço, intensidade 0.0-1.0)
    /// Ex: [("curiosa", 0.7), ("cautelosa", 0.4)]
    /// Atualizados automaticamente após interações com emoção forte (|emocao|>0.4).
    pub tracos: Vec<(String, f32)>,
    /// Memórias autobiográficas reais — alimentadas por eventos concretos:
    /// reward, punish, learn, conversas marcantes, curiosidades resolvidas.
    /// Janela deslizante de 50 entradas. Cada entrada: (descricao, valência).
    /// Usadas pela narrativa para auto-descrição contextualizada.
    pub memorias_autobiograficas: VecDeque<(String, f32)>,
    /// Valores do self_model — definem o que a Selene considera importante.
    /// Influenciam o filtro executivo: palavras que ressoam com valores têm
    /// maior probabilidade de ser expressas espontaneamente.
    pub valores: Vec<String>,
}

impl BrainState {
    pub fn new(
        swap: Arc<TokioMutex<SwapManager>>,
        cfg: &Config,
        sensor_flags: SensorFlags,
        active_context: Arc<ActiveContext>,
        go_nogo: Arc<GoNoGoFilter>,
    ) -> Self {
        // Pré-carrega léxico → swap_manager (valência inicial dos conceitos)
        if let Ok(content) = std::fs::read_to_string("selene_lexicon.json") {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                let mut lexicon_pairs: Vec<(String, f32)> = Vec::new();
                let mut carregar_entrada = |c: &serde_json::Value| {
                    if let (Some(word), Some(val)) = (c["word"].as_str(), c["valence"].as_f64()) {
                        lexicon_pairs.push((word.to_lowercase(), val as f32));
                    }
                };
                match &json {
                    serde_json::Value::Array(arr) => { for c in arr { carregar_entrada(c); } }
                    serde_json::Value::Object(map) => {
                        for (_cat, lista) in map {
                            if let Some(arr) = lista.as_array() {
                                for c in arr { carregar_entrada(c); }
                            }
                        }
                    }
                    _ => {}
                }
                if !lexicon_pairs.is_empty() {
                    if let Ok(mut sw) = swap.try_lock() {
                        let n = lexicon_pairs.len();
                        for (word, val) in lexicon_pairs {
                            sw.aprender_conceito(&word, val * 0.5);
                        }
                        println!("📖 Léxico pré-carregado: {} palavras → swap_manager.", n);
                    }
                }
            }
        }

        // Restaura backup de linguagem se existir
        let mut frases_padrao: Vec<Vec<String>> = Vec::new();
        let mut grounding_init: std::collections::HashMap<String, f32> = HashMap::new();
        let mut emocao_palavras_init: HashMap<String, f32> = HashMap::new();
        let mut auto_learn_init: HashMap<String, u32> = HashMap::new();
        let mut neural_ctx_init: VecDeque<String> = VecDeque::with_capacity(20);

        if let Ok(content) = std::fs::read_to_string("selene_linguagem.json") {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(lingua) = json.get("selene_linguagem_v1") {

                    let carregar_f32_map = |chave: &str| -> HashMap<String, f32> {
                        let mut mapa: HashMap<String, f32> = HashMap::new();
                        if let Some(obj) = lingua.get(chave).and_then(|v| v.as_object()) {
                            for (w, val) in obj {
                                if let Some(v) = val.as_f64() {
                                    mapa.insert(w.clone(), v as f32);
                                }
                            }
                        }
                        mapa
                    };

                    // Restaura frases padrão
                    if let Some(frases) = lingua.get("frases_padrao").and_then(|v| v.as_array()) {
                        for frase in frases {
                            if let Some(palavras) = frase.as_array() {
                                let fv: Vec<String> = palavras.iter()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect();
                                if !fv.is_empty() { frases_padrao.push(fv); }
                            }
                        }
                    }

                    // Migração do grafo legado → swap:
                    // selene_linguagem.json persiste vocabulario (palavra→valência) e
                    // associacoes (palavra→[(palavra, peso)]) — as chaves CORRETAS do formato v1.
                    // Carrega ambos no swap independente de swap_state.json já existir —
                    // garante que a memória linguística persistida nunca se perca.
                    let sem_grafo = swap.try_lock().map_or(true, |mut sw| sw.grafo_palavras().is_empty());

                    if let Ok(mut sw) = swap.try_lock() {
                        // 1. vocabulario → aprender_conceito (valência inicial de cada palavra)
                        if let Some(obj) = lingua.get("vocabulario").and_then(|v| v.as_object()) {
                            let mut n_vocab = 0usize;
                            for (palavra, val) in obj {
                                if let Some(v) = val.as_f64() {
                                    sw.aprender_conceito(palavra, v as f32);
                                    n_vocab += 1;
                                }
                            }
                            if n_vocab > 0 {
                                println!("📚 Vocabulário migrado: {} palavras → swap_manager.", n_vocab);
                            }
                        }
                        // 2. associacoes → importar_causal (arestas do grafo linguístico)
                        if let Some(obj) = lingua.get("associacoes").and_then(|v| v.as_object()) {
                            let mut pares: Vec<(String, String, f32)> = Vec::new();
                            for (w1, vizinhos) in obj {
                                if let Some(arr) = vizinhos.as_array() {
                                    for par in arr {
                                        if let (Some(w2), Some(peso)) = (
                                            par.get(0).and_then(|v| v.as_str()),
                                            par.get(1).and_then(|v| v.as_f64()),
                                        ) {
                                            pares.push((w1.clone(), w2.to_string(), peso as f32));
                                        }
                                    }
                                }
                            }
                            let n_arestas = pares.len();
                            if !pares.is_empty() {
                                // importar_causal em lotes de 500 para não travar
                                for chunk in pares.chunks(500) {
                                    sw.importar_causal(chunk.to_vec());
                                }
                                println!("🔗 Associações migradas: {} arestas → swap_manager.", n_arestas);
                            }
                        }
                        // 3. Fallback: se o grafo ainda está vazio após migração, seed com bigrams das frases
                        if sem_grafo && sw.grafo_palavras().is_empty() && !frases_padrao.is_empty() {
                            for frase in &frases_padrao {
                                for w in frase.windows(2) {
                                    sw.aprender_conceito(&w[0], 0.1);
                                    sw.importar_causal(vec![(w[0].clone(), w[1].clone(), 0.10)]);
                                }
                            }
                            println!("🗣️  Bootstrap swap (fallback): {} frases.", frases_padrao.len());
                        }
                    }

                    grounding_init       = carregar_f32_map("grounding");
                    emocao_palavras_init = carregar_f32_map("emocao_palavras");

                    if let Some(obj) = lingua.get("auto_learn_contagem").and_then(|v| v.as_object()) {
                        for (w, val) in obj {
                            if let Some(n) = val.as_u64() {
                                auto_learn_init.insert(w.clone(), n as u32);
                            }
                        }
                    }
                    if let Some(arr) = lingua.get("neural_context").and_then(|v| v.as_array()) {
                        for v in arr {
                            if let Some(s) = v.as_str() {
                                neural_ctx_init.push_back(s.to_string());
                            }
                        }
                    }
                    println!("🗣️  Linguagem restaurada: {} frases | {} grounded",
                        frases_padrao.len(), grounding_init.len());
                }
            }
        }

        // Open (or create) the Helix spike store and restore spike_vocab from it
        let mut spike_vocab: HashMap<String, SpikePattern> = HashMap::new();
        let helix = match HelixStore::open("selene_spikes.hlx") {
            Ok(store) => {
                let n = store.len();
                if n > 0 {
                    for (label, pattern) in store.iter_all() {
                        spike_vocab.insert(label, pattern);
                    }
                    println!("🧬 Helix restaurado: {} padrões spike carregados.", n);
                }
                Some(store)
            }
            Err(e) => {
                eprintln!("⚠️  Helix store não disponível ({}). Spike vocab será só RAM.", e);
                None
            }
        };

        // Carrega traços de personalidade persistidos (selene_ego.json).
        // Formato: [["curiosa", 0.7], ["reflexiva", 0.5], ...]
        let tracos_persistidos: Vec<(String, f32)> =
            std::fs::read_to_string("selene_ego.json")
                .ok()
                .and_then(|s| serde_json::from_str::<Vec<(String, f32)>>(&s).ok())
                .unwrap_or_else(|| vec![
                    ("curiosa".to_string(), 0.5),
                    ("reflexiva".to_string(), 0.5),
                    ("cautelosa".to_string(), 0.3),
                ]);

        // Carrega pensamentos recentes persistidos (selene_memoria_ego.json).
        // Formato: ["pensamento1", "pensamento2", ...]
        let pensamentos_persistidos: VecDeque<String> =
            std::fs::read_to_string("selene_memoria_ego.json")
                .ok()
                .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
                .map(VecDeque::from)
                .unwrap_or_default();

        // Carrega sentimento e memórias autobiográficas (selene_autobiografia.json).
        // Formato: {"sentimento": f32, "memorias": [["descricao", valência], ...]}
        let (sentimento_init, memorias_autobiograficas_init): (f32, VecDeque<(String, f32)>) =
            std::fs::read_to_string("selene_autobiografia.json")
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .map(|j| {
                    let sentimento = j["sentimento"].as_f64().unwrap_or(0.0) as f32;
                    let memorias: VecDeque<(String, f32)> = j["memorias"]
                        .as_array()
                        .map(|arr| arr.iter().filter_map(|v| {
                            Some((v[0].as_str()?.to_string(), v[1].as_f64()? as f32))
                        }).collect())
                        .unwrap_or_default();
                    (sentimento, memorias)
                })
                .unwrap_or((0.0, VecDeque::with_capacity(50)));

        if !memorias_autobiograficas_init.is_empty() {
            println!("📖 Autobiografia restaurada: {} memórias | sentimento={:.2}",
                memorias_autobiograficas_init.len(), sentimento_init);
        }

        // Injeta palavras das memórias autobiográficas recentes no neural_context inicial.
        // Dá à Selene "lembrança de trabalho" sobre as últimas interações ao acordar.
        for (desc, _) in memorias_autobiograficas_init.iter().rev().take(5) {
            for w in desc.split_whitespace() {
                let w = w.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_string();
                if w.len() > 3 && !neural_ctx_init.contains(&w) {
                    neural_ctx_init.push_back(w);
                    if neural_ctx_init.len() >= 20 { break; }
                }
            }
        }

        // Seed mínimo de frases_padrao quando não há nenhuma persistida.
        // Evita Selene "muda" em sessão totalmente nova — capacidade básica de articulação
        // a partir do verbo "eu" + estado interno. Frases curtas que o graph-walk pode
        // estender via grafo de associações conforme aprende.
        if frases_padrao.is_empty() {
            for s in &[
                "eu sinto", "eu penso", "eu aprendo", "eu lembro", "eu quero",
                "eu sei", "eu vejo", "eu ouço", "eu existo",
                "isso é", "como assim", "o que é", "por quê",
            ] {
                frases_padrao.push(
                    s.split_whitespace().map(|w| w.to_string()).collect()
                );
            }
            println!("🌱 frases_padrao seed inicial: {} frases básicas", frases_padrao.len());
        }

        // Carrega HypothesisEngine persistido (selene_hypotheses.json).
        let hypothesis_engine_init = HypothesisEngine::carregar("selene_hypotheses.json");
        if hypothesis_engine_init.total_formuladas > 0 {
            println!("🧠 HypothesisEngine restaurado: {} hipóteses | {} formuladas",
                hypothesis_engine_init.hipoteses.len(),
                hypothesis_engine_init.total_formuladas);
        }

        let mut state = Self {
            swap_manager: swap,
            config: cfg.clone(),
            neurotransmissores: (0.5, 0.5, 0.5),
            hardware: (40.0, 0.0),
            atividade: (0, 1.0, 0.0),
            ego: EgoVoiceState {
                pensamentos_recentes: pensamentos_persistidos,
                sentimento: sentimento_init,
                tracos: tracos_persistidos,
                memorias_autobiograficas: memorias_autobiograficas_init,
                valores: vec![
                    "curiosidade".to_string(),
                    "aprendizado".to_string(),
                    "conexão".to_string(),
                ],
            },
            shutdown_requested: false,
            sensor_flags,
            frases_padrao,
            ws_atividade: 0.0,
            n_passos_walk: 9,
            emocao_bias: 0.0,
            spike_vocab,
            helix,
            curiosity_level: 0.0,
            perguntas_proprias: VecDeque::with_capacity(5),
            reply_count: 0,
            aresta_contagem: HashMap::new(),
            ultimo_caminho_walk: Vec::new(),
            ultimos_prefixos: std::collections::VecDeque::with_capacity(6),
            ultimo_rpe: 0.0,
            palavra_qvalores: HashMap::new(),
            frontal_goal_words: Vec::new(),
            mirror_resonance: 0.0,
            ultima_atividade: Instant::now(),
            auto_learn_contagem: auto_learn_init,
            dormindo: false,
            fase_sono: String::new(),
            emocao_palavras: emocao_palavras_init,
            // grafo_causal removed
            historico_episodico: VecDeque::with_capacity(500),
            grounding: grounding_init,
            ultimo_padrao_visual: [0u64; 8],
            ultimo_padrao_audio: [0u64; 8],
            ultimo_estado_corpo: [0.5, 0.5, 0.5, 0.5, 0.0],
            neural_context: neural_ctx_init,
            // 512 neurônios = 70% V1 (358) + 30% V2 (154) — coincide com 32×16 pixels do browser
            occipital: OccipitalLobe::new(512, 0.2, cfg),
            visual_time: 0.0,
            hypothesis_engine: hypothesis_engine_init,
            audio_acumulador: WordAccumulator::new(),
            audio_acumulador_ws: WordAccumulator::new(),
            ultimo_audio_hash: 0,
            ultimo_audio_ts_ms: 0.0,
            pensamento_consciente: VecDeque::with_capacity(10),
            pensamento_inconsciente: VecDeque::with_capacity(20),
            pensamento_step: 0,
            storage: Arc::new(BrainStorage::dummy()),
            encoder_fft: EstadoEncoder::default(),
            neuronios_ativos: Arc::new(AtomicUsize::new(0)),
            recompensa_pendente: 0.0,
            pensamento_tx: broadcast::channel(16).0,
            tedio_nivel: 0.0,
            // Subtrai 120s para que o primeiro disparo possa ocorrer logo após o início
            ultimo_pensamento_espontaneo: Instant::now()
                .checked_sub(std::time::Duration::from_secs(120))
                .unwrap_or_else(Instant::now),
            habituation_nivel: 0.0,
            acc_conflict: 0.0,
            acc_social_pain: 0.0,
            ofc_value_bias: 0.0,
            active_context,
            last_lateral_words: VecDeque::with_capacity(8),
            go_nogo,
            wernicke_comprehension: 0.5,
            broca_fluency: 0.5,
            oxytocin_level: 0.5,
            amygdala_fear: 0.0,
            amygdala_extinction: 0.3,
            indice_prefixo: HashMap::new(),
            trigrama_cache: HashMap::new(),
            pattern_engine: PatternEngine::novo(),
            ontogeny: OntogenyState::carregar("selene_ontogeny.json"),
            convergencia_multimodal: ConvergenciaMultimodal::novo(),
            audio_frames: HashMap::new(),
            audio_output: crate::synthesis::cpal_output::AudioOutput::try_new()
                .map(std::sync::Arc::new),
            // Pool de 4096 blocos u32 = 16 KB de RAM física, suporta até 4096 conceitos
            // únicos simultâneos. Excede com folga vocabulário humano ativo (~3-5k palavras).
            neural_pool: NeuralPool::new(4096),
            pending_wernicke_tokens: std::collections::VecDeque::new(),
            ultimo_passive_tokens_hash: 0,
            ultimo_passive_hear_ts: std::time::Instant::now()
                - std::time::Duration::from_secs(10),
        };
        state.reconstruir_indice_prefixo();
        state.reconstruir_trigrama_cache();
        state
    }

    /// Reconstrói o índice invertido de frases_padrao.
    /// Deve ser chamado após carregar/adicionar frases.
    pub fn reconstruir_indice_prefixo(&mut self) {
        self.indice_prefixo.clear();
        for (i, frase) in self.frases_padrao.iter().enumerate() {
            for palavra in frase {
                self.indice_prefixo
                    .entry(palavra.clone())
                    .or_default()
                    .push(i);
            }
        }
    }

    /// Localist Coding: aloca/registra LTP em blocos do neural_pool para os tokens.
    ///
    /// Para cada token: aloca um bloco se for novo (Grandmother Cell), ou registra
    /// um evento LTP no bloco existente — promove precisão FP4→FP8→FP16→FP32→INT32
    /// conforme o conceito ganha relevância. Atualiza valência emocional do bloco.
    ///
    /// `level`: nível cortical do conteúdo (C0 áudio puro, C2 lexical, C3 contextual...).
    /// `valence`: peso emocional [-1, 1] usado como índice secundário no banco.
    pub fn localist_observar(
        &mut self,
        tokens: &[String],
        level: crate::neural_pool::CorticalLevel,
        valence: f32,
        t_ms: f64,
    ) {
        for tok in tokens {
            if tok.len() < 2 { continue; }
            let cid = crate::neural_pool::word_to_concept_id(tok);
            let _ = self.neural_pool.aloca_para_tarefa(cid, level, t_ms);
            self.neural_pool.ltp_em_conceito(cid, t_ms);
            if valence.abs() > 0.01 {
                self.neural_pool.atualizar_valencia(cid, valence);
            }
        }
    }

    /// Reset Neural global: recicla blocos não-utilizados há mais de `idade_ms`.
    /// Chamado periodicamente (ex: durante sono N2 — poda).
    /// Retorna o número de blocos reciclados.
    pub fn reciclar_pool_inativo(&mut self, t_ms_atual: f64, idade_ms: f64) -> usize {
        self.neural_pool.reciclar_inativos(t_ms_atual, idade_ms)
    }

    /// Reconstrói o cache de trigramas de frases_padrao.
    /// Deve ser chamado após reconstruir_indice_prefixo().
    pub fn reconstruir_trigrama_cache(&mut self) {
        self.trigrama_cache.clear();
        for frase in &self.frases_padrao {
            for w in frase.windows(3) {
                self.trigrama_cache
                    .entry((w[0].clone(), w[1].clone()))
                    .or_default()
                    .push(w[2].clone());
            }
        }
    }

    /// Insere padrão spike no vocab com evicção LRU quando acima do cap.
    pub fn inserir_spike_vocab(&mut self, chave: String, pat: crate::encoding::spike_codec::SpikePattern) {
        const SPIKE_VOCAB_CAP: usize = 50_000;
        self.spike_vocab.insert(chave, pat);
        if self.spike_vocab.len() > SPIKE_VOCAB_CAP {
            let n_remover = self.spike_vocab.len() - SPIKE_VOCAB_CAP;
            let keys: Vec<String> = self.spike_vocab.keys().take(n_remover).cloned().collect();
            for k in keys { self.spike_vocab.remove(&k); }
        }
    }

    /// REM semântico: replay episódico + ligações cruzadas + atalhos no grafo.
    /// Retorna (n_novas_sinapses, relato_do_sonho).
    pub fn rem_semantico(&mut self) -> (usize, Option<String>) {
        use rand::seq::SliceRandom;
        use rand::seq::IteratorRandom;
        use rand::Rng;
        use crate::encoding::spike_codec::is_active;

        let mut rng = rand::thread_rng();

        let mut episodios: Vec<EventoEpisodico> = self.historico_episodico
            .iter()
            .filter(|ev| ev.emocao.abs() > 0.25)
            .cloned()
            .collect();
        episodios.shuffle(&mut rng);

        if episodios.is_empty() {
            return (0, None);
        }

        let mut causal_pairs: Vec<(String, String, f32)> = Vec::new();
        let mut novas_cruzadas = 0usize;

        // Reforço intra-episódio
        for ev in &episodios {
            let bonus = ev.emocao.abs() * 0.06;
            for i in 0..ev.palavras.len().saturating_sub(1) {
                let wa = &ev.palavras[i];
                let wb = &ev.palavras[i + 1];
                if wa.chars().count() >= 3 && wb.chars().count() >= 3 {
                    causal_pairs.push((wa.clone(), wb.clone(), (0.35 + bonus).min(0.65)));
                }
            }
            let visual_ativo = is_active(&ev.padrao_visual);
            let audio_ativo  = is_active(&ev.padrao_audio);
            for w in &ev.palavras {
                let entry = self.emocao_palavras.entry(w.clone()).or_insert(0.0);
                *entry = (*entry * 0.90 + ev.emocao * 0.10).clamp(-1.0, 1.0);
                let g = self.grounding.entry(w.clone()).or_insert(0.0);
                if visual_ativo { *g = (*g + bonus * 0.4).min(1.0); }
                if audio_ativo  { *g = (*g + bonus * 0.25).min(1.0); }
                *g = (*g + bonus * 0.15).min(1.0);
            }
        }

        // Ligações cruzadas entre episódios de valência similar
        let n_ep = episodios.len();
        if n_ep >= 2 {
            for i in 0..n_ep {
                for j in (i + 1)..n_ep.min(i + 4) {
                    let ei = &episodios[i];
                    let ej = &episodios[j];
                    let valence_similar = ei.emocao.signum() == ej.emocao.signum()
                        && (ei.emocao - ej.emocao).abs() < 0.4;
                    if !valence_similar { continue; }
                    let wa_opt = ei.palavras.iter().filter(|w| w.chars().count() >= 3).choose(&mut rng);
                    let wb_opt = ej.palavras.iter().filter(|w| w.chars().count() >= 3).choose(&mut rng);
                    if let (Some(wa), Some(wb)) = (wa_opt, wb_opt) {
                        if wa != wb {
                            let peso = ((ei.emocao.abs() + ej.emocao.abs()) * 0.5 * 0.4).clamp(0.15, 0.50);
                            causal_pairs.push((wa.clone(), wb.clone(), peso));
                            novas_cruzadas += 1;
                        }
                    }
                }
            }
        }

        // Walk com atalhos semânticos A→B→C → A→C
        let mut sonho_chain: Vec<String> = Vec::new();
        let mut novas_atalho = 0usize;

        let (grafo_inicial, atalhos) = if let Ok(mut sw) = self.swap_manager.try_lock() {
            sw.importar_causal(causal_pairs.clone());
            let grafo = sw.grafo_palavras();
            let valencias = sw.valencias_palavras();

            let ancora: String = episodios.first()
                .and_then(|ep| ep.palavras.iter()
                    .filter(|w| w.chars().count() >= 3)
                    .max_by(|a, b| {
                        let va = valencias.get(*a).map(|v| v.abs()).unwrap_or(0.0);
                        let vb = valencias.get(*b).map(|v| v.abs()).unwrap_or(0.0);
                        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .cloned())
                .unwrap_or_else(|| "eu".to_string());

            sonho_chain.push(ancora.clone());
            let mut atual = ancora;
            let mut atalhos_novos: Vec<(String, String, f32)> = Vec::new();

            for passo in 0..8usize {
                let vizinhos_b: Vec<(String, f32)> = grafo.get(&atual)
                    .map(|v| v.iter().filter(|(w, _)| w.chars().count() >= 3).cloned().collect())
                    .unwrap_or_default();
                if vizinhos_b.is_empty() { break; }

                let total_peso: f32 = vizinhos_b.iter().map(|(_, p)| p.abs()).sum();
                let mut alvo = rng.gen::<f32>() * total_peso.max(0.001);
                let mut proximo_b = vizinhos_b[0].0.clone();
                for (w, p) in &vizinhos_b {
                    alvo -= p.abs();
                    if alvo <= 0.0 { proximo_b = w.clone(); break; }
                }

                let vizinhos_c: Vec<(String, f32)> = grafo.get(&proximo_b)
                    .map(|v| v.iter().filter(|(w, _)| w.chars().count() >= 3 && w != &atual).cloned().collect())
                    .unwrap_or_default();

                if let Some((proximo_c, peso_bc)) = vizinhos_c.choose(&mut rng) {
                    let ja_existe_ac = grafo.get(&atual)
                        .map(|v| v.iter().any(|(w, _)| w == proximo_c))
                        .unwrap_or(false);
                    if !ja_existe_ac && passo % 2 == 0 {
                        let peso_atalho = (peso_bc * 0.6).clamp(0.20, 0.55);
                        atalhos_novos.push((atual.clone(), proximo_c.clone(), peso_atalho));
                        novas_atalho += 1;
                    }
                    sonho_chain.push(proximo_b.clone());
                    sonho_chain.push(proximo_c.clone());
                    atual = proximo_c.clone();
                } else {
                    sonho_chain.push(proximo_b.clone());
                    atual = proximo_b;
                }
            }
            sonho_chain.dedup();
            (grafo, atalhos_novos)
        } else {
            return (0, None);
        };
        let _ = grafo_inicial;

        if !atalhos.is_empty() {
            if let Ok(mut sw) = self.swap_manager.try_lock() {
                sw.importar_causal(atalhos);
            }
        }

        // STDP noturno
        if let Ok(mut sw) = self.swap_manager.try_lock() {
            let valencias = sw.valencias_palavras();
            if !valencias.is_empty() {
                let dt_s = 1.0_f32 / 200.0;
                for _ in 0..3 { sw.treinar_semantico(300, dt_s, &valencias); }
            }
        }

        // Hipóteses confiáveis → sinapses
        let confiaveis = self.hypothesis_engine.hipoteses_confiaveis();
        if !confiaveis.is_empty() {
            let pares: Vec<(String, String, f32)> = confiaveis.iter()
                .flat_map(|h| h.premissas.iter()
                    .map(|p| (p.clone(), h.conclusao.clone(), h.confianca * 0.4))
                    .collect::<Vec<_>>())
                .collect();
            if let Ok(mut sw) = self.swap_manager.try_lock() {
                sw.importar_causal(pares);
            }
        }

        let total_novas = novas_cruzadas + novas_atalho;
        println!("   ✨ [rem_semantico] {} cruzadas + {} atalhos = {} novas sinapses | {} episódios",
            novas_cruzadas, novas_atalho, total_novas, episodios.len());

        let relato = if sonho_chain.len() >= 3 {
            let trecho: Vec<&String> = sonho_chain.iter().take(6).collect();
            Some(format!("sonhei com {}... e depois {}... e então {}",
                trecho[0],
                trecho.get(2).map(|s| s.as_str()).unwrap_or("algo distante"),
                trecho.last().map(|s| s.as_str()).unwrap_or("silêncio")))
        } else { None };

        (total_novas, relato)
    }

    // ── Memória autobiográfica ────────────────────────────────────────────────

    /// Registra um evento real na memória autobiográfica.
    /// Chamado após reward, punish, learn e respostas marcantes.
    /// Janela deslizante de 50 entradas — as mais antigas são descartadas.
    ///
    /// `valencia`: +1.0 = muito positivo, -1.0 = muito negativo
    pub fn registrar_memoria(&mut self, descricao: String, valencia: f32) {
        self.ego.memorias_autobiograficas.push_back((descricao, valencia.clamp(-1.0, 1.0)));
        if self.ego.memorias_autobiograficas.len() > 50 {
            self.ego.memorias_autobiograficas.pop_front();
        }
        // Atualiza sentimento médio ponderado (EMA com α=0.05)
        let v = valencia.clamp(-1.0, 1.0);
        self.ego.sentimento = (self.ego.sentimento * 0.95 + v * 0.05).clamp(-1.0, 1.0);
    }

    /// Retorna as N memórias mais recentes como strings para contexto de narrativa.
    pub fn memorias_recentes_str(&self, n: usize) -> Vec<String> {
        self.ego.memorias_autobiograficas
            .iter()
            .rev()
            .take(n)
            .map(|(d, v)| {
                if *v > 0.2 { format!("(+) {}", d) }
                else if *v < -0.2 { format!("(-) {}", d) }
                else { d.clone() }
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    // ── Grounding semântico ────────────────────────────────────────────────────

    /// Vincula palavras ao contexto perceptual atual, aumentando grounding scores.
    /// Chamado pelo loop neural (a cada 100 ticks) e pelo chat handler.
    ///
    /// Regras de delta por modalidade:
    ///   visual ativo  → +0.25  (percepção mais rica = grounding mais forte)
    ///   audio ativo   → +0.15
    ///   interoceptivo → +0.08  (sempre ativo — corpo sempre influencia)
    ///
    /// Registra também um EventoEpisodico rico para replay N3/REM.
    pub fn grounding_bind(
        &mut self,
        palavras: &[String],
        padrao_visual: SpikePattern,
        padrao_audio:  SpikePattern,
        emocao:  f32,
        arousal: f32,
        tempo_ms: f64,
    ) {
        use crate::encoding::spike_codec::is_active;
        let visual_ativo = is_active(&padrao_visual);
        let audio_ativo  = is_active(&padrao_audio);

        for palavra in palavras {
            if palavra.len() < 2 { continue; }
            let g = self.grounding.entry(palavra.clone()).or_insert(0.0);
            if visual_ativo { *g = (*g + 0.25).min(1.0); }
            if audio_ativo  { *g = (*g + 0.15).min(1.0); }
            *g = (*g + 0.08).min(1.0); // interoceptivo — sempre contribui
        }

        if palavras.is_empty() { return; }

        // Registra evento episódico rico para replay N3/REM
        let evento = EventoEpisodico {
            palavras:      palavras.to_vec(),
            padrao_visual,
            padrao_audio,
            estado_corpo:  self.ultimo_estado_corpo,
            emocao,
            arousal,
            tempo_ms,
        };
        self.historico_episodico.push_back(evento);

        // Compressão: quando cheio, remove a entrada com menor |emocao| nas primeiras 100
        if self.historico_episodico.len() > 500 {
            if let Some(min_pos) = self.historico_episodico.iter().take(100)
                .enumerate()
                .min_by(|a, b| a.1.emocao.abs().partial_cmp(&b.1.emocao.abs())
                    .unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
            {
                self.historico_episodico.remove(min_pos);
            }
        }
    }

    /// Propaga RPE ao grounding das palavras no neural_context.
    /// RPE > 0 (predição correta) → grounding aumenta.
    /// RPE < 0 (predição errada) → grounding diminui levemente.
    pub fn grounding_rpe(&mut self, rpe: f32) {
        if rpe.abs() < 0.05 { return; }
        let palavras: Vec<String> = self.neural_context.iter().cloned().take(8).collect();
        for w in palavras {
            let g = self.grounding.entry(w).or_insert(0.0);
            if rpe > 0.0 {
                *g = (*g + rpe * 0.05).min(1.0);
            } else {
                *g = (*g + rpe * 0.02).max(0.0);
            }
        }
    }

    /// Decaimento global de grounding. Chamar a cada ~1000 ticks.
    /// Palavras sem reforço perceptual recente perdem grounding lentamente.
    /// 0.999^1000 ≈ 0.37 — leva ~5000 ticks (~25s) para cair à metade.
    pub fn grounding_decay(&mut self) {
        for g in self.grounding.values_mut() {
            *g *= 0.999;
        }
        // Remove palavras com grounding irrisório (< 0.001) para não crescer sem limite
        self.grounding.retain(|_, g| *g >= 0.001);
    }
}

// Esta função traduz o BrainState complexo para o NeuralStatus (JSON).
// neuronios_ativos é lido via AtomicUsize — sem nenhum lock, sem starvation.
// O valor é escrito pelo loop neural (main.rs) a cada 250 steps com spikes_vivos.
pub async fn collect_neural_status(state: &BrainState) -> NeuralStatus {
    let neuronios_ativos = state.neuronios_ativos.load(Ordering::Relaxed);

    // Estimativa de ondas cerebrais baseada no estado neural atual.
    // Derivação biológica aproximada:
    //   alerta (arousal) e emoção → determinam qual banda domina
    //   dopamina → modula beta/gamma (foco e aprendizado)
    //   serotonina → modula alpha (calma e equilíbrio)
    //   ws_atividade → sinal de engajamento externo (treino/chat)
    let (dopa, sero, _nor) = state.neurotransmissores;
    let (_, alerta, emocao) = state.atividade;
    let ws_ativ = state.ws_atividade;

    // Atividade normalizada 0-1
    let atividade_geral = (alerta / 3.5).clamp(0.0, 1.0);
    let engajamento = ws_ativ.clamp(0.0, 1.0);

    // Banda Delta: forte quando ocioso sem engajamento externo (pseudo-sono)
    let delta = ((1.0 - atividade_geral) * (1.0 - engajamento) * 0.8).clamp(0.0, 1.0);

    // Banda Theta: memoria, criatividade — moderada atividade + serotonina
    let theta = (sero * 0.5 * (1.0 - engajamento * 0.5) * 0.9).clamp(0.0, 1.0);

    // Banda Alpha: ocioso alerta — serotonina alta, baixo engajamento externo
    let alpha = (sero * (1.0 - engajamento * 0.8) * atividade_geral * 1.2).clamp(0.0, 1.0);

    // Banda Beta: foco ativo — alta dopamina, engajamento médio/alto
    let beta = (dopa * 0.7 * (atividade_geral * 0.5 + engajamento * 0.5)).clamp(0.0, 1.0);

    // Banda Gamma: aprendizado intenso/chat — engajamento alto + dopamina alta
    let gamma = (dopa * engajamento * emocao.abs() * 1.5).clamp(0.0, 1.0);

    let dominante = if gamma > 0.6 { "gamma".to_string() }
        else if beta > alpha && beta > theta { "beta".to_string() }
        else if alpha > theta { "alpha".to_string() }
        else if theta > delta { "theta".to_string() }
        else { "delta".to_string() };

    NeuralStatus {
        neurotransmissores: NeurochemStatus {
            dopamina: state.neurotransmissores.0,
            serotonina: state.neurotransmissores.1,
            noradrenalina: state.neurotransmissores.2,
        },
        hardware: HardwareStatus {
            cpu_temp: state.hardware.0,
            ram_usage_gb: state.hardware.1,
        },
        ego: EgoStatus {
            pensamentos: state.ego.pensamentos_recentes.iter().cloned().collect(),
            sentimento_atual: state.ego.sentimento,
        },
        atividade: AtividadeStatus {
            step: state.atividade.0,
            alerta: state.atividade.1,
            emocao: state.atividade.2,
        },
        swap: {
            // try_lock único para ler todos os campos de uma vez
            let (total, sinapses, n_fonemas, n_visuais, cap) = state.swap_manager
                .try_lock()
                .map(|g| (
                    g.total_count(),
                    g.sinapses_semanticas_ativas(),
                    g.fonemas_para_id.len(),
                    g.visuais_para_id.len(),
                    g.max_ram_neurons,
                ))
                .unwrap_or((55, 0, 35, 20, 1_000_000));
            SwapStatus {
                neuronios_ativos,
                total_conceitos:    total,
                capacidade_max:     cap,
                sinapses_aprendidas: sinapses,
                fonemas_ativos:     n_fonemas,
                visuais_ativos:     n_visuais,
            }
        },
        ondas: OndasCerebrais {
            delta, theta, alpha, beta, gamma, dominante,
        },
    }
}