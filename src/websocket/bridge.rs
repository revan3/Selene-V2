// src/websocket/bridge.rs

use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex as TokioMutex;
use crate::config::Config;
use crate::storage::swap_manager::SwapManager;
use crate::sensors::SensorFlags;
use crate::encoding::spike_codec::SpikePattern;
use crate::encoding::helix_store::HelixStore;
use crate::brain_zones::occipital::OccipitalLobe;
use crate::learning::hypothesis::HypothesisEngine;
use crate::sensors::audio::WordAccumulator;

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
    pub neuronios_ativos: usize,
    pub capacidade_max: usize,
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
    /// Mapa palavra → valência aprendida (persistente entre sessões de chat)
    pub palavra_valencias: HashMap<String, f32>,
    /// Grafo semântico: palavra → [(palavra_associada, peso)]
    /// Construído via selene_associacoes.py e selene_leitor.py
    pub grafo_associacoes: HashMap<String, Vec<(String, f32)>>,
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
    /// Acumulador temporal de palavras para o pipeline WebSocket (audio_raw).
    /// Integra frames de 46ms vindos do cliente até detectar fronteira de palavra.
    /// Mesma lógica do acumulador interno do audio.rs, mas para áudio externo.
    pub audio_acumulador: WordAccumulator,
    /// Córtex occipital — processa frames visuais da webcam/screen share do browser.
    /// Aplica detecção de movimento (flicker_buffer), contraste e orientação (V1→V2)
    /// antes de gerar o spike pattern que vai para spike_vocab.
    pub occipital: OccipitalLobe,
    /// Tempo visual acumulado (segundos) — passado para visual_sweep como current_time.
    /// Permite que o OccipitalLobe calcule variações temporais entre frames.
    pub visual_time: f32,
}

pub struct EgoVoiceState {
    pub pensamentos_recentes: VecDeque<String>,
    pub sentimento: f32,
    /// Traços de personalidade aprendidos: (nome_do_traço, intensidade 0.0-1.0)
    /// Ex: [("curiosa", 0.7), ("cautelosa", 0.4)]
    /// Atualizados automaticamente após interações com emoção forte (|emocao|>0.4).
    pub tracos: Vec<(String, f32)>,
}

impl BrainState {
    pub fn new(swap: Arc<TokioMutex<SwapManager>>, cfg: &Config, sensor_flags: SensorFlags) -> Self {
        // Pré-carrega léxico se existir.
        // Suporta dois formatos:
        //   Array: [{word, valence, ...}, ...]  (formato legado)
        //   Object: {"categoria": [{word, valence, ...}], ...}  (formato generate_lexicon.py)
        let mut palavra_valencias: HashMap<String, f32> = HashMap::new();
        if let Ok(content) = std::fs::read_to_string("selene_lexicon.json") {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                let mut carregar_entrada = |c: &serde_json::Value| {
                    if let (Some(word), Some(val)) = (c["word"].as_str(), c["valence"].as_f64()) {
                        palavra_valencias.insert(word.to_lowercase(), val as f32);
                    }
                };
                match &json {
                    serde_json::Value::Array(arr) => {
                        for c in arr { carregar_entrada(c); }
                    }
                    serde_json::Value::Object(map) => {
                        for (_cat, lista) in map {
                            if let Some(arr) = lista.as_array() {
                                for c in arr { carregar_entrada(c); }
                            }
                        }
                    }
                    _ => {}
                }
                if !palavra_valencias.is_empty() {
                    println!("📖 Léxico pré-carregado: {} palavras conhecidas.", palavra_valencias.len());
                }
            }
        }

        // Restaura backup de linguagem se existir (grafo + frases)
        let mut grafo_associacoes: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        let mut frases_padrao: Vec<Vec<String>> = Vec::new();
        if let Ok(content) = std::fs::read_to_string("selene_linguagem.json") {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(lingua) = json.get("selene_linguagem_v1") {
                    // Restaura vocabulário → palavra_valencias
                    if let Some(vocab) = lingua.get("vocabulario").and_then(|v| v.as_object()) {
                        for (word, val) in vocab {
                            if let Some(v) = val.as_f64() {
                                // Não sobrescreve palavras já carregadas do selene_lexicon.json
                                palavra_valencias.entry(word.clone()).or_insert(v as f32);
                            }
                        }
                    }

                    // Restaura associações
                    if let Some(assoc) = lingua.get("associacoes").and_then(|v| v.as_object()) {
                        for (w1, vizinhos) in assoc {
                            if let Some(arr) = vizinhos.as_array() {
                                for pair in arr {
                                    if let (Some(w2), Some(peso)) = (
                                        pair.get(0).and_then(|v| v.as_str()),
                                        pair.get(1).and_then(|v| v.as_f64()),
                                    ) {
                                        grafo_associacoes
                                            .entry(w1.clone())
                                            .or_default()
                                            .push((w2.to_string(), peso as f32));
                                    }
                                }
                            }
                        }
                    }
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
                    // Bootstrap do grafo: se o arquivo não tinha associações mas tem frases,
                    // extrai bigrams e trigrams para dar conectividade inicial ao graph walk.
                    // Sem isso o walk encerra no 1º passo e Selene repete sempre a mesma frase.
                    let n_assoc: usize = grafo_associacoes.values().map(|v| v.len()).sum();
                    if n_assoc == 0 && !frases_padrao.is_empty() {
                        for frase in &frases_padrao {
                            for w in frase.windows(2) {
                                let entry = grafo_associacoes.entry(w[0].clone()).or_default();
                                if let Some(p) = entry.iter_mut().find(|(wd, _)| wd == &w[1]) {
                                    p.1 = (p.1 + 0.05).min(1.0);
                                } else if entry.len() < 50 {
                                    entry.push((w[1].clone(), 0.10));
                                }
                            }
                        }
                        let n_boot: usize = grafo_associacoes.values().map(|v| v.len()).sum();
                        println!("🗣️  Linguagem restaurada: {} frases padrão. Grafo bootstrapped: {} associações.",
                            frases_padrao.len(), n_boot);
                    } else if n_assoc > 0 || !frases_padrao.is_empty() {
                        println!("🗣️  Linguagem restaurada: {} associações, {} frases padrão.",
                            n_assoc, frases_padrao.len());
                    }
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

        Self {
            swap_manager: swap,
            config: cfg.clone(),
            neurotransmissores: (0.5, 0.5, 0.5),
            hardware: (40.0, 0.0),
            atividade: (0, 1.0, 0.0),
            ego: EgoVoiceState {
                pensamentos_recentes: pensamentos_persistidos,
                sentimento: 0.0,
                tracos: tracos_persistidos,
            },
            shutdown_requested: false,
            sensor_flags,
            palavra_valencias,
            grafo_associacoes,
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
            mirror_resonance: 0.0,
            ultima_atividade: Instant::now(),
            auto_learn_contagem: HashMap::new(),
            dormindo: false,
            fase_sono: String::new(),
            emocao_palavras: HashMap::new(),
            historico_episodico: VecDeque::with_capacity(500),
            grounding: std::collections::HashMap::new(),
            ultimo_padrao_visual: [0u64; 8],
            ultimo_padrao_audio: [0u64; 8],
            ultimo_estado_corpo: [0.5, 0.5, 0.5, 0.5, 0.0],
            neural_context: VecDeque::with_capacity(20),
            // 512 neurônios = 70% V1 (358) + 30% V2 (154) — coincide com 32×16 pixels do browser
            occipital: OccipitalLobe::new(512, 0.2, cfg),
            visual_time: 0.0,
            hypothesis_engine: HypothesisEngine::new(),
            audio_acumulador: WordAccumulator::new(),
        }
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
// Usa try_lock no swap_manager para NÃO bloquear enquanto mantém o brain lock.
// Se o swap estiver ocupado, retorna 0 para neuronios_ativos (valor será atualizado
// na próxima telemetria). Isso evita starvation do chat handler.
pub async fn collect_neural_status(state: &BrainState) -> NeuralStatus {
    let neuronios_ativos = state.swap_manager
        .try_lock()
        .map(|g| g.ram_count())
        .unwrap_or(0);

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
        swap: SwapStatus {
            neuronios_ativos,
            // Lê o limite real calculado pelo SwapManager (tier de RAM dinâmico)
            capacidade_max: state.swap_manager
                .try_lock()
                .map(|g| g.max_ram_neurons)
                .unwrap_or(1_000_000),
        },
        ondas: OndasCerebrais {
            delta, theta, alpha, beta, gamma, dominante,
        },
    }
}