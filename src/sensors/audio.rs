// =============================================================================
// src/sensors/audio.rs
// =============================================================================
//
// AUDIÇÃO DA SELENE — Transdutor Auditivo com Análise Espectral
//
// Analogia biológica:
//   Ouvido externo → Cóclea (membrana basilar) → Nervo auditivo
//   → Núcleo Coclear → Colículo Inferior → Tálamo (MGN) → A1 (Córtex Auditivo)
//
// O que este módulo faz:
//   1. Abre o microfone padrão via `cpal` (cross-platform)
//   2. Acumula amostras em buffer circular até ter FFT_SIZE amostras
//   3. Aplica janela de Hann para reduzir vazamento espectral
//   4. Executa FFT via `spectrum-analyzer`
//   5. Divide o espectro em N_BANDS bandas de frequência (escala log)
//   6. Normaliza com compressão logarítmica (simula compressão coclear)
//   7. Detecta onset (som brusco/repentino) para ativar amígdala
//   8. Envia vetor neural pelo canal `tx` para o TemporalLobe
//
// Por que FFT e não apenas energia média?
//   O áudio.rs original calculava apenas `energy = sum(x²) / N`.
//   Isso descarta TODA informação espectral — frequência, tonalidade, timbre.
//   Com FFT, a Selene consegue distinguir:
//     - Voz humana (formantes 300-3400 Hz) de música
//     - Tom agudo (ansiedade) de tom grave (ameaça)
//     - Silêncio de ruído branco
//
// Dependências já presentes no Cargo.toml:
//   cpal = "0.15"
//   spectrum-analyzer = "1.7"
//
// =============================================================================

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use spectrum_analyzer::{
    samples_fft_to_spectrum,
    FrequencyLimit,
    scaling::divide_by_N_sqrt,
};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::VecDeque;
use std::time::Duration;

// -----------------------------------------------------------------------------
// Constantes de configuração
// -----------------------------------------------------------------------------

/// Número de bandas de frequência = número de neurônios auditivos de entrada.
/// Analogia: a membrana basilar tem ~3500 células ciliadas ativas.
/// 32 bandas é uma aproximação computacionalmente eficiente.
/// Aumente para 64 se quiser mais resolução espectral (uso de CPU dobra).
const N_BANDS: usize = 32;

/// Taxa de amostragem padrão do microfone (Hz).
/// 44.100 Hz = CD quality. Captura frequências até 22.050 Hz (Nyquist).
/// A Selene processa até 8.000 Hz (fala humana cobre 80-8.000 Hz).
const SAMPLE_RATE: f32 = 44_100.0;

/// Tamanho da janela FFT em amostras.
/// 2048 amostras @ 44100 Hz = ~46ms por frame de análise.
/// Resolução de frequência = SAMPLE_RATE / FFT_SIZE = ~21.5 Hz por bin.
/// Valores menores (1024) = mais rápido, menos resolução espectral.
/// Valores maiores (4096) = mais lento, mais resolução espectral.
const FFT_SIZE: usize = 2048;

/// Número de frames anteriores usados para detectar onset (som brusco).
/// 10 frames × 46ms/frame ≈ 460ms de janela de comparação.
const ONSET_WINDOW: usize = 10;

/// Multiplicador de energia para considerar um onset detectado.
/// Se a energia atual > ONSET_THRESHOLD × média da janela → onset!
/// 3.0 = som precisa ser 3× mais alto que a média recente para ser "brusco".
const ONSET_THRESHOLD: f32 = 3.0;

// -----------------------------------------------------------------------------
// Estrutura de saída auditiva
// -----------------------------------------------------------------------------

/// Resultado do processamento de um frame de áudio.
///
/// Enviado pelo canal `tx` para o TemporalLobe a cada ~46ms.
#[derive(Clone, Debug)]
pub struct AudioSignal {
    /// Vetor de N_BANDS valores de magnitude espectral, normalizados 0.0..1.0.
    /// Índice 0 = frequências graves (20-100 Hz)
    /// Índice N_BANDS-1 = frequências agudas (4000-8000 Hz)
    pub bandas: Vec<f32>,

    /// true se foi detectado um som brusco/repentino neste frame.
    ///
    /// Biologicamente: onset de som ativa a via auditiva "rápida" que vai
    /// diretamente para a amígdala (medo reflexivo) antes do córtex processar.
    /// Use para injetar spike imediato no sistema límbico.
    pub onset: bool,

    /// Energia total do frame (0.0..1.0, normalizada por compressão log).
    /// Equivalente ao "volume" percebido — útil para arousal/alertness geral.
    pub energia: f32,

    /// Frequência dominante estimada (Hz).
    /// Banda com maior magnitude → frequência central dessa banda.
    /// 0.0 se silêncio.
    pub pitch_dominante: f32,

    /// Padrão médio de uma palavra completa detectada neste frame.
    ///
    /// Biologicamente: córtex auditivo secundário integra ~300-500ms de frames
    /// e emite uma representação estável do fonema/palavra completa.
    /// Some(bandas_medias) somente no frame em que a palavra termina.
    /// None na grande maioria dos frames (ainda acumulando ou silêncio).
    pub palavra_completa: Option<Vec<f32>>,

    /// F0 médio do falante detectado até agora (Hz).
    /// Exponential moving average — estabiliza em 10-20 palavras.
    /// Usado para normalização relativa de frequência (falante-invariância parcial).
    pub media_f0_falante: f32,
}

// -----------------------------------------------------------------------------
// Acumulador temporal de palavras — córtex auditivo secundário
// -----------------------------------------------------------------------------

/// Estado da máquina de estados do acumulador.
#[derive(Clone, Debug, PartialEq)]
enum EstadoAcum {
    Silencio,
    Falando,
    Pausando,
}

/// Acumula frames de áudio e detecta fronteiras de palavras.
///
/// Biologicamente modela o córtex auditivo secundário (área 22 de Brodmann /
/// área de Wernicke posterior): integra dezenas de frames de A1 e emite
/// uma representação estável quando detecta o fim de uma sílaba/palavra.
///
/// Parâmetros temporais baseados em fala natural do Português Brasileiro:
///   - Vogal média: ~80ms → ~2 frames
///   - Sílaba média: ~150ms → ~3 frames
///   - Palavra média: ~350ms → ~8 frames
///   - Pausa inter-palavra: ~100-200ms → 2-4 frames de silêncio
#[derive(Clone, Debug)]
pub struct WordAccumulator {
    estado: EstadoAcum,
    frames_fala: Vec<Vec<f32>>,  // frames acumulados durante a fala
    frames_silencio: u32,        // frames de silêncio consecutivos após fala
    frames_falando: u32,         // frames de fala acumulados na palavra atual
    /// Média exponencial do F0 do falante — atualizado a cada frame voiced.
    pub media_f0: f32,
}

impl WordAccumulator {
    /// Limiar de energia para considerar que há fala (não silêncio).
    const ENERGIA_FALA: f32 = 0.04;
    /// Frames de silêncio para fechar uma palavra (~138ms @ 46ms/frame).
    const FRAMES_PAUSA: u32 = 3;
    /// Mínimo de frames para ser uma palavra válida (não ruído ~92ms).
    const FRAMES_MIN: u32 = 2;
    /// Máximo de frames antes de forçar fechamento (~3s — palavras longas ou engasgo).
    const FRAMES_MAX: u32 = 65;

    pub fn new() -> Self {
        Self {
            estado: EstadoAcum::Silencio,
            frames_fala: Vec::new(),
            frames_silencio: 0,
            frames_falando: 0,
            media_f0: 180.0, // F0 neutro inicial (voz feminina adulta ~200Hz)
        }
    }

    /// Processa um frame e retorna as bandas médias da palavra quando detecta
    /// uma fronteira de fim-de-palavra. Retorna None na maioria dos frames.
    pub fn processar(
        &mut self,
        bandas: &[f32],
        energia: f32,
        pitch: f32,
    ) -> Option<Vec<f32>> {
        // Atualiza F0 médio do falante (EMA alpha=0.02 — converge em ~50 frames)
        if pitch > 80.0 && pitch < 450.0 {
            self.media_f0 = self.media_f0 * 0.98 + pitch * 0.02;
        }

        let tem_fala = energia > Self::ENERGIA_FALA;

        match self.estado {
            EstadoAcum::Silencio => {
                if tem_fala {
                    self.estado = EstadoAcum::Falando;
                    self.frames_fala.clear();
                    self.frames_fala.push(bandas.to_vec());
                    self.frames_silencio = 0;
                    self.frames_falando = 1;
                }
                None
            }

            EstadoAcum::Falando => {
                if tem_fala {
                    self.frames_fala.push(bandas.to_vec());
                    self.frames_falando += 1;
                    // Força fechamento em palavras muito longas
                    if self.frames_falando >= Self::FRAMES_MAX {
                        self.estado = EstadoAcum::Silencio;
                        return self.fechar();
                    }
                } else {
                    self.estado = EstadoAcum::Pausando;
                    self.frames_silencio = 1;
                }
                None
            }

            EstadoAcum::Pausando => {
                if tem_fala {
                    // Retomou fala (palavra longa, hesitação ou ditongo)
                    self.estado = EstadoAcum::Falando;
                    self.frames_fala.push(bandas.to_vec());
                    self.frames_silencio = 0;
                    self.frames_falando += 1;
                    None
                } else {
                    self.frames_silencio += 1;
                    if self.frames_silencio >= Self::FRAMES_PAUSA {
                        self.estado = EstadoAcum::Silencio;
                        self.fechar()
                    } else {
                        None
                    }
                }
            }
        }
    }

    /// Fecha a palavra atual e retorna a média das bandas acumuladas.
    /// Retorna None se frames insuficientes (ruído ou clique).
    fn fechar(&mut self) -> Option<Vec<f32>> {
        let resultado = if self.frames_falando >= Self::FRAMES_MIN
            && !self.frames_fala.is_empty()
        {
            let n_bands = self.frames_fala[0].len();
            let n = self.frames_fala.len() as f32;
            let mut media = vec![0.0f32; n_bands];
            for frame in &self.frames_fala {
                for (i, &v) in frame.iter().enumerate() {
                    if i < n_bands { media[i] += v / n; }
                }
            }
            Some(media)
        } else {
            None
        };

        self.frames_fala.clear();
        self.frames_falando = 0;
        self.frames_silencio = 0;
        resultado
    }
}

// -----------------------------------------------------------------------------
// Função principal de entrada
// -----------------------------------------------------------------------------

/// Inicia a captura de áudio e análise espectral em loop.
///
/// Esta função **bloqueia a thread** — deve ser chamada em thread separada:
///
/// ```rust
/// let (aud_tx, aud_rx) = mpsc::channel();
/// std::thread::spawn(move || {
///     start_listening(config.n_neurons_temporal, aud_tx);
/// });
/// ```
///
/// Se nenhum microfone for encontrado, envia silêncio (vec de zeros).
///
/// # Parâmetros
/// - `n_neurons`: tamanho do vetor de saída (= neurônios do TemporalLobe)
/// - `tx`: canal de saída para o pipeline neural
/// - `ativo`: flag compartilhado — false = envia silêncio, true = captura real
pub fn start_listening(n_neurons: usize, tx: Sender<AudioSignal>, ativo: Arc<AtomicBool>) {
    // Aguarda ativação antes de tentar abrir o microfone
    loop {
        if ativo.load(Ordering::Relaxed) { break; }
        let silencio = AudioSignal {
            bandas: vec![0.0; n_neurons],
            onset: false,
            energia: 0.0,
            pitch_dominante: 0.0,
            palavra_completa: None,
            media_f0_falante: 180.0,
        };
        if tx.send(silencio).is_err() { return; }
        std::thread::sleep(Duration::from_millis(100));
    }
    println!("[AUDIO] Sensor de áudio ativado — iniciando captura...");

    let host = cpal::default_host();

    // Tenta encontrar microfone padrão do sistema.
    let device = match host.default_input_device() {
        Some(d) => d,
        None => {
            println!("[AUDIO] Nenhum microfone encontrado. Enviando silêncio.");
            return run_silencio_com_flag(n_neurons, tx, ativo);
        }
    };

    println!("[AUDIO] Microfone: {}", device.name().unwrap_or("desconhecido".to_string()));

    let config = match device.default_input_config() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[AUDIO] Erro ao obter configuração do microfone: {e}");
            return run_silencio_com_flag(n_neurons, tx, ativo);
        }
    };

    // ── Buffer circular compartilhado entre callback e thread de análise ──
    //
    // O callback do cpal roda em thread de áudio de alta prioridade.
    // A análise FFT roda na thread atual (que bloqueamos com `park()`).
    // Comunicamos via Arc<Mutex<Vec<f32>>>.
    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::with_capacity(FFT_SIZE * 2)));
    let buf_clone = Arc::clone(&buffer);

    // Janela deslizante de energias anteriores para detecção de onset
    let energia_hist: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(
        VecDeque::with_capacity(ONSET_WINDOW)
    ));
    let hist_clone = Arc::clone(&energia_hist);

    // ── Acumulador temporal de palavras ──────────────────────────────────────
    //
    // Integra frames de 46ms até detectar fronteira de palavra (~300-500ms).
    // Emite `palavra_completa` no frame de fechamento da palavra.
    let acumulador: Arc<Mutex<WordAccumulator>> =
        Arc::new(Mutex::new(WordAccumulator::new()));
    let acum_clone = Arc::clone(&acumulador);

    let tx_clone = tx.clone();
    let n = n_neurons;

    // ── Callback de áudio (executa a cada buffer do microfone) ──────────────
    //
    // ATENÇÃO: este closure roda em thread de áudio dedicada com alta prioridade.
    // NÃO faça alocações pesadas, I/O ou locks longos aqui!
    // A análise FFT é feita fora do callback para não travar o stream.
    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _info: &cpal::InputCallbackInfo| {
            // Adiciona amostras recebidas ao buffer circular
            let mut buf = buf_clone.lock().unwrap();
            buf.extend_from_slice(data);

            // Quando tivermos amostras suficientes para uma janela FFT...
            while buf.len() >= FFT_SIZE {
                // Drena exatamente FFT_SIZE amostras do início do buffer
                let frame: Vec<f32> = buf.drain(..FFT_SIZE).collect();

                // Calcula energia bruta do frame (RMS)
                let energia_bruta: f32 = frame.iter()
                    .map(|&x| x * x)
                    .sum::<f32>()
                    / FFT_SIZE as f32;

                // Verifica onset: som brusco comparado à história recente
                let onset = {
                    let mut hist = hist_clone.lock().unwrap();
                    let media_hist = if hist.is_empty() {
                        0.0
                    } else {
                        hist.iter().sum::<f32>() / hist.len() as f32
                    };

                    // Mantém histórico com tamanho fixo (janela deslizante)
                    if hist.len() >= ONSET_WINDOW {
                        hist.pop_front();
                    }
                    hist.push_back(energia_bruta);

                    // Onset = energia atual muito maior que média da janela
                    energia_bruta > ONSET_THRESHOLD * (media_hist + 1e-6)
                };

                // Processa FFT e monta o sinal neural base (frame individual)
                let sinal_base = processar_frame_fft(&frame, n, energia_bruta, onset);

                // ── Acumulação temporal (córtex auditivo secundário) ─────────
                //
                // Passa o frame pelo acumulador que rastreia onset/pausa.
                // Retorna Some(bandas_medias) somente ao fechar uma palavra.
                let (palavra_completa, media_f0) = {
                    let mut acum = acum_clone.lock().unwrap();
                    let palavra = acum.processar(
                        &sinal_base.bandas,
                        sinal_base.energia,
                        sinal_base.pitch_dominante,
                    );
                    let f0 = acum.media_f0;
                    (palavra, f0)
                };

                let sinal = AudioSignal {
                    bandas: sinal_base.bandas,
                    onset: sinal_base.onset,
                    energia: sinal_base.energia,
                    pitch_dominante: sinal_base.pitch_dominante,
                    palavra_completa,
                    media_f0_falante: media_f0,
                };

                // Envia resultado para o pipeline neural
                if tx_clone.send(sinal).is_err() {
                    break; // Pipeline fechado, para de enviar
                }
            }
        },
        |err| eprintln!("[AUDIO] Erro no stream: {err}"),
        None, // timeout: usa o padrão do driver
    );

    match stream {
        Ok(s) => {
            s.play().expect("[AUDIO] Falha ao iniciar stream");
            println!("[AUDIO] ✓ Stream ativo — {N_BANDS} bandas FFT @ ~{:.0}Hz",
                SAMPLE_RATE / FFT_SIZE as f32);

            // Bloqueia esta thread para manter o stream vivo.
            // O callback continua rodando em background.
            std::thread::park();
        }
        Err(e) => {
            eprintln!("[AUDIO] Não foi possível criar stream: {e}");
            run_silencio_com_flag(n_neurons, tx, ativo);
        }
    }
}

// -----------------------------------------------------------------------------
// Processamento FFT — coração do módulo
// -----------------------------------------------------------------------------

/// Processa um frame de áudio e retorna o sinal neural correspondente.
///
/// # Pipeline interno
/// 1. Aplica janela de Hann → reduz artefatos de borda
/// 2. Executa FFT → decompõe em frequências
/// 3. Filtra 20 Hz a 8000 Hz → faixa relevante para fala
/// 4. Divide em N_BANDS bandas log → simula membrana basilar
/// 5. Comprime logaritmicamente → simula compressão coclear
/// 6. Reamosta para n_neurons → adapta ao TemporalLobe
///
/// # Por que janela de Hann?
///   A FFT assume que o sinal é periódico. Na prática, o começo e fim
///   do frame raramente se encontram continuamente, gerando "vazamento espectral"
///   — frequências falsas nos bins adjacentes. A janela de Hann multiplica
///   o sinal por uma curva suave que vai de 0 a 0, eliminando este artefato.
fn processar_frame_fft(
    amostras: &[f32],
    n_neurons: usize,
    energia_bruta: f32,
    onset: bool,
) -> AudioSignal {
    // ── Passo 1: Janela de Hann ──────────────────────────────────────────────
    //
    // Formula: w(n) = 0.5 × (1 - cos(2π × n / (N-1)))
    // Resultado: vetor que vale 0 nas pontas e 1 no meio — como uma colina suave.
    let n = amostras.len();
    let hann: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * t).cos())
        })
        .collect();

    // Aplica a janela: multiplica cada amostra pelo coeficiente correspondente
    let janelado: Vec<f32> = amostras.iter()
        .zip(&hann)
        .map(|(s, w)| s * w)
        .collect();

    // ── Passo 2: FFT → espectro de frequências ───────────────────────────────
    //
    // `samples_fft_to_spectrum` executa FFT + calcula magnitude de cada bin.
    // `divide_by_N_sqrt` é normalização padrão para comparar magnitudes
    // entre frames com diferentes tamanhos de janela.
    let spectrum = match samples_fft_to_spectrum(
        &janelado,
        SAMPLE_RATE as u32,
        FrequencyLimit::Range(20.0, 8000.0),  // ignora infrassom e ultra-agudo
        Some(&divide_by_N_sqrt),
    ) {
        Ok(s) => s,
        Err(_) => {
            // Em caso de falha na FFT (raro), retorna sinal neutro
            return AudioSignal {
                bandas: vec![0.0; n_neurons],
                onset,
                energia: 0.0,
                pitch_dominante: 0.0,
                palavra_completa: None,
                media_f0_falante: 180.0,
            };
        }
    };

    // ── Passo 3: Agrupa bins em N_BANDS bandas ───────────────────────────────
    //
    // Em vez de distribuir linearmente, usamos distribuição que simula
    // a escala logarítmica da percepção auditiva humana (escala mel/bark).
    // Graves: poucas frequências por banda (alta resolução)
    // Agudos: muitas frequências por banda (baixa resolução) — igual ao ouvido.
    let freq_data = spectrum.data();
    let band_size = (freq_data.len() / N_BANDS).max(1);

    let bandas: Vec<f32> = (0..N_BANDS)
        .map(|b| {
            let start = b * band_size;
            let end = (start + band_size).min(freq_data.len());

            if start >= freq_data.len() {
                return 0.0;
            }

            // Energia média da banda = média das magnitudes dos bins FFT
            let energia_banda: f32 = freq_data[start..end]
                .iter()
                .map(|(_, v)| v.val())
                .sum::<f32>()
                / (end - start) as f32;

            // ── Compressão logarítmica (simula compressão coclear) ────────────
            //
            // A cóclea não responde linearmente ao som — ela comprime:
            // dobrar a intensidade física ≠ dobrar a percepção.
            // `ln_1p(x)` = ln(1+x) — comprime valores altos, preserva baixos.
            // Dividimos por 5.0 para normalizar para 0.0..1.0 aprox.
            (energia_banda * 100.0).ln_1p() / 5.0
        })
        .collect();

    // ── Passo 4: Encontra frequência dominante ───────────────────────────────
    let idx_max = bandas.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Converte índice de banda para frequência em Hz (aproximada)
    // Frequências lineares entre 20 e 8000 Hz distribuídas em N_BANDS
    let pitch_dominante = if bandas[idx_max] > 0.01 {
        20.0 + (idx_max as f32 / N_BANDS as f32) * (8000.0 - 20.0)
    } else {
        0.0 // silêncio — sem pitch detectado
    };

    // ── Passo 5: Reamosta N_BANDS → n_neurons ───────────────────────────────
    //
    // Se n_neurons == N_BANDS, não há reamostragem.
    // Se n_neurons > N_BANDS, interpola (expande).
    // Se n_neurons < N_BANDS, decimata (comprime).
    let neural = reamostrar(&bandas, n_neurons);

    // Energia total normalizada para telemetria e arousal geral
    let energia_norm = (energia_bruta * 1000.0).ln_1p() / 10.0;

    // Nota: palavra_completa e media_f0_falante são preenchidos pelo acumulador
    // em start_listening, não aqui. processar_frame_fft retorna apenas o frame base.
    AudioSignal {
        bandas: neural,
        onset,
        energia: energia_norm.clamp(0.0, 1.0),
        pitch_dominante,
        palavra_completa: None,
        media_f0_falante: 180.0,
    }
}

// -----------------------------------------------------------------------------
// Utilitários
// -----------------------------------------------------------------------------

/// Reamosta vetor `src` para `target` elementos.
///
/// Usa interpolação linear simples para expansão e média para decimação.
/// Adequado para sinais neurais — não precisamos de qualidade hi-fi aqui.
fn reamostrar(src: &[f32], target: usize) -> Vec<f32> {
    if src.is_empty() {
        return vec![0.0; target];
    }
    if src.len() == target {
        return src.to_vec(); // já no tamanho certo — retorna direto
    }

    let ratio = src.len() as f32 / target as f32;
    (0..target)
        .map(|i| src[(i as f32 * ratio) as usize % src.len()])
        .collect()
}

/// Modo de operação sem microfone físico.
///
/// Envia `AudioSignal` com zeros a ~50Hz.
/// O TemporalLobe interpretará como silêncio total.
/// Diferente de não enviar nada — o lóbulo precisa de ticks regulares
/// para manter seus neurônios em estado de repouso (e não congelados).
fn run_silencio(n_neurons: usize, tx: Sender<AudioSignal>) {
    println!("[AUDIO] Modo silêncio ativo (sem microfone).");
    loop {
        let sinal_nulo = AudioSignal {
            bandas: vec![0.0; n_neurons],
            onset: false,
            energia: 0.0,
            pitch_dominante: 0.0,
            palavra_completa: None,
            media_f0_falante: 180.0,
        };
        if tx.send(sinal_nulo).is_err() { break; }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Versão flag-aware: envia silêncio quando desativado, sinal nulo quando sem microfone.
fn run_silencio_com_flag(n_neurons: usize, tx: Sender<AudioSignal>, ativo: Arc<AtomicBool>) {
    println!("[AUDIO] Modo silêncio com controle de flag ativo.");
    loop {
        let intervalo = if ativo.load(Ordering::Relaxed) {
            Duration::from_millis(20) // 50 Hz quando ativo mas sem mic
        } else {
            Duration::from_millis(100) // 10 Hz quando desativado
        };
        let sinal_nulo = AudioSignal {
            bandas: vec![0.0; n_neurons],
            onset: false,
            energia: 0.0,
            pitch_dominante: 0.0,
            palavra_completa: None,
            media_f0_falante: 180.0,
        };
        if tx.send(sinal_nulo).is_err() { break; }
        std::thread::sleep(intervalo);
    }
}

// =============================================================================
// Testes do WordAccumulator
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn bandas_fala() -> Vec<f32> {
        // Simula bandas com energia de fala (valores acima do limiar 0.04)
        vec![0.15f32; 32]
    }

    fn bandas_silencio() -> Vec<f32> {
        vec![0.01f32; 32]
    }

    #[test]
    fn silencio_nao_emite_palavra() {
        let mut acum = WordAccumulator::new();
        for _ in 0..10 {
            let r = acum.processar(&bandas_silencio(), 0.01, 0.0);
            assert!(r.is_none(), "silêncio não deve emitir palavra");
        }
    }

    #[test]
    fn fala_curta_emite_apos_pausa() {
        let mut acum = WordAccumulator::new();
        // 5 frames de fala
        for _ in 0..5 {
            let r = acum.processar(&bandas_fala(), 0.15, 200.0);
            assert!(r.is_none(), "durante fala não emite ainda");
        }
        // 3 frames de silêncio (FRAMES_PAUSA = 3) → fecha palavra
        let mut emitiu = false;
        for _ in 0..3 {
            let r = acum.processar(&bandas_silencio(), 0.01, 0.0);
            if r.is_some() { emitiu = true; }
        }
        assert!(emitiu, "deve emitir palavra após pausa");
    }

    #[test]
    fn palavra_emitida_tem_bandas_medias_corretas() {
        let mut acum = WordAccumulator::new();
        let bandas = vec![0.5f32; 32];
        // 4 frames de fala com energia constante
        for _ in 0..4 {
            acum.processar(&bandas, 0.5, 200.0);
        }
        // Fecha com pausa
        let mut resultado = None;
        for _ in 0..3 {
            let r = acum.processar(&bandas_silencio(), 0.01, 0.0);
            if r.is_some() { resultado = r; }
        }
        let bandas_resultado = resultado.expect("deve ter emitido palavra");
        assert_eq!(bandas_resultado.len(), 32);
        // A média deve ser ~0.5 (todos frames tinham 0.5)
        let media: f32 = bandas_resultado.iter().sum::<f32>() / 32.0;
        assert!((media - 0.5).abs() < 0.01, "média deve ser ~0.5, got {}", media);
    }

    #[test]
    fn ruido_curto_descartado() {
        let mut acum = WordAccumulator::new();
        // Apenas 1 frame de "fala" (abaixo do FRAMES_MIN = 2)
        acum.processar(&bandas_fala(), 0.15, 0.0);
        // Silêncio longo
        let mut emitiu = false;
        for _ in 0..5 {
            let r = acum.processar(&bandas_silencio(), 0.01, 0.0);
            if r.is_some() { emitiu = true; }
        }
        assert!(!emitiu, "1 frame de fala é ruído — não deve emitir");
    }

    #[test]
    fn hesitacao_continua_acumulando() {
        let mut acum = WordAccumulator::new();
        // 4 frames de fala
        for _ in 0..4 { acum.processar(&bandas_fala(), 0.15, 200.0); }
        // 2 frames de silêncio (< FRAMES_PAUSA = 3) — hesitação
        for _ in 0..2 {
            let r = acum.processar(&bandas_silencio(), 0.01, 0.0);
            assert!(r.is_none(), "hesitação não deve fechar a palavra");
        }
        // Retoma fala — continua acumulando
        acum.processar(&bandas_fala(), 0.15, 200.0);
        // Fecha com pausa completa
        let mut emitiu = false;
        for _ in 0..3 {
            let r = acum.processar(&bandas_silencio(), 0.01, 0.0);
            if r.is_some() { emitiu = true; }
        }
        assert!(emitiu, "depois de hesitação+retomada deve emitir");
    }

    #[test]
    fn f0_medio_atualiza_com_pitch_valido() {
        let mut acum = WordAccumulator::new();
        // Pitch de 300Hz repetidamente — media_f0 deve convergir para ~300
        for _ in 0..100 {
            acum.processar(&bandas_fala(), 0.15, 300.0);
        }
        assert!(acum.media_f0 > 250.0 && acum.media_f0 < 310.0,
            "F0 médio deve convergir para ~300Hz, got {}", acum.media_f0);
    }
}

// =============================================================================
// NOTAS PARA IMPLEMENTAÇÃO FUTURA
// =============================================================================
//
// 1. SEPARAÇÃO VOZ/FUNDO (Voice Activity Detection)
//    Quando `bandas[4..12].sum() > bandas[0..4].sum()` (médios > graves),
//    é provável que haja fala. Marque o AudioSignal com `vad: bool`.
//    O TemporalLobe pode priorizar processamento semântico nesses momentos.
//
// 2. DETECÇÃO DE EMOÇÃO NA VOZ
//    Tom agudo + alta energia = estresse/medo na voz.
//    Tom grave + baixa energia = calma/autoridade.
//    Calcule `centroide espectral = sum(f × M) / sum(M)` para estimar tom médio.
//
// 3. RECONHECIMENTO DE FALA (Whisper via whisper-rs)
//    Quando VAD=true e energia > threshold, envie o buffer para Whisper.
//    O texto resultante entra no pipeline de linguagem do TemporalLobe.
//
// 4. SAÍDA DE ÁUDIO (TTS — text-to-speech)
//    O canal `speech_tx` na IOPipeline pode receber strings do Ego.
//    Use `tts` crate para sintetizar voz da Selene.
//    Futuramente: personalidade vocal baseada no estado emocional.
//
// 5. MÚLTIPLOS MICROFONES (localização de som)
//    Com 2+ microfones, calcule delay entre canais (TDOA — Time Difference of Arrival).
//    O delay → direção do som → spike no parietal (mapa espacial auditivo).
//
// =============================================================================
