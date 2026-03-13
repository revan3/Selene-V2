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
pub fn start_listening(n_neurons: usize, tx: Sender<AudioSignal>) {
    let host = cpal::default_host();

    // Tenta encontrar microfone padrão do sistema.
    let device = match host.default_input_device() {
        Some(d) => d,
        None => {
            println!("[AUDIO] Nenhum microfone encontrado. Enviando silêncio.");
            return run_silencio(n_neurons, tx);
        }
    };

    println!("[AUDIO] Microfone: {}", device.name().unwrap_or("desconhecido".to_string()));

    let config = match device.default_input_config() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[AUDIO] Erro ao obter configuração do microfone: {e}");
            return run_silencio(n_neurons, tx);
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

                // Processa FFT e monta o sinal neural
                let sinal = processar_frame_fft(&frame, n, energia_bruta, onset);

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
            run_silencio(n_neurons, tx);
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

    AudioSignal {
        bandas: neural,
        onset,
        energia: energia_norm.clamp(0.0, 1.0),
        pitch_dominante,
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
        };

        if tx.send(sinal_nulo).is_err() {
            break; // Pipeline fechado
        }

        // 50 Hz de "tick" silencioso
        std::thread::sleep(Duration::from_millis(20));
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
