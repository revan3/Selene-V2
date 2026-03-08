// =============================================================================
// src/io/pipeline.rs
// =============================================================================
//
// BARRAMENTO SENSÓRIO-MOTOR DA SELENE — IOPipeline
//
// Analogia biológica:
//   Tálamo como hub central de informação sensorial:
//   - Toda informação sensorial (exceto olfato) passa pelo tálamo antes do córtex
//   - O tálamo não apenas retransmite — filtra, prioriza e modula os sinais
//   - O loop cortico-talâmico permite que o córtex "peça" ao tálamo para focar em algo
//
//   A IOPipeline é mais primitiva que o tálamo — é o nervo periférico.
//   Ela coleta os sinais dos sensores e os disponibiliza para o tálamo processar.
//
// O que este módulo faz:
//   1. Unifica todos os canais de entrada (câmera, microfone, texto, hardware)
//   2. Disponibiliza canais de saída (motor, fala, WebSocket)
//   3. Provê método `poll()` que drena o mais recente de cada canal
//   4. Broadcast de eventos para módulos que queiram "ouvir" sem consumir o canal
//
// Padrão de uso no main.rs:
//   ```rust
//   // Inicialização:
//   let (pipeline, vis_tx, aud_tx, txt_tx) = IOPipeline::new();
//
//   // Em threads separadas:
//   std::thread::spawn(move || VisualTransducer::new(N).run(vis_tx));
//   std::thread::spawn(move || audio::start_listening(N, aud_tx));
//
//   // No loop neural:
//   let (frame_visual, frame_audio, texto) = pipeline.poll();
//   if let Some(visual) = frame_visual { thalamus.relay_visual(visual) }
//   if let Some(audio)  = frame_audio  { thalamus.relay_audio(audio) }
//   ```
//
// Por que `poll()` em vez de bloqueante?
//   O loop neural tem frequência fixa (ex: 200Hz = 5ms por tick).
//   Se usássemos `recv()` bloqueante, o loop esperaria pelo próximo frame
//   da câmera (~33ms a 30fps) → o loop inteiro rodaria a apenas 30Hz.
//   Com `try_recv()` não-bloqueante, o loop mantém 200Hz e processa
//   frames quando disponíveis, pulando ticks sem novo frame.
//
// =============================================================================

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use tokio::sync::broadcast;

// Importa o tipo AudioSignal do módulo de áudio
// Se AudioSignal não existir ainda, use Vec<f32> por ora
use crate::sensors::audio::AudioSignal;

// -----------------------------------------------------------------------------
// Eventos de pipeline (para broadcast)
// -----------------------------------------------------------------------------

/// Evento que pode ser transmitido pelo barramento para múltiplos consumidores.
///
/// Diferença de canal `mpsc` normal:
///   - `mpsc`: um produtor, um consumidor, mensagem consumida por um só
///   - `broadcast`: um produtor, múltiplos consumidores, cada um recebe a cópia
///
/// Usado para que módulos como o `MetaCognitive` possam "ouvir" eventos
/// sem interferir no fluxo principal do loop neural.
#[derive(Clone, Debug)]
pub enum EventoPipeline {
    /// Novo frame visual disponível
    Visual(Vec<f32>),

    /// Novo frame de áudio disponível
    Audio(Vec<f32>),

    /// Novo texto recebido (do usuário ou de TTS externo)
    Texto(String),

    /// Onset de som detectado (som brusco — ativa amígdala)
    OnsetAudio,

    /// Comando motor emitido pelo sistema
    Motor(Vec<f32>),

    /// Texto a ser "falado" pela Selene (para TTS)
    Fala(String),

    /// Sinaliza encerramento do sistema (todas as threads devem parar)
    Encerrar,
}

// -----------------------------------------------------------------------------
// Estrutura principal
// -----------------------------------------------------------------------------

/// Barramento de entrada/saída — hub central de comunicação sensório-motor.
///
/// Criado uma vez no main.rs e usado ao longo de todo o ciclo de vida do sistema.
pub struct IOPipeline {
    // ── Receptores de entrada (lidos pelo loop neural) ──────────────────────

    /// Frames visuais da câmera (Vec<f32> com luminância normalizada 0..1).
    /// Frequência: ~30Hz (câmera física) ou ~30Hz simulado (placeholder).
    visual_rx: Receiver<Vec<f32>>,

    /// Frames de áudio do microfone (AudioSignal com bandas FFT e metadados).
    /// Frequência: ~22Hz (FFT_SIZE=2048 @ 44100Hz).
    audio_rx: Receiver<AudioSignal>,

    /// Texto de entrada (do usuário via WebSocket, stdin ou agente externo).
    /// Frequência: irregular (quando o usuário digita algo).
    texto_rx: Receiver<String>,

    // ── Transmissores de saída (escritos pelo loop neural) ──────────────────

    /// Comandos para sistema motor (braço robótico, síntese de movimento, etc.).
    /// Atualmente não conectado a hardware — reservado para expansão futura.
    pub motor_tx: Sender<Vec<f32>>,

    /// Texto para síntese de voz (TTS).
    /// Quando o Ego gerar uma frase, envie aqui para ser vocalizada.
    pub fala_tx: Sender<String>,

    // ── Broadcast (múltiplos assinantes podem receber o mesmo evento) ───────

    /// Canal de broadcast para eventos que múltiplos módulos querem observar.
    /// Ex: MetaCognitive, Logger, WebSocket dashboard — todos recebem cópias.
    broadcast_tx: broadcast::Sender<EventoPipeline>,
}

impl IOPipeline {
    // -------------------------------------------------------------------------
    // Construção
    // -------------------------------------------------------------------------

    /// Cria um novo pipeline e retorna os transmissores para uso nas threads de sensor.
    ///
    /// # Retorno
    /// Tupla `(IOPipeline, visual_tx, audio_tx, texto_tx)`:
    /// - `IOPipeline`: o hub central — fique com este no main.rs
    /// - `visual_tx`: passe para a thread da câmera (`VisualTransducer::run`)
    /// - `audio_tx`: passe para a thread do microfone (`audio::start_listening`)
    /// - `texto_tx`: passe para a thread do WebSocket ou stdin
    ///
    /// # Uso
    /// ```rust
    /// let (mut pipeline, vis_tx, aud_tx, txt_tx) = IOPipeline::new();
    ///
    /// std::thread::spawn(move || {
    ///     VisualTransducer::new(config.n_neurons).run(vis_tx);
    /// });
    /// std::thread::spawn(move || {
    ///     audio::start_listening(config.n_neurons, aud_tx);
    /// });
    /// ```
    pub fn new() -> (
        Self,
        Sender<Vec<f32>>,    // visual_tx — para thread da câmera
        Sender<AudioSignal>, // audio_tx  — para thread do microfone
        Sender<String>,      // texto_tx  — para WebSocket/stdin
    ) {
        // Canais de entrada (sensores → pipeline)
        // Buffer de 60 frames para absorver variações de timing
        // (câmera 30fps × 2 segundos de buffer = 60 frames)
        let (vis_tx, vis_rx) = mpsc::channel();
        let (aud_tx, aud_rx) = mpsc::channel();
        let (txt_tx, txt_rx) = mpsc::channel();

        // Canais de saída (pipeline → atuadores)
        let (motor_tx, _motor_rx) = mpsc::channel();
        let (fala_tx, _fala_rx)   = mpsc::channel();

        // Canal de broadcast (capacidade: 64 eventos em buffer)
        // Se um assinante estiver lento e o buffer encher, eventos antigos são descartados.
        let (broadcast_tx, _) = broadcast::channel(64);

        let pipeline = Self {
            visual_rx: vis_rx,
            audio_rx:  aud_rx,
            texto_rx:  txt_rx,
            motor_tx,
            fala_tx,
            broadcast_tx,
        };

        (pipeline, vis_tx, aud_tx, txt_tx)
    }

    // -------------------------------------------------------------------------
    // Poll — leitura não-bloqueante de todos os canais
    // -------------------------------------------------------------------------

    /// Drena todos os canais de entrada e retorna o dado mais recente de cada um.
    ///
    /// # Comportamento de drenagem
    /// Se a câmera enviou 5 frames desde o último tick, os 4 primeiros são
    /// descartados e apenas o mais recente é retornado.
    /// Isso é intencional: o loop neural processa a REALIDADE ATUAL,
    /// não o backlog de frames antigos.
    ///
    /// Analogia biológica: o tálamo não processa cada fotão individualmente —
    /// integra a cena atual em ~50ms de janela temporal.
    ///
    /// # Retorno
    /// `(visual, audio, texto)` — cada campo é `Some(dado)` ou `None` se vazio.
    pub fn poll(&mut self) -> (
        Option<Vec<f32>>,   // frame visual (luminância normalizada)
        Option<AudioSignal>,// frame de áudio (bandas FFT + metadados)
        Option<String>,     // texto de entrada
    ) {
        let visual = Self::drenar_ultimo(&self.visual_rx);
        let audio  = Self::drenar_ultimo(&self.audio_rx);
        let texto  = self.texto_rx.try_recv().ok(); // texto: pega apenas o próximo

        // Propaga onset de áudio para broadcast (para amígdala reagir)
        if let Some(ref sig) = audio {
            if sig.onset {
                // Onset detectado — broadcast para todos os assinantes
                let _ = self.broadcast_tx.send(EventoPipeline::OnsetAudio);
            }
        }

        (visual, audio, texto)
    }

    // -------------------------------------------------------------------------
    // Broadcast — assinatura de eventos
    // -------------------------------------------------------------------------

    /// Retorna um receptor de broadcast.
    ///
    /// Cada chamada retorna um receptor INDEPENDENTE — módulos diferentes
    /// podem assinar e cada um recebe cópias de todos os eventos.
    ///
    /// # Uso para MetaCognitive
    /// ```rust
    /// let mut meta_rx = pipeline.assinar_eventos();
    /// std::thread::spawn(move || {
    ///     loop {
    ///         match meta_rx.try_recv() {
    ///             Ok(EventoPipeline::OnsetAudio) => meta.registrar_sobressalto(),
    ///             Ok(EventoPipeline::Visual(v))  => meta.atualizar_foco_visual(&v),
    ///             _ => {}
    ///         }
    ///     }
    /// });
    /// ```
    pub fn assinar_eventos(&self) -> broadcast::Receiver<EventoPipeline> {
        self.broadcast_tx.subscribe()
    }

    /// Publica um evento no broadcast (para todos os assinantes).
    ///
    /// Útil para o loop neural notificar módulos secundários de eventos importantes.
    /// Ex: quando o Ego tomar uma decisão importante, publica um evento.
    pub fn publicar(&self, evento: EventoPipeline) {
        let _ = self.broadcast_tx.send(evento);
        // Ignora erro se não há assinantes — isso é esperado e OK.
    }

    // -------------------------------------------------------------------------
    // Métodos de saída
    // -------------------------------------------------------------------------

    /// Envia comando motor (para sistema físico futuro ou simulação).
    ///
    /// Formato: Vec<f32> com N valores -1.0..1.0, um por "músculo" ou eixo.
    /// Ex: robô com 6 articulações → vec![0.5, -0.3, 0.0, 0.8, -0.1, 0.2]
    pub fn enviar_motor(&self, comando: Vec<f32>) {
        let _ = self.motor_tx.send(comando.clone());
        let _ = self.broadcast_tx.send(EventoPipeline::Motor(comando));
    }

    /// Envia texto para síntese de voz (TTS).
    ///
    /// Futuramente conectado à crate `tts` para vocalização.
    /// Por enquanto apenas registra o evento no broadcast.
    pub fn falar(&self, texto: String) {
        let _ = self.fala_tx.send(texto.clone());
        let _ = self.broadcast_tx.send(EventoPipeline::Fala(texto));
    }

    // -------------------------------------------------------------------------
    // Utilitário privado: drenagem de canal
    // -------------------------------------------------------------------------

    /// Drena todos os itens de um canal e retorna apenas o último.
    ///
    /// # Por que descartar os anteriores?
    /// O loop neural tem frequência fixa. Se os sensores produzirem mais rápido
    /// do que o loop consome, os frames antigos acumulariam.
    /// Para não processar "o passado", descartamos tudo exceto o mais recente.
    ///
    /// # Cuidado
    /// Se o canal tiver dados importantíssimos que não podem ser descartados
    /// (ex: comandos de usuário), use `text_rx.try_recv()` diretamente.
    fn drenar_ultimo<T>(rx: &Receiver<T>) -> Option<T> {
        let mut ultimo = None;

        // Drena o canal até esvaziar, guardando sempre o mais novo
        loop {
            match rx.try_recv() {
                Ok(item)                   => { ultimo = Some(item); }
                Err(TryRecvError::Empty)   => { break; }  // canal vazio — para
                Err(TryRecvError::Disconnected) => { break; }  // produtor morreu
            }
        }

        ultimo
    }
}

// =============================================================================
// NOTAS PARA IMPLEMENTAÇÃO FUTURA
// =============================================================================
//
// 1. PRIORIZAÇÃO DE CANAIS
//    Adicione um sistema de prioridade: onset de áudio > frame de texto > frame visual.
//    Quando múltiplos canais têm dados simultaneamente, processe na ordem de prioridade.
//    Biologicamente: um barulho alto interrompe o processamento visual em curso.
//
// 2. TIMESTAMP DE FRAMES
//    Adicione timestamp a cada frame (Instant::now() no momento da captura).
//    Isso permite calcular latência (tempo entre captura e processamento)
//    e sincronizar múltiplos canais (áudio+vídeo juntos = percepção mais rica).
//
// 3. BUFFER COM JANELA TEMPORAL
//    Em vez de descartar frames antigos, mantenha uma janela de 100ms.
//    Isso permite detectar PADRÕES TEMPORAIS (ex: flash de luz seguido de som).
//    Biologicamente: a percepção de "causalidade" requer integração temporal.
//
// 4. CANAIS ADICIONAIS PARA FUTURO
//    - `haptic_rx`: feedback tátil (se houver hardware)
//    - `proprio_rx`: estado de articulações (robô)
//    - `gps_rx`: localização espacial
//    - `imu_rx`: acelerômetro/giroscópio (orientação)
//
// 5. PROTOCOLO DE SINCRONIZAÇÃO
//    Para visão binocular ou microfones múltiplos, sincronize os canais
//    por timestamp. Diferença de fase entre câmeras → profundidade 3D.
//    Diferença de fase entre microfones → direção do som (TDOA).
//
// =============================================================================
