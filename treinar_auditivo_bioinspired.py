#!/usr/bin/env python3
"""
Módulo de Treinamento Auditivo Bio-Inspirado — Selene V3.5
=========================================================

Objetivo: Executar ciclos de aprendizado auditivo baseados em:
  - Repetição semântica (núcleos de significado)
  - Variação acústica (velocidade, volume, timbre)
  - Consolidação REM (30 minutos pós-aprendizado)

Fases:
  1. Léxica: 300 palavras core com variação acústica
  2. Sintática: 300 frases contextualizadas
  3. REM: Consolidação sináptica com acesso exclusivo ao NVMe

Requisitos:
  - Python 3.10+
  - pyttsx3 (TTS)
  - numpy, scipy (áudio)
  - pandas (logging)
  - scipy.signal (ruído)
"""

import os
import sys
import csv
import time
import threading
import random
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple
import wave
import io

try:
    import pyttsx3
except ImportError:
    print("⚠️  pyttsx3 não encontrado. Instale: pip install pyttsx3")
    pyttsx3 = None

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("⚠️  sounddevice/soundfile não encontrado. Instale: pip install sounddevice soundfile")
    sd = None
    sf = None

# ==================== CONFIGURAÇÃO ====================

@dataclass
class TrainingConfig:
    """Configuração do módulo de treinamento"""
    # Dataset
    n_palavras: int = 300
    n_frases: int = 300
    amostra_hz: int = 44100  # Hz (44.1kHz)

    # Variação acústica
    variacao_velocidade: Tuple[float, float] = (0.8, 1.2)
    variacao_volume: Tuple[float, float] = (0.7, 1.0)
    taxa_ruido: float = 0.2  # 20% dos estímulos com ruído
    amplitude_ruido: float = 0.05

    # Timer de consolidação
    tempo_consolidacao_s: int = 1800  # 30 minutos

    # I/O
    log_file: str = "session_history.csv"
    audio_buffer_dir: str = ".audio_cache"

    # Hardware
    usar_threading: bool = True
    prioridade_sono: int = -20  # Nice level (menor = menor prioridade)


config = TrainingConfig()

# ==================== LOGGING ====================

logger = logging.getLogger("Selene.Training")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# ==================== DATASET ====================

class DatasetSemantico:
    """Gerencia 300 palavras e 300 frases agrupadas por "Núcleo de Significado" """

    def __init__(self):
        self.nucleos = self._gerar_nucleos()
        self.palavras = self._gerar_palavras()
        self.frases = self._gerar_frases()

    def _gerar_nucleos(self) -> Dict[str, List[str]]:
        """
        Define núcleos semânticos (grupos de sinônimos).
        Ex: Loop → ciclo, volta, iteração, repetição
        """
        nucleos = {
            "Loop": ["ciclo", "volta", "iteração", "repetição", "laço"],
            "Aprender": ["aprender", "estudar", "conhecer", "assimilar", "absorver"],
            "Memória": ["memória", "lembrança", "recordação", "trace", "engrama"],
            "Neurônio": ["neurônio", "célula", "nó", "unidade", "receptor"],
            "Sinapsia": ["sinapse", "conexão", "ligação", "junção", "acoplamento"],
            "Sono": ["sono", "repouso", "dormência", "inconsciência", "repouso"],
            "Vigília": ["vigília", "despertar", "alerta", "consciência", "atenção"],
            "Dopamina": ["dopamina", "recompensa", "satisfação", "prazer", "incentivo"],
            "Medo": ["medo", "terror", "pavor", "angústia", "fobia"],
            "Alegria": ["alegria", "felicidade", "gozo", "exultação", "júbilo"],
            "Atenção": ["atenção", "foco", "concentração", "saliência", "relevância"],
            "Erro": ["erro", "falha", "equívoco", "mistake", "desvio"],
            "Sucesso": ["sucesso", "êxito", "vitória", "acerto", "triunfo"],
            "Comunidade": ["comunidade", "grupo", "sociedade", "coletivo", "círculo"],
            "Oxitocina": ["oxitocina", "confiança", "vínculo", "empatia", "afeto"],
            "BDNF": ["BDNF", "crescimento", "plasticidade", "neurotrófico", "fator"],
            "Consolidação": ["consolidação", "integração", "síntese", "fusão", "estabilização"],
            "Reversão": ["reversão", "inverso", "retorno", "reversal", "desaprendizado"],
            "Causalidade": ["causalidade", "causa", "origem", "efeito", "consequência"],
            "Predição": ["predição", "previsão", "antecipação", "expectativa", "prognóstico"],
        }

        # Expandir para 60 núcleos (5 palavras cada = 300)
        nucleos_expandidos = {}
        temas_extra = [
            "Ritmo", "Padrão", "Abstração", "Concretude", "Movimento",
            "Pausa", "Aceleração", "Desaceleração", "Harmonia", "Dissonância",
            "Esperança", "Desespero", "Curiosidade", "Monotonia", "Surpreendimento",
            "Habituação", "Sensibilização", "Extinção", "Recuperação", "Renovação",
            "Identidade", "Alteridade", "Pertencimento", "Isolamento", "Fusão",
            "Criatividade", "Reprodução", "Inovação", "Tradição", "Transformação",
            "Limite", "Infinito", "Discreto", "Contínuo", "Estático",
            "Dinâmico", "Reversível", "Irreversível", "Gradual", "Súbito",
            "Silêncio", "Ruído", "Sinal", "Interferência", "Clareza",
            "Ambiguidade", "Certeza", "Dúvida", "Certeza", "Probabilidade",
            "Controle", "Liberdade", "Caos", "Ordem", "Entropia",
            "Seleção", "Eliminação", "Recombinação", "Mutação", "Herança",
            "Adaptação", "Rigidez", "Flexibilidade", "Resiliência", "Fragilidade",
        ]

        nucleos_expandidos.update(nucleos)
        for i, tema in enumerate(temas_extra[:40]):  # 40 mais, total 60
            nucleos_expandidos[tema] = [
                tema.lower(),
                f"{tema.lower()}_2",
                f"{tema.lower()}_3",
                f"{tema.lower()}_4",
                f"{tema.lower()}_5",
            ]

        return nucleos_expandidos

    def _gerar_palavras(self) -> List[str]:
        """Gerar 300 palavras a partir dos núcleos"""
        palavras = []
        for sincronimos in self.nucleos.values():
            palavras.extend(sincronimos)
        return palavras[:config.n_palavras]

    def _gerar_frases(self) -> List[str]:
        """Gerar 300 frases contextualizadas"""
        templates = [
            "A palavra {} é importante para aprendizado.",
            "Ao aprender {}, a Selene consolidou memória.",
            "O conceito de {} conecta com redes neurais.",
            "Durante o sono, {} foi reprocessado.",
            "{} modula a atividade neural.",
            "A plasticidade neural requer {}.",
            "Teste: qual é o significado de {}?",
            "Integre {} ao seu modelo mental.",
            "A ligação de {} com outras palavras é crucial.",
            "Detecte mudanças em {} ao longo do tempo.",
        ]

        frases = []
        for i, palavra in enumerate(self.palavras[:config.n_frases]):
            template = random.choice(templates)
            frase = template.format(palavra)
            frases.append(frase)

        return frases


# ==================== MOTOR DE ÁUDIO ====================

class MotorAudio:
    """Sintetiza e varia áudio, com suporte a pré-cache em RAM"""

    def __init__(self):
        self.engine = None
        self.audio_buffer: Dict[str, np.ndarray] = {}
        self.audio_cache_dir = Path(config.audio_buffer_dir)
        self.audio_cache_dir.mkdir(exist_ok=True)

        if pyttsx3:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # palavras/min
            logger.info("✅ TTS (pyttsx3) inicializado")
        else:
            logger.warning("⚠️  TTS não disponível; usando áudio simulado")

    def _sintetizar_wav(self, texto: str) -> np.ndarray:
        """Sintetizar texto em áudio WAV (numpy array)"""
        if not self.engine:
            # Fallback: gerar sinal sintético (tom)
            duracao_s = len(texto) * 0.1  # 100ms por caractere
            t = np.linspace(0, duracao_s, int(config.amostra_hz * duracao_s))
            frequencia = 440 + hash(texto) % 400  # Frequência hash-baseada
            audio = 0.3 * np.sin(2 * np.pi * frequencia * t)
            return audio.astype(np.float32)

        # TTS real
        try:
            temp_wav = "/tmp/selene_tts.wav"
            self.engine.save_to_file(texto, temp_wav)
            self.engine.runAndWait()

            # Carregar WAV
            if os.path.exists(temp_wav):
                with sf.SoundFile(temp_wav) as f:
                    audio = f.read()
                os.remove(temp_wav)
                return audio.astype(np.float32)
        except Exception as e:
            logger.warning(f"TTS falhou ({e}); usando fallback")

        # Fallback
        duracao_s = len(texto) * 0.1
        t = np.linspace(0, duracao_s, int(config.amostra_hz * duracao_s))
        frequencia = 440
        audio = 0.3 * np.sin(2 * np.pi * frequencia * t)
        return audio.astype(np.float32)

    def gerar_com_variacao(self, texto: str) -> np.ndarray:
        """Gerar áudio com variação acústica (velocidade, volume)"""
        # Sintetizar base
        if texto not in self.audio_buffer:
            self.audio_buffer[texto] = self._sintetizar_wav(texto)

        audio = self.audio_buffer[texto].copy()

        # Variação de velocidade (resample)
        velocidade = random.uniform(*config.variacao_velocidade)
        if velocidade != 1.0:
            novo_tamanho = int(len(audio) / velocidade)
            indices = np.linspace(0, len(audio) - 1, novo_tamanho)
            audio = np.interp(indices, np.arange(len(audio)), audio)

        # Variação de volume
        volume = random.uniform(*config.variacao_volume)
        audio *= volume

        # Injeção de ruído (20% dos estímulos)
        if random.random() < config.taxa_ruido:
            ruido = np.random.normal(0, config.amplitude_ruido, len(audio))
            audio += ruido

        # Normalizar para evitar clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95

        return audio.astype(np.float32)

    def reproduzir(self, audio: np.ndarray) -> None:
        """Reproduzir áudio (se sounddevice disponível)"""
        if sd is not None:
            try:
                sd.play(audio, samplerate=config.amostra_hz)
                sd.wait()
            except Exception as e:
                logger.debug(f"Reprodução falhou: {e}")
        else:
            # Simular reprodução (sleep pelo tempo do áudio)
            duracao_s = len(audio) / config.amostra_hz
            time.sleep(duracao_s)


# ==================== GERENCIADOR DE SESSÃO ====================

class GerenciadorSessao:
    """Coordena fases de aprendizado e logging"""

    def __init__(self, dataset: DatasetSemantico, motor: MotorAudio):
        self.dataset = dataset
        self.motor = motor
        self.sessao_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = config.log_file
        self.inicializar_log()
        self.lock_thread = threading.Lock()
        self.queue_audio = []

    def inicializar_log(self) -> None:
        """Criar arquivo CSV para logging"""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "session_id", "timestamp", "fase", "estimulo_tipo",
                "estimulo_texto", "duracao_ms", "variacao_vel", "variacao_vol"
            ])
        logger.info(f"📝 Log inicializado: {self.log_file}")

    def registrar_estimulo(self, fase: str, tipo: str, texto: str, duracao_ms: float) -> None:
        """Registrar estimulo no CSV"""
        with self.lock_thread:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.sessao_id,
                    datetime.now().isoformat(),
                    fase,
                    tipo,
                    texto[:50],  # Truncar para evitar overflow
                    f"{duracao_ms:.1f}",
                    "",
                    ""
                ])

    def fase_lexica(self) -> None:
        """Fase 1: Disparar 300 palavras com intervalos"""
        logger.info("\n🔤 FASE LÉXICA: Iniciando 300 palavras...")

        for i, palavra in enumerate(self.dataset.palavras):
            inicio = time.time()
            audio = self.motor.gerar_com_variacao(palavra)
            self.motor.reproduzir(audio)
            duracao_ms = (time.time() - inicio) * 1000

            self.registrar_estimulo("LEXICO", "palavra", palavra, duracao_ms)

            # Intervalo de 1 segundo
            if i < len(self.dataset.palavras) - 1:
                time.sleep(1.0)

            if (i + 1) % 50 == 0:
                logger.info(f"   {i + 1}/{len(self.dataset.palavras)} palavras")

        logger.info("✅ Fase Léxica completa")

    def fase_sintatica(self) -> None:
        """Fase 2: Disparar 300 frases contextualizadas"""
        logger.info("\n📝 FASE SINTÁTICA: Iniciando 300 frases...")

        for i, frase in enumerate(self.dataset.frases):
            inicio = time.time()
            audio = self.motor.gerar_com_variacao(frase)
            self.motor.reproduzir(audio)
            duracao_ms = (time.time() - inicio) * 1000

            self.registrar_estimulo("SINTATICO", "frase", frase, duracao_ms)

            # Intervalo de 1 segundo
            if i < len(self.dataset.frases) - 1:
                time.sleep(1.0)

            if (i + 1) % 50 == 0:
                logger.info(f"   {i + 1}/{len(self.dataset.frases)} frases")

        logger.info("✅ Fase Sintática completa")

    def fase_rem(self) -> None:
        """Fase 3: Consolidação REM (30 minutos)"""
        logger.info("\n💤 FASE REM: Consolidação sináptica iniciada...")
        logger.info(f"   Aguardando {config.tempo_consolidacao_s}s (~30 min)")

        tempo_total = config.tempo_consolidacao_s
        tempo_checkpoint = tempo_total // 6  # 6 checkpoints

        for checkpoint in range(6):
            for segundo in range(tempo_checkpoint):
                # Barra de progresso
                progresso = (checkpoint * tempo_checkpoint + segundo) / tempo_total
                barra = "█" * int(progresso * 40) + "░" * (40 - int(progresso * 40))
                print(f"\r   [{barra}] {progresso*100:.1f}%", end="", flush=True)
                time.sleep(1)

            logger.info(f"\n   Checkpoint {checkpoint + 1}/6 concluído")

        print("\n")
        logger.info("✅ Fase REM completa — Consolidação sináptica finalizada")

    def despertar_ping(self) -> None:
        """Emitir "ping" de alta frequência para sinalizar despertar"""
        logger.info("\n🔔 DESPERTAR: Gerando ping de retorno à vigília...")

        # Gerar tom de 2kHz por 200ms
        duracao_s = 0.2
        t = np.linspace(0, duracao_s, int(config.amostra_hz * duracao_s))
        frequencia = 2000
        ping = 0.5 * np.sin(2 * np.pi * frequencia * t).astype(np.float32)

        self.motor.reproduzir(ping)
        logger.info("✅ Despertar — Vigília restaurada")

    def executar(self) -> None:
        """Executar ciclo completo de treinamento"""
        logger.info("=" * 60)
        logger.info(f"🧠 SELENE V3.5 — TREINAMENTO AUDITIVO BIO-INSPIRADO")
        logger.info(f"   Sessão: {self.sessao_id}")
        logger.info(f"   Palavras: {len(self.dataset.palavras)}")
        logger.info(f"   Frases: {len(self.dataset.frases)}")
        logger.info("=" * 60)

        try:
            # Fase 1: Léxica
            self.fase_lexica()
            time.sleep(5)  # Pausa entre fases

            # Fase 2: Sintática
            self.fase_sintatica()
            time.sleep(5)

            # Fase 3: REM
            self.fase_rem()

            # Despertar
            self.despertar_ping()

            logger.info("\n" + "=" * 60)
            logger.info("✨ CICLO DE TREINAMENTO COMPLETO")
            logger.info(f"   Log: {self.log_file}")
            logger.info("=" * 60)

        except KeyboardInterrupt:
            logger.warning("\n⚠️  Treinamento interrompido pelo usuário")
        except Exception as e:
            logger.error(f"❌ Erro durante treinamento: {e}")
            raise


# ==================== MAIN ====================

def main():
    """Ponto de entrada do módulo"""
    logger.info("🚀 Inicializando Selene Treinamento Auditivo Bio-Inspirado...")

    # 1. Carregar dataset
    dataset = DatasetSemantico()
    logger.info(f"✅ Dataset carregado: {len(dataset.palavras)} palavras, {len(dataset.frases)} frases")

    # 2. Inicializar motor de áudio
    motor = MotorAudio()
    logger.info("✅ Motor de áudio inicializado")

    # 3. Criar gerenciador de sessão
    gerenciador = GerenciadorSessao(dataset, motor)

    # 4. Executar ciclo
    gerenciador.executar()


if __name__ == "__main__":
    main()
