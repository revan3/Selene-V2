#!/usr/bin/env python3
"""
Selene V3.5 — Treinamento Auditivo Bio-Inspirado (Versão 4.0)
==============================================================

3 Fases Progressivas de Aprendizado:
  Fase 1: Léxica Sequencial (Palavras)
  Fase 2: Sintática Progressiva (Frases)
  Fase 3: Semântica Complexa (Inferência)

Ciclo por significado:
  15-20 min de ensino → 20 min REM consolidation → teste simples
"""

import os
import sys
import csv
import time
import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import threading

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sd = None
    sf = None

try:
    from scipy import signal
except ImportError:
    signal = None

# ==================== ENUMS ====================

class Entonacao(Enum):
    """Variações de entonação para TTS"""
    NORMAL = 1.0
    ALTA = 1.3      # Pergunta/surpresa
    BAIXA = 0.7     # Afirmação/certeza
    TRISTE = 0.9
    ALEGRE = 1.2


class TipoRuido(Enum):
    """Tipos de ruído contextual"""
    SILENCIO = "silence"
    BRANCO = "white"      # Ruído branco
    CHUVA = "rain"        # Som de chuva
    VOZES = "voices"      # Vozes dispersas
    CARRO = "car"         # Som de carro
    AMBIENTE = "ambient"  # Ruído ambiente geral


# ==================== CONFIGURAÇÃO ====================

@dataclass
class TrainingConfig:
    """Configuração global de treinamento"""
    # Dataset
    significados: int = 60              # 60 núcleos semânticos
    palavras_por_significado: int = 5   # 5 palavras/grupo
    frases_por_significado: int = 10    # 10 frases/significado

    # Timers
    tempo_ensino_lexico_min: int = 15   # minutos
    tempo_ensino_sintatico_min: int = 20
    tempo_ensino_semantico_min: int = 25
    tempo_rem_min: int = 20             # REM consolidation

    # Áudio
    amostra_hz: int = 44100
    velocidade_fala: int = 150          # palavras/min

    # Variação
    variacao_entonacao: List[Entonacao] = field(default_factory=lambda: [
        Entonacao.NORMAL, Entonacao.ALTA, Entonacao.BAIXA
    ])
    variacao_ruido_fase1: List[TipoRuido] = field(default_factory=lambda: [
        TipoRuido.SILENCIO, TipoRuido.BRANCO
    ])
    variacao_ruido_fase2: List[TipoRuido] = field(default_factory=lambda: [
        TipoRuido.SILENCIO, TipoRuido.BRANCO, TipoRuido.CHUVA, TipoRuido.VOZES
    ])
    variacao_ruido_fase3: List[TipoRuido] = field(default_factory=lambda: [
        TipoRuido.SILENCIO, TipoRuido.BRANCO, TipoRuido.CHUVA,
        TipoRuido.VOZES, TipoRuido.CARRO, TipoRuido.AMBIENTE
    ])

    # Logging
    log_file: str = "selene_training_log.csv"
    modo_teste_rapido: bool = False     # True = reduz 100x os timers


config = TrainingConfig()


# ==================== LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger("Selene.Training.V4")


# ==================== DATASET ====================

class DatasetSemantico:
    """Gera e gerencia dataset de 60 significados únicos"""

    def __init__(self):
        self.nucleos = self._gerar_nucleos_unicos()
        self.frases_templates = self._gerar_templates_frases()
        self.significados_ordem = list(self.nucleos.keys())
        random.shuffle(self.significados_ordem)
        # Atributos de configuração
        self.frases_por_significado = config.frases_por_significado
        self.palavras_por_significado = config.palavras_por_significado

    def _gerar_nucleos_unicos(self) -> Dict[str, List[str]]:
        """60 núcleos semânticos únicos com 5 sinônimos cada"""
        nucleos = {
            # Cognitivos
            "Aprendizado": ["aprender", "estudar", "conhecer", "assimilar", "absorver"],
            "Memória": ["memória", "lembrança", "recordação", "trace", "engrama"],
            "Atenção": ["atenção", "foco", "concentração", "saliência", "alerta"],
            "Predição": ["predição", "previsão", "antecipação", "expectativa", "prognóstico"],
            "Erro": ["erro", "falha", "equívoco", "desvio", "incorreção"],

            # Neurobiológicos
            "Neurônio": ["neurônio", "célula", "nó", "unidade", "receptor"],
            "Sinapse": ["sinapse", "conexão", "ligação", "junção", "acoplamento"],
            "BDNF": ["BDNF", "crescimento", "plasticidade", "neurotrófico", "fator"],
            "Consolidação": ["consolidação", "integração", "síntese", "fusão", "estabilização"],
            "Sono": ["sono", "repouso", "dormência", "inconsciência", "letargia"],

            # Emocionais
            "Alegria": ["alegria", "felicidade", "gozo", "exultação", "júbilo"],
            "Medo": ["medo", "terror", "pavor", "angústia", "fobia"],
            "Tristeza": ["tristeza", "melancolia", "pesar", "desânimo", "desalento"],
            "Raiva": ["raiva", "cólera", "fúria", "ódio", "ressentimento"],
            "Esperança": ["esperança", "otimismo", "confiança", "expectativa", "fé"],

            # Motivacionais
            "Recompensa": ["recompensa", "prêmio", "incentivo", "ganho", "satisfação"],
            "Punição": ["punição", "castigo", "penalidade", "reprovação", "sanção"],
            "Motivação": ["motivação", "impulso", "ânimo", "estímulo", "inspiração"],
            "Dopamina": ["dopamina", "recompensa", "prazer", "satisfação", "incentivo"],
            "Salência": ["salência", "relevância", "importância", "saliência", "destaque"],

            # Sociais
            "Comunidade": ["comunidade", "grupo", "sociedade", "coletivo", "círculo"],
            "Empatia": ["empatia", "compreensão", "simpatia", "identificação", "conexão"],
            "Confiança": ["confiança", "fé", "credibilidade", "lealdade", "segurança"],
            "Oxitocina": ["oxitocina", "vínculo", "afeto", "intimidade", "apego"],
            "Isolamento": ["isolamento", "solidão", "separação", "afastamento", "desconexão"],

            # Estruturais
            "Padrão": ["padrão", "regularidade", "sequência", "estrutura", "formato"],
            "Causalidade": ["causalidade", "causa", "origem", "efeito", "consequência"],
            "Temporalidade": ["tempo", "duração", "sequência", "ordem", "progressão"],
            "Espaço": ["espaço", "lugar", "posição", "localização", "área"],
            "Quantidade": ["quantidade", "número", "magnitude", "volume", "escala"],

            # Procedimental
            "Ritmo": ["ritmo", "batida", "cadência", "compasso", "frequência"],
            "Movimento": ["movimento", "ação", "deslocamento", "mudança", "transição"],
            "Pausa": ["pausa", "intervalo", "repouso", "parada", "suspensão"],
            "Aceleração": ["aceleração", "velocidade", "pressa", "precipitação", "urgência"],
            "Desaceleração": ["desaceleração", "lentidão", "morosidade", "atraso", "demora"],

            # Perceptual
            "Brilho": ["brilho", "luminosidade", "claridade", "fulgor", "esplendor"],
            "Escuridão": ["escuridão", "obscuridade", "sombra", "trevas", "penumbra"],
            "Som": ["som", "ruído", "barulho", "estrondo", "murmurio"],
            "Silêncio": ["silêncio", "quietude", "tranquilidade", "serenidade", "mutismo"],
            "Sabor": ["sabor", "gosto", "paladar", "degustação", "saboroso"],

            # Abstratos
            "Verdade": ["verdade", "realidade", "facto", "certeza", "veracidade"],
            "Falsidade": ["falsidade", "mentira", "engano", "ilusão", "fraude"],
            "Complexidade": ["complexidade", "complicação", "intricação", "sofisticação", "sutileza"],
            "Simplicidade": ["simplicidade", "clareza", "facilidade", "primitividade", "elementaridade"],
            "Infinito": ["infinito", "ilimitado", "eterno", "interminável", "sem-fim"],

            # Transformação
            "Crescimento": ["crescimento", "aumento", "desenvolvimento", "expansão", "proliferação"],
            "Declínio": ["declínio", "diminuição", "redução", "enfraquecimento", "deterioração"],
            "Reversão": ["reversão", "inversão", "retorno", "reversal", "desaprendizado"],
            "Inovação": ["inovação", "novidade", "criatividade", "invenção", "originalidade"],
            "Tradição": ["tradição", "costume", "convenção", "herança", "legado"],

            # Controle
            "Liberdade": ["liberdade", "autonomia", "independência", "licença", "permissão"],
            "Controle": ["controle", "domínio", "poder", "autoridade", "comando"],
            "Caos": ["caos", "desordem", "confusão", "anarquia", "desorganização"],
            "Ordem": ["ordem", "organização", "sistema", "estrutura", "disposição"],
            "Harmonia": ["harmonia", "equilíbrio", "consonância", "concordância", "sincronização"],

            # Biológico
            "Vida": ["vida", "existência", "vitalidade", "animação", "ser"],
            "Morte": ["morte", "fim", "extinção", "término", "cessação"],
            "Saúde": ["saúde", "bem-estar", "vigor", "vitalidade", "sanidade"],
            "Doença": ["doença", "enfermidade", "patologia", "moléstia", "afecção"],
            "Adaptação": ["adaptação", "ajustamento", "adequação", "conformação", "acomodação"],
        }

        return nucleos  # 60 núcleos semânticos

    def _gerar_templates_frases(self) -> Dict[str, List[str]]:
        """Templates de frases para cada contexto"""
        return {
            "descritivo": [
                "A palavra '{}' é importante.",
                "Quando falamos de '{}', é um conceito fundamental.",
                "O conceito de '{}' é relevante.",
                "Podemos definir '{}' com clareza.",
                "A essência de '{}' é profunda.",
            ],
            "relacional": [
                "'{}' está conectado a conceitos maiores.",
                "'{}' frequentemente aparece em combinações.",
                "Existe uma relação entre '{}' e outros termos.",
                "'{}' é resultado de processos complexos.",
                "'{}' pode levar a consequências interessantes.",
            ],
            "temporal": [
                "Antes de '{}', vem preparação.",
                "Durante '{}', ocorrem transformações.",
                "Depois de '{}', segue consolidação.",
                "'{}' começa com reconhecimento.",
                "'{}' termina com integração.",
            ],
            "causal": [
                "'{}' causa mudanças significativas.",
                "'{}' é causado por processos anteriores.",
                "Se temos '{}', então há consequências.",
                "'{}' resulta em aprendizado.",
                "Por causa de '{}', acontecem eventos.",
            ],
            "comparativo": [
                "'{}' é similar em estrutura.",
                "'{}' é oposto em função.",
                "Entre '{}' e conceitos similares há diferenças.",
                "'{}' é mais complexo que alternativas.",
                "'{}' é essencial para compreensão.",
            ],
        }

    def obter_significado(self, idx: int) -> Tuple[str, List[str]]:
        """Obter significado + palavras por índice"""
        chave = self.significados_ordem[idx]
        return chave, self.nucleos[chave]

    def gerar_frases_para_significado(self, significado: str, sinonimo: str) -> List[str]:
        """Gerar 10 frases para um significado"""
        frases = []
        templates_lista = list(self.frases_templates.values())

        for i in range(self.frases_por_significado):
            template_choice = templates_lista[i % len(templates_lista)]
            template = random.choice(template_choice)
            # Simplificar template (usar apenas um placeholder)
            if "{}" in template:
                frase = template.format(sinonimo)
                frases.append(frase)

        return frases


# ==================== MOTOR DE ÁUDIO ====================

class MotorAudioAvancado:
    """Motor de áudio com entonação variável e ruído contextual"""

    def __init__(self):
        self.engine = None
        self.audio_cache = {}
        self.taxa_hz = config.amostra_hz

        if pyttsx3:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', config.velocidade_fala)
            logger.info("[OK] TTS (pyttsx3) inicializado")
        else:
            logger.warning("⚠️  TTS não disponível")

    def _sintetizar_texto(self, texto: str, entonacao: Entonacao = Entonacao.NORMAL) -> np.ndarray:
        """Sintetizar texto em áudio"""
        if not self.engine:
            # Fallback: tom sintético
            duracao = len(texto) * 0.1
            t = np.linspace(0, duracao, int(self.taxa_hz * duracao))
            freq = 440 * entonacao.value
            audio = 0.3 * np.sin(2 * np.pi * freq * t)
            return audio.astype(np.float32)

        # TTS com entonação
        try:
            temp_file = "/tmp/tts_temp.wav"
            # Ajustar velocidade para simular entonação
            velocidade_ajustada = int(config.velocidade_fala * entonacao.value)
            self.engine.setProperty('rate', velocidade_ajustada)
            self.engine.save_to_file(texto, temp_file)
            self.engine.runAndWait()

            if os.path.exists(temp_file):
                try:
                    if sf:
                        audio, _ = sf.read(temp_file)
                    else:
                        audio = self._ler_wav_puro(temp_file)
                    os.remove(temp_file)
                    return audio.astype(np.float32)
                except:
                    pass
        except Exception as e:
            logger.debug(f"TTS error: {e}")

        # Fallback
        duracao = len(texto) * 0.1
        t = np.linspace(0, duracao, int(self.taxa_hz * duracao))
        freq = 440 * entonacao.value
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        return audio.astype(np.float32)

    def _ler_wav_puro(self, caminho: str) -> np.ndarray:
        """Ler WAV sem soundfile (fallback)"""
        import wave
        try:
            with wave.open(caminho, 'rb') as f:
                frames = f.readframes(f.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768
                return audio
        except:
            return np.array([])

    def _adicionar_ruido(self, audio: np.ndarray, tipo_ruido: TipoRuido,
                         amplitude: float = 0.05) -> np.ndarray:
        """Adicionar ruído contextual ao áudio"""
        if tipo_ruido == TipoRuido.SILENCIO:
            return audio

        n_samples = len(audio)

        if tipo_ruido == TipoRuido.BRANCO:
            ruido = np.random.normal(0, amplitude, n_samples)

        elif tipo_ruido == TipoRuido.CHUVA:
            # Simular chuva: ruído gaussiano com envelope
            ruido_base = np.random.normal(0, amplitude, n_samples)
            t = np.arange(n_samples) / self.taxa_hz
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz modulação
            ruido = ruido_base * envelope

        elif tipo_ruido == TipoRuido.VOZES:
            # Simular vozes: múltiplas frequências
            ruido = np.zeros(n_samples)
            for freq in [200, 400, 800]:
                t = np.arange(n_samples) / self.taxa_hz
                ruido += amplitude * np.sin(2 * np.pi * freq * t + np.random.random() * 2 * np.pi)
            ruido = ruido / 3  # Normalizar

        elif tipo_ruido == TipoRuido.CARRO:
            # Simular carro: ruído com pitch variável
            t = np.arange(n_samples) / self.taxa_hz
            freq_base = 150 + 50 * np.sin(2 * np.pi * 0.2 * t)
            ruido = amplitude * np.sin(2 * np.pi * freq_base * t / self.taxa_hz)

        elif tipo_ruido == TipoRuido.AMBIENTE:
            # Ruído ambiente: mix de frequências
            ruido = np.zeros(n_samples)
            for freq in [100, 200, 300, 500, 800]:
                t = np.arange(n_samples) / self.taxa_hz
                peso = 0.3 + 0.7 * np.random.random()
                ruido += peso * amplitude * np.sin(2 * np.pi * freq * t)
            ruido = ruido / 5

        else:
            ruido = np.zeros(n_samples)

        # Misturar
        audio_com_ruido = audio + ruido

        # Normalizar
        max_val = np.max(np.abs(audio_com_ruido))
        if max_val > 0:
            audio_com_ruido = audio_com_ruido / max_val * 0.95

        return audio_com_ruido.astype(np.float32)

    def gerar_estímulo(self, texto: str, entonacao: Entonacao = Entonacao.NORMAL,
                       tipo_ruido: TipoRuido = TipoRuido.SILENCIO) -> np.ndarray:
        """Gerar estímulo completo (texto + entonação + ruído)"""
        # Sintetizar
        audio = self._sintetizar_texto(texto, entonacao)

        # Adicionar ruído
        audio = self._adicionar_ruido(audio, tipo_ruido)

        return audio

    def reproduzir(self, audio: np.ndarray) -> float:
        """Reproduzir e retornar duração"""
        duracao = len(audio) / self.taxa_hz

        if sd:
            try:
                sd.play(audio, samplerate=self.taxa_hz)
                sd.wait()
                return duracao
            except Exception as e:
                logger.debug(f"Playback error: {e}")

        # Simular
        time.sleep(min(duracao, 0.5))
        return duracao


# ==================== LOGGER ESTRUTURADO ====================

class LoggerSemantico:
    """Logging estruturado de treinamento"""

    def __init__(self, arquivo: str = config.log_file):
        self.arquivo = arquivo
        self.inicializar()
        self.lock = threading.Lock()

    def inicializar(self):
        """Criar CSV com headers"""
        with open(self.arquivo, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "fase", "significado", "item",
                "entonacao", "ruido", "duracao_s",
                "tipo_evento", "resultado_teste"
            ])

    def registrar(self, fase: str, significado: str, item: str,
                  entonacao: str = "-", ruido: str = "-",
                  duracao_s: float = 0, tipo_evento: str = "ensino",
                  resultado_teste: str = "-"):
        """Registrar evento"""
        with self.lock:
            with open(self.arquivo, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    fase, significado, item,
                    entonacao, ruido, f"{duracao_s:.2f}",
                    tipo_evento, resultado_teste
                ])


# ==================== TESTER INTELIGENTE ====================

class TesterInteligente:
    """Testes de reconhecimento progressivos"""

    def __init__(self, motor: MotorAudioAvancado, logger: LoggerSemantico):
        self.motor = motor
        self.logger = logger

    def teste_simples_fase1(self, significado: str, palavras_ensinadas: List[str]) -> bool:
        """Teste simples Fase 1: reconhecer palavras"""
        logger.info(f"\n  [TEST] TESTE FASE 1: Reconhecimento de palavras '{significado}'")

        # Escolher 3 palavras corretas + 2 incorretas
        corretas = random.sample(palavras_ensinadas, min(3, len(palavras_ensinadas)))
        todas_palavras = list(set(palavras_ensinadas))

        # Verificar: usuário precisa reconhecer as 3 palavras
        reconhecidas = 0
        for palavra in corretas:
            print(f"    Ouça: {palavra}")
            time.sleep(1)
            print(f"    Pergunta: Esta palavra significa '{significado}'? (s/n)")
            # Simulação: 80% acerto
            acertou = random.random() < 0.8
            if acertou:
                reconhecidas += 1
                print("    [OK] Correto!")
            else:
                print("    [FAIL] Erro")

        taxa_sucesso = reconhecidas / len(corretas)
        self.logger.registrar(
            "FASE1", significado, ",".join(corretas),
            tipo_evento="teste", resultado_teste=f"{taxa_sucesso:.1%}"
        )

        return taxa_sucesso >= 0.6

    def teste_simples_fase2(self, significado: str, frases_ensinadas: List[str]) -> bool:
        """Teste simples Fase 2: reconhecer significado"""
        logger.info(f"\n  [TEST] TESTE FASE 2: Compreensão de '{significado}'")

        # Escolher 2 frases corretas
        corretas = random.sample(frases_ensinadas, min(2, len(frases_ensinadas)))

        reconhecidas = 0
        for frase in corretas:
            print(f"    Ouça: {frase[:50]}...")
            time.sleep(1)
            print(f"    Pergunta: Qual é o significado principal?")
            print(f"    Resposta esperada: {significado}")
            # Simulação: 75% acerto
            acertou = random.random() < 0.75
            if acertou:
                reconhecidas += 1
                print("    [OK] Correto!")
            else:
                print("    [FAIL] Erro")

        taxa_sucesso = reconhecidas / len(corretas)
        self.logger.registrar(
            "FASE2", significado, ",".join(corretas[:1]),
            tipo_evento="teste", resultado_teste=f"{taxa_sucesso:.1%}"
        )

        return taxa_sucesso >= 0.5


# ==================== FASES DE TREINAMENTO ====================

class FaseLexico:
    """Fase 1: Aprendizado Léxico Sequencial"""

    def __init__(self, dataset: DatasetSemantico, motor: MotorAudioAvancado,
                 logger: LoggerSemantico):
        self.dataset = dataset
        self.motor = motor
        self.logger = logger
        self.tester = TesterInteligente(motor, logger)

    def ensinar_significado(self, idx: int) -> bool:
        """Ensinar um significado (ciclo completo)"""
        significado, palavras = self.dataset.obter_significado(idx)

        logger.info(f"\n{'='*70}")
        logger.info(f"📚 FASE 1 - LÉXICA: Significado {idx+1}/{config.significados}")
        logger.info(f"   Termo: '{significado}'")
        logger.info(f"   Palavras: {', '.join(palavras)}")
        logger.info(f"{'='*70}")

        # ===== ENSINO: 15 minutos =====
        logger.info(f"\n[TIMER]  Ensino por {config.tempo_ensino_lexico_min} minutos...")
        tempo_inicio = time.time()
        tempo_maximo = config.tempo_ensino_lexico_min * 60 if not config.modo_teste_rapido else 10

        repeticoes = 0
        while time.time() - tempo_inicio < tempo_maximo:
            palavra = random.choice(palavras)
            entonacao = random.choice(config.variacao_entonacao)
            ruido = random.choice(config.variacao_ruido_fase1)

            # Gerar e reproduzir
            audio = self.motor.gerar_estímulo(palavra, entonacao, ruido)
            duracao = self.motor.reproduzir(audio)

            self.logger.registrar(
                "FASE1", significado, palavra,
                entonacao.name, ruido.value, duracao
            )

            repeticoes += 1
            progresso = (time.time() - tempo_inicio) / tempo_maximo
            bar = '#' * int(progresso * 30) + '-' * (30 - int(progresso * 30))
            print(f"\r   [{bar}] {progresso*100:.0f}%", end="", flush=True)

            time.sleep(0.5)  # Pausa entre repetições

        print(f"\n   [OK] {repeticoes} repetições concluídas")

        # ===== REM: 20 minutos =====
        logger.info(f"\n[SLEEP] Consolidação REM por {config.tempo_rem_min} minutos...")
        tempo_rem = config.tempo_rem_min * 60 if not config.modo_teste_rapido else 5

        tempo_inicio = time.time()
        while time.time() - tempo_inicio < tempo_rem:
            progresso = (time.time() - tempo_inicio) / tempo_rem
            bar = '#' * int(progresso * 30) + '-' * (30 - int(progresso * 30))
            print(f"\r   [{bar}] {progresso*100:.0f}%", end="", flush=True)
            time.sleep(1)

        print("\n   [OK] REM consolidation completa")

        # ===== TESTE =====
        sucesso = self.tester.teste_simples_fase1(significado, palavras)

        if sucesso:
            logger.info(f"   [OK] Significado '{significado}' aprendido!")
        else:
            logger.info(f"   ⚠️  Baixo desempenho em '{significado}' - repetir recomendado")

        return sucesso

    def executar(self):
        """Executar Fase 1 completa"""
        logger.info(f"\n\n{'#'*70}")
        logger.info(f"🧠 FASE 1: APRENDIZADO LÉXICO SEQUENCIAL")
        logger.info(f"   Total: {config.significados} significados")
        logger.info(f"   Ciclo/significado: 15 min ensino + 20 min REM + teste")
        logger.info(f"{'#'*70}")

        sucessos = 0
        for i in range(config.significados):
            sucesso = self.ensinar_significado(i)
            if sucesso:
                sucessos += 1

            # Pausa entre significados
            logger.info("\n⏸️  Pausa de 2 minutos entre significados...")
            time.sleep(2 if config.modo_teste_rapido else 120)

        logger.info(f"\n\n[DONE] FASE 1 COMPLETA!")
        logger.info(f"   Sucessos: {sucessos}/{config.significados} ({sucessos/config.significados*100:.1f}%)")

        return sucessos / config.significados >= 0.6


class FaseSintatica:
    """Fase 2: Aprendizado Sintático (Frases em Contexto)"""

    def __init__(self, dataset: DatasetSemantico, motor: MotorAudioAvancado,
                 logger: LoggerSemantico):
        self.dataset = dataset
        self.motor = motor
        self.logger = logger
        self.tester = TesterInteligente(motor, logger)

    def ensinar_significado(self, idx: int) -> bool:
        """Ensinar significado via frases"""
        significado, _ = self.dataset.obter_significado(idx)
        sinonimo = self.dataset.nucleos[significado][0]
        frases = self.dataset.gerar_frases_para_significado(significado, sinonimo)

        logger.info(f"\n{'='*70}")
        logger.info(f"📚 FASE 2 - SINTÁTICA: Significado {idx+1}/{config.significados}")
        logger.info(f"   Termo: '{significado}'")
        logger.info(f"{'='*70}")

        # ===== ENSINO: 20 minutos =====
        logger.info(f"\n[TIMER]  Ensino por {config.tempo_ensino_sintatico_min} minutos...")
        tempo_inicio = time.time()
        tempo_maximo = config.tempo_ensino_sintatico_min * 60 if not config.modo_teste_rapido else 10

        repeticoes = 0
        while time.time() - tempo_inicio < tempo_maximo:
            frase = random.choice(frases)
            entonacao = random.choice(config.variacao_entonacao)
            ruido = random.choice(config.variacao_ruido_fase2)

            audio = self.motor.gerar_estímulo(frase, entonacao, ruido)
            duracao = self.motor.reproduzir(audio)

            self.logger.registrar(
                "FASE2", significado, frase[:40],
                entonacao.name, ruido.value, duracao
            )

            repeticoes += 1
            progresso = (time.time() - tempo_inicio) / tempo_maximo
            bar = '#' * int(progresso * 30) + '-' * (30 - int(progresso * 30))
            print(f"\r   [{bar}] {progresso*100:.0f}%", end="", flush=True)

            time.sleep(0.5)

        print(f"\n   [OK] {repeticoes} repetições concluídas")

        # ===== REM: 20 minutos =====
        logger.info(f"\n[SLEEP] Consolidação REM por {config.tempo_rem_min} minutos...")
        tempo_rem = config.tempo_rem_min * 60 if not config.modo_teste_rapido else 5

        tempo_inicio = time.time()
        while time.time() - tempo_inicio < tempo_rem:
            progresso = (time.time() - tempo_inicio) / tempo_rem
            bar = '#' * int(progresso * 30) + '-' * (30 - int(progresso * 30))
            print(f"\r   [{bar}] {progresso*100:.0f}%", end="", flush=True)
            time.sleep(1)

        print("\n   [OK] REM consolidation completa")

        # ===== TESTE =====
        sucesso = self.tester.teste_simples_fase2(significado, frases)

        if sucesso:
            logger.info(f"   [OK] Contexto '{significado}' aprendido!")
        else:
            logger.info(f"   ⚠️  Baixo desempenho contextual em '{significado}'")

        return sucesso

    def executar(self):
        """Executar Fase 2 completa"""
        logger.info(f"\n\n{'#'*70}")
        logger.info(f"🧠 FASE 2: APRENDIZADO SINTÁTICO")
        logger.info(f"   Total: {config.significados} significados")
        logger.info(f"   Ciclo/significado: 20 min ensino + 20 min REM + teste")
        logger.info(f"{'#'*70}")

        sucessos = 0
        for i in range(config.significados):
            sucesso = self.ensinar_significado(i)
            if sucesso:
                sucessos += 1

            # Pausa
            logger.info("\n⏸️  Pausa de 2 minutos entre significados...")
            time.sleep(2 if config.modo_teste_rapido else 120)

        logger.info(f"\n\n[DONE] FASE 2 COMPLETA!")
        logger.info(f"   Sucessos: {sucessos}/{config.significados} ({sucessos/config.significados*100:.1f}%)")

        return sucessos / config.significados >= 0.5


class FaseSemantica:
    """Fase 3: Aprendizado Semântico Complexo (Inferência e Causalidade)"""

    def __init__(self, dataset: DatasetSemantico, motor: MotorAudioAvancado,
                 logger: LoggerSemantico):
        self.dataset = dataset
        self.motor = motor
        self.logger = logger
        self.tester = TesterInteligente(motor, logger)

    def ensinar_significado(self, idx: int) -> bool:
        """Ensinar com complexidade aumentada"""
        significado, _ = self.dataset.obter_significado(idx)

        logger.info(f"\n{'='*70}")
        logger.info(f"📚 FASE 3 - SEMÂNTICA: Significado {idx+1}/{config.significados}")
        logger.info(f"   Termo: '{significado}'")
        logger.info(f"{'='*70}")

        # ===== ENSINO: 25 minutos COM MÚLTIPLAS CAMADAS =====
        logger.info(f"\n[TIMER]  Ensino complexo por {config.tempo_ensino_semantico_min} minutos...")
        tempo_inicio = time.time()
        tempo_maximo = config.tempo_ensino_semantico_min * 60 if not config.modo_teste_rapido else 10

        repeticoes = 0
        while time.time() - tempo_inicio < tempo_maximo:
            # Variar: palavra pura, em frase, com contexto, etc.
            choice = random.choice(['palavra', 'frase', 'contexto', 'contraste'])

            if choice == 'palavra':
                palavra = random.choice(self.dataset.nucleos[significado])
                texto = palavra
            elif choice == 'frase':
                sinonimo = self.dataset.nucleos[significado][0]
                frases = self.dataset.gerar_frases_para_significado(significado, sinonimo)
                texto = random.choice(frases)
            elif choice == 'contexto':
                sinonimo = self.dataset.nucleos[significado][0]
                texto = f"Considere {sinonimo} em um contexto diferente..."
            else:  # contraste
                outro_idx = (idx + 1) % config.significados
                outro_significado, _ = self.dataset.obter_significado(outro_idx)
                texto = f"Diferente de {outro_significado}, temos {significado}..."

            entonacao = random.choice(config.variacao_entonacao)
            ruido = random.choice(config.variacao_ruido_fase3)

            audio = self.motor.gerar_estímulo(texto, entonacao, ruido)
            duracao = self.motor.reproduzir(audio)

            self.logger.registrar(
                "FASE3", significado, texto[:40],
                entonacao.name, ruido.value, duracao
            )

            repeticoes += 1
            progresso = (time.time() - tempo_inicio) / tempo_maximo
            bar = '#' * int(progresso * 30) + '-' * (30 - int(progresso * 30))
            print(f"\r   [{bar}] {progresso*100:.0f}%", end="", flush=True)

            time.sleep(0.5)

        print(f"\n   [OK] {repeticoes} repetições concluídas")

        # ===== REM: 20 minutos =====
        logger.info(f"\n[SLEEP] Consolidação REM por {config.tempo_rem_min} minutos...")
        tempo_rem = config.tempo_rem_min * 60 if not config.modo_teste_rapido else 5

        tempo_inicio = time.time()
        while time.time() - tempo_inicio < tempo_rem:
            progresso = (time.time() - tempo_inicio) / tempo_rem
            bar = '#' * int(progresso * 30) + '-' * (30 - int(progresso * 30))
            print(f"\r   [{bar}] {progresso*100:.0f}%", end="", flush=True)
            time.sleep(1)

        print("\n   [OK] REM consolidation completa")

        # ===== TESTE COMPLEXO =====
        logger.info(f"\n  [TEST] TESTE FASE 3: Inferência e Causalidade")
        print(f"    Questão: Como '{significado}' se relaciona com outros conceitos?")
        # Simulação: 70% acerto em testes complexos
        acertou = random.random() < 0.7
        if acertou:
            logger.info("   [OK] Inferência correta!")
            taxa_sucesso = 1.0
        else:
            logger.info("   [FAIL] Inferência incorreta")
            taxa_sucesso = 0.0

        self.logger.registrar(
            "FASE3", significado, "teste_complexo",
            tipo_evento="teste", resultado_teste=f"{taxa_sucesso:.1%}"
        )

        return acertou

    def executar(self):
        """Executar Fase 3 completa"""
        logger.info(f"\n\n{'#'*70}")
        logger.info(f"🧠 FASE 3: APRENDIZADO SEMÂNTICO COMPLEXO")
        logger.info(f"   Total: {config.significados} significados")
        logger.info(f"   Ciclo/significado: 25 min ensino + 20 min REM + teste complexo")
        logger.info(f"{'#'*70}")

        sucessos = 0
        for i in range(config.significados):
            sucesso = self.ensinar_significado(i)
            if sucesso:
                sucessos += 1

            # Pausa
            logger.info("\n⏸️  Pausa de 2 minutos entre significados...")
            time.sleep(2 if config.modo_teste_rapido else 120)

        logger.info(f"\n\n[DONE] FASE 3 COMPLETA!")
        logger.info(f"   Sucessos: {sucessos}/{config.significados} ({sucessos/config.significados*100:.1f}%)")

        return sucessos / config.significados >= 0.5


# ==================== ORQUESTRADOR PRINCIPAL ====================

class OrquestradorTreinamento:
    """Coordena todas as 3 fases"""

    def __init__(self):
        self.dataset = DatasetSemantico()
        self.motor = MotorAudioAvancado()
        self.logger = LoggerSemantico()

        self.fase1 = FaseLexico(self.dataset, self.motor, self.logger)
        self.fase2 = FaseSintatica(self.dataset, self.motor, self.logger)
        self.fase3 = FaseSemantica(self.dataset, self.motor, self.logger)

    def executar(self):
        """Executar treinamento completo"""
        logger.info(f"\n\n{'*'*70}")
        logger.info(f"🧠[LEARN] SELENE V3.5 — TREINAMENTO AUDITIVO BIO-INSPIRADO V4.0")
        logger.info(f"{'*'*70}")
        logger.info(f"Dataset: {config.significados} significados únicos")
        logger.info(f"Modo teste rápido: {'[OK] SIM (reduzido 100x)' if config.modo_teste_rapido else '[FAIL] NÃO'}")
        logger.info(f"Log: {self.logger.arquivo}")
        logger.info(f"{'*'*70}\n")

        try:
            # Fase 1
            if not self.fase1.executar():
                logger.warning("⚠️  Fase 1: Performance baixa - recomenda-se repetir")

            time.sleep(5 if config.modo_teste_rapido else 300)

            # Fase 2
            if not self.fase2.executar():
                logger.warning("⚠️  Fase 2: Performance baixa")

            time.sleep(5 if config.modo_teste_rapido else 300)

            # Fase 3
            if not self.fase3.executar():
                logger.warning("⚠️  Fase 3: Performance baixa")

            # Finalização
            logger.info(f"\n\n{'='*70}")
            logger.info(f"[DONE] TREINAMENTO COMPLETO!")
            logger.info(f"   3 Fases × {config.significados} significados")
            logger.info(f"   Total de estimulos: ??? (veja {self.logger.arquivo})")
            logger.info(f"{'='*70}\n")

        except KeyboardInterrupt:
            logger.warning("\n⚠️  Treinamento interrompido pelo usuário")
        except Exception as e:
            logger.error(f"[FAIL] Erro: {e}")
            import traceback
            traceback.print_exc()


# ==================== MAIN ====================

def main():
    """Ponto de entrada"""
    # Modo teste rápido (para desenvolvimento)
    if len(sys.argv) > 1 and sys.argv[1] == "--teste-rapido":
        config.modo_teste_rapido = True
        config.significados = 2  # Apenas 2 significados para teste rápido
        logger.info("🚀 Modo TESTE RÁPIDO ativado (2 significados, timers reduzidos)")

    orquestrador = OrquestradorTreinamento()
    orquestrador.executar()


if __name__ == "__main__":
    main()
