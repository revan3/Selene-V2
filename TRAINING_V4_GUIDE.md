# 📖 Guia Completo — Selene Training V4.0
## Treinamento Auditivo Bio-Inspirado com 3 Fases Progressivas

---

## 🎯 Visão Geral

Script de treinamento completo para Selene V3.5 com **3 fases progressivas de aprendizado**:

| Fase | Foco | Duração/Significado | Ruído | Teste |
|------|------|-------------------|-------|-------|
| **1. Léxica** | Palavras isoladas | 15 min + 20 REM | Branco | Reconhecimento simples |
| **2. Sintática** | Frases em contexto | 20 min + 20 REM | Chuva, vozes | Compreensão contextual |
| **3. Semântica** | Inferência causal | 25 min + 20 REM | Carro, ambiente | Raciocínio complexo |

**Total:** 3 fases × 60 significados = ~60 horas de treinamento contínuo

---

## 📋 Especificação Detalhada

### Fase 1: Aprendizado Léxico Sequencial
**Objetivo:** Aquisição de vocabulário puro

```
Para cada um dos 60 significados:
├─ 15 minutos: Repetir 5 palavras do grupo
│   ├─ Variação: Entonação (normal, alta, baixa)
│   ├─ Ruído: Silêncio ou branco (20% dos estímulos)
│   └─ Intervalo: ~0.5s entre palavras
│
├─ 20 minutos: Consolidação REM
│   └─ Barra de progresso visual
│
└─ Teste: Reconhecer 3 das 5 palavras (≥60% acerto)
```

**Exemplo prático:**
```
Significado: "Memória"
Palavras: lembrança, recordação, trace, engrama, memória

Ensino (15 min):
  "lembrança" (entonação normal, sem ruído) - reproduz
  [0.5s pausa]
  "recordação" (entonação alta, com ruído branco)
  [0.5s pausa]
  "trace" (entonação baixa, sem ruído)
  [0.5s pausa]
  ... (repete ciclicamente)

REM (20 min):
  [████████████████████░░░░░░░░░░░░░░░░] 50%

Teste:
  Ouça "lembrança" → É memória? SIM ✅
  Ouça "recordação" → É memória? SIM ✅
  Ouça "trace" → É memória? SIM ✅
  Resultado: 3/3 = 100% ✅
```

---

### Fase 2: Aprendizado Sintático Progressivo
**Objetivo:** Consolidação contextual (phrase-level)

```
Para cada um dos 60 significados:
├─ 20 minutos: Repetir 10 frases diferentes
│   ├─ Contextos variados (causal, temporal, relacional)
│   ├─ Entonações: normal, alta, baixa, triste, alegre
│   ├─ Ruído: silêncio, branco, chuva, vozes (30% dos estímulos)
│   └─ Intervalo: ~0.5s entre frases
│
├─ 20 minutos: Consolidação REM
│
└─ Teste: Reconhecer significado em 2 frases (≥50% acerto)
```

**Exemplo prático:**
```
Significado: "Causalidade"
Frases:
  "A causa da memória é a consolidação."
  "Se temos aprendizado, então há causalidade."
  "Causalidade resulta em previsão."
  ... (10 frases)

Ensino (20 min):
  "A causa da memória é..." (entonação normal, chuva)
  [0.5s]
  "Se temos aprendizado..." (entonação alta, vozes)
  [0.5s]
  ... (repete)

Teste:
  Ouça "A causa da memória é..."
  Pergunta: Qual é o conceito principal?
  Resposta: Causalidade ✅
```

---

### Fase 3: Aprendizado Semântico Complexo
**Objetivo:** Inferência, causalidade, generalização

```
Para cada um dos 60 significados:
├─ 25 minutos: Múltiplas camadas de ensino
│   ├─ Camada 1: Palavra isolada
│   ├─ Camada 2: Frase contextual
│   ├─ Camada 3: Contexto expandido
│   ├─ Camada 4: Contraste com conceito oposto
│   ├─ Ruído: Todos os tipos (carro, ambiente, etc)
│   └─ Entonações: Todas variações
│
├─ 20 minutos: Consolidação REM
│
└─ Teste: Inferência causal (≥70% acerto)
```

**Exemplo prático:**
```
Significado: "Consolidação" 
Pergunta complexa: "Como 'Consolidação' se relaciona com 'Memória' e 'Erro'?"

Ensino (25 min):
  Camada 1: "consolidação" [palavra isolada]
  Camada 2: "A consolidação ocorre durante REM"
  Camada 3: "A consolidação é antagonista ao esquecimento"
  Camada 4: "Diferente de 'Reversão', consolidação é permanente"
  ... (com todos os ruídos)

Teste:
  Pergunta: "Consolidação causa ou é causada por aprendizado?"
  Resposta esperada: "Causa posterior a aprendizado" ✅
```

---

## 🚀 Como Usar

### Instalação

```bash
cd f:/Selene_brain_2.0
pip install -r requirements_training.txt
```

### Execução Full (60+ horas)

```bash
python treinar_selene_v4.py
```

**Saída esperada:**
```
**********************************************************************
🧠🎓 SELENE V3.5 — TREINAMENTO AUDITIVO BIO-INSPIRADO V4.0
**********************************************************************
Dataset: 60 significados únicos
Modo teste rápido: ❌ NÃO
Log: selene_training_log.csv

======================================================================
🧠 FASE 1: APRENDIZADO LÉXICO SEQUENCIAL
   Total: 60 significados
   Ciclo/significado: 15 min ensino + 20 min REM + teste
======================================================================

======================================================================
📚 FASE 1 - LÉXICA: Significado 1/60
   Termo: 'Aprendizado'
   Palavras: aprender, estudar, conhecer, assimilar, absorver
======================================================================

⏱️  Ensino por 15 minutos...
   [████████████████████░░░░░░░░░░░░░░░░] 50%

💤 Consolidação REM por 20 minutos...
   [████████████░░░░░░░░░░░░░░░░░░░░░░░░] 33%

  🧪 TESTE FASE 1: Reconhecimento de palavras 'Aprendizado'
    Ouça: aprender
    Pergunta: Esta palavra significa 'Aprendizado'? (s/n)
    ✅ Correto!
...
✨ FASE 1 COMPLETA!
   Sucessos: 48/60 (80.0%)
```

### Teste Rápido (60 segundos)

```bash
python treinar_selene_v4.py --teste-rapido
```

**Características:**
- Apenas 2 significados (vs 60)
- Timers reduzidos 100x
- Perfeito para desenvolvimento e debugging

---

## 📊 Dataset: 60 Significados Únicos

O script gera automaticamente 60 núcleos semânticos:

### Categorias:
- **Cognitivos (5):** Aprendizado, Memória, Atenção, Predição, Erro
- **Neurobiológicos (5):** Neurônio, Sinapse, BDNF, Consolidação, Sono
- **Emocionais (5):** Alegria, Medo, Tristeza, Raiva, Esperança
- **Motivacionais (5):** Recompensa, Punição, Motivação, Dopamina, Salência
- **Sociais (5):** Comunidade, Empatia, Confiança, Oxitocina, Isolamento
- **Estruturais (5):** Padrão, Causalidade, Temporalidade, Espaço, Quantidade
- **Procedimental (5):** Ritmo, Movimento, Pausa, Aceleração, Desaceleração
- **Perceptual (5):** Brilho, Escuridão, Som, Silêncio, Sabor
- **Abstratos (5):** Verdade, Falsidade, Complexidade, Simplicidade, Infinito
- **Transformação (5):** Crescimento, Declínio, Reversão, Inovação, Tradição
- **Controle (5):** Liberdade, Controle, Caos, Ordem, Harmonia
- **Biológico (5):** Vida, Morte, Saúde, Doença, Adaptação

**Total:** 12 categorias × 5 significados = **60 significados únicos**

Cada significado = 5 sinônimos (Fase 1) e 10 frases (Fase 2+)

---

## 🔊 Variações Acústicas

### Entonações Implementadas
```python
Entonacao.NORMAL  = 1.0x velocidade
Entonacao.ALTA    = 1.3x velocidade (pergunta/surpresa)
Entonacao.BAIXA   = 0.7x velocidade (afirmação/certeza)
Entonacao.TRISTE  = 0.9x velocidade
Entonacao.ALEGRE  = 1.2x velocidade
```

### Ruídos Contextual (Fase 1 → 3 progressivos)

**Fase 1 (simples):**
- Silêncio (0% ruído)
- Branco (5% amplitude)

**Fase 2 (intermediário):**
- + Chuva (envelope 0.5 Hz)
- + Vozes dispersas (3 frequências: 200, 400, 800 Hz)

**Fase 3 (complexo):**
- + Carro (frequência variável 150-200 Hz)
- + Ambiente (mix de 5 frequências: 100-800 Hz)

---

## 📝 Logging

### Arquivo: `selene_training_log.csv`

```csv
timestamp,fase,significado,item,entonacao,ruido,duracao_s,tipo_evento,resultado_teste
2026-05-04T14:30:00.123,FASE1,Aprendizado,aprender,NORMAL,white,0.85,ensino,-
2026-05-04T14:30:01.456,FASE1,Aprendizado,estudar,ALTA,silence,0.92,ensino,-
2026-05-04T14:30:02.789,FASE1,Aprendizado,conhecer,BAIXA,white,0.88,ensino,-
...
2026-05-04T14:50:00.000,FASE1,Aprendizado,test,NORMAL,silence,1.23,teste,100.0%
```

### Análise Posterior

```bash
# Visualizar estadísticas
python -c "
import pandas as pd
df = pd.read_csv('selene_training_log.csv')
print('Total eventos:', len(df))
print('Por fase:')
print(df.groupby('fase').size())
print('Taxa sucesso média:', df[df['resultado_teste'] != '-']['resultado_teste'].str.rstrip('%').astype(float).mean())
"
```

---

## 🔧 Configuração Avançada

### Modificar Timers

```python
# Em treinar_selene_v4.py, editar:
config = TrainingConfig(
    tempo_ensino_lexico_min=15,      # 15 min (vs padrão)
    tempo_ensino_sintatico_min=20,   # 20 min
    tempo_ensino_semantico_min=25,   # 25 min
    tempo_rem_min=20,                 # 20 min consolidação
)
```

### Modificar Dataset

```python
config = TrainingConfig(
    significados=30,                  # 30 (vs 60)
    palavras_por_significado=5,       # 5 sinônimos
    frases_por_significado=10,        # 10 frases
)
```

---

## 📊 Duração Total Estimada

```
Fase 1: 60 significados × (15 min + 20 min REM) = 35h
Fase 2: 60 significados × (20 min + 20 min REM) = 40h
Fase 3: 60 significados × (25 min + 20 min REM) = 45h
Pausas: 5min entre significados = 5h
───────────────────────────────────────────────────
TOTAL: ~125 horas = 5 dias contínuos (24h/dia)
```

**Para parar:** Ctrl+C (salva progresso em CSV)

---

## 🐛 Troubleshooting

### ❌ "pyttsx3 não encontrado"
```bash
pip install pyttsx3
# WSL: apt-get install espeak
```

### ❌ "sounddevice não encontrado"
```bash
pip install sounddevice soundfile
```

### ❌ "Reprodução muito lenta"
- Use `--teste-rapido` para verificar lógica
- Aumente `config.tempo_ensino_*_min` ou reduz `significados`

### ❌ "CSV muito grande"
- Arquivo cresce ~1MB/hora
- Normal: 125h = ~125MB
- Limpar: `rm selene_training_log.csv && python treinar_selene_v4.py`

---

## 🎓 Aprendizado Esperado

### Após Fase 1 (35h):
- ✅ Reconhecer 300 palavras (60 grupos × 5)
- ✅ Associar som a significado
- Taxa esperada: 80%+ acerto em testes

### Após Fase 2 (75h):
- ✅ Compreender contexto de 600 frases
- ✅ Detectar significado em frase
- Taxa esperada: 75%+ acerto em testes

### Após Fase 3 (120h):
- ✅ Inferir causalidade
- ✅ Generalizar conceitos
- ✅ Distinguir sutilezas semânticas
- Taxa esperada: 70%+ acerto em testes complexos

---

## 🚀 Roadmap Futuro

### V4.1
- [ ] Feedback em tempo real (gráficos de performance)
- [ ] Ajuste dinâmico de dificuldade

### V4.2
- [ ] Multi-idioma (português + inglês + espanhol)
- [ ] Integração com Selene WebSocket

### V4.3
- [ ] Neurofeedback (EEG/fNIRS opcional)
- [ ] Treinamento adaptativo

---

**Mantido por:** Rodrigo Luz (Pai) + Claude Haiku 4.5  
**Última atualização:** 2026-05-04  
**Status:** ✅ Pronto para produção
