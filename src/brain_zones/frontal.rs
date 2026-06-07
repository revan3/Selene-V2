// src/brain_zones/frontal.rs
// Córtex Pré-Frontal — Decisão executiva, working memory, controle inibitório
//
// Composição neuronal:
//   executive_layer: 80% RS (decisão) + 20% FS (inibição lateral)
//   inhibitory_layer: 100% FS (interneurônios GABAérgicos)
//
// Melhorias v2.6:
//   Working memory: buffer de 7 slots (Miller's Law) com gate dopaminérgico.
//     - Codificação: dopamina ≥ threshold → slot é gravado
//     - Manutenção: sustentação por recorrência (decay lento, 0.992)
//     - Limpeza: serotonina baixa → decay acelerado (ansiedade → distração)
//   Planejamento temporal: fila de até 5 goals com prioridade
//   Top-down suppression: sinal inibitório enviado de volta às áreas sensoriais

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::synaptic_core::{CamadaHibrida, PrecisionType, TipoNeuronal};
use crate::stem_cell::GestorNeurogenese;
use crate::encoding::spike_codec::SpikePattern;
use rand::{Rng, thread_rng};
use crate::config::Config;
use std::collections::VecDeque;

/// Capacidade do Episodic Buffer (Baddeley 2000): 4 episódios simultâneos.
const EPISODIC_BUFFER_CAP: usize = 4;

// ─────────────────────────────────────────────────────────────────────────────
// Working Memory — Modelo de Baddeley (1974) + Hasselmo (2006)
// ─────────────────────────────────────────────────────────────────────────────

/// Slots alocados estruturalmente (capacidade máxima do buffer).
const WM_SLOTS: usize = 7;
/// Capacidade real de chunks ativos (Cowan 2001, revisão de Miller 1956: 4±1).
/// Acima deste limite, o item menos saliente é evictado antes do encoding.
const WM_CHUNK_LIMIT: usize = 4;
/// Threshold de dopamina para gravar novo item na working memory.
const WM_ENCODE_THRESHOLD: f32 = 0.85;
/// Decay da working memory por tick (recorrência sustentada).
/// 0.992 ≈ meia-vida de ~87 ticks (0.44s @ 200Hz) — típico de delay tasks.
const WM_DECAY: f32 = 0.992;
/// Boost na atividade quando um slot é "lido" pelo processo de decisão.
const WM_READOUT_BOOST: f32 = 1.15;

/// Slot de working memory: padrão + saliência + timestamp.
#[derive(Clone, Debug)]
struct WmSlot {
    padrao:    Vec<f32>,  // padrão de ativação armazenado (normalizado)
    saliencia: f32,       // importância (dopamina no momento de encoding)
    idade:     u32,       // ticks desde encoding (para substituição por LRU)
    ativo:     bool,
}

impl WmSlot {
    fn vazio(n: usize) -> Self {
        Self { padrao: vec![0.0; n], saliencia: 0.0, idade: 0, ativo: false }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Goal Planning — Fila de objetivos com prioridade
// ─────────────────────────────────────────────────────────────────────────────

/// Objetivo representado como padrão de ativação alvo + prioridade.
#[derive(Clone, Debug)]
pub struct Goal {
    pub padrao:    Vec<f32>,
    pub prioridade: f32,  // 0..1
    pub descricao: String,
    pub ticks_vida: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// FrontalLobe
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct FrontalLobe {
    /// Camada executiva: RS (decisão) + FS (auto-inibição)
    pub executive_layer: CamadaHibrida,
    /// Camada inibitória pura: FS GABAérgico
    pub inhibitory_layer: CamadaHibrida,
    pub dopamine_level:   f32,
    pub serotonin_level:  f32,

    /// Working memory com 7 slots (Baddeley model)
    wm_slots: Vec<WmSlot>,
    /// Traço agregado exposto externamente (compatível com código anterior)
    pub working_memory_trace: Vec<f32>,

    /// Fila de goals planejados (máx 5)
    pub goal_queue: VecDeque<Goal>,

    /// Sinal de top-down suppression — enviado para áreas sensoriais.
    /// Valores positivos = suprimir (o frontal está focado em pensamento interno).
    pub suppression_signal: Vec<f32>,

    pub inhibition_strength: f32,
    pub noise_std: f32,
    n_exec: usize,

    /// Número de chunks ativos na WM agora [0, WM_CHUNK_LIMIT].
    /// Atualizado a cada tick de decide(). Expõe carga de WM externamente.
    pub wm_chunk_count: usize,

    /// Episodic Buffer de Baddeley (2000) — interface WM ↔ LTM.
    /// Armazena até 4 episódios hipocampais de alta saliência (arousal > 0.4).
    /// Palavras aqui têm boost no walk semântico inicial (acesso preferencial).
    episodic_buffer: VecDeque<(String, SpikePattern, f32)>,

    /// V4.6 — Neurogênese autônoma (célula-tronco). Avalia a necessidade da
    /// executive_layer e implanta/julga neurônios Hybrid APENAS durante o sono.
    /// Os novos neurônios entram na CAUDA (índice ≥ n_exec) → não tocam na WM
    /// (decide() só lê os primeiros n_exec).
    pub gestor_neurogenese: GestorNeurogenese,
}

impl FrontalLobe {
    pub fn new(n_executive: usize, inhibition_ratio: f32, noise_std: f32, config: &Config) -> Self {
        let n_inhib = (n_executive as f32 * inhibition_ratio).max(1.0) as usize;

        let exec_dist = vec![
            (PrecisionType::FP32, 0.05),
            (PrecisionType::FP16, 0.55),
            (PrecisionType::INT8, 0.30),
            (PrecisionType::INT4, 0.10),
        ];
        let inhib_dist = vec![
            (PrecisionType::FP16, 0.20),
            (PrecisionType::INT8, 0.60),
            (PrecisionType::INT4, 0.20),
        ];

        let escala = 50.0 / 127.0;

        let mut executive_layer = CamadaHibrida::new(
            n_executive, "frontal_exec",
            TipoNeuronal::RS,
            Some((TipoNeuronal::FS, 0.20)),
            Some(exec_dist),
            escala,
        );
        executive_layer.init_lateral_inhibition(6, 3.5);

        let inhibitory_layer = CamadaHibrida::new(
            n_inhib, "frontal_inhib",
            TipoNeuronal::FS,
            None,
            Some(inhib_dist),
            escala,
        );

        // Inicializa WM slots
        let wm_slots = (0..WM_SLOTS).map(|_| WmSlot::vazio(n_executive)).collect();

        Self {
            executive_layer,
            inhibitory_layer,
            dopamine_level: 1.0,
            serotonin_level: 1.0,
            wm_slots,
            working_memory_trace: vec![0.0; n_executive],
            goal_queue: VecDeque::with_capacity(5),
            suppression_signal: vec![0.0; n_executive],
            inhibition_strength: 6.5,
            noise_std,
            n_exec: n_executive,
            wm_chunk_count: 0,
            episodic_buffer: VecDeque::with_capacity(EPISODIC_BUFFER_CAP),
            // Cap 4 células em prova simultâneas → orçamento populacional (protege RAM).
            gestor_neurogenese: GestorNeurogenese::novo(4),
        }
    }

    /// V4.6 — Hook de neurogênese para ser chamado UMA vez por ciclo de sono
    /// (a partir de main.rs, ao lado de `amygdala.extinção_durante_sono()`).
    ///
    /// Julga as células-tronco em prova do ciclo anterior e, se a executive_layer
    /// estiver desequilibrada (runaway ou subutilizada), nasce/diferencia/implanta
    /// um neurônio Hybrid sob medida. Devolve (aceitas, rejeitadas, nascidas).
    pub fn neurogenese_no_sono(&mut self) -> (usize, usize, usize) {
        self.gestor_neurogenese.tick_sono(&mut self.executive_layer)
    }

    pub fn decide(
        &mut self,
        sensory_input: &[f32],
        goal_bias:     &[f32],
        dt:            f32,
        current_time:  f32,
        config:        &Config,
    ) -> Vec<f32> {
        let n = self.n_exec;
        let mut rng = thread_rng();
        let t_ms = current_time * 1000.0;

        // ── 1. Lê working memory (readout dos slots) ──────────────────────
        // Slot mais recente e mais saliente domina o traço de WM
        self.working_memory_trace = vec![0.0; n];
        let mut melhor_sal = 0.0f32;
        for slot in &mut self.wm_slots {
            if !slot.ativo { continue; }
            slot.idade += 1;
            // Decay da WM — serotonina baixa = decay mais rápido (distração)
            let decay = WM_DECAY * (0.85 + self.serotonin_level * 0.15);
            for v in &mut slot.padrao { *v *= decay; }
            // Saliência também decai
            slot.saliencia *= 0.998;
            if slot.saliencia < 0.02 { slot.ativo = false; }
            // Contribui para o traço com peso proporcional à saliência
            if slot.saliencia > melhor_sal {
                melhor_sal = slot.saliencia;
                for i in 0..n.min(slot.padrao.len()) {
                    self.working_memory_trace[i] = slot.padrao[i] * slot.saliencia * WM_READOUT_BOOST;
                }
            }
        }

        // ── 2. Goal atual (topo da fila) ──────────────────────────────────
        let goal_current: Vec<f32> = if let Some(goal) = self.goal_queue.front_mut() {
            goal.ticks_vida += 1;
            // Goal com mais de 2000 ticks (~10s) é removido (timeout)
            if goal.ticks_vida > 2000 {
                self.goal_queue.pop_front();
                vec![0.0; n]
            } else {
                let p = goal.prioridade;
                let pat = goal.padrao.clone();
                (0..n).map(|i| pat.get(i).copied().unwrap_or(0.0) * p).collect()
            }
        } else {
            vec![0.0; n]
        };

        // ── 3. Combina entradas ────────────────────────────────────────────
        let gain = 1.0 + self.dopamine_level * 0.8;
        let mut combined_input = vec![0.0f32; n];
        for i in 0..n {
            let noise = rng.gen_range(-self.noise_std..self.noise_std);
            combined_input[i] = (
                sensory_input.get(i).copied().unwrap_or(0.0)
                + goal_bias.get(i).copied().unwrap_or(0.0) * 1.5
                + goal_current.get(i).copied().unwrap_or(0.0) * 1.2
                + self.working_memory_trace[i]
                + noise
            ) * gain;
        }

        // ── 4. Atualiza camadas ────────────────────────────────────────────
        let executive_spikes = self.executive_layer.update(&combined_input, dt, t_ms);
        let active_count = executive_spikes.iter().filter(|&&s| s).count() as f32;
        let inhibition_input = vec![active_count * 0.4; self.inhibitory_layer.neuronios.len()];
        let inhibitory_spikes = self.inhibitory_layer.update(&inhibition_input, dt, t_ms);

        let n_inhib = self.inhibitory_layer.neuronios.len();
        let mut output = vec![0.0f32; n];

        for i in 0..n {
            if inhibitory_spikes[i % n_inhib] {
                let drop = self.inhibition_strength * (1.0 + self.dopamine_level * 0.2);
                if let Some(nr) = self.executive_layer.neuronios.get_mut(i) {
                    nr.v -= drop;
                }
            }
            output[i] = if executive_spikes[i] { 1.0 } else { 0.0 };
        }

        // ── 5. Grava na working memory se saliente ────────────────────────
        if self.dopamine_level >= WM_ENCODE_THRESHOLD {
            let sal_nova = (self.dopamine_level - WM_ENCODE_THRESHOLD) / (1.0 - WM_ENCODE_THRESHOLD);
            let n_ativos = self.wm_slots.iter().filter(|s| s.ativo).count();

            // Cowan (2001): limite real = 4±1 chunks — acima do limite, evicta o menos saliente
            let slot_idx = if n_ativos >= WM_CHUNK_LIMIT {
                self.wm_slots.iter().enumerate()
                    .filter(|(_, s)| s.ativo)
                    .min_by(|a, b| a.1.saliencia.partial_cmp(&b.1.saliencia).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            } else {
                // Abaixo do limite: prefere slot vazio, senão o menos saliente
                self.wm_slots.iter().enumerate()
                    .min_by(|a, b| {
                        let sa = if a.1.ativo { a.1.saliencia } else { -1.0 };
                        let sb = if b.1.ativo { b.1.saliencia } else { -1.0 };
                        sa.partial_cmp(&sb).unwrap()
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            };

            let slot = &mut self.wm_slots[slot_idx];
            slot.padrao = output.clone();
            slot.saliencia = sal_nova;
            slot.idade = 0;
            slot.ativo = true;
        }

        // Mantém wm_chunk_count sincronizado (inclui decays do passo 1)
        self.wm_chunk_count = self.wm_slots.iter().filter(|s| s.ativo).count();

        // ── 6. Top-down suppression ────────────────────────────────────────
        // Quando o frontal está ativamente deliberando (muitos spikes executivos),
        // envia sinal inibitório para áreas sensoriais — "não me perturbe agora".
        // Biologicamente: projeção glutamatérgica para interneurônios GABAérgicos do V1/temporal.
        let foco_interno = active_count / n as f32;  // 0..1, proporcional ao engajamento
        for i in 0..n {
            // Sinal de supressão proporcional ao output executivo
            self.suppression_signal[i] = output[i] * foco_interno * self.dopamine_level * 0.3;
        }

        output
    }

    /// Adiciona um novo goal à fila de planejamento.
    /// Goals com prioridade maior deslocam os mais antigos se a fila estiver cheia.
    pub fn planejar(&mut self, padrao: Vec<f32>, prioridade: f32, descricao: &str) {
        if self.goal_queue.len() >= 5 {
            // Remove o goal menos prioritário
            if let Some(min_idx) = self.goal_queue.iter().enumerate()
                .min_by(|a, b| a.1.prioridade.partial_cmp(&b.1.prioridade).unwrap())
                .map(|(i, _)| i)
            {
                if self.goal_queue[min_idx].prioridade < prioridade {
                    self.goal_queue.remove(min_idx);
                } else {
                    return; // goal novo é menos importante — descarta
                }
            }
        }
        // Insere na posição correta por prioridade
        let pos = self.goal_queue.iter().position(|g| g.prioridade < prioridade)
            .unwrap_or(self.goal_queue.len());
        self.goal_queue.insert(pos, Goal {
            padrao,
            prioridade: prioridade.clamp(0.0, 1.0),
            descricao: descricao.to_string(),
            ticks_vida: 0,
        });
    }

    /// Avalia se o goal atual foi atingido comparando o output atual com o padrão do goal.
    /// similarity > threshold → goal concluído (removido da fila), retorna RPE positivo.
    /// ticks_vida > timeout → goal expirado sem sucesso, retorna RPE negativo leve.
    /// Retorna (rpe: f32, goal_desc: Option<String>) onde rpe != 0 indica evento.
    pub fn avaliar_goal(&mut self, output_atual: &[f32]) -> (f32, Option<String>) {
        let Some(goal) = self.goal_queue.front() else { return (0.0, None) };
        let n = goal.padrao.len().min(output_atual.len());
        if n == 0 { return (0.0, None); }

        // Similaridade cosseno entre padrão do goal e output atual
        let dot: f32  = (0..n).map(|i| goal.padrao[i] * output_atual[i]).sum();
        let mag_g: f32 = (0..n).map(|i| goal.padrao[i].powi(2)).sum::<f32>().sqrt();
        let mag_o: f32 = (0..n).map(|i| output_atual[i].powi(2)).sum::<f32>().sqrt();
        let sim = if mag_g > 1e-6 && mag_o > 1e-6 { dot / (mag_g * mag_o) } else { 0.0 };

        let threshold = 0.65 * goal.prioridade.max(0.4);
        let prioridade = goal.prioridade; // copia antes de soltar o borrow
        if sim >= threshold {
            // Goal atingido → RPE positivo proporcional à prioridade
            let desc = self.goal_queue.pop_front().map(|g| g.descricao);
            let rpe = prioridade * 0.4;
            return (rpe, desc);
        }
        // Expirado sem sucesso (timeout já tratado no decide())
        (0.0, None)
    }

    pub fn set_dopamine(&mut self, level: f32) {
        self.dopamine_level = level.clamp(0.3, 2.5);
    }

    pub fn set_serotonin(&mut self, level: f32) {
        self.serotonin_level = level.clamp(0.0, 2.0);
    }

    /// Número de chunks ativos na WM [0..WM_CHUNK_LIMIT].
    pub fn wm_ocupacao(&self) -> usize {
        self.wm_chunk_count
    }

    /// Retorna os padrões de ativação média dos slots de WM ativos.
    /// Cada slot ativo contribui com sua saliência e padrão normalizado.
    /// Usado pelo neural_context para expor o que o frontal está "pensando".
    /// Retorna Vec de (saliência, padrão_resumo) para os slots mais salientes.
    pub fn wm_snapshots(&self) -> Vec<(f32, f32)> {
        self.wm_slots.iter()
            .filter(|s| s.ativo)
            .map(|s| {
                let media = s.padrao.iter().sum::<f32>() / s.padrao.len().max(1) as f32;
                (s.saliencia, media)
            })
            .collect()
    }

    /// Registra um episódio hipocampal no Episodic Buffer se arousal > 0.4.
    /// Se buffer cheio, descarta o item mais antigo (FIFO — recência importa).
    pub fn push_episodio(&mut self, palavra: &str, spike: SpikePattern, arousal: f32) {
        if arousal <= 0.4 { return; }
        if self.episodic_buffer.len() >= EPISODIC_BUFFER_CAP {
            self.episodic_buffer.pop_front();
        }
        self.episodic_buffer.push_back((palavra.to_string(), spike, arousal));
    }

    /// Retorna as palavras atualmente no Episodic Buffer (mais antigas primeiro).
    pub fn episodic_words(&self) -> Vec<String> {
        self.episodic_buffer.iter().map(|(w, _, _)| w.clone()).collect()
    }

    pub fn estatisticas(&self) -> FrontalStats {
        FrontalStats {
            executive:  self.executive_layer.estatisticas(),
            inhibitory: self.inhibitory_layer.estatisticas(),
            dopamine:   self.dopamine_level,
            wm_slots_ativos: self.wm_ocupacao(),
            n_goals: self.goal_queue.len(),
        }
    }
}

pub struct FrontalStats {
    pub executive:       crate::synaptic_core::CamadaStats,
    pub inhibitory:      crate::synaptic_core::CamadaStats,
    pub dopamine:        f32,
    pub wm_slots_ativos: usize,
    pub n_goals:         usize,
}
