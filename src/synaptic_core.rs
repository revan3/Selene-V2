// src/synaptic_core.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use half::f16;
use crate::config::Config;
use crate::compressor::salient::{SalientPoint, SalientCompressor};

// Tipos de precisão para neurônios
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PrecisionType {
    FP32,  // 4 bytes - precisão máxima (5% dos neurônios)
    FP16,  // 2 bytes - equilíbrio (35% dos neurônios)
    INT8,  // 1 byte - alta densidade (50% dos neurônios)
    INT4,  // 0.5 bytes - ultra-denso (10% dos neurônios)
}

// Representação compacta para INT4 (4 bits)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Int4(u8);  // Armazena dois valores de 4 bits

impl Int4 {
    pub fn new(valor: i8) -> Self {
        Self(valor.max(-8).min(7) as u8 & 0x0F)
    }
    
    pub fn valor(&self) -> i8 {
        let val = self.0 as i8;
        if val & 0x08 != 0 {
            (val | -16) as i8
        } else {
            val
        }
    }
}

// Peso do neurônio com escala
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PesoNeuronio {
    FP32(f32),
    FP16(f16),
    INT8(i8, f32),      // valor + fator de escala
    INT4(Int4, f32),    // valor 4-bit + fator de escala
}

impl PesoNeuronio {
    pub fn valor_f32(&self) -> f32 {
        match self {
            PesoNeuronio::FP32(v) => *v,
            PesoNeuronio::FP16(v) => v.to_f32(),
            PesoNeuronio::INT8(v, scale) => (*v as f32) * scale,
            PesoNeuronio::INT4(v, scale) => (v.valor() as f32) * scale,
        }
    }
    
    pub fn bytes_por_neuronio(&self) -> usize {
        match self {
            PesoNeuronio::FP32(_) => 4,
            PesoNeuronio::FP16(_) => 2,
            PesoNeuronio::INT8(_, _) => 1 + 4, // valor + scale (scale compartilhado)
            PesoNeuronio::INT4(_, _) => 1, // 2 valores por byte + scale compartilhado
        }
    }
}

// Neurônio híbrido
#[derive(Debug, Clone)]
pub struct NeuronioHibrido {
    pub id: Uuid,
    pub precisao: PrecisionType,
    pub peso: PesoNeuronio,
    
    // Estado (sempre FP32 para estabilidade)
    pub v: f32,           // potencial de membrana
    pub u: f32,           // variável de recuperação
    pub a: f32,           // parâmetros Izhikevich
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub g_scale_hh: f32, // ← ADICIONADO
    pub threshold: f32,
    pub refr_count: u16,
    
    // STDP
    pub trace: f32,
    pub last_spike_time: f32,
    
    // Opcional: região (para neurogênese)
    pub regiao: Option<crate::brain_zones::RegionType>,
}

impl NeuronioHibrido {
    pub fn new(precisao: PrecisionType) -> Self {
        let id = Uuid::new_v4();
        
        let peso = match precisao {
            PrecisionType::FP32 => PesoNeuronio::FP32(1.0),
            PrecisionType::FP16 => PesoNeuronio::FP16(f16::from_f32(1.0)),
            PrecisionType::INT8 => PesoNeuronio::INT8(100, 0.01), // 100 * 0.01 = 1.0
            PrecisionType::INT4 => {
                let val = Int4::new(8); // 8 * 0.125 = 1.0
                PesoNeuronio::INT4(val, 0.125)
            },
        };
        
        Self {
            id,
            precisao,
            peso,
            v: -65.0,
            u: 0.0,
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
            g_scale_hh: 1.0, // ← ADICIONADO
            threshold: 30.0,
            refr_count: 0,
            trace: 0.0,
            last_spike_time: -100.0,
            regiao: None,
        }
    }
    
    pub fn update(&mut self, input_current: f32, dt: f32, current_time: f32) -> bool {
        // Período refratário
        if self.refr_count > 0 {
            self.refr_count -= 1;
            self.v = -70.0;
            return false;
        }
        
        // Converte input para a precisão do neurônio
        let input_adaptado = match self.peso {
            PesoNeuronio::INT8(_, scale) => {
                let quantizado = (input_current / scale).round().max(-128.0).min(127.0) as i8;
                (quantizado as f32) * scale
            },
            PesoNeuronio::INT4(_, scale) => {
                let quantizado = (input_current / scale).round().max(-8.0).min(7.0) as i8;
                (quantizado as f32) * scale
            },
            _ => input_current,
        };
        
        // Integração Izhikevich
        self.v += dt * (0.04 * self.v.powi(2) + 5.0 * self.v + 140.0 - self.u + input_adaptado);
        self.u += dt * self.a * (self.b * self.v - self.u);
        
        // Decaimento do traço STDP
        self.trace *= (-dt / 20.0).exp();
        
        // Detecção de spike
        if self.v >= self.threshold {
            self.v = self.c;
            self.u += self.d;
            self.refr_count = (2.0 / dt) as u16;
            
            // STDP
            self.peso = match self.peso {
                PesoNeuronio::FP32(v) => PesoNeuronio::FP32((v + 0.01 * self.trace).min(2.0)),
                PesoNeuronio::FP16(v) => {
                    let novo = (v.to_f32() + 0.01 * self.trace).min(2.0);
                    PesoNeuronio::FP16(f16::from_f32(novo))
                },
                PesoNeuronio::INT8(v, scale) => {
                    let novo_f32 = (v as f32 * scale) + 0.01 * self.trace;
                    let novo_v = (novo_f32 / scale).round().max(-128.0).min(127.0) as i8;
                    PesoNeuronio::INT8(novo_v, scale)
                },
                PesoNeuronio::INT4(v, scale) => {
                    let novo_f32 = (v.valor() as f32 * scale) + 0.01 * self.trace;
                    let novo_v = Int4::new((novo_f32 / scale).round().max(-8.0).min(7.0) as i8);
                    PesoNeuronio::INT4(novo_v, scale)
                },
            };
            
            self.trace = 1.0;
            self.last_spike_time = current_time;
            true
        } else {
            false
        }
    }
}

// Camada de neurônios híbridos
#[derive(Debug)] 
pub struct CamadaHibrida {
    pub neuronios: Vec<NeuronioHibrido>,
    pub escala_compartilhada: Option<f32>,
    nome: String,
}

impl CamadaHibrida {
    pub fn new(n_neurons: usize, nome: &str, distribuicao: Option<Vec<(PrecisionType, f32)>>) -> Self {
        let mut neuronios = Vec::with_capacity(n_neurons);
        
        let dist = distribuicao.unwrap_or_else(|| vec![
            (PrecisionType::FP32, 0.05),
            (PrecisionType::FP16, 0.35),
            (PrecisionType::INT8, 0.50),
            (PrecisionType::INT4, 0.10),
        ]);
        
        let total: f32 = dist.iter().map(|(_, p)| p).sum();
        let dist: Vec<(PrecisionType, f32)> = dist.into_iter()
            .map(|(t, p)| (t, p / total))
            .collect();
        
        let mut acumulado = 0.0;
        let mut dist_iter = dist.iter().peekable();
        let (mut tipo_atual, mut prob_atual) = *dist_iter.next().unwrap(); // ← CORRIGIDO
        
        for i in 0..n_neurons {
            let progresso = i as f32 / n_neurons as f32;
            
            while progresso > acumulado + prob_atual { // ← CORRIGIDO
                acumulado += prob_atual; // ← CORRIGIDO
                if let Some((t, p)) = dist_iter.next() {
                    tipo_atual = *t; // ← CORRIGIDO
                    prob_atual = *p; // ← CORRIGIDO
                } else {
                    break;
                }
            }
            
            neuronios.push(NeuronioHibrido::new(tipo_atual)); // ← CORRIGIDO
        }
        
        Self {
            neuronios,
            escala_compartilhada: Some(0.01),
            nome: nome.to_string(),
        }
    }
    
    pub fn update(&mut self, input_currents: &[f32], dt: f32, current_time: f32) -> Vec<bool> {
        let mut spikes = Vec::with_capacity(self.neuronios.len());
        
        for (i, neuronio) in self.neuronios.iter_mut().enumerate() {
            let input = input_currents.get(i).copied().unwrap_or(0.0);
            let spike = neuronio.update(input, dt, current_time);
            spikes.push(spike);
        }
        
        spikes
    }
    
    /// Versão compactada: recebe spikes já comprimidos
    pub fn update_compact(
        &mut self, 
        input_points: &[Vec<SalientPoint>], 
        dt: f32, 
        current_time: f32, 
        compressor: &SalientCompressor
    ) -> Vec<bool> {
        let mut spikes = Vec::with_capacity(self.neuronios.len());
        
        for (i, neuronio) in self.neuronios.iter_mut().enumerate() {
            let pontos = input_points.get(i).cloned().unwrap_or_default();
            let spike_reconstruido = compressor.decompress(&pontos);
            let input_current = spike_reconstruido.iter().sum::<f32>() / spike_reconstruido.len().max(1) as f32;
            let spike = neuronio.update(input_current, dt, current_time);
            spikes.push(spike);
        }
        
        spikes
    }
    
    pub fn adicionar_neuronio(&mut self, precisao: PrecisionType) -> Uuid {
        let neuronio = NeuronioHibrido::new(precisao);
        let id = neuronio.id;
        self.neuronios.push(neuronio);
        id
    }
    
    pub fn estatisticas(&self) -> CamadaStats {
        let mut stats = CamadaStats::default();
        
        for neuronio in &self.neuronios {
            stats.total += 1;
            match neuronio.precisao {
                PrecisionType::FP32 => stats.fp32 += 1,
                PrecisionType::FP16 => stats.fp16 += 1,
                PrecisionType::INT8 => stats.int8 += 1,
                PrecisionType::INT4 => stats.int4 += 1,
            }
            
            stats.bytes_total += neuronio.peso.bytes_por_neuronio();
        }
        
        stats
    }
}

#[derive(Debug, Default)]
pub struct CamadaStats {
    pub total: usize,
    pub fp32: usize,
    pub fp16: usize,
    pub int8: usize,
    pub int4: usize,
    pub bytes_total: usize,
}

impl CamadaStats {
    pub fn media_bytes(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.bytes_total as f32 / self.total as f32 }
    }
}