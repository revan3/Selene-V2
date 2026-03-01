// src/compressor/salient.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use serde::{Serialize, Deserialize};

/// Representa um ponto saliente de um spike
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SalientPoint {
    pub index: u16,        // até 65535 amostras por spike
    pub amplitude: f32,
    pub point_type: PointType,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PointType {
    Start,
    End,
    LocalMax,
    LocalMin,
    Inflection,
}

/// Compressor que mantém apenas pontos salientes
pub struct SalientCompressor {
    tolerance: f32,
    max_points: usize,
}

impl SalientCompressor {
    pub fn new(tolerance: f32, max_points: usize) -> Self {
        Self { tolerance, max_points }
    }
    
    /// Comprime um spike (vetor de amplitudes) em poucos pontos salientes
    pub fn compress(&self, spike: &[f32]) -> Vec<SalientPoint> {
        if spike.len() < 3 {
            return spike.iter().enumerate().map(|(i, &v)| SalientPoint {
                index: i as u16,
                amplitude: v,
                point_type: PointType::Start,
            }).collect();
        }
        
        let mut salient = Vec::new();
        
        // Primeiro ponto
        salient.push(SalientPoint {
            index: 0,
            amplitude: spike[0],
            point_type: PointType::Start,
        });
        
        // Detecta extremos locais
        for i in 1..spike.len()-1 {
            let prev = spike[i-1];
            let curr = spike[i];
            let next = spike[i+1];
            
            // Máximo local
            if curr > prev && curr > next {
                salient.push(SalientPoint {
                    index: i as u16,
                    amplitude: curr,
                    point_type: PointType::LocalMax,
                });
            }
            // Mínimo local
            else if curr < prev && curr < next {
                salient.push(SalientPoint {
                    index: i as u16,
                    amplitude: curr,
                    point_type: PointType::LocalMin,
                });
            }
            // Ponto de inflexão
            else if (curr - prev).signum() != (next - curr).signum() {
                salient.push(SalientPoint {
                    index: i as u16,
                    amplitude: curr,
                    point_type: PointType::Inflection,
                });
            }
        }
        
        // Último ponto
        salient.push(SalientPoint {
            index: (spike.len()-1) as u16,
            amplitude: spike[spike.len()-1],
            point_type: PointType::End,
        });
        
        // Se excedeu max_points, faz seleção
        if salient.len() > self.max_points {
            salient = self.select_top_k(salient, self.max_points);
        }
        
        salient
    }
    
    /// Seleciona os K pontos mais importantes
    fn select_top_k(&self, points: Vec<SalientPoint>, k: usize) -> Vec<SalientPoint> {
        let mut sorted = points;
        sorted.sort_by(|a, b| {
            b.amplitude.abs().partial_cmp(&a.amplitude.abs()).unwrap()
        });
        
        let mut selected = Vec::new();
        selected.push(sorted.iter().find(|p| p.point_type == PointType::Start).unwrap().clone());
        selected.push(sorted.iter().find(|p| p.point_type == PointType::End).unwrap().clone());
        
        for p in sorted {
            if selected.len() >= k { break; }
            if p.point_type != PointType::Start && p.point_type != PointType::End {
                selected.push(p);
            }
        }
        
        selected.sort_by_key(|p| p.index);
        selected
    }
    
    /// Reconstrói spike a partir de pontos salientes (interpolação linear)
    pub fn decompress(&self, points: &[SalientPoint]) -> Vec<f32> {
        if points.is_empty() { return vec![]; }
        
        let last_index = points.last().unwrap().index as usize;
        let mut reconstructed = vec![0.0; last_index + 1];
        
        for window in points.windows(2) {
            let p1 = window[0];
            let p2 = window[1];
            
            let i1 = p1.index as usize;
            let i2 = p2.index as usize;
            
            for i in i1..=i2 {
                let t = (i - i1) as f32 / (i2 - i1) as f32;
                reconstructed[i] = p1.amplitude * (1.0 - t) + p2.amplitude * t;
            }
        }
        
        reconstructed
    }
}