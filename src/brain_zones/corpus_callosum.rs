//corpus_callosum.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use std::collections::VecDeque;
use rand::{Rng, thread_rng};
use rand::rngs::ThreadRng;
use crate::brain_zones::RegionType;

#[derive(Debug)] 
pub struct CorpusCallosum {
    latency_ms: f32,
    buffer_l_to_r: Vec<VecDeque<(f32, Vec<bool>)>>,
    buffer_r_to_l: Vec<VecDeque<(f32, Vec<bool>)>>,
    pub connectivity_factor: f32,
}

impl CorpusCallosum {
    pub fn new(latency: f32, num_channels: usize) -> Self {
        let mut l_to_r = Vec::with_capacity(num_channels);
        let mut r_to_l = Vec::with_capacity(num_channels);
        
        for _ in 0..num_channels {
            l_to_r.push(VecDeque::new());
            r_to_l.push(VecDeque::new());
        }

        Self {
            latency_ms: latency,
            buffer_l_to_r: l_to_r,
            buffer_r_to_l: r_to_l,
            connectivity_factor: 1.0,
        }
    }

    pub fn send_to_right(&mut self, channel: usize, spikes: Vec<bool>, current_time: f32) {
        let mut rng = thread_rng();
        
        if rng.gen_bool(self.connectivity_factor as f64) {
            let jitter = rng.gen_range(-1.5..1.5);
            let arrival = current_time + self.latency_ms + jitter;
            
            if let Some(buf) = self.buffer_l_to_r.get_mut(channel) {
                buf.push_back((arrival, spikes));
            }
        }
    }

    pub fn receive_at_right(&mut self, channel: usize, current_time: f32) -> Option<Vec<bool>> {
        if let Some(buf) = self.buffer_l_to_r.get_mut(channel) {
            if let Some((arrival, _)) = buf.front() {
                if current_time >= *arrival {
                    return buf.pop_front().map(|(_, spikes)| spikes);
                }
            }
        }
        None
    }

    pub fn send_to_left(&mut self, channel: usize, spikes: Vec<bool>, current_time: f32) {
        let mut rng = thread_rng();
        
        if rng.gen_bool(self.connectivity_factor as f64) {
            let jitter = rng.gen_range(-1.5..1.5);
            let arrival = current_time + self.latency_ms + jitter;
            
            if let Some(buf) = self.buffer_r_to_l.get_mut(channel) {
                buf.push_back((arrival, spikes));
            }
        }
    }

    pub fn receive_at_left(&mut self, channel: usize, current_time: f32) -> Option<Vec<bool>> {
        if let Some(buf) = self.buffer_r_to_l.get_mut(channel) {
            if let Some((arrival, _)) = buf.front() {
                if current_time >= *arrival {
                    return buf.pop_front().map(|(_, spikes)| spikes);
                }
            }
        }
        None
    }

    pub fn set_connectivity(&mut self, health: f32) {
        self.connectivity_factor = health.clamp(0.0, 1.0);
    }

    /// Ajusta a latência de transmissão inter-hemisférica.
    /// Arousal alto → transmissão mais rápida (min 4ms).
    /// Arousal baixo/sono → latência maior (max 20ms).
    pub fn set_latency(&mut self, latency_ms: f32) {
        self.latency_ms = latency_ms.clamp(4.0, 20.0);
    }
}