// src/io/pipeline.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
pub struct IOPipeline {
    visual_input: mpsc::Receiver<Vec<f32>>,
    audio_input: mpsc::Receiver<Vec<f32>>,
    text_input: mpsc::Receiver<String>,
    motor_output: mpsc::Sender<Vec<f32>>,
    text_output: mpsc::Sender<String>,
}