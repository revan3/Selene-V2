// src/websocket/converter.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
impl From<&BrainState> for NeuralStatus {
    fn from(brain: &BrainState) -> Self {
        NeuralStatus {
            regions: vec![
                RegionStatus {
                    id: "frontal".into(),
                    temp: brain.hardware.cpu_temp,
                    cpu: brain.hardware.cpu_usage,
                    spikes: brain.frontal.spike_rate,
                    chem: vec![
                        ChemLevel { key: "dopa".into(), val: brain.neurochem.dopamine * 100.0 },
                        ChemLevel { key: "sero".into(), val: brain.neurochem.serotonin * 100.0 },
                    ],
                    ..Default::default()
                },
                // ... outras regiões
            ],
            global: GlobalMetrics {
                cpu_avg: brain.global_cpu,
                freq_avg: brain.avg_frequency,
                total_synapses: brain.total_synapses,
                ..Default::default()
            },
        }
    }
}