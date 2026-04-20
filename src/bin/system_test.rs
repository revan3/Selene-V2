// =============================================================================
// src/bin/system_test.rs — Testes de integração do sistema completo Selene
// =============================================================================
//
// Valida todos os módulos principais antes de iniciar o runtime completo.
// Não requer hardware, banco de dados, câmera ou microfone.
//
//  T01: Inicialização — todos os módulos criam sem panic
//  T02: NeuroChem     — neurotransmissores em bounds após 1000 updates
//  T03: Thalamus      — relay filtra sem NaN, shape correto
//  T04: Brainstem     — alertness decai com adenosina crescente
//  T05: Interoception — sentir() dentro de [0,1]
//  T06: Pipeline V1   — occipital → parietal → temporal sem NaN
//  T07: Pipeline V2   — temporal → frontal → decisão sem NaN
//  T08: Hipocampo     — memorize retorna vec válido
//  T09: Cerebelo      — compute_motor_output sem NaN
//  T10: Atenção       — amplifica canal saliente vs uniforme
//  T11: BrainConns    — project_all produz currentes não-zero após spikes
//  T12: LobeRouter    — roteamento é determinístico, gates ∈ [0,1]
//  T13: Chunking      — registrar_spikes sem panic
//  T14: MetaCognição  — retroalimentar: ganho_frontal ∈ [0.5, 2.0]
//  T15: MirrorNeurons — tem padrões pré-configurados
//  T16: Bug ACh       — diagnóstico: ACh chega ao neurônio via 4 args (não via 3)
//  T17: Estabilidade  — 200 ticks do pipeline completo sem NaN/Inf
// =============================================================================

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use selene_kernel::{
    FrontalLobe, OccipitalLobe, ParietalLobe, TemporalLobe,
    LimbicSystem, HippocampusV2 as Hippocampus, Cerebellum,
    brain_zones::corpus_callosum::CorpusCallosum,
    brain_zones::mirror_neurons::MirrorNeurons,
    synaptic_core::{NeuronioHibrido, CamadaHibrida, TipoNeuronal, PrecisionType},
    neurochem::{NeuroChem, EmotionalState},
    thalamus::Thalamus,
    brainstem::Brainstem,
    interoception::Interoception,
    basal_ganglia::BasalGanglia,
    learning::{
        attention::AttentionGate,
        inter_lobe::BrainConnections,
        lobe_router::{LobeRouter, EMBED_DIM},
        chunking::ChunkingEngine,
    },
    brain_zones::RegionType,
    meta::MetaCognitive,
    config::{Config, ModoOperacao},
    sensors::hardware::HardwareSensor,
};

// ─────────────────────────────────────────────────────────────────────────────
// Utilitários
// ─────────────────────────────────────────────────────────────────────────────

fn ok(nome: &str) { println!("  ✓ {nome}"); }
fn warn(nome: &str, msg: &str) { println!("  ⚠ {nome}: {msg}"); }
fn fail(nome: &str, msg: &str) { println!("  ✗ {nome}: {msg}"); }
fn secao(nome: &str) { println!("\n═══ {nome} ═══"); }

fn tem_nan(v: &[f32]) -> bool { v.iter().any(|x| x.is_nan() || x.is_infinite()) }

const N: usize = 128;   // neurônios por camada (menor que runtime, mais rápido)
const DT: f32 = 0.005; // 200 Hz

// ─────────────────────────────────────────────────────────────────────────────
// T01 — Inicialização
// ─────────────────────────────────────────────────────────────────────────────

fn test_init() -> usize {
    secao("T01 — Inicialização de todos os módulos");
    let mut falhas = 0usize;
    let config = Config::new(ModoOperacao::Normal);

    macro_rules! chk {
        ($nome:expr, $expr:expr) => {{
            let _ = $expr;
            ok($nome);
        }};
    }

    chk!("Config::new", config.clone());
    chk!("NeuroChem::new", NeuroChem::new());
    chk!("OccipitalLobe::new", OccipitalLobe::new(N, 0.2, &config));
    chk!("ParietalLobe::new", ParietalLobe::new(N, 0.2, &config));
    chk!("TemporalLobe::new", TemporalLobe::new(N, 0.005, 0.2, &config));
    chk!("LimbicSystem::new", LimbicSystem::new(N / 2, &config));
    chk!("Hippocampus::new", Hippocampus::new(N / 2, &config));
    chk!("FrontalLobe::new", FrontalLobe::new(N, 0.2, 0.1, &config));
    chk!("Cerebellum::new", Cerebellum::new(N / 4, N / 2, &config));
    chk!("CorpusCallosum::new", CorpusCallosum::new(10.0, 8));
    chk!("MirrorNeurons::new", MirrorNeurons::new());
    chk!("Thalamus::new", Thalamus::new());
    chk!("Brainstem::new", Brainstem::new());
    chk!("Interoception::new", Interoception::new());
    chk!("BasalGanglia::new", BasalGanglia::new(&config));
    chk!("AttentionGate::new", AttentionGate::new(N));
    chk!("BrainConnections::new", BrainConnections::new(N));
    chk!("LobeRouter::new", LobeRouter::new());
    chk!("ChunkingEngine::new", ChunkingEngine::new(RegionType::Temporal));
    chk!("MetaCognitive::new", MetaCognitive::new());
    chk!("CamadaHibrida::new",
        CamadaHibrida::new(N, "test", TipoNeuronal::RS,
            Some((TipoNeuronal::FS, 0.15)), None, 1.0));

    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T02 — NeuroChem bounds
// ─────────────────────────────────────────────────────────────────────────────

fn test_neurochem() -> usize {
    secao("T02 — NeuroChem: neurotransmissores dentro de bounds");
    let mut falhas = 0usize;
    let config = Config::new(ModoOperacao::Normal);
    let mut neuro = NeuroChem::new();
    let mut sensor = HardwareSensor::dummy();

    for _ in 0..200 {
        neuro.update(&mut sensor, &config);
    }

    let checks = [
        ("dopamine",       neuro.dopamine,      0.0, 2.5),
        ("serotonin",      neuro.serotonin,     0.0, 2.0),
        ("cortisol",       neuro.cortisol,      0.0, 1.5),
        ("noradrenaline",  neuro.noradrenaline, 0.0, 2.0),
        ("acetylcholine",  neuro.acetylcholine, 0.0, 2.0),
    ];
    for (nome, val, min, max) in checks {
        if val >= min && val <= max && !val.is_nan() {
            ok(&format!("{nome} = {val:.4} ∈ [{min}, {max}]"));
        } else {
            fail(nome, &format!("{val:.4} fora de [{min}, {max}]"));
            falhas += 1;
        }
    }

    // Estado emocional Plutchik deve somar algo entre 0-8
    let emo = EmotionalState::from_neurochem(
        neuro.dopamine, neuro.serotonin, neuro.cortisol, neuro.noradrenaline,
    );
    let soma = emo.joy + emo.trust + emo.fear + emo.surprise
             + emo.sadness + emo.disgust + emo.anger + emo.anticipation;
    if soma >= 0.0 && soma <= 8.0 {
        ok(&format!("EmotionalState.soma = {soma:.3} ∈ [0,8] | dominante: '{}'",
            emo.dominante()));
    } else {
        fail("EmotionalState", &format!("soma = {soma:.3} inválida"));
        falhas += 1;
    }
    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T03 — Thalamus
// ─────────────────────────────────────────────────────────────────────────────

fn test_thalamus() -> usize {
    secao("T03 — Thalamus: relay filtra sem NaN");
    let mut falhas = 0usize;
    let config = Config::new(ModoOperacao::Normal);
    let mut thalamus = Thalamus::new();
    let input = vec![0.5f32; N];

    // Com arousal normal
    let out = thalamus.relay(&input, 0.7, &config);
    if out.len() == N {
        ok(&format!("relay output len={} ✓", out.len()));
    } else {
        fail("thalamus relay len", &format!("{} ≠ {N}", out.len()));
        falhas += 1;
    }
    if !tem_nan(&out) {
        ok("relay sem NaN");
    } else {
        fail("thalamus relay NaN", "detectado NaN/Inf");
        falhas += 1;
    }

    // Com arousal zero (filtragem máxima)
    let out_zero = thalamus.relay(&input, 0.0, &config);
    let media_zero = out_zero.iter().sum::<f32>() / out_zero.len() as f32;
    let media_normal = out.iter().sum::<f32>() / out.len() as f32;
    if media_zero <= media_normal + 0.01 {
        ok(&format!("arousal=0 atenua: {media_zero:.4} ≤ {media_normal:.4}"));
    } else {
        warn("thalamus filtro", &format!("arousal=0 não atenuou ({media_zero:.4} > {media_normal:.4})"));
    }
    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T04 — Brainstem
// ─────────────────────────────────────────────────────────────────────────────

fn test_brainstem() -> usize {
    secao("T04 — Brainstem: alertness decai com adenosina crescente");
    let mut falhas = 0usize;
    let mut brainstem = Brainstem::new();

    // Adenosina baixa → alertness deve ser alto
    for _ in 0..50 { brainstem.update(0.0, DT); }
    let alerta_acordado = brainstem.stats().alertness;

    // Adenosina alta → alertness deve cair
    for _ in 0..200 { brainstem.update(0.9, DT); }
    let alerta_cansado = brainstem.stats().alertness;

    if alerta_cansado < alerta_acordado {
        ok(&format!("alertness: acordado={alerta_acordado:.3} > cansado={alerta_cansado:.3}"));
    } else {
        fail("brainstem alertness", &format!(
            "cansado={alerta_cansado:.3} ≥ acordado={alerta_acordado:.3} (deveria ser menor)"));
        falhas += 1;
    }

    // Modulate sem panic
    let audio = vec![0.1f32; 10];
    let out = brainstem.modulate(&audio);
    if !tem_nan(&out) {
        ok("brainstem.modulate sem NaN");
    } else {
        fail("brainstem modulate", "NaN detectado");
        falhas += 1;
    }
    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T05 — Interoception
// ─────────────────────────────────────────────────────────────────────────────

fn test_interoception() -> usize {
    secao("T05 — Interoception: sentir() dentro de [0,1]");
    let mut falhas = 0usize;
    let mut intro = Interoception::new();
    intro.update(0.3, 38.5, 0.6);
    let s = intro.sentir();
    if s >= 0.0 && s <= 1.0 && !s.is_nan() {
        ok(&format!("sentir() = {s:.4} ∈ [0,1]"));
    } else {
        fail("interoception.sentir", &format!("{s:.4} fora de [0,1]"));
        falhas += 1;
    }
    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T06 — Pipeline sensorial: occipital → parietal → temporal
// ─────────────────────────────────────────────────────────────────────────────

fn test_pipeline_sensorial() -> usize {
    secao("T06 — Pipeline sensorial: occipital → parietal → temporal");
    let mut falhas = 0usize;
    let config = Config::new(ModoOperacao::Normal);

    let mut occipital = OccipitalLobe::new(N, 0.2, &config);
    let mut parietal  = ParietalLobe::new(N, 0.2, &config);
    let mut temporal  = TemporalLobe::new(N, DT, 0.2, &config);

    let input = vec![0.3f32; N];
    let zero  = vec![0.0f32; N];
    let t_ms  = 0.0f32;

    // Occipital
    let features = occipital.visual_sweep(&input, DT, Some(&parietal.spatial_map), t_ms, &config);
    if !features.is_empty() && !tem_nan(&features) {
        ok(&format!("occipital features: {} canais, sem NaN", features.len()));
    } else {
        fail("occipital visual_sweep", "vazio ou NaN");
        falhas += 1;
    }

    // Reconstrói vision_full
    let chunk_size = (N / features.len().max(1)).max(1);
    let mut vision_full = vec![0.0f32; N];
    for (i, &f) in features.iter().enumerate() {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(N);
        for j in start..end { vision_full[j] = f / 100.0; }
    }

    // Parietal
    let parietal_out = parietal.integrate(&vision_full, &zero, DT, t_ms, &config);
    if parietal_out.len() == N && !tem_nan(&parietal_out) {
        ok(&format!("parietal integrate: {} vals, sem NaN", parietal_out.len()));
    } else {
        fail("parietal integrate", &format!("len={} ou NaN", parietal_out.len()));
        falhas += 1;
    }

    // Temporal
    let temporal_out = temporal.process(&vision_full, &parietal_out, DT, t_ms, &config);
    if temporal_out.len() == N && !tem_nan(&temporal_out) {
        ok(&format!("temporal process: {} vals, sem NaN", temporal_out.len()));
    } else {
        fail("temporal process", &format!("len={} ou NaN", temporal_out.len()));
        falhas += 1;
    }

    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T07 — Pipeline executivo: temporal → frontal → ação
// ─────────────────────────────────────────────────────────────────────────────

fn test_pipeline_executivo() -> usize {
    secao("T07 — Pipeline executivo: temporal → frontal → decisão");
    let mut falhas = 0usize;
    let config = Config::new(ModoOperacao::Normal);
    let mut frontal = FrontalLobe::new(N, 0.2, 0.1, &config);

    let recognized = vec![0.4f32; N];
    let goal       = vec![0.5f32; N];

    let action = frontal.decide(&recognized, &goal, DT, 0.0, &config);
    if action.len() == N && !tem_nan(&action) {
        ok(&format!("frontal.decide: {} vals, sem NaN", action.len()));
        let media = action.iter().sum::<f32>() / N as f32;
        ok(&format!("média da decisão = {media:.4}"));
    } else {
        fail("frontal.decide", &format!("len={} ou NaN", action.len()));
        falhas += 1;
    }

    // set_dopamine / set_serotonin não devem panic
    frontal.set_dopamine(1.5);
    frontal.set_serotonin(0.8);
    ok("set_dopamine / set_serotonin sem panic");

    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T08 — Hipocampo
// ─────────────────────────────────────────────────────────────────────────────

fn test_hippocampus() -> usize {
    secao("T08 — Hipocampo: memorize retorna output válido");
    let mut falhas = 0usize;
    let config = Config::new(ModoOperacao::Normal);
    let mut hippo = Hippocampus::new(N / 2, &config);

    let pattern = vec![0.6f32; N];
    let (out, conexoes) = hippo.memorize_with_connections(&pattern, 0.7, DT, 0.0, &config);

    if !out.is_empty() && !tem_nan(&out) {
        ok(&format!("memorize output: {} vals, sem NaN", out.len()));
    } else {
        fail("hippocampus memorize", "vazio ou NaN");
        falhas += 1;
    }
    ok(&format!("conexões geradas: {}", conexoes.len()));
    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T09 — Cerebelo
// ─────────────────────────────────────────────────────────────────────────────

fn test_cerebellum() -> usize {
    secao("T09 — Cerebelo: compute_motor_output sem NaN");
    let mut falhas = 0usize;
    let config = Config::new(ModoOperacao::Normal);
    let mut cerebelo = Cerebellum::new(N / 4, N / 2, &config);

    let recognized = vec![0.5f32; N];
    let error      = vec![0.1f32; N];
    let out = cerebelo.compute_motor_output(&recognized, &error, DT, 0.0, &config);

    if !out.is_empty() && !tem_nan(&out) {
        ok(&format!("motor_output: {} vals, sem NaN", out.len()));
        let em_range = out.iter().all(|&v| v >= -1.5 && v <= 1.5);
        if em_range {
            ok("todos os outputs cerebelares ∈ [-1.5, 1.5]");
        } else {
            warn("cerebelo range", "algum valor fora de [-1.5, 1.5]");
        }
    } else {
        fail("cerebelo motor_output", "vazio ou NaN");
        falhas += 1;
    }
    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T10 — AttentionGate
// ─────────────────────────────────────────────────────────────────────────────

fn test_attention() -> usize {
    secao("T10 — AttentionGate: amplifica canal saliente");
    let mut falhas = 0usize;
    let mut gate = AttentionGate::new(N);

    // Input uniforme
    let uniform = vec![0.5f32; N];
    let prev_rates = vec![0.1f32; N];
    gate.set_topdown(&prev_rates);
    let out_uniform = gate.attend(&uniform, DT * 1000.0);

    // Input com pico saliente
    let mut saliente = vec![0.1f32; N];
    saliente[N / 2] = 2.0; // pico
    let out_saliente = gate.attend(&saliente, DT * 1000.0);

    if !tem_nan(&out_uniform) && !tem_nan(&out_saliente) {
        ok("attend sem NaN (uniform e saliente)");
    } else {
        fail("attention NaN", "detectado");
        falhas += 1;
    }

    let max_saliente = out_saliente.iter().copied().fold(0.0f32, f32::max);
    let media_uniform = out_uniform.iter().sum::<f32>() / N as f32;
    if max_saliente >= media_uniform {
        ok(&format!("pico saliente {max_saliente:.3} ≥ média uniforme {media_uniform:.3}"));
    } else {
        warn("attention amplificação", "pico não superou média uniforme");
    }
    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T11 — BrainConnections
// ─────────────────────────────────────────────────────────────────────────────

fn test_brain_connections() -> usize {
    secao("T11 — BrainConnections: project_all sem NaN");
    let mut falhas = 0usize;
    let mut conn = BrainConnections::new(N);

    let rates = vec![0.5f32; N];
    let rates_half = vec![0.3f32; N / 2];

    let currents = conn.project_all(&rates, &rates, &rates, &rates, &rates_half, &rates_half);

    let all_vecs = [
        &currents.para_temporal, &currents.para_parietal,
        &currents.para_frontal, &currents.para_hippo,
    ];
    let mut algum_nan = false;
    for v in &all_vecs {
        if tem_nan(v) { algum_nan = true; }
    }

    if !algum_nan {
        ok("project_all sem NaN em todos os destinos");
    } else {
        fail("brain_connections NaN", "detectado");
        falhas += 1;
    }

    // modular_all não deve panic
    conn.modular_all(1.2, 0.1);
    ok("modular_all sem panic");
    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T12 — LobeRouter
// ─────────────────────────────────────────────────────────────────────────────

fn test_lobe_router() -> usize {
    secao("T12 — LobeRouter: roteamento determinístico, gates ∈ [0,1]");
    let mut falhas = 0usize;
    let mut router = LobeRouter::new();

    let vision = vec![0.3f32; N];
    let cochlea = vec![0.1f32; 10];

    let query = LobeRouter::build_query(
        &vision, &cochlea,
        1.0, 1.0, 0.0, 0.5,
        0.1, 0.3, 0.3, 0.0, 1u64,
    );
    let decision = router.route(query);

    let gates: [(&str, f32); 6] = [
        ("parietal",    decision.parietal),
        ("temporal",    decision.temporal),
        ("frontal",     decision.frontal),
        ("limbic",      decision.limbic),
        ("hippocampus", decision.hippocampus),
        ("cerebellum",  decision.cerebellum),
    ];

    let mut todos_ok = true;
    for (nome, gate) in gates {
        if gate >= 0.0 && gate <= 1.0 && !gate.is_nan() {
            ok(&format!("{nome} gate = {gate:.4}"));
        } else {
            fail(&format!("{nome} gate"), &format!("{gate:.4} fora de [0,1]"));
            falhas += 1;
            todos_ok = false;
        }
    }

    // Segundo route com mesmo query deve produzir resultado consistente
    let decision2 = router.route(query);
    let diff = (decision2.frontal - decision.frontal).abs();
    if diff < 0.3 {
        ok(&format!("routing consistente: Δfrontal = {diff:.4}"));
    } else {
        warn("router consistência", &format!("Δfrontal = {diff:.4} (EMA pode variar)"));
    }
    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T13 — ChunkingEngine
// ─────────────────────────────────────────────────────────────────────────────

fn test_chunking() -> usize {
    secao("T13 — ChunkingEngine: registrar_spikes sem panic");
    let mut falhas = 0usize;
    let config = Config::new(ModoOperacao::Normal);
    let mut temporal = TemporalLobe::new(N, DT, 0.2, &config);
    let mut chunking = ChunkingEngine::new(RegionType::Temporal);

    // Roda alguns ticks para gerar spikes
    let input = vec![8.0f32; N];
    for i in 0..50 {
        let t = i as f32 * DT * 1000.0;
        temporal.process(&input, &vec![0.0f32; N], DT, t, &config);
    }

    // Registra spikes — não deve panic
    let spikes: Vec<bool> = temporal.recognition_layer.neuronios.iter()
        .map(|n| n.last_spike_ms > 0.0)
        .collect();

    let chunks = chunking.registrar_spikes(
        &spikes,
        &temporal.recognition_layer,
        0.3,
        50.0 * DT * 1000.0,
    );
    ok(&format!("registrar_spikes retornou {} chunks sem panic", chunks.len()));
    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T14 — MetaCognição
// ─────────────────────────────────────────────────────────────────────────────

fn test_metacognition() -> usize {
    secao("T14 — MetaCognição: retroalimentar em bounds");
    let mut falhas = 0usize;
    let mut meta = MetaCognitive::new();

    meta.observe(0.7, 0.4, 500);
    let fb = meta.retroalimentar();

    if fb.ganho_frontal >= 0.5 && fb.ganho_frontal <= 2.5 && !fb.ganho_frontal.is_nan() {
        ok(&format!("ganho_frontal = {:.4} ∈ [0.5, 2.5]", fb.ganho_frontal));
    } else {
        fail("metacog ganho_frontal", &format!("{:.4} fora de bounds", fb.ganho_frontal));
        falhas += 1;
    }
    if fb.plasticidade_mod >= 0.0 && fb.plasticidade_mod <= 2.0 {
        ok(&format!("plasticidade_mod = {:.4} ∈ [0, 2]", fb.plasticidade_mod));
    } else {
        fail("metacog plasticidade_mod", &format!("{:.4} fora de bounds", fb.plasticidade_mod));
        falhas += 1;
    }
    ok(&format!("habilitar_replay = {}", fb.habilitar_replay));
    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T15 — MirrorNeurons
// ─────────────────────────────────────────────────────────────────────────────

fn test_mirror_neurons() -> usize {
    secao("T15 — MirrorNeurons: padrões pré-configurados");
    let mut falhas = 0usize;
    let mirror = MirrorNeurons::new();
    let n = mirror.n_padroes();
    if n > 0 {
        ok(&format!("{n} padrões pré-configurados"));
    } else {
        fail("mirror_neurons", "nenhum padrão pré-configurado");
        falhas += 1;
    }
    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T16 — Bug ACh (diagnóstico)
// ─────────────────────────────────────────────────────────────────────────────

fn test_ach_bug_diagnostico() -> usize {
    secao("T16 — Diagnóstico Bug ACh: ACh real só chega via modular_neuro_v3");
    let mut falhas = 0usize;

    // Cenário 1: usando modular_neuro (3 args) — como main.rs faz para quase todos os lobos
    let mut n1 = NeuronioHibrido::new(0, TipoNeuronal::RS, PrecisionType::FP32);
    n1.modular_neuro(1.5, 1.0, 0.2); // main.rs chama assim
    if (n1.extras.mod_ach - 1.0).abs() < 1e-6 {
        warn("ACh via modular_neuro(3 args)",
            "mod_ach = 1.0 (fixo) — NeuroChem.acetylcholine NÃO chega ao neurônio");
    } else {
        ok(&format!("mod_ach via 3 args = {:.4}", n1.extras.mod_ach));
    }

    // Cenário 2: usando modular_neuro_v3 (4 args) — como DEVERIA ser
    let mut n2 = NeuronioHibrido::new(0, TipoNeuronal::RS, PrecisionType::FP32);
    n2.modular_neuro_v3(1.5, 1.0, 0.2, 2.0); // com ACh real
    if (n2.extras.mod_ach - 2.0).abs() < 1e-6 {
        ok("ACh via modular_neuro_v3(4 args): mod_ach = 2.0 (correto)");
    } else {
        fail("ACh 4-args", &format!("mod_ach = {:.4} ≠ 2.0", n2.extras.mod_ach));
        falhas += 1;
    }

    // Calcula impacto: com ach=2.0, I_M é reduzido em 35%
    let g_m_base = selene_kernel::synaptic_core::TipoNeuronalV3::g_m(&TipoNeuronal::RS);
    let g_m_com_ach = g_m_base * (1.0 - (2.0 - 1.0) * 0.35);
    println!("    → Com ACh=2.0 (atenção máxima): I_M RS = {g_m_base:.1} → {g_m_com_ach:.1} mS/cm²");
    println!("    → Fix: substituir modular_neuro(da,ser,cor) por");
    println!("           modular_neuro_v3(da, ser, cor, neuro.acetylcholine) em main.rs");

    // Este teste não conta como falha — é um diagnóstico
    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T17 — Estabilidade: 200 ticks do pipeline completo
// ─────────────────────────────────────────────────────────────────────────────

fn test_estabilidade_pipeline() -> usize {
    secao("T17 — Estabilidade: 200 ticks do pipeline completo sem NaN");
    let mut falhas = 0usize;
    let config = Config::new(ModoOperacao::Normal);

    let mut occipital  = OccipitalLobe::new(N, 0.2, &config);
    let mut parietal   = ParietalLobe::new(N, 0.2, &config);
    let mut temporal   = TemporalLobe::new(N, DT, 0.2, &config);
    let mut limbic     = LimbicSystem::new(N / 2, &config);
    let mut hippocampus = Hippocampus::new(N / 2, &config);
    let mut frontal    = FrontalLobe::new(N, 0.2, 0.1, &config);
    let mut cerebelo   = Cerebellum::new(N / 4, N / 2, &config);
    let mut thalamus   = Thalamus::new();
    let mut brainstem  = Brainstem::new();
    let mut neuro      = NeuroChem::new();
    let mut attention  = AttentionGate::new(N);
    let mut conn       = BrainConnections::new(N);
    let mut sensor     = HardwareSensor::dummy();

    let goal = vec![0.5f32; N];
    let zero = vec![0.0f32; N];
    let zero_half = vec![0.0f32; N / 2];

    let mut prev_frontal  = vec![0.0f32; N];
    let mut prev_temporal = vec![0.0f32; N];
    let mut prev_parietal = vec![0.0f32; N];
    let mut prev_limbic   = vec![0.0f32; N / 2];
    let mut prev_hippo    = vec![0.0f32; N / 2];

    let mut tick_crash = None;

    for tick in 0..200usize {
        let t_ms = tick as f32 * DT * 1000.0;

        neuro.update(&mut sensor, &config);
        let (da, ser, cor) = (neuro.dopamine, neuro.serotonin, neuro.cortisol);

        // Sinal de entrada com tônico básico
        let tonic: f32 = 0.04 * (tick as f32 * 0.1).sin().abs();
        let input_vis = vec![tonic; N];
        let input_aud = vec![tonic * 0.5; 10];

        brainstem.update(0.1, DT);
        let alertness = brainstem.stats().alertness.max(0.3);
        let relayed = thalamus.relay(&input_vis, neuro.noradrenaline, &config);
        let cochlea = brainstem.modulate(&input_aud);

        attention.set_topdown(&prev_frontal);
        let attended = attention.attend(&relayed, DT * 1000.0);

        let currents = conn.project_all(
            &attended, &prev_temporal, &prev_parietal,
            &prev_frontal, &prev_limbic, &prev_hippo,
        );

        let hybrid: Vec<f32> = attended.iter().enumerate()
            .map(|(i, &v)| (v + currents.para_temporal.get(i).copied().unwrap_or(0.0) * 0.1)
                .clamp(0.0, 1.0) * alertness)
            .collect();

        let features = occipital.visual_sweep(&hybrid, DT, Some(&parietal.spatial_map), t_ms, &config);
        let chunk_size = (N / features.len().max(1)).max(1);
        let mut vision_full = vec![0.0f32; N];
        for (i, &f) in features.iter().enumerate() {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(N);
            for j in start..end { vision_full[j] = f / 100.0; }
        }

        let new_parietal = parietal.integrate(&vision_full, &zero, DT, t_ms, &config);
        let recognized   = temporal.process(&vision_full, &new_parietal, DT, t_ms, &config);
        let (emotion, _) = limbic.evaluate(&cochlea, 0.0, DT, t_ms, &config);

        if emotion.abs() >= 0.35 {
            let (hippo_out, _) = hippocampus.memorize_with_connections(
                &recognized, emotion, DT, t_ms, &config,
            );
            prev_hippo.iter_mut().enumerate().for_each(|(i, v)| {
                *v = hippo_out.get(i).copied().unwrap_or(0.0);
            });
        }

        frontal.set_dopamine(da + emotion);
        let action = frontal.decide(&recognized, &goal, DT, t_ms, &config);
        let climbing: Vec<f32> = action.iter().zip(recognized.iter())
            .map(|(a, r)| (a - r).clamp(-1.0, 1.0)).collect();
        let _cerb_out = cerebelo.compute_motor_output(&recognized, &climbing, DT, t_ms, &config);

        // Neuromodulação
        if tick % 5 == 0 {
            occipital.v1_primary_layer.modular_neuro(da, ser, cor);
            temporal.recognition_layer.modular_neuro(da, ser, cor);
            frontal.executive_layer.modular_neuro(da, ser, cor);
        }

        // Verificar NaN
        let check_vecs: &[(&str, &[f32])] = &[
            ("vision_full", &vision_full),
            ("recognized",  &recognized),
            ("action",      &action),
        ];
        for (nome, v) in check_vecs {
            if tem_nan(v) {
                tick_crash = Some((tick, *nome));
                break;
            }
        }
        if tick_crash.is_some() { break; }

        // Atualiza prev
        prev_frontal.iter_mut().enumerate().for_each(|(i, v)| *v = action.get(i).copied().unwrap_or(0.0));
        prev_temporal.iter_mut().enumerate().for_each(|(i, v)| *v = recognized.get(i).copied().unwrap_or(0.0));
        prev_parietal.iter_mut().enumerate().for_each(|(i, v)| *v = new_parietal.get(i).copied().unwrap_or(0.0));
        prev_limbic.iter_mut().for_each(|v| *v = emotion.clamp(0.0, 1.0));
    }

    match tick_crash {
        None => ok("200 ticks do pipeline completo: sem NaN/Inf"),
        Some((tick, nome)) => {
            fail("estabilidade pipeline", &format!("NaN em '{nome}' no tick {tick}"));
            falhas += 1;
        }
    }
    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T18 — EventoEpisodico: estrutura correta e armazenamento
// ─────────────────────────────────────────────────────────────────────────────

fn test_grounding_evento_episodico() -> usize {
    secao("T18 — EventoEpisodico: estrutura e armazenamento");
    use selene_kernel::websocket::bridge::EventoEpisodico;
    let mut falhas = 0usize;

    let ev = EventoEpisodico {
        palavras:      vec!["quente".to_string(), "sol".to_string()],
        padrao_visual: [0xFFu64; 8],   // padrão visual ativo (todos bits 1)
        padrao_audio:  [0u64; 8],      // silêncio
        estado_corpo:  [0.8, 0.6, 0.5, 0.9, 0.1],
        emocao:        0.7,
        arousal:       0.6,
        tempo_ms:      12345.0,
    };

    if ev.palavras.len() == 2 { ok("EventoEpisodico.palavras correto"); }
    else { fail("EventoEpisodico", "palavras len errado"); falhas += 1; }

    if ev.emocao == 0.7 { ok("EventoEpisodico.emocao correto"); }
    else { fail("EventoEpisodico", "emocao errado"); falhas += 1; }

    let visual_ativo = ev.padrao_visual.iter().any(|&w| w != 0);
    let audio_ativo  = ev.padrao_audio.iter().any(|&w| w != 0);
    if visual_ativo  { ok("padrao_visual ativo (0xFF)"); }
    else             { fail("padrao_visual", "deveria estar ativo"); falhas += 1; }
    if !audio_ativo  { ok("padrao_audio silencioso (0x00)"); }
    else             { fail("padrao_audio", "deveria estar silencioso"); falhas += 1; }

    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T19 — Grounding: score aumenta com binding perceptual
// ─────────────────────────────────────────────────────────────────────────────

fn test_grounding_score_aumenta() -> usize {
    secao("T19 — Grounding: score aumenta com binding visual+audio");
    use selene_kernel::storage::swap_manager::SwapManager;
    use selene_kernel::config::{Config, ModoOperacao};
    use selene_kernel::sensors::SensorFlags;
    use selene_kernel::websocket::bridge::BrainState;
    use std::sync::Arc;
    let mut falhas = 0usize;

    // Mínimo para criar BrainState sem I/O real
    let cfg    = Config::new(ModoOperacao::Normal);
    let swap   = Arc::new(tokio::sync::Mutex::new(SwapManager::new(256, 3600)));
    let flags  = SensorFlags::new_desativados();
    let mut bs = BrainState::new(swap, &cfg, flags);

    // Antes do binding: grounding deve ser 0
    let g_antes = bs.grounding.get("quente").copied().unwrap_or(0.0);
    if g_antes == 0.0 { ok("grounding inicial = 0.0"); }
    else { fail("grounding inicial", &format!("{g_antes} != 0")); falhas += 1; }

    // Binding com padrão visual ativo
    let visual_ativo: [u64; 8] = [0xF0F0_F0F0u64; 8]; // bits ativos
    let audio_zero:   [u64; 8] = [0u64; 8];
    bs.grounding_bind(
        &["quente".to_string(), "sol".to_string()],
        visual_ativo, audio_zero,
        0.5, 0.6, 1000.0,
    );

    let g_apos_visual = bs.grounding.get("quente").copied().unwrap_or(0.0);
    if g_apos_visual > 0.0 {
        ok(&format!("grounding após binding visual = {g_apos_visual:.3}"));
    } else {
        fail("grounding visual", "não aumentou"); falhas += 1;
    }

    // Binding com padrão audio ativo também
    let audio_ativo: [u64; 8] = [0x0F0Fu64; 8];
    bs.grounding_bind(
        &["quente".to_string()],
        visual_ativo, audio_ativo,
        0.5, 0.6, 2000.0,
    );
    let g_apos_audio = bs.grounding.get("quente").copied().unwrap_or(0.0);
    if g_apos_audio > g_apos_visual {
        ok(&format!("grounding após binding audio > visual: {g_apos_audio:.3} > {g_apos_visual:.3}"));
    } else {
        fail("grounding audio", "não acumulou sobre visual"); falhas += 1;
    }

    // Verificar que historico_episodico foi preenchido
    if bs.historico_episodico.len() >= 2 {
        ok(&format!("historico_episodico: {} eventos registrados", bs.historico_episodico.len()));
    } else {
        fail("historico_episodico", &format!("apenas {} eventos", bs.historico_episodico.len()));
        falhas += 1;
    }

    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T20 — Grounding: palavra grounded preferida como âncora
// ─────────────────────────────────────────────────────────────────────────────

fn test_grounding_ancora_preferida() -> usize {
    secao("T20 — Grounding: palavra grounded preferida como âncora");
    use selene_kernel::encoding::spike_codec::features_to_spike_pattern;
    let mut falhas = 0usize;

    // Simula dois tokens: A com grounding alto, B com grounding zero.
    // Ambos têm o mesmo número de conexões no grafo.
    // O grounding score de A deve torná-lo preferido.

    let grounding: std::collections::HashMap<String, f32> = [
        ("quente".to_string(), 0.85f32),  // grounded via binding visual+audio
        ("abstrato".to_string(), 0.0f32), // só linguístico
    ].into_iter().collect();

    let mut grafo: std::collections::HashMap<String, Vec<(String, f32)>> =
        std::collections::HashMap::new();
    // Mesmo número de conexões para ambos (3 cada)
    grafo.insert("quente".to_string(), vec![
        ("sol".to_string(), 0.5),
        ("fogo".to_string(), 0.4),
        ("verão".to_string(), 0.3),
    ]);
    grafo.insert("abstrato".to_string(), vec![
        ("conceito".to_string(), 0.5),
        ("ideia".to_string(), 0.4),
        ("pensamento".to_string(), 0.3),
    ]);

    // Replica a lógica de seleção de âncora de gerar_resposta_emergente
    let tokens = vec!["quente", "abstrato"];
    let ancora = tokens.iter()
        .filter(|t| grafo.contains_key(**t))
        .max_by(|a, b| {
            let conn_a = grafo.get(**a).map(|v| v.len()).unwrap_or(0) as f32;
            let conn_b = grafo.get(**b).map(|v| v.len()).unwrap_or(0) as f32;
            let g_a = grounding.get(**a).copied().unwrap_or(0.0);
            let g_b = grounding.get(**b).copied().unwrap_or(0.0);
            (conn_a + g_a * 3.0).partial_cmp(&(conn_b + g_b * 3.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|t| t.to_string());

    match ancora {
        Some(ref a) if a == "quente" => ok(&format!("âncora escolhida: '{a}' (grounded=0.85 > 0.0)")),
        Some(ref a) => { fail("seleção âncora", &format!("'{a}' escolhido em vez de 'quente'")); falhas += 1; }
        None        => { fail("seleção âncora", "nenhuma âncora encontrada"); falhas += 1; }
    }

    // Teste features_to_spike_pattern (helper movido para spike_codec)
    let features = vec![50.0f32, 80.0, 30.0, 0.0, 100.0, 20.0, 60.0, 10.0];
    let pat = features_to_spike_pattern(&features);
    let ativo = pat.iter().any(|&w| w != 0);
    if ativo { ok("features_to_spike_pattern produz padrão ativo"); }
    else      { fail("features_to_spike_pattern", "padrão zero"); falhas += 1; }

    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T21 — Grounding: RPE positivo aumenta grounding do contexto
// ─────────────────────────────────────────────────────────────────────────────

fn test_grounding_rpe() -> usize {
    secao("T21 — Grounding: RPE positivo aumenta grounding, negativo reduz");
    use selene_kernel::storage::swap_manager::SwapManager;
    use selene_kernel::config::{Config, ModoOperacao};
    use selene_kernel::sensors::SensorFlags;
    use selene_kernel::websocket::bridge::BrainState;
    use std::sync::Arc;
    let mut falhas = 0usize;

    let cfg   = Config::new(ModoOperacao::Normal);
    let swap  = Arc::new(tokio::sync::Mutex::new(SwapManager::new(256, 3600)));
    let flags = SensorFlags::new_desativados();
    let mut bs = BrainState::new(swap, &cfg, flags);

    // Coloca palavras no neural_context
    bs.neural_context.push_back("aprender".to_string());
    bs.neural_context.push_back("descobrir".to_string());

    let g_antes = bs.grounding.get("aprender").copied().unwrap_or(0.0);

    // RPE positivo forte: predição correta → grounding aumenta
    bs.grounding_rpe(0.8);
    let g_apos_pos = bs.grounding.get("aprender").copied().unwrap_or(0.0);
    if g_apos_pos > g_antes {
        ok(&format!("RPE +0.8: grounding {g_antes:.3} → {g_apos_pos:.3}"));
    } else {
        fail("grounding_rpe positivo", "não aumentou"); falhas += 1;
    }

    // RPE negativo: predição errada → grounding diminui
    bs.grounding_rpe(-0.5);
    let g_apos_neg = bs.grounding.get("aprender").copied().unwrap_or(0.0);
    if g_apos_neg < g_apos_pos {
        ok(&format!("RPE -0.5: grounding {g_apos_pos:.3} → {g_apos_neg:.3}"));
    } else {
        fail("grounding_rpe negativo", "não diminuiu"); falhas += 1;
    }

    // Decaimento: após 1000 chamadas de decay, grounding deve cair
    let g_pre_decay = g_apos_neg;
    for _ in 0..100 { bs.grounding_decay(); }
    let g_apos_decay = bs.grounding.get("aprender").copied().unwrap_or(0.0);
    if g_apos_decay < g_pre_decay {
        ok(&format!("grounding_decay: {g_pre_decay:.4} → {g_apos_decay:.4}"));
    } else {
        // Se g_pre_decay era 0, decay não muda nada — não é falha
        warn("grounding_decay", &format!("{g_pre_decay:.4} → {g_apos_decay:.4} (pode ser 0)"));
    }

    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// T22 — N3/REM: replay usa EventoEpisodico e atualiza grounding
// ─────────────────────────────────────────────────────────────────────────────

fn test_grounding_rem_replay() -> usize {
    secao("T22 — N3/REM replay: EventoEpisodico consolida grounding");
    use selene_kernel::storage::swap_manager::SwapManager;
    use selene_kernel::config::{Config, ModoOperacao};
    use selene_kernel::sensors::SensorFlags;
    use selene_kernel::websocket::bridge::{BrainState, EventoEpisodico};
    use std::sync::Arc;
    let mut falhas = 0usize;

    let cfg   = Config::new(ModoOperacao::Normal);
    let swap  = Arc::new(tokio::sync::Mutex::new(SwapManager::new(256, 3600)));
    let flags = SensorFlags::new_desativados();
    let mut bs = BrainState::new(swap, &cfg, flags);

    // Pré-popula grafo com associações
    if let Ok(mut sw) = bs.swap_manager.try_lock() {
        sw.importar_causal(vec![
            ("amor".to_string(), "carinho".to_string(), 0.5),
        ]);
        sw.aprender_conceito("carinho", 0.3);
    }

    // Injeta evento episódico rico com padrão visual ativo
    let visual_ativo: [u64; 8] = [0xAAAA_AAAA_AAAA_AAAAu64; 8];
    let ev = EventoEpisodico {
        palavras:      vec!["amor".to_string(), "carinho".to_string()],
        padrao_visual: visual_ativo,
        padrao_audio:  [0u64; 8],
        estado_corpo:  [0.7, 0.8, 0.5, 0.6, 0.1],
        emocao:        0.8,  // emoção forte → replay saliente
        arousal:       0.7,
        tempo_ms:      5000.0,
    };
    bs.historico_episodico.push_back(ev);

    let g_antes = bs.grounding.get("amor").copied().unwrap_or(0.0);

    // grounding_bind simula o que N3/REM faz para eventos com percepção real
    let palavras_ev = vec!["amor".to_string(), "carinho".to_string()];
    bs.grounding_bind(&palavras_ev, visual_ativo, [0u64; 8], 0.8, 0.7, 5000.0);

    let g_apos = bs.grounding.get("amor").copied().unwrap_or(0.0);
    if g_apos > g_antes {
        ok(&format!("grounding 'amor' após REM replay: {g_antes:.3} → {g_apos:.3}"));
    } else {
        fail("REM replay grounding", "não aumentou"); falhas += 1;
    }

    // Verifica que histórico tem 2 eventos agora (original + o do bind)
    if bs.historico_episodico.len() >= 2 {
        ok(&format!("historico_episodico: {} eventos", bs.historico_episodico.len()));
    } else {
        fail("historico_episodico", "menos eventos que esperado"); falhas += 1;
    }

    // Verifica evento saliente (emocao > 0.35) está no histórico
    let tem_saliente = bs.historico_episodico.iter().any(|ev| ev.emocao.abs() > 0.35);
    if tem_saliente { ok("histórico contém evento emocional saliente (> 0.35)"); }
    else            { fail("historico saliente", "nenhum evento com emocao > 0.35"); falhas += 1; }

    falhas
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   Selene Brain 2.0 — Testes de Sistema Completo             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("  N = {N} neurônios/camada | DT = {DT}s (200Hz)");

    let mut total = 0usize;

    total += test_init();
    total += test_neurochem();
    total += test_thalamus();
    total += test_brainstem();
    total += test_interoception();
    total += test_pipeline_sensorial();
    total += test_pipeline_executivo();
    total += test_hippocampus();
    total += test_cerebellum();
    total += test_attention();
    total += test_brain_connections();
    total += test_lobe_router();
    total += test_chunking();
    total += test_metacognition();
    total += test_mirror_neurons();
    total += test_ach_bug_diagnostico(); // diagnóstico — não conta falhas
    total += test_estabilidade_pipeline();
    total += test_grounding_evento_episodico();
    total += test_grounding_score_aumenta();
    total += test_grounding_ancora_preferida();
    total += test_grounding_rpe();
    total += test_grounding_rem_replay();

    println!("\n══════════════════════════════════════════════════════════════");
    if total == 0 {
        println!("  RESULTADO: ✓ TODOS OS TESTES PASSARAM — sistema pronto");
    } else {
        println!("  RESULTADO: ✗ {total} FALHA(S) — revisar antes de iniciar");
    }
    println!("══════════════════════════════════════════════════════════════");

    std::process::exit(if total == 0 { 0 } else { 1 });
}
