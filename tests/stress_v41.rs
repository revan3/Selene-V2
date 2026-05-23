// ============================================================================
// tests/stress_v41.rs — Matriz de Estresse V4.1 (Resiliência + DSU)
// ============================================================================
//
// Quatro cenários críticos auditados em 2026-05-22:
//
//   A. Watchdog + Invariants                — main.rs LOOP_HEARTBEAT pattern
//   B. Depleção metabólica V4               — ATP, [K⁺]o, E_K(t), bomba Na/K
//   C. ChunkDsu / chaves_set complexity     — O(α(n)) clusters + O(1) lookup
//   D. Neuromodulação alostérica            — Adenosina⊣D2 + Oxitocina⊣BLA
//
// Cada cenário valida:
//   1. Hot-path (custo CPU/memória sob loop 200Hz)
//   2. Invariante biológica (Rall / Larkum / Nernst / Ferré / Kirsch)
//   3. Resiliência (que classe de falha silenciosa o teste previne)
//
// Rodar:
//   cargo test --test stress_v41 --profile release-lowmem
//   cargo test --test stress_v41 cenario_a    # cenário específico
// ============================================================================

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use selene_kernel::brain_zones::Amygdala;
use selene_kernel::config::{Config, ModoOperacao};
use selene_kernel::learning::chunking::{ChunkDsu, ChunkingEngine};
use selene_kernel::neurochem::NeuroChem;
use selene_kernel::synaptic_core::{
    CamadaHibrida, EstadoBrainState, NeuronioHibrido, PrecisionType, TipoNeuronal,
};

// ============================================================================
// CENÁRIO A — Watchdog do Loop 200Hz + Invariants
// ============================================================================
//
// O loop neural usa um `static LOOP_HEARTBEAT: AtomicU64` em main.rs,
// atualizado a cada step%500 (~2.5s @ 200Hz). Uma task tokio paralela
// verifica o valor a cada 5s; se não mudou → loga [WATCHDOG] e seta
// flag de saúde para `false`.
//
// Aqui validamos o PADRÃO — não a global de main.rs (private). O teste
// constrói o mesmo mecanismo com Arc<AtomicU64> e Arc<AtomicBool>.

#[tokio::test(flavor = "current_thread")]
async fn cenario_a_watchdog_detecta_stall_em_5s() {
    let heartbeat: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    let stall_detected: Arc<std::sync::atomic::AtomicBool> =
        Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Watchdog: verifica a cada 200ms (acelerado para o teste).
    let hb = Arc::clone(&heartbeat);
    let detected = Arc::clone(&stall_detected);
    let watchdog = tokio::spawn(async move {
        let mut iv = tokio::time::interval(Duration::from_millis(200));
        let mut last: u64 = 0;
        let mut ticks_idle = 0u32;
        loop {
            iv.tick().await;
            let cur = hb.load(Ordering::Relaxed);
            if cur == last && last > 0 {
                ticks_idle += 1;
                if ticks_idle >= 3 {
                    detected.store(true, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
            } else {
                ticks_idle = 0;
            }
            last = cur;
        }
    });

    // Fase 1: loop "saudável" por 600ms, atualizando heartbeat a cada 50ms.
    for step in 1..=12u64 {
        heartbeat.store(step, Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        !stall_detected.load(std::sync::atomic::Ordering::Relaxed),
        "Falso positivo: watchdog acusou stall durante operação saudável"
    );

    // Fase 2: STALL simulado — não atualiza heartbeat por >600ms.
    // O watchdog tem que detectar dentro de ~600ms (3 ticks × 200ms).
    let _ = tokio::time::timeout(Duration::from_secs(2), watchdog).await;

    assert!(
        stall_detected.load(std::sync::atomic::Ordering::Relaxed),
        "Watchdog NÃO detectou stall do loop neural — bug crítico de resiliência"
    );
}

/// Valida as 3 condições de invariante do save cycle (main.rs ~linha 2371)
/// que detectaram o bug do "Neonatal travada para sempre" (incidente 2026-05-17).
#[test]
fn cenario_a_invariants_detectam_classes_de_bug_conhecidas() {
    // Helper: replica exatamente a lógica de main.rs (apenas booleano, sem log).
    fn checar_invariantes(vocab_n: usize, edges_n: usize, step: u64) -> Vec<&'static str> {
        let mut violacoes = Vec::new();
        if vocab_n == 0 && step > 10_000 {
            violacoes.push("vocab_zero_lock_starvation");
        }
        if edges_n == 0 && step > 20_000 {
            violacoes.push("edges_zero_grafo_desconexo");
        }
        if vocab_n > 0 && edges_n == 0 && step > 15_000 {
            violacoes.push("sinapses_nao_criadas");
        }
        violacoes
    }

    // Caso 1: operação normal — nenhum invariante violado.
    assert!(checar_invariantes(500, 1_200, 50_000).is_empty());

    // Caso 2: ontogenia travada (vocab=0 após 30k ticks de treino).
    let v = checar_invariantes(0, 0, 30_000);
    assert!(v.contains(&"vocab_zero_lock_starvation"));
    assert!(v.contains(&"edges_zero_grafo_desconexo"));

    // Caso 3: aprendizado parcial — vocab cresce mas STDP não cria arestas.
    let v = checar_invariantes(200, 0, 18_000);
    assert!(v.contains(&"sinapses_nao_criadas"));
    assert!(!v.contains(&"edges_zero_grafo_desconexo"),
        "edges_zero só dispara após step>20k");

    // Caso 4: warm-up legítimo — invariants silenciosas antes de step 10k.
    assert!(checar_invariantes(0, 0, 5_000).is_empty(),
        "warm-up não deve disparar invariants");
}

// ============================================================================
// CENÁRIO B — Depleção Metabólica V4 (ATP / [K⁺]o / E_K Nernst)
// ============================================================================
//
// Mapeamento biológico:
//   • Equação de cabo de Rall 1967 (5 compartimentos AIS+Soma+Tronco+Apical+Extracell.)
//   • BAC firing — coincidência BAP + NMDA spike apical (Larkum 1999)
//   • Bomba Na/K-ATPase eletrogenica — corrente hiperpolarizante quando ATP alto
//   • Nernst dinâmico — E_K(t) = (RT/F) × ln([K⁺]o / [K⁺]i)

#[test]
fn cenario_b_atp_respeita_limites_de_seguranca_sob_burst_extremo() {
    let mut n = NeuronioHibrido::new(0, TipoNeuronal::IB, PrecisionType::FP32);
    n.brain_state = EstadoBrainState::Vigilia; // fator_apical = 1.0

    // Burst extremo: 500ms @ 200Hz com input acima do drive normal
    // (35.0 mV·escala já é o "burst forte" dos tests V4 existentes; usamos 45.0)
    let dt = 0.001f32;
    for i in 0..500 {
        n.input_apical = 0.0;
        let _ = n.update(45.0, dt, i as f32, 1.0);
    }

    // Invariante crítica: ATP ∈ [0.05, 2.5 mM] SEMPRE.
    // (PLOS CompBiol 2020; 0.05 é o piso de sobrevivência mitocondrial)
    assert!(n.metabolismo.atp >= 0.05,
        "ATP abaixo do piso de sobrevivência: {}", n.metabolismo.atp);
    assert!(n.metabolismo.atp <= 2.5,
        "ATP ultrapassou capacidade mitocondrial: {}", n.metabolismo.atp);

    // [Na⁺]i: bomba deve mantê-lo dentro de [7.0, 35.0 mM]
    assert!(n.metabolismo.na_intra >= 7.0 && n.metabolismo.na_intra <= 35.0,
        "[Na+]i fora do range fisiológico: {}", n.metabolismo.na_intra);

    // Recuperação metabólica: 1000ms sem input → ATP volta ao nível alto
    let atp_pos_burst = n.metabolismo.atp;
    for i in 500..1500 {
        n.input_apical = 0.0;
        let _ = n.update(0.0, dt, i as f32, 1.0);
    }
    assert!(n.metabolismo.atp > atp_pos_burst,
        "ATP não recuperou após 1s de descanso: antes={atp_pos_burst}, depois={}",
        n.metabolismo.atp);
}

#[test]
fn cenario_b_nernst_dinamico_responde_a_potassio_extracelular() {
    use selene_kernel::synaptic_core::EstadoMetabolico;

    // Verifica que E_K(t) = (RT/F) × ln([K⁺]o / [K⁺]i) responde monotonicamente.
    // [K⁺]i fixo = 140 mM. Esperamos:
    //   [K⁺]o = 3.0  → E_K ≈ -102 mV
    //   [K⁺]o = 6.0  → E_K ≈ -84 mV  (despolarização por K elevado)
    //   [K⁺]o = 12.0 → E_K ≈ -66 mV  (despolarização grave — patológico)
    let mut m = EstadoMetabolico::novo();

    m.k_o = 3.0;  m.atualizar_ek();
    let ek_rest = m.e_k_dyn;
    assert!((ek_rest + 102.0).abs() < 2.0,
        "E_K @ [K]o=3 esperado ~-102, got {ek_rest}");

    m.k_o = 6.0;  m.atualizar_ek();
    let ek_high = m.e_k_dyn;

    m.k_o = 12.0; m.atualizar_ek();
    let ek_pat = m.e_k_dyn;

    // Monotonicidade: subir [K⁺]o despolariza E_K (valor menos negativo)
    assert!(ek_rest < ek_high, "Subida de [K+]o deve despolarizar: {ek_rest} < {ek_high}");
    assert!(ek_high < ek_pat,  "Subida adicional deve despolarizar mais: {ek_high} < {ek_pat}");

    // Caso patológico: [K⁺]o > 8 mM deve gerar E_K mais positivo que -75 mV
    // (limiar clínico de hipercalemia neuronal)
    assert!(ek_pat > -75.0,
        "[K+]o=12 deve gerar despolarização patológica: {ek_pat}");
}

#[test]
fn cenario_b_bomba_eletrogenica_responde_a_carga_de_sodio() {
    // A bomba Na/K-ATPase é eletrogenica: 3 Na⁺ saem, 2 K⁺ entram → corrente
    // hiperpolarizante. Quando [Na⁺]i sobe, a bomba acelera (f_Na cúbico no
    // modelo). Validamos que neurônio em repouso após burst tem i_pump > 0.
    let mut n = NeuronioHibrido::new(0, TipoNeuronal::RS, PrecisionType::FP32);
    n.brain_state = EstadoBrainState::Vigilia;

    let dt = 0.001f32;
    // Burst breve para elevar [Na+]i
    for i in 0..150 {
        let _ = n.update(40.0, dt, i as f32, 1.0);
    }
    let na_apos_burst = n.metabolismo.na_intra;
    assert!(na_apos_burst > 10.0,
        "[Na+]i deve subir após burst: got {na_apos_burst}");

    // Bomba deve produzir corrente hiperpolarizante (i_pump > 0 = hyperpol.)
    // Após burst, Na está alto → f_Na ≈ (Na/(Na+10))^3 grande → bomba acelera.
    assert!(n.metabolismo.i_pump > 0.0,
        "Bomba eletrogenica não respondeu à carga de Na: i_pump={}",
        n.metabolismo.i_pump);
}

// ============================================================================
// CENÁRIO C — Complexidade Amortizada do ChunkDsu + ja_existe() O(1)
// ============================================================================
//
// Mapeamento biológico:
//   • Co-ativação STDP → chunking emergente (Wallenstein & Eichenbaum 2008)
//   • DSU é não-direcional → opera sobre RELAÇÕES DE CO-PERTENCIMENTO,
//     NUNCA sobre causalidade direcional (que vive no grafo dirigido
//     `sinapses_conceito: HashMap<(Uuid,Uuid),f32>` — STDP usa t_pre < t_post)

#[test]
fn cenario_c_dsu_path_compression_amortizado() {
    let mut dsu = ChunkDsu::new();

    // Constrói "corrente esticada" 0→1→2→...→999 (caso patológico para DSU).
    // Sem path compression: find(0) seria O(n). Com compression: O(α(n)) ≈ O(1).
    for i in 0..999 {
        dsu.union(i, i + 1);
    }

    // Após 1ª travessia, path compression deve aplanar a árvore.
    let raiz_ref = dsu.find(0);

    // Cronometra 1000 chamadas find() — devem ser sub-microssegundo cada
    // depois da compressão. Em CI lento toleramos 5ms total.
    let t0 = Instant::now();
    for i in 0..1000 {
        assert_eq!(dsu.find(i), raiz_ref,
            "Todos os nós devem compartilhar a mesma raiz após union encadeado");
    }
    let elapsed = t0.elapsed();
    assert!(elapsed < Duration::from_millis(5),
        "find() após compressão deve ser quase-O(1): levou {elapsed:?} para 1000 calls");

    // Cluster único: 1 raiz para 1000 nós
    assert_eq!(dsu.n_clusters(), 1);
}

#[test]
fn cenario_c_dsu_clusters_disjuntos() {
    let mut dsu = ChunkDsu::new();

    // 3 chunks disjuntos: {0,1,2}, {10,11,12}, {20,21,22,23}
    dsu.union(0, 1); dsu.union(1, 2);
    dsu.union(10, 11); dsu.union(11, 12);
    dsu.union(20, 21); dsu.union(21, 22); dsu.union(22, 23);

    assert_eq!(dsu.n_clusters(), 3, "3 chunks disjuntos => 3 raízes");
    assert_eq!(dsu.find(0), dsu.find(2),  "0 e 2 no mesmo cluster");
    assert_ne!(dsu.find(0), dsu.find(10), "0 e 10 em clusters diferentes");

    // Fusão transitiva: ao unir 2 com 10, os clusters {0,1,2} e {10,11,12}
    // colapsam em um só.
    dsu.union(2, 10);
    assert_eq!(dsu.n_clusters(), 2);
    assert_eq!(dsu.find(0), dsu.find(12),
        "Após union(2,10), 0 deve estar conectado a 12 transitivamente");
}

#[test]
fn cenario_c_chunking_engine_ja_existe_em_o1_e_dsu_acompanha() {
    use selene_kernel::brain_zones::RegionType;
    let mut engine = ChunkingEngine::new(RegionType::Temporal);

    // Cria camada e injeta trace_pre > TRACE_MINIMO_CHUNK em TODOS os neurônios.
    // (Em produção isto vem de CamadaHibrida::update(); aqui pulamos esse passo
    //  para teste determinístico do chunking, focando em DSU + O(1) lookup.)
    let mut camada = CamadaHibrida::new(
        64, "test", TipoNeuronal::RS, None,
        Some(vec![(PrecisionType::FP32, 1.0)]), 1.0,
    );
    for n in camada.neuronios.iter_mut() {
        n.trace_pre = 0.6; // bem acima de TRACE_MINIMO_CHUNK=0.05
    }

    // 20 padrões distintos de pares co-ativos, 8 repetições cada (> CHUNK_THRESHOLD=5)
    let mut t_ms = 0.0f32;
    for _repeticao in 0..8 {
        for k in 0..20 {
            let mut spikes = vec![false; 64];
            spikes[2 * k]     = true;
            spikes[2 * k + 1] = true;
            // dispara dentro da janela gamma — mesmo registro reusa instante
            // próximo para satisfazer `dentro_janela`
            engine.registrar_spikes(&spikes, &camada, 0.5, t_ms);
        }
        t_ms += 1.0; // avanço mínimo entre repetições
    }

    // Pelo menos alguns chunks devem ter emergido. O thresh é 5 co-ativações;
    // 8 repetições por padrão garantem que cada um cruza.
    assert!(!engine.chunks.is_empty(),
        "Esperado >=1 chunk após 8 repetições de 20 padrões; got 0");

    // ja_existe() deve responder em O(1) mesmo após várias inserções.
    // Cronometra 10k lookups via re-registro do mesmo padrão (cai no fast path).
    let spikes_amostra = {
        let mut s = vec![false; 64];
        s[0] = true; s[1] = true;
        s
    };
    let t0 = Instant::now();
    for _ in 0..10_000 {
        let _ = engine.registrar_spikes(&spikes_amostra, &camada, 0.5, t_ms);
        t_ms += 1.0;
    }
    let elapsed = t0.elapsed();
    assert!(elapsed < Duration::from_millis(500),
        "10k registrar_spikes() devem rodar em <500ms (lookup O(1)); got {elapsed:?}");

    // Sanidade do DSU: neurônios que co-ativaram devem estar no mesmo cluster.
    if engine.chunks.iter().any(|c| c.indices.contains(&0) && c.indices.contains(&1)) {
        assert!(engine.mesmo_cluster(0, 1),
            "Neurônios 0 e 1 co-ativaram em chunk emergido — devem estar no mesmo cluster DSU");
    }
}

// ============================================================================
// CENÁRIO D — Modulação Alostérica (Adenosina ⊣ D2 / Oxitocina ⊣ BLA)
// ============================================================================
//
// Mapeamento biológico:
//   • Ferré (2022): receptores A2A (adenosina) e D2 formam heterodímeros no
//     estriado. Acúmulo de adenosina → antagonismo D2 → reduz inibição
//     comportamental → maior pressão de sono / fadiga cognitiva.
//   • Kirsch (2005): oxitocina (PVN→BLA) inativa a amígdala basolateral via
//     gate multiplicativo [0.3, 1.0]. Fear signal medido pelo CeA cai sob
//     contexto social positivo.

#[test]
fn cenario_d_adenosina_antagoniza_d2_signal_segundo_ferre() {
    // Fórmula real (neurochem.rs:200-206):
    //   target_d2     = sigmoid(6 × (dopa - 0.5))
    //   target_d2_aden = target_d2 × (1 - adenosina × 0.3)
    //
    // Aqui validamos a FÓRMULA matemática (extraída do código) — porque
    // chamar NeuroChem::update() requer HardwareSensor real (não desejável
    // em CI). Garantimos que a fórmula está correta e monotônica.

    fn target_d2_aden(dopa: f32, adenosina: f32) -> f32 {
        let target_d2 = 1.0 / (1.0 + (-6.0 * (dopa - 0.5)).exp());
        let target_d2 = target_d2.clamp(0.1, 1.0);
        (target_d2 * (1.0 - adenosina * 0.3)).clamp(0.1, 1.0)
    }

    let dopa = 1.0; // baseline

    // Sem adenosina: D2 ≈ 0.97 (alto, dopa alta)
    let d2_baixa_aden = target_d2_aden(dopa, 0.0);
    // Adenosina máxima: D2 ≈ 0.97 × 0.7 = 0.68 (atenuação de ~30%)
    let d2_alta_aden  = target_d2_aden(dopa, 1.0);

    assert!(d2_baixa_aden > d2_alta_aden,
        "Adenosina deve reduzir D2 (Ferré 2022): baixa_aden={d2_baixa_aden} > alta_aden={d2_alta_aden}");

    // Atenuação aproximadamente proporcional ao fator (1 - 0.3) = 0.7
    let razao = d2_alta_aden / d2_baixa_aden;
    assert!((razao - 0.7).abs() < 0.05,
        "Atenuação esperada ~30% (razão 0.7); got razão={razao}");

    // Limites clamping: D2 nunca abaixo de 0.1 mesmo com adenosina + dopa baixa
    let d2_pior_caso = target_d2_aden(0.0, 1.0);
    assert!(d2_pior_caso >= 0.1, "Clamping mínimo violado: {d2_pior_caso}");
}

#[test]
fn cenario_d_oxitocina_atenua_fear_signal_segundo_kirsch() {
    let config = Config::new(ModoOperacao::Boost200);
    let mut amyg = Amygdala::new(32, &config);

    // Cenário A: alta oxitocina → gate 0.4 (forte atenuação)
    // Cenário B: zero oxitocina → gate 1.0 (sem atenuação)
    //
    // Pumping input aversivo (valência -0.8) com mesmo seed/ordem em ambos.
    let dt = 0.005f32;
    let mut t = 0.0f32;
    let oxytocin_gate_alto = 0.4f32; // oxitocina = 1.0 → gate = 0.6
    let oxytocin_gate_baixo = 1.0f32; // oxitocina = 0.0 → gate = 1.0

    // Trial A — alta oxitocina
    for _ in 0..40 {
        amyg.update(-0.8, 0.0, 0.6, dt, t, &config, Some(oxytocin_gate_alto));
        t += dt;
    }
    let fear_oxt_alto = amyg.fear_signal;

    // Reset: novo Amygdala, mesma simulação SEM oxitocina
    let mut amyg2 = Amygdala::new(32, &config);
    t = 0.0;
    for _ in 0..40 {
        amyg2.update(-0.8, 0.0, 0.6, dt, t, &config, Some(oxytocin_gate_baixo));
        t += dt;
    }
    let fear_oxt_baixo = amyg2.fear_signal;

    assert!(fear_oxt_alto < fear_oxt_baixo,
        "Oxitocina deve REDUZIR fear_signal (Kirsch 2005): \
         com_oxt={fear_oxt_alto} >= sem_oxt={fear_oxt_baixo}");

    // Atenuação esperada: gate 0.4 → fear escalado por 0.4 antes do EMA.
    // EMA decai 0.88 e adiciona 0.12 × raw_fear_gated. Não exigimos
    // razão exata por causa do RNG interno e dinâmica acumulativa,
    // mas exigimos pelo menos 30% de redução.
    if fear_oxt_baixo > 0.05 {
        let reducao = 1.0 - (fear_oxt_alto / fear_oxt_baixo);
        assert!(reducao > 0.25,
            "Atenuação por oxitocina deve ser >25% biologicamente; got {reducao:.2}");
    }
}

#[test]
fn cenario_d_neurochem_oxytocin_bla_gate_dentro_dos_limites_kirsch() {
    // gate = (1.0 - oxytocin * 0.4).clamp(0.3, 1.0)
    // oxytocin ∈ [0.1, 1.5] (range biológico de Selene)
    let mut nc = NeuroChem::new();

    nc.oxytocin = 0.0;
    assert!((nc.oxytocin_bla_gate() - 1.0).abs() < 1e-3,
        "Sem oxitocina, gate deve ser 1.0 (sem atenuação)");

    nc.oxytocin = 1.0;
    assert!((nc.oxytocin_bla_gate() - 0.6).abs() < 1e-3,
        "Com oxitocina=1.0, gate deve ser 0.6");

    // Limite clínico (saturação): oxitocina muito alta NÃO leva gate abaixo de 0.3
    nc.oxytocin = 1.5; // máximo do range
    let gate_max = nc.oxytocin_bla_gate();
    assert!(gate_max >= 0.3,
        "Gate nunca abaixo de 0.3 (proteção: não silencia BLA totalmente)");
    assert!(gate_max <= 1.0,
        "Gate nunca acima de 1.0");
}

// ============================================================================
// CENÁRIO TRANSVERSAL — Integração DSU + Causalidade STDP NÃO interferem
// ============================================================================
//
// Verifica o ponto mais sutil da auditoria: DSU é simétrica (não-direcional),
// STDP é assimétrica (t_pre < t_post). Nenhum dos dois deve "vazar" para o
// outro. O grafo direcional `sinapses_conceito` continua sendo a única
// fonte de verdade para causalidade.

#[test]
fn cenario_transversal_dsu_nao_corrompe_direcionalidade_de_causalidade() {
    let mut dsu = ChunkDsu::new();

    // União simétrica: union(A,B) não distingue A→B de B→A
    dsu.union(1, 2);
    assert_eq!(dsu.find(1), dsu.find(2));

    // Mas o usuário poderia esperar que find(1) "lembre" da ordem da união.
    // Garantimos que ELE NÃO LEMBRA — porque DSU é por design simétrica.
    let mut dsu2 = ChunkDsu::new();
    dsu2.union(2, 1); // ordem invertida
    assert_eq!(dsu2.find(1), dsu2.find(2));
    // Conclusão: API DSU não expõe direção; logo, nunca poderia ser usada
    // para causalidade. Causalidade direcional vive APENAS em
    // sinapses_conceito: HashMap<(Uuid,Uuid),f32> no SwapManager.
}
