// =============================================================================
// tests/neurogenese_hibridos.rs — V4.6
// Prova end-to-end: o sistema CRIA neurônios híbridos sob demanda e eles RESPONDEM.
// =============================================================================

use selene_kernel::stem_cell::{GestorNeurogenese, NecessidadeZona};
use selene_kernel::synaptic_core::{
    CamadaHibrida, NeuronalStatus, TipoNeuronal, gerar_especie_hibrida, NeuronioHibrido, PrecisionType,
};

const DT: f32 = 0.001;

/// Conduz a camada por N ticks com um drive uniforme (simula vigília).
fn simular_vigilia(zona: &mut CamadaHibrida, amp: f32, ticks: usize) {
    for t in 0..ticks {
        let inputs = vec![amp; zona.neuronios.len()];
        zona.update(&inputs, DT, t as f32);
    }
}

/// Marca uma fração dos neurônios como Dormant (zona subutilizada).
fn adormecer_fracao(zona: &mut CamadaHibrida, fracao: f32) {
    let total = zona.neuronios.len();
    let alvo = (total as f32 * fracao) as usize;
    for n in zona.neuronios.iter_mut().take(alvo) {
        n.status = NeuronalStatus::Dormant;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. O sistema DETECTA necessidade e CRIA um híbrido
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn sistema_cria_hibrido_quando_zona_subutilizada() {
    let mut zona = CamadaHibrida::new(40, "piloto", TipoNeuronal::RS, None, None, 1.0);
    adormecer_fracao(&mut zona, 0.6); // 60% dormindo → falta de drivers

    let necessidade = NecessidadeZona::avaliar(&zona);
    assert!(necessidade.precisa_intervir(),
        "zona 60% dormente deve sinalizar necessidade: {necessidade:?}");

    let n_antes = zona.neuronios.len();
    let mut gestor = GestorNeurogenese::novo(4);
    let (_aceitas, _rejeitadas, nascidas) = gestor.tick_sono(&mut zona);

    assert_eq!(nascidas, 1, "o sistema deve gerar exatamente 1 célula-tronco");
    assert_eq!(zona.neuronios.len(), n_antes + 1, "1 neurônio implantado");
    let implantado = zona.neuronios.last().unwrap();
    assert_eq!(implantado.tipo, TipoNeuronal::Hybrid, "o implante é um Hybrid");
    assert!(implantado.dna.is_some(), "o híbrido carrega um genoma (DNA)");
    assert_eq!(gestor.em_prova(), 1, "1 célula em período de prova");
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. O híbrido criado RESPONDE ao estímulo (dispara) e é JULGADO
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn hibrido_criado_responde_a_estimulo_e_e_julgado() {
    let mut zona = CamadaHibrida::new(40, "piloto2", TipoNeuronal::RS, None, None, 1.0);
    adormecer_fracao(&mut zona, 0.6);

    let mut gestor = GestorNeurogenese::novo(4);
    let (_, _, nascidas) = gestor.tick_sono(&mut zona);
    assert_eq!(nascidas, 1);
    let idx = zona.neuronios.len() - 1; // híbrido na cauda

    // ── Vigília: o híbrido recebe drive forte e deve RESPONDER ──
    for t in 0..3000 {
        let mut inputs = vec![3.0f32; zona.neuronios.len()];
        inputs[idx] = 40.0;
        zona.update(&inputs, DT, t as f32);
    }
    let hib = &zona.neuronios[idx];
    println!(
        "Híbrido [id={}]: e_inib={} g_m={:.2} thr={:.1} → activity_avg={:.4}",
        hib.id,
        hib.e_inibitorico_efetivo(),
        hib.g_m_efetivo(),
        hib.threshold,
        hib.activity_avg,
    );
    assert!(hib.activity_avg > 0.0,
        "o híbrido deve RESPONDER (disparar) ao estímulo — activity_avg={}",
        hib.activity_avg);

    // ── Próximo sono: a célula em prova é julgada ──
    let (aceitas, rejeitadas, _) = gestor.tick_sono(&mut zona);
    assert_eq!(aceitas + rejeitadas, 1,
        "a célula em prova deve ser julgada (aceita ou rejeitada)");
    if aceitas == 1 {
        assert!(gestor.registro.total() >= 1,
            "espécie aceita deve ser registrada com assinatura");
        println!("✅ Espécie ACEITA e registrada: {:?}",
            gestor.registro.especies.last().map(|e| &e.assinatura));
    } else {
        println!("🔻 Célula REJEITADA (apoptose) — fenótipo inviável neste contexto");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Crossover genético produz espécies VIÁVEIS que disparam
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn crossover_gera_especies_viaveis_de_pais_diferentes() {
    // Cruza tipos bem diferentes e confirma que o filho é instanciável e dispara.
    let casais = [
        (TipoNeuronal::RS, TipoNeuronal::FS),
        (TipoNeuronal::CH, TipoNeuronal::MSN),
        (TipoNeuronal::GridCell, TipoNeuronal::MirrorCell),
    ];
    for (pa, pb) in casais {
        let dna = gerar_especie_hibrida(&pa, &pb, 0.13);
        let mut n = NeuronioHibrido::novo_hibrido(7, dna, PrecisionType::FP32);
        let mut spikes = 0;
        for t in 0..3000 {
            if n.update(45.0, DT, t as f32, 1.0) { spikes += 1; }
        }
        println!("{pa:?} × {pb:?} → híbrido disparou {spikes}x em 3s");
        assert!(spikes > 0,
            "espécie híbrida {pa:?}×{pb:?} deve ser viável e disparar (got {spikes})");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Evolução ao longo de vários ciclos de sono — registro cresce, população limitada
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn evolucao_multiciclo_respeita_cap_populacional() {
    let mut zona = CamadaHibrida::new(60, "evo", TipoNeuronal::RS, None, None, 1.0);
    let mut gestor = GestorNeurogenese::novo(2); // cap baixo p/ testar limite

    for ciclo in 0..6 {
        // Mantém a zona "carente" (subutilizada) a cada ciclo.
        adormecer_fracao(&mut zona, 0.6);
        let (ac, rej, nasc) = gestor.tick_sono(&mut zona);
        // Vigília curta para as células em prova acumularem (ou não) atividade.
        simular_vigilia(&mut zona, 12.0, 800);
        println!("ciclo {ciclo}: nasc={nasc} ac={ac} rej={rej} em_prova={}", gestor.em_prova());
        // Nunca ultrapassa o cap de células em prova simultâneas.
        assert!(gestor.em_prova() <= 2,
            "população em prova deve respeitar o cap (got {})", gestor.em_prova());
    }
    // Ao longo dos ciclos, o sistema gerou gerações evolutivas.
    assert!(gestor.ciclos_sono >= 6, "deve ter contado os ciclos de sono");
}
