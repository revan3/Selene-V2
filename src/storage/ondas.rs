// src/storage/ondas.rs
// Tipos de onda e persistência da arquitetura wave-first da Selene.
//
// O DB nunca armazena texto ou símbolos fonéticos.
// Armazena apenas parâmetros físicos de onda:
//   - Som:    F0, F1, F2, F3, VOT, tipo de onset, delta de formantes
//   - Luz:    comprimento de onda nm, luminância, frequência espacial
//   - Interno: ritmo corporal (batimento, respiração, alfa, beta, gama)
//
// Tabelas SurrealDB usadas:
//   primitivas_onda     — instâncias individuais de percepção
//   bigramas_foneticos  — pares ordenados de primitivas (camada 2)
//   padroes_temporais   — sequências multi-primitiva (sílabas emergentes)
//   ondas_internas      — estado fisiológico gravado junto com percepções
#![allow(dead_code)]

use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use uuid::Uuid;

use super::tipos::{hash_bytes, CamadaFonetica, SpikeHash};

// ─── Enums base ───────────────────────────────────────────────────────────────

/// Modalidade sensorial da primitiva.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TipoOnda {
    Sonora,
    Luminosa,
    Interna,
}

/// Tipo de onset consonantal, detectado por distribuição espectral.
/// Crítico para distinguir consoantes com formantes vocálicos similares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TipoOnset {
    /// Oclusiva surda — silêncio (VOT longo) + burst de pressão
    OclusivaSurda,
    /// Oclusiva sonora — murmúrio voiced + burst (VOT curto/negativo)
    OclusivaSonora,
    /// Fricativa — ruído espectral contínuo (>3kHz)
    Fricativa,
    /// Nasal — murmúrio nasal ~250Hz + anti-formante
    Nasal,
    /// Lateral — L com energia em F3 específico
    Lateral,
    /// Aproximante — transição suave sem burst
    Aproximante,
    /// Vogal pura — sem onset consonantal
    Vogal,
    /// Silêncio / pausa
    Silencio,
}

// ─── PrimitivaOnda ────────────────────────────────────────────────────────────

/// Representação física de um momento de percepção.
/// Nunca contém texto, letras ou símbolos fonéticos.
/// O significado emerge da co-ativação repetida destas primitivas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitivaOnda {
    /// Hash determinístico dos parâmetros quantizados (chave de lookup).
    pub hash:        SpikeHash,
    pub tipo:        TipoOnda,

    // ── Parâmetros sonoros ────────────────────────────────────────────
    /// Frequência fundamental (pitch voiced). None = surdo.
    pub f0_hz:       Option<f32>,
    /// 1º formante — abertura da boca (200–900 Hz).
    pub f1_hz:       Option<f32>,
    /// 2º formante — posição da língua frente-atrás (700–2500 Hz).
    pub f2_hz:       Option<f32>,
    /// 3º formante — arredondamento labial (2000–3500 Hz).
    pub f3_hz:       Option<f32>,
    /// Taxa de variação de F1 (Hz/ms) — distingue onset de transição.
    pub delta_f1:    f32,
    /// Taxa de variação de F2 (Hz/ms) — índice de locus de consoantes.
    pub delta_f2:    f32,
    /// Voice Onset Time em ms (tempo entre onset e vozeamento).
    /// Negativo = pré-vozeamento (oclusivas sonoras do PT-BR).
    pub vot_ms:      f32,
    /// Tipo de onset detectado.
    pub onset:       TipoOnset,
    /// Energia total normalizada [0.0, 1.0].
    pub amplitude:   f32,
    /// Razão de energia em alta frequência (>3kHz) — índice de fricação.
    pub hf_ratio:    f32,

    // ── Parâmetros luminosos ──────────────────────────────────────────
    /// Comprimento de onda dominante em nm (380–700 nm). None = sonora.
    pub comprimento_onda_nm: Option<f32>,
    /// Luminância [0.0, 1.0].
    pub luminancia:          Option<f32>,
    /// Frequência espacial dominante (ciclos/pixel) — textura/borda.
    pub freq_espacial:       Option<f32>,
    /// Orientação dominante de borda em graus (0–180).
    pub orientacao_graus:    Option<f32>,
    /// Taxa de variação temporal — detecta movimento.
    pub taxa_variacao:       Option<f32>,

    // ── Parâmetros internos ───────────────────────────────────────────
    /// Frequência do ritmo interno (Hz). Ex: 1.0 = batimento cardíaco.
    pub freq_interna_hz:     Option<f32>,

    // ── Temporal ─────────────────────────────────────────────────────
    /// Duração da percepção em ms.
    pub duracao_ms:  u32,
    /// Timestamp Unix (s).
    pub timestamp:   f64,
    /// ID do episódio de binding temporal a que pertence.
    pub episodio_id: Option<String>,
}

impl PrimitivaOnda {
    /// Computa hash determinístico a partir dos parâmetros quantizados.
    /// Quantização a 10Hz evita ruído de ponto flutuante gerar hashes diferentes
    /// para o mesmo fonema.
    pub fn computar_hash(
        tipo:    &TipoOnda,
        f1:      Option<f32>,
        f2:      Option<f32>,
        onset:   &TipoOnset,
        hf:      f32,
    ) -> SpikeHash {
        let mut d = Vec::with_capacity(20);
        d.push(*tipo as u8);
        d.push(*onset as u8);
        // Quantiza a 10Hz para tolerância a jitter de medição
        let q = |f: f32| -> u32 { (f / 10.0).round() as u32 };
        if let Some(f) = f1 { d.extend_from_slice(&q(f).to_le_bytes()); }
        if let Some(f) = f2 { d.extend_from_slice(&q(f).to_le_bytes()); }
        d.extend_from_slice(&((hf * 100.0) as u32).to_le_bytes());
        hash_bytes(&d)
    }

    /// Cria uma PrimitivaOnda sonora com hash automático.
    pub fn sonora(
        f0: Option<f32>, f1: Option<f32>, f2: Option<f32>, f3: Option<f32>,
        delta_f1: f32, delta_f2: f32,
        vot_ms: f32, onset: TipoOnset,
        amplitude: f32, hf_ratio: f32,
        duracao_ms: u32, timestamp: f64,
    ) -> Self {
        let hash = Self::computar_hash(&TipoOnda::Sonora, f1, f2, &onset, hf_ratio);
        Self {
            hash, tipo: TipoOnda::Sonora,
            f0_hz: f0, f1_hz: f1, f2_hz: f2, f3_hz: f3,
            delta_f1, delta_f2, vot_ms, onset, amplitude, hf_ratio,
            comprimento_onda_nm: None, luminancia: None,
            freq_espacial: None, orientacao_graus: None, taxa_variacao: None,
            freq_interna_hz: None,
            duracao_ms, timestamp, episodio_id: None,
        }
    }

    /// Cria uma PrimitivaOnda luminosa com hash automático.
    pub fn luminosa(
        comprimento_onda_nm: f32, luminancia: f32,
        freq_espacial: f32, orientacao_graus: f32, taxa_variacao: f32,
        duracao_ms: u32, timestamp: f64,
    ) -> Self {
        let mut d = Vec::with_capacity(12);
        d.push(TipoOnda::Luminosa as u8);
        d.extend_from_slice(&((comprimento_onda_nm as u32 / 5) * 5).to_le_bytes());
        d.extend_from_slice(&((luminancia * 100.0) as u32).to_le_bytes());
        let hash = hash_bytes(&d);
        Self {
            hash, tipo: TipoOnda::Luminosa,
            f0_hz: None, f1_hz: None, f2_hz: None, f3_hz: None,
            delta_f1: 0.0, delta_f2: 0.0, vot_ms: 0.0,
            onset: TipoOnset::Silencio, amplitude: luminancia, hf_ratio: 0.0,
            comprimento_onda_nm: Some(comprimento_onda_nm),
            luminancia: Some(luminancia),
            freq_espacial: Some(freq_espacial),
            orientacao_graus: Some(orientacao_graus),
            taxa_variacao: Some(taxa_variacao),
            freq_interna_hz: None,
            duracao_ms, timestamp, episodio_id: None,
        }
    }

    /// Cria uma PrimitivaOnda de ritmo interno.
    pub fn interna(freq_hz: f32, amplitude: f32, timestamp: f64) -> Self {
        let mut d = [0u8; 8];
        d[0] = TipoOnda::Interna as u8;
        d[1..5].copy_from_slice(&((freq_hz * 100.0) as u32).to_le_bytes());
        let hash = hash_bytes(&d);
        Self {
            hash, tipo: TipoOnda::Interna,
            f0_hz: Some(freq_hz), f1_hz: None, f2_hz: None, f3_hz: None,
            delta_f1: 0.0, delta_f2: 0.0, vot_ms: 0.0,
            onset: TipoOnset::Silencio, amplitude, hf_ratio: 0.0,
            comprimento_onda_nm: None, luminancia: None,
            freq_espacial: None, orientacao_graus: None, taxa_variacao: None,
            freq_interna_hz: Some(freq_hz),
            duracao_ms: 0, timestamp, episodio_id: None,
        }
    }
}

// ─── BigramaFonetico ──────────────────────────────────────────────────────────

/// Par ordenado de primitivas de onda — camada 2 da memória.
/// Emerge da co-ativação sequencial repetida.
/// NUNCA armazena "ba" ou "A→M" — armazena hashes de padrões acústicos.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BigramaFonetico {
    /// Hash do par (hash_de + hash_para).
    pub hash:      SpikeHash,
    /// Primitiva anterior.
    pub de:        SpikeHash,
    /// Primitiva seguinte.
    pub para:      SpikeHash,
    /// Posição na sequência (0 = início de sílaba).
    pub posicao:   u8,
    /// Valência emocional emergente desta transição [-1.0, 1.0].
    pub valencia:  f32,
    /// Número de co-ativações observadas.
    pub contagem:  u32,
    /// Timestamp do último reforço.
    pub ultimo_uso: f64,
}

impl BigramaFonetico {
    pub fn novo(de: SpikeHash, para: SpikeHash, posicao: u8) -> Self {
        let hash = hash_bytes((de.as_str().to_string() + para.as_str()).as_bytes());
        let agora = agora_f64();
        Self { hash, de, para, posicao, valencia: 0.0, contagem: 1, ultimo_uso: agora }
    }
}

// ─── PadraoTemporal ───────────────────────────────────────────────────────────

/// Sequência de primitivas co-ativadas dentro de uma janela temporal.
/// Sílabas e palavras emergem como padrões de alta contagem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PadraoTemporal {
    /// Hash da sequência completa.
    pub hash:              SpikeHash,
    /// Hashes das primitivas em ordem temporal.
    pub sequencia:         Vec<SpikeHash>,
    /// Valência emocional emergente [-1.0, 1.0].
    pub valencia:          f32,
    /// Reforço acumulado (STDP).
    pub peso:              f32,
    /// Número de ocorrências.
    pub contagem:          u32,
    /// Camada hierárquica a que este padrão pertence.
    pub camada:            CamadaFonetica,
    /// Hashes de outros padrões que co-ocorrem frequentemente (camada 3).
    pub contextos:         Vec<SpikeHash>,
    pub timestamp_criacao: f64,
    pub ultimo_reforco:    f64,
}

impl PadraoTemporal {
    pub fn de_primitivas(primitivas: &[&PrimitivaOnda], camada: CamadaFonetica) -> Self {
        let sequencia: Vec<SpikeHash> = primitivas.iter().map(|p| p.hash.clone()).collect();
        let hash = hash_bytes(sequencia.join("|").as_bytes());
        let agora = agora_f64();
        Self {
            hash, sequencia, valencia: 0.0, peso: 1.0,
            contagem: 1, camada, contextos: Vec::new(),
            timestamp_criacao: agora, ultimo_reforco: agora,
        }
    }
}

// ─── Auxiliar de tempo ────────────────────────────────────────────────────────

fn agora_f64() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64()
}

// ─── Persistência SurrealDB ───────────────────────────────────────────────────

/// Cria índices das novas tabelas de onda. Idempotente.
pub async fn inicializar_schema_ondas(db: &Surreal<Db>) -> surrealdb::Result<()> {
    // primitivas_onda — lookup por hash e por tipo
    let _ = db.query("DEFINE INDEX ponda_hash_idx ON TABLE primitivas_onda FIELDS hash UNIQUE;").await;
    let _ = db.query("DEFINE INDEX ponda_tipo_idx  ON TABLE primitivas_onda FIELDS tipo;").await;
    let _ = db.query("DEFINE INDEX ponda_ts_idx    ON TABLE primitivas_onda FIELDS timestamp;").await;

    // bigramas_foneticos — lookup por (de, para) e por de sozinho
    let _ = db.query("DEFINE INDEX bigrama_hash_idx ON TABLE bigramas_foneticos FIELDS hash UNIQUE;").await;
    let _ = db.query("DEFINE INDEX bigrama_de_idx   ON TABLE bigramas_foneticos FIELDS de;").await;

    // padroes_temporais — lookup por hash e por camada
    let _ = db.query("DEFINE INDEX padrao_hash_idx   ON TABLE padroes_temporais FIELDS hash UNIQUE;").await;
    let _ = db.query("DEFINE INDEX padrao_camada_idx ON TABLE padroes_temporais FIELDS camada;").await;
    let _ = db.query("DEFINE INDEX padrao_peso_idx   ON TABLE padroes_temporais FIELDS peso;").await;

    Ok(())
}

/// Persiste uma PrimitivaOnda (idempotente por hash).
pub async fn put_primitiva(db: &Surreal<Db>, p: &PrimitivaOnda) -> surrealdb::Result<()> {
    // INSERT ... ON DUPLICATE KEY UPDATE atualiza apenas campos mutáveis
    db.query(
        "INSERT INTO primitivas_onda $rec \
         ON DUPLICATE KEY UPDATE amplitude = $rec.amplitude, \
         timestamp = $rec.timestamp, episodio_id = $rec.episodio_id RETURN NONE"
    )
    .bind(("rec", p))
    .await?;
    Ok(())
}

/// Busca PrimitivaOnda pelo hash.
pub async fn get_primitiva(db: &Surreal<Db>, hash: &str) -> Option<PrimitivaOnda> {
    let mut r = db
        .query("SELECT * FROM primitivas_onda WHERE hash = $h LIMIT 1")
        .bind(("h", hash))
        .await.ok()?;
    let v: Vec<PrimitivaOnda> = r.take(0).ok()?;
    v.into_iter().next()
}

/// Persiste ou reforça um BigramaFonetico.
/// Se já existe (mesmo hash), incrementa contagem e atualiza valência por média móvel.
pub async fn put_bigrama(db: &Surreal<Db>, b: &BigramaFonetico) -> surrealdb::Result<()> {
    db.query(
        "INSERT INTO bigramas_foneticos $rec \
         ON DUPLICATE KEY UPDATE \
           contagem  = contagem + 1, \
           valencia  = (valencia * contagem + $rec.valencia) / (contagem + 1), \
           ultimo_uso = $rec.ultimo_uso \
         RETURN NONE"
    )
    .bind(("rec", b))
    .await?;
    Ok(())
}

/// Retorna todos os bigramas que partem de `from_hash`, ordenados por contagem desc.
pub async fn get_bigramas_de(db: &Surreal<Db>, from_hash: &str) -> Vec<BigramaFonetico> {
    let r = db
        .query("SELECT * FROM bigramas_foneticos WHERE de = $h ORDER BY contagem DESC")
        .bind(("h", from_hash))
        .await;
    match r {
        Ok(mut res) => res.take(0).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

/// Persiste ou reforça um PadraoTemporal (STDP: peso += 1 por ocorrência).
pub async fn put_padrao(db: &Surreal<Db>, p: &PadraoTemporal) -> surrealdb::Result<()> {
    db.query(
        "INSERT INTO padroes_temporais $rec \
         ON DUPLICATE KEY UPDATE \
           contagem      = contagem + 1, \
           peso          = peso + 1.0, \
           valencia      = (valencia * contagem + $rec.valencia) / (contagem + 1), \
           ultimo_reforco = $rec.ultimo_reforco \
         RETURN NONE"
    )
    .bind(("rec", p))
    .await?;
    Ok(())
}

/// Busca os N padrões de maior peso de uma camada específica.
pub async fn get_padroes_fortes(
    db: &Surreal<Db>,
    camada: CamadaFonetica,
    n: usize,
) -> Vec<PadraoTemporal> {
    let r = db
        .query("SELECT * FROM padroes_temporais WHERE camada = $c ORDER BY peso DESC LIMIT $n")
        .bind(("c", camada))
        .bind(("n", n as u64))
        .await;
    match r {
        Ok(mut res) => res.take(0).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

/// Aplica decaimento exponencial aos padrões com `ultimo_reforco` antigo.
/// Chamado periodicamente (ex: a cada ciclo de sono).
/// `taxa`: fração de decaimento por hora (ex: 0.05 = 5% por hora).
pub async fn decair_padroes(db: &Surreal<Db>, agora: f64, taxa_por_hora: f32) -> surrealdb::Result<()> {
    db.query(
        "UPDATE padroes_temporais \
         SET peso = peso * (1.0 - $taxa * ((($agora - ultimo_reforco) / 3600.0))) \
         WHERE peso > 0.01"
    )
    .bind(("taxa",  taxa_por_hora))
    .bind(("agora", agora))
    .await?;
    // Remove padrões com peso abaixo de threshold
    db.query("DELETE padroes_temporais WHERE peso < 0.01").await?;
    Ok(())
}
