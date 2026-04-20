// src/sleep_cycle.rs
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::time::Duration;
use crate::storage::memory_tier::MemoryTier;
use crate::storage::memory_graph::{MemoryTierV2, ConexaoSinaptica, ContextoSemantico};
use crate::storage::{backup_to_hdd, BrainStorage};
use crate::config::Config;
use uuid::Uuid;
use chrono;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

const DB_PATH: &str = "selene_memories.db";
const BACKUP_ROOT: &str = "D:/Selene_Backup_RAM";
const ARCHIVE_PATH: &str = "D:/Selene_Archive";

// ================== TIPOS AUXILIARES ==================

#[derive(Debug, Clone)]
pub struct Experiencia {
    pub id: String,
    pub timestamp: f64,
    pub contexto: String,
    pub emocao: f32,
    pub neurons_ativos: Vec<usize>,
    pub importancia: f32,
    pub frequencia_acesso: f32,
}

#[derive(Debug, Clone)]
pub struct Hipotese {
    pub descricao: String,
    pub conexoes_novas: Vec<ConexaoSinaptica>,
    pub probabilidade: f32,
    pub origem: Vec<String>,
    pub testada: bool,
    pub valida: Option<bool>,
}

// ================== FASE DO SONO ==================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FaseSono {
    N1, // Consolidação leve — persiste conexões recentes
    N2, // Revisão e Poda — remove sinapses negativas por contexto
    N3, // REM (Sonho) — recombina conexões esquecidas
    N4, // Backup — snapshot do RocksDB para o HDD
}

impl FaseSono {
    pub fn nome(&self) -> &'static str {
        match self {
            FaseSono::N1 => "N1 - Consolidação",
            FaseSono::N2 => "N2 - Revisão e Poda",
            FaseSono::N3 => "N3 - REM (Sonho)",
            FaseSono::N4 => "N4 - Backup",
        }
    }

    pub fn duracao_minutos(&self) -> u64 { 90 }
}

// ================== CICLO DE SONO ==================

pub struct CicloSono {
    pub fase_atual: FaseSono,
    progresso: f32,
    inicio_fase: std::time::Instant,
    memorias_do_dia: Vec<Experiencia>,
    hipoteses_geradas: Vec<Hipotese>,
    versao_sistema: String,
    /// Grafo de memória sináptica compartilhado com o main loop
    pub grafo: Option<Arc<TokioMutex<MemoryTierV2>>>,
    /// Banco de dados — necessário para persistir mutações sinápticas
    pub db: Option<Arc<BrainStorage>>,
    /// BrainState compartilhado — necessário para REM semântico real
    pub brain_state: Option<Arc<TokioMutex<crate::websocket::bridge::BrainState>>>,
}

impl CicloSono {
    pub fn new() -> Self {
        Self {
            fase_atual: FaseSono::N1,
            progresso: 0.0,
            inicio_fase: std::time::Instant::now(),
            memorias_do_dia: Vec::new(),
            hipoteses_geradas: Vec::new(),
            versao_sistema: env!("CARGO_PKG_VERSION").to_string(),
            grafo: None,
            db: None,
            brain_state: None,
        }
    }

    pub fn with_grafo(mut self, grafo: Arc<TokioMutex<MemoryTierV2>>) -> Self {
        self.grafo = Some(grafo);
        self
    }

    pub fn with_db(mut self, db: Arc<BrainStorage>) -> Self {
        self.db = Some(db);
        self
    }

    pub fn with_brain_state(mut self, bs: Arc<TokioMutex<crate::websocket::bridge::BrainState>>) -> Self {
        self.brain_state = Some(bs);
        self
    }

    // ================================================================
    // FASE N1 — Consolidação
    // Move conexões recentes do hipocampo para storage persistente
    // Reforça conexões com emocao_media alta
    // ================================================================
    async fn fase_n1(&mut self, memoria: &mut MemoryTier) {
        println!("\n🌙 FASE 1: {}", FaseSono::N1.nome());

        memoria.flush_to_l3();
        println!("   ✅ L1 → L3: memória de trabalho consolidada no NVMe");

        if let Some(grafo_arc) = &self.grafo {
            let mut grafo = grafo_arc.lock().await;

            // Reforça conexões emocionalmente relevantes e coleta as atualizadas
            let mut atualizadas: Vec<(Uuid, f32, f32, f64, u32)> = Vec::new();
            for c in grafo.grafo_completo.conexoes.values_mut() {
                if c.emocao_media > 0.7 && c.marcador_poda >= 0.0 {
                    c.reforcar(0.05);
                    let uso = c.ultimo_uso.unwrap_or(c.criada_em);
                    atualizadas.push((c.id, c.peso, c.marcador_poda, uso, c.total_usos));
                }
            }
            println!("   💪 {} conexões emocionalmente relevantes reforçadas", atualizadas.len());

            // Persiste os pesos atualizados no DB
            if let Some(db) = &self.db {
                let mut erros = 0usize;
                for (id, peso, mp, uso, n) in &atualizadas {
                    if db.update_conexao_peso(*id, *peso, *mp, *uso, *n).await.is_err() {
                        erros += 1;
                    }
                }
                if erros > 0 {
                    println!("   ⚠️  {} falhas ao persistir pesos no DB", erros);
                } else if !atualizadas.is_empty() {
                    println!("   💾 {} pesos persistidos no DB", atualizadas.len());
                }
            }

            // Promove conexões dormentes com alto uso para ativas
            let dormentes_ids: Vec<Uuid> = grafo.conexoes_dormentes
                .values()
                .filter(|c| c.total_usos > 5 && c.emocao_media > 0.5)
                .map(|c| c.id)
                .collect();
            let promovidas = dormentes_ids.len();
            for id in dormentes_ids {
                grafo.ativar_conexao(id).await;
            }
            println!("   📈 {} conexões dormentes promovidas para ativas", promovidas);
        } else {
            println!("   ⚠️  Grafo sináptico não conectado ao ciclo de sono");
        }
    }

    // ================================================================
    // FASE N2 — Revisão e Poda Contextual
    // Remove sinapses com peso < 0 ou marcador_poda < 0
    // Respeita o contexto semântico (Fantasia/Sonho são poupados)
    // ================================================================
    async fn fase_n2(&mut self) {
        println!("\n🌙 FASE 2: {}", FaseSono::N2.nome());

        if let Some(grafo_arc) = &self.grafo {
            let mut grafo = grafo_arc.lock().await;

            // Decaimento natural do marcador de poda para sinapses inativas
            {
                let agora = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64();

                for c in grafo.grafo_completo.conexoes.values_mut() {
                    let inativo_ha = c.ultimo_uso
                        .map(|u| agora - u)
                        .unwrap_or(agora - c.criada_em);

                    let taxa = match c.contexto_semantico {
                        ContextoSemantico::Fantasia | ContextoSemantico::Sonho => 0.001,
                        ContextoSemantico::Habito => 0.0005,
                        _ => 0.01,
                    };
                    if inativo_ha > 3600.0 {
                        c.marcador_poda -= taxa * (inativo_ha / 3600.0).min(10.0) as f32;
                    }
                }
            }

            // Executa poda — retorna IDs removidos
            let podadas_ids = grafo.podar_sinapses();
            println!("   ✂️  Total podado: {} sinapses (neurônios preservados)", podadas_ids.len());
            println!("   📊 Conexões ativas:    {}", grafo.conexoes_ativas.len());
            println!("   📊 Conexões dormentes: {}", grafo.conexoes_dormentes.len());

            // Propaga a poda para o DB
            if let Some(db) = &self.db {
                match db.delete_conexoes_batch(&podadas_ids).await {
                    Ok(_) if !podadas_ids.is_empty() =>
                        println!("   💾 {} sinapses removidas do DB", podadas_ids.len()),
                    Err(e) =>
                        println!("   ⚠️  Falha ao remover sinapses do DB: {}", e),
                    _ => {}
                }
            }
        } else {
            println!("   ⚠️  Grafo sináptico não conectado ao ciclo de sono");
        }
    }

    // ================================================================
    // FASE N3 — REM (Sonho)
    // Recombinação criativa de conexões esquecidas
    // ================================================================
    async fn fase_n3(&mut self) {
        println!("\n✨ FASE 3: {}", FaseSono::N3.nome());
        println!("   🌟 Selene está sonhando...");

        if let Some(grafo_arc) = &self.grafo {
            let mut grafo = grafo_arc.lock().await;
            let novas_conexoes = grafo.ciclo_rem().await;

            // Persiste as novas conexões criadas pelo REM no DB
            if let Some(db) = &self.db {
                let mut salvas = 0usize;
                let mut erros = 0usize;
                for c in &novas_conexoes {
                    match db.save_conexao(c).await {
                        Ok(_)  => salvas += 1,
                        Err(_) => erros += 1,
                    }
                }
                if salvas > 0 {
                    println!("   💾 {} conexões REM persistidas no DB", salvas);
                }
                if erros > 0 {
                    println!("   ⚠️  {} falhas ao salvar conexões REM", erros);
                }
            }
        } else {
            println!("   ⚠️  Grafo sináptico não conectado ao ciclo REM");
        }

        // REM semântico: replay episódico + atalhos no grafo de palavras
        if let Some(bs_arc) = &self.brain_state {
            if let Ok(mut bs) = bs_arc.try_lock() {
                let (novas, relato) = bs.rem_semantico();
                if novas > 0 { println!("   🧠 REM semântico: {} novas sinapses", novas); }
                if let Some(r) = relato { println!("   💭 Sonho: {}", r); }

                // P4.1 — Replay reverso: reforça caminhos causais de trás para frente.
                // Biologicamente: hippocampus replay é tanto forward quanto reverse.
                // Reverse replay propaga recompensas de volta à causa — aprende causalidade.
                let n_reverso = {
                    let episodios_rev: Vec<_> = bs.historico_episodico
                        .iter()
                        .filter(|ep| ep.emocao.abs() > 0.5)
                        .rev()
                        .take(20)
                        .flat_map(|ep| ep.palavras.iter().rev().cloned())
                        .collect();
                    let mut n_rev = 0usize;
                    // Cria atalhos reversos: se B→A não existe e emocao alta, cria com peso menor
                    for w in episodios_rev.windows(2) {
                        if let [a, b] = w {
                            if a != b && a.len() >= 2 && b.len() >= 2 {
                                // arc reverso (recompensa → causa) com peso reduzido
                                if let Ok(mut sw) = bs.swap_manager.try_lock() {
                                    sw.importar_causal(vec![(b.clone(), a.clone(), 0.05)]);
                                }
                                n_rev += 1;
                            }
                        }
                    }
                    n_rev
                };
                if n_reverso > 0 {
                    println!("   ↩️  Replay reverso: {} atalhos causais revertidos", n_reverso);
                }

                // PatternEngine — extração e consolidação de padrões durante REM
                let t_s = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64();
                let novos_candidatos = bs.pattern_engine.extrair_padroes(t_s);
                let novos_consolidados = bs.pattern_engine.consolidar(t_s);
                if novos_candidatos > 0 || novos_consolidados > 0 {
                    println!("   🔍 PatternEngine: +{} candidatos, +{} consolidados",
                        novos_candidatos, novos_consolidados);
                }
            }
        }
    }

    // ================================================================
    // FASE N4 — Backup para HDD
    // Snapshot do RocksDB → D:/Selene_Backup_RAM/backup_TIMESTAMP/
    // ================================================================
    async fn fase_n4(&mut self) {
        println!("\n💾 FASE 4: {}", FaseSono::N4.nome());

        // Garante que os diretórios de destino existem
        let _ = std::fs::create_dir_all(BACKUP_ROOT);
        let _ = std::fs::create_dir_all(ARCHIVE_PATH);

        match backup_to_hdd(DB_PATH, BACKUP_ROOT).await {
            Ok(dest) => {
                println!("   ✅ Backup concluído: {}", dest.display());

                // Mantém apenas os 5 backups mais recentes para não lotar o HDD
                Self::limpar_backups_antigos(BACKUP_ROOT, 5);
            }
            Err(e) => {
                println!("   ❌ Falha no backup: {} — DB permanece íntegro no NVMe", e);
            }
        }
    }

    /// Remove backups antigos, mantendo apenas os `manter` mais recentes
    fn limpar_backups_antigos(root: &str, manter: usize) {
        let Ok(entradas) = std::fs::read_dir(root) else { return };

        let mut dirs: Vec<_> = entradas
            .flatten()
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .filter(|e| e.file_name().to_string_lossy().starts_with("backup_"))
            .collect();

        // Ordena por nome (que inclui timestamp YYYYMMDD_HHMMSS)
        dirs.sort_by_key(|e| e.file_name());

        // Remove os mais antigos se exceder o limite
        if dirs.len() > manter {
            for old in &dirs[..dirs.len() - manter] {
                if std::fs::remove_dir_all(old.path()).is_ok() {
                    println!("   🗑️  Backup antigo removido: {}", old.file_name().to_string_lossy());
                }
            }
        }
    }

    // ================================================================
    // LOOP PRINCIPAL DO SONO
    // ================================================================
    pub async fn dormir(&mut self, memoria: &mut MemoryTier, config: &Config) {
        println!("\n{}", "=".repeat(60));
        println!("💤 SELENE INICIANDO CICLO DE SONO");
        println!("{}", "=".repeat(60));

        self.fase_n1(memoria).await;
        self.fase_n2().await;
        self.fase_n3().await;
        self.fase_n4().await;

        println!("\n{}", "=".repeat(60));
        println!("🔆 SELENE DESPERTOU!");
        println!("   Energia: {:.1}W | Versão: {}", config.energia_watts, self.versao_sistema);
        println!("{}", "=".repeat(60));
    }

    /// Shutdown gracioso — chamado antes de encerrar o processo.
    /// Faz flush + backup imediato sem passar pelo ciclo completo.
    pub async fn shutdown_gracioso(memoria: &mut MemoryTier) {
        println!("\n🛑 [SHUTDOWN] Iniciando desligamento gracioso da Selene...");

        // Flush final da memória ativa
        memoria.flush_to_l3();
        println!("   ✅ Memória ativa flushed para NVMe");

        // Backup de emergência
        let _ = std::fs::create_dir_all(BACKUP_ROOT);
        match backup_to_hdd(DB_PATH, BACKUP_ROOT).await {
            Ok(dest) => println!("   ✅ Backup de shutdown salvo: {}", dest.display()),
            Err(e)   => println!("   ⚠️  Backup falhou: {} — DB intacto no NVMe", e),
        }

        println!("   🧠 Selene encerrada com segurança. Até logo.");
    }
}
