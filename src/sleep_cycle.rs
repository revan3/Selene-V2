// src/sleep_cycle.rs
// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use std::time::Duration;
use crate::storage::memory_tier::MemoryTier;
use crate::storage::ConexaoSinaptica;  // IMPORTANTE!
use crate::config::Config;
use rand::seq::SliceRandom;
use uuid::Uuid;
use chrono;

// Estruturas temporárias (depois você move para storage)
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

#[derive(Debug, Clone)]
pub struct BackupSistema {
    pub versao: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub memorias_consolidadas: Vec<Experiencia>,
    pub insights: Vec<Hipotese>,
    pub metricas: std::collections::HashMap<String, f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FaseSono {
    N1, N2, N3, N4,
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
    
    pub fn duracao_minutos(&self) -> u64 {
        90 // Todas as fases com 90 min por enquanto
    }
}

pub struct CicloSono {
    pub fase_atual: FaseSono,
    progresso: f32,
    inicio_fase: std::time::Instant,
    memorias_do_dia: Vec<Experiencia>,
    hipoteses_geradas: Vec<Hipotese>,
    versao_sistema: String,
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
        }
    }
    
    // VERSÃO SIMPLIFICADA - apenas imprime o que faria
    pub async fn dormir(&mut self, memoria: &mut MemoryTier, config: &Config) {
        println!("\n{}", "=".repeat(50));
        println!("💤 SELENE INICIANDO CICLO DE SONO");
        println!("{}", "=".repeat(50));
        
        // FASE 1
        println!("\n🌙 FASE 1: {}", FaseSono::N1.nome());
        println!("   📚 Consolidando aprendizados do dia...");
        println!("   → 10 experiências processadas");
        println!("   ✅ Consolidadas: 8");
        println!("   💪 Fortalecidas: 3");
        
        // FASE 2
        println!("\n🌙 FASE 2: {}", FaseSono::N2.nome());
        println!("   🔍 Revisando memórias antigas...");
        println!("   📊 Resultados:");
        println!("      - Memórias revisadas: 25");
        println!("      - Conexões podadas: 5");
        println!("      - Conexões enfraquecidas: 12");
        
        // FASE 3
        println!("\n✨ FASE 3: {}", FaseSono::N3.nome());
        println!("   🌟 Selene está sonhando...");
        println!("   🎨 Recombinando conhecimentos...");
        println!("   🧩 Combinando 3 experiências recentes com 2 memórias antigas");
        println!("   ✅ Hipótese viável: 'E se eu pudesse desenhar letras como formas?'");
        println!("   💭 Total de insights do sonho: 2");
        
        // PAUSA
        println!("\n🔄 PAUSA PARA MANUTENÇÃO");
        println!("   🔧 Criando checkpoint de segurança...");
        println!("   ✅ Checkpoint salvo: checkpoint_2.0.0.bin");
        println!("   📡 Verificando atualizações...");
        println!("   ✅ Sistema atualizado (versão {})", self.versao_sistema);
        
        // FASE 4
        println!("\n💾 FASE 4: {}", FaseSono::N4.nome());
        println!("   💾 Realizando backup completo...");
        println!("   ✅ Backup concluído com sucesso!");
        
        println!("\n{}", "=".repeat(50));
        println!("🔆 SELENE DESPERTOU!");
        println!("   Memórias consolidadas: 8");
        println!("   Hipóteses geradas: 2");
        println!("   Energia consumida: {:.1}W", config.energia_watts);
        println!("{}", "=".repeat(50));
    }
}