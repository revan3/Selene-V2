// src/storage/checkpoint.rs
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::time::{Duration, Instant};
use std::path::{Path, PathBuf};

const CHECKPOINT_INTERVAL_SECS: u64 = 4 * 60 * 60;
const CHECKPOINT_DIR: &str = "selene_checkpoints";

const ARQUIVOS_ESTADO: &[(&str, &str)] = &[
    ("selene_linguagem.json",   "selene_linguagem.json"),
    ("selene_qtable.bin",       "selene_qtable.bin"),
    ("selene_hippo_ltp.json",   "selene_hippo_ltp.json"),
    ("selene_memoria_ego.json", "selene_memoria_ego.json"),
    ("selene_ego.json",         "selene_ego.json"),
    ("selene_swap_state.json",  "selene_swap_state.json"),
];

pub struct CheckpointSystem {
    last_save:     Instant,
    interval:      Duration,
    n_checkpoints: u32,
}

impl CheckpointSystem {
    pub fn new() -> Self {
        Self {
            last_save:     Instant::now(),
            interval:      Duration::from_secs(CHECKPOINT_INTERVAL_SECS),
            n_checkpoints: 0,
        }
    }

    pub fn com_intervalo_horas(horas: u64) -> Self {
        Self {
            last_save:     Instant::now(),
            interval:      Duration::from_secs(horas * 60 * 60),
            n_checkpoints: 0,
        }
    }

    /// Deve ser chamado frequentemente no loop principal.
    /// Retorna true se um checkpoint foi executado neste tick.
    pub fn tick(&mut self) -> bool {
        if self.last_save.elapsed() < self.interval {
            return false;
        }
        self.executar();
        self.last_save     = Instant::now();
        self.n_checkpoints += 1;
        true
    }

    pub fn forcar(&mut self) {
        self.executar();
        self.last_save     = Instant::now();
        self.n_checkpoints += 1;
    }

    pub fn proximo_em(&self) -> Duration {
        self.interval
            .checked_sub(self.last_save.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    pub fn n_checkpoints(&self) -> u32 {
        self.n_checkpoints
    }

    fn executar(&self) {
        let timestamp = chrono::Utc::now()
            .format("%Y%m%d_%H%M%S")
            .to_string();
        let dest = PathBuf::from(CHECKPOINT_DIR).join(&timestamp);

        if let Err(e) = std::fs::create_dir_all(&dest) {
            log::warn!("[Checkpoint] Não foi possível criar pasta {}: {}", dest.display(), e);
            return;
        }

        let mut copiados = 0u32;
        let mut ausentes  = 0u32;

        for (src_nome, dst_nome) in ARQUIVOS_ESTADO {
            let src = Path::new(src_nome);
            if src.exists() {
                let dst = dest.join(dst_nome);
                match std::fs::copy(src, &dst) {
                    Ok(_)  => copiados += 1,
                    Err(e) => log::warn!("[Checkpoint] Falha ao copiar {}: {}", src_nome, e),
                }
            } else {
                ausentes += 1;
            }
        }

        let manifesto = format!(
            "Selene Brain 2.0 — Checkpoint #{n}\n\
             Timestamp : {ts}\n\
             Arquivos  : {ok} copiados, {miss} ausentes\n",
            n    = self.n_checkpoints + 1,
            ts   = timestamp,
            ok   = copiados,
            miss = ausentes,
        );
        let _ = std::fs::write(dest.join("CHECKPOINT.txt"), &manifesto);

        println!(
            "💾 [Checkpoint #{}] {} arquivos salvos → selene_checkpoints/{}  (próximo em {:.1}h)",
            self.n_checkpoints + 1,
            copiados,
            timestamp,
            CHECKPOINT_INTERVAL_SECS as f32 / 3600.0,
        );
        log::info!("[Checkpoint] #{} → {}", self.n_checkpoints + 1, dest.display());
    }
}
