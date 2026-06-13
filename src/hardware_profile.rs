// =============================================================================
// src/hardware_profile.rs — V4.6.1
// Perfil de hardware: ajusta a Selene à máquina onde corre, com um único switch.
//
// Seleção:  env SELENE_HW = "avell" | "ideapad" | (ausente → auto-detecção)
// Auto:     ≥8 núcleos lógicos → Avell; senão IdeaPad.
//
// Hoje ajusta: descrição/banner + nº de threads + flag de GPU (reservada p/ Fase 2,
// quando o backend wgpu existir). O cap de RAM continua adaptativo em
// `swap_manager::calcular_cap` (lê sysinfo em tempo real) — este perfil não o
// sobrepõe, só o complementa.
// =============================================================================

use sysinfo::System;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwProfile {
    IdeaPad,
    Avell,
}

#[derive(Debug, Clone)]
pub struct HardwareConfig {
    pub perfil: HwProfile,
    pub nome: &'static str,
    pub nucleos: usize,
    pub ram_total_gb: f64,
    /// Usar backend GPU (wgpu) para o update neural. Reservado para a Fase 2:
    /// hoje sempre `false` até o `GpuBackend` existir. O perfil Avell o ligará.
    pub usar_gpu: bool,
    /// Threads sugeridas para o pool Rayon (0 = deixar o Rayon decidir).
    pub threads_rayon: usize,
    /// Perfil de build recomendado para compilar nesta máquina.
    pub build_profile_recomendado: &'static str,
}

impl HardwareConfig {
    /// Detecta o perfil a partir de `SELENE_HW` ou por heurística de hardware.
    pub fn detectar() -> Self {
        let env = std::env::var("SELENE_HW").unwrap_or_default().to_lowercase();
        let nucleos = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

        let mut sys = System::new();
        sys.refresh_memory();
        let ram_total_gb = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

        let perfil = match env.as_str() {
            "avell"   => HwProfile::Avell,
            "ideapad" => HwProfile::IdeaPad,
            // Auto: o Avell tem 8c/16t Zen4; o IdeaPad 3500U tem 4c/8t.
            _ => if nucleos >= 8 { HwProfile::Avell } else { HwProfile::IdeaPad },
        };

        match perfil {
            HwProfile::Avell => Self {
                perfil,
                nome: "Avell Storm 450 — Ryzen 7 8745HS (Zen4) + RTX 4050 6GB",
                nucleos,
                ram_total_gb,
                // GPU fica desligada até o backend wgpu existir (Fase 2).
                usar_gpu: false,
                threads_rayon: nucleos,
                build_profile_recomendado: "release-avell",
            },
            HwProfile::IdeaPad => Self {
                perfil,
                nome: "IdeaPad S145 — Ryzen 5 3500U (Zen+) / genérico",
                nucleos,
                ram_total_gb,
                usar_gpu: false,
                // Deixa 1 núcleo livre para o SO/áudio na máquina mais fraca.
                threads_rayon: nucleos.saturating_sub(1).max(1),
                build_profile_recomendado: "release-lowmem",
            },
        }
    }

    /// Aplica as configurações que dependem do perfil (hoje: pool Rayon).
    /// Chamar UMA vez no arranque, antes de qualquer trabalho paralelo.
    pub fn aplicar(&self) {
        if self.threads_rayon > 0 {
            // Ignora erro se o pool global já foi inicializado.
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(self.threads_rayon)
                .build_global();
        }
    }

    pub fn banner(&self) -> String {
        format!(
            "🖥️  Perfil de hardware: {}\n   → núcleos: {} | RAM: {:.1} GB | GPU neural: {} | build: --profile {}",
            self.nome,
            self.nucleos,
            self.ram_total_gb,
            if self.usar_gpu { "ON (wgpu)" } else { "OFF (CPU/Rayon)" },
            self.build_profile_recomendado,
        )
    }
}
