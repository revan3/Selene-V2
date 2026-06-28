// =============================================================================
// src/motor_primitives.rs — Primitivas Motoras = o domínio MOTOR dos templates
// =============================================================================
//
// IDEIA (Rodrigo, 2026-06): o cérebro NÃO decide músculo a músculo. Ele SELECIONA
// um movimento já mapeado (pular, agarrar, socar) e os circuitos locais EXECUTAM.
// Isso é o conectoma da Drosophila (controle motor distribuído) + a neurociência
// real (primitivas motoras de Bizzi, CPGs da medula, DMPs de Ijspeert, seleção de
// ação dos gânglios da base).
//
// UNIFICAÇÃO: uma primitiva motora É um `template` do domínio Motor (templates.rs):
//   • PADRÃO persistente (keyframes do movimento) — a "topologia"
//   • SLOTS em branco (amplitude, velocidade) — preenchidos pela situação, efêmeros
//   • EVOLUI com uso validado: Nascente → Desenvolvendo → Consolidado → Automático
// O mesmo mecanismo serve cognição (templates.rs) E motor — como no cérebro real.
//
// FLUXO: gânglios da base → `selecionar()` qual primitiva → `trajetoria()` executa
// localmente (interpola keyframes × slots) → ângulos no tempo para o corpo.
// =============================================================================

#![allow(dead_code)]

use std::collections::HashMap;

/// Pose = ângulos das juntas, normalizados em [-1, 1]. Tamanho = nº de juntas do corpo.
pub type Pose = Vec<f32>;

/// Slots "em branco" de uma primitiva: parâmetros preenchidos pela SITUAÇÃO (efêmeros,
/// como nos templates cognitivos). A estrutura (keyframes) persiste; isto aqui não.
#[derive(Debug, Clone, Copy)]
pub struct Slots {
    pub amplitude: f32,   // escala do movimento (força/extensão): 1.0 = nominal
    pub velocidade: f32,  // ritmo de execução: 2.0 = duas vezes mais rápido
}

impl Default for Slots {
    fn default() -> Self { Slots { amplitude: 1.0, velocidade: 1.0 } }
}

/// Ciclo de vida — espelha `EstadoTemplate`: a habilidade motora amadurece com uso.
/// Automático = "anda sem pensar" (ativa sem esforço, plasticidade mínima).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EstadoMotor {
    Nascente,       // 0–2 usos: desajeitado, muito maleável
    Desenvolvendo,  // 3–19: refinando
    Consolidado,    // 20–99: estável, pode gerar variantes
    Automatico,     // ≥100: sem esforço
    Arquivado,      // força < 0.05: dormente
}

const USOS_DESENVOLVENDO: u32 = 3;
const USOS_CONSOLIDADO: u32 = 20;
const USOS_AUTOMATICO: u32 = 100;
const FORCA_MINIMA: f32 = 0.05;
const PASSOS_POR_KEYFRAME: usize = 8;   // resolução base da interpolação

/// Uma PRIMITIVA MOTORA = template do domínio Motor.
#[derive(Debug, Clone)]
pub struct Primitiva {
    pub nome: String,
    pub keyframes: Vec<Pose>,   // a topologia do movimento (poses no tempo)
    pub usos: u32,
    pub forca: f32,             // qualidade/confiança 0..1, sobe com recompensa
    pub estado: EstadoMotor,
}

impl Primitiva {
    pub fn nova(nome: &str, keyframes: Vec<Pose>) -> Self {
        Primitiva {
            nome: nome.to_string(),
            keyframes,
            usos: 0,
            forca: 0.3,                // começa incerta
            estado: EstadoMotor::Nascente,
        }
    }

    /// ATIVA a primitiva: gera a trajetória executável interpolando os keyframes e
    /// aplicando os slots (amplitude escala a pose; velocidade rareia os passos).
    /// É a EXECUÇÃO LOCAL — o cérebro só escolheu QUAL; aqui vira ângulos no tempo.
    pub fn trajetoria(&self, slots: Slots) -> Vec<Pose> {
        if self.keyframes.len() < 2 {
            return self.keyframes.iter()
                .map(|p| escalar(p, slots.amplitude)).collect();
        }
        let passos = (PASSOS_POR_KEYFRAME as f32 / slots.velocidade.max(0.1))
            .round().max(1.0) as usize;
        let mut traj = Vec::new();
        for par in self.keyframes.windows(2) {
            for s in 0..passos {
                let t = s as f32 / passos as f32;        // interpola par[0]→par[1]
                let pose: Pose = par[0].iter().zip(&par[1])
                    .map(|(a, b)| (a + (b - a) * t) * slots.amplitude)
                    .collect();
                traj.push(pose);
            }
        }
        traj.push(escalar(self.keyframes.last().unwrap(), slots.amplitude));
        traj
    }

    /// REFORÇA: uso validado. A força segue a recompensa (EMA) e o estado amadurece.
    pub fn reforcar(&mut self, recompensa: f32) {
        self.usos += 1;
        let alvo = (0.5 + recompensa * 0.5).clamp(0.0, 1.0);   // reward [-1,1] → [0,1]
        self.forca = self.forca * 0.85 + alvo * 0.15;
        self.estado = if self.forca < FORCA_MINIMA {
            EstadoMotor::Arquivado
        } else if self.usos >= USOS_AUTOMATICO {
            EstadoMotor::Automatico
        } else if self.usos >= USOS_CONSOLIDADO {
            EstadoMotor::Consolidado
        } else if self.usos >= USOS_DESENVOLVENDO {
            EstadoMotor::Desenvolvendo
        } else {
            EstadoMotor::Nascente
        };
    }
}

/// REPERTÓRIO MOTOR: a biblioteca de primitivas de onde o cérebro escolhe.
pub struct RepertorioMotor {
    pub primitivas: HashMap<String, Primitiva>,
    pub n_juntas: usize,
}

impl RepertorioMotor {
    /// Cria o vocabulário motor inicial pré-mapeado (poses simbólicas — o mapeamento
    /// fino para o corpo real, ex. URDF do Webots, é calibrado depois por uso).
    pub fn novo(n_juntas: usize) -> Self {
        let z = || vec![0.0f32; n_juntas];               // repouso
        let u = |v: f32| vec![v; n_juntas];              // pose uniforme
        let mut primitivas = HashMap::new();
        let mut add = |nome: &str, kf: Vec<Pose>| {
            primitivas.insert(nome.to_string(), Primitiva::nova(nome, kf));
        };
        add("repouso", vec![z()]);
        add("agachar", vec![z(), u(-0.5)]);
        add("levantar", vec![u(-0.5), z()]);
        add("pular",   vec![z(), u(-0.6), u(0.8), z()]);
        add("agarrar", vec![z(), u(0.5), u(0.9)]);
        add("socar",   vec![z(), u(-0.3), u(1.0), z()]);
        add("acenar",  vec![z(), u(0.7), u(-0.2), u(0.7), z()]);
        RepertorioMotor { primitivas, n_juntas }
    }

    pub fn adicionar(&mut self, p: Primitiva) {
        self.primitivas.insert(p.nome.clone(), p);
    }

    /// SELEÇÃO (placeholder do gânglio basal): escolhe a primitiva de maior preferência
    /// vinda do cérebro, ponderada pela força (qualidade aprendida). Argmax simples por
    /// ora; quando os gânglios da base assumirem, isto vira a saída deles (Go/NoGo).
    pub fn selecionar(&self, preferencias: &HashMap<String, f32>) -> Option<&Primitiva> {
        self.primitivas.values()
            .filter(|p| p.estado != EstadoMotor::Arquivado)
            .max_by(|a, b| {
                let va = preferencias.get(&a.nome).copied().unwrap_or(0.0) * a.forca;
                let vb = preferencias.get(&b.nome).copied().unwrap_or(0.0) * b.forca;
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// COMPÕE duas primitivas em SEQUÊNCIA (ex. agachar→pular) → nova primitiva.
    /// É a operação "Empilhamento/Combinação" dos templates, no domínio motor.
    pub fn compor(&mut self, a: &str, b: &str, nome_novo: &str) -> bool {
        let (ka, kb) = match (self.primitivas.get(a), self.primitivas.get(b)) {
            (Some(pa), Some(pb)) => (pa.keyframes.clone(), pb.keyframes.clone()),
            _ => return false,
        };
        let mut kf = ka;
        kf.extend(kb);
        self.adicionar(Primitiva::nova(nome_novo, kf));
        true
    }
}

fn escalar(p: &Pose, k: f32) -> Pose {
    p.iter().map(|v| v * k).collect()
}

// ── Testes ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn repertorio_tem_vocabulario_inicial() {
        let r = RepertorioMotor::novo(6);
        assert!(r.primitivas.contains_key("pular"));
        assert!(r.primitivas.contains_key("agarrar"));
        assert_eq!(r.primitivas["pular"].estado, EstadoMotor::Nascente);
    }

    #[test]
    fn ativar_gera_trajetoria_parametrizada() {
        let r = RepertorioMotor::novo(4);
        let traj = r.primitivas["pular"].trajetoria(Slots::default());
        assert!(traj.len() > 4);                       // interpolou os keyframes
        assert!(traj.iter().all(|p| p.len() == 4));    // mantém nº de juntas
        // amplitude 0 = sem movimento (todas as poses zeradas)
        let parado = r.primitivas["pular"]
            .trajetoria(Slots { amplitude: 0.0, velocidade: 1.0 });
        assert!(parado.iter().all(|p| p.iter().all(|&v| v.abs() < 1e-6)));
    }

    #[test]
    fn primitiva_amadurece_com_uso() {
        let mut p = Primitiva::nova("teste", vec![vec![0.0], vec![1.0]]);
        for _ in 0..25 { p.reforcar(1.0); }
        assert_eq!(p.estado, EstadoMotor::Consolidado);
        assert!(p.forca > 0.8);
    }

    #[test]
    fn compor_encadeia_movimentos() {
        let mut r = RepertorioMotor::novo(3);
        let n0 = r.primitivas["agachar"].keyframes.len()
               + r.primitivas["pular"].keyframes.len();
        assert!(r.compor("agachar", "pular", "agachar_pular"));
        assert_eq!(r.primitivas["agachar_pular"].keyframes.len(), n0);
    }

    #[test]
    fn selecao_pondera_preferencia_e_forca() {
        let r = RepertorioMotor::novo(3);
        let mut pref = HashMap::new();
        pref.insert("socar".to_string(), 0.9);
        pref.insert("acenar".to_string(), 0.2);
        assert_eq!(r.selecionar(&pref).unwrap().nome, "socar");
    }
}
