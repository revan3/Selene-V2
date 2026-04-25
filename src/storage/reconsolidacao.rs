// src/storage/reconsolidacao.rs
// Reconsolidação de Memória — Nader, Schafe & LeDoux (2000) / Misanin et al. (1968)
//
// IDEIA CENTRAL:
//   Memórias não são armazenadas uma vez e ficam fixas. Ao serem EVOCADAS,
//   elas entram em um estado de LABILIDADE — ficam instáveis e maleáveis por
//   uma janela de tempo (minutos a horas). Após essa janela, precisam ser
//   RECONSOLIDADAS (re-estabilizadas) para persistir.
//
//   Durante a janela lábil:
//   - Informação CONSISTENTE → reforça e atualiza a memória
//   - Informação CONTRADITÓRIA → enfraquece ou modifica a memória
//   - Ausência de reforço → pequena erosão (esquecimento adaptativo)
//   - Bloqueio da consolidação → memória não reconsolidada → perda
//
// DIFERENÇA DO SONO (consolidação):
//   Sono: transferência hipocampo→córtex de memórias recentes (off-line)
//   Reconsolidação: destabilização por EVOCAÇÃO + janela de modificação (on-line)
//
// RELEVÂNCIA PARA SELENE:
//   - Cada vez que Selene acessa um conceito em resposta a input → fica lábil
//   - Se o usuário confirma → reconsolidação fortalecida
//   - Se o usuário contradiz → memória enfraquecida e/ou substituída
//   - Permite que Selene "reescreva" conceitos com feedback iterativo
//   - Explica como trauma terapêutico funciona: evocação + novo contexto
//
// INTEGRAÇÃO:
//   SwapManager.reconsolidacao: RegistroReconsolidacao
//   Chamado em:
//     - treinar_semantico() → marca pares (a,b) recém-usados como lábeis
//     - aprender_conceito() → acesso ao conceito → labilidade
//     - validar_ultimo_template() → reconsolida com reforço positivo
//   Processado em: liberar_apos_sono() → janelas expiradas reconsolidam

#![allow(dead_code)]

use std::collections::HashMap;
use uuid::Uuid;

// ── Constantes ────────────────────────────────────────────────────────────────

/// Janela de labilidade padrão em segundos (1 hora)
const JANELA_LABIL_S: f64          = 3_600.0;
/// Janela de labilidade para memórias antigas (mais curta)
const JANELA_LABIL_ANTIGA_S: f64   = 900.0;  // 15min
/// Erosão por tick durante janela sem reforço (por segundo)
const EROSAO_SEM_REFORCO: f32      = 0.00005;
/// Ganho de força ao reconsolidar com reforço
const GANHO_RECONSOLIDACAO: f32    = 0.08;
/// Perda de força ao reconsolidar com contradição
const PERDA_CONTRADIÇÃO: f32       = 0.12;
/// Força mínima para uma memória sobreviver a reconsolidação
const FORCA_MINIMA_SOBREVIVENCIA: f32 = 0.05;

// ── Estado de Labilidade ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum EstadoMemoria {
    Estavel,
    Labil,
    Reconsolidando,
    Apagada,
}

/// O tipo de modificação ocorrida durante a janela lábil.
#[derive(Debug, Clone)]
pub enum ModificacaoLabil {
    Reforco { magnitude: f32 },
    Contradicao { magnitude: f32, novo_peso: Option<f32> },
    Atualizacao { novo_contexto: String },
}

/// Uma memória (par de conceitos com peso sináptico) em estado lábil.
#[derive(Debug, Clone)]
pub struct MemoriaLabil {
    pub id:          Uuid,
    /// Conceito de origem no grafo semântico
    pub conceito_a:  String,
    /// Conceito de destino no grafo semântico
    pub conceito_b:  String,
    /// Peso ORIGINAL (antes de qualquer modificação lábil)
    pub peso_original: f32,
    /// Peso ATUAL (pode ser modificado durante janela)
    pub peso_atual:  f32,
    /// Tempo de ativação (início da janela lábil), em segundos de simulação
    pub t_ativacao:  f64,
    /// Duração da janela lábil
    pub janela_s:    f64,
    /// Modificações aplicadas durante a janela
    pub modificacoes: Vec<ModificacaoLabil>,
    pub estado:      EstadoMemoria,
    /// Quantas vezes esta memória já foi reconsolidada
    pub n_reconsolidacoes: u32,
}

impl MemoriaLabil {
    pub fn nova(a: &str, b: &str, peso: f32, t_atual: f64, n_reconsolidacoes: u32) -> Self {
        // Memórias mais antigas (mais reconsolidações) têm janela mais curta
        let janela = if n_reconsolidacoes > 3 {
            JANELA_LABIL_ANTIGA_S
        } else {
            JANELA_LABIL_S
        };
        Self {
            id:               Uuid::new_v4(),
            conceito_a:       a.to_string(),
            conceito_b:       b.to_string(),
            peso_original:    peso,
            peso_atual:       peso,
            t_ativacao:       t_atual,
            janela_s:         janela,
            modificacoes:     Vec::new(),
            estado:           EstadoMemoria::Labil,
            n_reconsolidacoes,
        }
    }

    /// Verifica se a janela lábil ainda está aberta.
    pub fn janela_aberta(&self, t_atual: f64) -> bool {
        t_atual - self.t_ativacao < self.janela_s
            && self.estado == EstadoMemoria::Labil
    }

    /// Aplica erosão passiva (sem reforço durante a janela).
    pub fn erosao_passiva(&mut self, dt_s: f64) {
        if self.estado != EstadoMemoria::Labil { return; }
        self.peso_atual -= EROSAO_SEM_REFORCO * dt_s as f32;
        if self.peso_atual < FORCA_MINIMA_SOBREVIVENCIA {
            self.estado = EstadoMemoria::Apagada;
        }
    }

    /// Aplica um reforço durante a janela lábil.
    pub fn reforcar(&mut self, magnitude: f32) {
        if self.estado != EstadoMemoria::Labil { return; }
        self.peso_atual = (self.peso_atual + magnitude * GANHO_RECONSOLIDACAO).min(1.0);
        self.modificacoes.push(ModificacaoLabil::Reforco { magnitude });
    }

    /// Aplica uma contradição durante a janela lábil.
    /// `novo_peso` = se fornecido, substitui o peso por um valor diferente
    pub fn contradizer(&mut self, magnitude: f32, novo_peso: Option<f32>) {
        if self.estado != EstadoMemoria::Labil { return; }
        if let Some(np) = novo_peso {
            self.peso_atual = np.clamp(0.0, 1.0);
        } else {
            self.peso_atual = (self.peso_atual - magnitude * PERDA_CONTRADIÇÃO).max(0.0);
        }
        self.modificacoes.push(ModificacaoLabil::Contradicao { magnitude, novo_peso });
        if self.peso_atual < FORCA_MINIMA_SOBREVIVENCIA {
            self.estado = EstadoMemoria::Apagada;
        }
    }

    /// Adiciona contexto novo à memória durante a janela.
    pub fn atualizar_contexto(&mut self, contexto: String) {
        if self.estado != EstadoMemoria::Labil { return; }
        self.modificacoes.push(ModificacaoLabil::Atualizacao { novo_contexto: contexto });
    }

    /// Reconsolidação: fecha a janela e estabiliza o peso atual.
    /// Retorna (conceito_a, conceito_b, peso_final, deve_persistir).
    pub fn reconsolidar(&mut self) -> (String, String, f32, bool) {
        self.estado = EstadoMemoria::Reconsolidando;
        self.n_reconsolidacoes += 1;
        let deve_persistir = self.peso_atual > FORCA_MINIMA_SOBREVIVENCIA
            && self.estado != EstadoMemoria::Apagada;
        (
            self.conceito_a.clone(),
            self.conceito_b.clone(),
            self.peso_atual,
            deve_persistir,
        )
    }

    /// Resumo das modificações para debug.
    pub fn resumo_modificacoes(&self) -> String {
        if self.modificacoes.is_empty() {
            return format!("{}→{}: sem modificações", self.conceito_a, self.conceito_b);
        }
        let mods = self.modificacoes.len();
        format!(
            "{}→{}: peso {:.3}→{:.3} ({} modificações)",
            self.conceito_a, self.conceito_b,
            self.peso_original, self.peso_atual, mods
        )
    }
}

// ── Registro de Reconsolidação ────────────────────────────────────────────────

/// Rastreia todas as memórias lábeis ativas e gerencia o processo de reconsolidação.
pub struct RegistroReconsolidacao {
    /// Memórias atualmente em estado lábil, indexadas por (a, b)
    labil: HashMap<(String, String), MemoriaLabil>,
    /// Log de reconsolidações concluídas (últimas 256)
    historico: std::collections::VecDeque<String>,
    /// Tempo da última chamada a processar_janelas
    t_ultimo_processamento: f64,
    /// Total de reconsolidações realizadas
    pub n_total: u64,
    /// Total de memórias apagadas por contradição/erosão
    pub n_apagadas: u64,
}

impl RegistroReconsolidacao {
    pub fn novo() -> Self {
        Self {
            labil:                  HashMap::new(),
            historico:              std::collections::VecDeque::with_capacity(256),
            t_ultimo_processamento: 0.0,
            n_total:                0,
            n_apagadas:             0,
        }
    }

    // ── API principal ─────────────────────────────────────────────────────────

    /// Marca um par (a, b) como lábil ao ser acessado/evocado.
    /// Se já estava lábil, reinicia a janela (re-ativação).
    pub fn ativar(&mut self, a: &str, b: &str, peso_atual: f32, t_atual: f64) {
        let chave = (a.to_string(), b.to_string());
        let n_recons = self.labil.get(&chave)
            .map(|m| m.n_reconsolidacoes)
            .unwrap_or(0);
        self.labil.insert(
            chave,
            MemoriaLabil::nova(a, b, peso_atual, t_atual, n_recons),
        );
    }

    /// Aplica reforço a uma memória lábil (ex: usuário confirma o conceito).
    pub fn reforcar(&mut self, a: &str, b: &str, magnitude: f32, t_atual: f64) {
        let chave = (a.to_string(), b.to_string());
        if let Some(m) = self.labil.get_mut(&chave) {
            if m.janela_aberta(t_atual) {
                m.reforcar(magnitude);
            }
        }
    }

    /// Aplica contradição a uma memória lábil (ex: usuário corrige conceito).
    pub fn contradizer(
        &mut self,
        a: &str,
        b: &str,
        magnitude: f32,
        novo_peso: Option<f32>,
        t_atual: f64,
    ) {
        let chave = (a.to_string(), b.to_string());
        if let Some(m) = self.labil.get_mut(&chave) {
            if m.janela_aberta(t_atual) {
                m.contradizer(magnitude, novo_peso);
            }
        }
    }

    /// Processa todas as janelas lábeis: fecha as expiradas e reconsolida.
    /// Retorna lista de (a, b, peso_final, deve_persistir) para atualização
    /// no grafo semântico.
    pub fn processar_janelas(
        &mut self,
        t_atual: f64,
    ) -> Vec<(String, String, f32, bool)> {
        let dt = t_atual - self.t_ultimo_processamento;
        self.t_ultimo_processamento = t_atual;

        let mut para_reconsolidar: Vec<(String, String)> = Vec::new();
        let mut para_erosao: Vec<(String, String)> = Vec::new();

        for (chave, mem) in &self.labil {
            if !mem.janela_aberta(t_atual) {
                para_reconsolidar.push(chave.clone());
            } else if mem.modificacoes.is_empty() {
                para_erosao.push(chave.clone());
            }
        }

        // Aplica erosão passiva
        for chave in para_erosao {
            if let Some(m) = self.labil.get_mut(&chave) {
                m.erosao_passiva(dt);
            }
        }

        // Reconsolida memórias com janela expirada
        let mut resultados = Vec::new();
        for chave in para_reconsolidar {
            if let Some(mut m) = self.labil.remove(&chave) {
                let resumo = m.resumo_modificacoes();
                let (a, b, peso, persiste) = m.reconsolidar();

                if !persiste { self.n_apagadas += 1; }
                self.n_total += 1;
                if self.historico.len() >= 256 { self.historico.pop_front(); }
                self.historico.push_back(resumo);

                resultados.push((a, b, peso, persiste));
            }
        }
        resultados
    }

    /// Quantas memórias estão em estado lábil agora.
    pub fn n_labeis(&self) -> usize { self.labil.len() }

    /// Verifica se um par está em estado lábil.
    pub fn esta_labil(&self, a: &str, b: &str) -> bool {
        self.labil.contains_key(&(a.to_string(), b.to_string()))
    }

    /// Retorna o peso atual de uma memória lábil (se existir).
    pub fn peso_labil(&self, a: &str, b: &str) -> Option<f32> {
        self.labil.get(&(a.to_string(), b.to_string()))
            .map(|m| m.peso_atual)
    }

    /// Aplica reforço em bloco: todas as memórias lábeis que contêm
    /// qualquer um dos tokens recebem reforço.
    /// Útil quando o usuário valida um turno de conversa inteiro.
    pub fn reforcar_contexto(&mut self, tokens: &[String], magnitude: f32, t_atual: f64) {
        let chaves: Vec<(String, String)> = self.labil.keys()
            .filter(|(a, b)| tokens.contains(a) || tokens.contains(b))
            .cloned()
            .collect();
        for chave in chaves {
            if let Some(m) = self.labil.get_mut(&chave) {
                if m.janela_aberta(t_atual) {
                    m.reforcar(magnitude);
                }
            }
        }
    }

    /// Aplica contradição em bloco: todos os pares lábeis que contradizem
    /// os tokens recebem enfraquecimento.
    pub fn contradizer_contexto(&mut self, tokens: &[String], magnitude: f32, t_atual: f64) {
        let chaves: Vec<(String, String)> = self.labil.keys()
            .filter(|(a, b)| tokens.contains(a) || tokens.contains(b))
            .cloned()
            .collect();
        for chave in chaves {
            if let Some(m) = self.labil.get_mut(&chave) {
                if m.janela_aberta(t_atual) {
                    m.contradizer(magnitude, None);
                }
            }
        }
    }
}

// ── Testes ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn janela_labila_abre_e_fecha() {
        let mut reg = RegistroReconsolidacao::novo();
        reg.ativar("agua", "frio", 0.6, 0.0);
        assert!(reg.esta_labil("agua", "frio"));

        // Simula passagem da janela
        let resultados = reg.processar_janelas(JANELA_LABIL_S + 1.0);
        assert!(!reg.esta_labil("agua", "frio"), "janela deve ter expirado");
        assert_eq!(resultados.len(), 1);
    }

    #[test]
    fn reforco_aumenta_peso() {
        let mut reg = RegistroReconsolidacao::novo();
        reg.ativar("sol", "calor", 0.5, 0.0);
        reg.reforcar("sol", "calor", 1.0, 100.0);
        let peso = reg.peso_labil("sol", "calor").unwrap();
        assert!(peso > 0.5, "reforço deveria aumentar peso: {}", peso);
    }

    #[test]
    fn contradicao_enfraquece_memoria() {
        let mut reg = RegistroReconsolidacao::novo();
        reg.ativar("gelo", "quente", 0.7, 0.0);
        reg.contradizer("gelo", "quente", 1.0, None, 100.0);
        let peso = reg.peso_labil("gelo", "quente").unwrap();
        assert!(peso < 0.7, "contradição deveria enfraquecer: {}", peso);
    }

    #[test]
    fn erosao_passiva_enfraquece() {
        let mut reg = RegistroReconsolidacao::novo();
        reg.ativar("x", "y", 0.5, 0.0);
        // Processa sem nenhum reforço — apenas erosão
        reg.processar_janelas(100.0); // dt = 100s
        // Peso deve ter decaído ou memória ainda lábil
        if let Some(peso) = reg.peso_labil("x", "y") {
            assert!(peso <= 0.5);
        }
        // (memória pode ter reconsolidado se dt >= janela, mas erosão é discreta)
    }

    #[test]
    fn reconsolidacao_retorna_peso_modificado() {
        let mut reg = RegistroReconsolidacao::novo();
        reg.ativar("a", "b", 0.4, 0.0);
        reg.reforcar("a", "b", 1.0, 100.0);
        // Força expiração da janela
        let resultados = reg.processar_janelas(JANELA_LABIL_S + 1.0);
        let (_, _, peso, persiste) = &resultados[0];
        assert!(*persiste, "memória reforçada deve persistir");
        assert!(*peso > 0.4, "peso reconsolidado deve ser maior: {}", peso);
    }

    #[test]
    fn reforco_contexto_em_bloco() {
        let mut reg = RegistroReconsolidacao::novo();
        reg.ativar("fogo", "calor", 0.5, 0.0);
        reg.ativar("fogo", "luz",   0.5, 0.0);
        reg.ativar("agua", "frio",  0.5, 0.0);

        let tokens = vec!["fogo".to_string()];
        reg.reforcar_contexto(&tokens, 1.0, 100.0);

        assert!(reg.peso_labil("fogo", "calor").unwrap() > 0.5);
        assert!(reg.peso_labil("fogo", "luz").unwrap()   > 0.5);
        // agua/frio não foi afetado
        assert_eq!(reg.peso_labil("agua", "frio").unwrap(), 0.5);
    }
}
