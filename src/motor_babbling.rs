// =============================================================================
// src/motor_babbling.rs — Fase B: controle motor contínuo + aprender o corpo
// =============================================================================
//
// Dá à Selene o controle das N juntas do corpo (Webots) com o aprendizado que
// bebês fazem: MOTOR BABBLING guiado por CURIOSIDADE.
//
//   1. A Selene GERA os comandos de junta (não o controlador) — exploração.
//   2. Aprende um FORWARD MODEL: prevê a mudança de propriocepção de cada comando
//      (= "modelo do próprio corpo"; papel do cerebelo).
//   3. O ERRO de predição vira CURIOSIDADE → recompensa intrínseca: ela explora
//      o que ainda não domina (Oudeyer 2007, Schmidhuber 1991).
//   4. Conforme o modelo melhora (erro cai), o babbling diminui → controle emerge.
// =============================================================================

pub struct MotorBabbling {
    n: usize,                 // número de juntas (definido no 1º env_step)
    comando: Vec<f32>,        // comando atual [-1,1] por junta
    w: Vec<f32>,              // forward model linear achatado [n*n]: dProprio ≈ W·comando
    epsilon: f32,             // intensidade do babbling (decai com o aprendizado)
    pub erro_predicao: f32,   // curiosidade = |real - predito| médio
    prev_proprio: Vec<f32>,
    prev_comando: Vec<f32>,
    rng: u64,
    iniciado: bool,
}

impl Default for MotorBabbling {
    fn default() -> Self {
        Self::new()
    }
}

impl MotorBabbling {
    pub fn new() -> Self {
        Self {
            n: 0,
            comando: Vec::new(),
            w: Vec::new(),
            epsilon: 0.4,
            erro_predicao: 0.0,
            prev_proprio: Vec::new(),
            prev_comando: Vec::new(),
            rng: 0x1234_5678_9ABC_DEF0,
            iniciado: false,
        }
    }

    #[inline]
    fn rand(&mut self) -> f32 {
        // xorshift64* — barato e determinístico, sem dependência externa.
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        ((self.rng.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u64 << 24) as f32
    }

    fn iniciar(&mut self, n: usize) {
        self.n = n;
        self.comando = vec![0.0; n];
        self.w = vec![0.0; n * n];
        self.prev_proprio = vec![0.0; n];
        self.prev_comando = vec![0.0; n];
        self.iniciado = true;
    }

    /// Aprende o modelo do corpo com a propriocepção observada e devolve a
    /// CURIOSIDADE (erro de predição) — usada como recompensa intrínseca.
    pub fn aprender(&mut self, proprio: &[f32]) -> f32 {
        if !self.iniciado || self.n != proprio.len() {
            self.iniciar(proprio.len());
            self.prev_proprio.copy_from_slice(proprio);
            return 0.0;
        }
        let mut erro_total = 0.0;
        for i in 0..self.n {
            let real = proprio[i] - self.prev_proprio[i]; // mudança real da junta i
            let base = i * self.n;
            let pred: f32 = (0..self.n).map(|j| self.w[base + j] * self.prev_comando[j]).sum();
            let erro = real - pred;
            erro_total += erro.abs();
            for j in 0..self.n {
                // gradiente do forward model (aprende qual comando move qual junta)
                self.w[base + j] += 0.02 * erro * self.prev_comando[j];
            }
        }
        self.erro_predicao = erro_total / self.n.max(1) as f32;
        // o babbling encolhe conforme o corpo é dominado (erro baixo → menos exploração)
        self.epsilon = (self.erro_predicao * 4.0 + 0.05).clamp(0.05, 0.5);
        self.prev_proprio.copy_from_slice(proprio);
        self.erro_predicao
    }

    /// Gera o próximo comando de junta (random-walk suave com retorno ao centro).
    /// AGORA é a Selene quem gera o movimento — não mais o controlador.
    pub fn gerar_comando(&mut self) -> Vec<f32> {
        if !self.iniciado {
            return Vec::new();
        }
        for i in 0..self.n {
            let ruido = (self.rand() * 2.0 - 1.0) * self.epsilon;
            self.comando[i] = (self.comando[i] * 0.9 + ruido).clamp(-1.0, 1.0);
        }
        self.prev_comando.copy_from_slice(&self.comando);
        self.comando.clone()
    }
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn aprende_modelo_do_corpo() {
        let mut mb = MotorBabbling::new();
        // 2 juntas; a propriocepção segue o comando (corpo simples).
        let mut proprio = vec![0.0, 0.0];
        mb.aprender(&proprio); // inicia
        let mut erro_inicial = 1.0;
        for k in 0..200 {
            let cmd = mb.gerar_comando();
            // simula um corpo: cada junta se move ~na direção do seu comando
            for i in 0..2 {
                proprio[i] += cmd[i] * 0.1;
            }
            let e = mb.aprender(&proprio);
            if k == 5 {
                erro_inicial = e;
            }
        }
        assert!(mb.erro_predicao <= erro_inicial + 1e-3,
                "erro de predição deve cair ao aprender o corpo (de {erro_inicial} p/ {})",
                mb.erro_predicao);
    }
}
