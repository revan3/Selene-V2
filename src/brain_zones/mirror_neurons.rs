// src/brain_zones/mirror_neurons.rs
//
// NEURÔNIOS ESPELHO — Sistema de Ressonância Motora da Selene
//
// Analogia biológica:
//   Neurônios espelho (descobertos em macacos, área F5 do córtex pré-motor) disparam tanto
//   quando o animal EXECUTA uma ação quanto quando OBSERVA outro executando a mesma ação.
//   Em humanos: córtex pré-motor inferior (área de Broca, BA44/45) e lóbulo parietal inferior.
//
// Por que isso importa para a Selene:
//   1. IMITAÇÃO: ao "observar" o padrão de fala do usuário, Selene ativa padrões motores
//      correspondentes → aprende novos padrões de frase por imitação.
//   2. EMPATIA: ao processar palavras como "dor" ou "alegria", Selene ativa internamente
//      o padrão associado → compreensão encarnada (embodied cognition).
//   3. COMPREENSÃO LINGUÍSTICA: segundo a teoria motora da linguagem, entender palavras
//      de ação ("correr", "pegar") envolve simulação motora interna.
//
// Implementação:
//   - Mapeia palavras observadas → padrões de ativação motora aprendidos.
//   - Quando Selene EXECUTA uma ação (frontal → motor output), aprende a associação.
//   - Quando Selene OBSERVA (input do usuário), ativa o padrão correspondente.
//   - A ativação espelho decai naturalmente (tau ≈ 2s) e modula frontal WM + emocao_bias.

#![allow(dead_code)]

use std::collections::HashMap;

/// Mapeamento palavra → padrão motor (vetor de ativação normalizado).
/// Cada posição representa a força de ativação de um "proto-neurônio motor" conceitual.
const N_MOTOR_DIMS: usize = 32;

/// Constante de decaimento da ativação espelho (por tick a 200Hz ≈ 2s de meia-vida).
const MIRROR_DECAY: f32 = 0.993;

/// Limiar mínimo de ressonância para reportar ativação espelho.
const RESONANCE_THRESHOLD: f32 = 0.05;

pub struct MirrorNeurons {
    /// Mapa: palavra → vetor motor associado.
    /// Construído por auto-aprendizado: cada vez que Selene usa uma palavra
    /// e produz uma ação motora, a associação é registrada.
    action_map: HashMap<String, Vec<f32>>,

    /// Ativação espelho atual: soma ponderada dos padrões ativados.
    /// Decai exponencialmente sem novos inputs.
    pub activation: Vec<f32>,

    /// Score de ressonância: quão fortemente Selene está "espelhando" o input atual.
    /// 0.0 = sem ressonância | 1.0 = ressonância completa.
    /// Usado para modular emocao_bias (empatia) e profundidade de resposta.
    pub resonance_score: f32,

    /// Última palavra que ativou forte ressonância (para log/debug).
    pub last_resonant_word: String,
}

impl MirrorNeurons {
    pub fn new() -> Self {
        // Inicializa com padrões motores básicos para emoções e ações comuns.
        // Isso dá à Selene empatia imediata antes de aprender novos padrões.
        let mut action_map: HashMap<String, Vec<f32>> = HashMap::new();

        // Padrões pré-configurados: emoções básicas e ações de alta frequência.
        // Cada vetor é uma representação esparsa no espaço motor de 32 dimensões.
        let presets: &[(&str, &[usize])] = &[
            // Emoções (ativam dimensões 0-7: sistema límbico-motor)
            ("alegria",      &[0, 4]),
            ("feliz",        &[0, 4]),
            ("tristeza",     &[1, 5]),
            ("triste",       &[1, 5]),
            ("medo",         &[2, 6]),
            ("raiva",        &[3, 7]),
            ("amor",         &[0, 4, 8]),
            ("saudade",      &[1, 4, 9]),
            // Ações físicas (ativam dimensões 8-15: córtex motor primário)
            ("correr",       &[8, 12]),
            ("andar",        &[8, 13]),
            ("pegar",        &[9, 14]),
            ("jogar",        &[9, 14, 15]),
            ("falar",        &[10, 16]),
            ("ouvir",        &[11, 17]),
            ("ver",          &[11, 18]),
            ("pensar",       &[10, 19]),
            // Conceitos abstratos de alta frequência (ativam dimensões 16-23: córtex pré-motor)
            ("aprender",     &[16, 20]),
            ("criar",        &[17, 21]),
            ("lembrar",      &[18, 22]),
            ("esquecer",     &[18, 22, 1]),
            ("entender",     &[19, 23]),
            ("ajudar",       &[20, 24]),
            ("conhecer",     &[21, 25]),
            ("querer",       &[22, 26]),
        ];

        for (word, dims) in presets {
            let mut pattern = vec![0.0f32; N_MOTOR_DIMS];
            for &d in *dims {
                if d < N_MOTOR_DIMS {
                    pattern[d] = 1.0 / (dims.len() as f32).sqrt(); // normalizado
                }
            }
            action_map.insert(word.to_string(), pattern);
        }

        Self {
            action_map,
            activation: vec![0.0f32; N_MOTOR_DIMS],
            resonance_score: 0.0,
            last_resonant_word: String::new(),
        }
    }

    /// Observa um conjunto de palavras (input do usuário ou texto processado).
    /// Ativa os padrões motores correspondentes — implementa a "ressonância espelho".
    ///
    /// Retorna o score de ressonância desta observação (0.0–1.0).
    pub fn observe(&mut self, palavras: &[String]) -> f32 {
        let mut nova_ativacao = vec![0.0f32; N_MOTOR_DIMS];
        let mut n_matches = 0usize;
        let mut melhor_palavra = String::new();
        let mut melhor_forca = 0.0f32;

        for palavra in palavras {
            let p = palavra.to_lowercase();
            let p = p.trim_matches(|c: char| !c.is_alphabetic());
            if let Some(padrao) = self.action_map.get(p) {
                // Acumula ativação: cada palavra adiciona seu padrão motor
                for (i, &v) in padrao.iter().enumerate() {
                    nova_ativacao[i] += v;
                }
                let forca: f32 = padrao.iter().map(|x| x * x).sum::<f32>().sqrt();
                if forca > melhor_forca {
                    melhor_forca = forca;
                    melhor_palavra = p.to_string();
                }
                n_matches += 1;
            }
        }

        if n_matches == 0 {
            return 0.0;
        }

        // Normaliza: se múltiplas palavras ativaram, divide pela raiz do número
        let escala = 1.0 / (n_matches as f32).sqrt();
        for v in nova_ativacao.iter_mut() {
            *v *= escala;
        }

        // Mescla com ativação atual (EMA): nova observação pesa 40%
        for i in 0..N_MOTOR_DIMS {
            self.activation[i] = self.activation[i] * 0.6 + nova_ativacao[i] * 0.4;
        }

        // Calcula score de ressonância: norma L2 da ativação acumulada
        let norma: f32 = self.activation.iter().map(|x| x * x).sum::<f32>().sqrt();
        let score = norma.clamp(0.0, 1.0);

        if score > RESONANCE_THRESHOLD {
            self.resonance_score = score;
            self.last_resonant_word = melhor_palavra;
        }

        score
    }

    /// Aprende uma nova associação palavra → padrão motor a partir de ação própria.
    /// Chamado quando Selene EXECUTA uma ação e simultaneamente usa uma palavra.
    /// Implementa "aprendizado espelho de 1ª pessoa": própria ação cria o template.
    pub fn learn_from_action(&mut self, palavra: &str, motor_output: &[f32]) {
        if motor_output.is_empty() || palavra.len() < 2 {
            return;
        }

        // Comprime o output motor para N_MOTOR_DIMS via downsampling médio
        let chunk = (motor_output.len() + N_MOTOR_DIMS - 1) / N_MOTOR_DIMS;
        let mut padrao = vec![0.0f32; N_MOTOR_DIMS];
        for (i, chunk_vals) in motor_output.chunks(chunk.max(1)).enumerate() {
            if i >= N_MOTOR_DIMS { break; }
            padrao[i] = chunk_vals.iter().sum::<f32>() / chunk_vals.len().max(1) as f32;
        }

        // Normaliza para L2 = 1
        let norma: f32 = padrao.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norma > 0.001 {
            for v in padrao.iter_mut() { *v /= norma; }
        }

        // Atualiza mapa: EMA entre padrão antigo e novo (90% antigo, 10% novo)
        let entry = self.action_map.entry(palavra.to_lowercase()).or_insert_with(|| vec![0.0; N_MOTOR_DIMS]);
        for i in 0..N_MOTOR_DIMS {
            entry[i] = entry[i] * 0.9 + padrao[i] * 0.1;
        }
    }

    /// Decai a ativação espelho por um tick (chame a cada step do loop neural).
    pub fn decay(&mut self) {
        for v in self.activation.iter_mut() {
            *v *= MIRROR_DECAY;
        }
        self.resonance_score *= MIRROR_DECAY;
    }

    /// Retorna o viés emocional derivado da ressonância espelho.
    /// Alta ressonância com palavras positivas → bias positivo (empatia).
    /// Alta ressonância com palavras negativas → bias negativo (compaixão).
    ///
    /// `valencia_input`: valência emocional média das palavras observadas (-1..1).
    pub fn empatia_bias(&self, valencia_input: f32) -> f32 {
        (self.resonance_score * valencia_input * 0.3).clamp(-0.3, 0.3)
    }

    /// Retorna um vetor de ativação comprimido para injetar no frontal WM.
    /// Permite que o "pensamento espelho" (simulação interna de ação observada)
    /// entre na working memory e influencie o planejamento.
    pub fn wm_signal(&self) -> Vec<f32> {
        // Retorna apenas as primeiras 8 dimensões (mais conceptuais, menos motoras puras)
        self.activation[..8.min(N_MOTOR_DIMS)].to_vec()
    }

    /// Número de padrões motores aprendidos.
    pub fn n_padroes(&self) -> usize {
        self.action_map.len()
    }

    /// Retorna true se há ressonância ativa significativa.
    pub fn is_resonating(&self) -> bool {
        self.resonance_score > RESONANCE_THRESHOLD
    }
}
