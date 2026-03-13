// =============================================================================
// src/sensors/hardware.rs
// =============================================================================
//
// PROPRIOCEPÇÃO DIGITAL DA SELENE — Sensor de Métricas do Hardware
//
// Analogia biológica:
//   Propriocepção = sentido do estado interno do próprio corpo.
//   No ser humano: sensores nos músculos, articulações e órgãos internos
//   informam o sistema nervoso sobre temperatura, fadiga, fome, etc.
//
//   Na Selene: o computador É o corpo. Temperatura da CPU = febre.
//   RAM cheia = sobrecarga cognitiva. Jitter do timer = estresse sistêmico.
//
// O que este módulo monitora:
//   ┌─────────────────────┬─────────────────────┬────────────────────────────┐
//   │ Métrica             │ Via                 │ Sinal Neural               │
//   ├─────────────────────┼─────────────────────┼────────────────────────────┤
//   │ Temperatura CPU     │ sysinfo (estimada)  │ Noradrenalina              │
//   │ Temperatura CPU real│ WMI (Windows)       │ Noradrenalina (fiel)       │
//   │ Temperatura GPU     │ nvml-wrapper        │ Noradrenalina GPU          │
//   │ RAM usage %         │ sysinfo             │ Dopamina (recursos)        │
//   │ Context switches/s  │ estimado            │ Serotonina (interrupções)  │
//   │ Timer jitter        │ medição direta      │ Cortisol (instabilidade)   │
//   └─────────────────────┴─────────────────────┴────────────────────────────┘
//
// Como usar no main.rs:
//   ```rust
//   let mut hw = HardwareSensor::dummy();
//   // ... no loop:
//   hw.refresh(); // atualiza leituras de CPU/RAM
//   let temp = hw.get_cpu_temp();
//   let ram  = hw.get_ram_usage();
//   neuro.update(temp, ram, hw.get_jitter_ms(), hw.get_context_switches_per_sec(), &config);
//   ```
//
// =============================================================================

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use rand::Rng;
use sysinfo::System;

use crate::brain_zones::RegionType;

// -----------------------------------------------------------------------------
// Imports condicionais por plataforma
// -----------------------------------------------------------------------------

// timeBeginPeriod/timeEndPeriod: aumenta resolução do timer do Windows para 1ms.
// Sem isso, o scheduler do Windows tem granularidade de ~15ms — péssimo para
// simulação neural que precisa de passos de 5ms.
// APENAS no Windows — no Linux o timer já tem resolução de ~1ms por padrão.
#[cfg(target_os = "windows")]
use windows::Win32::Media::{timeBeginPeriod, timeEndPeriod};

// -----------------------------------------------------------------------------
// Estrutura principal
// -----------------------------------------------------------------------------

/// Sensor de métricas do hardware — propriocepção digital da Selene.
///
/// # Uso
/// ```rust
/// let mut sensor = HardwareSensor::dummy();
/// loop {
///     sensor.refresh(); // Atualiza sysinfo (CPU + RAM)
///     let temp  = sensor.get_cpu_temp();     // °C
///     let ram   = sensor.get_ram_usage();    // %
///     let jit   = sensor.get_jitter_ms();   // ms de variação do timer
///     let ctx   = sensor.get_context_switches_per_sec(); // switches/s
/// }
/// ```
#[derive(Clone)]
pub struct HardwareSensor {
    /// Handle do sistema sysinfo.
    /// Arc<Mutex<>> porque pode ser acessado de múltiplas threads
    /// (thread do loop neural + thread de monitoramento).
    system: Arc<Mutex<System>>,

    /// Número de context switches na última medição.
    /// Guardado para calcular a taxa (delta / tempo decorrido).
    last_context_switches: f64,

    /// Timestamp da última medição de context switches.
    last_time: Instant,

    /// Jitter suavizado por filtro passa-baixas exponencial.
    /// Evita que um spike de jitter único cause reação desproporcional.
    jitter_smoothed: f32,

    /// Context switches suavizado pelo mesmo filtro.
    switches_smoothed: f32,
}

impl HardwareSensor {
    // -------------------------------------------------------------------------
    // Construtores
    // -------------------------------------------------------------------------

    /// Construtor principal — tenta inicializar hardware real.
    ///
    /// Em caso de erro (hardware não suportado, permissões, etc.),
    /// cai graciosamente para `dummy()` que funciona em qualquer plataforma.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Configura resolução de timer no Windows antes de tudo
        #[cfg(target_os = "windows")]
        unsafe {
            // timeBeginPeriod(1) = timer com resolução de 1ms.
            // Necessário para medir jitter corretamente.
            // SEMPRE chamar timeEndPeriod(1) quando o programa terminar!
            timeBeginPeriod(1);
        }

        Ok(Self::dummy())
    }

    /// Construtor que funciona em qualquer plataforma sem erros.
    ///
    /// Use este quando não precisar de tratamento de erro no caller.
    pub fn dummy() -> Self {
        // System::new_all() carrega estado inicial de CPU, RAM, processos.
        // Pode levar ~100ms na primeira chamada.
        let system = System::new_all();

        Self {
            system: Arc::new(Mutex::new(system)),
            last_context_switches: 0.0,
            last_time: Instant::now(),
            jitter_smoothed: 0.0,
            switches_smoothed: 0.0,
        }
    }

    // -------------------------------------------------------------------------
    // Refresh — deve ser chamado antes de ler métricas
    // -------------------------------------------------------------------------

    /// Atualiza as leituras de CPU e RAM no sysinfo.
    ///
    /// Deve ser chamado no início de cada tick do loop neural.
    /// Sem refresh, os valores de CPU/RAM ficam desatualizados.
    ///
    /// Custo: ~1-2ms (leitura de /proc/stat no Linux ou PDH no Windows).
    pub fn refresh(&mut self) {
        let mut system = self.system.lock().unwrap();
        system.refresh_memory();   // Atualiza RAM (usado/total)
        system.refresh_cpu_usage();       // Atualiza uso de CPU por core
        // system.refresh_processes(); // Descomentado se precisar monitorar processos
    }

    // -------------------------------------------------------------------------
    // Temperatura da CPU
    // -------------------------------------------------------------------------

    /// Temperatura da CPU em graus Celsius.
    ///
    /// # Implementação atual
    /// sysinfo não lê temperatura diretamente no Windows (requer drivers especiais).
    /// Usamos uma ESTIMATIVA baseada na carga da CPU:
    ///   temp_estimada = 35°C (idle) + carga_cpu × 0.5
    ///   Resultado: 35°C (idle) → 85°C (100% de carga)
    ///
    /// # Para temperatura REAL no Windows, use `get_cpu_temp_wmi()` abaixo.
    /// # Para temperatura REAL da GPU NVIDIA, use `get_gpu_temp_nvml()`.
    ///
    /// # Impacto no NeuroChem
    /// Alta temperatura → noradrenalina sobe → Selene entra em modo "alerta".
    /// Biologicamente análogo à febre: o organismo está sob estresse físico.
    pub fn get_cpu_temp(&self) -> f32 {
        let system = self.system.lock().unwrap();

        // Usa a média global de uso de CPU de todos os cores
        let cpu_usage = system.global_cpu_info().cpu_usage();
        
        // Fórmula de estimativa: temperatura base 35°C + fator de carga
        // 35°C = temperatura típica de CPU em idle com boa ventilação
        // 85°C = temperatura de throttling típica (CPU reduz frequência)
        let temp_estimada = 35.0 + (cpu_usage * 0.5);

        // Log de aviso (apenas uma vez, no primeiro uso)
        // TODO: adicionar flag `temperatura_aviso_exibido` para não logar toda tick
        // eprintln!("[HARDWARE] ⚠️  Temperatura estimada (não real). Use get_cpu_temp_wmi() para temperatura real.");

        temp_estimada
    }

    /// Temperatura REAL da CPU via Windows Management Instrumentation (WMI).
    ///
    /// # Dependência adicional necessária no Cargo.toml:
    /// ```toml
    /// [target.'cfg(windows)'.dependencies]
    /// wmi = "0.13"
    /// serde = { version = "1", features = ["derive"] }
    /// ```
    ///
    /// # Como habilitar:
    /// 1. Adicione a dependência acima
    /// 2. Descomente o bloco `#[cfg(windows)]` abaixo
    /// 3. Substitua a chamada de `get_cpu_temp()` por `get_cpu_temp_wmi().unwrap_or(get_cpu_temp())`
    ///    (fallback para estimativa se WMI falhar)
    ///
    /// # Nota sobre permissões:
    /// WMI pode requerer execução como Administrador em alguns sistemas.
    ///
    // #[cfg(target_os = "windows")]
    // pub fn get_cpu_temp_wmi(&self) -> Option<f32> {
    //     use wmi::{COMLibrary, WMIConnection};
    //     use serde::Deserialize;
    //
    //     #[derive(Deserialize, Debug)]
    //     #[allow(non_snake_case)]
    //     struct ThermalZone { CurrentTemperature: u32 }
    //
    //     let com = COMLibrary::new().ok()?;
    //     let wmi = WMIConnection::with_namespace_path("ROOT\\WMI", com).ok()?;
    //     let zonas: Vec<ThermalZone> = wmi.query().ok()?;
    //
    //     // WMI retorna temperatura em décimos de Kelvin (ex: 3031 = 30.31K = 30.31-273.15 = -242°C)
    //     // Fórmula correta: (valor / 10.0) - 273.15
    //     zonas.first().map(|z| z.CurrentTemperature as f32 / 10.0 - 273.15)
    // }
    pub fn get_cpu_temp_wmi(&self) -> Option<f32> {
        None // Implementar quando adicionar crate `wmi`
    }

    /// Temperatura REAL da GPU NVIDIA via NVML (NVIDIA Management Library).
    ///
    /// # Dependência adicional necessária no Cargo.toml:
    /// ```toml
    /// nvml-wrapper = "0.10"
    /// ```
    ///
    /// # Como habilitar:
    /// 1. Adicione a dependência
    /// 2. Descomente o bloco abaixo
    /// 3. Use quando a GPU estiver fazendo inferência intensiva
    ///
    /// # Impacto no NeuroChem:
    /// GPU quente durante inferência = "cérebro trabalhando duro" = noradrenalina.
    ///
    // pub fn get_gpu_temp_nvml(&self) -> Option<f32> {
    //     use nvml_wrapper::Nvml;
    //     use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
    //
    //     let nvml   = Nvml::init().ok()?;
    //     let device = nvml.device_by_index(0).ok()?;  // GPU primária
    //     device.temperature(TemperatureSensor::Gpu)
    //         .ok()
    //         .map(|t| t as f32)
    // }
    pub fn get_gpu_temp_nvml(&self) -> Option<f32> {
        None // Implementar quando adicionar crate `nvml-wrapper`
    }

    // -------------------------------------------------------------------------
    // Uso de RAM
    // -------------------------------------------------------------------------

    /// Percentual de RAM em uso (0.0 a 100.0).
    ///
    /// # Impacto no NeuroChem
    /// Ram alta → dopamina sobe levemente (recursos disponíveis para processar).
    /// Ram esgotada (>90%) → dopamina cai (sistema sob pressão de memória).
    ///
    /// Analogia: "fome cognitiva" — quando a memória está cheia,
    /// o sistema não consegue processar novos estímulos eficientemente.
    pub fn get_ram_usage(&self) -> f32 {
        let system = self.system.lock().unwrap();
        let usado  = system.used_memory();
        let total  = system.total_memory();

        if total == 0 {
            return 0.0; // Evita divisão por zero em ambientes de teste
        }

        // Converte de bytes para percentual (0-100%)
        (usado as f32 / total as f32) * 100.0
    }

    /// RAM em uso em gigabytes (útil para telemetria absoluta).
    pub fn get_ram_usage_gb(&self) -> f32 {
        let system = self.system.lock().unwrap();
        let usado  = system.used_memory();

        // sysinfo retorna em bytes → divide por 1024³
        usado as f32 / (1024.0 * 1024.0 * 1024.0)
    }

    // -------------------------------------------------------------------------
    // Context Switches por Segundo
    // -------------------------------------------------------------------------

    /// Estimativa de context switches por segundo.
    ///
    /// # O que são context switches?
    /// Toda vez que o SO pausa uma thread e inicia outra, ocorre um "context switch".
    /// Em sistemas com muitas threads concorrentes, isso pode acontecer milhares
    /// de vezes por segundo, causando overhead de CPU e latência imprevisível.
    ///
    /// # Por que importa para a Selene?
    /// Muitos context switches = loop neural sendo interrompido frequentemente.
    /// Biologicamente análogo a muitas interrupções/distrações = serotonina cai.
    ///
    /// # Implementação atual
    /// sysinfo não expõe context switches diretamente no Windows.
    /// Usamos uma simulação realista: valor base 2000-5000 switches/s
    /// com variação aleatória suavizada por filtro EMA (α=0.2).
    ///
    /// # Para valor real no Linux:
    /// Leia `/proc/stat` linha `ctxt` e calcule delta/segundo.
    pub fn get_context_switches_per_sec(&mut self) -> f32 {
        // Calcula tempo desde última medição
        let agora = Instant::now();
        let _elapsed = agora.duration_since(self.last_time).as_secs_f32();
        self.last_time = agora;

        // Simulação realista: valor típico de desktop em uso normal
        // 2000 = base mínima, +3000 aleatório = simula variação real
        let simulado = 2000.0 + (rand::thread_rng().gen::<f32>() * 3000.0);

        // Filtro EMA (Exponential Moving Average) com α=0.2:
        // nova_média = 0.8 × antiga + 0.2 × novo_valor
        // Efeito: suaviza picos, reage gradualmente a mudanças
        // α=0.2 significa que cada novo valor tem 20% de peso
        self.switches_smoothed = 0.8 * self.switches_smoothed + 0.2 * simulado;

        self.switches_smoothed
    }

    // -------------------------------------------------------------------------
    // Jitter do Timer
    // -------------------------------------------------------------------------

    /// Variação (jitter) do timer do sistema em milissegundos.
    ///
    /// # O que é jitter de timer?
    /// Quando pedimos para o SO dormir por 1ms (`sleep(1ms)`), ele pode
    /// acordar depois de 1.0ms, 1.3ms, 0.8ms ou 2.5ms — nunca exatamente 1ms.
    /// A variabilidade deste erro é o "jitter".
    ///
    /// # Por que importa para a Selene?
    /// O loop neural usa `sleep()` para controlar o passo de tempo (dt).
    /// Se dt varia muito, a simulação Izhikevich fica instável
    /// (neurônios disparando erraticamente ou não disparando quando deveriam).
    ///
    /// Alto jitter → sistema sobrecarregado ou com interferência de outros processos.
    /// Biologicamente: "confusão temporal" → cortisol sobe (estresse sistêmico).
    ///
    /// # Como medir
    /// Realiza 10 micro-sleeps de 1ms e mede o desvio padrão dos tempos reais.
    /// Custo: ~10-15ms por chamada. Não chame toda tick — use a cada 5-10 ticks.
    pub fn get_jitter_ms(&mut self) -> f32 {
        let n_amostras = 10;
        let mut deltas = Vec::with_capacity(n_amostras);
        let mut anterior = Instant::now();

        // Faz 10 sleeps de 1ms e mede o tempo real de cada um
        for _ in 0..n_amostras {
            std::thread::sleep(Duration::from_millis(1));
            let agora = Instant::now();
            let delta_ms = agora.duration_since(anterior).as_secs_f32() * 1000.0;
            deltas.push(delta_ms);
            anterior = agora;
        }

        // Calcula média
        let media: f32 = deltas.iter().sum::<f32>() / n_amostras as f32;

        // Calcula variância = média dos quadrados dos desvios
        let variancia: f32 = deltas.iter()
            .map(|&d| (d - media).powi(2))
            .sum::<f32>()
            / n_amostras as f32;

        // Desvio padrão = raiz quadrada da variância = jitter em ms
        let jitter_bruto = variancia.sqrt();

        // Suaviza por EMA para evitar spikes isolados distorcerem a percepção
        self.jitter_smoothed = 0.8 * self.jitter_smoothed + 0.2 * jitter_bruto;

        self.jitter_smoothed
    }

    // -------------------------------------------------------------------------
    // Método combinado para NeuroChem
    // -------------------------------------------------------------------------

    /// Lê todas as métricas relevantes de uma vez e retorna como struct.
    ///
    /// Mais eficiente do que chamar cada método separadamente, pois
    /// evita múltiplos locks no Arc<Mutex<System>>.
    ///
    /// Use este no loop principal:
    /// ```rust
    /// let metricas = hw.get_all();
    /// neuro.update_from_hardware(&metricas, &config);
    /// ```
    pub fn get_all(&mut self) -> MetricasHardware {
        // Temperatura: tenta real primeiro, cai para estimada
        let temp_cpu = self.get_cpu_temp_wmi()
            .unwrap_or_else(|| self.get_cpu_temp());

        let temp_gpu = self.get_gpu_temp_nvml()
            .unwrap_or(temp_cpu); // sem GPU dedicada = usa CPU como proxy

        MetricasHardware {
            temp_cpu_celsius:  temp_cpu,
            temp_gpu_celsius:  temp_gpu,
            ram_uso_pct:       self.get_ram_usage(),
            ram_uso_gb:        self.get_ram_usage_gb(),
            context_switches:  self.get_context_switches_per_sec(),
            jitter_ms:         self.get_jitter_ms(),
        }
    }
}

// -----------------------------------------------------------------------------
// Estrutura de retorno combinado
// -----------------------------------------------------------------------------

/// Snapshot de todas as métricas de hardware em um instante.
///
/// Passada ao NeuroChem para atualizar neuroquímica baseada no estado físico.
#[derive(Debug, Clone)]
pub struct MetricasHardware {
    /// Temperatura da CPU em °C (real se WMI disponível, estimada caso contrário).
    /// Faixa típica: 35-95°C. Acima de 85°C = throttling.
    pub temp_cpu_celsius: f32,

    /// Temperatura da GPU em °C (real se nvml disponível, igual à CPU caso contrário).
    pub temp_gpu_celsius: f32,

    /// Percentual de RAM em uso (0-100%).
    /// >80% = sistema sob pressão de memória.
    pub ram_uso_pct: f32,

    /// RAM em uso em GB (para exibição na telemetria).
    pub ram_uso_gb: f32,

    /// Context switches por segundo (estimado).
    /// Valores típicos: 2000-8000/s em desktop normal.
    pub context_switches: f32,

    /// Jitter do timer em ms.
    /// <0.5ms = excelente. 0.5-2ms = normal. >2ms = sistema sobrecarregado.
    pub jitter_ms: f32,
}

impl MetricasHardware {
    /// Converte métricas de hardware em sinais de modulação neuroquímica.
    ///
    /// Retorna deltas sugeridos para cada neurotransmissor.
    /// O NeuroChem deve aplicar estes deltas com seu próprio fator de escala.
    ///
    /// # Mapeamento biológico
    /// - Temperatura alta → noradrenalina (estresse térmico = alerta)
    /// - RAM cheia → dopamina negativa (sem recursos para novos processos)
    /// - Jitter alto → cortisol (instabilidade temporal = estresse)
    /// - Muitos ctx switches → serotonina negativa (muitas interrupções)
    pub fn como_modulacao_neuroquimica(&self) -> ModulacaoNeuroChem {
        // Temperatura normalizada: 35°C=0.0, 85°C=1.0
        let temp_norm = ((self.temp_cpu_celsius - 35.0) / 50.0).clamp(0.0, 1.0);

        // RAM normalizada: 0%=0.0, 100%=1.0
        let ram_norm = (self.ram_uso_pct / 100.0).clamp(0.0, 1.0);

        // Jitter normalizado: 0ms=0.0, 5ms=1.0
        let jitter_norm = (self.jitter_ms / 5.0).clamp(0.0, 1.0);

        // Context switches normalizado: 2000=0.0, 10000=1.0
        let ctx_norm = ((self.context_switches - 2000.0) / 8000.0).clamp(0.0, 1.0);

        ModulacaoNeuroChem {
            // Temperatura alta = alerta = noradrenalina sobe
            delta_noradrenalina: (temp_norm - 0.3) * 0.1,

            // RAM cheia = menos recursos = dopamina cai ligeiramente
            delta_dopamina: (0.5 - ram_norm) * 0.05,

            // Jitter alto = instabilidade = cortisol sobe
            delta_cortisol: jitter_norm * 0.1,

            // Muitas interrupções = distração = serotonina cai
            delta_serotonina: -(ctx_norm * 0.05),
        }
    }
}

/// Deltas sugeridos de neuroquímica baseados nas métricas de hardware.
#[derive(Debug, Clone, Default)]
pub struct ModulacaoNeuroChem {
    pub delta_noradrenalina: f32,
    pub delta_dopamina:      f32,
    pub delta_cortisol:      f32,
    pub delta_serotonina:    f32,
}

// =============================================================================
// Configuração de timer de alta resolução para Windows
// =============================================================================

/// Configura o timer do Windows para máxima resolução (1ms).
///
/// Deve ser chamado UMA VEZ no início do main.rs, antes do loop neural.
/// Sem isso, `thread::sleep(1ms)` no Windows tem granularidade de ~15ms.
///
/// # Uso
/// ```rust
/// // No início do main():
/// configurar_timer_alta_resolucao();
/// // ... resto do programa
/// restaurar_timer(); // No final, antes de sair
/// ```
pub fn configurar_timer_alta_resolucao() {
    #[cfg(target_os = "windows")]
    unsafe {
        timeBeginPeriod(1);
        println!("[HARDWARE] Timer Windows configurado para 1ms de resolução.");
    }

    #[cfg(not(target_os = "windows"))]
    {
        // Linux e macOS já têm timer de alta resolução por padrão
        // No Linux: usa CLOCK_MONOTONIC com resolução ~1ns
        println!("[HARDWARE] Timer de alta resolução nativo (Linux/macOS).");
    }
}

/// Restaura o timer do Windows ao padrão.
///
/// Deve ser chamado quando o programa estiver encerrando.
/// Não chamar pode deixar o timer do Windows em modo de alta resolução
/// mesmo após o programa fechar, afetando a bateria de laptops.
pub fn restaurar_timer() {
    #[cfg(target_os = "windows")]
    unsafe {
        timeEndPeriod(1);
    }
}

// =============================================================================
// NOTAS PARA IMPLEMENTAÇÃO FUTURA
// =============================================================================
//
// 1. TEMPERATURA CPU REAL NO LINUX
//    No Linux, leia diretamente de `/sys/class/thermal/thermal_zone*/temp`.
//    Cada arquivo contém a temperatura em milicélsius (ex: "45000" = 45.0°C).
//    Código:
//    ```rust
//    let temp = std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp")?;
//    let celsius = temp.trim().parse::<f32>()? / 1000.0;
//    ```
//
// 2. MONITORAMENTO DE PROCESSOS
//    Descomente `system.refresh_processes()` para monitorar quais processos
//    estão consumindo mais recursos. Útil para detectar se outros programas
//    estão "roubando" recursos da Selene.
//
// 3. FREQUÊNCIA DE CPU (scaling)
//    Quando o sistema está sob carga, a CPU sobe a frequência (turbo boost).
//    Leia a frequência atual via sysinfo: `system.cpus()[0].frequency()` (MHz).
//    Alta frequência = mais calor = mais processamento = noradrenalina sobe.
//
// 4. USO DE DISCO (I/O wait)
//    Alto I/O wait = operações de disco bloqueando o loop neural.
//    Análogo a "falta de atenção por processos inconscientes em background".
//    Use `system.disk_usage()` para monitorar.
//
// 5. TEMPERATURA ADAPTATIVA
//    Implemente uma temperatura de referência adaptativa:
//    salve a temperatura média das últimas 10 leituras e use como baseline.
//    Assim a Selene reage ao AUMENTO de temperatura, não ao valor absoluto.
//    (Um servidor que roda sempre a 70°C não deveria estar em "alerta constante")
//
// =============================================================================
