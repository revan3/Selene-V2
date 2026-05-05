#!/usr/bin/env python3
"""
Script de Teste Rápido — Selene Training (60 segundos)
=======================================================

Para testar o módulo de treinamento sem aguardar 30 minutos de consolidação.
Reduz fase REM para 30 segundos.

Uso: python test_treinamento_rapido.py
"""

import sys
sys.path.insert(0, '.')

from treinar_auditivo_bioinspired import (
    TrainingConfig, DatasetSemantico, MotorAudio, GerenciadorSessao
)

def test_rapido():
    """Teste rápido (60 segundos total)"""
    print("\n🧪 TESTE RÁPIDO — Selene Training (60s)\n")

    # Customizar config para teste rápido
    config_teste = TrainingConfig(
        n_palavras=10,  # Apenas 10 palavras (vs 300)
        n_frases=10,    # Apenas 10 frases (vs 300)
        tempo_consolidacao_s=30,  # 30 segundos (vs 1800s = 30 min)
    )

    # Monkey-patch a config global
    import treinar_auditivo_bioinspired
    treinar_auditivo_bioinspired.config = config_teste

    # 1. Dataset
    dataset = DatasetSemantico()
    dataset.palavras = dataset.palavras[:10]
    dataset.frases = dataset.frases[:10]
    print(f"✅ Dataset reduzido: {len(dataset.palavras)} palavras, {len(dataset.frases)} frases\n")

    # 2. Motor
    motor = MotorAudio()
    print("✅ Motor de áudio inicializado\n")

    # 3. Gerenciador
    gerenciador = GerenciadorSessao(dataset, motor)
    gerenciador.log_file = "test_session_history.csv"

    # 4. Executar
    print("🚀 Iniciando teste rápido...")
    print("   Fase 1 (Léxica): ~10s")
    print("   Fase 2 (Sintática): ~10s")
    print("   Fase 3 (REM): ~30s")
    print("   Total esperado: ~60s\n")

    try:
        gerenciador.executar()
        print("\n✅ Teste completado com sucesso!")
        print(f"   Log salvo em: {gerenciador.log_file}")

        # Mostrar conteúdo do log
        print("\n📊 Preview do log:")
        try:
            with open(gerenciador.log_file, 'r') as f:
                for i, linha in enumerate(f):
                    if i < 5:
                        print(f"   {linha.strip()}")
                    elif i == 5:
                        print("   ...")
                        break
        except:
            pass

    except KeyboardInterrupt:
        print("\n⚠️  Teste interrompido")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_rapido()
