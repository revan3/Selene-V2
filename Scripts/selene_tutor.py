# scripts/selene_tutor.py
import asyncio
import websockets
import json
import time

URL = "ws://127.0.0.1:3030/ws"

async def train():
    try:
        with open("selene_lexicon.json", "r", encoding="utf-8") as f:
            lexico = json.load(f)
        
        async with websockets.connect(URL) as ws:
            print("🎓 Iniciando treinamento da Selene...")
            
            for categoria, palavras in lexico.items():
                # Define a carga química (Valência)
                valence = 1.0 if "positivas" in categoria else (-1.0 if "negativas" in categoria else 0.0)
                
                for palavra in palavras:
                    # Payload compatível com o seu bridge.rs
                    payload = {
                        "action": "learn",
                        "text": palavra,
                        "valence": valence
                    }
                    
                    await ws.send(json.dumps(payload))
                    
                    # Recebe o NeuralStatus (telemetria)
                    response = await ws.recv()
                    status = json.loads(response)
                    
                    # Log de progresso solicitado
                    print(f"📖 Palavra: {palavra.upper()} | Tipo: {categoria}")
                    print(f"   🧬 Neurônios: {status.get('total_neurons')} | 🔗 Sinapses: {status.get('total_synapses')}")
                    print(f"   🧪 Humor: {status.get('humor')}% | Alerta: {status.get('alertness')}")
                    print("-" * 30)
                    
                    # Respeitando o benchmark de 18.4 palavras/s
                    await asyncio.sleep(0.06) 

            print("✅ Ciclo de aprendizado básico finalizado.")

    except FileNotFoundError:
        print("❌ Erro: Gere o léxico primeiro com o script anterior!")
    except Exception as e:
        print(f"❌ Erro na conexão: {e}")

if __name__ == "__main__":
    asyncio.run(train())