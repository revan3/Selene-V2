# scripts/selene_exam.py
import asyncio
import websockets
import json

URL = "ws://127.0.0.1:3030/ws"

async def exam():
    async with websockets.connect(URL) as ws:
        print("📝 Iniciando Exame de Lógica e Associação...")
        
        # Casais de palavras para testar a sinapse formada
        desafios = [
            ("gato", "bom"),    # Se ela associou gato a algo positivo
            ("fogo", "errado"), # Se ela associou fogo a perigo/negativo
            ("selene", "inteligente")
        ]

        for p1, p2 in desafios:
            print(f"❓ Desafio: Existe conexão entre '{p1}' e '{p2}'?")
            
            await ws.send(json.dumps({
                "action": "check_connection",
                "pair": [p1, p2]
            }))
            
            resp = await ws.recv()
            data = json.loads(resp)
            
            # Lógica de recompensa: se a conexão existe (> 0.5 de peso)
            if data.get("strength", 0) > 0.5:
                print("🏆 ACERTOU! Enviando Dopamina (+0.5)")
                await ws.send(json.dumps({"action": "reward", "value": 0.5}))
            else:
                print("❌ ERRO! Aplicando Noradrenalina (+0.3) para forçar novo aprendizado.")
                await ws.send(json.dumps({"action": "punish", "value": 0.3}))

if __name__ == "__main__":
    asyncio.run(exam())