# scripts/generate_lexicon.py
import json

def run():
    # Categorias de base para o aprendizado inicial
    # Em um cenário real, você pode expandir estas listas para 1000 palavras cada
    positivas = ["bom", "correto", "vitoria", "inteligente", "feliz", "amigo", "paz", "sim", "sucesso", "alegria"]
    negativas = ["ruim", "errado", "derrota", "burro", "triste", "inimigo", "guerra", "não", "falha", "odio"]
    neutras = ["gato", "fogo", "agua", "casa", "livro", "pedra", "arvore", "computador", "selene", "celular"]

    lexico = {
        "positivas": positivas,
        "negativas: ": negativas,
        "neutras": neutras
    }

    with open("selene_lexicon.json", "w", encoding="utf-8") as f:
        json.dump(lexico, f, ensure_ascii=False, indent=2)
    
    print("✅ Arquivo 'selene_lexicon.json' gerado com sucesso!")

if __name__ == "__main__":
    run()