import json, sys
sys.stdout.reconfigure(encoding='utf-8')

# ── Carrega vocabulário e associações ──────────────────
with open('selene_linguagem.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
lang  = data['selene_linguagem_v1']
vocab = lang['vocabulario']
assoc = lang.get('associacoes', {})

# ── Carrega categorias editáveis ────────────────────────
try:
    with open('memoria_categorias.json', 'r', encoding='utf-8') as f:
        cats = json.load(f)
except FileNotFoundError:
    cats = {}

cat_map = {}
for grupo, palavras in cats.items():
    if grupo.startswith('_'): continue
    for p in palavras:
        cat_map[p.lower()] = grupo

# ── Constrói grafo ─────────────────────────────────────
# Seeds: palavras das categorias + top vocabulário por peso
seed = set(cat_map.keys())
top_vocab = set(w for w,_ in sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:80])
seed |= top_vocab

nodes_set = set(seed)
edges = []
for word in list(seed):
    if word in assoc:
        for neighbor, weight in assoc[word]:
            if weight >= 0.45:
                nodes_set.add(neighbor)
                edges.append({'source': word, 'target': neighbor, 'weight': round(weight, 3)})

# Arestas internas
seen = set()
for word in list(nodes_set):
    if word in assoc:
        for neighbor, weight in assoc[word]:
            if neighbor in nodes_set and weight >= 0.45:
                key = tuple(sorted([word, neighbor]))
                if key not in seen:
                    seen.add(key)
                    edges.append({'source': word, 'target': neighbor, 'weight': round(weight, 3)})

def get_group(w):
    if w in cat_map:      return cat_map[w]
    if vocab.get(w, 0) > 0.012: return 'core'
    return 'assoc'

node_list = [{'id': n, 'weight': round(vocab.get(n, 0.005), 4), 'group': get_group(n)}
             for n in nodes_set]

# ── Salva grafo JSON ────────────────────────────────────
graph = {'nodes': node_list, 'links': edges}
with open('grafo_selene.json', 'w', encoding='utf-8') as f:
    json.dump(graph, f, ensure_ascii=False, indent=2)

# ── Gera HTML com D3 e dados embutidos ─────────────────
with open('d3.v7.min.js', 'r', encoding='utf-8') as f:
    d3_code = f.read()
graph_json = json.dumps(graph, ensure_ascii=False)
TEMPLATE = open('grafo_template.html', 'r', encoding='utf-8').read()
html = TEMPLATE.replace('__D3_CODE__', d3_code).replace('__GRAPH_DATA__', graph_json)
with open('grafo_selene.html', 'w', encoding='utf-8') as f:
    f.write(html)

grupos = {}
for n in node_list:
    g = n['group']
    grupos[g] = grupos.get(g, 0) + 1

print(f'Nós: {len(node_list)}  Arestas: {len(edges)}')
print('Grupos:', {k: v for k, v in sorted(grupos.items())})
print(f'HTML: {len(html)//1024} KB')
