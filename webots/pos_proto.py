"""Pós-processa o PROTO gerado pelo urdf2webots:
  1. corrige paths de mesh (\\ -> /)
  2. remove a física (corpo estático — sem explosão; física correta vem na Fase B)
Uso: python pos_proto.py <caminho_do_proto>"""
import sys
import os


def remove_blocos(texto, marcador):
    out, i, n = [], 0, 0
    while True:
        idx = texto.find(marcador, i)
        if idx == -1:
            out.append(texto[i:])
            break
        out.append(texto[i:idx])
        j = texto.find("{", idx)
        depth, k = 1, j + 1
        while k < len(texto) and depth > 0:
            if texto[k] == "{":
                depth += 1
            elif texto[k] == "}":
                depth -= 1
            k += 1
        i = k
        n += 1
    return "".join(out), n


# 2 olhos (visão estéreo) inseridos no Solid da cabeça. A direção/posição vêm do
# eye_cfg.txt (calculado pelo converter via bone 'headfront' = frente do rosto);
# se não existir, cai num chute.
def montar_olhos(proto_path):
    cfg = os.path.join(os.path.dirname(proto_path), "eye_cfg.txt")
    rot = "1 0 0 1.5708"
    pe, pd = (-0.012, 0.030, 0.040), (0.012, 0.030, 0.040)
    if os.path.exists(cfg):
        ln = open(cfg, encoding="utf-8").read().splitlines()
        a = ln[0].split()
        rot = f"{a[0]} {a[1]} {a[2]} {a[3]}"
        pe = tuple(float(x) for x in ln[1].split())
        pd = tuple(float(x) for x in ln[2].split())

    def cam(nome, p):
        return (f"      Camera {{\n"
                f"        translation {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n"
                f"        rotation {rot}\n"
                f'        name "{nome}"\n'
                f"        width 80\n        height 80\n        fieldOfView 1.2\n"
                f"      }}\n")
    return cam("olho_esq", pe) + cam("olho_dir", pd)


def inserir_olhos(texto, proto_path):
    idx = texto.find("DEF Head Mesh")
    if idx == -1:
        return texto, 0
    j = texto.find("}", idx)          # fecha o Mesh
    j2 = texto.find("}", j + 1)       # fecha o Shape
    nl = texto.find("\n", j2)
    if nl == -1:
        return texto, 0
    return texto[:nl + 1] + montar_olhos(proto_path) + texto[nl + 1:], 2


p = sys.argv[1]
sem_fisica = "--sem-fisica" in sys.argv
s = open(p, encoding="utf-8").read()
np_ = s.count("meshes\\")
s = s.replace("meshes\\", "meshes/")
nf = 0
if sem_fisica:           # física só some se pedido (ex.: se ainda explodir)
    s, nf = remove_blocos(s, "physics Physics {")
s, nolhos = inserir_olhos(s, p)
open(p, "w", encoding="utf-8", newline="").write(s)
print(f"[pos_proto] paths={np_} | fisica removida={nf} | olhos={nolhos}")
