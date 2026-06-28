"""Gera os 4 corpos da Selene (bebê/5/15/30 anos) do .glb → URDF → PROTO → pós.
Decima cabeça, escala real, olhos, física de bebê. Copia o bebê pro projeto Webots.
Uso: python gerar_corpos.py"""
import subprocess
import glob
import os
import shutil

BLENDER = r"F:/programas/blender/blender.exe"
PY = r"f:/Selene_brain_2.0/venv/Scripts/python.exe"
CONV = r"f:/Selene_brain_2.0/webots/converter_urdf.py"
POS = r"f:/Selene_brain_2.0/webots/pos_proto.py"
BASE = r"f:/Selene_brain_2.0/Avatar_3D"
PROJ = r"F:/projetos/quarto_selene/protos"

# (pasta_glb, dir_saida, nome_urdf, nome_proto, altura_m, copia_pro_projeto)
CORPOS = [
    ("bebe", "urdf_bebe", "selene_bebe", "SeleneBebe", 0.55, True),
    ("5 anos", "urdf_5anos", "selene_5anos", "Selene5anos", 1.10, False),
    ("15 anos", "urdf_15anos", "selene_15anos", "Selene15anos", 1.60, False),
    ("Adulta", "urdf_30anos", "selene_30anos", "Selene30anos", 1.65, False),
]

for pasta, saida, nome, proto, altura, copiar in CORPOS:
    glbs = glob.glob(f"{BASE}/Webots avatares/{pasta}/**/*Character_output.glb",
                     recursive=True)
    if not glbs:
        print(f"[{pasta}] glb NAO encontrado")
        continue
    outdir = f"{BASE}/{saida}"
    print(f"=== {pasta} -> {nome} ({altura} m) ===")
    # 1) URDF (decima + escala + centra)
    r = subprocess.run([BLENDER, "--background", "--python", CONV, "--",
                        glbs[0], outdir, nome, str(altura)],
                       capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if "escala]" in ln or "OK] URDF" in ln or "Error" in ln:
            print("  " + ln.strip())
    # 2) PROTO
    proto_path = f"{outdir}/{proto}.proto"
    subprocess.run([PY, "-m", "urdf2webots.importer", "--input",
                    f"{outdir}/{nome}.urdf", "--output", proto_path],
                   capture_output=True, text=True)
    # 3) pos (paths + olhos + física)
    r3 = subprocess.run([PY, POS, proto_path], capture_output=True, text=True)
    print("  " + r3.stdout.strip())
    # 4) copia o bebê pro projeto Webots
    if copiar:
        shutil.copy(proto_path, f"{PROJ}/{proto}.proto")
        mesh_dst = f"{PROJ}/meshes"
        if os.path.exists(mesh_dst):
            shutil.rmtree(mesh_dst)
        shutil.copytree(f"{outdir}/meshes", mesh_dst)
        print(f"  copiado {proto} + meshes -> projeto")

print("TODOS OS CORPOS PRONTOS")
