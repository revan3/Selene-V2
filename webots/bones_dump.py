"""Dump da orientacao real dos ossos do modelo (rodar no Blender headless).
Diz, pra cada osso-chave: direcao do osso + eixos locais X/Z no mundo.
Com isso descobrimos o EIXO DE FLEXAO anatomico de cada junta (sem chutar).
Uso: blender --background --python bones_dump.py -- <glb>"""
import bpy
import sys

argv = sys.argv[sys.argv.index("--") + 1:]
glb = argv[0]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
mw = arm.matrix_world

# convencao do mundo (ver origins do URDF): X=lateral, Y=frente/tras, Z=vertical
alvos = ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
         "LeftUpLeg", "LeftLeg", "LeftFoot", "Spine", "Spine01",
         "Spine02", "Neck", "Head"]

print("=== ORIENTACAO DOS OSSOS (no mundo) ===")
for nome in alvos:
    b = arm.data.bones.get(nome)
    if b is None:
        print(f"{nome:12s}: NAO ACHADO")
        continue
    head = mw @ b.head_local
    tail = mw @ b.tail_local
    direc = (tail - head).normalized()
    xax = (mw.to_3x3() @ b.x_axis).normalized()
    zax = (mw.to_3x3() @ b.z_axis).normalized()
    print(f"{nome:12s}: dir=({direc.x:+.2f},{direc.y:+.2f},{direc.z:+.2f})"
          f"  Xloc=({xax.x:+.2f},{xax.y:+.2f},{xax.z:+.2f})"
          f"  Zloc=({zax.x:+.2f},{zax.y:+.2f},{zax.z:+.2f})")
