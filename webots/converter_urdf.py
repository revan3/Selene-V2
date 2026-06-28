"""
converter_urdf.py — converte um modelo riggado (.glb/.fbx) em URDF articulado p/ Webots.

Para cada OSSO da armature gera: 1 link (com a fatia da malha daquele osso) + 1 joint
revolute ligando ao osso-pai. Massas/inércias estimadas pela bounding box do segmento.

⚠️ É a 1ª versão — eixos de junta e limites saem genéricos; rodamos no Webots, vemos o
que move errado e ajustamos. O que JÁ fica certo: estrutura, hierarquia, malhas, massas.

USO (fora do Blender, no terminal):
  blender --background --python converter_urdf.py -- "ENTRADA.glb" "PASTA_SAIDA" [nome]

Ex.:
  blender --background --python converter_urdf.py -- ^
    "Avatar_3D/Webots avatares/bebe/Meshy_AI_.../Meshy_AI_..._Character_output.glb" ^
    "Avatar_3D/urdf_bebe" selene_bebe

Gera: PASTA_SAIDA/<nome>.urdf  +  PASTA_SAIDA/meshes/<osso>.stl
"""
import bpy, bmesh, os, sys, math
from mathutils import Vector, Matrix

# ===== ARGUMENTOS =====
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
INPUT = argv[0] if len(argv) > 0 else ""
OUTDIR = argv[1] if len(argv) > 1 else "urdf_out"
NOME = argv[2] if len(argv) > 2 else "robo"
ALTURA = float(argv[3]) if len(argv) > 3 else 1.0   # altura-alvo em metros (escala real)
DENSIDADE = 985.0          # kg/m³ (~densidade do corpo humano)
LIMITE_TRIS = 8000         # decima malhas acima disto (a cabeça vem com ~35k → trava o Webots)
LIM = math.radians(90)     # limite genérico de cada junta (±90°); ajustar depois

# Eixo + limites ANATÔMICOS por junta (rad), medidos via bones_dump.py.
# Frame do mundo: X=lateral, Y=frente/trás, Z=vertical (frente = -Y). Como o joint
# usa rpy=0, o <axis> está no mundo. Pernas/coluna fletem em X; ombro em Y; cotovelo
# e punho em Z. A ORDEM importa: chaves específicas antes das genéricas.
JUNTAS_ANAT = [
    ("ForeArm",  "0 0 1", -2.40,  0.10),   # cotovelo: dobra só num sentido (~135°)
    ("UpLeg",    "1 0 0", -2.00,  0.60),   # quadril: flexão ~115° / extensão ~35°
    ("ToeBase",  "1 0 0", -0.30,  0.30),   # dedos do pé
    ("Shoulder", "0 1 0", -0.30,  0.30),   # clavícula: pouco
    ("Foot",     "1 0 0", -0.70,  0.70),   # tornozelo: dorsi/plantar
    ("Hand",     "0 0 1", -0.80,  0.80),   # punho
    ("Arm",      "0 1 0", -0.30,  1.60),   # ombro: abaixa o braço (T → lado do corpo)
    ("Leg",      "1 0 0", -0.10,  2.30),   # joelho: dobra só num sentido (~135°)
    ("Spine",    "1 0 0", -0.40,  0.50),   # tronco: flexão moderada
    ("Neck",     "1 0 0", -0.60,  0.60),   # pescoço
    ("Head",     "1 0 0", -0.50,  0.50),   # cabeça
]


def anat_junta(nome):
    """Devolve (eixo, lower, upper) anatômico pelo nome do osso; fallback flexão X."""
    for chave, eixo, lo, hi in JUNTAS_ANAT:
        if chave in nome:
            return eixo, lo, hi
    return "1 0 0", -1.50, 1.50


def perpendicular_ao_osso(eixo_str, bone, mw):
    """Remove do eixo a componente PARALELA ao osso → a junta DOBRA, nunca TORCE
    (rotação axial = saca-rolha). Mantém o plano de dobra; só tira o componente de
    torção (ombro tinha ~20%, quadril ~11%). Devolve o eixo unitário corrigido."""
    ev = Vector([float(x) for x in eixo_str.split()])
    d = mw.to_3x3() @ (bone.tail_local - bone.head_local)
    if d.length < 1e-6:
        return eixo_str
    d.normalize()
    ev = ev - ev.dot(d) * d                 # projeta perpendicular ao osso
    if ev.length < 1e-4:                     # eixo era ~paralelo → mantém o original
        return eixo_str
    ev.normalize()
    return f"{ev.x:.4f} {ev.y:.4f} {ev.z:.4f}"


def eixo_dobra(b, eixo_anat_str, mw):
    """Eixo de flexão = NORMAL do plano (osso-pai, osso-filho) = a 'dobradiça' real
    medida da geometria. Garante dobrar no plano natural (sem torcer), respeitando a
    orientação individual de cada osso. Alinha o sentido ao eixo anatômico (preserva
    lower/upper). Fallback p/ ossos colineares ou raiz: anatômico ⊥ ao osso."""
    R = mw.to_3x3()
    d_f = R @ (b.tail_local - b.head_local)
    ev_anat = Vector([float(x) for x in eixo_anat_str.split()])
    if b.parent is not None and d_f.length > 1e-6:
        d_p = R @ (b.parent.tail_local - b.parent.head_local)
        if d_p.length > 1e-6:
            cruz = d_p.normalized().cross(d_f.normalized())
            if cruz.length > 0.15:                 # ossos não-colineares → dobradiça boa
                cruz.normalize()
                if cruz.dot(ev_anat) < 0:          # alinha o sentido ao anatômico
                    cruz = -cruz
                return f"{cruz.x:.4f} {cruz.y:.4f} {cruz.z:.4f}"
    return perpendicular_ao_osso(eixo_anat_str, b, mw)   # fallback

MESHDIR = os.path.join(OUTDIR, "meshes")
os.makedirs(MESHDIR, exist_ok=True)


# ===== 1. CENA LIMPA + IMPORTA =====
def importar(caminho):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = os.path.splitext(caminho)[1].lower()
    if ext == ".glb" or ext == ".gltf":
        bpy.ops.import_scene.gltf(filepath=caminho)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=caminho)
    else:
        raise SystemExit(f"formato nao suportado: {ext} (use .glb ou .fbx)")
    arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    mesh = next((o for o in bpy.data.objects if o.type == "MESH"), None)
    if not arm or not mesh:
        raise SystemExit("nao achei armature+mesh no arquivo")
    return arm, mesh


# ===== 2. APLICA TRANSFORMAÇÕES (escala/rotação reais) =====
def aplicar_transformacoes(arm, mesh):
    for ob in (arm, mesh):
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        ob.select_set(False)


# ===== 2b. NORMALIZA A ESCALA PARA A ALTURA REAL (modelo vem gigante) =====
def normalizar_escala(arm, mesh, altura_alvo):
    """Escala armature+mesh (a partir da origem) p/ a maior dimensão = altura_alvo."""
    coords = [mesh.matrix_world @ v.co for v in mesh.data.vertices]
    dim = max(
        max(c.z for c in coords) - min(c.z for c in coords),
        max(c.y for c in coords) - min(c.y for c in coords),
        max(c.x for c in coords) - min(c.x for c in coords),
    )
    if dim < 1e-6:
        return
    f = altura_alvo / dim
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    bpy.context.scene.tool_settings.transform_pivot_point = "CURSOR"
    bpy.ops.transform.resize(value=(f, f, f))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.select_all(action="DESELECT")
    print(f"[escala] dim atual={dim:.3f}m -> alvo={altura_alvo:.2f}m (fator {f:.4f})")


# ===== 3. OSSO DOMINANTE DE CADA VÉRTICE (maior peso) =====
def vertices_por_osso(mesh):
    """Devolve {nome_osso: [indices de vertices]}. Cada vértice entra no osso dominante
    E em qualquer osso com peso >= 50% do dominante → OVERLAP nas juntas (preenche os
    gaps entre as partes, senão ficam lacunas visíveis nas articulações)."""
    grupos = {g.index: g.name for g in mesh.vertex_groups}
    por_osso = {}
    for v in mesh.data.vertices:
        pesos = [(g.weight, grupos[g.group]) for g in v.groups if g.group in grupos]
        if not pesos:
            continue
        melhor_peso = max(p for p, _ in pesos)
        for p, nome in pesos:
            if p >= melhor_peso * 0.5:
                por_osso.setdefault(nome, []).append(v.index)
    return por_osso


# ===== 4. FATIA A MALHA POR OSSO E EXPORTA STL =====
def exportar_fatia(arm, mesh, bone, indices):
    """Duplica a malha, apaga os vértices que NÃO são do osso, CENTRA no head do osso
    e salva STL. Devolve (caminho_stl, tamanho) ou None se vazia."""
    bpy.ops.object.select_all(action="DESELECT")
    dup = mesh.copy(); dup.data = mesh.data.copy()
    bpy.context.collection.objects.link(dup)
    manter = set(indices)
    bm = bmesh.new(); bm.from_mesh(dup.data); bm.verts.ensure_lookup_table()
    apagar = [v for v in bm.verts if v.index not in manter]
    bmesh.ops.delete(bm, geom=apagar, context="VERTS")
    if len(bm.verts) == 0:
        bm.free(); bpy.data.objects.remove(dup, do_unlink=True); return None
    bm.to_mesh(dup.data); bm.free()
    # CENTRA o mesh na ORIGEM DO JOINT (head do osso). Sem isso os vértices ficam em
    # coords globais e, como o URDF já SOMA a posição do link, o corpo "explode"
    # (cada peça afastada do centro pela própria distância).
    head_g = arm.matrix_world @ bone.head_local
    mw = dup.matrix_world.copy()
    for v in dup.data.vertices:
        v.co = (mw @ v.co) - head_g
    dup.parent = None
    dup.matrix_basis = Matrix.Identity(4)
    # decima malhas muito densas (a cabeça vem ~35k tris → trava física e sombras no Webots)
    if len(dup.data.polygons) > LIMITE_TRIS:
        bpy.context.view_layer.objects.active = dup
        dec = dup.modifiers.new(name="dec", type="DECIMATE")
        dec.ratio = LIMITE_TRIS / max(len(dup.data.polygons), 1)
        bpy.ops.object.modifier_apply(modifier="dec")
    # bounding box (já relativa ao head) p/ inércia
    coords = [v.co for v in dup.data.vertices]
    mn = Vector((min(c.x for c in coords), min(c.y for c in coords), min(c.z for c in coords)))
    mx = Vector((max(c.x for c in coords), max(c.y for c in coords), max(c.z for c in coords)))
    tam = Vector((max(mx.x - mn.x, 1e-3), max(mx.y - mn.y, 1e-3), max(mx.z - mn.z, 1e-3)))
    centro = (mn + mx) / 2.0   # centro do segmento (rel. ao head) p/ a caixa de colisão
    # exporta STL
    bpy.ops.object.select_all(action="DESELECT")
    dup.select_set(True); bpy.context.view_layer.objects.active = dup
    caminho = os.path.join(MESHDIR, f"{bone.name}.stl")
    # Blender 4.x usa wm.stl_export; versões antigas usam export_mesh.stl.
    try:
        bpy.ops.wm.stl_export(filepath=caminho, export_selected_objects=True)
    except (AttributeError, RuntimeError, TypeError):
        bpy.ops.export_mesh.stl(filepath=caminho, use_selection=True)
    bpy.data.objects.remove(dup, do_unlink=True)
    return caminho, centro, tam


# ===== 5. INÉRCIA DE CAIXA SÓLIDA =====
def inercia_caixa(massa, tam):
    x, y, z = tam
    ixx = massa * (y * y + z * z) / 12.0
    iyy = massa * (x * x + z * z) / 12.0
    izz = massa * (x * x + y * y) / 12.0
    return ixx, iyy, izz


# ===== 6. MONTA O URDF =====
def gerar_urdf(arm, mesh):
    bones = arm.data.bones
    por_osso = vertices_por_osso(mesh)
    linhas = [f'<?xml version="1.0"?>', f'<robot name="{NOME}">']

    for b in bones:
        info = exportar_fatia(arm, mesh, b, por_osso.get(b.name, []))
        if info is None:
            # osso sem malha própria (ex.: ponta) → link minúsculo só p/ a cadeia
            caminho, centro, tam = None, Vector((0, 0, 0)), Vector((0.02, 0.02, 0.02))
        else:
            caminho, centro, tam = info
        vol = max(tam.x * tam.y * tam.z, 1e-6)
        massa = max(vol * DENSIDADE, 0.01)
        ixx, iyy, izz = inercia_caixa(massa, tam)

        linhas.append(f'  <link name="{b.name}">')
        linhas.append(f'    <inertial>')
        linhas.append(f'      <mass value="{massa:.4f}"/>')
        linhas.append(f'      <inertia ixx="{ixx:.5f}" ixy="0" ixz="0" iyy="{iyy:.5f}" iyz="0" izz="{izz:.5f}"/>')
        linhas.append(f'    </inertial>')
        if caminho:
            rel = f"meshes/{b.name}.stl"
            # visual = mesh detalhado (aparência)
            linhas.append(f'    <visual>')
            linhas.append(f'      <geometry><mesh filename="{rel}"/></geometry>')
            linhas.append(f'    </visual>')
            # colisão = CAIXA simples (mesh detalhado trava a física do Webots)
            linhas.append(f'    <collision>')
            linhas.append(f'      <origin xyz="{centro.x:.4f} {centro.y:.4f} {centro.z:.4f}" rpy="0 0 0"/>')
            linhas.append(f'      <geometry><box size="{tam.x:.4f} {tam.y:.4f} {tam.z:.4f}"/></geometry>')
            linhas.append(f'    </collision>')
        else:
            # osso-ponta (sem malha): caixinha de colisão → o Webots calcula a inércia
            # (senão dá "Undefined inertia matrix: using the identity matrix").
            linhas.append(f'    <collision>')
            linhas.append(f'      <geometry><box size="0.02 0.02 0.02"/></geometry>')
            linhas.append(f'    </collision>')
        linhas.append(f'  </link>')

    # joints: cada osso com pai vira um revolute (origem = head do osso - head do pai)
    for b in bones:
        if b.parent is None:
            continue
        origem = b.head_local - b.parent.head_local
        eixo, lo, hi = anat_junta(b.name)
        eixo = eixo_dobra(b, eixo, arm.matrix_world)
        linhas.append(f'  <joint name="j_{b.name}" type="revolute">')
        linhas.append(f'    <parent link="{b.parent.name}"/>')
        linhas.append(f'    <child link="{b.name}"/>')
        linhas.append(f'    <origin xyz="{origem.x:.4f} {origem.y:.4f} {origem.z:.4f}" rpy="0 0 0"/>')
        linhas.append(f'    <axis xyz="{eixo}"/>')
        linhas.append(f'    <limit lower="{lo:.3f}" upper="{hi:.3f}" effort="30" velocity="3"/>')
        linhas.append(f'  </joint>')

    linhas.append("</robot>")
    return "\n".join(linhas)


# ===== 7. OLHOS: direção/posição via bone "headfront" (frente do rosto) =====
def calcular_olhos(arm):
    """Usa o bone 'headfront' (nariz/chifres) como referência da FRENTE do rosto p/
    orientar as câmeras. Frame da câmera = head do bone Head, orientação do MUNDO
    (joints usam rpy=0). Grava eye_cfg.txt: rotação + posição de cada olho."""
    bones = arm.data.bones
    head = bones.get("Head")
    front = bones.get("headfront")
    if head is None or front is None:
        print("[olhos] bones Head/headfront nao achados — usando chute do pos_proto")
        return
    mw = arm.matrix_world
    head_c = mw @ head.head_local
    front_c = mw @ ((front.head_local + front.tail_local) / 2.0)
    frente = front_c - head_c
    if frente.length < 1e-6:
        return
    frente.normalize()
    # eixo óptico da câmera Webots = -Z; gira p/ apontar na 'frente' do rosto
    quat = Vector((0.0, 0.0, -1.0)).rotation_difference(frente)
    ax, ang = quat.to_axis_angle()
    # up = direção do bone da cabeça; lateral = frente × up (separa os 2 olhos)
    up = mw.to_3x3() @ (head.tail_local - head.head_local)
    up = up.normalized() if up.length > 1e-6 else Vector((0.0, 0.0, 1.0))
    lateral = frente.cross(up).normalized()
    # posições proporcionais à ALTURA (a magnitude dos bones do rig não é confiável —
    # o headfront pode estar a metros de distância). Usamos só a DIREÇÃO deles.
    centro = frente * (ALTURA * 0.05) + up * (ALTURA * 0.04)   # à frente e na altura dos olhos
    d_lado = ALTURA * 0.015                                    # meia distância interpupilar
    pe, pd = centro + lateral * d_lado, centro - lateral * d_lado
    with open(os.path.join(OUTDIR, "eye_cfg.txt"), "w", encoding="utf-8") as f:
        f.write(f"{ax.x:.5f} {ax.y:.5f} {ax.z:.5f} {ang:.5f}\n")
        f.write(f"{pe.x:.5f} {pe.y:.5f} {pe.z:.5f}\n")
        f.write(f"{pd.x:.5f} {pd.y:.5f} {pd.z:.5f}\n")
    print(f"[olhos] direcao via headfront: eixo=({ax.x:.2f},{ax.y:.2f},{ax.z:.2f}) ang={ang:.2f}")


# ===== MAIN =====
if not INPUT:
    raise SystemExit("faltou o arquivo de entrada. Veja o cabecalho do script.")
arm, mesh = importar(INPUT)
aplicar_transformacoes(arm, mesh)
normalizar_escala(arm, mesh, ALTURA)
urdf = gerar_urdf(arm, mesh)
calcular_olhos(arm)
saida = os.path.join(OUTDIR, f"{NOME}.urdf")
with open(saida, "w", encoding="utf-8") as f:
    f.write(urdf)
print(f"[OK] URDF gerado: {saida}")
print(f"[OK] malhas em:   {MESHDIR}")
print(f"[!] eixos/limites sao genericos — rode no Webots e me diga o que move errado.")
