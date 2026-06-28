"""
Controlador Webots — liga o robô (NAO ou corpo URDF da Selene) ao cérebro dela.

VÊ:    câmera(s) → grid 16x16 → occipital  (NAO: 1 câmera; corpo: 2 olhos estéreo)
SENTE: ângulo de cada junta (propriocepção) + sensores de toque (tato)
AGE:   - corpo URDF  → a Selene devolve COMANDOS DE JUNTA (Fase B: motor babbling)
       - NAO         → a Selene devolve uma tecla → toca um .motion pré-gravado

Tudo via o mesmo `env_step` que o jogo 3D usa.

SETUP: pip install websocket-client (no Python do Webots) · LIGUE a Selene antes do Play.
"""
from controller import Robot, Motion, Node
import json
import math
import random

try:
    import websocket  # websocket-client (API síncrona)
except ImportError:
    websocket = None

# ===== CONFIG =====
SELENE_URI = "ws://127.0.0.1:3030/selene"
VW, VH = 80, 80       # resolução da visão (80x80=6400; n_neurons da Selene vai até 8192)
ENVIA_A_CADA = 8
LIMIAR_TOQUE = 0.05
TOQUE_FORTE = 0.5
AMPLITUDE_JUNTA = 0.6        # comando [-1,1] da Selene → ±0.6 rad na junta
MODO_TESTE_ROM = True        # True: testa cada junta isolada (diagnóstico de ROM);
#                              False: a Selene controla o corpo (babbling normal)
SEGUNDOS_POR_JUNTA = 4.0     # tempo varrendo cada junta no modo diagnóstico
MOTION_DIR = "F:/programas/Webots/projects/robots/softbank/nao/motions/"
MOTIONS = {"up": "Forwards.motion", "down": "Backwards.motion",
           "left": "TurnLeft60.motion", "right": "TurnRight60.motion"}


# ===== MOVIMENTOS (só NAO) =====
def carregar_motions():
    carregados = {}
    for tecla, arquivo in MOTIONS.items():
        mot = Motion(MOTION_DIR + arquivo)
        if mot.isValid():
            carregados[tecla] = mot
    return carregados


# ===== SENSORES / MOTORES =====
def descobrir(robot, timestep):
    """Acha motores rotacionais, o position sensor de cada (pareado pelo nome) e os
    sensores de toque. Pareia motor 'j_X' com sensor 'j_X_sensor' → ordem consistente."""
    motores, sensores_jnt, toques = [], [], []
    for i in range(robot.getNumberOfDevices()):
        dev = robot.getDeviceByIndex(i)
        if dev.getNodeType() == Node.TOUCH_SENSOR:
            dev.enable(timestep)
            toques.append(dev)
    for i in range(robot.getNumberOfDevices()):
        dev = robot.getDeviceByIndex(i)
        if dev.getNodeType() == Node.ROTATIONAL_MOTOR:
            motores.append(dev)
            s = robot.getDevice(dev.getName() + "_sensor")
            if s:
                s.enable(timestep)
            sensores_jnt.append(s)
    print(f"[selene] motores: {len(motores)} | toque: {len(toques)} sensores")
    return motores, sensores_jnt, toques


def ler_juntas(sensores):
    """Ângulo de cada junta (ordem dos motores); NaN/None vira 0."""
    out = []
    for s in sensores:
        try:
            v = s.getValue()
            out.append(v if v == v else 0.0)
        except Exception:
            out.append(0.0)
    return out


def ler_toque(sensores):
    total = 0.0
    for s in sensores:
        try:
            if s.getType() == 2:                 # FORCE3D (vetor)
                vx, vy, vz = s.getValues()
                total = max(total, (vx * vx + vy * vy + vz * vz) ** 0.5)
            else:
                total = max(total, s.getValue())
        except Exception:
            pass
    return total


# ===== VISÃO =====
def capturar_grid(cameras):
    """Grid VWxVH de luminância. Com 2 câmeras (olhos), faz a média = visão combinada."""
    if not cameras:
        return [0.0] * (VW * VH)
    soma = [0.0] * (VW * VH)
    for cam in cameras:
        img = cam.getImage()
        cw, ch = cam.getWidth(), cam.getHeight()
        for gy in range(VH):
            sy = int((gy + 0.5) / VH * ch)
            for gx in range(VW):
                sx = int((gx + 0.5) / VW * cw)
                i = (sy * cw + sx) * 4
                b, g, r = img[i], img[i + 1], img[i + 2]
                soma[gy * VW + gx] += (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    n = len(cameras)
    return [v / n for v in soma]


# ===== CONEXÃO =====
def conectar():
    if websocket is None:
        print("[selene] FALTA: pip install websocket-client (no Python do Webots)")
        return None
    try:
        ws = websocket.create_connection(SELENE_URI, timeout=5)
        ws.settimeout(0.2)
        print("[selene] conectado ao cérebro ✅")
        return ws
    except Exception as e:
        print(f"[selene] sem conexão ({e}); a Selene está ligada?")
        return None


def pedir_acao(ws, grid, joints):
    """Manda visão+propriocepção; devolve a resposta motora (motor_action ou motor_joints)."""
    ws.send(json.dumps({"action": "env_step", "reward": 0.0,
                        "grid": grid, "joints": joints, "done": False}))
    for _ in range(20):
        try:
            msg = json.loads(ws.recv())
        except Exception:
            break
        if isinstance(msg, dict) and msg.get("type") in ("motor_action", "motor_joints"):
            return msg
    return None


def avisar_toque(ws, intensidade):
    tipo = "carinho" if intensidade < TOQUE_FORTE else "beliscao"
    ws.send(json.dumps({"action": "touch", "type": tipo, "intensity": min(intensidade, 1.0)}))


def aplicar_comandos(motores, comandos):
    """A Selene comanda cada junta. O comando [-1,1] varre o RANGE REAL da junta
    (-1→limite mín, +1→limite máx, a 90% pra dar folga do batente), em vez de um
    ±0.6 fixo — assim ela usa TODA a amplitude, não só ~38% dela."""
    for i, m in enumerate(motores):
        if i >= len(comandos):
            break
        lo, hi = m.getMinPosition(), m.getMaxPosition()
        if lo < hi:                       # junta com limite → usa o range inteiro
            centro, meia = (lo + hi) / 2.0, (hi - lo) / 2.0
            alvo = max(lo, min(hi, centro + comandos[i] * meia * 0.9))
        else:                             # junta livre (continuous) → ±AMPLITUDE_JUNTA
            alvo = comandos[i] * AMPLITUDE_JUNTA
        try:
            m.setPosition(alvo)
        except Exception:
            pass


# ===== MODO DIAGNÓSTICO (testar ROM junta a junta) =====
def testar_rom(motores, contador, timestep):
    """Move UMA junta por vez (senoide min↔max), as outras em repouso. Mostra
    nome+range pra avaliar o movimento ISOLADO de cada junta (dobra? torce?)."""
    t = contador * timestep / 1000.0
    idx = int(t / SEGUNDOS_POR_JUNTA) % len(motores)
    fase = (t % SEGUNDOS_POR_JUNTA) / SEGUNDOS_POR_JUNTA * 2.0 * math.pi
    for i, m in enumerate(motores):
        lo, hi = m.getMinPosition(), m.getMaxPosition()
        if not (lo < hi):
            continue
        if i == idx:
            centro, meia = (lo + hi) / 2.0, (hi - lo) / 2.0
            m.setPosition(centro + meia * 0.95 * math.sin(fase))
        else:
            m.setPosition(min(hi, max(lo, 0.0)))   # outras em repouso (~0)
    if contador % 16 == 0:
        a = motores[idx]
        print(f"[TESTE-ROM] >>> {a.getName()}  "
              f"range=[{a.getMinPosition():+.2f},{a.getMaxPosition():+.2f}] rad")


# ===== LOOP PRINCIPAL =====
robot = Robot()
timestep = int(robot.getBasicTimeStep())

cameras = []
for _i in range(robot.getNumberOfDevices()):
    _dev = robot.getDeviceByIndex(_i)
    if _dev.getNodeType() == Node.CAMERA:
        _dev.enable(timestep)
        cameras.append(_dev)
print(f"[selene] câmeras: {len(cameras)} {[c.getName() for c in cameras]}")

motores, sensores_jnt, toques = descobrir(robot, timestep)
# Corpo URDF da Selene (juntas "j_*") → controle de junta; senão NAO → .motion.
corpo_urdf = any(m.getName().startswith("j_") for m in motores)
motions = {} if corpo_urdf else carregar_motions()
print(f"[selene] corpo: {'URDF (controle de junta)' if corpo_urdf else 'NAO (.motion)'}")

ws = None
movimento_atual = None
toque_anterior = 0.0
contador = 0

while robot.step(timestep) != -1:
    contador += 1
    if MODO_TESTE_ROM:                     # diagnóstico: move 1 junta por vez (ignora a Selene)
        if motores:
            testar_rom(motores, contador, timestep)
        continue
    if ws is None:
        ws = conectar()
        if ws is None:
            continue
    if contador % ENVIA_A_CADA != 0:
        continue

    # VÊ + SENTE → cérebro → resposta motora
    grid = capturar_grid(cameras)
    angulos = ler_juntas(sensores_jnt)
    try:
        resp = pedir_acao(ws, grid, angulos)
    except Exception as e:
        print(f"[selene] conexão caiu ({e}); reconectando...")
        ws = None
        continue

    # tato: avisa quando muda
    toque = ler_toque(toques)
    if toque > LIMIAR_TOQUE and abs(toque - toque_anterior) > LIMIAR_TOQUE:
        try:
            avisar_toque(ws, toque)
        except Exception:
            ws = None
    toque_anterior = toque

    # AGE conforme a resposta
    if resp and resp.get("type") == "motor_joints":
        # FASE B: a Selene comanda as juntas (controle motor contínuo).
        aplicar_comandos(motores, resp.get("comandos", []))
        if contador % 40 == 0:
            lum = sum(grid) / len(grid)
            var = sum((g - lum) ** 2 for g in grid) / len(grid)
            cur = resp.get("curiosidade", 0.0)
            print(f"[selene] VISAO luz={lum:.3f} var={var:.4f} | curiosidade={cur:.4f} | juntas={len(angulos)}")
    elif resp and resp.get("type") == "motor_action":
        # NAO: tecla → .motion pré-gravado.
        if movimento_atual is not None and movimento_atual.isValid() and not movimento_atual.isOver():
            continue
        mot = motions.get(resp.get("key"))
        if mot is not None and mot.isValid():
            mot.play()
            movimento_atual = mot
