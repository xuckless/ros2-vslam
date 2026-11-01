"""
Offline tests for COSC_4366EL_A3_G5.py — ROS 2 and YOLO are stubbed out.

Checks the floor projection, the band line tracker on synthetic tape masks,
lane gating at a + junction, and closed-loop junction scenarios (A–F) with
detection dropouts, position noise and LEFT/RIGHT label flicker.

    python tests/sim_junctions.py            # needs numpy + opencv-python
"""
import os, sys, types, math, random, importlib.util
import numpy as np, cv2

# ---------- stubs ----------
class _Log:
    def info(self, m): LOG.append(m)
    def warn(self, m): LOG.append("WARN " + m)
    def error(self, m): LOG.append("ERR " + m)
    def debug(self, m): pass
LOG = []
SIM_T = [0.0]
class _Clock:
    def now(self): return types.SimpleNamespace(nanoseconds=int(SIM_T[0] * 1e9))
class Node:
    def __init__(self, name): pass
    def get_logger(self): return _Log()
    def get_clock(self): return _Clock()
    def create_subscription(self, *a, **k): pass
    def create_publisher(self, *a, **k): return types.SimpleNamespace(publish=lambda t: None)
    def create_timer(self, *a, **k): pass
    def destroy_node(self): pass
class Twist:
    def __init__(self):
        self.linear = types.SimpleNamespace(x=0.0); self.angular = types.SimpleNamespace(z=0.0)
mods = {
 'rclpy': types.SimpleNamespace(init=lambda **k: None, spin=None, ok=lambda: True, shutdown=lambda: None),
 'rclpy.node': types.SimpleNamespace(Node=Node),
 'rclpy.qos': types.SimpleNamespace(QoSProfile=lambda **k: None, ReliabilityPolicy=types.SimpleNamespace(RELIABLE=1),
     DurabilityPolicy=types.SimpleNamespace(VOLATILE=1), HistoryPolicy=types.SimpleNamespace(KEEP_LAST=1), qos_profile_sensor_data=None),
 'geometry_msgs': types.ModuleType('geometry_msgs'), 'geometry_msgs.msg': types.SimpleNamespace(Twist=Twist),
 'nav_msgs': types.ModuleType('nav_msgs'), 'nav_msgs.msg': types.SimpleNamespace(Odometry=object),
 'ultralytics': types.SimpleNamespace(YOLO=lambda p: (_ for _ in ()).throw(RuntimeError("no model offline"))),
}
sys.modules.update(mods)
NODE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "COSC_4366EL_A3_G5.py")
spec = importlib.util.spec_from_file_location("a3", NODE); a3 = importlib.util.module_from_spec(spec); sys.modules["a3"] = a3; spec.loader.exec_module(a3)

fails = 0
def check(name, cond, extra=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    fails += (not cond)

# ---------- 1. geometry ----------
for f, l in [(10, 0), (20, 10), (35, -15), (45, 25)]:
    u, v = a3.ground_to_pixel(f, l); g = a3.pixel_to_ground(u, v)
    check(f"roundtrip ({f},{l})", abs(g[0]-f) < 0.6 and abs(g[1]-l) < 0.6, f"{g[0]:.1f},{g[1]:.1f}")
def old_est(y):
    rel = (240 - y) / 240; ang = 54 + rel * 24
    return 20 * math.tan(math.radians(ang)) - 5
for y in (480, 400, 300, 240, 150):
    print(f"   row {y}: old model {old_est(y):5.1f}cm  new {a3.pixel_to_ground(320, y)[0]:5.1f}cm")

# ---------- 2. line tracker on synthetic masks ----------
def draw_tape(mask, p0, p1, width_cm=5.0):
    """Draw tape between two floor points (fwd, lat) cm as a polygon."""
    (f0, l0), (f1, l1) = p0, p1
    d = np.array([f1 - f0, l1 - l0], float); d /= np.linalg.norm(d); n = np.array([-d[1], d[0]]) * width_cm / 2
    poly = []
    for fx, lx in [(f0 + n[0], l0 + n[1]), (f1 + n[0], l1 + n[1]), (f1 - n[0], l1 - n[1]), (f0 - n[0], l0 - n[1])]:
        p = a3.ground_to_pixel(max(fx, -3.0), lx); poly.append(p)
    cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
vp = a3.VisionProcessor("none")
def run(mask):
    vp._last_x = a3.CX
    return vp._track_line(cv2.bitwise_and(mask, vp._roi)), vp._find_junction(mask)

m = np.zeros((480, 640), np.uint8); draw_tape(m, (-3, 0), (80, 0))
ln, j = run(m); check("straight centred", ln.ok and abs(ln.lateral) < 0.05 and abs(ln.heading) < 0.05, f"{ln.lateral:+.3f},{ln.heading:+.3f}"); check("no junction on straight", j is None, str(j))
m = np.zeros((480, 640), np.uint8); draw_tape(m, (-3, 4), (80, 4))
ln, _ = run(m); check("tape right → lateral +", ln.ok and ln.lateral > 0.05, f"{ln.lateral:+.3f}")
m = np.zeros((480, 640), np.uint8); draw_tape(m, (-3, 0), (80, 25))
ln, _ = run(m); check("line leaning right → heading +", ln.ok and ln.heading > 0.05, f"{ln.heading:+.3f}")
m = np.zeros((480, 640), np.uint8); draw_tape(m, (-3, 0), (80, 0)); draw_tape(m, (25, -60), (25, 60))
ln, j = run(m); check("+ junction: still centred", ln.ok and abs(ln.lateral) < 0.05 and abs(ln.heading) < 0.05, f"{ln.lateral:+.3f},{ln.heading:+.3f}")
check("+ junction distance ≈ 22.5cm (near edge)", j is not None and abs(j - 22.5) < 3, str(j))
m = np.zeros((480, 640), np.uint8); draw_tape(m, (-3, 0), (80, 0)); draw_tape(m, (20, 0), (40, 40))
ln, _ = run(m); check("Y branch right: keeps straight", ln.ok and abs(ln.lateral) < 0.08 and abs(ln.heading) < 0.1, f"{ln.lateral:+.3f},{ln.heading:+.3f}")
m = np.zeros((480, 640), np.uint8)
for k in range(0, 80, 12): draw_tape(m, (k - 3, 0), (k + 3, 0))
ln, _ = run(m); check("dashed line tracked", ln.ok and abs(ln.lateral) < 0.05, f"pts={len(ln.points)}")

# ---------- 3. lane gating at a + junction 30cm ahead (near edge 27.5) ----------
J = 27.5
def S(label, f, l): s = a3.Sign(label, 0.8, (0, 0, 1, 1), f - 3, l, f); s.reject = a3.VisionProcessor._lane_check(s, J); return s
cases = [("ours Q4 GO FORWARD", S("GO FORWARD", 22, 12), ""),
         ("east-bound sign (our Q3) read as TURN RIGHT", S("TURN RIGHT", 22, -12), "left side"),
         ("west-bound sign (our Q1) read as TURN LEFT", S("TURN LEFT", 38, 12), "past junction"),
         ("south-bound sign (our Q2)", S("TURN LEFT", 38, -12), "left side"),
         ("next junction sign", S("TURN RIGHT", 70, 12), "too far"),
         ("dead end top of T", S("DEAD END", 34, 0), ""),
         ("west-bound NO ENTRY (Q1)", S("NO ENTRY", 38, 14), "past junction"),
         ("STOP paired Q3 (ignored, Q4 twin counts)", S("STOP", 22, -12), "left side")]
for name, s, want in cases: check(f"gate: {name}", s.reject == want, repr(s.reject))

# ---------- 4. closed-loop junction simulation ----------
a3.SHOW_WINDOWS = False
class FakeCap:
    def __init__(self, *a): pass
    def set(self, *a): pass
    def isOpened(self): return True
    def read(self): import time; time.sleep(0.01); return False, None
    def release(self): pass
a3.cv2.VideoCapture = FakeCap

def visible(f, l):
    if f < 1: return False
    p = a3.ground_to_pixel(f, l); return p is not None and 0 <= p[0] < 640 and 0 <= p[1] < 480

def simulate(signs, roads, flicker=None, steps=1400, seed=1):
    """signs: [(label, wx, wy)]  roads: headings (rad) available at the junction at x=1.0"""
    random.seed(seed); LOG.clear(); SIM_T[0] = 0.0
    n = a3.VisionSlamNode(); n._running = False; n.odom_ok = True
    states = []
    for k in range(steps):
        SIM_T[0] = k * 0.05
        # robot-relative sign observations
        c, s_ = math.cos(n.yaw), math.sin(n.yaw)
        obs = []
        for lbl, wx, wy in signs:
            dx, dy = wx - n.x, wy - n.y
            f, l = (dx * c + dy * s_) * 100, (dx * s_ - dy * c) * 100
            if visible(f, l) and random.random() < 0.8:
                if flicker and lbl in flicker and random.random() < 0.3: lbl = flicker[lbl]
                f += random.gauss(0, 1.0); l += random.gauss(0, 1.0)
                obs.append(a3.Sign(lbl, random.uniform(0.55, 0.9), (0, 0, 1, 1), f - 3, l, f))
        # junction crossing tape (cross road along y at x=1.0) — only when roughly facing +x
        jf = ((1.0 - 0.025) - n.x) * 100 if abs(a3.wrap(n.yaw)) < 0.3 else None
        jcm = jf if jf is not None and 5 < jf < 60 else None
        for o in obs: o.reject = a3.VisionProcessor._lane_check(o, jcm)
        # line: visible if heading matches a road at robot's location
        # main road along y=0 (ends at x=1.0 for a T); cross road along x=1.0
        avail = []
        if abs(n.y) < 0.15 and (0.0 in roads or n.x < 1.05): avail += [0.0, math.pi]
        if abs(n.x - 1.0) < 0.15 and n.y > -0.15 and math.pi / 2 in roads: avail += [math.pi / 2, -math.pi / 2]
        if abs(n.x - 1.0) < 0.15 and n.y < 0.15 and -math.pi / 2 in roads: avail += [math.pi / 2, -math.pi / 2]
        err = min((abs(a3.wrap(n.yaw - r)) for r in avail), default=9)
        line = a3.Line(ok=err < 0.25, lateral=0.0, heading=a3.wrap(n.yaw - min(avail, key=lambda r: abs(a3.wrap(n.yaw - r))))) if avail else a3.Line()
        vis = a3.VisionResult(line, obs, jcm, None, None)
        with n._vis_lock: n._vis = vis; n._vis_seq += 1
        tw = Twist()
        n._vis_seq  # control loop body (without publish)
        with n._vis_lock: v, seq = n._vis, n._vis_seq
        if seq != n._last_seq: n._last_seq = seq; n._on_new_frame(v)
        n._tick(tw, v.line)
        # kinematics
        n.yaw = a3.wrap(n.yaw + tw.angular.z * 0.05)
        n.x += tw.linear.x * math.cos(n.yaw) * 0.05; n.y += tw.linear.x * math.sin(n.yaw) * 0.05
        n.odometer += abs(tw.linear.x) * 0.05
        if not states or states[-1] != n.state: states.append(n.state)
    return n, states

R = -0.12  # right of path (north-bound robot travels +x; right = -y)
plus = [0.0, math.pi / 2, -math.pi / 2, math.pi]
cross_traffic = [("TURN RIGHT", 0.88, 0.12), ("TURN LEFT", 1.12, R), ("TURN LEFT", 1.12, 0.12)]

n, st = simulate([("GO FORWARD", 0.88, R)] + cross_traffic, plus, flicker={"GO FORWARD": "TURN RIGHT"})
check("A: + junction, GO FORWARD + cross-traffic signs → straight", "TURN" not in st and "CROSS" in st and abs(n.y) < 0.05 and n.x > 1.5, f"{st} end=({n.x:.2f},{n.y:.2f},{math.degrees(n.yaw):.0f}°)")

n, st = simulate([("TURN RIGHT", 0.88, R), ("DEAD END", 1.06, 0.0)], [math.pi / 2, -math.pi / 2, math.pi], flicker={"TURN RIGHT": "TURN LEFT"})
check("B: T junction, TURN RIGHT + DEAD END → turns right", "TURN" in st and "UTURN" not in st and abs(a3.wrap(n.yaw + math.pi / 2)) < 0.3, f"{st} yaw={math.degrees(n.yaw):.0f}°")

n, st = simulate([("STOP", 0.86, R), ("TURN LEFT", 0.88, -0.2)] + cross_traffic, plus)
check("C: STOP + TURN LEFT → stop, then left", st[:3] == ["FOLLOW", "STOP", "FOLLOW"] and "TURN" in st and abs(a3.wrap(n.yaw - math.pi / 2)) < 0.3, f"{st} yaw={math.degrees(n.yaw):.0f}°")

n, st = simulate([("DEAD END", 1.06, 0.0)], [math.pi], steps=900)
check("D: DEAD END alone → U-turn", "UTURN" in st and abs(a3.wrap(n.yaw - math.pi)) < 0.3, f"{st} yaw={math.degrees(n.yaw):.0f}°")

n, st = simulate(cross_traffic, plus)
check("E: no sign for us, only cross-traffic signs → straight", "TURN" not in st and abs(n.y) < 0.05 and n.x > 1.5, f"{st}")

n, st = simulate([("NO ENTRY", 0.86, R), ("NO ENTRY", 0.86, 0.12), ("TURN LEFT", 0.88, -0.2)], plus)
check("F: NO ENTRY pair + TURN LEFT → follows the turn", "TURN" in st and "UTURN" not in st and abs(a3.wrap(n.yaw - math.pi / 2)) < 0.3, f"{st} yaw={math.degrees(n.yaw):.0f}°")

print("\nlast log lines (F):"); [print("  ", l) for l in LOG[-8:]]
print(f"\n{fails} failure(s)")
sys.exit(1 if fails else 0)
