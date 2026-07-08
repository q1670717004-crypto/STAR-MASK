# kuiper_snapshot.py
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import networkx as nx

# --- 常量 ---
R_E = 6371.0                  # 地球半径 (km)
MU  = 398600.4418             # 地心引力常数 (km^3/s^2)
OMEGA_E = 7.2921159e-5        # 地球自转角速度 (rad/s)
C_KMPS = 299792.458           # 光速 (km/s)

# --- 基本工具 ---
def deg2rad(d): return d * math.pi / 180.0
def rot_z(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)
def rot_x(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)

def eci_to_ecef(r_eci: np.ndarray, t: float) -> np.ndarray:
    # 简化：ECI -> ECEF 仅绕 z 轴自转
    return rot_z(OMEGA_E * t) @ r_eci

def llh_to_ecef(lat_deg, lon_deg, alt_km=0.0) -> np.ndarray:
    lat, lon = deg2rad(lat_deg), deg2rad(lon_deg)
    x = (R_E + alt_km) * math.cos(lat) * math.cos(lon)
    y = (R_E + alt_km) * math.cos(lat) * math.sin(lon)
    z = (R_E + alt_km) * math.sin(lat)
    return np.array([x,y,z], float)

def euclid_dist_km(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def elevation_deg_from_vectors(sat_ecef: np.ndarray, gs_ecef: np.ndarray) -> float:
    # 快速仰角近似（够用做“可见性+择优”）
    u = gs_ecef / np.linalg.norm(gs_ecef)
    v = sat_ecef / np.linalg.norm(sat_ecef)
    central = math.degrees(math.acos(np.clip(u.dot(v), -1, 1)))
    return 90.0 - central

def wrap360(x: float) -> float:
    x = x % 360.0
    return x + 360.0 if x < 0 else x

def circ_dist_deg(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)

# --- Walker-Delta 单组 ---
@dataclass
class WalkerDelta:
    P: int                 # 轨道面数
    S: int                 # 每面卫星数
    inc_deg: float         # 倾角
    alt_km: float          # 圆轨道高度
    F: int = 0             # Walker 相位参数
    a0_RAAN_deg: float = 0.0
    a0_M_deg: float   = 0.0

    def mean_motion(self) -> float:
        a = R_E + self.alt_km
        return math.sqrt(MU / (a**3))  # rad/s

    def sat_eci(self, t: float, p: int, s: int) -> np.ndarray:
        a = R_E + self.alt_km
        n = self.mean_motion()
        inc = deg2rad(self.inc_deg)
        raan = deg2rad(self.a0_RAAN_deg + 360.0 * p / self.P)

        # —— 关键修正：跨面相位使用 F·360/(P·S) —— #
        M0_deg = self.a0_M_deg \
                 + (360.0 / self.S) * s \
                 + (360.0 / (self.P * self.S)) * (self.F * p)

        u = deg2rad(M0_deg) + n * t  # 圆轨：真近点角≈平近点角
        r_pf = np.array([a * math.cos(u), a * math.sin(u), 0.0])
        return rot_z(raan) @ (rot_x(inc) @ r_pf)

# --- 地面站 ---
@dataclass
class GroundStation:
    name: str
    lat: float
    lon: float
    elev_mask_deg: float = 25.0  # 可见性仰角门限

# --- 快照生成器（单组 Kuiper 第二壳层） ---
class Snapshotter:
    """
    graph_at(t_sec) -> (G, hnodes)
    G: DiGraph
      - 卫星 = 交换机：节点名 '0'..'1295'
      - 地面 = 主机：  节点名 'h0','h1',...
      - 边属性: weight = 传播时延 (毫秒, float)
    规则：
      - ISL：同面相邻（环）+ 邻面对齐（p 与 p+1）；
      - GSL：每个地面站只连“仰角最高”的一颗可见卫星（双向）。
    """
    def __init__(self,
                 w: WalkerDelta,
                 grounds: List[GroundStation],
                 isl_max_km: float = 6000.0,
                 interplane: bool = True):
        self.w = w
        self.grounds = grounds
        self.isl_max_km = isl_max_km
        self.interplane = interplane

        self.hnodes = [f"h{i}" for i,_ in enumerate(grounds)]
        self.gs_ecef = {f"h{i}": llh_to_ecef(gs.lat, gs.lon, 0.0)
                        for i, gs in enumerate(grounds)}

    def _sat_positions(self, t: float) -> Dict[Tuple[int,int], np.ndarray]:
        pos = {}
        for p in range(self.w.P):
            for s in range(self.w.S):
                r_eci  = self.w.sat_eci(t, p, s)
                pos[(p,s)] = eci_to_ecef(r_eci, t)
        return pos

    def graph_at(self, t: float):
        w = self.w
        sat_pos = self._sat_positions(t)

        def sat_idx(p, s): return p * w.S + s          # 连续编号
        def sat_name(p, s): return str(sat_idx(p, s))  # 字符串，便于与你现有 env 兼容

        G = nx.DiGraph()
        nsat = w.P * w.S  # 36*36=1296
        G.add_nodes_from([str(i) for i in range(nsat)])  # 卫星=交换机
        G.add_nodes_from(self.hnodes)                    # 地面=主机

        def add_edge(u: str, v: str, dist_km: float):
            delay_ms = 1000.0 * (dist_km / C_KMPS)
            G.add_edge(u, v, weight=delay_ms)

        # — ISL：同面相邻（每星两条，环形）—
        for p in range(w.P):
            for s in range(w.S):
                me  = sat_name(p, s)
                nxt = sat_name(p, (s+1) % w.S)
                prv = sat_name(p, (s-1) % w.S)
                a = sat_pos[(p, s)]
                b = sat_pos[(p, (s+1) % w.S)]
                c = sat_pos[(p, (s-1) % w.S)]
                d_ab = euclid_dist_km(a, b)
                d_ac = euclid_dist_km(a, c)
                if d_ab <= self.isl_max_km:
                    add_edge(me, nxt, d_ab); add_edge(nxt, me, d_ab)
                if d_ac <= self.isl_max_km:
                    add_edge(me, prv, d_ac); add_edge(prv, me, d_ac)

        # — ISL：邻面对齐（每星两条：右侧 p+1 & 左侧 p-1），按 Walker 相位就近匹配 —
        if self.interplane:
            # Walker-Delta 跨面相位：beta = F * 360 / (P*S)
            beta_deg = (w.F * 360.0) / (w.P * w.S)

            def mean_anomaly_deg(p_idx: int, s_idx: int, t_sec: float) -> float:
                # 与 sat_eci 的公式保持一致（单位：度）
                base = w.a0_M_deg \
                       + (360.0 / w.S) * s_idx \
                       + (360.0 / (w.P * w.S)) * (w.F * p_idx)
                return wrap360(base + math.degrees(w.mean_motion()) * t_sec)

            for p in range(w.P):
                for s in range(w.S):
                    a_name = sat_name(p, s)
                    Ma = mean_anomaly_deg(p, s, t)

                    # 右侧面 p+1：目标相位 = Ma + beta
                    q = (p + 1) % w.P
                    target_r = wrap360(Ma + beta_deg)
                    best_m_r, best_err_r = None, 1e9
                    for m in range(w.S):
                        Mqm = mean_anomaly_deg(q, m, t)
                        err = circ_dist_deg(Mqm, target_r)
                        if err < best_err_r:
                            best_err_r = err;
                            best_m_r = m
                    b_name = sat_name(q, best_m_r)
                    d_rb = euclid_dist_km(sat_pos[(p, s)], sat_pos[(q, best_m_r)])
                    if d_rb <= self.isl_max_km:
                        add_edge(a_name, b_name, d_rb);
                        add_edge(b_name, a_name, d_rb)

                    # 左侧面 p-1：目标相位 = Ma - beta
                    r = (p - 1) % w.P
                    target_l = wrap360(Ma - beta_deg)
                    best_m_l, best_err_l = None, 1e9
                    for m in range(w.S):
                        Mrm = mean_anomaly_deg(r, m, t)
                        err = circ_dist_deg(Mrm, target_l)
                        if err < best_err_l:
                            best_err_l = err;
                            best_m_l = m
                    c_name = sat_name(r, best_m_l)
                    d_lb = euclid_dist_km(sat_pos[(p, s)], sat_pos[(r, best_m_l)])
                    if d_lb <= self.isl_max_km:
                        add_edge(a_name, c_name, d_lb);
                        add_edge(c_name, a_name, d_lb)

        # — GSL：每个地面站只连“仰角最高”的可见卫星 —
        for i, gs in enumerate(self.grounds):
            hid = f"h{i}"
            r_gs = self.gs_ecef[hid]
            best = None  # (elev, dist, (p,s))
            for p in range(w.P):
                for s in range(w.S):
                    r_sat = sat_pos[(p,s)]
                    elev = elevation_deg_from_vectors(r_sat, r_gs)
                    if elev < gs.elev_mask_deg:
                        continue
                    d = euclid_dist_km(r_sat, r_gs)
                    if (best is None) or (elev > best[0]) or (elev == best[0] and d < best[1]):
                        best = (elev, d, (p, s))
            if best is not None:
                _, d, (p, s) = best
                sn = sat_name(p, s)
                add_edge(hid, sn, d)
                add_edge(sn, hid, d)

        return G, self.hnodes

# --- 直接给出 “Kuiper 第二壳层” 的配置 ---
def build_kuiper_shell2_snapshotter(
    grounds: List[GroundStation],
    isl_max_km: float = 6000.0,
    interplane: bool = True,
    F: int = 11,                # <—— 新增：默认 11
):
    """
    Kuiper 第二壳层：
      P=36, S=36, inc≈42°, alt≈610 km
    """
    w = WalkerDelta(P=36, S=36, inc_deg=42.0, alt_km=610.0, F=F)
    return Snapshotter(w, grounds, isl_max_km=isl_max_km, interplane=interplane)

# --- NeLS（10×12=120 星） ---
def build_nels_snapshotter(
    grounds: List[GroundStation],
    isl_max_km: float = 6000.0,
    interplane: bool = True,
    F: int = 3,                 # Walker 相位，可按需调整
    inc_deg: float = 42.0,      # 默认沿用 Kuiper 壳层的倾角/高度
    alt_km: float = 610.0,
):
    """
    NeLS 小星座：
      P=10, S=12, 默认 inc≈42°, alt≈610 km
    """
    w = WalkerDelta(P=10, S=12, inc_deg=inc_deg, alt_km=alt_km, F=F)
    return Snapshotter(w, grounds, isl_max_km=isl_max_km, interplane=interplane)