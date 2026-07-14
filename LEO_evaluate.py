import os, random, numpy as np, torch
import argparse
from evaluate_T import TSNEnv, DQNAgent, DuelingDQN, OCSQFOfflineScheduler, FLOW, generate_sn_dn
from Kuiper_Shell import build_nels_snapshotter, GroundStation
from collections import defaultdict
from collections import Counter
import random
import json
import networkx as nx
# ====== 1) 在文件顶部补 ======
import os
def _host_to_sat(G0, hname:str):
    # host只连到一个卫星：取其邻居的卫星ID（字符串->int）
    for w in list(G0.successors(hname)) + list(G0.predecessors(hname)):
        if isinstance(w, str) and not w.startswith('h'):
            try: return int(w)
            except: pass
    return None

def set_seed(seed: int):
    """让评估可复现：固定随机种子"""
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 让部分后端尽量确定性（可能稍慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def expand_to_bidir(edges, G):
    """把有向边列表扩展为双向；只对卫星-卫星边且反向边存在于图中时生效。"""
    out = set()
    for (u, v) in edges:
        out.add((u, v))
        # 只处理卫星-卫星，且图里确实有反向边
        if (not str(u).startswith('h')) and (not str(v).startswith('h')) and G.has_edge(v, u):
            out.add((v, u))
    return list(out)

def is_sat(n: str) -> bool:
    """你的工程里 host 以 'h' 开头，这里只把非 host 视为卫星节点。"""
    return not str(n).startswith('h')

def choose_important_satellite(env, G, top_n_flows: int = 20, k_edges: int = 40,
                               avoid_nodes=None):
    """
    选择“重要卫星”的策略：
    ① 先用你已有的 pick_important_edges_from_flows 找一批热点边（按流负载/最短路热度）；
       统计这些边的端点出现次数（只计卫星节点），出现次数越多越“重要”。
    ② 若没有拿到热点边（或冲突被过滤），退回到“卫星子图的介数中心性”挑最大者。
    ③ 再不行就从所有卫星里随机挑一个（兜底）。
    """
    if avoid_nodes is None:
        avoid_nodes = set()

    # --- ① 基于“热点边”的节点热度 ---
    hot_edges = []
    try:
        # disjoint=False 允许覆盖更广的边；k_edges 越大采样越多
        hot_edges = pick_important_edges_from_flows(env, k=k_edges, top_n_flows=top_n_flows, disjoint=False)
    except Exception:
        pass

    cnt = Counter()
    for (u, v) in hot_edges:
        if is_sat(u): cnt[u] += 1
        if is_sat(v): cnt[v] += 1

    # 去掉不想选的节点（比如之前已被 SEU/掐断波及的端点）
    for n in list(cnt.keys()):
        if n in avoid_nodes:
            del cnt[n]

    if cnt:
        # 并列时用度数打破平局（越“多连”越优先）
        cand_sorted = sorted(cnt.items(), key=lambda kv: ( -kv[1], -G.degree(kv[0]) ))
        return cand_sorted[0][0]

    # --- ② 退回中心性（只在“卫星子图”上计算） ---
    try:
        H = G.copy()
        hosts = [n for n in H.nodes if not is_sat(n)]
        H.remove_nodes_from(hosts)
        if H.number_of_nodes() > 0:
            bc = nx.betweenness_centrality(H, weight='weight', normalized=True)  # 有向图也可用
            for n, _ in sorted(bc.items(), key=lambda kv: -kv[1]):
                if n not in avoid_nodes:
                    return n
    except Exception:
        pass

    # --- ③ 兜底随机 ---
    sats = [n for n in G.nodes if is_sat(n) and n not in avoid_nodes]
    if not sats:
        sats = [n for n in G.nodes if is_sat(n)]
    return random.choice(sats)

def pick_important_edges_from_flows(env, k=2, top_n_flows=5, disjoint=True):
    """
    从前 top_n_flows 条流的最短时延路径中，统计卫星-卫星边出现频次。
    若 disjoint=True，则在选出前 k 条时，保证所选边两两端点不相交（不共享任何卫星）。
    """
    counter = defaultdict(int)

    base_flows = env.scheduled[:top_n_flows]
    for f in base_flows:
        t_sec = getattr(f, "t_start_ms", 0) / 1000.0
        Gt, _ = env.snapshotter.graph_at(t_sec)

        src, dst = f.sn_dn  # 'hX' 到 'hY'
        try:
            path = nx.dijkstra_path(Gt, src, dst, weight='weight')
        except nx.NetworkXNoPath:
            continue

        # 只统计卫星-卫星的有向边
        for u, v in zip(path, path[1:]):
            if not str(u).startswith('h') and not str(v).startswith('h'):
                counter[(u, v)] += 1

    if not counter:
        return []

    # 频次降序；同频次按端点字典序稳定排序，避免每次抖动
    edges_sorted = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))

    if not disjoint:
        return [e for (e, _) in edges_sorted[:k]]

    # 保证端点不相交的贪心选择
    chosen = []
    used_nodes = set()  # 已被占用的卫星节点
    for (e, cnt) in edges_sorted:
        u, v = e
        if u in used_nodes or v in used_nodes:
            continue
        chosen.append(e)
        used_nodes.add(u)
        used_nodes.add(v)
        if len(chosen) >= k:
            break

    # 如果可选的“端点不相交”边不足 k 条，就返回能找到的那几条
    return chosen

def build_shared_agent_and_net(offline, device, ckpt_path):
    """
    创建共享网络，并加载 checkpoint；构造每个交换机的 DQNAgent 绑定同一网络。
    返回：shared_net, shared_target, optimizer, agents(dict)
    """
    # 和训练时保持一致的维度：每端口 feat_dim=4，obs = (max_outdeg, feat_dim) 展平
    max_outdeg = offline.max_outdeg
    obs_dim = 6 * max_outdeg
    act_dim = max_outdeg

    # 共享网络 & 目标网络
    shared_net    = DuelingDQN(obs_dim, act_dim).to(device)
    shared_target = DuelingDQN(obs_dim, act_dim).to(device)
    optimizer     = torch.optim.Adam(shared_net.parameters(), lr=5e-5)

    # 加载 checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    shared_net.load_state_dict(ckpt['model_state_dict'])
    # 评估其实不需要优化器状态；如果你想继续训练再加载
    if 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception:
            pass
    shared_target.load_state_dict(shared_net.state_dict())

    shared_net.eval()  # 关掉dropout/BN
    torch.set_grad_enabled(False)

    # 构造每个交换机的 agent，并挂共享网络；评估时关探索/关噪声
    G = offline.G
    agents = {}
    switches = [n for n in G.nodes if not str(n).startswith('h')]
    for sw in switches:
        ag = DQNAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            lr=1e-4, gamma=0.99, buffer_size=1, batch_size=1  # 这些评估用不到
        )
        ag.net        = shared_net
        ag.target_net = shared_target
        ag.optimizer  = optimizer
        ag.epsilon    = 0.0
        if hasattr(ag, "noise_sigma"):
            ag.noise_sigma = 0.0
        agents[sw] = ag

    return shared_net, shared_target, optimizer, agents, obs_dim, act_dim

def apply_events_to_tsn_env(env, events, T_cycle_ms=1):
    for e in events.get("edge_outages", []):
        u, v = str(e["u"]), str(e["v"])
        start_slot = int(e["t0_ms"] / T_cycle_ms)
        until_slot = int(e["t1_ms"] / T_cycle_ms)
        env.ban_edges([(u,v)], start_slot=start_slot, until_slot=until_slot)

def evaluate_once(snapshotter, offline, agents, anomaly_prob=0.0,
                  seed=42, run_idx=1, outdir="eval_suites"):
    set_seed(seed)
    G0, hnodes0 = snapshotter.graph_at(0.0)
    flows = offline.generate_flow(5)
    env = TSNEnv(G0, flows, offline, snapshotter, anomaly_prob=anomaly_prob)

    record_suite = True
    # REC = {"edge_outages": []}
    REC = {"edge_outages": [], "node_seu": []}
    T_CYCLE_MS = 1  # 你的仿真就是 1ms

    # 包一层 ban_edges 以记录所有被禁用的卫星-卫星边（转毫秒）
    env._ban_edges_orig = env.ban_edges

    def _ban_edges_record(edges, start_slot=None, until_slot=None):
        s = env.global_slot if start_slot is None else int(start_slot)
        e = int(until_slot)
        for (u, v) in edges:
            if (not str(u).startswith('h')) and (not str(v).startswith('h')):
                REC["edge_outages"].append({
                    "u": int(u), "v": int(v),
                    "t0_ms": int(s * T_CYCLE_MS),
                    "t1_ms": int(e * T_CYCLE_MS),
                })
        return env._ban_edges_orig(edges, until_slot=e, start_slot=s)

    env.ban_edges = _ban_edges_record

    # === 安排 3 次“掐断×5s”（每次4条边，双向） + 1 次“SEU×10s” ===
    slots_per_sec = int(1000 / offline.T_cycle)  # =1000
    DUR_CUT = 5 * slots_per_sec
    DUR_SEU = 10 * slots_per_sec
    WIN_60S = 60 * slots_per_sec
    base_slot = 200

    EDGES_PER_EVENT = 4
    K_CANDIDATES = EDGES_PER_EVENT * 3

    num_cuts = 3
    used_edges = set()

    for i in range(num_cuts):
        candidates = pick_important_edges_from_flows(
            env, k=K_CANDIDATES, top_n_flows=10, disjoint=True
        )
        candidates = [e for e in candidates
                      if (e not in used_edges and (e[1], e[0]) not in used_edges)]

        cut_edges, event_nodes = [], set()
        for (u, v) in candidates:
            if u in event_nodes or v in event_nodes:
                continue
            cut_edges.append((u, v))
            event_nodes.update([u, v])
            if len(cut_edges) == EDGES_PER_EVENT:
                break

        if len(cut_edges) < EDGES_PER_EVENT:
            sat_edges = [(u, v) for (u, v) in G0.edges()
                         if not str(u).startswith('h') and not str(v).startswith('h')]
            random.shuffle(sat_edges)
            for (u, v) in sat_edges:
                if (u in event_nodes) or (v in event_nodes):
                    continue
                if (u, v) in used_edges or (v, u) in used_edges:
                    continue
                cut_edges.append((u, v))
                event_nodes.update([u, v])
                if len(cut_edges) == EDGES_PER_EVENT:
                    break

        cut_edges = expand_to_bidir(cut_edges, env.G)
        for e in cut_edges:
            used_edges.add(e)

        latest_start = base_slot + max(0, WIN_60S - DUR_CUT)
        start_i = random.randint(base_slot, latest_start)
        env.ban_edges(cut_edges, start_slot=start_i, until_slot=start_i + DUR_CUT)
        print(f"[CUT#{i + 1}] edges={cut_edges} slots=[{start_i},{start_i + DUR_CUT}] (~5s, bidir)")


    # 如果你上面维护了 used_edges，可以顺带把端点也避开，避免与掐断事件完全重叠
    avoid_nodes = set()
    for (u, v) in list(used_edges):
        avoid_nodes.add(u)
        avoid_nodes.add(v)

    # 再安排 1 次 SEU：选择“重要卫星”，把它所有星间入/出边禁 10s
    seu_node = choose_important_satellite(env, G0, top_n_flows=20, k_edges=40)  # ← 不避开 used_edges

    # 收集该星的所有 ISL（入 + 出），仅限卫星-卫星
    seu_edges = [(u, v) for (u, v) in G0.out_edges(seu_node) if not str(v).startswith('h')]
    seu_edges += [(u, v) for (u, v) in G0.in_edges(seu_node) if not str(u).startswith('h')]
    # 去重
    seu_edges = list({e: None for e in seu_edges}.keys())

    start_seu = random.randint(base_slot, base_slot + max(0, WIN_60S - DUR_SEU))
    env.ban_edges(seu_edges, start_slot=start_seu, until_slot=start_seu + DUR_SEU)
    print(
        f"[SEU] IMPORTANT node={seu_node} edges={len(seu_edges)} slots=[{start_seu},{start_seu + DUR_SEU}] (~10s)")
    # === 仅追加节点级记录，边级记录已由 env.ban_edges 包装器自动写入 ===
    start_ms = int(start_seu * T_CYCLE_MS)
    end_ms = int((start_seu + DUR_SEU) * T_CYCLE_MS)
    REC["node_seu"].append({
        "n": str(seu_node),  # 节点ID统一成字符串，和其它json风格一致
        "t0_ms": start_ms,
        "t1_ms": end_ms
    })

    # 可达性自检（保持原逻辑）
    impossible = 0
    impossible_list = []
    for fid, st in env.flow_states.items():
        src, dst = st['pos'], st['dst']
        try:
            sp = nx.dijkstra_path_length(env.G, src, dst, weight='weight')
        except nx.NetworkXNoPath:
            sp = float('inf')
        ddl = float(st['deadline'])
        if sp > ddl:
            impossible += 1
            impossible_list.append((fid, sp, ddl))
    print(f"[CHECK] Shortest-propagation>DDL: {impossible}/{len(env.flow_states)}")
    for x in impossible_list[:10]:
        print(f"  fid={x[0]} sp={x[1]:.1f}ms ddl={x[2]:.1f}ms")

    # 初始 obs
    obs_for = {}
    # 先把“无路可走”的流直接失败并移除
    for fid in list(env.flow_states.keys()):
        st = env.flow_states[fid]
        Gt, _ = env._graph_for_fid(fid)
        if not env._has_path_quick(Gt, st['pos'], st['dst']):
            env._fail_flow_now(fid, reason="no_path")
            continue
        # 只有可达的才取观测，后续才参与决策
        obs_for[fid] = env._get_obs_for(fid)
    # obs_for = {fid: env._get_obs_for(fid) for fid in env.flow_states.keys()}
    total_flows = len(env.flow_states)

    # 主循环（保持原逻辑）
    while len(env.flow_states) > 0:
        valid_masks = {}
        # 先把当前时隙“无路可走”的流直接判失败并移除
        for fid in list(env.flow_states.keys()):
            st = env.flow_states[fid]
            Gt, _ = env._graph_for_fid(fid)
            if not env._has_path_quick(Gt, st['pos'], st['dst']):
                env._fail_flow_now(fid, reason="no_path")

        if not env.flow_states:
            break

        for fid, st in env.flow_states.items():
            sw = st['pos']
            Gt, _ = env._graph_for_fid(fid)
            nbrs = list(Gt.successors(sw))
            act_dim = next(iter(agents.values())).act_dim
            mask = np.zeros(act_dim, dtype=bool)
            mask[:len(nbrs)] = True
            valid_masks[fid] = mask

        actions = {}
        for fid, st in env.flow_states.items():
            sw = st['pos']
            # obs_fid = env._get_obs_for(fid)
            actions[fid] = agents[sw].select_action(obs_for[fid], valid_masks[fid])

        next_obs_all, rewards_all, finished_ids, _ = env.step(actions, obs_for)

        obs_for = {fid: next_obs_all[fid] for fid in next_obs_all if fid not in finished_ids}
        for fid in finished_ids:
            env.flow_states.pop(fid)
            # print(f"流{fid}已完成")

    # print("[FAIL STATS]", env.fail_stats)
    success = (total_flows - env.failed_count) / total_flows
    avg_delay_ms = env.get_average_delay_ms()
    print(f"[METRIC] avg_delay_ms = {avg_delay_ms:.3f} ms")

    # ---------- 仅在“生成场景”模式下写 JSON ----------
    if record_suite:
        def _dump_suite(path_json, G0, flows, seed_val, rec, t_cycle_ms):
            host_pairs = []
            for f in flows:
                host_pairs.append({
                    "src": f.sn_dn[0], "dst": f.sn_dn[1],
                    "ddl_ms": int(f.ddl), "pf_ms": int(f.pf),
                    "pkt_num": int(f.pkt_num),
                    "t_start_ms": int(getattr(f, "t_start_ms", 0)),
                    "t_life_ms": int(getattr(f, "t_life_ms", 60000)),
                })
            sat_pairs = []
            for f in host_pairs:
                us = _host_to_sat(G0, f["src"])
                vs = _host_to_sat(G0, f["dst"])
                if us is not None and vs is not None:
                    sat_pairs.append({
                        "src": us, "dst": vs,
                        "ddl_ms": f["ddl_ms"], "pf_ms": f["pf_ms"],
                        "pkt_num": f["pkt_num"],
                        "t_start_ms": f["t_start_ms"], "t_life_ms": f["t_life_ms"],
                    })
            suite_out = {
                "seed": seed_val, "T_cycle_ms": t_cycle_ms,
                "flows": {"host_pairs": host_pairs, "sat_pairs": sat_pairs},
                "events": {
                    "edge_outages": rec.get("edge_outages", []),
                    "node_seu": rec.get("node_seu", [])
                },
            }
            os.makedirs(outdir, exist_ok=True)
            with open(path_json, "w", encoding="utf-8") as f:
                json.dump(suite_out, f, indent=2)

        os.makedirs(outdir, exist_ok=True)
        _dump_suite(os.path.join(outdir, f"run_50{run_idx:02d}.json"), G0, env.flows, seed, REC, T_CYCLE_MS)

    return float(success)

def evaluate_avg(snapshotter, offline, agents, runs=5, anomaly_prob=0.0,
                 base_seed=1000, outdir="eval_suites"):
    """
    多次评估取均值和方差；每次 seed 变化 → 不同但可复现的一批流/异常
    """
    scores = []

    for i in range(0, runs):
        sr = evaluate_once(snapshotter, offline, agents,
                           anomaly_prob=anomaly_prob,
                           seed=base_seed + i,
                           run_idx=i + 1,
                           outdir=outdir)
        scores.append(sr)
    return float(np.mean(scores)), float(np.std(scores)), scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved CSQF DQN Agent")
    parser.add_argument("--ckpt", type=str, required=True,
                        help=r"Checkpoint 路径，如 D:\Agent_revised\csqf_agent_ep10000.pth")
    parser.add_argument("--runs", type=int, default=5, help="重复评估次数")
    # parser.add_argument("--flows", type=int, default=700, help="每次评估流数量")
    parser.add_argument("--anomaly", type=float, default=0.0, help="评估阶段异常概率")
    parser.add_argument("--seed", type=int, default=1000, help="评估基准种子")
    parser.add_argument("--outdir", type=str, default="eval_suites",
                        help="生成场景 JSON 的输出目录（仅当未提供 --scenario 时生效）")
    parser.add_argument("--run-idx", type=int, default=None,
                        help="生成场景时写 run_XX.json 的 XX（1-based）；不提供则按循环次序自动赋值")
    args = parser.parse_args()

    ckpt_path = args.ckpt
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint：{ckpt_path}")

    device = torch.device("cpu")

    # 1) 构建 LEO 快照器
    # w = WalkerDelta(P=8, S=12, inc_deg=53.0, alt_km=600.0, F=0)
    grounds = [
        GroundStation("Singapore", 1.3521, 103.8198, 25.0),
        GroundStation("Tokyo",     35.6762, 139.6503, 25.0),
        GroundStation("Frankfurt", 50.1109,   8.6821, 25.0),
        GroundStation("NewYork",   40.7128, -74.0060, 25.0),
        GroundStation("LA",        34.0522,-118.2437, 25.0),
        GroundStation("Sydney",   -33.8688, 151.2093, 25.0),
        GroundStation("Dubai",     25.2048,  55.2708, 25.0),
        GroundStation("SaoPaulo", -23.5505, -46.6333, 25.0),
    ]
    # grounds = [
    #     GroundStation("Seattle", 47.6062, -122.3321, 25.0),
    #     GroundStation("LosAngeles", 34.0522, -118.2437, 25.0),
    #     GroundStation("SanFrancisco", 37.7749, -122.4194, 25.0),
    #     GroundStation("Vancouver", 49.2827, -123.1207, 25.0),
    #     GroundStation("MexicoCity", 19.4326, -99.1332, 25.0),
    #     GroundStation("NewYork", 40.7128, -74.0060, 25.0),
    #     GroundStation("Miami", 25.7617, -80.1918, 25.0),
    #     GroundStation("Toronto", 43.6532, -79.3832, 25.0),
    #
    #     GroundStation("SaoPaulo", -23.5505, -46.6333, 25.0),
    #     GroundStation("BuenosAires", -34.6037, -58.3816, 25.0),
    #     GroundStation("Santiago", -33.4489, -70.6693, 25.0),
    #     GroundStation("Lima", -12.0464, -77.0428, 25.0),
    #
    #     GroundStation("London", 51.5074, -0.1278, 25.0),
    #     GroundStation("Frankfurt", 50.1109, 8.6821, 25.0),
    #     GroundStation("Madrid", 40.4168, -3.7038, 25.0),
    #     GroundStation("Istanbul", 41.0082, 28.9784, 25.0),
    #
    #     GroundStation("Cairo", 30.0444, 31.2357, 25.0),
    #     GroundStation("Nairobi", -1.2921, 36.8219, 25.0),
    #     GroundStation("Johannesburg", -26.2041, 28.0473, 25.0),
    #     GroundStation("Dubai", 25.2048, 55.2708, 25.0),
    #
    #     GroundStation("Mumbai", 19.0760, 72.8777, 25.0),
    #     GroundStation("Singapore", 1.3521, 103.8198, 25.0),
    #     GroundStation("Tokyo", 35.6762, 139.6503, 25.0),
    #     GroundStation("Sydney", -33.8688, 151.2093, 25.0),
    # ]
    # snapshotter = Snapshotter(w, grounds, isl_max_km=6000.0, interplane=True)
    # snapshotter = build_kuiper_shell2_snapshotter(grounds, isl_max_km=6000.0, interplane=True)
    snapshotter = build_nels_snapshotter(grounds, isl_max_km=6000.0, interplane=True)

    # 离线调度器 & 共享网络/Agent
    G0, hnodes0 = snapshotter.graph_at(0.0)
    offline = OCSQFOfflineScheduler(G0, hnodes0)
    shared_net, shared_target, optimizer, agents, obs_dim, act_dim = \
        build_shared_agent_and_net(offline, device, ckpt_path)
    # 评估
    mean_sr, std_sr, all_scores = evaluate_avg(
        snapshotter, offline, agents,
        runs=args.runs, anomaly_prob=args.anomaly,
        base_seed=args.seed, outdir=args.outdir
    )
    print(f"[EVAL] ε=0, no-noise  -> success = {mean_sr * 100:.2f}% ± {std_sr * 100:.2f}% "
          f"(runs={args.runs}, anomaly={args.anomaly})")
    print("per-run success:", [f"{s * 100:.2f}%" for s in all_scores])

# python LEO_evaluate.py  --ckpt "D:\Agent_test\csqf_agent_ep10000.pth" --runs 5 --seed 500 --outdir eval_suites
if __name__ == "__main__":
    main()

