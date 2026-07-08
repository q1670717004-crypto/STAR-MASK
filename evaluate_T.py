import time
import math
import pandas
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from Kuiper_Shell import Snapshotter, WalkerDelta, GroundStation
from tianshou.data import PrioritizedReplayBuffer
from tianshou.data import Batch
import copy
# 该图有31个主机，13个交换机，53条双向边，所以有106条单向边
device = torch.device("cpu")
print(f"Using device: {device}")


# G = nx.DiGraph()
# G.add_nodes_from(nodes)
# G.add_weighted_edges_from(edges[i] for i in range(len(edges)))
#
# pos = nx.spring_layout(G,k=1,iterations=100)
# nx.draw_networkx_nodes(G, pos,node_size=400,alpha=0.3,node_color='#b3e2cd')
# nx.draw_networkx_edges(G, pos, width=1)
# nx.draw_networkx_labels(G, pos)
# # nx.draw_networkx_edge_labels(G, pos, {(edges[i][0],edges[i][1]):edges[i][2] for i in range(len(edges))})
# plt.tight_layout()
# plt.show()
# 网络系统中的所有数据流在开始时就已经全部确定，不会在中途添加新的数据流进来
# 每条流的路径信息都已经确定，经过的节点也已经确定，所以我们只需规划该流的数据包在接收队列的位置即可
# 我们就先设定6条流和每条流长度为10
# 同步误差先不管，都是0

class FLOW () :
    def __init__(self):
        self.id = 0
        self.offset = 0
        self.ddl = 0
        # self.jitter = 0.0
        self.t_start_ms = 0
        self.t_life_ms = 0
        self.sn_dn = None
        self.pf = 0       # period
        self.pkt_num = 0
        self.path = None
        self.RO = []     # 接收偏移量
        self.QO = []     # 队列偏移量

def generate_sn_dn(hnodes):   # 生成数据流的源节点和目的节点
    while True:
        sn = random.choice(hnodes)
        dn = random.choice(hnodes)
        if sn == dn:
            continue
        else:
            break
    return (sn,dn)

def generate_path(G, snodes, dnodes):
    shortest_weighted_path = nx.dijkstra_path(G, snodes, dnodes)   # dijkstra算法算源节点和目的节点的最短路径
    # print("节点%s到节点%s最短加权路径为：" % (snodes, dnodes), shortest_weighted_path)
    return shortest_weighted_path
# shortest_weighted_path = generate_path("h1", "h5")
# print(shortest_weighted_path)
def paths_to_edges(paths):  # 将流的路径转化成边的组合
    flow_edge_list = []
    for j in range(len(paths) - 1):
        edge_tmp = (paths[j], paths[j + 1])
        flow_edge_list.append(edge_tmp)          # 列表里面嵌套元组
    # print("节点%s到节点%s最短加权路径流经的边为：" % (paths[0], paths[-1]), flow_edge_list[1:])
    return flow_edge_list


class OCSQFOfflineScheduler:
    def __init__(self, G, hnodes,
                 T_cycle=1, Hyper_cycle=60000,
                 queue_num=4, queue_length=10):
        # 拓扑和参数
        self.G = G
        self.hnodes = hnodes
        self.T_cycle = T_cycle
        self.Hyper_cycle = Hyper_cycle
        self.total_cycle_num = int(Hyper_cycle / T_cycle)
        self.queue_num = queue_num
        self.queue_length = queue_length

        # 出度最大值，用于状态 padding
        # self.max_outdeg = max(len(list(G.successors(sw)))
        #                       for sw in G.nodes if not sw.startswith('h'))
        self.max_outdeg = 9

        # 初始化“全局时序队列”数据结构
        # self.queue[t][sw][nbr] = 长度 queue_num*queue_length 的 0/1 列表
        self._init_empty_queue()

    def _init_empty_queue(self):
        # 和原脚本里那段 queue = [] 完全一样
        self.queue = []
        # nodes = [n for n in self.G.nodes if not str(n).startswith('h')]
        # for t in range(self.total_cycle_num+5):
        #     per_node = {}
        #     for sw in nodes:
        #         per_node[sw] = {}
        #         for u,v,_ in self.G.out_edges(sw, data='weight'):
        #             per_node[sw][v] = [0] * (self.queue_num * self.queue_length)
        #     self.queue.append(per_node)

    def generate_flow(self, num):
        # 把原脚本的 generate_flow 拷进去，返回 FLOW 对象列表
        flow_list_show = []
        flow_list_obj = []
        for i in range(num):
            new_flow = FLOW()
            new_flow.id = i + 1
            new_flow.ddl = random.randint(150,200)# random.choice(range(220,280))  # ms
            new_flow.sn_dn = generate_sn_dn(self.hnodes)
            new_flow.pf = random.choice([6,12,18]) # ms
            new_flow.pkt_num = random.choice([1, 2, 3])  # 每个流只有1-3个数据包
            if i % 2 == 0:
                new_flow.t_start_ms = random.choice(tuple(range(6, 15001, 6)))
            else:
                new_flow.t_start_ms = 0
            new_flow.t_life_ms = 60000
            tup_flow = (new_flow.id, new_flow.sn_dn[0], new_flow.sn_dn[1], new_flow.pf, new_flow.pkt_num)  # 流的信息，id、源节点和目的节点，period和数据包个数
            flow_list_obj.append(new_flow)  # 列表里记录每条流对象，记录每条流的所有信息
            flow_list_show.append(tup_flow)  # 同上，只不过每条流只记录5个信息
            # print ("第%d条流的参数为"%(i+1), flow_list_show[i])
        return flow_list_obj  # 对象里含 id, ddl, sn_dn, pf, pkt_num

    # def generate_flow_3i2m(self,
    #                        num_initial=3, num_middle=2,
    #                        ddl_range=(220,280), # (220, 280),  # ms
    #                        pf_choices=(6, 12, 18),  # ms
    #                        pkt_choices=(1, 2, 3),
    #                        middle_starts_ms=tuple(range(6, 61, 6)),  # {6,12,...,15000} ms
    #                        life_ms=60000  # 每条流的生命周期，按需改
    #                        ):
    #     """
    #     返回 FLOW 对象列表。每条 FLOW 新增两个字段：
    #       - t_start_ms: 绝对起始时间（ms）；初始流=0，中途流∈{6,12,...,15000}
    #       - t_life_ms:  生命周期（ms）；用来决定何时停止按周期注入
    #     其余字段与你现有代码一致：id, ddl, jitter, sn_dn, pf, pkt_num
    #     另外保留 'offset' 作为 CSQF 每周期内的相位（与你现有 schedule() 兼容）
    #     """
    #     flow_list_obj = []
    #
    #     def _make_flow(fid: int, start_ms: int):
    #         f = FLOW()
    #         f.id = fid
    #         f.ddl = random.randint(ddl_range[0], ddl_range[1])  # ms
    #         # f.jitter = 0.1
    #         f.sn_dn = generate_sn_dn(self.hnodes)  # (src_host, dst_host)
    #         f.pf = random.choice(pf_choices)  # ms
    #         f.pkt_num = random.choice(pkt_choices)
    #         # —— 新增：绝对起始时间 & 生命周期（供运行期注入使用）——
    #         f.t_start_ms = int(start_ms)
    #         f.t_life_ms = int(life_ms)
    #
    #         # —— 保留“每周期内相位 offset”以兼容你现有的 schedule() ——
    #         #    （注意：这是周期内的相位，不是绝对起始时间）
    #         # max_off = max(1, int(f.pf / self.T_cycle))  # T_cycle 与你项目一致，如 0.1ms
    #         # f.offset = random.randrange(max_off) * self.T_cycle  # 与你的 schedule() 用法对齐
    #         f.offset = start_ms
    #         return f
    #
    #     fid = 1
    #     # 3 条“初始流”——绝对起始时间 = 0 ms
    #     for _ in range(num_initial):
    #         flow_list_obj.append(_make_flow(fid, start_ms=0))
    #         fid += 1
    #
    #     # 2 条“中途流”——在集合 {6,12,...,15000} ms 中随机选择起始时间
    #     for _ in range(num_middle):
    #         start_ms = random.choice(middle_starts_ms)
    #         flow_list_obj.append(_make_flow(fid, start_ms=start_ms))
    #         fid += 1
    #
    #     return flow_list_obj

# -----------------------
# Online Environment: TSNEnv
# -----------------------
class TSNEnv:
    """
    Multi-agent TSN environment with anomalies.
    Agents: one per switch node.
    State (per agent):
      - per (out_link, queue)  两个特征：(min_block_idx, legal_block_count)
        -> min_block_idx/Qmax ∈ [0,1], legal_block_count/Qmax ∈ [0,1]
      - 全局标量: elapsed_time, rem_delay, slack
    Action (per agent): flattened index = link_idx * Qmax + block_start_idx
    Reward:
      - instant: (slack_{t+1} - slack_t) - alpha * hop_latency
      - early stop: if slack<0 at any hop, give -R_MISS and mark flow done
      - terminal: +R_HIT if flow meets deadline, else -R_MISS
    """

    def __init__(self, G_static, flows, offline_scheduler,snapshotter=None,
                 alpha=0.01, R_HIT=20, R_MISS=10,
                 anomaly_prob=0.1):
        # self.G = G
        self.G = G_static
        self.snapshotter = snapshotter  # <—— 新增：可为 None（静态图），或 Snapshotter 实例
        self.flows = flows
        self.offline = offline_scheduler
        self.switches = [n for n in G_static.nodes if not str(n).startswith('h')]
        self.alpha = alpha
        self.R_HIT = R_HIT
        self.R_MISS = R_MISS

        # 与论文式(18)对齐的权重
        self.rho_c = 4.0  # ρc：时延代价权重
        self.sigma_c = 0.5  # σc：队列代价权重

        # 距离势函数塑形（可设大，但它不是代价项）
        self.eta_dist = 5.0  # 你原来是 *10，可先从 4 起

        self.anomaly_prob = anomaly_prob
        # 用来决定在哪个 global_slot 触发一次异常
        # 每次 reset() 时都会重置下面两个属性：
        self.anomaly_triggered = False
        self.anomaly_slot = 0
        # self.scheduled = self.offline.schedule(flows)
        self.scheduled = flows # self.offline.generate_flow(100)
        # self.runtime_queue = self.offline.queue.copy()
        # per-link queue capacity
        self.Qmax = self.offline.queue_num * self.offline.queue_length
        # number of actions per neighbor = Qmax
        #  一个距离上界，用来替换 inf
        # 这里简单取节点数 * 最大链路权重
        # all_pairs = nx.all_pairs_dijkstra_path_length(G, weight='weight')
        # self.max_dist = max(d for _, dist_dict in all_pairs for d in dist_dict.values()
        #                     if d < float('inf'))
        self.global_slot = 0  # 新增：离散时隙计数器，每次 step() ++
        self.failed_count = 0
        self.T_cycle = 1
        self.fail_stats = {"no_slots": 0, "loop": 0, "slack": 0, "bad_host": 0, "link_fail": 0}
        self.t_slot = []
        self.banned_edges = {}  # dict[(u,v)] = until_slot（含）
        self.flow_delay = defaultdict(float)  # 每个流累计时延（ms）
        self.flow_delay_success = []  # 成功流的端到端时延列表（ms）
        self.reset()

        # print("Scheduled flow IDs:", [f.id for f in self.scheduled])

    def _init_runtime(self):
        # 初始化在线态
        self.link_delays = {
            e: self.G.edges[e]['weight']
            for e in self.G.edges
        }

        # 初始化 runtime_queue，先全 0
        self.runtime_queue = {}
        Qmax = self.offline.queue_num * self.offline.queue_length
        # port_registry = self._build_port_registry(1.0, 1.0)
        # total_slots = self.offline.total_cycle_num + 5  # 你原来的 +5 缓冲
        #
        # for t in range(total_slots):
        #     per_node = {}
        #     for sw in self.switches:  # 非 'h' 节点（卫星=交换机）
        #         per_node[sw] = {}
        #         # 给“可能出现过的所有邻居”预分配队列
        #         for nbr in port_registry[sw]:
        #             # t < anomaly_slot 用离线占位，之后全 0（按你之前逻辑）
        #             if t < self.anomaly_slot:
        #                 # 注意：离线表里不一定有这个 nbr（因为离线时未出现）
        #                 if (t < len(self.offline.queue) and
        #                         sw in self.offline.queue[t] and
        #                         nbr in self.offline.queue[t][sw]):
        #                     per_node[sw][nbr] = self.offline.queue[t][sw][nbr].copy()
        #                 else:
        #                     per_node[sw][nbr] = [0] * Qmax
        #             else:
        #                 per_node[sw][nbr] = [0] * Qmax
        #     self.runtime_queue.append(per_node)

        tem_flow_states = []

        # self.finish_slot = {}  # map: flow.id -> int
        for f in self.scheduled:
            # 计算这条流每一趟的结束时隙
            # self.finish_slot[f.id] = f.finish_slot

            # 得到这条流的趟数
            max_v = int((f.t_life_ms-f.t_start_ms-f.ddl) / f.pf)

            for v in range(max_v):
                total_slot_v = int((f.offset + v * f.pf) / self.T_cycle)
                elapsed = f.offset + v * f.ddl
                if self.anomaly_slot <= total_slot_v:  # 产生异常的时隙小于等于这条流的offset，那把这条流所有趟都应该放入flow_states
                    # 每趟流的开始时间，对这个时间去一个网络拓扑快照生成这条流最开始安排的路径
                    t = f.t_start_ms + v * f.pf
                    G,_ = self.snapshotter.graph_at(t*0.001)
                    f.path = generate_path(G, f.sn_dn[0], f.sn_dn[1])
                    flow_state = {
                        'pos': f.path[1],
                        'dst': f.sn_dn[1],  # 新增：流的目的节点
                        'hop': 1,
                        # 'elapsed': elapsed + 0.004,
                        'elapsed': G.edges[f.path[0], f.path[1]]['weight'],
                        # 'deadline': f.ddl * (v + 1),
                        'deadline': f.ddl,
                        'path': f.path,
                        'pkt_num': f.pkt_num,
                        'QO': f.QO,
                        'RO': f.RO,
                        'offset': f.offset + v * f.ddl
                    }
                    tem_flow_states.append(flow_state)
                    # 因为要直接跳过初始主机所以要+1
                    self.t_slot.append(total_slot_v+int(G.edges[f.path[0], f.path[1]]['weight']/self.T_cycle))

            # # 得到这条流的趟数
            # max_v = int(36 / f.pf)
            #
            # for v in range(max_v):
            #     total_slot_v = int((f.offset + v * f.pf) / self.T_cycle)
            #     elapsed = f.offset + v * f.ddl
            #     if self.anomaly_slot <= total_slot_v:  # 产生异常的时隙小于等于这条流的offset，那把这条流所有趟都应该放入flow_states
            #         # 每趟流的开始时间，对这个时间去一个网络拓扑快照生成这条流最开始安排的路径
            #         t = f.offset + v * f.pf
            #         G, _ = self.snapshotter.graph_at(t * 0.001)
            #         f.path = generate_path(G, f.sn_dn[0], f.sn_dn[1])
            #         flow_state = {
            #             'pos': f.path[1],
            #             'dst': f.sn_dn[1],  # 新增：流的目的节点
            #             'hop': 1,
            #             # 'elapsed': elapsed + 0.004,
            #             'elapsed': G.edges[f.path[0], f.path[1]]['weight'],
            #             # 'deadline': f.ddl * (v + 1),
            #             'deadline': f.ddl,
            #             'path': f.path,
            #             'pkt_num': f.pkt_num,
            #             'QO': f.QO,
            #             'RO': f.RO,
            #             'offset': f.offset + v * f.ddl
            #         }
            #         tem_flow_states.append(flow_state)
            #         # 因为要直接跳过初始主机所以要+1
            #         self.t_slot.append(total_slot_v + int(G.edges[f.path[0], f.path[1]]['weight'] / self.T_cycle))


        # 新增：每流的“已访问交换机”集合，初始只有源节点
        self.visited = []
        for f in tem_flow_states:
            visit = []
            for i in range(f['hop'] + 1):
                visit.append(f['path'][i])
            self.visited.append(visit)

        self.flow_states = {}
        # 把flow_states变成字典
        for fid in range(len(tem_flow_states)):
            self.flow_states[fid] = tem_flow_states[fid]
            # self.t_slot.append(0)
        print(len(self.flow_states))

    def _ensure_bitmap(self, slot, sw, nbr):
        return (self.runtime_queue
                .setdefault(slot, {})
                .setdefault(sw, {})
                .setdefault(nbr, [0] * self.Qmax))

    def reset(self):
        # reset link delays

        # 只重置 runtime 相关，不再调用 schedule()
        self._init_runtime()
        # self.global_slot = 0  # 新增：离散时隙计数器，每次 step() ++
        self.failed_count = 0
        self.flow_delay.clear()
        self.flow_delay_success.clear()
        # return self._get_obs_for(0)
        return

    def _build_port_registry(self, horizon_s: float, dt_s: float):
        """
        扫描 [0, horizon_s] 内的快照，收集每个交换机 sw 可能出现过的所有后继 nbr。
        返回: dict[str, set[str]]，比如 {'0': {'1','2','h3',...}, ...}
        """
        reg = {sw: set() for sw in self.switches}  # self.switches 是非 'h' 开头的卫星
        t = 0.0
        while t <= horizon_s:
            Gt,_ = self.snapshotter.graph_at(t)  # 你已有的生成快照函数：t(秒)->DiGraph
            for sw in self.switches:
                for nbr in Gt.successors(sw):
                    reg[sw].add(nbr)
            t += dt_s
        return reg

    def _has_path_quick(self, G, src, dst) -> bool:
        """不算最短路，纯连通性判定；无路返回 False。"""
        try:
            return nx.has_path(G, src, dst)
        except Exception:
            return False

    def _safe_dijkstra_len(self, G, src, dst, weight='weight') -> float:
        """安全版最短路：无路返回 inf，避免抛出 NetworkXNoPath。"""
        try:
            return nx.dijkstra_path_length(G, src, dst, weight=weight)
        except (nx.NetworkXNoPath, KeyError):
            return float('inf')

    def _fail_flow_now(self, fid: int, reason: str = "no_path") -> None:
        """立刻将该流标记为失败并从 env.flow_states 中移除。"""
        self.failed_count += 1
        self.fail_stats[reason] = self.fail_stats.get(reason, 0) + 1
        self.flow_states.pop(fid, None)

    # def get_queue_bitmap(self, sw, nbr, slot):
    #     # 得到目标端口当前slot的队列资源快照
    #     Q = self.offline.queue_num * self.offline.queue_length
    #     bmp = [0] * Q
    #
    #     # off = self.offline.get_offline_bitmap(sw, nbr)
    #     rt = self.runtime_queue[slot][sw][nbr]
    #     for i in range(Q):
    #         bmp[i] = rt[i]
    #     return bmp

    def get_queue_bitmap(self, sw, nbr, slot):
        Q = self.Qmax
        return (self.runtime_queue.get(slot, {})
                                 .get(sw, {})
                                 .get(nbr, [0] * Q))

    # 一个小工具：给 fid 返回（Gt, t_sec）
    # def _graph_for_fid(self, fid):
    #     if self.snapshotter is None:
    #         return self.G, 0.0
    #     t_sec = self.t_slot[fid] * self.T_cycle * 0.001  # 也可加上该流 offset（若你保留 offset）
    #     Gt, _ = self.snapshotter.graph_at(t_sec)
    #     return Gt, t_sec
    def _graph_for_fid(self, fid):
        """
        你原本的“按 fid 的 t_slot 取快照”的函数里，调用 _apply_bans_to_graph。
        假设你已有 snapshotter.graph_at(t_sec) -> (G, hnodes)
        """
        t_sec = self.t_slot[fid] * self.offline.T_cycle  # 或你现有的换算
        Gt, hnodes = self.snapshotter.graph_at(t_sec* 0.001)
        # 应用禁边（按当前流实例的 slot）
        self._apply_bans_to_graph(Gt, self.t_slot[fid])
        return Gt, hnodes

    def _get_obs_for(self, fid: int) -> np.ndarray:
        st = self.flow_states[fid]
        sw = st['pos']
        Gt, _ = self._graph_for_fid(fid)
        nbrs = list(Gt.successors(sw))
        # dis = nx.dijkstra_path_length(self.G, st['pos'], st['dst'], weight='weight')
        dis = self._safe_dijkstra_len(Gt, st['pos'], st['dst'], weight='weight')
        obs_list = []
        # —— per-link × per-queue 特征（同你原来逻辑） ——
        for v in nbrs:
            feats = []
            bitmap = self.get_queue_bitmap(sw, v, self.t_slot[fid])  # list of 0/1 length Qmax
            # k = 0
            # dis_next = nx.dijkstra_path_length(self.G, v, st['dst'], weight='weight')
            dis_next = self._safe_dijkstra_len(Gt, v, st['dst'], weight='weight')
            # # pick a representative flow currently at sw for pkt_num
            # for fid, fs in self.flow_states.items():
            #     if fs['pos'] == sw:
            #         k = fs['pkt_num']
            #         break
            k = self.flow_states[fid]['pkt_num']
            # find legal block starts
            LB = [
                p % self.Qmax
                for p in range(((self.t_slot[fid] % self.offline.queue_num) + 1) * self.offline.queue_length,
                               ((self.t_slot[fid] % self.offline.queue_num) + 1) * self.offline.queue_length + (
                                           self.offline.queue_num - 1) * self.offline.queue_length - k + 1)
                if all(bitmap[(p + i) % self.Qmax] == 0 for i in range(k))
            ]
            # 判断是否有合法位置
            if LB:
                feats.append(1)
            else:
                feats.append(0)

            # 判断是否有环路
            visited_flag = 0.0
            if st['pos'] == sw and v in self.visited[fid]:
                visited_flag = 1.0

            feats.append(visited_flag)

            if math.isinf(dis) and math.isinf(dis_next):
                delta = 0.0
            elif math.isinf(dis):
                delta = +100.0  # 自己无路但邻居有路 → 强烈鼓励
            elif math.isinf(dis_next):
                delta = -100.0  # 邻居无路 → 强烈惩罚
            else:
                delta = dis - dis_next
            feats.append(max(-100.0, min(100.0, delta)) / 8)
            # feats.append(float(dis_next - dis) / 4)
            # if dis_next < dis:
            #     feats.append(float(dis-dis_next)/10)
            # else:
            #     feats.append(float(dis_next-dis)/10)

            is_dest = 1.0 if v == st['dst'] else 0.0
            feats.append(is_dest)
            is_host = 1.0 if v.startswith('h') else 0.0
            feats.append(is_host)

            # 占用率：0~1
            util = float(np.sum(bitmap)) / float(self.Qmax)
            feats.append(util)

            obs_list.append(feats)

        # pad to max_outdeg:
        pad = self.offline.max_outdeg - len(nbrs)
        if pad > 0:
            for i in range(pad):
                obs_list.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(obs_list, dtype=np.float32)


    # def estimate_remaining_delay_flow(self, fs):
    #     """最简单的下界：剩余跳数 × 平均链路时延"""
    #     src, dst = fs['pos'], fs['dst']
    #     try:
    #         # 注意 Graph 存的是 (u,v,weight) 初始值，但在运行时我们可能已更新 self.link_delays
    #         return nx.dijkstra_path_length(
    #             self.G, src, dst,
    #             weight=lambda u, v, data: self.link_delays.get((u, v), data.get('weight', 1.0))
    #         )
    #     except nx.NetworkXNoPath:
    #         # 如果真没有路，返回无穷
    #         return float('inf')

    def ban_edges(self, edges, until_slot: int, start_slot: int = None):
        """
        支持同一条 edge 有多个禁用窗口：banned_edges[(u,v)] = [(s1,e1), (s2,e2), ...]
        兼容旧格式：tuple 或 int 会自动升级为列表。
        """
        if start_slot is None:
            start_slot = self.global_slot
        s, e = int(start_slot), int(until_slot)
        for edge in edges:
            win = self.banned_edges.get(edge)
            if not win:
                self.banned_edges[edge] = [(s, e)]
            else:
                # 兼容旧格式
                if isinstance(win, tuple):
                    win = [win]
                elif isinstance(win, int):
                    win = [(0, win)]
                win.append((s, e))
                self.banned_edges[edge] = win

    def _apply_bans_to_graph(self, Gt, cur_slot: int):
        """
        在当前 slot，把仍处于任一禁用窗口内的边从快照图里删除。
        """
        if not self.banned_edges:
            return
        to_remove = []
        for (u, v), win in self.banned_edges.items():
            # 归一化为列表
            if isinstance(win, tuple):
                windows = [win]
            elif isinstance(win, int):
                windows = [(0, win)]
            else:
                windows = win
            # 只要有任意一个窗口命中就移除
            if any(s <= cur_slot <= e for (s, e) in windows) and Gt.has_edge(u, v):
                to_remove.append((u, v))
        if to_remove:
            Gt.remove_edges_from(to_remove)

    def _edge_banned_overlap(self, edge, a_slot: int, b_slot: int) -> bool:
        """
        判断 [a_slot, b_slot] 是否与该 edge 的任意禁用窗口重叠。
        """
        win = self.banned_edges.get(edge)
        if not win:
            return False
        if isinstance(win, tuple):
            windows = [win]
        elif isinstance(win, int):
            windows = [(0, win)]
        else:
            windows = win
        for (s, e) in windows:
            if not (b_slot < s or a_slot > e):
                return True
        return False

    def _simulate_anomaly(self):
        d = True
        e = random.choice(list(self.link_delays.keys()))
        while d:
            if e[0][0] != 'h' and e[1][0] != 'h':
                self.link_delays[e] = float('inf')
                d = False
            else:
                e = random.choice(list(self.link_delays.keys()))
        # kind = random.choice([1, 2, 3])
        # if kind == 1:
        #     # increase random link delay
        #     e = random.choice(list(self.link_delays.keys()))
        #     self.link_delays[e] *= random.uniform(0.01, 0.04)
        # elif kind == 2:
        #     # break a random link
        #     e = random.choice(list(self.link_delays.keys()))
        #     self.link_delays[e] = float('inf')
        # else:
        #     # add random new link
        #     u, v = random.sample(self.switches, 2)
        #     self.link_delays[(u, v)] = random.uniform(0.004, 0.004)

    def get_average_delay_ms(self) -> float:
        """成功流的平均端到端时延（ms）。若无成功流，返回 0.0。"""
        if not self.flow_delay_success:
            return 0.0
        return float(sum(self.flow_delay_success) / len(self.flow_delay_success))

    def _apply_action(self, fid: int, sw: str, act: int, obs_for: dict):
        """
        为 flow id=fid，在交换机 sw 上执行动作 act。
        返回 (t_delay, reward_delta, done_flag) 三元组。
        """
        fs = self.flow_states[fid]
        Gt, _ = self._graph_for_fid(fid)
        nbrs = list(Gt.successors(sw))

        # 解码动作：仅 link_idx = port 索引
        link_idx = act
        if link_idx < len(nbrs):
            nxt = nbrs[link_idx]
        else:
            # 超出范围直接尝试发往目的节点
            nxt = fs['dst']

        pkt_num = fs['pkt_num']

        # —— 队列资源检查 ——（无可用位置则早停）
        if obs_for[fid][link_idx][0] == 0:
            self.failed_count += 1
            self.fail_stats["no_slots"] += 1
            return 0.0, - self.R_MISS * 0.6, True

        # —— 环路检查 ——（命中环路则早停）
        if obs_for[fid][link_idx][1] == 1:
            self.failed_count += 1
            self.fail_stats["loop"] += 1
            return 0.0, - self.R_MISS, True
        # 记录访问
        self.visited[fid].append(nxt)

        # —— 计算“可用块起点”但先不写 bitmap（先做断链重叠判定）——
        bmp = self.get_queue_bitmap(sw, nxt, self.t_slot[fid])
        legal_slots = [
            p % self.Qmax
            for p in range(
                ((self.t_slot[fid] % self.offline.queue_num) + 1) * self.offline.queue_length,
                ((self.t_slot[fid] % self.offline.queue_num) + 1) * self.offline.queue_length
                + (self.offline.queue_num - 1) * self.offline.queue_length - pkt_num + 1
            )
            if all(bmp[(p + i) % self.Qmax] == 0 for i in range(pkt_num))
        ]
        if len(legal_slots) == 0:
            self.failed_count += 1
            self.fail_stats["no_slots"] += 1
            return 0.0, - self.R_MISS * 0.6, True

        blk_idx = legal_slots[0]
        queue_id = blk_idx // self.offline.queue_length
        send_q = self.t_slot[fid] % self.offline.queue_num
        wait_cycles = (queue_id - send_q) % self.offline.queue_num

        # 队列内位置等待对应的“子槽”时间
        pos_in_q = blk_idx % self.offline.queue_length

        # 物理链路时延（单位与全局一致；你前面已完成单位统一的话，这里直接使用即可）
        link_d = Gt[sw][nxt]['weight']
        # 以“槽”为粒度的链路传输跨度
        link_slots = max(1, int(math.ceil(link_d / self.T_cycle)))

        now_slot = self.t_slot[fid]
        start_tx_slot = now_slot + wait_cycles
        end_tx_slot = start_tx_slot + link_slots - 1

        # —— 断链窗口重叠判定：两种情况都要 fail ——
        # ① 排队阶段（预约后尚未开始发）：[now_slot, start_tx_slot-1]
        if start_tx_slot - 1 >= now_slot and self._edge_banned_overlap((sw, nxt), now_slot, start_tx_slot - 1):
            self.failed_count += 1
            self.fail_stats["link_fail"] += 1
            return 0.0, - self.R_MISS, True

        # ② 传输阶段（in-flight）：[start_tx_slot, end_tx_slot]
        if self._edge_banned_overlap((sw, nxt), start_tx_slot, end_tx_slot):
            self.failed_count += 1
            self.fail_stats["link_fail"] += 1
            return 0.0, - self.R_MISS, True

        # —— 走到这里说明安全：可以真正写入 bitmap 占位，并推进时钟 ——
        # 先把“预约等待阶段”的每个 slot 的占位写入（与原逻辑一致）
        for p in range(blk_idx, blk_idx + pkt_num):
            if ((blk_idx // self.offline.queue_length) - (
                    self.t_slot[fid] % self.offline.queue_num)) % self.offline.queue_num >= 1:
                for q in range(0, ((blk_idx // self.offline.queue_length) - (
                        self.t_slot[fid] % self.offline.queue_num)) % self.offline.queue_num):
                    bmp_q = self._ensure_bitmap(self.t_slot[fid] + q, sw, nxt)
                    bmp_q[p % self.Qmax] = 1

        u = float(obs_for[fid][link_idx][5])  # 0~1
        reward = 1-self.sigma_c * (u ** 2)
        reward += obs_for[fid][link_idx][2] * self.eta_dist

        # 是否邻居就是目的主机：即便如此也要走上面的断链检查（你现在已经做了）
        if obs_for[fid][link_idx][3] == 1:
            reward += self.R_HIT
            self.flow_delay_success.append(self.flow_delay[fid])
            return link_d, reward, True

        # —— 计算总时延并推进时钟 ——
        base_wait = wait_cycles * self.offline.T_cycle
        intra_wait = pos_in_q * (self.offline.T_cycle / self.offline.queue_length)
        t_delay = base_wait + intra_wait + link_d
        # —— 累计到该流的端到端时延 ——
        self.flow_delay[fid] += float(t_delay)

        self.t_slot[fid] += wait_cycles
        self.t_slot[fid] += link_slots

        # 更新位置
        fs['pos'] = nxt
        return t_delay, reward, False

    def step(self, actions, obs_for):
        """
        actions: dict {switch: action_int}
        return obs, rewards, done, info
        """
        # # maybe anomaly
        # if random.random() < self.anomaly_prob:
        #     self.anomaly_prob = 0.0
        #     self._simulate_anomaly()

        rewards = {fid: 0.0 for fid in self.flow_states.keys()}
        done = False
        finished_ids = []
        for fid, st in self.flow_states.items():
            cur = st['pos']
            if cur != st['dst']:

                # nbrs = list(self.G.successors(cur))

                act = actions[fid]
                t_delay, r_delta, failed = self._apply_action(fid, cur, act, obs_for)

                # 2) 执行 hop：更新 elapsed、pos、hop
                prev_slack = st['deadline'] - st['elapsed']
                st['elapsed'] += t_delay
                st['hop'] += 1

                # 3) 计算 slack 并做 Early‐stop（≤0）
                curr_slack = st['deadline'] - st['elapsed']
                if curr_slack < 0:
                    # 立刻给一个惩罚，然后删流
                    rewards[fid] -= self.R_MISS
                    self.failed_count += 1
                    self.fail_stats["slack"] += 1
                    finished_ids.append(fid)
                    continue

                # r = (prev_slack - curr_slack) - self.alpha * t_delay
                # print("及时奖励：",r)
                rewards[fid] -= self.rho_c * (t_delay / st['deadline'])
                # self.global_slot += 1
                rewards[fid] += r_delta
                if failed:
                    finished_ids.append(fid)
                    continue

                # 4) 如果是交换机 hop，再给即时奖励。如果不是交换机是主机，
                nxt = st['pos']
                if nxt.startswith('h'):
                    # 下一跳是主机且不是目标主机，直接Early_stop
                    rewards[fid] -= self.R_MISS * 0.8
                    self.failed_count += 1
                    self.fail_stats["bad_host"] += 1
                    finished_ids.append(fid)
                    st['pos'] = nxt

        # 做完这一步后返回下一步前的obs
        next_obs = {}
        for fid in self.flow_states.keys():
            if fid not in finished_ids:
                # print(fid)
                # print(self.flow_states[fid])
                obs = self._get_obs_for(fid)
            else:
                # 已完成的流，用一个“同样结构但全 0”的占位 obs
                # 1) 知道每个端口特征维度：
                feat_dim = 6
                # 2) 知道最大端口数：
                max_ports = self.offline.max_outdeg
                # 3) 构造一个列表：max_ports 个全 0 向量
                obs = [np.zeros(feat_dim, dtype=np.float32)
                       for _ in range(max_ports)]
            next_obs[fid] = obs

        rewards = {fid: rw / 10 for fid, rw in rewards.items()}
        return next_obs, rewards, finished_ids, {}

# -----------------------

class DuelingDQN(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # 公共特征提取层
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # 优势流
        self.adv_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)                   # (B, 1)
        a = self.adv_stream(f)                     # (B, act_dim)
        # 合并：Q = V + (A - mean(A))
        return v + a - a.mean(dim=1, keepdim=True)

# DQN Agent
# -----------------------
class DQNAgent:
    def __init__(self, obs_dim, act_dim,
                 lr=5e-4, gamma=0.99,
                 buffer_size=50000, batch_size=128,
                noise_start = 0.05,  # 初始噪声强度 σ
                noise_end = 0.01,  # 最小噪声强度
                noise_decay = 0.9999  # 每步衰减比例
                ):
        self.obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_var = np.ones(obs_dim, dtype=np.float32)
        self.obs_std = np.sqrt(self.obs_var)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.batch_size = batch_size
        # self.memory = PrioritizedReplayBuffer(buffer_size, alpha=0.6)

        self.beta_start = 0.4
        self.beta_frames = 100000
        self.memory = PrioritizedReplayBuffer(
                    buffer_size,
                    alpha = 0.6,
                    beta = self.beta_start,
                )


        self.epsilon = 1.0
        self.noise_sigma   = noise_start
        self.noise_end     = noise_end
        self.noise_decay   = noise_decay
        # self.eps_decay = 0.995
        # self.eps_min = 0.05
        self.loss_history = []
        self.device = torch.device("cpu")
        # 创建网络并搬到 device
        self.net = DuelingDQN(obs_dim, act_dim).to(self.device)
        self.target_net = DuelingDQN(obs_dim, act_dim).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def select_action(self, obs: list[np.ndarray], valid_mask):
        """
        obs: List[np.ndarray]，长度 ≤ act_dim，每个元素是 feat_dim 维向量
        valid_mask: np.ndarray(bool)，长度 = act_dim
        """

        # 1) 重建定长状态向量：pad 到 (act_dim, feat_dim)，再 flatten
        feat_dim = self.obs_dim // self.act_dim
        state_arr = np.zeros(self.obs_dim, dtype=np.float32)

        for i, feat in enumerate(obs):
            start = i * feat_dim
            state_arr[start:start + feat_dim] = feat

        # 2) 标准化 & to(device)
        norm = (state_arr - self.obs_mean) / (self.obs_std + 1e-6)
        x = torch.tensor(norm, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, obs_dim)

        # 3) 一次过网络，得到全端口 Q 值 (act_dim,)

        with torch.no_grad():
            q_vals = self.net(x)[0].cpu().numpy()
        noise = np.random.normal(
            loc=0.0,
            scale=self.noise_sigma,
            size=q_vals.shape
        )
        q_vals = q_vals + noise

        # 4) mask 掉不存在的端口
        q_vals[~valid_mask] = -np.inf

        # 5) ε-贪心
        choices = np.nonzero(valid_mask)[0]
        if choices.size == 0:
            # 防止万一没有合法动作，兜底选 0
            return 0

        if random.random() < self.epsilon:
            # 随机探索
            action = int(np.random.choice(choices))

        else:
            # 贪心利用：选 Q 值最大的动作
            action = int(np.argmax(q_vals))
            # return int(np.nanargmax(q_vals))

        self.noise_sigma = max(self.noise_end, self.noise_sigma * self.noise_decay)

        return action

    def store(self, s, a, r, s2, done):

        self.memory.add(
            obs = s,
            act = a,
            rew = r,
            next_obs = s2,
            done = done
        )

    def train_step(self, frame_idx):
        if len(self.memory) < self.batch_size:
            return
        states_arr, actions_arr, rewards_arr, next_states_arr, dones_arr, idxs, is_weights = self.memory.sample(self.batch_size)
        is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        states = states_arr  # shape (B, obs_dim)
        actions = actions_arr  # shape (B,)
        rewards = rewards_arr  # shape (B,)
        next_states = next_states_arr  # shape (B, obs_dim)
        dones = dones_arr  # shape (B,)

        states_np = np.stack(states).astype(np.float32)  # (B, obs_dim)
        next_states_np = np.stack(next_states).astype(np.float32)

        alpha = 1e-3
        for s in states_arr:  # s 是 shape=(obs_dim,) 的 numpy 数组
            delta = s - self.obs_mean
            self.obs_mean = (1 - alpha) * self.obs_mean + alpha * s
            self.obs_var = (1 - alpha) * self.obs_var + alpha * (delta * delta)
        self.obs_std = np.sqrt(self.obs_var + 1e-6)

        s_v = torch.from_numpy(states_np).to(self.device)  # float32
        s2_v = torch.from_numpy(next_states_np).to(self.device)
        a_v = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        r_v = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_v = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_vals = self.net(s_v).gather(1, a_v).squeeze(-1)
        # 取下一个状态的 greedy 动作
        with torch.no_grad():
            next_q_vals = self.net(s2_v)  # online net 评估动作
            next_actions = next_q_vals.argmax(dim=1, keepdim=True)  # 选动作
            q_next_target = self.target_net(s2_v).gather(1, next_actions).squeeze(1)
        q_target = r_v + self.gamma * q_next_target * (1 - done_v)
        td_errors = (q_target - q_vals).abs().detach().cpu().numpy()
        self.memory.update_priorities(idxs, td_errors + 1e-6)

        loss = F.smooth_l1_loss(q_vals, q_target, reduction='none')
        # 再乘以 is_weights：
        loss = (loss * is_weights).mean()

        loss_val = loss.item()
        self.loss_history.append(loss_val)

        self.optimizer.zero_grad()

        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        self.optimizer.step()
        if frame_idx % 1000 == 0:
            avg = np.mean(self.loss_history[-1000:])
            print(f"[Step {frame_idx}] 平均 TD Loss (上 1000 步) = {avg:.4f}")
        # self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())


