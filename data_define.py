import math
import networkx as nx
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tianshou.data import PrioritizedReplayBuffer
# 该图有31个主机，13个交换机，53条双向边，所以有106条单向边

device = torch.device("cpu")
print(f"Using device: {device}")

# 主机
hnodes = ['h0','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12'
    ,'h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23','h24'
    ,'h25','h26','h27','h28','h29','h30']

# 总结点数
nodes = ['0','1','2','3','4','5','6','7','8','9','10','11','12'
    ,'h0','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12'
    ,'h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23','h24'
    ,'h25','h26','h27','h28','h29','h30']

# 边
edges = [('h0','0',4),('0','h0',4),('h1','0',4),('0','h1',4),('h2','0',4),('0','h2',4)
    ,('h3','1',4),('h4','1',4),('1','h3',4),('1','h4',4),('2','h5',4),('h5','2',4),('2','h6',4)
    ,('h6','2',4),('3','h7',4),('h7','3',4),('3','h8',4),('h8','3',4),('4','h9',4),('h9','4',4)
    ,('5','h10',4),('h10','5',4),('5','h11',4),('h11','5',4),('5','h12',4),('h12','5',4)
    ,('6','h13',4),('h13','6',4),('6','h14',4),('h14','6',4),('7','h15',4),('h15','7',4)
    ,('7','h16',4),('h16','7',4),('7','h17',4),('h17','7',4),('7','h18',4),('h18','7',4)
    ,('7','h19',4),('h19','7',4),('8','h20',4),('h20','8',4),('8','h21',4),('h21','8',4)
    ,('9','h22',4),('h22','9',4),('9','h23',4),('h23','9',4),('9','h24',4),('h24','9',4)
    ,('10','h25',4),('h25','10',4),('10','h26',4),('h26','10',4),('11','h27',4),('h27','11',4)
    ,('11','h28',4),('h28','11',4),('12','h29',4),('h29','12',4),('12','h30',4),('h30','12',4)
    ,('0','4',4),('0','8',4),('1','2',4),('1','4',4),('1','5',4),('2','1',4),('2','5',4)
    ,('3','4',4),('3','8',4),('4','0',4),('4','1',4),('4','3',4),('4','5',4),('4','6',4),('4','7',4)
    ,('4','10',4),('5','1',4),('5','2',4),('5','4',4),('5','6',4),('6','4',4),('6','5',4)
    ,('6','8',4),('6','9',4),('7','4',4),('7','8',4),('8','0',4),('8','3',4),('8','6',4)
    ,('8','7',4),('8','9',4),('8','10',4),('8','11',4),('9','6',4),('9','8',4),('9','11',4)
    ,('9','12',4),('10','4',4),('10','8',4),('11','8',4),('11','9',4),('11','12',4),('12','9',4),('12','11',4)]

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_weighted_edges_from(edges[i] for i in range(len(edges)))

class FLOW () :
    def __init__(self):
        self.id = 0
        self.offset = 0
        self.ddl = 0
        self.jitter = 0.0
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

def same_queue_segment(p, k, queue_length):
    return (p // queue_length) == ((p + k - 1) // queue_length)

class OCSQFOfflineScheduler:
    def __init__(self, G, hnodes,
                 T_cycle=0.1, Hyper_cycle=32,
                 queue_num=5, queue_length=10):
        # 拓扑和参数
        self.G = G
        self.hnodes = hnodes
        self.T_cycle = T_cycle
        self.Hyper_cycle = Hyper_cycle
        self.total_cycle_num = int(Hyper_cycle / T_cycle)
        self.queue_num = queue_num
        self.queue_length = queue_length

        # 出度最大值，用于状态 padding
        self.max_outdeg = max(len(list(G.successors(sw)))
                              for sw in G.nodes if not sw.startswith('h'))

        # 初始化“全局时序队列”数据结构
        # self.queue[t][sw][nbr] = 长度 queue_num*queue_length 的 0/1 列表
        self._init_empty_queue()

    def _init_empty_queue(self):
        # 和原脚本里那段 queue = [] 完全一样
        self.queue = []
        nodes = [n for n in self.G.nodes if not str(n).startswith('h')]
        for t in range(self.total_cycle_num+5):
            per_node = {}
            for sw in nodes:
                per_node[sw] = {}
                for u,v,_ in self.G.out_edges(sw, data='weight'):
                    per_node[sw][v] = [0] * (self.queue_num * self.queue_length)
            self.queue.append(per_node)

    def generate_flow(self, num):
        # 把原脚本的 generate_flow 拷进去，返回 FLOW 对象列表
        flow_list_show = []
        flow_list_obj = []
        for i in range(num):
            new_flow = FLOW()
            new_flow.id = i + 1
            new_flow.ddl = random.choice([4,8,16,32])# random.choice(range(220,280))  # ms
            # new_flow.jitter = 0.1
            new_flow.offset = random.randint(0, (new_flow.ddl-2)*10) / 10
            new_flow.sn_dn = generate_sn_dn(self.hnodes)
            new_flow.pf = new_flow.ddl # ms
            new_flow.pkt_num = random.choice([1, 2, 3])  # 每个流只有1-3个数据包
            tup_flow = (new_flow.id, new_flow.sn_dn[0], new_flow.sn_dn[1], new_flow.pf, new_flow.pkt_num)  # 流的信息，id、源节点和目的节点，period和数据包个数
            flow_list_obj.append(new_flow)  # 列表里记录每条流对象，记录每条流的所有信息
            flow_list_show.append(tup_flow)  # 同上，只不过每条流只记录5个信息
            # print ("第%d条流的参数为"%(i+1), flow_list_show[i])
        return flow_list_obj  # 对象里含 id, ddl, sn_dn, pf, pkt_num

    # def generate_flow_3i2m(self,
    #                        num_initial=3, num_middle=2,
    #                        ddl_range=(220, 280),  # ms
    #                        pf_choices=(6, 12, 18),  # ms
    #                        pkt_choices=(1, 2, 3),
    #                        middle_starts_ms=tuple(range(6, 15001, 6)),  # {6,12,...,15000} ms
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

    def schedule(self, flow_list_obj):
        # 把 sche_flow 的主体逻辑提取出来：
        #  1) 生成每条 flow.path
        #  2) 维护 self.queue，给每个 FLOW 填充 flow.RO, flow.QO
        #  3) 返回那些成功调度的 FLOW 列表
        #
        # 注意：每次调用 schedule 前应清空 self.queue
        self._init_empty_queue()
        scheduled = []
        sche_flow_num = 0  # 调度的数据流个数

        for i in range(len(flow_list_obj)):
            flow_list_obj[i].RO = []
            flow_list_obj[i].QO = []
            # start_time = time.time()
            # print(f'第{i + 1}条流调度，参数为：{flow_list_show[i]}')
            result_path = generate_path(self.G, flow_list_obj[i].sn_dn[0], flow_list_obj[i].sn_dn[1])  # 生成该条流的最短路径
            flow_list_obj[i].path = result_path
            result_edges = paths_to_edges(result_path)  # 该最短路径的边
            Max_offset = int(flow_list_obj[i].pf / self.T_cycle)
            succ_tag = 0  # 如果所有周期都能调度，则让succ_tag = 1，然后sche_flow_num+1
            for j in range(0, Max_offset):
                flow_list_obj[i].offset = j * self.T_cycle
                Max_v = int(self.Hyper_cycle / flow_list_obj[i].pf)
                min_Latency = 0.0
                max_Latency = 0.0
                min_Latency_v = 0.0
                max_Latency_v = 0.0
                tag2 = 0  # 用来判断是完成了所有周期调度还是没有完成
                all_Latency = 0
                for v in range(1, Max_v + 1):

                    if v == 1:
                        max_Latency_v = flow_list_obj[i].ddl - (flow_list_obj[i].pkt_num * 0.01 + 0.004) - flow_list_obj[i].offset
                        min_Latency_v = max(max_Latency_v - flow_list_obj[i].jitter, 0)
                    else:
                        max_Latency_v = min(flow_list_obj[i].ddl, min_Latency + flow_list_obj[i].jitter) - (flow_list_obj[i].pkt_num * 0.01 + 0.004) - flow_list_obj[i].offset
                        min_Latency_v = max(max_Latency - flow_list_obj[i].jitter - (flow_list_obj[i].pkt_num * 0.01 + 0.004) - flow_list_obj[i].offset, 0)
                    # jitter_v = max_Latency_v - min_Latency_v
                    # avg_jitter = jitter_v / (len(result_edges)-1)
                    avg_min_Latency_v = min_Latency_v / (len(result_edges) - 1)
                    avg_max_Latency_v = max_Latency_v / (len(result_edges) - 1)

                    # 当前是第几个周期，作为队列的下标索引
                    T_num = int((v - 1) * flow_list_obj[i].pf / self.T_cycle) + int(flow_list_obj[i].offset)
                    queue_output_num = T_num % self.queue_num
                    # 因为我们刚开始就减去了从第一个主机到第一个交换机的延迟
                    # 和那条边的传输延迟，所以我们可以将其余的结点和边看作n-2份，一份的延迟就是位置*10微秒+4微秒
                    LP = [[]]  # 因为k从1开始，所以我把下标为0的LP置为空
                    subscript = [0, ]
                    tag1 = 0  # 用来标记一条流的最后一个数据包在某一跳放到了队列的最后

                    Latency_tem = 0.0  # 用来更新avg_Latency
                    Latency = flow_list_obj[i].pkt_num * 0.01 + 0.004 + flow_list_obj[i].offset  # 用来记录这一周期的延迟

                    for k in range(1, len(result_edges)):  # 初始化
                        subscript.append(0)  # 用来记录每跳的LP_k列表的下标，表示当前这一跳的位置，如果后面因为某些原因回溯到当前这一跳需要重新选择位置时，只需将该值+1就能得到BD第二小的位置
                    k = 1
                    while k < (len(result_edges)):
                        try:
                            len(LP[k])
                        except:
                            LP_k = []
                            # subscript[k] = 0   # 用来记录每跳的LP_k列表的下标，表示当前这一跳的位置，如果后面因为某些原因回溯到当前这一跳需要重新选择位置时，只需将该值+1就能得到BD第二小的位置
                            time_L = []
                            # calculate the legal position
                            # queue[T_num][result_edges[k]]
                            for n in range(flow_list_obj[i].pkt_num,(self.queue_num - 1) * self.queue_length + 1):  # 因为我们假设同一条流的数据包是连续的，所以我们只要算最后一个数据包的发送时间
                                # m为队列资源的下标，用来判断该队列资源是否被使用
                                m = (n - 1 + (queue_output_num + 1) * self.queue_length) % (self.queue_length * self.queue_num)
                                if self.queue[T_num][result_path[k]][result_path[k + 1]][m] == 0:
                                    Latency_avg = n * 0.01 + 0.004
                                    # print(Latency_avg)
                                    # print(avg_min_Latency_v)
                                    # print(avg_max_Latency_v)
                                    if Latency_avg < avg_min_Latency_v:
                                        continue
                                    elif Latency_avg <= avg_max_Latency_v:
                                        # 当Latency大于等于最小，小于等于最大延迟就说明该位置合法，把它放入LP列表中
                                        for source in range(m - flow_list_obj[i].pkt_num + 1, m + 1):
                                            if self.queue[T_num][result_path[k]][result_path[k + 1]][source] == 0:
                                                if source < m:
                                                    continue
                                                else:
                                                    LP_k.append(source)
                                            else:
                                                break
                                    else:
                                        break
                                else:
                                    continue
                            if len(LP_k) == 0:  # LP为空
                                if k == 1:  # 并且该跳是第一跳，也就是第一个交换机，说明当前的周期偏移量没有LP给该流
                                    # 用个标志，标记一下是第一跳且LP为空的，需要break到offset的循环来换一个offset
                                    tag2 = 1
                                    break
                                else:  # 该跳不是第一跳，所以就回到上一跳
                                    # 回溯，释放上一跳所占的队列资源，根据subscript[k]找到新的位置，重新分配队列资源
                                    k -= 1
                                    T_num -= 1
                                    # 要分情况，如果回溯到上一跳，上一跳还有LP可以往后选择就回溯，如果没有LP可以选择了就tag2=1然后换个offset（这个offset失败了，需要换)
                                    if len(LP[k]) > subscript[k]:
                                        for p in range(LP[k][subscript[k] - 1] - flow_list_obj[i].pkt_num + 1,LP[k][subscript[k] - 1] + 1):
                                            self.queue[T_num][result_path[k]][result_path[k + 1]][p] = 0
                                        # 还需要回溯avg_Latency
                                        max_Latency_v += Latency_tem
                                        min_Latency_v += Latency_tem
                                        avg_max_Latency_v = max_Latency_v / (len(result_edges) - k)
                                        avg_min_Latency_v = min_Latency_v / (len(result_edges) - k)
                                        # avg_max_Latency_v = avg_max_Latency_v * (len(result_edges) - k -1) + Latency_tem
                                        # avg_min_Latency_v = avg_min_Latency_v * (len(result_edges) - k -1) + Latency_tem
                                        Latency -= Latency_tem
                                        if tag1 == 1:  # 如果上一跳让tag1=1，那么回溯时因为不使用上一跳的原先位置，所以要把tag1也变成0
                                            tag1 = 0
                                        continue
                                    else:  # 回溯到上一个但是没有LP可以选择了，就说明该offset不行
                                        tag2 = 1
                                        break
                            else:
                                LP.append(LP_k)
                            # 找到了所有的LP_k，然后计算timeL和标准差来找到最平均的位置
                            tem_dict = {}  # 用来存储位置和BD的键值对，方便后面进行排序
                            # if flow_list_obj[i].pf == 16:
                            #     print(f'该流在第{v}个周期和第{k}跳有{len(LP_k)}个合法位置,周期偏移量为：{j}')
                            for p in LP_k:
                                time_L = []
                                num = 0
                                for pkt in range(p - flow_list_obj[i].pkt_num + 1, p + 1):
                                    self.queue[T_num][result_path[k]][result_path[k + 1]][pkt] = 1
                                for f in range((queue_output_num + 1) * self.queue_length,
                                               (queue_output_num + 1) * self.queue_length + (self.queue_num - 1) * self.queue_length):
                                    q = f % (self.queue_length * self.queue_num)
                                    if self.queue[T_num][result_path[k]][result_path[k + 1]][q] == 0:
                                        num += 1
                                    elif num != 0:
                                        time_L.append(num)
                                        num = 0
                                    else:
                                        continue
                                    if f == (queue_output_num + 1) * self.queue_length + (self.queue_num - 1) * self.queue_length - 1:
                                        if self.queue[T_num][result_path[k]][result_path[k + 1]][q] == 0:
                                            time_L.append(num)
                                if len(time_L) == 0:
                                    break
                                M = len(time_L)
                                # print(M)
                                u_t = 0.0  # 公式2的左边
                                o_t = 0.0  # 公式3的左边
                                BD = 0.0  # 公式1的左边
                                for g in range(M):
                                    u_t += time_L[g]
                                u_t /= M
                                for h in range(M):
                                    o_t += (time_L[h] - u_t) ** 2
                                o_t /= M
                                o_t = math.sqrt(o_t)
                                BD = 0.5 * M * u_t + (1 - 0.5) * o_t / u_t
                                tem_dict[p] = BD
                                # 每次结尾都要把该位置的queue值变回来
                                for pkt in range(p - flow_list_obj[i].pkt_num + 1, p + 1):
                                    self.queue[T_num][result_path[k]][result_path[k + 1]][pkt] = 0
                            # 将BD的值进行排序，把BD小的位置放前面
                            # print(f'tem_dict的长度为{len(tem_dict)}个')
                            # print(f'合法位置有{len(LP_k)}个')
                            sorted_tem_list = sorted(tem_dict.items(), key=lambda x: x[1])
                            LP_k_sorted = []
                            # if len(LP_k) == 1:
                            #     LP_k_sorted.append(LP_k[0])
                            for key, value in sorted_tem_list:
                                if (key % self.queue_length == (self.queue_length - 1)) and (tag1 == 1):
                                    continue
                                LP_k_sorted.append(key)
                            # 所有满足tag1 = 1且对队列长度取余得9的位置都没去除
                            if tag1 == 1:
                                tag1 = 0
                            if not LP_k_sorted:
                                # 方案一：退回 LP_k（不做该过滤）
                                LP_k_sorted = LP_k[:]  # <= 常见兜底
                            # 将原本的LP_k换成排好序的，方便回溯
                            LP[k] = LP_k_sorted
                        # print(k)
                        # print(subscript[k])
                        # print(LP_k_sorted[subscript[k]])
                        if LP[k][subscript[k]] % self.queue_length == (self.queue_length - 1):
                            if tag1 == 0:
                                tag1 = 1
                        for p in range(LP[k][subscript[k]] - flow_list_obj[i].pkt_num + 1,LP[k][subscript[k]] + 1):
                            self.queue[T_num][result_path[k]][result_path[k + 1]][p] = 1
                        flow_list_obj[i].QO.append(LP[k][subscript[k]] // self.queue_length)  # 哪个队列
                        flow_list_obj[i].RO.append(LP[k][subscript[k]] % self.queue_length)  # 队列的哪个位置
                        # 计算延迟，更新avg_Latency
                        if LP[k][subscript[k]] >= (queue_output_num + 1) * self.queue_length:
                            Latency_tem = (LP[k][subscript[k]] - (queue_output_num + 1) * self.queue_length + 1) * 0.01 + 0.004
                        else:
                            Latency_tem = ((self.queue_num - (queue_output_num + 1)) * self.queue_length + LP[k][subscript[k]] + 1) * 0.01 + 0.004
                        Latency += Latency_tem
                        if len(result_edges) - k - 1 > 0:
                            max_Latency_v -= Latency_tem
                            min_Latency_v -= Latency_tem
                            avg_max_Latency_v = max_Latency_v / (len(result_edges) - k - 1)
                            avg_min_Latency_v = min_Latency_v / (len(result_edges) - k - 1)
                        # 将LP_k_sorted列表的下标+1，如果之后回溯就可以直接去到下一个位置
                        subscript[k] += 1
                        k += 1
                        T_num += 1
                    # 运行到这，要么是break出来的要么是正常运行出来的，break出来的话就说明tag2 = 1,
                    if tag2 == 1:
                        break
                    max_Latency = max(Latency, max_Latency)
                    if min_Latency == 0.0:
                        min_Latency = Latency
                    else:
                        min_Latency = min(Latency, min_Latency)
                    all_Latency += Latency
                if tag2 == 1:
                    continue  # 换个offset
                else:
                    succ_tag = 1
                    sche_flow_num += 1
                    # print(f'第{i+1}条流的最大延迟是：{max_Latency}ms')
                    # print(f'第{i+1}条流的最小延迟是：{min_Latency}ms')
                    Jitter = max_Latency - min_Latency
                    max_Latency = math.floor(max_Latency * 1000) / 1000
                    min_Latency = math.floor(min_Latency * 1000) / 1000
                    Jitter = math.floor(Jitter * 1000) / 1000
                    # print(f'第{i + 1}条流的最大延迟是：{max_Latency}ms')
                    # print(f'第{i + 1}条流的最小延迟是：{min_Latency}ms')
                    # print(f'第{i + 1}条流的抖动为{Jitter}ms')
                    # print(f'第{i+1}条流的平均时延为{all_Latency/Max_v}ms')
                    break
            if succ_tag == 1:
                scheduled.append(flow_list_obj[i])
                # print(f"第{i + 1}条流调度成功")
            else:
                # print(f'第{i + 1}条流调度失败，这条流的周期为：{flow_list_obj[i].pf}')
                continue
        # print(sche_flow_num)
        # … 复制 sche_flow 里面的双层循环逻辑，
        #    并且当你“真正”在 queue[t][sw][nbr][slot]=1”
        #    的时候就是在填 self.queue
        #
        #    如果一条 flow 最终 succ_tag==1，就把它 append 到 scheduled，
        #    并且它的 flow.RO/flow.QO 数组里记录每跳的 (queue_id, slot_id)。
        return scheduled

    def get_offline_bitmap(self, sw, nbr):
        """
        返回长度为 queue_num*queue_length 的 0/1 列表，
        取 self.queue 的所有 t 的 OR（或者只取当前 t，取决你在线环境的设定）。
        在线环境里，我们一般只需要“这个队列已经被预占了哪些 slot”这个快照，
        所以可以把所有超周期 t 上的占用做一次 OR。
        """
        Q = self.queue_num * self.queue_length
        bitmap = [0]*Q
        for t in range(self.total_cycle_num):
            bits = self.queue[t][sw][nbr]
            for i in range(Q):
                if bits[i]:
                    bitmap[i] = 1
        return bitmap


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

    def __init__(self, G, flows, offline_scheduler,
                 alpha=0.01, R_HIT=20, R_MISS=10,
                 anomaly_prob=0.1):
        self.G = G
        self.flows = flows
        self.offline = offline_scheduler
        self.switches = [n for n in G.nodes if not str(n).startswith('h')]
        self.alpha = alpha
        self.R_HIT = R_HIT
        self.R_MISS = R_MISS
        # 与论文式(18)对齐的权重
        self.rho_c = 4.0  # ρc：时延代价权重
        self.sigma_c = 0.5  # σc：队列代价权重

        # 距离势函数塑形（可设大，但它不是代价项）
        self.eta_dist = 4.0  # 你原来是 *10，可先从 4 起
        self.anomaly_prob = anomaly_prob
        # 用来决定在哪个 global_slot 触发一次异常
        # 每次 reset() 时都会重置下面两个属性：
        self.anomaly_triggered = False
        self.anomaly_slot = 0
        # self.scheduled = self.offline.schedule(flows)
        self.scheduled = flows
        # self.runtime_queue = self.offline.queue.copy()
        # per-link queue capacity
        self.Qmax = self.offline.queue_num * self.offline.queue_length
        # number of actions per neighbor = Qmax
        #  一个距离上界，用来替换 inf
        # 这里简单取节点数 * 最大链路权重
        all_pairs = nx.all_pairs_dijkstra_path_length(G, weight='weight')
        self.max_dist = max(d for _, dist_dict in all_pairs for d in dist_dict.values()
                            if d < float('inf'))
        self.global_slot = 0  # 新增：离散时隙计数器，每次 step() ++
        self.fail_stats = {"no_slots": 0, "loop": 0, "slack": 0, "bad_host": 0, "link_fail": 0}
        self.failed_count = 0
        self.T_cycle = 0.1
        self.t_slot = []
        self.reset()

        # print("Scheduled flow IDs:", [f.id for f in self.scheduled])

    def _init_runtime(self):
        # 初始化在线态
        self.link_delays = {
            e:  0.004 # self.G.edges[e]['weight']
            for e in self.G.edges
        }

        # 初始化 runtime_queue，先全 0
        self.runtime_queue = []
        for t in range(self.offline.total_cycle_num+5):
            per_node = {}
            for sw in self.G.nodes():
                # 只对交换机做字典，主机可忽略
                if str(sw).startswith('h'):
                    continue
                per_node[sw] = {}
                for _, nbr, _ in self.G.out_edges(sw, data='weight'):
                    # 早于 anomaly_slot，从离线数据里拷贝；否则全 0
                    if t < len(self.offline.queue):
                        # 离线排期表就是 offline.queue
                        per_node[sw][nbr] = self.offline.queue[t][sw][nbr].copy()
                    else:
                        per_node[sw][nbr] = [0] * self.Qmax
            self.runtime_queue.append(per_node)

        tem_flow_states = []
        # self.finish_slot = {}  # map: flow.id -> int
        for f in self.scheduled:
            # 计算这条流每一趟的结束时隙
            # self.finish_slot[f.id] = f.finish_slot
            # 得到这条流的趟数
            # max_v = int(32 / f.pf)
            #
            # for v in range(max_v):
            #     total_slot_v = int((f.offset + v * f.pf) / self.T_cycle)
            #     elapsed = f.offset + v * f.ddl
            #     if self.anomaly_slot <= total_slot_v:  # 产生异常的时隙小于等于这条流的offset，那把这条流所有趟都应该放入flow_states
            #         # 每趟流的开始时间，对这个时间去一个网络拓扑快照生成这条流最开始安排的路径
            #         t = f.offset + v * f.pf
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
            # 得到这条流的趟数
            max_v = int(32 / f.pf)

            for v in range(max_v):
                total_slot_v = int((f.offset + v * f.pf) / self.T_cycle)
                elapsed = f.offset + v * f.ddl
                if self.anomaly_slot <= total_slot_v:  # 产生异常的时隙小于等于这条流的offset，那把这条流所有趟都应该放入flow_states
                    # 每趟流的开始时间，对这个时间去一个网络拓扑快照生成这条流最开始安排的路径
                    # t = f.t_start_ms + v * f.pf
                    f.path = generate_path(G, f.sn_dn[0], f.sn_dn[1])
                    flow_state = {
                        'pos': f.path[1],
                        'dst': f.sn_dn[1],  # 新增：流的目的节点
                        'hop': 1,
                        'elapsed': elapsed + 0.004,
                        # 'elapsed': G.edges[f.path[0], f.path[1]]['weight'],
                        'deadline': f.ddl * (v + 1),
                        # 'deadline': f.ddl,
                        'path': f.path,
                        'pkt_num': f.pkt_num,
                        'QO': f.QO,
                        'RO': f.RO,
                        'offset': f.offset + v * f.ddl
                    }
                    tem_flow_states.append(flow_state)
                    # 因为要直接跳过初始主机所以要+1
                    self.t_slot.append(total_slot_v+1)
                # else:  # 这条流的某一趟的发出时隙小于等于异常时隙，那就遍历这一趟的所有跳，看看到哪一跳才开始大于等于异常时隙
                #     for k in range(-1, (len(f.path) - 3)):
                #         if k == -1:
                #             total_slot_v += 1
                #             elapsed += 0.004
                #         else:
                #             total_slot_v += f.QO[v * (len(f.path) - 2) + k]
                #             elapsed += ((f.QO[v * (len(f.path) - 2) + k - 1] - 1) * 0.1 + f.RO[v * (len(f.path) - 2) + k - 1] * 0.01) + 0.004
                #         if total_slot_v >= self.anomaly_slot:
                #             # 在某一趟的某一跳时，出现了异常，把没完成的这一趟和后续的趟都放入flow_states里面
                #             flow_state = {
                #                 'pos': f.path[k + 2],  # 这个应该要变成当前节点，而不是起始节点
                #                 'dst': f.sn_dn[1],  # 新增：流的目的节点
                #                 'hop': k + 2,
                #                 'elapsed': elapsed,
                #                 'deadline': f.ddl * (v + 1),
                #                 'path': f.path,
                #                 'pkt_num': f.pkt_num,
                #                 'QO': f.QO,
                #                 'RO': f.RO,
                #                 'offset': f.offset + v * f.ddl
                #             }
                #             tem_flow_states.append(flow_state)
                #             self.t_slot.append(total_slot_v)
                #             break

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


    def reset(self):
        # reset link delays

        # 只重置 runtime 相关，不再调用 schedule()
        self._init_runtime()
        # self.global_slot = 0  # 新增：离散时隙计数器，每次 step() ++
        self.failed_count = 0
        # return self._get_obs_for(0)
        return

    def get_queue_bitmap(self, sw, nbr, slot):
        # 得到目标端口当前slot的队列资源快照
        Q = self.offline.queue_num * self.offline.queue_length
        bmp = [0] * Q

        # off = self.offline.get_offline_bitmap(sw, nbr)
        rt = self.runtime_queue[slot][sw][nbr]
        for i in range(Q):
            bmp[i] = rt[i]
        return bmp

    # 一个小工具：给 fid 返回（Gt, t_sec）
    def _graph_for_fid(self, fid):
        if self.snapshotter is None:
            return self.G, 0.0
        t_sec = self.t_slot[fid] * self.T_cycle  # 也可加上该流 offset（若你保留 offset）
        Gt, _ = self.snapshotter.graph_at(t_sec)
        return Gt, t_sec

    def _get_obs_for(self, fid: int) -> np.ndarray:
        st = self.flow_states[fid]
        sw = st['pos']
        nbrs = list(self.G.successors(sw))
        dis = nx.dijkstra_path_length(self.G, st['pos'], st['dst'], weight='weight')

        obs_list = []
        # —— per-link × per-queue 特征（同你原来逻辑） ——
        for v in nbrs:
            feats = []
            bitmap = self.get_queue_bitmap(sw, v, self.t_slot[fid])  # list of 0/1 length Qmax
            # k = 0
            dis_next = nx.dijkstra_path_length(self.G, v, st['dst'], weight='weight')
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
                feats.append(1.0)
            else:
                feats.append(0.0)

            # 判断是否有环路
            visited_flag = 0.0
            if st['pos'] == sw and v in self.visited[fid]:
                visited_flag = 1.0

            feats.append(visited_flag)

            feats.append(float(dis - dis_next) / 4)

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

    def _simulate_anomaly(self):
        kind = random.choice([1, 2, 3])
        if kind == 1:
            # increase random link delay
            e = random.choice(list(self.link_delays.keys()))
            self.link_delays[e] *= random.uniform(0.01, 0.04)
        elif kind == 2:
            # break a random link
            e = random.choice(list(self.link_delays.keys()))
            self.link_delays[e] = float('inf')
        else:
            # add random new link
            u, v = random.sample(self.switches, 2)
            self.link_delays[(u, v)] = random.uniform(0.004, 0.004)

    def _apply_action(self, fid: int, sw: str, act: int, obs_for: dict):
        """
        为 flow id=fid，在交换机 sw 上执行动作 act。
        返回 (t_delay, reward_delta, done_flag) 三元组。
        """
        fs = self.flow_states[fid]
        nbrs = list(self.G.successors(sw))

        # # 终点检测，如果邻居有终点，直接发往终点
        # for nbr in nbrs:
        #     if nbr == fs["dst"]:
        #         return 0.0, self.R_HIT, True

        # 解码动作：仅 link_idx = port 索引
        link_idx = act
        if link_idx < len(nbrs):
            nxt = nbrs[link_idx]
        else:
        # 超出范围直接尝试发往目的节点
            nxt = fs['dst']


        pkt_num = fs['pkt_num']
        # —— 队列资源检查 ——
        # agent选择的端口没有队列资源就Early-stop
        if obs_for[fid][link_idx][0] == 0:
            self.failed_count += 1
            self.fail_stats["no_slots"] += 1
            return 0.0, - self.R_MISS*0.6, True

        # —— 环路检测 ——
        # 如果 nxt 已经在这条流的 visited 里，视为严重错误，Early-stop
        if obs_for[fid][link_idx][1] == 1:
            self.failed_count += 1
            self.fail_stats["loop"] += 1
            return 0.0, - self.R_MISS, True
        # 记下这条流访问过 nxt
        self.visited[fid].append(nxt)

        # —— 自动找第一个可用 slot ——
        bmp = self.get_queue_bitmap(sw, nxt, self.t_slot[fid])
        # free_slots = [i for i, b in enumerate(bmp) if b == 0]
        # ((self.t_slot[fid] % self.offline.queue_num)+1)*10
        # legal_slots = [
        #     p
        #     for p in range(0, self.Qmax - pkt_num + 1)
        #     if all(bmp[p + i] == 0 for i in range(pkt_num))
        # ]
        # legal_slots = [
        #     p % self.Qmax
        #     for p in range(((self.t_slot[fid] % self.offline.queue_num)+1)*self.offline.queue_length, ((self.t_slot[fid] % self.offline.queue_num)+1)*self.offline.queue_length + (self.offline.queue_num - 1) * self.offline.queue_length-pkt_num+1)
        #     if same_queue_segment(p, pkt_num, self.offline.queue_length)
        #     and all(bmp[(p+ i)% self.Qmax] == 0 for i in range(pkt_num))
        # ]
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
        # 空间不足，Early-stop
            self.failed_count += 1
            self.fail_stats["no_slots"] += 1
            return 0.0, - self.R_MISS*0.6, True

        # 取最早的连续 pkt_num 个 slot 起始位置
        blk_idx = legal_slots[0]

        # print(fs)
        for p in range(blk_idx, blk_idx + pkt_num):
            # if(self.t_slot[fid] >= self.offline.total_cycle_num):
            #     print(f"fid={fid}, t_slot={self.t_slot[fid]}, total_cycle_num={self.offline.total_cycle_num}")
            # 不只是当前时隙要占位置，如果选择的位置距离发送队列的位置>=2
            if ((blk_idx // self.offline.queue_length) - (self.t_slot[fid] % self.offline.queue_num)) % self.offline.queue_num >= 1:
                for q in range(0, ((blk_idx // self.offline.queue_length) - (self.t_slot[fid] % self.offline.queue_num)) % self.offline.queue_num):
                    # if(p>=50):
                    #     print(p)
                    self.runtime_queue[self.t_slot[fid]+q][sw][nxt][p % self.Qmax] = 1

        # 简单的“占用奖励”
        u = float(obs_for[fid][link_idx][5])  # 0~1
        # 占用奖励（原来是 1），现在按 (1-u)2 衰减
        reward = 1-self.sigma_c * (u ** 2)
        # 是否前进
        reward += obs_for[fid][link_idx][2] * self.eta_dist
        if obs_for[fid][link_idx][3] == 1:
            reward += self.R_HIT
            return self.link_delays.get((sw, nxt), float('inf')), reward, True
        # —— 计算实际时延 t_delay ——
        # 1) 跨队列循环等待
        queue_id = blk_idx // self.offline.queue_length
        # 通过t_slot得到发送队列，用来计算等待时间
        send_q = self.t_slot[fid] % self.offline.queue_num
        wait_cycles = (queue_id - send_q) % self.offline.queue_num
        self.t_slot[fid] += wait_cycles
        base_wait = wait_cycles * self.offline.T_cycle
        # 2) 队列内位置等待
        pos_in_q = blk_idx % self.offline.queue_length
        intra_wait = pos_in_q * (self.offline.T_cycle / self.offline.queue_length)
        # 3) 物理链路延迟
        link_d = self.link_delays.get((sw, nxt), float('inf'))
        t_delay = base_wait + intra_wait + link_d
        # if not np.isfinite(link_d):
        #     return t_delay, 0.0, True
        # self.t_slot[fid] += int(t_delay/self.T_cycle)
        # —— 更新流的位置 ——
        fs['pos'] = nxt
        # 注意：elapsed 的累加要在 step() 里统一执行

        return t_delay, reward, False

    def step(self, actions, obs_for):
        """
        actions: dict {switch: action_int}
        return obs, rewards, done, info
        """
        # maybe anomaly
        if random.random() < self.anomaly_prob:
            self._simulate_anomaly()

        # rewards = [0.0 for i in range(len(self.flow_states))]
        rewards = {fid: 0.0 for fid in self.flow_states.keys()}
        done = False
        finished_ids = []
        for fid, st in self.flow_states.items():
            cur = st['pos']
            if cur != st['dst']:
                # 1) 计算下一跳和 t_delay
                # if cur.startswith('h'):
                #     nxt = list(self.G.successors(cur))[0]
                #     t_delay = self.link_delays.get((cur, nxt), 0.0)
                #     st['pos'] = nxt
                # else:
                nbrs = list(self.G.successors(cur))
                # if st['dst'] in nbrs:
                #     nxt = st['dst']
                #     r_delta = self.R_HIT
                #     failed = True  # or True 来表示这一流完成
                #     t_delay = self.link_delays.get((cur, st['dst']), float('inf'))
                #     st['pos'] = nxt
                # else:
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
                # if r_delta > 0:
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
                    rewards[fid] -= self.R_MISS*0.8
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
        # done 对应 terminated，truncated 始终 False
        # batch = Batch(
        #     obs=s,
        #     act=a,
        #     rew=r,
        #     terminated=done,
        #     truncated=False,
        #     obs_next=s2
        # )
        # self.memory.add(batch)
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


