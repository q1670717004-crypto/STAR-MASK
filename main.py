import time
import math
import networkx as nx
import os, csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import os
from data_define import OCSQFOfflineScheduler, TSNEnv, DuelingDQN, DQNAgent   # 你的环境定义
from ReplayBuffer import PrioritizedReplayBuffer       # 我们自定义的PER缓冲


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
# # 6 台主机
# hnodes = ['h0','h1','h2','h3','h4','h5']
#
# # 4 个交换机 + 6 台主机
# nodes = [
#     '0','1','2','3',  # 交换机
#     'h0','h1','h2','h3','h4','h5'
# ]
#
# # 边列表：双向都有
# edges = [
#     # 主机 ↔ 交换机
#     ('h0','0',4), ('0','h0',4),
#     ('h1','0',4), ('0','h1',4),
#     ('h2','1',4), ('1','h2',4),
#     ('h3','1',4), ('1','h3',4),
#     ('h4','2',4), ('2','h4',4),
#     ('h5','2',4), ('2','h5',4),
#
#     # 交换机环形互联
#     ('0','1',4), ('1','0',4),
#     ('1','2',4), ('2','1',4),
#     ('2','3',4), ('3','2',4),
#     ('3','0',4), ('0','3',4),
# ]

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_weighted_edges_from(edges[i] for i in range(len(edges)))
# for u, v in G.edges():
#     G.edges[u, v]['weight'] *= 1e-2

pos = nx.spring_layout(G,k=1,iterations=100)
nx.draw_networkx_nodes(G, pos,node_size=400,alpha=0.3,node_color='#b3e2cd')
nx.draw_networkx_edges(G, pos, width=1)
nx.draw_networkx_labels(G, pos)
# nx.draw_networkx_edge_labels(G, pos, {(edges[i][0],edges[i][1]):edges[i][2] for i in range(len(edges))})
plt.tight_layout()
# 指定保存目录
ckpt_dir = r"D:\Agent_test"
os.makedirs(ckpt_dir, exist_ok=True)

def main():
    # 1. 离线调度和环境初始化
    offline = OCSQFOfflineScheduler(G, hnodes)
    flows = offline.generate_flow(200)   # 或者改为 curriculum learning 动态流量
    env = TSNEnv(G, flows, offline)



    # Qmax        = offline.queue_num * offline.queue_length
    max_outdeg  = offline.max_outdeg
    obs_dim = 6 * max_outdeg
    shared_act_dim = max_outdeg

    # 3. 创建共享网络、目标网络及优化器
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    shared_net    = DuelingDQN(obs_dim, shared_act_dim, ).to(device)
    shared_target = DuelingDQN(obs_dim, shared_act_dim, ).to(device)
    shared_target.load_state_dict(shared_net.state_dict())
    optimizer = optim.Adam(shared_net.parameters(), lr=5e-5)
    max_obs_dim = max_outdeg
    # 4. 创建共享的优先经验回放缓冲
    shared_memory = PrioritizedReplayBuffer(
        capacity=60000, alpha=0.6, beta=0.4, max_obs_dim = max_obs_dim
    )

    # 5. 为每个交换机初始化 agent，并挂载共享组件
    agents = {}
    for sw in env.switches:
        # 构造 agent 时仍传入统一的 obs_dim 和 action_dim
        agent = DQNAgent(
            obs_dim=obs_dim,
            act_dim=shared_act_dim,
            # max_obs_dim= max_obs_dim,
            lr=1e-4,
            gamma=0.99,
            buffer_size=20000,
            batch_size=128
        )
        # **覆盖** agent 内部自建网络、buffer、优化器为共享的
        agent.net        = shared_net
        agent.target_net = shared_target
        agent.optimizer  = optimizer
        agent.memory     = shared_memory
        agents[sw] = agent

    # 6. 训练循环（保持原逻辑，网络和缓冲是共享的）
    episodes = 10000

    target_update_frames = 2000  # 每2000帧同步一次目标网络
    success_rates = []
    reward_history = []  # 存每个 episode 的平均奖励
    save_dir = r"D:\Agent_test"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "train_metrics.csv")

    curriculum = [
        (1, 2000, 200),  # ep 1–100: 100 条流
        (2001, 10000, 1000),  # ep 201–1000: 1000 条流
    ]
    # Paper setting: episodes 1-2000 use 200 flows, episodes >=2001 use 700 flows.
    # curriculum[1] = (2001, episodes, 700)
    # current_flow_stage = None
    # 2) 定义“噪声阶段”：简单用布尔标志
    # current_noise_flag = None

    # 3) 占位，后面按需重建
    # flows = None
    # env   = None
    # anomaly_prob = 0
    shared_agent = next(iter(agents.values()))  # 只取第一个，网络/buffer 都是共享的
    global_env_step = 0
    global_update_step = 0
    UPDATES_PER_ENV_STEP = 8  # 先用 4 或 8 试

    for ep in range(1, episodes + 1):
        print(f"\n=== Starting Episode {ep}/{episodes} ===")
        # obs = env.reset()
        # ———— 4) 计算当前流量阶段 ————
        for start, end, num in curriculum:
            if start <= ep <= end:
                flow_stage = (start, end, num)
                break
        for agent in agents.values():
            if ep <= 1000:
                agent.epsilon = 1.0
            elif ep <= 2500:
                # 线性从 1 → 0.05
                frac = (ep - 1000) / (2500 - 1000)
                agent.epsilon = 1.0 - frac * (1.0 - 0.01)
            elif ep <= 3500:
                frac = (ep - 2500) / (3500 - 2500)
                agent.epsilon = 0.01-frac * (0.01 - 0.005)
            else:
                agent.epsilon = 0.005
        # 阶段变化时（或第一次）重建 flows
        # ———— 5) 计算当前噪声阶段 ————
        noise_flag = (ep > 2000)  # False 表示前 200 集 anomaly=0，True 表示之后 anomaly=0.1

        # ———— 6) 仅在阶段变化时重建 flows/env ————
        # if flow_stage != current_flow_stage or noise_flag != current_noise_flag:
        #     current_flow_stage = flow_stage
        #     current_noise_flag = noise_flag
        #     _, _, num_flows = flow_stage
        #     flows = offline.generate_flow(num_flows)
        #     anomaly_prob = 0.1 # if noise_flag else 0.0
        #     env = TSNEnv(G, flows, offline, anomaly_prob=anomaly_prob)
        # else:
        _, _, num_flows = flow_stage
        flows = offline.generate_flow(num_flows)
        # anomaly_prob = 1 if noise_flag else 0
        # Keep anomaly injection aligned with the staged training description.
        anomaly_prob = 0.1 if noise_flag else 0.0
        env = TSNEnv(G, flows, offline, anomaly_prob=anomaly_prob)

        true_num = len(env.flow_states)

        obs_for = {fid : env._get_obs_for(fid) for fid in env.flow_states.keys()}
        # for fs in env.flow_states.values():
        #     print(fs)
        done = False
        frame_idx = 0
        # ep_reward = {sw: 0.0 for sw in env.switches}
        rewards = {fid: 0.0 for fid in env.flow_states.keys()}

        while not done:

            # 1) 先由上一步的 obs_for（字典 fid->state 向量）构造一个 valid_masks 字典
            valid_masks = {}
            for fid, st in env.flow_states.items():
                sw = st['pos']
                nbrs = list(G.successors(sw))
                # if len(nbrs) == 0:
                #     print(sw)
                # 给这个交换机构造一个全 False 的 mask
                mask = np.zeros(shared_act_dim, dtype=bool)
                mask[: len(nbrs)] = True
                valid_masks[fid] = mask
            # a) 所有交换机并行选动作
            # 1) 为每个流分别选动作

            actions = {}         # 字典
            for fid, st in env.flow_states.items():
                sw = st['pos']
                # obs_fid = env._get_obs_for(fid)
                mask = valid_masks[fid]
                # print(st)
                a = agents[sw].select_action(obs_for[fid], mask)
                actions[fid] = a

            sw_before = {fid: st['pos'] for fid, st in env.flow_states.items()}
            # # 2) 传给 env.step
            # next_obs, rewards, done, info = env.step(actions)

            # 2) 把所有流的动作一次性传给环境
            next_obs_all, rewards_all, finished_ids, info = env.step(actions, obs_for)
            # print("rewards_all", rewards_all)
            # 3) 把每条流的 transition 存到共享经验池
            for fid, st in env.flow_states.items():
                # sw = st['pos']
                agent = agents[sw_before[fid]]
                obs_fid = obs_for[fid]
                nxt_obs = next_obs_all[fid]
                r = rewards_all[fid]
                if fid in finished_ids:
                    # 这条流在这一step完成了
                    agent.store(obs_fid, actions[fid], r, nxt_obs, True)
                else:
                    agent.store(obs_fid, actions[fid], r, nxt_obs, False)
                rewards[fid] += rewards_all[fid]
            # 先存储再弹出
            for fid in finished_ids:
                env.flow_states.pop(fid)

            if len(env.flow_states) == 0:
                done = True
            # # 4) **只调用一次** train_step
            # shared_agent.train_step(frame_idx)
            #
            # # 5) target net 同步也只做一次
            # if frame_idx % target_update_frames == 0:
            #     shared_agent.update_target()
            #
            # frame_idx += 1

            # 本 env step 写入了多少条 transition（近似）
            N_new = len(obs_for)  # 或 len(actions)，二者基本一致

            K = max(1, math.ceil(N_new / shared_agent.batch_size))
            K = min(K, 8)  # 上限，CPU 建议 4~8；GPU 可更大

            # 4) 每个 env step 做多次梯度更新
            for _ in range(K):
                shared_agent.train_step(global_update_step)
                global_update_step += 1

            # 5) target net 同步仍按“环境步”频率来（更符合你原意）
            if global_env_step % target_update_frames == 0:
                shared_agent.update_target()

            global_env_step += 1

            obs_for = next_obs_all

        # Episode 结束后统计 success rate
        success = (true_num - env.failed_count) / true_num
        success_rates.append(success)
        avg_reward = np.mean(list(rewards.values()))
        reward_history.append(avg_reward)
        # print(env.fail_stats)
        # print(f"Episode {ep} success rate: {success:.1%}, avg reward: {avg_reward:.2f}, rewards: {list(rewards.values())}:")
        print(f"Episode {ep} success rate: {success:.1%}, avg reward: {avg_reward:.2f}")
        # 追加写入 CSV（首行写表头，只在 ep==1 时）
        write_header = (ep == 1 and not os.path.exists(csv_path))
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["episode", "avg_reward", "success_rate"])
            w.writerow([ep, float(avg_reward), float(success)])

        if ep % 500 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"csqf_agent_ep{ep}.pth")
            torch.save({
                'model_state_dict': shared_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': ep,
                'epsilon': shared_agent.epsilon
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")
    # 结束所有 episode 后再存
    ckpt_path = os.path.join(ckpt_dir, f"csqf_agent_ep{episodes}.pth")
    torch.save({
        'model_state_dict': shared_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episodes,
        'epsilon': shared_agent.epsilon
    }, ckpt_path)
    print(f"Training complete. Final model saved to {ckpt_path}")

    # df = pd.read_csv(r"D:\Agent_final\train_metrics.csv")
    # plt.figure()
    # plt.plot(df["episode"], df["avg_reward"], label="Avg Reward / Episode")
    # plt.plot(df["episode"], df["success_rate"], label="Success Rate", alpha=0.7)
    # plt.xlabel("Episode")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure()
    # plt.plot(shared_agent.loss_history)
    # plt.title('TD Loss 随训练步数的变化')
    # plt.xlabel('Train Step')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    main()
