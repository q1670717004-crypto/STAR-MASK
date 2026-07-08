import numpy as np

def pad_obs(obs, max_obs_dim):
    if obs is None:
        return np.zeros(max_obs_dim * 6, dtype=np.float32)

    obs_arr = np.array(obs, dtype=np.float32)
    if obs_arr.ndim == 1:
        return obs_arr.astype(np.float32)
    if obs_arr.ndim != 2:
        return np.zeros(max_obs_dim * 6, dtype=np.float32)

    n, feat_dim = obs_arr.shape
    # pad 到 (max_obs_dim, feat_dim)
    padded = np.zeros((max_obs_dim, feat_dim), dtype=np.float32)
    length = min(n, max_obs_dim)
    padded[:length, :] = obs_arr[:length, :]
    # flatten 成 (max_obs_dim * feat_dim,)
    return padded.reshape(-1)
    # arr = np.zeros(max_obs_dim, dtype=np.float32)
    # length = min(len(obs), max_obs_dim)
    # arr[:length] = obs[:length]
    # return arr


class SumTree:
    """用于存储优先级的二叉树结构，支持高效采样和更新。"""
    def __init__(self, capacity):
        self.capacity = capacity  # 叶子节点（经验条目）的最大数量
        # 二叉树的总节点数 = 2*capacity - 1
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data_index = 0  # 下一个插入数据的索引（循环覆盖）
        self.size = 0        # 当前存储的经验数量

    def total_priority(self):
        return self.tree[0]  # 树根节点存储所有叶子优先值之和

    def add(self, priority):
        """在树中新增一个叶子节点（优先值），并更新相关父节点。"""
        leaf_index = self.data_index + self.capacity - 1  # 计算对应叶子节点的索引
        self.update(leaf_index, priority)
        # 更新数据索引及大小
        self.data_index = (self.data_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_index, new_priority):
        """将指定树索引的节点更新为 new_priority，并向上传播更新父节点值。"""
        change = new_priority - self.tree[tree_index]
        self.tree[tree_index] = new_priority
        # 迭代更新父节点
        parent = (tree_index - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get_leaf(self, value):
        """根据给定的累积和随机值 value，查找对应的叶子节点及其索引、优先值。"""
        parent = 0
        # 从根开始向下寻找叶子：比较给定值与左子树权重决定走向
        while True:
            left_child = 2 * parent + 1
            right_child = left_child + 1
            if left_child >= len(self.tree):  # 到达叶子节点
                leaf_index = parent
                break
            # 决定向左还是向右
            if value <= self.tree[left_child]:
                parent = left_child
            else:
                value -= self.tree[left_child]
                parent = right_child
        leaf_index = parent
        priority = self.tree[leaf_index]
        data_index = leaf_index - (self.capacity - 1)  # 计算对应的数据索引
        return leaf_index, priority, data_index

class PrioritizedReplayBuffer:
    """支持按优先级采样的经验回放缓冲。"""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=None, epsilon=1e-6, max_obs_dim=None):
        """
        参数:
        - capacity: 缓冲区最大容量
        - alpha: 优先采样概率的放大系数 (0表示不区分优先级)
        - beta:  重要性采样权重的初始修正系数
        - beta_increment: 每次采样后 beta 的增加量（用于逐步提高 beta 至1）
        - epsilon: 微小值，避免优先级为0导致样本永远不被采样
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment  # 如果需要逐步增加beta，可指定每一步增加量
        self.epsilon = epsilon
        # 初始化 SumTree 和经验存储容器
        self.tree = SumTree(capacity)
        self.data = {
            "obs": np.zeros((capacity,), dtype=object),       # 状态，可以是高维向量，这里用object以存任意类型
            "act": np.zeros((capacity,), dtype=np.int64),
            "rew": np.zeros((capacity,), dtype=np.float32),
            "next_obs": np.zeros((capacity,), dtype=object),
            "done": np.zeros((capacity,), dtype=np.float32)
        }
        # 记录当前最大优先级，确保新经验以最高优先级加入
        self.max_priority = 1.0
        if max_obs_dim is None:
            raise ValueError("PrioritizedReplayBuffer requires max_obs_dim")
        self.max_obs_dim = max_obs_dim

    def __len__(self):
        """当前存储的经验数（缓冲区长度）。"""
        return self.tree.size

    def _is_valid_obs(self, obs):
        if obs is None:
            return False
        arr = np.asarray(obs, dtype=object)
        return arr.ndim >= 1 and arr.size > 0

    def _valid_indices(self):
        return [
            i for i in range(len(self))
            if self._is_valid_obs(self.data["obs"][i])
            and self._is_valid_obs(self.data["next_obs"][i])
        ]

    def add(self, obs, act, rew, next_obs, done):
        """向缓冲区添加一个新经验。"""
        # 确定用于新经验的优先级: 使用当前最大优先级，保证新样本至少被采样一次
        priority = self.max_priority ** self.alpha
        # 在 SumTree 中添加优先值
        self.tree.add(priority)
        # 获取插入位置的索引
        idx = (self.tree.data_index - 1) % self.capacity
        # 存储经验数据
        self.data["obs"][idx] = obs
        self.data["act"][idx] = act
        self.data["rew"][idx] = rew
        self.data["next_obs"][idx] = next_obs
        self.data["done"][idx] = 1.0 if done else 0.0

    def sample(self, batch_size, beta=None):
        if beta is None:
            beta = self.beta
        else:
            self.beta = beta

        N = len(self)
        if N == 0:
            raise ValueError("Cannot sample from an empty buffer")
        valid_indices = self._valid_indices()
        if not valid_indices:
            raise ValueError("Cannot sample because the buffer has no valid transitions")

        total_p = self.tree.total_priority()
        # 如果 total_p 非法，则退回均匀采样
        if not np.isfinite(total_p) or total_p <= 0:
            indices = np.random.choice(valid_indices, batch_size, replace=(batch_size > len(valid_indices)))
            # pad 后再 stack
            states_arr = np.stack([pad_obs(self.data["obs"][i], self.max_obs_dim)
                                   for i in indices]).astype(np.float32)
            actions_arr = np.array([self.data["act"][i] for i in indices], dtype=np.int64)
            rewards_arr = np.array([self.data["rew"][i] for i in indices], dtype=np.float32)
            next_states_arr = np.stack([pad_obs(self.data["next_obs"][i], self.max_obs_dim)
                                        for i in indices]).astype(np.float32)
            dones_arr = np.array([self.data["done"][i] for i in indices], dtype=np.float32)
            is_weights = np.ones(batch_size, dtype=np.float32)
            return (states_arr, actions_arr, rewards_arr,
                    next_states_arr, dones_arr,
                    np.array(indices, dtype=np.int32),
                    is_weights)

            # 优先采样分支
        segment = total_p / batch_size
        batch_idxs, priorities = [], []
        raw_states, raw_actions, raw_rewards, raw_next, raw_dones = [], [], [], [], []
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            v = np.random.uniform(a, b)
            leaf_idx, p, idx = self.tree.get_leaf(v)
            if idx >= N or not self._is_valid_obs(self.data["obs"][idx]) or not self._is_valid_obs(self.data["next_obs"][idx]):
                idx = int(np.random.choice(valid_indices))
                leaf_idx = idx + self.tree.capacity - 1
                p = max(float(self.tree.tree[leaf_idx]), self.epsilon)
            batch_idxs.append(idx)
            priorities.append(p)
            raw_states.append(self.data["obs"][idx])
            raw_actions.append(self.data["act"][idx])
            raw_rewards.append(self.data["rew"][idx])
            raw_next.append(self.data["next_obs"][idx])
            raw_dones.append(self.data["done"][idx])

        # pad 后再 stack/array
        states_arr = np.stack([pad_obs(s, self.max_obs_dim) for s in raw_states]).astype(np.float32)
        actions_arr = np.array(raw_actions, dtype=np.int64)
        rewards_arr = np.array(raw_rewards, dtype=np.float32)
        next_states_arr = np.stack([pad_obs(s, self.max_obs_dim) for s in raw_next]).astype(np.float32)
        dones_arr = np.array(raw_dones, dtype=np.float32)
        idxs_arr = np.array(batch_idxs, dtype=np.int32)

        # … 剩余概率和 is_weights 计算不变 …
        pri_arr = np.array(priorities, dtype=np.float32)
        probs = pri_arr / (pri_arr.sum() + 1e-6)
        weights = np.power(N * probs, -beta)
        finite = np.isfinite(weights)
        if not np.any(finite):
            weights = np.ones_like(weights)
        else:
            max_w = weights[finite].max()
            weights[~finite] = max_w
            weights = weights / (max_w + 1e-6)

        return (states_arr, actions_arr, rewards_arr,
                next_states_arr, dones_arr,
                idxs_arr, weights.astype(np.float32))

    def update_priorities(self, indices, td_errors):
        """
        根据给定的索引和对应TD误差更新优先级。
        indices 为 sample 返回的 data 索引列表，td_errors 为对应的TD误差数组。
        """
        for idx, error in zip(indices, td_errors):
            # 计算新的优先级
            priority = ((abs(error) + self.epsilon) ** self.alpha)
            # 更新SumTree中对应叶子的优先值
            tree_index = idx + self.tree.capacity - 1  # 叶子索引 = 数据索引 + capacity - 1
            self.tree.update(tree_index, priority)
            # 更新当前最大优先值，供新样本添加时使用
            if priority > self.max_priority:
                self.max_priority = priority
