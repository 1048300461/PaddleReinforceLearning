import numpy as np

class SarsaAgent(object):
    def __init__(self,
                 obs_n,
                 act_n,
                 learning_rate=0.01,
                 gamma=0.9,
                 e_greed=0.1):
        self.act_n = act_n  # 动作维度
        self.lr = learning_rate # 学习率
        self.gamma = gamma  # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选择动作
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观测值，采样输出动作值
    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # 根据table的Q值旋动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)   # 有一定概率随机探索选取一个动作
        return action

    # 根据输入观测值，预测输出的动作值
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]   # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    # 学习方法，更新Q表格
    def learn(self, obs, action, reward, next_obs, next_action, done):
        """
            on-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            next_action: 根据当前Q表格, 针对next_obs会选择的动作, a_t+1
            done: episode是否结束
        """

        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward   # 没有下一个状态，target值即为reward值
        else:
            target_Q = reward + self.gamma * self.Q[next_obs, next_action]  # Sarsa
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)

    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')