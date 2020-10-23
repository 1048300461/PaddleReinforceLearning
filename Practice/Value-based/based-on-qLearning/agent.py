import numpy as np

class QLearningAgent(object):
    def __init__(self,
                 obs_n,
                 act_n,
                 learning_rate=0.01,
                 gamma=0.9,
                 e_greed=0.1):
        self.act_n = act_n  # 动作维度
        self.lr = learning_rate # 学习率
        self.gamma = gamma  # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))   # Q表格

    # 根据输入观测值obs，采样带探索的动作值
    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # 根据Q表格的Q值选动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)   # 随机探索选取一个动作
        return action

    # 根据输入观测值，预测输出的动作值
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]   # maxQ可能有多个，所以全部挑选出来，随机选择一个
        action = np.random.choice(action_list)
        return action

    # 学习方法，也就是更新Q表格的方法
    def learn(self, obs, action, reward, next_obs, done):
        """ off-policy
            obs：交互前的obs，即s_t
            action：本次交互选择的action，即a_t
            reward：本次动作获得的奖励r
            next_obs：本次交互后获得的obs，即s_t+1
            done：episode是否结束
        """   
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward   # 表示没有下一个状态，target即为reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :])
        self.Q[obs, action] += self.lr * (target_Q - predict_Q) # 修正q

    # 将Q表格的数据保存到文件汇总
    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    # 从文件中读取数据到Q表格
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')
    