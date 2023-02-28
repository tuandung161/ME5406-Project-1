from Environment import Frozen_Lake_Env

class QLearning():
    def __init__(self, map_size=4, n_episode = 1000, learn_rate=0.05, epsilon=0.5, gamma=0.8):
        self.map_size = map_size
        self.env = Frozen_Lake_Env(self.map_size)

        self.epsilon = epsilon
        self.gamma = gamma

        self.n_episode = n_episode
        self.learn_rate = learn_rate

        

