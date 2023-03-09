from Environment import Frozen_Lake_Env
import random
import numpy as np

class QLearning():
    def __init__(self, map_size=10, n_episode = 10000, learn_rate=0.1, epsilon=0.1, gamma=0.9):
        self.map_size = map_size
        self.env = Frozen_Lake_Env(self.map_size)
        
        
        self.n_observation = self.env.len_x * self.env.len_y
        self.n_action = len(self.env.action)
        self.epsilon = epsilon
        self.gamma = gamma

        self.n_episode = n_episode
        self.learn_rate = learn_rate

        self.Q_table = self.initialize()

        self.train_success = 0
        self.train_fail = 0
        
        self.total_reward = []
        self.avg_reward = 0


    def initialize(self):
        Q_table = np.zeros((self.n_observation, self.n_action))
        return Q_table

    def greedy_policy(self, state):
        if random.uniform(0,1) < self.epsilon:
            return random.randint(0,3)
        else:
            policy = []
            for i in range(self.n_action):
                policy.append(self.Q_table[state][i])
            max_ = max(policy)
            max_index = random.choice([i for i in range(len(policy)) if policy[i] == max_])
            #print("Max index: ", max_index)
            return max_index


    def run_episode(self): 
        self.env.reset()
        state = self.env.state

        current_step = 0
        
        action = self.greedy_policy(state)

        while True:
            #print(action)
            next_state, reward, complete = self.env.step(action)
            #print(state, complete)
            next_action = self.greedy_policy(next_state)
            
            self.Q_table[(state,action)] += self.learn_rate * (reward + self.gamma * np.max(self.Q_table[next_state, :]) - self.Q_table[state, action])

            current_step += 1
            
            state = next_state
            action = next_action

            if self.env.complete:
                if reward > 0: 
                    self.train_success += 1
                else: 
                    self.train_fail += 1
                self.total_reward.append(reward)
                #print("Reward: " + str(reward))
                print("Episode Compete: " + str(current_step) + " steps")
                break

    def train(self):
        for i in range(self.n_episode):
            self.run_episode()
            self.avg_reward = sum(self.total_reward)/(i+1)
            print("AVG reward: " + str(self.avg_reward))
            print("Episode {} completed".format(i))
        return self.Q_table

if __name__ == "__main__":
    robot = QLearning()
    Q = robot.train()


