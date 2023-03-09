from Environment import Frozen_Lake_Env
import random
import numpy as np

class MonteCarlo():
    def __init__(self, map_size, n_step=1000, n_episode = 100000, epsilon=0.1, gamma=0.9):
        self.map_size = map_size
        self.env = Frozen_Lake_Env(self.map_size)
        self.n_observation = self.env.len_x * self.env.len_y
        self.n_action = len(self.env.action)
        self.epsilon = epsilon
        self.gamma = gamma
        
        self.Q_table, self.Return_table, self.N_table = self.initialize()
        
        #Parameter for train set
        self.n_episode = n_episode
        self.n_step = n_step

        self.train_success = 0
        self.train_fail = 0
        
        self.total_reward = []
        self.avg_reward = 0

        #Parameter for test set
        self.test_eps = 0
        self.test_sucess = 0
        self.test_fail= 0
        self.test_reward = []
        self.test_avg_reward = []


    def initialize(self):
        Q_table = np.zeros((self.n_observation, self.n_action))
        Return_table = np.zeros((self.n_observation, self.n_action))
        N_table = np.zeros((self.n_observation, self.n_action))
                
        return Q_table, Return_table, N_table
                        
                           
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
    
    def optimal_policy(self,state):
        return np.argmax(self.Q_table[state])
    
    def run_episode(self):
        state_action = []
        reward_ls = []

        #Reset and update initial state
        self.env.reset()
        state = self.env.state

        current_step = 0

        for i in range(self.n_step): 
            action = self.greedy_policy(state)
            state_action.append((state, action))
            state, reward, complete = self.env.step(action)
            reward_ls.append(reward)
            current_step += 1
            
            if complete:
                if reward > 0:
                    self.train_success += 1
                else: 
                    self.train_fail += 1
                
                #print("Episode Compete: " + str(current_step) + " steps")
                break
        #print("Episode Compete: " + str(current_step) + " steps")
        return state_action, reward_ls, reward
    
    def train(self):
        episode = 0
        for i in range(self.n_episode):
            state_action, reward, last_reward = self.run_episode()
            episode += 1
            G = 0

            #Compute G
            for j in range(len(state_action)-1,-1,-1):
                state_j, action_j = state_action[j]
                G = self.gamma * G + reward[j]
                if not (state_j, action_j) in state_action[:j]:
                    self.Return_table[state_j][action_j] += G
                    self.N_table[state_j][action_j] += 1
                    self.Q_table[state_j][action_j] = self.Return_table[state_j][action_j]/self.N_table[state_j][action_j]
            
            self.total_reward.append(last_reward)
            self.avg_reward = sum(self.total_reward)/(i+1)
            print("Episode ", episode, " completed")
            print("AVG reward: " + str(self.avg_reward))
        return self.Q_table
    
    def test(self, test_eps):
        self.test_eps = test_eps
        
        for episode in range(self.test_eps):
            self.env.reset()
            state = self.env.state
            for step in range(self.n_step):
                action = self.optimal_policy(state)
                state, reward, complete = self.env.step(action)
                
                if complete or step == self.n_step - 1:
                    if reward <= 0:
                        self.test_fail += 1
                    elif reward > 0:
                        self.test_sucess += 1
                    break
            self.test_reward.append(reward)
            avg_reward = sum(self.test_reward)/(episode+1)
            self.test_avg_reward.append(avg_reward)
        print("Test average reward: ", avg_reward)
        print("Test success percentage: ", self.test_sucess/self.test_eps * 100, "%")

if __name__ == "__main__": 
    #Initialize & train robot:
    map_size = int(input("Input Map Size: "))
    robot = MonteCarlo(map_size=map_size)
    Q = robot.train()





            







    


        
