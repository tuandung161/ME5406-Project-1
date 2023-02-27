from Environment import Frozen_Lake_Env
import random

class MonteCarlo():
    def __init__(self, map_size=4, n_step=100, n_episode = 1000, epsilon=0.5, gamma=0.5):
        self.map_size = map_size
        self.env = Frozen_Lake_Env(self.map_size)
        self.n_observation = self.env.len_x * self.env.len_y
        self.n_action = len(self.env.action)
        self.epsilon = epsilon
        self.gamma = gamma
        
        self.Q_table, self.Return_table, self.N_table = self.initialize()

        self.n_episode = n_episode
        self.n_step = n_step

        self.train_success = 0
        self.train_fail = 0


    def initialize(self):
        Q_table, Return_table, N_table = {}, {}, {}

        for i in range(self.n_observation):
            for j in range(self.n_action):
                Q_table[(i,j)], N_table[(i,j)], Return_table[(i,j)] = 0, 0, 0
                
        return Q_table, Return_table, N_table
                        
    
    def greedy_policy(self, state):
        if self.epsilon < random.uniform(0,1):
            return random.randint(0,3)
        else:
            return max(list(range(self.n_action)), key=lambda x: self.Q_table[(state, x)])
    
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
            self.N_table[(state, action)] += 1
            state, reward, complete = self.env.step(action)
            reward_ls.append(reward)
            current_step += 1
            
            if self.env.complete:
                if reward > 0:
                    self.train_success += 1
                else: 
                    self.train_fail += 1
                print("Episode Compete: " + str(current_step) + " steps")
                break
        return state_action, reward_ls
    
    def train(self):
        for i in range(self.n_episode):
            state_action, reward = self.run_episode()
            G = 0

            #Compute G
            for j in range(len(state_action)-1,-1,-1):
                state_j, action_j = state_action[j]
                G = self.gamma * G + reward[j]
                if not (state_j, action_j) in state_action[:j]:
                    self.Return_table[(state_j, action_j)] += G
                    self.Q_table[(state_j, action_j)] = self.Return_table[(state_j, action_j)]/self.N_table[(state_j, action_j)]
        
        return self.Q_table


if __name__ == "__main__": 
    #Initialize & train robot:
    robot = MonteCarlo()
    Q = robot.train()





            







    


        
