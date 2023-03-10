from Environment import Frozen_Lake_Env
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import time
import math

class MonteCarlo():
    def __init__(self, map_size, n_step=1000, n_episode = 10000, epsilon=0.1, gamma=0.9, epsilon_mode=False):
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
        self.step_count = []
        self.episode_count = []

        self.train_success = 0
        self.train_fail = 0
        
        self.total_reward = []
        self.avg_reward = []
        self.train_time = 0
        self.epsilon_mode = epsilon_mode

        #Parameter for test set
        self.test_eps = 0
        self.test_sucess = 0
        self.test_fail= 0
        self.test_reward = []
        self.test_avg_reward = []
        self.test_step_count = []
        self.test_episode_count = []


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
            #print(max_)
            max_index = random.choice([i for i in range(len(policy)) if policy[i] == max_])
            #print("Max index: ", max_index)
            return max_index
        
    def decay_epsilon(self, epsilon_start=0.3, epsilon_end=0.1, decay_rate=10000, t=0):
        epsilon = epsilon_start * (epsilon_end / epsilon_start) ** (t / decay_rate)
        return epsilon
    
    def optimal_policy(self,state):
        return np.argmax(self.Q_table[state])
    
    def run_episode(self):
        state_action = []
        reward_ls = []

        #Reset and update initial state
        self.env.reset()
        state = self.env.state

        current_step = 0
        
        if self.epsilon_mode == True:
            self.epsilon = self.decay_epsilon(t=self.episode_count[-1])
        print("Epsilon Value: ", self.epsilon)

        for i in range(self.n_step): 
            current_step += 1
            action = self.greedy_policy(state)
            state_action.append((state, action))
            state, reward, complete = self.env.step(action)
            reward_ls.append(reward)
            
            if complete:
                if reward > 0:
                    self.train_success += 1
                else: 
                    self.train_fail += 1
                
                #print("Episode Compete: " + str(current_step) + " steps")
                break
        #print("Episode Compete: " + str(current_step) + " steps")
        self.step_count.append(current_step)
        return state_action, reward_ls, reward
    
    def train(self):
        episode = 0
        start_time = time.time()
        for i in range(self.n_episode):
            episode += 1
            self.episode_count.append(episode)
            state_action, reward, last_reward = self.run_episode()
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
            self.avg_reward.append(sum(self.total_reward)/(i+1))
            print("Episode ", episode, " completed")
            print("AVG reward: " + str(self.avg_reward[-1]))
        end_time = time.time()
        self.train_time = round(end_time-start_time,2)
        print("Train success percentage: ", self.train_success/self.n_episode * 100, "%")
        print("Train Duration: ", self.train_time, " seconds")
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
                    self.test_step_count.append(step+1)
                    break
            self.test_reward.append(reward)
            avg_reward = sum(self.test_reward)/(episode+1)
            self.test_avg_reward.append(avg_reward)
            self.test_episode_count.append(episode+1)
        print("Test average reward: ", avg_reward)
        print("Test success percentage: ", self.test_sucess/self.test_eps * 100, "%")

    def plot(self):
        terminate = False
        i = 0
        while not terminate: 
            fig, axes = plt.subplots(1,1)
            i += 1
            #print_mode = int(input("Print mode: "))
            print_mode = i
            #plt.tight_layout(pad=10)
            #plt.figure(figsize=(60,40))
            #fig.autoscale(True)
            if print_mode == 1: 
                #Plot success vs fail count for train set
                axes.set_title("[Train] Success and Fail Count")
                axes.bar(["Success", "Fail"], height=[int(self.train_success), int(self.train_fail)], align='center', color=["green", "red"])
                axes.set(ylabel="Count")
                fig.savefig("figure/Monte_Carlo/%sx%s/[Train] Success and Fail Count.png"%(str(self.map_size),str(self.map_size)))
            elif print_mode == 2: 
                #Plot success vs fail count for test set
                axes.set_title("[Test] Success and Fail Count")
                axes.bar(["Success", "Fail"], height=[int(self.test_sucess), int(self.test_fail)], align='center', color=["green", "red"])
                axes.set(ylabel="Count")
                fig.savefig("figure/Monte_Carlo/%sx%s/[Test] Success and Fail Count.png"%(str(self.map_size),str(self.map_size)))
            elif print_mode == 3:
                #Plot average reward over time for train set
                axes.set_title("[Train] Average reward vs episode")
                axes.set(xlabel="Episode", ylabel="Average Reward")
                axes.plot(self.avg_reward)
                axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                axes.set_ylim(ymax=1,ymin=-1)
                fig.savefig("figure/Monte_Carlo/%sx%s/[Train] Average reward vs episode.png"%(str(self.map_size),str(self.map_size)))
            elif print_mode == 4:
                #Plot average reward over time for test set
                axes.set_title("[Test] Average reward vs episode")
                axes.set(xlabel="Episode", ylabel="Average Reward")
                axes.plot(self.test_avg_reward)
                axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                axes.set_ylim(ymax=1.1,ymin=-1.1)
                fig.savefig("figure/Monte_Carlo/%sx%s/[Test] Average reward vs episode.png"%(str(self.map_size),str(self.map_size)))
            elif print_mode == 5:
                #Plot steps per episode for train set
                axes.set_title("[Train] Number of step per episode")
                axes.scatter(self.episode_count, self.step_count, marker=".")
                axes.set(xlabel="Episode", ylabel="Steps")
                axes.set_ylim(ymax=1000,ymin=0)
                fig.savefig("figure/Monte_Carlo/%sx%s/[Train] Number of step per episode.png"%(str(self.map_size),str(self.map_size)))
            elif print_mode == 6:
                #Plot steps per episode for test set
                axes.set_title("[Test] Number of step per episode")
                axes.scatter(self.test_episode_count, self.test_step_count, marker=".")
                axes.set(xlabel="Episode", ylabel="Steps")
                axes.set_ylim(ymax=25,ymin=0)
                fig.savefig("figure/Monte_Carlo/%sx%s/[Test] Number of step per episode.png"%(str(self.map_size),str(self.map_size)))
            elif print_mode == 7:
                print("Plot completed and saved")
                terminate = True

if __name__ == "__main__": 
    #Initialize & train robot:
    map_size = int(input("Input Map Size: "))
    dynamic_epsilon = int(input("Epsilon Mode: "))
    if dynamic_epsilon == 0:
        robot = MonteCarlo(map_size=map_size)
    elif dynamic_epsilon == 1: 
        robot = MonteCarlo(map_size=map_size, epsilon_mode=True)
    Q = robot.train()
    robot.test(1000)
    robot.plot()





            







    


        
