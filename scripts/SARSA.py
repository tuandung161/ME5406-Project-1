from Environment import Frozen_Lake_Env
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import time

class SARSA():
    def __init__(self, map_size=10, n_episode = 10000, learn_rate=0.1, epsilon=0.1, gamma=0.9,epsilon_mode = False):
        self.map_size = map_size
        self.env = Frozen_Lake_Env(self.map_size)

        self.n_observation = self.env.len_x * self.env.len_y
        self.n_action = len(self.env.action)
        self.epsilon = epsilon
        self.gamma = gamma
        self.learn_rate = learn_rate

        self.Q_table = self.initialize()

        #Parameter for train set
        self.n_episode = n_episode
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
        self.test_step_limit = 100
        self.test_reward = []
        self.test_avg_reward = []
        self.test_step_count = []
        self.test_episode_count = []

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
        
    def decay_epsilon(self, epsilon_start=0.1, epsilon_end=0.01, decay_rate=10000, t=0):
        epsilon = epsilon_start * (epsilon_end / epsilon_start) ** (t / decay_rate)
        return epsilon
    
    def decay_learn_rate(self, lr_start=0.9, lr_end=0.3, decay_rate=10000, t=0):
        lr = lr_start * (lr_end / lr_start) ** (t / decay_rate)
        return lr

    def optimal_policy(self,state):
        return np.argmax(self.Q_table[state])

    def run_episode(self):
        self.env.reset()
        state = self.env.state

        current_step = 0
        
        if self.epsilon_mode == True:
            self.epsilon = self.decay_epsilon(t=self.episode_count[-1])
            self.learn_rate = self.decay_learn_rate(t=self.episode_count[-1])
        print("Epsilon Value: ", self.epsilon)
        print("Learn Rate: ", self.learn_rate)
        
        action = self.greedy_policy(state)

        while True:
            #print(action)
            next_state, reward, complete = self.env.step(action)
            #print(state, complete)
            next_action = self.greedy_policy(next_state)
            self.Q_table[state][action] += self.learn_rate * (reward + self.gamma * self.Q_table[next_state][next_action] - self.Q_table[state][action])
            
            #print("Current Step: " + str(current_step))
            
            current_step += 1
            
            state = next_state
            action = next_action
            
            if complete:
                if reward > 0: 
                    self.train_success += 1
                else: 
                    self.train_fail += 1
                self.total_reward.append(reward)
                self.step_count.append(current_step)
                #print("Reward: " + str(reward))
                print("Episode Compete: " + str(current_step) + " steps")
                break

    def train(self):
        start_time = time.time()
        for i in range(self.n_episode):
            self.episode_count.append(i+1)
            self.run_episode()
            self.avg_reward.append(sum(self.total_reward)/(i+1))
            print("AVG reward: " + str(self.avg_reward[-1]))
            print("Episode {} completed".format(i))
        end_time = time.time()
        self.train_time = round(end_time-start_time,2)
        print("Train success percentage: ", self.train_success/self.n_episode * 100, "%")
        print("Train Duration: ", self.train_time, " seconds")
        return self.Q_table
    
    def test(self, test_eps):
        self.test_eps = test_eps

        for episode in range(self.test_eps):
            self.env.reset()
            current_step = 0
            state = self.env.state

            while True:
                action = self.optimal_policy(state)
                state, reward, complete = self.env.step(action)
                current_step += 1
                if complete or current_step == self.test_step_limit:
                    if reward <= 0:
                        self.test_fail += 1
                    elif reward > 0:
                        self.test_sucess += 1
                    self.test_step_count.append(current_step)
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
                fig.savefig("figure/SARSA/%sx%s/[Train] Success and Fail Count.png"%(str(self.map_size),str(self.map_size)))
            elif print_mode == 2: 
                #Plot success vs fail count for test set
                axes.set_title("[Test] Success and Fail Count")
                axes.bar(["Success", "Fail"], height=[int(self.test_sucess), int(self.test_fail)], align='center', color=["green", "red"])
                axes.set(ylabel="Count")
                fig.savefig("figure/SARSA/%sx%s/[Test] Success and Fail Count.png"%(str(self.map_size),str(self.map_size)))
            elif print_mode == 3:
                #Plot average reward over time for train set
                axes.set_title("[Train] Average reward vs episode")
                axes.set(xlabel="Episode", ylabel="Average Reward")
                axes.plot(self.avg_reward)
                axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                axes.set_ylim(ymax=1,ymin=-1)
                fig.savefig("figure/SARSA/%sx%s/[Train] Average reward vs episode.png"%(str(self.map_size),str(self.map_size)))
            elif print_mode == 4:
                #Plot average reward over time for test set
                axes.set_title("[Test] Average reward vs episode")
                axes.set(xlabel="Episode", ylabel="Average Reward")
                axes.plot(self.test_avg_reward)
                axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                axes.set_ylim(ymax=1.1,ymin=-1.1)
                fig.savefig("figure/SARSA/%sx%s/[Test] Average reward vs episode.png"%(str(self.map_size),str(self.map_size)))
            elif print_mode == 5:
                #Plot steps per episode for train set
                axes.set_title("[Train] Number of step per episode")
                axes.scatter(self.episode_count, self.step_count, marker=".")
                axes.set(xlabel="Episode", ylabel="Steps")
                axes.set_ylim(ymax=10000,ymin=0)
                fig.savefig("figure/SARSA/%sx%s/[Train] Number of step per episode.png"%(str(self.map_size),str(self.map_size)))
            elif print_mode == 6:
                #Plot steps per episode for test set
                axes.set_title("[Test] Number of step per episode")
                axes.scatter(self.test_episode_count, self.test_step_count, marker=".")
                axes.set(xlabel="Episode", ylabel="Steps")
                axes.set_ylim(ymax=25,ymin=0)
                fig.savefig("figure/SARSA/%sx%s/[Test] Number of step per episode.png"%(str(self.map_size),str(self.map_size)))
            elif print_mode == 7:
                print("Plot completed and saved")
                terminate = True
    
if __name__ == "__main__":
    map_size = int(input("Input Map Size: "))
    dynamic_epsilon = int(input("Epsilon Mode: "))
    if dynamic_epsilon == 0:
        robot = SARSA(map_size=map_size)
    elif dynamic_epsilon == 1: 
        robot = SARSA(map_size=map_size, epsilon_mode=True)
    Q = robot.train()
    robot.test(1000)
    robot.plot()


