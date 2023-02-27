import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Frozen_Lake_Env:
    def __init__(self, size):
        self.env_list = {4:"./map_4_4.p"}
        self.env = np.array(pickle.load(open(self.env_list[size], "rb")))
        self.len_x, self.len_y = self.env.shape
        self.type = {"start":0,
                     "ice":1,
                     "hole":2,
                     "goal":3}
        
        self.action = {"left":  [0,-1],
                       "right": [0,1],
                       "up":    [-1,0],
                       "down":  [1,0]}
        
        self.position = []
        self.state = 0
        self.complete = False
        

    #Visualize Environment
    def draw_map(self):
        fig, axes = plt.subplots(self.len_x, self.len_y, figsize = (8,8))
        j,k = 0,0
        for i, ax in enumerate(axes.flat):
            if self.env[j][k] == self.type["start"]:
                ax.imshow(Image.open("icons/start.png"))
            elif self.env[j][k] == self.type["hole"]:
                ax.imshow(Image.open("icons/hole.png"))
            elif self.env[j][k] == self.type["goal"]:
                ax.imshow(Image.open("icons/goal.png"))
            if k < 3:
                k += 1
            else:
                k = 0
                j += 1
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    def get_init_pos(self):
        for x in range(self.len_x):
            for y in range(self.len_y):
                if self.env[x][y] == self.type["start"]:
                    return [x,y]
    
    def pos_to_state(self, position): 
        n = 0
        d = {}
        for i in range(self.len_x):
            for j in range(self.len_y):
                d[(i,j)] = n
                n += 1
        return d[(position[0],position[1])]

    #Run 1 step then return reward & next step
    def step(self, direction):
        [dx, dy] = list(self.action.values())[direction]
        new_pos = [self.position[0]+dx, self.position[1]+dy]   

        reward = 0 

        #check if out of bound
        if not (0 <= new_pos[0] < self.len_x and 0 <= new_pos[1] < self.len_y):
            return self.state , reward, self.complete
        else: # if not then update new position
            self.position = new_pos
            self.state = self.pos_to_state(self.position)
            if self.env[self.position[0]][self.position[1]] == self.type["start"] or self.env[self.position[0]][self.position[1]] == self.type["ice"]:
                reward += 0
            elif self.env[self.position[0]][self.position[1]] == self.type["hole"]:
                reward -= 1
                self.complete = True
            elif self.env[self.position[0]][self.position[1]] == self.type["goal"]:
                reward += 1
                self.complete = True
            return self.state, reward, self.complete
        
    def reset(self):
        self.position = self.get_init_pos()
        self.complete = False
        self.state = self.pos_to_state(self.position)
        


if __name__ == '__main__':
    map_size = int(input("Select Map Size: "))
    env = Frozen_Lake_Env(map_size)
    env.draw_map()
    print(env.env)
    env.reset()
    print("Initial Position: " + str(env.position))
    i = 0
    for i in range(6):
        direction = int(input("Input Action: "))
        state, reward, complete = env.step(direction)
        print("Current Position: " + str(env.position))
        print("Current State: " + str(env.state))
        print("Complete: " + str(env.complete))
        print("Reward: " + str(reward))
        if complete == True:
            break