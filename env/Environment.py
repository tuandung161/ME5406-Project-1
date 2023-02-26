import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Frozen_Lake_Env:
    def __init__(self, size):
        self.map_list = {4:"./map_4_4.p"}
        self.map = np.array(pickle.load(open(self.map_list[size], "rb")))
        self.len_x, self.len_y = self.map.shape
        self.action = {"left":  (-1,0),
                       "right": (0,1),
                       "up":    (1,0),
                       "down":  (0,-1)}
        self.type = {"start":0,
                     "ice":1,
                     "hole":2,
                     "goal":3}

    def draw_map(self):
        fig, axes = plt.subplots(self.len_x, self.len_y, figsize = (8,8))
        j,k = 0,0
        for i, ax in enumerate(axes.flat):
            if self.map[j][k] == self.type["start"]:
                ax.imshow(Image.open("icons/start.png"))
            elif self.map[j][k] == self.type["hole"]:
                ax.imshow(Image.open("icons/hole.png"))
            elif self.map[j][k] == self.type["goal"]:
                ax.imshow(Image.open("icons/goal.png"))
            if k < 3:
                k += 1
            else:
                k = 0
                j += 1
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)




if __name__ == '__main__':
    map_size = int(input("Select Map Size: "))
    env = Frozen_Lake_Env(map_size)
    env.draw_map()
    #print(env.map)

