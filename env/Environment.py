import pickle
import numpy as np

class Frozen_Lake_Env:
    def __init__(self, size):
        self.map_list = {4:"./map_4_4.p"}
        self.map = np.array(pickle.load(open(self.map_list[size], "rb")))
        self.len_x, self.len_y = self.map.shape


if __name__ == '__main__':
    map_size = int(input("Select Map Size: "))
    env = Frozen_Lake_Env(map_size)
    #print(env.map, env.len_x, env.len_y)
