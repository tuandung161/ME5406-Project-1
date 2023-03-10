from MonteCarlo import MonteCarlo
from SARSA import SARSA
from QLearning import QLearning

if __name__ == "__main__": 
    #Initialize & train robot:
    map_size = int(input("Map Option: 4 or 10\nInput Map Size: "))
    algo = int(input("1: Monte Carlo\n2: SARSA\n3: Q-Learning\nChoose learning algorithms: "))
    if algo == 1: 
        dynamic_epsilon = int(input("1: default\n2: decaying\nEpsilon Mode: "))
        if dynamic_epsilon == 1:
            robot = MonteCarlo(map_size=map_size)
        elif dynamic_epsilon == 2: 
            robot = MonteCarlo(map_size=map_size, epsilon_mode=True)
    elif algo == 2:
        dynamic_epsilon = int(input("1: default\n2: decaying\nEpsilon Mode: "))
        if dynamic_epsilon == 1:
            robot = SARSA(map_size=map_size)
        elif dynamic_epsilon == 2: 
            robot = SARSA(map_size=map_size, epsilon_mode=True)
    elif algo == 3:
        robot = QLearning(map_size=map_size) 
    Q = robot.train()
    robot.test(1000)
    robot.plot()