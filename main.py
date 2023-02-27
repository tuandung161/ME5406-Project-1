from scripts.MonteCarlo import MonteCarlo

if __name__ == "__main__":
    #Initialize & train robot:
    robot = MonteCarlo()
    Q = robot.train()