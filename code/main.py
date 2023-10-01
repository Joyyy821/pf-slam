# File: main.py
# Author: Yang Jiao
# Last update: March 7, 2023
# --------------------------
# Code to start the project.

from matplotlib import pyplot as plt

from models import Robot
from pr2_utils import tic, toc

def main_slam():
    # SLAM
    N_pt = 5
    rob_slam = Robot(dataset=20, mode=2, N_pt=N_pt)
    ts = tic()
    rob_slam.SLAM()
    toc(ts, "Particle filter with "+str(N_pt)+" particles")
    rob_slam.generateGIF()

    # Plot the trajectory
    if rob_slam.mode >= 1:
        rob_slam.showParticles()

    plt.ioff()
    plt.figure()
    plt.plot(rob_slam.states[:,0], rob_slam.states[:,1])
    plt.title("Robot state x-y trajectory")

    plt.show()

def main_texture_mapping():
    rob_texture_map = Robot(dataset=20, mode=3)
    rob_texture_map.textureMapping()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # Remove the comment for the corresponding parts to run the program

    main_slam()

    # main_texture_mapping()
