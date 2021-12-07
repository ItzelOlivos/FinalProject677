# importing libraries

import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Arrow
from agents import *

# ======== Predator properties ========
PRED_RAD = 10
PRED_TRANS = .9
PRED_PN = 5
PRED_ON = 10
PRED_MAX_VEL = 1

# ======== Agent properties ===========
NUM_AGENTS = 30
AGENT_WIDTH = 5
AGENT_MAX_VEL = 1

R = 100
D = 20
DO = 200
E = 0.9
C1 = 1
C2 = 10
C3 = 20
C1T = .1
C2T = .1
H = 0.9

# ====== Environment properties =======
TIME_STEPS = 300
BKG_SCALE = 400
TARGET_MAX_VEL = 5


def preparing_workspace():
    fig = plt.figure(figsize=(7, 7))
    plt.axis([-BKG_SCALE, BKG_SCALE, -BKG_SCALE, BKG_SCALE])
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.add_patch(Rectangle(init_pos, 10, 10, color="black"))
    ax.add_patch(Rectangle(target_pos0, 10, 10, color="orange"))
    return [fig, ax]


def animate(i):
    ax.clear()
    plt.axis([-BKG_SCALE, BKG_SCALE, -BKG_SCALE, BKG_SCALE])
    ax.add_patch(Rectangle(target.pos, 10, 10, color="orange"))
    for agent in agents:
        # shape = Arrow(agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1], width=AGENT_WIDTH, color="royalblue")
        # ax.add_patch(shape)
        ax.plot(agent.pos[0], agent.pos[1], '.', color='pink')
        for id in agent.neighbors:
            plt.plot([agent.pos[0], agents[id].pos[0]], [agent.pos[1], agents[id].pos[1]], color='black', linewidth=0.1)

        if agent.obstacle is not None:
            plt.plot([agent.pos[0], agent.obstacle[0]], [agent.pos[1], agent.obstacle[1]], color='blue',
                     linewidth=0.1)

    shape = Circle((predator.pos[0], predator.pos[1]), radius=PRED_RAD, color="crimson")
    ax.add_patch(shape)

    return [ax]


def signorm(z):
    return (np.sqrt(1 + E * np.linalg.norm(z) ** 2) - 1) / E


def bumpfunc(z):
    if 0 <= z < H:
        return 1
    elif H <= z <= 1:
        return (1 + np.cos(np.pi * (z - H) / (1 - H))) / 2
    else:
        return 0


def phiFunc(j, i):
    a = .25
    b = .5
    c = abs(a - b) / np.sqrt(4 * a * b)
    s = signorm(agents[j].pos - agents[i].pos)
    z = s - signorm(agents[j].pos - agents[i].pos)
    zbis = z + c
    sigma = zbis / np.sqrt(1 + zbis ** 2)
    phi = ((a + b) * sigma + (a - b)) / 2
    return bumpfunc(s / signorm(R)) * phi


def phiFuncBeta(pos, i):
    s = signorm(pos - agents[i].pos)
    z = s - DO
    sigma = z / np.sqrt(1 + z ** 2)
    return bumpfunc(s / signorm(DO)) * (sigma - 1)


def getNeighbors(i):
    # Sensing devices within rad R
    candidates = [id for id in range(NUM_AGENTS) if id != i]
    res = []
    for c in candidates:
        if np.linalg.norm(agents[c].pos - agents[i].pos) < R:
            res.append(c)

    agents[i].neighbors = res


def findPredator(i):
    # Change this line for an estimate:
    estimated_pos = predator.pos

    if np.linalg.norm(estimated_pos - agents[i].pos) < DO:
        agents[i].obstacle = estimated_pos
    else:
        agents[i].obstacle = None


# ============= Main code =============

init_pos = np.asarray([380, 380])
object_pos = np.asarray([400, 400])
target_pos0 = np.asarray([-400, 400])

[fig, ax] = preparing_workspace()
plt.ion()
plt.show()

agents = [Agent(idx, init_pos + AGENT_WIDTH * np.random.randn(), np.random.randn(), AGENT_MAX_VEL) for idx in
          range(NUM_AGENTS)]
predator = Predator(object_pos, PRED_TRANS * np.eye(2), PRED_PN, PRED_ON)
target = Target(target_pos0)

for i in range(NUM_AGENTS):
    getNeighbors(i)
    findPredator(i)

actions = [np.zeros(2) for _ in range(NUM_AGENTS)]
for t in range(TIME_STEPS):
    anim = animation.FuncAnimation(fig, animate, interval=1, blit=True)

    for i in range(NUM_AGENTS):
        agents[i].step(actions[i])

    obs = predator.step(np.array([PRED_MAX_VEL, PRED_MAX_VEL]))
    target.step(np.array([TARGET_MAX_VEL, -TARGET_MAX_VEL]), BKG_SCALE)

    actions = []
    for i in range(NUM_AGENTS):
        getNeighbors(i)
        findPredator(i)

        fg = C1 * sum([phiFunc(j, i) * (agents[j].pos - agents[i].pos) / np.sqrt(
            1 + E * np.linalg.norm(agents[j].pos - agents[i].pos) ** 2) + bumpfunc(
            signorm(agents[j].pos - agents[i].pos) / signorm(R)) * (agents[j].vel - agents[i].vel) for j in
                       agents[i].neighbors])

        fo = C3 * sum([phiFuncBeta(agents[i].obstacle, i) * (agents[i].obstacle - agents[i].pos) / np.sqrt(1 + E * np.linalg.norm(agents[i].obstacle - agents[i].pos) ** 2) + bumpfunc(signorm(agents[i].obstacle - agents[i].pos) / signorm(DO)) * (PRED_MAX_VEL - agents[i].vel) if agents[i].obstacle is not None else 0])

        fd = C2 * sum(
            [bumpfunc(signorm(agents[j].pos - agents[i].pos) / signorm(R)) * (agents[j].vel - agents[i].vel) for j in
             agents[i].neighbors])

        # fo = C3 * sum([bumpfunc(signorm(agents[i].obstacle - agents[i].pos) / signorm(DO)) * (PRED_MAX_VEL - agents[i].vel) if agents[i].obstacle is not None else 0])

        ft = - C1T * (agents[i].pos - target.pos) - C2T * (TARGET_MAX_VEL - agents[i].vel)

        actions.append(fg + fo + fd + ft)

    plt.pause(0.1)
