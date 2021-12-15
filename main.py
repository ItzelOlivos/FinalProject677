import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Arrow
from agents import *

VIS = True

# ======== Predator properties ========
PRED_RAD = 10
PRED_TRANS = 1.9
PRED_PN = 20
PRED_MAX_VEL = PRED_PN

# Predator's cost function
Qx = np.array([[0.001, 0], [0, 0.001]])
Qu = np.array([[100, 0], [0, 100]])
QT = np.array([[0.001, 0], [0, 0.001]])

# ======== Agent properties ===========
NUM_AGENTS = 20
AGENT_WIDTH = 5
AGENT_MAX_VEL = 1
SENSOR_VAR = 10
INIT_GUESS = np.array([0, 0])
COM_GRAPH = np.zeros([NUM_AGENTS, NUM_AGENTS])
COM_REACH = .8
ROUNDS = 1

# RN and COM_REACH
# E1: RN=500, REACH=.8 (h SC)
# E2: RN=100, REACH=.8 (m SC)
# E3: RN=60, REACH=.8 (l SC)
# E4: RN=60, REACH=.8, R=20 (m SC w cons)
# ======= Flocking control params =====
RN = 100
RD = 50
RP = 150
RDP = 100
RC = RD + (RN - RD)*COM_REACH

TEST_ID = f"ignore_me"

E = 0.9
K1 = 5.
K2 = .1
K3 = 10.
K4 = .1

H = 0.9

# ====== Environment properties =======
TIME_STEPS = 150
BKG_SCALE = 400
TARGET_MAX_VEL = 5

# =========== Statistics ==============
messages = np.zeros([TIME_STEPS, NUM_AGENTS])
dist2neighbors = np.zeros([TIME_STEPS, NUM_AGENTS])
dist2target = np.zeros([TIME_STEPS, NUM_AGENTS])
dist2obstacle = np.zeros([TIME_STEPS, NUM_AGENTS])
abs_error = np.zeros([TIME_STEPS, NUM_AGENTS, 2])
disconnections = np.zeros(TIME_STEPS)


def preparing_workspace():
    fig = plt.figure(figsize=(7, 7))
    plt.axis([-BKG_SCALE, BKG_SCALE, -BKG_SCALE, BKG_SCALE])
    ax = plt.gca()
    ax.set_aspect('equal')
    return [fig, ax]


def animate(i):
    ax.clear()
    plt.axis([-BKG_SCALE, BKG_SCALE, -BKG_SCALE, BKG_SCALE])
    ax.add_patch(Arrow(target.pos[0], target.pos[1], 20, -20, width=50, color="gold"))
    for agent in agents:
        # shape = Arrow(agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1], width=AGENT_WIDTH, color="royalblue")
        # ax.add_patch(shape)
        ax.plot(agent.pos[0], agent.pos[1], '.', color='olivedrab')
        for id in agent.connected:
            plt.plot([agent.pos[0], agents[id].pos[0]], [agent.pos[1], agents[id].pos[1]], 'k--', linewidth=0.1)

        if agent.obstacle is not None:
            plt.plot([agent.pos[0], agent.obstacle[0]], [agent.pos[1], agent.obstacle[1]], 'r--',linewidth=0.1)

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


def sigma1(z):
    return z / np.sqrt(1 + z ** 2)


def phi(z):
    a = .25
    b = .5
    c = abs(a - b) / np.sqrt(4 * a * b)
    return ((a + b) * sigma1(z + c) + (a - b)) / 2


def phiAlpha(j, i):
    z = signorm(agents[j].pos - agents[i].pos)
    return bumpfunc(z / signorm(RN)) * phi(z - signorm(RD))


def phiBeta(pos, i):
    z = signorm(pos - agents[i].pos)
    return bumpfunc(z / signorm(RDP)) * (sigma1(z - signorm(RDP)) - 1)


def getNeighbors(i):
    # Sensing devices within rad R
    candidates = [id for id in range(NUM_AGENTS) if id != i]
    res = []
    res_conn = []
    for c in candidates:
        if np.linalg.norm(agents[c].pos - agents[i].pos) <= RN:
            res.append(c)
            if np.linalg.norm(agents[c].pos - agents[i].pos) <= RC:
                res_conn.append(c)
                COM_GRAPH[i, c] = -1
                COM_GRAPH[c, i] = -1

    agents[i].neighbors = res
    agents[i].connected = res_conn


def findPredator(i, obs):
    # Only agents within the range receive an observation
    if np.linalg.norm(predator.pos - agents[i].pos) <= RP:
        estimated_pos = (A - B.dot(L[i]) - K[i].dot(A) + K[i].dot(B.dot(L[i]))).dot(agents[i].estimate) + K[i].dot(obs)
        agents[i].estimate = estimated_pos

# ============= Main code =============

init_pos = np.asarray([380, 380])
pred_pos0 = np.asarray([400, 400])
target_pos0 = np.asarray([-400, 400])

[fig, ax] = preparing_workspace()
plt.ion()
plt.show()

# Predator's control:
S = np.zeros([TIME_STEPS + 1, 2, 2])
L = np.zeros([TIME_STEPS + 1, 2, 2])
S[TIME_STEPS] = QT
A = PRED_TRANS * np.eye(2)
B = np.eye(2)
for k in range(TIME_STEPS-1, -1, -1):
    S[k] = A.T.dot(S[k+1] - S[k+1].dot(B).dot(np.linalg.inv(B.T.dot(S[k+1].dot(B)) + Qu).dot(B.T).dot(S[k+1]))).dot(A) + Qx
    L[k] = np.linalg.inv(B.T.dot(S[k+1].dot(B)) + Qu).dot(B.T.dot(S[k+1].dot(A)))

# Agents' estimator error covariance
PN = PRED_PN * np.eye(2)
SN = SENSOR_VAR * np.eye(2)

P = np.zeros([TIME_STEPS + 1, 2, 2])
K = np.zeros([TIME_STEPS, 2, 2])
P[0] = SENSOR_VAR
for k in range(TIME_STEPS):
    P[k+1] = A.dot(P[k] - P[k].dot(np.linalg.inv(P[k] + SN).dot(S[k]))).dot(A.T) + PN
    K[k] = P[k].dot(np.linalg.inv(P[k] + SN))

agents = [Agent(idx, init_pos + AGENT_WIDTH * np.random.randn(), np.random.randn(), AGENT_MAX_VEL, INIT_GUESS) for idx in range(NUM_AGENTS)]
predator = Predator(pred_pos0, A, PRED_PN, SENSOR_VAR)
target = Target(target_pos0)

for i in range(NUM_AGENTS):
    getNeighbors(i)
    findPredator(i, pred_pos0)

actions = [np.zeros(2) for _ in range(NUM_AGENTS)]
for t in range(TIME_STEPS):
    if VIS:
        anim = animation.FuncAnimation(fig, animate, interval=1, blit=True)

    # Measuring avg distance to target
    dist2target[t] = np.array([np.linalg.norm(agents[i].pos - target.pos) for i in range(NUM_AGENTS)])

    # Measuring avg distance to obstacle
    dist2obstacle[t] = np.array([np.linalg.norm(agents[i].pos - predator.pos) for i in range(NUM_AGENTS)])

    # Measuring avg distance between neighbors
    dist2neighbors[t] = np.array([np.mean([np.linalg.norm(agents[i].pos - agents[id].pos) for id in agents[i].neighbors]) if len(agents[i].neighbors)!=0 else 0 for i in range(NUM_AGENTS)])

    # Measuring avg MSE in estimated position of obstacle
    abs_error[t] = np.array([(agents[i].estimate - predator.pos)**2 for i in range(NUM_AGENTS)])

    for i in range(NUM_AGENTS):
        agents[i].step(actions[i])

    obs = predator.step(-L[t].dot(predator.pos))
    target.step(np.array([TARGET_MAX_VEL, -TARGET_MAX_VEL]), BKG_SCALE)

    # Individual measurements
    COM_GRAPH = np.zeros([NUM_AGENTS, NUM_AGENTS])
    for i in range(NUM_AGENTS):
        getNeighbors(i)
        findPredator(i, obs)

        # Counting the number of disconnected nodes
        if len(agents[i].neighbors) == 0:
            disconnections[t] += 1

    # We need a graph that is irreducible (ensures connectivity), aperiodic (a limit exists), and doubly stochastic (consensus)
    laplacian = COM_GRAPH - np.sum(COM_GRAPH, axis=1) * np.eye(NUM_AGENTS)
    epsilon = .9 * 1 / max(np.diag(laplacian))
    COM_GRAPH = np.eye(NUM_AGENTS) - epsilon * laplacian

    # Collective estimation
    for _ in range(ROUNDS):
        Collective = []
        for i in range(NUM_AGENTS):
            # Each agent i receives m messages
            messages[t, i] += len(COM_GRAPH[i, :][COM_GRAPH[i, :] > 0])
            Collective.append(np.sum([COM_GRAPH[i, j] * agents[j].estimate for j in range(NUM_AGENTS)], axis=0))

    actions = []
    for i in range(NUM_AGENTS):

        agents[i].estimate = Collective[i]
        if np.linalg.norm(agents[i].estimate - agents[i].pos) < RP:
            agents[i].obstacle = agents[i].estimate
        else:
            agents[i].obstacle = None

        fc = K1 * sum([phiAlpha(j, i) * (agents[j].pos - agents[i].pos) / np.sqrt(
            1 + E * np.linalg.norm(agents[j].pos - agents[i].pos) ** 2) + bumpfunc(
            signorm(agents[j].pos - agents[i].pos) / signorm(RN)) * (agents[j].vel - agents[i].vel) for j in
                       agents[i].neighbors])

        fg = - K2 * ((agents[i].pos - target.pos) + (agents[i].vel - TARGET_MAX_VEL))

        fo = 0
        if agents[i].obstacle is not None:
            fo = K3 * phiBeta(agents[i].obstacle[0], i) * (agents[i].obstacle[0] - agents[i].pos) / np.sqrt(
            1 + E * np.linalg.norm(agents[i].obstacle[0] - agents[i].pos) ** 2) + bumpfunc(
            signorm(agents[i].obstacle[0] - agents[i].pos) / signorm(RDP)) * (PRED_MAX_VEL - agents[i].vel)

        fv = K4 * sum(
            [bumpfunc(signorm(agents[j].pos - agents[i].pos) / signorm(RN)) * (agents[j].vel - agents[i].vel) for j in
             agents[i].neighbors])

        actions.append(fc + fg + fo + fv)

    plt.pause(0.001)

np.save(f'Stats/messages_{TEST_ID}.npy', messages)
np.save(f'Stats/dist2neighbors_{TEST_ID}.npy', dist2neighbors)
np.save(f'Stats/dist2target_{TEST_ID}.npy', dist2target)
np.save(f'Stats/dist2obstacle_{TEST_ID}.npy', dist2obstacle)
np.save(f'Stats/abs_error_{TEST_ID}.npy', abs_error)
np.save(f'Stats/disconnections_{TEST_ID}.npy', disconnections)

stamps = np.arange(TIME_STEPS)

plt.close()
plt.figure()
mean_messages = np.mean(messages, axis=1)
std_messages = 2*np.std(messages, axis=1)
plt.fill_between(stamps, mean_messages + std_messages, mean_messages - std_messages, color='grey', alpha=0.5, label='CI')
plt.plot(stamps, mean_messages, label='mean', color='midnightblue')
plt.ylabel('Messages')
plt.xlabel('time stamps')
plt.legend()
plt.show()

plt.figure()
plt.bar(stamps, disconnections, color='midnightblue')
plt.ylabel('Disconnected agents')
plt.xlabel('time stamps')
plt.show()

plt.figure()
mean_dist2neighbors = np.mean(dist2neighbors, axis=1)
std_dist2neighbors = 2*np.std(dist2neighbors, axis=1)
plt.fill_between(stamps, mean_dist2neighbors + std_dist2neighbors, mean_dist2neighbors - std_dist2neighbors, color='grey', alpha=0.5, label='CI')
plt.plot(stamps, mean_dist2neighbors, label='mean', color='midnightblue')
plt.ylabel('Average distance to neighbors')
plt.xlabel('time stamps')
plt.legend()
plt.show()

plt.figure()
mean_dist2target = np.mean(dist2target, axis=1)
std_dist2target = 2*np.std(dist2target, axis=1)
plt.fill_between(stamps, mean_dist2target + std_dist2target, mean_dist2target - std_dist2target, color='grey', alpha=0.5, label='CI')
plt.plot(stamps, mean_dist2target, label='mean', color='midnightblue')
plt.ylabel('Distance to target')
plt.xlabel('time stamps')
plt.legend()
plt.show()

plt.figure()
mean_dist2obstacle = np.mean(dist2obstacle, axis=1)
std_dist2obstacle = 2*np.std(dist2obstacle, axis=1)
plt.fill_between(stamps, mean_dist2obstacle + std_dist2obstacle, mean_dist2obstacle - std_dist2obstacle, color='grey', alpha=0.5, label='CI')
plt.plot(stamps, mean_dist2obstacle, label='mean', color='midnightblue')
plt.ylabel('Distance to obstacle')
plt.xlabel('time stamps')
plt.legend()
plt.show()

mean_MSE_x = np.mean(abs_error[:, :, 0], axis=1)
std_MSE_x = 2*np.std(abs_error[:, :, 0], axis=1)
mean_MSE_y = np.mean(abs_error[:, :, 1], axis=1)
std_MSE_y = 2*np.std(abs_error[:, :, 1], axis=1)
plt.figure()
plt.fill_between(stamps[:-2], mean_MSE_x[:-2] + std_MSE_x[:-2], mean_MSE_x[:-2] - std_MSE_x[:-2], color='grey', alpha=0.5, label='CI (x)')
plt.plot(stamps[:-2], mean_MSE_x[:-2], label='mean (x)', color='midnightblue')
plt.ylabel('Absolute error')
plt.xlabel('time stamps')
plt.legend()
plt.show()
input()

