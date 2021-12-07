import numpy as np

class Agent:
    def __init__(self, id, pos0, vel0, max_speed, init_guess):
        self.id = id
        self.pos = pos0
        self.vel = vel0
        self.neighbors = []
        self.obstacle = None
        self.max_speed = max_speed
        self.estimate = init_guess

    def step(self, u):
        self.vel += u
        self.pos += self.vel
        # limit
        if np.linalg.norm(self.vel) > self.max_speed:
            self.vel = self.vel / np.linalg.norm(self.vel) * self.max_speed

    def measure(self):
        pass


class Predator:
    def __init__(self, pos0, A, sig_p, sig_z):
        self.pos = pos0
        self.A = A
        self.proc_std = sig_p
        self.obs_std = sig_z

    def step(self, u):
        self.pos = self.A.dot(self.pos) + u + self.proc_std * np.random.randn(2)
        # self.pos = np.mod(self.pos + u, top)
        return self.pos + self.obs_std * np.random.randn(2)


class Target:
    def __init__(self, pos0):
        self.pos = pos0

    def step(self, u, top):
        next = self.pos + u

        if next[0] > top:
            next = np.array([-top, top])

        self.pos = next
