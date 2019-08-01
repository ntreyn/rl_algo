import numpy as np
import random

from collections import defaultdict

from rl_agent import RLAgent
from rl_utils import Transition

class MCOnPolicy(RLAgent):

    def __init__(self, env, params):
        super().__init__(env, **params)

        self.env = env

        action_size = self.env.action_size
        self.Q = defaultdict(lambda: np.zeros(action_size))

        self.episode = 0

        self.memory = []
        self.done = False
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)

    def act(self, state):
        state = tuple(state)

        open_actions = self.env.possible_actions()
        state_q = np.array([-float('Inf')] * self.env.action_size)
        state_q[open_actions] = self.Q[state][open_actions]
        max_actions = np.argwhere(state_q == np.max(state_q)).flatten()
        best_action = np.random.choice(max_actions)

        if self.training:
            A = np.zeros(self.env.action_size, dtype=float)
            A[open_actions] = np.ones(len(open_actions), dtype=float) * self.epsilon / len(open_actions)
            A[best_action] += (1.0 - self.epsilon)
            action = np.random.choice(np.arange(len(A)), p=A)
        else:
            action = best_action
        
        return action

    def learn(self, *args):
        if not self.done:
            return
        
        self.episode = args[0]

        if self.var_epsilon:
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * self.episode)

        sa_in_episode = set([(tuple(x[0]), x[1]) for x in self.memory])

        for state, action in sa_in_episode:
            sa_pair = (state, action)

            first_idx = next(i for i, x in enumerate(self.memory) if tuple(x[0]) == state and x[1] == action)
            G = sum([x[2] * (self.gamma**i) for i, x in enumerate(self.memory[first_idx:])])

            self.returns_sum[sa_pair] += G
            self.returns_count[sa_pair] += 1.0
            self.Q[state][action] = self.returns_sum[sa_pair] / self.returns_count[sa_pair]

        self.memory = []

    def push(self, *args):
        self.done = args[-1]
        self.memory.append(args[:-2])