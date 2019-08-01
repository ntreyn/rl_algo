import numpy as np
import random

from collections import deque, defaultdict

from rl_agent import RLAgent
from rl_utils import Transition

class new_q(RLAgent):
    def __init__(self, env, args):
        super().__init__(env, **args)

        self.env = env
        self.memory = deque(maxlen=1)

        action_size = self.env.action_size
        self.Q = defaultdict(lambda: np.zeros(action_size))

        self.episode = 0
        self.player = 'X'

    def act(self, state):

        if self.training and random.uniform(0,1) < self.epsilon:
            action = self.env.sample_action()
        else:
            state = tuple(state)
            max_actions, _ = self.get_best_actions(state=state)
            action = np.random.choice(max_actions)

        return action

    def get_best_actions(self, state):
        if self.env.player == self.player:
            open_actions = self.env.possible_actions()
            state_q = np.array([-float('Inf')] * self.env.action_size)
            state_q[open_actions] = self.Q[state][open_actions]
            best_q = np.max(state_q)
            best_actions = np.argwhere(state_q == best_q).flatten()
        else:
            open_actions = self.env.possible_actions()
            state_q = np.array([float('Inf')] * self.env.action_size)
            state_q[open_actions] = self.Q[state][open_actions]
            best_q = np.min(state_q)
            best_actions = np.argwhere(state_q == best_q).flatten()
        return best_actions, best_q
        

    def learn(self, *args):

        if self.var_epsilon:
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * self.episode)

        self.episode = args[0]

        mem = self.memory.pop()
        state = tuple(mem.state)
        action = mem.action
        reward = mem.reward
        done = mem.done

        if done:
            if self.env.player != self.player:
                reward = -reward  
            self.Q[state][action] = self.Q[state][action] + self.alpha * reward
        else:
            next_state = tuple(mem.next_state)

            _, next_max_q = self.get_best_actions(state=next_state)

            if self.env.player != self.player:
                reward = -reward  
            self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * next_max_q - self.Q[state][action])


    def push(self, *args):
        self.memory.append(Transition(*args))