import numpy as np
import random

from collections import deque, defaultdict

from rl_agent import RLAgent
from rl_utils import Transition

class QLearning(RLAgent):
    def __init__(self, env, args):
        super().__init__(env, **args)

        self.env = env
        self.memory = deque(maxlen=1)

        action_size = self.env.action_size
        self.Q = defaultdict(lambda: np.zeros(action_size))

        self.epsilon = 1.0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.1
        self.decay_rate = 0.0001

        self.episode = 0
        self.var_epsilon = True

    def act(self, state):

        if self.training and random.uniform(0,1) < self.epsilon:
            action = self.env.sample_action()
        else:
            state = tuple(state)
            
            """
            ttt specific action selection
            """
            
            open_actions = self.env.possible_actions()
            state_q = np.array([-float('Inf')] * self.env.action_size)
            state_q[open_actions] = self.Q[state][open_actions]
            max_actions = np.argwhere(state_q == np.max(state_q)).flatten()

            """
            general action selection
            """
            # max_actions = np.argwhere(self.Q[state] == np.max(self.Q[state])).flatten()
            action = np.random.choice(max_actions)
        return action

    def max_op_max_resp(self, state):

        # Best opponent action (from state)
        op_actions = self.env.possible_actions(state=state)

        if len(op_actions) == 0:
            return 0

        op_qs = np.array([-float('Inf')] * self.env.action_size)
        op_qs[op_actions] = self.Q[state][op_actions]
        op_max_q = np.max(op_qs)
        op_max_actions = np.argwhere(op_qs == op_max_q).flatten()

        if len(op_actions) == 1:
            return -op_max_q

        best_opponent_q_val = op_max_q
        best_response_q_val = -float('Inf')

        # Best action after each opponent action
        for op_action in op_max_actions:
            temp_state = state[:op_action] + tuple(self.env.opponent) + state[op_action+1:]
            p_actions = self.env.possible_actions(state=temp_state)
            p_qs = np.array([-float('Inf')] * self.env.action_size)
            p_qs[p_actions] = self.Q[temp_state][p_actions]
            p_max_q = np.max(p_qs)
            if p_max_q > best_response_q_val:
                best_response_q_val = p_max_q
        
        # best_response_q_val = 0
        best_opponent_q_val = 0
        
        return -best_opponent_q_val + best_response_q_val

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
            self.Q[state][action] = self.Q[state][action] + self.alpha * reward
        else:
            next_state = tuple(mem.next_state)
            next_max_q = self.max_op_max_resp(next_state)
            self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * next_max_q - self.Q[state][action])

            """
            next_state = tuple(mem.next_state)
            open_actions = self.env.possible_actions()
            next_state_q = np.array([-float('Inf')] * self.env.action_size)
            next_state_q[open_actions] = self.Q[next_state][open_actions]
            self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(next_state_q) - self.Q[state][action])
            """
            # self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])

    def push(self, *args):
        self.memory.append(Transition(*args))