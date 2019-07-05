from rl_utils import NeuralNet, ReplayMemory, Transition
from rl_agent import RLAgent

from collections import namedtuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class DQN(RLAgent):
    def __init__(self, env, params):
        super().__init__(env, **params.agent)

        self.env = env

        if params.device == 'cpu' or params.device == 'cuda':
            self.device = torch.device(params.device)
        else:
            print("Invalid device: default to cpu")
            self.device = torch.device('cpu')

        self.input_size = 9
        self.hidden_size = 128
        self.output_size = self.env.action_size

        self.memory_size = 2560
        self.batch_size = 128
        self.clip = 10
        self.cur_step = 0
        self.episode = 0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.1
        self.decay_rate = 1000
        self.var_epsilon = True

        dqn_net = NeuralNet(self.input_size, self.hidden_size, self.output_size)
        self.policy_net = dqn_net.to(self.device)
        self.target_net = dqn_net.to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())

        self.memory = ReplayMemory(self.memory_size)
    
    def act(self, state):
        state = self.state_to_tensor(state)
        exp_exp_tradeoff = random.uniform(0,1)

        if self.training:
            self.cur_step += 1

        if self.training and exp_exp_tradeoff < self.epsilon:
            action = self.env.sample_action()
        else:
            self.policy_net.eval()
            with torch.no_grad():
                policy_out = self.policy_net(state.to(self.device))
            self.policy_net.train()

            open_actions = self.env.possible_actions()
            open_policy = np.array([-float('Inf')] * self.env.action_size)
            open_policy[open_actions] = np.array(policy_out.flatten().tolist())[open_actions]
            max_actions = np.argwhere(open_policy == np.max(open_policy)).flatten()
            action = np.random.choice(max_actions)
            # action = policy_out.max(1)[1].view(1, 1).item()

        return action
    
    def learn(self, *args):
        self.episode = args[0]

        if self.var_epsilon:
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-1.0 * self.cur_step / self.decay_rate)

        if self.memory.size() <= self.batch_size:
            return

        batch = self.sample()

        non_final_mask = torch.tensor(tuple([s is not None for s in batch.next_state]),
                                        device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)

        state_batch = state_batch.to(self.device)
        non_final_next_states = non_final_next_states.to(self.device)

        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_q_vals = self.policy_net(state_batch).gather(1, action_batch)

        next_state_vals = torch.zeros(self.batch_size, device=self.device)
        
        # print(non_final_mask)
        # print(non_final_next_states)
        # print(self.target_net(non_final_next_states).max(1))
        # print(self.target_net(non_final_next_states).max(1)[0])
        # print(self.target_net(non_final_next_states).max(1)[0].detach())
        # print('\n')
        
        next_state_vals[non_final_mask] = self.max_opp_max_resp(non_final_next_states)

        # next_state_vals[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_q_values = (next_state_vals * self.gamma) + reward_batch

        loss = F.mse_loss(state_q_vals, expected_state_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip)
        self.optimizer.step()

        if self.cur_step % 2 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def push(self, *args):
        state = self.state_to_tensor(args[0])
        action = torch.tensor([[args[1]]])
        reward = torch.tensor([args[2]], dtype=torch.float, device=self.device)
        done = torch.tensor([args[4]])

        if not done:
            next_state = self.state_to_tensor(args[3])
        else:
            next_state = args[3]

        self.memory.push(state, action, reward, next_state, done)

    def max_opp_max_resp(self, non_final_next_states):
        """
        For now, not checking if either action is legal
        """
        vals = []

        for state_tensor in non_final_next_states:
            state_tuple = self.tensor_to_state(state_tensor)
            op_open_actions = self.env.possible_actions(state=state_tuple)
            op_open_mask = torch.tensor([op_open_actions], dtype=torch.long, device=self.device)

            op_open_vals = torch.tensor([[-10000] * self.env.action_size], dtype=torch.float, device=self.device)
            op_open_vals[0][op_open_mask] = self.target_net(state_tensor.unsqueeze(0))[0][op_open_mask]

            op_action_tensor = op_open_vals.max(1)
            op_action = op_action_tensor[1].detach().item()
            op_value = op_action_tensor[0].detach().item()

            if op_action not in op_open_actions:
                import pdb; pdb.set_trace()
                print("INVALID OP ACTION")
                print(self.target_net(state_tensor.unsqueeze(0)))
                print(op_action)
                print(op_open_actions)
            
            if len(op_open_actions) == 1:
                vals.append(-op_value)
                continue

            player = tuple(self.env.get_next_player(state_tuple))

            temp_state = state_tuple[:op_action] + player + state_tuple[op_action+1:]
            temp_state_tensor = self.state_to_tensor(temp_state)
            temp_state_tensor = temp_state_tensor.to(self.device)
            open_actions = self.env.possible_actions(state=temp_state)
            open_mask = torch.tensor([open_actions], dtype=torch.long, device=self.device)

            open_vals = torch.tensor([[-10000] * self.env.action_size], dtype=torch.float, device=self.device)
            open_vals[0][open_mask] = self.target_net(temp_state_tensor)[0][open_mask]

            action_tensor = open_vals.max(1)
            action = action_tensor[1].detach().item()
            value = action_tensor[0].detach().item()

            if action not in open_actions:
                import pdb; pdb.set_trace()
                print("INVALID ACTION")
                print(self.target_net(temp_state_tensor))
                print(action)
                print(open_actions)

            q_val = -op_value + value
            vals.append(q_val)
        
        return torch.tensor(vals, dtype=torch.float, device=self.device)        

    def sample(self):
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*list(zip(*transitions)))
        return batch

    def state_to_tensor(self, state):
        state_list = []
        tile_to_num = {
            ' ': 0,
            'X': 1,
            'O': -1
        }

        for tile in state:
            state_list.append(tile_to_num[tile])

        state_tensor = torch.tensor([state_list], dtype=torch.float)
        return state_tensor
    
    def tensor_to_state(self, tensor):
        state_list = tensor.tolist()
        num_to_tile = {
            0: ' ',
            1: 'X',
            -1: 'O'
        }
        state = tuple([num_to_tile[num] for num in state_list])
        return state
