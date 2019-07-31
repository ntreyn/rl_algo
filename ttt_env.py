from gym import spaces
import numpy as np
import random

from collections import Counter

class ttt_env:

    def __init__(self, im_reward=True):
        self.num_tiles = 9
        self.action_size = 9
        self.state_size = 19683 # * 2 # 3^9 * 2
        """
            3^9 does not accuracy describe the number of actually possible states
            however, for now, the computation for determining states is simpler
        """
        self.streaks = [ [1,2,3], [1,4,7], [1,5,9], [2,5,8], [3,6,9], [3,5,7], [4,5,6], [7,8,9] ]
        self.intermediate_reward = im_reward
        self.reset()

    def reset(self):
        self.board = [' '] * self.num_tiles
        self.done = False
        self.player = 'X'
        self.opponent = 'O'
        return self.get_state()

    def step(self, action):
        if self.done:
            print("Game already over")
            quit()    
        if self.board[action] != ' ':
            print("Invalid action:", action)
            quit()

        reward = self.get_intermediate_reward(action)

        self.board[action] = self.player
        new_state = self.get_state()

        if reward >= 1000:
            self.done = True
            new_state = None
            if not self.intermediate_reward:
                reward = 1
        elif self.board_full():
            self.done = True
            new_state = None
            if not self.intermediate_reward:
                reward = 0
        else:
            self.change_turn()
            if not self.intermediate_reward:
                reward = 0

        return new_state, reward, self.done

    def get_intermediate_reward(self, a, state=None, player=None):
        action = a + 1
        reward = 0

        if state is None:
            state = self.board
        if player is None:
            player = self.player

        for streak in self.streaks:
            if action in streak:
                sums = self.streak_contains(streak, state)
                if sums[' '] == 3:
                    # Create streak(s) of 1
                    # +2 per streak
                    # continue
                    reward += 2

                elif sums[' '] == 2:
                    if sums[player] == 1:
                        # If friendly streak
                        # Create streak(s) of 2
                        # +5 per streak
                        # continue
                        reward += 5
                    else:
                        # Else
                        # Block streak(s) of 1
                        # +1 per streak
                        # continue
                        reward += 1

                elif sums[' '] == 1:
                    if sums[player] == 2:
                        # If friendly streak
                        # Create streak of 3
                        # +1000
                        reward += 1000
                    elif sums[player] == 1:
                        # Else if mixed streak
                        # +0
                        continue
                    else:
                        # Else if opponent streak
                        # Block streak(s) of 2
                        # +20 per streak
                        # continue
                        reward += 20
                else:
                    print("Error: action for full streak")
                    exit()

        return reward

    def streak_contains(self, streak, board_state):

        sums = {}
        sums['X'] = 0
        sums['O'] = 0
        sums[' '] = 0

        for space in streak:
            tile = board_state[space - 1]
            if tile == 'X':
                sums['X'] += 1
            elif tile == 'O':
                sums['O'] += 1
            else:
                sums[' '] += 1

        return sums

    def change_turn(self):
        if self.player == 'X':
            self.player = 'O'
            self.opponent = 'X'
        else:
            self.player = 'X'
            self.opponent = 'O'

    def board_full(self):
        spaces = self.possible_actions()
        return not spaces

    def render(self):
        print('-------------')
        for row in range(0, 9, 3):
            rowString = '| ' + self.board[row] + ' | ' + self.board[row + 1] + ' | ' + self.board[row + 2] + ' |'
            print(rowString)
            print('-------------')

    def possible_actions(self, state=None):
        if state is None:
            state = self.board
        return [i for i, s in enumerate(state) if s == ' ']

    def impossible_actions(self, state=None):
        if state is None:
            state = self.board
        return [i for i, s in enumerate(state) if s != ' ']

    def sample_action(self):
        return random.choice(self.possible_actions())

    def get_state(self):
        return tuple(self.board)
    
    def get_next_player(self, state):
        state_list = list(state)
        counts = Counter(state_list)
        
        if counts['X'] == counts['O']:
            return 'X'
        else:
            return 'O'
