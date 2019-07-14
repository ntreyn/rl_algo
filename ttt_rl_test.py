from qlearning import QLearning
from mc_on_policy import MCOnPolicy
from dqn import DQN

from ttt_env import ttt_env
from parameters import core_argparser, extra_params

import numpy as np
import argparse

class human_agent:
    def __init__(self):
        self.training = False

    def act(self, state):
        move = int(input("Choose move: "))
        return move - 1
    
    def eval(self):
        self.training = False

def eval_agent(agent1, agent2, env, render, eval_iter=10):
    agent1.eval()  # enable eval mode
    agent2.eval()

    results = {
        "a1": 0,
        "a2": 0,
        "draw": 0
    }

    for i_episode in range(eval_iter):

        rand = np.random.randint(0, 2)
        if rand == 0:
            status = play(agent1, agent2, env, render)
            if status == 'X':
                results["a1"] += 1
            elif status == 'O':
                results["a2"] += 1
            else:
                results["draw"] += 1
        else:
            status = play(agent2, agent1, env, render)
            if status == 'X':
                results["a2"] += 1
            elif status == 'O':
                results["a1"] += 1
            else:
                results["draw"] += 1

    return results

def play(X, O, env, render, episode=0):
    max_steps = 10

    X_state = env.reset()
    done = False
    first = True
    status = 'D'

    if render:
        env.render()

    for step in range(max_steps):

        # X turn
        X_action = X.act(X_state)
        X_next_state, X_reward, done = env.step(X_action)

        if render:
            env.render()

        # If X ends game
        if done:
            if X.training:
                X.push(X_state, X_action, X_reward, X_next_state, done)
                X.learn(episode)
            if X_reward >= 1000:
                status = 'X'
            break
                
        # O learns
        if not first and O.training:
            O.push(O_state, O_action, O_reward, X_next_state, done)
            O.learn(episode)
        else:
            first = False

        O_state = X_next_state

        # O turn
        O_action = O.act(O_state)
        O_next_state, O_reward, done = env.step(O_action)

        if render:
            env.render()

        # If O ends game
        if done:
            if O.training:
                O.push(O_state, O_action, O_reward, O_next_state, done)
                O.learn(episode)
            if O_reward >= 1000:
                status = 'O'
            break

        # X learns
        if X.training:
            X.push(X_state, X_action, X_reward, O_next_state, done)
            X.learn(episode)

        X_state = O_next_state
    
    return status

def train_mc_on(env, args):

    total_episodes = args.episodes

    agent = MCOnPolicy(env, args.agent)
    agent.train()

    for episode in range(total_episodes):
        print(episode, agent.epsilon, end='\r')

        max_steps = 10
        state = env.reset()
        done = False
        player = 'X'
        X_mem = []
        O_mem = []

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            if player == 'X':
                X_mem.append((state, action, reward, next_state, done))
            else:
                O_mem.append((state, action, reward, next_state, done))

            # agent.push(state, action, reward, next_state, done)
            agent.learn(episode)

            state = next_state
            if done:
                if reward >= 1000:
                    if player == 'X':
                        s, a, r, n, d = O_mem.pop(-1)
                        r += -2000
                        O_mem.append((s, a, r, n, d))
                    else:
                        s, a, r, n, d = X_mem.pop(-1)
                        r += -2000
                        X_mem.append((s, a, r, n, d))
                else:
                    s, a, r, n, d = O_mem.pop(-1)
                    r += 500
                    O_mem.append((s, a, r, n, d))
                    s, a, r, n, d = X_mem.pop(-1)
                    r += 500
                    X_mem.append((s, a, r, n, d))
                
                for mem in X_mem:
                    agent.push(*mem)
                agent.learn(episode)
                for mem in O_mem:
                    agent.push(*mem)
                agent.learn(episode)
                break
            
            if player == 'X':
                player = 'O'
            else:
                player = 'X'
    print()
    return agent

def train_q(env, args):

    total_episodes = args.episodes

    agent = QLearning(env, args.agent)
    agent.train()

    for episode in range(total_episodes):
        print(episode, agent.epsilon, end='\r')

        max_steps = 10
        state = env.reset()
        done = False

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.push(state, action, reward, next_state, done)
            agent.learn(episode)

            state = next_state
            if done:
                break
    
    return agent

def train_dqn(env, args):
    
    total_episodes = args.episodes
    max_steps = 10

    agent = DQN(env, args)
    agent.train()

    for episode in range(total_episodes):
        print(episode, agent.epsilon, end='\r')

        state = env.reset()
        done = False

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.push(state, action, reward, next_state, done)
            agent.learn(episode)

            state = next_state

            if done:
                break

    return agent

def main(args):
    render = args.render
    
    env = ttt_env()
    dqn_agent = train_dqn(env, args)
    # mc_agent = train_mc_on(env, args)
    q_agent = train_q(env, args)
    

    human = human_agent()

    results = eval_agent(dqn_agent, q_agent, env, False, eval_iter=10000)
    print("Agent 1 wins: {}, Agent 2 wins: {}, Draws: {}".format(results["a1"], results["a2"], results["draw"]))
    
    #results = eval_agent(mc_agent, q_agent, env, render, eval_iter=5)
    # results = eval_agent(mc_agent, human, env, render, eval_iter=5)
    results = eval_agent(dqn_agent, human, env, render, eval_iter=5)
                
if __name__ == "__main__":
    ARGPARSER = argparse.ArgumentParser(parents=[core_argparser()])
    PARAMS = extra_params(ARGPARSER.parse_args())
    main(PARAMS)