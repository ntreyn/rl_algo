from qlearning import QLearning
from new_q import new_q
from mc_on_policy import MCOnPolicy
from dqn import DQN

from ttt_env import ttt_env
from parameters import core_argparser, extra_params

import numpy as np
import argparse
import os

from collections import defaultdict

class human_agent:
    def __init__(self):
        self.training = False
        self.Q = defaultdict(lambda: np.zeros(9))

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
            if render:
                print("X: agent1")
                print("O: agent2")
            status = play(agent1, agent2, env, render)
            if status == 'X':
                results["a1"] += 1
            elif status == 'O':
                results["a2"] += 1
            else:
                results["draw"] += 1
        else:
            if render:
                print("X: agent2")
                print("O: agent1")
            status = play(agent2, agent1, env, render)
            if status == 'X':
                results["a2"] += 1
            elif status == 'O':
                results["a1"] += 1
            else:
                results["draw"] += 1

    return results

def play(X, O, env, render):
    max_steps = 10

    turn_map = {
        'X': X,
        'O': O
    }

    state = env.reset()
    done = False
    status = 'D'

    if render:
        env.render()

    for step in range(max_steps):
        turn = env.player
        action = turn_map[turn].act(state)
        next_state, reward, done = env.step(action)

        if render:
            print(turn_map[turn].Q[state])
            env.render()

        if done:
            if reward >= 1000:
                status = env.player
            break

        state = next_state
    
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
    print()
    
    return agent

def train_new_q(env, args):

    total_episodes = args.episodes

    agent = new_q(env, args.agent)
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
    print()
    
    return agent

def train_dqn(env, args):
    
    total_episodes = args.episodes
    max_steps = 10

    agent = DQN(env, args)

    if args.load_ckpt:
        agent.load(args.ckpt_path, args.load_ckpt)

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
    print()
    
    if args.save_ckpt:
        if not os.path.exists(args.ckpt_path):
            os.mkdir(args.ckpt_path)
        agent.save(args.ckpt_path, args.save_file)

    return agent

def test_agents(agent1, agent2, env, render, iterations=1):
    results = eval_agent(agent1, agent2, env, render, eval_iter=iterations)
    print("Agent 1 wins: {}, Agent 2 wins: {}, Draws: {}".format(results["a1"], results["a2"], results["draw"]))

def main(args):
    render = args.render
    
    env_im = ttt_env(im_reward=True)
    env_no_im = ttt_env(im_reward=False)

    q_agent = train_q(env_im, args)
    #q_agent = train_q(env_no_im, args)
    #q_agent.env = env_im

    new_q_agent = train_new_q(env_no_im, args)
    new_q_agent.env = env_im

    mc_agent = train_mc_on(env_im, args)

    # dqn_agent_im = train_dqn(env_im, args)
    
    human = human_agent()

    test_agents(mc_agent, new_q_agent, env_im, False, 10000)
    test_agents(mc_agent, q_agent, env_im, False, 10000)
    test_agents(q_agent, new_q_agent, env_im, False, 10000)

    test_agents(mc_agent, new_q_agent, env_im, True, 10)
    test_agents(mc_agent, q_agent, env_im, True, 10)
    test_agents(q_agent, new_q_agent, env_im, True, 10)
    """
    test_agents(mc_agent, new_q_agent, env_im, render, 10)
    test_agents(new_q_agent, human, env_im, render, 10)
    test_agents(mc_agent, human, env_im, render, 3)
    """
    
                
if __name__ == "__main__":
    ARGPARSER = argparse.ArgumentParser(parents=[core_argparser()])
    PARAMS = extra_params(ARGPARSER.parse_args())
    main(PARAMS)