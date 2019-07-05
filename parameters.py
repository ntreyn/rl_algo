import argparse

from rl_utils import AttrDict
from mc_on_policy import MCOnPolicy
from qlearning import QLearning
from dqn import DQN

MODEL_MAP = {
    'qlearn': QLearning,
    'mc_on': MCOnPolicy,
    'dqn': DQN
}

def core_argparser():
    argparser = argparse.ArgumentParser(add_help=False)
    argparser.add_argument(
        '-g', '--gamma', 
        type=float, 
        default=0.9, 
        help='discount (default: 0.9)'
    )
    argparser.add_argument(
        '-a', '--alpha', 
        type=float, 
        default=0.7, 
        help='step size (default: 0.7)'
    )
    argparser.add_argument(
        '-e', '--epsilon', 
        type=float, 
        default=0.1, 
        help='exploration chance (default: 1.0)'
    )
    argparser.add_argument(
        '-r', '--render', 
        action='store_true',
        help='render game'
    )
    argparser.add_argument(
        '--episodes', 
        type=int, 
        default=100000, 
        help='num episodes (default: 100000)'
    )
    argparser.add_argument(
        '--device',
        default='cpu',
        type=str,
        help='cpu or cuda (default: cpu)'
    )
    return argparser


def extra_params(params):
    params.agent = AttrDict()
    params.agent.gamma = params.gamma
    params.agent.alpha = params.alpha
    params.agent.epsilon = params.epsilon
    return params