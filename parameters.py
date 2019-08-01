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
        default=0.99, 
        help='discount (default: 0.9)'
    )
    argparser.add_argument(
        '-a', '--alpha', 
        type=float, 
        default=0.9, 
        help='step size (default: 0.1)'
    )
    argparser.add_argument(
        '-e', '--epsilon', 
        type=float, 
        default=0.1, 
        help='exploration chance (default: 1.0)'
    )
    argparser.add_argument(
        '--var_epsilon',
        action='store_true'
    )
    argparser.add_argument(
        '--decay_rate', 
        type=float, 
        default=0.0001, 
        help='epsilon decay rate (default: 0.0001)'
    )
    argparser.add_argument(
        '--min_epsilon', 
        type=float, 
        default=0.1, 
        help='epsilon lower bound (default: 0.1)'
    )
    argparser.add_argument(
        '--max_epsilon', 
        type=float, 
        default=1.0, 
        help='epsilon starting point (default: 1.0)'
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
    argparser.add_argument(
        '--save_ckpt',
        action='store_true'
    )
    argparser.add_argument(
        '--ckpt_path',
        default='checkpoints',
        type=str
    )
    argparser.add_argument(
        '--load_ckpt',
        default='',
        type=str
    )
    argparser.add_argument(
        '--info',
        type=str,
        default=''
    )
    return argparser


def extra_params(params):
    params.agent = AttrDict()
    params.agent.gamma = params.gamma
    params.agent.alpha = params.alpha
    params.agent.epsilon = params.epsilon
    params.agent.var_epsilon = params.var_epsilon
    params.agent.min_epsilon = params.min_epsilon
    params.agent.max_epsilon = params.max_epsilon
    params.agent.decay_rate = params.decay_rate

    if params.save_ckpt:
        params.save_file = 'checkpoint_e_{}_g_{}_episodes_{}'.format(params.epsilon, params.gamma, params.episodes)
        if params.info:
            params.save_file += '_{}'.format(params.info)

    return params