class RLAgent:

    def __init__(self, env, **kwargs):
        """agent initialization
        Arguments:
            env {gym.env} -- OpenAI Gym environment
        """

        # self.action_space = env.action_space
        for key, value in list(kwargs.items()):
            setattr(self, key, value)
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def act(self, state):
        raise NotImplementedError

    def push(self, *args):
        raise NotImplementedError

    def learn(self, *args):
        raise NotImplementedError
