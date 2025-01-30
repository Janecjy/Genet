class RLTrain():

    def __init__(self, fixed=True):
        self.fixed=fixed

    def select_action(self, state):
        raise NotImplementedError

    def evaluate(self, net_env):
        raise NotImplementedError