import numpy as np

from rl.core import Agent
from keras.layers import Concatenate
from os.path import splitext

class InterleavedAgent(Agent):
    def __init__(self, agents):
        self.agents = agents
        self.cur_agent = -1
        self.n = len(agents)

        self.compiled = False
        self.m_names = []
        self._training = False
        self._step = 0

        super(InterleavedAgent, self).__init__()

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self,t):
        self._training = t
        for agent in self.agents:
            agent.training = t

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self,s):
        #print "setting step %i" % s
        self._step = s
        for agent in self.agents:
            agent.step = s
    
    def reset_states(self):
        self.cur_agent = 1
        for agent in self.agents:
            agent.reset_states()

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.
        # Argument
            observation (object): The current observation from the environment.
        # Returns
            The next action to be executed in the environment.
        """

        # Determine current player
        self.cur_agent = (self.cur_agent+1)%self.n

        return self.agents[self.cur_agent].forward(observation)

    def backward(self, reward, terminal):
        """Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.
        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.
        # Returns
            List of metrics values
        """
        return self.agents[self.cur_agent].backward(reward,terminal)[:len(self.m_names)]

    def compile(self, optimizer, metrics=[]):
        """Compiles an agent and the underlaying models to be used for training and testing.
        # Arguments
            optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
            metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
        """
        for i,agent in enumerate(self.agents):
            if not agent.compiled:
                agent.compile(optimizer[i],metrics)

        # Just truncate the list of metrics if one has more (assume prefixes match)
        if len(self.agents[0].metrics_names)<=len(self.agents[1].metrics_names):
            self.m_names = self.agents[0].metrics_names
        else:
            self.m_names = self.agents[1].metrics_names

        self.compiled = True

    def load_weights(self, filepath):
        """Loads the weights of an agent from an HDF5 file.
        # Arguments
            filepath (str): The path to the HDF5 file.
        """
        fbase, fext = splitext(filepath)
        for i, agent in enumerate(self.agents):
            agent.load_weights('%s%i%s' % (fbase,i,fext))

    def save_weights(self, filepath, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.
        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        fbase, fext = splitext(filepath)
        for i, agent in enumerate(self.agents):
            agent.save_weights('%s%i%s' % (fbase,i,fext), overwrite)

    @property
    def layers(self):
        """Returns all layers of the underlying model(s).
        If the concrete implementation uses multiple internal models,
        this method returns them in a concatenated list.
        # Returns
            A list of the model's layers
        """
        return [ layer for agent in self.agents
                    for layer in agent.layers() ]

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        # Returns
            A list of metric's names (string)
        """
        # Assumes all agents share metric names
        return self.m_names

    def _on_train_begin(self):
        """Callback that is called before training begins."
        """
        for agent in self.agents:
            agent._on_train_begin()

    def _on_train_end(self):
        """Callback that is called after training ends."
        """
        for agent in self.agents:
            agent._on_train_end()

    def _on_test_begin(self):
        """Callback that is called before testing begins."
        """
        for agent in self.agents:
            agent._on_test_begin()

    def _on_test_end(self):
        """Callback that is called after testing ends."
        """
        for agent in self.agents:
            agent._on_test_end()