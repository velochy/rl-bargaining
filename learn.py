# Modified from keras-rl ddpg-pendulum.py example
import numpy as np
import gym

from keras.models import Sequential, Model, clone_model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Add
from keras.optimizers import Adam
from keras.regularizers import l1, l2

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess

from bargaining import Bargaining
from interleaved import InterleavedAgent

#ENV_NAME = 'Pendulum-v0'
#env = gym.make(ENV_NAME)

ENV_NAME = 'Bargaining'
env = Bargaining()

#np.random.seed(123)
#env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

MEMORY_SIZE = 100000
MEMORY_WINDOW = 4
LAYER_SIZE = 64
N_WARMUP = 5000

# Next, we build a very simple model.

observation_input = Input(shape=(MEMORY_WINDOW,) + env.observation_space.shape, name='observation_input')
x = Flatten()(observation_input)
x = Dense(LAYER_SIZE,kernel_regularizer=l2(0.01))(x)
x = Activation('relu')(x)
y = Dense(LAYER_SIZE,kernel_regularizer=l2(0.01))(x)
y = Activation('relu')(y)
x = Add()([x,y]) # ResNet
y = Dense(LAYER_SIZE,kernel_regularizer=l2(0.01))(x)
y = Activation('relu')(y)
x = Add()([x,y]) # ResNet
x = Dense(1)(x)
x = Activation('sigmoid')(x)
actor = Model(inputs=[observation_input], outputs=x)
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(MEMORY_WINDOW,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(LAYER_SIZE,kernel_regularizer=l2(0.01))(x)
x = Activation('relu')(x)
y = Dense(LAYER_SIZE,kernel_regularizer=l2(0.01))(x)
y = Activation('relu')(y)
x = Add()([x,y]) # ResNet
y = Dense(LAYER_SIZE,kernel_regularizer=l2(0.01))(x)
y = Activation('relu')(y)
x = Add()([x,y]) # ResNet
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

memory = SequentialMemory(limit=MEMORY_SIZE, window_length=MEMORY_WINDOW)
random_process = GaussianWhiteNoiseProcess(mu=0.0,sigma=0.2,sigma_min=0.03,n_steps_annealing=100000)#OrnsteinUhlenbeckProcess(size=nb_actions, theta=.05, mu=0., sigma=.05)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=N_WARMUP, nb_steps_warmup_actor=N_WARMUP,
                  random_process=random_process, gamma=.999, target_model_update=1e-3,
                  batch_size=64)
#agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Use the same network for both sides (self-play)
agent = InterleavedAgent([agent,agent])

# Make a copy of these networks for player 2 (buyer)
'''
actor_b = clone_model(actor)
critic_b = clone_model(critic)
action_input_b = critic_b.inputs[0]

memory_b = SequentialMemory(limit=MEMORY_SIZE, window_length=MEMORY_WINDOW)
random_process_b = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent_b = DDPGAgent(nb_actions=nb_actions, actor=actor_b, critic=critic_b, critic_action_input=action_input_b,
                  memory=memory_b, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                  random_process=random_process_b, gamma=.99, target_model_update=1e-3,
                  batch_size=128)
agent = InterleavedAgent([agent,agent_b])
'''

agent.compile([Adam(lr=1e-3),Adam(lr=1e-3)], metrics=['mae'])

#agent.load_weights('ddpg_{}_weights.h5f'.format(ENV_NAME))

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=800000, visualize=False, verbose=1, nb_max_episode_steps=None)

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)