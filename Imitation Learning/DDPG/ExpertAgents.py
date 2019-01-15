from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import logging
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import gym
gym.undo_logger_setup()  # NOQA
from gym import spaces
import gym.wrappers

from osim.env import ProstheticsEnv


import chainer
from chainer import optimizers
from chainerrl.agents.ddpg import DDPG
from chainerrl.agents.ddpg import DDPGModel
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import misc
from chainerrl import policy
from chainerrl import q_functions
from chainerrl import replay_buffer

style.use('ggplot')



# Chainer's settings
seed=0
gpu=0


# Network Setting

actor_hidden_layers=3
actor_hidden_units=300
actor_lr=1e-4


critic_hidden_layers=3
critic_hidden_units=300
critic_lr=1e-3


# other settings

number_of_episodes=2000
max_episode_length=1000

replay_buffer_size=5 * 10 ** 5
replay_start_size=5000
number_of_update_times=1

target_update_interval=1
target_update_method='soft'

soft_update_tau=1e-2
update_interval=4
number_of_eval_runs=100
eval_interval=10 ** 5

final_exploration_steps=10 ** 6

gamma=0.995
minibatch_size=128

#_______________________________________________ Helper's functions ______________________________________________________

def clip_action_filter(a):
    return np.clip(a, action_space.low, action_space.high)

def reward_filter(r):
    return r * 1


def phi(obs):
    obs=np.array(obs)
    return obs.astype(np.float32)

def random_action():
    a = action_space.sample()
    if isinstance(a, np.ndarray):
        a = a.astype(np.float32)
    return a


def make_env(test,render=False):
    env = ProstheticsEnv(visualize=render)
    # Use different random seeds for train and test envs
    env_seed = 2 ** 32 - 1 - seed if test else seed
    env.seed(env_seed)
    #if args.monitor:
        #env = gym.wrappers.Monitor(env, args.outdir)
    if isinstance(env.action_space, spaces.Box):
        misc.env_modifiers.make_action_filtered(env, clip_action_filter)
    if not test:
        misc.env_modifiers.make_reward_filtered(env, reward_filter)
    if render and not test:
        misc.env_modifiers.make_rendered(env)
    return env


# Set a random seed used in ChainerRL
misc.set_random_seed(seed)

# Environment Initialization

env = make_env(test=False,render=False)
#timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
obs_size = np.asarray(env.observation_space.shape).prod()
action_space = env.action_space

action_size = np.asarray(action_space.shape).prod()



# Critic Network

q_func = q_functions.FCSAQFunction(
            obs_size, 
            action_size,
            n_hidden_channels=critic_hidden_units,
            n_hidden_layers=critic_hidden_layers)

pi = policy.FCDeterministicPolicy(
            obs_size, 
            action_size=action_size,
            n_hidden_channels=actor_hidden_units,
            n_hidden_layers=actor_hidden_layers,
            min_action=action_space.low, 
            max_action=action_space.high,
            bound_action=True)


# The Model

model = DDPGModel(q_func=q_func, policy=pi)
opt_actor = optimizers.Adam(alpha=actor_lr)
opt_critic = optimizers.Adam(alpha=critic_lr)
opt_actor.setup(model['policy'])
opt_critic.setup(model['q_function'])
opt_actor.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_a')
opt_critic.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_c')

rbuf = replay_buffer.ReplayBuffer(replay_buffer_size)
ou_sigma = (action_space.high - action_space.low) * 0.2

explorer = explorers.AdditiveOU(sigma=ou_sigma)


# The agent
agent = DDPG(model, opt_actor, opt_critic, rbuf, gamma=gamma,
                 explorer=explorer, replay_start_size=replay_start_size,
                 target_update_method=target_update_method,
                 target_update_interval=target_update_interval,
                 update_interval=update_interval,
                 soft_update_tau=soft_update_tau,
                 n_times_update=number_of_update_times,
                 phi=phi,minibatch_size=minibatch_size
            )




class ExpertDDPGAgent:
    
    def __init__(self,path):
        self.path=path
        
    def load_agent(self):
        agent.load(self.path)
        return agent