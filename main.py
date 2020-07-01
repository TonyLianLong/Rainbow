#%%
import random
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from segment_tree import MinSegmentTree, SumSegmentTree
from DQN_agent import DQNAgent

#%%
env_id = "CartPole-v0"
env = gym.make(env_id)

deterministic = False
if deterministic:
    seed = 777

    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)

# parameters
num_frames = 20000
memory_size = 1000
batch_size = 32
target_update = 100

#%%
# train
agent = DQNAgent(env, memory_size, batch_size, target_update)

#%%
agent.train(num_frames, plot=False)

#%%
# agent.env = gym.wrappers.Monitor(env, "videos", force=True)
agent.test(render=True)

