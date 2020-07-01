#%%
import random
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from segment_tree import MinSegmentTree, SumSegmentTree
from DQN_agent import DQNAgent
from algorithm import Algorithm
from network import Network
from parl.utils import logger

#%%
env_id = "CartPole-v0"
env = gym.make(env_id)

deterministic = True
if deterministic:
    seed = 772

    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)

logger.set_dir('./logs')

# parameters
num_frames = 20000
memory_size = 1000
batch_size = 32
target_update = 100

atom_size = 51
v_min = 0.0
v_max = 200.0

n_step = 3

gamma = 0.99
alpha = 0.2
beta = 0.6

prior_eps = 1e-6

#%%
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
logger.info("Device: {}".format(device))

support = torch.linspace(
    v_min, v_max, atom_size
).to(device)

dqn = Network(
    obs_dim, action_dim, atom_size, support
).to(device)
dqn_target = Network(
    obs_dim, action_dim, atom_size, support
).to(device)

algorithm = Algorithm(model=dqn, model_target=dqn_target, device=device, gamma=gamma, prior_eps=prior_eps, alpha=alpha, beta=beta, v_min=v_min, v_max=v_max, atom_size=atom_size, support=support, batch_size=batch_size)

# train
agent = DQNAgent(algorithm=algorithm, env=env, memory_size=memory_size, batch_size=batch_size, obs_dim=obs_dim, action_dim=action_dim, target_update=target_update, gamma=gamma, alpha=alpha, beta=beta, n_step=n_step, device=device)

#%%
agent.train(num_frames, plot=False)

#%%
# agent.env = gym.wrappers.Monitor(env, "videos", force=True)
agent.test(render=True)

