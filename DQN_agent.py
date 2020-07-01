import gym
import numpy as np
from typing import Deque, Dict, List, Tuple
import torch
from IPython.display import clear_output
from prioritized_replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from network import Network
import torch.optim as optim
import matplotlib.pyplot as plt
import parl
from parl.utils import logger, tensorboard

class DQNAgent(parl.Agent):
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
        self, 
        algorithm, 
        device, 
        obs_dim,
        action_dim,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        n_step: int = 3,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
    ):  
        self.env = env
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        # NoisyNet: All attributes related to epsilon are removed
        
        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha
        )
        
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
            )

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def sample(self, state: np.ndarray):
        state, selected_action = self.algorithm.sample(state)
        self.transition = [state, selected_action]
        return selected_action

    def predict(self, state: np.ndarray):
        return self.algorithm.predict(state)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done

    def learn(self) -> torch.Tensor:
        if self.use_n_step:
            return self.algorithm.learn(self.memory, self.memory_n)
        else:
            return self.algorithm.learn(self.memory, memory_n = None)
        
    def train(self, num_frames: int, plotting_interval: int = 200, plot: bool = False):
        """Train the agent."""
        self.is_test = False
        
        state = self.env.reset()
        update_cnt = 0
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.sample(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
            
            # NoisyNet: removed decrease of epsilon
            
            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                tensorboard.add_scalar('score', score)
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.learn()
                losses.append(loss)
                tensorboard.add_scalar('loss', loss)
                update_cnt += 1
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self.algorithm._target_hard_update()

            # plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, plot=plot)

                
                
        self.env.close()
                
    def test(self, render: bool) -> None:
        """Test the agent."""
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        while not done:
            # self.env.render()
            action = self.sample(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        logger.info("score: {}".format(score))
        self.env.close()
                
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float],
        plot: bool
    ):
        """Plot the training progresses."""
        logger.info("Frame: {}, Score: {:.1f}, loss: {:.2f}".format(frame_idx, np.mean(scores[-10:]), np.mean(losses)))
        
        if plot:
            clear_output(True)
            plt.figure(figsize=(20, 5))
            plt.subplot(131)
            plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
            plt.plot(scores)
            plt.subplot(132)
            plt.title('loss')
            plt.plot(losses)
            plt.show()
            plt.close()
