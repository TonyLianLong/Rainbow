import parl
from typing import Deque, Dict, List, Tuple
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class Algorithm(parl.Algorithm):
    def __init__(self,
        model,
        model_target,
        v_min,
        v_max,
        atom_size,
        batch_size,
        support,
        prior_eps,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        n_step: int = 3,
        device: str = "cpu"):
        self.model = model
        self.model_target = model_target
        self.gamma = gamma
        self.beta = beta
        self.support = support

        self.prior_eps = prior_eps

        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.batch_size = batch_size

        self.n_step = n_step
        
        # networks: dqn, model_target
        
        self.model_target.load_state_dict(self.model.state_dict())
        self.model_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters())

        self.device = device
    
    def predict(self, state): # Called in test mode
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        selected_action = self.model(
            torch.tensor(state, dtype=torch.float, device=self.device)
        ).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        
        return selected_action

    def sample(self, state):
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        selected_action = self.model(
            torch.tensor(state, dtype=torch.float, device=self.device)
        ).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        
        return state, selected_action

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.model(next_state).argmax(1)
            next_dist = self.model_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.model.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def learn(self, memory, memory_n) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if memory_n:
            gamma = self.gamma ** self.n_step
            samples = memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        self.model.reset_noise()
        self.model_target.reset_noise()

        return loss.item()
    
    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.model_target.load_state_dict(self.model.state_dict())

    sync_target = _target_hard_update