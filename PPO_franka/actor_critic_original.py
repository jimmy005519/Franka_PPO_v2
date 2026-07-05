import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class PPOActor(nn.Module):
    """Actor network for PPO"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(PPOActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.relu = nn.ReLU()
        
    def forward(self, obs):
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        mean = self.mean(x)
        std = self.log_std.exp()
        return mean, std

class PPOCritic(nn.Module):
    """Critic network for PPO"""
    def __init__(self, obs_dim, hidden_dim=256):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, obs):
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        value = self.value(x)
        return value

class PPOAgent:
    """PPO Agent for robotic control"""
    def __init__(self, obs_dim, action_dim, device, lr=3e-4):
        self.device = device
        self.actor = PPOActor(obs_dim, action_dim).to(device)
        self.critic = PPOCritic(obs_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
    def get_action(self, obs, deterministic=False):
        mean, std = self.actor(obs)
        dist = Normal(mean, std)
        if deterministic:
            # The legacy training loop always unpacks (action, log_prob).
            action = mean
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def get_value(self, obs):
        return self.critic(obs)
    
    def update(self, rollout_data, epochs=5, clip_ratio=0.2, value_coeff=0.5, entropy_coeff=0.01):
        """Update actor and critic networks"""
        obs, actions, returns, advantages, old_log_probs = rollout_data
        
        for _ in range(epochs):
            # Actor update
            new_means, new_stds = self.actor(obs)
            new_dist = Normal(new_means, new_stds)
            new_log_probs = new_dist.log_prob(actions).sum(dim=-1)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy = new_dist.entropy().mean()
            
            self.actor_optimizer.zero_grad()
            (actor_loss - entropy_coeff * entropy).backward()
            self.actor_optimizer.step()
            
            # Critic update
            values = self.critic(obs).squeeze(-1)
            critic_loss = ((values - returns) ** 2).mean()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation"""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    advantages = torch.tensor(advantages, device=values.device)
    returns = advantages + values
    return advantages, returns

def get_observation(hand_pos, hand_rot, hand_vel, box_pos, box_rot, box_vel, goal_pos, gripper_pos):
    """Construct observation from state"""
    obs = torch.cat([
        hand_pos,           # 3
        hand_rot,           # 4
        hand_vel,           # 3
        box_pos,            # 3
        box_rot,            # 4
        box_vel,            # 3
        goal_pos,           # 3
        gripper_pos,        # 1
    ], dim=-1)
    return obs
