import torch
import torch.nn as nn
from torch.distributions import Normal


class PPOActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Smaller initial exploration than std=1 avoids violent 5 cm target jumps.
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))

    def forward(self, obs):
        mean = self.net(obs)
        std = self.log_std.clamp(-4.0, 1.0).exp().expand_as(mean)
        return mean, std


class PPOCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.net(obs)


class PPOAgent:
    def __init__(self, obs_dim, action_dim, device, lr=3e-4):
        self.actor = PPOActor(obs_dim, action_dim).to(device)
        self.critic = PPOCritic(obs_dim).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    @torch.no_grad()
    def get_action(self, obs, deterministic=False):
        mean, std = self.actor(obs)
        dist = Normal(mean, std)
        raw_action = mean if deterministic else dist.sample()
        return raw_action, dist.log_prob(raw_action).sum(-1), self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

    def update(self, obs, actions, returns, advantages, old_log_probs,
               epochs=5, clip_ratio=0.2, value_coeff=0.5, entropy_coeff=0.005):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(epochs):
            mean, std = self.actor(obs)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(-1)
            ratio = (log_probs - old_log_probs).exp()
            unclipped = ratio * advantages
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = (self.critic(obs).squeeze(-1) - returns).square().mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = actor_loss + value_coeff * value_loss - entropy_coeff * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), 1.0
            )
            self.optimizer.step()
