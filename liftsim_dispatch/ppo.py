"""Pairwise PPO model and checkpoint loading."""

import torch
from torch import nn
from torch.distributions import Categorical


class PairwisePPO(nn.Module):
    def __init__(self, obs_dim, action_dim, feat_dim):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(obs_dim,128), nn.LayerNorm(128), nn.Tanh(),
                                     nn.Linear(128,96), nn.Tanh())
        self.pair_actor = nn.Sequential(nn.Linear(96+feat_dim,128), nn.Tanh(),
                                        nn.Linear(128,64), nn.Tanh(), nn.Linear(64,1))
        self.critic = nn.Sequential(nn.Linear(96,128),nn.Tanh(),nn.Linear(128,1))
        nn.init.orthogonal_(self.pair_actor[-1].weight, gain=.02)
        nn.init.zeros_(self.pair_actor[-1].bias)
        self.action_dim=action_dim

    def forward(self, x, features):
        context = self.context(x)
        ctx = context.unsqueeze(1).expand(-1,features.size(1),-1)
        logits = self.pair_actor(torch.cat((ctx,features),dim=-1)).squeeze(-1)
        return logits, self.critic(context).squeeze(-1)

    def distribution(self, states, masks, priors, features):
        logits, vals = self(states, features)
        # Prior supplies useful floor-selection behavior initially; PPO learns corrections.
        logits = (logits + PRIOR_STRENGTH * priors).masked_fill(~masks, -1e9)
        return Categorical(logits=logits), vals

PRIOR_STRENGTH = 0.65


@torch.no_grad()
def agent_choice(agent, obs, mask, prior, features, greedy=False):
    device = next(agent.parameters()).device
    state = torch.as_tensor(obs,dtype=torch.float32,device=device).unsqueeze(0)
    m = torch.as_tensor(mask,dtype=torch.bool,device=device).unsqueeze(0)
    p = torch.as_tensor(prior,dtype=torch.float32,device=device).unsqueeze(0)
    f = torch.as_tensor(features,dtype=torch.float32,device=device).unsqueeze(0)
    dist,value = agent.distribution(state,m,p,f)
    action = dist.probs.argmax(-1) if greedy else dist.sample()
    return int(action.item()),float(dist.log_prob(action).item()),float(value.item())


def ppo_policy(agent):
    agent.eval()

    def choose(decision):
        return agent_choice(
            agent, decision.observation, decision.mask, decision.prior,
            decision.features, greedy=True,
        )[0]

    return choose


def load_policy(checkpoint):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    agent = PairwisePPO(data["obs_dim"], data["action_dim"], data["feat_dim"])
    agent.load_state_dict(data["model_state_dict"])
    return ppo_policy(agent)
