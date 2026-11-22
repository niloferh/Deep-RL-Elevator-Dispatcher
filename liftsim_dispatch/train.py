"""Train the notebook's imitation-initialized, masked PPO/GAE policy."""

import argparse
import copy
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import trange

from .env import CallAwareLiftSim, DEFAULT_CONFIG, MAX_SIM_TICKS
from .evaluation import evaluate_episode, score_of
from .ppo import PairwisePPO, agent_choice, ppo_policy

VALID_SEEDS = range(710, 716)
HORIZON = 768
MINIBATCH = 192
PPO_EPOCHS = 4
GAMMA,LAMBDA = .993,.95
CLIP_EPS=.18
VALUE_COEF=.4
ENTROPY_COEF=.004
TARGET_KL=.035


def imitate(agent, optimizer, device, config_path, demo_episodes, imitation_epochs):
    examples=[]
    for ep in trange(demo_episodes,desc='Gather dispatch demonstrations'):
        env=CallAwareLiftSim(seed=1100+ep, config_path=config_path)
        try:
            obs=env.observation()
            while env.tick < MAX_SIM_TICKS:
                mask,prior,features=env.action_info()
                a=int(np.argmax(np.where(mask, prior, -np.inf)))
                if mask.sum()>1:
                    examples.append((obs.copy(),mask.copy(),prior.copy(),features.copy(),a))
                obs,_,done,_=env.step(a)
                if done:break
        finally:
            env.close()
    if not examples:
        raise ValueError("No dispatch choices collected for imitation. Check the traffic configuration.")
    print('Expert decisions:',len(examples))

    xs=torch.as_tensor(np.stack([r[0] for r in examples]),dtype=torch.float32,device=device)
    ms=torch.as_tensor(np.stack([r[1] for r in examples]),dtype=torch.bool,device=device)
    ps=torch.as_tensor(np.stack([r[2] for r in examples]),dtype=torch.float32,device=device)
    fs=torch.as_tensor(np.stack([r[3] for r in examples]),dtype=torch.float32,device=device)
    ys=torch.as_tensor([r[4] for r in examples],dtype=torch.long,device=device)
    for epoch in range(imitation_epochs):
        losses=[]
        for idx in torch.randperm(len(xs),device=device).split(96):
            dist,_=agent.distribution(xs[idx],ms[idx],ps[idx],fs[idx])
            imitation_loss=-dist.log_prob(ys[idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            imitation_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(),.5)
            optimizer.step()
            losses.append(float(imitation_loss.item()))
        if (epoch+1)%4==0:print('Imitation epoch',epoch+1,'loss',round(float(np.mean(losses)),4))
    agent.eval()


def validation_score(agent, config_path):
    policy = ppo_policy(agent)
    scores = [
        score_of(evaluate_episode(policy, seed, config_path=config_path))
        for seed in VALID_SEEDS
    ]
    return float(np.mean(scores))


def fit_ppo(agent, optimizer, train_env, device, config_path, total_decisions):
    best_val = validation_score(agent, config_path)
    best_weights = copy.deepcopy(agent.state_dict())
    updates = math.ceil(total_decisions / HORIZON)
    obs=train_env.observation()
    training_returns=[]
    val_history=[(0,best_val)]
    episode_reward=0.0
    for update in trange(updates,desc='Train PPO'):
        size=min(HORIZON,total_decisions-update*HORIZON)
        buffer=[]
        for _ in range(size):
            mask,prior,features=train_env.action_info()
            a,logp,value=agent_choice(agent, obs,mask,prior,features)
            next_obs,reward,done,_=train_env.step(a)
            episode_reward+=reward
            buffer.append((obs.copy(),mask.copy(),prior.copy(),features.copy(),
                           a,logp,value,reward,done))
            obs=next_obs
            if done:
                training_returns.append(episode_reward)
                episode_reward=0.
                obs=train_env.reset()

        # Correct GAE at rollout boundary: bootstrap if last step was not terminal.
        with torch.no_grad():
            _,end_val=agent(torch.as_tensor(obs,dtype=torch.float32,device=device)[None],
                            torch.as_tensor(train_env.action_info()[2],dtype=torch.float32,device=device)[None])
            end_val=float(end_val.item())
        adv=np.zeros(size,dtype=np.float32)
        next_value=end_val
        gae=0.
        for t in reversed(range(size)):
            state,mask,prior,features,a,lp,v,r,done=buffer[t]
            nonterm=1.0-float(done)
            delta=r+GAMMA*next_value*nonterm-v
            gae=delta+GAMMA*LAMBDA*nonterm*gae
            adv[t]=gae
            next_value=v

        states=torch.as_tensor(np.stack([r[0] for r in buffer]),dtype=torch.float32,device=device)
        masks=torch.as_tensor(np.stack([r[1] for r in buffer]),dtype=torch.bool,device=device)
        priors=torch.as_tensor(np.stack([r[2] for r in buffer]),dtype=torch.float32,device=device)
        feat=torch.as_tensor(np.stack([r[3] for r in buffer]),dtype=torch.float32,device=device)
        actions=torch.as_tensor([r[4] for r in buffer],dtype=torch.long,device=device)
        oldlp=torch.as_tensor([r[5] for r in buffer],dtype=torch.float32,device=device)
        oldvalue=torch.as_tensor([r[6] for r in buffer],dtype=torch.float32,device=device)
        advantages=torch.as_tensor(adv,dtype=torch.float32,device=device)
        targets=oldvalue+advantages
        normalized=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
        agent.train()
        stop=False
        for epoch in range(PPO_EPOCHS):
            for ix in torch.randperm(size,device=device).split(MINIBATCH):
                dist,v=agent.distribution(states[ix],masks[ix],priors[ix],feat[ix])
                new_lp=dist.log_prob(actions[ix])
                ratio=torch.exp((new_lp-oldlp[ix]).clamp(-20,20))
                pi=-torch.minimum(ratio*normalized[ix],
                     ratio.clamp(1-CLIP_EPS,1+CLIP_EPS)*normalized[ix]).mean()
                vf=.5*(v-targets[ix]).square().mean()
                loss=pi+VALUE_COEF*vf-ENTROPY_COEF*dist.entropy().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(),.5)
                optimizer.step()
                kl=(oldlp[ix]-new_lp.detach()).mean().item()
                if kl > TARGET_KL:
                    stop=True
                    break
            if stop:break
        agent.eval()
        if (update+1)%8==0 or (update+1)==updates:
            val=validation_score(agent, config_path)
            val_history.append((min(total_decisions,(update+1)*HORIZON),val))
            if val>best_val:
                best_val=val
                best_weights=copy.deepcopy(agent.state_dict())
            print(f'Update {update+1}/{updates}: val={val:.2f} best={best_val:.2f} training_eps={len(training_returns)}')
    return best_weights, best_val, training_returns, val_history


def train(output_dir, *, config_path=DEFAULT_CONFIG, total_decisions=60_000,
          demo_episodes=24, imitation_epochs=16, device=None):
    if total_decisions < 1 or demo_episodes < 0 or imitation_epochs < 0:
        raise ValueError("Decisions must be positive and imitation counts nonnegative")
    torch.manual_seed(42)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    env = CallAwareLiftSim(seed=42, config_path=config_path)
    try:
        agent = PairwisePPO(env.obs_dim, env.A, env.FEAT_DIM).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)
        if demo_episodes and imitation_epochs:
            imitate(agent, optimizer, device, config_path, demo_episodes, imitation_epochs)
        weights, score, returns, validation = fit_ppo(
            agent, optimizer, env, device, config_path, total_decisions
        )
    finally:
        env.close()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "LiftSim_PPO_Improved_best.pt"
    torch.save({
        "model_state_dict": weights, "obs_dim": env.obs_dim,
        "action_dim": env.A, "feat_dim": env.FEAT_DIM,
        "best_validation_score": score, "seed": 42,
        "config_file": str(config_path), "algorithm": "masked_pairwise_PPO_GAE",
    }, checkpoint)
    save_curves(output_dir, returns, validation, score)
    return checkpoint


def save_curves(output_dir, training_returns, val_history, best_val):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 4))
    axs = fig.subplots(1, 2)
    axs[0].plot(training_returns,alpha=.35)
    if len(training_returns)>=10:
        axs[0].plot(np.convolve(training_returns,np.ones(10)/10,mode='valid'))
    axs[0].set(title='Shaped training return',xlabel='Episode',ylabel='Reward')
    axs[1].plot(*zip(*val_history),marker='o')
    axs[1].axhline(best_val,color='gray',ls='--')
    axs[1].set(title='Held-out validation score',xlabel='Dispatch decisions',ylabel='Score')
    fig.tight_layout()
    fig.savefig(output_dir / "LiftSim_PPO_learning_curves.png", dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--decisions", type=int, default=60_000)
    parser.add_argument("--demo-episodes", type=int, default=24)
    parser.add_argument("--imitation-epochs", type=int, default=16)
    parser.add_argument("--device")
    args = parser.parse_args()
    torch.set_num_threads(min(os.cpu_count() or 1, 4))
    print(train(args.output_dir, config_path=args.config, total_decisions=args.decisions,
                demo_episodes=args.demo_episodes, imitation_epochs=args.imitation_epochs,
                device=args.device))


if __name__ == "__main__":
    main()
