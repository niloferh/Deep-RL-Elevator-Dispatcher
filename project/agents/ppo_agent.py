"""Masked pairwise PPO from lizsun, adapted to the shared LiftSim interface."""

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

PRIOR_STRENGTH = 0.65
MAX_CANDIDATES = 16
TICKS_PER_ACTION = 3
HORIZON = 768
MINIBATCH = 192
PPO_EPOCHS = 4
GAMMA, LAMBDA = .993, .95
CLIP_EPS = .18
VALUE_COEF = .4
ENTROPY_COEF = .004
TARGET_KL = .035


class PairwisePPO(nn.Module):
    def __init__(self, obs_dim, action_dim, feat_dim):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(obs_dim, 128), nn.LayerNorm(128), nn.Tanh(),
                                     nn.Linear(128, 96), nn.Tanh())
        self.pair_actor = nn.Sequential(nn.Linear(96 + feat_dim, 128), nn.Tanh(),
                                        nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.critic = nn.Sequential(nn.Linear(96, 128), nn.Tanh(), nn.Linear(128, 1))
        nn.init.orthogonal_(self.pair_actor[-1].weight, gain=.02)
        nn.init.zeros_(self.pair_actor[-1].bias)
        self.action_dim = action_dim

    def forward(self, x, features):
        context = self.context(x)
        ctx = context.unsqueeze(1).expand(-1, features.size(1), -1)
        logits = self.pair_actor(torch.cat((ctx, features), dim=-1)).squeeze(-1)
        return logits, self.critic(context).squeeze(-1)

    def distribution(self, states, masks, priors, features):
        logits, vals = self(states, features)
        logits = (logits + PRIOR_STRENGTH * priors).masked_fill(~masks, -1e9)
        return Categorical(logits=logits), vals


class PPOAgent:
    """Use an existing LiftSim; act(observation) returns its native action list.

    fit() trains on the supplied environment before evaluation. No environment
    construction, evaluation metrics, CSV output or rendering lives here.
    """

    FEAT_DIM = 14

    def __init__(self, env, seed=42, checkpoint=None, device=None):
        self.env = env
        self.F = int(env.attribute.NumberOfFloor)
        self.E = int(env.attribute.ElevatorNumber)
        self.A = 1 + self.E * 2 * self.F
        self.obs_dim = self.E * (9 + self.F) + 8 * self.F + 4
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        # Initialize reproducibly without changing the caller's CPU RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.model = PairwisePPO(self.obs_dim, self.A, self.FEAT_DIM)
        self.model.to(self.device).eval()
        self.seed = seed
        self._trained = False
        self.reset()
        if checkpoint is not None:
            self.load(checkpoint)

    def reset(self):
        """Reset decision cadence at the start of a shared evaluation episode."""
        self._remaining = 0

    def queues(self):
        return self.env._mansion.waiting_queue

    def is_free(self, i, observation):
        state = observation.ElevatorStates[i]
        phys = self.env._mansion._elevators[i]
        return (state.CurrentDispatchTarget <= 0 and
                len(state.ReservedTargetFloors) == 0 and
                abs(state.Velocity) < 0.04 and state.DoorState < 0.03 and
                phys.loaded_people_num == 0 and
                not phys._entering_person and not phys._exiting_person)

    def encode(self, car, floor, direction):
        j = floor - 1 if direction == 1 else self.F + floor - 1
        return 1 + car * (2 * self.F) + j

    def decode(self, a):
        car, loc = divmod(int(a) - 1, 2 * self.F)
        direction = 1 if loc < self.F else -1
        floor = loc + 1 if loc < self.F else loc - self.F + 1
        return car, floor, direction

    def observation(self, observation):
        obs = []
        for i, car in enumerate(observation.ElevatorStates):
            booked = set(car.ReservedTargetFloors)
            obs.extend([
                car.Floor / self.F, car.Velocity / max(car.MaximumSpeed, 0.1),
                float(car.Direction), float(car.DoorState),
                car.CurrentDispatchTarget / self.F, float(car.DispatchTargetDirection),
                car.LoadWeight / max(car.MaximumLoad, 1.0),
                len(booked) / self.F, float(self.is_free(i, observation)),
            ])
            obs.extend(float(f in booked) for f in range(1, self.F + 1))
        up, down = self.queues()
        now = self.env._mansion._config.raw_time
        for q in list(up) + list(down):
            age = now - q[-1].AppearTime if q else 0.
            obs.extend([min(len(q), 15) / 5., np.clip(age / 90., 0., 4.)])
        for q in list(up) + list(down):
            if q:
                ages = [max(0., now - p.AppearTime) for p in q]
                obs.extend([np.clip(np.mean(ages) / 120., 0., 4.),
                            np.clip(np.max(ages) / 300., 0., 3.)])
            else:
                obs.extend([0., 0.])
        free_count = sum(self.is_free(i, observation) for i in range(self.E))
        obs.extend([free_count / self.E, sum(map(len, up)) / 30.,
                    sum(map(len, down)) / 30., min(now / 300., 2.)])
        return np.asarray(obs, dtype=np.float32)

    def action_info(self, observation):
        mask = np.zeros(self.A, dtype=bool)
        prior = np.full(self.A, -12., dtype=np.float32)
        mask[0] = True
        prior[0] = 0.
        up, down = self.queues()
        states = observation.ElevatorStates
        occupied = {(int(car.CurrentDispatchTarget), int(car.DispatchTargetDirection))
                    for car in states if car.CurrentDispatchTarget > 0}
        now = self.env._mansion._config.raw_time
        idle = [i for i in range(self.E) if self.is_free(i, observation)]
        pairs = []
        for i in idle:
            car = states[i]
            for direction, qs in ((1, up), (-1, down)):
                for floor, q in enumerate(qs, start=1):
                    if not q or (floor, direction) in occupied:
                        continue
                    distance = abs(float(car.Floor) - floor)
                    oldest = max(0., now - q[-1].AppearTime)
                    preference = -distance + .022 * min(oldest, 300.) + 1.25 * math.log1p(len(q))
                    pairs.append((preference, self.encode(i, floor, direction)))
        if pairs:
            pairs.sort(reverse=True)
            mask[0] = False
            for score, index in pairs[:MAX_CANDIDATES]:
                mask[index], prior[index] = True, score

        feats = np.zeros((self.A, self.FEAT_DIM), dtype=np.float32)
        active = sum(bool(q) for q in list(up) + list(down))
        waiting = sum(map(len, up)) + sum(map(len, down))
        for a in np.flatnonzero(mask):
            if a == 0:
                continue
            i, floor, direction = self.decode(a)
            car = states[i]
            q = (up if direction == 1 else down)[floor - 1]
            ages = [max(0., now - p.AppearTime) for p in q]
            nearest_other = min((abs(states[j].Floor - floor) for j in idle if j != i), default=self.F)
            mean_ride = (self.F - floor if direction == 1 else floor - 1) * .5
            feats[a] = [float(car.Floor) / self.F, floor / self.F,
                        abs(float(car.Floor) - floor) / self.F, direction,
                        np.log1p(len(q)) / 3., min(max(ages) / 300., 2.),
                        min(float(np.mean(ages)) / 300., 2.), mean_ride / self.F,
                        nearest_other / self.F, len(idle) / self.E, active / (2 * self.F),
                        min(waiting / 30., 3.), min(now / 300., 2.),
                        np.sign(floor - float(car.Floor))]
        return mask, prior, feats

    def _inputs(self, observation):
        return (self.observation(observation), *self.action_info(observation))

    def _tensor(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    @torch.no_grad()
    def _choice(self, inputs, greedy=False):
        obs, mask, prior, feats = inputs
        dist, value = self.model.distribution(self._tensor(obs)[None],
            self._tensor(mask, torch.bool)[None], self._tensor(prior)[None], self._tensor(feats)[None])
        a = dist.probs.argmax(-1) if greedy else dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    def _command(self, action):
        # -1 leaves a car's existing dispatch target unchanged in LiftSim.
        command = [-1, 0] * self.E
        if action:
            i, floor, direction = self.decode(action)
            command[2 * i:2 * i + 2] = [int(floor), int(direction)]
        return command

    def act(self, observation):
        if not self._trained:
            raise RuntimeError('Call fit() or load() before evaluating PPO.')
        if self._remaining:
            self._remaining -= 1
            return self._command(0)
        action, _, _ = self._choice(self._inputs(observation), greedy=True)
        self._remaining = TICKS_PER_ACTION - 1
        return self._command(action)

    def _advance(self, action, steps):
        """Training only: use the same native steps and cadence as act()."""
        reward = 0.
        for i in range(min(TICKS_PER_ACTION, steps)):
            observation, r, done, _ = self.env.step(self._command(action if i == 0 else 0))
            reward += float(r)
            if done:
                break
        return observation, reward, done, i + 1

    def _update(self, buffer, end_value, optimizer):
        adv = np.zeros(len(buffer), dtype=np.float32)
        gae = 0.
        next_value = end_value
        for t in reversed(range(len(buffer))):
            _, _, _, _, _, _, v, r, done = buffer[t]
            nonterm = 1. - float(done)
            delta = r + GAMMA * next_value * nonterm - v
            gae = delta + GAMMA * LAMBDA * nonterm * gae
            adv[t], next_value = gae, v
        states = self._tensor(np.stack([r[0] for r in buffer]))
        masks = self._tensor(np.stack([r[1] for r in buffer]), torch.bool)
        priors = self._tensor(np.stack([r[2] for r in buffer]))
        feats = self._tensor(np.stack([r[3] for r in buffer]))
        actions = self._tensor([r[4] for r in buffer], torch.long)
        oldlp = self._tensor([r[5] for r in buffer])
        oldvalue = self._tensor([r[6] for r in buffer])
        advantages = self._tensor(adv)
        targets = oldvalue + advantages
        normalized = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        self.model.train()
        for _ in range(PPO_EPOCHS):
            stop = False
            for ix in torch.randperm(len(buffer), device=self.device).split(MINIBATCH):
                dist, v = self.model.distribution(states[ix], masks[ix], priors[ix], feats[ix])
                newlp = dist.log_prob(actions[ix])
                ratio = torch.exp((newlp - oldlp[ix]).clamp(-20, 20))
                pi = -torch.minimum(ratio * normalized[ix],
                                    ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * normalized[ix]).mean()
                vf = .5 * (v - targets[ix]).square().mean()
                loss = pi + VALUE_COEF * vf - ENTROPY_COEF * dist.entropy().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), .5)
                optimizer.step()
                if (oldlp[ix] - newlp.detach()).mean().item() > TARGET_KL:
                    stop = True
                    break
            if stop:
                break
        self.model.eval()

    def _imitate(self, optimizer, demo_episodes, imitation_epochs, episode_steps):
        examples = []
        for ep in range(demo_episodes):
            self.env.seed(1100 + ep)
            observation = self.env.reset()
            step = 0
            while step < episode_steps:
                inputs = self._inputs(observation)
                _, mask, prior, _ = inputs
                action = int(np.argmax(np.where(mask, prior, -np.inf)))
                if mask.sum() > 1:
                    examples.append((*inputs, action))
                observation, _, done, n = self._advance(action, episode_steps - step)
                step += n
                if done:
                    break
        if not examples:
            return
        self.model.train()
        for _ in range(imitation_epochs):
            for ix in torch.randperm(len(examples)).split(96):
                batch = [examples[i] for i in ix.tolist()]
                obs, mask, prior, feats, action = zip(*batch)
                dist, _ = self.model.distribution(self._tensor(np.stack(obs)),
                    self._tensor(np.stack(mask), torch.bool), self._tensor(np.stack(prior)),
                    self._tensor(np.stack(feats)))
                loss = -dist.log_prob(self._tensor(action, torch.long)).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), .5)
                optimizer.step()
        self.model.eval()

    def fit(self, total_decisions=60_000, episode_steps=3000, demo_episodes=24,
            imitation_epochs=16):
        """Train using shared LiftSim's native reward; no separate metric system."""
        if min(total_decisions, episode_steps) < 1 or min(demo_episodes, imitation_epochs) < 0:
            raise ValueError('Invalid PPO training lengths.')
        import random
        py_state, np_state = random.getstate(), np.random.get_state()
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == 'cuda' else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(self.seed)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(self.seed)
                optimizer = torch.optim.Adam(self.model.parameters(), lr=2.5e-4, eps=1e-5)
                if demo_episodes and imitation_epochs:
                    self._imitate(optimizer, demo_episodes, imitation_epochs, episode_steps)
                self.env.seed(self.seed)
                observation = self.env.reset()
                step = 0
                for start in range(0, total_decisions, HORIZON):
                    buffer = []
                    for _ in range(min(HORIZON, total_decisions - start)):
                        inputs = self._inputs(observation)
                        action, logp, value = self._choice(inputs)
                        observation, reward, done, n = self._advance(action, episode_steps - step)
                        step += n
                        done = bool(done or step >= episode_steps)
                        buffer.append((*inputs, action, logp, value, reward, done))
                        if done:
                            observation = self.env.reset()
                            step = 0
                    end_value = 0. if buffer[-1][-1] else self._choice(self._inputs(observation), greedy=True)[2]
                    self._update(buffer, end_value, optimizer)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            self.model.eval()
            self.reset()
        self._trained = True
        return self

    def save(self, checkpoint):
        if not self._trained:
            raise RuntimeError('Cannot save an untrained PPO agent.')
        path = Path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': self.model.state_dict(), 'obs_dim': self.obs_dim,
                    'action_dim': self.A, 'feat_dim': self.FEAT_DIM}, path)

    def load(self, checkpoint):
        data = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if (data['obs_dim'], data['action_dim'], data['feat_dim']) != (self.obs_dim, self.A, self.FEAT_DIM):
            raise ValueError('PPO checkpoint does not match this building configuration.')
        self.model.load_state_dict(data['model_state_dict'])
        self.model.eval()
        self._trained = True
        self.reset()
        return self
