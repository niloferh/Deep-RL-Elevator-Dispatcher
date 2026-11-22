"""Run a dispatch policy on one seeded LiftSim episode."""

from .env import CallAwareLiftSim, DEFAULT_CONFIG
from .policies import Decision

import numpy as np


def snapshot(env, cumulative_reward):
    state=env.base.state
    up,down=env.queues()
    return {'step':int(env.tick),'reward':float(cumulative_reward),
            'cars':[{'floor':float(car.Floor),'door':float(car.DoorState),
                     'direction':int(car.Direction),'load':int(load),
                     'target':int(car.CurrentDispatchTarget)}
                    for car,load in zip(state.ElevatorStates,env.base._mansion.loaded_people)],
            'up':[len(q) for q in up],'down':[len(q) for q in down],
            'stats':dict(env.base.statistics)}


def evaluate_episode(policy, seed, *, config_path=DEFAULT_CONFIG, record=False):
    """Evaluate a callable accepting a Decision and returning a legal action ID."""
    env = CallAwareLiftSim(seed=int(seed), config_path=config_path)
    rng = np.random.default_rng(int(seed) + 113)
    total_raw = 0.0
    frames = [snapshot(env, total_raw)] if record else None

    def take_frame(simulator, reward):
        nonlocal total_raw
        total_raw += reward
        frames.append(snapshot(simulator, total_raw))

    try:
        obs = env.observation()
        done = False
        while not done:
            mask, prior, features = env.action_info()
            candidates = {int(a): env.decode(a) for a in np.flatnonzero(mask) if a}
            action = policy(Decision(obs, mask, prior, features, candidates, rng))
            obs, _, done, reward = env.step(action, on_tick=take_frame if record else None)
            if not record:
                total_raw += reward
        return {"seed": seed, "return": total_raw, "stats": dict(env.base.statistics),
                "frames": frames}
    finally:
        env.close()


def score_of(result):
    """Validation objective: passenger throughput first, then wait/energy/abandonments."""
    s=result['stats']
    return (float(s['DeliveredPersons(10Minutes)'])
            - .008*float(s['TotalWaitingTime(10Minutes)'])
            - .000005*float(s['EnergyConsumption(10Minutes)'])
            - 3*float(s['AbandonedPersons(10Minutes)']))
