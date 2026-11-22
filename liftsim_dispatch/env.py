"""Headless LiftSim environment and the shared dispatch action space."""

import math
import operator
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import numpy as np

DEFAULT_CONFIG = Path(__file__).with_name("config.ini")
TICKS_PER_ACTION = 3
MAX_SIM_TICKS = 600
MAX_CANDIDATES = 16


def _liftsim_class():
    # MetaGym imports its native pyglet renderer eagerly, which can fail on macOS.
    name = "metagym.liftsim.environment.animation.rendering"
    if name in sys.modules:
        from metagym.liftsim.environment.env import LiftSim

        return LiftSim
    sys.modules[name] = ModuleType(name)
    try:
        from metagym.liftsim.environment.env import LiftSim
    finally:
        del sys.modules[name]
    return LiftSim


class CallAwareLiftSim:
    """Discrete choice 0=wait, 1..(2*F*E)=assign an idle car to an active hall call.

    No changes to LiftSim mechanics; each step sends [-1, 0] to busy cars so passenger
    destinations and valid dispatch targets are not overwritten.
    """
    def __init__(self, seed=42, config_path=DEFAULT_CONFIG):
        self.raw = _liftsim_class()(config_file=str(config_path))
        self._random_state = random.Random(seed).getstate()
        self._numpy_state = np.random.RandomState(seed).get_state()
        self.base = self.raw
        self.F = int(self.base.attribute.NumberOfFloor)
        self.E = int(self.base.attribute.ElevatorNumber)
        self.A = 1 + self.E * 2 * self.F
        self.tick = 0
        self.reset(seed)
        self.obs_dim = len(self.observation())

    def reset(self, seed=None):
        if seed is not None:
            self._random_state = random.Random(int(seed)).getstate()
            self._numpy_state = np.random.RandomState(int(seed)).get_state()
        with self._simulator_rng():
            self.raw.reset()
        mansion = self.base._mansion
        # LiftSim.reset() does not clear these rolling stats.
        for name in ('_delivered_person', '_generated_person', '_abandoned_person',
                     '_cumulative_waiting_time', '_cumulative_energy_consumption'):
            getattr(mansion, name).clear()
        self.tick = 0
        return self.observation()

    def queues(self):
        return self.base._mansion.waiting_queue

    def is_free(self, i):
        state = self.base.state.ElevatorStates[i]
        phys = self.base._mansion._elevators[i]
        return (state.CurrentDispatchTarget <= 0 and
                len(state.ReservedTargetFloors) == 0 and
                abs(state.Velocity) < 0.04 and state.DoorState < 0.03 and
                phys.loaded_people_num == 0 and
                not phys._entering_person and not phys._exiting_person)

    def _core_observation(self):
        s = self.base.state.ElevatorStates
        obs = []
        for i, car in enumerate(s):
            booked = set(car.ReservedTargetFloors)
            obs.extend([
                car.Floor / self.F, car.Velocity / max(car.MaximumSpeed, 0.1),
                float(car.Direction), float(car.DoorState),
                car.CurrentDispatchTarget / self.F,
                float(car.DispatchTargetDirection),
                car.LoadWeight / max(car.MaximumLoad, 1.0),
                len(booked) / self.F, float(self.is_free(i)),
            ])
            obs.extend([float(f in booked) for f in range(1, self.F+1)])
        now = self.base._mansion._config.raw_time
        for direction_queues in self.queues():
            for q in direction_queues:
                obs.append(min(len(q), 15) / 5.0)
                age = (now - q[-1].AppearTime) if len(q) else 0.0
                obs.append(np.clip(age / 90.0, 0.0, 4.0))
        return np.asarray(obs, dtype=np.float32)

    def encode(self, car, floor, direction):
        j = (floor-1) if direction == 1 else self.F + floor-1
        return 1 + car * (2*self.F) + j

    def decode(self, a):
        k = int(a)-1
        car, loc = divmod(k, 2*self.F)
        direction = 1 if loc < self.F else -1
        floor = loc+1 if loc < self.F else loc-self.F+1
        return car, floor, direction

    def _mask_and_prior(self):
        """Boolean mask and prior logits (a deterministic nearest-call inductive bias)."""
        mask = np.zeros(self.A, dtype=bool)
        prior = np.full(self.A, -12.0, dtype=np.float32)
        mask[0] = True; prior[0] = 0.0
        up, down = self.queues()
        occupied = set()
        for car in self.base.state.ElevatorStates:
            if car.CurrentDispatchTarget > 0:
                occupied.add((int(car.CurrentDispatchTarget), int(car.DispatchTargetDirection)))
        now = self.base._mansion._config.raw_time
        pairs = []
        for car_id, car in enumerate(self.base.state.ElevatorStates):
            if not self.is_free(car_id):
                continue
            for direction, qs in ((1, up), (-1, down)):
                for floor, q in enumerate(qs, start=1):
                    if not q or (floor, direction) in occupied:
                        continue
                    distance = abs(float(car.Floor) - floor)
                    oldest = max(0.0, now - q[-1].AppearTime)
                    # The prior is not a hardcoded action: PPO learns corrections to it.
                    preference = (-1.00 * distance + 0.022 * min(oldest, 300.0)
                                  + 1.25 * math.log1p(len(q)))
                    pairs.append((preference, self.encode(car_id, floor, direction)))
        # Limit combinatorial branching; wait is used only if there is no valid call.
        if pairs:
            pairs.sort(reverse=True)
            mask[0] = False
            for score, index in pairs[:MAX_CANDIDATES]:
                mask[index] = True
                prior[index] = float(score)
        return mask, prior

    def step(self, action, on_tick=None):
        if self.tick >= MAX_SIM_TICKS:
            raise RuntimeError("Reset the environment before stepping a finished episode")
        a = operator.index(action)
        if not 0 <= a < self.A:
            raise ValueError(f'Action {a} out of bounds')
        mask, _ = self._mask_and_prior()
        if not mask[a]:
            raise ValueError(f'Illegal call assignment {a}; use action mask')
        commands = [-1, 0] * self.E
        if a != 0:
            car_id, floor, direction = self.decode(a)
            commands[2*car_id:2*car_id+2] = [int(floor), int(direction)]
        prev_stats = dict(self.base.statistics)
        raw_reward = 0.0
        for t in range(TICKS_PER_ACTION):
            cmd = commands if t == 0 else [-1, 0] * self.E
            with self._simulator_rng():
                _, r, _, _ = self.raw.step(cmd)
            raw_reward += float(r)
            self.tick += 1
            if on_tick is not None:
                on_tick(self, float(r))
            if self.tick >= MAX_SIM_TICKS:
                break
        train_reward = self.training_reward(prev_stats, dict(self.base.statistics))
        done = (self.tick >= MAX_SIM_TICKS)
        return self.observation(), float(train_reward), done, float(raw_reward)

    def close(self):
        self.raw.close()

    FEAT_DIM = 14

    def observation(self):
        core = list(self._core_observation())
        up, down = self.queues()
        now = self.base._mansion._config.raw_time
        # The oldest person is at the RIGHT of LiftSim's deque (appendleft).
        for q in list(up) + list(down):
            if q:
                ages = [max(0., now-p.AppearTime) for p in q]
                core.extend([np.clip(np.mean(ages)/120.,0,4),
                             np.clip(np.max(ages)/300.,0,3)])
            else:
                core.extend([0.,0.])
        free_count = sum(self.is_free(i) for i in range(self.E))
        core += [free_count/self.E, sum(map(len,up))/30., sum(map(len,down))/30.,
                 min(now/300., 2.0)]
        return np.asarray(core,dtype=np.float32)

    @staticmethod
    def training_reward(before, after):
        """Dense, measured objective; all components are per-STEP deltas, not totals."""
        def delta(key): return max(0., after[key]-before[key])
        delivered=delta('DeliveredPersons(10Minutes)')
        abandoned=delta('AbandonedPersons(10Minutes)')
        waiting=delta('TotalWaitingTime(10Minutes)')
        energy=delta('EnergyConsumption(10Minutes)')
        return float(2.0*delivered - 0.012*waiting - 0.000006*energy - 5.0*abandoned)

    def action_info(self):
        mask, prior = self._mask_and_prior()
        feats = np.zeros((self.A, self.FEAT_DIM), dtype=np.float32)
        available = np.flatnonzero(mask)
        now = self.base._mansion._config.raw_time
        waiting = self.queues()
        states = self.base.state.ElevatorStates
        idle_idx = [i for i in range(self.E) if self.is_free(i)]
        active_queue_count = sum(bool(q) for direction in waiting for q in direction)
        global_wait = sum(len(q) for direction in waiting for q in direction)
        # Features are candidate-specific; the actor shares scorer weights across all floors/cars.
        for action in available:
            if action == 0: continue
            car_id, floor, direction = self.decode(action)
            car = states[car_id]
            q = waiting[0 if direction == 1 else 1][floor-1]
            ages = [max(0., now-p.AppearTime) for p in q]
            nearest_other = min((abs(states[j].Floor-floor) for j in idle_idx if j!=car_id), default=self.F)
            max_age = max(ages) if ages else 0.
            mean_age = float(np.mean(ages)) if ages else 0.
            # Hall-call systems know only up/down, not riders' destination floors.
            # Use a direction-based expected journey length; no private destinations.
            mean_ride = (self.F-floor if direction==1 else floor-1) * .5
            feats[action] = np.array([
                float(car.Floor)/self.F, floor/self.F, abs(float(car.Floor)-floor)/self.F,
                direction, np.log1p(len(q))/3., min(max_age/300.,2.),
                min(mean_age/300.,2.), mean_ride/self.F,
                nearest_other/self.F, len(idle_idx)/self.E,
                active_queue_count/(2*self.F), min(global_wait/30.,3.),
                min(now/300.,2.), np.sign(floor-float(car.Floor))
            ], dtype=np.float32)
        return mask, prior, feats


    @contextmanager
    def _simulator_rng(self):
        # MetaGym uses global RNGs. Keep each simulation stream independent of
        # policy code and of validation episodes interleaved with training.
        random_state = random.getstate()
        numpy_state = np.random.get_state()
        random.setstate(self._random_state)
        np.random.set_state(self._numpy_state)
        try:
            yield
        finally:
            self._random_state = random.getstate()
            self._numpy_state = np.random.get_state()
            random.setstate(random_state)
            np.random.set_state(numpy_state)
