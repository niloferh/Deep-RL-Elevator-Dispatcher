# Deep RL Elevator Dispatcher

Shared LiftSim environment and paired benchmarks for classic dispatch, PPO, and team policies such as Q-learning. This package extracts the simulator configuration, imitation warm start, PPO/GAE training, evaluation, and headless animation from `LiftSim_PPO_Improved_Training_Animation(2).ipynb`.

## Run the benchmark

Use Python 3.10–3.12. The standard benchmark runs 30 held-out seeds with the notebook's 10-floor, 4-car uniform traffic scenario. Every policy gets the same seeds and 600 physics ticks per episode.

```bash
python -m pip install -e .
python -m liftsim_dispatch.benchmark
```

This runs the notebook's expert heuristic, a classic nearest-car heuristic, and a random valid-call baseline. The runner writes `outputs/benchmark_results.csv` with official LiftSim reward, deliveries, generated passengers, abandonments, passenger time, energy, and the notebook's validation objective.

## Train and evaluate PPO

```bash
python -m pip install -e '.[ppo,report]'
python -m liftsim_dispatch.train
python -m liftsim_dispatch.benchmark \
  --checkpoint outputs/LiftSim_PPO_Improved_best.pt \
  --plot --video PPO Random
```

Training saves the best validation checkpoint and learning curves under `outputs/`. The benchmark prints PPO–Random paired bootstrap intervals and saves the optional comparison plot and side-by-side MP4 there. For a quick smoke run, pass `--decisions 256 --demo-episodes 1 --imitation-epochs 1` to training and `--seeds 12001 12002` to the benchmark. This short run checks the pipeline. Use the default training budget and seeds for performance comparisons.

## Add a team policy

Place a module such as `my_policy.py` at the repository root. Expose a factory that takes the episode seed and returns a callable. The runner calls the factory for each episode so policy state can start fresh.

```python
import numpy as np

def make_policy(seed):
    def choose(decision):
        valid = np.flatnonzero(decision.mask)
        # Replace the heuristic scores with your fitted Q(state, action) values.
        scores = decision.prior
        return int(valid[np.argmax(scores[valid])])

    return choose
```

```bash
python -m liftsim_dispatch.benchmark --policy 'My policy=my_policy:make_policy'
```

| Decision field | Contents |
| --- | --- |
| `observation` | Float32 vector with car state, hall-call queue lengths and ages, and aggregate traffic |
| `mask` | Boolean vector of legal action IDs for this decision |
| `prior` | Notebook heuristic score for each action |
| `features` | Matrix of 14 pair features per action. See `env.py` for feature order. |
| `candidates` | Legal nonzero IDs mapped to `(zero_based_car, one_based_floor, direction)` |
| `rng` | Episode-seeded NumPy generator for the policy |

Action 0 waits when no assignment is available. The remaining IDs encode one idle-car-to-hall-call assignment, with up `+1` and down `-1`. The notebook's heuristic caps the candidate set at 16. All policies share that cap and the same three physics ticks per decision. Policies should select only IDs allowed by the mask.

To train a different agent, instantiate `CallAwareLiftSim` from `liftsim_dispatch.env`. Use `reset(seed)` for the initial observation, `action_info()` for `(mask, prior, features)`, and `step(action)` for `(observation, shaped_reward, done, official_reward)`. Call `close()` when finished. For custom scenarios, pass the same `--config` to training and evaluation. Keep checkpoints paired with their training configuration.

| File | Responsibility |
| --- | --- |
| `env.py`, `config.ini` | Headless simulator, traffic, observations, legal actions, and rewards |
| `policies.py` | Policy interface and reference dispatchers |
| `ppo.py`, `train.py` | PPO model, checkpoint loading, expert imitation, and PPO/GAE training |
| `evaluation.py`, `benchmark.py` | Episode evaluation, matched-seed comparisons, and metrics |
| `render.py` | Side-by-side MP4 rendering |
