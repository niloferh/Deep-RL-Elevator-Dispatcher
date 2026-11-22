# Deep RL Elevator Dispatcher

Train an elevator dispatcher with Proximal Policy Optimization (PPO) and compare it with random dispatch or teammate policies in the same LiftSim traffic scenario.

## Quick start

Run these commands from the repository root with Python 3.10–3.12. The activation command is for macOS and Linux.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m liftsim_dispatch.benchmark --seeds 12001 12002
```

The tests check that the benchmark creates fresh policies for matched seeds, video recording preserves episode results, and invalid assignments are rejected. The benchmark command runs the random valid-call baseline and writes two rows to `outputs/benchmark_results.csv`.

## Test PPO end to end

```bash
python -m pip install -e '.[ppo,report]'
python -m liftsim_dispatch.train \
  --decisions 256 --demo-episodes 1 --imitation-epochs 1 \
  --output-dir outputs/smoke
python -m liftsim_dispatch.benchmark \
  --checkpoint outputs/smoke/LiftSim_PPO_Improved_best.pt \
  --seeds 12001 12002 --output-dir outputs/smoke \
  --plot --video PPO Random
```

The smoke test exercises training, checkpoint loading, a matched comparison, and MP4 export. It writes four benchmark rows, one for each policy and seed. Use the full run to measure performance.

## Run the full experiment

With the PPO and reporting extras installed, use the default 60,000-decision training budget and 30 held-out benchmark seeds.

```bash
python -m liftsim_dispatch.train
python -m liftsim_dispatch.benchmark \
  --checkpoint outputs/LiftSim_PPO_Improved_best.pt \
  --plot --video PPO Random
```

| Output under `outputs/` | Contents |
| --- | --- |
| `LiftSim_PPO_Improved_best.pt` | PPO checkpoint selected on validation seeds |
| `LiftSim_PPO_learning_curves.png` | Shaped training return and validation score |
| `benchmark_results.csv` | Per-policy, per-seed official LiftSim reward, deliveries, generated passengers, abandonments, passenger time, energy, and validation objective |
| `benchmark_comparison.png` | Metric distributions across benchmark seeds when `--plot` is enabled |
| `benchmark_comparison.mp4` | Side-by-side replay on a separate matched seed when `--video` is enabled |

## How the method works

1. **Build legal assignments.** At each decision, LiftSim exposes car positions, hall-call queues, and waiting times. The action mask limits dispatch to eligible idle-car and hall-call pairs. Busy cars keep their existing destinations.
2. **Train the policy.** PPO first imitates a heuristic that scores assignments by distance, queue size, and wait age. A shared neural scorer then learns corrections to those scores through PPO with generalized advantage estimation (GAE). The training reward favors completed rides and penalizes waiting, energy, and abandonments.
3. **Compare on matched traffic.** The benchmark evaluates PPO, random dispatch, and any supplied team policies on the same held-out seeds. Every policy gets a 10-floor, 4-car building and 600 physics ticks per episode under the default scenario. The report uses official LiftSim reward alongside operational metrics and prints paired bootstrap intervals for PPO versus Random.

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

## Repository files

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Declares the package, supported Python versions, and optional PPO and reporting dependencies |
| `liftsim_dispatch/__init__.py` | Defines the importable Python package |
| `liftsim_dispatch/config.ini` | Defines the default building, simulation time step, and passenger traffic |
| `liftsim_dispatch/env.py` | Runs headless LiftSim and exposes observations, legal dispatch actions, and rewards |
| `liftsim_dispatch/policies.py` | Defines the policy interface, decision fields, and random baseline |
| `liftsim_dispatch/ppo.py` | Implements the pairwise actor–critic model and loads trained checkpoints |
| `liftsim_dispatch/train.py` | Runs heuristic imitation, PPO/GAE training, and checkpoint selection |
| `liftsim_dispatch/evaluation.py` | Runs one seeded episode and records metrics and optional frame data |
| `liftsim_dispatch/benchmark.py` | Runs matched-seed comparisons, loads team policies, and writes reports |
| `liftsim_dispatch/render.py` | Turns two recorded episodes into a side-by-side MP4 |
| `tests/test_evaluation.py` | Regression tests for matched seeds, recording, and valid actions |
| `LiftSim_PPO_Improved_Training_Animation.ipynb` | Original notebook workflow |
| `.gitignore` | Excludes generated outputs, virtual environments, and Python build files |
