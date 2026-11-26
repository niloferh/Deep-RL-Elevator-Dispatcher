# PPO elevator dispatch agent

The PPO contribution lives in `project/agents/ppo_agent.py` and is intended to be used with the shared LiftSim environment and evaluation framework on the [khalesah branch](https://github.com/niloferh/Deep-RL-Elevator-Dispatcher/tree/khalesah).

Requires NumPy and PyTorch, in addition to the shared LiftSim dependencies.

```python
from project.agents.ppo_agent import PPOAgent

# env is the existing LiftSim instance from the shared framework.
agent = PPOAgent(env)
agent.fit(total_decisions=60_000, episode_steps=3000)

observation = env.reset()
agent.reset()
action = agent.act(observation)
```

To reuse trained weights, call `agent.save("ppo.pt")` after training and initialize with `PPOAgent(env, checkpoint="ppo.pt")` for evaluation. The shared framework owns environment configuration, rendering, evaluation and metrics. This branch is not a standalone simulator.
