"""Compare dispatch policies on matched LiftSim seeds and write a per-seed CSV."""

import argparse
import csv
from importlib import import_module
from pathlib import Path

import numpy as np

from .env import DEFAULT_CONFIG
from .evaluation import evaluate_episode, score_of
from .policies import random_valid

TEST_SEEDS = range(12001, 12031)
STAT_KEYS = {
    "delivered": "DeliveredPersons(10Minutes)",
    "generated": "GeneratedPersons(10Minutes)",
    "abandoned": "AbandonedPersons(10Minutes)",
    "waiting_seconds": "TotalWaitingTime(10Minutes)",
    "energy": "EnergyConsumption(10Minutes)",
}
METRICS = ("raw_reward", "delivered", "abandoned", "waiting_seconds", "energy")


def run_benchmark(policy_factories, seeds=TEST_SEEDS, *, config_path=DEFAULT_CONFIG):
    """Build a fresh policy via factory(seed) for each matched seed and policy."""
    rows = []
    for seed in seeds:
        for name, factory in policy_factories.items():
            result = evaluate_episode(factory(seed), seed, config_path=config_path)
            row = {"seed": seed, "policy": name, "raw_reward": result["return"]}
            row.update({column: result["stats"][key] for column, key in STAT_KEYS.items()})
            row["validation_objective"] = score_of(result)
            rows.append(row)
    return rows


def write_results(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(rows, names):
    grouped = {name: [row for row in rows if row["policy"] == name] for name in names}
    for name, group in grouped.items():
        print(name)
        for metric in METRICS:
            values = np.array([row[metric] for row in group])
            std = values.std(ddof=1) if len(values) > 1 else float("nan")
            print(f"  {metric:16s} mean={values.mean():12.3f}  sd={std:12.3f}")

    if "PPO" in grouped and "Random" in grouped:
        rng = np.random.default_rng(2026)
        print("PPO minus Random, with paired bootstrap intervals")
        for metric in METRICS:
            delta = np.array([a[metric] - b[metric]
                              for a, b in zip(grouped["PPO"], grouped["Random"])])
            draws = rng.choice(delta, size=(10_000, len(delta)), replace=True)
            low, high = np.quantile(draws.mean(axis=1), [0.025, 0.975])
            wins = delta > 0 if metric in ("raw_reward", "delivered") else delta < 0
            print(f"  {metric:16s} Δ={delta.mean():+.3f}  95% CI [{low:+.3f}, {high:+.3f}]"
                  f"  PPO win rate={wins.mean():.1%}")
        # Keep the notebook's throughput comparison separate from official reward.
        ppo = np.mean([r["delivered"] for r in grouped["PPO"]])
        random = np.mean([r["delivered"] for r in grouped["Random"]])
        if random:
            print(f"PPO throughput improvement: {ppo / random - 1:+.2%}")
        print("PASS: PPO beats random on deliveries" if ppo > random
              else "FAIL: PPO did not beat random on deliveries")


def plot_results(path, rows, names):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(16, 4))
    axes = fig.subplots(1, 3)
    for ax, metric in zip(axes, ("delivered", "waiting_seconds", "energy")):
        groups = [[r[metric] for r in rows if r["policy"] == name] for name in names]
        ax.boxplot(groups, showmeans=True)
        ax.set_xticks(range(1, len(names) + 1), labels=list(names), rotation=20)
        ax.set_title(metric.replace("_", " ").capitalize())
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def load_factory(spec):
    module, separator, attr = spec.partition(":")
    if not module or not separator or not attr:
        raise ValueError("Policy factory must be specified as module:function")
    return getattr(import_module(module), attr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Include PPO from a saved checkpoint")
    parser.add_argument("--policy", action="append", default=[], metavar="NAME=MODULE:FACTORY")
    parser.add_argument("--seeds", nargs="+", type=int, default=TEST_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--video", nargs=2, metavar=("LEFT", "RIGHT"),
                        help="Render two policy names on a separate matched seed")
    parser.add_argument("--video-seed", type=int, default=12040)
    args = parser.parse_args()

    factories = {"Random": lambda seed: random_valid}
    if args.checkpoint:
        import torch

        from .ppo import load_policy

        torch.set_num_threads(1)
        ppo = load_policy(args.checkpoint)
        factories["PPO"] = lambda seed: ppo

    for entry in args.policy:
        name, separator, spec = entry.partition("=")
        if not name or not separator or name in factories:
            parser.error(f"Expected a unique policy name in NAME=MODULE:FACTORY: {entry}")
        factories[name] = load_factory(spec)

    if args.video and any(name not in factories for name in args.video):
        parser.error(f"Video policy names must be among: {', '.join(factories)}")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Benchmark seeds must be unique")

    rows = run_benchmark(factories, args.seeds, config_path=args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_results(args.output_dir / "benchmark_results.csv", rows)
    report(rows, factories)
    if args.plot:
        plot_results(args.output_dir / "benchmark_comparison.png", rows, factories)
    if args.video:
        from .render import write_video

        recordings = [
            evaluate_episode(factories[name](args.video_seed), args.video_seed,
                             config_path=args.config, record=True)["frames"]
            for name in args.video
        ]
        write_video(args.output_dir / "benchmark_comparison.mp4", *recordings, args.video)


if __name__ == "__main__":
    main()
