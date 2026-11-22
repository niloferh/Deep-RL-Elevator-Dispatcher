import random
import unittest

import numpy as np

from liftsim_dispatch.benchmark import run_benchmark
from liftsim_dispatch.env import CallAwareLiftSim, MAX_SIM_TICKS
from liftsim_dispatch.evaluation import evaluate_episode
from liftsim_dispatch.policies import expert, nearest_car


class EvaluationTests(unittest.TestCase):
    def test_matched_seeds_and_policy_factory(self):
        seeds_seen = []

        def make_policy(seed):
            seeds_seen.append(seed)
            return nearest_car

        seeds = [12001, 12002]
        rows = run_benchmark({"Classic": make_policy, "Expert": lambda seed: expert}, seeds)
        self.assertEqual(seeds_seen, seeds)
        self.assertEqual([(row["seed"], row["policy"]) for row in rows],
                         [(seed, name) for seed in seeds for name in ("Classic", "Expert")])
        for classic, reference in zip(rows[::2], rows[1::2]):
            self.assertGreater(classic["delivered"], 0)
            self.assertEqual(classic["generated"], reference["generated"])

    def test_recording_and_policy_randomness_preserve_episode(self):
        def choose(decision):
            random.random()
            np.random.random()
            return expert(decision)

        plain = evaluate_episode(expert, 12001)
        recorded = evaluate_episode(choose, 12001, record=True)
        self.assertEqual(plain["stats"], recorded["stats"])
        self.assertAlmostEqual(plain["return"], recorded["return"])
        self.assertEqual(len(recorded["frames"]), MAX_SIM_TICKS + 1)

    def test_action_encoding_and_illegal_action(self):
        env = CallAwareLiftSim()
        try:
            self.assertEqual(env.decode(env.encode(1, 7, -1)), (1, 7, -1))
            self.assertTrue(env.action_info()[0][0])
            with self.assertRaises(ValueError):
                env.step(1)
            self.assertEqual(env.tick, 0)
        finally:
            env.close()
