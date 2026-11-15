"""Parameter-search arm: tune S3-FIFO's knobs under the same experiment budget.

No agent and no views: sample configurations, keep the best on the development
suite, and score it exactly like an agent's submission. If the visual agent
cannot beat this at equal budget, its edits are not adding anything that
tuning could not.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

from .lab import Lab

GRID = {
    "SMALL_RATIO": (0.02, 0.05, 0.1, 0.15, 0.2, 0.3),
    "GHOST_RATIO": (0.5, 0.9, 1.5, 2.0),
    "MOVE_THRESHOLD": (1, 2, 3),
    "MAX_FREQ": (1, 2, 3, 5),
}
DEFAULT = {"SMALL_RATIO": 0.1, "GHOST_RATIO": 0.9, "MOVE_THRESHOLD": 2, "MAX_FREQ": 3}


def policy_source(config: dict[str, Any]) -> str:
    body = "\n".join(f"    {name} = {value!r}" for name, value in config.items())
    return f"from s3fifo import S3FIFO\n\n\nclass CachePolicy(S3FIFO):\n{body}\n"


def search(lab: Lab, *, budget: int, seed: int = 0, prune: bool = False) -> dict[str, Any]:
    names = list(GRID)
    configurations = [dict(zip(names, values, strict=True)) for values in itertools.product(*GRID.values())]
    configurations = [config for config in configurations if config != DEFAULT]
    random.Random(seed).shuffle(configurations)
    trials = []
    for config in configurations[:budget]:
        name = "params " + " ".join(f"{k}={v}" for k, v in config.items())
        experiment, comparison = lab.dev_experiment(name, policy_source(config), prune=prune)
        change = comparison["mean_relative_change"] if comparison is not None else None
        trials.append({"experiment": experiment.id, "config": config, "dev_mean_relative_change": change})
    scored = [trial for trial in trials if trial["dev_mean_relative_change"] is not None]
    if not scored:
        return {"arm": "parameter-search", "budget": budget, "seed": seed, "trials": trials, "score": None}
    best = min(scored, key=lambda trial: trial["dev_mean_relative_change"])
    evaluation = lab.final_evaluation(best["experiment"], prune=prune)
    return {
        "arm": "parameter-search",
        "budget": budget,
        "seed": seed,
        "trials": trials,
        "selected": best,
        "evaluation": evaluation,
        "score": evaluation["score"],
    }
