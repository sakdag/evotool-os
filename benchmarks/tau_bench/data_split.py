"""
τ-Bench data split utilities.

Creates reproducible Strain / Ssel / test splits from the τ-Bench task lists.
- Retail:  uses train split for Strain+Ssel; official test split for evaluation.
- Airline: only has a test split; we carve out Strain+Ssel from the first N tasks
           using a fixed seed; remaining tasks are the evaluation set.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TauBenchSplits:
    train_indices: list[int]
    sel_indices: list[int]
    test_indices: list[int]


def make_splits(
    total: int,
    train_size: int,
    sel_size: int,
    seed: int = 42,
    test_offset: int = 0,
) -> TauBenchSplits:
    """
    Partition task indices [0, total) into Strain, Ssel, test.

    For airline (test_offset > 0): the first `test_offset` indices are used
    for Strain+Ssel; indices >= test_offset form the evaluation set.
    """
    rng = random.Random(seed)
    pool = list(range(total))
    rng.shuffle(pool)

    if test_offset > 0:
        # Carve Strain+Ssel from [0, test_offset); rest is evaluation
        train_pool = pool[:min(test_offset, total)]
        test_pool = pool[min(test_offset, total):]
    else:
        train_pool = pool
        test_pool = []

    train = train_pool[:train_size]
    sel = train_pool[train_size: train_size + sel_size]

    if test_offset == 0:
        # For retail: test indices are from a fresh load of the test split
        # (handled by calling env with task_split="test")
        test = []
    else:
        test = test_pool

    return TauBenchSplits(train_indices=train, sel_indices=sel, test_indices=test)
