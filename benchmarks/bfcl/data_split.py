"""
BFCL data split utilities.

Creates reproducible Strain / Ssel / test splits from BFCL categories.
Categories are loaded from the bfcl-eval package's bundled data files.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# BFCL V3 single-turn categories used for "Single" metric
SINGLE_TURN_CATEGORIES = [
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
]

# BFCL V3 multi-turn categories used for "Multi" metric
MULTI_TURN_CATEGORIES = [
    "multi_turn_base",
]


@dataclass
class BFCLSplits:
    train_indices: list[tuple[str, int]]   # (category, index) pairs
    sel_indices: list[tuple[str, int]]
    test_indices: list[tuple[str, int]]


def make_bfcl_splits(
    categories: list[str],
    all_instances: list[dict],
    train_size: int,
    sel_size: int,
    seed: int = 42,
) -> BFCLSplits:
    """
    Split a flat list of BFCL instances into Strain / Ssel / test.
    Each element in all_instances must have a "category" field.
    """
    rng = random.Random(seed)
    indexed = list(enumerate(all_instances))
    rng.shuffle(indexed)

    train_flat = [(all_instances[i]["category"], i) for i, _ in indexed[:train_size]]
    sel_flat = [(all_instances[i]["category"], i) for i, _ in indexed[train_size: train_size + sel_size]]
    test_flat = [(all_instances[i]["category"], i) for i, _ in indexed[train_size + sel_size:]]

    return BFCLSplits(
        train_indices=train_flat,
        sel_indices=sel_flat,
        test_indices=test_flat,
    )


def get_bfcl_data_dir() -> Path:
    """Find the bfcl-eval package's data directory."""
    try:
        import bfcl_eval
        pkg_dir = Path(bfcl_eval.__file__).parent
        # Try common data dir names
        for candidate in ["data", "dataset", "gorilla_openfunctions_v3_data", "bfcl_data"]:
            d = pkg_dir / candidate
            if d.exists():
                return d
        return pkg_dir
    except ImportError:
        raise ImportError("bfcl-eval not installed. Run: pip install bfcl-eval")


def load_bfcl_category(category: str) -> list[dict]:
    """
    Load all instances from a BFCL category.
    Returns a list of dicts with fields: id, question, function, category.
    """
    import json

    data_dir = get_bfcl_data_dir()

    # Try multiple file naming conventions (V3 and V4)
    candidates = [
        data_dir / f"BFCL_v3_{category}.json",
        data_dir / f"BFCL_v4_{category}.json",
        data_dir / f"{category}.json",
    ]
    # Also search recursively
    found = None
    for c in candidates:
        if c.exists():
            found = c
            break
    if found is None:
        matches = list(data_dir.rglob(f"*{category}*.json"))
        matches = [m for m in matches if "possible_answer" not in str(m) and "score" not in str(m)]
        if matches:
            found = matches[0]

    if found is None:
        raise FileNotFoundError(
            f"BFCL data file for category '{category}' not found in {data_dir}. "
            "Run: pip install --upgrade bfcl-eval"
        )

    instances = []
    with open(found) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    obj["category"] = category
                    instances.append(obj)
                except json.JSONDecodeError:
                    pass

    logger.info("Loaded %d instances from %s (%s)", len(instances), found.name, category)
    return instances


def load_bfcl_ground_truth(category: str) -> dict[str, Any]:
    """
    Load ground truth answers for a BFCL category.
    Returns a dict mapping instance_id → ground_truth.
    """
    import json

    data_dir = get_bfcl_data_dir()
    candidates = [
        data_dir / "possible_answer" / f"BFCL_v3_{category}.json",
        data_dir / "possible_answer" / f"BFCL_v4_{category}.json",
        data_dir / "possible_answer" / f"{category}.json",
    ]
    found = None
    for c in candidates:
        if c.exists():
            found = c
            break
    if found is None:
        matches = list(data_dir.rglob(f"*possible_answer*{category}*.json"))
        if matches:
            found = matches[0]

    if found is None:
        logger.warning("Ground truth not found for category '%s'.", category)
        return {}

    gt: dict[str, Any] = {}
    with open(found) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    gt[obj["id"]] = obj.get("ground_truth", [])
                except json.JSONDecodeError:
                    pass
    return gt


def load_all_bfcl(categories: list[str]) -> tuple[list[dict], dict[str, Any]]:
    """Load instances and ground truth for multiple BFCL categories."""
    all_instances: list[dict] = []
    all_gt: dict[str, Any] = {}
    for cat in categories:
        try:
            instances = load_bfcl_category(cat)
            gt = load_bfcl_ground_truth(cat)
            all_instances.extend(instances)
            all_gt.update(gt)
        except Exception as exc:
            logger.warning("Failed to load BFCL category '%s': %s", cat, exc)
    return all_instances, all_gt
