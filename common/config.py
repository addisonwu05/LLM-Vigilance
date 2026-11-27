"""Paths, YAML loading, and the model registry.

Each experiment has its own folder (exp1/, exp2/, exp3/) holding its prompts.yaml, models.yaml, data/ and
outputs/; common/ holds what the experiments share.
"""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("VIGILANCE_ROOT", Path(__file__).resolve().parents[1]))
COMMON = ROOT / "common"
WEBPPL_MODEL = COMMON / "motivated_model.wppl"
POSTERIOR_TABLE = COMMON / "outputs" / "posterior_table.json"   # regenerable: `vigilance posteriors`


def data_dir(experiment: str) -> Path:
    return ROOT / experiment / "data"


def output_dir(experiment: str) -> Path:
    """Tables and figures rebuilt by the `vigilance` commands (git-ignored)."""
    return ROOT / experiment / "outputs"


@lru_cache(maxsize=None)
def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(folder: str, name: str = "prompts") -> dict:
    """<folder>/<name>.yaml: load_prompts("exp2") is exp2/prompts.yaml."""
    return _load(ROOT / folder / f"{name}.yaml")


def load_models() -> dict:
    """common/models.yaml: the API providers and the model registry."""
    return _load(COMMON / "models.yaml")


def _runs(experiment: str) -> dict:
    """<experiment>/models.yaml: the models an experiment ran, and their settings."""
    return _load(ROOT / experiment / "models.yaml")


def merge(*layers: dict) -> dict:
    """Combine mappings: later layers win, and nested mappings are merged key by key."""
    out: dict = {}
    for layer in layers:
        for key, value in layer.items():
            both = isinstance(value, dict) and isinstance(out.get(key), dict)
            out[key] = merge(out[key], value) if both else value
    return out


def models(experiment: str) -> list[str]:
    """Models run in an experiment ('exp1', 'exp2' or 'exp3'), in the paper's table order."""
    return list(_runs(experiment)["runs"])


def model_spec(experiment: str, model: str) -> dict:
    """A model's registry entry (label, provider, id), overlaid with the experiment's defaults
    and then this model's own run settings. Returns a fresh copy."""
    runs = _runs(experiment)
    return copy.deepcopy(merge(load_models()["models"][model], runs.get("defaults", {}), runs["runs"][model]))


def label(model: str) -> str:
    """Display name used in the paper's tables."""
    return load_models()["models"][model]["label"]
