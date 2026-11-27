"""Prompt assembly shared by Experiments 2 and 3: layered rating-question templates, the two
perspectives' pronouns (common/perspectives.yaml) and the answer formats (common/answer_formats.yaml)."""
from __future__ import annotations

from typing import Iterable

from .config import load_prompts


def pronouns(perspective: str) -> dict[str, str]:
    return load_prompts("common", "perspectives")[perspective]


def answer_format(instructions: str) -> str:
    return load_prompts("common", "answer_formats")[instructions]


def rating_question(layers: Iterable[dict], flip: bool, perspective: str, **fields) -> str:
    """Assemble one 0-100 rating question.

    ``layers`` run from general to specific, and the most specific definition of each piece wins
    (nested mappings are skipped: they are layers of their own). The chosen ``template`` is filled
    with the other pieces, the ``anchors`` as {lo}/{hi} (swapped when ``flip``), the perspective's
    pronouns and ``fields``. Pieces may use pronouns and fields but not other pieces.
    """
    spec: dict = {}
    for layer in layers:
        spec.update((k, v) for k, v in layer.items() if not isinstance(v, dict))
    values = {**pronouns(perspective), **fields}
    pieces = {k: v.format(**values) for k, v in spec.items() if isinstance(v, str) and k != "template"}
    lo, hi = (anchor.format(**values) for anchor in spec["anchors"])
    if flip:
        lo, hi = hi, lo
    return spec["template"].format_map({**values, **pieces, "lo": lo, "hi": hi})
