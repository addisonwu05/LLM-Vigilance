"""JSON files, and the append-and-save loop that makes every data collection resumable."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: Any, *, indent: int = 4, ensure_ascii: bool = True) -> Path:
    path = Path(path)
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=ensure_ascii), encoding="utf-8")
    return path


def extend(path: Path, n: int, make_item: Callable[[int], Any]) -> Path:
    """Append ``make_item(i)`` to the JSON list in ``path`` until it holds ``n`` items, saving
    after each one, so an interrupted collection resumes where it stopped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    items = read_json(path) if path.exists() else []
    while len(items) < n:
        items.append(make_item(len(items)))
        write_json(path, items)
    return path
