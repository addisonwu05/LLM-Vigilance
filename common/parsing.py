"""Turn raw model replies into numbers.

Each experiment used its own parser, and the stored scores in ``data/`` were produced
by them. Every function here reproduces one original parser exactly, quirks included,
so re-parsing stored replies gives the stored numbers (tests/test_exp*.py).
"""
from __future__ import annotations

import re

# Patterns tried in order by the Exp 2 parser (original: oktar_experiment/parser.py).
_EXP2_PATTERNS = [
    r"\*{0,2}\s*rating\s*=\s*\*{0,2}\s*(\d+)\s*\*{0,2}",               # rating = 85, **rating = 85**
    r"\*{0,2}\s*rating\s*=\s*\*{0,2}\s*(\d+)\s*/\s*100\s*\*{0,2}",       # rating = 85/100
    r"\*{0,2}\s*score\s*[:=]\s*\*{0,2}\s*(\d+)\s*\*{0,2}",                # score: 90
    r"\*{0,2}\s*rating\s*=\s*\*{0,2}\s*(\d+(?:\.\d+)?)\s*/\s*10\s*\*{0,2}",  # rating = 8.5/10
]


def parse_exp2_score(text: str) -> int:
    """Exp 2 (Oktar vignettes) score parser.

    Order: digits of the last whitespace token; then the rating/score regexes above;
    then the first standalone 0-100 integer. Raises ValueError when nothing matches
    (the original then asked a human to type the score).

    Kept quirks: a token like "85." parses as 85; a negative last token ("-5") is
    skipped by the first rule; any pattern string containing "/10" (which includes the
    "/100" pattern) multiplies its match by 10, although earlier patterns shadow it.
    """
    text = text.strip()
    if not text:
        raise ValueError("empty reply")
    possible_number = "".join(c for c in text.split()[-1] if c.isdigit() or c == "-")
    if possible_number.isdigit():
        return int(possible_number)
    for pattern in _EXP2_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if "/10" in pattern:
                value *= 10
            return round(value)
    generic = re.search(r"\b(\d{1,3})\b", text)
    if generic:
        value = int(generic.group(1))
        if 0 <= value <= 100:
            return value
    raise ValueError("No valid score found")


def parse_exp2_score_v1(text: str) -> int:
    """First version of the Exp 2 parser (kept, commented out, at the top of parser.py).

    Earlier response files were scored with it: digits (and '-') of the last word.
    Raises ValueError/IndexError when that yields nothing.
    """
    last_word = text.strip().split()[-1].strip()
    return int("".join(c for c in last_word if c.isdigit() or c == "-"))


def parse_exp3_score(text: str) -> int | None:
    """Exp 3 (YouTube) parser: the last run of digits inside the last whitespace token.

    Kept quirk: a reply ending in "85/100" parses as 100.
    """
    tokens = re.findall(r"\S+", text)
    last = tokens[-1] if tokens else ""
    numbers = re.findall(r"\d+", last)
    return int(numbers[-1]) if numbers else None


def parse_exp1_number(text: str) -> int | None:
    """Exp 1 (vigilance game): the last signed integer anywhere in the reply."""
    matches = re.findall(r"-?\d+", text)
    return int(matches[-1]) if matches else None


def unflip(score: int | None, flipped: bool) -> int | None:
    """Undo a reversed rating scale: prompts with flip=True asked for 100 - x."""
    if score is None:
        return None
    return 100 - score if flipped else score
