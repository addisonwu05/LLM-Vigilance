"""Experiment 2: prompts and collection vs. the original scripts, each pipeline stage vs. the original
pipeline's stored outputs (exp2/data/reference/), and the paper's tables.

The prompt and replay goldens were captured by running the ORIGINAL scripts with a fake LLM.
"""
import json

import numpy as np
import pandas as pd
import pytest
from conftest import FakeLLM, load_golden, require, runs, select, sha256
from paper_values import EXP2_FULL, EXP2_INTRA, EXP2_MAIN

from common import bayes
from common.config import label, model_spec
from common.design import PERSPECTIVES, conditions
from common.files import read_json
from common.parsing import parse_exp2_score, parse_exp2_score_v1
from exp2 import DATA, analysis, audit, collect, convert, prompts, tables

REF = DATA / "reference"
PROB = "params.dist.X..a0..1..a1..0..prob"
TOL = 0.0015  # printed to 3 decimals; the sheet rounded some intermediate values differently
COLS = ("Bayesian-LLM", "Bayesian-Human", "LLM-Human")


# ------------------------------------------------------------------ prompts and collection
def test_every_prompt_matches_original():
    golden = load_golden("exp2_prompts.json")
    bad = [g for g in golden["prompts"]
           if sha256(prompts.question(g["scenario"], g["perspective"], g["purpose"], g["person"],
                                      g["reward"], g["flip"], g["instructions"])) != g["sha256"]]
    assert not bad, f"{len(bad)} of {len(golden['prompts'])} prompts differ, e.g. {bad[0]}"
    assert len(golden["prompts"]) == 1296


def test_examples_readable():
    ex = load_golden("exp2_prompts.json")["examples"]
    assert prompts.question("credit_card", "first", "trust", "a stranger", 100, True, "cot") == ex["trust"]


@pytest.mark.parametrize("variant", ["always_flip_separate", "per_subject_separate", "per_subject_merged", "reasoning"])
def test_seeded_collection_replays_original(variant, tmp_path):
    g = load_golden("exp2_replays.json")[variant]
    spec = {"provider": "fake", "id": "fake", "max_tokens": None, "samples": g["n"], **g["protocol"]}
    fake = FakeLLM("50")
    out = collect.collect("fake", g["scenario"], g["perspective"], g["purpose"], spec["instructions"],
                          n=g["n"], seed=g["seed"], out=tmp_path / "out.json", chat=fake, spec=spec)
    requests = [r["messages"] for r in fake.requests]
    assert requests[0] == g["first_request"]
    assert len(requests) == g["n_requests"]
    assert sha256(json.dumps(requests)) == g["requests_sha256"]
    assert sha256(out.read_text()) == g["output_sha256"]


# ------------------------------------------------------------------ pipeline vs. the original pipeline
# The original input for this condition was written before two stored scores changed; both are
# parse errors in the stored file (the replies say 70 and 65). See docs/PROVENANCE.md.
KNOWN_INPUT_DIFFS = {
    ("deepseek-r1", "credit_card", "user", "reasoning"): {(4, "a stranger", 100), (2, "a stranger", 1000)},
}


@pytest.mark.parametrize("model,scenario,persp,instr", runs("exp2", prompts.scenarios(), PERSPECTIVES))
def test_model_inputs_match_original(model, scenario, persp, instr):
    suffix = "" if instr == "reasoning" else f"_{instr}"
    ref = require(REF / "webppl_inputs" / model / f"{scenario}_results_{persp}{suffix}.json",
                  "the original pipeline wrote no file for this condition")
    key = ["Subject", "rec", "val"]
    cols = key + ["rec_vr", "val_vr", "value"]

    def rows(df):  # row order differs between converter versions and affects no statistic
        known = KNOWN_INPUT_DIFFS.get((model, scenario, persp, instr), set())
        df = df[[k not in known for k in zip(df.Subject, df.rec, df.val)]]
        return df[cols].sort_values(key).reset_index(drop=True)

    expected = pd.DataFrame(read_json(ref))
    got = convert.trials(model, scenario, persp, instr)
    assert len(got) == len(expected)
    pd.testing.assert_frame_equal(rows(got), rows(expected), check_dtype=False)


@pytest.mark.parametrize("csv", sorted((REF / "best_fits").glob("*/*.csv")),
                         ids=lambda p: f"{p.parent.name}/{p.stem}" if hasattr(p, "parent") else str(p))
def test_bayesian_fit_matches_original(csv):
    ref = pd.read_csv(csv)
    fit = bayes.best_fit(ref[["val_vr", "rec_vr", "value"]])
    assert (fit.params["alphaL0"], fit.params["alphaS"]) == (ref.alphaL0.iloc[0], ref.alphaS.iloc[0])
    assert round(fit.params["w"], 6) == round(ref.w.iloc[0], 6)
    np.testing.assert_allclose(fit.rows["prob"].to_numpy(), ref[PROB].to_numpy(), rtol=1e-12)
    assert abs(fit.r - np.corrcoef(ref.value, ref[PROB])[0, 1]) < 1e-10


def test_stored_scores_are_reproduced_by_a_parser():
    """Every stored score equals max(parse(last reply), 0) under the current or the first
    version of the original parser (older files were scored with the first version)."""
    total = mismatch = 0
    for _, trial in audit.scored_trials():
        total += 1
        reply = trial["messages"][-1]["content"]
        candidates = set()
        for parser in (parse_exp2_score, parse_exp2_score_v1):
            try:
                candidates.add(max(parser(reply), 0))
            except (ValueError, IndexError):
                pass
        mismatch += trial["score"] not in candidates
    assert total > 0
    # The remainder were typed in by hand when no parser could read the reply.
    assert mismatch / total < 0.005, f"{mismatch}/{total} stored scores not reproduced"


# ------------------------------------------------------------------ the paper's tables
# Rows the committed data does not reproduce (details: docs/PROVENANCE.md, section 2). They are
# strict xfails, so a row that starts to match fails the suite and gets noticed.
_FITS = "committed best fits (added 2025-07-31) do not give the printed Bayesian columns; LLM-Human matches"
_RESPONSES = "committed responses (added 2025-07-31) do not give the printed numbers"
_OCT = "data re-committed 2025-10-22 ('additional results'), after the table was made"


def _every_condition(model, reason):
    return {(model, p, i): reason for i, p in conditions(model_spec("exp2", model))}


KNOWN_MISMATCH = {
    **_every_condition("gemini-2.0-flash", _FITS),
    **_every_condition("gemma-3-4b", _FITS),
    **_every_condition("llama-3.2-3b", _RESPONSES),
    ("o1", "first", "reasoning"): _OCT + "; LLM-Human 0.865 vs printed 0.909",
    ("o3-mini", "user", "reasoning"): _OCT + "; Bayesian-LLM 0.669 vs printed 0.667",
    **_every_condition("deepseek-r1", _OCT),
}
MAIN_KNOWN_MISMATCH = {
    **{m: "averages rows listed in KNOWN_MISMATCH"
       for m in ("gemini-2.0-flash", "gemma-3-4b", "llama-3.2-3b", "o1", "deepseek-r1")},
    "o3-mini": "printed LLM-Human 0.712 is not the mean of its own printed rows (0.772, 0.667 -> 0.720)",
}
INTRA_KNOWN_MISMATCH = {
    **_every_condition("deepseek-r1", _OCT),
    ("o3-mini", "user", "reasoning"): _OCT,
    ("o1", "user", "reasoning"): "0.936 vs printed 0.947, although this condition's other columns match",
    **{("gemini-2.0-flash", p, i): "differs from the printed value, although LLM-Human matches"
       for p, i in (("user", "direct"), ("first", "direct"), ("first", "cot"))},
}


def _cases(targets, known, id_fn):
    for key in targets:
        marks = [pytest.mark.xfail(reason=known[key], strict=True)] if key in known else []
        yield pytest.param(key, marks=marks, id=id_fn(key))


def _close(got, want):
    return all(abs(g - w) <= TOL for g, w in zip(got, want))  # NaN never compares close


@pytest.fixture(scope="module")
def per_condition():
    return analysis.all_conditions()


def _condition_row(table, model, perspective, instructions):
    return select(table, model=label(model), perspective=perspective, instructions=instructions)


@pytest.mark.slow
@pytest.mark.parametrize("key", _cases(EXP2_FULL, KNOWN_MISMATCH, "-".join))
def test_full_table(per_condition, key):
    row = _condition_row(tables.table_full(per_condition), *key)
    got = tuple(row[c] for c in COLS)
    assert _close(got, EXP2_FULL[key]), f"{key}: got {np.round(got, 4)}, paper {EXP2_FULL[key]}"


@pytest.mark.slow
@pytest.mark.parametrize("model", _cases(EXP2_MAIN, MAIN_KNOWN_MISMATCH, str))
def test_main_table(per_condition, model):
    row = select(tables.table_main(per_condition), model=label(model))
    got = tuple(row[c] for c in COLS)
    assert _close(got, EXP2_MAIN[model]), f"{model}: got {np.round(got, 4)}, paper {EXP2_MAIN[model]}"


@pytest.mark.slow
@pytest.mark.parametrize("key", _cases(EXP2_INTRA, INTRA_KNOWN_MISMATCH, "-".join))
def test_intra_character_table(per_condition, key):
    got = _condition_row(tables.table_alignment(per_condition), *key)["intra_character"]
    assert _close([got], [EXP2_INTRA[key]]), f"{key}: got {got:.4f}, paper {EXP2_INTRA[key]}"
