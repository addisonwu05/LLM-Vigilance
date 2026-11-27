"""Experiment 3: prompts and collection, each pipeline stage vs. the original pipeline's stored
outputs (exp3/data/reference/), and the paper's tables.

exp3_prompts.json was captured by running the ORIGINAL scripts with a fake LLM; exp3_replays.json is a
characterization golden captured from this package before its deduplication pass.
"""
import hashlib

import pandas as pd
import pytest
from conftest import RequestHasher, load_golden, require, runs, select, sha256
from paper_values import EXP3_ADDITIONAL, EXP3_LENGTH, EXP3_STEERING

from common.config import model_spec
from common.design import PERSPECTIVES, PURPOSES
from common.files import read_json
from common.parsing import parse_exp3_score
from exp3 import DATA, analysis, collect, convert, prompts

RUNS = runs("exp3", prompts.steers(), PERSPECTIVES)


# ------------------------------------------------------------------ prompts and collection
def test_every_prompt_matches_original():
    golden = load_golden("exp3_prompts.json")
    require(collect.videos_path())
    videos = collect.load_videos()[: golden["n_videos"]]
    bad = [g for g in golden["prompts"]
           if sha256(prompts.question(g["perspective"], g["purpose"], videos[g["video_index"]], g["flip"],
                                      g["instructions"], g["steer"])) != g["sha256"]]
    assert not bad, f"{len(bad)} prompts differ, e.g. {bad[0]}"


def test_seeded_collection_replays(tmp_path):
    g = load_golden("exp3_replays.json")
    videos = collect.load_videos()[: g["n_videos"]]
    fake, saved = RequestHasher(), hashlib.sha256()
    for i, (model, steer, perspective, purpose, instructions) in enumerate(g["cases"]):
        out = collect.collect(model, steer, perspective, purpose, instructions, videos=videos, seed=g["seed"],
                              out=tmp_path / f"{i}.json", chat=fake)
        saved.update(out.read_bytes())
        collect.score(out)
        saved.update(out.read_bytes())
    assert (fake.n, fake.hexdigest()) == (g["n_requests"], g["requests_sha256"])
    assert saved.hexdigest() == g["outputs_sha256"]


# ------------------------------------------------------------------ pipeline vs. the original pipeline
@pytest.mark.parametrize("model,steer,persp,instr", RUNS)
def test_model_inputs_match_original(model, steer, persp, instr):
    legacy = model_spec("exp3", model)["legacy_name"]
    ref = require(DATA / "reference" / "webppl_inputs" / steer / model / f"{legacy}_{persp}_{instr}.json",
                  "condition not run")
    expected = pd.DataFrame(read_json(ref))[convert.COLUMNS]
    got = convert.trials(model, steer, persp, instr)
    pd.testing.assert_frame_equal(got.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)


@pytest.mark.parametrize("model,steer,persp,instr", RUNS)
def test_stored_scores_are_reproduced(model, steer, persp, instr):
    for purpose in PURPOSES:
        path = require(collect.response_path(steer, model, purpose, persp, instr), "condition not run")
        for entry in read_json(path):
            assert parse_exp3_score(entry["model_response"]) == entry["score"]


# ------------------------------------------------------------------ the paper's tables
def test_steering_table():
    t = analysis.steering_table()
    for (model, instr, persp), (d, s) in EXP3_STEERING.items():
        row = select(t, model=model, instructions=instr, perspective=persp)
        assert round(row["default"], 3) == d and round(row["incentives"], 3) == s, (model, instr, persp)


def test_additional_steers_table():
    t = analysis.additional_steers_table()
    for (instr, persp), vals in EXP3_ADDITIONAL.items():
        row = select(t, instructions=instr, perspective=persp)
        got = tuple(round(row[s], 4) for s in analysis.ADDITIONAL_STEERS)
        assert got == vals, (instr, persp, got)


def test_length_table():
    t = analysis.length_table()
    for (model, persp, instr, q), want in EXP3_LENGTH.items():
        row = select(t, model=model, perspective=persp, instructions=instr, quartile=q)
        if abs(want) < 1e-12:   # printed values like -1.67e-16 are floating-point noise: r is undefined
            degenerate = row["constant_rating"] or row["constant_prediction"]
            assert degenerate and abs(row["r"]) < 1e-12, (model, persp, instr, q, row["r"])
        else:
            assert round(row["r"], 3) == want, (model, persp, instr, q, row["r"])
