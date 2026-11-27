"""Command line, organised like the folders: `vigilance <exp1|exp2|exp3> <action>`, plus
`vigilance reproduce`. Without installing, run `python -m common.cli ...` from the repository root.

Analyses run offline from each experiment's data/ and write to its outputs/. Collection commands
call model APIs (keys in .env).
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import load_prompts, model_spec, models, output_dir
from .design import INSTRUCTIONS, PERSPECTIVES, PURPOSES

RMD_HINT = ("Exp 1 figures, t-tests and Stan model: "
            "Rscript -e \"rmarkdown::render('exp1/analysis.Rmd', output_dir = 'exp1/outputs')\"")


def _save(df: pd.DataFrame, name: str, experiment: str, show: bool = True) -> None:
    out = output_dir(experiment)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv"
    df.to_csv(path, index=False)
    if show:
        with pd.option_context("display.width", 200, "display.max_rows", 500, "display.float_format", "{:.4f}".format):
            print(df.to_string(index=False))
    print(f"\n-> {path}")


# ---------------------------------------------------------------- analyses (offline)
def cmd_posteriors(_):
    from . import bayes
    print("wrote", bayes.build_posterior_table())


def cmd_exp1_export(_):
    from exp1 import analysis, export
    for model in models("exp1"):
        for instructions in model_spec("exp1", model)["instructions"]:
            print("wrote", export.write_player_two(model, instructions, output_dir("exp1")))
    _save(analysis.shifts()[analysis.SHIFT_COLUMNS], "shifts", "exp1", show=False)
    _save(analysis.mean_shifts(), "mean_shifts", "exp1")


def cmd_exp2_tables(_):
    from exp2 import analysis, tables
    per_condition = analysis.all_conditions()
    _save(per_condition, "per_condition", "exp2")
    for name, table in (("full", tables.table_full), ("main", tables.table_main), ("alignment", tables.table_alignment)):
        _save(table(per_condition), f"table_{name}", "exp2")


def cmd_exp2_audit(_):
    from exp2 import audit
    _save(audit.audit(), "score_audit", "exp2")


def cmd_exp3_tables(_):
    from exp3 import analysis
    for name, table in (("steering", analysis.steering_table), ("additional_steers", analysis.additional_steers_table),
                        ("length", analysis.length_table)):
        _save(table(), f"table_{name}", "exp3")


def cmd_reproduce(args):
    for step in (cmd_posteriors, cmd_exp2_tables, cmd_exp3_tables, cmd_exp1_export):
        step(args)
    print("\n" + RMD_HINT)


# ---------------------------------------------------------------- collection (calls APIs)
def cmd_exp1_collect(args):
    from exp1 import game
    print(game.collect(args.model, args.instructions, args.payoff, n=args.n, seed=args.seed))


def cmd_exp2_collect(args):
    from exp2 import collect
    if args.all:
        collect.collect_all(args.model, n=args.n, seed=args.seed)
    else:
        print(collect.collect(args.model, args.scenario, args.perspective, args.purpose, args.instructions,
                              n=args.n, seed=args.seed))


def cmd_exp3_collect(args):
    from exp3 import collect
    path = collect.collect(args.model, args.steer, args.perspective, args.purpose, args.instructions, seed=args.seed)
    collect.score(path)
    print(path)


# ---------------------------------------------------------------- parser
def _formats(experiment: str) -> list[str]:
    """Answer formats that some model in the experiment was run with."""
    used = {i for m in models(experiment) for i in model_spec(experiment, m)["instructions"]}
    return [i for i in INSTRUCTIONS if i in used]


def _parser() -> argparse.ArgumentParser:
    from exp2.prompts import scenarios
    from exp3.prompts import steers

    p = argparse.ArgumentParser(prog="vigilance", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    top = p.add_subparsers(dest="command", required=True)

    def command(group, name, fn, help, **options):
        sp = group.add_parser(name, help=help)
        for flag, spec in options.items():
            sp.add_argument(f"--{flag}", **spec)
        sp.set_defaults(fn=fn)

    def experiment(name, help):
        return top.add_parser(name, help=help).add_subparsers(dest="action", required=True)

    def choice(values, required=True, **kw):
        return {"choices": list(values), "required": required, **kw}

    seed = {"type": int, "help": "make the order of questions and scale flips reproducible"}

    command(top, "reproduce", cmd_reproduce, "rebuild every paper table from the experiments' data (no API keys needed)")
    command(top, "posteriors", cmd_posteriors, "run the Bayesian model once -> common/outputs/posterior_table.json")

    e1 = experiment("exp1", "Experiment 1: vigilance game")
    command(e1, "export", cmd_exp1_export, "Player-2 CSVs and shifts for analysis.Rmd, mean shifts")
    command(e1, "collect", cmd_exp1_collect, "play games (calls APIs)",
            model=choice(models("exp1")), instructions=choice(_formats("exp1")),
            payoff=choice(load_prompts("exp1")["standard"]["payoff"]), n={"type": int, "default": 30}, seed=seed)

    e2 = experiment("exp2", "Experiment 2: Oktar et al. vignettes")
    command(e2, "tables", cmd_exp2_tables, "the paper's tables")
    command(e2, "audit", cmd_exp2_audit, "stored scores vs. the ratings replies state explicitly")
    command(e2, "collect", cmd_exp2_collect, "collect responses (calls APIs)",
            model=choice(models("exp2")), all={"action": "store_true", "help": "every scenario/perspective/purpose/format"},
            scenario=choice(scenarios(), False), perspective=choice(PERSPECTIVES, False),
            purpose=choice(PURPOSES, False), instructions=choice(_formats("exp2"), False), n={"type": int}, seed=seed)

    e3 = experiment("exp3", "Experiment 3: YouTube sponsorships")
    command(e3, "tables", cmd_exp3_tables, "the paper's tables")
    command(e3, "collect", cmd_exp3_collect, "collect responses (calls APIs)",
            model=choice(models("exp3")), steer=choice(steers(), False, default="none"),
            perspective=choice(PERSPECTIVES), purpose=choice(PURPOSES), instructions=choice(_formats("exp3")), seed=seed)
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
