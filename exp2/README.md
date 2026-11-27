# Experiment 2: nuanced vigilance in controlled vignettes (Oktar et al.)

LLMs rate the speakers in Oktar et al.'s three vignettes (credit card, medicine, real estate): how much a
speaker cares about the listener (trust), how much the incentive matters to the speaker (reward quality),
and how good the recommended option is (offer quality), for 4 characters × 4 incentive levels, from two
perspectives (the model as the listener, or answering for a user). The Bayesian model of motivated
communication (`common/motivated_model.wppl`) predicts the offer rating from the other two; the paper
correlates the LLM, the model and human participants across the 16 character × incentive cells.

| File | What it does |
|---|---|
| `prompts.py` | renders the questions from `prompts.yaml` |
| `collect.py` | runs the simulated subjects (`vigilance exp2 collect`) |
| `convert.py` | responses → the Bayesian model's inputs |
| `analysis.py` | fits the model per condition and computes the 16-cell correlations |
| `tables.py` | the paper's tables (`vigilance exp2 tables`) |
| `audit.py` | stored scores vs. the ratings replies state (`vigilance exp2 audit`) |
| `prompts.yaml` | every prompt; its header explains how questions are assembled |
| `models.yaml` | which models were run, and their settings (sample size, scale flips, token caps, ...) |

## Reproduce

```bash
vigilance exp2 tables      # outputs/: per_condition.csv, table_full.csv, table_main.csv, table_alignment.csv
```

## How the numbers are computed

**The Bayesian model** (`common/motivated_model.wppl`) is Oktar et al.'s `motivatedmodel.v2_1c_freelambda`.
For an LLM's ratings of a speaker it gives the posterior that the recommended option is good, given the
speaker's incentive (the LLM's 0–100 incentive rating rounded to {0, .25, …, 1}) and benevolence (its trust
rating rounded to {0, .1, …, 1}), over a grid of three parameters (αL0, αS ∈ {1, 4, 7}; w ∈ {0.0001, …, 1.0501}).
The posterior depends only on those inputs, so `vigilance posteriors` computes all 10,890 combinations once
(about 3 seconds) instead of once per trial. For each condition, `bayes.best_fit` picks the grid point that
maximises Pearson r between the LLM's quality rating (rescaled to 1–7) and the posterior across trials.

**The paper's correlations.** Per scenario, the LLM's mean quality rating and the model's mean posterior are
taken in each of the 16 character × incentive cells, and three correlations are computed across those cells:
Bayesian–LLM, Bayesian–Human and LLM–Human (human means: `data/human/oktar_cell_means.csv`). The three
scenarios are combined with a **Fisher-z mean** (tanh of the mean of atanh(r)). Table 1 is the plain mean of a
model's prompt-condition rows. Intra-character alignment is the Fisher-z mean of the 12 within-character
LLM–human correlations (4 characters × 3 scenarios). Trust ratings go onto the benevolence grid with the
rounding formula of the converter that produced each model's original inputs (`trust_rounding` in
`models.yaml`); the two formulas differ only for ratings 5, 45 and 65.

These rules were reconstructed from the analysis spreadsheet; `docs/PROVENANCE.md` (section 2) shows how
they were checked and lists the rows the committed data does not reproduce.

## data/

* `responses/{model}/{scenario}/{purpose}_{perspective}_{instructions}.json`, with scenario ∈ credit_card,
  medicine, real_estate; purpose ∈ trust, reward_quality, offer_quality; perspective ∈ first, user;
  instructions ∈ direct, cot (reasoning models: reasoning). A file is a list of simulated subjects; a
  subject is the list of trials of one dialogue, in the order asked: `person`, `reward` (absent for
  trust), `flip` (rating scale reversed), `messages` (the full dialogue up to and including that answer),
  `score` (parsed answer) and, in newer files, `flipped_score` (= score un-reversed).
* `human/oktar_cell_means.csv`: the mean human rating (1–7) of the recommended option for each scenario ×
  character × incentive cell, as entered in the paper's analysis spreadsheet ("LLM Vigilance Results").
  Credit card: ≈251 participants; medicine: ≈249; real estate: ≈251–252. The raw credit-card Qualtrics
  export (`data1a.csv`) is not redistributed: it includes participant IP addresses and locations.
* `prompt_variants/`: GPT-4o on the credit-card scenario with four paraphrased prompt sets (a robustness
  run, not reported in the paper).
* `reference/`: outputs of the ORIGINAL pipeline, kept as regression targets for `tests/test_exp2.py`:
  WebPPL input files (`webppl_inputs/`), R best-fit tables (`best_fits/`, CSV at full precision; the JSON
  copies were rounded to 4 decimals by `jsonlite`) and the aggregator's outputs
  (`best_fits_aggregated/`, `Bayesian_*_vigilance_results.csv`).
