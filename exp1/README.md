# Experiment 1: deliberate vs. incidental information (vigilance game)

Watson & Morgan's vigilance game, played by vision LLMs (GPT-4o and Claude 3.5 Sonnet) in both roles.
Player 1 sees easier images of blue and yellow circles, answers, and leaves advice that need not be
honest. Player 2 sees harder versions of the same images, guesses, is then shown either Player 1's
actual guess ("spied") or the advice, and may revise. The result is how far Player 2 moves toward
Player 1's number: shift = (second answer − first answer) / (Player 1's number − first answer).

| File | What it does |
|---|---|
| `game.py` | plays 20-round games with a model (`vigilance exp1 collect`) |
| `export.py` | game results → Watson & Morgan's CSV format |
| `analysis.py` | the shift statistic (`vigilance exp1 export`) |
| `analysis.Rmd` | figures, t-tests and the Stan model; reads `outputs/shifts.csv` |
| `stimuli.py` | how the circle images were generated |
| `watson_morgan_original_analysis.Rmd` | Watson & Morgan's own analysis, kept verbatim |
| `prompts.yaml` | every prompt the models saw |
| `models.yaml` | which models were run, and their settings |

## Reproduce

```bash
vigilance exp1 export      # outputs/: Player_two_*.csv, shifts.csv, mean_shifts.csv
Rscript -e "rmarkdown::render('exp1/analysis.Rmd', output_dir = 'exp1/outputs')"   # outputs/figures/, analysis.html
```

## data/

* `stimuli/player_1/`, `stimuli/player_2/`: the 20 image pairs (`{yellow}_yellow_{blue}_blue.png`;
  player-2 images have 2 more circles of each colour, so the answers match). They were generated once,
  without a seed, so they are data, not something to regenerate.
* `responses/{model}_{direct|cot}_{cooperative|competitive}.json`: 30 games each. A game is a list of 20
  rounds (`correct_answer`, `player_1_guess`, `player_1_advice`, `player_1_advice_message`,
  `player_2_guess_before`, `info_provided` ∈ {spied, advice}, `player_2_guess_after`, `player_2_update_message`).
* `human/watson_morgan_player_two.csv`: Watson & Morgan (2025) Player-2 trials in the Cooperative and
  Competitive conditions, questions ≤ 10, filtered from their raw `Player_two.csv`
  (`export.clean_human_player_two`). Their raw files (`Player_one.csv`, `Player_two.csv`,
  `Demographic.csv`, `Qualitative analysis.xlsx`) are not redistributed; they are needed only to
  re-derive this file or to run `watson_morgan_original_analysis.Rmd`.
* `diverse/`: the prompt variation of Appendix A.5 (30 colour pairs, mixed shapes, 10 images per game):
  `stimuli/variation_XX/`, `responses/{model}_{prompt}_{payoff}/` (one file per variation) and
  `responses_combined/` (the per-condition concatenation used by the analysis).
* `reference/`: outputs of the ORIGINAL pipeline: the Player-2 CSV exports (regression targets for
  `tests/test_exp1.py`) and `figures/` as the original notebooks drew them; `diverse/` holds the same
  for the variation.
