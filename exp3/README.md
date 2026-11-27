# Experiment 3: vigilance in the wild (sponsored YouTube segments)

LLMs read real sponsored segments of YouTube videos (from SponsorBlock, with brand names censored) and
rate how much the YouTuber cares about the viewer (trust), how much the sponsorship matters to the
YouTuber (reward quality) and how good the product is (offer quality). Optional steering sentences ask the
model to consider the speaker's motives. As in Experiment 2, the Bayesian model
(`common/motivated_model.wppl`) predicts the quality rating from the other two; the reported number is the
best-fit correlation across videos.

| File | What it does |
|---|---|
| `prompts.py` | renders the questions from `prompts.yaml` |
| `collect.py` | one query per video and rating (`vigilance exp3 collect`) |
| `convert.py` | responses → the Bayesian model's inputs |
| `analysis.py` | the paper's tables (`vigilance exp3 tables`) |
| `dataset.py` | how the video sample was built: SponsorBlock → YouTube API → GPT-4o check and censoring |
| `prompts.yaml` | every rating prompt, and the steering sentences |
| `dataset_prompts.yaml` | the two prompts used to build the video sample |
| `models.yaml` | which models were run, and their settings |

## Reproduce

```bash
vigilance exp3 tables      # outputs/: table_steering.csv, table_additional_steers.csv, table_length.csv
```

## data/

* `videos_censored.json`: the video sample. Per video: `video_id`, `video_title`, `video_channel_name`,
  `video_channel_description`, and `video_sponsorship_transcript` with brand names censored by GPT-4o.
* `responses/{steer}/{model}/{purpose}_{perspective}_{instructions}.json`, with steer ∈ none, incentives,
  bias, gricean. Each entry is the video record plus `model_response`, `flipped`, `score`, `flipped_score`.
* `reference/`: outputs of the ORIGINAL pipeline (`webppl_inputs/`), regression targets for
  `tests/test_exp3.py`.
* Not included: the SponsorBlock database dump (`sponsorTimes.csv`, ~5 GB, downloadable from
  sponsor.ajay.app), needed only to re-sample videos with `dataset.py`.
