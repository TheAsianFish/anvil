# Pilot: a local open model as the teacher (2026-10-21)

Before planning M8 I checked that a small open model can label the `make eval-classify`
samples on a laptop, how noisy it is against gold, and how fast. This is a pilot, not an
eval: it is not a RESULTS.md number and `make eval-classify TEACHER=ollama` (M8) replaces it.

## Method

Ollama 0.40.7 (the Linux tarball, served locally, 4 CPUs, no GPU), models `qwen2.5:3b` and
`qwen2.5:7b` as pulled on 2026-10-21 (q4 default tags). For each input: `POST /api/chat` with
the task's own system prompt from `anvil.teacher.claude.system_prompt` as the system message,
the input text as the user message, the task's JSON schema from
`anvil.teacher.claude.output_schema` as `format`, `temperature 0`, `num_ctx 2048`,
`num_predict 64`. Inputs: the first 60 items of the `make eval-classify` sample of each
dataset (`eval.data.sample` with the config's size and seed, so the same rows the eval
labels). The paraphrase pass uses `paraphrase_spec()` from `eval/classify/banking77.yaml`
(the ceiling prompt) on the same 60 inputs. Time is wall seconds per request, measured in
the client; the first request of a run pays the model load (19 s) and is included in the
mean, not the median.

The script was 40 lines outside the repository (it reads the eval config and sample through
the repo's own modules and posts to Ollama); the rows it wrote are next to this note, one
JSON object per input with the gold label, the teacher's label, seconds, and Ollama's
prompt and output token counts.

## Numbers

| Dataset | Model | Prompt | Agree with gold | Kappa vs gold | Invalid | Median s/label | Prompt tokens |
|---|---|---|---|---|---|---|---|
| banking77 | qwen2.5:3b | task | 33/60 | 0.54 | 0 | 1.7 | 505 |
| banking77 | qwen2.5:7b | task | 36/60 | 0.59 | 0 | 3.8 | 505 |
| banking77 | qwen2.5:3b | paraphrase (ceiling) | 36/60 | 0.59 | 0 | 1.8 | 504 |
| ag_news | qwen2.5:3b | task | 46/60 | 0.67 | 0 | 2.5 | 142 |

- Paraphrase self-agreement of `qwen2.5:3b` on Banking77: 47/60 (the two passes agree on
  the label whether or not it is gold).
- `qwen2.5:3b` and `qwen2.5:7b` agree with each other on 40/60 Banking77 inputs.
- Repeated 3B confusions on Banking77: `reverted_card_payment` labelled `cancel_transfer`
  (2), `automatic_top_up` labelled `top_up_by_card_charge` (2). The 7B model makes the first
  one too.
- Kappa here is Cohen's kappa on the 60 pairs (observed agreement against the agreement
  expected from the two label distributions); with 77 classes and 60 items the chance term
  is small, so kappa sits just under the raw agreement.
- No output ever failed the schema: Ollama's `format` constrains decoding to the enum.
- Throughput: the 505-token Banking77 system prompt is prefix-cached by the server, so a
  3B label costs about 1.7 s after the first request. Labelling the 2,700 inputs of one
  dataset (500 held-out, 2,000 pool, 200 ceiling) is about 1.5 h on 4 CPUs for 3B and about
  3 h for 7B.

## What it decides

- M8 uses `qwen2.5:3b` as its teacher: the 7B model buys 3 more agreements in 60 (within the
  noise of 60 items) for 2.2x the time per label, and the point of the milestone is a teacher
  that is wrong often enough to measure what the student does with its errors.
- The hypotheses in PLAN.md's evolution section (teacher-vs-gold kappa 0.50-0.60 on
  Banking77, a ceiling near 0.75, the student's confusion matrix repeating the teacher's
  pairs) are these numbers scaled to the full sample.
