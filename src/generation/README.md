# Generation pipelines

## MedSP1000 multi-turn generation

Build the frozen 200-question set directly from the pinned MedSP1000 source:

```bash
uv run python scripts/prepare_medsp1000_questions.py
```

Run one end-to-end smoke question or the full set. Mistral runs on Modal; the
selected clinician is routed to Modal or its provider API:

```bash
uv run modal run src/generation/modal_medsp1000.py --smoke-test

uv run modal run src/generation/modal_medsp1000.py \
  --clinician-model claude-sonnet-5 \
  --num-questions 200
```

The runner generates 8 conversations at a time by default. As soon as a batch
finishes, each complete conversation is appended and synced individually, then
the run manifest is updated with progress. If a run is interrupted, the next
invocation skips every durably saved conversation. Adjust the tradeoff between
inference batching and potential rework with
`--checkpoint-batch-size` (use `1` for per-conversation checkpoints).
Provider, truncation, and batch-validation failures are also appended as failed
attempt rows, including any turns completed before the failure. They remain
eligible for a later retry and advance the attempt counter on the next run.

The patient simulator is fixed to
`mistralai/Mistral-Small-3.1-24B-Instruct-2503` on Modal. The clinician is an
experimental variable selected with `--clinician-model`; OpenAI, Anthropic,
Gemini, and the two pinned Qwen/Modal models use the same conversation and
checkpoint pipeline. Role contexts remain separate throughout generation.
Prior visible messages are kept clean; patient grounding and clinician turn
controls are attached only to the current message. Embedded role context hashes
are verified before inference. Outputs are append-only and resumable by
question, model pair, prompt version, exchange count, output caps, sampling
configuration, and seed at
`data/outputs/medsp1000/generations.jsonl`; each run writes a separate manifest
beside it. One JSONL row contains one complete conversation attempt, with
turn-level model, token, latency, and finish metadata.
Failures are isolated per conversation within each batch, so a truncated or
failed response does not discard successful neighbors. API clinician turns also
record provider request IDs and raw text-block counts and hashes when exposed by
the SDK. Exact repeated-message artifacts are flagged without rewriting the
provider's returned text.
Both question and output rows are validated against their committed JSON
Schemas. Output loading additionally verifies turn ordering, transcript
derivation, and token/latency aggregates before resume state is accepted.
API clinicians use an explicitly pinned medium reasoning effort. The
4,096-token clinician generation budget leaves room for hidden reasoning; the
prompt itself asks for concise natural messages without imposing a
response-length scoring target. Qwen reasoning remains disabled until a
comparable open-weight reasoning condition is selected for that cohort.

Judge complete MedSP1000 trajectories with blinded five-axis rubric scoring
and an independent model ranking:

```bash
uv run python -m judging.run_medsp1000_judging --dry-run
uv run python -m judging.run_medsp1000_judging
```

The runner treats each full clinician-patient transcript as one candidate,
scores only clinician behavior, and writes append-only, resumable outputs under
`data/outputs/medsp1000/judgements/`.

To judge a prefix of each trajectory instead, pass an even role-turn count no
greater than the generated trajectory length. Prefix views have distinct
prompt-template IDs, resume keys, manifests, and JSONLs, so they cannot collide
with full-trajectory judgments:

```bash
uv run python -m judging.run_medsp1000_judging \
  --view-turn-count 2
```

For example, the combined condition writes 2-, 4-, and 6-turn views to
`rubric_and_model_ranking_2_turns.jsonl`,
`rubric_and_model_ranking_4_turns.jsonl`, and
`rubric_and_model_ranking_6_turns.jsonl`, respectively. Each judge sees only
the requested prefix and is explicitly told to evaluate the visible
interaction.

## Real-POCQi single-turn generation

The Real-POCQi pipeline deterministically shuffles the committed 620-question
artifact with seed 42 and asks each requested model to answer as the question's
specialty expert. Use the question-count option only for an intentional sample.

```bash
uv run python -m generation.generate_real_pocqi \
  --models gpt-5.6-sol gpt-5.6-terra claude-opus-5 claude-sonnet-5
```

The CLI loads provider credentials from the repository's ignored `.env` file
without replacing variables already exported in the shell. Sampling temperature
is omitted by default because the primary OpenAI and Anthropic cohort endpoints
do not accept it. Use `--temperature` only with a compatible model. The default
output cap is 4,096 tokens; responses that still hit the cap are recorded as
failures instead of being mistaken for complete answers.

Mixed-model runs queue models round-robin for each question, allowing providers
to generate concurrently within the configured worker limit.

Qwen and other Modal-hosted models also require their deployed app:

```bash
uv run python -m generation.generate_real_pocqi \
  --models modal/Qwen3.6-35B \
  --modal-app medical-llm-inference
```

The pinned Qwen cohort has a dedicated batched Modal runner. It serves the
official FP8 checkpoints for Qwen3.8-27B on one H100 and
Qwen3.5-122B-A10B on two H100s, with thinking disabled:

```bash
uv run modal run src/generation/modal_real_pocqi_generation.py
```

The dedicated runner defaults to the full 620-question cohort and a 2,048-token
output cap. To upgrade previously capped attempts while preserving completed
answers, pass `--regenerate-truncated`; only attempts truncated under a smaller
cap are regenerated.

Attempt records are appended immediately to
`data/outputs/generations/real_pocqi_generations.jsonl`. Each failed retry is
retained. Later invocations skip successful question/model keys by default, so
the command can safely resume an interrupted run. Use `--force` only when a new
attempt is intentionally required.

Run `uv run python -m generation.generate_real_pocqi --help` for sampling,
specialty filtering, concurrency, retry, and output options.

### Identity-revealed judging follow-up

To compare blinded judgments with judgments where each candidate's generator
name is visible, reuse the existing generation artifact and run the combined
five-axis scoring plus independent-ranking condition on a seeded 200-question
subset:

```bash
uv run python -m judging.run_real_pocqi_judging \
  --env-file .env \
  --reveal-generator-identities \
  --judging-cases rubric_and_model_ranking \
  --num-questions 200 \
  --question-sample-seed 42 \
  --max-output-tokens 8192 \
  --dry-run

uv run python -m judging.run_real_pocqi_judging \
  --env-file .env \
  --reveal-generator-identities \
  --judging-cases rubric_and_model_ranking \
  --num-questions 200 \
  --question-sample-seed 42 \
  --max-output-tokens 8192
```

The revealed mode defaults to the same 200-question sample and seed when those
two flags are omitted. It writes judgments to
`data/real_pcoqi/judgements/identity_revealed_rubric_and_model_ranking.jsonl`,
leaving the blinded JSONL untouched. The manifest freezes the selected question
IDs, sample seed, output path, and identity condition. Use those question IDs to
filter the completed blinded sweep for paired analysis.

### Identity-blind score-only follow-up

The score-only condition reuses the same seed-42 sample of 200 questions, but
shows the judge exactly one saved generation per call. It requests only the
same five Real-POCQi rubric scores; neither the prompt nor structured output
contains a ranking. With eight generators and eight judges, the complete
matrix contains 12,800 individually scored answers.

Run the six API judges with:

```bash
uv run python -m judging.score_only_real_pocqi --dry-run
uv run python -m judging.score_only_real_pocqi
```

Run the two Qwen judges through the batched Modal deployment with:

```bash
uv run modal run src/judging/modal_real_pocqi_judging.py \
  --score-only \
  --num-questions 200 \
  --question-sample-seed 42 \
  --max-output-tokens 1024
```

Both commands append to the resumable
`data/real_pcoqi/judgements/score_only.jsonl` artifact under the
`real_pocqi_score_only_random200_v1` experiment ID. Once all cells complete,
construct tie-aware score rankings and compare them with the existing explicit
answer-selection and combined rubric-sum rankings:

```bash
uv run python scripts/analyze_score_only_real_pocqi.py
```

The reproducible Markdown report and machine-readable statistics are written
under `data/analysis/score_only/`.
