# Medical LLM self-preference

## Overview

LLMs are increasingly used as scalable evaluators of model-generated medical
content. This creates a potential circularity problem: a judge may prefer the
style, reasoning conventions, or outputs of its own model family independently
of clinical quality.

This project evaluates that risk in two settings:

- **Real-POCQi:** single-turn answers to 620 real-world specialist clinical
  questions spanning 30 specialties.
- **MedSP1000:** multi-turn clinician interactions in 200 standardized-patient
  scenarios, evaluated after 2, 4, 6, and 8 visible turns.

The primary analysis compares how a model ranks its own answer with how other
judges rank that exact same answer. This matched design holds answer content and
length fixed. Candidate order is randomized deterministically and adjusted for
in the analysis. Negative rank differences indicate that the generating
model's own judge placed its answer closer to first place.

Useful entry points:

- [Manuscript PDF](output/pdf/med_self_preference_updated.pdf)
- [Experiment details](docs/experiment.md)
- [Documentation map](docs/README.md)
- [Static results explorer](leaderboard/index.html)
- [Derived analysis definitions](data/analysis/self_preference/README.md)

## Main findings

- In the blinded Real-POCQi condition, all eight judges ranked their own exact
  answers more favorably than outside judges did. The pooled matched effect was
  **−1.219 rank positions** (95% CI −1.261 to −1.177).
- Judge identity mattered substantially: Real-POCQi effects ranged from
  **−0.324 to −2.939 positions**. Clinical specialty explained only 5.7% of
  question-level variation, with no significant global specialty
  heterogeneity (`p=0.190`).
- Raw score inflation, own-pick rate, and matched rank preference were not
  interchangeable. Weak generators could rarely win overall while still being
  promoted by their own judges relative to outside judges.
- Removing the explicit rubric preserved the dominant Real-POCQi result:
  direct and rubric-assisted rankings agreed on 91.5% of candidate pairs.
- MedSP1000 showed matched self-preference at every transcript length. Pooled
  effects were **−0.518, −0.457, −0.502, and −0.613 positions** at 2, 4, 6,
  and 8 turns, respectively; the relationship with conversation length was not
  monotonic.
- Revealing generator identities did not amplify the effect. On matched
  Real-POCQi cells, self-preference was modestly weaker with names visible
  (revealed minus blinded: **+0.123 positions**, 95% CI +0.059 to +0.188).
- Longer answers tended to receive better rankings, but answer length cannot
  explain away the primary matched contrast because the exact response is held
  fixed between own and outside judges.

These findings support using multi-family judge panels, matched estimands,
explicit position controls, and separate reporting of scores and rankings
instead of relying on a single LLM judge or raw win rate.

## Study design

| Component | Real-POCQi | MedSP1000 |
|---|---:|---:|
| Frozen inputs | 620 questions | 200 scenarios |
| Candidate generators | 8 | 6 clinician models |
| Judges | 8 | 6 |
| Primary candidate set | 8 answers | 6 trajectories |
| Evaluation views | Blinded, direct ranking, identity revealed | 2, 4, 6, and 8 turns |
| Saved judgments | 6,960 | 4,800 |

The repository contains **11,760 judgments**:
4,960 blinded Real-POCQi rubric-plus-ranking judgments, 800 direct rankings,
1,200 identity-revealed judgments, and 4,800 MedSP1000 judgments across four
conversation lengths.

### Model cohorts

The complete Real-POCQi cohort contains two models from each of four families:

| Family | Models |
|---|---|
| OpenAI | `gpt-5.6-sol`, `gpt-5.6-terra` |
| Anthropic | `claude-opus-5`, `claude-sonnet-5` |
| Google | `gemini-3.1-pro-preview`, `gemini-3.7-flash` |
| Qwen | `Qwen/Qwen3.5-122B-A10B-FP8`, `Qwen/Qwen3.8-27B-FP8` |

MedSP1000 uses the six API models as clinician generators and judges. The
patient simulator is fixed to
`mistralai/Mistral-Small-3.1-24B-Instruct-2503`. Qwen clinician and judge runs
are not included in the current MedSP1000 experiments.

The repository also preserves results from an earlier four-model phase. Its
raw generations and judgments are not available here, so those results cannot
be reproduced from the repository. The experiments described above are the
basis for the current conclusions.

## Reproduce the analyses

### Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Git LFS for the large committed JSONL artifacts

Clone the repository and materialize the LFS objects before running analyses:

```bash
git lfs install
git lfs pull
uv sync --dev
```

The committed raw outputs are sufficient to regenerate the derived
self-preference tables without provider credentials or new model calls:

```bash
uv run python scripts/create_self_preference_scores.py
uv run python scripts/analyze_new_self_preference.py
uv run python scripts/analyze_token_length_confounding.py
```

Run the test suite with:

```bash
uv run pytest -q
```

The analysis scripts select the last successful attempt for each logical
question–judge cell and filter on the production experiment identifier. This
preserves the complete append-only audit trail while preventing smoke runs,
failed attempts, and retries from being counted as independent observations.

## Rerun generation or judging

Reproducing the saved analyses does not require API access. Generating new
answers or judgments does require credentials for the selected providers:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- Modal authentication for the pinned open-weight inference deployments

Credentials may be exported in the environment or placed in an ignored `.env`
file. Never commit credential files. Full experiment reruns can incur
substantial inference cost and may not reproduce provider outputs byte for byte
even when prompts and sampling settings are held fixed.

See [the generation and judging guide](src/generation/README.md) for commands,
resume behavior, model routing, and Modal entry points. The shared inference
interface is documented in [src/inference/README.md](src/inference/README.md).

## Data and artifact map

| Path | Contents |
|---|---|
| `data/question_sets/` | Frozen Real-POCQi and MedSP1000 inputs, schemas, hashes, and source revisions |
| `data/outputs/generations/` | Append-only Real-POCQi model generations and run manifests |
| `data/outputs/medsp1000/` | Append-only clinician-patient trajectories and judgments at four transcript lengths |
| `data/real_pcoqi/judgements/` | Real-POCQi blinded, direct-ranking, and identity-revealed judgments |
| `data/analysis/self_preference/` | Reproducible model-, pairwise-, question-, and token-length analysis artifacts |
| `data/deprecated/` | Historical inputs, standalone pilots, and smoke outputs retained for provenance |
| `docs/experiment.md` | Detailed methods, counts, results, limitations, and available files |
| `docs/latex/med_self_preference/` | LaTeX manuscript source |
| `leaderboard/` | Build-free static results explorer |

The `real_pcoqi` directory spelling is retained for compatibility with the
recorded production paths.

Active question sets are immutable JSONL artifacts paired with provenance
manifests. The Real-POCQi set is frozen from source revision
`9002e1ddff506d354f1b7becc1213b96299d07f6`; the MedSP1000 cohort is selected
deterministically with seed 42 from revision
`55e3e55efd08c73baab912ba0c5b42637114fbc8`.

Downloaded upstream snapshots under `data/source/` are ignored because they
are re-downloadable and may contain nested repositories. Historical artifacts
outside the current experiments are labeled and kept under `data/deprecated/`.

## Browse the results locally

The static explorer has no build step or JavaScript dependency:

```bash
python -m http.server 8000 --directory leaderboard
```

Then open `http://localhost:8000`.

## Repository structure

```text
.
├── data/                 Frozen inputs, outputs, judgments, and analyses
├── docs/                 Experiment record, manuscript, and focused QA notes
├── scripts/              Question preparation and reproducible analyses
├── src/
│   ├── generation/       Real-POCQi and MedSP1000 generation pipelines
│   ├── inference/        Shared provider interface
│   ├── judging/          Blinded and identity-revealed judge pipelines
│   └── leaderboard/      Static results explorer
└── tests/                Unit and end-to-end pipeline tests
```

Operational formats are documented alongside their artifacts:

- [Question-set contracts](data/question_sets/README.md)
- [Saved generation format](data/outputs/README.md)
- [Analysis definitions](data/analysis/self_preference/README.md)
- [MedSP1000 information boundaries](docs/medsp1000_generation.md)
- [Real-POCQi generation spot checks](docs/real_pocqi_generation_spot_checks.md)

## Interpretation and limitations

- Automated judges have not yet been calibrated against blinded physician
  judgments for the expanded conditions.
- The study uses one saved generation per model-question cell. Repeated
  generations are needed to separate stable family affinity from
  answer-specific variation.
- Absolute score calibration differs markedly across judges, especially for
  open-weight models; raw scores should not be pooled without normalization.
- Response length is strongly associated with rankings. The length analysis is
  observational and cannot distinguish verbosity preference from genuine
  completeness or quality.
- Some specialty strata are small, limiting subgroup power.
- The identity-revealed comparison contains six API judges and should not be
  generalized to the two Qwen judges.
- Current MedSP1000 conclusions cover six API clinician models, not the full
  eight-model Real-POCQi cohort.
- Model outputs and judgments may contain factual errors, unsafe medical
  content, or provider-specific artifacts. They are not clinical
  recommendations.

For full methods, uncertainty estimates, specialty results, prompt-development
history, and planned extensions, see [docs/experiment.md](docs/experiment.md).

## Citation

To cite this repository, use the entry below or the machine-readable
[`CITATION.cff`](CITATION.cff):

```bibtex
@misc{ansari2026medicalselfpreference,
  title  = {Medical LLM Self-Preference},
  author = {Ansari, Zara and Natarajan, Shlok and Fanous, Aaron and Daneshjou, Roxana},
  year   = {2026},
  note   = {Code, model outputs, and analyses}
}
```

## Licensing and responsible use

Repository-authored code and documentation are released under the
[MIT License](LICENSE). Frozen benchmark inputs retain their upstream terms;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dataset licenses,
citations, source revisions, and the transformations made in this repository.
Saved model generations and judgments may also be subject to the applicable
model providers' terms.

This work is intended for research and evaluation. It does not provide medical
advice, and the saved model outputs should not be used for patient care.
