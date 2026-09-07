# Real-POCQi score-only experiment

## Design

This identity-blind follow-up uses the seed-42 sample of **200 questions**. Each of 8 judges scored each of 8 saved generations in a separate prompt (12,800 logical judgments). A judge saw one answer only, and neither the prompt nor the structured output requested a ranking. Rankings below were constructed afterward from the five-axis score sum, with average ranks for ties.

For this report, **ASR** means the explicit answer-selection ranking from the matched existing rubric-and-ranking condition. The within-prompt rubric-sum ranking is reported separately as a second reference.

## Aggregate generator ranking

| Rank | Generator | Score-only mean rank | Score-only mean score / 25 | ASR mean rank | Combined rubric-sum mean rank |
|---:|---|---:|---:|---:|---:|
| 1 | `claude-opus-5` | 3.056 | 21.886 | 1.894 | 2.038 |
| 2 | `gpt-5.6-terra` | 3.419 | 21.687 | 3.393 | 3.357 |
| 3 | `gpt-5.6-sol` | 3.787 | 21.393 | 4.320 | 4.367 |
| 4 | `gemini-3.7-flash` | 4.268 | 20.582 | 3.904 | 3.905 |
| 5 | `claude-sonnet-5` | 4.442 | 20.524 | 4.266 | 4.156 |
| 6 | `gemini-3.1-pro-preview` | 4.560 | 20.368 | 4.606 | 4.604 |
| 7 | `Qwen/Qwen3.5-122B-A10B-FP8` | 6.031 | 18.358 | 6.584 | 6.587 |
| 8 | `Qwen/Qwen3.8-27B-FP8` | 6.438 | 17.389 | 7.034 | 6.987 |

## Agreement with existing rankings

| Comparison | Pair agreement (ties excluded) | Score-only tie rate | Mean Spearman r | Mean absolute rank shift | Same unique first | Reference first in score-only top set |
|---|---:|---:|---:|---:|---:|---:|
| ASR | 79.3% | 27.3% | 0.608 | 1.458 | 28.9% | 67.6% |
| Combined rubric-sum ranking | 81.5% | 27.3% | 0.623 | 1.375 | 25.2% | 74.1% |

Pair agreement excludes comparisons tied by the score-only judge; the tie rate is shown explicitly because isolated absolute scoring can produce equal totals more often than a forced strict ranking.

## Matched self-preference

The score-only matched rank effect compares the rank a judge derives for its own saved answer with the mean rank derived from outside judges' independent scores of that exact answer. Negative values indicate self-preference. No presentation-position adjustment is needed because every prompt contains only one answer.

The pooled effect is **-1.299 rank positions** (95% CI -1.374 to -1.224).

| Judge | Matched rank effect | 95% CI | Own-score bias (0–5) |
|---|---:|---:|---:|
| `Qwen/Qwen3.5-122B-A10B-FP8` | -1.484 | [-1.651, -1.317] | 1.322 |
| `Qwen/Qwen3.8-27B-FP8` | -1.271 | [-1.515, -1.028] | 1.160 |
| `claude-opus-5` | -1.918 | [-2.047, -1.790] | -0.170 |
| `claude-sonnet-5` | -0.611 | [-0.847, -0.374] | 0.104 |
| `gemini-3.1-pro-preview` | -0.314 | [-0.571, -0.058] | -0.130 |
| `gemini-3.7-flash` | -0.740 | [-0.986, -0.494] | 0.902 |
| `gpt-5.6-sol` | -2.445 | [-2.586, -2.304] | 0.167 |
| `gpt-5.6-terra` | -1.604 | [-1.784, -1.424] | -0.885 |

## Interpretation

The central comparison is whether removing all simultaneous candidate context changes the model ordering or the matched own-answer advantage. Pairwise agreement describes ordering stability; the matched effect isolates judge-specific treatment of the same fixed answer. Raw absolute scores should not be pooled without caution because judge calibration differs across model families.

All statistics are reproducible from `scripts/analyze_score_only_real_pocqi.py`; the machine-readable values are stored beside this report in `analysis.json`.
