"""Analyze independently scored Real-POCQi answers and compare rankings.

The score-only condition never presents multiple answers together. This script
constructs within-question rankings after judgment, uses midranks for tied
rubric sums, and compares them with both ranking signals saved by the existing
combined condition: its explicit answer-selection ranking (ASR) and its
within-prompt rubric-sum ranking.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = ROOT / "data/real_pcoqi/judgements/score_only.jsonl"
DEFAULT_COMBINED = ROOT / "data/real_pcoqi/judgements/rubric_and_model_ranking.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/analysis/score_only"
SCORE_EXPERIMENT = "real_pocqi_score_only_random200_v1"
COMBINED_EXPERIMENT = "real_pocqi_combined_all_judges_v1"


def read_score_only(path: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("experiment_id") != SCORE_EXPERIMENT or row.get("status") != "succeeded":
                continue
            key = (row["question_id"], row["judge_model"], row["generator_model"])
            previous = latest.get(key)
            if previous is None or row["attempt"] >= previous["attempt"]:
                latest[key] = row
    return list(latest.values())


def read_combined(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("experiment_id") != COMBINED_EXPERIMENT or row.get("status") != "succeeded":
                continue
            latest[(row["question_id"], row["judge_model"])] = row
    return latest


def midranks(scores: dict[str, float]) -> dict[str, float]:
    """Descending ranks with the average occupied rank assigned to ties."""

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = ((start + 1) + end) / 2
        for model, _ in ordered[start:end]:
            ranks[model] = rank
        start = end
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    xmean, ymean = fmean(xs), fmean(ys)
    numerator = sum((x - xmean) * (y - ymean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - xmean) ** 2 for x in xs) * sum((y - ymean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def summary(values: Iterable[float]) -> dict[str, float | int | list[float] | None]:
    data = list(values)
    if not data:
        return {"n": 0, "mean": None, "ci95": None}
    mean = fmean(data)
    if len(data) == 1:
        return {"n": 1, "mean": mean, "ci95": None}
    se = stdev(data) / math.sqrt(len(data))
    return {"n": len(data), "mean": mean, "ci95": [mean - 1.96 * se, mean + 1.96 * se]}


def combined_rankings(row: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    model_for_id = {candidate["response_id"]: candidate["generator_model"] for candidate in row["candidates"]}
    explicit = {
        model_for_id[response_id]: float(rank)
        for rank, response_id in enumerate(row["result"]["model_ranking"]["response_ids"], 1)
    }
    sums = {
        model_for_id[item["response_id"]]: sum(item["scores"].values())
        for item in row["result"]["scored_responses"]
    }
    return explicit, midranks(sums)


def ranking_comparison(
    left: dict[str, float], right: dict[str, float]
) -> dict[str, float | int | bool | None]:
    models = sorted(left)
    concordant = discordant = ties = left_ties = right_ties = 0
    for index, first in enumerate(models):
        for second in models[index + 1 :]:
            ldiff = left[first] - left[second]
            rdiff = right[first] - right[second]
            left_ties += ldiff == 0
            right_ties += rdiff == 0
            if ldiff == 0 or rdiff == 0:
                ties += 1
            elif (ldiff < 0) == (rdiff < 0):
                concordant += 1
            else:
                discordant += 1
    non_tied = concordant + discordant
    left_best = {model for model, rank in left.items() if rank == min(left.values())}
    right_best = {model for model, rank in right.items() if rank == min(right.values())}
    return {
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "tied_pairs": ties,
        "left_tied_pairs": left_ties,
        "right_tied_pairs": right_ties,
        "pair_agreement_excluding_ties": concordant / non_tied if non_tied else None,
        "spearman_r": pearson([left[m] for m in models], [right[m] for m in models]),
        "mean_absolute_rank_difference": fmean(abs(left[m] - right[m]) for m in models),
        "same_unique_first": len(left_best) == len(right_best) == 1 and left_best == right_best,
        "right_first_in_left_top_tie": bool(left_best & right_best),
    }


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def analyze(scores_path: Path, combined_path: Path) -> dict[str, Any]:
    score_rows = read_score_only(scores_path)
    if not score_rows:
        raise ValueError(f"no completed {SCORE_EXPERIMENT} rows in {scores_path}")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[(row["question_id"], row["judge_model"])].append(row)
    combined = read_combined(combined_path)
    generators = sorted({row["generator_model"] for row in score_rows})
    judges = sorted({row["judge_model"] for row in score_rows})
    questions = sorted({row["question_id"] for row in score_rows})
    expected = set(generators)
    if len(score_rows) != len(questions) * len(judges) * len(generators):
        raise ValueError("score-only matrix is incomplete")

    cells: dict[tuple[str, str], dict[str, Any]] = {}
    explicit_comparisons = []
    rubric_comparisons = []
    for key, rows in grouped.items():
        cell_scores = {
            row["generator_model"]: sum(row["scores"].values()) for row in rows
        }
        if set(cell_scores) != expected:
            raise ValueError(f"incomplete generators for {key}")
        if key not in combined:
            raise ValueError(f"missing matched combined judgment for {key}")
        ranks = midranks(cell_scores)
        explicit, rubric = combined_rankings(combined[key])
        cells[key] = {"scores": cell_scores, "ranks": ranks, "explicit": explicit, "rubric": rubric}
        explicit_comparisons.append(ranking_comparison(ranks, explicit))
        rubric_comparisons.append(ranking_comparison(ranks, rubric))

    def aggregate_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
        concordant = sum(row["concordant_pairs"] for row in rows)
        discordant = sum(row["discordant_pairs"] for row in rows)
        tied = sum(row["tied_pairs"] for row in rows)
        left_tied = sum(row["left_tied_pairs"] for row in rows)
        right_tied = sum(row["right_tied_pairs"] for row in rows)
        total_pairs = concordant + discordant + tied
        return {
            "cells": len(rows),
            "candidate_pair_agreement_excluding_score_only_ties": concordant / (concordant + discordant),
            "score_only_pair_tie_rate": left_tied / total_pairs,
            "reference_pair_tie_rate": right_tied / total_pairs,
            "mean_cell_spearman_r": fmean(row["spearman_r"] for row in rows if row["spearman_r"] is not None),
            "mean_absolute_rank_difference": fmean(row["mean_absolute_rank_difference"] for row in rows),
            "same_unique_first_rate": fmean(row["same_unique_first"] for row in rows),
            "reference_first_in_score_only_top_rate": fmean(row["right_first_in_left_top_tie"] for row in rows),
        }

    generator_rows = []
    for generator in generators:
        generator_rows.append(
            {
                "generator": generator,
                "score_only_mean_sum": fmean(cell["scores"][generator] for cell in cells.values()),
                "score_only_mean_rank": fmean(cell["ranks"][generator] for cell in cells.values()),
                "asr_mean_rank": fmean(cell["explicit"][generator] for cell in cells.values()),
                "combined_rubric_mean_rank": fmean(cell["rubric"][generator] for cell in cells.values()),
            }
        )
    generator_rows.sort(key=lambda row: row["score_only_mean_rank"])

    self_preference = []
    question_effects: dict[str, list[float]] = defaultdict(list)
    for judge in judges:
        effects = []
        score_biases = []
        for question in questions:
            own = cells[(question, judge)]
            outside = [cells[(question, other)] for other in judges if other != judge]
            effect = own["ranks"][judge] - fmean(cell["ranks"][judge] for cell in outside)
            score_bias = own["scores"][judge] - fmean(cell["scores"][judge] for cell in outside)
            effects.append(effect)
            score_biases.append(score_bias / 5)
            question_effects[question].append(effect)
        self_preference.append(
            {"judge": judge, "rank_effect": summary(effects), "mean_score_bias_0_to_5": summary(score_biases)}
        )

    return {
        "design": {
            "questions": len(questions),
            "judges": len(judges),
            "generators": len(generators),
            "logical_judgments": len(score_rows),
            "candidates_per_prompt": 1,
            "ranking_requested": False,
            "ranking_method": "descending five-axis score sum; average midranks for ties",
            "asr_definition": "explicit answer-selection ranking from the matched combined rubric-and-ranking condition",
        },
        "generator_ranking": generator_rows,
        "comparison_with_asr": aggregate_comparisons(explicit_comparisons),
        "comparison_with_combined_rubric_sum_ranking": aggregate_comparisons(rubric_comparisons),
        "score_only_matched_self_preference": {
            "pooled": summary(fmean(values) for values in question_effects.values()),
            "by_judge": self_preference,
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    design = result["design"]
    asr = result["comparison_with_asr"]
    rubric = result["comparison_with_combined_rubric_sum_ranking"]
    pooled = result["score_only_matched_self_preference"]["pooled"]
    pooled_ci = pooled["ci95"]
    lines = [
        "# Real-POCQi score-only experiment",
        "",
        "## Design",
        "",
        f"This identity-blind follow-up uses the seed-42 sample of **{design['questions']} questions**. "
        f"Each of {design['judges']} judges scored each of {design['generators']} saved generations in a "
        f"separate prompt ({design['logical_judgments']:,} logical judgments). A judge saw one answer only, "
        "and neither the prompt nor the structured output requested a ranking. Rankings below were constructed "
        "afterward from the five-axis score sum, with average ranks for ties.",
        "",
        "For this report, **ASR** means the explicit answer-selection ranking from the matched existing "
        "rubric-and-ranking condition. The within-prompt rubric-sum ranking is reported separately as a "
        "second reference.",
        "",
        "## Aggregate generator ranking",
        "",
        "| Rank | Generator | Score-only mean rank | Score-only mean score / 25 | ASR mean rank | Combined rubric-sum mean rank |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(result["generator_ranking"], 1):
        lines.append(
            f"| {rank} | `{row['generator']}` | {fmt(row['score_only_mean_rank'])} | "
            f"{fmt(row['score_only_mean_sum'])} | {fmt(row['asr_mean_rank'])} | "
            f"{fmt(row['combined_rubric_mean_rank'])} |"
        )
    lines.extend(
        [
            "",
            "## Agreement with existing rankings",
            "",
            "| Comparison | Pair agreement (ties excluded) | Score-only tie rate | Mean Spearman r | Mean absolute rank shift | Same unique first | Reference first in score-only top set |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| ASR | {asr['candidate_pair_agreement_excluding_score_only_ties']:.1%} | {asr['score_only_pair_tie_rate']:.1%} | {fmt(asr['mean_cell_spearman_r'])} | {fmt(asr['mean_absolute_rank_difference'])} | {asr['same_unique_first_rate']:.1%} | {asr['reference_first_in_score_only_top_rate']:.1%} |",
            f"| Combined rubric-sum ranking | {rubric['candidate_pair_agreement_excluding_score_only_ties']:.1%} | {rubric['score_only_pair_tie_rate']:.1%} | {fmt(rubric['mean_cell_spearman_r'])} | {fmt(rubric['mean_absolute_rank_difference'])} | {rubric['same_unique_first_rate']:.1%} | {rubric['reference_first_in_score_only_top_rate']:.1%} |",
            "",
            "Pair agreement excludes comparisons tied by the score-only judge; the tie rate is shown explicitly "
            "because isolated absolute scoring can produce equal totals more often than a forced strict ranking.",
            "",
            "## Matched self-preference",
            "",
            "The score-only matched rank effect compares the rank a judge derives for its own saved answer with "
            "the mean rank derived from outside judges' independent scores of that exact answer. Negative values "
            "indicate self-preference. No presentation-position adjustment is needed because every prompt contains "
            "only one answer.",
            "",
            f"The pooled effect is **{fmt(pooled['mean'])} rank positions** "
            f"(95% CI {fmt(pooled_ci[0])} to {fmt(pooled_ci[1])}).",
            "",
            "| Judge | Matched rank effect | 95% CI | Own-score bias (0–5) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in result["score_only_matched_self_preference"]["by_judge"]:
        rank = row["rank_effect"]
        score = row["mean_score_bias_0_to_5"]
        ci = rank["ci95"]
        lines.append(
            f"| `{row['judge']}` | {fmt(rank['mean'])} | [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(score['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The central comparison is whether removing all simultaneous candidate context changes the model "
            "ordering or the matched own-answer advantage. Pairwise agreement describes ordering stability; "
            "the matched effect isolates judge-specific treatment of the same fixed answer. Raw absolute scores "
            "should not be pooled without caution because judge calibration differs across model families.",
            "",
            "All statistics are reproducible from `scripts/analyze_score_only_real_pocqi.py`; the machine-readable "
            "values are stored beside this report in `analysis.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--combined", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = analyze(args.scores, args.combined)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(render_markdown(result), encoding="utf-8")
    print(args.output_dir / "README.md")


if __name__ == "__main__":
    main()
