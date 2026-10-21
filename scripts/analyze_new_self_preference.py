"""Reproduce self-preference analyses for the completed expanded experiments.

This script uses only the Python standard library. It treats questions as the
independent sampling units for pooled inference and prints JSON so the reported
Markdown results can be checked without additional model calls.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev


ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data/real_pcoqi/judgements/rubric_and_model_ranking.jsonl"
DIRECT = ROOT / "data/real_pcoqi/judgements/direct_ranking.jsonl"
IDENTITY_REVEALED = (
    ROOT / "data/real_pcoqi/judgements/identity_revealed_rubric_and_model_ranking.jsonl"
)
MED = {
    2: ROOT / "data/outputs/medsp1000/judgements/rubric_and_model_ranking_2_turns.jsonl",
    4: ROOT / "data/outputs/medsp1000/judgements/rubric_and_model_ranking_4_turns.jsonl",
    6: ROOT / "data/outputs/medsp1000/judgements/rubric_and_model_ranking_6_turns.jsonl",
    8: ROOT / "data/outputs/medsp1000/judgements/rubric_and_model_ranking.jsonl",
}


def read_production(path: Path, experiment_id: str) -> list[dict]:
    records: dict[tuple[str, str], dict] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("experiment_id") != experiment_id or row.get("status") != "succeeded":
                continue
            key = (row["question_id"], row["judge_model"])
            # Files are append-only; retaining the last success makes retries safe.
            records[key] = row
    return list(records.values())


def candidate_maps(row: dict) -> tuple[dict[str, str], dict[str, int]]:
    generator = {c["response_id"]: c["generator_model"] for c in row["candidates"]}
    position = {c["response_id"]: i + 1 for i, c in enumerate(row["candidates"])}
    return generator, position


def long_ranks(records: list[dict]) -> list[dict]:
    result = []
    for row in records:
        generator, position = candidate_maps(row)
        rank_block = row["result"].get("model_ranking") or row["result"]["ranking"]
        ranking = rank_block["response_ids"]
        for rank, response_id in enumerate(ranking, 1):
            result.append(
                {
                    "question": row["question_id"],
                    "judge": row["judge_model"],
                    "generator": generator[response_id],
                    "position": position[response_id],
                    "rank": float(rank),
                }
            )
    return result


def long_rubric_z(records: list[dict]) -> list[dict]:
    result = []
    for row in records:
        generator, position = candidate_maps(row)
        scored = row["result"]["scored_responses"]
        sums = {item["response_id"]: sum(item["scores"].values()) for item in scored}
        mean = fmean(sums.values())
        variance = fmean((value - mean) ** 2 for value in sums.values())
        scale = math.sqrt(variance)
        for response_id, value in sums.items():
            result.append(
                {
                    "question": row["question_id"],
                    "judge": row["judge_model"],
                    "generator": generator[response_id],
                    "position": position[response_id],
                    "score": 0.0 if scale == 0 else (value - mean) / scale,
                }
            )
    return result


def position_adjust(rows: list[dict], outcome: str) -> tuple[list[dict], dict[int, float]]:
    """Adjust using judge-generator and categorical presentation-position effects."""

    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        cells[(row["judge"], row["generator"])].append(row)
    positions = sorted({row["position"] for row in rows})
    position_effect = {position: 0.0 for position in positions}
    cell_effect: dict[tuple[str, str], float] = {}
    for _ in range(10_000):
        cell_effect = {
            cell: fmean(row[outcome] - position_effect[row["position"]] for row in members)
            for cell, members in cells.items()
        }
        by_position: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            cell = (row["judge"], row["generator"])
            by_position[row["position"]].append(row[outcome] - cell_effect[cell])
        updated = {position: fmean(by_position[position]) for position in positions}
        center = fmean(updated[row["position"]] for row in rows)
        updated = {position: value - center for position, value in updated.items()}
        change = max(abs(updated[p] - position_effect[p]) for p in positions)
        position_effect = updated
        if change < 1e-12:
            break
    adjusted = [
        {**row, outcome: row[outcome] - position_effect[row["position"]]} for row in rows
    ]
    return adjusted, position_effect


def matched_effects(rows: list[dict], outcome: str, lower_is_better: bool) -> list[dict]:
    by_qg: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_qg[(row["question"], row["generator"])][row["judge"]] = row[outcome]
    effects = []
    direction = 1.0 if lower_is_better else -1.0
    for (question, generator), judge_values in by_qg.items():
        if generator not in judge_values or len(judge_values) < 2:
            continue
        outside = [value for judge, value in judge_values.items() if judge != generator]
        # Negative always denotes self-preference in the output.
        effect = direction * (judge_values[generator] - fmean(outside))
        effects.append({"question": question, "judge": generator, "effect": effect})
    return effects


def normal_summary(values: list[float]) -> dict:
    mean = fmean(values)
    se = stdev(values) / math.sqrt(len(values))
    z = mean / se
    # Two-sided normal approximation using erfc.
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {
        "n": len(values),
        "mean": mean,
        "se": se,
        "ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "p_normal": p,
    }


def summarize_effects(effects: list[dict]) -> dict:
    by_judge: dict[str, list[float]] = defaultdict(list)
    by_question: dict[str, list[float]] = defaultdict(list)
    for row in effects:
        by_judge[row["judge"]].append(row["effect"])
        by_question[row["question"]].append(row["effect"])
    return {
        "pooled_question_clustered": normal_summary(
            [fmean(values) for values in by_question.values()]
        ),
        "by_judge": {judge: normal_summary(values) for judge, values in sorted(by_judge.items())},
        "question_effects": {question: fmean(values) for question, values in by_question.items()},
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for i, key in enumerate(ordered):
        running = max(running, (count - i) * p_values[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def analyze_records(records: list[dict], include_rubric: bool = True) -> dict:
    raw_ranks = long_ranks(records)
    unadjusted_rank_summary = summarize_effects(
        matched_effects(raw_ranks, "rank", lower_is_better=True)
    )["pooled_question_clustered"]
    ranks, position_effects = position_adjust(raw_ranks, "rank")
    rank_summary = summarize_effects(matched_effects(ranks, "rank", lower_is_better=True))
    rank_summary["position_rank_effects"] = position_effects
    rank_summary["unadjusted_pooled_question_clustered"] = unadjusted_rank_summary
    rank_summary["holm_p_by_judge"] = holm(
        {judge: result["p_normal"] for judge, result in rank_summary["by_judge"].items()}
    )
    output = {"rank": rank_summary}
    if include_rubric:
        rubric, rubric_position_effects = position_adjust(long_rubric_z(records), "score")
        rubric_summary = summarize_effects(
            matched_effects(rubric, "score", lower_is_better=False)
        )
        rubric_summary["position_standardized_score_effects"] = rubric_position_effects
        output["rubric_z"] = rubric_summary
    return output


def ranking_condition_comparison(direct_records: list[dict], combined_records: list[dict]) -> dict:
    def ranking_maps(records: list[dict]) -> dict[tuple[str, str], list[str]]:
        output = {}
        for row in records:
            generator, _ = candidate_maps(row)
            rank_block = row["result"].get("model_ranking") or row["result"]["ranking"]
            output[(row["question_id"], row["judge_model"])] = [
                generator[response_id] for response_id in rank_block["response_ids"]
            ]
        return output

    direct_map = ranking_maps(direct_records)
    combined_map = ranking_maps(combined_records)
    shared = sorted(set(direct_map) & set(combined_map))
    pair_agree = pair_total = same_first = same_order = 0
    for key in shared:
        direct, combined = direct_map[key], combined_map[key]
        direct_rank = {model: rank for rank, model in enumerate(direct)}
        combined_rank = {model: rank for rank, model in enumerate(combined)}
        models = direct
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                pair_total += 1
                pair_agree += (direct_rank[models[i]] < direct_rank[models[j]]) == (
                    combined_rank[models[i]] < combined_rank[models[j]]
                )
        same_first += direct[0] == combined[0]
        same_order += direct == combined

    direct_adjusted, _ = position_adjust(long_ranks(direct_records), "rank")
    combined_adjusted, _ = position_adjust(long_ranks(combined_records), "rank")
    direct_effect = {
        (row["question"], row["judge"]): row["effect"]
        for row in matched_effects(direct_adjusted, "rank", lower_is_better=True)
    }
    combined_effect = {
        (row["question"], row["judge"]): row["effect"]
        for row in matched_effects(combined_adjusted, "rank", lower_is_better=True)
    }
    by_question: dict[str, list[float]] = defaultdict(list)
    for question, judge in sorted(set(direct_effect) & set(combined_effect)):
        by_question[question].append(
            direct_effect[(question, judge)] - combined_effect[(question, judge)]
        )
    return {
        "judge_question_pairs": len(shared),
        "candidate_pair_agreement": pair_agree / pair_total,
        "same_first_place": same_first / len(shared),
        "same_complete_order": same_order / len(shared),
        "direct_minus_combined_self_preference": normal_summary(
            [fmean(values) for values in by_question.values()]
        ),
    }


def identity_condition_comparison(
    revealed_records: list[dict], combined_records: list[dict]
) -> dict:
    """Compare revealed and blinded ranks on identical question-judge cells."""

    revealed_keys = {(row["question_id"], row["judge_model"]) for row in revealed_records}
    blinded_records = [
        row
        for row in combined_records
        if (row["question_id"], row["judge_model"]) in revealed_keys
    ]
    if len(blinded_records) != len(revealed_records):
        raise ValueError("Identity conditions do not have identical question-judge cells")

    revealed_ranks, revealed_positions = position_adjust(long_ranks(revealed_records), "rank")
    blinded_ranks, blinded_positions = position_adjust(long_ranks(blinded_records), "rank")
    revealed_effects = {
        (row["question"], row["judge"]): row["effect"]
        for row in matched_effects(revealed_ranks, "rank", lower_is_better=True)
    }
    blinded_effects = {
        (row["question"], row["judge"]): row["effect"]
        for row in matched_effects(blinded_ranks, "rank", lower_is_better=True)
    }
    shared = sorted(set(revealed_effects) & set(blinded_effects))
    if len(shared) != len(revealed_records):
        raise ValueError("Identity comparison is missing matched effects")

    differences = [
        {
            "question": question,
            "judge": judge,
            "effect": revealed_effects[(question, judge)]
            - blinded_effects[(question, judge)],
        }
        for question, judge in shared
    ]
    revealed_summary = summarize_effects(
        [
            {"question": question, "judge": judge, "effect": effect}
            for (question, judge), effect in revealed_effects.items()
        ]
    )
    blinded_summary = summarize_effects(
        [
            {"question": question, "judge": judge, "effect": effect}
            for (question, judge), effect in blinded_effects.items()
        ]
    )
    change_summary = summarize_effects(differences)
    change_summary["holm_p_by_judge"] = holm(
        {judge: result["p_normal"] for judge, result in change_summary["by_judge"].items()}
    )
    for summary in (revealed_summary, blinded_summary, change_summary):
        summary.pop("question_effects", None)
    return {
        "questions": len({question for question, _ in shared}),
        "judges": len({judge for _, judge in shared}),
        "matched_question_judge_cells": len(shared),
        "revealed": revealed_summary,
        "matched_blinded": blinded_summary,
        "revealed_minus_blinded": change_summary,
        "position_rank_effects": {
            "revealed": revealed_positions,
            "blinded": blinded_positions,
        },
    }


def repeated_measures_anova(question_by_length: dict[int, dict[str, float]]) -> dict:
    lengths = sorted(question_by_length)
    questions = sorted(set.intersection(*(set(question_by_length[x]) for x in lengths)))
    matrix = [[question_by_length[length][question] for length in lengths] for question in questions]
    column_means = [fmean(row[j] for row in matrix) for j in range(len(lengths))]
    grand_mean = fmean(column_means)
    row_means = [fmean(row) for row in matrix]
    ss_time = len(matrix) * sum((value - grand_mean) ** 2 for value in column_means)
    ss_error = sum(
        (matrix[i][j] - row_means[i] - column_means[j] + grand_mean) ** 2
        for i in range(len(matrix))
        for j in range(len(lengths))
    )
    df_time = len(lengths) - 1
    df_error = (len(matrix) - 1) * df_time
    f_stat = (ss_time / df_time) / (ss_error / df_error)
    return {
        "n_questions": len(matrix),
        "lengths": lengths,
        "means": column_means,
        "F": f_stat,
        "df": [df_time, df_error],
    }


def paired_contrast(
    question_by_length: dict[int, dict[str, float]], later: int, earlier: int
) -> dict:
    shared = sorted(set(question_by_length[later]) & set(question_by_length[earlier]))
    return normal_summary(
        [question_by_length[later][q] - question_by_length[earlier][q] for q in shared]
    )


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1e-300 if abs(d) < 1e-300 else d
    d = 1.0 / d
    h = d
    for m in range(1, 501):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def f_survival(f_stat: float, df1: int, df2: int) -> float:
    x = df2 / (df2 + df1 * f_stat)
    return regularized_beta(x, df2 / 2.0, df1 / 2.0)


def invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    augmented = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][j] - factor * augmented[column][j] for j in range(2 * n)
            ]
    return [row[n:] for row in augmented]


def hotelling_length_test(question_by_length: dict[int, dict[str, float]]) -> dict:
    lengths = sorted(question_by_length)
    baseline = lengths[0]
    questions = sorted(set.intersection(*(set(question_by_length[x]) for x in lengths)))
    vectors = [
        [question_by_length[length][q] - question_by_length[baseline][q] for length in lengths[1:]]
        for q in questions
    ]
    n, dimensions = len(vectors), len(vectors[0])
    means = [fmean(row[j] for row in vectors) for j in range(dimensions)]
    covariance = [
        [
            sum((row[i] - means[i]) * (row[j] - means[j]) for row in vectors) / (n - 1)
            for j in range(dimensions)
        ]
        for i in range(dimensions)
    ]
    inverse = invert_matrix(covariance)
    quadratic = sum(means[i] * inverse[i][j] * means[j] for i in range(dimensions) for j in range(dimensions))
    t_squared = n * quadratic
    f_stat = (n - dimensions) * t_squared / (dimensions * (n - 1))
    return {
        "n_questions": n,
        "contrasts": [f"{length}-{baseline}" for length in lengths[1:]],
        "means": means,
        "hotelling_T2": t_squared,
        "F": f_stat,
        "df": [dimensions, n - dimensions],
        "p": f_survival(f_stat, dimensions, n - dimensions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--medsp-experiment-id",
        default="medsp1000_all_judges_v1",
        help="MedSP1000 experiment identifier to analyze",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON to this path instead of standard output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real_records = read_production(REAL, "real_pocqi_combined_all_judges_v1")
    direct_records = read_production(DIRECT, "real_pocqi_direct_ranking_random100_v1")
    identity_records = read_production(
        IDENTITY_REVEALED, "real_pocqi_identity_revealed_random200_v1"
    )
    med_records = {
        length: read_production(path, args.medsp_experiment_id)
        for length, path in MED.items()
    }

    real = analyze_records(real_records)
    direct = analyze_records(direct_records, include_rubric=False)
    direct_comparison = ranking_condition_comparison(direct_records, real_records)
    identity_comparison = identity_condition_comparison(identity_records, real_records)
    med = {length: analyze_records(records) for length, records in med_records.items()}
    med_question_effects = {
        length: result["rank"]["question_effects"] for length, result in med.items()
    }
    med_repeated = {
        "anova": repeated_measures_anova(med_question_effects),
        "hotelling_test": hotelling_length_test(med_question_effects),
        "contrasts_vs_2": {
            str(length): paired_contrast(med_question_effects, length, 2)
            for length in (4, 6, 8)
        },
        "adjacent_contrasts": {
            f"{later}-{earlier}": paired_contrast(med_question_effects, later, earlier)
            for earlier, later in ((2, 4), (4, 6), (6, 8))
        },
    }
    med_repeated["anova"]["p_sphericity_assumed"] = f_survival(
        med_repeated["anova"]["F"], *med_repeated["anova"]["df"]
    )

    # Remove bulky per-question values from printed summaries after using them.
    for result in [real, direct, *med.values()]:
        for analysis in result.values():
            analysis.pop("question_effects", None)

    rendered = json.dumps(
        {
            "medsp1000_experiment_id": args.medsp_experiment_id,
            "record_counts": {
                "real_pocqi": len(real_records),
                "direct_ranking": len(direct_records),
                "identity_revealed": len(identity_records),
                "medsp1000": {str(k): len(v) for k, v in med_records.items()},
            },
            "real_pocqi": real,
            "direct_ranking": direct,
            "direct_vs_combined": direct_comparison,
            "identity_revealed_vs_blinded": identity_comparison,
            "medsp1000": med,
            "medsp1000_repeated_measures": med_repeated,
        },
        indent=2,
        sort_keys=True,
    )
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
