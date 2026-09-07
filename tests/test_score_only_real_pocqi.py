"""Tests for independent, single-answer Real-POCQi scoring."""

from __future__ import annotations

import json
from pathlib import Path

from generation.real_pocqi import GenerationStatus, RealPocqiOutput
from inference import ModelResponse, Provider, TokenUsage
from judging.judge_real_pocqi import PocqiResponseInput
from judging.real_pocqi import RealPocqiScores
from judging.score_only_real_pocqi import (
    DEFAULT_EXPERIMENT_ID,
    ResumeTracker,
    judge_one_response,
    parse_args,
    run,
)
from scripts.analyze_score_only_real_pocqi import midranks, ranking_comparison


class FakeCaller:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, model: str, input: str, **kwargs) -> ModelResponse:
        self.calls.append({"model": model, "input": input, **kwargs})
        scores = RealPocqiScores(
            accuracy=5,
            clinical_utility=4,
            source_quality=3,
            verifiability=2,
            completeness=1,
        )
        return ModelResponse(
            text=scores.model_dump_json(),
            parsed=scores,
            provider=Provider.OPENAI,
            model=model,
            request_id=f"request-{len(self.calls)}",
            finish_reason="completed",
            usage=TokenUsage(input_tokens=50, output_tokens=10),
        )


def _response(model: str = "generator-a") -> PocqiResponseInput:
    return PocqiResponseInput(
        generation_id=f"generation-{model}",
        generator_family="hidden-family",
        generator_model=model,
        response_text=f"Answer from {model}",
    )


def _generation(question: int, model: str) -> RealPocqiOutput:
    return RealPocqiOutput(
        experiment_id="generation-experiment",
        run_id="generation-run",
        generation_key=f"question-{question}__{model}",
        generation_id=f"generation-{question}-{model}",
        attempt=1,
        status=GenerationStatus.SUCCEEDED,
        created_at="2026-09-06T12:00:00+00:00",
        question_id=f"question-{question}",
        question_text=f"Clinical question {question}?",
        specialty="Medicine",
        generator_family="hidden-family",
        generator_model=model,
        prompt_template_id="generation-v1",
        system_prompt="system",
        user_prompt="user",
        response_text=f"Answer from {model}",
    )


def test_one_response_prompt_is_blinded_and_has_no_ranking(tmp_path: Path) -> None:
    path = tmp_path / "score_only.jsonl"
    caller = FakeCaller()
    tracker = ResumeTracker(path)
    record = judge_one_response(
        question_id="question-1",
        question_text="What should be done?",
        specialty="Medicine",
        response=_response(),
        judge_model="openai/gpt-test",
        experiment_id="score-only-test",
        run_id="run-1",
        tracker=tracker,
        model_caller=caller,
        retry_delay_seconds=0,
    )

    assert len(caller.calls) == 1
    visible = caller.calls[0]["system"] + "\n" + caller.calls[0]["input"]
    assert "Answer from generator-a" in visible
    assert "generator-a" not in visible.replace("Answer from generator-a", "")
    assert "generation-generator-a" not in visible
    assert "do not provide a ranking" in visible
    assert caller.calls[0]["response_format"] is RealPocqiScores
    assert record.identity_blinded is True
    assert record.candidates_per_prompt == 1
    assert record.ranking_requested is False
    assert record.scores is not None

    restored = json.loads(path.read_text())
    assert "ranking" not in restored
    assert set(restored["scores"]) == set(RealPocqiScores.model_fields)

    resumed = judge_one_response(
        question_id="question-1",
        question_text="What should be done?",
        specialty="Medicine",
        response=_response(),
        judge_model="openai/gpt-test",
        experiment_id="score-only-test",
        run_id="run-2",
        tracker=tracker,
        model_caller=caller,
        retry_delay_seconds=0,
    )
    assert resumed.judgment_id == record.judgment_id
    assert len(caller.calls) == 1


def test_batch_defaults_to_seeded_200_question_experiment() -> None:
    args = parse_args([])
    assert args.num_questions == 200
    assert args.question_sample_seed == 42
    assert args.experiment_id == DEFAULT_EXPERIMENT_ID


def test_batch_dry_run_counts_individual_answers(tmp_path: Path, capsys) -> None:
    generations = tmp_path / "generations.jsonl"
    models = ("generator-a", "generator-b")
    records = [
        _generation(question, model)
        for question in (1, 2)
        for model in models
    ]
    generations.write_text(
        "\n".join(record.to_json() for record in records) + "\n",
        encoding="utf-8",
    )
    caller = FakeCaller()
    args = parse_args(
        [
            "--input-generations", str(generations),
            "--output-path", str(tmp_path / "scores.jsonl"),
            "--generator-models", *models,
            "--judge-models", "gpt-test", "claude-test",
            "--num-questions", "2",
            "--dry-run",
        ]
    )
    assert run(args, model_caller=caller) == 0
    assert caller.calls == []
    assert "8 logical judgments, 8 pending, 0 skipped" in capsys.readouterr().out


def test_offline_rank_construction_preserves_ties() -> None:
    ranks = midranks({"a": 20, "b": 20, "c": 10})
    assert ranks == {"a": 1.5, "b": 1.5, "c": 3.0}

    comparison = ranking_comparison(ranks, {"a": 1, "b": 2, "c": 3})
    assert comparison["concordant_pairs"] == 2
    assert comparison["discordant_pairs"] == 0
    assert comparison["tied_pairs"] == 1
    assert comparison["pair_agreement_excluding_ties"] == 1.0
