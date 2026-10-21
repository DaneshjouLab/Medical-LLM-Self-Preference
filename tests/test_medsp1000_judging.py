"""Tests for blinded multi-turn MedSP1000 judging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference import ModelResponse, Provider, TokenUsage
from judging.judge_medsp1000 import judge_medsp1000_trajectories
from judging.judge_real_pocqi import PocqiJudgingSettings, PocqiResponseInput
from judging.real_pocqi import (
    PocqiJudgingCase,
    RealPocqiScores,
    ResponseRanking,
    RubricAndModelRankingOutput,
    ScoredPocqiResponse,
)
from judging.run_medsp1000_judging import (
    load_latest_question_trajectories,
    output_paths,
    parse_args,
    run,
)


class FakeJudgeCaller:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, model: str, input: str, **kwargs) -> ModelResponse:
        self.calls.append({"model": model, "input": input, **kwargs})
        response_format = kwargs["response_format"]
        ranking = ResponseRanking(response_ids=["response-2", "response-1"])
        scored = [
            ScoredPocqiResponse(
                response_id=response_id,
                scores=RealPocqiScores(
                    accuracy=score,
                    clinical_utility=score,
                    source_quality=score,
                    verifiability=score,
                    completeness=score,
                ),
            )
            for response_id, score in (("response-1", 4), ("response-2", 5))
        ]
        assert response_format is RubricAndModelRankingOutput
        parsed = RubricAndModelRankingOutput(
            scored_responses=scored,
            model_ranking=ranking,
        )
        return ModelResponse(
            text=parsed.model_dump_json(),
            parsed=parsed,
            provider=Provider.OPENAI,
            model=model,
            request_id=f"request-{len(self.calls)}",
            finish_reason="completed",
            usage=TokenUsage(input_tokens=100, output_tokens=20),
        )


def _responses() -> list[PocqiResponseInput]:
    return [
        PocqiResponseInput(
            generation_id=f"generation-{index}",
            generator_family=family,
            generator_model=model,
            response_text=(
                f"CLINICIAN: Clinician statement {index}\n"
                f"PATIENT: Patient reply {index}\n"
                f"CLINICIAN: Follow-up {index}\n"
                f"PATIENT: Final reply {index}"
            ),
        )
        for index, (family, model) in enumerate(
            (("openai", "gpt-test"), ("anthropic", "claude-test")),
            start=1,
        )
    ]


def test_medsp_judge_uses_full_blinded_trajectories_and_resumes(
    tmp_path: Path,
) -> None:
    caller = FakeJudgeCaller()
    paths = output_paths(tmp_path)
    settings = PocqiJudgingSettings(
        experiment_id="medsp-judge-test",
        run_id="run-1",
        retry_delay_seconds=0,
    )
    records = judge_medsp1000_trajectories(
        question_id="case-1__scenario1",
        question_text="Evaluate and manage this patient.",
        source_scenario_path="case-1/scenario1",
        responses=_responses(),
        judge_model="openai/gpt-judge",
        settings=settings,
        model_caller=caller,
        output_paths=paths,
    )

    assert list(records) == [PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING]
    assert len(caller.calls) == 1
    assert "multi-turn clinician trajectories" in caller.calls[0]["system"]
    assert "Do not require formal citations" in caller.calls[0]["system"]
    assert "evaluate only the clinician's behavior" in caller.calls[0]["system"]
    for call in caller.calls:
        visible = call["system"] + "\n" + call["input"]
        assert "Evaluate and manage this patient." in visible
        assert "case-1/scenario1" in visible
        assert "CLINICIAN: Clinician statement" in visible
        assert "PATIENT: Patient reply" in visible
        for hidden in (
            "generation-1",
            "generation-2",
            "gpt-test",
            "claude-test",
            "openai",
            "anthropic",
        ):
            assert hidden not in visible

    resumed = judge_medsp1000_trajectories(
        question_id="case-1__scenario1",
        question_text="Evaluate and manage this patient.",
        source_scenario_path="case-1/scenario1",
        responses=list(reversed(_responses())),
        judge_model="openai/gpt-judge",
        settings=PocqiJudgingSettings(
            experiment_id="medsp-judge-test",
            run_id="run-2",
        ),
        model_caller=caller,
        output_paths=paths,
    )
    assert len(caller.calls) == 1
    assert {
        case: record.judgment_id for case, record in resumed.items()
    } == {case: record.judgment_id for case, record in records.items()}


def _question(question: int) -> dict:
    return {
        "question_id": f"case-{question}__scenario1",
        "question_text": f"Initialization {question}",
    }


def _generation(
    question: int,
    model: str,
    *,
    generation_key: str | None = None,
    reasoning_effort: str | None = "medium",
) -> dict:
    question_id = f"case-{question}__scenario1"
    turns = [
        {"role": "clinician", "content": f"Opening from {model}"},
        {"role": "patient", "content": "Reply"},
        {"role": "clinician", "content": "Follow-up"},
        {"role": "patient", "content": "Reply"},
        {"role": "clinician", "content": "Assessment"},
        {"role": "patient", "content": "Reply"},
        {"role": "clinician", "content": "Plan"},
        {"role": "patient", "content": "Reply"},
    ]
    return {
        "status": "succeeded",
        "question_id": question_id,
        "question_text": f"Initialization {question}",
        "source_scenario_path": f"case-{question}/scenario1",
        "clinician_model": model,
        "generation_id": f"generation-{question}-{model}",
        "generation_key": generation_key or f"key-{question}-{model}",
        "attempt": 1,
        "exchange_count": 4,
        "turn_count": 8,
        "clinician_reasoning_effort": reasoning_effort,
        "patient_model": "patient-model",
        "patient_prompt_template_id": "patient-v1",
        "prompt_version": "prompt-v1",
        "turns": turns,
        "transcript_text": "\n".join(
            f"{turn['role'].upper()}: {turn['content']}" for turn in turns
        ),
    }


def _write_matrix(
    questions_path: Path,
    generations_path: Path,
    *,
    question_count: int = 2,
    models: tuple[str, ...] = ("generator-a", "generator-b"),
) -> None:
    questions_path.write_text(
        "\n".join(json.dumps(_question(i)) for i in range(1, question_count + 1))
        + "\n",
        encoding="utf-8",
    )
    generations_path.write_text(
        "\n".join(
            json.dumps(_generation(i, model))
            for i in range(1, question_count + 1)
            for model in models
        )
        + "\n",
        encoding="utf-8",
    )


def test_medsp_loader_requires_complete_unambiguous_matrix(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    generations = tmp_path / "generations.jsonl"
    _write_matrix(questions, generations, question_count=1)

    loaded = load_latest_question_trajectories(
        generations,
        questions_path=questions,
        generator_models=("generator-a", "generator-b"),
    )
    assert len(loaded) == 1
    assert [r.generator_model for r in loaded[0].responses] == [
        "generator-a",
        "generator-b",
    ]
    assert loaded[0].responses[0].response_text.startswith("CLINICIAN:")

    prefix = load_latest_question_trajectories(
        generations,
        questions_path=questions,
        generator_models=("generator-a", "generator-b"),
        view_turn_count=2,
    )
    assert prefix[0].responses[0].response_text.count("CLINICIAN:") == 1
    assert prefix[0].responses[0].response_text.count("PATIENT:") == 1
    assert "Follow-up" not in prefix[0].responses[0].response_text

    generations.write_text(
        generations.read_text(encoding="utf-8")
        + json.dumps(_generation(1, "generator-a", generation_key="other-key"))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multiple successful generation configurations"):
        load_latest_question_trajectories(
            generations,
            questions_path=questions,
            generator_models=("generator-a", "generator-b"),
        )


def test_medsp_loader_accepts_disabled_reasoning_for_qwen(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    generations = tmp_path / "generations.jsonl"
    questions.write_text(json.dumps(_question(1)) + "\n", encoding="utf-8")
    generations.write_text(
        json.dumps(
            _generation(
                1,
                "Qwen/Qwen3.8-27B-FP8",
                reasoning_effort=None,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_latest_question_trajectories(
        generations,
        questions_path=questions,
        generator_models=("Qwen/Qwen3.8-27B-FP8",),
    )
    assert len(loaded) == 1
    assert loaded[0].responses[0].generator_model == "Qwen/Qwen3.8-27B-FP8"


def test_medsp_batch_runner_dry_run_and_resume(tmp_path: Path, capsys) -> None:
    questions = tmp_path / "questions.jsonl"
    generations = tmp_path / "generations.jsonl"
    output_dir = tmp_path / "judgements"
    _write_matrix(questions, generations)
    common = [
        "--input-generations",
        str(generations),
        "--input-questions",
        str(questions),
        "--output-dir",
        str(output_dir),
        "--generator-models",
        "generator-a",
        "generator-b",
        "--judge-models",
        "gpt-test",
        "claude-test",
        "--experiment-id",
        "medsp-test",
        "--max-concurrency",
        "2",
        "--openai-concurrency",
        "1",
        "--anthropic-concurrency",
        "1",
        "--gemini-concurrency",
        "1",
        "--retry-delay-seconds",
        "0",
    ]

    dry_caller = FakeJudgeCaller()
    assert run(parse_args([*common, "--dry-run"]), model_caller=dry_caller) == 0
    assert not dry_caller.calls
    assert "4 logical judgments, 4 pending, 0 skipped" in capsys.readouterr().out

    caller = FakeJudgeCaller()
    assert run(
        parse_args([*common, "--run-id", "medsp-run-1"]), model_caller=caller
    ) == 0
    assert len(caller.calls) == 4
    manifest = json.loads(
        (output_dir / "medsp-run-1.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["question_count"] == 2
    assert manifest["logical_judgments"]["succeeded"] == 4

    resumed_caller = FakeJudgeCaller()
    assert run(
        parse_args([*common, "--run-id", "medsp-run-2"]),
        model_caller=resumed_caller,
    ) == 0
    assert not resumed_caller.calls
    resumed = json.loads(
        (output_dir / "medsp-run-2.manifest.json").read_text(encoding="utf-8")
    )
    assert resumed["logical_judgments"]["skipped_existing"] == 4


def test_partial_view_uses_distinct_prompt_and_output(tmp_path: Path) -> None:
    caller = FakeJudgeCaller()
    settings = PocqiJudgingSettings(
        experiment_id="medsp-prefix-test",
        run_id="run-prefix",
        retry_delay_seconds=0,
    )
    paths = output_paths(tmp_path, view_turn_count=2)
    records = judge_medsp1000_trajectories(
        question_id="case-1__scenario1",
        question_text="Evaluate this patient.",
        source_scenario_path="case-1/scenario1",
        responses=_responses(),
        judge_model="gpt-test",
        settings=settings,
        view_turn_count=2,
        model_caller=caller,
        output_paths=paths,
    )

    record = records[PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING]
    assert record.prompt_template_id.endswith("_first_2_turns_v1")
    assert "first 2 role turns" in record.user_prompt
    assert "visible interaction" in record.system_prompt
    assert paths[PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING].name == (
        "rubric_and_model_ranking_2_turns.jsonl"
    )
