"""Tests for the batched Modal Real-POCQi judge adapter."""

from __future__ import annotations

from judging import DirectRankingOutput
from judging.modal_real_pocqi_judging import (
    ModalBatchCaller,
    _runner_argv,
    _score_only_runner_argv,
)
from judging.run_real_pocqi_judging import parse_args
from judging.score_only_real_pocqi import parse_args as parse_score_only_args


class _FakeRemoteMethod:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def remote(
        self,
        conversations,
        response_schema,
        temperature,
        max_tokens,
    ):
        self.calls.append(
            (conversations, response_schema, temperature, max_tokens)
        )
        count = len(conversations)
        return {
            "texts": [
                '{"ranking":{"response_ids":["response-2","response-1"]}}'
                for _ in range(count)
            ],
            "input_tokens": [100] * count,
            "output_tokens": [20] * count,
            "finish_reasons": ["stop"] * count,
            "batch_latency_seconds": 1.0,
        }


class _FakeRemote:
    def __init__(self) -> None:
        self.generate_batch = _FakeRemoteMethod()


def test_modal_batch_caller_parses_structured_response() -> None:
    remote = _FakeRemote()
    caller = ModalBatchCaller(
        {"Qwen-test": remote},
        batch_size=2,
        flush_seconds=0,
    )
    try:
        response = caller(
            "modal/Qwen-test",
            "Rank the responses",
            system="Judge overall quality",
            response_format=DirectRankingOutput,
            max_output_tokens=256,
            temperature=0,
        )
    finally:
        caller.close()

    assert response.provider.value == "modal"
    assert response.model == "Qwen-test"
    assert response.parsed.ranking.response_ids == ["response-2", "response-1"]
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 20
    conversations, schema, temperature, max_tokens = (
        remote.generate_batch.calls[0]
    )
    assert conversations == [
        [
            {"role": "system", "content": "Judge overall quality"},
            {"role": "user", "content": "Rank the responses"},
        ]
    ]
    assert schema["title"] == "DirectRankingOutput"
    assert temperature == 0
    assert max_tokens == 256


def test_modal_runner_argv_preserves_seeded_subset() -> None:
    args = parse_args(
        _runner_argv(
            input_generations="generations.jsonl",
            output_dir="judgements",
            experiment_id="direct-random-100",
            run_id="run-1",
            judging_case="direct_ranking",
            models=("Qwen-test",),
            num_questions=100,
            question_sample_seed=20260824,
            presentation_seed=42,
            max_output_tokens=512,
            max_concurrency=8,
            retries=2,
            retry_delay_seconds=1,
            reveal_generator_identities=True,
            force=False,
        )
    )

    assert args.judge_models == ["modal/Qwen-test"]
    assert args.judging_cases == ["direct_ranking"]
    assert args.num_questions == 100
    assert args.question_sample_seed == 20260824
    assert args.modal_concurrency == 8
    assert args.reveal_generator_identities is True


def test_modal_score_only_argv_uses_individual_scoring_runner() -> None:
    args = parse_score_only_args(
        _score_only_runner_argv(
            input_generations="generations.jsonl",
            output_path="score_only.jsonl",
            experiment_id="score-only",
            run_id="run-1",
            models=("Qwen-test",),
            num_questions=200,
            question_sample_seed=42,
            max_output_tokens=512,
            max_concurrency=16,
            retries=2,
            retry_delay_seconds=1,
            force=False,
        )
    )
    assert args.judge_models == ["modal/Qwen-test"]
    assert args.num_questions == 200
    assert args.question_sample_seed == 42
    assert args.output_path.name == "score_only.jsonl"
