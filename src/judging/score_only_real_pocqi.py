"""Judge Real-POCQi generations one at a time with the five-axis rubric.

Unlike the listwise judging conditions, each model call sees exactly one
generated answer. Rankings are deliberately absent from both the prompt and
the structured response and are constructed only during offline analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from generation.generate_real_pocqi import load_dotenv
from inference import ModelResponse, Provider, call_model, resolve_provider

from .judge_real_pocqi import PocqiResponseInput, RUBRIC_SYSTEM_PROMPT, _judge_family
from .real_pocqi import PocqiJudgmentStatus, RealPocqiScores
from .run_real_pocqi_judging import (
    DEFAULT_GENERATIONS_PATH,
    DEFAULT_GENERATOR_MODELS,
    DEFAULT_JUDGE_MODELS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUESTION_SAMPLE_SEED,
    ProviderLimitedCaller,
    load_latest_question_responses,
    validate_judge_models,
)


DEFAULT_EXPERIMENT_ID = "real_pocqi_score_only_random200_v1"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "score_only.jsonl"
PROMPT_TEMPLATE_ID = "pocqi_individual_rubric_scoring_v1"
SCORE_ONLY_INSTRUCTION = (
    "Score the single response on all five rubric axes. Evaluate it on its own; "
    "do not compare it with other possible answers and do not provide a ranking."
)
_APPEND_LOCK = threading.Lock()

ModelCaller = Callable[..., ModelResponse[Any]]


class ScoreOnlyJudgmentRecord(BaseModel):
    """Append-only record for one answer scored by one judge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    experiment_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    judgment_key: str = Field(min_length=1)
    judgment_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    status: PocqiJudgmentStatus
    created_at: datetime

    question_id: str = Field(min_length=1)
    question_text: str = Field(min_length=1)
    specialty: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    generator_family: str = Field(min_length=1)
    generator_model: str = Field(min_length=1)

    judge_family: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    prompt_template_id: Literal["pocqi_individual_rubric_scoring_v1"] = (
        PROMPT_TEMPLATE_ID
    )
    identity_blinded: Literal[True] = True
    candidates_per_prompt: Literal[1] = 1
    ranking_requested: Literal[False] = False
    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)

    scores: RealPocqiScores | None = None
    judge_response_text: str | None = None
    finish_reason: str | None = None
    temperature: float | None = None
    max_output_tokens: int = Field(ge=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)
    provider_request_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_terminal_state(self) -> "ScoreOnlyJudgmentRecord":
        if self.status is PocqiJudgmentStatus.SUCCEEDED:
            if self.scores is None:
                raise ValueError("a succeeded judgment requires scores")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("a succeeded judgment cannot contain an error")
        elif self.scores is not None:
            raise ValueError("a failed judgment cannot contain scores")
        return self

    @staticmethod
    def build_judgment_key(
        *, experiment_id: str, question_id: str, generation_id: str, judge_model: str
    ) -> str:
        payload = json.dumps(
            {
                "experiment_id": experiment_id,
                "question_id": question_id,
                "generation_id": generation_id,
                "judge_model": judge_model,
                "prompt_template_id": PROMPT_TEMPLATE_ID,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
        return "__".join((experiment_id, question_id, judge_model, digest))


class ResumeTracker:
    """Thread-safe latest-success and attempt index."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.succeeded: dict[str, ScoreOnlyJudgmentRecord] = {}
        self.highest_attempt: dict[str, int] = {}
        if not path.exists():
            return
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = ScoreOnlyJudgmentRecord.model_validate_json(line)
                except ValueError as exc:
                    raise ValueError(f"invalid judgment at {path}:{line_number}: {exc}") from exc
                key = record.judgment_key
                self.highest_attempt[key] = max(
                    record.attempt, self.highest_attempt.get(key, 0)
                )
                if record.status is PocqiJudgmentStatus.SUCCEEDED:
                    previous = self.succeeded.get(key)
                    if previous is None or record.attempt >= previous.attempt:
                        self.succeeded[key] = record

    def lookup(self, key: str) -> tuple[ScoreOnlyJudgmentRecord | None, int]:
        with self._lock:
            return self.succeeded.get(key), self.highest_attempt.get(key, 0)

    def append(self, record: ScoreOnlyJudgmentRecord) -> None:
        line = record.model_dump_json() + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _APPEND_LOCK, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        with self._lock:
            key = record.judgment_key
            self.highest_attempt[key] = max(
                record.attempt, self.highest_attempt.get(key, 0)
            )
            if record.status is PocqiJudgmentStatus.SUCCEEDED:
                self.succeeded[key] = record


def _user_prompt(question_text: str, specialty: str, response_text: str) -> str:
    return f"""CLINICAL SPECIALTY:
{specialty}

CLINICAL QUESTION:
{question_text}

RESPONSE TO SCORE:
<candidate_response>
{response_text}
</candidate_response>

TASK:
{SCORE_ONLY_INSTRUCTION}"""


def judge_one_response(
    *,
    question_id: str,
    question_text: str,
    specialty: str,
    response: PocqiResponseInput,
    judge_model: str,
    experiment_id: str,
    run_id: str,
    tracker: ResumeTracker,
    model_caller: ModelCaller = call_model,
    temperature: float | None = None,
    max_output_tokens: int = 1024,
    retries: int = 2,
    retry_delay_seconds: float = 2.0,
    force: bool = False,
) -> ScoreOnlyJudgmentRecord:
    """Score one blinded answer, saving every attempt immediately."""

    provider, native_judge_model = resolve_provider(judge_model)
    key = ScoreOnlyJudgmentRecord.build_judgment_key(
        experiment_id=experiment_id,
        question_id=question_id,
        generation_id=response.generation_id,
        judge_model=native_judge_model,
    )
    succeeded, highest_attempt = tracker.lookup(key)
    if not force and succeeded is not None:
        return succeeded
    prompt = _user_prompt(question_text, specialty, response.response_text)
    last: ScoreOnlyJudgmentRecord | None = None
    for retry_index in range(retries + 1):
        started = time.perf_counter()
        common: dict[str, Any] = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "judgment_key": key,
            "judgment_id": f"judgment-{uuid.uuid4()}",
            "attempt": highest_attempt + retry_index + 1,
            "created_at": datetime.now(UTC),
            "question_id": question_id,
            "question_text": question_text,
            "specialty": specialty,
            "generation_id": response.generation_id,
            "generator_family": response.generator_family,
            "generator_model": response.generator_model,
            "judge_family": _judge_family(judge_model),
            "judge_model": native_judge_model,
            "system_prompt": RUBRIC_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        try:
            options: dict[str, Any] = {}
            if temperature is not None:
                options["temperature"] = temperature
            model_response = model_caller(
                judge_model,
                prompt,
                system=RUBRIC_SYSTEM_PROMPT,
                response_format=RealPocqiScores,
                max_output_tokens=max_output_tokens,
                **options,
            )
            if not isinstance(model_response.parsed, RealPocqiScores):
                raise TypeError("judge did not return RealPocqiScores")
            last = ScoreOnlyJudgmentRecord(
                **common,
                status=PocqiJudgmentStatus.SUCCEEDED,
                scores=model_response.parsed,
                judge_response_text=model_response.text or None,
                finish_reason=model_response.finish_reason,
                input_tokens=model_response.usage.input_tokens,
                output_tokens=model_response.usage.output_tokens,
                latency_ms=round((time.perf_counter() - started) * 1000),
                provider_request_id=model_response.request_id,
            )
        except Exception as exc:
            last = ScoreOnlyJudgmentRecord(
                **common,
                status=PocqiJudgmentStatus.FAILED,
                latency_ms=round((time.perf_counter() - started) * 1000),
                error_type=type(exc).__name__,
                error_message=str(exc) or repr(exc),
            )
        tracker.append(last)
        if last.status is PocqiJudgmentStatus.SUCCEEDED:
            return last
        if retry_index < retries and retry_delay_seconds:
            time.sleep(retry_delay_seconds * (2**retry_index))
    assert last is not None
    return last


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-generations", type=Path, default=DEFAULT_GENERATIONS_PATH)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--generator-models", nargs="+", default=DEFAULT_GENERATOR_MODELS)
    parser.add_argument("--judge-models", nargs="+", default=DEFAULT_JUDGE_MODELS)
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--question-sample-seed", type=int, default=DEFAULT_QUESTION_SAMPLE_SEED)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--max-concurrency", type=int, default=24)
    parser.add_argument("--openai-concurrency", type=int, default=8)
    parser.add_argument("--anthropic-concurrency", type=int, default=8)
    parser.add_argument("--gemini-concurrency", type=int, default=8)
    parser.add_argument("--modal-concurrency", type=int, default=16)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace, *, model_caller: ModelCaller = call_model) -> int:
    for name in (
        "num_questions", "max_output_tokens", "max_concurrency",
        "openai_concurrency", "anthropic_concurrency", "gemini_concurrency",
        "modal_concurrency",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.retries < 0 or args.retry_delay_seconds < 0:
        raise ValueError("retry settings cannot be negative")
    judges = validate_judge_models(args.judge_models)
    questions = load_latest_question_responses(
        args.input_generations,
        generator_models=args.generator_models,
        num_questions=args.num_questions,
        question_sample_seed=args.question_sample_seed,
    )
    run_id = args.run_id or (
        f"real-pocqi-score-only-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    tracker = ResumeTracker(args.output_path)
    jobs = []
    for question in questions:
        for judge in judges:
            _, native_judge = resolve_provider(judge)
            for response in question.responses:
                key = ScoreOnlyJudgmentRecord.build_judgment_key(
                    experiment_id=args.experiment_id,
                    question_id=question.question_id,
                    generation_id=response.generation_id,
                    judge_model=native_judge,
                )
                existing, _ = tracker.lookup(key)
                if args.force or existing is None:
                    jobs.append((question, judge, response))
    total = len(questions) * len(judges) * len(args.generator_models)
    skipped = total - len(jobs)
    print(
        f"Real-POCQi score-only {run_id}: {len(questions)} questions, "
        f"{len(judges)} judges, {len(args.generator_models)} answers each, "
        f"{total} logical judgments, {len(jobs)} pending, {skipped} skipped"
    )
    if args.dry_run:
        return 0
    limited = ProviderLimitedCaller(
        model_caller,
        {
            Provider.OPENAI: args.openai_concurrency,
            Provider.ANTHROPIC: args.anthropic_concurrency,
            Provider.GEMINI: args.gemini_concurrency,
            Provider.MODAL: args.modal_concurrency,
        },
    )
    succeeded = failed = completed = 0

    def execute(job: tuple[Any, str, PocqiResponseInput]) -> ScoreOnlyJudgmentRecord:
        question, judge, response = job
        return judge_one_response(
            question_id=question.question_id,
            question_text=question.question_text,
            specialty=question.specialty,
            response=response,
            judge_model=judge,
            experiment_id=args.experiment_id,
            run_id=run_id,
            tracker=tracker,
            model_caller=limited,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            retries=args.retries,
            retry_delay_seconds=args.retry_delay_seconds,
            force=args.force,
        )

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        futures = {executor.submit(execute, job): job for job in jobs}
        for future in as_completed(futures):
            record = future.result()
            completed += 1
            if record.status is PocqiJudgmentStatus.SUCCEEDED:
                succeeded += 1
            else:
                failed += 1
                print(
                    f"FAILED {record.question_id} {record.judge_model} "
                    f"{record.generator_model}: {record.error_type}: {record.error_message}"
                )
            if completed % 100 == 0 or completed == len(jobs):
                print(f"Progress: {completed}/{len(jobs)}; {succeeded} succeeded, {failed} failed")

    manifest = {
        "schema_version": "1.0",
        "experiment_id": args.experiment_id,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "input_generations": str(args.input_generations),
        "output_path": str(args.output_path),
        "question_count": len(questions),
        "question_ids": [question.question_id for question in questions],
        "question_sample_seed": args.question_sample_seed,
        "generator_models": list(args.generator_models),
        "judge_models": list(judges),
        "identity_blinded": True,
        "candidates_per_prompt": 1,
        "ranking_requested": False,
        "rubric_axes": list(RealPocqiScores.model_fields),
        "logical_judgments": {
            "total": total,
            "pending_at_start": len(jobs),
            "skipped_existing": skipped,
            "succeeded": succeeded,
            "failed": failed,
        },
    }
    manifest_path = args.output_path.parent / f"{run_id}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Complete: {succeeded} succeeded, {failed} failed; manifest={manifest_path}")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(args.env_file)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
