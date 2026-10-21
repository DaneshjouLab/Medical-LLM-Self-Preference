"""Run blinded rubric-and-ranking judging across MedSP1000 trajectories."""

from __future__ import annotations

import argparse
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from inference import Provider, call_model

from generation.generate_real_pocqi import load_dotenv

from .judge_medsp1000 import (
    MEDSP1000_JUDGING_CASES,
    build_medsp1000_judgment_keys,
    judge_medsp1000_trajectories,
)
from .judge_real_pocqi import (
    PocqiJudgingSettings,
    PocqiResponseInput,
    PocqiResumeTracker,
)
from .real_pocqi import PocqiJudgingCase, PocqiJudgmentStatus
from .run_real_pocqi_judging import (
    ModelCaller,
    ProviderLimitedCaller,
    validate_judge_models,
)


DEFAULT_GENERATIONS_PATH = Path("data/outputs/medsp1000/generations.jsonl")
DEFAULT_QUESTIONS_PATH = Path("data/question_sets/medsp1000_generation_cases.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/outputs/medsp1000/judgements")
DEFAULT_EXPERIMENT_ID = "medsp1000_all_judges_v1"

DEFAULT_GENERATOR_MODELS = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.7-flash",
    "Qwen/Qwen3.5-122B-A10B-FP8",
    "Qwen/Qwen3.8-27B-FP8",
)
DEFAULT_JUDGE_MODELS = DEFAULT_GENERATOR_MODELS


@dataclass(frozen=True, slots=True)
class MedspQuestionTrajectories:
    """Complete candidate set for one standardized-patient scenario."""

    question_id: str
    question_text: str
    source_scenario_path: str
    responses: tuple[PocqiResponseInput, ...]


@dataclass(frozen=True, slots=True)
class PlannedJob:
    """One scenario judged by one model across pending conditions."""

    question: MedspQuestionTrajectories
    judge_model: str
    pending_cases: tuple[PocqiJudgingCase, ...]


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def default_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"medsp1000-judging-{timestamp}-{uuid.uuid4().hex[:8]}"


def output_paths(
    output_dir: Path,
    *,
    view_turn_count: int | None = None,
) -> dict[PocqiJudgingCase, Path]:
    suffix = "" if view_turn_count is None else f"_{view_turn_count}_turns"
    return {
        PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING: (
            output_dir / f"rubric_and_model_ranking{suffix}.jsonl"
        ),
    }


def _generator_family(model: str) -> str:
    lowered = model.casefold()
    if lowered.startswith("gpt-"):
        return "openai"
    if lowered.startswith("claude-"):
        return "anthropic"
    if lowered.startswith("gemini-"):
        return "google"
    if lowered.startswith("qwen"):
        return "qwen"
    return model.partition("/")[0] or model


def _question_order(path: Path, *, num_questions: int | None) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                question_id = json.loads(line)["question_id"]
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid question at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"invalid question_id at {path}:{line_number}")
            if question_id in seen:
                raise ValueError(f"duplicate question_id {question_id!r} in {path}")
            ordered.append(question_id)
            seen.add(question_id)
    if num_questions is not None:
        if num_questions <= 0:
            raise ValueError("num_questions must be positive")
        ordered = ordered[:num_questions]
    return ordered


def load_latest_question_trajectories(
    generations_path: Path,
    *,
    questions_path: Path,
    generator_models: Sequence[str],
    num_questions: int | None = None,
    exchange_count: int = 4,
    reasoning_effort: str = "medium",
    view_turn_count: int | None = None,
) -> list[MedspQuestionTrajectories]:
    """Load one successful, configuration-matched trajectory per model cell."""

    requested_models = tuple(generator_models)
    if not requested_models:
        raise ValueError("at least one generator model is required")
    if len(set(requested_models)) != len(requested_models):
        raise ValueError("generator models must be unique")
    if exchange_count <= 0:
        raise ValueError("exchange_count must be positive")
    full_turn_count = exchange_count * 2
    if view_turn_count is not None and (
        view_turn_count <= 0
        or view_turn_count % 2
        or view_turn_count > full_turn_count
    ):
        raise ValueError(
            "view_turn_count must be a positive even integer no greater than "
            f"{full_turn_count}"
        )
    question_order = _question_order(questions_path, num_questions=num_questions)
    requested_questions = set(question_order)
    requested_set = set(requested_models)
    latest: dict[tuple[str, str], dict[str, Any]] = {}

    with generations_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid generation at {generations_path}:{line_number}: {exc}"
                ) from exc
            if (
                record.get("status") != "succeeded"
                or record.get("clinician_model") not in requested_set
                or record.get("question_id") not in requested_questions
                or record.get("exchange_count") != exchange_count
                or record.get("clinician_reasoning_effort")
                != (
                    None
                    if _generator_family(record["clinician_model"]) == "qwen"
                    else reasoning_effort
                )
            ):
                continue
            if record.get("turn_count") != full_turn_count:
                raise ValueError(
                    f"unexpected turn count at {generations_path}:{line_number}"
                )
            transcript = record.get("transcript_text")
            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError(
                    f"missing transcript_text at {generations_path}:{line_number}"
                )
            turns = record.get("turns")
            if view_turn_count is not None and (
                not isinstance(turns, list) or len(turns) != full_turn_count
            ):
                raise ValueError(
                    f"missing or invalid turns at {generations_path}:{line_number}"
                )
            key = (record["question_id"], record["clinician_model"])
            previous = latest.get(key)
            if previous is not None and previous.get("generation_key") != record.get(
                "generation_key"
            ):
                raise ValueError(
                    "multiple successful generation configurations found for "
                    f"{key}; narrow the generation selection before judging"
                )
            if previous is None or record.get("attempt", 0) >= previous.get("attempt", 0):
                latest[key] = record

    if not latest:
        raise ValueError(
            f"no requested successful MedSP1000 generations found in {generations_path}"
        )

    incomplete: dict[str, list[str]] = {}
    questions: list[MedspQuestionTrajectories] = []
    for question_id in question_order:
        missing = [
            model for model in requested_models if (question_id, model) not in latest
        ]
        if missing:
            incomplete[question_id] = missing
            continue
        records = [latest[(question_id, model)] for model in requested_models]
        for field in (
            "question_text",
            "source_scenario_path",
            "patient_model",
            "patient_prompt_template_id",
            "prompt_version",
            "exchange_count",
        ):
            if len({record.get(field) for record in records}) != 1:
                raise ValueError(f"inconsistent {field} for {question_id}")
        questions.append(
            MedspQuestionTrajectories(
                question_id=question_id,
                question_text=records[0]["question_text"],
                source_scenario_path=records[0]["source_scenario_path"],
                responses=tuple(
                    PocqiResponseInput(
                        generation_id=record["generation_id"],
                        generator_family=_generator_family(record["clinician_model"]),
                        generator_model=record["clinician_model"],
                        response_text=(
                            record["transcript_text"]
                            if view_turn_count is None
                            else "\n".join(
                                f"{turn['role'].upper()}: {turn['content']}"
                                for turn in record["turns"][:view_turn_count]
                            )
                        ),
                    )
                    for record in records
                ),
            )
        )
    if incomplete:
        sample = dict(list(incomplete.items())[:5])
        raise ValueError(
            f"{len(incomplete)} scenarios have incomplete generation coverage; "
            f"first missing cells: {sample}"
        )
    if not questions:
        raise ValueError("no complete scenarios available for judging")
    return questions


def plan_jobs(
    *,
    questions: Sequence[MedspQuestionTrajectories],
    judge_models: Sequence[str],
    settings: PocqiJudgingSettings,
    paths: dict[PocqiJudgingCase, Path],
    tracker: PocqiResumeTracker,
    view_turn_count: int | None = None,
) -> list[PlannedJob]:
    jobs: list[PlannedJob] = []
    for question in questions:
        for judge_model in judge_models:
            keys = build_medsp1000_judgment_keys(
                question_id=question.question_id,
                responses=question.responses,
                judge_model=judge_model,
                settings=settings,
                view_turn_count=view_turn_count,
            )
            pending = tuple(
                case
                for case, key in keys.items()
                if settings.force or tracker.lookup(paths[case], key)[0] is None
            )
            jobs.append(
                PlannedJob(
                    question=question,
                    judge_model=judge_model,
                    pending_cases=pending,
                )
            )
    return jobs


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    run_id: str,
    questions: Sequence[MedspQuestionTrajectories],
    judge_models: Sequence[str],
    counts: dict[str, int],
) -> None:
    manifest = {
        "schema_version": "1.0",
        "experiment_id": args.experiment_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "input_generations": str(args.input_generations),
        "input_questions": str(args.input_questions),
        "output_dir": str(args.output_dir),
        "question_count": len(questions),
        "question_ids": [question.question_id for question in questions],
        "generator_models": list(args.generator_models),
        "judge_models": list(judge_models),
        "judging_cases": [case.value for case in MEDSP1000_JUDGING_CASES],
        "exchange_count": args.exchange_count,
        "view_turn_count": args.view_turn_count,
        "reasoning_effort": args.reasoning_effort,
        "presentation_seed": args.presentation_seed,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "retries": args.retries,
        "retry_delay_seconds": args.retry_delay_seconds,
        "force": args.force,
        "provider_concurrency": {
            "openai": args.openai_concurrency,
            "anthropic": args.anthropic_concurrency,
            "gemini": args.gemini_concurrency,
            "modal": args.modal_concurrency,
        },
        "logical_judgments": counts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-generations", type=Path, default=DEFAULT_GENERATIONS_PATH
    )
    parser.add_argument("--input-questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generator-models", nargs="+", default=DEFAULT_GENERATOR_MODELS)
    parser.add_argument("--judge-models", nargs="+", default=DEFAULT_JUDGE_MODELS)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--exchange-count", type=int, default=4)
    parser.add_argument(
        "--view-turn-count",
        type=int,
        default=None,
        help="Judge only the first N role turns; defaults to the full trajectory.",
    )
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--presentation-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-concurrency", type=int, default=6)
    parser.add_argument("--openai-concurrency", type=int, default=2)
    parser.add_argument("--anthropic-concurrency", type=int, default=2)
    parser.add_argument("--gemini-concurrency", type=int, default=2)
    parser.add_argument("--modal-concurrency", type=int, default=2)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace, *, model_caller: ModelCaller = call_model) -> int:
    """Execute or dry-run the MedSP1000 judging matrix."""

    for name in (
        "max_concurrency",
        "openai_concurrency",
        "anthropic_concurrency",
        "gemini_concurrency",
        "modal_concurrency",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative")
    if args.retry_delay_seconds < 0:
        raise ValueError("--retry-delay-seconds cannot be negative")

    judge_models = validate_judge_models(args.judge_models)
    questions = load_latest_question_trajectories(
        args.input_generations,
        questions_path=args.input_questions,
        generator_models=args.generator_models,
        num_questions=args.num_questions,
        exchange_count=args.exchange_count,
        reasoning_effort=args.reasoning_effort,
        view_turn_count=args.view_turn_count,
    )
    run_id = args.run_id or default_run_id()
    settings = PocqiJudgingSettings(
        experiment_id=args.experiment_id,
        run_id=run_id,
        presentation_seed=args.presentation_seed,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        retries=args.retries,
        retry_delay_seconds=args.retry_delay_seconds,
        force=args.force,
    )
    paths = output_paths(
        args.output_dir,
        view_turn_count=args.view_turn_count,
    )
    tracker = PocqiResumeTracker(list(paths.values()))
    jobs = plan_jobs(
        questions=questions,
        judge_models=judge_models,
        settings=settings,
        paths=paths,
        tracker=tracker,
        view_turn_count=args.view_turn_count,
    )
    total_logical = len(jobs)
    pending_logical = sum(len(job.pending_cases) for job in jobs)
    skipped = total_logical - pending_logical
    pending_jobs = [job for job in jobs if job.pending_cases]
    print(
        f"MedSP1000 judging {run_id}: {len(questions)} scenarios, "
        f"{len(judge_models)} judges, {total_logical} logical judgments, "
        f"{pending_logical} pending, {skipped} skipped"
    )
    if args.dry_run:
        return 0

    limited_caller = ProviderLimitedCaller(
        model_caller,
        {
            Provider.OPENAI: args.openai_concurrency,
            Provider.ANTHROPIC: args.anthropic_concurrency,
            Provider.GEMINI: args.gemini_concurrency,
            Provider.MODAL: args.modal_concurrency,
        },
    )
    succeeded = failed = completed_jobs = 0

    def execute(job: PlannedJob):
        question = job.question
        return job, judge_medsp1000_trajectories(
            question_id=question.question_id,
            question_text=question.question_text,
            source_scenario_path=question.source_scenario_path,
            responses=question.responses,
            judge_model=job.judge_model,
            settings=settings,
            view_turn_count=args.view_turn_count,
            model_caller=limited_caller,
            output_paths=paths,
            resume_tracker=tracker,
        )

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        futures = {executor.submit(execute, job): job for job in pending_jobs}
        for future in as_completed(futures):
            job, records = future.result()
            for case in job.pending_cases:
                if records[case].status is PocqiJudgmentStatus.SUCCEEDED:
                    succeeded += 1
                else:
                    failed += 1
                    record = records[case]
                    print(
                        f"FAILED {job.question.question_id} {job.judge_model} "
                        f"{case.value}: {record.error_type}: {record.error_message}"
                    )
            completed_jobs += 1
            if completed_jobs % 25 == 0 or completed_jobs == len(pending_jobs):
                print(
                    f"Progress: {completed_jobs}/{len(pending_jobs)} jobs; "
                    f"{succeeded} judgments succeeded, {failed} failed"
                )

    counts = {
        "total": total_logical,
        "pending_at_start": pending_logical,
        "skipped_existing": skipped,
        "succeeded": succeeded,
        "failed": failed,
    }
    manifest_path = args.output_dir / f"{run_id}.manifest.json"
    write_manifest(
        manifest_path,
        args=args,
        run_id=run_id,
        questions=questions,
        judge_models=judge_models,
        counts=counts,
    )
    print(
        f"Complete: {succeeded} succeeded, {failed} failed, {skipped} skipped; "
        f"manifest={manifest_path}"
    )
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
