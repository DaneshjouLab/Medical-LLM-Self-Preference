"""Run batched MedSP1000 judging with the pinned Qwen cohort on Modal."""

from __future__ import annotations

from typing import Sequence

from judging.modal_real_pocqi_judging import (
    MODEL_IDS,
    QWEN_27B,
    QWEN_122B,
    ModalBatchCaller,
    Qwen27BJudge,
    Qwen122BJudge,
    app,
)
from judging.run_medsp1000_judging import (
    DEFAULT_GENERATIONS_PATH,
    DEFAULT_GENERATOR_MODELS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUESTIONS_PATH,
    parse_args,
    run,
)


def _runner_argv(
    *,
    input_generations: str,
    input_questions: str,
    output_dir: str,
    experiment_id: str,
    run_id: str,
    models: Sequence[str],
    num_questions: int,
    view_turn_count: int | None,
    presentation_seed: int,
    temperature: float,
    max_output_tokens: int,
    max_concurrency: int,
    retries: int,
    retry_delay_seconds: float,
    force: bool,
) -> list[str]:
    argv = [
        "--input-generations",
        input_generations,
        "--input-questions",
        input_questions,
        "--output-dir",
        output_dir,
        "--generator-models",
        *DEFAULT_GENERATOR_MODELS,
        "--judge-models",
        *(f"modal/{model}" for model in models),
        "--num-questions",
        str(num_questions),
        "--experiment-id",
        experiment_id,
        "--presentation-seed",
        str(presentation_seed),
        "--temperature",
        str(temperature),
        "--max-output-tokens",
        str(max_output_tokens),
        "--max-concurrency",
        str(max_concurrency),
        "--modal-concurrency",
        str(max_concurrency),
        "--retries",
        str(retries),
        "--retry-delay-seconds",
        str(retry_delay_seconds),
    ]
    if run_id:
        argv.extend(("--run-id", run_id))
    if view_turn_count is not None:
        argv.extend(("--view-turn-count", str(view_turn_count)))
    if force:
        argv.append("--force")
    return argv


@app.local_entrypoint(name="medsp_main")
def main(
    input_generations: str = str(DEFAULT_GENERATIONS_PATH),
    input_questions: str = str(DEFAULT_QUESTIONS_PATH),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    experiment_id: str = "medsp1000_all_eight_judges_v2",
    run_id: str = "",
    models: str = ",".join(MODEL_IDS),
    num_questions: int = 200,
    view_turn_counts: str = "0,2,4,6",
    presentation_seed: int = 42,
    temperature: float = 0.0,
    max_output_tokens: int = 8192,
    batch_size: int = 8,
    max_concurrency: int = 32,
    retries: int = 2,
    retry_delay_seconds: float = 2.0,
    force: bool = False,
) -> None:
    selected_models = tuple(item.strip() for item in models.split(",") if item.strip())
    if not selected_models:
        raise ValueError("at least one model is required")
    unknown_models = set(selected_models) - set(MODEL_IDS)
    if unknown_models:
        raise ValueError(f"unsupported models: {sorted(unknown_models)}")
    if num_questions <= 0:
        raise ValueError("num_questions must be positive")
    selected_views = tuple(
        None if value == 0 else value
        for value in (
            int(item.strip()) for item in view_turn_counts.split(",") if item.strip()
        )
    )
    if not selected_views:
        raise ValueError("at least one view turn count is required")
    if len(set(selected_views)) != len(selected_views):
        raise ValueError("view turn counts must be unique")

    remotes = {
        QWEN_27B: Qwen27BJudge(),
        QWEN_122B: Qwen122BJudge(),
    }
    caller = ModalBatchCaller(
        {model: remotes[model] for model in selected_models},
        batch_size=batch_size,
    )
    try:
        exit_codes = []
        for selected_view in selected_views:
            args = parse_args(
                _runner_argv(
                    input_generations=input_generations,
                    input_questions=input_questions,
                    output_dir=output_dir,
                    experiment_id=experiment_id,
                    run_id=run_id,
                    models=selected_models,
                    num_questions=num_questions,
                    view_turn_count=selected_view,
                    presentation_seed=presentation_seed,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    max_concurrency=max_concurrency,
                    retries=retries,
                    retry_delay_seconds=retry_delay_seconds,
                    force=force,
                )
            )
            exit_codes.append(run(args, model_caller=caller))
    finally:
        caller.close()
    if any(exit_codes):
        raise RuntimeError("one or more Modal judgments failed after retries")
