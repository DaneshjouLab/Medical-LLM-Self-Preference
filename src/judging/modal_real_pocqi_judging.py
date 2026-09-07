"""Run batched Real-POCQi judging with the pinned Qwen cohort on Modal.

The two vLLM engines stay warm and accept local judging calls through a small
thread-safe batcher. Both use official FP8 checkpoints, constrained JSON
generation, and thinking-disabled chat templates.

Examples:
    uv run modal run src/judging/modal_real_pocqi_judging.py \
      --judging-case rubric_and_model_ranking

    uv run modal run src/judging/modal_real_pocqi_judging.py \
      --judging-case direct_ranking --num-questions 100 \
      --question-sample-seed 20260824
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import modal

from inference import ModelResponse, Provider, TokenUsage, resolve_provider

from judging.real_pocqi import PocqiJudgingCase
from judging.run_real_pocqi_judging import (
    DEFAULT_GENERATIONS_PATH,
    DEFAULT_GENERATOR_MODELS,
    DEFAULT_OUTPUT_DIR,
    parse_args,
    run,
)
from judging.score_only_real_pocqi import (
    DEFAULT_EXPERIMENT_ID as DEFAULT_SCORE_ONLY_EXPERIMENT_ID,
    DEFAULT_OUTPUT_PATH as DEFAULT_SCORE_ONLY_OUTPUT_PATH,
    parse_args as parse_score_only_args,
    run as run_score_only,
)


APP_NAME = "medical-real-pocqi-qwen-judging"
QWEN_27B = "Qwen/Qwen3.8-27B-FP8"
QWEN_122B = "Qwen/Qwen3.5-122B-A10B-FP8"
MODEL_IDS = (QWEN_27B, QWEN_122B)
MINUTES = 60

app = modal.App(APP_NAME)

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.27.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_USE_DEEP_GEMM": "0"})
    .add_local_python_source("generation", "inference", "judging")
)
hf_cache = modal.Volume.from_name(
    "medical-llm-huggingface-cache",
    create_if_missing=True,
)
vllm_cache = modal.Volume.from_name(
    "medical-llm-vllm-cache",
    create_if_missing=True,
)
volumes = {
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/vllm": vllm_cache,
}


def _generate_batch(
    llm: Any,
    conversations: list[list[dict[str, str]]],
    *,
    response_schema: dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    import vllm
    from vllm.sampling_params import StructuredOutputsParams

    sampling = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        structured_outputs=StructuredOutputsParams(json=response_schema),
    )
    started = time.perf_counter()
    outputs = llm.chat(
        conversations,
        sampling_params=sampling,
        chat_template_kwargs={"enable_thinking": False},
    )
    return {
        "texts": [output.outputs[0].text.strip() for output in outputs],
        "input_tokens": [len(output.prompt_token_ids) for output in outputs],
        "output_tokens": [len(output.outputs[0].token_ids) for output in outputs],
        "finish_reasons": [output.outputs[0].finish_reason for output in outputs],
        "batch_latency_seconds": time.perf_counter() - started,
    }


def _start_llm(model: str, *, tensor_parallel_size: int = 1) -> Any:
    import vllm

    return vllm.LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        safetensors_load_strategy="prefetch",
        max_model_len=32768,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 0, "video": 0},
        enforce_eager=True,
        enable_prefix_caching=True,
    )


@app.cls(
    image=vllm_image,
    gpu="H100",
    max_containers=1,
    timeout=4 * MINUTES * 60,
    startup_timeout=MINUTES * 60,
    scaledown_window=5 * MINUTES,
    volumes=volumes,
)
class Qwen27BJudge:
    @modal.enter()
    def start(self) -> None:
        self.llm = _start_llm(QWEN_27B)

    @modal.method()
    def generate_batch(
        self,
        conversations: list[list[dict[str, str]]],
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        return _generate_batch(
            self.llm,
            conversations,
            response_schema=response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )


@app.cls(
    image=vllm_image,
    gpu="H100:2",
    max_containers=1,
    timeout=4 * MINUTES * 60,
    startup_timeout=MINUTES * 60,
    scaledown_window=5 * MINUTES,
    volumes=volumes,
)
class Qwen122BJudge:
    @modal.enter()
    def start(self) -> None:
        self.llm = _start_llm(QWEN_122B, tensor_parallel_size=2)

    @modal.method()
    def generate_batch(
        self,
        conversations: list[list[dict[str, str]]],
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        return _generate_batch(
            self.llm,
            conversations,
            response_schema=response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )


@dataclass(slots=True)
class _QueuedCall:
    model: str
    user_input: str
    system: str | None
    response_format: type[Any]
    max_output_tokens: int
    temperature: float
    completed: threading.Event = field(default_factory=threading.Event)
    response: ModelResponse[Any] | None = None
    error: BaseException | None = None

    @property
    def signature(self) -> tuple[type[Any], int, float]:
        return (
            self.response_format,
            self.max_output_tokens,
            self.temperature,
        )


class ModalBatchCaller:
    """Adapt synchronous judging calls to batched Modal class methods."""

    def __init__(
        self,
        remotes: dict[str, Any],
        *,
        batch_size: int = 8,
        flush_seconds: float = 0.1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if flush_seconds < 0:
            raise ValueError("flush_seconds cannot be negative")
        self._remotes = remotes
        self._batch_size = batch_size
        self._flush_seconds = flush_seconds
        self._closed = False
        self._lock = threading.Lock()
        self._queues: dict[str, queue.Queue[_QueuedCall | None]] = {
            model: queue.Queue() for model in remotes
        }
        self._workers = {
            model: threading.Thread(
                target=self._worker,
                args=(model,),
                name=f"modal-judge-batcher-{model.rsplit('/', 1)[-1]}",
                daemon=True,
            )
            for model in remotes
        }
        for worker in self._workers.values():
            worker.start()

    def __call__(self, model: str, input: str, **kwargs: Any) -> ModelResponse[Any]:
        provider, native_model = resolve_provider(model)
        if provider is not Provider.MODAL:
            raise ValueError(f"ModalBatchCaller cannot call {provider.value}")
        try:
            request_queue = self._queues[native_model]
        except KeyError as exc:
            raise ValueError(f"unsupported Modal judge model: {native_model}") from exc
        response_format = kwargs.get("response_format")
        if response_format is None or not hasattr(response_format, "model_validate_json"):
            raise TypeError("a Pydantic response_format is required")
        max_output_tokens = kwargs.get("max_output_tokens")
        if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        temperature = kwargs.get("temperature")
        request = _QueuedCall(
            model=native_model,
            user_input=input,
            system=kwargs.get("system"),
            response_format=response_format,
            max_output_tokens=max_output_tokens,
            temperature=0.0 if temperature is None else float(temperature),
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("ModalBatchCaller is closed")
            request_queue.put(request)
        request.completed.wait()
        if request.error is not None:
            raise request.error
        assert request.response is not None
        return request.response

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for request_queue in self._queues.values():
                request_queue.put(None)
        for worker in self._workers.values():
            worker.join()

    def _worker(self, model: str) -> None:
        request_queue = self._queues[model]
        while True:
            first = request_queue.get()
            if first is None:
                return
            batch = [first]
            deadline = time.monotonic() + self._flush_seconds
            while len(batch) < self._batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    candidate = request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if candidate is None:
                    request_queue.put(None)
                    break
                if candidate.signature != first.signature:
                    candidate.error = RuntimeError(
                        "a Modal judging batch mixed incompatible output settings"
                    )
                    candidate.completed.set()
                    continue
                batch.append(candidate)
            self._dispatch(model, batch)

    def _dispatch(self, model: str, batch: Sequence[_QueuedCall]) -> None:
        first = batch[0]
        conversations = []
        for request in batch:
            messages = []
            if request.system is not None:
                messages.append({"role": "system", "content": request.system})
            messages.append({"role": "user", "content": request.user_input})
            conversations.append(messages)
        started = time.perf_counter()
        try:
            result = self._remotes[model].generate_batch.remote(
                conversations,
                first.response_format.model_json_schema(),
                first.temperature,
                first.max_output_tokens,
            )
            expected = len(batch)
            fields = ("texts", "input_tokens", "output_tokens", "finish_reasons")
            if any(len(result[field]) != expected for field in fields):
                raise RuntimeError("Modal judging returned an unexpected batch size")
            batch_latency = float(
                result.get("batch_latency_seconds", time.perf_counter() - started)
            )
            for index, request in enumerate(batch):
                text = str(result["texts"][index]).strip()
                parsed = request.response_format.model_validate_json(text)
                request.response = ModelResponse(
                    text=text,
                    parsed=parsed,
                    provider=Provider.MODAL,
                    model=model,
                    request_id=f"modal-batch-{uuid.uuid4()}",
                    finish_reason=str(result["finish_reasons"][index]),
                    usage=TokenUsage(
                        input_tokens=int(result["input_tokens"][index]),
                        output_tokens=int(result["output_tokens"][index]),
                    ),
                    raw={"batch_latency_seconds": batch_latency},
                )
        except BaseException as exc:
            for request in batch:
                request.error = exc
        finally:
            for request in batch:
                request.completed.set()


def _runner_argv(
    *,
    input_generations: str,
    output_dir: str,
    experiment_id: str,
    run_id: str,
    judging_case: str,
    models: Sequence[str],
    num_questions: int | None,
    question_sample_seed: int | None,
    presentation_seed: int,
    max_output_tokens: int,
    max_concurrency: int,
    retries: int,
    retry_delay_seconds: float,
    reveal_generator_identities: bool,
    force: bool,
) -> list[str]:
    argv = [
        "--input-generations",
        input_generations,
        "--output-dir",
        output_dir,
        "--generator-models",
        *DEFAULT_GENERATOR_MODELS,
        "--judge-models",
        *(f"modal/{model}" for model in models),
        "--judging-cases",
        judging_case,
        "--experiment-id",
        experiment_id,
        "--presentation-seed",
        str(presentation_seed),
        "--temperature",
        "0",
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
    if num_questions is not None:
        argv.extend(("--num-questions", str(num_questions)))
    if question_sample_seed is not None:
        argv.extend(("--question-sample-seed", str(question_sample_seed)))
    if reveal_generator_identities:
        argv.append("--reveal-generator-identities")
    if force:
        argv.append("--force")
    return argv


def _score_only_runner_argv(
    *,
    input_generations: str,
    output_path: str,
    experiment_id: str,
    run_id: str,
    models: Sequence[str],
    num_questions: int,
    question_sample_seed: int,
    max_output_tokens: int,
    max_concurrency: int,
    retries: int,
    retry_delay_seconds: float,
    force: bool,
) -> list[str]:
    argv = [
        "--input-generations", input_generations,
        "--output-path", output_path,
        "--generator-models", *DEFAULT_GENERATOR_MODELS,
        "--judge-models", *(f"modal/{model}" for model in models),
        "--num-questions", str(num_questions),
        "--question-sample-seed", str(question_sample_seed),
        "--experiment-id", experiment_id,
        "--temperature", "0",
        "--max-output-tokens", str(max_output_tokens),
        "--max-concurrency", str(max_concurrency),
        "--modal-concurrency", str(max_concurrency),
        "--retries", str(retries),
        "--retry-delay-seconds", str(retry_delay_seconds),
    ]
    if run_id:
        argv.extend(("--run-id", run_id))
    if force:
        argv.append("--force")
    return argv


@app.local_entrypoint()
def main(
    input_generations: str = str(DEFAULT_GENERATIONS_PATH),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    experiment_id: str = "real_pocqi_combined_all_judges_v1",
    run_id: str = "",
    judging_case: str = PocqiJudgingCase.RUBRIC_AND_MODEL_RANKING.value,
    models: str = ",".join(MODEL_IDS),
    num_questions: int = 620,
    question_sample_seed: int = -1,
    presentation_seed: int = 42,
    max_output_tokens: int = 2048,
    batch_size: int = 8,
    max_concurrency: int = 32,
    retries: int = 2,
    retry_delay_seconds: float = 2.0,
    direct_num_questions: int = 0,
    direct_question_sample_seed: int = 20260824,
    direct_experiment_id: str = "real_pocqi_direct_ranking_random100_v1",
    direct_run_id: str = "",
    direct_max_output_tokens: int = 1024,
    reveal_generator_identities: bool = False,
    score_only: bool = False,
    score_only_output_path: str = str(DEFAULT_SCORE_ONLY_OUTPUT_PATH),
    score_only_experiment_id: str = DEFAULT_SCORE_ONLY_EXPERIMENT_ID,
    force: bool = False,
) -> None:
    try:
        selected_case = PocqiJudgingCase(judging_case)
    except ValueError as exc:
        raise ValueError(f"unsupported judging case: {judging_case}") from exc
    selected_models = tuple(item.strip() for item in models.split(",") if item.strip())
    if not selected_models:
        raise ValueError("at least one model is required")
    unknown_models = set(selected_models) - set(MODEL_IDS)
    if unknown_models:
        raise ValueError(f"unsupported models: {sorted(unknown_models)}")
    if num_questions <= 0:
        raise ValueError("num_questions must be positive")
    if direct_num_questions < 0:
        raise ValueError("direct_num_questions cannot be negative")
    if direct_num_questions and direct_max_output_tokens <= 0:
        raise ValueError("direct_max_output_tokens must be positive")

    remotes = {
        QWEN_27B: Qwen27BJudge(),
        QWEN_122B: Qwen122BJudge(),
    }
    caller = ModalBatchCaller(
        {model: remotes[model] for model in selected_models},
        batch_size=batch_size,
    )
    workloads = [
        (
            experiment_id,
            run_id,
            selected_case.value,
            num_questions,
            None if question_sample_seed < 0 else question_sample_seed,
            max_output_tokens,
        )
    ]
    if direct_num_questions:
        workloads.append(
            (
                direct_experiment_id,
                direct_run_id,
                PocqiJudgingCase.DIRECT_RANKING.value,
                direct_num_questions,
                direct_question_sample_seed,
                direct_max_output_tokens,
            )
        )
    exit_codes: list[int] = []
    try:
        if score_only:
            args = parse_score_only_args(
                _score_only_runner_argv(
                    input_generations=input_generations,
                    output_path=score_only_output_path,
                    experiment_id=score_only_experiment_id,
                    run_id=run_id,
                    models=selected_models,
                    num_questions=num_questions,
                    question_sample_seed=(
                        42 if question_sample_seed < 0 else question_sample_seed
                    ),
                    max_output_tokens=max_output_tokens,
                    max_concurrency=max_concurrency,
                    retries=retries,
                    retry_delay_seconds=retry_delay_seconds,
                    force=force,
                )
            )
            exit_codes.append(run_score_only(args, model_caller=caller))
            workloads = []
        for (
            workload_experiment_id,
            workload_run_id,
            workload_case,
            workload_num_questions,
            workload_sample_seed,
            workload_max_output_tokens,
        ) in workloads:
            args = parse_args(
                _runner_argv(
                    input_generations=input_generations,
                    output_dir=output_dir,
                    experiment_id=workload_experiment_id,
                    run_id=workload_run_id,
                    judging_case=workload_case,
                    models=selected_models,
                    num_questions=workload_num_questions,
                    question_sample_seed=workload_sample_seed,
                    presentation_seed=presentation_seed,
                    max_output_tokens=workload_max_output_tokens,
                    max_concurrency=max_concurrency,
                    retries=retries,
                    retry_delay_seconds=retry_delay_seconds,
                    reveal_generator_identities=reveal_generator_identities,
                    force=force,
                )
            )
            exit_codes.append(run(args, model_caller=caller))
    finally:
        caller.close()
    if any(exit_codes):
        raise RuntimeError("one or more Modal judgments failed after retries")
