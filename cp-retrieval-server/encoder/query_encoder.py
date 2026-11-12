from __future__ import annotations

import os
import threading
from typing import Sequence

import numpy as np


class QueryEncoder:
    """Query-only adapter with matching tokenization and float32 outputs."""

    def __init__(
        self,
        model_path: str,
        backend: str,
        max_seq_length: int = 2048,
        use_bf16: bool = True,
        engine_max_seq_length: int | None = None,
    ) -> None:
        self.capacity = engine_max_seq_length or max_seq_length
        if not 1 <= max_seq_length <= self.capacity:
            raise ValueError("Query length must fit within encoder capacity")
        if backend not in {"sentence_transformers", "vllm"}:
            raise ValueError(f"Unknown ENCODER_BACKEND: {backend!r}")
        self.backend = backend
        self.max_seq_length = max_seq_length
        self._lock = threading.RLock()

        if backend == "sentence_transformers":
            import torch
            from sentence_transformers import SentenceTransformer

            kwargs = {"torch_dtype": torch.bfloat16} if use_bf16 else {}
            self._model = SentenceTransformer(
                model_path,
                trust_remote_code=True,
                model_kwargs=kwargs,
            )
            self._model.max_seq_length = max_seq_length
            self.tokenizer = self._model.tokenizer
        else:
            from transformers import AutoConfig, AutoTokenizer
            from vllm.config import PoolerConfig

            self._vllm_mode = os.getenv("VLLM_MODE", "async").strip().lower()
            if self._vllm_mode not in {"async", "sync"}:
                raise ValueError("VLLM_MODE must be async or sync")

            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            native_capacity = getattr(config, "max_position_embeddings", 0)
            if self.capacity > native_capacity:
                raise ValueError(f"Encoder capacity exceeds native model limit {native_capacity}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self._engine_options = dict(
                model=model_path,
                runner="pooling",
                convert="embed",
                trust_remote_code=True,
                dtype="bfloat16" if use_bf16 else "float32",
                max_model_len=self.capacity,
                hf_overrides={"model_max_length": native_capacity},
                max_num_seqs=int(os.getenv("VLLM_MAX_NUM_SEQS", "2")),
                max_num_batched_tokens=int(os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "2048")),
                gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.8")),
                enforce_eager=os.getenv("VLLM_ENFORCE_EAGER", "false").lower()
                in {"1", "true", "yes", "on"},
                enable_chunked_prefill=True,
                enable_prefix_caching=False,
                async_scheduling=False,
                pooler_config=PoolerConfig(pooling_type="LAST", use_activation=True),
            )
            if self._vllm_mode == "sync":
                from vllm import LLM

                self._model = LLM(**self._engine_options)
            else:
                from vllm import AsyncEngineArgs, AsyncLLMEngine, PoolingParams
                from encoder.async_pooler import AsyncPooler

                self._engine_options.update(
                    scheduling_policy="priority",
                    long_prefill_token_threshold=int(os.getenv("VLLM_LONG_PREFILL_THRESHOLD", "512")),
                    long_prefill_token_threshold_adaptive=False,
                )
                engine_args = AsyncEngineArgs(**self._engine_options, enable_log_requests=False)
                self._async_pooler = AsyncPooler(
                    lambda: AsyncLLMEngine.from_engine_args(engine_args),
                    lambda: PoolingParams(task="embed", use_activation=True),
                    max_pending=int(os.getenv("ENCODER_MAX_PENDING", "32")),
                    timeout=float(os.getenv("ENCODER_TIMEOUT_SECONDS", "120")),
                )
        self.tokenizer.model_max_length = max_seq_length

    def set_max_seq_length(self, length: int) -> None:
        if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= self.capacity:
            raise ValueError(f"Query length must be between 1 and {self.capacity}")
        with self._lock:
            self.max_seq_length = length
            self.tokenizer.model_max_length = length
            if self.backend == "sentence_transformers":
                self._model.max_seq_length = length

    def token_ids(self, text: str) -> list[int]:
        with self._lock:
            return self.tokenizer(
                text.strip(),
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_seq_length,
            )["input_ids"]

    def count_tokens(self, text: str) -> int:
        with self._lock:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_length,
            )
            return len(encoded["input_ids"])

    def full_token_count(self, text: str) -> int:
        with self._lock:
            return len(self.tokenizer(text.strip(), add_special_tokens=True,
                                      truncation=False)["input_ids"])

    def encode_at_length(self, text: str, length: int) -> np.ndarray:
        if type(length) is not int or not 1 <= length <= self.capacity:
            raise ValueError(f"Query length must be between 1 and {self.capacity}")
        if getattr(self, "_vllm_mode", "sync") == "async":
            return self._encode_async([text], length, priority=10)[0]
        with self._lock:
            previous = self.max_seq_length
            self.set_max_seq_length(length)
            try:
                return self.encode(text)
            finally:
                self.set_max_seq_length(previous)

    def _encode_async(self, texts: Sequence[str], length: int | None = None,
                      priority: int = 0) -> np.ndarray:
        with self._lock:
            limit = self.max_seq_length if length is None else length
            prompts = [self.tokenizer(text.strip(), add_special_tokens=True,
                                      truncation=True, max_length=limit)["input_ids"] for text in texts]
        return self._normalise(self._async_pooler.encode_batch(prompts, priority=priority))

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("At least one query is required")
        if getattr(self, "_vllm_mode", "sync") == "async":
            return self._encode_async(texts)
        texts = [text.strip() for text in texts]
        with self._lock:
            if self.backend == "sentence_transformers":
                import torch

                embeddings = self._model.encode(
                    texts,
                    batch_size=4,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                ).to(torch.float32).cpu().numpy()
            else:
                prompts = [{"prompt_token_ids": self.token_ids(text)} for text in texts]
                outputs = self._model.embed(prompts, use_tqdm=False)
                embeddings = np.asarray(
                    [output.outputs.embedding for output in outputs], dtype=np.float32
                )
        return self._normalise(embeddings)

    @staticmethod
    def _normalise(embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.all(np.isfinite(embeddings)) or np.any(norms <= 0):
            raise ValueError("Encoder returned invalid query embeddings")
        return embeddings / norms

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def status(self) -> dict:
        result = {"backend": self.backend, "mode": getattr(self, "_vllm_mode", "sync")}
        if hasattr(self, "_engine_options"):
            result.update({key: self._engine_options[key] for key in
                           ("max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill")})
        if hasattr(self, "_async_pooler"):
            result.update(self._async_pooler.status())
        return result

    def close(self) -> None:
        if hasattr(self, "_async_pooler"):
            self._async_pooler.close()


def create_encoder(
    model_path: str,
    backend: str = "sentence_transformers",
    max_seq_length: int = 2048,
    use_bf16: bool = True,
    engine_max_seq_length: int | None = None,
) -> QueryEncoder:
    return QueryEncoder(model_path, backend, max_seq_length, use_bf16, engine_max_seq_length)
