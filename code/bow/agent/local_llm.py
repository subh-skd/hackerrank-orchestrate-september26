"""Local Qwen chat model on CPU, as a LangChain BaseChatModel.

Weights come from the Hugging Face Hub and run through transformers on CPU,
with no API and no GPU. Wrapping the model as a BaseChatModel keeps it a
first-class LangChain component: LangGraph nodes invoke it like any chat model,
the global LLM cache replays its answers, and callbacks receive
usage_metadata for the token report.

Why not ChatHuggingFace: it does not forward chat-template arguments, and
Qwen3 needs enable_thinking=False to answer directly instead of producing
long reasoning traces, which would be far too slow on a CPU.

Measured on a Ryzen 5 5600H (6 cores): Qwen3-1.7B in float32 decides a compact
case file in about 12 s and chose the best plan on 5/5 benchmark samples.
float32 is used because bfloat16 has no fast kernels on AVX2 CPUs (it was
about 6x slower in testing).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.load import dumps, loads
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult, Generation

DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-1.7B"
_ROOT = Path(__file__).resolve().parents[3]


class DiskCache(BaseCache):
    """LangChain LLM cache persisted as one JSON file per response.

    Keyed by (prompt, model configuration). Rerunning the graph replays the
    identical model answers, and deserialization is restricted to an explicit
    allowlist of output classes.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, prompt: str, llm_string: str) -> Path:
        digest = hashlib.sha256(f"{llm_string}\x00{prompt}".encode("utf-8")).hexdigest()
        return self.directory / f"{digest[:32]}.json"

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        path = self._path(prompt, llm_string)
        if not path.exists():
            self.misses += 1
            return None
        self.hits += 1
        allowed = [ChatGeneration, Generation, AIMessage]
        return [
            loads(item, allowed_objects=allowed)
            for item in json.loads(path.read_text(encoding="utf-8"))
        ]

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        self._path(prompt, llm_string).write_text(
            json.dumps([dumps(g) for g in return_val]), encoding="utf-8"
        )

    def clear(self, **kwargs: Any) -> None:
        for path in self.directory.glob("*.json"):
            path.unlink()


def ensure_hf_home() -> None:
    os.environ.setdefault("HF_HOME", str(_ROOT / ".cache" / "huggingface"))


class _Weights:
    """Process-wide singleton: load the model once, serialize generation."""

    lock = threading.Lock()
    loaded: dict[str, tuple[Any, Any]] = {}

    @classmethod
    def get(cls, model_id: str):
        with cls.lock:
            if model_id not in cls.loaded:
                ensure_hf_home()
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                torch.set_num_threads(
                    max(1, os.cpu_count() // 2 if os.cpu_count() else 4)
                )
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, dtype=torch.float32
                ).eval()
                cls.loaded[model_id] = (tokenizer, model)
            return cls.loaded[model_id]


class LocalQwenChat(BaseChatModel):
    model_id: str = DEFAULT_LOCAL_MODEL
    max_new_tokens: int = 96
    enable_thinking: bool = False

    @property
    def _llm_type(self) -> str:
        return "local-huggingface-transformers"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_new_tokens": self.max_new_tokens,
            "enable_thinking": self.enable_thinking,
            "decoding": "greedy",
        }

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        import torch

        tokenizer, model = _Weights.get(self.model_id)
        chat = []
        for message in messages:
            role = (
                "system"
                if isinstance(message, SystemMessage)
                else ("assistant" if isinstance(message, AIMessage) else "user")
            )
            content = (
                message.content
                if isinstance(message.content, str)
                else " ".join(
                    part.get("text", "")
                    for part in message.content
                    if isinstance(part, dict)
                )
            )
            chat.append({"role": role, "content": content})

        template_kwargs = (
            {"enable_thinking": self.enable_thinking}
            if "Qwen3" in self.model_id
            else {}
        )
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        n_in = int(inputs["input_ids"].shape[1])

        started = time.monotonic()
        with _Weights.lock, torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        seconds = time.monotonic() - started

        n_out = int(output.shape[1] - n_in)
        text = tokenizer.decode(output[0, n_in:], skip_special_tokens=True).strip()
        message = AIMessage(
            content=text,
            usage_metadata={
                "input_tokens": n_in,
                "output_tokens": n_out,
                "total_tokens": n_in + n_out,
            },
            response_metadata={
                "model_name": self.model_id,
                "seconds": round(seconds, 2),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def local_chat_model() -> LocalQwenChat:
    return LocalQwenChat(
        model_id=os.environ.get("BOW_LOCAL_MODEL", DEFAULT_LOCAL_MODEL)
    )


__all__ = ["LocalQwenChat", "local_chat_model", "HumanMessage"]
