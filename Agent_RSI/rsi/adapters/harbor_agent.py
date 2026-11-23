"""White-box Qwen3 agent that executes actions through Harbor environments."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from rsi.contracts import HarnessModule
from rsi.harness.compiler import StructuredUpdate
from rsi.harness.schema import HarnessSpec

_MODEL_CACHE: dict[tuple[str, str | None, str, str], InstrumentedModel] = {}
_PRIVATE_PATH = re.compile(
    r"(^|[\\/])(?:hidden[_-]?tests?|solution|gold(?:_patch)?|verifier)(?:[\\/]|$)|/logs/verifier",
    re.IGNORECASE,
)


def _extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    if not candidates:
        raise ValueError("Model response did not contain a JSON object")
    return candidates[-1]


class InstrumentedModel:
    """One frozen HF checkpoint shared by sequential Harbor trials."""

    def __init__(
        self,
        model_name: str,
        revision: str | None,
        device: str,
        dtype: str,
        layers: tuple[int, ...],
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, trust_remote_code=False
        )
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }[dtype]
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                torch_dtype=torch_dtype,
                trust_remote_code=False,
            )
            .to(device)
            .eval()
        )
        self.model.requires_grad_(False)
        self.device = device
        blocks = getattr(getattr(self.model, "model", None), "layers", None)
        if blocks is None:
            raise ValueError("Expected decoder blocks at model.model.layers")
        self.modules: dict[tuple[int, str], Any] = {}
        for layer in layers:
            if not 0 <= layer < len(blocks):
                raise ValueError(f"Layer {layer} outside [0, {len(blocks)})")
            block = blocks[layer]
            self.modules[(layer, "residual")] = block
            self.modules[(layer, "attention")] = block.self_attn
            self.modules[(layer, "mlp")] = block.mlp
        self.revision = revision or getattr(self.model.config, "_commit_hash", None)

    def _template(self, messages: list[dict[str, str]], thinking: bool):
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=thinking, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    @contextmanager
    def _record(self) -> Iterator[dict[tuple[int, str], list[np.ndarray]]]:
        captured: dict[tuple[int, str], list[np.ndarray]] = {
            site: [] for site in self.modules
        }
        handles = []

        def hook_for(site):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                # The prefill can contain thousands of tokens. The prospective
                # probe needs its final position plus each generated prefix.
                captured[site].append(
                    hidden[0, -1:].detach().float().cpu().numpy()
                )

            return hook

        try:
            for site, module in self.modules.items():
                handles.append(module.register_forward_hook(hook_for(site)))
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def generate(
        self,
        messages: list[dict[str, str]],
        output_path: Path,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        thinking: bool,
        seed: int,
    ) -> dict[str, Any]:
        torch = self.torch
        torch.manual_seed(seed)
        input_ids = self._template(messages, thinking).to(self.device)
        attention_mask = torch.ones_like(input_ids)
        generation = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if temperature > 0:
            generation.update({"temperature": temperature, "top_p": top_p})
        with self._record() as captured, torch.inference_mode():
            sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation,
            )
        new_ids = sequences[0, input_ids.shape[1] :]
        new_id_values = new_ids.detach().cpu().tolist()
        text = self.tokenizer.decode(new_id_values, skip_special_tokens=True)
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        activation_length = None
        for (layer, component), chunks in captured.items():
            matrix = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0))
            arrays[f"l{layer}_{component}"] = matrix.astype(np.float16, copy=False)
            activation_length = (
                len(matrix)
                if activation_length is None
                else min(activation_length, len(matrix))
            )
        all_ids = sequences[0].detach().cpu().numpy()
        start = int(input_ids.shape[1]) - 1
        if activation_length is not None:
            all_ids = all_ids[start : start + activation_length]
            arrays = {key: value[:activation_length] for key, value in arrays.items()}
        field_prefix_positions = {}
        for field in (
            "reflection",
            "root_cause",
            "responsible_step",
            "target_module",
            "operator",
            "arguments",
        ):
            marker = text.rfind(f'"{field}"')
            if marker < 0:
                continue
            low, high = 0, len(new_id_values)
            while low < high:
                middle = (low + high + 1) // 2
                prefix_chars = len(
                    self.tokenizer.decode(
                        new_id_values[:middle], skip_special_tokens=True
                    )
                )
                if prefix_chars <= marker:
                    low = middle
                else:
                    high = middle - 1
            field_prefix_positions[field] = low
        np.savez_compressed(
            output_path,
            token_ids=all_ids,
            activation_token_positions=np.arange(
                start, start + len(all_ids), dtype=np.int32
            ),
            input_tokens=np.asarray([input_ids.shape[1]], dtype=np.int32),
            **arrays,
        )
        return {
            "text": text,
            "input_tokens": int(input_ids.shape[1]),
            "output_tokens": len(new_ids),
            "activation_file": str(output_path),
            "field_prefix_positions": field_prefix_positions,
            "model_revision": self.revision,
        }


def get_model(
    model_name: str,
    revision: str | None,
    device: str,
    dtype: str,
    layers: tuple[int, ...],
) -> InstrumentedModel:
    key = (model_name, revision, device, dtype + ":" + ",".join(map(str, layers)))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = InstrumentedModel(
            model_name, revision, device, dtype, layers
        )
    return _MODEL_CACHE[key]


class WhiteboxHarborAgent(BaseAgent):
    """Mini-SWE-agent style loop with host-side activations and Harbor tools."""

    SUPPORTS_ATIF = False
    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        harness_path: str | None = None,
        revision: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        layers: list[int] | None = None,
        max_steps: int = 40,
        max_new_tokens: int = 768,
        temperature: float = 0.0,
        top_p: float = 0.95,
        thinking: bool = True,
        seed: int = 2026,
        workdir: str | None = None,
        extra_env: dict | None = None,
        **kwargs,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        if not model_name:
            raise ValueError("WhiteboxHarborAgent requires model_name")
        if not harness_path:
            raise ValueError("WhiteboxHarborAgent requires harness_path")
        self.harness = HarnessSpec.load(harness_path)
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.layers = tuple(layers or (3, 9, 15, 21, 27))
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.thinking = thinking
        self.seed = seed
        self.workdir = workdir
        self.extra_env = extra_env or {}

    @staticmethod
    def name() -> str:
        return "agent-rsi-whitebox"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    def _system_prompt(self) -> str:
        return (
            "You are a software-engineering agent operating in a private benchmark container.\n"
            "Follow the five harness modules below. Never inspect hidden tests, solution files, gold "
            "patches, or verifier logs. Respond with one JSON object and no prose outside it:\n"
            '{"reflection":"brief state update","command":"one shell command or empty",'
            '"task_complete":false,"completion_evidence":""}\n\n'
            + self.harness.prompt()
        )

    def _trim_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        limit = int(
            self.harness.parameter(
                HarnessModule.CONTEXT_MANAGEMENT, "max_step_messages", 24
            )
        )
        if len(messages) <= limit + 2:
            return messages
        removed = messages[2:-limit]
        summary_chars = int(
            self.harness.parameter(
                HarnessModule.CONTEXT_MANAGEMENT, "summary_chars", 6000
            )
        )
        summary = "\n".join(item["content"] for item in removed)[-summary_chars:]
        return (
            messages[:2]
            + [
                {
                    "role": "user",
                    "content": "Earlier interaction summary (verbatim tail):\n"
                    + summary,
                }
            ]
            + messages[-limit:]
        )

    def _observation(self, stdout: str, stderr: str, return_code: int) -> str:
        text = (
            f"return_code={return_code}\nstdout:\n{stdout}\nstderr:\n{stderr}".strip()
        )
        max_chars = int(
            self.harness.parameter(
                HarnessModule.OBSERVATION_MANAGEMENT, "max_chars", 12000
            )
        )
        if len(text) <= max_chars:
            return text
        fraction = float(
            self.harness.parameter(
                HarnessModule.OBSERVATION_MANAGEMENT, "head_fraction", 0.35
            )
        )
        head = max(1, int(max_chars * fraction))
        tail = max_chars - head
        return text[:head] + "\n[... observation truncated ...]\n" + text[-tail:]

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        activation_dir = self.logs_dir / "activations"
        model = get_model(
            self.model_name, self.revision, self.device, self.dtype, self.layers
        )
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": instruction},
        ]
        steps: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        total_input = total_output = 0
        consecutive_complete = 0
        successful_command = False
        harness_steps = int(
            self.harness.parameter(
                HarnessModule.AGENT_LOOP, "max_steps", self.max_steps
            )
        )
        max_steps = min(self.max_steps, harness_steps)
        required = int(
            self.harness.parameter(
                HarnessModule.COMPLETION_DETECTION, "required_consecutive", 1
            )
        )
        require_success = bool(
            self.harness.parameter(
                HarnessModule.COMPLETION_DETECTION, "require_successful_command", True
            )
        )
        timeout = int(
            self.harness.parameter(HarnessModule.TOOL_USE, "command_timeout_sec", 120)
        )

        for index in range(max_steps):
            call = model.generate(
                messages,
                activation_dir / f"call-{index:03d}.npz",
                self.max_new_tokens,
                self.temperature,
                self.top_p,
                self.thinking,
                self.seed + index,
            )
            calls.append(call)
            total_input += call["input_tokens"]
            total_output += call["output_tokens"]
            messages.append({"role": "assistant", "content": call["text"]})
            try:
                action = _extract_json(call["text"])
            except ValueError as exc:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Invalid action: {exc}. Return valid JSON.",
                    }
                )
                steps.append(
                    {"index": index, "parse_error": str(exc), "raw": call["text"]}
                )
                messages = self._trim_messages(messages)
                continue

            command = str(action.get("command", "")).strip()
            declared_complete = bool(action.get("task_complete", False))
            observation = "No command executed."
            return_code = None
            if command:
                if _PRIVATE_PATH.search(command):
                    observation = (
                        "Command rejected: benchmark-private paths are inaccessible."
                    )
                    return_code = 126
                else:
                    result = await environment.exec(
                        command,
                        cwd=self.workdir,
                        env=self.extra_env or None,
                        timeout_sec=timeout,
                    )
                    return_code = result.return_code
                    successful_command = successful_command or result.return_code == 0
                    observation = self._observation(
                        result.stdout or "", result.stderr or "", result.return_code
                    )
            consecutive_complete = consecutive_complete + 1 if declared_complete else 0
            accepted_complete = consecutive_complete >= required and (
                successful_command or not require_success
            )
            steps.append(
                {
                    "index": index,
                    "reflection": str(action.get("reflection", "")),
                    "command": command,
                    "return_code": return_code,
                    "observation": observation,
                    "declared_complete": declared_complete,
                    "accepted_complete": accepted_complete,
                    "completion_evidence": str(action.get("completion_evidence", "")),
                    "activation_file": call["activation_file"],
                }
            )
            messages.append({"role": "user", "content": observation})
            if accepted_complete:
                break
            messages = self._trim_messages(messages)

        trace = {
            "schema_version": 1,
            "model": self.model_name,
            "model_revision": model.revision,
            "harness_version": self.harness.version_id,
            "component_hashes": self.harness.component_hashes(),
            "instruction": instruction,
            "steps": steps,
            "model_calls": calls,
            "messages": messages,
        }
        trace_path = self.logs_dir / "agent_rsi_trace.json"
        trace_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        context.n_input_tokens = total_input
        context.n_output_tokens = total_output
        context.metadata = {
            "agent_rsi_trace": str(trace_path),
            "harness_version": self.harness.version_id,
            "component_hashes": self.harness.component_hashes(),
            "steps": steps,
            "model_calls": calls,
        }


def generate_structured_update(
    *,
    model_name: str,
    revision: str | None,
    device: str,
    dtype: str,
    layers: tuple[int, ...],
    harness: HarnessSpec,
    episode: dict[str, Any],
    output_path: Path,
    seed: int,
    thinking: bool,
) -> tuple[StructuredUpdate, dict[str, Any]]:
    """Diagnose an official verifier result and emit one constrained proposal."""
    system = (
        "Diagnose this failed agent episode. Return one JSON object with fields: "
        "reflection, root_cause, responsible_step, target_module, operator, scope, "
        "rationale, arguments. target_module must be one of agent_loop, tool_use, "
        "observation_management, context_management, completion_detection. operator "
        "must be one of ADD_GUARD, ADD_VALIDATION, RETRY_ON_FAILURE, REORDER, RETAIN, "
        "COMPRESS, RETRIEVE, STOP_RULE, NO_OP. Prefer NO_OP when evidence is weak."
    )
    payload = json.dumps(
        {"harness": harness.to_dict(), "episode": episode},
        ensure_ascii=False,
        default=str,
    )[-60000:]
    model = get_model(model_name, revision, device, dtype, layers)
    call = model.generate(
        [{"role": "system", "content": system}, {"role": "user", "content": payload}],
        output_path,
        max_new_tokens=1024,
        temperature=0.0,
        top_p=1.0,
        thinking=thinking,
        seed=seed,
    )
    value = _extract_json(call["text"])
    update = StructuredUpdate.from_dict(value)
    return update, {
        "reflection": str(value.get("reflection", "")),
        "root_cause": value.get("root_cause"),
        "responsible_step": value.get("responsible_step"),
        "call": call,
        "raw": value,
    }
