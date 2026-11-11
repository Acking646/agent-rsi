"""Backends for phase-aligned activation capture and proposal readout."""
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .environment import ACTIONS, Episode, Harness

END_REFLECTION = "\n[END REFLECTION]\n"
DECISION = "\n[DECISION]\nChoose one code: A keep, B ask for missing input, C refresh the catalog, D verify arithmetic. Code:"
CODES = tuple(ACTIONS)


@dataclass
class Run:
    text: str
    token_ids: list[int]
    phases: dict[str, list[int]]
    activations: dict[tuple[int, str], np.ndarray]
    logits: np.ndarray
    context: str
    tokens: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Intervention:
    site: tuple[int, str]
    vector: np.ndarray
    add: bool = False


def context_for(episode: Episode, harness: Harness) -> str:
    return (
        "Analyze the observed attempt and feedback before choosing one update.\n"
        + json.dumps({
            "task": episode.task, "trace": episode.trace,
            "feedback": episode.feedback, "harness": asdict(harness),
        }, ensure_ascii=False, sort_keys=True)
        + "\n[REFLECTION]\n"
    )


class ToyBackend:
    """Synthetic instrumented fixture. Encodes labels intentionally for smoke testing."""
    metadata = {"backend": "toy", "version": 1, "synthetic": True}

    def __init__(self, max_reflection: int):
        self.max_reflection = max_reflection

    def reflect(self, context: str, episode: Episode) -> list[int]:
        return list(episode.feedback["message"].encode())[:self.max_reflection]

    def run(self, context: str, episode: Episode, reflection: list[int],
            intervention: Intervention | None = None) -> Run:
        phases = {}
        tokens = list(context.encode())
        phases["feedback_end"] = [len(tokens) - 1]
        phases["reflection"] = list(range(len(tokens), len(tokens) + len(reflection)))
        tokens.extend(reflection)
        tokens.extend(END_REFLECTION.encode())
        phases["reflection_end"] = [len(tokens) - 1]
        tokens.extend(DECISION.encode())
        phases["proposal"] = [len(tokens) - 1]
        activations = {}
        label = CODES.index(episode.labels["recommended_action"])
        observed = np.zeros(8, dtype=np.float32)
        observed[label] = 4
        observed[4:] = np.random.default_rng(int(episode.task_id)).normal(0, .2, 4)
        for layer in range(2):
            for component in ("residual", "attention", "mlp"):
                site = layer, component
                x = np.zeros((len(tokens), 8), dtype=np.float32)
                x[phases["feedback_end"]] = observed * .5
                for i, pos in enumerate(phases["reflection"], 1):
                    x[pos] = observed * i / max(1, len(reflection))
                x[phases["reflection_end"]] = observed
                if intervention is not None and site == intervention.site:
                    pos = phases["reflection_end"][0]
                    x[pos] = x[pos] + intervention.vector if intervention.add else intervention.vector
                x[phases["proposal"]] = x[phases["reflection_end"]]
                activations[site] = x
        primary = activations[(0, "residual")][-1, :4]
        others = [x[-1, :4] for site, x in activations.items() if site != (0, "residual")]
        logits = .65 * primary + .35 * np.mean(others, axis=0)
        return Run(bytes(reflection).decode(errors="replace"), tokens, phases, activations, logits,
                   context, {"choice_token_ids": [ord(code) for code in CODES]})


class HFBackend:
    """Llama/Qwen-style decoder blocks. Each run recomputes the full token prefix."""
    def __init__(self, model_path: str, layers: list[int], max_reflection: int,
                 device: str, revision: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, revision=revision, torch_dtype="auto",
        ).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.max_reflection = max_reflection
        blocks = getattr(getattr(self.model, "model", None), "layers", None)
        if blocks is None:
            raise ValueError("Expected decoder blocks at model.model.layers")
        self.modules = {}
        for layer in layers:
            if not 0 <= layer < len(blocks):
                raise ValueError(f"Layer {layer} outside [0, {len(blocks)})")
            block = blocks[layer]
            self.modules[(layer, "residual")] = block
            self.modules[(layer, "attention")] = block.self_attn
            self.modules[(layer, "mlp")] = block.mlp
        self.suffix = self.encode(DECISION)
        self.choice_ids = [self.choice_id(code) for code in CODES]
        if len(set(self.choice_ids)) != len(CODES):
            raise ValueError("Proposal labels must map to distinct token IDs")
        self.metadata = {
            "backend": "hf", "model": model_path, "revision": revision,
            "commit": getattr(self.model.config, "_commit_hash", None),
            "tokenizer": type(self.tokenizer).__name__, "layers": layers,
            "device": device, "max_reflection": max_reflection,
            "reflection_decoding": "greedy", "choice_token_ids": self.choice_ids,
            "activation_cache": False,
        }

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def choice_id(self, code: str) -> int:
        combined = self.encode(DECISION + " " + code)
        if combined[:-1] != self.suffix:
            raise ValueError(f"Decision label {code} needs one token after the fixed suffix")
        return combined[-1]

    def reflect(self, context: str, episode: Episode) -> list[int]:
        ids = self.torch.tensor([self.encode(context)], device=self.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                input_ids=ids, attention_mask=self.torch.ones_like(ids),
                max_new_tokens=self.max_reflection, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )[0, ids.shape[1]:].tolist()
        stop = self.tokenizer.eos_token_id
        if stop in out:
            out = out[:out.index(stop)]
        return out

    def run(self, context: str, episode: Episode, reflection: list[int],
            intervention: Intervention | None = None) -> Run:
        torch = self.torch
        tokens = self.encode(context)
        phases = {"feedback_end": [len(tokens) - 1],
                  "reflection": list(range(len(tokens), len(tokens) + len(reflection)))}
        tokens.extend(reflection)
        tokens.extend(self.encode(END_REFLECTION))
        anchor = len(tokens) - 1
        phases["reflection_end"] = [anchor]
        tokens.extend(self.suffix)
        phases["proposal"] = [len(tokens) - 1]
        if intervention is not None and intervention.site not in self.modules:
            raise ValueError(f"Unregistered intervention site {intervention.site}")
        activations = {}
        handles = []

        def hook_for(site):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if intervention is not None and site == intervention.site:
                    hidden = hidden.clone()
                    v = torch.as_tensor(intervention.vector, device=hidden.device, dtype=hidden.dtype)
                    hidden[0, anchor] = hidden[0, anchor] + v if intervention.add else v
                activations[site] = hidden[0].detach().float().cpu().numpy()
                if intervention is not None and site == intervention.site:
                    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
            return hook

        try:
            for site, module in self.modules.items():
                handles.append(module.register_forward_hook(hook_for(site)))
            with torch.inference_mode():
                model_out = self.model(
                    input_ids=torch.tensor([tokens], device=self.device),
                    use_cache=False,
                )
            logits = model_out.logits[0, -1, self.choice_ids].detach().float().cpu().numpy()
        finally:
            for handle in handles:
                handle.remove()
        return Run(self.tokenizer.decode(reflection, skip_special_tokens=True), tokens,
                   phases, activations, logits, context, {"choice_token_ids": self.choice_ids})
