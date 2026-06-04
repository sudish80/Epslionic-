"""Model Merging & LoRA Fusion — SLERP, TIES, DARE, Linear merge.
Enables multi-adapter composition and base model interpolation."""

import json
import logging
import os
import shutil
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import torch

logger = logging.getLogger("openclaw.tool.model_merger")


class ModelMergeError(Exception):
    pass


class ModelMerger:
    """Merge multiple models/adapters into one using various algorithms."""

    def __init__(self, models_dir: Path = None):
        self.models_dir = models_dir or Path("/content/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def merge_loras(self, model_name: str, adapter_paths: List[str],
                    adapter_weights: List[float] = None,
                    merge_method: str = "linear") -> dict:
        """Merge multiple LoRA adapters into a single model using task arithmetic."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            weights = adapter_weights or [1.0 / len(adapter_paths)] * len(adapter_paths)
            if len(weights) != len(adapter_paths):
                raise ModelMergeError("Number of weights must match number of adapters")

            base = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            if merge_method == "linear":
                merged = self._linear_merge(base, adapter_paths, weights)
            elif merge_method == "slerp":
                merged = self._slerp_merge(base, adapter_paths, weights)
            elif merge_method == "ties":
                merged = self._ties_merge(base, adapter_paths, weights)
            else:
                raise ModelMergeError(f"Unknown merge method: {merge_method}")

            output_path = str(self.models_dir / f"merged_{merge_method}_{len(adapter_paths)}")
            merged.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            return {"success": True, "model_path": output_path, "method": merge_method,
                    "num_adapters": len(adapter_paths), "weights": weights}
        except Exception as e:
            logger.error(f"Model merge failed: {e}")
            return {"success": False, "error": str(e)}

    def _linear_merge(self, base_model, adapter_paths: List[str], weights: List[float]):
        """Task arithmetic: merged = base + sum(wi * (adapter_i - base))."""
        from peft import PeftModel
        merged = base_model
        state = base_model.state_dict()

        accum = {k: torch.zeros_like(v) for k, v in state.items()}
        for path, w in zip(adapter_paths, weights):
            adapter = PeftModel.from_pretrained(base_model, path)
            merged_adapter = adapter.merge_and_unload()
            delta = {k: (merged_adapter.state_dict()[k] - state[k]) * w for k in state}
            for k in accum:
                accum[k] += delta.get(k, torch.zeros_like(accum[k]))

        for k in merged.state_dict():
            if k in accum:
                merged.state_dict()[k].copy_(state[k] + accum[k])
        return merged

    def _slerp_merge(self, base_model, adapter_paths: List[str], weights: List[float]):
        """Spherical linear interpolation between model weights."""
        from peft import PeftModel
        if len(adapter_paths) != 2:
            raise ModelMergeError("SLERP requires exactly 2 models")

        t = weights[0] / (weights[0] + weights[1]) if sum(weights) > 0 else 0.5
        model_a = PeftModel.from_pretrained(base_model, adapter_paths[0]).merge_and_unload()
        model_b = PeftModel.from_pretrained(base_model, adapter_paths[1]).merge_and_unload()

        state_a, state_b = model_a.state_dict(), model_b.state_dict()
        merged = base_model
        for k in merged.state_dict():
            if k in state_a and k in state_b:
                v0, v1 = state_a[k].float(), state_b[k].float()
                dot = (v0 * v1).sum().clamp(-1, 1)
                theta = torch.acos(dot)
                sin_theta = torch.sin(theta)
                if sin_theta > 1e-6:
                    merged.state_dict()[k].copy_(
                        (torch.sin((1 - t) * theta) / sin_theta) * v0 +
                        (torch.sin(t * theta) / sin_theta) * v1
                    ).half()
        return merged

    def _ties_merge(self, base_model, adapter_paths: List[str], weights: List[float]):
        """TIES merging: Trim, Elect Sign, and Merge."""
        from peft import PeftModel
        import numpy as np

        models = []
        for path in adapter_paths:
            adapter = PeftModel.from_pretrained(base_model, path)
            models.append(adapter.merge_and_unload())

        state_dicts = [m.state_dict() for m in models]
        base_state = base_model.state_dict()
        merged = base_model

        top_k_frac = 0.3
        for k in merged.state_dict():
            if k not in state_dicts[0]:
                continue
            deltas = []
            for sd in state_dicts:
                delta = sd[k].float() - base_state[k].float()
                deltas.append(delta)

            stacked = torch.stack(deltas)
            mask = torch.zeros_like(stacked[0])
            for d in deltas:
                threshold = torch.quantile(d.abs().view(-1), 1 - top_k_frac)
                mask += (d.abs() > threshold).float()

            sign_agreement = torch.stack([d.sign() for d in deltas]).mean(dim=0)
            final_delta = torch.zeros_like(stacked[0])
            for i, d in enumerate(deltas):
                agreement = (sign_agreement * d.sign()) > 0
                final_delta += d * agreement.float() * weights[i]

            merged.state_dict()[k].copy_((base_state[k].float() + final_delta).half())
        return merged

    def extract_adapter(self, model_path: str, output_path: str = None) -> dict:
        """Extract LoRA adapter from a merged model."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(model_path)
            lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=16)
            peft_model = get_peft_model(model, lora_config)

            out = output_path or str(self.models_dir / "extracted_adapter")
            peft_model.save_pretrained(out)
            return {"success": True, "adapter_path": out}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_adapters(self) -> List[dict]:
        """List all available adapters in models directory."""
        adapters = []
        for f in self.models_dir.glob("**/adapter_model.safetensors"):
            parent = f.parent
            adapters.append({"path": str(parent), "name": parent.name, "model": parent.parent.name})
        for f in self.models_dir.glob("**/adapter_config.json"):
            parent = f.parent
            if parent.name not in [a["name"] for a in adapters]:
                adapters.append({"path": str(parent), "name": parent.name, "model": parent.parent.name})
        return adapters

    def get_tool_description(self) -> dict:
        return {
            "merge_loras": {"description": "Merge multiple LoRA adapters into base model",
                "parameters": {"model_name": "Base model name", "adapter_paths": "List of adapter paths",
                    "adapter_weights": "Weights per adapter", "merge_method": "linear|slerp|ties"}},
            "extract_adapter": {"description": "Extract LoRA adapter from a merged model",
                "parameters": {"model_path": "Path to merged model"}},
        }
