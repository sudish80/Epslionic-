"""Post-Training Quantization — GPTQ, AWQ, GGUF export.
Converts trained models to deployment-optimized formats."""

import json
import logging
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger("epsionic.tool.quantizer")


class ModelQuantizer:
    """Quantize trained models for deployment (GPTQ, AWQ, GGUF)."""

    def __init__(self, models_dir: Path = None):
        self.models_dir = models_dir or Path("/content/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def quantize_gptq(self, model_path: str, bits: int = 4, group_size: int = 128,
                      dataset: str = "wikitext2", output_name: str = None) -> dict:
        """Quantize model using GPTQ (GPU-optimized)."""
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            from transformers import AutoTokenizer

            quantize_config = BaseQuantizeConfig(
                bits=bits, group_size=group_size, desc_act=False,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)

            calib_data = self._get_calibration_data(model_path, dataset)
            model.quantize(calib_data)

            out = output_name or f"{Path(model_path).name}_gptq_{bits}bit"
            output_path = str(self.models_dir / out)
            model.save_quantized(output_path)
            tokenizer.save_pretrained(output_path)

            return {"success": True, "output_path": output_path, "method": "gptq",
                    "bits": bits, "size_gb": self._folder_size(output_path)}
        except ImportError:
            return {"success": False, "error": "auto_gptq not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def quantize_awq(self, model_path: str, bits: int = 4, output_name: str = None) -> dict:
        """Quantize model using AWQ (Activation-Aware)."""
        try:
            from awq import AutoAWQForCausalLM

            model = AutoAWQForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path) if hasattr(self, '_') else None

            from awq import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            calib_data = self._get_calibration_data(model_path, "wikitext2")
            model.quantize(tokenizer, calib_data, bits=bits, group_size=128)

            out = output_name or f"{Path(model_path).name}_awq_{bits}bit"
            output_path = str(self.models_dir / out)
            model.save_quantized(output_path)
            tokenizer.save_pretrained(output_path)

            return {"success": True, "output_path": output_path, "method": "awq", "bits": bits}
        except ImportError:
            return {"success": False, "error": "awq not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def export_gguf(self, model_path: str, output_name: str = None,
                    quantize: str = "q4_k_m") -> dict:
        """Export model to GGUF format (CPU/llama.cpp compatible)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            import subprocess
            import tempfile

            out = output_name or f"{Path(model_path).name}_gguf"
            output_path = str(self.models_dir / f"{out}.gguf")

            with tempfile.TemporaryDirectory() as tmp:
                tmp_model = Path(tmp) / "model"
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model.save_pretrained(str(tmp_model))
                tokenizer.save_pretrained(str(tmp_model))

                result = subprocess.run(
                    ["python", "-m", "llama_cpp.convert", str(tmp_model), "--outfile", output_path,
                     "--outtype", quantize.replace("q", "q").upper()],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode != 0:
                    return {"success": False, "error": result.stderr[:500]}

            return {"success": True, "output_path": output_path, "method": "gguf", "quantization": quantize}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_calibration_data(self, model_path: str, dataset: str, num_samples: int = 128):
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset, split="train", streaming=True)
            samples = [next(iter(ds))["text"][:512] for _ in range(num_samples)]
            return samples
        except Exception:
            return ["The quick brown fox jumps over the lazy dog."] * num_samples

    def _folder_size(self, path: str) -> float:
        total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        return total / 1e9

    def get_tool_description(self) -> dict:
        return {
            "quantize_gptq": {"description": "GPTQ quantization for GPU deployment",
                "parameters": {"model_path": "Input model", "bits": "4 or 8", "group_size": "128 or 64"}},
            "quantize_awq": {"description": "AWQ activation-aware quantization",
                "parameters": {"model_path": "Input model", "bits": "4"}},
            "export_gguf": {"description": "Export to GGUF for CPU/llama.cpp",
                "parameters": {"model_path": "Input model", "quantize": "q4_k_m, q5_k_m, q8_0"}},
        }
