"""
Auto-Fixer Tool - Detects and fixes common ML training errors.
OpenClaw skill: self-healing capability that monitors and corrects issues.
Designed for Colab environment with common GPU/training errors.
"""

import json
import logging
import re
import traceback
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

logger = logging.getLogger("openclaw.tool.auto_fixer")


class AutoFixerTool:
    def __init__(self, memory_store=None):
        self.memory = memory_store
        self._fix_count = 0
        self._fix_rules = self._build_fix_rules()

    def _build_fix_rules(self) -> List[dict]:
        return [
            {
                "pattern": r"CUDA out of memory",
                "severity": "critical",
                "fix": "Reduce batch size or enable gradient checkpointing",
                "param_fix": {
                    "batch_size": "halve",
                    "gradient_accumulation_steps": "double",
                    "use_gradient_checkpointing": True
                },
                "confidence": 0.9,
            },
            {
                "pattern": r"is None|NoneType|object has no attribute",
                "severity": "critical",
                "fix": "Check model/dataset initialization. Ensure dataset is properly loaded.",
                "param_fix": {"force_reload": True, "verify_integrity": True},
                "confidence": 0.6,
            },
            {
                "pattern": r"expected scalar type (Half|Float).*found (Float|Half)",
                "severity": "warning",
                "fix": "Mixed precision dtype mismatch. Set torch_dtype consistently.",
                "param_fix": {"torch_dtype": "auto", "use_mixed_precision": True},
                "confidence": 0.8,
            },
            {
                "pattern": r"ConnectionError|timeout|Connection refused",
                "severity": "warning",
                "fix": "Network issue retrying connection",
                "param_fix": {"retry_count": 3, "retry_delay": 5},
                "confidence": 0.5,
            },
            {
                "pattern": r"HTTP Error 503|HTTP Error 429",
                "severity": "warning",
                "fix": "Hugging Face rate limit. Wait and retry.",
                "param_fix": {"retry_delay": 30, "use_mirror": True},
                "confidence": 0.7,
            },
            {
                "pattern": r"No module named|ModuleNotFoundError|ImportError",
                "severity": "critical",
                "fix": "Install missing Python package",
                "param_fix": {"install_missing": True},
                "confidence": 0.95,
            },
            {
                "pattern": r"FileNotFoundError|No such file or directory",
                "severity": "warning",
                "fix": "Create missing directory or download file",
                "param_fix": {"create_dirs": True, "download_if_missing": True},
                "confidence": 0.7,
            },
            {
                "pattern": r"Permission denied|Access denied",
                "severity": "warning",
                "fix": "Fix file permissions or use alternative path",
                "param_fix": {"chmod": True, "alt_path": "/tmp/"},
                "confidence": 0.5,
            },
            {
                "pattern": r"NaN|inf|division by zero",
                "severity": "warning",
                "fix": "Numerical instability. Reduce learning rate or add gradient clipping.",
                "param_fix": {
                    "learning_rate": "divide_by_10",
                    "max_grad_norm": 1.0,
                },
                "confidence": 0.8,
            },
            {
                "pattern": r"tokenizer|vocab|embedding",
                "severity": "warning",
                "fix": "Tokenizer/vocab mismatch. Ensure model and tokenizer are compatible.",
                "param_fix": {"force_tokenizer_reload": True},
                "confidence": 0.6,
            },
            {
                "pattern": r"graph\.py|derivative|backward",
                "severity": "warning",
                "fix": "Autograd graph issue. Enable `retain_graph=True` or check model architecture.",
                "param_fix": {"retain_graph": True},
                "confidence": 0.5,
            },
            {
                "pattern": r"out of memory|OOM|Killed",
                "severity": "critical",
                "fix": "System out of memory. Clear cache, reduce model size or batch size.",
                "param_fix": {
                    "batch_size": "halve",
                    "clear_cache": True,
                    "use_cpu_offload": True,
                },
                "confidence": 0.85,
            },
        ]

    def fix(self, error: str = "", context: dict = None,
            brain_fix: dict = None) -> dict:
        """
        Public entry point: analyze an error and attempt a fix.
        Wraps analyze_and_fix with the OpenClaw expected return shape.
        """
        result = self.analyze_and_fix(error, context, brain_fix)
        return {
            "fixed": result.get("success", False),
            "conf": result.get("confidence", 0.0),
            "desc": result.get("fix_description", ""),
            "fix_attempted": result.get("fix_attempted", False),
            "param_fix": result.get("param_fix", {}),
            "error_id": result.get("error_id"),
            "adjusted": result.get("adjusted_params", {}),
            "matched": result.get("matched_rule"),
        }

    def analyze_and_fix(self, error_message: str, context: dict = None,
                        brain_analysis: dict = None) -> dict:
        """
        Analyze an error and attempt to fix it.
        Uses rule-based matching first, then LLM analysis if available.
        """
        context = context or {}

        result = {
            "original_error": error_message[:500],
            "matched_rule": None,
            "fix_attempted": False,
            "fix_description": "",
            "param_fix": {},
            "confidence": 0.0,
            "success": False,
        }

        # Step 1: Try rule-based matching
        for rule in self._fix_rules:
            if re.search(rule["pattern"], error_message, re.IGNORECASE):
                result["matched_rule"] = rule["pattern"]
                result["fix_description"] = rule["fix"]
                result["param_fix"] = rule["param_fix"]
                result["confidence"] = rule["confidence"]
                result["fix_attempted"] = True
                logger.info(f"Matched fix rule: {rule['pattern']} -> {rule['fix']}")
                break

        # Step 2: Use LLM brain analysis if provided
        if brain_analysis:
            llm_fix = brain_analysis.get("retry_with_params", {})
            llm_confidence = brain_analysis.get("confidence", 0.0)
            if llm_confidence > result["confidence"]:
                result["param_fix"] = llm_fix
                result["fix_description"] = brain_analysis.get("suggested_fix", "")
                result["confidence"] = llm_confidence
                result["fix_attempted"] = True

        # Step 3: If no rule matched but we have context, try generic fixes
        if not result["fix_attempted"]:
            generic_fix = self._generic_fix_attempt(error_message, context)
            if generic_fix:
                result.update(generic_fix)

        # Step 4: Apply the param fix to context
        if result["fix_attempted"]:
            result["success"] = True
            result["adjusted_params"] = self._resolve_param_fix(
                result["param_fix"], context
            )

        # Log to memory
        if self.memory:
            error_id = self.memory.log_error(
                source=context.get("source", "auto_fixer"),
                error=error_message[:500],
                context={"fix_result": result}
            )
            result["error_id"] = error_id
            if result["success"]:
                self.memory.mark_error_fixed(error_id, result["fix_description"])
                self._fix_count += 1

        return result

    def _generic_fix_attempt(self, error: str, context: dict) -> Optional[dict]:
        """Generic fallback fix for unknown errors."""
        if "size" in error.lower() and ("batch" in error.lower() or "dimension" in error.lower()):
            return {
                "matched_rule": "generic_size_mismatch",
                "fix_description": "Tensor size mismatch. Try reducing batch size or sequence length.",
                "param_fix": {"batch_size": "halve", "max_seq_length": "halve"},
                "confidence": 0.4,
                "fix_attempted": True,
            }

        if "install" in error.lower() or "package" in error.lower():
            return {
                "matched_rule": "generic_install",
                "fix_description": "Missing package detected",
                "param_fix": {"install_missing": True},
                "confidence": 0.5,
                "fix_attempted": True,
            }

        return None

    def _resolve_param_fix(self, param_fix: dict, original_params: dict) -> dict:
        """Resolve relative param fixes (e.g., 'halve' batch_size) into concrete values."""
        resolved = {}

        for key, value in param_fix.items():
            if value == "halve" and key in original_params:
                resolved[key] = max(1, original_params[key] // 2)
            elif value == "double" and key in original_params:
                resolved[key] = original_params[key] * 2
            elif value == "divide_by_10" and key in original_params:
                resolved[key] = original_params.get(key, 0.001) / 10
            elif isinstance(value, bool) or isinstance(value, (int, float, str)):
                resolved[key] = value
            else:
                resolved[key] = value

        return resolved

    def auto_install_missing(self, error_message: str) -> Dict[str, bool]:
        """Detect and auto-install missing Python packages."""
        installed = {}
        package_map = {
            "transformers": "transformers",
            "torch": "torch",
            "datasets": "datasets",
            "accelerate": "accelerate",
            "peft": "peft",
            "trl": "trl",
            "bitsandbytes": "bitsandbytes",
            "unsloth": "unsloth",
            "scipy": "scipy",
            "sentencepiece": "sentencepiece",
            "protobuf": "protobuf",
            "wandb": "wandb",
            "huggingface_hub": "huggingface_hub",
            "einops": "einops",
        }

        for import_name, pip_name in package_map.items():
            if import_name in error_message.lower().replace(" ", ""):
                try:
                    __import__(import_name.replace(" ", ""))
                    installed[pip_name] = True
                except ImportError:
                    logger.info(f"Auto-installing {pip_name}...")
                    import subprocess
                    try:
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", "-q", pip_name],
                            timeout=120
                        )
                        installed[pip_name] = True
                        logger.info(f"Installed {pip_name}")
                    except Exception as e:
                        logger.error(f"Failed to install {pip_name}: {e}")
                        installed[pip_name] = False

        return installed

    def fix_cuda_oom(self, original_config: dict) -> dict:
        """Specialized fix for CUDA out of memory."""
        fix = dict(original_config)
        fix["batch_size"] = max(1, original_config.get("batch_size", 2) // 2)
        fix["gradient_accumulation_steps"] = original_config.get("gradient_accumulation_steps", 4) * 2
        fix["use_gradient_checkpointing"] = True
        fix["max_seq_length"] = max(64, original_config.get("max_seq_length", 2048) // 2)
        if "num_train_epochs" in fix:
            fix["num_train_epochs"] = max(1, fix["num_train_epochs"] - 1)
        logger.info(f"CUDA OOM fix applied: batch_size={fix['batch_size']}, "
                    f"grad_accum={fix['gradient_accumulation_steps']}")
        return fix

    def get_fix_summary(self) -> str:
        return f"Auto-fixer: {self._fix_count} errors fixed, {len(self._fix_rules)} rules loaded"

    def report_known_issues(self) -> List[dict]:
        """Report common Colab and ML training issues."""
        return [
            {
                "issue": "Runtime disconnects",
                "fix": "Use JavaScript reconnect snippet in Colab. Save checkpoints to Drive.",
                "common": True,
            },
            {
                "issue": "CUDA OOM on T4 (16GB)",
                "fix": "Use 4-bit quantization, batch_size=1-2, gradient checkpointing.",
                "common": True,
            },
            {
                "issue": "Slow dataset downloads",
                "fix": "Use streaming=True or cache to Google Drive.",
                "common": True,
            },
            {
                "issue": "Package conflicts in Colab",
                "fix": "Restart runtime after installs. Use `pip install --upgrade`.",
                "common": True,
            },
        ]
