"""Evaluation Benchmarks — auto-run MMLU, GSM8K, HumanEval after training."""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("epsionic.benchmarks")


_BENCHMARK_DEFS = {
    "mmlu": {
        "name": "MMLU",
        "description": "Massive Multitask Language Understanding (57 subjects)",
        "dataset": "mmlu",
        "metric": "accuracy",
        "command_template": "lm_eval --model hf --model_args pretrained={model_path} --tasks mmlu --num_fewshot 5 --batch_size auto --output_path {output}",
    },
    "gsm8k": {
        "name": "GSM8K",
        "description": "Grade School Math (8.5K math word problems)",
        "dataset": "gsm8k",
        "metric": "exact_match,flexible_extract",
        "command_template": "lm_eval --model hf --model_args pretrained={model_path} --tasks gsm8k --num_fewshot 5 --batch_size auto --output_path {output}",
    },
    "humaneval": {
        "name": "HumanEval",
        "description": "Hand-Written Coding Challenges (164 Python problems)",
        "dataset": "humaneval",
        "metric": "pass@1",
        "command_template": "lm_eval --model hf --model_args pretrained={model_path} --tasks humaneval --num_fewshot 0 --batch_size auto --output_path {output}",
    },
    "hellaswag": {
        "name": "HellaSwag",
        "description": "Commonsense NLI (sentence completion)",
        "dataset": "hellaswag",
        "metric": "accuracy",
        "command_template": "lm_eval --model hf --model_args pretrained={model_path} --tasks hellaswag --num_fewshot 0 --batch_size auto --output_path {output}",
    },
    "arc_challenge": {
        "name": "ARC Challenge",
        "description": "AI2 Reasoning Challenge (grade-school science)",
        "dataset": "arc_challenge",
        "metric": "accuracy",
        "command_template": "lm_eval --model hf --model_args pretrained={model_path} --tasks arc_challenge --num_fewshot 25 --batch_size auto --output_path {output}",
    },
}


class BenchmarkRunner:
    """Run standard evaluation benchmarks on trained models using lm-eval-harness."""

    def __init__(self, model_path: str | Path = None):
        self.model_path = str(model_path) if model_path else ""
        self.results: Dict[str, dict] = {}

    @staticmethod
    def list_benchmarks() -> Dict[str, dict]:
        return dict(_BENCHMARK_DEFS)

    def run(self, benchmark_key: str, model_path: str = None) -> dict:
        """Run a single benchmark and return results."""
        bench = _BENCHMARK_DEFS.get(benchmark_key)
        if not bench:
            valid = list(_BENCHMARK_DEFS.keys())
            raise ValueError(f"Unknown benchmark '{benchmark_key}'. Valid: {valid}")

        mp = model_path or self.model_path
        if not mp:
            raise ValueError("model_path is required")

        output_dir = tempfile.mkdtemp(prefix=f"bench_{benchmark_key}_")
        cmd = bench["command_template"].format(model_path=mp, output=output_dir)

        logger.info("Running benchmark %s: %s", benchmark_key, cmd)
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            return {"error": "Benchmark timed out after 3600s", "benchmark": benchmark_key}
        except FileNotFoundError:
            return {
                "error": "lm_eval not installed. Install with: pip install lm-eval",
                "benchmark": benchmark_key,
            }

        # Try to parse results from output file
        import glob
        results_files = list(Path(output_dir).rglob("results*.json"))
        metrics = {}
        if results_files:
            try:
                data = json.loads(results_files[0].read_text())
                metrics = data.get("results", {})
            except Exception:
                pass

        self.results[benchmark_key] = {
            "benchmark": bench["name"],
            "model": mp,
            "metrics": metrics,
            "stdout": stdout[-2000:],
            "stderr": stderr[-1000:],
            "returncode": returncode,
            "output_dir": output_dir,
        }
        return self.results[benchmark_key]

    def run_all(self, model_path: str = None) -> Dict[str, dict]:
        """Run all available benchmarks."""
        for key in _BENCHMARK_DEFS:
            try:
                self.run(key, model_path)
            except Exception as e:
                logger.error("Benchmark %s failed: %s", key, e)
                self.results[key] = {"error": str(e)}
        return self.results

    def summary(self) -> str:
        lines = ["## Benchmark Results", ""]
        for key, res in self.results.items():
            name = _BENCHMARK_DEFS.get(key, {}).get("name", key)
            if "error" in res:
                lines.append(f"- **{name}**: ❌ {res['error']}")
            else:
                metrics = res.get("metrics", {})
                scores = ", ".join(f"{k}={v}" for k, v in metrics.items())
                lines.append(f"- **{name}**: {scores or 'no metrics'}")
        return "\n".join(lines)

    def get_tool_description(self) -> dict:
        return {
            "name": "benchmark_runner",
            "description": "Run standard evaluation benchmarks (MMLU, GSM8K, HumanEval, HellaSwag, ARC) on trained models",
            "parameters": {
                "action": {"type": "string", "enum": ["list", "run", "run_all", "summary"]},
                "benchmark": {"type": "string", "optional": True},
                "model_path": {"type": "string", "optional": True},
            },
        }
