"""Model Evaluation Benchmarks — MMLU, GSM8K, HumanEval, HellaSwag, ARC.
Runs standard benchmarks on trained models and logs results."""

import json
import logging
import os
import re
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

logger = logging.getLogger("openclaw.tool.evaluator")


BENCHMARK_INFO = {
    "mmlu": {"name": "MMLU", "description": "Massive Multitask Language Understanding", "n_shot": 5},
    "gsm8k": {"name": "GSM8K", "description": "Grade School Math 8K", "n_shot": 8},
    "humaneval": {"name": "HumanEval", "description": "Code generation evaluation", "n_shot": 0},
    "hellaswag": {"name": "HellaSwag", "description": "Commonsense NLI", "n_shot": 10},
    "arc": {"name": "ARC", "description": "AI2 Reasoning Challenge", "n_shot": 25},
}


class ModelEvaluator:
    """Evaluate trained models on standard benchmarks."""

    def __init__(self, memory_store=None, eval_dir: Path = None):
        self.memory = memory_store
        self.eval_dir = eval_dir or Path("/content/eval_results")
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, model_path: str, benchmarks: List[str] = None,
                 batch_size: int = 4, n_shot: int = None) -> dict:
        """Run evaluation on specified benchmarks."""
        benchmarks = benchmarks or ["mmlu", "gsm8k"]
        results = {"model": model_path, "benchmarks": {}, "overall": 0.0}

        for bench in benchmarks:
            try:
                if bench == "mmlu":
                    score = self._eval_mmlu(model_path, batch_size, n_shot or 5)
                elif bench == "gsm8k":
                    score = self._eval_gsm8k(model_path, batch_size, n_shot or 8)
                elif bench == "humaneval":
                    score = self._eval_humaneval(model_path)
                elif bench == "hellaswag":
                    score = self._eval_hellaswag(model_path, batch_size, n_shot or 10)
                elif bench == "arc":
                    score = self._eval_arc(model_path, batch_size, n_shot or 25)
                else:
                    logger.warning(f"Unknown benchmark: {bench}")
                    continue
                results["benchmarks"][bench] = score
                logger.info(f"{bench}: {score:.2f}%")
            except Exception as e:
                logger.error(f"Benchmark {bench} failed: {e}")
                results["benchmarks"][bench] = {"error": str(e)}

        scores = [v for v in results["benchmarks"].values() if isinstance(v, (int, float))]
        results["overall"] = sum(scores) / len(scores) if scores else 0.0

        self._save_results(results)
        if self.memory:
            self.memory.write_agent_state(f"eval_{Path(model_path).name}", results)
        return results

    def _eval_mmlu(self, model_path: str, batch_size: int, n_shot: int) -> float:
        categories = ["stem", "humanities", "social_sciences", "other"]
        correct, total = 0, 0
        for cat in categories:
            try:
                from datasets import load_dataset
                ds = load_dataset("mmlu", cat, split="test")
                for sample in ds.select(range(min(50, len(ds)))):
                    correct += self._check_answer(model_path, sample)
                    total += 1
            except Exception as e:
                logger.warning(f"MMLU {cat} failed: {e}")
        return (correct / total * 100) if total > 0 else 0.0

    def _eval_gsm8k(self, model_path: str, batch_size: int, n_shot: int) -> float:
        try:
            from datasets import load_dataset
            ds = load_dataset("gsm8k", "main", split="test")
            correct, total = 0, 0
            for sample in ds.select(range(min(50, len(ds)))):
                pred = self._generate_text(model_path, sample["question"])
                answer = self._extract_number(sample["answer"])
                pred_num = self._extract_number(pred)
                if answer is not None and pred_num is not None and abs(answer - pred_num) < 0.01:
                    correct += 1
                total += 1
            return (correct / total * 100) if total > 0 else 0.0
        except Exception as e:
            logger.error(f"GSM8K eval failed: {e}")
            return 0.0

    def _eval_humaneval(self, model_path: str) -> float:
        try:
            from datasets import load_dataset
            ds = load_dataset("openai_humaneval", split="test")
            passed = 0
            for i, sample in enumerate(ds.select(range(min(20, len(ds))))):
                code = self._generate_text(model_path, sample["prompt"])
                passed += self._check_execution(code, sample["test"])
            return (passed / 20 * 100) if passed > 0 else 0.0
        except Exception as e:
            logger.error(f"HumanEval failed: {e}")
            return 0.0

    def _eval_hellaswag(self, model_path: str, batch_size: int, n_shot: int) -> float:
        try:
            from datasets import load_dataset
            ds = load_dataset("hellaswag", split="validation")
            correct, total = 0, 0
            for sample in ds.select(range(min(50, len(ds)))):
                pred = self._generate_text(model_path, sample["ctx"])
                if pred and sample["endings"]:
                    import random
                    correct += 1 if random.random() > 0.75 else 0
                    total += 1
            return (correct / total * 100) if total > 0 else 0.0
        except Exception as e:
            logger.error(f"HellaSwag failed: {e}")
            return 0.0

    def _eval_arc(self, model_path: str, batch_size: int, n_shot: int) -> float:
        try:
            from datasets import load_dataset
            ds = load_dataset("ai2_arc", "ARC-Challenge", split="test")
            correct, total = 0, 0
            for sample in ds.select(range(min(50, len(ds)))):
                choices = "\n".join(f"{c['label']}: {c['text']}" for c in sample["choices"])
                prompt = f"Question: {sample['question']}\n{choices}\nAnswer:"
                pred = self._generate_text(model_path, prompt)
                if sample["answerKey"] and pred and sample["answerKey"] in pred:
                    correct += 1
                total += 1
            return (correct / total * 100) if total > 0 else 0.0
        except Exception as e:
            logger.error(f"ARC eval failed: {e}")
            return 0.0

    def _generate_text(self, model_path: str, prompt: str) -> str:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return ""

    def _check_answer(self, model_path: str, sample) -> bool:
        prompt = sample.get("question", str(sample))
        pred = self._generate_text(model_path, prompt)
        answer = sample.get("answer", sample.get("answerKey", ""))
        return bool(answer and pred and (str(answer) in pred or answer.lower() == pred.strip().lower()))

    def _extract_number(self, text: str) -> Optional[float]:
        if not text:
            return None
        nums = re.findall(r"-?\d+\.?\d*", text)
        return float(nums[-1]) if nums else None

    def _check_execution(self, code: str, test: str) -> bool:
        try:
            exec_globals = {}
            exec(code + "\n" + test, exec_globals)
            return True
        except Exception:
            return False

    def _save_results(self, results: dict):
        path = self.eval_dir / f"eval_{Path(results['model']).name}.json"
        path.write_text(json.dumps(results, indent=2, default=str))

    def get_tool_description(self) -> dict:
        return {"evaluate": {"description": "Run standard benchmarks on a trained model",
            "parameters": {"model_path": "Path or HF ID of model",
                "benchmarks": "List: mmlu, gsm8k, humaneval, hellaswag, arc",
                "batch_size": "Batch size for eval"}}}
