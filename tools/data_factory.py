"""Synthetic Data Generation — Self-Instruct, Evol-Instruct, augmentation.
Bootstraps training datasets from seed tasks using teacher LLMs."""

import json
import logging
import os
import random
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger("openclaw.tool.data_factory")

SYNTHETIC_SEED_TASKS = {
    "math": ["Solve this math problem step by step", "Explain the concept of derivatives",
             "Prove the Pythagorean theorem", "What is the limit of (sin x)/x as x approaches 0?"],
    "code": ["Write a Python function to sort a list", "Explain the difference between lists and tuples",
             "Debug this code: print('hello'", "Write a binary search implementation"],
    "general": ["Explain quantum computing in simple terms", "Write a short story about AI",
                "Compare democracy and autocracy", "What are the causes of climate change?"],
}


class SyntheticDataGenerator:
    """Generate synthetic training data using teacher LLMs."""

    def __init__(self, memory_store=None, datasets_dir: Path = None,
                 api_key: str = None):
        self.memory = memory_store
        self.datasets_dir = datasets_dir or Path("/content/datasets")
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def self_instruct(self, seed_tasks: List[str], num_samples: int = 100,
                      teacher_model: str = "gpt-4o-mini", output_name: str = "synthetic_self_instruct") -> dict:
        """Self-Instruct pipeline: seed -> generate -> filter -> deduplicate."""
        generated = []
        batch_size = min(10, num_samples)

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            prompts = [random.choice(seed_tasks) for _ in range(batch_end - batch_start)]
            batch_results = self._batch_generate(prompts, teacher_model)
            generated.extend(batch_results)
            logger.info(f"Generated {len(generated)}/{num_samples} samples")

        filtered = self._filter_quality(generated)
        deduplicated = self._deduplicate(filtered)

        output_path = self._save_dataset(deduplicated, output_name)
        if self.memory:
            self.memory.record_dataset(f"synthetic/{output_name}", {
                "num_samples": len(deduplicated), "teacher": teacher_model,
                "method": "self_instruct", "raw_generated": len(generated),
                "filtered": len(filtered), "deduplicated": len(deduplicated),
            })
        return {"success": True, "path": output_path, "num_samples": len(deduplicated),
                "generated": len(generated), "filtered": len(filtered)}

    def evol_instruct(self, seed_instructions: List[str], num_rounds: int = 2,
                      teacher_model: str = "gpt-4o-mini", output_name: str = "synthetic_evol") -> dict:
        """Evol-Instruct: deepen, complicate, and constrain seed instructions."""
        evolved = list(seed_instructions)
        evolution_types = ["add_constraint", "deepen", "complicate", "concretize"]

        for round_num in range(num_rounds):
            new_instructions = []
            for inst in evolved:
                evo_type = random.choice(evolution_types)
                prompt = self._build_evolution_prompt(inst, evo_type)
                result = self._llm_generate(prompt, teacher_model)
                if result and len(result) > len(inst) * 1.2:
                    new_instructions.append(result)
            evolved.extend(new_instructions)
            logger.info(f"Evol round {round_num + 1}: {len(evolved)} instructions")

        responses = self._batch_generate(evolved, teacher_model)
        pairs = [{"instruction": evolved[i], "response": responses[i]}
                 for i in range(len(responses)) if responses[i]]

        output_path = self._save_dataset(pairs, output_name)
        return {"success": True, "path": output_path, "num_samples": len(pairs)}

    def augment_dataset(self, dataset_id: str, teacher_model: str = "gpt-4o-mini",
                        augmentation_factor: int = 2, output_name: str = None) -> dict:
        """Augment an existing dataset by generating paraphrases and variations."""
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_id, split="train")
            samples = list(ds.select(range(min(100, len(ds)))))
        except Exception as e:
            return {"success": False, "error": str(e)}

        augmented = []
        for sample in samples:
            text = str(sample.get("text", sample.get("instruction", json.dumps(sample))))
            prompt = f"Paraphrase this text in 3 different ways:\n{text[:500]}"
            result = self._llm_generate(prompt, teacher_model)
            if result:
                variations = result.strip().split("\n")
                for var in variations[:augmentation_factor]:
                    if var.strip():
                        augmented.append({"original": text[:200], "variation": var.strip()})

        output_name = output_name or f"{dataset_id.replace('/', '_')}_augmented"
        output_path = self._save_dataset(augmented, output_name)
        return {"success": True, "path": output_path, "num_samples": len(augmented)}

    def _batch_generate(self, prompts: List[str], model: str) -> List[str]:
        results = []
        for prompt in prompts:
            result = self._llm_generate(
                f"Generate a high-quality response for this instruction:\n{prompt}\n\nResponse:",
                model
            )
            results.append(result or "")
        return results

    def _llm_generate(self, prompt: str, model: str) -> Optional[str]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=model, messages=[
                    {"role": "system", "content": "You generate high-quality training data."},
                    {"role": "user", "content": prompt},
                ], temperature=0.8, max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return None

    def _build_evolution_prompt(self, instruction: str, evo_type: str) -> str:
        templates = {
            "add_constraint": f"Rewrite this instruction adding a specific constraint: {instruction}",
            "deepen": f"Make this instruction more challenging and requiring deeper reasoning: {instruction}",
            "complicate": f"Add complexity to this instruction: {instruction}",
            "concretize": f"Make this instruction more specific and concrete: {instruction}",
        }
        return templates.get(evo_type, templates["deepen"])

    def _filter_quality(self, samples: List[str]) -> List[str]:
        filtered = []
        for s in samples:
            if len(s) < 20 or len(s) > 4096:
                continue
            if s.count(" ") < 3:
                continue
            filtered.append(s)
        return filtered

    def _deduplicate(self, samples: List[str]) -> List[str]:
        seen = set()
        unique = []
        for s in samples:
            h = hash(s[:100])
            if h not in seen:
                seen.add(h)
                unique.append(s)
        return unique

    def _save_dataset(self, data: List, name: str) -> str:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.datasets_dir / f"{name}_{ts}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
        return str(path)

    def get_tool_description(self) -> dict:
        return {
            "self_instruct": {"description": "Generate synthetic data from seed tasks using Self-Instruct",
                "parameters": {"seed_tasks": "List of seed instructions", "num_samples": "Number to generate",
                    "teacher_model": "LLM to generate with"}},
            "evol_instruct": {"description": "Evolve instructions to be more complex (Evol-Instruct)",
                "parameters": {"seed_instructions": "Base instructions", "num_rounds": "Evolution rounds"}},
            "augment_dataset": {"description": "Augment existing dataset with paraphrases",
                "parameters": {"dataset_id": "HF dataset ID", "augmentation_factor": "Multiplier"}},
        }
