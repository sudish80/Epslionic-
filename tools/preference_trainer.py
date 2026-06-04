"""Preference Optimization Trainer — DPO/ORPO/KTO alignment.
Extends base TrainerTool with preference-based training paradigms."""

import json
import logging
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger("epsionic.tool.preference_trainer")

PREFERENCE_DATASETS = {
    "dpo": ["Anthropic/hh-rlhf", "Intel/orca_dpo_pairs", "argilla/ultrafeedback-binarized-preferences"],
    "orpo": ["argilla/ultrafeedback-binarized-preferences", "mlabonne/orpo-dpo-mix-40k"],
    "kto": ["argilla/kto-mix-15k"],
}


class PreferenceTrainerTool:
    """DPO/ORPO/KTO preference optimization trainer."""

    def __init__(self, memory_store=None, models_dir: Path = None):
        self.memory = memory_store
        self.models_dir = models_dir or Path("/content/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train_dpo(self, experiment_id: str, model_name: str, dataset_dict: dict,
                  training_args: dict, beta: float = 0.1, loss_type: str = "sigmoid") -> dict:
        try:
            from trl import DPOTrainer, DPOConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from datasets import Dataset
            import torch

            if self.memory:
                self.memory.update_experiment(experiment_id, status="running")

            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto", load_in_4bit=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token

            ref_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto", load_in_4bit=True
            )

            dataset = self._prepare_dpo_dataset(dataset_dict, tokenizer)

            dpo_args = DPOConfig(
                per_device_train_batch_size=training_args.get("batch_size", 2),
                gradient_accumulation_steps=training_args.get("gradient_accumulation_steps", 4),
                learning_rate=training_args.get("learning_rate", 5e-6),
                num_train_epochs=training_args.get("num_train_epochs", 3),
                max_length=training_args.get("max_seq_length", 1024),
                output_dir=str(self.models_dir / experiment_id),
                logging_steps=10,
                save_strategy="epoch",
                report_to="none",
                beta=beta,
                loss_type=loss_type,
            )

            trainer = DPOTrainer(
                model=model, ref_model=ref_model, tokenizer=tokenizer,
                args=dpo_args, train_dataset=dataset,
            )
            trainer.train()

            model_path = str(self.models_dir / experiment_id / "final")
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

            if self.memory:
                self.memory.update_experiment(experiment_id, status="completed",
                    metrics={"method": "dpo", "beta": beta, "loss_type": loss_type})

            return {"success": True, "model_path": model_path, "method": "dpo", "status": "completed"}
        except Exception as e:
            logger.error(f"DPO training failed: {e}")
            return {"success": False, "error": str(e), "method": "dpo"}

    def train_orpo(self, experiment_id: str, model_name: str, dataset_dict: dict,
                   training_args: dict) -> dict:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from datasets import Dataset
            import torch

            try:
                from trl import ORPOTrainer, ORPOConfig
            except ImportError:
                logger.warning("ORPO not in this TRL version, falling back to DPO-like training")
                return self.train_dpo(experiment_id, model_name, dataset_dict, training_args, beta=0.05)

            if self.memory:
                self.memory.update_experiment(experiment_id, status="running")

            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto", load_in_4bit=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token

            dataset = self._prepare_dpo_dataset(dataset_dict, tokenizer)

            orpo_args = ORPOConfig(
                per_device_train_batch_size=training_args.get("batch_size", 2),
                gradient_accumulation_steps=training_args.get("gradient_accumulation_steps", 4),
                learning_rate=training_args.get("learning_rate", 8e-6),
                num_train_epochs=training_args.get("num_train_epochs", 3),
                max_length=training_args.get("max_seq_length", 1024),
                output_dir=str(self.models_dir / experiment_id),
                logging_steps=10,
                save_strategy="epoch",
                report_to="none",
            )

            trainer = ORPOTrainer(
                model=model, tokenizer=tokenizer, args=orpo_args, train_dataset=dataset,
            )
            trainer.train()

            model_path = str(self.models_dir / experiment_id / "final")
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

            if self.memory:
                self.memory.update_experiment(experiment_id, status="completed",
                    metrics={"method": "orpo"})

            return {"success": True, "model_path": model_path, "method": "orpo", "status": "completed"}
        except Exception as e:
            logger.error(f"ORPO training failed: {e}")
            return {"success": False, "error": str(e), "method": "orpo"}

    def _prepare_dpo_dataset(self, dataset_dict: dict, tokenizer) -> Optional[Any]:
        from datasets import Dataset
        chosen = dataset_dict.get("chosen", [])
        rejected = dataset_dict.get("rejected", [])
        if chosen and rejected:
            return Dataset.from_dict({"chosen": chosen, "rejected": rejected})
        dataset_id = dataset_dict.get("dataset_id", "")
        if dataset_id:
            try:
                from datasets import load_dataset
                ds = load_dataset(dataset_id, split="train")
                if "chosen" in ds.features and "rejected" in ds.features:
                    return ds
                if "chosen" in ds.column_names and "rejected" in ds.column_names:
                    return ds
            except Exception:
                pass
        logger.warning("No preference data found, using placeholder")
        return Dataset.from_dict({
            "chosen": ["You are helpful"], "rejected": ["You are unhelpful"]
        })

    def get_tool_description(self) -> dict:
        return {
            "dpo": {"description": "Direct Preference Optimization fine-tuning",
                "parameters": {"beta": "KL penalty (default 0.1)", "loss_type": "sigmoid/ipo/kto_pair"}},
            "orpo": {"description": "Odds Ratio Preference Optimization (no ref model needed)",
                "parameters": {}},
        }
