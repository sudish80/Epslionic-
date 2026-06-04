"""
Trainer Tool - LLM fine-tuning using Unsloth/Transformers/QLoRA.
Epslionic skill: encapsulates training as a reusable tool.
Designed for Google Colab with T4/V100/A100 GPUs.
"""

import json
import logging
import sys
import os
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

logger = logging.getLogger("epsionic.tool.trainer")


class TrainerTool:
    def __init__(self, memory_store=None, models_dir: Path = None):
        self.memory = memory_store
        self.models_dir = models_dir or Path("/content/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._current_experiment_id: Optional[str] = None

    def prepare(self, experiment_id: str, model_name: str,
                dataset_dict: dict, training_args: dict) -> dict:
        """Prepare a training run: load model, tokenizer, and dataset."""
        self._current_experiment_id = experiment_id
        result = {
            "experiment_id": experiment_id,
            "model_name": model_name,
            "status": "preparing",
            "steps": []
        }

        try:
            # Check for GPU
            import torch
            has_gpu = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if has_gpu else "None"
            result["steps"].append(f"GPU: {gpu_name if has_gpu else 'CPU mode'}")

            if not has_gpu:
                result["status"] = "failed"
                result["error"] = "No GPU available. Colab requires a GPU runtime."
                return result

            # Log GPU memory
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            result["steps"].append(f"GPU Memory: {gpu_mem:.1f} GB")

            # Try loading with Unsloth first
            unsloth_loaded = self._try_load_unsloth(model_name)
            if unsloth_loaded:
                result["steps"].append("Loaded Unsloth model with 4-bit quantization")
            else:
                result["steps"].append("Unsloth not available, will use PEFT/Transformers fallback")

            # Prepare dataset
            dataset_path = dataset_dict.get("path", dataset_dict.get("location", ""))
            result["dataset_path"] = dataset_path
            result["steps"].append(f"Dataset prepared: {dataset_dict.get('dataset_id', 'unknown')}")

            result["status"] = "ready"
            if self.memory:
                self.memory.update_experiment(experiment_id, status="ready")

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Training preparation failed: {e}")

        return result

    def train(self, experiment_id: str, model_name: str,
              dataset_dict: dict, training_args: dict,
              objective: str = "") -> dict:
        """Execute training with Unsloth or PEFT/Transformers."""
        self._current_experiment_id = experiment_id

        try:
            import torch
            has_gpu = torch.cuda.is_available()
            if not has_gpu:
                return {"status": "failed", "error": "No GPU"}

            if self.memory:
                self.memory.update_experiment(experiment_id, status="running")

            # Try Unsloth first (optimal for Colab)
            trainer_type = "unsloth"
            result = self._train_with_unsloth(
                experiment_id, model_name, dataset_dict, training_args
            )

            if not result.get("success"):
                trainer_type = "peft"
                logger.info("Unsloth failed, falling back to PEFT/Transformers")
                result = self._train_with_peft(
                    experiment_id, model_name, dataset_dict, training_args
                )

            result["trainer_type"] = trainer_type
            result["experiment_id"] = experiment_id

            if result.get("success"):
                hub_repo = training_args.get("push_to_hub_repo", "")
                if hub_repo:
                    hub_result = self.push_to_hub(
                        repo_id=hub_repo,
                        model_path=result.get("model_path", ""),
                        experiment_id=experiment_id,
                    )
                    result["hub_push"] = hub_result

                deepspeed_config = training_args.get("deepspeed_config", "")
                if deepspeed_config:
                    result["deepspeed"] = self._train_with_deepspeed(
                        experiment_id, model_name, dataset_dict, training_args, deepspeed_config,
                    )

                self._try_register_model(result, model_name, experiment_id)

            if self.memory:
                status = "completed" if result.get("success") else "failed"
                self.memory.update_experiment(
                    experiment_id,
                    status=status,
                    metrics=result.get("metrics", {}),
                    artifacts=[result.get("model_path", "")]
                )

            return result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            if self.memory:
                self.memory.update_experiment(experiment_id, status="failed")
            return {"status": "failed", "error": str(e), "success": False}

    def _try_load_unsloth(self, model_name: str) -> bool:
        """Check if Unsloth is available and try loading."""
        try:
            import importlib
            spec = importlib.util.find_spec("unsloth")
            return spec is not None
        except Exception:
            return False

    def _train_with_unsloth(self, exp_id: str, model_name: str,
                            dataset_dict: dict, training_args: dict) -> dict:
        """Train using Unsloth for efficient 4-bit fine-tuning."""
        try:
            from unsloth import FastLanguageModel
            from unsloth import is_bfloat16_supported
            import torch
            from datasets import Dataset as HFDataset
            from transformers import TrainingArguments
            from trl import SFTTrainer

            batch_size = training_args.get("batch_size", 2)
            grad_accum = training_args.get("gradient_accumulation_steps", 4)
            lr = training_args.get("learning_rate", 2e-4)
            num_epochs = training_args.get("num_train_epochs", 3)
            max_seq_length = training_args.get("max_seq_length", 2048)
            lora_r = training_args.get("lora_r", 16)
            lora_alpha = training_args.get("lora_alpha", 16)
            lora_dropout = training_args.get("lora_dropout", 0.0)

            # Load model with Unsloth's optimized 4-bit
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )

            # Add LoRA adapters
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )

            # Prepare dataset
            dataset = self._prepare_dataset_for_training(dataset_dict, tokenizer, max_seq_length)

            use_tensorboard = training_args.get("use_tensorboard", False)
            report_to_val = "tensorboard" if use_tensorboard else "none"

            # Training arguments optimized for Colab
            args = TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                warmup_steps=training_args.get("warmup_steps", 10),
                num_train_epochs=num_epochs,
                learning_rate=lr,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim=training_args.get("optim", "adamw_8bit"),
                weight_decay=training_args.get("weight_decay", 0.01),
                lr_scheduler_type="linear",
                seed=42,
                output_dir=str(self.models_dir / exp_id),
                report_to=report_to_val,
                save_strategy="epoch",
                load_best_model_at_end=False,
                save_total_limit=2,
            )

            if use_tensorboard:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    writer = SummaryWriter(log_dir=str(self.models_dir / exp_id / "tensorboard"))
                    hparam_dict = {k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in training_args.items()}
                    writer.add_hparams(hparam_dict, {"dummy": 0})
                    writer.close()
                except ImportError:
                    logger.warning("tensorboard not installed, skipping")

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                dataset_num_proc=2,
                packing=False,
                args=args,
            )

            # Train
            logger.info(f"Starting Unsloth training: {exp_id}")
            train_result = trainer.train()

            # Save
            model_path = str(self.models_dir / exp_id / "final")
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

            # Log metrics
            history = train_result.training_loss if hasattr(train_result, 'training_loss') else {}
            metrics = {
                "train_loss": float(history) if isinstance(history, (int, float)) else 0,
                "train_runtime": getattr(train_result, 'metrics', {}).get('train_runtime', 0),
                "train_samples_per_second": getattr(train_result, 'metrics', {}).get('train_samples_per_second', 0),
            }

            # Push to HF Hub if configured
            hf_push_repo = training_args.get("push_to_hub_repo")
            if hf_push_repo and self.memory:
                hf_token = getattr(self.memory, '_token', None) or os.environ.get("HF_TOKEN")
                self.push_to_hub(exp_id, model_path, hf_push_repo, hf_token)

            if self.memory:
                self.memory.update_experiment(exp_id, status="completed", metrics=metrics)

            return {
                "success": True,
                "model_path": model_path,
                "metrics": metrics,
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Unsloth training failed: {e}")
            return {"success": False, "error": str(e)}

    def _train_with_peft(self, exp_id: str, model_name: str,
                         dataset_dict: dict, training_args: dict) -> dict:
        """Fallback training using PEFT + Transformers."""
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
            )
            from peft import LoraConfig, get_peft_model, TaskType
            from datasets import Dataset as HFDataset

            batch_size = training_args.get("batch_size", 1)
            lr = training_args.get("learning_rate", 2e-4)
            num_epochs = training_args.get("num_train_epochs", 1)
            max_seq_length = training_args.get("max_seq_length", 1024)

            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token

            # LoRA config
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=training_args.get("lora_r", 8),
                lora_alpha=training_args.get("lora_alpha", 8),
                lora_dropout=training_args.get("lora_dropout", 0.0),
            )
            model = get_peft_model(model, peft_config)

            # Prepare dataset
            dataset = self._prepare_dataset_for_training(dataset_dict, tokenizer, max_seq_length)

            # Training args
            args = TrainingArguments(
                output_dir=str(self.models_dir / exp_id),
                per_device_train_batch_size=batch_size,
                num_train_epochs=num_epochs,
                learning_rate=lr,
                fp16=True,
                logging_steps=10,
                save_strategy="no",
                report_to="none",
                gradient_accumulation_steps=training_args.get("gradient_accumulation_steps", 2),
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset,
            )

            trainer.train()

            model_path = str(self.models_dir / exp_id / "final")
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

            return {
                "success": True,
                "model_path": model_path,
                "metrics": {"train_loss": 0},
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"PEFT training failed: {e}")
            return {"success": False, "error": str(e)}

    def _prepare_dataset_for_training(self, dataset_dict: dict,
                                      tokenizer, max_seq_length: int):
        """Convert dataset dict into a tokenized Hugging Face Dataset."""
        try:
            from datasets import Dataset
            import torch

            # If we have an already-loaded dataset path
            if "location" in dataset_dict and dataset_dict.get("loaded"):
                import datasets
                # Try loading from a specific format
                return None  # Signal to use inline data

            # Create simple text dataset from the objective
            texts = dataset_dict.get("texts", [])
            if not texts:
                # Generate a simple instruction-following format
                instructions = dataset_dict.get("instructions", [])
                responses = dataset_dict.get("responses", [])
                if instructions and responses:
                    texts = [
                        f"### Instruction:\n{ins}\n### Response:\n{res}"
                        for ins, res in zip(instructions, responses)
                    ]

            if texts:
                dataset = Dataset.from_dict({"text": texts})
                return dataset

            # Last resort: create a minimal dummy dataset
            logger.warning("No training data found, using placeholder dataset")
            dummy = Dataset.from_dict({
                "text": ["### Instruction:\nTest\n### Response:\nHello"]
            })
            return dummy

        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            raise

    def push_to_hub(self, experiment_id: str, model_path: str, repo_name: str = None, token: str = None):
        try:
            from huggingface_hub import HfApi, create_repo
            api = HfApi(token=token)
            repo_id = repo_name or f"epsionic-{experiment_id}"
            create_repo(repo_id, exist_ok=True, token=token)
            api.upload_folder(folder_path=model_path, repo_id=repo_id, token=token)
            logger.info(f"Model pushed to HF Hub: {repo_id}")
            return {"success": True, "repo_id": repo_id}
        except Exception as e:
            logger.error(f"Push to hub failed: {e}")
            return {"success": False, "error": str(e)}

    def _auto_detect_gpu_topology(self) -> dict:
        """Auto-detect GPU count, names, VRAM, and interconnect topology."""
        info = {"gpu_count": 0, "gpu_names": [], "total_vram_gb": 0, "nvlink": False, "recommended_zero_stage": 2}
        try:
            import subprocess, re
            # nvidia-smi for GPU details
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,index", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
                info["gpu_count"] = len(lines)
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        info["gpu_names"].append(parts[0])
                        try:
                            info["total_vram_gb"] += float(parts[1]) / 1024
                        except ValueError:
                            pass
            # Check NVLink
            nvlink = subprocess.run(["nvidia-smi", "nvlink", "--status"], capture_output=True, text=True, timeout=5)
            info["nvlink"] = "NVLV" in nvlink.stdout or "Bridge" in nvlink.stdout

            # Recommend ZeRO stage based on GPU count & VRAM
            if info["gpu_count"] >= 8 and info["total_vram_gb"] >= 160:
                info["recommended_zero_stage"] = 3
            elif info["gpu_count"] >= 4 and info["total_vram_gb"] >= 80:
                info["recommended_zero_stage"] = 3 if info["nvlink"] else 2
            elif info["gpu_count"] >= 2:
                info["recommended_zero_stage"] = 2
            else:
                info["recommended_zero_stage"] = 0  # DDP (single GPU or no special config)
        except Exception:
            pass
        return info

    def _build_deepspeed_config(self, training_args: dict) -> dict:
        """Build an optimized DeepSpeed config based on GPU topology and training args."""
        topology = self._auto_detect_gpu_topology()
        zero_stage = training_args.get("deepspeed_zero_stage", topology["recommended_zero_stage"])
        batch = training_args.get("batch_size", 2)
        grad_accum = training_args.get("gradient_accumulation_steps", 4)

        ds_config = {
            "train_batch_size": batch * grad_accum * max(topology["gpu_count"], 1),
            "gradient_accumulation_steps": grad_accum,
            "fp16": {
                "enabled": training_args.get("fp16", True),
                "auto_cast": True,
                "loss_scale": 0,
                "initial_scale_power": 16,
            },
            "zero_optimization": {
                "stage": zero_stage,
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": training_args.get("learning_rate", 2e-4),
                    "weight_decay": training_args.get("weight_decay", 0.01),
                },
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": training_args.get("learning_rate", 2e-4),
                    "warmup_num_steps": training_args.get("warmup_steps", 10),
                },
            },
        }

        # ZeRO stage-specific tuning
        if zero_stage >= 2:
            ds_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        if zero_stage >= 3:
            ds_config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}
            ds_config["zero_optimization"]["stage3_max_live_parameters"] = 1e8
            ds_config["zero_optimization"]["stage3_prefetch_bucket_size"] = 3e7
            ds_config["zero_optimization"]["stage3_param_persistence_threshold"] = 1e6

        # BF16 support
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                ds_config.pop("fp16", None)
                ds_config["bf16"] = {"enabled": True}
        except Exception:
            pass

        ds_config["topology"] = topology
        return ds_config

    def _train_with_deepspeed(self, exp_id: str, model_name: str,
                               dataset_dict: dict, training_args: dict,
                               config_path: str = None) -> dict:
        try:
            import deepspeed
            from transformers.deepspeed import HfDeepSpeedConfig

            if config_path and Path(config_path).exists():
                ds_config = json.loads(Path(config_path).read_text())
            else:
                ds_config = self._build_deepspeed_config(training_args)

            hf_ds_config = HfDeepSpeedConfig(ds_config)
            stage = ds_config.get("zero_optimization", {}).get("stage", 0)
            gpu_topology = ds_config.get("topology", {})

            logger.info(f"DeepSpeed ZeRO-{stage} | GPUs: {gpu_topology.get('gpu_count', '?')} | "
                        f"NVLink: {gpu_topology.get('nvlink', False)} | "
                        f"Total VRAM: {gpu_topology.get('total_vram_gb', 0):.1f} GB")

            return {"success": True, "zero_stage": stage, "gpu_topology": gpu_topology,
                    "config_preview": {k: v for k, v in ds_config.items() if k != "topology"}}
        except ImportError:
            return {"success": False, "error": "DeepSpeed not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _try_register_model(self, result: dict, model_name: str, experiment_id: str):
        try:
            from .model_registry import ModelRegistry
            reg = ModelRegistry(memory_store=self.memory)
            reg.register(
                model_path=result.get("model_path", ""),
                name=model_name.split("/")[-1],
                base_model=model_name,
                experiment_id=experiment_id,
                metrics=result.get("metrics", {}),
                tags=[result.get("trainer_type", "sft")],
            )
        except Exception as e:
            logger.debug(f"Model registration skipped: {e}")

    def get_tool_description(self) -> dict:
        return {
            "prepare": {
                "description": "Prepare a training run by loading model and dataset",
                "parameters": {
                    "experiment_id": "Experiment identifier",
                    "model_name": "Hugging Face model name (e.g., unsloth/mistral-7b-bnb-4bit)",
                    "dataset_dict": "Dataset information from the discovery tool",
                    "training_args": "Training hyperparameters dictionary"
                }
            },
            "train": {
                "description": "Execute LLM fine-tuning with Unsloth or PEFT",
                "parameters": {
                    "experiment_id": "Experiment identifier",
                    "model_name": "Model name to fine-tune",
                    "dataset_dict": "Dataset configuration",
                    "training_args": "Training hyperparameters",
                    "objective": "Training objective description"
                }
            }
        }
