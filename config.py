import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


ENV_PREFIX = "OPENCLAW_"

__test__ = False


@dataclass
class AgentConfig:
    workspace_root: Path = Path("/content/openclaw_workspace")
    memory_dir: Path = Path("/content/openclaw_workspace/memory")
    skills_dir: Path = Path("/content/openclaw_workspace/skills")
    logs_dir: Path = Path("/content/openclaw_workspace/logs")
    models_dir: Path = Path("/content/openclaw_workspace/models")
    datasets_dir: Path = Path("/content/openclaw_workspace/datasets")

    huggingface_token: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    wandb_api_key: Optional[str] = None

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_ollama_url: str = "http://localhost:11434"

    heartbeat_interval_seconds: int = 60
    max_retries_on_error: int = 3
    max_concurrent_sessions: int = 2

    domain: Optional[str] = None
    domain_config: Optional[dict] = None

    default_training_config: dict = field(default_factory=lambda: {
        "model_name": "unsloth/mistral-7b-bnb-4bit",
        "max_seq_length": 2048,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "optim": "adamw_8bit",
        "warmup_steps": 10,
    })

    @classmethod
    def from_file(cls, path: str) -> "AgentConfig":
        cfg = cls()
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text())
            known = {f.name for f in cls.__dataclass_fields__.values()}
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    @classmethod
    def from_env(cls) -> "AgentConfig":
        cfg = cls()
        for field_info in cls.__dataclass_fields__.values():
            name = field_info.name
            env_key = ENV_PREFIX + name.upper()
            val = os.environ.get(env_key)
            if val is not None:
                python_type = field_info.type
                if python_type is bool or python_type == bool:
                    setattr(cfg, name, val.lower() in ("1", "true", "yes"))
                elif python_type is int or python_type == int:
                    setattr(cfg, name, int(val))
                elif python_type is float or python_type == float:
                    setattr(cfg, name, float(val))
                elif python_type is Path or python_type == Path:
                    setattr(cfg, name, Path(val))
                else:
                    setattr(cfg, name, val)
        return cfg

    def resolve_env(self):
        self.huggingface_token = self.huggingface_token or os.environ.get("HF_TOKEN")
        self.openai_api_key = self.openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.wandb_api_key = self.wandb_api_key or os.environ.get("WANDB_API_KEY")
        return self

    def to_dict(self) -> dict:
        d = {}
        for field_info in self.__dataclass_fields__.values():
            name = field_info.name
            val = getattr(self, name)
            if isinstance(val, Path):
                val = str(val)
            d[name] = val
        return d

    def validate(self, strict: bool = False) -> list:
        errors = []
        if self.llm_temperature < 0.0 or self.llm_temperature > 2.0:
            errors.append("llm_temperature must be between 0.0 and 2.0")
        if self.heartbeat_interval_seconds < 5:
            errors.append("heartbeat_interval_seconds must be >= 5")
        if self.max_concurrent_sessions < 1:
            errors.append("max_concurrent_sessions must be >= 1")
        if strict:
            if not self.openai_api_key and not self.anthropic_api_key:
                errors.append("No LLM API key configured (set OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        return errors
