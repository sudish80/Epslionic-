__version__ = "1.0.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))

__all__ = [
    "__version__",
    "AgentConfig",
    "EpslionicError",
    "ConfigurationError",
    "EnvironmentError",
    "GPUError",
    "TrainingError",
    "DomainError",
    "AuthenticationError",
    "error_code",
    "get_gateway",
    "get_dashboard",
    "mount_drive",
    "get_secret",
    "save_checkpoint",
    "load_checkpoint",
    "safe_print",
    "strip_emoji",
    "set_emoji_mode",
]


def __getattr__(name):
    import importlib

    _LAZY_MAP = {
        "AgentConfig": (".config", "AgentConfig"),
        "EpslionicError": (".exceptions", "EpslionicError"),
        "ConfigurationError": (".exceptions", "ConfigurationError"),
        "EnvironmentError": (".exceptions", "EnvironmentError"),
        "GPUError": (".exceptions", "GPUError"),
        "TrainingError": (".exceptions", "TrainingError"),
        "DomainError": (".exceptions", "DomainError"),
        "AuthenticationError": (".exceptions", "AuthenticationError"),
        "error_code": (".exceptions", "error_code"),
        "Gateway": (".core.gateway", "Gateway"),
        "DomainSelector": (".core.domain", "DomainSelector"),
        "DOMAINS": (".core.domain", "DOMAINS"),
        "MemoryStore": (".core.memory", "MemoryStore"),
    "DatasetDiscoveryTool": (".tools.dataset_discovery", "DatasetDiscoveryTool"),
    "TrainerTool": (".tools.trainer", "TrainerTool"),
    "AutoFixerTool": (".tools.auto_fixer", "AutoFixerTool"),
    "PreferenceTrainerTool": (".tools.preference_trainer", "PreferenceTrainerTool"),
    "ModelMerger": (".tools.model_merger", "ModelMerger"),
    "SyntheticDataGenerator": (".tools.data_factory", "SyntheticDataGenerator"),
    "ModelEvaluator": (".tools.evaluator", "ModelEvaluator"),
    "ModelServer": (".tools.model_server", "ModelServer"),
    "ModelPlayground": (".tools.playground", "ModelPlayground"),
    "ExperimentTracker": (".tools.experiment_tracker", "ExperimentTracker"),
    "HyperparameterOptimizer": (".tools.optimizer", "HyperparameterOptimizer"),
    "ModelQuantizer": (".tools.quantizer", "ModelQuantizer"),
    "CostTracker": (".tools.cost_tracker", "CostTracker"),
    "Notifier": (".tools.notifier", "Notifier"),
    "mount_drive": (".utils.colab", "mount_drive"),
    "get_secret": (".utils.colab", "get_secret"),
    "save_checkpoint": (".utils.colab", "save_checkpoint"),
    "load_checkpoint": (".utils.colab", "load_checkpoint"),
    "safe_print": (".utils.format", "safe_print"),
    "strip_emoji": (".utils.format", "strip_emoji"),
    "set_emoji_mode": (".utils.format", "set_emoji_mode"),
    }

    if name in _LAZY_MAP:
        mod_path, attr_name = _LAZY_MAP[name]
        mod = importlib.import_module(mod_path, __package__)
        return getattr(mod, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_gateway(config=None, auto_install_deps=True):
    from pathlib import Path
    from .config import AgentConfig
    from .core.gateway import Gateway
    from .tools.dataset_discovery import DatasetDiscoveryTool
    from .tools.trainer import TrainerTool
    from .tools.auto_fixer import AutoFixerTool
    from .tools.preference_trainer import PreferenceTrainerTool
    from .tools.model_merger import ModelMerger
    from .tools.data_factory import SyntheticDataGenerator
    from .tools.evaluator import ModelEvaluator
    from .tools.model_server import ModelServer
    from .tools.playground import ModelPlayground
    from .tools.experiment_tracker import ExperimentTracker
    from .tools.optimizer import HyperparameterOptimizer
    from .tools.quantizer import ModelQuantizer
    from .tools.cost_tracker import CostTracker
    from .tools.notifier import Notifier

    cfg = config or AgentConfig.from_env().resolve_env()

    if auto_install_deps:
        try:
            from .epsionic_agent_complete import auto_install
            has_gpu = False
            try:
                import torch; has_gpu = torch.cuda.is_available()
            except ImportError:
                pass
            auto_install(essential_only=not has_gpu)
        except ImportError:
            pass

    gw = Gateway(cfg)

    ds_tool = DatasetDiscoveryTool(gw.memory, Path(cfg.datasets_dir) if not isinstance(cfg.datasets_dir, Path) else cfg.datasets_dir, cfg.huggingface_token)
    tr_tool = TrainerTool(gw.memory, cfg.models_dir)
    fx_tool = AutoFixerTool(gw.memory)
    pref_tool = PreferenceTrainerTool(gw.memory, cfg.models_dir)
    merger_tool = ModelMerger(cfg.models_dir)
    factory_tool = SyntheticDataGenerator()
    evaluator_tool = ModelEvaluator()
    server_tool = ModelServer()
    playground_tool = ModelPlayground()
    tracker_tool = ExperimentTracker()
    opt_tool = HyperparameterOptimizer(memory_store=gw.memory)
    quant_tool = ModelQuantizer(models_dir=cfg.models_dir)
    cost_tool = CostTracker(memory_store=gw.memory)
    notif_tool = Notifier()

    gw.register_tool("discover", "Search HuggingFace datasets",
        lambda **kw: (
            ds_tool.search(keywords=kw.get("query", ""), max_results=15)
            if kw.get("query") else
            ds_tool.load(dataset_id=kw.get("dataset_id", ""))
        ))
    gw.register_tool("train", "Fine-tune LLM",
        lambda **kw: tr_tool.train(
            experiment_id=kw.get("experiment_id", "exp"),
            model_name=kw.get("model_name", cfg.default_training_config.get("model_name", "unsloth/mistral-7b-bnb-4bit")),
            dataset_dict=kw.get("dataset_dict", {}),
            training_args=kw.get("training_args", cfg.default_training_config),
            objective=kw.get("objective", ""),
        ))
    gw.register_tool("prepare", "Prepare training environment",
        lambda **kw: tr_tool.prepare(
            experiment_id=kw.get("experiment_id", "exp"),
            model_name=kw.get("model_name", cfg.default_training_config.get("model_name", "unsloth/mistral-7b-bnb-4bit")),
        ))
    gw.register_tool("auto_fix", "Analyze and fix errors",
        lambda **kw: fx_tool.fix(
            error=kw.get("error", ""),
            context=kw.get("context", {}),
            brain_fix=kw.get("brain_fix"),
        ))
    gw.register_tool("preference_train", "DPO/ORPO alignment training",
        lambda **kw: pref_tool.train_dpo(
            experiment_id=kw.get("experiment_id", "pref_exp"),
            model_name=kw.get("model_name", cfg.default_training_config.get("model_name", "unsloth/mistral-7b-bnb-4bit")),
            dataset_dict=kw.get("dataset_dict", {}),
            training_args=kw.get("training_args", cfg.default_training_config),
        ))
    gw.register_tool("merge_models", "Merge LoRA adapters (SLERP/TIES/Linear)",
        lambda **kw: merger_tool.merge_loras(
            model_name=kw.get("model_name", cfg.default_training_config.get("model_name", "unsloth/mistral-7b-bnb-4bit")),
            adapter_paths=kw.get("adapter_paths", []),
            merge_method=kw.get("method", "linear"),
        ))
    gw.register_tool("generate_data", "Generate synthetic training data",
        lambda **kw: factory_tool.self_instruct(
            seed_tasks=kw.get("seed_tasks", ["Generate a training example"]),
            num_samples=kw.get("num_samples", 100),
        ))
    gw.register_tool("evaluate", "Run evaluation benchmarks",
        lambda **kw: evaluator_tool.evaluate(
            model_path=kw.get("model_path", ""),
            benchmarks=kw.get("benchmarks", ["mmlu"]),
        ))
    gw.register_tool("serve_model", "Start model serving endpoint",
        lambda **kw: server_tool.deploy_vllm(
            model_path=kw.get("model_path", ""),
            port=kw.get("port", 8001),
        ))
    gw.register_tool("chat", "Interactive model playground",
        lambda **kw: playground_tool.generate(
            message=kw.get("message", ""),
            system_prompt=kw.get("system_prompt", ""),
            max_tokens=kw.get("max_tokens", 256),
        ))
    gw.register_tool("track_run", "Initialize experiment tracking",
        lambda **kw: tracker_tool.init_run(
            project=kw.get("project", "epsionic"),
            name=kw.get("name", None),
            config=kw.get("config", {}),
        ))
    gw.register_tool("optimize_hp", "Run hyperparameter optimization",
        lambda **kw: opt_tool.create_study(
            domain=kw.get("domain", "general"),
            n_trials=kw.get("n_trials", 10),
        ))
    gw.register_tool("quantize", "Quantize model for deployment",
        lambda **kw: quant_tool.quantize_gptq(
            model_path=kw.get("model_path", ""),
            bits=kw.get("bits", 4),
        ))
    gw.register_tool("track_cost", "Track training cost",
        lambda **kw: cost_tool.estimate_training_cost(
            gpu_type=kw.get("gpu_type", None),
            hours=kw.get("hours", 1.0),
            gpus=kw.get("gpus", 1),
        ))
    gw.register_tool("notify", "Send notification",
        lambda **kw: notif_tool.send(
            title=kw.get("title", "Epslionic"),
            message=kw.get("message", ""),
            channels=kw.get("channels", ["log"]),
            level=kw.get("level", "info"),
        ))

    gw.load_skills()
    gw.start_heartbeat()
    return gw


def get_dashboard(gateway=None, memory=None, share=True, port=7860):
    try:
        from .tools.dashboard import serve
        if gateway and not memory:
            memory = gateway.memory
        serve(memory or gateway.memory, gateway, share=share, port=port)
    except ImportError:
        print("gradio not installed. Install with: pip install gradio")
