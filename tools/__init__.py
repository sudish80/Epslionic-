"""Tool implementations — agent-accessible operations."""
# Each tool is a class instantiated by the agent with (memory_store, dirs).
# Tools expose get_tool_description() for the agent's tool registry.

from .dataset_discovery import DatasetDiscoveryTool
from .trainer import TrainerTool
from .auto_fixer import AutoFixerTool
from .dashboard import serve, make_ui

from .preference_trainer import PreferenceTrainerTool
from .model_merger import ModelMerger
from .data_factory import SyntheticDataGenerator
from .evaluator import ModelEvaluator
from .model_server import ModelServer
from .playground import ModelPlayground

from .experiment_tracker import ExperimentTracker
from .optimizer import HyperparameterOptimizer
from .quantizer import ModelQuantizer
from .cost_tracker import CostTracker
from .notifier import Notifier
from .flow import FlowEngine

__all__ = [
    "DatasetDiscoveryTool", "TrainerTool", "AutoFixerTool", "serve", "make_ui",
    "PreferenceTrainerTool", "ModelMerger", "SyntheticDataGenerator", "ModelEvaluator", "ModelServer", "ModelPlayground",
    "ExperimentTracker", "HyperparameterOptimizer", "ModelQuantizer", "CostTracker", "Notifier",
    "FlowEngine",
]
