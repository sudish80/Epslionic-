#!/usr/bin/env python3
"""
Launch script for the OpenClaw-Colab Agent.
This can run in Google Colab or locally.

Usage in Colab:
    !python launch.py --objective "Fine-tune Llama 3.2 on math reasoning"
    !python launch.py --interactive
"""

import sys
import os
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("launch")


def setup_colab_environment():
    """Install required packages in Colab."""
    packages = [
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "peft>=0.6.0",
        "trl>=0.7.0",
        "bitsandbytes>=0.41.0",
        "scipy",
        "sentencepiece",
        "huggingface_hub",
    ]

    import subprocess
    for pkg in packages:
        try:
            __import__(pkg.split(">=")[0].split("==")[0].replace("-", "_"))
            logger.info(f"  ✓ {pkg}")
        except ImportError:
            logger.info(f"  Installing {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                timeout=120
            )

    # Try installing Unsloth separately (optional, best for Colab)
    try:
        import unsloth
        logger.info("  ✓ unsloth (optimized training)")
    except ImportError:
        logger.info("  Optional: install unsloth for faster training")
        logger.info("  !pip install unsloth")


def launch_interactive(gateway):
    """Launch an interactive session."""
    print("\n" + "=" * 60)
    print("  OpenClaw-Colab ML Training Agent")
    print("  Type 'quit' to exit, 'status' for report")
    print("=" * 60)

    while True:
        try:
            objective = input("\n🎯 Enter training objective: ").strip()
            if not objective:
                continue
            if objective.lower() in ("quit", "exit", "q"):
                break
            if objective.lower() in ("status", "report"):
                print(gateway.get_status_markdown())
                continue
            if objective.lower() == "heartbeat":
                print(gateway.heartbeat.get_heartbeat_markdown())
                continue

            print(f"\n🔬 Processing: {objective}")
            result = gateway.run_objective(objective)
            print(f"\n✅ Complete! {len(result['steps'])} steps executed.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")

    gateway.shutdown()
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="OpenClaw-Colab ML Training Agent")
    parser.add_argument("--objective", "-o", type=str, help="Training objective")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--setup", action="store_true", help="Install dependencies")
    parser.add_argument("--model", type=str, default="unsloth/mistral-7b-bnb-4bit",
                       help="Base model for training")
    parser.add_argument("--dataset", type=str, help="Dataset ID to use")
    parser.add_argument("--llm-key", type=str, help="OpenAI/Anthropic API key")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")
    args = parser.parse_args()

    if args.setup:
        print("Setting up Colab environment...")
        setup_colab_environment()
        print("Setup complete!")
        return

    # Initialize the gateway
    from openclaw_colab_agent.config import AgentConfig
    from openclaw_colab_agent.core import Gateway

    config = AgentConfig()
    if args.llm_key:
        config.openai_api_key = args.llm_key
    if args.hf_token:
        config.huggingface_token = args.hf_token

    gateway = Gateway(config)

    # Register tools
    from openclaw_colab_agent.tools import DatasetDiscoveryTool, TrainerTool, AutoFixerTool

    discovery = DatasetDiscoveryTool(
        memory_store=gateway.memory,
        datasets_dir=config.datasets_dir,
        huggingface_token=config.huggingface_token,
    )
    trainer = TrainerTool(
        memory_store=gateway.memory,
        models_dir=config.models_dir,
    )
    fixer = AutoFixerTool(memory_store=gateway.memory)

    # Register with the gateway
    gateway.register_tool(
        "discover", "Search and load Hugging Face datasets",
        lambda **kw: discovery.search(**kw) if kw.get("query") else discovery.load(kw.get("dataset_id", "")),
        {"query": "Search query", "dataset_id": "HF dataset ID"}
    )
    gateway.register_tool(
        "train", "Fine-tune an LLM on a dataset",
        lambda **kw: trainer.train(
            kw.get("experiment_id", "exp_001"),
            kw.get("model_name", args.model),
            kw.get("dataset_dict", {}),
            kw.get("training_args", config.default_training_config),
            kw.get("objective", "")
        ),
        {"experiment_id": "str", "model_name": "str", "dataset_dict": "dict", "training_args": "dict"}
    )
    gateway.register_tool(
        "auto_fix", "Analyze and fix training errors",
        lambda **kw: fixer.analyze_and_fix(kw.get("error", ""), kw.get("context", {})),
        {"error": "Error message", "context": "Context dict"}
    )
    gateway.register_tool(
        "prepare", "Prepare training environment",
        lambda **kw: trainer.prepare(
            kw.get("experiment_id", "exp_001"),
            kw.get("model_name", args.model),
            kw.get("dataset_dict", {}),
            kw.get("training_args", config.default_training_config),
        ),
        {"experiment_id": "str", "model_name": "str", "dataset_dict": "dict", "training_args": "dict"}
    )

    # Load skills
    gateway.load_skills(config.skills_dir)

    # Start heartbeat
    gateway.start_heartbeat()
    logger.info("Gateway is running!")

    if args.interactive:
        launch_interactive(gateway)
    elif args.objective:
        print(f"\n🎯 Running: {args.objective}")
        result = gateway.run_objective(args.objective)
        print(f"\n✅ Result: {len(result['steps'])} steps")
        if result.get("outputs"):
            for tool, output in result["outputs"].items():
                print(f"\n  📊 {tool}: {str(output)[:200]}")
        if result.get("error"):
            print(f"\n  ❌ Error: {result['error']}")
        gateway.shutdown()
    else:
        print(gateway.get_status_markdown())
        print("\nTip: Use --objective or --interactive to run training")
        gateway.shutdown()


if __name__ == "__main__":
    main()
