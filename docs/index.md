# OpenClaw-Colab Agent

Production-grade autonomous LLM training agent with domain auto-selection, dataset discovery, fine-tuning, auto-fix, web dashboard, and REST API.

## Quick Start

```bash
pip install openclaw-colab-agent
openclaw --interactive
```

## Key Features

- **9 Preset Domains**: math, code, medical, legal, creative, science, finance, chat, general
- **Autonomous Mode**: VRAM-based domain selection, zero user input
- **HuggingFace Dataset Discovery**: Auto-search and configure
- **Unsloth-first Training**: With PEFT/Transformers fallback
- **Auto-Fix**: 9 rule-based patterns + LLM-powered recovery
- **Web Dashboard**: Gradio multi-tab UI
- **REST API**: FastAPI server
- **Plugin System**: Lifecycle hooks (AutoGPT-style)
- **Chain Pipeline**: Composition with pipe operator (LangChain-style)
