# Epslionic-Colab Agent

[![CI](https://github.com/epsionic/epsionic/actions/workflows/ci.yml/badge.svg)](https://github.com/epsionic/epsionic/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/epsionic.svg)](https://pypi.org/project/epsionic/)
[![Python versions](https://img.shields.io/pypi/pyversions/epsionic.svg)](https://pypi.org/project/epsionic/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Production-grade autonomous LLM training agent — domain auto-selection, dataset discovery, fine-tuning, auto-fix, web dashboard, REST API, plugin system, and more.

## Quick Start

```bash
pip install epsionic

# Interactive mode — agent asks what domain to train
epsionic --interactive

# Autonomous mode — detects environment, picks domain, runs
epsionic --autonomous

# Launch web dashboard
epsionic --dashboard
```

## Architecture (Epslionic)

```
Gateway (Orchestrator)
├── DomainSelector → 9 preset domains + custom
├── Pipeline (Discover → Select → Prepare → Train)
├── SessionManager (Lane queues)
├── MemoryStore (File-based JSON)
├── HeartbeatMonitor (Proactive checks)
├── LLMBrain (OpenAI/Anthropic reasoning)
├── PluginSystem (Lifecycle hooks)
├── DeviceManager (CUDA/MPS/CPU)
└── StateMachine (Valid transitions)
```

## Features

- **Domain Auto-Selection**: 9 preset domains (math, code, medical, legal, creative, science, finance, chat, general) + custom domain definition
- **Autonomous Mode**: VRAM-based domain picking, zero user input
- **Dataset Discovery**: HuggingFace search, mock search fallback
- **Training**: Unsloth-first fine-tuning with PEFT/Transformers fallback
- **Auto-Fix**: 9 rule-based + LLM-powered error recovery patterns
- **Web Dashboard**: Gradio multi-tab UI (Status, Experiments, Datasets, Errors, Heartbeat, Control)
- **REST API**: FastAPI server with /health, /train/{domain}, /experiments, /status
- **Plugin System**: AutoGPT-style lifecycle hooks (on_register, before_tool, after_tool, on_error)
- **Chain Pipeline**: LangChain-style composition with pipe operator
- **State Machine**: Home-Assistant-style valid transitions
- **Device Manager**: PyTorch-style CUDA/MPS/CPU auto-detection
- **Graceful Fallbacks**: Windows cp1252 safe_print, GPU-less essential_only install

## CLI Reference

```
epsionic [OPTIONS]

Modes:
  -i, --interactive     Interactive REPL
  -a, --autonomous      Fully autonomous (no input)
  -c, --continuous      Cycle domains continuously
  -db, --dashboard      Launch Gradio web UI
  --serve               Start FastAPI REST server

Domain:
  -d, --domain KEY      Domain (math, code, medical...)
  -o, --objective TEXT  Training objective
  --skip-domain         Skip domain interview

Config:
  -c, --config FILE     Path to config YAML/JSON
  --api-key KEY         OpenAI/Anthropic API key
  --hf-token TOKEN      HuggingFace token
  --verbose             Debug logging
  --setup-only          Install deps and exit
```

## Docker

```bash
docker compose up agent           # Autonomous mode
docker compose --profile dashboard up  # Dashboard only
```

## Development

```bash
git clone https://github.com/epsionic/epsionic
cd epsionic
pip install -e ".[dev,dashboard]"
pre-commit install

make test            # Run tests
make lint            # Ruff check
make typecheck       # MyPy
```

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/epsionic/epsionic/blob/main/Epslionic_Colab_Agent.ipynb)

Or paste in a single cell:
```python
!pip install epsionic
!epsionic --interactive
```

## API Server

```bash
epsionic --serve
curl http://localhost:8000/health
curl http://localhost:8000/train/math -X POST
curl http://localhost:8000/experiments
```

## Project Status

Beta. 47+ tests, CI/CD, Docker, PyPI package ready.

## License

MIT
