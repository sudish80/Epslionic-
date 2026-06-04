# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-15

### Added
- Epslionic architecture: Gateway, Sessions (lane queues), Memory (file-based), Heartbeat (monitoring), Brain (LLM reasoning), Tools (skills)
- Domain auto-selection with 9 preset domains (math, code, medical, legal, creative, science, finance, chat, general)
- Autonomous mode with VRAM-based domain picking
- Interactive domain interview mode
- HuggingFace dataset discovery and auto-configuration
- Unsloth-first training with PEFT/Transformers fallback
- 9-pattern rule-based error auto-fixer with LLM fallback
- Gradio multi-tab web dashboard (Status, Experiments, Datasets, Errors, Heartbeat, Control)
- FastAPI REST API server (/health, /train/{domain}, /experiments, /status)
- Plugin system (AutoGPT-style) with lifecycle hooks
- Chain pipeline (LangChain-style) with pipe operator
- State machine (Home-Assistant-style) with valid transition validation
- Device manager (PyTorch-style) with CUDA/MPS/CPU auto-detect
- Exception hierarchy with 11 typed error classes
- _safe_print() fallback for Windows cp1252 terminals
- Docker multi-stage build (90MB) + docker-compose
- CI/CD with GitHub Actions (ruff, mypy, pytest, Docker)
- 47 unit tests across 4 test files

### Fixed
- Windows cp1252 terminal emoji crash with _safe_print()
- Gradio lazy import to avoid import-time dependency
- GPU-less auto_install skipping heavy ML packages
