# Getting Started

## Installation

```bash
pip install openclaw-colab-agent
```

For development:
```bash
git clone https://github.com/openclaw/openclaw-colab-agent
cd openclaw-colab-agent
pip install -e ".[dev,dashboard]"
```

## Quick Usage

### Interactive Mode
```bash
openclaw --interactive
```
The agent will show a domain menu. Pick one, and it runs the full pipeline.

### Autonomous Mode
```bash
openclaw --autonomous
```
The agent detects your environment (GPU, VRAM) and auto-selects the best domain.

### Web Dashboard
```bash
openclaw --dashboard
```
Launches a Gradio UI at `http://localhost:7860`.

### REST API
```bash
openclaw --serve
curl http://localhost:8000/health
```

### Docker
```bash
docker compose up agent
```
