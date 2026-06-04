# Getting Started

## Installation

```bash
pip install epsionic
```

For development:
```bash
git clone https://github.com/epsionic/epsionic
cd epsionic
pip install -e ".[dev,dashboard]"
```

## Quick Usage

### Interactive Mode
```bash
epsionic --interactive
```
The agent will show a domain menu. Pick one, and it runs the full pipeline.

### Autonomous Mode
```bash
epsionic --autonomous
```
The agent detects your environment (GPU, VRAM) and auto-selects the best domain.

### Web Dashboard
```bash
epsionic --dashboard
```
Launches a Gradio UI at `http://localhost:7860`.

### REST API
```bash
epsionic --serve
curl http://localhost:8000/health
```

### Docker
```bash
docker compose up agent
```
