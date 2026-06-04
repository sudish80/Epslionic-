# Contributing to OpenClaw-Colab Agent

We love contributions! Here's how to get started.

## Development Setup

```bash
git clone <your-fork>
cd openclaw-colab-agent
pip install -e ".[dev,dashboard]"
pre-commit install
```

## Code Style

- **Formatting**: Ruff (`ruff format .`)
- **Linting**: Ruff (`ruff check .`)
- **Type checking**: mypy (`mypy openclaw_colab_agent`)
- **Imports**: isort with black profile

Run all checks:
```bash
make style lint typecheck
```

## Testing

```bash
make test        # Quick tests
make test-cov    # With coverage
```

Write tests in `tests/` following existing patterns. Mark GPU/network/slow tests:
```python
@pytest.mark.gpu
@pytest.mark.network
@pytest.mark.slow
```

## Pull Request Process

1. Fork the repo and create a feature branch
2. Make your changes with tests
3. Run `make lint && make test` — all must pass
4. Update CHANGELOG.md if applicable
5. Open a PR with a clear description

## Project Structure

```
openclaw_colab_agent/
├── core/          # Gateway, Domain, Memory, Session, Heartbeat, Brain, State, Chain, Plugin
├── tools/         # DatasetDiscovery, Trainer, AutoFixer, Dashboard
├── utils/         # DeviceManager
├── tests/         # 47+ tests
├── cli.py         # Production CLI entry point
├── config.py      # AgentConfig
└── exceptions.py  # Error hierarchy
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
