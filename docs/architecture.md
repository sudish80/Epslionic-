# Architecture

The OpenClaw architecture is composed of six core layers:

```
Gateway (Orchestrator)
├── DomainSelector → 9 preset domains + custom
├── Pipeline (Discover → Select → Prepare → Train)
├── SessionManager (Lane queues for concurrency)
├── MemoryStore (File-based JSON persistence)
├── HeartbeatMonitor (Proactive health checks)
├── LLMBrain (OpenAI/Anthropic reasoning)
├── PluginSystem (Lifecycle hooks)
├── DeviceManager (CUDA/MPS/CPU)
└── StateMachine (Valid transition enforcement)
```

## Data Flow

1. **Domain Selection**: User picks or auto-selects a domain
2. **Dataset Discovery**: Searches HuggingFace for relevant datasets
3. **Pipeline**: 4-step chain (discover → select → prepare → train)
4. **Training**: Unsloth-first, PEFT fallback, auto-fix on errors
5. **Monitoring**: Heartbeat checks training progress, device, plugins
6. **Results**: Stored in MemoryStore, viewable via Dashboard or API
