import os
import sys
import json
import time
import logging
import argparse
import subprocess
import threading
from pathlib import Path
from typing import Optional, Tuple

_STRIP_EMOJI = False

from . import __version__
from .config import AgentConfig
from .core.gateway import Gateway
from .core.domain import DomainSelector, DOMAINS
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

logger = logging.getLogger('openclaw.cli')

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.logging import RichHandler
    _HAS_RICH = True
    _console = Console()
except ImportError:
    _HAS_RICH = False
    _console = None


def _echo(msg="", style=None):
    if _console:
        _console.print(msg)
    else:
        print(msg)


def _status_panel(title, content):
    if _console:
        _console.print(Panel(content, title=title, border_style="blue"))
    else:
        print(f"=== {title} ===")
        print(content)


def _make_table(title, columns, rows):
    if _console:
        table = Table(title=title)
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*[str(c) for c in row])
        _console.print(table)
    else:
        header = " | ".join(columns)
        sep = "-+-".join("-" * len(c) for c in columns)
        print(f"\n{title}")
        print(header)
        print(sep)
        for row in rows:
            print(" | ".join(str(c) for c in row))


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    level = logging.DEBUG if verbose else logging.INFO
    handlers = []

    if _HAS_RICH and not log_file:
        handlers.append(RichHandler(rich_tracebacks=True, markup=True))
    else:
        handlers.append(logging.StreamHandler(sys.stdout))

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        handlers=handlers,
    )


def build_gateway(cfg: AgentConfig) -> Gateway:
    gw = Gateway(cfg)

    from pathlib import Path
    ds_dir = Path(cfg.datasets_dir) if not isinstance(cfg.datasets_dir, Path) else cfg.datasets_dir
    ds_tool = DatasetDiscoveryTool(gw.memory, ds_dir, cfg.huggingface_token)
    tr_tool = TrainerTool(gw.memory, cfg.models_dir)
    fx_tool = AutoFixerTool(gw.memory)
    pref_tool = PreferenceTrainerTool(gw.memory, cfg.models_dir)
    merger_tool = ModelMerger(cfg.models_dir)
    factory_tool = SyntheticDataGenerator()
    evaluator_tool = ModelEvaluator()
    server_tool = ModelServer()
    playground_tool = ModelPlayground()
    tracker_tool = ExperimentTracker(backend=os.environ.get("TRACKER_BACKEND", "none"))
    opt_tool = HyperparameterOptimizer(memory_store=gw.memory)
    quant_tool = ModelQuantizer(models_dir=cfg.models_dir)
    cost_tool = CostTracker(memory_store=gw.memory, budget_limit=cfg.get("budget_limit", 50.0))
    notif_tool = Notifier()

    gw.register_tool('discover', 'Search HuggingFace datasets',
        lambda **kw: (
            ds_tool.search(keywords=kw.get('query', ''), max_results=15)
            if kw.get('query') else
            ds_tool.load(dataset_id=kw.get('dataset_id', ''))
        ))
    gw.register_tool('train', 'Fine-tune LLM',
        lambda **kw: tr_tool.train(
            experiment_id=kw.get('experiment_id', 'exp'),
            model_name=kw.get('model_name', cfg.default_training_config.get('model_name', 'unsloth/mistral-7b-bnb-4bit')),
            dataset_dict=kw.get('dataset_dict', {}),
            training_args=kw.get('training_args', cfg.default_training_config),
            objective=kw.get('objective', ''),
        ))
    gw.register_tool('prepare', 'Prepare training environment',
        lambda **kw: tr_tool.prepare(
            experiment_id=kw.get('experiment_id', 'exp'),
            model_name=kw.get('model_name', cfg.default_training_config.get('model_name', 'unsloth/mistral-7b-bnb-4bit')),
        ))
    gw.register_tool('auto_fix', 'Analyze and fix errors',
        lambda **kw: fx_tool.fix(
            error=kw.get('error', ''),
            context=kw.get('context', {}),
            brain_fix=kw.get('brain_fix'),
        ))
    gw.register_tool('preference_train', 'DPO/ORPO alignment training',
        lambda **kw: pref_tool.train_dpo(
            experiment_id=kw.get('experiment_id', 'pref_exp'),
            model_name=kw.get('model_name', cfg.default_training_config.get('model_name', 'unsloth/mistral-7b-bnb-4bit')),
            dataset_dict=kw.get('dataset_dict', {}),
            training_args=kw.get('training_args', cfg.default_training_config),
        ))
    gw.register_tool('merge_models', 'Merge LoRA adapters (SLERP/TIES/Linear)',
        lambda **kw: merger_tool.merge_loras(
            model_name=kw.get('model_name', cfg.default_training_config.get('model_name', 'unsloth/mistral-7b-bnb-4bit')),
            adapter_paths=kw.get('adapter_paths', []),
            merge_method=kw.get('method', 'linear'),
        ))
    gw.register_tool('generate_data', 'Generate synthetic training data',
        lambda **kw: factory_tool.self_instruct(
            seed_tasks=kw.get('seed_tasks', ['Generate a training example']),
            num_samples=kw.get('num_samples', 100),
        ))
    gw.register_tool('evaluate', 'Run evaluation benchmarks',
        lambda **kw: evaluator_tool.evaluate(
            model_path=kw.get('model_path', ''),
            benchmarks=kw.get('benchmarks', ['mmlu']),
        ))
    gw.register_tool('serve_model', 'Start model serving endpoint',
        lambda **kw: server_tool.deploy_vllm(
            model_path=kw.get('model_path', ''),
            port=kw.get('port', 8001),
        ))
    gw.register_tool('chat', 'Interactive model playground',
        lambda **kw: playground_tool.generate(
            message=kw.get('message', ''),
            system_prompt=kw.get('system_prompt', ''),
            max_tokens=kw.get('max_tokens', 256),
        ))
    gw.register_tool('track_run', 'Initialize experiment tracking',
        lambda **kw: tracker_tool.init_run(
            project=kw.get('project', 'openclaw'),
            name=kw.get('name', None),
            config=kw.get('config', {}),
        ))
    gw.register_tool('optimize_hp', 'Run hyperparameter optimization',
        lambda **kw: opt_tool.create_study(
            domain=kw.get('domain', 'general'),
            n_trials=kw.get('n_trials', 10),
        ))
    gw.register_tool('quantize', 'Quantize model for deployment',
        lambda **kw: quant_tool.quantize_gptq(
            model_path=kw.get('model_path', ''),
            bits=kw.get('bits', 4),
        ))
    gw.register_tool('track_cost', 'Track training cost',
        lambda **kw: cost_tool.estimate_training_cost(
            gpu_type=kw.get('gpu_type', None),
            hours=kw.get('hours', 1.0),
            gpus=kw.get('gpus', 1),
        ))
    gw.register_tool('notify', 'Send notification',
        lambda **kw: notif_tool.send(
            title=kw.get('title', 'OpenClaw'),
            message=kw.get('message', ''),
            channels=kw.get('channels', ['log']),
            level=kw.get('level', 'info'),
        ))

    gw.load_skills()
    gw.start_heartbeat()
    _echo(f"[Device] {gw.device.name} ({gw.device.backend}, {gw.device.vram_gb:.1f} GB)")
    _echo(f"[Plugins] {len(gw.plugins)} loaded: {', '.join(gw.plugins.keys()) or 'none'}")
    return gw


def cmd_autonomous(gw: Gateway, continuous: bool = False) -> int:
    logger.info("Starting autonomous mode (continuous=%s)", continuous)

    if _HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Processing domains...", total=None)
            completed = gw.run_autonomous_loop(continuous=continuous)
    else:
        completed = gw.run_autonomous_loop(continuous=continuous)

    _echo(f"[Done] Autonomous complete: {len(completed)} domains: {', '.join(completed)}")
    return 0


def cmd_domain_pipeline(gw: Gateway, domain_key: str, domain_config: dict) -> int:
    logger.info("Starting domain pipeline: %s (%s)", domain_config.get('name', domain_key), domain_key)
    result = gw.run_domain(domain_key, domain_config)
    status = 'completed' if not result.get('error') else 'failed'

    steps = result.get('steps', [])
    _make_table(
        f"Pipeline Results: {domain_key}",
        ["Step", "Tool", "Status"],
        [(s.get('step', '?'), s.get('tool', '?'), s.get('status', '?')) for s in steps]
    )

    logger.info("Pipeline %s: %d steps", status, len(result['steps']))
    return 0 if status == 'completed' else 1


def cmd_interactive(gw: Gateway) -> int:
    logger.info("Starting interactive mode")
    _echo("Commands: <objective>, status, device, plugins, domain, quit")
    try:
        while True:
            inp = input('\n>> ').strip()
            if not inp:
                continue
            if inp == 'quit':
                break
            if inp == 'status':
                _echo(gw.get_status_markdown())
                continue
            if inp == 'device':
                _echo(gw.device.summary())
                continue
            if inp == 'plugins':
                _echo(gw.plugin_summary())
                continue
            if inp == 'domain':
                sel = DomainSelector()
                dk, dc = sel.ask()
                gw.state.domain = dk
                gw.state.domain_config = dc
                r = gw.run_domain(dk, dc)
                _echo(f"Done: {len(r['steps'])} steps")
                continue
            r = gw.run_objective(inp)
            _echo(f"Done: {len(r['steps'])} steps")
    except (KeyboardInterrupt, EOFError):
        pass
    return 0


def cmd_dashboard(gw: Gateway, share: bool = True, port: int = 7860) -> int:
    try:
        from .tools.dashboard import serve as dashboard_serve
        logger.info("Launching dashboard on port %d (share=%s)", port, share)
        dashboard_serve(gw.memory, gw, share=share, port=port)
    except ImportError:
        logger.error("gradio not installed. Install with: pip install openclaw-colab-agent[dashboard]")
        return 1
    except Exception as e:
        logger.error("Dashboard launch failed: %s", e)
        return 1
    return 0


def cmd_serve(gw: Gateway, host: str = '0.0.0.0', port: int = 8000) -> int:
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        import uvicorn
    except ImportError:
        logger.error("fastapi/uvicorn not installed. Install with: pip install fastapi uvicorn")
        return 1

    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="OpenClaw-Colab Agent API", version=__version__,
                  docs_url="/docs", redoc_url="/redoc")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "domain": gw.state.domain or "not set",
            "tools": len(gw._tool_registry),
            "uptime": gw.state.started_at,
        }

    @app.post("/train/{domain}")
    def train_domain(domain: str):
        if domain not in DOMAINS:
            raise HTTPException(status_code=400, detail=f"Unknown domain: {domain}")
        cfg = DOMAINS[domain]
        gw.state.domain = domain
        gw.state.domain_config = cfg
        result = gw.run_domain(domain, cfg)
        return {
            "status": "completed" if not result.get('error') else "failed",
            "steps": len(result['steps']),
            "experiment": result.get('experiment_id'),
        }

    @app.get("/experiments")
    def list_experiments():
        exps = gw.memory.list_experiments()
        return {
            "count": len(exps),
            "experiments": [{'id': e.get('id'), 'name': e.get('name'),
                            'status': e.get('status')} for e in exps[:20]]
        }

    @app.get("/status")
    def status():
        return {
            "status": gw.state.status,
            "domain": gw.state.domain,
            "tools": gw.state.tools_loaded,
            "sessions": gw.state.sessions_created,
            "errors": gw.state.errors_encountered,
            "fixed": gw.state.errors_fixed,
            "started": gw.state.started_at,
        }

    @app.get("/experiments/{exp_id}")
    def get_experiment(exp_id: str):
        exp = gw.memory.get_experiment(exp_id)
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return exp

    @app.get("/experiments/compare")
    def compare_experiments(a: str, b: str):
        exp_a = gw.memory.get_experiment(a)
        exp_b = gw.memory.get_experiment(b)
        if not exp_a or not exp_b:
            raise HTTPException(status_code=404, detail="One or both experiments not found")
        diff = {}
        for k in set(list(exp_a.keys()) + list(exp_b.keys())):
            va, vb = exp_a.get(k), exp_b.get(k)
            if va != vb:
                diff[k] = {"a": str(va)[:200], "b": str(vb)[:200]}
        return {"experiment_a": a, "experiment_b": b, "differences": diff}

    @app.get("/plugins")
    def list_plugins():
        return {
            "count": len(gw.plugins),
            "plugins": [{"name": n, "enabled": p.enabled, "hooks": list(p._hooks.keys())} for n, p in gw.plugins.items()]
        }

    @app.get("/device")
    def device_info():
        return {
            "name": gw.device.name,
            "backend": gw.device.backend,
            "vram_gb": gw.device.vram_gb,
            "batch_recommendation": gw.device.recommended_batch_size(),
            "quantization": gw.device.recommended_quantization(),
        }

    @app.post("/experiments/export")
    def export_experiments():
        path = gw.memory.export_experiments_csv()
        return {"path": path, "count": len(gw.memory.list_experiments())}

    @app.post("/jobs")
    def create_job(job_type: str, params: dict = None, priority: int = 0):
        job_id = gw.memory.create_job(job_type, params or {}, priority)
        return {"job_id": job_id, "status": "queued"}

    @app.get("/jobs")
    def list_jobs(status: str = None, job_type: str = None):
        jobs = gw.memory.list_jobs(status=status, job_type=job_type)
        return {"count": len(jobs), "jobs": jobs[:50]}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job = gw.memory.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        ok = gw.memory.cancel_job(job_id)
        if not ok:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled or not found")
        return {"job_id": job_id, "status": "cancelled"}

    @app.get("/experiments/trends")
    def experiment_trends():
        exps = gw.memory.list_experiments()
        if not exps:
            return {"experiments": 0, "trends": {}}
        successful = sum(1 for e in exps if e.get("status") == "completed")
        failed = sum(1 for e in exps if e.get("status") == "failed")
        by_domain = {}
        for e in exps:
            cfg = e.get("config", {}) or {}
            d = cfg.get("domain", "unknown") if isinstance(cfg, dict) else "unknown"
            by_domain[d] = by_domain.get(d, 0) + 1
        return {"experiments": len(exps), "successful": successful, "failed": failed,
                "by_domain": by_domain,
                "avg_loss": round(sum(e.get("metrics", {}).get("eval_loss", 0) or 0 for e in exps) / max(len(exps), 1), 4)}

    @app.get("/logs")
    def get_logs(lines: int = 50):
        import subprocess
        try:
            result = subprocess.run(["tail", "-n", str(lines), str(Path(cfg.logs_dir) / "openclaw.log")],
                                    capture_output=True, text=True, timeout=5)
            return {"logs": result.stdout.split("\n")}
        except Exception:
            return {"logs": ["Log file not available"]}

    @app.post("/plugins/{name}/toggle")
    def toggle_plugin(name: str):
        if name not in gw.plugins:
            raise HTTPException(status_code=404, detail=f"Plugin '{name}' not found")
        p = gw.plugins[name]
        if p.enabled:
            gw._plugin_manager.disable(name)
        else:
            gw._plugin_manager.enable(name)
        return {"name": name, "enabled": not p.enabled}

    _echo(f"Starting API server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
    return 0


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog='openclaw',
        description=f'OpenClaw-Colab Agent v{__version__} — Production LLM Training Agent',
    )
    p.add_argument('--version', '-v', action='version', version=f'%(prog)s {__version__}')

    p.add_argument('--config', '-c', default=None, help='Path to config YAML/JSON file')
    p.add_argument('--verbose', action='store_true', help='Enable debug logging')
    p.add_argument('--log-file', default=None, help='Path to log file')

    domain_help = f"Domain to train on. Options: {', '.join(DOMAINS.keys())}"
    p.add_argument('--domain', '-d', default=None, help=domain_help)
    p.add_argument('--objective', '-o', default=None, help='Training objective (combined with --domain)')
    p.add_argument('--skip-domain', action='store_true', help='Skip domain interview')

    p.add_argument('--autonomous', '-a', action='store_true', help='Fully autonomous mode — no user input')
    p.add_argument('--continuous', action='store_true', help='With --autonomous: cycle through domains')
    p.add_argument('--interactive', '-i', action='store_true', help='Enter interactive REPL after setup')
    p.add_argument('--dashboard', '-db', action='store_true', help='Launch Gradio web dashboard')
    p.add_argument('--serve', action='store_true', help='Start FastAPI REST API server')
    p.add_argument('--setup-only', action='store_true', help='Just install dependencies and exit')

    p.add_argument('--host', default='0.0.0.0', help='API server host (default: 0.0.0.0)')
    p.add_argument('--port', type=int, default=8000, help='API server port (default: 8000)')
    p.add_argument('--dashboard-port', type=int, default=7860, help='Dashboard port (default: 7860)')

    p.add_argument('--api-key', default=None, help='OpenAI/Anthropic API key')
    p.add_argument('--hf-token', default=None, help='HuggingFace token')

    p.add_argument('--no-emoji', action='store_true', help='Strip emoji for CI/terminal compat')
    p.add_argument('--install-completion', action='store_true', help='Install tab-completion for your shell')

    return p.parse_args(argv)


def _auto_install(essential_only: bool = False):
    packages = [
        "huggingface_hub>=0.20.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
    ]
    if not essential_only:
        packages.extend([
            "transformers>=4.36.0",
            "datasets>=2.14.0",
            "accelerate>=0.24.0",
            "peft>=0.6.0",
            "trl>=0.7.0",
            "bitsandbytes>=0.41.0",
            "scipy",
            "sentencepiece",
        ])
    import subprocess
    for pkg in packages:
        name = pkg.split(">=")[0].split("==")[0].replace("-", "_")
        try:
            __import__(name)
        except ImportError:
            logger.info(f"Installing {pkg}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg], timeout=120)
            except Exception as e:
                logger.warning(f"Failed to install {pkg}: {e}")


def _setup_signal_handlers(gw):
    import signal
    def _handler(signum, frame):
        logger.info(f"Signal {signum} received, shutting down...")
        if gw:
            gw.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    setup_logging(verbose=args.verbose, log_file=args.log_file)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    cfg = AgentConfig()
    if args.config:
        cfg = AgentConfig.from_file(args.config)
    else:
        for p in ['openclaw.json', 'openclaw.yaml', 'openclaw.yml', 'config/openclaw.json', '~/.config/openclaw/config.json']:
            resolved = Path(p).expanduser()
            if resolved.exists():
                cfg = AgentConfig.from_file(str(resolved))
                logger.info("Loaded config from %s", resolved)
                break

    cfg.resolve_env()
    if args.api_key:
        cfg.openai_api_key = args.api_key
    if args.hf_token:
        cfg.huggingface_token = args.hf_token

    errors = cfg.validate()
    if errors:
        for e in errors:
            logger.error("Config validation: %s", e)
        return 1

    if args.install_completion:
        import subprocess
        shell = os.environ.get("SHELL", "")
        if "zsh" in shell:
            rc = subprocess.call(["openclaw", "--help"])
            print("\nAdd to ~/.zshrc: eval \"$(_OPENCLAW_COMPLETE=zsh_source openclaw)\"")
        elif "bash" in shell:
            print("\nAdd to ~/.bashrc: eval \"$(_OPENCLAW_COMPLETE=bash_source openclaw)\"")
        else:
            print("Tab-completion: add to your shell rc file or use `openclaw --help`")
        return 0

    if args.no_emoji:
        from .utils.format import set_emoji_mode
        set_emoji_mode(True)

    if args.setup_only:
        _auto_install()
        return 0

    has_gpu = False
    try:
        import torch; has_gpu = torch.cuda.is_available()
    except ImportError:
        pass
    _auto_install(essential_only=not has_gpu)

    gw = build_gateway(cfg)
    _setup_signal_handlers(gw)
    logger.info("Gateway ready — %d tools registered", gw.state.tools_loaded)

    if args.autonomous:
        return cmd_autonomous(gw, continuous=args.continuous)

    if args.dashboard:
        t = threading.Thread(target=cmd_dashboard, args=(gw,), kwargs={'share': True, 'port': args.dashboard_port}, daemon=True)
        t.start()
        logger.info("Dashboard launching in background thread")

    if args.serve:
        return cmd_serve(gw, host=args.host, port=args.port)

    domain_key = None
    domain_config = None

    if args.domain:
        if args.domain in DOMAINS:
            domain_key = args.domain
            domain_config = DOMAINS[domain_key]
            logger.info("Domain: %s (%s)", domain_config['name'], domain_key)
        else:
            logger.error("Unknown domain '%s'. Options: %s", args.domain, ', '.join(DOMAINS.keys()))
            return 1
        cfg.domain = domain_key
        cfg.domain_config = domain_config
    elif not args.skip_domain and not args.objective:
        sel = DomainSelector()
        domain_key, domain_config = sel.ask()
        cfg.domain = domain_key
        cfg.domain_config = domain_config

    if domain_key:
        rc = cmd_domain_pipeline(gw, domain_key, domain_config)
        if rc != 0:
            return rc

    if args.interactive or (not domain_key and not args.objective):
        return cmd_interactive(gw)

    if args.objective:
        logger.info("Objective: %s", args.objective)
        result = gw.run_objective(args.objective)
        logger.info("Done: %d steps", len(result['steps']))
        return 0 if not result.get('error') else 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
