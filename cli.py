import os
import sys
import json
import time
import logging
import argparse
import subprocess
import threading
from pathlib import Path
from typing import Optional, Tuple, List

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

logger = logging.getLogger('epsionic.cli')

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
            project=kw.get('project', 'epsionic'),
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
            title=kw.get('title', 'Epslionic'),
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
        logger.error("gradio not installed. Install with: pip install epsionic[dashboard]")
        return 1
    except Exception as e:
        logger.error("Dashboard launch failed: %s", e)
        return 1
    return 0


def _agent_chat(gw, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Simple agent chat — uses LLMBrain to generate a response."""
    try:
        result = gw.brain.think_and_act(
            prompt,
            context=f"Respond helpfully. Max tokens: {max_tokens}",
            max_steps=1,
        )
        return result.reasoning or result.action_params.get("response", "No response generated.")
    except Exception as e:
        return f"Error: {e}"


def make_app(gw: Gateway) -> "FastAPI":
    """Build the FastAPI application for a given Gateway instance."""
    from fastapi import FastAPI, HTTPException, Depends, Security, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from starlette.middleware.base import BaseHTTPMiddleware
    from pydantic import BaseModel

    _api_key = os.environ.get("EPSIONIC_API_KEY", "")
    _security = HTTPBearer(auto_error=False)

    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "connect-src 'self'; "
                "frame-ancestors 'none';"
            )
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
            return response

    def _verify_key(credentials: Optional[HTTPAuthorizationCredentials] = Security(_security)):
        if not _api_key:
            return True
        if not credentials or credentials.credentials != _api_key:
            raise HTTPException(status_code=403, detail="Invalid API key")
        return True

    app = FastAPI(title="Epslionic-Colab Agent API", version=__version__,
                  docs_url="/docs", redoc_url="/redoc")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SecurityHeadersMiddleware)

    # Input size limit middleware
    from starlette.datastructures import MutableHeaders
    import time as _time
    from collections import defaultdict
    _rate_limit_buckets: dict = defaultdict(list)
    _RATE_LIMIT_WINDOW = 60  # seconds
    _RATE_LIMIT_MAX = 100    # requests per window

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = _time.time()
        bucket = _rate_limit_buckets[client_ip]
        bucket[:] = [t for t in bucket if now - t < _RATE_LIMIT_WINDOW]
        if len(bucket) >= _RATE_LIMIT_MAX:
            return JSONResponse(
                {"error": "Rate limit exceeded", "retry_after": _RATE_LIMIT_WINDOW},
                status_code=429,
                headers={"Retry-After": str(_RATE_LIMIT_WINDOW)},
            )
        bucket.append(now)
        return await call_next(request)

    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and int(cl) > 10 * 1024 * 1024:  # 10MB max
            return JSONResponse({"error": "Request body too large"}, status_code=413)
        return await call_next(request)

    @app.get("/health")
    def health(auth: bool = Depends(_verify_key)):
        return {
            "status": "ok",
            "version": __version__,
            "domain": gw.state.domain or "not set",
            "tools": len(gw._tool_registry),
            "uptime": gw.state.started_at,
            "schedules": len(gw._schedule),
            "budget_halted": getattr(gw, "_budget_halt", False),
        }

    @app.get("/ready")
    def readiness(auth: bool = Depends(_verify_key)):
        issues = []
        if not getattr(gw, "_tool_registry", None):
            issues.append("no tools registered")
        if not getattr(gw, "_running", None) and gw.state.status != "running":
            pass
        if issues:
            return JSONResponse(status_code=503, content={"status": "not ready", "issues": issues})
        return {"status": "ready", "uptime": gw.state.started_at}

    # ── Version endpoint ───────────────────────────────────────────────
    @app.get("/version")
    def version_info(auth: bool = Depends(_verify_key)):
        return {
            "version": __version__,
            "api_version": "v1",
            "schema_version": getattr(gw.memory, "_get_schema_version", lambda: 0)(),
        }

    # ── Prometheus metrics ─────────────────────────────────────────────
    @app.get("/metrics")
    def metrics(auth: bool = Depends(_verify_key)):
        import psutil, platform
        lines = [
            "# HELP epsionic_build_info Build metadata",
            "# TYPE epsionic_build_info gauge",
            f'epsionic_build_info{{version="{__version__}",python="{platform.python_version()}"}} 1',
            "# HELP epsionic_tools_total Number of registered tools",
            "# TYPE epsionic_tools_total gauge",
            f"epsionic_tools_total {len(gw._tool_registry)}",
            "# HELP epsionic_experiments_total Total experiments created",
            "# TYPE epsionic_experiments_total gauge",
            f"epsionic_experiments_total {gw.state.sessions_created}",
            "# HELP epsionic_errors_total Total errors encountered",
            "# TYPE epsionic_errors_total gauge",
            f"epsionic_errors_total {gw.state.errors_encountered}",
            "# HELP epsionic_budget_halted Budget halt status",
            "# TYPE epsionic_budget_halted gauge",
            f"epsionic_budget_halted {1 if getattr(gw, '_budget_halt', False) else 0}",
            "# HELP epsionic_uptime_seconds Agent uptime",
            "# TYPE epsionic_uptime_seconds gauge",
            f"epsionic_uptime_seconds {(datetime.now() - datetime.fromisoformat(gw.state.started_at)).total_seconds() if gw.state.started_at else 0}",
        ]
        try:
            lines.append("# HELP python_gc_objects_collected_total Objects collected by GC")
            lines.append("# TYPE python_gc_objects_collected_total counter")
            lines.append(f"python_gc_objects_collected_total {len(__import__('gc').get_objects())}")
            lines.append("# HELP process_cpu_seconds_total Total CPU seconds")
            lines.append("# TYPE process_cpu_seconds_total counter")
            lines.append(f"process_cpu_seconds_total {psutil.Process().cpu_times().user + psutil.Process().cpu_times().system:.2f}")
            lines.append("# HELP process_memory_bytes Process memory in bytes")
            lines.append("# TYPE process_memory_bytes gauge")
            lines.append(f"process_memory_bytes {psutil.Process().memory_info().rss}")
        except Exception:
            pass
        return PlainTextResponse("\n".join(lines) + "\n")

    # ── OpenAI-compatible API ─────────────────────────────────────────
    from pydantic import BaseModel

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = "default"
        messages: List[ChatMessage]
        temperature: float = 0.7
        max_tokens: int = 512
        stream: bool = False

    class CompletionRequest(BaseModel):
        model: str = "default"
        prompt: str
        max_tokens: int = 512
        temperature: float = 0.7

    @app.get("/v1/models")
    def list_models(auth: bool = Depends(_verify_key)):
        models = [{"id": "default", "object": "model", "created": int(datetime.now().timestamp()), "owned_by": "epsionic"}]
        from ..tools.model_server import list_models as ls
        try:
            models.extend(ls())
        except Exception:
            pass
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    def chat_completion(req: ChatRequest, auth: bool = Depends(_verify_key)):
        prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
        output = _agent_chat(gw, prompt, req.max_tokens, req.temperature)
        return {
            "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": output}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(output.split()), "total_tokens": len(prompt.split()) + len(output.split())},
        }

    @app.post("/v1/completions")
    def text_completion(req: CompletionRequest, auth: bool = Depends(_verify_key)):
        output = _agent_chat(gw, req.prompt, req.max_tokens, req.temperature)
        return {
            "id": f"cmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "object": "text_completion",
            "created": int(datetime.now().timestamp()),
            "model": req.model,
            "choices": [{"index": 0, "text": output, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(req.prompt.split()), "completion_tokens": len(output.split()), "total_tokens": len(req.prompt.split()) + len(output.split())},
        }

    # ── WebSocket Dashboard ────────────────────────────────────────────
    from fastapi import WebSocket, WebSocketDisconnect

    _ws_clients: list = []

    @app.websocket("/ws/dashboard")
    async def ws_dashboard(websocket: WebSocket):
        await websocket.accept()
        _ws_clients.append(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                # Echo back or handle commands
                await websocket.send_text(f'{{"echo": "{data}"}}')
        except WebSocketDisconnect:
            if websocket in _ws_clients:
                _ws_clients.remove(websocket)

    # ── Webhook Integration ────────────────────────────────────────────
    from ..core.webhooks import WebhookManager
    gw._webhook_manager = WebhookManager()

    def _fire_webhooks(event: str, **kw):
        if hasattr(gw, "_webhook_manager"):
            gw._webhook_manager.fire(event, kw)
    gw.hook("after_tool", lambda gw_obj, **kw: _fire_webhooks("tool_complete", **kw))
    gw.hook("on_error", lambda gw_obj, **kw: _fire_webhooks("error", **kw))
    gw.hook("on_complete", lambda gw_obj, **kw: _fire_webhooks("objective_complete", **kw))

    # ── Webhook API endpoints ──────────────────────────────────────────

    @app.post("/webhooks")
    def register_webhook(url: str, events: List[str], auth: bool = Depends(_verify_key)):
        if hasattr(gw, "_webhook_manager"):
            hook = gw._webhook_manager.register(url, events)
            return {"status": "registered", "hook": hook}
        return {"error": "WebhookManager not available"}

    @app.get("/webhooks")
    def list_webhooks(auth: bool = Depends(_verify_key)):
        if hasattr(gw, "_webhook_manager"):
            return {"webhooks": gw._webhook_manager.list()}
        return {"webhooks": []}

    @app.delete("/webhooks/{hook_id}")
    def remove_webhook(hook_id: str, auth: bool = Depends(_verify_key)):
        if hasattr(gw, "_webhook_manager") and gw._webhook_manager.remove(hook_id):
            return {"status": "removed"}
        raise HTTPException(404, "Webhook not found")

    # ── Recipe / Marketplace endpoints ─────────────────────────────────

    @app.get("/recipes")
    def list_recipes(domain: str = None, auth: bool = Depends(_verify_key)):
        from ..tools.recipe_manager import RecipeManager
        rm = RecipeManager()
        return {"recipes": rm.list_recipes(domain)}

    @app.get("/recipes/search")
    def search_registry(query: str = "", auth: bool = Depends(_verify_key)):
        from ..tools.recipe_manager import RecipeManager
        rm = RecipeManager()
        return {"results": rm.search_registry(query)}

    @app.post("/recipes/download")
    def download_recipe(name: str, auth: bool = Depends(_verify_key)):
        from ..tools.recipe_manager import RecipeManager
        rm = RecipeManager()
        ok = rm.download_recipe(name)
        return {"status": "downloaded" if ok else "failed"}

    @app.get("/marketplace")
    def marketplace_list(category: str = "recipes", auth: bool = Depends(_verify_key)):
        from ..tools.marketplace import Marketplace
        m = Marketplace(gw.config.workspace_root)
        return {"items": m.list_available(category)}

    @app.post("/marketplace/download")
    def marketplace_download(category: str, name: str, auth: bool = Depends(_verify_key)):
        from ..tools.marketplace import Marketplace
        m = Marketplace(gw.config.workspace_root)
        ok = m.download(category, name)
        return {"status": "downloaded" if ok else "failed"}

    # ── Tenant / Multi-Tenant endpoints ────────────────────────────────

    @app.post("/tenants")
    def create_tenant(name: str, auth: bool = Depends(_verify_key)):
        from ..core.workspace import WorkspaceManager
        wm = WorkspaceManager(Path(str(gw.config.workspace_root)) / "tenants.db")
        tenant = wm.create_tenant(name, gw.config.workspace_root)
        return tenant

    @app.get("/tenants")
    def list_tenants(auth: bool = Depends(_verify_key)):
        from ..core.workspace import WorkspaceManager
        wm = WorkspaceManager(Path(str(gw.config.workspace_root)) / "tenants.db")
        return {"tenants": wm.list_tenants()}

    # ── Prompt Management endpoints ────────────────────────────────────

    @app.post("/prompts")
    def create_prompt(name: str, template: str, auth: bool = Depends(_verify_key)):
        from ..tools.prompt_manager import PromptManager
        pm = PromptManager(Path(str(gw.config.workspace_root)) / "prompts.db")
        result = pm.create_template(name, template)
        return result

    @app.get("/prompts")
    def list_prompts(tag: str = None, auth: bool = Depends(_verify_key)):
        from ..tools.prompt_manager import PromptManager
        pm = PromptManager(Path(str(gw.config.workspace_root)) / "prompts.db")
        return {"prompts": pm.list_templates(tag)}

    @app.post("/prompts/{prompt_id}/render")
    def render_prompt(prompt_id: str, variables: dict = None, auth: bool = Depends(_verify_key)):
        from ..tools.prompt_manager import PromptManager
        pm = PromptManager(Path(str(gw.config.workspace_root)) / "prompts.db")
        result = pm.render(prompt_id, variables or {})
        if result is None:
            raise HTTPException(404, "Prompt not found")
        return {"rendered": result}

    @app.post("/ab-tests")
    def create_ab_test(name: str, prompt_a_id: str, prompt_b_id: str, auth: bool = Depends(_verify_key)):
        from ..tools.prompt_manager import PromptManager
        pm = PromptManager(Path(str(gw.config.workspace_root)) / "prompts.db")
        return pm.create_ab_test(name, prompt_a_id, prompt_b_id)

    @app.get("/ab-tests/{test_id}")
    def get_ab_test(test_id: str, auth: bool = Depends(_verify_key)):
        from ..tools.prompt_manager import PromptManager
        pm = PromptManager(Path(str(gw.config.workspace_root)) / "prompts.db")
        return pm.get_ab_summary(test_id)

    # ── Benchmark endpoints ────────────────────────────────────────────

    @app.get("/benchmarks")
    def list_benchmarks(auth: bool = Depends(_verify_key)):
        from ..tools.benchmarks import BenchmarkRunner
        return {"benchmarks": BenchmarkRunner.list_benchmarks()}

    @app.post("/benchmarks/run")
    def run_benchmark(benchmark: str, model_path: str = None, auth: bool = Depends(_verify_key)):
        from ..tools.benchmarks import BenchmarkRunner
        br = BenchmarkRunner(model_path)
        try:
            result = br.run(benchmark)
            return result
        except ValueError as e:
            raise HTTPException(400, str(e))

    # ── Colab Deploy endpoint ──────────────────────────────────────────

    @app.post("/deploy/colab")
    def deploy_colab(api_key: str = "", hf_token: str = "", server_api_key: str = "epsionic-local-key",
                     auth: bool = Depends(_verify_key)):
        from ..tools.colab_deploy import generate_colab_notebook
        notebook = generate_colab_notebook(api_key, hf_token, server_api_key)
        return {"status": "generated", "cells": len(notebook.splitlines()), "notebook": notebook[:500]}

    import re
    _DOMAIN_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")

    @app.post("/train/{domain}")
    def train_domain(domain: str, auth: bool = Depends(_verify_key)):
        if not _DOMAIN_PATTERN.match(domain) or domain not in DOMAINS:
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
    def list_experiments(auth: bool = Depends(_verify_key)):
        exps = gw.memory.list_experiments()
        return {"count": len(exps), "experiments": [{'id': e.get('id'), 'name': e.get('name'),
                    'status': e.get('status'), 'tags': e.get('tags', [])} for e in exps[:50]]}

    @app.get("/status")
    def status(auth: bool = Depends(_verify_key)):
        return {"status": gw.state.status, "domain": gw.state.domain, "tools": gw.state.tools_loaded,
                "sessions": gw.state.sessions_created, "errors": gw.state.errors_encountered,
                "fixed": gw.state.errors_fixed, "started": gw.state.started_at,
                "budget": gw.check_budget() if hasattr(gw, 'check_budget') else True}

    _EXP_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+_\d{8}_\d{6}$")

    @app.get("/experiments/{exp_id}")
    def get_experiment(exp_id: str, auth: bool = Depends(_verify_key)):
        if not _EXP_ID_PATTERN.match(exp_id):
            raise HTTPException(status_code=400, detail="Invalid experiment ID format")
        exp = gw.memory.get_experiment(exp_id)
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return exp

    @app.get("/experiments/compare")
    def compare_experiments(a: str, b: str, auth: bool = Depends(_verify_key)):
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

    @app.get("/experiments/trends")
    def experiment_trends(auth: bool = Depends(_verify_key)):
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
                "avg_loss": round(sum(e.get("metrics", {}).get("eval_loss", 0) or 0 for e in exps) / max(len(exps), 1), 4),
                "budget_halted": False}

    @app.post("/experiments/export")
    def export_experiments(auth: bool = Depends(_verify_key)):
        path = gw.memory.export_experiments_csv()
        return {"path": path, "count": len(gw.memory.list_experiments())}

    from ..utils.security import sanitize_plugin_name

    @app.post("/jobs")
    def create_job(job_type: str, params: dict = None, priority: int = 0, auth: bool = Depends(_verify_key)):
        safe_type = sanitize_plugin_name(job_type)
        if not safe_type:
            raise HTTPException(status_code=400, detail="Invalid job type")
        priority = min(max(int(priority), 0), 100)
        job_id = gw.memory.create_job(safe_type, params or {}, priority)
        return {"job_id": job_id, "status": "queued"}

    @app.get("/jobs")
    def list_jobs(status: str = None, job_type: str = None, auth: bool = Depends(_verify_key)):
        jobs = gw.memory.list_jobs(status=status, job_type=job_type)
        return {"count": len(jobs), "jobs": jobs[:50]}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, auth: bool = Depends(_verify_key)):
        job = gw.memory.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, auth: bool = Depends(_verify_key)):
        ok = gw.memory.cancel_job(job_id)
        if not ok:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled or not found")
        return {"job_id": job_id, "status": "cancelled"}

    @app.get("/plugins")
    def list_plugins(auth: bool = Depends(_verify_key)):
        return {"count": len(gw.plugins), "plugins": [
            {"name": n, "enabled": p.enabled, "hooks": list(p._hooks.keys())} for n, p in gw.plugins.items()]}

    @app.post("/plugins/{name}/toggle")
    def toggle_plugin(name: str, auth: bool = Depends(_verify_key)):
        if name not in gw.plugins:
            raise HTTPException(status_code=404, detail=f"Plugin '{name}' not found")
        p = gw.plugins[name]
        if p.enabled:
            gw._plugin_manager.disable(name)
        else:
            gw._plugin_manager.enable(name)
        return {"name": name, "enabled": not p.enabled}

    # Plugin Registry / Marketplace endpoints
    @app.get("/plugins/registry")
    def list_plugin_registry(auth: bool = Depends(_verify_key)):
        try:
            from .tools.plugin_registry import PluginRegistry
            reg = PluginRegistry(memory_store=gw.memory)
            return {"plugins": reg.list_plugins()}
        except Exception as e:
            return {"plugins": [], "error": str(e)}

    @app.post("/plugins/install/pypi")
    def install_plugin_pypi(package: str, name: str = None, auth: bool = Depends(_verify_key)):
        from .tools.plugin_registry import PluginRegistry
        reg = PluginRegistry(memory_store=gw.memory)
        result = reg.install_from_pypi(package, name)
        return result

    @app.post("/plugins/install/github")
    def install_plugin_github(repo: str, name: str = None, auth: bool = Depends(_verify_key)):
        from .tools.plugin_registry import PluginRegistry
        reg = PluginRegistry(memory_store=gw.memory)
        result = reg.install_from_github(repo, name)
        return result

    @app.post("/plugins/install/path")
    def install_plugin_path(path: str, name: str = None, auth: bool = Depends(_verify_key)):
        from .tools.plugin_registry import PluginRegistry
        reg = PluginRegistry(memory_store=gw.memory)
        result = reg.install_from_path(path, name)
        return result

    @app.post("/plugins/registry/{name}/enable")
    def enable_plugin_registry(name: str, auth: bool = Depends(_verify_key)):
        from .tools.plugin_registry import PluginRegistry
        reg = PluginRegistry(memory_store=gw.memory)
        ok = reg.enable(name)
        return {"success": ok, "name": name}

    @app.post("/plugins/registry/{name}/disable")
    def disable_plugin_registry(name: str, auth: bool = Depends(_verify_key)):
        from .tools.plugin_registry import PluginRegistry
        reg = PluginRegistry(memory_store=gw.memory)
        ok = reg.disable(name)
        return {"success": ok, "name": name}

    @app.delete("/plugins/registry/{name}")
    def uninstall_plugin(name: str, auth: bool = Depends(_verify_key)):
        from .tools.plugin_registry import PluginRegistry
        reg = PluginRegistry(memory_store=gw.memory)
        ok = reg.uninstall(name)
        return {"success": ok, "name": name}

    # Scheduler endpoints
    @app.get("/schedules")
    def list_schedules(auth: bool = Depends(_verify_key)):
        return {"schedules": gw.list_schedules()}

    @app.post("/schedules")
    def add_schedule(domain: str, interval_hours: float = 24.0, auth: bool = Depends(_verify_key)):
        return gw.schedule_training(domain, interval_hours)

    @app.delete("/schedules/{domain}")
    def remove_schedule(domain: str, auth: bool = Depends(_verify_key)):
        ok = gw.remove_schedule(domain)
        return {"success": ok, "domain": domain}

    # State persistence endpoints
    @app.post("/state/save")
    def save_state(auth: bool = Depends(_verify_key)):
        result = gw.memory.save_state()
        return result

    @app.post("/state/restore")
    def restore_state(auth: bool = Depends(_verify_key)):
        result = gw.memory.restore_state()
        return result

    @app.get("/device")
    def device_info(auth: bool = Depends(_verify_key)):
        return {"name": gw.device.name, "backend": gw.device.backend,
                "vram_gb": gw.device.vram_gb,
                "batch_recommendation": gw.device.recommended_batch_size(),
                "quantization": gw.device.recommended_quantization()}

    @app.get("/logs")
    def get_logs(lines: int = 50, auth: bool = Depends(_verify_key)):
        import subprocess
        try:
            n = min(max(int(lines), 1), 5000)
            log_path = Path(cfg.logs_dir).resolve() / "epsionic.log"
            # Prevent path traversal
            if not str(log_path).startswith(str(Path(cfg.logs_dir).resolve())):
                return {"logs": ["Invalid log path"]}
            if not log_path.exists():
                return {"logs": ["Log file not found"]}
            result = subprocess.run(["tail", "-n", str(n), str(log_path)],
                                    capture_output=True, text=True, timeout=5)
            return {"logs": result.stdout.split("\n")}
        except Exception:
            return {"logs": ["Log file not available"]}

    # Registry endpoints
    @app.get("/registry")
    def list_registry(auth: bool = Depends(_verify_key)):
        try:
            from .tools.model_registry import ModelRegistry
            reg = ModelRegistry(memory_store=gw.memory)
            return {"models": reg.list()}
        except Exception as e:
            return {"models": [], "error": str(e)}

    @app.get("/registry/{name}")
    def get_registry_model(name: str, auth: bool = Depends(_verify_key)):
        try:
            from .tools.model_registry import ModelRegistry
            reg = ModelRegistry(memory_store=gw.memory)
            latest = reg.get_latest(name)
            if not latest:
                raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
            return latest
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    _NODE_TYPE_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")

    # Flow engine endpoints
    @app.post("/flows")
    def create_flow(name: str, description: str = "", auth: bool = Depends(_verify_key)):
        if not isinstance(name, str) or len(name) > 200 or len(name) < 1:
            raise HTTPException(status_code=400, detail="Invalid flow name")
        try:
            from .tools.flow import FlowEngine
            engine = FlowEngine()
            fid = engine.create_flow(name[:200], description[:1000] if isinstance(description, str) else "")
            return {"flow_id": fid, "status": "created"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/flows")
    def list_flows(auth: bool = Depends(_verify_key)):
        try:
            from .tools.flow import FlowEngine
            engine = FlowEngine()
            return {"flows": engine.list_flows()}
        except Exception as e:
            return {"flows": [], "error": str(e)}

    _FLOW_ID_PATTERN = re.compile(r"^flow_\d{8}_\d{6}_\d{6}$")

    @app.get("/flows/{flow_id}")
    def get_flow(flow_id: str, auth: bool = Depends(_verify_key)):
        if not _FLOW_ID_PATTERN.match(flow_id):
            raise HTTPException(status_code=400, detail="Invalid flow ID format")
        try:
            from .tools.flow import FlowEngine
            engine = FlowEngine()
            flow = engine.get_flow(flow_id)
            if not flow:
                raise HTTPException(status_code=404, detail="Flow not found")
            return flow
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/flows/{flow_id}/trigger")
    def trigger_flow(flow_id: str, payload: dict = None, auth: bool = Depends(_verify_key)):
        if not _FLOW_ID_PATTERN.match(flow_id):
            raise HTTPException(status_code=400, detail="Invalid flow ID format")
        try:
            from .tools.flow import FlowEngine
            engine = FlowEngine()
            result = engine.trigger_webhook(flow_id, payload or {},
                                            tool_executor=lambda t, **kw: {"called": t, "params": kw})
            return {"flow_id": flow_id, "result": result, "status": "triggered"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/flows/{flow_id}/nodes")
    def add_flow_node(flow_id: str, node_type: str, params: dict = None,
                      after_node: str = None, auth: bool = Depends(_verify_key)):
        if not _FLOW_ID_PATTERN.match(flow_id):
            raise HTTPException(status_code=400, detail="Invalid flow ID format")
        if not _NODE_TYPE_PATTERN.match(node_type):
            raise HTTPException(status_code=400, detail="Invalid node type")
        try:
            from .tools.flow import FlowEngine
            engine = FlowEngine()
            nid = engine.add_node(flow_id, node_type, params or {}, after_node=after_node)
            return {"node_id": nid, "status": "added"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/flows/editor")
    def flow_editor(auth: bool = Depends(_verify_key)):
        html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self';">
<title>Epslionic Flow Editor</title>
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;height:100vh;display:flex;flex-direction:column}
#toolbar{background:#1e293b;padding:12px 20px;display:flex;gap:12px;align-items:center;border-bottom:1px solid #334155}
#toolbar button{padding:8px 16px;background:#3b82f6;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:14px}
#toolbar button:hover{background:#2563eb}
#toolbar select{padding:8px;border-radius:6px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;font-size:14px}
#cy{flex:1}
#panel{width:320px;background:#1e293b;padding:16px;border-left:1px solid #334155;overflow-y:auto;display:flex;flex-direction:column;gap:12px}
#panel input,#panel textarea{width:100%;padding:8px;background:#0f172a;border:1px solid #475569;border-radius:4px;color:#e2e8f0;font-size:13px}
#panel label{font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px}
</style></head>
<body>
<div id="toolbar">
<strong style="color:#3b82f6;font-size:16px">Epslionic Flow Editor</strong>
<select id="nodeType"><option value="train">Train</option><option value="discover">Discover</option><option value="evaluate">Evaluate</option><option value="preference_train">Preference Train</option><option value="merge_models">Merge</option><option value="generate_data">Generate Data</option><option value="quantize">Quantize</option></select>
<button onclick="addNode()">+ Add Node</button>
<button onclick="saveFlow()">Save</button>
<button onclick="runFlow()">Run</button>
<button onclick="loadFlows()">Load</button>
<span id="flowLabel" style="color:#94a3b8;font-size:13px">No flow loaded</span>
</div>
<div style="display:flex;flex:1">
<div id="cy"></div>
<div id="panel">
<label>Flow Name</label><input id="flowName" placeholder="my-training-flow" value="untitled">
<label>Node ID</label><input id="nodeId" placeholder="auto">
<label>Params (JSON)</label><textarea id="nodeParams" rows="4" placeholder='{"model":"mistral-7b"}'></textarea>
<label>Retry on fail</label><input id="retryCount" type="number" value="0" min="0">
</div></div>
<script>
function esc(s){var d=document.createElement('div');d.appendChild(document.createTextNode(s));return d.innerHTML}
var cy,currentFlowId=null,nodes=[],edges=[];
var API=window.location.origin;
document.addEventListener('DOMContentLoaded',function(){
cy=cytoscape({container:document.getElementById('cy'),style:[{selector:'node',style:{'background-color':'#3b82f6','label':'data(label)','text-valign':'bottom','color':'#e2e8f0','font-size':'11px','width':60,'height':60,'shape':'round-rectangle'}},{selector:'edge',style:{'width':2,'line-color':'#475569','target-arrow-color':'#475569','target-arrow-shape':'triangle','curve-style':'bezier'}},{selector:':selected',style:{'border-width':3,'border-color':'#f59e0b'}}],layout:{name:'grid'},wheelSensitivity:0.3});
cy.on('tap','node',function(e){var n=e.target;document.getElementById('nodeId').value=n.id();document.getElementById('nodeParams').value=JSON.stringify(n.data('params')||{},null,2);document.getElementById('retryCount').value=n.data('retry')||0});
cy.on('dragfree','node',function(){positionNodes()});
});
function addNode(){var t=document.getElementById('nodeType').value;var p=document.getElementById('nodeParams').value;var params=p?JSON.parse(p):{};var retry=parseInt(document.getElementById('retryCount').value)||0;var nid=document.getElementById('nodeId').value||'node_'+(nodes.length+1);var label=t.charAt(0).toUpperCase()+t.slice(1).replace('_',' ');nodes.push({id:nid,type:t,params:params,retry_on_fail:retry});cy.add({group:'nodes',data:{id:nid,label:label,type:t,params:params,retry:retry}});positionNodes();document.getElementById('nodeId').value='';}
function positionNodes(){var n=cy.nodes();var cols=Math.ceil(Math.sqrt(n.length));n.forEach(function(node,i){var col=i%cols,row=Math.floor(i/cols);node.position({x:100+col*160,y:80+row*120})});cy.layout({name:'preset',fit:true,padding:30}).run();}
function saveFlow(){var name=esc(document.getElementById('flowName').value||'untitled');fetch(API+'/flows',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name})}).then(function(r){return r.json()}).then(function(res){currentFlowId=res.flow_id;document.getElementById('flowLabel').textContent='Flow: '+res.flow_id;var p=[];nodes.forEach(function(n,i){var after=i>0?nodes[i-1].id:null;p.push(fetch(API+'/flows/'+currentFlowId+'/nodes?node_type='+encodeURIComponent(n.type)+'&after_node='+(after||''),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({params:n.params||{}})}))});Promise.all(p).then(function(){alert('Flow saved: '+res.flow_id)}).catch(function(e){console.error(e)})}).catch(function(e){alert('Save failed')});}
function loadFlows(){fetch(API+'/flows').then(function(r){return r.json()}).then(function(d){if(!d.flows||!d.flows.length){alert('No flows found');return}var msg='Available flows:\\n';d.flows.forEach(function(f){msg+=f.id+' - '+esc(f.name||'?')+'\\n'});var id=prompt(msg+'\\nEnter Flow ID:');if(!id)return;fetch(API+'/flows/'+encodeURIComponent(id)).then(function(r){return r.json()}).then(function(flow){currentFlowId=id;document.getElementById('flowLabel').textContent='Flow: '+id;document.getElementById('flowName').value=flow.name||id;nodes=[];edges=[];cy.elements().remove();(flow.nodes||[]).forEach(function(n){nodes.push(n);cy.add({group:'nodes',data:{id:n.id,label:esc(n.type||'?').charAt(0).toUpperCase()+esc(n.type||'?').slice(1),type:n.type,params:n.params,retry:n.retry_on_fail}});if(n.connections&&n.connections.output){var tgt=n.connections.output;edges.push({source:n.id,target:tgt});cy.add({group:'edges',data:{source:n.id,target:tgt,id:'e_'+n.id+'_'+tgt}})}});positionNodes()})})}
function runFlow(){if(!currentFlowId){alert('Save or load a flow first');return}fetch(API+'/flows/'+currentFlowId+'/trigger',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}).then(function(r){return r.json()}).then(function(r){var ok=r.result&&r.result.success?'Success':'Failed';alert('Run '+esc(currentFlowId)+': '+ok)}).catch(function(e){alert('Run failed')});}
</script></body></html>"""
        return HTMLResponse(content=html)

    return app


def cmd_serve(gw: Gateway, host: str = '0.0.0.0', port: int = 8000) -> int:
    try:
        import uvicorn
    except ImportError:
        logger.error("fastapi/uvicorn not installed. Install with: pip install fastapi uvicorn")
        return 1

    app = make_app(gw)
    _echo(f"Starting API server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
    return 0


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog='epsionic',
        description=f'Epslionic-Colab Agent v{__version__} — Production LLM Training Agent',
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
    p.add_argument('--server-api-key', default=None, help='API key for FastAPI auth (env: EPSIONIC_API_KEY)')

    # Flow engine subcommands
    flow_sub = p.add_argument_group('Flow Engine')
    flow_sub.add_argument('--flow-list', action='store_true', help='List all flow engine pipelines')
    flow_sub.add_argument('--flow-create', type=str, default=None, metavar='NAME', help='Create a new flow')
    flow_sub.add_argument('--flow-run', type=str, default=None, metavar='FLOW_ID', help='Execute a flow')
    flow_sub.add_argument('--flow-mermaid', type=str, default=None, metavar='FLOW_ID', help='Export flow as Mermaid')
    flow_sub.add_argument('--flow-export', type=str, default=None, metavar='FLOW_ID', help='Export flow as YAML')

    # Plugin registry subcommands
    plugin_sub = p.add_argument_group('Plugin Registry')
    plugin_sub.add_argument('--plugin-list', action='store_true', help='List all registered plugins')
    plugin_sub.add_argument('--plugin-install', type=str, default=None, metavar='PYPI_PKG', help='Install plugin from PyPI')
    plugin_sub.add_argument('--plugin-github', type=str, default=None, metavar='REPO_URL', help='Install plugin from GitHub')
    plugin_sub.add_argument('--plugin-disable', type=str, default=None, metavar='NAME', help='Disable a plugin')
    plugin_sub.add_argument('--plugin-enable', type=str, default=None, metavar='NAME', help='Enable a plugin')
    plugin_sub.add_argument('--plugin-uninstall', type=str, default=None, metavar='NAME', help='Uninstall a plugin')

    # Scheduler subcommands
    sched_sub = p.add_argument_group('Scheduler')
    sched_sub.add_argument('--schedule-list', action='store_true', help='List scheduled training jobs')
    sched_sub.add_argument('--schedule-add', type=str, default=None, metavar='DOMAIN', help='Schedule recurring training')
    sched_sub.add_argument('--schedule-interval', type=float, default=24.0, help='Interval in hours (default: 24)')

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
    """Delegate to gateway's built-in graceful shutdown handlers."""
    if hasattr(gw, "_setup_graceful_shutdown"):
        gw._setup_graceful_shutdown()
    else:
        logger.warning("Gateway has no graceful shutdown handler")


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
        for p in ['epsionic.json', 'epsionic.yaml', 'epsionic.yml', 'config/epsionic.json', '~/.config/epsionic/config.json']:
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
    if args.server_api_key:
        os.environ["EPSIONIC_API_KEY"] = args.server_api_key

    errors = cfg.validate()
    if errors:
        for e in errors:
            logger.error("Config validation: %s", e)
        return 1

    if args.install_completion:
        import subprocess
        shell = os.environ.get("SHELL", "")
        if "zsh" in shell:
            rc = subprocess.call(["epsionic", "--help"])
            print("\nAdd to ~/.zshrc: eval \"$(_EPSIONIC_COMPLETE=zsh_source epsionic)\"")
        elif "bash" in shell:
            print("\nAdd to ~/.bashrc: eval \"$(_EPSIONIC_COMPLETE=bash_source epsionic)\"")
        else:
            print("Tab-completion: add to your shell rc file or use `epsionic --help`")
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

    # Flow engine CLI commands
    if args.flow_list:
        from .tools.flow import FlowEngine
        engine = FlowEngine()
        flows = engine.list_flows()
        if not flows:
            _echo("No flows found")
        else:
            _make_table("Flow Pipelines", ["ID", "Name", "Nodes", "Description"],
                        [(f["id"], f["name"], str(f["nodes"]), f.get("description", "")) for f in flows])
        return 0

    if args.flow_create:
        from .tools.flow import FlowEngine
        engine = FlowEngine()
        fid = engine.create_flow(args.flow_create)
        _echo(f"Flow created: {fid}")
        return 0

    if args.flow_run:
        from .tools.flow import FlowEngine
        engine = FlowEngine()
        result = engine.execute(args.flow_run, tool_executor=lambda t, **kw: {"called": t})
        _status_panel("Flow Execution", f"Success: {result['success']}\nNodes: {len(result['nodes'])}")
        return 0

    if args.flow_mermaid:
        from .tools.flow import FlowEngine
        engine = FlowEngine()
        _echo(engine.export_mermaid(args.flow_mermaid))
        return 0

    if args.flow_export:
        from .tools.flow import FlowEngine
        engine = FlowEngine()
        _echo(engine.export_yaml(args.flow_export))
        return 0

    # Plugin registry CLI commands
    if args.plugin_list:
        from .tools.plugin_registry import PluginRegistry
        reg = PluginRegistry()
        reg.discover_local()
        plugins = reg.list_plugins()
        if not plugins:
            _echo("No plugins registered")
        else:
            _make_table("Plugin Registry", ["Name", "Source", "Enabled", "Description"],
                        [(p["name"], p["source"], "Yes" if p.get("enabled") else "No", p.get("description", "")) for p in plugins])
        return 0

    if args.plugin_install:
        from .tools.plugin_registry import PluginRegistry
        reg = PluginRegistry()
        result = reg.install_from_pypi(args.plugin_install)
        if result.get("success"):
            _echo(f"Installed plugin: {result['plugin']}")
        else:
            _echo(f"Failed: {result.get('error', 'unknown')}")
        return 0

    if args.plugin_github:
        from .tools.plugin_registry import PluginRegistry
        reg = PluginRegistry()
        result = reg.install_from_github(args.plugin_github)
        if result.get("success"):
            _echo(f"Installed plugin from GitHub: {result['plugin']}")
        else:
            _echo(f"Failed: {result.get('error', 'unknown')}")
        return 0

    if args.plugin_disable:
        from .tools.plugin_registry import PluginRegistry
        reg = PluginRegistry()
        ok = reg.disable(args.plugin_disable)
        _echo(f"Plugin '{args.plugin_disable}' disabled: {ok}")
        return 0

    if args.plugin_enable:
        from .tools.plugin_registry import PluginRegistry
        reg = PluginRegistry()
        ok = reg.enable(args.plugin_enable)
        _echo(f"Plugin '{args.plugin_enable}' enabled: {ok}")
        return 0

    if args.plugin_uninstall:
        from .tools.plugin_registry import PluginRegistry
        reg = PluginRegistry()
        ok = reg.uninstall(args.plugin_uninstall)
        _echo(f"Plugin '{args.plugin_uninstall}' uninstalled: {ok}")
        return 0

    # Scheduler CLI commands
    if args.schedule_list:
        schedules = gw.list_schedules()
        if not schedules:
            _echo("No scheduled training jobs")
        else:
            _make_table("Scheduled Training", ["Domain", "Interval (h)", "Last Run", "Enabled"],
                        [(s["domain"], str(s["interval_hours"]), s.get("last_run", "never")[:19], "Yes" if s.get("enabled") else "No") for s in schedules])
        return 0

    if args.schedule_add:
        result = gw.schedule_training(args.schedule_add, args.schedule_interval)
        _echo(f"Scheduled: {result['domain']} every {result['interval_hours']}h")
        return 0

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
