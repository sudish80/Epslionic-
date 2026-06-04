import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field, asdict

from ..config import AgentConfig
from ..exceptions import (
    EpslionicError, ToolError, ConfigurationError,
    error_code, as_error
)
from .session import SessionManager, SessionStatus
from .memory import MemoryStore
from .heartbeat import HeartbeatMonitor
from .brain import LLMBrain, BrainAction
from .domain import DomainSelector, DOMAINS
from .state import StateMachine, StateTransitionError
from .plugin import PluginManager
from .chain import ChainBuilder, Pipeline, Chain
from ..utils.device import DeviceManager

logger = logging.getLogger("epsionic.gateway")

_BUDGET_HALT = False


@dataclass
class AgentState:
    status: str = "idle"
    current_objective: str = ""
    current_task: str = ""
    started_at: str = ""
    tools_loaded: int = 0
    sessions_created: int = 0
    errors_encountered: int = 0
    errors_fixed: int = 0
    domain: str = ""
    domain_config: dict = field(default_factory=dict)


class Gateway:
    """Central orchestrator — integrates FSM, plugins, chain pipeline, device, heartbeat."""

    def __init__(self, config: AgentConfig):
        self.config = config.resolve_env()
        self.state = AgentState()
        self.state.started_at = datetime.now().isoformat()
        self._fsm = StateMachine("idle")

        self.memory = MemoryStore(self.config.memory_dir)
        self.session_manager = SessionManager(max_concurrent=config.max_concurrent_sessions)
        self.brain = LLMBrain(
            provider=config.llm_provider, model=config.llm_model,
            api_key=config.openai_api_key or config.anthropic_api_key,
            temperature=config.llm_temperature,
        )
        self.heartbeat = HeartbeatMonitor(self.memory, interval_seconds=config.heartbeat_interval_seconds)
        self.device = DeviceManager()

        device_cfg = {"gpu": self.device.is_cuda, "vram_gb": self.device.vram_gb}
        self.domain_selector = DomainSelector(brain=self.brain, device_info=device_cfg)

        self._tool_registry: Dict[str, Callable] = {}
        self._running = False
        self._skill_files: List[Path] = []
        self._hooks: Dict[str, List[Callable]] = {e: [] for e in
            ["before_tool", "after_tool", "on_error", "on_start", "on_complete"]}

        plugins_dir = Path(str(self.config.workspace_root)) / "plugins"
        self._plugin_manager = PluginManager(plugins_dir)
        self._plugin_manager.discover()
        self._plugin_manager.create_example()
        self._plugin_manager.discover()

        self._cost_tracker = None
        budget = getattr(config, 'budget_limit', 50.0)
        if budget > 0:
            try:
                from ..tools.cost_tracker import CostTracker
                self._cost_tracker = CostTracker(
                    memory_store=self.memory,
                    budget_limit=budget,
                    cost_file=Path(str(self.config.workspace_root)) / "costs.jsonl",
                )
            except Exception:
                pass

        # Scheduler for recurring training
        self._scheduler_thread = None
        self._scheduler_running = False
        self._schedule = []  # list of {"domain": str, "interval_hours": float, "last_run": str, "enabled": bool}

        logger.info(f"Gateway initialized — device: {self.device.name}, plugins: {len(self._plugin_manager.plugins)}")

    # ── Plugin integration ──────────────────────────────────────────────

    def discover_plugins(self, create_example: bool = True):
        found = self._plugin_manager.discover()
        if not found and create_example:
            self._plugin_manager.create_example()
            found = self._plugin_manager.discover()
        if found:
            logger.info(f"Discovered plugins: {', '.join(found)}")

    @property
    def plugins(self):
        return self._plugin_manager.plugins

    def plugin_summary(self) -> str:
        return self._plugin_manager.summary()

    # ── State machine ───────────────────────────────────────────────────

    def _state_transition(self, target: str) -> str:
        try:
            self._fsm.transition(target)
            self.state.status = self._fsm.state
            self._log_audit("state_transition", {"from": self._fsm.history[-2] if len(self._fsm.history) > 1 else "initial", "to": target})
        except StateTransitionError as e:
            logger.warning(str(e))
        return self._fsm.state

    # ── Tool registry ───────────────────────────────────────────────────

    def register_tool(self, name: str, description: str,
                      handler: Callable, parameters: dict = None):
        self._tool_registry[name] = handler
        self.brain.register_tool(name, description, handler, parameters or {})
        self.state.tools_loaded = len(self._tool_registry)
        logger.info(f"Registered tool: {name}")

    def hook(self, event: str, fn: Callable):
        if event in self._hooks:
            self._hooks[event].append(fn)
        else:
            raise ConfigurationError(f"Unknown hook event: {event}", context={"available": list(self._hooks.keys())})

    def _run_hooks(self, event: str, **kwargs):
        for fn in self._hooks.get(event, []):
            try:
                fn(self, **kwargs)
            except Exception as e:
                logger.warning(f"Hook {event}/{fn.__name__}: {e}")
        self._plugin_manager.run_hook(event, **kwargs)

    # ── Skills ──────────────────────────────────────────────────────────

    def load_skills(self, skills_dir: Path = None):
        skills_dir = Path(skills_dir or self.config.skills_dir)
        if not skills_dir.exists():
            logger.warning(f"Skills directory not found: {skills_dir}")
            return
        for skill_file in skills_dir.glob("**/*.md"):
            self._skill_files.append(skill_file)
            content = skill_file.read_text()
            title, description = "Untitled Skill", "No description"
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    for line in parts[1].strip().split("\n"):
                        if line.startswith("title:"):
                            title = line.split(":", 1)[1].strip().strip('"')
                        elif line.startswith("description:"):
                            description = line.split(":", 1)[1].strip().strip('"')
            skill_name = skill_file.stem.lower().replace(" ", "_")
            self.register_tool(name=f"skill_{skill_name}", description=description,
                handler=lambda c=content: f"Skill instructions loaded:\n\n{c[:3000]}")
            logger.info(f"Loaded skill: {title} ({skill_file.name})")

    # ── Heartbeat ───────────────────────────────────────────────────────

    def start_heartbeat(self):
        self.heartbeat.add_check("training_progress", self._check_training_progress)
        self.heartbeat.add_check("error_check", self._check_errors)
        self.heartbeat.add_check("device_check", self._check_device)
        self.heartbeat.add_check("plugin_check", self._check_plugins)
        self.heartbeat.start()

    def _check_training_progress(self) -> str:
        issues = []
        for exp in self.memory.list_experiments("running"):
            loss = exp.get("metrics", {}).get("loss", 0)
            if loss and (loss != loss or loss > 1e10):
                issues.append(f"Invalid loss ({loss}) in {exp['name']}")
        return "; ".join(issues)

    def _check_errors(self) -> str:
        errors = self.memory.get_unfixed_errors()
        return f"{len(errors)} unresolved errors" if errors else ""

    def _check_device(self) -> str:
        issues = []
        if not self.device.is_cuda and not self.device.is_mps:
            issues.append("No GPU available — training disabled")
        if self.device.is_cuda and self.device.vram_gb < 8:
            issues.append(f"Low VRAM ({self.device.vram_gb:.1f} GB) — use 4-bit")
        return "; ".join(issues)

    def _check_plugins(self) -> str:
        enabled = list(self._plugin_manager.enabled_plugins.keys())
        return f"Plugins: {', '.join(enabled)}" if enabled else ""

    # ── Chain pipeline ──────────────────────────────────────────────────

    def build_pipeline(self, domain_key: str = None) -> Pipeline:
        p = Pipeline()
        p.add("discover", lambda ctx: self._run_chain_step("discover", ctx))
        p.add("select", lambda ctx: self._run_chain_step("select", ctx))
        p.add("prepare", lambda ctx: self._run_chain_step("prepare", ctx))
        p.add("train", lambda ctx: self._run_chain_step("train", ctx))
        return p

    def _run_chain_step(self, step_name: str, ctx: dict) -> dict:
        if step_name == "discover":
            domain_config = ctx.get("domain_config", {})
            search_query = domain_config.get("search_queries", [ctx.get("domain", "general")])[0]
            tool = self._tool_registry.get("discover")
            if tool:
                datasets = tool(search=search_query)
                return {"datasets": datasets, "dataset_count": len(datasets)}
            return {"datasets": [], "dataset_count": 0}
        if step_name == "select":
            domain_config = ctx.get("domain_config", {})
            preferred = domain_config.get("datasets", [])
            selected = preferred[0] if preferred else "gsm8k"
            return {"selected_dataset": selected}
        if step_name == "prepare":
            model_name = ctx.get("domain_config", {}).get("models", ["unsloth/mistral-7b-bnb-4bit"])[0]
            hints = ctx.get("domain_config", {}).get("training_hints", {})
            tc = dict(self.config.default_training_config)
            tc.update({"model_name": model_name,
                "learning_rate": hints.get("learning_rate", tc["learning_rate"]),
                "lora_r": hints.get("lora_r", tc["lora_r"]),
                "max_seq_length": hints.get("max_seq_length", tc["max_seq_length"]),
                "domain": ctx.get("domain", "")})
            tool = self._tool_registry.get("prepare")
            if tool:
                result = tool(experiment_id=ctx.get("experiment_id", ""), model_name=model_name,
                              dataset_dict={"dataset_id": ctx.get("selected_dataset", "")},
                              training_args=tc)
                return {"training_config": tc, "prep_result": result}
            return {"training_config": tc}
        if step_name == "train":
            if not self.check_budget():
                return {"error": "Budget exceeded", "halted": True}
            tc = ctx.get("training_config", self.config.default_training_config)
            tool = self._tool_registry.get("train")
            if tool:
                result = tool(experiment_id=ctx.get("experiment_id", ""),
                    model_name=tc.get("model_name"), dataset_dict={"dataset_id": ctx.get("selected_dataset", "")},
                    training_args=tc, objective=ctx.get("objective", ""))
                if self._cost_tracker:
                    self._cost_tracker.track_training(
                        duration_hours=result.get("hours", 1.0),
                        gpu_type=self.device.name,
                    )
                return {"train_result": result}
            return {"train_result": {"status": "queued"}}
        return {}

    # ── Core execution ──────────────────────────────────────────────────

    def run_objective(self, objective: str, max_steps: int = 10) -> Dict[str, Any]:
        self._state_transition("running")
        self.state.current_objective = objective
        self._run_hooks("on_start", objective=objective)

        session = self.session_manager.create_session(name=objective[:40], config={"objective": objective})
        self.state.sessions_created += 1
        logger.info(f"Running objective: {objective}")

        result = {"objective": objective, "session_id": session.id, "steps": [], "outputs": {}, "error": None}
        for step in range(max_steps):
            context = self._build_context()
            action = self.brain.think_and_act(objective, context)
            sr = {"step": step + 1, "tool": action.tool, "reasoning": action.reasoning,
                  "params": action.params, "status": "pending"}
            if action.tool in self._tool_registry:
                if action.tool in ("train", "prepare", "preference_train", "generate_data", "evaluate"):
                    if not self.check_budget():
                        sr["status"], sr["error"] = "skipped", "Budget exceeded - training halted"
                        result["error"] = "Budget exceeded"
                        result["steps"].append(sr)
                        break
                try:
                    self._run_hooks("before_tool", tool=action.tool, params=action.params)
                    handler = self._tool_registry[action.tool]
                    sr["params"].update({"objective": objective, "session_id": session.id})
                    output = handler(**action.params)
                    sr["output"], sr["status"] = str(output)[:500], "completed"
                    result["outputs"][action.tool] = output
                    self._run_hooks("after_tool", tool=action.tool, result=output)
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}"
                    sr["status"], sr["error"] = "failed", error_msg
                    self.memory.log_error(source=f"tool_{action.tool}", error=error_msg,
                        context={"step": step, "params": action.params})
                    self.state.errors_encountered += 1
                    result["error"] = error_msg
                    self._run_hooks("on_error", tool=action.tool, error=e, step=step, params=action.params)
                    if step < max_steps - 1:
                        fix = self.brain.analyze_error(error_msg, action.params)
                        if fix.get("retry_with_params"):
                            action.params.update(fix["retry_with_params"])
                            sr["auto_fix"] = fix
            else:
                sr["status"], sr["error"] = "skipped", f"Tool '{action.tool}' not found"
            result["steps"].append(sr)
            if sr["status"] == "completed" and action.tool in ("train", "finalize"):
                break
        self._state_transition("idle")
        self._run_hooks("on_complete", result=result)
        return result

    def _build_context(self) -> str:
        parts = []
        mem = self.memory.get_training_summary_markdown()
        if mem:
            parts.append(f"## Memory\n{mem[:1000]}")
        parts.append(self.session_manager.get_status_markdown())
        parts.append(f"## Errors\n{self.memory.get_error_summary()}")
        return "\n\n".join(parts)

    def ask_domain_and_run(self) -> Dict[str, Any]:
        dk, dc = self.domain_selector.ask()
        self.state.domain = dk
        self.state.domain_config = dc
        print(f"\n{self.domain_selector.get_domain_info_markdown(dk, dc)}\n")
        return self.run_with_domain(self.domain_selector.build_objective(dk, dc), dk, dc)

    def run_with_domain(self, objective: str, domain_key: str,
                        domain_config: dict) -> Dict[str, Any]:
        self._state_transition("running")
        self.state.current_objective, self.state.domain, self.state.domain_config = objective, domain_key, domain_config
        logger.info(f"Domain pipeline: {domain_config['name']}")

        exp_id = self.memory.create_experiment(name=f"{domain_key}_training", config=domain_config)
        self.state.sessions_created += 1

        result = {"objective": objective, "domain": domain_key, "domain_config": domain_config,
                  "experiment_id": exp_id, "steps": [], "outputs": {}, "error": None}

        ctx = {"objective": objective, "domain": domain_key, "domain_config": domain_config, "experiment_id": exp_id}
        pipeline = self.build_pipeline(domain_key)
        pipeline_ctx = pipeline.execute(ctx)

        for s in pipeline._steps:
            name = s.name
            step_data = pipeline_ctx.get(name, {})
            step = {"step": len(result["steps"]) + 1, "tool": name, "status": "completed",
                    "reasoning": f"Pipeline step: {name}", "output": str(step_data)[:200]}
            if "_error" in step_data:
                step["status"] = "failed"
                step["error"] = step_data["_error"]
                result["error"] = step_data["_error"]
            result["steps"].append(step)
            result["outputs"][name] = step_data
            print(f"  Step {step['step']}/4: {name} — {step['status']}")

        self._state_transition("idle")
        return result

    # ── Status ──────────────────────────────────────────────────────────

    def get_status_markdown(self) -> str:
        plugin_names = list(self._plugin_manager.plugins.keys())
        enabled_names = list(self._plugin_manager.enabled_plugins.keys())
        lines = [
            "# Gateway Status", "",
            f"Status: {self.state.status}",
            f"FSM: {self._fsm.status()}",
            f"Device: {self.device.name} ({self.device.backend}, {self.device.vram_gb:.1f} GB)",
            f"Started: {self.state.started_at}",
            f"Domain: {self.state.domain or 'not set'}",
            f"Tools: {self.state.tools_loaded} | Sessions: {self.state.sessions_created}",
            f"Errors: {self.state.errors_encountered} / fixed: {self.state.errors_fixed}",
            f"Objective: {self.state.current_objective or 'none'}",
            f"Plugins: {', '.join(enabled_names) if enabled_names else 'none'} ({len(plugin_names)} total)",
            "",
        ]
        lines.append(self.session_manager.get_status_markdown())
        lines.append("")
        lines.append(self.heartbeat.get_heartbeat_markdown())
        return "\n".join(lines)

    # ── Domain pipeline ─────────────────────────────────────────────────

    def run_domain(self, domain_key: str, domain_config: dict) -> Dict[str, Any]:
        self._state_transition("running")
        objective = self.domain_selector.build_objective(domain_key, domain_config)
        from ..exceptions import as_error
        try:
            result = self.run_with_domain(objective, domain_key, domain_config)
            return result
        except Exception as e:
            wrapped = as_error(e)
            logger.error(f"Domain {domain_key} failed: {wrapped}")
            return {"error": str(wrapped), "domain": domain_key, "steps": [], "outputs": {}}
        finally:
            self._state_transition("idle")

    def run_autonomous_loop(self, continuous: bool = False) -> List[str]:
        completed = []
        self._state_transition("running")
        self._running = True

        if continuous:
            import itertools
            cycle = itertools.cycle(DOMAINS.items())
            try:
                count = 0
                while self._running and count < 10:
                    for dk, dc in cycle:
                        if not self._running:
                            break
                        logger.info(f"Auto loop: {dc['name']}")
                        try:
                            self.run_domain(dk, dc)
                            completed.append(dk)
                        except Exception as e:
                            logger.error(f"Domain {dk}: {e}")
                        count += 1
                        if count >= 10:
                            break
            except KeyboardInterrupt:
                logger.info("Auto loop interrupted")
        else:
            self.domain_selector._device_info = {"gpu": self.device.is_cuda, "vram_gb": self.device.vram_gb}
            dk, dc = self.domain_selector.auto_select()
            logger.info(f"Auto-selected: {dc['name']}")
            result = self.run_domain(dk, dc)
            if not result.get("error"):
                completed.append(dk)

        self._state_transition("idle")
        self._running = False
        return completed

    def check_budget(self) -> bool:
        """Return False if over budget (training should halt)."""
        global _BUDGET_HALT
        if _BUDGET_HALT:
            return False
        if self._cost_tracker and self._cost_tracker.is_over_budget():
            _BUDGET_HALT = True
            msg = f"Training halted: budget ${self._cost_tracker.get_session_cost():.2f} exceeds limit"
            logger.warning(msg)
            self.memory.write_agent_state("budget_halt", {
                "cost": self._cost_tracker.get_session_cost(),
                "limit": self._cost_tracker.budget_limit,
                "halted_at": datetime.now().isoformat(),
            })
            self._run_hooks("on_error", tool="budget", error=msg, step=-1, params={})
            return False
        if self._cost_tracker:
            remaining = self._cost_tracker.get_budget_remaining()
            if remaining < self._cost_tracker.budget_limit * 0.1:
                logger.info(f"Budget warning: ${remaining:.2f} remaining")
        return True

    def _log_audit(self, action: str, details: dict = None):
        try:
            self.memory.write_audit_log(action, details)
        except Exception:
            pass

    # ── Scheduler  ───────────────────────────────────────────────────────

    def schedule_training(self, domain: str, interval_hours: float = 24.0):
        """Schedule recurring training for a domain."""
        self._schedule.append({
            "domain": domain, "interval_hours": interval_hours,
            "last_run": "", "enabled": True,
        })
        if not self._scheduler_running:
            self._start_scheduler()
        self._log_audit("schedule_added", {"domain": domain, "interval_hours": interval_hours})
        return {"success": True, "domain": domain, "interval_hours": interval_hours}

    def list_schedules(self) -> list:
        return list(self._schedule)

    def remove_schedule(self, domain: str) -> bool:
        before = len(self._schedule)
        self._schedule[:] = [s for s in self._schedule if s["domain"] != domain]
        return len(self._schedule) < before

    def _start_scheduler(self):
        """Start background thread that checks and executes scheduled training."""
        self._scheduler_running = True

        def _loop():
            while self._scheduler_running:
                try:
                    now = datetime.now()
                    for sched in self._schedule:
                        if not sched["enabled"]:
                            continue
                        last = sched.get("last_run", "")
                        if not last:
                            sched["last_run"] = now.isoformat()
                            continue
                        try:
                            last_dt = datetime.fromisoformat(last)
                        except Exception:
                            last_dt = now
                        elapsed = (now - last_dt).total_seconds() / 3600
                        if elapsed >= sched["interval_hours"]:
                            domain_key = sched["domain"]
                            domain_config = DOMAINS.get(domain_key)
                            if domain_config:
                                logger.info(f"Scheduled training: {domain_key}")
                                self.run_domain(domain_key, domain_config)
                                sched["last_run"] = datetime.now().isoformat()
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                time.sleep(60)  # Check every 60s

        self._scheduler_thread = threading.Thread(target=_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("Scheduler started with %d schedules", len(self._schedule))

    def shutdown(self):
        self._scheduler_running = False
        self._log_audit("shutdown", {"status": self.state.status, "domain": self.state.domain, "tools": self.state.tools_loaded})
        self._plugin_manager.run_hook("on_shutdown")
        self.heartbeat.stop()
        self.session_manager.shutdown()
        self._state_transition("stopped")
        self._running = False
        logger.info("Gateway shutdown complete")
