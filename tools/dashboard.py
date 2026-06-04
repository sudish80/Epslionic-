"""Dashboard: Gradio web UI for Epslionic-Colab agent.
Launches a multi-tab dashboard in Colab with shareable public URL.

Tabs: Status | Experiments | Datasets | Errors | Heartbeat | Control
"""

import os, json, time, threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.domain import DOMAINS

try:
    import gradio as gr
except ImportError:
    gr = None


def _fmt_time(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts[:19]


def _safe_json(d, indent=2):
    return json.dumps(d, indent=indent, default=str)


# ---- Data loaders (read from MemoryStore files) ----

def load_status(memory) -> str:
    lines = []
    exp_dir = memory.exp_dir
    err_dir = memory.err_dir
    heart_dir = memory.heart_dir
    total_exps = len(list(exp_dir.glob("*.json")))
    running = sum(1 for f in exp_dir.glob("*.json") if json.loads(f.read_text()).get("status") == "running")
    completed = sum(1 for f in exp_dir.glob("*.json") if json.loads(f.read_text()).get("status") == "completed")
    failed = sum(1 for f in exp_dir.glob("*.json") if json.loads(f.read_text()).get("status") == "failed")
    unfixed = sum(1 for f in err_dir.glob("*.json") if not json.loads(f.read_text()).get("fixed"))
    total_errors = len(list(err_dir.glob("*.json")))
    hb_files = list(heart_dir.glob("*.jsonl"))
    hb_lines = 0
    for hf in hb_files:
        try:
            hb_lines += len(hf.read_text().strip().split("\n"))
        except Exception:
            pass
    lines.append("## Gateway Status")
    lines.append("")
    lines.append(f"- **Workspace:** `{memory.base}`")
    lines.append(f"- **Experiments:** {total_exps} total ({running} running, {completed} done, {failed} failed)")
    lines.append(f"- **Errors:** {total_errors} total ({unfixed} unresolved)")
    lines.append(f"- **Heartbeat ticks:** {hb_lines}")
    lines.append("")
    lines.append("---")
    lines.append(f"_Last refreshed: {datetime.now():%H:%M:%S}_")
    return "\n".join(lines)


def load_experiments(memory) -> str:
    exps = memory.list_experiments()
    if not exps:
        return "No experiments yet. Launch training from the **Control** tab."
    lines = ["## Training Experiments", "", "| # | Name | Status | Created | Metrics |", "|---|---|---|---|---|"]
    for i, e in enumerate(exps[:20], 1):
        name = e.get("name", "?")
        status = e.get("status", "?")
        created = _fmt_time(e.get("created_at", ""))
        metrics = e.get("metrics", {})
        metric_str = "; ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        if not metric_str:
            metric_str = "—"
        lines.append(f"| {i} | {name} | {status} | {created} | {metric_str} |")
    lines.append("")
    lines.append(f"_Showing {min(len(exps), 20)} of {len(exps)} experiments_")
    # Details for latest
    latest = exps[0]
    lines.append("")
    lines.append("### Latest Experiment Details")
    lines.append("")
    lines.append(f"- **ID:** `{latest.get('id', '?')}`")
    lines.append(f"- **Name:** {latest.get('name', '?')}")
    lines.append(f"- **Status:** {latest.get('status', '?')}")
    lines.append(f"- **Created:** {_fmt_time(latest.get('created_at', ''))}")
    if latest.get("config"):
        lines.append(f"- **Config:** {_safe_json(latest['config'])}")
    if latest.get("errors"):
        lines.append(f"- **Errors ({len(latest['errors'])}):** {'; '.join(str(e)[:100] for e in latest['errors'])}")
    lines.append("")
    lines.append(f"_Last refreshed: {datetime.now():%H:%M:%S}_")
    return "\n".join(lines)


def load_datasets(memory) -> str:
    ds_dir = memory.ds_dir
    files = sorted(ds_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
    if not files:
        return "No datasets recorded yet. Discover datasets, then check here."
    lines = ["## Discovered Datasets", "", "| # | Dataset | Samples | Features | Loaded |", "|---|---|---|---|---|"]
    for i, f in enumerate(files[:20], 1):
        try:
            d = json.loads(f.read_text())
            did = d.get("dataset_id", f.stem)
            samples = d.get("num_samples", "?")
            features = ", ".join(d.get("features", []))[:60]
            loaded = "\u2705" if d.get("loaded") else "\u274c"
            lines.append(f"| {i} | {did} | {samples} | {features} | {loaded} |")
        except Exception:
            pass
    lines.append("")
    lines.append(f"_Showing {min(len(files), 20)} datasets_")
    lines.append("")
    lines.append(f"_Last refreshed: {datetime.now():%H:%M:%S}_")
    return "\n".join(lines)


def load_errors(memory) -> str:
    unfixed = memory.get_unfixed()
    err_dir = memory.err_dir
    all_errs = []
    for f in sorted(err_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            all_errs.append(json.loads(f.read_text()))
        except Exception:
            pass
    if not all_errs:
        return "No errors logged. All clear!"
    lines = ["## Error Log", "", "| # | Source | Error | Fixed | Time |", "|---|---|---|---|---|"]
    for i, e in enumerate(all_errs[:20], 1):
        src = e.get("source", "?")
        err = str(e.get("error", "?"))[:60]
        fixed = "\u2705" if e.get("fixed") else "\u26a0\ufe0f"
        ts = _fmt_time(e.get("timestamp", ""))
        lines.append(f"| {i} | {src} | {err} | {fixed} | {ts} |")
    lines.append("")
    lines.append(f"- **Unresolved:** {len(unfixed)}")
    lines.append(f"- **Total logged:** {len(all_errs)}")
    lines.append(f"- **Fix rate:** {(len(all_errs) - len(unfixed))}/{len(all_errs)} ({((len(all_errs)-len(unfixed))/len(all_errs)*100):.0f}%)")
    if unfixed:
        lines.append("")
        lines.append("### Unresolved Errors (requires attention)")
        for e in unfixed[:5]:
            lines.append(f"- `{e.get('source', '?')}`: {str(e.get('error', '?'))[:100]}")
    lines.append("")
    lines.append(f"_Last refreshed: {datetime.now():%H:%M:%S}_")
    return "\n".join(lines)


def load_heartbeat(memory) -> str:
    heart_dir = memory.heart_dir
    files = sorted(heart_dir.glob("*.jsonl"), reverse=True)
    if not files:
        return "No heartbeat data yet. Heartbeat runs every 60s."
    hb_file = files[0]
    try:
        lines_text = hb_file.read_text().strip().split("\n")
        beats = [json.loads(l) for l in lines_text if l.strip()]
    except Exception:
        return "Could not parse heartbeat log."
    lines = ["## Heartbeat Timeline", "", "| Timestamp | Status | Details |", "|---|---|---|"]
    ok_count = sum(1 for b in beats if b.get("status") == "HEARTBEAT_OK")
    issue_count = sum(1 for b in beats if b.get("status") != "HEARTBEAT_OK")
    for b in beats[-30:]:
        ts = _fmt_time(b.get("timestamp", ""))
        status = b.get("status", "?")
        details = str(b.get("details", ""))[:80]
        lines.append(f"| {ts} | {status} | {details} |")
    lines.insert(2, f"_Healthy ticks: {ok_count} | Issues: {issue_count}_")
    lines.append("")
    lines.append(f"_Last refreshed: {datetime.now():%H:%M:%S}_")
    return "\n".join(lines)


# ---- Control actions ----

def run_action(memory, gateway, action: str, params: str) -> str:
    """Run a control action from the dashboard."""
    if action == "refresh":
        return "Refreshing all tabs..."
    if action == "status":
        return load_status(memory)
    if action == "domain" and gateway:
        try:
            # domain params expected as JSON: {"domain_key": "math", ...}
            p = json.loads(params) if params else {}
            dk = p.get("domain_key", "")
            if dk and dk in DOMAINS:
                dc = DOMAINS[dk]
                gateway.state.domain = dk
                gateway.state.domain_config = dc
                t = threading.Thread(target=gateway.run_domain, args=(dk, dc), daemon=True)
                t.start()
                return f"Launched {dk} training pipeline in background thread."
            else:
                domains = list(DOMAINS.keys())
                return f"Unknown domain '{dk}'. Choose from: {', '.join(domains)}"
        except Exception as e:
            return f"Domain action failed: {e}"
    if action == "experiment":
        results = f"Action: {action}\nParams: {params}"
        return results
    return f"Unknown action: {action}"


# ---- Build Gradio UI ----

def make_ui(memory, gateway=None, queue=None):
    """Construct the Gradio Blocks UI."""
    with gr.Blocks(
        title="Epslionic-Colab Dashboard",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
        css="""
        .refresh-btn { min-width: 140px; }
        footer { display: none !important; }
        """,
    ) as ui:
        gr.Markdown(
            "# \U0001f9e0 Epslionic-Colab Dashboard",
        )
        gr.Markdown(
            "Overview of training experiments, datasets, errors, and heartbeat. "
            "Click a refresh button on any tab to update."
        )

        with gr.Tabs() as tabs:
            # ---- Tab 1: Status ----
            with gr.Tab("\U0001f4ca Status"):
                status_out = gr.Markdown(load_status(memory))
                status_refresh = gr.Button("\U0001f504 Refresh Status", variant="secondary", elem_classes="refresh-btn")
                status_refresh.click(fn=lambda: load_status(memory), outputs=status_out)

            # ---- Tab 2: Experiments ----
            with gr.Tab("\U0001f4c8 Experiments"):
                exp_out = gr.Markdown(load_experiments(memory))
                exp_refresh = gr.Button("\U0001f504 Refresh Experiments", variant="secondary", elem_classes="refresh-btn")
                exp_refresh.click(fn=lambda: load_experiments(memory), outputs=exp_out)

            # ---- Tab 3: Datasets ----
            with gr.Tab("\U0001f4e6 Datasets"):
                ds_out = gr.Markdown(load_datasets(memory))
                ds_refresh = gr.Button("\U0001f504 Refresh Datasets", variant="secondary", elem_classes="refresh-btn")
                ds_refresh.click(fn=lambda: load_datasets(memory), outputs=ds_out)

            # ---- Tab 4: Errors ----
            with gr.Tab("\u26a0\ufe0f Errors"):
                err_out = gr.Markdown(load_errors(memory))
                err_refresh = gr.Button("\U0001f504 Refresh Errors", variant="secondary", elem_classes="refresh-btn")
                err_refresh.click(fn=lambda: load_errors(memory), outputs=err_out)

            # ---- Tab 5: Heartbeat ----
            with gr.Tab("\U0001f493 Heartbeat"):
                hb_out = gr.Markdown(load_heartbeat(memory))
                hb_refresh = gr.Button("\U0001f504 Refresh Heartbeat", variant="secondary", elem_classes="refresh-btn")
                hb_refresh.click(fn=lambda: load_heartbeat(memory), outputs=hb_out)

            # ---- Tab 6: Control ----
            with gr.Tab("\U0001f3f7\ufe0f Control"):
                gr.Markdown(
                    "## Control Panel\n\n"
                    "Launch training or run commands from the dashboard."
                )

                with gr.Row():
                    domain_dd = gr.Dropdown(
                        choices=["math","code","medical","legal","creative","science","finance","chat","general","custom"],
                        value="math", label="Domain", info="Select a domain to train on"
                    )
                    launch_btn = gr.Button("\U0001f680 Launch Domain Training", variant="primary", scale=2)

                control_out = gr.Markdown("Ready to launch training.")

                launch_btn.click(
                    fn=lambda d: run_action(memory, gateway, "domain", json.dumps({"domain_key": d})),
                    inputs=domain_dd, outputs=control_out
                )

                gr.Markdown("---")
                gr.Markdown("### Quick Status")

                with gr.Row():
                    quick_status = gr.Button("\U0001f4ca Refresh All Stats")
                    quick_errors = gr.Button("\u26a0\ufe0f Check Errors")

                quick_out = gr.Markdown("_Click a button above._")

                quick_status.click(fn=lambda: load_status(memory), outputs=quick_out)
                quick_errors.click(fn=lambda: load_errors(memory), outputs=quick_out)

                gr.Markdown("---")
                gr.Markdown("### Run Custom Command")
                cmd_text = gr.Textbox(label="Command", placeholder="e.g., discovery datasets for math reasoning")
                cmd_btn = gr.Button("Execute", variant="secondary")
                cmd_out = gr.Markdown("")
                if queue is not None:
                    cmd_btn.click(
                        fn=lambda c: queue.put(c) or f"Sent to queue: {c}",
                        inputs=cmd_text, outputs=cmd_out
                    )
                else:
                    cmd_btn.click(fn=lambda: "Queue not available (run with gateway)", outputs=cmd_out)

        gr.Markdown(
            "---\n"
            f"Epslionic-Colab Dashboard | "
            f"Updated: {datetime.now():%H:%M:%S}"
        )

    return ui


def serve(memory, gateway=None, share=True, port=7860, queue=None):
    """Launch the Gradio dashboard. Returns the Gradio app."""
    if gr is None:
        print("ERROR: gradio not installed. Run: pip install -q gradio")
        return None
    ui = make_ui(memory, gateway, queue)
    print(f"\n  {'='*50}")
    print(f"  Epslionic-Colab Dashboard")
    print(f"  {'='*50}")
    print(f"  Launching dashboard...")
    ui.queue(default_concurrency_limit=5)
    ui.launch(share=share, server_port=port, server_name="0.0.0.0", quiet=True)
    return ui
