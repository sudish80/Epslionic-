#!/usr/bin/env python3
"""
OpenClaw-Colab Auto-Deployer
=============================
Automatically generates a Colab notebook from the agent code
and deploys it via GitHub for one-click Colab loading.

Usage:
  python deploy_to_colab.py                   # Generate notebook + GitHub setup
  python deploy_to_colab.py --gist            # Create a GitHub Gist
  python deploy_to_colab.py --no-github       # Just generate the .ipynb file
"""

import os
import sys
import json
import base64
import subprocess
import argparse
import webbrowser
from pathlib import Path
from datetime import datetime

COLAB_NOTEBOOK_PATH = Path("OpenClaw_Colab_Agent.ipynb")


def read_agent_code():
    """Read the complete agent file (thin wrapper)."""
    return '''#!/usr/bin/env python3
"""OpenClaw-Colab: LLM Training Agent - Thin wrapper for Colab."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from openclaw_colab_agent.cli import main
if __name__ == "__main__":
    sys.exit(main())
'''


def generate_notebook(agent_code):
    """Generate a complete .ipynb file with one cell containing all agent code."""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"id": "title"},
                "source": [
                    "# OpenClaw-Colab: LLM Training Agent\\n",
                    "\\n",
                    "OpenClaw architecture: **Gateway -> Sessions -> Memory -> Heartbeat -> Brain -> Tools**\\n",
                    "\\n",
                    "Capabilities:\\n",
                    "- **LLM Training** via Unsloth/QLoRA (optimized for Colab T4)\\n",
                    "- **Auto-Fix Errors** with rule-based + LLM-powered recovery\\n",
                    "- **Dataset Discovery** from Hugging Face\\n",
                    "- **LLM Brain** for reasoning and planning\\n",
                    "\\n",
                    "*(Inspired by [github.com/openclaw/openclaw](https://github.com/openclaw/openclaw))*\\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "arch"},
                "source": [
                    "---\\n",
                    "## One-Click Setup\\n",
                    "\\n",
                    "All cells below contain the complete agent. Just **Runtime -> Run all** (or run each cell in order).\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {
                    "id": "setup",
                    "colab": {"base_uri": "https://localhost:8080/"},
                },
                "outputs": [],
                "source": [
                    "# ============================================================\n",
                    "# CELL 1: Auto-install dependencies + SET API KEYS\n",
                    "# ============================================================\n",
                    "import sys, subprocess, importlib, os\n",
                    "\n",
                    "# SET YOUR API KEYS HERE or use Colab secrets (key icon on left)\n",
                    "os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')  # <-- PASTE YOUR KEY\n",
                    "os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')              # <-- PASTE YOUR HF TOKEN\n",
                    "\n",
                    "packages = [\n",
                    "    'transformers>=4.36.0', 'datasets>=2.14.0', 'accelerate>=0.24.0',\n",
                    "    'peft>=0.6.0', 'trl>=0.7.0', 'bitsandbytes>=0.41.0',\n",
                    "    'scipy', 'sentencepiece', 'huggingface_hub', 'openai',\n",
                    "]\n",
                    "for pkg in packages:\n",
                    "    name = pkg.split('>=')[0].split('==')[0].replace('-', '_')\n",
                    "    try:\n",
                    "        importlib.import_module(name)\n",
                    "        print(f'  OK {pkg}')\n",
                    "    except ImportError:\n",
                    "        print(f'  Installing {pkg}...')\n",
                    "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg], timeout=120)\n",
                    "try:\n",
                    "    import unsloth; print('  OK unsloth')\n",
                    "except ImportError:\n",
                    "    print('  Installing unsloth...')\n",
                    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'unsloth'], timeout=180)\n",
                    "print('\\nAll dependencies ready!')\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "agent_code_header"},
                "source": [
                    "---\\n",
                    "## Agent Source Code\\n",
                    "\\n",
                    "Run this cell to load the complete OpenClaw-inspired agent (all classes).\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "agent_code"},
                "outputs": [],
                "source": [
                    "# ============================================================\n",
                    "# CELL 2: Complete Agent Code\n",
                    "# ============================================================\n",
                    f"\n{agent_code}\n",
                    "\n",
                    "print('\\nAgent code loaded!')\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {"id": "domain_interview_header"},
                "source": [
                    "---\\n",
                    "## \\U0001f9e0  Domain Interview\\n",
                    "\\n",
                    "**The agent will now ask you what domain you want to train on.**\\n",
                    "\\n",
                    "It will then auto-discover datasets, configure training, and execute.\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "domain_interview"},
                "outputs": [],
                "source": [
                    "# ============================================================\n",
                    "# CELL 3: THE AGENT ASKS: \"WHAT DOMAIN DO YOU WANT TO TRAIN?\"\n",
                    "# ============================================================\n",
                    "# Run this cell. The agent will show a menu. You pick a domain.\n",
                    "# It then runs the full pipeline: discover -> select -> configure -> train\n",
                    "\n",
                    "from openclaw_agent_complete import ask_domain, domain_to_objective, DOMAINS\n",
                    "from openclaw_agent_complete import Config, Gateway, DatasetTool, TrainerTool, FixerTool\n",
                    "\n",
                    "cfg = Config()\n",
                    "cfg.openai_api_key = os.environ.get('OPENAI_API_KEY', '')\n",
                    "cfg.hf_token = os.environ.get('HF_TOKEN', '')\n",
                    "\n",
                    "gateway = Gateway(cfg)\n",
                    "ds_tool = DatasetTool(gateway.memory, cfg.datasets_dir, cfg.hf_token)\n",
                    "tr_tool = TrainerTool(gateway.memory, cfg.models_dir)\n",
                    "fx_tool = FixerTool(gateway.memory)\n",
                    "\n",
                    "gateway.tool('discover', 'Search HuggingFace datasets',\n",
                    "    lambda **kw: ds_tool.search(kw.get('query',''))\n",
                    "    if kw.get('query') else ds_tool.load(kw.get('dataset_id','')))\n",
                    "gateway.tool('train', 'Fine-tune LLM',\n",
                    "    lambda **kw: tr_tool.train(kw.get('experiment_id','exp'),\n",
                    "        kw.get('model_name',cfg.train_config['model_name']),\n",
                    "        kw.get('dataset_dict',{}), kw.get('training_args',cfg.train_config)))\n",
                    "gateway.tool('prepare', 'Prepare training',\n",
                    "    lambda **kw: tr_tool.prepare(kw.get('experiment_id','exp'),\n",
                    "        kw.get('model_name',cfg.train_config['model_name'])))\n",
                    "gateway.tool('auto_fix', 'Analyze & fix errors',\n",
                    "    lambda **kw: fx_tool.fix(kw.get('error',''), kw.get('context',{})))\n",
                    "\n",
                    "gateway.load_skills()\n",
                    "gateway.start_heartbeat()\n",
                    "\n",
                    "# THE AGENT ASKS: which domain?\n",
                    "domain_key, domain_config = ask_domain()\n",
                    "gateway.domain_key = domain_key\n",
                    "gateway.domain_config = domain_config\n",
                    "\n",
                    "# Auto-run the domain pipeline\n",
                    "print(f'\\n{\"=\"*50}')\n",
                    "print(f'  Starting: {domain_config[\"emoji\"]} {domain_config[\"name\"]} Pipeline')\n",
                    "print(f'{\"=\"*50}')\n",
                    "result = gateway.run_domain(domain_key, domain_config)\n",
                    "\n",
                    "print(f'\\nPipeline complete: {len(result[\"steps\"])} steps')\n",
                    "for s in result['steps']:\n",
                    "    icon = '\\u2705' if s['status']=='completed' else '\\u26a0\\ufe0f'\n",
                    "    print(f'  {icon} Step {s[\"step\"]}: {s[\"tool\"]}')\n",
                    "if result.get('error'):\n",
                    "    print(f'  Error: {result[\"error\"][:100]}')\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "dashboard"},
                "outputs": [],
                "source": [
                    "# ============================================================\n",
                    "# CELL 4: [OPTIONAL] Launch Web Dashboard\n",
                    "# ============================================================\n",
                    "# Run this cell after setting up gateway (Cell 3) to launch\n",
                    "# a Gradio dashboard with public shareable URL.\n",
                    "# Tabs: Status | Experiments | Datasets | Errors | Heartbeat | Control\n",
                    "\n",
                    "try:\n",
                    "    import gradio\n",
                    "    from tools.dashboard import serve as dashboard_serve\n",
                    "    print('Launching dashboard (this will block until you close it)...')\n",
                    "    dashboard_serve(gateway.memory, gateway, share=True)\n",
                    "except ImportError:\n",
                    "    import subprocess, sys\n",
                    "    print('Installing gradio...')\n",
                    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'gradio'])\n",
                    "    from tools.dashboard import serve as dashboard_serve\n",
                    "    print('Launching dashboard...')\n",
                    "    dashboard_serve(gateway.memory, gateway, share=True)\n",
                    "except Exception as e:\n",
                    "    print(f'Dashboard error: {e}')\n",
                    "    print('Access your agent via command-line instead.')\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "demo_discovery"},
                "outputs": [],
                "source": [
                    "# ============================================================\n",
                    "# DEMO: Dataset Discovery\n",
                    "# ============================================================\n",
                    "print('Searching HuggingFace datasets...')\n",
                    "results = ds_tool.search(query='math reasoning', max_r=5)\n",
                    "for i, ds in enumerate(results, 1):\n",
                    "    print(f'  {i}. {ds[\"id\"]}')\n",
                    "    print(f'     {ds[\"description\"][:100]}')\n",
                    "    print(f'     Downloads: {ds[\"downloads\"]}')\n",
                    "print(f'\\\\nFound {len(results)} datasets')\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "demo_fixer"},
                "outputs": [],
                "source": [
                    "# ============================================================\n",
                    "# DEMO: Auto-Fix\n",
                    "# ============================================================\n",
                    "errors = [\n",
                    "    'CUDA out of memory. Tried to allocate 2.00 GiB.',\n",
                    "    \"No module named 'bitsandbytes'\",\n",
                    "    'RuntimeError: expected scalar type Half but found Float',\n",
                    "    'Loss is NaN at step 47',\n",
                    "]\n",
                    "for err in errors:\n",
                    "    r = fx_tool.fix(err, {'batch_size': 4, 'learning_rate': 2e-4})\n",
                    "    print(f'\\\\n  Error: {err[:50]}...')\n",
                    "    print(f'  Fix: {r[\"desc\"]}')\n",
                    "    print(f'  Conf: {r[\"conf\"]:.0%}')\n",
                    "    if r.get('adjusted'): print(f'  Adjusted: {r[\"adjusted\"]}')\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "demo_agent"},
                "outputs": [],
                "source": [
                    "# ============================================================\n",
                    "# DEMO: Full Agent Loop (LLM Brain decides actions)\n",
                    "# ============================================================\n",
                    "result = gateway.run(\n",
                    "    'Find math datasets and prepare training environment',\n",
                    "    max_steps=5\n",
                    ")\n",
                    "print(f'Steps: {len(result[\"steps\"])}')\n",
                    "for s in result['steps']:\n",
                    "    icon = 'OK' if s['status']=='completed' else 'FAIL'\n",
                    "    print(f'  [{icon}] {s[\"tool\"]}: {s[\"reasoning\"][:80]}')\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "interactive"},
                "outputs": [],
                "source": [
                    "# ============================================================\n",
                    "# INTERACTIVE MODE\n",
                    "# ============================================================\n",
                    "print('OpenClaw-Colab Interactive Agent')\n",
                    "print('Commands: <objective>, status, quit')\n",
                    "while True:\n",
                    "    try:\n",
                    "        inp = input('\\\\n>> ').strip()\n",
                    "        if not inp: continue\n",
                    "        if inp == 'quit': break\n",
                    "        if inp == 'status': print(gateway.status_md()); continue\n",
                    "        r = gateway.run(inp); print(f'Done: {len(r[\"steps\"])} steps')\n",
                    "    except KeyboardInterrupt: break\n",
                    "    except EOFError: break\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "training"},
                "outputs": [],
                "source": [
                    "# ============================================================\n",
                    "[OPTIONAL] LLM Fine-Tuning\n",
                    "# Uncomment to run actual training on T4 GPU\n",
                    "# ============================================================\n",
                    "# exp_id = gateway.memory.create_experiment('math_finetune', cfg.train_config)\n",
                    "# ds = ds_tool.load('gsm8k', split='train', max_s=50)\n",
                    "# if ds.get('loaded'):\n",
                    "#     print(f'Dataset: {ds[\"dataset_id\"]} ({ds[\"num_samples\"]} samples)')\n",
                    "#     result = tr_tool.train(exp_id, 'unsloth/mistral-7b-bnb-4bit',\n",
                    "#         dataset_dict=ds, training_args=cfg.train_config)\n",
                    "#     print(f'Training: {result.get(\"status\", \"done\")}')\n",
                    "#     print(f'Model: {result.get(\"model_path\", \"N/A\")}')\n",
                    "# else:\n",
                    "#     print('Dataset load failed. Check HF token or internet.')\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "status"},
                "outputs": [],
                "source": [
                    "# ============================================================\n",
                    "# Status Report\n",
                    "# ============================================================\n",
                    "print(gateway.status_md())\n",
                    "print(f'\\\\nWorkspace: {cfg.workspace_root}')\n",
                    "print('Save to Drive: !cp -r /content/openclaw_workspace /content/drive/MyDrive/')\n",
                ],
            },
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4",
                "toc_visible": True,
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    with open(COLAB_NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"  Generated: {COLAB_NOTEBOOK_PATH}")
    print(f"  Cells: {len([c for c in notebook['cells'] if c['cell_type']=='code'])} code cells")
    # Update cell 3 to use the modular package
    notebook["cells"][6]["source"] = [
        "# ============================================================\n",
        "# CELL 3: Launch the agent using the modular package\n",
        "# ============================================================\n",
        "from openclaw_colab_agent import get_gateway, get_dashboard\n",
        "from openclaw_colab_agent.core.domain import DomainSelector, DOMAINS\n",
        "\n",
        "gateway = get_gateway(auto_install_deps=True)\n",
        "print('Gateway ready!')\n",
        "\n",
        "# Option 1: Autonomous mode\n",
        "# dk, dc = DomainSelector().auto_select()\n",
        "# result = gateway.run_with_domain(...)\n",
        "\n",
        "# Option 2: Domain interview (uncomment below)\n",
        "# dk, dc = DomainSelector().ask()\n",
        "# result = gateway.run_domain(dk, dc)\n",
    ]

    return COLAB_NOTEBOOK_PATH


def github_deploy():
    """Deploy to GitHub for one-click Colab access."""
    print("\n  Deploying to GitHub...")

    # Check if git is available
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  git not found. Install git or use --no-github")
        return None

    # Check if we're in a git repo
    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        print("  Initializing git repository...")
        subprocess.run(["git", "init"], check=True)

    # Check if remote exists
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        print()
        print("  No GitHub remote found.")
        print()
        print("  To deploy to GitHub:")
        print(f"    1. Create a repo at https://github.com/new")
        print(f"    2. Run these commands:")
        print(f'       git remote add origin https://github.com/YOUR_USER/openclaw-colab-agent.git')
        print(f'       git add -A')
        print(f'       git commit -m "Initial commit: OpenClaw-Colab agent"')
        print(f'       git branch -M main')
        print(f'       git push -u origin main')
        print()
        print(f"  Then in Colab, run ONE cell:")
        print(f"  ----------------------------------------")
        print(f"  !git clone https://github.com/YOUR_USER/openclaw-colab-agent.git")
        print(f"  %cd openclaw-colab-agent")
        print(f"  !python openclaw_agent_complete.py --interactive")
        print(f"  ----------------------------------------")
        return None

    # Git repo exists with remote
    print(f"  Remote: {result.stdout.strip()}")

    # Stage, commit, push
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Auto-deploy {datetime.now():%Y-%m-%d %H:%M}"],
        capture_output=True,
    )

    print("  Pushing to GitHub...")
    result = subprocess.run(["git", "push", "-u", "origin", "HEAD"], capture_output=True, text=True)
    if result.returncode == 0:
        print("  Push successful!")
        remote_url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True
        ).stdout.strip()

        # Convert to HTTPS if SSH
        if remote_url.startswith("git@"):
            parts = remote_url.replace(":", "/").replace("git@", "https://")
            remote_url = parts

        colab_url = f"https://colab.research.google.com/github{remote_url.replace('https://github.com', '')}/blob/main/{COLAB_NOTEBOOK_PATH.name}"

        print()
        print(f"  ONE-CLICK COLAB OPEN:")
        print(f"  {colab_url}")
        print()
        print(f"  Or run this in a new Colab notebook:")
        print(f"  ----------------------------------------")
        print(f"  !git clone {remote_url}")
        print(f"  %cd {os.path.basename(remote_url).replace('.git','')}")
        print(f"  !python openclaw_agent_complete.py --interactive")
        print(f"  ----------------------------------------")

        try:
            webbrowser.open(colab_url)
            print("  (Colab link opened in browser)")
        except Exception:
            pass

        return colab_url

    print(f"  Push failed: {result.stderr}")
    return None


def create_gist():
    """Create a GitHub Gist from the agent code."""
    code = read_agent_code()

    try:
        result = subprocess.run(
            ["gh", "gist", "create", "--public",
             AGENT_FILE.name,
             "-d", "OpenClaw-Colab: LLM Training Agent"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            gist_url = result.stdout.strip()
            raw_url = f"{gist_url}/raw/{AGENT_FILE.name}"

            print(f"\n  Gist created: {gist_url}")
            print()
            print(f"  In Colab, run ONE cell:")
            print(f"  ----------------------------------------")
            print(f"  !pip install -q openai && !wget -q {raw_url} -O openclaw_agent.py")
            print(f"  !python openclaw_agent.py --interactive")
            print(f"  ----------------------------------------")

            try:
                webbrowser.open(gist_url)
            except Exception:
                pass
            return gist_url
        else:
            print(f"  gh gist failed: {result.stderr}")
            print("  Install GitHub CLI: https://cli.github.com/")
            return None
    except FileNotFoundError:
        print("  GitHub CLI (gh) not found.")
        print("  Install from: https://cli.github.com/")
        return None


def main():
    parser = argparse.ArgumentParser(description="Deploy OpenClaw-Colab Agent")
    parser.add_argument("--gist", action="store_true", help="Create GitHub Gist")
    parser.add_argument("--no-github", action="store_true", help="Skip GitHub deploy")
    args = parser.parse_args()

    print("=" * 50)
    print("  OpenClaw-Colab Auto-Deployer")
    print("=" * 50)
    print()

    # Step 1: Read agent code
    print("Step 1: Reading agent code...")
    agent_code = read_agent_code()
    print(f"  Loaded: {AGENT_FILE.name} ({len(agent_code)} bytes)")

    # Step 2: Generate notebook
    print("\nStep 2: Generating Colab notebook...")
    notebook_path = generate_notebook(agent_code)
    notebook_size = notebook_path.stat().st_size
    print(f"  Size: {notebook_size/1024:.0f} KB")

    # Step 3: Deploy
    print("\nStep 3: Deploying...")

    if args.gist:
        create_gist()
    elif args.no_github:
        print("\n  Notebook generated. Upload to Colab manually:")
        print(f"    https://colab.research.google.com -> Upload -> {COLAB_NOTEBOOK_PATH.name}")
    else:
        github_deploy()

    print()
    print("=" * 50)
    print("  Deployment complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
