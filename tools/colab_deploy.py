"""Colab Deploy — One-click Colab notebook generation with public tunnel."""

import json
import logging
import textwrap
from pathlib import Path
from typing import Optional

logger = logging.getLogger("epsionic.colab_deploy")

_COLAB_NOTEBOOK_TEMPLATE = r'''# @title Epslionic Agent — One-Click Deploy
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install Epslionic
!pip install -q epsionic[serve]

# Install tunnel
!pip install -q pyngrok

# Set up auth
import os
os.environ["OPENAI_API_KEY"] = "{api_key}"
os.environ["HF_TOKEN"] = "{hf_token}"
os.environ["EPSIONIC_API_KEY"] = "{server_api_key}"

# Start agent in background with REST API
import threading, time, logging
logging.basicConfig(level=logging.INFO)

from epsionic.config import AgentConfig
from epsionic.core.gateway import Gateway

cfg = AgentConfig()
cfg.workspace_root = "/content/workspace"
gw = Gateway(cfg)
gw.start_heartbeat()

def run_api():
    from epsionic.cli import cmd_serve
    cmd_serve(gw, host="0.0.0.0", port=8000)

threading.Thread(target=run_api, daemon=True).start()
time.sleep(3)

# Start ngrok tunnel
from pyngrok import ngrok
public_url = ngrok.connect(8000, bind_tls=True).public_url
print(f"\n🚀 Epslionic API: {{public_url}}")
print(f"📘 Swagger Docs: {{public_url}}/docs")
print(f"📗 ReDoc:        {{public_url}}/redoc")
print(f"🔑 API Key:      {{os.environ['EPSIONIC_API_KEY']}}")

# Keep alive
from IPython.display import display, HTML
display(HTML(f"""
<h3>Epslionic Agent Running</h3>
<p><a href="{{public_url}}" target="_blank">Open API</a></p>
<p><a href="{{public_url}}/docs" target="_blank">Swagger Docs</a></p>
"""))

# Keep notebook alive
import time
while True:
    time.sleep(60)
    print(".", end="")
'''


def generate_colab_notebook(
    api_key: str = "",
    hf_token: str = "",
    server_api_key: str = "epsionic-local-key",
    output_path: Optional[str] = None,
) -> str:
    """Generate a Colab notebook Python script with embedded deployment logic."""
    notebook = _COLAB_NOTEBOOK_TEMPLATE.format(
        api_key=api_key,
        hf_token=hf_token,
        server_api_key=server_api_key,
    )
    if output_path:
        Path(output_path).write_text(notebook, encoding="utf-8")
        logger.info("Colab deploy script written to %s", output_path)
    return notebook


def get_tunnel_url(port: int = 8000) -> Optional[str]:
    """Get the public ngrok tunnel URL for the given local port."""
    try:
        from pyngrok import ngrok
        tunnels = ngrok.get_tunnels()
        for t in tunnels:
            if f":{port}" in t.config.get("addr", ""):
                return t.public_url
        return None
    except ImportError:
        logger.warning("pyngrok not installed. Install with: pip install pyngrok")
        return None
    except Exception as e:
        logger.warning("Tunnel error: %s", e)
        return None


def get_tool_description() -> dict:
    return {
        "name": "colab_deploy",
        "description": "Generate one-click Colab deployment notebook with public ngrok tunnel",
        "parameters": {
            "api_key": {"type": "string", "optional": True, "description": "OpenAI API key"},
            "hf_token": {"type": "string", "optional": True, "description": "HuggingFace token"},
            "server_api_key": {"type": "string", "optional": True, "description": "API key for server auth"},
            "output_path": {"type": "string", "optional": True, "description": "Where to write the notebook script"},
        },
    }
