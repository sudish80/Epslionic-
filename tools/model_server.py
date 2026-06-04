"""Production Model Serving — vLLM / TGI / Ollama / OpenAI proxy.
Deploys trained models as OpenAI-compatible API endpoints with multi-provider support."""

import json
import logging
import os
import threading
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger("epsionic.tool.model_server")


class ModelServer:
    """Serve trained models via vLLM, TGI, Ollama, or proxy through OpenAI with unified API."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._process = None
        self._endpoint_url = None
        self._running = False
        self._provider = None

    def deploy(self, model_path: str, provider: str = "auto", port: int = 8000,
               gpu_memory_utilization: float = 0.9, max_model_len: int = 4096,
               api_base: str = None, api_key: str = None) -> dict:
        """Auto-select provider based on availability and user preference."""
        provider = provider.lower() if provider else "auto"
        if provider == "vllm":
            return self.deploy_vllm(model_path, port, gpu_memory_utilization, max_model_len)
        elif provider == "transformers":
            return self.deploy_transformers(model_path, port)
        elif provider == "tgi":
            return self.deploy_tgi(model_path, port)
        elif provider == "ollama":
            return self.deploy_ollama(model_path, port)
        elif provider == "openai":
            return self.deploy_openai_proxy(model_path, api_base=api_base, api_key=api_key)
        # Auto-detect
        for p in ["vllm", "tgi", "ollama"]:
            try:
                result = getattr(self, f"deploy_{p}")(model_path, port)
                if result.get("success"):
                    return result
            except Exception:
                continue
        return self.deploy_transformers(model_path, port)

    def deploy_vllm(self, model_path: str, port: int = 8000,
                    gpu_memory_utilization: float = 0.9,
                    max_model_len: int = 4096) -> dict:
        """Deploy model using vLLM's OpenAI-compatible server."""
        try:
            import subprocess
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_path,
                "--port", str(port),
                "--gpu-memory-utilization", str(gpu_memory_utilization),
                "--max-model-len", str(max_model_len),
                "--trust-remote-code",
                "--dtype", "float16",
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"

            self._process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self._endpoint_url = f"http://localhost:{port}/v1"
            self._running = True
            self._provider = "vllm"

            def _monitor():
                time.sleep(15)
                if self._process and self._process.poll() is not None:
                    stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                    logger.error(f"vLLM server exited: {stderr[:500]}")
                    self._running = False
                    self._process = None

            threading.Thread(target=_monitor, daemon=True).start()
            return {"success": True, "endpoint": self._endpoint_url, "provider": "vllm",
                    "model": model_path, "port": port}
        except Exception as e:
            logger.error(f"vLLM deploy failed: {e}")
            return {"success": False, "error": str(e), "provider": "vllm"}

    def deploy_tgi(self, model_path: str, port: int = 8000) -> dict:
        """Deploy using HuggingFace TGI (text-generation-inference)."""
        try:
            import subprocess
            cmd = [
                "text-generation-launcher",
                "--model-id", model_path,
                "--port", str(port),
                "--max-input-length", "2048",
                "--max-total-tokens", "4096",
                "--trust-remote-code",
            ]
            self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self._endpoint_url = f"http://localhost:{port}"
            self._running = True
            self._provider = "tgi"
            return {"success": True, "endpoint": self._endpoint_url, "provider": "tgi",
                    "model": model_path, "port": port}
        except Exception as e:
            logger.warning(f"TGI deploy failed (may not be installed): {e}")
            return {"success": False, "error": str(e), "provider": "tgi"}

    def deploy_ollama(self, model_path: str, port: int = 8000) -> dict:
        """Use Ollama to serve a model from its library or a local GGUF."""
        try:
            import subprocess, shutil
            ollama_path = shutil.which("ollama")
            if not ollama_path:
                return {"success": False, "error": "Ollama not found in PATH", "provider": "ollama"}
            # If it's a local file, create a Modelfile
            model_name = Path(model_path).stem if Path(model_path).exists() else model_path
            if Path(model_path).exists():
                modelfile = f"FROM {model_path}\n"
                modelfile_path = Path(model_path).parent / "Modelfile"
                modelfile_path.write_text(modelfile)
                subprocess.run([ollama_path, "create", model_name, "-f", str(modelfile_path)],
                               capture_output=True, timeout=300)
            # Run ollama serve in background if not running
            self._process = subprocess.Popen([ollama_path, "serve"],
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(3)
            self._endpoint_url = f"http://localhost:11434"
            self._running = True
            self._provider = "ollama"
            return {"success": True, "endpoint": self._endpoint_url, "provider": "ollama",
                    "model": model_name, "port": 11434}
        except Exception as e:
            logger.warning(f"Ollama deploy failed: {e}")
            return {"success": False, "error": str(e), "provider": "ollama"}

    def deploy_openai_proxy(self, model_path: str, api_base: str = None,
                            api_key: str = None) -> dict:
        """Proxy through an existing OpenAI-compatible endpoint."""
        self._endpoint_url = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self._running = True
        self._provider = "openai"
        self.config["openai_api_key"] = api_key or os.environ.get("OPENAI_API_KEY", "")
        return {"success": True, "endpoint": self._endpoint_url, "provider": "openai",
                "model": model_path}

    def deploy_transformers(self, model_path: str, port: int = 8001) -> dict:
        """Deploy using a lightweight FastAPI + Transformers server."""
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        app = FastAPI(title=f"Epslionic Model: {Path(model_path).name}")

        logger.info(f"Loading model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Model loaded successfully")

        class ChatRequest(BaseModel):
            messages: List[dict]
            max_tokens: int = 512
            temperature: float = 0.7
            top_p: float = 0.9
            stream: bool = False

        class ChatResponse(BaseModel):
            id: str = "chatcmpl-epsionic"
            object: str = "chat.completion"
            choices: List[dict] = None
            usage: dict = None

        @app.get("/health")
        def health():
            return {"status": "ok", "model": Path(model_path).name}

        @app.post("/v1/chat/completions")
        def chat_completions(req: ChatRequest):
            prompt = tokenizer.apply_chat_template(req.messages, tokenize=False)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=req.max_tokens,
                    temperature=req.temperature, top_p=req.top_p,
                    do_sample=True, pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return ChatResponse(
                choices=[{"index": 0, "message": {"role": "assistant", "content": response}}],
                usage={"prompt_tokens": inputs.input_ids.shape[1], "completion_tokens": len(outputs[0]) - inputs.input_ids.shape[1]},
            )

        @app.post("/v1/completions")
        def completions(prompt: str = "", max_tokens: int = 256, temperature: float = 0.7):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_tokens, temperature=temperature,
                    do_sample=True, pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return {"choices": [{"text": text}]}

        def _run():
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        endpoint = f"http://localhost:{port}"
        self._endpoint_url = endpoint
        self._running = True
        self._provider = "transformers"
        return {"success": True, "endpoint": endpoint, "provider": "transformers",
                "model": model_path, "port": port}

    def stop(self):
        if self._process:
            self._process.terminate()
            self._process = None
        self._running = False
        self._endpoint_url = None
        self._provider = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def endpoint(self) -> Optional[str]:
        return self._endpoint_url

    @property
    def provider(self) -> Optional[str]:
        return self._provider

    def get_tool_description(self) -> dict:
        return {
            "deploy": {"description": "Auto-deploy model with best available provider",
                "parameters": {"model_path": "Model path", "provider": "auto|vllm|tgi|ollama|openai|transformers",
                    "port": "Server port"}},
            "deploy_vllm": {"description": "Deploy model as OpenAI-compatible API via vLLM",
                "parameters": {"model_path": "Model path", "port": "Server port", "gpu_memory_utilization": "0.0-1.0"}},
            "deploy_transformers": {"description": "Deploy model as FastAPI + Transformers server",
                "parameters": {"model_path": "Model path", "port": "Server port"}},
            "deploy_tgi": {"description": "Deploy via HuggingFace TGI",
                "parameters": {"model_path": "Model path", "port": "Server port"}},
            "deploy_ollama": {"description": "Deploy via Ollama (local GGUF/models)",
                "parameters": {"model_path": "Model path", "port": "Server port"}},
            "deploy_openai_proxy": {"description": "Proxy through existing OpenAI-compatible endpoint",
                "parameters": {"model_path": "Model name", "api_base": "API base URL", "api_key": "API key"}},
            "stop": {"description": "Stop the running model server"},
        }
