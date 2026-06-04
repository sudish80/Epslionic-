"""Production Model Serving — vLLM / TGI integration.
Deploys trained models as OpenAI-compatible API endpoints."""

import json
import logging
import os
import threading
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger("openclaw.tool.model_server")


class ModelServer:
    """Serve trained models via vLLM with OpenAI-compatible API."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._process = None
        self._endpoint_url = None
        self._running = False

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

            def _monitor():
                time.sleep(15)
                if self._process and self._process.poll() is not None:
                    stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                    logger.error(f"vLLM server exited: {stderr[:500]}")
                    self._running = False
                    self._process = None

            threading.Thread(target=_monitor, daemon=True).start()
            return {"success": True, "endpoint": self._endpoint_url, "model": model_path, "port": port}
        except Exception as e:
            logger.error(f"vLLM deploy failed: {e}")
            return {"success": False, "error": str(e)}

    def deploy_transformers(self, model_path: str, port: int = 8001) -> dict:
        """Deploy using a lightweight FastAPI + Transformers server."""
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        app = FastAPI(title=f"OpenClaw Model: {Path(model_path).name}")

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
            id: str = "chatcmpl-openclaw"
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
        return {"success": True, "endpoint": endpoint, "model": model_path, "port": port}

    def stop(self):
        if self._process:
            self._process.terminate()
            self._process = None
        self._running = False
        self._endpoint_url = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def endpoint(self) -> Optional[str]:
        return self._endpoint_url

    def get_tool_description(self) -> dict:
        return {
            "deploy_vllm": {"description": "Deploy model as OpenAI-compatible API via vLLM",
                "parameters": {"model_path": "Model path", "port": "Server port", "gpu_memory_utilization": "0.0-1.0"}},
            "deploy_transformers": {"description": "Deploy model as FastAPI + Transformers server",
                "parameters": {"model_path": "Model path", "port": "Server port"}},
            "stop": {"description": "Stop the running model server"},
        }
