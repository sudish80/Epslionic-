"""Interactive Playground — Chat with trained models, compare versions.
Gradio-based chat UI with streaming, multi-turn, and A/B comparison."""

import json
import logging
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger("epsionic.tool.playground")


class ModelPlayground:
    """Interactive chat playground for trained models."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._model = None
        self._tokenizer = None
        self._model_path = None
        self._conversation_history: List[dict] = []
        self._comparison_models: Dict[str, tuple] = {}

    def load_model(self, model_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self._model_path = model_path
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._conversation_history = []
        logger.info(f"Playground loaded model: {model_path}")

    def generate(self, message: str, system_prompt: str = "",
                 temperature: float = 0.7, max_tokens: int = 512,
                 top_p: float = 0.9, stream: bool = False) -> str:
        if self._model is None:
            return "No model loaded. Call load_model() first."
        import torch
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self._conversation_history)
        messages.append({"role": "user", "content": message})

        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=temperature, top_p=top_p,
                do_sample=temperature > 0, pad_token_id=self._tokenizer.eos_token_id,
            )
        response = self._tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        self._conversation_history.append({"role": "user", "content": message})
        self._conversation_history.append({"role": "assistant", "content": response})
        return response

    def compare(self, message: str, model_path_a: str, model_path_b: str,
                temperature: float = 0.7) -> dict:
        """Compare responses from two models side by side."""
        original_model = self._model_path

        self.load_model(model_path_a)
        response_a = self.generate(message, temperature=temperature)

        self.load_model(model_path_b)
        response_b = self.generate(message, temperature=temperature)

        if original_model:
            self.load_model(original_model)

        return {
            "model_a": model_path_a, "response_a": response_a,
            "model_b": model_path_b, "response_b": response_b,
        }

    def reset_conversation(self):
        self._conversation_history = []

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def current_model(self) -> Optional[str]:
        return self._model_path

    def unload(self):
        import torch
        self._model = None
        self._tokenizer = None
        self._model_path = None
        self._conversation_history = []
        torch.cuda.empty_cache()

    def get_tool_description(self) -> dict:
        return {
            "load_model": {"description": "Load a model into the playground",
                "parameters": {"model_path": "Path or HF ID of model"}},
            "generate": {"description": "Generate a response to a message",
                "parameters": {"message": "User message", "temperature": "0.0-2.0", "max_tokens": "Max tokens"}},
            "compare": {"description": "Compare two models side by side",
                "parameters": {"message": "Test prompt", "model_path_a": "First model", "model_path_b": "Second model"}},
        }
