"""
LLM Brain - The reasoning engine of the agent.
Uses an LLM (OpenAI/Anthropic) to decide actions, analyze errors, plan training.
Inspired by OpenClaw's model-agnostic bring-your-own-keys approach.
"""

import json
import logging
import re
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("openclaw.brain")


@dataclass
class BrainAction:
    action: str
    tool: str
    params: dict = field(default_factory=dict)
    reasoning: str = ""


class LLMBrain:
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None, temperature: float = 0.3):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self._client = None
        self._tools_registry: Dict[str, dict] = {}
        self._conversation_history: List[dict] = []

    def register_tool(self, name: str, description: str,
                      handler: Callable, parameters: dict):
        self._tools_registry[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "parameters": parameters,
        }

    def get_tool_descriptions(self) -> str:
        lines = ["## Available Tools", ""]
        for name, info in self._tools_registry.items():
            lines.append(f"### {name}")
            lines.append(f"{info['description']}")
            lines.append(f"Parameters: {json.dumps(info['parameters'], indent=2)}")
            lines.append("")
        return "\n".join(lines)

    def think_and_act(self, objective: str, context: str = "") -> BrainAction:
        """Ask the LLM to reason about the current state and decide what tool to call."""
        prompt = self._build_prompt(objective, context)
        response = self._query_llm(prompt)
        action = self._parse_action(response)
        self._conversation_history.append({
            "role": "user", "content": objective
        })
        self._conversation_history.append({
            "role": "assistant", "content": response
        })
        return action

    def analyze_error(self, error_message: str, context: dict = None) -> dict:
        """Analyze a training error and suggest a fix."""
        prompt = f"""You are an ML training error analyst. Analyze this error and suggest a fix.

Error:
{error_message}

Context:
{json.dumps(context or {}, indent=2)}

Respond with JSON:
{{
    "root_cause": "brief description of root cause",
    "severity": "critical/warning/info",
    "suggested_fix": "what to change",
    "confidence": 0.0-1.0,
    "retry_with_params": {{ "param_to_change": "new_value" }}
}}"""
        response = self._query_llm(prompt)
        return self._parse_json_safely(response, {
            "root_cause": "Unknown",
            "severity": "warning",
            "suggested_fix": "No suggestion",
            "confidence": 0.0,
            "retry_with_params": {}
        })

    def discover_datasets(self, query: str) -> dict:
        """Plan a dataset discovery strategy based on a natural language query."""
        prompt = f"""Given this training objective, determine the best Hugging Face datasets to use.

Objective: {query}

Respond with JSON:
{{
    "search_queries": ["query1", "query2"],
    "filters": {{"task_categories": [], "languages": []}},
    "reasoning": "why these datasets would work"
}}"""
        response = self._query_llm(prompt)
        return self._parse_json_safely(response, {
            "search_queries": [query],
            "filters": {},
            "reasoning": "Direct search"
        })

    def plan_training(self, objective: str, dataset_info: dict,
                      hardware: str = "T4 (16GB VRAM)") -> dict:
        """Plan a full training configuration based on objective and dataset."""
        prompt = f"""You are an ML training planner. Design a training configuration.

Objective: {objective}
Hardware: {hardware}
Dataset: {json.dumps(dataset_info, indent=2)}

Respond with JSON:
{{
    "model_name": "model to fine-tune",
    "reasoning": "why this model",
    "lora_config": {{ "r": 16, "alpha": 16, "dropout": 0.0 }},
    "training_args": {{
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
        "optim": "adamw_8bit",
        "warmup_steps": 10
    }},
    "expected_vram_gb": 12.0,
    "warnings": ["potential issue"]
}}"""
        response = self._query_llm(prompt)
        return self._parse_json_safely(response, {
            "model_name": "unsloth/mistral-7b-bnb-4bit",
            "reasoning": "Default selection",
            "lora_config": {"r": 16, "alpha": 16, "dropout": 0.0},
            "training_args": {"batch_size": 2, "learning_rate": 2e-4},
            "expected_vram_gb": 12.0,
            "warnings": []
        })

    def synthesize_results(self, experiment_data: List[dict]) -> str:
        """Generate a human-readable summary of experiment results."""
        prompt = f"""Summarize these training experiment results in markdown:
{json.dumps(experiment_data, indent=2, default=str)}

Include: what worked, what didn't, and recommendations for next steps."""
        return self._query_llm(prompt)

    def _build_prompt(self, objective: str, context: str) -> str:
        tools_desc = self.get_tool_descriptions()
        memory = ""

        return f"""You are an AI agent using OpenClaw architecture principles.
You have access to tools. Decide which tool to call next to achieve the objective.

## Objective
{objective}

## Context
{context}

## Previously
{self._format_history()}

{tools_desc}

Respond with exactly:
ACTION: tool_name
PARAMS: {{"key": "value"}}
REASONING: why this action
"""

    def _format_history(self) -> str:
        if not self._conversation_history:
            return "No previous actions."
        return "\n".join(
            f"{m['role']}: {m['content'][:200]}"
            for m in self._conversation_history[-4:]
        )

    def _query_llm(self, prompt: str) -> str:
        if self.provider == "openai":
            return self._query_openai(prompt)
        elif self.provider == "anthropic":
            return self._query_anthropic(prompt)
        elif self.provider == "ollama":
            return self._query_ollama(prompt)
        else:
            return self._query_openai(prompt)

    def _query_ollama(self, prompt: str) -> str:
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=60,
            )
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception as e:
            logger.warning(f"Ollama query failed: {e}")
            return f"ACTION: analyze\nPARAMS: {{}}\nREASONING: Ollama unavailable."

    def _query_openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a precise ML training agent."},
                          {"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"OpenAI query failed: {e}. Using fallback reasoning.")
            return f"ACTION: analyze\nPARAMS: {{}}\nREASONING: LLM unavailable, using rule-based fallback."

    def _query_anthropic(self, prompt: str) -> str:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.warning(f"Anthropic query failed: {e}")
            return f"ACTION: analyze\nPARAMS: {{}}\nREASONING: LLM unavailable."

    def _parse_action(self, response: str) -> BrainAction:
        action_match = re.search(r"ACTION:\s*(\w+)", response)
        params_match = re.search(r"PARAMS:\s*(\{.*\})", response, re.DOTALL)
        reasoning_match = re.search(r"REASONING:\s*(.+)", response, re.DOTALL)

        tool_name = action_match.group(1) if action_match else "analyze"
        params = {}
        if params_match:
            try:
                params = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                params = {"raw": params_match.group(1)}
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

        return BrainAction(
            action="call_tool",
            tool=tool_name,
            params=params,
            reasoning=reasoning,
        )

    def _parse_json_safely(self, text: str, default: dict) -> dict:
        try:
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                return json.loads(json_match.group(0))
            return default
        except json.JSONDecodeError:
            return default
