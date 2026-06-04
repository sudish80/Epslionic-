"""
Domain Selector - Makes the agent ask which domain to train on.
The agent interviews the user and auto-configures based on the domain.
"""

import os
import re
from typing import Dict, List, Optional, Tuple


# Comprehensive domain knowledge base
DOMAINS = {
    "math": {
        "name": "Mathematical Reasoning",
        "emoji": "🔢",
        "description": "Math word problems, theorem proving, numerical reasoning",
        "search_queries": ["math reasoning", "mathematics", "arithmetic"],
        "datasets": ["gsm8k", "math_dataset", "MetaMathQA", "proofwiki"],
        "models": [
            "unsloth/mistral-7b-bnb-4bit",
            "unsloth/llama-3-8b-bnb-4bit",
        ],
        "training_hints": {
            "learning_rate": 2e-4,
            "lora_r": 16,
            "max_seq_length": 1024,
        },
        "format_hint": "### Instruction\n{question}\n### Response\n{answer}",
    },
    "code": {
        "name": "Code Generation",
        "emoji": "💻",
        "description": "Code completion, bug fixing, code explanation, programming",
        "search_queries": ["code generation", "programming", "coding"],
        "datasets": ["code_alpaca", "CodeFeedback", "codeparrot/github-code"],
        "models": [
            "unsloth/codellama-7b-bnb-4bit",
            "unsloth/deepseek-coder-6.7b-bnb-4bit",
        ],
        "training_hints": {
            "learning_rate": 3e-4,
            "lora_r": 8,
            "max_seq_length": 2048,
        },
        "format_hint": "### Instruction\n{question}\n### Response\n```python\n{answer}\n```",
    },
    "medical": {
        "name": "Medical / Biomedical",
        "emoji": "🏥",
        "description": "Medical Q&A, clinical notes, biomedical literature",
        "search_queries": ["medical", "biomedical", "clinical", "healthcare"],
        "datasets": ["med_qa", "pubmed_qa", "BioMed", "health_fact"],
        "models": [
            "unsloth/mistral-7b-bnb-4bit",
            "unsloth/llama-3-8b-bnb-4bit",
        ],
        "training_hints": {
            "learning_rate": 1e-4,
            "lora_r": 8,
            "max_seq_length": 512,
        },
        "format_hint": "### Question\n{question}\n### Medical Answer\n{answer}",
    },
    "legal": {
        "name": "Legal / Law",
        "emoji": "⚖️",
        "description": "Legal document analysis, case law, contract review",
        "search_queries": ["legal", "law", "case law", "contract"],
        "datasets": ["legal_bench", "case_hold", "contract_nli"],
        "models": [
            "unsloth/mistral-7b-bnb-4bit",
            "unsloth/llama-3-8b-bnb-4bit",
        ],
        "training_hints": {
            "learning_rate": 1e-4,
            "lora_r": 16,
            "max_seq_length": 2048,
        },
        "format_hint": "### Legal Question\n{question}\n### Legal Analysis\n{answer}",
    },
    "creative": {
        "name": "Creative Writing",
        "emoji": "✍️",
        "description": "Story writing, poetry, marketing copy, creative content",
        "search_queries": ["creative writing", "story", "narrative"],
        "datasets": ["dolly", "alpaca", "OpenOrca", "story_generation"],
        "models": [
            "unsloth/llama-3-8b-bnb-4bit",
            "unsloth/mistral-7b-bnb-4bit",
        ],
        "training_hints": {
            "learning_rate": 2e-4,
            "lora_r": 16,
            "max_seq_length": 2048,
        },
        "format_hint": "### Writing Prompt\n{question}\n### Story\n{answer}",
    },
    "science": {
        "name": "Science / Research",
        "emoji": "🔬",
        "description": "Scientific Q&A, physics, chemistry, biology research",
        "search_queries": ["science", "physics", "chemistry", "biology"],
        "datasets": ["sciq", "qed", "wikiscience"],
        "models": [
            "unsloth/mistral-7b-bnb-4bit",
            "unsloth/llama-3-8b-bnb-4bit",
        ],
        "training_hints": {
            "learning_rate": 2e-4,
            "lora_r": 8,
            "max_seq_length": 1024,
        },
        "format_hint": "### Scientific Question\n{question}\n### Answer\n{answer}",
    },
    "finance": {
        "name": "Finance / Business",
        "emoji": "💰",
        "description": "Financial analysis, stock market, accounting, business strategy",
        "search_queries": ["finance", "financial", "business", "economics"],
        "datasets": ["finance_bench", "fp_qa", "fin_qa"],
        "models": [
            "unsloth/mistral-7b-bnb-4bit",
            "unsloth/llama-3-8b-bnb-4bit",
        ],
        "training_hints": {
            "learning_rate": 1e-4,
            "lora_r": 8,
            "max_seq_length": 1024,
        },
        "format_hint": "### Financial Question\n{question}\n### Financial Analysis\n{answer}",
    },
    "chat": {
        "name": "Conversation / Chat",
        "emoji": "💬",
        "description": "General conversation, customer support, dialogue",
        "search_queries": ["conversation", "dialogue", "chat", "instruction"],
        "datasets": ["oasst1", "sharegpt", "dolly", "alpaca"],
        "models": [
            "unsloth/llama-3-8b-bnb-4bit",
            "unsloth/zephyr-7b-bnb-4bit",
        ],
        "training_hints": {
            "learning_rate": 2e-4,
            "lora_r": 8,
            "max_seq_length": 1024,
        },
        "format_hint": "<|user|>\n{question}\n<|assistant|>\n{answer}",
    },
    "general": {
        "name": "General Instruction Following",
        "emoji": "🧠",
        "description": "General-purpose instruction tuning, QA, reasoning",
        "search_queries": ["instruction tuning", "general qa", "reasoning"],
        "datasets": ["alpaca", "dolly", "OpenOrca", "oasst1"],
        "models": [
            "unsloth/mistral-7b-bnb-4bit",
            "unsloth/llama-3-8b-bnb-4bit",
        ],
        "training_hints": {
            "learning_rate": 2e-4,
            "lora_r": 16,
            "max_seq_length": 2048,
        },
        "format_hint": "### Instruction\n{question}\n### Response\n{answer}",
    },
}


class DomainSelector:
    """Asks the user which domain to train on and returns the config."""

    def __init__(self, brain=None, device_info: dict = None):
        self.brain = brain
        self._device_info = device_info or {}

    def show_menu(self) -> str:
        print()
        print("=" * 56)
        print("  🧠  WHAT DOMAIN DO YOU WANT TO TRAIN?")
        print("=" * 56)
        print()
        for key, info in DOMAINS.items():
            print(f"  {info['emoji']}  [{key}]\t{info['name']}")
            print(f"      {info['description']}")
            print()
        print(f"  📋  [custom]\tDescribe your own domain")
        print(f"  ❌  [quit]\tExit")
        print()
        return input("  Enter domain name or number >> ").strip().lower()

    def ask(self) -> Tuple[str, dict]:
        """
        Ask the user for the training domain.
        Returns (domain_key, domain_config).
        """
        while True:
            choice = self.show_menu()

            if choice in ("quit", "q", "exit"):
                print("\n  Goodbye!")
                raise SystemExit(0)

            # Check for custom domain
            if choice in ("custom", "c", "other"):
                return self._handle_custom()

            # Check for numeric choice
            if choice.isdigit():
                keys = list(DOMAINS.keys())
                idx = int(choice) - 1
                if 0 <= idx < len(keys):
                    choice = keys[idx]
                else:
                    print(f"\n  ❌  Invalid number. Choose 1-{len(DOMAINS)}")
                    continue

            # Match by key or partial name
            matched = self._fuzzy_match(choice)
            if matched:
                config = DOMAINS[matched]
                print(f"\n  ✅  Selected: {config['emoji']} {config['name']}")
                print(f"  📊  Datasets: {', '.join(config['datasets'][:3])}")
                print(f"  🤖  Models: {', '.join(config['models'])}")
                return matched, config

            print(f"\n  ❌  Unknown domain '{choice}'. Try again or type 'custom'.")

    def _fuzzy_match(self, text: str) -> Optional[str]:
        """Match user input to a domain key by name, alias, or partial."""
        text = text.lower().strip()

        # Direct key match
        if text in DOMAINS:
            return text

        # Match by name
        for key, info in DOMAINS.items():
            if info["name"].lower() == text:
                return key
            if info["name"].lower().startswith(text):
                return key
            if text in info["name"].lower():
                return key
            if text in key:
                return key

        # Use LLM brain to interpret if available
        if self.brain and self.brain.api_key:
            return self._llm_match(text)

        return None

    def _llm_match(self, text: str) -> Optional[str]:
        """Use LLM to match free-form text to a domain."""
        try:
            keys = list(DOMAINS.keys())
            names = [DOMAINS[k]["name"] for k in keys]
            prompt = (
                f"User wants to train an LLM on '{text}'. "
                f"Which domain matches best? Options: {', '.join(f'{k}: {n}' for k, n in zip(keys, names))}\n"
                f"Reply with just the key name."
            )
            from openai import OpenAI
            response = OpenAI(api_key=self.brain.api_key).chat.completions.create(
                model="gpt-4o-mini", messages=[
                    {"role": "system", "content": "Reply with a single domain key."},
                    {"role": "user", "content": prompt},
                ], temperature=0, max_tokens=20
            )
            matched = response.choices[0].message.content.strip().lower()
            if matched in DOMAINS:
                return matched
        except Exception:
            pass
        return None

    def _handle_custom(self) -> Tuple[str, dict]:
        """Handle custom domain input."""
        print("\n  📋  Describe your custom domain:")
        print("  (e.g., 'I want to train on cybersecurity threat reports')")
        desc = input("  >> ").strip()

        if self.brain and self.brain.api_key:
            plan = self.brain.plan(desc, {"custom": True}, "T4 16GB")
            config = {
                "name": desc[:30],
                "search_queries": [desc],
                "datasets": ["custom"],
                "models": [plan.get("model_name", "unsloth/mistral-7b-bnb-4bit")],
                "training_hints": plan.get("training_args", {}),
                "format_hint": "### Instruction\n{question}\n### Response\n{answer}",
            }
            print(f"  🤖  LLM recommends: {config['models'][0]}")
            return "custom", config

        return "custom", {
            "name": desc[:30],
            "search_queries": [desc],
            "datasets": [],
            "models": ["unsloth/mistral-7b-bnb-4bit"],
            "training_hints": {},
            "format_hint": "### Instruction\n{question}\n### Response\n{answer}",
        }

    def auto_select(self) -> Tuple[str, dict]:
        """Automatically select a domain based on environment detection. No user input."""
        env = self._detect_environment()
        print(f"\n  {'='*50}")
        print(f"  DomainSelector: Auto-selecting domain")
        print(f"  {'='*50}")
        print(f"  GPU: {'YES' if env.get('gpu') else 'NO'} | VRAM: {env.get('vram_gb',0):.0f}GB")
        print(f"  Internet: {'YES' if env.get('internet') else 'NO'}")
        print(f"  API Key: {'YES' if env.get('api_key') else 'NO'}")

        domain_key = self._auto_pick(env)
        config = DOMAINS[domain_key]
        print(f"\n  Selected: {config['emoji']} {config['name']}")
        print(f"  Dataset: {config['datasets'][0]} | Model: {config['models'][0]}")
        return domain_key, config

    def _detect_environment(self) -> dict:
        """Detect GPU, VRAM, internet, API keys — no user input."""
        env = {'gpu': False, 'vram_gb': 0, 'gpu_name': 'None',
               'internet': False, 'hf_token': False, 'api_key': False}
        try:
            import torch
            env['gpu'] = torch.cuda.is_available()
            if env['gpu']:
                env['vram_gb'] = torch.cuda.get_device_properties(0).total_mem / 1e9
                env['gpu_name'] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        try:
            import urllib.request
            urllib.request.urlopen('https://huggingface.co', timeout=3)
            env['internet'] = True
        except Exception:
            pass
        env['hf_token'] = bool(os.environ.get('HF_TOKEN', ''))
        env['api_key'] = bool(os.environ.get('OPENAI_API_KEY', ''))
        return env

    def _auto_pick(self, env: dict) -> str:
        """Pick a domain based on detected hardware. No input() calls."""
        if self.brain and self.brain.api_key:
            try:
                import json
                action = self.brain.think(
                    'Select the best domain for LLM training from the options.',
                    f"Environment: {json.dumps(env)}\nDomains: {list(DOMAINS.keys())}")
                if action.tool in DOMAINS:
                    return action.tool
            except Exception:
                pass
        if env.get('gpu'):
            vram = env.get('vram_gb', 0)
            if vram >= 20: return 'code'
            if vram >= 14: return 'science'
            if vram >= 12: return 'math'
            if vram >= 10: return 'medical'
            if vram >= 8:  return 'general'
            return 'chat'
        if env.get('internet'):
            return 'general'
        return 'math'

    def build_objective(self, domain_key: str, domain_config: dict) -> str:
        """Build a training objective string from domain selection."""
        ds = domain_config.get("datasets", [])
        models = domain_config.get("models", [])
        return (
            f"Train an LLM for {domain_config['name']}. "
            f"Search datasets like {', '.join(ds[:3])} from HuggingFace. "
            f"Use model {models[0] if models else 'Mistral 7B'}. "
            f"Configure training with {domain_config.get('training_hints', {})}. "
            f"Format: {domain_config.get('format_hint', 'instruction-response')}"
        )

    def get_domain_info_markdown(self, domain_key: str, domain_config: dict) -> str:
        """Generate a markdown summary of the domain selection."""
        lines = [
            f"# Domain: {domain_config['name']}",
            "",
            f"**Datasets:** {', '.join(domain_config.get('datasets', ['auto-discover']))}",
            f"**Models:** {', '.join(domain_config.get('models', ['auto-select']))}",
            f"**Learning Rate:** {domain_config.get('training_hints', {}).get('learning_rate', 'auto')}",
            f"**LoRA Rank:** {domain_config.get('training_hints', {}).get('lora_r', 'auto')}",
            f"**Max Length:** {domain_config.get('training_hints', {}).get('max_seq_length', 'auto')}",
            "",
        ]
        return "\n".join(lines)
