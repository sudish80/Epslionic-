"""Chain composition - composable pipeline steps.
Pattern from LangChain (138k stars): RunnableSequence, RunnableParallel."""

import logging
from typing import Dict, Any, List, Callable, Optional, Union

logger = logging.getLogger("epsionic.chain")


class Chain:
    """A single pipeline step with typed inputs/outputs."""

    def __init__(self, name: str, fn: Callable, description: str = ""):
        self.name = name
        self._fn = fn
        self.description = description or name
        self._next: Optional[Chain] = None

    def __or__(self, other: "Chain") -> "Chain":
        """Pipe operator: chain1 | chain2"""
        self._next = other
        return other

    def __rshift__(self, other: "Chain") -> "Chain":
        self._next = other
        return other

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Chain [{self.name}]: {self.description}")
        try:
            result = self._fn(inputs)
            if not isinstance(result, dict):
                result = {self.name: result}
            result["_chain"] = self.name
        except Exception as e:
            logger.error(f"Chain [{self.name}] failed: {e}")
            result = {"_chain": self.name, "_error": str(e)}

        if self._next:
            return self._next.invoke({**inputs, **result})
        return result


class ChainBuilder:
    """Build composable pipelines.
    
    Usage:
        pipeline = ChainBuilder()
        pipeline.add_step("discover", lambda ctx: {...})
        pipeline.add_step("train", lambda ctx: {...})
        result = pipeline.run({"objective": ...})
    """

    def __init__(self):
        self._steps: List[Chain] = []

    def add_step(self, name: str, fn: Callable, description: str = ""):
        chain = Chain(name, fn, description)
        if self._steps:
            self._steps[-1] >> chain
        self._steps.append(chain)
        return self

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self._steps:
            return inputs
        return self._steps[0].invoke(inputs)

    def to_list(self) -> List[str]:
        return [s.name for s in self._steps]


class Pipeline:
    """High-level pipeline that wraps ChainBuilder for domain training.
    Pattern from scikit-learn (66k stars): pipeline.fit() / pipeline.transform()."""

    def __init__(self):
        self._steps: List[Chain] = []
        self._results: Dict[str, Any] = {}

    def add(self, name: str, fn: Callable, description: str = ""):
        self._steps.append(Chain(name, fn, description))
        return self

    def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = dict(context or {})
        for step in self._steps:
            logger.info(f"Pipeline step: {step.name}")
            result = step.invoke(context)
            self._results[step.name] = result
            context.update(result)
        return context

    @property
    def results(self) -> Dict[str, Any]:
        return dict(self._results)

    def summary(self) -> str:
        lines = ["# Pipeline Steps", ""]
        for s in self._steps:
            status = "done" if s.name in self._results else "pending"
            lines.append(f"- {s.name}: {status}")
        return "\n".join(lines)


def step(name: str, description: str = ""):
    """Decorator to create a Chain step from a function."""
    def decorator(fn):
        return Chain(name, fn, description)
    return decorator
