"""Visual Flow Engine — n8n-inspired node-based pipeline orchestration.
Define training workflows as JSON/YAML DAGs; execute, visualize, and share."""

import json
import logging
import time
import yaml
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("openclaw.tool.flow")


NODE_REGISTRY = {}


def register_node_type(type_name: str):
    def decorator(fn):
        NODE_REGISTRY[type_name] = fn
        return fn
    return decorator


@dataclass
class FlowNode:
    id: str
    type: str
    params: dict = field(default_factory=dict)
    connections: dict = field(default_factory=dict)  # {"output": "next_node_id"}
    retry_on_fail: int = 0
    timeout: int = 3600


@dataclass
class FlowDefinition:
    name: str
    description: str = ""
    version: str = "1.0"
    triggers: list = field(default_factory=list)
    nodes: list = field(default_factory=list)


class FlowError(Exception):
    pass


class FlowEngine:
    """Execute DAG-based workflows with retry, error handling, and state persistence."""

    def __init__(self, memory_store=None, flows_dir: Path = None):
        self.memory = memory_store
        self.flows_dir = flows_dir or Path("/content/flows")
        self.flows_dir.mkdir(parents=True, exist_ok=True)

    def create_flow(self, name: str, description: str = "",
                    triggers: list = None) -> str:
        flow_id = f"flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        flow = FlowDefinition(
            name=name, description=description, triggers=triggers or [],
        )
        path = self.flows_dir / f"{flow_id}.json"
        path.write_text(json.dumps(asdict(flow), indent=2))
        return flow_id

    def add_node(self, flow_id: str, node_type: str, params: dict = None,
                 connections: dict = None, after_node: str = None) -> str:
        path = self.flows_dir / f"{flow_id}.json"
        if not path.exists():
            raise FlowError(f"Flow {flow_id} not found")
        flow_data = json.loads(path.read_text())

        node_id = f"node_{len(flow_data['nodes']) + 1}"
        node = FlowNode(
            id=node_id, type=node_type,
            params=params or {},
            connections=connections or {},
        )

        if after_node:
            for n in flow_data["nodes"]:
                if n["id"] == after_node:
                    n["connections"]["output"] = node_id
                    break

        flow_data["nodes"].append(asdict(node))
        path.write_text(json.dumps(flow_data, indent=2))
        return node_id

    def execute(self, flow_id: str, tool_executor: Callable = None,
                context: dict = None) -> dict:
        path = self.flows_dir / f"{flow_id}.json"
        if not path.exists():
            raise FlowError(f"Flow {flow_id} not found")
        flow_data = json.loads(path.read_text())

        results = {}
        node_map = {n["id"]: n for n in flow_data["nodes"]}
        entry_nodes = [n for n in flow_data["nodes"]
                       if not any(n["id"] in other["connections"].get("output", "")
                                  for other in flow_data["nodes"])]

        if not entry_nodes and flow_data["nodes"]:
            entry_nodes = [flow_data["nodes"][0]]

        visited = set()
        queue = list(entry_nodes)
        ctx = dict(context or {})

        while queue:
            node = queue.pop(0)
            if node["id"] in visited:
                continue
            visited.add(node["id"])

            node_type = node["type"]
            params = dict(node["params"])
            params.update(ctx)

            if tool_executor and node_type in NODE_REGISTRY:
                fn = NODE_REGISTRY[node_type]
            elif tool_executor:
                fn = lambda **kw: tool_executor(node_type, **kw)
            else:
                fn = None

            result = None
            errors = []
            if fn:
                retries = node.get("retry_on_fail", 0) + 1
                for attempt in range(retries):
                    try:
                        result = fn(**params)
                        break
                    except Exception as e:
                        errors.append(str(e))
                        if attempt < retries - 1:
                            logger.warning(f"Retry {attempt+1}/{retries} for {node['id']}: {e}")
                            time.sleep(2 ** attempt)

            results[node["id"]] = {
                "type": node_type, "params": params, "result": result,
                "errors": errors, "success": len(errors) == 0,
            }
            ctx[f"{node['id']}_output"] = result

            output_target = node.get("connections", {}).get("output")
            if output_target and output_target in node_map:
                queue.append(node_map[output_target])

            outputs = node.get("connections", {}).get("outputs", {})
            for key, target in outputs.items():
                if target in node_map:
                    ctx[f"{target}_input_key"] = key
                    queue.append(node_map[target])

        return {"flow_id": flow_id, "nodes": results, "context": ctx,
                "success": all(n["success"] for n in results.values())}

    def list_flows(self) -> list:
        flows = []
        for f in self.flows_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                flows.append({
                    "id": f.stem, "name": data.get("name", f.stem),
                    "nodes": len(data.get("nodes", [])),
                    "description": data.get("description", ""),
                })
            except Exception:
                pass
        return flows

    def get_flow(self, flow_id: str) -> Optional[dict]:
        path = self.flows_dir / f"{flow_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def export_mermaid(self, flow_id: str) -> str:
        flow = self.get_flow(flow_id)
        if not flow:
            return "Flow not found"
        lines = ["graph TD;"]
        for node in flow.get("nodes", []):
            nid = node["id"]
            label = f"{node['type']}"
            lines.append(f"    {nid}[{label}];")
            out = node.get("connections", {}).get("output")
            if out:
                lines.append(f"    {nid} --> {out};")
            outputs = node.get("connections", {}).get("outputs", {})
            for key, target in outputs.items():
                lines.append(f"    {nid} -->|{key}| {target};")
        return "\n".join(lines)

    def export_yaml(self, flow_id: str) -> str:
        flow = self.get_flow(flow_id)
        if not flow:
            return ""
        return yaml.dump(flow, default_flow_style=False)


def get_tool_description() -> dict:
    return {
        "create_flow": {"description": "Create an n8n-style workflow DAG",
            "parameters": {"name": "Flow name", "description": "What this flow does",
                "triggers": "List of trigger configs"}},
        "add_node": {"description": "Add a processing node to a flow",
            "parameters": {"flow_id": "Target flow", "node_type": "Node type",
                "params": "Node parameters", "after_node": "Connect after this node ID"}},
        "execute": {"description": "Run a flow DAG",
            "parameters": {"flow_id": "Flow to execute"}},
    }
