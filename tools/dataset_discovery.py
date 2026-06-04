"""
Dataset Discovery Tool - Automated Hugging Face dataset search and loading.
Uses the Hugging Face Datasets API for discovery and retrieval.
Epslionic skill: encapsulates a reusable capability as a tool.
"""

import json
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path

logger = logging.getLogger("epsionic.tool.dataset_discovery")


class DatasetDiscoveryTool:
    def __init__(self, memory_store=None, datasets_dir: Path = None,
                 huggingface_token: Optional[str] = None):
        self.memory = memory_store
        self.datasets_dir = datasets_dir or Path("/content/datasets")
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.huggingface_token = huggingface_token
        self._dataset_cache: Dict[str, dict] = {}

    def search(self, query: str = "", task: str = "",
               language: str = "", max_results: int = 10) -> List[dict]:
        """Search Hugging Face datasets by query, task, or language."""
        try:
            from huggingface_hub import list_datasets

            results = []
            filters = {}

            if task:
                filters["task_categories"] = task

            # Search via huggingface_hub
            for ds in list_datasets(search=query or None, **filters):
                info = {
                    "id": ds.id,
                    "description": getattr(ds, 'description', '') or '',
                    "tags": getattr(ds, 'tags', []) or [],
                    "downloads": getattr(ds, 'downloads', 0) or 0,
                    "last_modified": str(getattr(ds, 'last_modified', '')),
                }
                results.append(info)
                if len(results) >= max_results:
                    break

            if self.memory:
                self.memory.write_agent_state("last_dataset_search", {
                    "query": query, "results_count": len(results),
                    "timestamp": __import__('datetime').datetime.now().isoformat()
                })

            return results

        except ImportError:
            logger.warning("huggingface_hub not installed. Using mock search.")
            return self._mock_search(query)
        except Exception as e:
            logger.error(f"Dataset search failed: {e}")
            return []

    def load(self, dataset_id: str, split: str = "train",
             subset: Optional[str] = None, max_samples: Optional[int] = None,
             streaming: bool = False) -> dict:
        """Load a dataset from Hugging Face, optionally streaming."""
        try:
            from datasets import load_dataset

            logger.info(f"Loading dataset: {dataset_id} (streaming={streaming})")

            kwargs = {"split": split}
            if streaming:
                kwargs["streaming"] = True
            if subset:
                kwargs["name"] = subset

            dataset = load_dataset(dataset_id, **kwargs)

            result = {
                "dataset_id": dataset_id,
                "split": split,
                "num_samples": len(dataset) if not streaming else "streaming",
                "streaming": streaming,
                "features": list(dataset.features.keys()) if hasattr(dataset, 'features') else [],
                "sample": dataset[0] if len(dataset) > 0 else None,
                "loaded": True,
                "location": str(self.datasets_dir / dataset_id.replace("/", "_")),
            }

            if max_samples and len(dataset) > max_samples:
                if streaming:
                    dataset = dataset.take(max_samples)
                else:
                    dataset = dataset.select(range(max_samples))
                result["num_samples"] = max_samples
                result["truncated"] = True

            # Save a sample for inspection
            sample_path = self.datasets_dir / f"{dataset_id.replace('/', '_')}_sample.json"
            sample_path.write_text(json.dumps(result["sample"], default=str, indent=2))

            if self.memory:
                self.memory.record_dataset(dataset_id, result)

            return result

        except ImportError:
            logger.warning("datasets library not installed")
            return {"dataset_id": dataset_id, "loaded": False, "error": "datasets library not installed"}
        except Exception as e:
            logger.error(f"Dataset load failed for {dataset_id}: {e}")
            return {"dataset_id": dataset_id, "loaded": False, "error": str(e)}

    def analyze(self, dataset_id: str) -> dict:
        """Analyze a dataset's structure and statistics."""
        try:
            from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names

            info = {
                "dataset_id": dataset_id,
                "configs": [],
                "splits": [],
                "features": {},
            }

            try:
                info["configs"] = get_dataset_config_names(dataset_id)
            except Exception:
                pass

            try:
                info["splits"] = get_dataset_split_names(dataset_id)
            except Exception:
                pass

            # Load a small sample to inspect features
            ds = load_dataset(dataset_id, split="train", streaming=True)
            sample = next(iter(ds))
            info["features"] = {
                k: str(type(v).__name__) for k, v in sample.items()
            }
            info["sample"] = {k: str(v)[:200] for k, v in sample.items()}

            return info

        except Exception as e:
            logger.error(f"Dataset analysis failed: {e}")
            return {"dataset_id": dataset_id, "error": str(e)}

    def get_tool_description(self) -> dict:
        return {
            "search": {
                "description": "Search Hugging Face datasets by query, task, or language",
                "parameters": {
                    "query": "Search query string",
                    "task": "Filter by task category (e.g., text-classification, summarization)",
                    "language": "Filter by language",
                    "max_results": "Maximum results to return (default 10)"
                }
            },
            "load": {
                "description": "Load a dataset from Hugging Face into memory",
                "parameters": {
                    "dataset_id": "Hugging Face dataset ID (e.g., 'gsm8k')",
                    "split": "Dataset split (train, test, validation)",
                    "subset": "Subset/config name if applicable",
                    "max_samples": "Limit number of samples to load"
                }
            },
            "analyze": {
                "description": "Analyze dataset structure, features, and statistics",
                "parameters": {
                    "dataset_id": "Hugging Face dataset ID"
                }
            }
        }

    def _mock_search(self, query: str) -> List[dict]:
        """Mock search for when HF hub is unavailable."""
        popular = [
            {"id": "gsm8k", "description": "Math word problems", "downloads": 50000},
            {"id": "wikitext", "description": "Wikipedia text corpus", "downloads": 80000},
            {"id": "openwebtext", "description": "Open web text corpus", "downloads": 30000},
            {"id": "c4", "description": "Colossal Clean Crawled Corpus", "downloads": 100000},
            {"id": "alpaca", "description": "Instruction-following dataset", "downloads": 60000},
            {"id": "dolly", "description": "Databricks instruction dataset", "downloads": 40000},
            {"id": "oasst1", "description": "Open Assistant conversations", "downloads": 35000},
            {"id": "code_alpaca", "description": "Code generation instructions", "downloads": 25000},
            {"id": "squad", "description": "Question answering dataset", "downloads": 70000},
            {"id": "imdb", "description": "Movie review sentiment", "downloads": 90000},
        ]
        if query:
            q = query.lower()
            return [d for d in popular if q in d["id"].lower() or q in d["description"].lower()]
        return popular
