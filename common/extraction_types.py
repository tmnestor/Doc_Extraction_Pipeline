"""Foundation dataclasses for the agentic extraction engine.

Zero dependencies on existing modules -- safe to import anywhere.
All graph execution state, results, and session types live here.

Named ``NodeGenParams`` (not ``GenerationParams``) to avoid collision
with ``models/backend.py:GenerationParams`` which is a frozen dataclass
used by the orchestrator pipeline.
"""

from dataclasses import dataclass, field
from typing import Any

_MISSING = object()  # sentinel for WorkflowState.get() default


@dataclass
class NodeGenParams:
    """Per-node generation parameters built from YAML node definition.

    The generate_fn wrapper dispatches via ``match`` on these fields
    to build backend-specific sampling parameters (vLLM SamplingParams,
    HF GenerationConfig, etc.).
    """

    max_tokens: int  # explicitly set by GraphExecutor._resolve_max_tokens()
    temperature: float = 0.0
    stop: list[str] | None = None
    output_schema: dict | None = None  # JSON schema -> vLLM StructuredOutputsParams
    logprobs: int | None = None  # top-k logprobs per token
    # Pre-tiling tile budget for the vLLM backend; max_tiles None -> single-image
    # path. Populated from the per-doc-type image budget when the graph path needs
    # high-resolution tiling. min_tiles is the floor forced into the crop so
    # clean-aspect docs don't under-tile (see plans/2026-06-04-adaptive-tiling-dense-bank.md).
    max_tiles: int | None = None
    min_tiles: int = 1


@dataclass
class NodeResult:
    """Result of executing a single graph node."""

    key: str  # matches YAML node key
    image_ref: str  # which image this node ran against
    prompt_sent: str  # after template injection (empty for routers/validators)
    raw_response: str  # raw model output (empty for routers/validators)
    parsed: dict[str, Any]  # structured parse of this node's output
    elapsed: float  # seconds
    attempt: int  # 1-indexed retry attempt number
    edge_taken: str  # which edge was followed


@dataclass
class WorkflowState:
    """Typed state accumulated across graph execution.

    Implements the MetaGPT pattern: structured outputs between nodes
    prevent garbage propagation through the graph.
    """

    node_results: dict[str, NodeResult] = field(default_factory=dict)
    image_name: str = ""  # set by GraphExecutor._walk for diagnostic logging

    def get(self, dot_path: str, default: Any = _MISSING) -> Any:
        """Resolve a dot-path reference.

        Example: ``'detect_headers.column_mapping.balance'``

        If *default* is provided, return it when the path cannot be
        resolved (missing node or missing key).  Without a default,
        raise ``KeyError`` with a diagnostic message.
        """
        parts = dot_path.split(".")
        node_key = parts[0]
        if node_key not in self.node_results:
            if default is not _MISSING:
                return default
            msg = f"Node '{node_key}' not found in state (have: {list(self.node_results)})"
            raise KeyError(msg)
        result: Any = self.node_results[node_key].parsed
        for part in parts[1:]:
            try:
                if isinstance(result, dict):
                    result = result[part]
                else:
                    result = getattr(result, part)
            except (KeyError, AttributeError):
                if default is not _MISSING:
                    return default
                error_ctx = self.node_results[node_key].parsed.get("error", "")
                hint = f" (node parse failed: {error_ctx})" if error_ctx else ""
                msg = f"Cannot resolve '{dot_path}': key '{part}' missing{hint}"
                raise KeyError(msg) from None
        return result

    def has(self, node_key: str) -> bool:
        """Check whether a node has produced a result."""
        return node_key in self.node_results
