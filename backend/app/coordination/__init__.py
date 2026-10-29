"""Central coordination primitives for document, code and web tasks.

This package is deliberately import-light. Service-level adapters (database,
ForgeFlow, retrieval graph) are imported inside functions so the planner and
URL guard can be unit-tested on a host that does not have the full RAG stack.
"""

from app.coordination.planner import (
    MAX_SUBTASKS,
    Plan,
    SubTask,
    TaskKind,
    plan_request,
)

__all__ = ["MAX_SUBTASKS", "Plan", "SubTask", "TaskKind", "plan_request"]
