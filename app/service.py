"""
Capa de servicio: ejecuta turnos de conversación sobre el grafo.

Cada conversación es un `thread_id` del checkpointer. Como el estado vive en
Redis (AsyncRedisSaver), un hilo puede continuarse — o un HITL pendiente
aprobarse — después de reiniciar la API.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.agents.common import message_text
from app.agents.hitl import ApprovalDecision
from app.agents.state import new_turn_input
from app.api.schemas import ChatResponse, ChatTurn, PendingApproval, Source, Step, ThreadState
from app.observability import trace_session

logger = logging.getLogger(__name__)

WORKER_NODES = ("knowledge", "research", "analyst")


class ThreadBusyError(Exception):
    """El hilo ya está procesando otro mensaje."""


class ApprovalStateError(Exception):
    """El hilo no está (o sí está) esperando aprobación humana."""


class IntelligenceService:
    def __init__(self, graph: CompiledStateGraph):
        self.graph = graph
        self._locks: dict[str, asyncio.Lock] = {}

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def new_thread_id() -> str:
        return uuid.uuid4().hex[:16]

    @staticmethod
    def _config(thread_id: str, operation: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "run_name": f"intelligence_system.{operation}",
            "tags": ["intelligence-system", operation],
            "metadata": {"thread_id": thread_id, "operation": operation},
        }

    def _lock(self, thread_id: str) -> asyncio.Lock:
        lock = self._locks.setdefault(thread_id, asyncio.Lock())
        if lock.locked():
            raise ThreadBusyError(f"El hilo '{thread_id}' está procesando otro mensaje")
        return lock

    async def _snapshot(self, thread_id: str):
        return await self.graph.aget_state({"configurable": {"thread_id": thread_id}})

    @staticmethod
    def _pending(snapshot) -> PendingApproval | None:
        interrupts = getattr(snapshot, "interrupts", ()) or ()
        if not interrupts:
            return None
        value = interrupts[0].value
        return PendingApproval.model_validate(value) if isinstance(value, dict) else None

    def _build_response(self, thread_id: str, snapshot, t0: float) -> ChatResponse:
        values = snapshot.values or {}
        pending = self._pending(snapshot)
        if pending is not None:
            status = "awaiting_approval"
        elif values.get("is_approved") is False:
            status = "rejected"
        else:
            status = "completed"
        steps = [Step.model_validate(s) for s in values.get("steps", [])]
        agents_used = list(dict.fromkeys(s.node for s in steps if s.node in WORKER_NODES))
        return ChatResponse(
            thread_id=thread_id,
            status=status,
            answer=values.get("final_answer") if pending is None else None,
            sources=[Source.model_validate(s) for s in values.get("sources", [])],
            agents_used=agents_used,
            steps=steps,
            iterations=values.get("iteration_count", 0),
            pending_approval=pending,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    # ── operaciones ─────────────────────────────────────────────────────────

    async def chat(self, message: str, thread_id: str | None = None) -> ChatResponse:
        thread_id = thread_id or self.new_thread_id()
        t0 = time.perf_counter()
        async with self._lock(thread_id):
            if self._pending(await self._snapshot(thread_id)):
                raise ApprovalStateError(
                    f"El hilo '{thread_id}' tiene una aprobación pendiente: resolvela con POST /v1/threads/{thread_id}/approve"
                )
            with trace_session(thread_id, operation="chat"):
                await self.graph.ainvoke(new_turn_input(message), self._config(thread_id, "chat"))
            return self._build_response(thread_id, await self._snapshot(thread_id), t0)

    async def resume(self, thread_id: str, approved: bool, comment: str | None = None) -> ChatResponse:
        t0 = time.perf_counter()
        async with self._lock(thread_id):
            if self._pending(await self._snapshot(thread_id)) is None:
                raise ApprovalStateError(f"El hilo '{thread_id}' no tiene aprobaciones pendientes")
            decision = ApprovalDecision(approved=approved, comment=comment)
            with trace_session(thread_id, operation="approve"):
                await self.graph.ainvoke(Command(resume=decision.model_dump()), self._config(thread_id, "approve"))
            return self._build_response(thread_id, await self._snapshot(thread_id), t0)

    async def ensure_can_start(self, thread_id: str) -> None:
        """Validaciones previas (para endpoints que no pueden fallar a mitad de un stream)."""
        lock = self._locks.get(thread_id)
        if lock is not None and lock.locked():
            raise ThreadBusyError(f"El hilo '{thread_id}' está procesando otro mensaje")
        if self._pending(await self._snapshot(thread_id)):
            raise ApprovalStateError(f"El hilo '{thread_id}' tiene una aprobación pendiente")

    async def stream(self, message: str, thread_id: str | None = None) -> AsyncIterator[dict[str, Any]]:
        """Emite un evento por cada nodo ejecutado y un evento final con la respuesta."""
        thread_id = thread_id or self.new_thread_id()
        t0 = time.perf_counter()
        async with self._lock(thread_id):
            if self._pending(await self._snapshot(thread_id)):
                raise ApprovalStateError(f"El hilo '{thread_id}' tiene una aprobación pendiente")
            yield {"event": "start", "data": {"thread_id": thread_id}}
            with trace_session(thread_id, operation="stream"):
                async for update in self.graph.astream(
                    new_turn_input(message), self._config(thread_id, "stream"), stream_mode="updates"
                ):
                    for node, payload in update.items():
                        if node == "__interrupt__":
                            continue
                        last_step = (payload or {}).get("steps", [{}])[-1] if payload else {}
                        yield {
                            "event": "step",
                            "data": {"node": node, "summary": last_step.get("summary"), "decision": last_step.get("decision")},
                        }
            response = self._build_response(thread_id, await self._snapshot(thread_id), t0)
            yield {"event": "final", "data": response.model_dump()}

    async def thread_state(self, thread_id: str) -> ThreadState | None:
        snapshot = await self._snapshot(thread_id)
        values = snapshot.values or {}
        if not values:
            return None
        pending = self._pending(snapshot)
        if pending is not None:
            status = "awaiting_approval"
        elif values.get("is_approved") is False:
            status = "rejected"
        elif values.get("final_answer"):
            status = "completed"
        else:
            status = "empty"
        messages = [
            ChatTurn(role="user" if isinstance(m, HumanMessage) else "assistant", content=message_text(m))
            for m in values.get("messages", [])
        ]
        return ThreadState(
            thread_id=thread_id,
            status=status,
            messages=messages,
            last_steps=[Step.model_validate(s) for s in values.get("steps", [])],
            pending_approval=pending,
            checkpoint_id=(snapshot.config or {}).get("configurable", {}).get("checkpoint_id"),
        )
