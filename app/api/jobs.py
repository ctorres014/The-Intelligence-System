"""
Tareas en segundo plano con estado persistido en Redis (cliente async).

    pending → running → completed | awaiting_approval | rejected | failed

`POST /v1/tasks` responde 202 al instante; el grafo corre en una tarea de
asyncio (sin threads, porque todo el pipeline es async) y el cliente hace
polling a `GET /v1/tasks/{job_id}`.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from redis.asyncio import Redis

from app.api.schemas import ChatResponse, JobRecord, JobStatus
from app.service import IntelligenceService

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, redis: Redis, ttl_seconds: int):
        self._redis = redis
        self._ttl = ttl_seconds

    @staticmethod
    def _key(job_id: str) -> str:
        return f"job:{job_id}"

    async def create(self, thread_id: str, message: str) -> JobRecord:
        now = _now()
        job = JobRecord(
            job_id=uuid.uuid4().hex, thread_id=thread_id, status="pending",
            message=message, created_at=now, updated_at=now,
        )
        await self._redis.set(self._key(job.job_id), job.model_dump_json(), ex=self._ttl)
        return job

    async def get(self, job_id: str) -> JobRecord | None:
        raw = await self._redis.get(self._key(job_id))
        return JobRecord.model_validate_json(raw) if raw else None

    async def update(self, job_id: str, **fields) -> JobRecord | None:
        job = await self.get(job_id)
        if job is None:
            return None
        job = job.model_copy(update={**fields, "updated_at": _now()})
        await self._redis.set(self._key(job_id), job.model_dump_json(), ex=self._ttl)
        return job


class JobRunner:
    def __init__(self, store: JobStore, service: IntelligenceService):
        self.store = store
        self._service = service
        self._tasks: set[asyncio.Task] = set()

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _finish(self, job_id: str, response: ChatResponse) -> None:
        status: JobStatus = response.status
        await self.store.update(job_id, status=status, result=response)

    async def _run(self, job: JobRecord) -> None:
        await self.store.update(job.job_id, status="running")
        try:
            await self._finish(job.job_id, await self._service.chat(job.message, job.thread_id))
        except Exception as exc:
            logger.exception("Job %s falló", job.job_id)
            await self.store.update(job.job_id, status="failed", error=str(exc))

    async def _resume(self, job: JobRecord, approved: bool, comment: str | None) -> None:
        await self.store.update(job.job_id, status="running")
        try:
            await self._finish(job.job_id, await self._service.resume(job.thread_id, approved, comment))
        except Exception as exc:
            logger.exception("Job %s falló al reanudar", job.job_id)
            await self.store.update(job.job_id, status="failed", error=str(exc))

    async def submit(self, message: str, thread_id: str) -> JobRecord:
        job = await self.store.create(thread_id, message)
        self._spawn(self._run(job))
        return job

    def resume(self, job: JobRecord, approved: bool, comment: str | None) -> None:
        self._spawn(self._resume(job, approved, comment))

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
