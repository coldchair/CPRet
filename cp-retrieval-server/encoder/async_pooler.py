from __future__ import annotations

import asyncio
from concurrent.futures import Future, InvalidStateError, TimeoutError as FutureTimeoutError
import threading
import uuid

import numpy as np


class EncoderBusyError(RuntimeError):
    pass


class EncoderTimeoutError(RuntimeError):
    pass


class AsyncPooler:
    """One engine/event loop shared by synchronous callers; no inference-wide lock."""

    def __init__(self, factory, params_factory, max_pending: int = 32, timeout: float = 120) -> None:
        if max_pending < 1 or timeout <= 0:
            raise ValueError("Encoder queue size and timeout must be positive")
        self._factory = factory
        self._params_factory = params_factory
        self._timeout = timeout
        self._slots = threading.BoundedSemaphore(max_pending)
        self._state_lock = threading.Lock()
        self._ready = Future()
        self._closed = False
        self._pending = set()
        self._active = 0
        self._peak = 0
        self._completed = 0
        self._errors = 0
        self._thread = threading.Thread(target=self._run, name="vllm-async-pooling", daemon=True)
        self._thread.start()
        self._ready.result()

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as error:
            if not self._ready.done():
                self._ready.set_exception(error)
            with self._state_lock:
                self._closed = True
                pending = list(self._pending)
            for future in pending:
                future.cancel()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        self._tasks = set()
        self._engine = self._factory()
        self._ready.set_result(None)
        try:
            await self._stop.wait()
        finally:
            tasks = list(self._tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._engine.shutdown(timeout=10)

    async def _encode(self, token_ids: list[int], priority: int) -> np.ndarray:
        request_id = uuid.uuid4().hex
        try:
            result = None
            async for output in self._engine.encode(
                {"prompt_token_ids": token_ids}, self._params_factory(), request_id,
                priority=priority,
            ):
                if getattr(output, "error", None) is not None:
                    raise RuntimeError(f"Embedding request failed: {output.error}")
                if output.finished:
                    data = output.outputs.data
                    if hasattr(data, "detach"):
                        data = data.detach().float().cpu().numpy()
                    result = np.asarray(data, dtype=np.float32)
            if result is None or result.ndim != 1:
                raise ValueError("Async encoder did not return a final embedding vector")
            return result
        except BaseException:
            try:
                await self._engine.abort(request_id)
            except Exception:
                pass
            raise

    def submit(self, token_ids: list[int], priority: int = 0) -> Future:
        if not self._slots.acquire(blocking=False):
            raise EncoderBusyError("Encoder request queue is full; retry later")
        future = Future()
        try:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Async encoder is closed")
                self._pending.add(future)
                self._active += 1
                self._peak = max(self._peak, self._active)
                self._loop.call_soon_threadsafe(self._start, future, token_ids, priority)
            return future
        except BaseException:
            with self._state_lock:
                if future in self._pending:
                    self._pending.remove(future)
                    self._active -= 1
            self._slots.release()
            raise

    def _start(self, future: Future, token_ids: list[int], priority: int) -> None:
        if future.cancelled():
            self._release(future, failed=True)
            return
        task = self._loop.create_task(self._encode(token_ids, priority))
        self._tasks.add(task)
        task.add_done_callback(lambda completed: self._finished(future, completed))

        def cancel(completed):
            if completed.cancelled() and not task.done():
                self._loop.call_soon_threadsafe(task.cancel)

        future.add_done_callback(cancel)

    def _finished(self, future: Future, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        failed = task.cancelled() or task.exception() is not None
        self._release(future, failed=failed)
        if future.done():
            return
        try:
            if task.cancelled():
                future.cancel()
            elif failed:
                future.set_exception(task.exception())
            else:
                future.set_result(task.result())
        except InvalidStateError:
            pass

    def _release(self, future: Future, failed: bool) -> None:
        with self._state_lock:
            self._pending.remove(future)
            self._active -= 1
            if failed:
                self._errors += 1
            else:
                self._completed += 1
        self._slots.release()

    def encode_batch(self, prompts: list[list[int]], priority: int = 0) -> np.ndarray:
        futures = []
        try:
            for token_ids in prompts:
                futures.append(self.submit(token_ids, priority))

            async def collect():
                results = await asyncio.wait_for(
                    asyncio.gather(*(asyncio.wrap_future(future) for future in futures)),
                    timeout=self._timeout,
                )
                return np.asarray(results, dtype=np.float32)

            combined = asyncio.run_coroutine_threadsafe(collect(), self._loop)
            try:
                return combined.result(timeout=self._timeout + 5)
            except (TimeoutError, FutureTimeoutError) as error:
                combined.cancel()
                raise EncoderTimeoutError("Encoder request timed out") from error
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    def status(self) -> dict:
        with self._state_lock:
            return {"submitted_active": self._active, "peak_submitted_active": self._peak,
                    "completed": self._completed, "errors": self._errors,
                    "closed": self._closed}

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            pending = list(self._pending)
        for future in pending:
            future.cancel()
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=15)
