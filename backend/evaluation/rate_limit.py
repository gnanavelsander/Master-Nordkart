import asyncio
import time
from dataclasses import dataclass


@dataclass
class RateLimiter:
    calls_per_minute: int   = 60
    max_concurrent:   int   = 5
    threshold_pause:  float = 0.9
    threshold_resume: float = 0.5

    def __post_init__(self):
        self.semaphore  = asyncio.Semaphore(self.max_concurrent)
        self.call_times = []
        self.lock       = asyncio.Lock()
        self.paused     = False

    def _usage(self) -> float:
        now = time.time()
        self.call_times = [t for t in self.call_times if now - t < 60]
        return len(self.call_times) / self.calls_per_minute

    def _log(self, msg: str):
        print(f"  [RateLimiter] {msg}")

    async def _wait_until_safe(self):
        while True:
            wait_time = 5
            async with self.lock:
                u = self._usage()

                if u >= self.threshold_pause:
                    self.paused = True
                    oldest    = self.call_times[0] if self.call_times else time.time()
                    wait_time = max(60 - (time.time() - oldest), 1)
                    self._log(f"usage {u:.0%} — pausing {wait_time:.1f}s")

                elif self.paused and u <= self.threshold_resume:
                    self.paused = False
                    self._log(f"usage {u:.0%} — resuming")

                if not self.paused:
                    return

            await asyncio.sleep(wait_time)

    async def handle_429(self, model_id: str, attempt: int):
        wait = 30 * (attempt + 1)
        self._log(f"{model_id} got 429 — waiting {wait}s")
        await asyncio.sleep(wait)

    async def __aenter__(self):
        await self._wait_until_safe()
        await self.semaphore.acquire()
        async with self.lock:
            self.call_times.append(time.time())
        return self

    async def __aexit__(self, *_):
        self.semaphore.release()

    async def status(self) -> dict:
        async with self.lock:
            u = self._usage()
            return {
                "usage":   f"{u:.0%}",
                "paused":  self.paused,
                "calls":   len(self.call_times),
                "limit":   self.calls_per_minute,
            }