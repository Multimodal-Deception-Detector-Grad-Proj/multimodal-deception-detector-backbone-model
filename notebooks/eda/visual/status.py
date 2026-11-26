"""Progress bars, parallel task running, and per-stage reports."""

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional

import pandas as pd

try:  # tqdm is optional; everything works without it
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


def progress(iterable, **kwargs):
    """Wrap ``iterable`` in a tqdm bar when tqdm is installed."""
    return iterable if _tqdm is None else _tqdm(iterable, **kwargs)


def run_tasks(
    fn: Callable,
    tasks: Iterable,
    workers: int = 1,
    desc: str = "",
    processes: bool = False,
    on_error: Optional[Callable] = None,
) -> Iterator:
    """Yield ``fn(task)`` results as they finish, with a progress bar.

    * ``workers <= 1`` runs in-process (easy to debug, real tracebacks).
    * ``processes=True`` uses *spawn* worker processes (``fn`` and tasks must be
      picklable, i.e. ``fn`` must be a module-level function). Spawn avoids the
      fork-after-OpenCV/torch deadlocks that make notebook pools appear to hang.
    * ``on_error(task, exc)`` turns a worker crash into a normal result.
    """
    tasks = list(tasks)
    if not tasks:
        return
    if workers <= 1:
        for task in progress(tasks, desc=desc):
            yield fn(task)
        return

    if processes:
        pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(fn, task): task for task in tasks}
    try:
        for fut in progress(as_completed(futures), total=len(futures), desc=desc):
            try:
                yield fut.result()
            except Exception as exc:
                if on_error is None:
                    raise
                yield on_error(futures[fut], exc)
    except BaseException:  # includes KeyboardInterrupt / GeneratorExit
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)


@dataclass
class StageReport:
    """What a pipeline stage did: cached vs computed vs failed, and how long."""

    name: str
    total: int = 0
    cached: int = 0
    computed: int = 0
    failed: List[Dict[str, str]] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    elapsed_sec: float = 0.0
    _t0: float = field(default=0.0, repr=False)

    def start(self) -> "StageReport":
        self._t0 = time.perf_counter()
        return self

    def finish(self, verbose: bool = True) -> "StageReport":
        self.elapsed_sec = time.perf_counter() - self._t0
        if verbose:
            print(self.summary())
        return self

    def fail(self, video: str, error: str) -> None:
        self.failed.append({"video": video, "error": error})

    def summary(self) -> str:
        line = (
            f"[{self.name}] {self.total} videos | {self.cached} cached | "
            f"{self.computed} computed | {len(self.failed)} failed | {self.elapsed_sec:.1f}s"
        )
        if self.timings:
            mean = sum(self.timings.values()) / len(self.timings)
            slow = sorted(self.timings.items(), key=lambda kv: -kv[1])[:3]
            line += f"\n    mean {mean:.2f}s per computed video; slowest: " + ", ".join(
                f"{k} ({v:.1f}s)" for k, v in slow
            )
        return line

    def failures_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.failed, columns=["video", "error"])
