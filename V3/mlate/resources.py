"""
Compute budget
==============

One place that decides how much of the machine a training run may use, so no
stage has to guess. Import and call `claim()` at the top of any script that
trains something.

    from mlate import resources as res
    res.claim()                 # sets thread counts, GPU fractions, env vars
    print(res.BUDGET)

Policy (set in config.py):
    CPU     80% of logical cores
    RAM     85% of physical memory
    GPU     90% of each visible device's memory

The CPU share is applied three ways, because they are enforced by different
layers and setting only one leaves the others unbounded:
  - n_jobs on scikit-learn / XGBoost / LightGBM estimators
  - OMP/MKL/OPENBLAS thread counts, which control the BLAS underneath them
  - torch's intra-op and inter-op thread pools

Oversubscription warning: if you run N estimators in parallel and each is given
N_JOBS threads, you request N x N_JOBS threads and the machine thrashes. Use
`inner_jobs(n_parallel)` to divide the budget when nesting parallelism.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from mlate import config as cfg


@dataclass(frozen=True)
class Budget:
    cpu_total: int
    n_jobs: int
    ram_total_gib: float
    ram_limit_gib: float
    gpus: list[str] = field(default_factory=list)
    gpu_fraction: float = cfg.GPU_MEMORY_FRACTION

    def __str__(self) -> str:
        gpu = ", ".join(f"[{i}] {n}" for i, n in enumerate(self.gpus)) or "none"
        return (f"CPU {self.n_jobs}/{self.cpu_total} threads | "
                f"RAM {self.ram_limit_gib:.0f}/{self.ram_total_gib:.0f} GiB | "
                f"GPU {self.gpu_fraction:.0%} of {gpu}")


def _detect() -> Budget:
    try:
        import psutil
        ram = psutil.virtual_memory().total / 2 ** 30
    except Exception:
        ram = 0.0

    gpus: list[str] = []
    try:
        import torch
        if torch.cuda.is_available():
            gpus = [torch.cuda.get_device_properties(i).name
                    for i in range(torch.cuda.device_count())]
    except Exception:
        pass

    return Budget(
        cpu_total=cfg.CPU_TOTAL,
        n_jobs=cfg.N_JOBS,
        ram_total_gib=ram,
        ram_limit_gib=ram * cfg.RAM_FRACTION,
        gpus=gpus,
    )


BUDGET = _detect()


def claim(verbose: bool = True) -> Budget:
    """Apply the budget to this process. Call once, before importing heavy work."""
    threads = str(BUDGET.n_jobs)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, threads)

    try:
        import torch
        torch.set_num_threads(BUDGET.n_jobs)
        torch.set_num_interop_threads(max(1, BUDGET.n_jobs // 8))
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(
                    cfg.GPU_MEMORY_FRACTION, device=i)
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if verbose:
        print(f"compute budget: {BUDGET}")
    return BUDGET


def single_thread():
    """
    Context manager forcing every native thread pool to one thread.

    Use this around any fit executed inside a worker process. Setting
    OMP_NUM_THREADS and friends from inside the worker does NOT work, and the
    failure is silent and expensive: OpenMP reads its environment when the
    runtime initialises, which has already happened by the time a worker
    function runs. Estimators that thread through OpenMP rather than through an
    n_jobs argument - HistGradientBoosting above all, plus anything reaching
    BLAS, such as MLP - therefore keep spawning one thread per core inside
    every worker.

    Measured cost of getting this wrong: with 54 workers each spawning ~64
    OpenMP threads (3,456 threads on 64 cores), a HistGradientBoosting fit that
    takes 1.46 s alone measured 246 s - a 160x slowdown, with mean, median and
    p95 all equal, which is the signature of contention rather than of
    expensive hyper-parameters.

    threadpoolctl reaches the pools at runtime through their C interfaces, so
    it works regardless of import order. Estimators that expose n_jobs or
    thread_count should still be passed 1 as well; this catches the ones that
    do not.
    """
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=1)


def inner_jobs(n_parallel: int) -> int:
    """
    Threads per worker when running `n_parallel` jobs at once.

    Keeps total demand at the CPU budget instead of n_parallel times it.
    """
    return max(1, BUDGET.n_jobs // max(1, n_parallel))


def devices() -> list[str]:
    """Torch device strings for every visible GPU, or CPU if there are none."""
    return [f"cuda:{i}" for i in range(len(BUDGET.gpus))] or ["cpu"]


def device_for(index: int) -> str:
    """Round-robin a unit of work across the available GPUs."""
    d = devices()
    return d[index % len(d)]
