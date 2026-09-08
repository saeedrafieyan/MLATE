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
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=1)


def inner_jobs(n_parallel: int) -> int:
    return max(1, BUDGET.n_jobs // max(1, n_parallel))


def devices() -> list[str]:
    return [f"cuda:{i}" for i in range(len(BUDGET.gpus))] or ["cpu"]


def device_for(index: int) -> str:
    d = devices()
    return d[index % len(d)]
