"""RAD50 card-free, explicit-assumption critical-path model.

The existing RAD42 path overlaps CPU combining chunk i+1 with the engine GPU
pulling and merging chunk i. A CPU *copy after combine* can serialize the CPU
producer: raw PCIe GB/s alone cannot justify this optimization.

This deliberately is NOT a calibrated performance prediction. Its purpose is
to expose the unknown handoff/wakeup/CPU-store overhead and eliminate designs
that cannot win even with optimistic assumptions. Every run prints inputs.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, replace
from math import ceil, isfinite


@dataclass(frozen=True)
class Inputs:
    rows: int = 2048
    hidden: int = 2560
    chunk_rows: int = 256
    element_bytes: int = 2
    gpu_pull_gbps: float = 6.8  # RAD40: XTX1 *under* XTX0 host-read contention
    cpu_push_gbps: float = 10.3  # RAD40: XTX1 CPU BAR push, same contention
    producer_first_us: float = 0.0  # unmeasured assumptions below
    producer_interval_us: float = 100.0
    combine_us_per_chunk: float = 150.0
    gpu_add_us_per_chunk: float = 30.0
    gpu_pull_setup_us: float = 0.0
    cpu_push_setup_us: float = 0.0
    gpu_local_read_us_per_chunk: float = 0.0
    signal_us_per_chunk: float = 0.0
    cpu_wake_us_per_chunk: float = 0.0
    direct_output_extra_us_per_chunk: float = 0.0
    # An illustrative lower-bound for replacing CPU stores into host memory
    # with direct stores to GPU VRAM; it is NOT derived from copy bandwidth.
    # Explicitly calibrate direct_output_extra_us_per_chunk on real hardware.

    def validate(self):
        if min(self.rows, self.hidden, self.chunk_rows, self.element_bytes) <= 0:
            raise ValueError("all shapes/widths must be positive")
        if self.rows > 4096 or self.hidden > 65536:
            raise ValueError("outside the experimental bounds")
        if self.gpu_pull_gbps <= 0 or self.cpu_push_gbps <= 0:
            raise ValueError("bandwidth must be positive")
        for name in (
            "gpu_pull_gbps", "cpu_push_gbps", "producer_first_us",
            "producer_interval_us", "combine_us_per_chunk", "gpu_add_us_per_chunk",
            "gpu_pull_setup_us", "cpu_push_setup_us", "gpu_local_read_us_per_chunk",
            "signal_us_per_chunk", "cpu_wake_us_per_chunk",
            "direct_output_extra_us_per_chunk"
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def sizes(self):
        self.validate()
        return [
            min(self.chunk_rows, self.rows - i) * self.hidden * self.element_bytes
            for i in range(0, self.rows, self.chunk_rows)
        ]


def bandwidth_us(size_bytes: int, gbps: float) -> float:
    if size_bytes < 0 or not isfinite(gbps) or gbps <= 0:
        raise ValueError("bad bytes or bandwidth")
    return size_bytes / (gbps * 1000.0)


@dataclass(frozen=True)
class Result:
    first_chunk_us: float
    makespan_us: float
    cpu_finish_us: float
    chunks: int


def simulate(x: Inputs, mode: str) -> Result:
    """Baseline = GPU pull; copy = combine then CPU copy; direct = CPU combine
    stores directly into VRAM with independently measured *extra* CPU time.

    CPU combine and the subsequent copy share the same producer thread in
    copy mode. This is conservative for a threaded copy pipeline, not a bound
    on a separately offloaded CPU copy. The GPU add for each chunk is serialized
    in the single consumer stream, but overlaps the *next* CPU combine.
    """
    if mode not in ("baseline", "copy", "direct"):
        raise ValueError("invalid mode")
    chunks = x.sizes()
    cpu_time = 0.0
    gpu_time = 0.0
    first = None
    for i, size in enumerate(chunks):
        gpu_ready = x.producer_first_us + i * x.producer_interval_us
        cpu_time = max(cpu_time, gpu_ready)
        cpu_time += x.cpu_wake_us_per_chunk + x.combine_us_per_chunk
        if mode == "baseline":
            ready = cpu_time
            gpu_service = (
                bandwidth_us(size, x.gpu_pull_gbps) +
                x.gpu_pull_setup_us + x.gpu_add_us_per_chunk
            )
        else:
            if mode == "copy":
                cpu_time += bandwidth_us(size, x.cpu_push_gbps) + x.cpu_push_setup_us
            else:
                cpu_time += x.direct_output_extra_us_per_chunk
            cpu_time += x.signal_us_per_chunk
            ready = cpu_time
            gpu_service = x.gpu_local_read_us_per_chunk + x.gpu_add_us_per_chunk
        gpu_time = max(gpu_time, ready) + gpu_service
        if first is None:
            first = gpu_time
    return Result(first if first is not None else 0.0, gpu_time,
                  cpu_time, len(chunks))


def break_even_signal_us(x: Inputs, mode: str, tolerance_us: float = 1e-4) -> float:
    """Maximum additional signalling cost per chunk that could break even.
    Negative => candidate already loses at zero additional signal cost.
    This is an analytic *scenario threshold*, NOT a measured tolerance.
    """
    if mode not in ("copy", "direct"):
        raise ValueError("mode must be copy or direct")
    base = simulate(x, "baseline").makespan_us
    zero = replace(x, signal_us_per_chunk=0.0)
    if simulate(zero, mode).makespan_us >= base:
        return -1.0
    lo, hi = 0.0, max(base, 1.0)
    # At this bound even one positive signal must exhaust the entire
    # baseline makespan, so the candidate necessarily loses.
    for _ in range(55):
        mid = (lo + hi) / 2.0
        if simulate(replace(zero, signal_us_per_chunk=mid), mode).makespan_us < base:
            lo = mid
        else:
            hi = mid
        if hi - lo < tolerance_us:
            break
    return (lo + hi) / 2.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--chunk-rows", type=int, default=256)
    p.add_argument("--pull-gbps", type=float, default=6.8)
    p.add_argument("--push-gbps", type=float, default=10.3)
    p.add_argument("--combine-us", type=float, default=150.0)
    p.add_argument("--producer-interval-us", type=float, default=100.0)
    p.add_argument("--gpu-add-us", type=float, default=30.0)
    p.add_argument("--cpu-wake-us", type=float, default=0.0)
    p.add_argument("--signal-us", type=float, default=0.0)
    p.add_argument("--copy-setup-us", type=float, default=0.0)
    p.add_argument("--direct-extra-us", type=float, default=0.0)
    a = p.parse_args()
    x = Inputs(rows=a.rows, chunk_rows=a.chunk_rows,
               gpu_pull_gbps=a.pull_gbps, cpu_push_gbps=a.push_gbps,
               combine_us_per_chunk=a.combine_us,
               producer_interval_us=a.producer_interval_us,
               gpu_add_us_per_chunk=a.gpu_add_us,
               cpu_wake_us_per_chunk=a.cpu_wake_us,
               signal_us_per_chunk=a.signal_us,
               cpu_push_setup_us=a.copy_setup_us,
               direct_output_extra_us_per_chunk=a.direct_extra_us)
    x.validate()
    print("MODELLING ONLY — no hardware measurements from this run.")
    print(f"inputs={x!r}")
    base = simulate(x, "baseline")
    print(f"baseline first={base.first_chunk_us:.1f}us makespan={base.makespan_us:.1f}us")
    for mode in ("copy", "direct"):
        r = simulate(x, mode)
        gain = base.makespan_us - r.makespan_us
        print(f"{mode:8} first={r.first_chunk_us:.1f}us "
              f"makespan={r.makespan_us:.1f}us "
              f"delta={gain:+.1f}us ({100*gain/base.makespan_us:+.1f}%) "
              f"max_extra_signal={break_even_signal_us(x, mode):.1f}us/chunk")
    print("CAUTION: direct mode assumes independently calibrated extra CPU "
          "VRAM-store cost; zero is an optimistic placeholder. The baseline "
          "already overlaps CPU combine and GPU pull.")


if __name__ == "__main__":
    main()
