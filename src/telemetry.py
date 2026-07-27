"""
GPU telemetry sampler (pynvml) — Month-2 "monitoring" task.

Use as a context manager around a workload to capture VRAM / temperature / power /
utilization over time, then dump a CSV you can chart (or later feed to Grafana).

    from telemetry import GpuTelemetry
    with GpuTelemetry("outputs/telemetry.csv") as t:
        ...run generations...
    print(t.summary())

Standalone smoke test:  python src/telemetry.py --seconds 10
"""
from __future__ import annotations

import argparse
import csv
import threading
import time
from pathlib import Path


class GpuTelemetry:
    def __init__(self, csv_path: str, index: int = 0, interval_s: float = 0.5):
        self.csv_path = Path(csv_path)
        self.index = index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rows: list[dict] = []

    def _sample_loop(self, handle, pynvml):
        t0 = time.time()
        while not self._stop.is_set():
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = -1
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                power_w = -1
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except Exception:
                util = -1
            self._rows.append({
                "t_s": round(time.time() - t0, 2),
                "vram_used_gb": round(mem.used / 1e9, 3),
                "temp_c": temp,
                "power_w": round(power_w, 1),
                "util_pct": util,
            })
            self._stop.wait(self.interval_s)

    def __enter__(self):
        import pynvml
        pynvml.nvmlInit()
        self._pynvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        self._thread = threading.Thread(
            target=self._sample_loop, args=(self._handle, pynvml), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._pynvml.nvmlShutdown()
        if self._rows:
            self.csv_path.parent.mkdir(exist_ok=True, parents=True)
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
                w.writeheader()
                w.writerows(self._rows)

    def summary(self) -> dict:
        if not self._rows:
            return {}
        peak = max(r["vram_used_gb"] for r in self._rows)
        tmax = max((r["temp_c"] for r in self._rows if r["temp_c"] >= 0), default=-1)
        pmax = max((r["power_w"] for r in self._rows if r["power_w"] >= 0), default=-1)
        return {"samples": len(self._rows), "peak_vram_gb": peak,
                "max_temp_c": tmax, "max_power_w": pmax, "csv": str(self.csv_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample GPU telemetry for N seconds")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--out", default="outputs/telemetry.csv")
    args = ap.parse_args()
    with GpuTelemetry(args.out) as t:
        time.sleep(args.seconds)
    print(t.summary())


if __name__ == "__main__":
    main()
