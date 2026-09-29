"""Long-Run Stability & Memory Leak Test Suite
=============================================
Runs UnifiedRetailPipeline continuously, measuring:
- Process RAM (MB)
- CPU Usage (%)
- Capture FPS and AI FPS
- Inference latency (ms)
- Queue sizes and reconnect counts
Outputs a structured summary verifying that memory does not leak.
"""

import argparse
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mivolo_system"))

import numpy as np
import psutil

from services.unified_pipeline import UnifiedRetailPipeline


def run_stability_test(
    duration_seconds: float,
    video_path: str = "test_videos/Outlet_Sample_2_Banasree.avi",
    enable_demographics: bool = False,
):
    print("=" * 70)
    print(f"  Starting Long-Run Stability Test for {duration_seconds:.0f} seconds ({duration_seconds/60:.1f} min)")
    print(f"  Target Source: {video_path}")
    print(f"  Demographics:  {'Enabled' if enable_demographics else 'Disabled'}")
    print("=" * 70)

    proc = psutil.Process()
    ram_startup = proc.memory_info().rss / (1024.0 * 1024.0)

    pipeline = UnifiedRetailPipeline(
        video_source=video_path,
        display=False,
        enable_demographics=enable_demographics,
    )

    pipe_thread = threading.Thread(target=pipeline.run, daemon=True)
    pipe_thread.start()

    time.sleep(2.0)  # Warmup

    samples = []
    t_start = time.time()
    next_sample_time = t_start

    print(f"{'Elapsed':<10} | {'RAM (MB)':<10} | {'CPU (%)':<8} | {'CAM FPS':<9} | {'AI FPS':<8} | {'Latency':<8} | {'Active Trk':<10}")
    print("-" * 75)

    try:
        while (time.time() - t_start) < duration_seconds:
            now = time.time()
            if now >= next_sample_time:
                next_sample_time = now + 5.0  # Sample every 5 seconds
                elapsed = now - t_start

                ram_mb = proc.memory_info().rss / (1024.0 * 1024.0)
                cpu_pct = psutil.cpu_percent(interval=None)
                cam_fps = pipeline.perf_monitor.capture_fps
                ai_fps = pipeline.perf_monitor.ai_fps
                lat_ms = pipeline.perf_monitor.inference_ms
                active_trk = len(pipeline.cached_persons)

                sample = {
                    "elapsed_sec": round(elapsed, 1),
                    "ram_mb": round(ram_mb, 1),
                    "cpu_percent": round(cpu_pct, 1),
                    "capture_fps": round(cam_fps, 1),
                    "ai_fps": round(ai_fps, 1),
                    "latency_ms": round(lat_ms, 1),
                    "active_tracks": active_trk,
                }
                samples.append(sample)

                print(f"{elapsed:8.1f}s | {ram_mb:9.1f}  | {cpu_pct:7.1f}  | {cam_fps:8.1f}  | {ai_fps:7.1f}  | {lat_ms:6.0f}ms | {active_trk:8d}")

            time.sleep(0.5)

    finally:
        print("\nStopping pipeline...")
        pipeline.stop()
        pipe_thread.join(timeout=3.0)

    # Post-run analysis
    rams = [s["ram_mb"] for s in samples]
    cpus = [s["cpu_percent"] for s in samples]
    ai_fps_list = [s["ai_fps"] for s in samples if s["ai_fps"] > 0]

    ram_final = rams[-1] if rams else ram_startup
    ram_peak = max(rams) if rams else ram_startup
    ram_drift = ram_final - rams[0] if len(rams) > 1 else 0.0

    print("=" * 70)
    print("  STABILITY TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f" Test Duration:       {duration_seconds:.1f}s")
    print(f" Startup RAM:         {ram_startup:.1f} MB")
    print(f" Initial Sample RAM:  {rams[0] if rams else 0:.1f} MB")
    print(f" Final RAM:           {ram_final:.1f} MB")
    print(f" Peak RAM:            {ram_peak:.1f} MB")
    print(f" RAM Drift (Net):     {ram_drift:+.1f} MB")
    print(f" Average CPU:         {np.mean(cpus):.1f}%")
    print(f" Average AI FPS:      {np.mean(ai_fps_list) if ai_fps_list else 0:.1f} FPS")
    print(f" Entries Recorded:    {pipeline.event_manager.entry_count}")
    print(f" Exits Recorded:      {pipeline.event_manager.exit_count}")
    print(f" Bag Exits Confirmed: {pipeline.cached_total_bag_exits}")
    print(f" RTSP Reconnects:     {pipeline.reader.reconnect_count}")
    print("=" * 70)

    # Acceptance threshold checks
    ram_limit = 2600.0 if enable_demographics else 1750.0
    assert ram_peak <= ram_limit, f"Peak RAM ({ram_peak:.1f} MB) exceeded safety limit ({ram_limit} MB)"
    # RAM drift over the run must not continuously explode (within 50MB allowance for GC cycles)
    if len(rams) > 5:
        # Check slope over last half
        half_idx = len(rams) // 2
        late_drift = rams[-1] - rams[half_idx]
        assert late_drift < 50.0, f"Memory leak detected! RAM drifted +{late_drift:.1f} MB in second half."

    print("STABILITY & LEAK CHECK: PASSED (Zero unbounded memory growth detected)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0, help="Test duration in seconds.")
    parser.add_argument("--demographics", action="store_true", help="Enable MiVOLO demographic profiling.")
    args = parser.parse_args()
    run_stability_test(args.seconds, enable_demographics=args.demographics)
