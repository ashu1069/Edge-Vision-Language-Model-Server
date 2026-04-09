#!/usr/bin/env python3
"""
Latency benchmark for Edge VLM Server components.

Measures YOLO detection and VLM inference latency with proper
warmup, statistical analysis (mean, P50, P95, P99), and device info.

Usage:
    # YOLO-only benchmark (fast, no model download)
    python benchmark.py --yolo

    # VLM-only benchmark (requires model download)
    python benchmark.py --vlm

    # Full benchmark (both)
    python benchmark.py --all

    # Customize runs and model
    python benchmark.py --all --yolo-runs 100 --vlm-runs 10
    python benchmark.py --vlm --vlm-model HuggingFaceTB/SmolVLM-Instruct
"""

import argparse
import base64
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_image() -> str:
    """Load test.jpg as base64, or generate a synthetic 640x480 image."""
    test_path = Path(__file__).parent / "test.jpg"
    if test_path.exists():
        with open(test_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    # Fallback: synthetic image
    import numpy as np
    from PIL import Image
    import io

    img = Image.fromarray(
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def compute_stats(times: list[float]) -> dict:
    """Compute latency statistics from a list of durations (seconds)."""
    times_ms = [t * 1000 for t in times]
    times_ms.sort()
    n = len(times_ms)
    return {
        "runs": n,
        "mean_ms": round(statistics.mean(times_ms), 2),
        "median_ms": round(statistics.median(times_ms), 2),
        "p95_ms": round(times_ms[int(n * 0.95)], 2) if n >= 20 else None,
        "p99_ms": round(times_ms[int(n * 0.99)], 2) if n >= 100 else None,
        "min_ms": round(min(times_ms), 2),
        "max_ms": round(max(times_ms), 2),
        "std_ms": round(statistics.stdev(times_ms), 2) if n > 1 else 0,
    }


def get_system_info() -> dict:
    """Collect hardware and software info for reproducibility."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
    }

    # CPU brand
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            info["cpu"] = result.stdout.strip()
    except FileNotFoundError:
        pass

    # RAM
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            info["ram_gb"] = round(int(result.stdout.strip()) / (1024**3), 1)
    except (FileNotFoundError, ValueError):
        pass

    # GPU / device
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["device"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_mem / (1024**3), 1)
        elif torch.backends.mps.is_available():
            info["device"] = "Apple Silicon (MPS)"
        else:
            info["device"] = "CPU"
    except ImportError:
        info["device"] = "CPU (torch not installed)"

    # Ultralytics version
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass

    # Transformers version
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass

    return info


def print_table(title: str, stats: dict):
    """Print a formatted stats table."""
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")
    print(f"  {'Runs':<12} {stats['runs']}")
    print(f"  {'Mean':<12} {stats['mean_ms']:.2f} ms")
    print(f"  {'Median':<12} {stats['median_ms']:.2f} ms")
    if stats['p95_ms'] is not None:
        print(f"  {'P95':<12} {stats['p95_ms']:.2f} ms")
    if stats['p99_ms'] is not None:
        print(f"  {'P99':<12} {stats['p99_ms']:.2f} ms")
    print(f"  {'Min':<12} {stats['min_ms']:.2f} ms")
    print(f"  {'Max':<12} {stats['max_ms']:.2f} ms")
    print(f"  {'Std Dev':<12} {stats['std_ms']:.2f} ms")
    print(f"{'=' * 55}")


# ---------------------------------------------------------------------------
# YOLO Benchmark
# ---------------------------------------------------------------------------

def benchmark_yolo(
    image_b64: str,
    model_name: str = "yolov8n.pt",
    warmup: int = 5,
    runs: int = 50,
    conf: float = 0.5,
) -> dict:
    """Benchmark YOLOv8 inference latency."""
    from app.vision import VisionModel

    print(f"\n[YOLO] Loading model: {model_name}")
    model = VisionModel(model_name=model_name)

    # Warmup (not measured)
    print(f"[YOLO] Warming up ({warmup} runs)...")
    for _ in range(warmup):
        model.predict(image_b64, conf_threshold=conf)

    # Timed runs
    print(f"[YOLO] Benchmarking ({runs} runs)...")
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        result = model.predict(image_b64, conf_threshold=conf)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        if i == 0:
            det_count = result.get("count", 0)
            print(f"[YOLO] First run: {det_count} detections in {elapsed*1000:.1f}ms")

    stats = compute_stats(times)
    stats["model"] = model_name
    return stats


# ---------------------------------------------------------------------------
# VLM Benchmark
# ---------------------------------------------------------------------------

def benchmark_vlm(
    image_b64: str,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    warmup: int = 2,
    runs: int = 5,
    prompt: str = "Describe this image briefly.",
    max_new_tokens: int = 50,
) -> dict:
    """Benchmark VLM inference latency."""
    from app.vlm import VLMModel

    print(f"\n[VLM] Loading model: {model_name}")
    print("[VLM] This may take a while on first run (model download)...")
    model = VLMModel(model_name=model_name, lazy_load=False)

    # Warmup
    print(f"[VLM] Warming up ({warmup} runs)...")
    for _ in range(warmup):
        model.predict(image_b64, prompt=prompt, max_new_tokens=max_new_tokens)

    # Timed runs
    print(f"[VLM] Benchmarking ({runs} runs)...")
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        result = model.predict(
            image_b64, prompt=prompt, max_new_tokens=max_new_tokens
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        if i == 0:
            resp = result.get("response", result.get("error", ""))
            preview = resp[:80] + "..." if len(resp) > 80 else resp
            print(f"[VLM] First run: {elapsed*1000:.0f}ms — \"{preview}\"")

    stats = compute_stats(times)
    stats["model"] = model_name
    stats["max_new_tokens"] = max_new_tokens
    return stats


# ---------------------------------------------------------------------------
# End-to-end pipeline benchmark
# ---------------------------------------------------------------------------

def benchmark_pipeline(
    image_b64: str,
    yolo_model: str = "yolov8n.pt",
    vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct",
    warmup: int = 2,
    runs: int = 5,
) -> dict:
    """Benchmark the full YOLO → VLM pipeline (DETECTION_AND_VLM task)."""
    from app.vision import VisionModel
    from app.vlm import VLMModel

    print(f"\n[Pipeline] Loading YOLO: {yolo_model}")
    yolo = VisionModel(model_name=yolo_model)
    print(f"[Pipeline] Loading VLM: {vlm_model}")
    vlm = VLMModel(model_name=vlm_model, lazy_load=False)

    prompt = "Describe what is happening in this scene."

    # Warmup
    print(f"[Pipeline] Warming up ({warmup} runs)...")
    for _ in range(warmup):
        det = yolo.predict(image_b64, conf_threshold=0.5)
        vlm.predict(image_b64, prompt=prompt, max_new_tokens=50, detection_context=det)

    # Timed runs
    print(f"[Pipeline] Benchmarking ({runs} runs)...")
    times = []
    yolo_times = []
    vlm_times = []
    for _ in range(runs):
        t0 = time.perf_counter()

        t_yolo = time.perf_counter()
        det = yolo.predict(image_b64, conf_threshold=0.5)
        yolo_elapsed = time.perf_counter() - t_yolo

        t_vlm = time.perf_counter()
        vlm.predict(
            image_b64, prompt=prompt, max_new_tokens=50, detection_context=det
        )
        vlm_elapsed = time.perf_counter() - t_vlm

        total = time.perf_counter() - t0
        times.append(total)
        yolo_times.append(yolo_elapsed)
        vlm_times.append(vlm_elapsed)

    stats = compute_stats(times)
    stats["yolo_mean_ms"] = round(statistics.mean(yolo_times) * 1000, 2)
    stats["vlm_mean_ms"] = round(statistics.mean(vlm_times) * 1000, 2)
    return stats


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    system: dict,
    yolo_stats: dict | None = None,
    vlm_stats: dict | None = None,
    pipeline_stats: dict | None = None,
):
    """Generate a markdown-friendly benchmark report."""
    report_lines = [
        "",
        "## Benchmark Report",
        "",
        "### System",
        f"- **CPU:** {system.get('cpu', system.get('processor', 'unknown'))}",
        f"- **RAM:** {system.get('ram_gb', '?')} GB",
        f"- **Device:** {system.get('device', 'unknown')}",
        f"- **Python:** {system.get('python', '?')}",
    ]
    if "torch_version" in system:
        report_lines.append(f"- **PyTorch:** {system['torch_version']}")
    if "ultralytics" in system:
        report_lines.append(f"- **Ultralytics:** {system['ultralytics']}")

    if yolo_stats:
        report_lines += [
            "",
            f"### YOLOv8 Detection ({yolo_stats['model']})",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean | {yolo_stats['mean_ms']:.2f} ms |",
            f"| Median (P50) | {yolo_stats['median_ms']:.2f} ms |",
        ]
        if yolo_stats['p95_ms'] is not None:
            report_lines.append(
                f"| P95 | {yolo_stats['p95_ms']:.2f} ms |"
            )
        report_lines += [
            f"| Min | {yolo_stats['min_ms']:.2f} ms |",
            f"| Max | {yolo_stats['max_ms']:.2f} ms |",
            f"| Runs | {yolo_stats['runs']} |",
        ]

    if vlm_stats:
        report_lines += [
            "",
            f"### VLM Inference ({vlm_stats['model']})",
            f"*max_new_tokens={vlm_stats.get('max_new_tokens', 50)}*",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean | {vlm_stats['mean_ms']:.0f} ms |",
            f"| Median (P50) | {vlm_stats['median_ms']:.0f} ms |",
            f"| Min | {vlm_stats['min_ms']:.0f} ms |",
            f"| Max | {vlm_stats['max_ms']:.0f} ms |",
            f"| Runs | {vlm_stats['runs']} |",
        ]

    if pipeline_stats:
        report_lines += [
            "",
            "### End-to-End Pipeline (YOLO + VLM)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Mean | {pipeline_stats['mean_ms']:.0f} ms |",
            f"| YOLO Mean | {pipeline_stats['yolo_mean_ms']:.2f} ms |",
            f"| VLM Mean | {pipeline_stats['vlm_mean_ms']:.0f} ms |",
            f"| Median (P50) | {pipeline_stats['median_ms']:.0f} ms |",
            f"| Runs | {pipeline_stats['runs']} |",
        ]

    report = "\n".join(report_lines)
    print(report)

    # Save to file
    out_path = Path(__file__).parent / "BENCHMARKS.md"
    with open(out_path, "w") as f:
        f.write(f"# Edge VLM Server — Benchmarks\n\n")
        f.write(f"*Auto-generated by `benchmark.py`*\n")
        f.write(report)
        f.write("\n")
    print(f"\nReport saved to {out_path}")

    # Also save raw JSON
    raw = {
        "system": system,
        "yolo": yolo_stats,
        "vlm": vlm_stats,
        "pipeline": pipeline_stats,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    json_path = Path(__file__).parent / "benchmarks.json"
    with open(json_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"Raw data saved to {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Edge VLM Server latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--yolo", action="store_true", help="Benchmark YOLO detection"
    )
    parser.add_argument(
        "--vlm", action="store_true", help="Benchmark VLM inference"
    )
    parser.add_argument(
        "--pipeline", action="store_true", help="Benchmark end-to-end pipeline"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all benchmarks"
    )
    parser.add_argument(
        "--yolo-model", default="yolov8n.pt", help="YOLO model (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--vlm-model", default=None,
        help="VLM model (default: env VLM_MODEL or Qwen/Qwen2-VL-2B-Instruct)",
    )
    parser.add_argument(
        "--yolo-runs", type=int, default=50, help="YOLO benchmark iterations"
    )
    parser.add_argument(
        "--vlm-runs", type=int, default=5, help="VLM benchmark iterations"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=50,
        help="VLM max tokens to generate per run",
    )

    args = parser.parse_args()

    if not any([args.yolo, args.vlm, args.pipeline, args.all]):
        parser.print_help()
        print("\nError: specify at least one of --yolo, --vlm, --pipeline, or --all")
        sys.exit(1)

    run_yolo = args.yolo or args.all
    run_vlm = args.vlm or args.all
    run_pipeline = args.pipeline or args.all

    vlm_model = args.vlm_model or os.getenv(
        "VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct"
    )

    # Collect system info
    print("Collecting system info...")
    system = get_system_info()
    print(f"  Device: {system.get('device', 'unknown')}")
    print(f"  CPU: {system.get('cpu', system.get('processor', 'unknown'))}")

    # Load test image
    print("Loading test image...")
    image_b64 = load_test_image()

    yolo_stats = None
    vlm_stats = None
    pipeline_stats = None

    if run_yolo:
        yolo_stats = benchmark_yolo(
            image_b64,
            model_name=args.yolo_model,
            runs=args.yolo_runs,
        )
        print_table(f"YOLO ({args.yolo_model})", yolo_stats)

    if run_vlm:
        vlm_stats = benchmark_vlm(
            image_b64,
            model_name=vlm_model,
            runs=args.vlm_runs,
            max_new_tokens=args.max_new_tokens,
        )
        print_table(f"VLM ({vlm_model})", vlm_stats)

    if run_pipeline:
        pipeline_stats = benchmark_pipeline(
            image_b64,
            yolo_model=args.yolo_model,
            vlm_model=vlm_model,
            runs=args.vlm_runs,
        )
        print_table("End-to-End Pipeline", pipeline_stats)

    # Generate report
    generate_report(system, yolo_stats, vlm_stats, pipeline_stats)


if __name__ == "__main__":
    main()
