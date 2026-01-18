#!/usr/bin/env python3
"""
SIFT Benchmark Script

Compares execution time of sift-rs (CPU/GPU) vs OpenCV SIFT.
Usage: uv run benchmark.py [image_path] [--runs N]
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-python>=4.8",
#     "numpy>=1.24",
#     "rich>=13.0",
# ]
# ///

import subprocess
import time
import statistics
import argparse
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

# SIFT parameters matching sift-rs defaults
SIFT_PARAMS = {
    "nfeatures": 0,  # 0 = detect all
    "nOctaveLayers": 3,  # DEFAULT_NUM_INTERVALS
    "contrastThreshold": 0.04 / 3,  # DEFAULT_CONTRAST_THRESHOLD / intervals
    "edgeThreshold": 10.0,  # DEFAULT_EDGE_THRESHOLD
    "sigma": 1.6,  # DEFAULT_SIGMA
}


def run_opencv_sift(image_path: Path) -> dict:
    """Run OpenCV SIFT and measure time."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": f"Failed to load image: {image_path}"}

    sift = cv2.SIFT_create(
        nfeatures=SIFT_PARAMS["nfeatures"],
        nOctaveLayers=SIFT_PARAMS["nOctaveLayers"],
        contrastThreshold=SIFT_PARAMS["contrastThreshold"],
        edgeThreshold=SIFT_PARAMS["edgeThreshold"],
        sigma=SIFT_PARAMS["sigma"],
    )

    start = time.perf_counter()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    end = time.perf_counter()

    return {
        "sift_time_ms": (end - start) * 1000,
        "keypoints": len(keypoints),
        "kp_objects": keypoints,
        "descriptors": descriptors,
        "success": True,
    }


def run_sift_rs(binary_path: Path, image_path: Path, backend: str) -> dict:
    """Run sift-rs and capture timing information."""
    start = time.perf_counter()

    result = subprocess.run(
        [str(binary_path), "--backend", backend, str(image_path)],
        capture_output=True,
        text=True,
    )

    end = time.perf_counter()
    wall_time = (end - start) * 1000  # ms

    # Parse output for internal timing and keypoints
    sift_time = None
    keypoints = None

    for line in result.stdout.split("\n"):
        if "SIFT processing took:" in line:
            time_str = line.split(":")[-1].strip()
            if time_str.endswith("ms"):
                sift_time = float(time_str[:-2])
        elif "Found" in line and "keypoints." in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "Found" and i + 1 < len(parts):
                    try:
                        keypoints = int(parts[i + 1])
                    except ValueError:
                        pass

    return {
        "wall_time_ms": wall_time,
        "sift_time_ms": sift_time,
        "keypoints": keypoints,
        "success": result.returncode == 0,
        "stderr": result.stderr if result.returncode != 0 else None,
    }


def benchmark_opencv(image_path: Path, runs: int) -> dict:
    """Run multiple OpenCV benchmarks and compute statistics."""
    results = []

    for _ in track(range(runs), description="[cyan]Benchmarking OpenCV...[/cyan]"):
        result = run_opencv_sift(image_path)
        if result["success"]:
            results.append(result)

    if not results:
        return {"error": "No successful runs"}

    sift_times = [r["sift_time_ms"] for r in results]
    keypoints = results[0]["keypoints"]
    kp_objects = results[-1]["kp_objects"]
    descriptors = results[-1]["descriptors"]

    return {
        "backend": "OpenCV",
        "runs": len(results),
        "keypoints": keypoints,
        "kp_objects": kp_objects,
        "descriptors": descriptors,
        "sift_time": {
            "mean": statistics.mean(sift_times),
            "min": min(sift_times),
            "max": max(sift_times),
            "stdev": statistics.stdev(sift_times) if len(sift_times) > 1 else 0,
        },
    }


def benchmark_sift_rs(
    binary_path: Path, image_path: Path, backend: str, runs: int
) -> dict:
    """Run multiple sift-rs benchmarks and compute statistics."""
    results = []

    for _ in track(
        range(runs), description=f"[cyan]Benchmarking sift-rs ({backend})...[/cyan]"
    ):
        result = run_sift_rs(binary_path, image_path, backend)
        if result["success"] and result["sift_time_ms"]:
            results.append(result)

    if not results:
        return {"error": "No successful runs"}

    sift_times = [r["sift_time_ms"] for r in results]
    keypoints = results[0]["keypoints"]

    return {
        "backend": f"sift-rs ({backend.upper()})",
        "runs": len(results),
        "keypoints": keypoints,
        "sift_time": {
            "mean": statistics.mean(sift_times),
            "min": min(sift_times),
            "max": max(sift_times),
            "stdev": statistics.stdev(sift_times) if len(sift_times) > 1 else 0,
        },
    }


def draw_keypoints_opencv(image_path: Path, output_path: Path, keypoints) -> None:
    """Draw keypoints using OpenCV."""
    img = cv2.imread(str(image_path))
    img_with_kp = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        color=(0, 0, 255),  # Red (BGR)
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    cv2.imwrite(str(output_path), img_with_kp)
    console.print(f"[green]OpenCV keypoints saved to: {output_path}[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SIFT: sift-rs vs OpenCV"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="../data/1.jpg",
        help="Path to test image (default: ../data/1.jpg)",
    )
    parser.add_argument(
        "--runs",
        "-n",
        type=int,
        default=5,
        help="Number of benchmark runs (default: 5)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip keypoint visualization",
    )
    args = parser.parse_args()

    # Find binary
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    binary_path = project_root / "target" / "release" / "sift"

    if not binary_path.exists():
        console.print(
            "[red]Error: Binary not found. Run 'cargo build --release' first.[/red]"
        )
        return 1

    # Resolve image path
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = script_dir / image_path

    if not image_path.exists():
        console.print(f"[red]Error: Image not found: {image_path}[/red]")
        return 1

    # Get image dimensions
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    console.print(f"\n[bold]SIFT Benchmark: sift-rs vs OpenCV[/bold]")
    console.print(f"Image: {image_path.name} ({w}x{h})")
    console.print(f"Runs: {args.runs}")
    console.print(f"\n[dim]SIFT Parameters:[/dim]")
    console.print(f"  nOctaveLayers: {SIFT_PARAMS['nOctaveLayers']}")
    console.print(f"  contrastThreshold: {SIFT_PARAMS['contrastThreshold']:.4f}")
    console.print(f"  edgeThreshold: {SIFT_PARAMS['edgeThreshold']}")
    console.print(f"  sigma: {SIFT_PARAMS['sigma']}\n")

    # Run benchmarks
    results = {}

    # OpenCV
    results["opencv"] = benchmark_opencv(image_path, args.runs)

    # sift-rs CPU
    results["cpu"] = benchmark_sift_rs(binary_path, image_path, "cpu", args.runs)

    # sift-rs GPU
    results["gpu"] = benchmark_sift_rs(binary_path, image_path, "gpuv2", args.runs)

    # Display results table
    table = Table(title="Benchmark Results")
    table.add_column("Implementation", style="cyan")
    table.add_column("Keypoints", justify="right")
    table.add_column("Mean (ms)", justify="right", style="green")
    table.add_column("Min (ms)", justify="right")
    table.add_column("Max (ms)", justify="right")
    table.add_column("Stdev (ms)", justify="right")

    for key, data in results.items():
        if "error" in data:
            table.add_row(key, "-", data["error"], "-", "-", "-")
        else:
            table.add_row(
                data["backend"],
                str(data["keypoints"] or "-"),
                f"{data['sift_time']['mean']:.2f}",
                f"{data['sift_time']['min']:.2f}",
                f"{data['sift_time']['max']:.2f}",
                f"{data['sift_time']['stdev']:.2f}",
            )

    console.print(table)

    # Speedup comparisons
    console.print("\n[bold]Speedup Analysis:[/bold]")

    opencv_mean = results["opencv"]["sift_time"]["mean"] if "error" not in results["opencv"] else None
    cpu_mean = results["cpu"]["sift_time"]["mean"] if "error" not in results["cpu"] else None
    gpu_mean = results["gpu"]["sift_time"]["mean"] if "error" not in results["gpu"] else None

    if opencv_mean and cpu_mean:
        if cpu_mean < opencv_mean:
            speedup = opencv_mean / cpu_mean
            console.print(f"  sift-rs (CPU) is [green]{speedup:.2f}x faster[/green] than OpenCV")
        else:
            slowdown = cpu_mean / opencv_mean
            console.print(f"  sift-rs (CPU) is [yellow]{slowdown:.2f}x slower[/yellow] than OpenCV")

    if opencv_mean and gpu_mean:
        if gpu_mean < opencv_mean:
            speedup = opencv_mean / gpu_mean
            console.print(f"  sift-rs (GPU) is [green]{speedup:.2f}x faster[/green] than OpenCV")
        else:
            slowdown = gpu_mean / opencv_mean
            console.print(f"  sift-rs (GPU) is [yellow]{slowdown:.2f}x slower[/yellow] than OpenCV")

    if cpu_mean and gpu_mean:
        if gpu_mean < cpu_mean:
            speedup = cpu_mean / gpu_mean
            console.print(f"  sift-rs (GPU) is [green]{speedup:.2f}x faster[/green] than sift-rs (CPU)")
        else:
            slowdown = gpu_mean / cpu_mean
            console.print(f"  sift-rs (GPU) is [yellow]{slowdown:.2f}x slower[/yellow] than sift-rs (CPU)")

    # Visualize keypoints
    if not args.no_visualize and "kp_objects" in results["opencv"]:
        console.print("\n[bold]Generating visualizations...[/bold]")
        
        # OpenCV visualization
        opencv_output = script_dir / f"output_opencv.png"
        draw_keypoints_opencv(
            image_path, opencv_output, results["opencv"]["kp_objects"]
        )
        
        # sift-rs outputs are already generated by the binary
        cpu_output = project_root / "data" / "output_Cpu.png"
        gpu_output = project_root / "data" / "output_WebGpu.png"
        
        if cpu_output.exists():
            console.print(f"[green]sift-rs (CPU) keypoints: {cpu_output}[/green]")
        if gpu_output.exists():
            console.print(f"[green]sift-rs (GPU) keypoints: {gpu_output}[/green]")

    return 0


if __name__ == "__main__":
    exit(main())
