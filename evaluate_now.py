"""
Evaluation script for DECA models on the NoW Dataset.
Runs Docker-based official evaluation (noweval) on pre-computed predictions
and parses 3D reconstruction errors (Mean and Median in mm).

Usage:
    # Run evaluation on predictions directory:
    python evaluate_now.py --predictions_dir /path/to/predictions/
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional


def is_running_in_wsl() -> bool:
    """
    Checks if current environment is running inside Windows Subsystem for Linux (WSL).
    """
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/version", "r", encoding="utf-8") as f:
                content = f.read().lower()
                return "microsoft" in content or "wsl" in content
        except Exception:
            return False
    return False


def to_docker_mount_path(path: str, docker_is_windows_exe: bool) -> str:
    """
    Converts path to the appropriate format expected by Docker volume mounts.
    If running inside WSL2 and invoking Windows host docker.exe, converts /mnt/c/... to C:\\...
    """
    abs_path = os.path.abspath(path)
    if docker_is_windows_exe and is_running_in_wsl():
        try:
            proc = subprocess.run(["wslpath", "-w", abs_path], capture_output=True, text=True, check=True)
            win_path = proc.stdout.strip()
            if win_path:
                return win_path
        except Exception as e:
            print(f"[WARN] Failed to convert path using wslpath -w: {e}")
    return abs_path


def find_docker_executable(custom_docker_path: Optional[str] = None) -> Tuple[str, bool]:
    """
    Finds the appropriate docker executable.
    Returns (docker_executable_path, is_windows_exe).
    """
    if custom_docker_path:
        is_win = custom_docker_path.lower().endswith(".exe")
        return custom_docker_path, is_win

    # 1. If not running in WSL (e.g. native Windows or native Linux)
    if not is_running_in_wsl():
        docker_cmd = shutil.which("docker") or shutil.which("docker.exe")
        if docker_cmd:
            return docker_cmd, docker_cmd.lower().endswith(".exe")
        return "docker", False

    # 2. Running inside WSL2:
    linux_docker = shutil.which("docker")
    if linux_docker:
        try:
            res = subprocess.run([linux_docker, "info"], capture_output=True, text=True, timeout=3)
            if res.returncode == 0:
                return linux_docker, False
        except Exception:
            pass

    windows_docker = shutil.which("docker.exe")
    if windows_docker:
        return windows_docker, True

    common_win_docker_paths = [
        "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe",
        "/mnt/c/Program Files/Docker/Docker/docker.exe"
    ]
    for p in common_win_docker_paths:
        if os.path.exists(p):
            return p, True

    return "docker", False


def resolve_now_paths(dataset_path: str) -> Tuple[str, str]:
    """
    Resolves both:
      1) now_root_dir: directory containing scans/, scans_lmks_onlypp/, and imagepaths*.txt (mounted to /dataset in Docker)
      2) images_dir: directory containing the subject folders (e.g. FaMoS_*)
    """
    path_abs = os.path.abspath(dataset_path)

    subdirs = [d for d in os.listdir(path_abs) if os.path.isdir(os.path.join(path_abs, d))] if os.path.isdir(path_abs) else []
    has_subject_dirs = any(d.startswith("FaMoS_") for d in subdirs)

    standard_iphone_subpath = os.path.join(path_abs, "NoW_Dataset", "final_release_version", "iphone_pictures")

    if has_subject_dirs:
        images_dir = path_abs
    elif os.path.isdir(standard_iphone_subpath):
        images_dir = standard_iphone_subpath
    else:
        found_img_dir = None
        for root, dirs, _ in os.walk(path_abs):
            if os.path.basename(root) == "iphone_pictures" or any(d.startswith("FaMoS_") for d in dirs):
                found_img_dir = root
                break
        images_dir = found_img_dir if found_img_dir else path_abs

    curr = path_abs
    now_root_dir = path_abs
    while curr:
        if os.path.isdir(os.path.join(curr, "scans")):
            now_root_dir = curr
            break
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent

    return now_root_dir, images_dir


def run_docker_evaluation(dataset_abs: str, model_output_abs: str, custom_docker_path: Optional[str] = None) -> None:
    """
    Runs NoW evaluation Docker container:
        docker run --ipc host --rm -v <dataset_mount>:/dataset -v <model_output_mount>:/preds noweval
    Handles WSL2 to Windows Host docker.exe bridging and path translation.
    """
    docker_bin, is_win_exe = find_docker_executable(custom_docker_path)
    mount_dataset = to_docker_mount_path(dataset_abs, docker_is_windows_exe=is_win_exe)
    mount_output = to_docker_mount_path(model_output_abs, docker_is_windows_exe=is_win_exe)

    cmd = [
        docker_bin, "run", "--ipc", "host", "--rm",
        "-v", f"{mount_dataset}:/dataset",
        "-v", f"{mount_output}:/preds",
        "noweval"
    ]
    cmd_str = " ".join(cmd)
    print(f"\n[INFO] Executing Docker evaluation command via {docker_bin}:\n  {cmd_str}\n")

    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print("[INFO] Docker evaluation output:")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Docker evaluation failed with return code {e.returncode}:")
        if e.stdout:
            print(f"Stdout:\n{e.stdout}")
        if e.stderr:
            print(f"Stderr:\n{e.stderr}")
        print("\n[TROUBLESHOOTING TIP] If Docker Desktop on Windows is not running, please start it.\n")
        raise e
    except FileNotFoundError:
        print(f"[ERROR] Docker executable '{docker_bin}' not found on system PATH.")
        raise


def evaluation_results_exist(model_output_dir: str) -> bool:
    """
    Checks if evaluation metric files (.npy or .meanmedian) already exist inside model_output_dir.
    """
    model_output_abs = os.path.abspath(model_output_dir)
    npy_files = glob.glob(os.path.join(model_output_abs, "**", "*.npy"), recursive=True)
    meanmedian_files = glob.glob(os.path.join(model_output_abs, "**", "*.meanmedian"), recursive=True)
    return len(npy_files) > 0 or len(meanmedian_files) > 0


def extract_metrics(model_output_dir: str) -> Tuple[float, float]:
    """
    Locates generated evaluation files inside model_output_dir.
    Returns (mean_error, median_error).
    """
    import numpy as np

    model_output_abs = os.path.abspath(model_output_dir)
    npy_files = glob.glob(os.path.join(model_output_abs, "**", "*.npy"), recursive=True)

    if not npy_files:
        raise FileNotFoundError(f"No .npy evaluation metric files found in {model_output_abs}")

    distances_file = None
    for f in npy_files:
        if "distance" in os.path.basename(f).lower():
            distances_file = f
            break
    if distances_file is None:
        distances_file = npy_files[0]

    print(f"[INFO] Reading evaluation metrics from: {distances_file}")
    data = np.load(distances_file, allow_pickle=True)

    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        content = data.item()
    else:
        content = data

    if isinstance(content, dict):
        if "mean" in content and "median" in content:
            return float(content["mean"]), float(content["median"])

        if "computed_distances" in content:
            raw_dists = content["computed_distances"]
            if isinstance(raw_dists, list):
                all_dists = np.concatenate([np.asarray(x).ravel() for x in raw_dists if len(x) > 0])
            else:
                all_dists = np.asarray(raw_dists).ravel()
            return float(np.mean(all_dists)), float(np.median(all_dists))

    if isinstance(content, np.ndarray) and content.size > 0:
        all_dists = content.ravel()
        return float(np.mean(all_dists)), float(np.median(all_dists))

    raise ValueError(f"Could not parse mean and median error metrics from {distances_file}")


def append_summary(summary_file: str, line: str):
    """
    Appends a single line to the summary file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(summary_file)), exist_ok=True)
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pre-computed predictions on the NoW Dataset.")
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Path to directory containing predicted .obj and .txt files (or folder of step_* predictions).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/NoW/",
        help="Path to NoW dataset root directory. Default: 'datasets/NoW/'.",
    )
    parser.add_argument(
        "--docker_executable",
        type=str,
        default=None,
        help="Optional explicit path to docker binary.",
    )
    parser.add_argument(
        "--re_evaluate",
        action="store_true",
        help="If set, re-runs Docker evaluation even if result files already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    preds_abs = os.path.abspath(args.predictions_dir)

    if not os.path.isdir(preds_abs):
        raise NotADirectoryError(f"Predictions directory does not exist: {preds_abs}")

    now_root_abs, _ = resolve_now_paths(args.dataset_path)

    # Check if predictions_dir contains multiple subdirectories (e.g. step_00001000, step_00002000)
    subdirs = [os.path.join(preds_abs, d) for d in os.listdir(preds_abs) if os.path.isdir(os.path.join(preds_abs, d))]
    has_step_subdirs = any(os.path.basename(d).startswith("step_") for d in subdirs)

    target_dirs = [d for d in subdirs if os.path.basename(d).startswith("step_")] if has_step_subdirs else [preds_abs]

    print("=" * 70)
    print("NoW Dataset Docker Evaluation Pipeline")
    print(f"Predictions Target(s): {len(target_dirs)}")
    print(f"Dataset root: {now_root_abs}")
    print("=" * 70)

    summary_file = os.path.join(preds_abs, "evaluation_summary.txt")
    records: Dict[str, Dict[str, float]] = {}

    for target_dir in target_dirs:
        dir_name = os.path.basename(target_dir)
        print(f"\nEvaluating: {dir_name}")

        if evaluation_results_exist(target_dir) and not args.re_evaluate:
            print(f"[INFO] Evaluation results already exist for {dir_name}. Skipping Docker evaluation.")
        else:
            try:
                run_docker_evaluation(
                    dataset_abs=now_root_abs,
                    model_output_abs=target_dir,
                    custom_docker_path=args.docker_executable,
                )
            except subprocess.CalledProcessError:
                print(f"[WARNING] Docker evaluation failed for '{dir_name}'. Skipping.")
                continue

        try:
            mean_err, median_err = extract_metrics(target_dir)
            records[dir_name] = {"mean": mean_err, "median": median_err}
            print(f"[RESULT] {dir_name} -> Mean: {mean_err:.4f} mm, Median: {median_err:.4f} mm")
            append_summary(summary_file, f"{dir_name}: Mean = {mean_err:.4f} mm, Median = {median_err:.4f} mm")
        except Exception as e:
            print(f"[WARNING] Could not extract metrics for '{dir_name}': {e}")

    if records:
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY RESULTS")
        print("=" * 70)
        for name, metrics in records.items():
            print(f"{name:<25} Mean: {metrics['mean']:.4f} mm | Median: {metrics['median']:.4f} mm")


if __name__ == "__main__":
    main()
