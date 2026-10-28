"""
Benchmarking script for DECA models on the NoW Dataset.
Supports cross-platform execution (WSL2 Linux DECA inference + Windows Host Docker noweval evaluation),
as well as decoupled stages (--mode predict_only, eval_only, or all).

Usage:
    # Full end-to-end run from WSL2 or Windows:
    python evaluate_now.py --model_path /path/to/checkpoint.tar

    # Run only DECA inference (e.g. inside WSL2 without Docker needed):
    python evaluate_now.py --model_path /path/to/checkpoints_dir/ --mode predict_only

    # Run only Docker evaluation & metrics on pre-computed predictions (from Windows or WSL2):
    python evaluate_now.py --model_path /path/to/checkpoints_dir/ --mode eval_only
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Exact zero-based indices required for NoW evaluation (7 landmark points)
NOW_LANDMARK_INDICES = [36, 39, 42, 45, 33, 48, 54]
SUPPORTED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


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
            # Use wslpath -w to translate /mnt/c/... to C:\...
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
    # First, test if native Linux docker works
    linux_docker = shutil.which("docker")
    if linux_docker:
        try:
            # Check if docker daemon responds
            res = subprocess.run([linux_docker, "info"], capture_output=True, text=True, timeout=3)
            if res.returncode == 0:
                return linux_docker, False
        except Exception:
            pass

    # Fall back to Windows host docker.exe accessible from WSL
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DECA model(s) on the NoW Dataset.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to a single DECA checkpoint (.tar) OR a directory containing multiple .tar checkpoints.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/NoW/",
        help="Path to input dataset images. Default: 'datasets/NoW/'.",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default=None,
        help="Base directory for predictions and results. Default: '<model_folder>/../NoW_Evaluation'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device ('cuda' or 'cpu'). Default: 'cuda'.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "predict_only", "eval_only"],
        default="all",
        help="Execution mode: 'all' (default, end-to-end), 'predict_only' (DECA inference only), or 'eval_only' (Docker evaluation only).",
    )
    parser.add_argument(
        "--docker_executable",
        type=str,
        default=None,
        help="Optional explicit path to docker binary (e.g. /mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe or docker.exe).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, recomputes and overwrites existing predictions (.txt / .obj). Otherwise, skips images that already have both files generated.",
    )
    parser.add_argument(
        "--re_evaluate",
        action="store_true",
        help="If set, re-runs Docker evaluation even if result files already exist. Otherwise (by default), skips Docker evaluation and loads existing metrics.",
    )
    return parser.parse_args()


def get_checkpoint_list(model_path: str) -> List[str]:
    """
    Returns a sorted list of checkpoint (.tar) file paths.
    """
    path = os.path.abspath(model_path)
    if os.path.isdir(path):
        pattern = os.path.join(path, "*.tar")
        checkpoints = sorted(glob.glob(pattern))
        if not checkpoints:
            raise FileNotFoundError(f"No .tar checkpoint files found in directory: {path}")
        return checkpoints
    elif os.path.isfile(path):
        if not path.endswith(".tar"):
            print(f"[WARN] Checkpoint path does not end with .tar: {path}")
        return [path]
    else:
        raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")


def resolve_now_paths(dataset_path: str) -> Tuple[str, str]:
    """
    Resolves both:
      1) now_root_dir: directory containing scans/, scans_lmks_onlypp/, and imagepaths*.txt (mounted to /dataset in Docker)
      2) images_dir: directory containing the subject folders (e.g. FaMoS_*) so relative paths start directly with <subject>/<exp>/<img_name>
    """
    path_abs = os.path.abspath(dataset_path)

    # If user pointed to the images folder (iphone_pictures or similar containing FaMoS_*)
    # Check if this folder itself directly contains subject directories (e.g. FaMoS_*)
    subdirs = [d for d in os.listdir(path_abs) if os.path.isdir(os.path.join(path_abs, d))] if os.path.isdir(path_abs) else []
    has_subject_dirs = any(d.startswith("FaMoS_") for d in subdirs)

    # Standard relative structure in NoW Dataset
    standard_iphone_subpath = os.path.join(path_abs, "NoW_Dataset", "final_release_version", "iphone_pictures")

    if has_subject_dirs:
        images_dir = path_abs
    elif os.path.isdir(standard_iphone_subpath):
        images_dir = standard_iphone_subpath
    else:
        # Search recursively for iphone_pictures or directory containing FaMoS_
        found_img_dir = None
        for root, dirs, _ in os.walk(path_abs):
            if os.path.basename(root) == "iphone_pictures" or any(d.startswith("FaMoS_") for d in dirs):
                found_img_dir = root
                break
        images_dir = found_img_dir if found_img_dir else path_abs

    # Find the now_root_dir containing scans/
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
    """
    Recursively finds all supported images inside dataset_path.
    Returns list of tuples: (abs_image_path, relative_path_from_dataset_root).
    """
    dataset_abs = os.path.abspath(dataset_path)
    if not os.path.isdir(dataset_abs):
        raise NotADirectoryError(f"Dataset path does not exist or is not a directory: {dataset_abs}")

def find_dataset_images(images_dir: str) -> List[Tuple[str, str]]:
    """
    Recursively finds all supported images inside images_dir.
    Returns list of tuples: (abs_image_path, relative_path_from_images_dir).
    The relative path starts directly with <subject>/<exp>/<image_name>, so predictions
    are saved directly under <model_output_dir>/<subject>/... without extra wrapper directories.
    """
    images_abs = os.path.abspath(images_dir)
    if not os.path.isdir(images_abs):
        raise NotADirectoryError(f"Images directory does not exist or is not a directory: {images_abs}")

    image_records = []
    for root, _, files in os.walk(images_abs):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTS:
                abs_file = os.path.join(root, file)
                rel_file = os.path.relpath(abs_file, images_abs)
                image_records.append((abs_file, rel_file))

    image_records.sort(key=lambda x: x[1])
    return image_records


def load_image_tensor(image_path: str, target_size: int = 224):
    """
    Loads and pre-processes an image for DECA inference.
    """
    from PIL import Image
    import numpy as np
    import torch

    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        if img_rgb.size != (target_size, target_size):
            img_rgb = img_rgb.resize((target_size, target_size), Image.BILINEAR)
        img_np = np.array(img_rgb, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return tensor


def instantiate_deca(checkpoint_path: str, device: str):
    """
    Instantiates DECA model with weights loaded from checkpoint_path.
    """
    import torch
    # Ensure DECA directory is on sys.path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from decalib.deca import DECA
    from decalib.utils.config import cfg as deca_cfg

    cfg = deca_cfg.clone()
    cfg.pretrained_modelpath = checkpoint_path
    cfg.rasterizer_type = "pytorch3d"

    torch_device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"\n[INFO] Instantiating DECA on {torch_device} with checkpoint: {checkpoint_path}")
    model = DECA(config=cfg, device=torch_device)
    model.eval()
    return model


def predictions_exist(model_output_dir: str, images: List[Tuple[str, str]]) -> bool:
    """
    Checks if all required prediction files (.txt and .obj) already exist for the model.
    Returns True only if every image in `images` has both non-empty files generated in `model_output_dir`.
    """
    if not os.path.isdir(model_output_dir):
        return False

    for _, rel_img_path in images:
        rel_base_no_ext = os.path.splitext(rel_img_path)[0]
        dst_base = os.path.join(model_output_dir, rel_base_no_ext)
        txt_path = dst_base + ".txt"
        obj_path = dst_base + ".obj"
        if not (os.path.isfile(txt_path) and os.path.getsize(txt_path) > 0 and
                os.path.isfile(obj_path) and os.path.getsize(obj_path) > 0):
            return False
    return True


def run_inference_for_model(
    deca,
    images: List[Tuple[str, str]],
    model_output_dir: str,
    device: str,
    overwrite: bool = False
):
    """
    Runs DECA inference over all images, saving:
      - 7 extracted landmarks (.txt) with (X, Y, Z) coordinates
      - reconstructed 3D head mesh (.obj)
    If overwrite is False, skips images where both .txt and .obj already exist.
    """
    import torch
    from decalib.utils import util
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, *args, **kwargs):
            return iterable

    torch_device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"[INFO] Running inference on {len(images)} images -> {model_output_dir} (overwrite={overwrite})")

    skipped_count = 0
    for abs_img_path, rel_img_path in tqdm(images, desc="Predicting", unit="img"):
        rel_base_no_ext = os.path.splitext(rel_img_path)[0]
        dst_base = os.path.join(model_output_dir, rel_base_no_ext)
        os.makedirs(os.path.dirname(dst_base), exist_ok=True)

        txt_path = dst_base + ".txt"
        obj_path = dst_base + ".obj"

        # Check if already generated
        if not overwrite and os.path.isfile(txt_path) and os.path.getsize(txt_path) > 0 and os.path.isfile(obj_path) and os.path.getsize(obj_path) > 0:
            skipped_count += 1
            continue

        # Load image tensor
        image_tensor = load_image_tensor(abs_img_path, target_size=deca.image_size).to(torch_device)

        with torch.no_grad():
            codedict = deca.encode(image_tensor, use_detail=False)
            opdict = deca.decode(codedict, rendering=False, return_vis=False, use_detail=False, vis_lmk=False)

        # a) & b) Extract opdict['landmarks3d_world'] and select 7 landmark points
        lmks3d_world = opdict['landmarks3d_world'][0].cpu().numpy()
        selected_lmks = lmks3d_world[NOW_LANDMARK_INDICES, :]

        # c) Slice off 4th column (W / alpha channel) so only (X, Y, Z) remain
        if selected_lmks.shape[1] > 3:
            selected_lmks = selected_lmks[:, :3]
        elif selected_lmks.shape[1] < 3:
            raise ValueError(f"Landmark tensor has fewer than 3 coordinates: {selected_lmks.shape}")

        # d) Save landmarks as .txt: exactly 7 lines, formatted as 'X Y Z' separated by a single space
        with open(txt_path, "w", encoding="utf-8") as f:
            for pt in selected_lmks:
                f.write(f"{pt[0]:.18e} {pt[1]:.18e} {pt[2]:.18e}\n")

        # e) Save reconstructed 3D head mesh as .obj file
        verts = opdict['verts'][0].cpu().numpy()
        faces = deca.render.faces[0].cpu().numpy()
        util.write_obj(obj_path, verts, faces)

    if skipped_count > 0:
        print(f"[INFO] Skipped {skipped_count} already predicted images.")


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
        print("\n[TROUBLESHOOTING TIP] If Docker Desktop on Windows is not running, please start it.")
        print("Alternatively, you can complete inference using '--mode predict_only', then run evaluation from Windows using '--mode eval_only'.\n")
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
    Looks for .npy files containing mean/median distances or companion .meanmedian files.
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


def main():
    args = parse_args()

    checkpoints = get_checkpoint_list(args.model_path)
    model_path_abs = os.path.abspath(args.model_path)
    model_folder = model_path_abs if os.path.isdir(model_path_abs) else os.path.dirname(model_path_abs)

    if args.output_base is not None:
        output_base_abs = os.path.abspath(args.output_base)
    else:
        output_base_abs = os.path.abspath(os.path.join(model_folder, "..", "NoW_Evaluation"))

    summary_file = os.path.join(output_base_abs, "summary.txt")

    # Resolve NoW root directory (with scans/) and images directory (with FaMoS_* subjects)
    now_root_abs, images_dir_abs = resolve_now_paths(args.dataset_path)

    print("=" * 70)
    print("DECA NoW Dataset Benchmark Pipeline")
    print(f"Mode: {args.mode}")
    print(f"Checkpoints ({len(checkpoints)}): {checkpoints}")
    print(f"Dataset root: {now_root_abs}")
    print(f"Images directory: {images_dir_abs}")
    print(f"Output base directory: {output_base_abs}")
    print(f"PyTorch device: {args.device}")
    print("=" * 70)

    # 1. Prepare images if prediction stage is enabled
    images = None
    if args.mode in ["all", "predict_only"]:
        images = find_dataset_images(images_dir_abs)
        if not images:
            raise FileNotFoundError(f"No valid images found under {images_dir_abs}")
        print(f"[INFO] Found {len(images)} images in dataset.")

    records: Dict[str, Dict[str, float]] = {}

    for ckpt_path in checkpoints:
        ckpt_filename = os.path.basename(ckpt_path)
        model_name = os.path.splitext(ckpt_filename)[0]
        model_output_dir = os.path.join(output_base_abs, model_name)

        print("\n" + "-" * 70)
        print(f"Processing model: {model_name} (from {ckpt_path})")
        print("-" * 70)

        # STAGE 1: Prediction
        if args.mode in ["all", "predict_only"]:
            already_predicted = predictions_exist(model_output_dir, images)

            if already_predicted and not args.overwrite:
                print(f"[INFO] All predictions already exist for {model_name} in {model_output_dir}. Skipping model inference (use --overwrite to force).")
            else:
                import torch
                deca = instantiate_deca(ckpt_path, device=args.device)
                run_inference_for_model(
                    deca=deca,
                    images=images,
                    model_output_dir=model_output_dir,
                    device=args.device,
                    overwrite=args.overwrite
                )
                del deca
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if args.mode == "predict_only":
                print(f"[SUCCESS] Predictions complete for {model_name} in {model_output_dir}")
                continue

        # STAGE 2: Docker Evaluation
        if args.mode in ["all", "eval_only"]:
            already_evaluated = evaluation_results_exist(model_output_dir)

            if already_evaluated and not args.re_evaluate:
                print(f"[INFO] Evaluation results already exist for {model_name}. Skipping Docker evaluation (use --re_evaluate to force).")
            else:
                try:
                    run_docker_evaluation(
                        dataset_abs=now_root_abs,
                        model_output_abs=model_output_dir,
                        custom_docker_path=args.docker_executable
                    )
                except subprocess.CalledProcessError as e:
                    print(f"\n[WARNING] Docker evaluation failed for model '{model_name}'.")
                    print(f"[WARNING] Skipping evaluation for '{model_name}' due to corrupted/unreadable prediction files or evaluation error.")
                    continue

            # STAGE 3: Metrics Extraction
            try:
                mean_err, median_err = extract_metrics(model_output_dir)
                records[model_name] = {"mean": mean_err, "median": median_err}
                print(f"[RESULT] {model_name} -> Mean: {mean_err:.6f}, Median: {median_err:.6f}")

                # Append to evaluation_summary.txt if not already recorded or if re-evaluating
                append_summary(summary_file, f"{model_name} {mean_err:.6f} {median_err:.6f}")
            except Exception as e:
                print(f"[WARNING] Could not extract metrics for '{model_name}': {e}. Skipping.")

    # Final Comparison
    if args.mode in ["all", "eval_only"] and records:
        best_mean_model = min(records.keys(), key=lambda m: records[m]["mean"])
        best_median_model = min(records.keys(), key=lambda m: records[m]["median"])

        print("\n" + "=" * 70)
        print("FINAL EVALUATION BENCHMARK RESULTS")
        print("=" * 70)
        for m_name, m_metrics in records.items():
            print(f"Model: {m_name:<25} Mean: {m_metrics['mean']:.6f}   Median: {m_metrics['median']:.6f}")

        verdict_lines = [
            "",
            "--- Best Model Verdict ---",
            f"Lowest Mean Error:   {best_mean_model} ({records[best_mean_model]['mean']:.6f})",
            f"Lowest Median Error: {best_median_model} ({records[best_median_model]['median']:.6f})"
        ]

        for line in verdict_lines:
            print(line)
            append_summary(summary_file, line)

        print(f"\n[INFO] Summary appended to: {summary_file}")
    elif args.mode == "predict_only":
        print("\n[INFO] Prediction stage complete for all models.")


if __name__ == "__main__":
    main()
