import os
from pathlib import Path

import numpy as np

import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points

from pathlib import Path
import shutil

SRC_ROOT = Path("datasets/NoW/NoW_Dataset/final_release_version/iphone_pictures")
EXTENSIONS = {".jpg", ".png", ".jpeg"}

def folder_has_supported_files(folder: Path) -> bool:
    return any(
        p.is_file() and p.suffix.lower() in EXTENSIONS
        for p in folder.iterdir()
    )

def iter_source_folders(root: Path):
    folders = [root] if root.is_dir() and folder_has_supported_files(root) else []
    folders.extend(
        p for p in root.rglob("*")
        if p.is_dir() and folder_has_supported_files(p)
    )
    return sorted(set(folders))

def read_image_list(file_list_path: Path):
    image_paths = []
    with file_list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            if not p.is_absolute():
                p = SRC_ROOT / p
            p = p.resolve()
            if p.suffix.lower() in EXTENSIONS and p.is_file():
                image_paths.append(p)
            else:
                print(f"[WARN] Skipping invalid image path: {line}")
    return image_paths


def process_images(image_paths, deca: DECA, device: torch.device, dst_root: Path) -> None:
    testdata = datasets.TestData([str(p) for p in image_paths], scale=1.6)

    for i in tqdm(range(len(testdata)), desc="Processing listed images"):
        sample = testdata[i]
        images = sample["image"].to(device)[None, ...]

        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict)

        src_image_path = Path(testdata.imagepath_list[i]).resolve()
        rel_path = src_image_path.relative_to(SRC_ROOT.resolve())
        dst_base = (dst_root / rel_path).with_suffix("")
        dst_base.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(dst_base) + '_vis.jpg', deca.visualize(visdict))
        np.savetxt(str(dst_base) + ".txt", opdict["landmarks3d_world"][0].cpu().numpy())
        deca.save_obj(str(dst_base) + ".obj", opdict)

        print(f"[OK] {src_image_path} -> {dst_base}.txt/.obj")


def is_image_already_processed(src_image_path: Path, dst_root: Path) -> bool:
    rel_path = src_image_path.resolve().relative_to(SRC_ROOT.resolve())
    dst_base = (dst_root / rel_path).with_suffix("")
    return (
            Path(str(dst_base) + ".txt").exists()
            and Path(str(dst_base) + ".obj").exists()
            and Path(str(dst_base) + "_vis.jpg").exists()
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file-list",
        type=str,
        required=True,
        help="Path to a text file containing image paths (one per line). "
             "Relative paths are resolved from SRC_ROOT.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="NoW_Evaluation/evaluation_preds2",
        help="Directory where output files will be saved.",
    )

    args = parser.parse_args()

    dst_root = Path(args.output_dir)

    deca_cfg.rasterizer_type = "pytorch3d"
    device = torch.device("cuda")
    deca = DECA(config=deca_cfg, device=device)

    image_paths = read_image_list(Path(args.file_list))
    if not image_paths:
        print(f"No valid images found in list: {args.file_list}")
        return

    pending = []
    for p in image_paths:
        if is_image_already_processed(p, dst_root):
            print(f"[SKIP] Already processed: {p}")
            continue
        pending.append(p)

    if not pending:
        print("All listed images are already processed.")
        return

    process_images(pending, deca, device, dst_root)


if __name__ == "__main__":
    main()