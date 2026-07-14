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
DST_ROOT = Path("NoW_Evaluation/evaluation_preds")

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

def process_folder(src_folder: Path, deca: DECA, device: torch.device) -> None:
    testdata = datasets.TestData(str(src_folder))

    for i in tqdm(range(len(testdata)), desc=f"Processing {src_folder}"):
        sample = testdata[i]
        images = sample["image"].to(device)[None, ...]

        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict)

        src_image_path = Path(testdata.imagepath_list[i])
        rel_path = src_image_path.relative_to(SRC_ROOT)
        dst_base = (DST_ROOT / rel_path).with_suffix("")
        dst_base.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(dst_base) + '_vis.jpg', deca.visualize(visdict))
        np.savetxt(str(dst_base) + ".txt", opdict["landmarks3d"][0].cpu().numpy())
        deca.save_obj(str(dst_base) + ".obj", opdict)

        print(f"[OK] {src_image_path} -> {dst_base}.txt/.obj")
def is_folder_already_processed(src_folder: Path) -> bool:
    image_files = [
        p for p in src_folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]
    if not image_files:
        return False

    for src_image_path in image_files:
        rel_path = src_image_path.relative_to(SRC_ROOT)
        dst_base = (DST_ROOT / rel_path).with_suffix("")
        if not (
                Path(str(dst_base) + ".txt").exists()
                and Path(str(dst_base) + ".obj").exists()
                and Path(str(dst_base) + "_vis.jpg").exists()
        ):
            return False
    return True

def main() -> None:
    deca_cfg.rasterizer_type = "pytorch3d"
    device = torch.device("cuda")
    deca = DECA(config=deca_cfg, device=device)

    source_folders = iter_source_folders(SRC_ROOT)
    if not source_folders:
        print(f"No supported image folders found under: {SRC_ROOT}")
        return

    for src_folder in source_folders:
        if is_folder_already_processed(src_folder):
            print(f"[SKIP] Folder already processed: {src_folder}")
            continue
        process_folder(src_folder, deca, device)

if __name__ == "__main__":
    main()
