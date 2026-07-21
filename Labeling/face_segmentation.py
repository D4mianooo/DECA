import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

parser = argparse.ArgumentParser(
    description="Extract face-skin masks recursively and preserve folder structure."
)
parser.add_argument("-i", "--input-folder", required=True, help="Root input image folder (e.g. datasets/VGGFACE2/train)")
parser.add_argument("-o", "--output-image-folder", required=True, help="Root output folder for binary mask PNGs")
parser.add_argument("--num-workers", type=int, default=1, help="Total number of workers for sharding")
parser.add_argument("--worker-id", type=int, default=0, help="Current worker index in [0, num_workers-1]")
parser.add_argument("--model-path", type=str, default="selfie_multiclass_256x256.tflite",
                    help="Path to the TFLite model file")
args = parser.parse_args()

model_path = args.model_path
base_options = BaseOptions(model_asset_path=model_path)

ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ImageSegmenterOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,
    output_confidence_masks=True,
    output_category_mask=True,
)

input_root = Path(args.input_folder)
out_img_root = Path(args.output_image_folder)

out_img_root.mkdir(parents=True, exist_ok=True)

valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
image_paths = sorted([p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts])

if args.num_workers < 1:
    raise ValueError("--num-workers must be >= 1")
if args.worker_id < 0 or args.worker_id >= args.num_workers:
    raise ValueError("--worker-id must be in [0, num_workers-1]")

image_paths = image_paths[args.worker_id::args.num_workers]

processed = 0
skipped = 0

pending_paths = []
for image_path in image_paths:
    rel_path = image_path.relative_to(input_root)
    rel_no_ext = rel_path.with_suffix("")
    out_npy_path = out_img_root / rel_no_ext.with_suffix(".npy")

    if out_npy_path.exists():
        skipped += 1
        continue

    pending_paths.append(image_path)

with ImageSegmenter.create_from_options(options) as segmenter:
    for image_path in tqdm(pending_paths, desc=f"Worker {args.worker_id}/{args.num_workers - 1}"):
        rel_path = image_path.relative_to(input_root)
        rel_no_ext = rel_path.with_suffix("")

        out_npy_path = out_img_root / rel_no_ext.with_suffix(".npy")

        out_npy_path.parent.mkdir(parents=True, exist_ok=True)

        mp_image = mp.Image.create_from_file(str(image_path))
        segmented_masks = segmenter.segment(mp_image)
        confidence_masks = segmented_masks.confidence_masks

        if not confidence_masks:
            print(f"Skipped (no confidence masks): {image_path}")
            skipped += 1
            continue

        face_skin_mask = confidence_masks[3]  # face-skin class
        mask_array = face_skin_mask.numpy_view()

        np.save(str(out_npy_path), mask_array)
