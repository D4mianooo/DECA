import argparse
import hashlib
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from tqdm import tqdm

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract face-skin masks recursively and preserve folder structure."
    )
    parser.add_argument(
        "-i",
        "--input-folder",
        required=True,
        help="Root input image folder (or a single image path).",
    )
    parser.add_argument(
        "-o",
        "--output-image-folder",
        required=True,
        help="Root output folder for .npy masks.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Total number of workers for sharding.",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Current worker index in [0, num_workers-1].",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./data/selfie_multiclass_256x256.tflite",
        help="Path to the TFLite model file.",
    )
    return parser.parse_args()


def validate_worker_args(num_workers: int, worker_id: int):
    if num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("--worker-id must be in [0, num_workers-1]")


def create_segmenter_options(model_path: str):
    base_options = BaseOptions(model_asset_path=model_path)
    image_segmenter_options = mp.tasks.vision.ImageSegmenterOptions
    vision_running_mode = mp.tasks.vision.RunningMode

    return image_segmenter_options(
        base_options=base_options,
        running_mode=vision_running_mode.IMAGE,
        output_confidence_masks=True,
        output_category_mask=True,
    )


def iter_images(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            annotated_fan_path = input_path.parent / "race_per_7000_annotated_fan" / input_path.with_suffix(".npy").name
            if annotated_fan_path.exists():
                yield input_path
        return

    for path in input_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            relative_to_input = path.relative_to(input_path)
            annotated_fan_path = input_path.parent / "race_per_7000_annotated_fan" / relative_to_input.with_suffix(".npy")
            if annotated_fan_path.exists():
                yield path

def belongs_to_worker(img_path: Path, input_root: Path, num_workers: int, worker_id: int) -> bool:
    relative_path = img_path.relative_to(input_root).as_posix()
    digest = hashlib.blake2b(relative_path.encode("utf-8"), digest_size=8).digest()
    file_id = int.from_bytes(digest, byteorder="big")
    return file_id % num_workers == worker_id


def build_output_npy_path(image_path: Path, input_root: Path, output_root: Path) -> Path:
    rel_path = image_path.relative_to(input_root)
    rel_no_ext = rel_path.with_suffix("")
    return output_root / rel_no_ext.with_suffix(".npy")


def iter_unprocessed_worker_images(
        input_path: Path,
        input_root: Path,
        output_root: Path,
        num_workers: int,
        worker_id: int,
):
    for img_path in iter_images(input_path):
        if not belongs_to_worker(img_path, input_root, num_workers, worker_id):
            continue

        out_npy_path = build_output_npy_path(img_path, input_root, output_root)
        if out_npy_path.exists():
            continue

        yield img_path


def process_images(image_iter, input_root: Path, output_root: Path, options, worker_id: int, num_workers: int):
    image_segmenter = mp.tasks.vision.ImageSegmenter
    skipped = 0
    processed = 0

    with image_segmenter.create_from_options(options) as segmenter:
        for image_path in tqdm(image_iter, desc=f"Worker {worker_id}/{num_workers - 1}"):
            out_npy_path = build_output_npy_path(image_path, input_root, output_root)
            out_npy_path.parent.mkdir(parents=True, exist_ok=True)

            mp_image = mp.Image.create_from_file(str(image_path))
            segmented_masks = segmenter.segment(mp_image)
            confidence_masks = segmented_masks.confidence_masks

            if not confidence_masks:
                print(f"Skipped (no confidence masks): {image_path}")
                skipped += 1
                continue

            face_skin_mask = confidence_masks[3]
            mask_array = face_skin_mask.numpy_view()
            np.save(str(out_npy_path), mask_array)
            processed += 1

    return processed, skipped


def main():
    args = parse_args()
    validate_worker_args(args.num_workers, args.worker_id)

    input_path = Path(args.input_folder).resolve()
    output_root = Path(args.output_image_folder).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    input_root = input_path if input_path.is_dir() else input_path.parent

    image_iter = iter_unprocessed_worker_images(
        input_path=input_path,
        input_root=input_root,
        output_root=output_root,
        num_workers=args.num_workers,
        worker_id=args.worker_id,
    )

    options = create_segmenter_options(args.model_path)
    processed, skipped = process_images(
        image_iter=image_iter,
        input_root=input_root,
        output_root=output_root,
        options=options,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
    )

    print(f"Done. Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()