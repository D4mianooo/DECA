import argparse
import hashlib
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import face_alignment
import tqdm

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

def iter_images(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield input_path
        return

    for path in input_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path

def relative_stem(img_path: Path, input_root: Path) -> Path:
    return img_path.relative_to(input_root).with_suffix("")

def belongs_to_worker(img_path: Path, input_root: Path, num_workers: int, worker_id: int) -> bool:
    relative_path = img_path.relative_to(input_root).as_posix()

    digest = hashlib.blake2b(
        relative_path.encode("utf-8"),
        digest_size=8
    ).digest()

    file_id = int.from_bytes(digest, byteorder="big")

    return file_id % num_workers == worker_id

def iter_unprocessed_worker_images(input_path: Path, input_root: Path, output_dir: Path, num_workers: int, worker_id: int, overwrite: bool):
    for img_path in iter_images(input_path):
        rel_stem = relative_stem(img_path, input_root)
        npy_path = output_dir / rel_stem.with_suffix(".npy")

        if npy_path.exists() and not overwrite:
            continue

        yield img_path


def draw_landmarks(
        image_bgr: np.ndarray,
        kpt: np.ndarray
) -> np.ndarray:
    vis = image_bgr.copy()

    for x, y in kpt:
        cv2.circle(
            vis,
            (int(round(x)), int(round(y))),
            1,
            (0, 255, 0),
            -1
        )

    return vis


def main():
    parser = argparse.ArgumentParser(
        "FAN-only 2D landmark extraction"
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input image or folder"
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output folder"
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device"
    )

    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save image with landmark overlay"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute existing outputs"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Total number of worker processes"
    )

    parser.add_argument(
        "--worker_id",
        type=int,
        default=0,
        help="Worker ID from 0 to num_workers - 1"
    )

    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num_workers must be at least 1")

    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError(
            f"--worker_id must be between 0 and "
            f"{args.num_workers - 1}"
        )

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    input_root = (
        input_path
        if input_path.is_dir()
        else input_path.parent
    )

    images = iter_unprocessed_worker_images(
        input_path=input_path,
        input_root=input_root,
        output_dir=output_dir,
        num_workers=args.num_workers,
        worker_id=args.worker_id,
        overwrite=args.overwrite
    )

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=args.device
    )

    processed = 0
    detected = 0
    failed_read = 0
    no_face = 0

    for img_path in tqdm.tqdm(images, desc=f"Worker {args.worker_id}/{args.num_workers}", position=args.worker_id, leave=True):
        rel_stem = relative_stem(img_path, input_root)

        npy_path = output_dir / rel_stem.with_suffix(".npy")
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        bgr = cv2.imread(str(img_path))

        if bgr is None:
            failed_read += 1
            tqdm.tqdm.write(
                f"[Worker {args.worker_id}] "
                f"Failed to read: {img_path}"
            )
            continue

        print(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        try:
            preds = fa.get_landmarks_from_image(rgb)
        except RuntimeError as e:
            print(f"[WARN] Skipping: {e}")
            continue

        if preds is None or len(preds) == 0:
            no_face += 1
            tqdm.tqdm.write(
                f"[Worker {args.worker_id}] "
                f"No face: {img_path}"
            )
            continue

        kpt = np.asarray(
            preds[0],
            dtype=np.float32
        )

        np.save(str(npy_path), kpt)
        detected += 1

        if args.save_vis:
            vis = draw_landmarks(bgr, kpt)

            vis_path = (
                    output_dir
                    / rel_stem.with_name(
                f"{rel_stem.name}_vis"
            ).with_suffix(".jpg")
            )

            vis_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            cv2.imwrite(str(vis_path), vis)

    print(f"\n[Worker {args.worker_id}] Done.")
    print(f"[Worker {args.worker_id}] Processed: {processed}")
    print(f"[Worker {args.worker_id}] Detected faces: {detected}")
    print(f"[Worker {args.worker_id}] No-face images: {no_face}")
    print(f"[Worker {args.worker_id}] Read failures: {failed_read}")


if __name__ == "__main__":
    main()