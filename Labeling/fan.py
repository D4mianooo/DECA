import argparse
from pathlib import Path

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


def list_images(input_path: Path):
    if input_path.is_file():
        return [input_path]

    return sorted(
        p
        for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def draw_landmarks(image_bgr: np.ndarray, kpt: np.ndarray):
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


def relative_stem(img_path: Path, input_root: Path):
    rel = img_path.relative_to(input_root)
    return rel.with_suffix("")

def filter_unprocessed_images(images, input_root: Path, output_dir: Path):
    unprocessed_images = []
    skipped_existing = 0

    for img_path in images:
        rel_stem = relative_stem(img_path, input_root)
        npy_path = output_dir / rel_stem.with_suffix(".npy")

        if npy_path.exists():
            skipped_existing += 1
        else:
            unprocessed_images.append(img_path)

    return unprocessed_images, skipped_existing


def split_images_between_workers(images, num_workers: int, worker_id: int):
    if num_workers < 1:
        raise ValueError("--num_workers must be at least 1")

    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError(
            f"--worker_id must be between 0 and {num_workers - 1}"
        )

    return images[worker_id::num_workers]


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
        help="Recompute even if output .npy already exists"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Total number of parallel worker processes"
    )

    parser.add_argument(
        "--worker_id",
        type=int,
        default=0,
        help="ID of this worker, from 0 to num_workers - 1"
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_path)

    if not images:
        print(f"[Worker {args.worker_id}] No images found in: {input_path}")
        return

    input_root = (
        input_path
        if input_path.is_dir()
        else input_path.parent
    )

    total_found = len(images)
    skipped_existing = 0

    if not args.overwrite:
        images, skipped_existing = filter_unprocessed_images(
            images=images,
            input_root=input_root,
            output_dir=output_dir
        )

    if not images:
        print(
            f"[Worker {args.worker_id}] "
            "All images have already been processed."
        )
        print(
            f"[Worker {args.worker_id}] "
            f"Skipped existing: {skipped_existing}"
        )
        return

    total_unprocessed = len(images)

    images = split_images_between_workers(
        images=images,
        num_workers=args.num_workers,
        worker_id=args.worker_id
    )

    worker_total = len(images)

    print(
        f"[Worker {args.worker_id}/{args.num_workers}] "
        f"Images found: {total_found}"
    )
    print(
        f"[Worker {args.worker_id}/{args.num_workers}] "
        f"Unprocessed images: {total_unprocessed}"
    )
    print(
        f"[Worker {args.worker_id}/{args.num_workers}] "
        f"Assigned images: {worker_total}"
    )

    if not images:
        print(
            f"[Worker {args.worker_id}] "
            "No images were assigned to this worker."
        )
        return

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=args.device,
        compile=False
    )

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

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(rgb)

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
    print(
        f"[Worker {args.worker_id}] "
        f"Assigned images: {worker_total}"
    )
    print(
        f"[Worker {args.worker_id}] "
        f"Detected faces: {detected}/{worker_total}"
    )
    print(
        f"[Worker {args.worker_id}] "
        f"Globally skipped existing: {skipped_existing}"
    )
    print(
        f"[Worker {args.worker_id}] "
        f"No-face images: {no_face}"
    )
    print(
        f"[Worker {args.worker_id}] "
        f"Read failures: {failed_read}"
    )


if __name__ == "__main__":
    main()
