from pathlib import Path
import argparse
from collections import defaultdict
import os
import json

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError
from skimage.io import imread
import face_alignment
from tqdm import tqdm

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SUPPORTED_KPT_EXTENSIONS = {".npy"}
SHIFT_RATIO = 0.05
EXPAND_TOP = 0.10
EXPAND_LEFT = 0.20
EXPAND_RIGHT = 0.20
EXPAND_BOTTOM = 0.20
LANDMARK_THRESHOLD = 0.10


def build_grouped_list(sample_ids, group_size, min_per_identity, seed):
    by_id = defaultdict(list)
    for sample_id in sample_ids:
        identity = sample_id.split("/")[0]
        by_id[identity].append(sample_id)

    rows = []
    rng = np.random.default_rng(seed)

    for samples in by_id.values():
        if len(samples) < min_per_identity:
            continue

        samples = sorted(samples)
        for i in range(0, len(samples) - group_size + 1, group_size):
            rows.append(samples[i:i + group_size])

    if not rows:
        for samples in by_id.values():
            if not samples:
                continue
            rows.append(
                list(
                    rng.choice(
                        samples,
                        size=group_size,
                        replace=len(samples) < group_size,
                    )
                )
            )

    return np.array(rows, dtype=str)


def second_cleaning_output_path(image_path: Path) -> Path:
    image_folder = image_path if image_path.is_dir() else image_path.parent
    return image_folder.parent / f"{image_folder.name}_second_cleaning.npy"


def second_cleaning_resume_path(image_path: Path) -> Path:
    image_folder = image_path if image_path.is_dir() else image_path.parent
    return image_folder.parent / f"{image_folder.name}_second_cleaning_passed_ids.jsonl"


def sample_id_for(image_path: Path, image_root: Path) -> str:
    relative_stem = image_path.relative_to(image_root).with_suffix("")
    return relative_stem.as_posix()

def _validate_file(path: Path, allowed_extensions, kind: str) -> Path:
    if not path.is_file():
        raise ValueError(f"{kind} path is not a file: {path}")
    if path.suffix.lower() not in allowed_extensions:
        raise ValueError(f"Unsupported {kind} file extension: {path}")
    return path


def _iter_files_recursive(folder: Path, allowed_extensions, kind: str):
    if not folder.is_dir():
        raise ValueError(f"{kind} path is not a folder: {folder}")
    for root, dirnames, filenames in os.walk(folder):
        dirnames.sort()
        filenames.sort()
        root_path = Path(root)
        for filename in filenames:
            p = root_path / filename
            if p.suffix.lower() in allowed_extensions:
                yield p


def _has_matching_image_for_rel_stem(image_root: Path, rel_stem: Path) -> bool:
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        if (image_root / rel_stem).with_suffix(ext).is_file():
            return True
    return False


def count_image_kpt_pairs(image_path: Path, kpt_path: Path) -> int:
    image_is_file = image_path.is_file()
    image_is_dir = image_path.is_dir()

    if image_is_file:
        image_file = _validate_file(image_path, SUPPORTED_IMAGE_EXTENSIONS, "image")
        if kpt_path.is_dir():
            kpt_file = (kpt_path / image_file.name).with_suffix(".npy")
        else:
            kpt_file = image_file.with_suffix(".npy") if str(kpt_path) == "" else kpt_path

        if not kpt_file.is_file():
            return 0
        _validate_file(kpt_file, SUPPORTED_KPT_EXTENSIONS, "kpt")
        return 1

    if image_is_dir:
        if not kpt_path.is_dir():
            raise ValueError(f"kpt path must be a folder when image path is a folder: {kpt_path}")

        count = 0
        for image_file in _iter_files_recursive(image_path, SUPPORTED_IMAGE_EXTENSIONS, "image"):
            rel_image = image_file.relative_to(image_path)
            kpt_file = (kpt_path / rel_image).with_suffix(".npy")
            if not kpt_file.is_file():
                continue
            count += 1

        return count

    raise ValueError(f"image path must be a file or folder: {image_path}")


def iter_image_kpt_pairs(image_path: Path, kpt_path: Path):
    image_is_file = image_path.is_file()
    image_is_dir = image_path.is_dir()

    if image_is_file:
        image_file = _validate_file(image_path, SUPPORTED_IMAGE_EXTENSIONS, "image")
        if kpt_path.is_dir():
            kpt_file = (kpt_path / image_file.name).with_suffix(".npy")
        else:
            kpt_file = kpt_path

        if not kpt_file.is_file():
            tqdm.write(f"[SKIP] Missing kpt file for image: {image_file.name}")
            return

        _validate_file(kpt_file, SUPPORTED_KPT_EXTENSIONS, "kpt")
        yield image_file, kpt_file
        return

    if image_is_dir:
        if not kpt_path.is_dir():
            raise ValueError(f"kpt path must be a folder when image path is a folder: {kpt_path}")

        found_image = False
        for image_file in _iter_files_recursive(image_path, SUPPORTED_IMAGE_EXTENSIONS, "image"):
            found_image = True
            rel_image = image_file.relative_to(image_path)
            kpt_file = (kpt_path / rel_image).with_suffix(".npy")

            if not kpt_file.is_file():
                tqdm.write(f"[SKIP] Missing kpt file: {rel_image.as_posix()}")
                continue

            yield image_file, kpt_file

        if not found_image:
            raise ValueError(f"No supported image files found in folder: {image_path}")
        return

    raise ValueError(f"image path must be a file or folder: {image_path}")

def bbox_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.ndim != 2 or landmarks.shape[0] == 0 or landmarks.shape[1] != 2:
        raise ValueError(
            f"Expected nonempty landmarks with shape (N, 2), got {landmarks.shape}"
        )
    if not np.all(np.isfinite(landmarks)):
        raise ValueError("Landmarks contain non-finite coordinates")

    return np.array(
        [
            np.min(landmarks[:, 0]),
            np.min(landmarks[:, 1]),
            np.max(landmarks[:, 0]),
            np.max(landmarks[:, 1]),
        ],
        dtype=np.float32,
    )


def bbox_dimensions(bbox: np.ndarray) -> np.ndarray:
    bbox = np.asarray(bbox, dtype=np.float32)
    if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
        raise ValueError(f"Expected a finite bounding box with shape (4,), got {bbox}")

    dimensions = bbox[2:] - bbox[:2]
    if np.any(dimensions <= 0):
        raise ValueError(f"Bounding-box dimensions must be positive, got {dimensions}")
    return dimensions


def shift_bbox(bbox: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    bbox = np.asarray(bbox, dtype=np.float32)
    epsilon = np.asarray(epsilon, dtype=np.float32)
    if bbox.shape != (4,) or epsilon.shape != (2,):
        raise ValueError("Expected bbox shape (4,) and epsilon shape (2,)")
    return bbox + np.tile(epsilon, 2)


def expand_bbox(bbox: np.ndarray, bbox_size: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    bbox_width, bbox_height = np.asarray(bbox_size, dtype=np.float32)
    return np.array(
        [
            x1 - EXPAND_LEFT * bbox_width,
            y1 - EXPAND_TOP * bbox_height,
            x2 + EXPAND_RIGHT * bbox_width,
            y2 + EXPAND_BOTTOM * bbox_height,
            ],
        dtype=np.float32,
    )


def run_fan(fa, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    predictions = fa.get_landmarks_from_image(
        image,
        detected_faces=[np.asarray(bbox, dtype=np.float32)],
    )
    if predictions is None or len(predictions) != 1:
        raise ValueError("FAN did not return exactly one landmark set")

    landmarks = np.asarray(predictions[0], dtype=np.float32)
    if landmarks.ndim != 2 or landmarks.shape[0] == 0 or landmarks.shape[1] != 2:
        raise ValueError(f"Expected FAN landmarks with shape (N, 2), got {landmarks.shape}")
    if not np.all(np.isfinite(landmarks)):
        raise ValueError("FAN returned non-finite landmarks")
    return landmarks


def landmark_consistency_score(
        original_landmarks: np.ndarray,
        shifted_landmarks: np.ndarray,
        epsilon: np.ndarray,
        bbox_size: np.ndarray,
) -> float:
    original_landmarks = np.asarray(original_landmarks, dtype=np.float32)
    shifted_landmarks = np.asarray(shifted_landmarks, dtype=np.float32)
    epsilon = np.asarray(epsilon, dtype=np.float32)
    bbox_size = np.asarray(bbox_size, dtype=np.float32)

    if original_landmarks.shape != shifted_landmarks.shape:
        raise ValueError("Original and shifted landmarks must have identical shapes")
    if original_landmarks.ndim != 2 or original_landmarks.shape[0] == 0:
        raise ValueError("Landmark arrays must be nonempty and two-dimensional")
    if original_landmarks.shape[1] != 2:
        raise ValueError("Landmark arrays must contain x/y coordinate pairs")
    if epsilon.shape != (2,) or bbox_size.shape != (2,):
        raise ValueError("epsilon and bbox_size must both have shape (2,)")
    if np.any(bbox_size <= 0):
        raise ValueError("Bounding-box dimensions must be positive")

    delta = shifted_landmarks - epsilon - original_landmarks
    distances = np.linalg.norm(delta / bbox_size, axis=1)
    if not np.all(np.isfinite(distances)):
        raise ValueError("Landmark distances contain non-finite values")
    return float(np.max(distances))


def load_passed_ids_jsonl(path: Path):
    if not path.exists():
        return []
    passed = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "id" in obj:
                    passed.append(str(obj["id"]))
                else:
                    passed.append(line)
            except json.JSONDecodeError:
                passed.append(line)
    return list(dict.fromkeys(passed))


def append_passed_id_jsonl(path: Path, sample_id: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": sample_id}, ensure_ascii=False) + "\n")

def load_image_strict(image_path: Path) -> np.ndarray:
    try:
        with Image.open(str(image_path)) as img:
            img = img.convert("RGB")
            return np.asarray(img)
    except (OSError, UnidentifiedImageError) as error:
        raise RuntimeError("Failed to decode image {}: {}".format(image_path, error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-path",
        required=True,
        type=Path,
        help="Path to a single image or a folder containing images",
    )
    parser.add_argument(
        "--kpt-path",
        required=True,
        type=Path,
        help="Path to a single keypoint file or a folder containing keypoint files",
    )
    parser.add_argument("--group-size", type=int, default=5, help="Number of samples per row")
    parser.add_argument(
        "--min-per-identity",
        type=int,
        default=5,
        help="Minimum samples needed per identity",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.group_size < 1:
        raise ValueError("--group-size must be at least 1")
    if args.min_per_identity < 1:
        raise ValueError("--min-per-identity must be at least 1")

    pair_iter = iter_image_kpt_pairs(args.image_path, args.kpt_path)

    image_folder = args.image_path if args.image_path.is_dir() else args.image_path.parent
    output_path = second_cleaning_output_path(args.image_path)
    resume_path = second_cleaning_resume_path(args.image_path)

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device="cuda",
    )

    passed_sample_ids = load_passed_ids_jsonl(resume_path)
    if passed_sample_ids:
        tqdm.write(f"[RESUME] Loaded {len(passed_sample_ids)} passed samples from {resume_path}")

    passed_sample_id_set = set(passed_sample_ids)
    passed_images = len(passed_sample_ids)
    skipped_images = 0

    for image_path, kpt_path in tqdm(pair_iter):
        sample_id = sample_id_for(image_path, image_folder)
        if sample_id in passed_sample_id_set:
            continue

        try:
            image = load_image_strict(Path(image_path))
        except RuntimeError as error:
            print("[WARN] {}".format(error))
            continue
        source_landmarks = np.asarray(np.load(str(kpt_path))[:, :2], dtype=np.float32)

        try:
            original_bbox = bbox_from_landmarks(source_landmarks)
            bbox_size = bbox_dimensions(original_bbox)
            epsilon = SHIFT_RATIO * bbox_size
            shifted_bbox = shift_bbox(original_bbox, epsilon)

            expanded_original_bbox = expand_bbox(original_bbox, bbox_size)
            expanded_shifted_bbox = expand_bbox(shifted_bbox, bbox_size)

            kpt1 = run_fan(fa, image, expanded_original_bbox)
            kpt2 = run_fan(fa, image, expanded_shifted_bbox)
            max_distance = landmark_consistency_score(
                original_landmarks=kpt1,
                shifted_landmarks=kpt2,
                epsilon=epsilon,
                bbox_size=bbox_size,
            )
        except (RuntimeError, ValueError) as error:
            skipped_images += 1
            tqdm.write(f"[SKIP] {image_path.name}: {error}")
            continue

        if max_distance >= LANDMARK_THRESHOLD:
            skipped_images += 1
            tqdm.write(
                f"[SKIP] {image_path.name}: max_distance={max_distance:.6f} "
                f">= {LANDMARK_THRESHOLD}"
            )
            continue

        passed_images += 1
        passed_sample_ids.append(sample_id)
        passed_sample_id_set.add(sample_id)
        append_passed_id_jsonl(resume_path, sample_id)
        tqdm.write(f"[PASS] {image_path.name}: max_distance={max_distance:.6f}")

    grouped_list = build_grouped_list(
        passed_sample_ids,
        group_size=args.group_size,
        min_per_identity=args.min_per_identity,
        seed=args.seed,
    )
    np.save(output_path, grouped_list)
    written_images = int(grouped_list.size)

    print(f"Second cleaning finished: {passed_images:,} images passed.")
    print(f"Created list with {written_images:,} images: {output_path}")
    if grouped_list.shape[0] > 0:
        print(f"First row: {grouped_list[0].tolist()}")
        print(f"Last row: {grouped_list[-1].tolist()}")
    else:
        print("First row: <empty>")
        print("Last row: <empty>")


if __name__ == "__main__":
    main()