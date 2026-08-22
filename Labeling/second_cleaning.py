from pathlib import Path
import argparse

import numpy as np
from skimage.io import imread
import face_alignment

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SUPPORTED_KPT_EXTENSIONS = {".npy"}
SHIFT_RATIO = 0.05
EXPAND_TOP = 0.10
EXPAND_LEFT = 0.20
EXPAND_RIGHT = 0.20
EXPAND_BOTTOM = 0.20
LANDMARK_THRESHOLD = 0.10


def _validate_file(path: Path, allowed_extensions, kind: str) -> Path:
    if not path.is_file():
        raise ValueError(f"{kind} path is not a file: {path}")
    if path.suffix.lower() not in allowed_extensions:
        raise ValueError(f"Unsupported {kind} file extension: {path}")
    return path


def _collect_files_from_folder(folder: Path, allowed_extensions, kind: str):
    if not folder.is_dir():
        raise ValueError(f"{kind} path is not a folder: {folder}")
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_extensions
    )
    if not files:
        raise ValueError(f"No supported {kind} files found in folder: {folder}")
    return files


def _build_stem_index(files, kind: str):
    stem_to_file = {}
    for p in files:
        stem = p.stem
        if stem in stem_to_file:
            raise ValueError(
                f"Duplicate {kind} stem '{stem}' found: {stem_to_file[stem]} and {p}"
            )
        stem_to_file[stem] = p
    return stem_to_file


def collect_image_kpt_pairs(image_path: Path, kpt_path: Path):
    image_is_file = image_path.is_file()
    image_is_dir = image_path.is_dir()
    kpt_is_file = kpt_path.is_file()
    kpt_is_dir = kpt_path.is_dir()

    if image_is_file and kpt_is_file:
        image_file = _validate_file(image_path, SUPPORTED_IMAGE_EXTENSIONS, "image")
        kpt_file = _validate_file(kpt_path, SUPPORTED_KPT_EXTENSIONS, "kpt")
        if image_file.stem != kpt_file.stem:
            raise ValueError(
                f"Image and kpt names must match (without extension): {image_file.name} vs {kpt_file.name}"
            )
        return [(image_file, kpt_file)]

    if image_is_dir and kpt_is_dir:
        image_files = _collect_files_from_folder(
            image_path, SUPPORTED_IMAGE_EXTENSIONS, "image"
        )
        kpt_files = _collect_files_from_folder(
            kpt_path, SUPPORTED_KPT_EXTENSIONS, "kpt"
        )

        image_index = _build_stem_index(image_files, "image")
        kpt_index = _build_stem_index(kpt_files, "kpt")

        missing_kpt = sorted(stem for stem in image_index if stem not in kpt_index)
        if missing_kpt:
            raise ValueError(
                "Missing kpt files for image stems: {}".format(", ".join(missing_kpt))
            )

        extra_kpt = sorted(stem for stem in kpt_index if stem not in image_index)
        if extra_kpt:
            raise ValueError(
                "Kpt files without matching image stems: {}".format(", ".join(extra_kpt))
            )

        stems = sorted(image_index.keys())
        return [(image_index[s], kpt_index[s]) for s in stems]

    raise ValueError("image_path and kpt_path must both be files or both be folders.")


def bbox_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Return the tight [x1, y1, x2, y2] box around 2D landmarks."""
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
    """Return [width, height] for an [x1, y1, x2, y2] box."""
    bbox = np.asarray(bbox, dtype=np.float32)
    if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
        raise ValueError(f"Expected a finite bounding box with shape (4,), got {bbox}")

    dimensions = bbox[2:] - bbox[:2]
    if np.any(dimensions <= 0):
        raise ValueError(f"Bounding-box dimensions must be positive, got {dimensions}")
    return dimensions


def shift_bbox(bbox: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    """Translate an [x1, y1, x2, y2] box by the [x, y] offset epsilon."""
    bbox = np.asarray(bbox, dtype=np.float32)
    epsilon = np.asarray(epsilon, dtype=np.float32)
    if bbox.shape != (4,) or epsilon.shape != (2,):
        raise ValueError("Expected bbox shape (4,) and epsilon shape (2,)")
    return bbox + np.tile(epsilon, 2)


def expand_bbox(bbox: np.ndarray, bbox_size: np.ndarray) -> np.ndarray:
    """Expand a box using the paper's asymmetric margins."""
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
    """Run FAN for one supplied box and return landmarks in image coordinates."""
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
    """Compute max_i ||diag(bw, bh)^-1 (k2_i - epsilon - k1_i)||."""
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
    args = parser.parse_args()

    pairs = collect_image_kpt_pairs(args.image_path, args.kpt_path)

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device='cuda'
    )

    total_images = len(pairs)
    passed_images = 0
    skipped_images = 0
    for image_path, kpt_path in pairs:
        image = imread(str(image_path))
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
            print(f"[SKIP] {image_path.name}: {error}")
            continue

        if max_distance >= LANDMARK_THRESHOLD:
            skipped_images += 1
            print(
                f"[SKIP] {image_path.name}: max_distance={max_distance:.6f} "
                f">= {LANDMARK_THRESHOLD}"
            )
            continue

        passed_images += 1
        print(f"[PASS] {image_path.name}: max_distance={max_distance:.6f}")
    print(
        f"[DEBUG] Passed filter: {passed_images}/{total_images} "
        f"({(passed_images / total_images * 100.0) if total_images else 0.0:.2f}%), "
        f"Skipped: {skipped_images}/{total_images}"
    )
if __name__ == "__main__":
    main()
