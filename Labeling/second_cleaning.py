from pathlib import Path
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import estimate_transform, warp
import face_alignment

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SUPPORTED_KPT_EXTENSIONS = {".npy"}
IMAGE_SIZE = 224


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


def crop_from_kpt(
        image: np.ndarray,
        kpt: np.ndarray,
        image_size: int = IMAGE_SIZE,
        height_ratio_shift: float = 0.0,
        width_ratio_shift: float = 0.0,
        top_expand: float = 0.0,
        bottom_expand: float = 0.0,
        left_expand: float = 0.0,
        right_expand: float = 0.0,
):
    left = np.min(kpt[:, 0])
    right = np.max(kpt[:, 0])
    top = np.min(kpt[:, 1])
    bottom = np.max(kpt[:, 1])

    bw = right - left
    bh = bottom - top

    left = left - left_expand * bw
    right = right + right_expand * bw
    top = top - top_expand * bh
    bottom = bottom + bottom_expand * bh

    bw = right - left
    bh = bottom - top

    shift = np.array([width_ratio_shift * bw, height_ratio_shift * bh], dtype=np.float32)

    old_size = (bw + bh) / 2.0
    center = np.array(
        [right - bw / 2.0, bottom - bh / 2.0],
        dtype=np.float32,
    )
    center = center + shift
    size = int(old_size)

    src_pts = np.array(
        [
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2],
        ],
        dtype=np.float32,
    )
    dst_pts = np.array(
        [[0, 0], [0, image_size - 1], [image_size - 1, 0]],
        dtype=np.float32,
    )

    tform = estimate_transform("similarity", src_pts, dst_pts)
    cropped_image = warp(image, tform.inverse, output_shape=(image_size, image_size))
    return cropped_image


def show_comparison(
        image: np.ndarray,
        cropped_image: np.ndarray,
        shifted_cropped_image: np.ndarray,
        title: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cropped_image)
    axes[1].set_title(f"Cropped ({IMAGE_SIZE}x{IMAGE_SIZE})")
    axes[1].axis("off")

    axes[2].imshow(shifted_cropped_image)
    axes[2].set_title(f"Shifted +5% ({IMAGE_SIZE}x{IMAGE_SIZE})")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def _plot_image_with_landmarks(ax, image, kpt: np.ndarray, title: str):
    ax.imshow(image)
    ax.scatter(kpt[:, 0], kpt[:, 1], s=8, c="lime")
    ax.set_title(title)
    ax.axis("off")

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

    for image_path, kpt_path in pairs:
        image = imread(str(image_path))
        kpt = np.load(str(kpt_path))[:, :2]

        shifted_expanded_cropped_image = crop_from_kpt(
            image=image,
            kpt=kpt,
            image_size=IMAGE_SIZE,
            width_ratio_shift=0.05,
            height_ratio_shift=0.05,
            top_expand=0.10,
            bottom_expand=0.20,
            left_expand=0.20,
            right_expand=0.20,
        )
        img = shifted_expanded_cropped_image

        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        preds = fa.get_landmarks_from_image(img)

        kpt2 = np.asarray(
            preds[0],
            dtype=np.float32
        )

        _plot_image_with_landmarks(axes[0], image, kpt, "Original")
        _plot_image_with_landmarks(axes[1], img, kpt2, "Original")

        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    main()