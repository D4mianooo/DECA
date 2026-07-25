#!/usr/bin/env python3
"""Generate a list of image paths from a directory of landmark .npy files."""

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np


LANDMARK_DIR_SUFFIX = "_annotated_fan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert landmark .npy paths to corresponding image paths and "
            "write them relative to the directory where the script is run."
        )
    )
    parser.add_argument(
        "landmarks_dir",
        type=Path,
        help="Directory containing landmark .npy files.",
    )
    parser.add_argument(
        "-i",
        "--images-dir",
        type=Path,
        help=(
            "Root image directory. By default, a landmarks directory ending "
            "in '_annotated_fan' is mapped to its sibling without that suffix."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output NumPy file (default: <images-dir>_image_paths.npy).",
    )
    parser.add_argument(
        "--extension",
        default=".jpg",
        help="Image extension to use (default: .jpg).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Fail if any generated image path does not exist.",
    )
    return parser.parse_args()


def infer_images_dir(landmarks_dir: Path) -> Path:
    if landmarks_dir.name.endswith(LANDMARK_DIR_SUFFIX):
        image_dir_name = landmarks_dir.name[: -len(LANDMARK_DIR_SUFFIX)]
        return landmarks_dir.with_name(image_dir_name)
    return landmarks_dir


def iter_image_paths(
    landmarks_dir: Path,
    images_dir: Path,
    extension: str,
) -> Iterable[Path]:
    for landmark_path in sorted(landmarks_dir.rglob("*.npy")):
        relative_path = landmark_path.relative_to(landmarks_dir)
        yield images_dir / relative_path.with_suffix(extension)


def default_output_path(images_dir: Path) -> Path:
    return images_dir.parent / "{}_image_paths.npy".format(images_dir.name)


def relative_to_working_directory(path: Path, working_dir: Path) -> str:
    return Path(os.path.relpath(str(path), str(working_dir))).as_posix()


def main() -> None:
    args = parse_args()
    working_dir = Path.cwd().resolve()
    landmarks_dir = args.landmarks_dir.resolve()

    if not landmarks_dir.is_dir():
        raise NotADirectoryError(
            "Landmarks directory does not exist: {}".format(landmarks_dir)
        )

    images_dir = (
        args.images_dir.resolve()
        if args.images_dir is not None
        else infer_images_dir(landmarks_dir)
    )

    extension = args.extension
    if not extension.startswith("."):
        extension = "." + extension

    output_path = (
        args.output.resolve()
        if args.output is not None
        else default_output_path(images_dir)
    )

    if output_path.suffix.lower() != ".npy":
        raise ValueError("--output must use the .npy extension")

    relative_paths = []
    missing_paths = []
    existing_image_paths = (
        set(images_dir.rglob("*{}".format(extension)))
        if args.validate
        else None
    )

    for image_path in iter_image_paths(landmarks_dir, images_dir, extension):
        if (
            existing_image_paths is not None
            and image_path not in existing_image_paths
        ):
            missing_paths.append(image_path)
        relative_paths.append(
            relative_to_working_directory(image_path, working_dir)
        )

    if not relative_paths:
        raise RuntimeError(
            "No .npy files found under: {}".format(landmarks_dir)
        )

    if missing_paths:
        preview = "\n".join(str(path) for path in missing_paths[:10])
        raise FileNotFoundError(
            "{} corresponding image(s) are missing. First missing paths:\n{}"
            .format(len(missing_paths), preview)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(relative_paths, dtype=np.str_))

    print("Saved: {}".format(output_path))
    print("Working directory: {}".format(working_dir))
    print("Landmarks directory: {}".format(landmarks_dir))
    print("Images directory: {}".format(images_dir))
    print("Paths written: {}".format(len(relative_paths)))
    print("First path: {}".format(relative_paths[0]))
    print("Last path: {}".format(relative_paths[-1]))


if __name__ == "__main__":
    main()
