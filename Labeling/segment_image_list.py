#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

# Ubuntu's Caffe bindings are built for the system Python. If the active
# environment cannot load them, restart with the matching interpreter rather
# than mixing binary packages from incompatible Python environments.
try:
    import caffe
except (ImportError, OSError):
    system_python = Path("/usr/bin/python3")
    if Path(sys.executable) != system_python and system_python.exists():
        os.execv(str(system_python), [str(system_python), *sys.argv])
    raise

import numpy as np
from PIL import Image


MEAN_BGR = np.array(
    (104.00698793, 116.66876762, 122.67891434),
    dtype=np.float32,
)
BACKGROUND_CHANNEL = 0
FACE_CHANNEL = 15

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Segment images from a NumPy or text list and save .npy masks while "
            "preserving their hierarchy under a different parent directory."
        )
    )
    parser.add_argument(
        "file_list",
        type=Path,
        help="A .npy string array or text file containing image paths.",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        required=True,
        help="Parent directory under which .npy masks will be saved.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        help=(
            "Image hierarchy root. If omitted, it is inferred as the common "
            "parent directory of all listed images."
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        default=root / "data" / "face_seg_fcn8s.caffemodel",
        help="Path to the Caffe model weights.",
    )
    parser.add_argument(
        "-d",
        "--deploy",
        type=Path,
        default=root / "data" / "face_seg_fcn8s_deploy.prototxt",
        help="Path to the Caffe deploy prototxt.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Caffe inference device (default: cpu).",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device ID used with --device gpu.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=500,
        help=(
            "Maximum width and height used for inference while preserving "
            "aspect ratio (default: 500)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute masks that already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many entries; useful for validation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print path mapping without loading the model.",
    )
    return parser.parse_args()


def path_values(file_list: Path):
    if file_list.suffix.lower() == ".npy":
        values = np.load(file_list, mmap_mode="r", allow_pickle=False)
        if values.ndim != 1:
            raise ValueError(
                "Expected a one-dimensional path array, got shape {}"
                .format(values.shape)
            )
        return values
    return file_list.read_text(encoding="utf-8").splitlines()


def read_image_paths(
    file_list: Path,
    working_dir: Path,
    limit: Optional[int],
) -> List[Path]:
    paths = []
    for raw_value in path_values(file_list):
        if isinstance(raw_value, bytes):
            raw_value = raw_value.decode("utf-8")
        value = str(raw_value).strip()
        if not value or value.startswith("#"):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = working_dir / path
        paths.append(Path(os.path.abspath(str(path))))
        if limit is not None and len(paths) >= limit:
            break

    if not paths:
        raise ValueError("No image paths found in: {}".format(file_list))
    return paths


def infer_input_root(image_paths: Sequence[Path]) -> Path:
    common_parent = os.path.commonpath(
        [str(path.parent) for path in image_paths]
    )
    return Path(common_parent)


def output_path_for(
    image_path: Path,
    input_root: Path,
    output_root: Path,
) -> Path:
    try:
        relative_path = image_path.relative_to(input_root)
    except ValueError as error:
        raise ValueError(
            "Image is outside --input-root: {}".format(image_path)
        ) from error
    return output_root / relative_path.with_suffix(".npy")


def resize_for_inference(
    image: Image.Image,
    max_size: int,
) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_size / width, max_size / height)
    if scale == 1.0:
        return image
    size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(size, Image.BICUBIC)


def prepare_input(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    bgr = rgb[:, :, ::-1].copy()
    bgr -= MEAN_BGR
    return bgr.transpose((2, 0, 1))


def segment(net: "caffe.Net", image_path: Path, max_size: int) -> np.ndarray:
    with Image.open(str(image_path)) as source:
        source = source.convert("RGB")
        original_size = source.size
        inference_image = resize_for_inference(source, max_size)
        network_input = prepare_input(inference_image)

    net.blobs["data"].reshape(1, *network_input.shape)
    net.blobs["data"].data[...] = network_input
    net.forward()

    scores = net.blobs["score"].data[0]
    if scores.shape[0] <= FACE_CHANNEL:
        raise RuntimeError(
            "Expected at least {} output channels, got {}"
            .format(FACE_CHANNEL + 1, scores.shape[0])
        )

    mask = (
        scores[FACE_CHANNEL] > scores[BACKGROUND_CHANNEL]
    ).astype(np.uint8)

    if (mask.shape[1], mask.shape[0]) != original_size:
        mask_image = Image.fromarray(mask, mode="L")
        mask_image = mask_image.resize(original_size, Image.NEAREST)
        mask = np.asarray(mask_image, dtype=np.uint8)
    return mask


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as output_file:
        np.save(output_file, mask)
    temporary_path.replace(path)


def configure_caffe(device: str, gpu_id: int) -> None:
    if device == "gpu":
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()


def main() -> None:
    args = parse_args()
    if args.max_size < 1:
        raise ValueError("--max-size must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")

    working_dir = Path.cwd().resolve()
    file_list = args.file_list.resolve()
    output_root = args.output_root.resolve()
    image_paths = read_image_paths(file_list, working_dir, args.limit)

    input_root = (
        args.input_root.resolve()
        if args.input_root is not None
        else infer_input_root(image_paths)
    )

    mappings = [
        (
            image_path,
            output_path_for(image_path, input_root, output_root),
        )
        for image_path in image_paths
    ]


    print("Images: {}".format(len(mappings)))
    print("Input root: {}".format(input_root))
    print("Output root: {}".format(output_root))
    print("First mapping: {} -> {}".format(*mappings[0]))

    if args.dry_run:
        print("Dry run complete.")
        return

    model_path = args.model.resolve()
    deploy_path = args.deploy.resolve()
    if not model_path.is_file():
        raise FileNotFoundError("Model file not found: {}".format(model_path))
    if not deploy_path.is_file():
        raise FileNotFoundError(
            "Deploy file not found: {}".format(deploy_path)
        )

    configure_caffe(args.device, args.gpu_id)
    net = caffe.Net(
        str(deploy_path),
        caffe.TEST,
        weights=str(model_path),
    )

    processed = 0
    skipped = 0
    failures = []

    for index, (image_path, output_path) in enumerate(mappings, start=1):
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            mask = segment(net, image_path, args.max_size)
            save_mask(output_path, mask)
            processed += 1
        except Exception as error:
            failures.append((image_path, str(error)))
            print(
                "[{}/{}] FAILED {}: {}"
                .format(index, len(mappings), image_path, error),
                file=sys.stderr,
            )
        if index == 1 or index % 100 == 0 or index == len(mappings):
            print(
                "[{}/{}] processed={}, skipped={}, failed={}"
                .format(
                    index,
                    len(mappings),
                    processed,
                    skipped,
                    len(failures),
                )
            )

    print(
        "Done. Processed: {}, skipped: {}, failed: {}."
        .format(processed, skipped, len(failures))
    )
    if failures:
        raise RuntimeError(
            "Face segmentation failed for {} image(s)."
            .format(len(failures))
        )


if __name__ == "__main__":
    main()
