#!/usr/bin/env python3
"""Extract exactly one frame from every MP4 video.

Dependencies:
    pip install opencv-python tqdm

Examples:
    python extract_frames.py \
        --input /path/to/videos \
        --output /path/to/frames

    python extract_frames.py \
        --input /path/to/videos \
        --output /path/to/frames \
        --recursive \
        --strategy random \
        --seed 42
"""

import argparse
import hashlib
import random
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one middle or random frame from every MP4 video."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory containing MP4 files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory in which JPG frames will be saved.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively in input subdirectories.",
    )
    parser.add_argument(
        "--strategy",
        choices=("middle", "random"),
        default="middle",
        help="Frame selection strategy (default: middle).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for the random strategy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an already existing output frame.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Total number of workers for sharding (default: 1).",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Current worker index in [0, num-workers-1] (default: 0).",
    )
    return parser.parse_args()


def validate_worker_args(num_workers: int, worker_id: int) -> None:
    if num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("--worker-id must be in [0, num-workers-1]")

def iter_videos(input_dir: Path, recursive: bool) -> Iterable[Path]:
    paths = input_dir.rglob("*") if recursive else input_dir.iterdir()
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".mp4":
            yield path

def belongs_to_worker(
    video_path: Path,
    input_dir: Path,
    num_workers: int,
    worker_id: int,
) -> bool:
    """Return whether a video belongs to this deterministic worker shard."""
    relative_path = video_path.relative_to(input_dir).as_posix()
    digest = hashlib.blake2b(
        relative_path.encode("utf-8"),
        digest_size=8,
    ).digest()
    file_id = int.from_bytes(digest, byteorder="big")
    return file_id % num_workers == worker_id


def iter_worker_videos(
    input_dir: Path,
    recursive: bool,
    num_workers: int,
    worker_id: int,
) -> Iterable[Path]:
    """Yield this worker's videos one at a time."""
    for video_path in iter_videos(input_dir, recursive):
        if belongs_to_worker(
            video_path,
            input_dir,
            num_workers,
            worker_id,
        ):
            yield video_path


def choose_frame_index(
    frame_count: int,
    strategy: str,
    rng: random.Random,
) -> int:
    """Return a valid zero-based frame index."""
    if frame_count <= 0:
        raise ValueError("video has no frames")
    if strategy == "middle":
        return frame_count // 2
    return rng.randrange(frame_count)


def output_stem(video_path: Path, input_dir: Path, recursive: bool) -> Path:
    """Build a collision-resistant relative output stem.

    In recursive mode, the first directory is preserved and all remaining
    path components are joined with ``__``. For example:

        id00012/video_001/clip_00003.mp4
        -> id00012/video_001__clip_00003
    """
    relative_video = video_path.relative_to(input_dir).with_suffix("")
    if not recursive or len(relative_video.parts) == 1:
        return Path(relative_video.stem)

    top_level = relative_video.parts[0]
    flattened_name = "__".join(relative_video.parts[1:])
    return Path(top_level) / flattened_name


def create_output_name(
    video_path: Path,
    input_dir: Path,
    frame_index: int,
    recursive: bool,
) -> Path:
    """Return the relative JPG path for a selected frame."""
    stem = output_stem(video_path, input_dir, recursive)
    return stem.with_name(
        "{}__frame_{:06d}.jpg".format(stem.name, frame_index)
    )


def find_existing_output(
    output_dir: Path,
    video_path: Path,
    input_dir: Path,
    recursive: bool,
) -> Optional[Path]:
    """Find a previously extracted frame for a video, if one exists."""
    stem = output_stem(video_path, input_dir, recursive)
    parent = output_dir / stem.parent
    matches = sorted(parent.glob("{}__frame_*.jpg".format(stem.name)))
    return matches[0] if matches else None


def extract_frame(
    video_path: Path,
    output_dir: Path,
    input_dir: Path,
    recursive: bool,
    strategy: str,
    rng: random.Random,
    overwrite: bool,
) -> Tuple[str, Optional[Path]]:
    """Extract one frame and return ``(status, output_path)``.

    Status is one of ``saved``, ``skipped``, or ``error``.
    """
    if not overwrite:
        existing_output = find_existing_output(
            output_dir,
            video_path,
            input_dir,
            recursive,
        )
        if existing_output is not None:
            return "skipped", existing_output

    video = cv2.VideoCapture(str(video_path))
    try:
        if not video.isOpened():
            tqdm.write(
                "WARNING: cannot open video: {}".format(video_path),
                file=sys.stderr,
            )
            return "error", None

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            tqdm.write(
                "WARNING: empty video or invalid frame count: {}"
                .format(video_path),
                file=sys.stderr,
            )
            return "error", None

        frame_index = choose_frame_index(frame_count, strategy, rng)
        relative_output = create_output_name(
            video_path,
            input_dir,
            frame_index,
            recursive,
        )
        output_path = output_dir / relative_output

        if output_path.exists() and not overwrite:
            return "skipped", output_path

        seek_succeeded = video.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )
        if not seek_succeeded:
            tqdm.write(
                "WARNING: cannot seek to frame {} in: {}"
                .format(frame_index, video_path),
                file=sys.stderr,
            )
            return "error", None

        read_succeeded, frame = video.read()
        if (
            not read_succeeded
            or frame is None
            or frame.size == 0
            or frame.ndim not in (2, 3)
        ):
            tqdm.write(
                "WARNING: cannot read frame {} from: {}"
                .format(frame_index, video_path),
                file=sys.stderr,
            )
            return "error", None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_succeeded = cv2.imwrite(str(output_path), frame)
        except cv2.error as error:
            tqdm.write(
                "WARNING: cannot save frame from {}: {}"
                .format(video_path, error),
                file=sys.stderr,
            )
            return "error", None

        if not write_succeeded or not output_path.is_file():
            tqdm.write(
                "WARNING: OpenCV failed to save: {}".format(output_path),
                file=sys.stderr,
            )
            return "error", None

        return "saved", output_path
    finally:
        video.release()


def validate_args(args: argparse.Namespace) -> Tuple[Path, Path]:
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(
            "Input directory does not exist: {}".format(input_dir)
        )
    if output_dir == input_dir:
        raise ValueError(
            "Input and output directories must be different."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def main() -> int:
    args = parse_args()
    try:
        validate_worker_args(args.num_workers, args.worker_id)
        input_dir, output_dir = validate_args(args)
    except (OSError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    video_paths = iter_worker_videos(
        input_dir=input_dir,
        recursive=args.recursive,
        num_workers=args.num_workers,
        worker_id=args.worker_id,
    )

    videos_found = 0
    saved = 0
    skipped = 0
    errors = 0

    for video_path in tqdm(
        video_paths,
        desc="Worker {}/{}".format(args.worker_id, args.num_workers - 1),
        unit="video",
    ):
        videos_found += 1
        try:
            status, _ = extract_frame(
                video_path=video_path,
                output_dir=output_dir,
                input_dir=input_dir,
                recursive=args.recursive,
                strategy=args.strategy,
                rng=rng,
                overwrite=args.overwrite,
            )
        except Exception as error:
            tqdm.write(
                "WARNING: unexpected error for {}: {}"
                .format(video_path, error),
                file=sys.stderr,
            )
            status = "error"

        if status == "saved":
            saved += 1
        elif status == "skipped":
            skipped += 1
        else:
            errors += 1

    print("\nSummary")
    print("  Worker: {}/{}".format(args.worker_id, args.num_workers - 1))
    print("  Videos assigned: {}".format(videos_found))
    print("  Frames saved: {}".format(saved))
    print("  Videos skipped: {}".format(skipped))
    print("  Errors: {}".format(errors))

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
