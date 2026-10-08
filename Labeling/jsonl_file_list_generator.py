import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL sample IDs into grouped DECA training rows."
    )
    parser.add_argument(
        "--input_path",
        "-i",
        type=Path,
        default=Path("datasets/VGGFACE2/train_second_cleaning_passed_ids.jsonl"),
        help="JSONL file containing one string-valued 'id' per record.",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=Path,
        default=Path("datasets/VGGFACE2/train_second_cleaning_list.npy"),
        help="New .npy file to create (existing files are not overwritten).",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=5,
        help="Number of samples from one identity in each row.",
    )
    args = parser.parse_args()

    if args.group_size <= 0:
        raise ValueError("--group_size must be positive")
    if args.output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {args.output_path}")

    by_identity = defaultdict(list)
    with args.input_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("id")
            if not isinstance(sample_id, str) or "/" not in sample_id:
                raise ValueError(
                    f"Line {line_number} has no valid string 'id': {record!r}"
                )
            identity = sample_id.split("/", 1)[0]
            by_identity[identity].append(sample_id)

    rows = []
    for samples in by_identity.values():
        for start in range(0, len(samples) - args.group_size + 1, args.group_size):
            rows.append(samples[start : start + args.group_size])

    if not rows:
        raise ValueError("Input produced no complete groups")

    output = np.asarray(rows, dtype=str)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, output)

    discarded = sum(len(samples) % args.group_size for samples in by_identity.values())
    print(f"Saved: {args.output_path}")
    print(f"Shape: {output.shape}")
    print(f"Discarded incomplete-group samples: {discarded}")
    print(f"First row: {output[0]}")


if __name__ == "__main__":
    main()
