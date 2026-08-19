import os
import argparse
import numpy as np
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Generate grouped train list from keypoint .npy files")
    parser.add_argument(
        "--input_path", "-i",
        type=str,
        default="./datasets/VGGFACE2/train_annotated_fan",
        help="Path to root folder containing .npy keypoint files",
    )
    parser.add_argument(
        "--output_path", "-o",
        type=str,
        default="./datasets/VGGFACE2/my_vggface2_train_list.npy",
        help="Path to save output .npy file",
    )
    parser.add_argument("--group_size", type=int, default=5, help="Number of samples per row")
    parser.add_argument(
        "--min_per_identity",
        type=int,
        default=5,
        help="Minimum samples needed per identity",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()



    all_ids = []
    for root, _, files in os.walk(args.input_path):
        for f in files:
            if "BUPT" in root:
                if "African" in root or "Asia" in root:
                    if f.endswith(".npy"):
                        rel = os.path.relpath(os.path.join(root, f), args.input_path)
                        sample_id = os.path.splitext(rel)[0].replace("\\", "/")
                        all_ids.append(sample_id)
            else:
                if f.endswith(".npy"):
                    rel = os.path.relpath(os.path.join(root, f), args.input_path)
                    sample_id = os.path.splitext(rel)[0].replace("\\", "/")
                    all_ids.append(sample_id)

    by_id = defaultdict(list)
    for sid in all_ids:
        identity = sid.split("/")[0]
        by_id[identity].append(sid)

    rows = []
    rng = np.random.default_rng(args.seed)

    for identity, samples in by_id.items():
        if len(samples) < args.min_per_identity:
            continue

        samples = sorted(samples)

        for i in range(0, len(samples) - args.group_size + 1, args.group_size):
            row = samples[i:i + args.group_size]
            rows.append(row)

    if not rows:
        for identity, samples in by_id.items():
            if len(samples) == 0:
                continue
            row = list(
                rng.choice(
                    samples,
                    size=args.group_size,
                    replace=len(samples) < args.group_size,
                )
            )
            rows.append(row)

    arr = np.array(rows, dtype=str)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True) if os.path.dirname(args.output_path) else None
    np.save(args.output_path, arr)

    print(f"Saved: {args.output_path}")
    print(f"Shape: {arr.shape}")
    print("First row:", arr[0] if len(arr) else "EMPTY")

if __name__ == "__main__":
    main()