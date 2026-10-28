'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this computer program. 
Using this computer program means that you agree to the terms in the LICENSE file (https://ringnet.is.tue.mpg.de/license). 
Any use not explicitly granted by the LICENSE is prohibited.
Copyright 2020 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its 
Max Planck Institute for Intelligent Systems. All rights reserved.
More information about the NoW Challenge is available at https://ringnet.is.tue.mpg.de/challenge.
For comments or questions, please email us at ringnet@tue.mpg.de
'''

import os
import sys
from pathlib import Path
import argparse
import numpy as np

# Non-interactive Matplotlib backend for headless / Docker execution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_path(filename):
    return os.path.dirname(filename)


def cumulative_error(errors, nbins=100000):
    errors = errors.ravel()
    values, base = np.histogram(errors, bins=nbins)
    cumsum = np.array(np.cumsum(values), dtype=float)
    cumulative = 100.0 * cumsum / float(errors.shape[0])
    return (base[:-1], cumulative)


def generating_cumulative_error_plots(method_error_fnames: list, method_identifiers: list, out_fname: str):
    """
    Generate cumulative error plots for a list of errors.
    :param method_error_fnames: list of benchmark output files (.npy)
    :param method_identifiers: list of names of methods that created the output files in the order corresponding to method_error_fnames
    :param out_fname: output plot filename
    """
    method_errors = []
    for fname in method_error_fnames:
        method_errors.append(np.load(fname, allow_pickle=True, encoding="latin1").item()['computed_distances'])

    for i in range(len(method_identifiers)):
        print('%s - median: %.2f, mean: %.2f, std: %.2f' % (
            method_identifiers[i],
            np.round(np.median(np.hstack(method_errors[i])), 2),
            np.round(np.mean(np.hstack(method_errors[i])), 2),
            np.round(np.std(np.hstack(method_errors[i])), 2)
        ))

    cumulative_errors = []
    for error in method_errors:
        cumulative_errors.append(cumulative_error(np.hstack(error)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 8])
    ax.set_xticks(np.arange(0, 8, 1.0))
    ax.set_ylim([0, 100])
    ax.set_yticks(np.arange(0, 101, 20.0))

    for i, method_id in enumerate(method_identifiers):
        plt.plot(cumulative_errors[i][0], cumulative_errors[i][1], label=method_id)

    plt.xlabel('Error [mm]', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    lgd = ax.legend(loc='lower right')
    
    # Ensure destination directory exists
    out_dir = os.path.dirname(out_fname)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(out_fname)
    plt.clf()
    print(f"[SUCCESS] Cumulative error plot saved to: {out_fname}")

    _, ext = os.path.splitext(out_fname)
    if ext == '.svg':
        from subprocess import call
        out_fname_pdf = out_fname.replace('.svg', '.pdf')
        args = ['rsvg-convert', '-f', 'pdf', '-o', out_fname_pdf, out_fname]
        call(args)


def find_npy_files_and_identifiers(input_path: str):
    """
    Dynamically loads all valid .npy files from either:
      - A single .npy file path
      - A directory containing .npy files (or model subdirectories with results/*.npy)
    Derives each plot line's label by extracting the prefix of the file name before the first underscore:
      filename.stem.split('_')[0]
    """
    p = Path(input_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    fnames = []
    identifiers = []

    if p.is_file():
        if p.suffix.lower() == ".npy":
            fnames.append(str(p))
            ident = p.stem.split('_')[0]
            identifiers.append(ident)
        else:
            raise ValueError(f"Specified file is not a .npy file: {p}")
    elif p.is_dir():
        # Look for .npy files recursively
        all_npy = sorted(p.glob("**/*.npy"))
        for f in all_npy:
            if not f.is_file():
                continue
            stem_prefix = f.stem.split('_')[0]
            # If the file is inside <model>/results/RECON_computed_distances.npy,
            # 'RECON' is generic, so use the model folder prefix
            if stem_prefix.upper() in ["RECON", "COMPUTED"] and f.parent.name in ["results", "preds"]:
                model_name = f.parent.parent.name
                ident = model_name.split('_')[0]
            else:
                ident = stem_prefix
            fnames.append(str(f))
            identifiers.append(ident)

    return fnames, identifiers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate cumulative error plots using NoW cumulative_errors calculation."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to a single .npy file OR a directory containing .npy files.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="chart.png",
        help="Output image path for the plot. Default: 'chart.png'.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    fnames, identifiers = find_npy_files_and_identifiers(args.input)

    if not fnames:
        print(f"[ERROR] No valid .npy error files found in: {args.input}")
        sys.exit(1)

    print(f"[INFO] Found {len(fnames)} error file(s):")
    for fname, ident in zip(fnames, identifiers):
        print(f"  [{ident}] -> {fname}")

    generating_cumulative_error_plots(fnames, identifiers, args.output)
