#!/usr/bin/env python3
"""
aggregate_importances.py

Usage:
    python aggregate_importances.py <session_folder> [--last_k K] [--threshold T]

Loads per-window feature importances (importances.npy and features.json) from a walk-forward session,
computes mean importance for each feature, saves a CSV of importances, and optionally outputs a JSON
list of features whose mean importance exceeds the threshold.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

def load_window_importances(window_folder):
    """
    Load importances and feature names from a single window folder.
    Returns: (features_list, importances_array) or (None, None) if files missing.
    """
    imp_path = os.path.join(window_folder, 'importances.npy')
    feat_path = os.path.join(window_folder, 'features.json')
    if not os.path.exists(imp_path) or not os.path.exists(feat_path):
        return None, None
    importances = np.load(imp_path)
    features = json.load(open(feat_path))
    return features, importances


def aggregate_importances(session_folder, last_k=None):
    """
    Aggregate mean importances across windows in session_folder.
    If last_k is provided, only use the last k windows.
    Returns a DataFrame with columns ['feature', 'mean_importance'] sorted descending.
    """
    # Identify window directories, possibly under a 'models' subdirectory
    models_subdir = os.path.join(session_folder, 'models')
    if os.path.isdir(models_subdir):
        root = models_subdir
    else:
        root = session_folder
    all_dirs = [d for d in os.listdir(root) if d.startswith('window_') and os.path.isdir(os.path.join(root, d))]
    all_dirs.sort(key=lambda x: int(x.split('_')[-1]))
    if last_k is not None and last_k > 0:
        dirs = all_dirs[-last_k:]
    else:
        dirs = all_dirs

    importance_dicts = []
    all_features = set()
    for d in dirs:
        folder = os.path.join(root, d)
        feats, imps = load_window_importances(folder)
        if feats is None:
            continue
        # build mapping
        dmap = dict(zip(feats, imps))
        importance_dicts.append(dmap)
        all_features.update(feats)

    if not importance_dicts:
        print(f"No importances found in any window under {session_folder}")
        sys.exit(1)

    # Compute mean importance
    mean_imp = {}
    for feat in sorted(all_features):
        vals = [dmap.get(feat, 0.0) for dmap in importance_dicts]
        mean_imp[feat] = float(np.mean(vals))

    df = pd.DataFrame(
        list(mean_imp.items()),
        columns=['feature','mean_importance']
    )
    df = df.sort_values('mean_importance', ascending=False)
    return df


def main():
    parser = argparse.ArgumentParser(description='Aggregate feature importances from a walk-forward session')
    parser.add_argument('session_folder', help='Path to models/session_<timestamp> folder')
    parser.add_argument('--last_k', '-k', type=int, default=None,
                        help='Number of most recent windows to include (default all)')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                        help='Mean importance threshold for pruning')
    parser.add_argument('--out_csv', '-o', default='aggregated_importances.csv',
                        help='Output CSV filename for aggregated importances')
    parser.add_argument('--pruned_out', '-p', default='pruned_features.json',
                        help='Output JSON filename for pruned features')
    args = parser.parse_args()

    session_folder = args.session_folder
    if not os.path.isdir(session_folder):
        print(f"Error: {session_folder} is not a directory.")
        sys.exit(1)

    df = aggregate_importances(session_folder, args.last_k)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved aggregated importances to {args.out_csv}")

    if args.threshold is not None:
        pruned = df.loc[df['mean_importance'] >= args.threshold, 'feature'].tolist()
        with open(args.pruned_out, 'w') as f:
            json.dump(pruned, f, indent=4)
        print(f"Saved pruned features (threshold {args.threshold}) to {args.pruned_out}")

if __name__ == '__main__':
    main() 