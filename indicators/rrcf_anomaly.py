"""
Robust Random Cut Forest (RRCF) anomaly detection indicator module.

RRCF is a tree-based anomaly detection algorithm designed for streaming data.
It builds an ensemble of trees where each point's anomaly score is based on
the Collusive Displacement (CoDisp) - the expected change in model complexity
when that point is removed.

This implementation uses the official rrcf library for accurate anomaly detection.
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, List
import warnings
from tqdm import tqdm
import rrcf

logger = logging.getLogger(__name__)


def calculate_rrcf_anomaly(df: pd.DataFrame,
                           feature_cols: Optional[List[str]] = None,
                           num_trees: int = 100,
                           tree_size: int = 256,
                           target_col: str = 'RRCF_Anomaly',
                           random_seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Calculate RRCF anomaly scores for timeseries data using the official rrcf library.

    The anomaly score is based on Collusive Displacement (CoDisp), which measures
    how much the model complexity changes when a point is removed. Higher scores
    indicate more anomalous points.

    Args:
        df: DataFrame with price/feature data
        feature_cols: List of column names to use as features. If None, uses ['close'] if available
        num_trees: Number of trees in the forest (default: 100)
        tree_size: Maximum size of each tree (default: 256)
        target_col: Name of the output column (default: 'RRCF_Anomaly')
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        DataFrame: Original DataFrame with RRCF anomaly score column added (values in [0, 1])
    """
    try:
        logger.info(f"Calculating {target_col} with num_trees={num_trees}, tree_size={tree_size}")

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Determine feature columns
        if feature_cols is None:
            available_cols = df.columns.tolist()
            feature_cols = []

            # Use close price as primary feature
            for col in ['close', 'Close', 'CLOSE']:
                if col in available_cols:
                    feature_cols.append(col)
                    break

            if not feature_cols:
                # Fallback to first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    feature_cols = [numeric_cols[0]]
                else:
                    raise ValueError("No numeric columns found for anomaly detection")

        logger.info(f"Using feature columns: {feature_cols}")

        # Validate feature columns exist
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Feature columns not found in DataFrame: {missing_cols}")

        # Extract features and handle missing values
        features_df = df[feature_cols].copy()
        features_df = features_df.ffill().bfill().fillna(0)

        # Convert to numpy array
        points = features_df.values
        n_points = len(points)

        if n_points == 0:
            result_df[target_col] = 0.5
            return result_df

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize anomaly scores dictionary
        avg_codisp = {}

        # Create a forest of random cut trees
        logger.info(f"Building {num_trees} random cut trees...")
        forest = []
        for _ in range(num_trees):
            tree = rrcf.RCTree()
            forest.append(tree)

        # Process points in a streaming fashion
        # For each point, insert into all trees and compute CoDisp
        logger.info("Computing anomaly scores...")
        for index in tqdm(range(n_points), desc="RRCF scoring"):
            point = points[index]

            # For each tree, insert the point and compute displacement
            for tree in forest:
                # If tree is at capacity, drop the oldest point
                if len(tree.leaves) >= tree_size:
                    oldest = min(tree.leaves.keys())
                    tree.forget_point(oldest)

                # Insert the new point
                tree.insert_point(point, index=index)

                # Compute codisp for this point
                if index not in avg_codisp:
                    avg_codisp[index] = 0.0

                # CoDisp is the collusive displacement
                new_codisp = tree.codisp(index)
                avg_codisp[index] += new_codisp

        # Average CoDisp across all trees
        for index in avg_codisp:
            avg_codisp[index] /= num_trees

        # Convert to array
        anomaly_scores = np.array([avg_codisp.get(i, 0.0) for i in range(n_points)])

        # Normalize scores to [0, 1] range using robust scaling
        # Use percentile-based normalization to handle outliers
        if len(anomaly_scores) > 0:
            # Use 1st and 99th percentile for robust normalization
            p1 = np.percentile(anomaly_scores, 1)
            p99 = np.percentile(anomaly_scores, 99)

            if p99 > p1:
                # Clip to percentile range and normalize
                anomaly_scores = np.clip(anomaly_scores, p1, p99)
                anomaly_scores = (anomaly_scores - p1) / (p99 - p1)
            else:
                # All scores are the same, set to middle value
                anomaly_scores = np.full(n_points, 0.5)

        # Add anomaly scores to result DataFrame
        result_df[target_col] = anomaly_scores

        # Log statistics
        logger.info(f"RRCF anomaly scores - Min: {anomaly_scores.min():.4f}, "
                    f"Max: {anomaly_scores.max():.4f}, "
                    f"Mean: {anomaly_scores.mean():.4f}, "
                    f"Std: {anomaly_scores.std():.4f}")

        return result_df

    except Exception as e:
        logger.error(f"Error calculating RRCF anomaly scores: {e}")
        import traceback
        traceback.print_exc()
        # Add the column with default values in case of error
        if target_col not in df.columns:
            df = df.copy()
            df[target_col] = 0.5  # Neutral anomaly score as default
        return df
