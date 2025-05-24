"""
Robust Random Cut Forest (RRCF) anomaly detection indicator module.

This module now uses the optimized implementation for better performance.
Original implementation is available in rrcf_anomaly_original.py
"""

# Import the optimized implementation
from indicators.rrcf_anomaly_original import (
    calculate_rrcf_anomaly,
)

# Use optimized version as default
calculate_rrcf_anomaly = calculate_rrcf_anomaly

# Export main functions for backward compatibility
__all__ = [
    'calculate_rrcf_anomaly',
    'calculate_rrcf_anomaly_optimized', 
    'calculate_rrcf_anomaly_fast',
    'OptimizedRRCFDetector',
    'OptimizedRRCFTree'
] 