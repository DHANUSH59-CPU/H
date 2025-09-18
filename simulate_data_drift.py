#!/usr/bin/env python3
"""
Data drift simulation for monitoring demonstrations.
This script creates various types of data drift scenarios to test monitoring systems.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime, timedelta


class DriftSimulator:
    """Simulate different types of data drift for monitoring demonstrations."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize drift simulator with random seed for reproducibility."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def gradual_drift(self, data: pd.DataFrame, 
                     columns: List[str], 
                     drift_rate: float = 0.01,
                     drift_direction: str = 'positive') -> pd.DataFrame:
        """
        Simulate gradual drift whe