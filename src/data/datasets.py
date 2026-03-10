"""Data loading and preprocessing utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DatasetMetadata:
    """Metadata for datasets including feature information and constraints."""
    
    def __init__(
        self,
        feature_names: List[str],
        feature_types: List[str],
        target_names: List[str],
        sensitive_attributes: Optional[List[str]] = None,
        monotonic_features: Optional[List[str]] = None,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """Initialize dataset metadata.
        
        Args:
            feature_names: List of feature names.
            feature_types: List of feature types ('numerical', 'categorical').
            target_names: List of target class names.
            sensitive_attributes: List of sensitive attribute names.
            monotonic_features: List of features with monotonicity constraints.
            feature_ranges: Dictionary mapping feature names to (min, max) ranges.
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.target_names = target_names
        self.sensitive_attributes = sensitive_attributes or []
        self.monotonic_features = monotonic_features or []
        self.feature_ranges = feature_ranges or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "feature_names": self.feature_names,
            "feature_types": self.feature_types,
            "target_names": self.target_names,
            "sensitive_attributes": self.sensitive_attributes,
            "monotonic_features": self.monotonic_features,
            "feature_ranges": self.feature_ranges,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        """Create metadata from dictionary."""
        return cls(
            feature_names=data["feature_names"],
            feature_types=data["feature_types"],
            target_names=data["target_names"],
            sensitive_attributes=data.get("sensitive_attributes"),
            monotonic_features=data.get("monotonic_features"),
            feature_ranges=data.get("feature_ranges"),
        )


def load_iris_dataset() -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """Load the Iris dataset with metadata.
    
    Returns:
        Tuple of (features, targets, metadata).
    """
    data = load_iris()
    X = data.data
    y = data.target
    
    metadata = DatasetMetadata(
        feature_names=data.feature_names.tolist(),
        feature_types=["numerical"] * len(data.feature_names),
        target_names=data.target_names.tolist(),
    )
    
    return X, y, metadata


def load_wine_dataset() -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """Load the Wine dataset with metadata.
    
    Returns:
        Tuple of (features, targets, metadata).
    """
    data = load_wine()
    X = data.data
    y = data.target
    
    metadata = DatasetMetadata(
        feature_names=data.feature_names.tolist(),
        feature_types=["numerical"] * len(data.feature_names),
        target_names=data.target_names.tolist(),
    )
    
    return X, y, metadata


def load_breast_cancer_dataset() -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """Load the Breast Cancer dataset with metadata.
    
    Returns:
        Tuple of (features, targets, metadata).
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    metadata = DatasetMetadata(
        feature_names=data.feature_names.tolist(),
        feature_types=["numerical"] * len(data.feature_names),
        target_names=data.target_names.tolist(),
    )
    
    return X, y, metadata


def create_synthetic_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    noise: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """Create a synthetic dataset for testing.
    
    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        n_classes: Number of classes.
        noise: Amount of noise to add.
        random_state: Random seed.
        
    Returns:
        Tuple of (features, targets, metadata).
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate targets based on linear combination of features
    weights = np.random.randn(n_features)
    logits = X @ weights + np.random.randn(n_samples) * noise
    
    if n_classes == 2:
        y = (logits > 0).astype(int)
        target_names = ["Class 0", "Class 1"]
    else:
        # Multi-class
        thresholds = np.percentile(logits, np.linspace(0, 100, n_classes + 1)[1:-1])
        y = np.digitize(logits, thresholds)
        target_names = [f"Class {i}" for i in range(n_classes)]
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    metadata = DatasetMetadata(
        feature_names=feature_names,
        feature_types=["numerical"] * n_features,
        target_names=target_names,
    )
    
    return X, y, metadata


def preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """Preprocess data with train/test split and optional scaling.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction of data to use for testing.
        random_state: Random seed.
        scale_features: Whether to scale features.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler).
    """
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler


def save_dataset_metadata(
    metadata: DatasetMetadata, 
    filepath: Union[str, Path]
) -> None:
    """Save dataset metadata to JSON file.
    
    Args:
        metadata: Dataset metadata.
        filepath: Path to save metadata.
    """
    with open(filepath, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)


def load_dataset_metadata(filepath: Union[str, Path]) -> DatasetMetadata:
    """Load dataset metadata from JSON file.
    
    Args:
        filepath: Path to metadata file.
        
    Returns:
        Dataset metadata.
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return DatasetMetadata.from_dict(data)
