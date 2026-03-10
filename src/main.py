"""Modernized Trust Calibration in AI Systems - Main Script.

This script demonstrates comprehensive trust calibration and uncertainty quantification
methods for AI systems, including Platt Scaling, Isotonic Regression, and advanced
calibration techniques.

WARNING: This project is for research and educational purposes only.
XAI outputs may be unstable or misleading. DO NOT USE FOR REGULATED DECISIONS WITHOUT HUMAN REVIEW.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data.datasets import (
    DatasetMetadata,
    create_synthetic_dataset,
    load_breast_cancer_dataset,
    load_iris_dataset,
    load_wine_dataset,
    preprocess_data,
    save_dataset_metadata,
)
from src.eval.metrics import CalibrationEvaluator, ModelComparison
from src.models.calibration import CalibratedModel
from src.utils.device import get_device, set_seed
from src.viz.plots import CalibrationVisualizer


def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """Load specified dataset.
    
    Args:
        dataset_name: Name of dataset to load.
        
    Returns:
        Tuple of (features, targets, metadata).
    """
    if dataset_name == "iris":
        return load_iris_dataset()
    elif dataset_name == "wine":
        return load_wine_dataset()
    elif dataset_name == "breast_cancer":
        return load_breast_cancer_dataset()
    elif dataset_name == "synthetic":
        return create_synthetic_dataset(n_samples=1000, n_features=10, n_classes=2)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def train_models(
    X_train: np.ndarray, 
    y_train: np.ndarray
) -> Dict[str, Any]:
    """Train multiple models for comparison.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        
    Returns:
        Dictionary of trained models.
    """
    models = {}
    
    # Base Random Forest
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    models["Random Forest"].fit(X_train, y_train)
    
    # Platt Scaling
    models["Platt Scaling"] = CalibratedModel(
        base_model=RandomForestClassifier(n_estimators=100, random_state=42),
        calibration_method="platt"
    )
    models["Platt Scaling"].fit(X_train, y_train)
    
    # Isotonic Regression
    models["Isotonic Regression"] = CalibratedModel(
        base_model=RandomForestClassifier(n_estimators=100, random_state=42),
        calibration_method="isotonic"
    )
    models["Isotonic Regression"].fit(X_train, y_train)
    
    # Logistic Regression (baseline)
    models["Logistic Regression"] = LogisticRegression(random_state=42)
    models["Logistic Regression"].fit(X_train, y_train)
    
    return models


def evaluate_models(
    models: Dict[str, Any], 
    X_test: np.ndarray, 
    y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Evaluate all models and return metrics.
    
    Args:
        models: Dictionary of trained models.
        X_test: Test features.
        y_test: Test targets.
        
    Returns:
        Dictionary of model evaluation results.
    """
    evaluator = CalibrationEvaluator()
    return evaluator.evaluate_multiple_models(models, X_test, y_test)


def create_visualizations(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    results: Dict[str, Dict[str, float]],
    output_dir: Path,
) -> None:
    """Create and save visualization plots.
    
    Args:
        models: Dictionary of trained models.
        X_test: Test features.
        y_test: Test targets.
        results: Model evaluation results.
        output_dir: Directory to save plots.
    """
    visualizer = CalibrationVisualizer()
    
    # Create calibration curves for each model
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)
        
        # For multi-class, use max probability
        if y_prob.shape[1] > 2:
            y_prob_positive = np.max(y_prob, axis=1)
        else:
            y_prob_positive = y_prob[:, 1]
        
        # Plot calibration curve
        fig = visualizer.plot_calibration_curve(
            y_test, y_prob_positive, model_name=name
        )
        visualizer.save_plot(
            fig, output_dir / f"calibration_curve_{name.lower().replace(' ', '_')}.png"
        )
        
        # Plot reliability diagram
        fig = visualizer.plot_reliability_diagram(
            y_test, y_prob_positive, model_name=name
        )
        visualizer.save_plot(
            fig, output_dir / f"reliability_diagram_{name.lower().replace(' ', '_')}.png"
        )
        
        # Plot confidence histogram
        fig = visualizer.plot_confidence_histogram(
            y_prob_positive, y_test, model_name=name
        )
        visualizer.save_plot(
            fig, output_dir / f"confidence_histogram_{name.lower().replace(' ', '_')}.png"
        )
    
    # Plot model comparison
    fig = visualizer.plot_model_comparison(results)
    visualizer.save_plot(fig, output_dir / "model_comparison.png")


def print_results(results: Dict[str, Dict[str, float]]) -> None:
    """Print formatted results.
    
    Args:
        results: Model evaluation results.
    """
    print("\n" + "="*80)
    print("TRUST CALIBRATION EVALUATION RESULTS")
    print("="*80)
    
    # Print header
    metrics = list(next(iter(results.values())).keys())
    header = f"{'Model':<20} " + " ".join(f"{metric:>12}" for metric in metrics)
    print(header)
    print("-" * len(header))
    
    # Print results for each model
    for model_name, metrics_dict in results.items():
        row = f"{model_name:<20} "
        for metric in metrics:
            row += f"{metrics_dict[metric]:>12.4f} "
        print(row)
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("- Lower ECE (Expected Calibration Error) is better")
    print("- Lower MCE (Maximum Calibration Error) is better")
    print("- Lower Brier Score is better")
    print("- Higher Accuracy is better")
    print("- Higher ROC AUC is better (for binary classification)")
    print("="*80)


def main():
    """Main function to run trust calibration analysis."""
    parser = argparse.ArgumentParser(
        description="Trust Calibration in AI Systems - Comprehensive Analysis"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="iris",
        choices=["iris", "wine", "breast_cancer", "synthetic"],
        help="Dataset to use for analysis"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="assets/results",
        help="Directory to save results and plots"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.2,
        help="Fraction of data to use for testing"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Trust Calibration in AI Systems")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    X, y, metadata = load_dataset(args.dataset)
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Feature names: {metadata.feature_names}")
    print()
    
    # Save dataset metadata
    save_dataset_metadata(metadata, output_dir / "dataset_metadata.json")
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print()
    
    # Train models
    print("Training models...")
    models = train_models(X_train, y_train)
    print(f"Trained {len(models)} models: {list(models.keys())}")
    print()
    
    # Evaluate models
    print("Evaluating models...")
    results = evaluate_models(models, X_test, y_test)
    print()
    
    # Print results
    print_results(results)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(models, X_test, y_test, results, output_dir)
    print(f"Plots saved to: {output_dir}")
    
    # Save results to JSON
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    # Create leaderboard
    print("\nCreating model leaderboard...")
    comparison = ModelComparison()
    leaderboard = comparison.create_leaderboard(models, X_test, y_test)
    comparison.print_leaderboard(leaderboard)
    
    # Save leaderboard
    leaderboard_file = output_dir / "leaderboard.json"
    with open(leaderboard_file, "w") as f:
        json.dump(leaderboard, f, indent=2)
    print(f"Leaderboard saved to: {leaderboard_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("IMPORTANT DISCLAIMERS:")
    print("- Results are for research and educational purposes only")
    print("- Calibration methods may not generalize across different datasets")
    print("- Always validate AI system outputs with domain experts")
    print("- Consider potential biases in training data and model predictions")
    print("- DO NOT USE FOR REGULATED DECISIONS WITHOUT HUMAN REVIEW")
    print("="*80)


if __name__ == "__main__":
    main()
