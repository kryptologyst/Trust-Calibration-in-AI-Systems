#!/usr/bin/env python3
"""Evaluation script for Trust Calibration in AI Systems."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from src.data.datasets import load_dataset
from src.eval.metrics import CalibrationEvaluator, ModelComparison
from src.models.calibration import CalibratedModel
from src.utils.device import set_seed


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def evaluate_model_robustness(
    model: Any, 
    X: np.ndarray, 
    y: np.ndarray, 
    cv_folds: int = 5
) -> Dict[str, float]:
    """Evaluate model robustness using cross-validation.
    
    Args:
        model: Model to evaluate.
        X: Features.
        y: Targets.
        cv_folds: Number of CV folds.
        
    Returns:
        Dictionary of robustness metrics.
    """
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
    
    return {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "cv_min": cv_scores.min(),
        "cv_max": cv_scores.max()
    }


def run_comprehensive_evaluation(
    dataset_name: str,
    output_dir: Path,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Run comprehensive evaluation of calibration methods.
    
    Args:
        dataset_name: Name of dataset to use.
        output_dir: Output directory for results.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Comprehensive evaluation results.
    """
    set_seed(random_seed)
    
    # Load dataset
    X, y, metadata = load_dataset(dataset_name)
    
    # Preprocess data
    from src.data.datasets import preprocess_data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X, y, test_size=0.2, random_state=random_seed
    )
    
    # Train models
    models = {
        "Random Forest": CalibratedModel(calibration_method="platt"),
        "Platt Scaling": CalibratedModel(calibration_method="platt"),
        "Isotonic Regression": CalibratedModel(calibration_method="isotonic"),
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    # Evaluate models
    evaluator = CalibrationEvaluator()
    results = evaluator.evaluate_multiple_models(models, X_test, y_test)
    
    # Add robustness evaluation
    robustness_results = {}
    for name, model in models.items():
        robustness_results[name] = evaluate_model_robustness(model, X_train, y_train)
    
    # Create comprehensive results
    comprehensive_results = {
        "dataset": dataset_name,
        "dataset_info": {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
            "feature_names": metadata.feature_names,
            "target_names": metadata.target_names
        },
        "performance": results,
        "robustness": robustness_results,
        "config": {
            "random_seed": random_seed,
            "test_size": 0.2,
            "cv_folds": 5
        }
    }
    
    # Save results
    results_file = output_dir / f"comprehensive_evaluation_{dataset_name}.json"
    with open(results_file, "w") as f:
        json.dump(comprehensive_results, f, indent=2)
    
    # Create summary report
    summary_file = output_dir / f"evaluation_summary_{dataset_name}.txt"
    with open(summary_file, "w") as f:
        f.write(f"Comprehensive Evaluation Report - {dataset_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"- Samples: {len(X)}\n")
        f.write(f"- Features: {X.shape[1]}\n")
        f.write(f"- Classes: {len(np.unique(y))}\n\n")
        
        f.write("Model Performance:\n")
        for model_name, metrics in results.items():
            f.write(f"\n{model_name}:\n")
            for metric, value in metrics.items():
                f.write(f"  - {metric}: {value:.4f}\n")
        
        f.write("\nRobustness Analysis:\n")
        for model_name, robustness in robustness_results.items():
            f.write(f"\n{model_name}:\n")
            for metric, value in robustness.items():
                f.write(f"  - {metric}: {value:.4f}\n")
    
    return comprehensive_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Trust Calibration Models")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="iris",
        choices=["iris", "wine", "breast_cancer", "synthetic"],
        help="Dataset to evaluate"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="assets/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("DEBUG" if args.verbose else "INFO")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    logging.info(f"Starting comprehensive evaluation on {args.dataset}")
    
    try:
        results = run_comprehensive_evaluation(
            args.dataset, 
            output_dir, 
            args.seed
        )
        
        logging.info("Evaluation completed successfully")
        logging.info(f"Results saved to: {output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for model_name, metrics in results["performance"].items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        print(f"\nDetailed results saved to: {output_dir}")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
