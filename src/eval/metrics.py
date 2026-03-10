"""Comprehensive evaluation metrics for calibration and uncertainty quantification."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


class CalibrationEvaluator:
    """Evaluator for calibration metrics and uncertainty quantification."""
    
    def __init__(self, n_bins: int = 10):
        """Initialize CalibrationEvaluator.
        
        Args:
            n_bins: Number of bins for calibration curve calculation.
        """
        self.n_bins = n_bins
    
    def expected_calibration_error(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray
    ) -> float:
        """Calculate Expected Calibration Error (ECE).
        
        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities for positive class.
            
        Returns:
            ECE value.
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def maximum_calibration_error(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray
    ) -> float:
        """Calculate Maximum Calibration Error (MCE).
        
        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities for positive class.
            
        Returns:
            MCE value.
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def reliability_diagram_data(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get data for reliability diagram.
        
        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities for positive class.
            
        Returns:
            Tuple of (bin_centers, bin_accuracies, bin_counts).
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.sum()
            
            if prop_in_bin > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(y_true[in_bin].mean())
                bin_counts.append(prop_in_bin)
            else:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        return np.array(bin_centers), np.array(bin_accuracies), np.array(bin_counts)
    
    def evaluate(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model calibration and performance.
        
        Args:
            model: Trained model with predict_proba method.
            X: Test features.
            y: Test targets.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Get predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        
        # For multi-class, use max probability
        if y_prob.shape[1] > 2:
            y_prob_max = np.max(y_prob, axis=1)
            y_prob_positive = y_prob_max
        else:
            y_prob_positive = y_prob[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "brier_score": brier_score_loss(y, y_prob_positive),
            "log_loss": log_loss(y, y_prob),
            "ece": self.expected_calibration_error(y, y_prob_positive),
            "mce": self.maximum_calibration_error(y, y_prob_positive),
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(y)) == 2:
            metrics["roc_auc"] = roc_auc_score(y, y_prob_positive)
        
        return metrics
    
    def evaluate_multiple_models(
        self, 
        models: Dict[str, Any], 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate multiple models and return comparison.
        
        Args:
            models: Dictionary of model names to models.
            X: Test features.
            y: Test targets.
            
        Returns:
            Dictionary of model names to metrics.
        """
        results = {}
        for name, model in models.items():
            results[name] = self.evaluate(model, X, y)
        return results


class UncertaintyEvaluator:
    """Evaluator for uncertainty quantification methods."""
    
    def __init__(self):
        """Initialize UncertaintyEvaluator."""
        pass
    
    def prediction_entropy(self, y_prob: np.ndarray) -> np.ndarray:
        """Calculate prediction entropy.
        
        Args:
            y_prob: Predicted probabilities.
            
        Returns:
            Entropy values.
        """
        # Avoid log(0) by adding small epsilon
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return -np.sum(y_prob * np.log(y_prob), axis=1)
    
    def mutual_information(self, y_prob_samples: np.ndarray) -> np.ndarray:
        """Calculate mutual information from MC samples.
        
        Args:
            y_prob_samples: MC samples of predicted probabilities.
            
        Returns:
            Mutual information values.
        """
        # Average over samples
        y_prob_mean = np.mean(y_prob_samples, axis=0)
        
        # Calculate entropy of mean
        entropy_mean = self.prediction_entropy(y_prob_mean)
        
        # Calculate average entropy
        entropy_samples = np.array([self.prediction_entropy(sample) for sample in y_prob_samples])
        entropy_avg = np.mean(entropy_samples, axis=0)
        
        # Mutual information = entropy_mean - entropy_avg
        return entropy_mean - entropy_avg
    
    def aleatoric_epistemic_uncertainty(
        self, 
        y_prob_samples: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Separate aleatoric and epistemic uncertainty.
        
        Args:
            y_prob_samples: MC samples of predicted probabilities.
            
        Returns:
            Tuple of (aleatoric_uncertainty, epistemic_uncertainty).
        """
        # Average over samples
        y_prob_mean = np.mean(y_prob_samples, axis=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = self.prediction_entropy(y_prob_mean)
        
        # Aleatoric uncertainty (data uncertainty)
        entropy_samples = np.array([self.prediction_entropy(sample) for sample in y_prob_samples])
        aleatoric = np.mean(entropy_samples, axis=0)
        
        return aleatoric, epistemic


class ModelComparison:
    """Compare multiple models across different metrics."""
    
    def __init__(self):
        """Initialize ModelComparison."""
        self.calibration_evaluator = CalibrationEvaluator()
        self.uncertainty_evaluator = UncertaintyEvaluator()
    
    def create_leaderboard(
        self, 
        models: Dict[str, Any], 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Create a leaderboard comparing models.
        
        Args:
            models: Dictionary of model names to models.
            X: Test features.
            y: Test targets.
            
        Returns:
            Leaderboard with rankings and metrics.
        """
        # Evaluate all models
        results = self.calibration_evaluator.evaluate_multiple_models(models, X, y)
        
        # Create rankings for each metric
        leaderboard = {
            "models": results,
            "rankings": {}
        }
        
        # Get all metric names
        metric_names = list(next(iter(results.values())).keys())
        
        for metric in metric_names:
            # Sort models by metric (lower is better for most metrics)
            sorted_models = sorted(
                results.items(), 
                key=lambda x: x[1][metric],
                reverse=(metric in ["accuracy", "roc_auc"])  # Higher is better for these
            )
            leaderboard["rankings"][metric] = sorted_models
        
        return leaderboard
    
    def print_leaderboard(self, leaderboard: Dict[str, Any]) -> None:
        """Print formatted leaderboard.
        
        Args:
            leaderboard: Leaderboard data.
        """
        print("Model Performance Leaderboard")
        print("=" * 50)
        
        for metric, ranking in leaderboard["rankings"].items():
            print(f"\n{metric.upper()}:")
            for i, (model_name, metrics) in enumerate(ranking, 1):
                print(f"  {i}. {model_name}: {metrics[metric]:.4f}")
