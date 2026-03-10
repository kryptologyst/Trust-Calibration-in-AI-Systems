"""Visualization utilities for calibration and uncertainty analysis."""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure


class CalibrationVisualizer:
    """Visualizer for calibration analysis and uncertainty quantification."""
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (10, 8)):
        """Initialize CalibrationVisualizer.
        
        Args:
            style: Matplotlib style to use.
            figsize: Default figure size.
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
        n_bins: int = 10,
        ax: Optional[plt.Axes] = None,
    ) -> Figure:
        """Plot calibration curve.
        
        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities for positive class.
            model_name: Name of the model for legend.
            n_bins: Number of bins for calibration curve.
            ax: Matplotlib axes to plot on.
            
        Returns:
            Matplotlib figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        # Calculate calibration curve
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        # Plot calibration curve
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2, 
                label=f'{model_name} Calibration Curve', color=self.colors[0])
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', 
                label='Perfect Calibration', alpha=0.7)
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(f'{model_name} Calibration Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_reliability_diagram(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
        n_bins: int = 10,
        ax: Optional[plt.Axes] = None,
    ) -> Figure:
        """Plot reliability diagram with bin counts.
        
        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities for positive class.
            model_name: Name of the model for legend.
            n_bins: Number of bins for reliability diagram.
            ax: Matplotlib axes to plot on.
            
        Returns:
            Matplotlib figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        # Calculate bin data
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
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
        
        bin_centers = np.array(bin_centers)
        bin_accuracies = np.array(bin_accuracies)
        bin_counts = np.array(bin_counts)
        
        # Create subplot with two y-axes
        ax2 = ax.twinx()
        
        # Plot calibration curve
        ax.plot(bin_centers, bin_accuracies, marker='o', linewidth=2,
                label=f'{model_name} Calibration', color=self.colors[0])
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray',
                label='Perfect Calibration', alpha=0.7)
        
        # Plot bin counts as bars
        ax2.bar(bin_centers, bin_counts, width=0.8/n_bins, alpha=0.3,
                color=self.colors[1], label='Sample Count')
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12, color=self.colors[0])
        ax2.set_ylabel('Number of Samples', fontsize=12, color=self.colors[1])
        ax.set_title(f'{model_name} Reliability Diagram', fontsize=14, fontweight='bold')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_confidence_histogram(
        self,
        y_prob: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        model_name: str = "Model",
        n_bins: int = 20,
        ax: Optional[plt.Axes] = None,
    ) -> Figure:
        """Plot histogram of predicted probabilities.
        
        Args:
            y_prob: Predicted probabilities.
            y_true: True labels (optional, for colored histogram).
            model_name: Name of the model for legend.
            n_bins: Number of bins for histogram.
            ax: Matplotlib axes to plot on.
            
        Returns:
            Matplotlib figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        if y_true is not None:
            # Plot separate histograms for each class
            for class_label in np.unique(y_true):
                mask = y_true == class_label
                ax.hist(y_prob[mask], bins=n_bins, alpha=0.7, 
                       label=f'Class {class_label}', density=True)
        else:
            # Plot single histogram
            ax.hist(y_prob, bins=n_bins, alpha=0.7, 
                   label=f'{model_name} Predictions', density=True)
        
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{model_name} Confidence Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_uncertainty_analysis(
        self,
        y_prob_samples: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        model_name: str = "Model",
        ax: Optional[plt.Axes] = None,
    ) -> Figure:
        """Plot uncertainty analysis from MC samples.
        
        Args:
            y_prob_samples: MC samples of predicted probabilities.
            y_true: True labels (optional).
            model_name: Name of the model for legend.
            ax: Matplotlib axes to plot on.
            
        Returns:
            Matplotlib figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        # Calculate mean and std
        y_prob_mean = np.mean(y_prob_samples, axis=0)
        y_prob_std = np.std(y_prob_samples, axis=0)
        
        # Sort by mean probability for better visualization
        sort_idx = np.argsort(y_prob_mean)
        y_prob_mean_sorted = y_prob_mean[sort_idx]
        y_prob_std_sorted = y_prob_std[sort_idx]
        
        # Plot mean with error bars
        x = np.arange(len(y_prob_mean_sorted))
        ax.errorbar(x, y_prob_mean_sorted, yerr=y_prob_std_sorted, 
                   marker='o', capsize=3, capthick=2, linewidth=2,
                   label=f'{model_name} Mean ± Std', color=self.colors[0])
        
        if y_true is not None:
            y_true_sorted = y_true[sort_idx]
            ax.scatter(x, y_true_sorted, alpha=0.6, s=30,
                      label='True Labels', color=self.colors[1])
        
        ax.set_xlabel('Sample Index (sorted by mean probability)', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'{model_name} Uncertainty Analysis', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> Figure:
        """Plot comparison of multiple models.
        
        Args:
            results: Dictionary of model results.
            metrics: List of metrics to plot.
            ax: Matplotlib axes to plot on.
            
        Returns:
            Matplotlib figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        if metrics is None:
            metrics = list(next(iter(results.values())).keys())
        
        model_names = list(results.keys())
        x = np.arange(len(model_names))
        width = 0.8 / len(metrics)
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            ax.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def save_plot(self, fig: Figure, filepath: str, dpi: int = 300) -> None:
        """Save plot to file.
        
        Args:
            fig: Matplotlib figure.
            filepath: Path to save the plot.
            dpi: DPI for saving.
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
