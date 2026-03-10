"""Tests for Trust Calibration in AI Systems."""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.data.datasets import create_synthetic_dataset, DatasetMetadata
from src.eval.metrics import CalibrationEvaluator, UncertaintyEvaluator
from src.models.calibration import CalibratedModel, PlattScaling, IsotonicCalibration
from src.utils.device import get_device, set_seed
from src.viz.plots import CalibrationVisualizer


class TestDatasetUtils:
    """Test dataset utilities."""
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        X, y, metadata = create_synthetic_dataset(n_samples=100, n_features=5, n_classes=2)
        
        assert X.shape == (100, 5)
        assert y.shape == (100,)
        assert len(np.unique(y)) == 2
        assert len(metadata.feature_names) == 5
        assert len(metadata.target_names) == 2
    
    def test_dataset_metadata(self):
        """Test dataset metadata functionality."""
        metadata = DatasetMetadata(
            feature_names=["f1", "f2"],
            feature_types=["numerical", "categorical"],
            target_names=["class1", "class2"],
            sensitive_attributes=["f1"],
            monotonic_features=["f2"]
        )
        
        # Test to_dict
        data_dict = metadata.to_dict()
        assert "feature_names" in data_dict
        assert "sensitive_attributes" in data_dict
        
        # Test from_dict
        metadata_restored = DatasetMetadata.from_dict(data_dict)
        assert metadata_restored.feature_names == metadata.feature_names
        assert metadata_restored.sensitive_attributes == metadata.sensitive_attributes


class TestCalibrationMethods:
    """Test calibration methods."""
    
    def setup_method(self):
        """Setup test data."""
        set_seed(42)
        X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    
    def test_platt_scaling(self):
        """Test Platt Scaling calibration."""
        calibrator = PlattScaling()
        calibrator.fit(self.X_train, self.y_train)
        
        proba = calibrator.predict_proba(self.X_test)
        assert proba.shape == (len(self.X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-10)
    
    def test_isotonic_calibration(self):
        """Test Isotonic Regression calibration."""
        calibrator = IsotonicCalibration()
        calibrator.fit(self.X_train, self.y_train)
        
        proba = calibrator.predict_proba(self.X_test)
        assert proba.shape == (len(self.X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-10)
    
    def test_calibrated_model(self):
        """Test CalibratedModel wrapper."""
        model = CalibratedModel(calibration_method="platt")
        model.fit(self.X_train, self.y_train)
        
        # Test prediction
        y_pred = model.predict(self.X_test)
        assert len(y_pred) == len(self.X_test)
        
        # Test probability prediction
        y_proba = model.predict_proba(self.X_test)
        assert y_proba.shape == (len(self.X_test), 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-10)
        
        # Test calibration score
        score = model.get_calibration_score(self.X_test, self.y_test)
        assert isinstance(score, float)
        assert 0 <= score <= 1


class TestEvaluationMetrics:
    """Test evaluation metrics."""
    
    def setup_method(self):
        """Setup test data."""
        set_seed(42)
        X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train a simple model
        self.model = CalibratedModel(calibration_method="platt")
        self.model.fit(self.X_train, self.y_train)
    
    def test_calibration_evaluator(self):
        """Test CalibrationEvaluator."""
        evaluator = CalibrationEvaluator()
        
        # Test ECE calculation
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        ece = evaluator.expected_calibration_error(self.y_test, y_prob)
        assert isinstance(ece, float)
        assert 0 <= ece <= 1
        
        # Test MCE calculation
        mce = evaluator.maximum_calibration_error(self.y_test, y_prob)
        assert isinstance(mce, float)
        assert 0 <= mce <= 1
        
        # Test full evaluation
        results = evaluator.evaluate(self.model, self.X_test, self.y_test)
        assert isinstance(results, dict)
        assert "accuracy" in results
        assert "brier_score" in results
        assert "ece" in results
        assert "mce" in results
    
    def test_uncertainty_evaluator(self):
        """Test UncertaintyEvaluator."""
        evaluator = UncertaintyEvaluator()
        
        # Test prediction entropy
        y_prob = self.model.predict_proba(self.X_test)
        entropy = evaluator.prediction_entropy(y_prob)
        assert len(entropy) == len(self.X_test)
        assert np.all(entropy >= 0)
        
        # Test mutual information (with mock MC samples)
        y_prob_samples = np.random.dirichlet([1, 1], size=(10, len(self.X_test)))
        mi = evaluator.mutual_information(y_prob_samples)
        assert len(mi) == len(self.X_test)
        assert np.all(mi >= 0)
        
        # Test aleatoric/epistemic uncertainty separation
        aleatoric, epistemic = evaluator.aleatoric_epistemic_uncertainty(y_prob_samples)
        assert len(aleatoric) == len(self.X_test)
        assert len(epistemic) == len(self.X_test)
        assert np.all(aleatoric >= 0)
        assert np.all(epistemic >= 0)


class TestVisualization:
    """Test visualization utilities."""
    
    def setup_method(self):
        """Setup test data."""
        set_seed(42)
        X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        self.model = CalibratedModel(calibration_method="platt")
        self.model.fit(self.X_train, self.y_train)
        
        self.visualizer = CalibrationVisualizer()
    
    def test_calibration_curve_plot(self):
        """Test calibration curve plotting."""
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        fig = self.visualizer.plot_calibration_curve(
            self.y_test, y_prob, model_name="Test Model"
        )
        
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_reliability_diagram_plot(self):
        """Test reliability diagram plotting."""
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        fig = self.visualizer.plot_reliability_diagram(
            self.y_test, y_prob, model_name="Test Model"
        )
        
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_confidence_histogram_plot(self):
        """Test confidence histogram plotting."""
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        fig = self.visualizer.plot_confidence_histogram(
            y_prob, self.y_test, model_name="Test Model"
        )
        
        assert fig is not None
        assert len(fig.axes) > 0


class TestDeviceUtils:
    """Test device utilities."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device is not None
        assert hasattr(device, 'type')
    
    def test_set_seed(self):
        """Test seed setting."""
        # This should not raise an exception
        set_seed(42)
        set_seed(123)
    
    def test_device_info(self):
        """Test device info retrieval."""
        from src.utils.device import get_device_info
        
        info = get_device_info()
        assert isinstance(info, dict)
        assert "device_type" in info
        assert "cuda_available" in info


if __name__ == "__main__":
    pytest.main([__file__])
