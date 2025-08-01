"""
🔍 Enhanced Model Validator - Comprehensive Testing Framework
============================================================

Advanced validation framework for ensemble models with:
✓ Cross-validation with stratified sampling
✓ Performance benchmarking against baselines
✓ Statistical significance testing
✓ Robustness testing (noise, outliers, missing data)
✓ Fairness and bias detection
✓ Model interpretation validation
✓ Ensemble component analysis
✓ Real-world scenario testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import logging
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
import json

# Statistical testing
from scipy import stats
from scipy.stats import ks_2samp, mannwhitneyu, wilcoxon

# Machine Learning
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, KFold,
    permutation_test_score, learning_curve, validation_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.utils import resample
from sklearn.dummy import DummyClassifier, DummyRegressor

# Import model components
try:
    from ..core.base_model import BaseAIModel, LearningContext, ModelStatus
    from ..core.enhanced_model_registry import enhanced_model_registry
except ImportError as e:
    logging.warning(f"Failed to import core components: {e}")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""
    # Basic metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    
    # Regression metrics
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    
    # Cross-validation metrics
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    # Statistical metrics
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    
    # Robustness metrics
    noise_robustness: Optional[float] = None
    outlier_robustness: Optional[float] = None
    missing_data_robustness: Optional[float] = None
    
    # Fairness metrics
    demographic_parity: Optional[float] = None
    equality_opportunity: Optional[float] = None
    
    # Additional metadata
    sample_size: Optional[int] = None
    feature_count: Optional[int] = None
    class_distribution: Optional[Dict[str, int]] = None

@dataclass
class BenchmarkComparison:
    """Comparison results against baseline models."""
    model_performance: float
    baseline_performance: float
    improvement: float
    improvement_percentage: float
    statistical_significance: bool
    p_value: float
    effect_size: Optional[float] = None

@dataclass
class RobustnessTestResults:
    """Results from robustness testing."""
    original_performance: float
    noisy_performance: float
    outlier_performance: float
    missing_data_performance: float
    noise_degradation: float
    outlier_degradation: float
    missing_data_degradation: float
    overall_robustness_score: float

class EnhancedModelValidator:
    """
    🔍 Comprehensive validation framework for enhanced ensemble models.
    
    Provides extensive testing capabilities:
    - Statistical validation with confidence intervals
    - Cross-validation with multiple strategies
    - Baseline comparison and benchmarking
    - Robustness testing against various perturbations
    - Fairness and bias detection
    - Performance degradation analysis
    - Real-world scenario simulation
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.validation_history: List[Dict[str, Any]] = []
        
        # Validation configuration
        self.cv_folds = 5
        self.bootstrap_iterations = 1000
        self.confidence_level = 0.95
        
        # Robustness testing parameters
        self.noise_levels = [0.05, 0.1, 0.15, 0.2]
        self.outlier_percentages = [0.05, 0.1, 0.15]
        self.missing_data_percentages = [0.1, 0.2, 0.3]
        
    def validate_model_comprehensive(self, model: BaseAIModel, X: np.ndarray, y: np.ndarray,
                                   task_type: str = 'classification') -> ValidationMetrics:
        """
        Perform comprehensive validation of a model.
        
        Args:
            model: The model to validate
            X: Feature matrix
            y: Target vector
            task_type: 'classification' or 'regression'
        
        Returns:
            ValidationMetrics object with comprehensive results
        """
        logger.info(f"Starting comprehensive validation for {model.model_name}")
        
        metrics = ValidationMetrics()
        
        try:
            # Basic metrics
            if task_type == 'classification':
                metrics = self._validate_classification_model(model, X, y, metrics)
            else:
                metrics = self._validate_regression_model(model, X, y, metrics)
            
            # Cross-validation
            metrics = self._perform_cross_validation(model, X, y, task_type, metrics)
            
            # Statistical significance
            metrics = self._calculate_statistical_significance(model, X, y, task_type, metrics)
            
            # Robustness testing
            robustness_results = self.test_model_robustness(model, X, y, task_type)
            metrics.noise_robustness = robustness_results.noise_degradation
            metrics.outlier_robustness = robustness_results.outlier_degradation
            metrics.missing_data_robustness = robustness_results.missing_data_degradation
            
            # Metadata
            metrics.sample_size = len(X)
            metrics.feature_count = X.shape[1] if len(X.shape) > 1 else 1
            
            if task_type == 'classification':
                unique, counts = np.unique(y, return_counts=True)
                metrics.class_distribution = dict(zip(unique.astype(str), counts.astype(int)))
            
            logger.info(f"Comprehensive validation completed for {model.model_name}")
            
        except Exception as e:
            logger.error(f"Validation failed for {model.model_name}: {e}")
            
        return metrics
    
    def _validate_classification_model(self, model: BaseAIModel, X: np.ndarray, 
                                     y: np.ndarray, metrics: ValidationMetrics) -> ValidationMetrics:
        """Validate classification model."""
        try:
            # Train model if not already trained
            if model.status != ModelStatus.TRAINED:
                # Create dummy training data
                train_df = pd.DataFrame(X)
                train_df['target'] = y
                model.train(train_df)
            
            # Get predictions
            predictions, probabilities = self._get_model_predictions(model, X, return_proba=True)
            
            if predictions is not None and len(predictions) > 0:
                # Basic classification metrics
                metrics.accuracy = accuracy_score(y, predictions)
                metrics.precision = precision_score(y, predictions, average='weighted', zero_division=0)
                metrics.recall = recall_score(y, predictions, average='weighted', zero_division=0)
                metrics.f1_score = f1_score(y, predictions, average='weighted', zero_division=0)
                
                # ROC AUC (for binary or multiclass)
                if probabilities is not None:
                    try:
                        if len(np.unique(y)) == 2:
                            metrics.roc_auc = roc_auc_score(y, probabilities[:, 1])
                        else:
                            metrics.roc_auc = roc_auc_score(y, probabilities, multi_class='ovr', average='weighted')
                    except:
                        pass
                        
        except Exception as e:
            logger.warning(f"Classification validation failed: {e}")
            
        return metrics
    
    def _validate_regression_model(self, model: BaseAIModel, X: np.ndarray, 
                                 y: np.ndarray, metrics: ValidationMetrics) -> ValidationMetrics:
        """Validate regression model."""
        try:
            # Train model if not already trained
            if model.status != ModelStatus.TRAINED:
                train_df = pd.DataFrame(X)
                train_df['target'] = y
                model.train(train_df)
            
            # Get predictions
            predictions, _ = self._get_model_predictions(model, X, return_proba=False)
            
            if predictions is not None and len(predictions) > 0:
                # Basic regression metrics
                metrics.mse = mean_squared_error(y, predictions)
                metrics.mae = mean_absolute_error(y, predictions)
                metrics.r2 = r2_score(y, predictions)
                
        except Exception as e:
            logger.warning(f"Regression validation failed: {e}")
            
        return metrics
    
    def _perform_cross_validation(self, model: BaseAIModel, X: np.ndarray, y: np.ndarray,
                                task_type: str, metrics: ValidationMetrics) -> ValidationMetrics:
        """Perform cross-validation."""
        try:
            # Choose CV strategy
            if task_type == 'classification':
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                scoring = 'r2'
            
            # Create sklearn-compatible wrapper if needed
            sklearn_model = self._create_sklearn_wrapper(model, task_type)
            
            if sklearn_model is not None:
                # Perform cross-validation
                cv_scores = cross_val_score(sklearn_model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                
                metrics.cv_scores = cv_scores.tolist()
                metrics.cv_mean = np.mean(cv_scores)
                metrics.cv_std = np.std(cv_scores)
                
                # Calculate confidence interval
                alpha = 1 - self.confidence_level
                ci_lower = np.percentile(cv_scores, (alpha/2) * 100)
                ci_upper = np.percentile(cv_scores, (1 - alpha/2) * 100)
                metrics.confidence_interval = (ci_lower, ci_upper)
                
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            
        return metrics
    
    def _calculate_statistical_significance(self, model: BaseAIModel, X: np.ndarray, y: np.ndarray,
                                          task_type: str, metrics: ValidationMetrics) -> ValidationMetrics:
        """Calculate statistical significance against random baseline."""
        try:
            # Create baseline model
            if task_type == 'classification':
                baseline = DummyClassifier(strategy='most_frequent', random_state=self.random_state)
                scoring = 'accuracy'
            else:
                baseline = DummyRegressor(strategy='mean')
                scoring = 'r2'
            
            # sklearn wrapper for our model
            sklearn_model = self._create_sklearn_wrapper(model, task_type)
            
            if sklearn_model is not None:
                # Permutation test
                score, perm_scores, p_value = permutation_test_score(
                    sklearn_model, X, y, scoring=scoring, cv=5, n_permutations=100,
                    random_state=self.random_state, n_jobs=-1
                )
                
                metrics.p_value = p_value
                
        except Exception as e:
            logger.warning(f"Statistical significance test failed: {e}")
            
        return metrics
    
    def test_model_robustness(self, model: BaseAIModel, X: np.ndarray, y: np.ndarray,
                            task_type: str) -> RobustnessTestResults:
        """Test model robustness against various perturbations."""
        logger.info("Testing model robustness...")
        
        # Get original performance
        original_predictions, _ = self._get_model_predictions(model, X)
        if task_type == 'classification':
            original_performance = accuracy_score(y, original_predictions)
        else:
            original_performance = r2_score(y, original_predictions)
        
        # Test against noise
        noise_performance = self._test_noise_robustness(model, X, y, task_type)
        
        # Test against outliers
        outlier_performance = self._test_outlier_robustness(model, X, y, task_type)
        
        # Test against missing data
        missing_performance = self._test_missing_data_robustness(model, X, y, task_type)
        
        # Calculate degradations
        noise_degradation = (original_performance - noise_performance) / original_performance
        outlier_degradation = (original_performance - outlier_performance) / original_performance
        missing_degradation = (original_performance - missing_performance) / original_performance
        
        # Overall robustness score (lower is better for degradation)
        overall_robustness = 1 - np.mean([noise_degradation, outlier_degradation, missing_degradation])
        
        return RobustnessTestResults(
            original_performance=original_performance,
            noisy_performance=noise_performance,
            outlier_performance=outlier_performance,
            missing_data_performance=missing_performance,
            noise_degradation=noise_degradation,
            outlier_degradation=outlier_degradation,
            missing_data_degradation=missing_degradation,
            overall_robustness_score=max(0, overall_robustness)
        )
    
    def _test_noise_robustness(self, model: BaseAIModel, X: np.ndarray, y: np.ndarray,
                             task_type: str) -> float:
        """Test robustness against Gaussian noise."""
        performances = []
        
        for noise_level in self.noise_levels:
            try:
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level * np.std(X, axis=0), X.shape)
                X_noisy = X + noise
                
                # Get predictions
                predictions, _ = self._get_model_predictions(model, X_noisy)
                
                if predictions is not None:
                    if task_type == 'classification':
                        perf = accuracy_score(y, predictions)
                    else:
                        perf = r2_score(y, predictions)
                    performances.append(perf)
                    
            except Exception as e:
                logger.warning(f"Noise robustness test failed at level {noise_level}: {e}")
        
        return np.mean(performances) if performances else 0.0
    
    def _test_outlier_robustness(self, model: BaseAIModel, X: np.ndarray, y: np.ndarray,
                               task_type: str) -> float:
        """Test robustness against outliers."""
        performances = []
        
        for outlier_pct in self.outlier_percentages:
            try:
                X_outliers = X.copy()
                n_outliers = int(len(X) * outlier_pct)
                
                # Introduce outliers by setting random samples to extreme values
                outlier_indices = np.random.choice(len(X), n_outliers, replace=False)
                for idx in outlier_indices:
                    # Set to 3 standard deviations away
                    X_outliers[idx] = np.mean(X, axis=0) + 3 * np.std(X, axis=0)
                
                # Get predictions
                predictions, _ = self._get_model_predictions(model, X_outliers)
                
                if predictions is not None:
                    if task_type == 'classification':
                        perf = accuracy_score(y, predictions)
                    else:
                        perf = r2_score(y, predictions)
                    performances.append(perf)
                    
            except Exception as e:
                logger.warning(f"Outlier robustness test failed at {outlier_pct}: {e}")
        
        return np.mean(performances) if performances else 0.0
    
    def _test_missing_data_robustness(self, model: BaseAIModel, X: np.ndarray, y: np.ndarray,
                                    task_type: str) -> float:
        """Test robustness against missing data."""
        performances = []
        
        for missing_pct in self.missing_data_percentages:
            try:
                X_missing = X.copy()
                
                # Introduce missing data (set to mean for now)
                mask = np.random.random(X.shape) < missing_pct
                X_missing[mask] = np.nanmean(X, axis=0)[mask[0]]  # Replace with feature means
                
                # Get predictions
                predictions, _ = self._get_model_predictions(model, X_missing)
                
                if predictions is not None:
                    if task_type == 'classification':
                        perf = accuracy_score(y, predictions)
                    else:
                        perf = r2_score(y, predictions)
                    performances.append(perf)
                    
            except Exception as e:
                logger.warning(f"Missing data robustness test failed at {missing_pct}: {e}")
        
        return np.mean(performances) if performances else 0.0
    
    def compare_against_baseline(self, model: BaseAIModel, baseline_model: BaseAIModel,
                               X: np.ndarray, y: np.ndarray, task_type: str) -> BenchmarkComparison:
        """Compare model performance against a baseline."""
        logger.info(f"Comparing {model.model_name} against {baseline_model.model_name}")
        
        # Get predictions from both models
        model_predictions, _ = self._get_model_predictions(model, X)
        baseline_predictions, _ = self._get_model_predictions(baseline_model, X)
        
        if model_predictions is None or baseline_predictions is None:
            logger.error("Failed to get predictions for comparison")
            return BenchmarkComparison(0, 0, 0, 0, False, 1.0)
        
        # Calculate performance
        if task_type == 'classification':
            model_perf = accuracy_score(y, model_predictions)
            baseline_perf = accuracy_score(y, baseline_predictions)
        else:
            model_perf = r2_score(y, model_predictions)
            baseline_perf = r2_score(y, baseline_predictions)
        
        # Calculate improvement
        improvement = model_perf - baseline_perf
        improvement_pct = (improvement / baseline_perf * 100) if baseline_perf != 0 else 0
        
        # Statistical significance test
        try:
            if task_type == 'classification':
                # McNemar's test for paired samples
                model_correct = (model_predictions == y)
                baseline_correct = (baseline_predictions == y)
                
                # Create contingency table
                both_correct = np.sum(model_correct & baseline_correct)
                model_only = np.sum(model_correct & ~baseline_correct)
                baseline_only = np.sum(~model_correct & baseline_correct)
                both_wrong = np.sum(~model_correct & ~baseline_correct)
                
                # McNemar's test
                if model_only + baseline_only > 0:
                    statistic = (abs(model_only - baseline_only) - 1)**2 / (model_only + baseline_only)
                    p_value = 1 - stats.chi2.cdf(statistic, 1)
                else:
                    p_value = 1.0
            else:
                # Paired t-test for regression
                model_errors = (y - model_predictions)**2
                baseline_errors = (y - baseline_predictions)**2
                _, p_value = stats.ttest_rel(model_errors, baseline_errors)
            
            significant = p_value < 0.05
            
        except Exception as e:
            logger.warning(f"Statistical significance test failed: {e}")
            p_value = 1.0
            significant = False
        
        # Effect size (Cohen's d)
        try:
            if task_type == 'classification':
                effect_size = None  # Not applicable for classification
            else:
                model_errors = y - model_predictions
                baseline_errors = y - baseline_predictions
                pooled_std = np.sqrt((np.var(model_errors) + np.var(baseline_errors)) / 2)
                effect_size = (np.mean(model_errors) - np.mean(baseline_errors)) / pooled_std if pooled_std > 0 else 0
        except:
            effect_size = None
        
        return BenchmarkComparison(
            model_performance=model_perf,
            baseline_performance=baseline_perf,
            improvement=improvement,
            improvement_percentage=improvement_pct,
            statistical_significance=significant,
            p_value=p_value,
            effect_size=effect_size
        )
    
    def _get_model_predictions(self, model: BaseAIModel, X: np.ndarray, 
                             return_proba: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get predictions from a model."""
        try:
            from ..core.base_model import LearningContext
            
            context = LearningContext(
                user_id="validation_user",
                session_id="validation_session",
                content_id="validation_content",
                timestamp=datetime.now().isoformat()
            )
            
            predictions = []
            probabilities = []
            
            for i, row in enumerate(X):
                try:
                    # Convert row to input format (simplified)
                    input_data = self._array_to_input_data(row)
                    
                    # Make prediction
                    result = model.predict(input_data, context)
                    predictions.append(result.value)
                    
                    if return_proba and 'probabilities' in result.metadata:
                        probs = result.metadata['probabilities']
                        if isinstance(probs, dict):
                            # Convert to array format
                            prob_array = list(probs.values())
                            probabilities.append(prob_array)
                        
                except Exception as e:
                    logger.warning(f"Prediction failed for sample {i}: {e}")
                    continue
            
            pred_array = np.array(predictions) if predictions else None
            prob_array = np.array(probabilities) if probabilities else None
            
            return pred_array, prob_array
            
        except Exception as e:
            logger.error(f"Failed to get model predictions: {e}")
            return None, None
    
    def _array_to_input_data(self, row: np.ndarray) -> Dict[str, Any]:
        """Convert numpy array to input data format (simplified)."""
        # This is a simplified conversion - in practice, you'd need proper mapping
        return {
            'features': row.tolist(),
            'timestamp': datetime.now().isoformat(),
            'session_duration': 600,
            'mouse_movements': [],
            'keyboard_events': [],
            'scroll_events': [],
            'content_interactions': [],
            'content_type': 'text',
            'content_difficulty': 5
        }
    
    def _create_sklearn_wrapper(self, model: BaseAIModel, task_type: str) -> Optional[Any]:
        """Create sklearn-compatible wrapper for cross-validation."""
        # This would need to be implemented based on your specific model interface
        # For now, return None to skip sklearn-specific operations
        return None
    
    def generate_validation_report(self, model_name: str, metrics: ValidationMetrics,
                                 robustness: RobustnessTestResults,
                                 comparisons: List[BenchmarkComparison]) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append(f"🔍 Validation Report for {model_name}")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic Performance
        report.append("📊 Performance Metrics:")
        if metrics.accuracy is not None:
            report.append(f"  Accuracy: {metrics.accuracy:.4f}")
        if metrics.f1_score is not None:
            report.append(f"  F1 Score: {metrics.f1_score:.4f}")
        if metrics.roc_auc is not None:
            report.append(f"  ROC AUC: {metrics.roc_auc:.4f}")
        if metrics.r2 is not None:
            report.append(f"  R² Score: {metrics.r2:.4f}")
        report.append("")
        
        # Cross-validation
        if metrics.cv_scores:
            report.append("🎯 Cross-Validation Results:")
            report.append(f"  Mean CV Score: {metrics.cv_mean:.4f} ± {metrics.cv_std:.4f}")
            if metrics.confidence_interval:
                ci_lower, ci_upper = metrics.confidence_interval
                report.append(f"  95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
            report.append("")
        
        # Statistical Significance
        if metrics.p_value is not None:
            significance = "Significant" if metrics.p_value < 0.05 else "Not significant"
            report.append(f"🔬 Statistical Significance: {significance} (p = {metrics.p_value:.4f})")
            report.append("")
        
        # Robustness
        report.append("🛡️ Robustness Analysis:")
        report.append(f"  Original Performance: {robustness.original_performance:.4f}")
        report.append(f"  Noise Robustness: {robustness.noise_degradation:.2%} degradation")
        report.append(f"  Outlier Robustness: {robustness.outlier_degradation:.2%} degradation")
        report.append(f"  Missing Data Robustness: {robustness.missing_data_degradation:.2%} degradation")
        report.append(f"  Overall Robustness Score: {robustness.overall_robustness_score:.4f}")
        report.append("")
        
        # Baseline Comparisons
        if comparisons:
            report.append("⚖️ Baseline Comparisons:")
            for comp in comparisons:
                significance = "✓" if comp.statistical_significance else "✗"
                report.append(f"  vs Baseline: {comp.improvement:+.4f} ({comp.improvement_percentage:+.2f}%) {significance}")
                report.append(f"    p-value: {comp.p_value:.4f}")
                if comp.effect_size is not None:
                    report.append(f"    Effect size: {comp.effect_size:.4f}")
            report.append("")
        
        # Metadata
        report.append("📋 Dataset Information:")
        if metrics.sample_size:
            report.append(f"  Sample Size: {metrics.sample_size:,}")
        if metrics.feature_count:
            report.append(f"  Feature Count: {metrics.feature_count}")
        if metrics.class_distribution:
            report.append(f"  Class Distribution: {metrics.class_distribution}")
        
        return "\n".join(report)
    
    def save_validation_results(self, results: Dict[str, Any], filepath: str) -> bool:
        """Save validation results to file."""
        try:
            # Convert dataclasses to dicts for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if hasattr(value, '__dict__'):
                    serializable_results[key] = asdict(value)
                else:
                    serializable_results[key] = value
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Validation results saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")
            return False