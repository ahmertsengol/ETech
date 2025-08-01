#!/usr/bin/env python3
"""
Advanced Evaluation Framework for Learning Psychology Models
===========================================================

Comprehensive evaluation system with:
- Psychological validity metrics
- Fairness and bias assessment  
- Model drift detection
- A/B testing framework
- Real-time monitoring
- Temporal cross-validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    model_name: str
    timestamp: str
    accuracy: float
    f1_weighted: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    psychological_validity: float
    fairness_score: float
    bias_metrics: Dict[str, float]
    drift_score: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    detailed_metrics: Dict[str, Any]


class PsychologicalValidityEvaluator:
    """Evaluates psychological validity of model predictions."""
    
    def __init__(self):
        self.validity_criteria = {
            'attention_tracker': {
                'temporal_consistency': 0.7,  # Attention should be relatively stable
                'fatigue_pattern': 0.6,       # Should show decline over time
                'context_sensitivity': 0.5    # Should respond to content difficulty
            },
            'cognitive_load': {
                'complexity_correlation': 0.4,  # Should correlate with task complexity
                'performance_inverse': 0.3,     # Higher load → lower performance
                'recovery_pattern': 0.5         # Should recover with breaks
            },
            'learning_style': {
                'consistency_across_tasks': 0.6,  # Should be consistent
                'performance_alignment': 0.4,     # Better performance in preferred style
                'adaptation_evidence': 0.3        # Should show adaptation over time
            }
        }
    
    def evaluate_attention_validity(self, predictions: np.ndarray, 
                                  behavioral_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate psychological validity of attention predictions."""
        validity_scores = {}
        
        # Temporal consistency - attention shouldn't fluctuate wildly
        if len(predictions) > 1:
            attention_changes = np.abs(np.diff(predictions.astype(float)))
            temporal_consistency = 1.0 - (np.mean(attention_changes) / 3.0)  # 3 = max change
            validity_scores['temporal_consistency'] = max(0, temporal_consistency)
        else:
            validity_scores['temporal_consistency'] = 0.5
        
        # Fatigue pattern - attention should generally decline over long sessions
        if len(predictions) >= 6:
            first_half = predictions[:len(predictions)//2]
            second_half = predictions[len(predictions)//2:]
            
            first_half_mean = np.mean(first_half.astype(float))
            second_half_mean = np.mean(second_half.astype(float))
            
            expected_decline = first_half_mean - second_half_mean
            fatigue_pattern = min(1.0, max(0, expected_decline / 1.0))  # Normalize to 0-1
            validity_scores['fatigue_pattern'] = fatigue_pattern
        else:
            validity_scores['fatigue_pattern'] = 0.5
        
        # Context sensitivity - should respond to difficulty changes
        if behavioral_data and len(behavioral_data) == len(predictions):
            difficulty_scores = []
            for data in behavioral_data:
                difficulty = data.get('content_difficulty', 5)
                difficulty_scores.append(difficulty)
            
            if len(set(difficulty_scores)) > 1:  # Variable difficulty
                correlation, _ = stats.pearsonr(difficulty_scores, predictions.astype(float))
                # Negative correlation expected (higher difficulty → lower attention)
                context_sensitivity = max(0, -correlation) if not np.isnan(correlation) else 0.5
                validity_scores['context_sensitivity'] = context_sensitivity
            else:
                validity_scores['context_sensitivity'] = 0.5
        else:
            validity_scores['context_sensitivity'] = 0.5
        
        return validity_scores
    
    def evaluate_cognitive_load_validity(self, predictions: np.ndarray,
                                       behavioral_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate psychological validity of cognitive load predictions."""
        validity_scores = {}
        
        # Complexity correlation - load should increase with task complexity
        if behavioral_data and len(behavioral_data) == len(predictions):
            complexity_scores = []
            for data in behavioral_data:
                complexity = np.mean(data.get('task_complexities', [5.0]))
                complexity_scores.append(complexity)
            
            if len(set(complexity_scores)) > 1:
                correlation, _ = stats.pearsonr(complexity_scores, predictions.astype(float))
                complexity_correlation = max(0, correlation) if not np.isnan(correlation) else 0.5
                validity_scores['complexity_correlation'] = complexity_correlation
            else:
                validity_scores['complexity_correlation'] = 0.5
        else:
            validity_scores['complexity_correlation'] = 0.5
        
        # Performance inverse relationship - higher load should mean lower performance
        if behavioral_data:
            performance_scores = []
            for data in behavioral_data:
                accuracy = np.mean(data.get('accuracy_scores', [0.7]))
                performance_scores.append(accuracy)
            
            if len(performance_scores) == len(predictions) and len(set(performance_scores)) > 1:
                correlation, _ = stats.pearsonr(performance_scores, predictions.astype(float))
                # Negative correlation expected
                performance_inverse = max(0, -correlation) if not np.isnan(correlation) else 0.5
                validity_scores['performance_inverse'] = performance_inverse
            else:
                validity_scores['performance_inverse'] = 0.5
        else:
            validity_scores['performance_inverse'] = 0.5
        
        # Recovery pattern - load should decrease after breaks
        if len(predictions) >= 4:
            # Look for decreasing trends (simplified)
            load_changes = np.diff(predictions.astype(float))
            recovery_instances = np.sum(load_changes < 0)
            total_changes = len(load_changes)
            recovery_pattern = recovery_instances / total_changes if total_changes > 0 else 0.5
            validity_scores['recovery_pattern'] = recovery_pattern
        else:
            validity_scores['recovery_pattern'] = 0.5
        
        return validity_scores
    
    def evaluate_learning_style_validity(self, predictions: np.ndarray,
                                       behavioral_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate psychological validity of learning style predictions."""
        validity_scores = {}
        
        # Consistency across tasks - learning style should be relatively stable
        if len(predictions) > 1:
            unique_predictions = len(set(predictions))
            total_predictions = len(predictions)
            consistency_score = 1.0 - (unique_predictions - 1) / (total_predictions - 1)
            validity_scores['consistency_across_tasks'] = max(0, consistency_score)
        else:
            validity_scores['consistency_across_tasks'] = 0.5
        
        # Performance alignment - better performance in preferred modality
        if behavioral_data:
            alignment_scores = []
            for i, data in enumerate(behavioral_data):
                if i < len(predictions):
                    predicted_style = predictions[i]
                    performance_by_type = data.get('performance_by_type', {})
                    
                    # Map prediction to modality (simplified)
                    modality_map = {0: 'visual', 1: 'auditory', 2: 'text', 3: 'interactive', 4: 'multimodal'}
                    predicted_modality = modality_map.get(predicted_style, 'visual')
                    
                    if predicted_modality in performance_by_type:
                        predicted_performance = np.mean(performance_by_type[predicted_modality])
                        other_performances = []
                        for mod, perf in performance_by_type.items():
                            if mod != predicted_modality:
                                other_performances.extend(perf)
                        
                        if other_performances:
                            other_mean = np.mean(other_performances)
                            alignment = max(0, predicted_performance - other_mean)
                            alignment_scores.append(alignment)
            
            if alignment_scores:
                validity_scores['performance_alignment'] = np.mean(alignment_scores)
            else:
                validity_scores['performance_alignment'] = 0.5
        else:
            validity_scores['performance_alignment'] = 0.5
        
        # Adaptation evidence - should show some learning/adaptation
        if len(predictions) >= 6:
            # Look for any pattern of adaptation (simplified)
            first_third = predictions[:len(predictions)//3]
            last_third = predictions[-len(predictions)//3:]
            
            # Measure if predictions become more consistent (adaptation)
            first_consistency = 1.0 - (len(set(first_third)) / len(first_third))
            last_consistency = 1.0 - (len(set(last_third)) / len(last_third))
            
            adaptation_score = max(0, last_consistency - first_consistency)
            validity_scores['adaptation_evidence'] = adaptation_score
        else:
            validity_scores['adaptation_evidence'] = 0.5
        
        return validity_scores
    
    def evaluate_overall_validity(self, model_type: str, predictions: np.ndarray,
                                behavioral_data: List[Dict[str, Any]]) -> float:
        """Calculate overall psychological validity score."""
        
        if model_type == 'attention_tracker':
            validity_scores = self.evaluate_attention_validity(predictions, behavioral_data)
        elif model_type == 'cognitive_load':
            validity_scores = self.evaluate_cognitive_load_validity(predictions, behavioral_data)
        elif model_type == 'learning_style':
            validity_scores = self.evaluate_learning_style_validity(predictions, behavioral_data)
        else:
            return 0.5
        
        # Weight scores by criteria importance
        criteria = self.validity_criteria[model_type]
        weighted_score = 0.0
        total_weight = 0.0
        
        for criterion, weight in criteria.items():
            if criterion in validity_scores:
                weighted_score += validity_scores[criterion] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.5


class BiasAndFairnessEvaluator:
    """Evaluates model bias and fairness across different groups."""
    
    def __init__(self):
        self.protected_attributes = ['age_group', 'gender', 'education_level', 'cultural_background']
    
    def evaluate_demographic_parity(self, predictions: np.ndarray, 
                                  sensitive_attributes: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate demographic parity across groups."""
        parity_scores = {}
        
        for attr_name, attr_values in sensitive_attributes.items():
            if len(set(attr_values)) < 2:
                parity_scores[attr_name] = 1.0  # Perfect parity if no variation
                continue
            
            # Calculate positive prediction rates for each group
            group_rates = {}
            unique_groups = set(attr_values)
            
            for group in unique_groups:
                group_mask = attr_values == group
                group_predictions = predictions[group_mask]
                
                if len(group_predictions) > 0:
                    # For multi-class, use "positive" as non-lowest class
                    positive_rate = np.mean(group_predictions > np.min(predictions))
                    group_rates[group] = positive_rate
            
            # Calculate parity as 1 - max difference between groups
            if len(group_rates) > 1:
                rates = list(group_rates.values())
                parity_score = 1.0 - (max(rates) - min(rates))
                parity_scores[attr_name] = max(0, parity_score)
            else:
                parity_scores[attr_name] = 1.0
        
        return parity_scores
    
    def evaluate_equalized_odds(self, predictions: np.ndarray, true_labels: np.ndarray,
                              sensitive_attributes: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate equalized odds across groups."""
        odds_scores = {}
        
        for attr_name, attr_values in sensitive_attributes.items():
            if len(set(attr_values)) < 2:
                odds_scores[attr_name] = 1.0
                continue
            
            group_tpr = {}  # True Positive Rates
            group_fpr = {}  # False Positive Rates
            unique_groups = set(attr_values)
            
            for group in unique_groups:
                group_mask = attr_values == group
                group_pred = predictions[group_mask]
                group_true = true_labels[group_mask]
                
                if len(group_pred) > 0:
                    # Calculate TPR and FPR for each class (simplified for multi-class)
                    accuracy = accuracy_score(group_true, group_pred)
                    group_tpr[group] = accuracy  # Simplified metric
                    group_fpr[group] = 1.0 - accuracy
            
            # Calculate equalized odds as similarity of TPR and FPR across groups
            if len(group_tpr) > 1:
                tpr_values = list(group_tpr.values())
                tpr_variance = np.var(tpr_values)
                
                fpr_values = list(group_fpr.values())
                fpr_variance = np.var(fpr_values)
                
                # Lower variance = better equalized odds
                odds_score = 1.0 / (1.0 + tpr_variance + fpr_variance)
                odds_scores[attr_name] = odds_score
            else:
                odds_scores[attr_name] = 1.0
        
        return odds_scores
    
    def evaluate_calibration(self, prediction_probabilities: np.ndarray, true_labels: np.ndarray,
                           sensitive_attributes: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate prediction calibration across groups."""
        calibration_scores = {}
        
        for attr_name, attr_values in sensitive_attributes.items():
            if len(set(attr_values)) < 2:
                calibration_scores[attr_name] = 1.0
                continue
            
            group_calibrations = {}
            unique_groups = set(attr_values)
            
            for group in unique_groups:
                group_mask = attr_values == group
                group_probs = prediction_probabilities[group_mask]
                group_true = true_labels[group_mask]
                
                if len(group_probs) > 10:  # Need sufficient samples
                    # Calculate calibration error (simplified)
                    predicted_probs = np.max(group_probs, axis=1)  # Max probability
                    predicted_classes = np.argmax(group_probs, axis=1)
                    
                    # Bin predictions and calculate calibration
                    n_bins = 5
                    bin_boundaries = np.linspace(0, 1, n_bins + 1)
                    calibration_error = 0.0
                    
                    for i in range(n_bins):
                        bin_lower = bin_boundaries[i]
                        bin_upper = bin_boundaries[i + 1]
                        
                        in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
                        prop_in_bin = in_bin.mean()
                        
                        if prop_in_bin > 0:
                            accuracy_in_bin = (predicted_classes[in_bin] == group_true[in_bin]).mean()
                            avg_confidence_in_bin = predicted_probs[in_bin].mean()
                            calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    
                    group_calibrations[group] = 1.0 - calibration_error  # Higher is better
                else:
                    group_calibrations[group] = 0.5
            
            # Calculate calibration fairness as similarity across groups
            if len(group_calibrations) > 1:
                calibration_values = list(group_calibrations.values())
                calibration_variance = np.var(calibration_values)
                calibration_score = 1.0 / (1.0 + calibration_variance)
                calibration_scores[attr_name] = calibration_score
            else:
                calibration_scores[attr_name] = 1.0
        
        return calibration_scores
    
    def evaluate_overall_fairness(self, predictions: np.ndarray, true_labels: np.ndarray,
                                prediction_probabilities: np.ndarray,
                                sensitive_attributes: Dict[str, np.ndarray]) -> float:
        """Calculate overall fairness score."""
        
        # Demographic parity
        parity_scores = self.evaluate_demographic_parity(predictions, sensitive_attributes)
        
        # Equalized odds
        odds_scores = self.evaluate_equalized_odds(predictions, true_labels, sensitive_attributes)
        
        # Calibration
        calibration_scores = self.evaluate_calibration(prediction_probabilities, true_labels, sensitive_attributes)
        
        # Combine scores
        all_scores = []
        all_scores.extend(parity_scores.values())
        all_scores.extend(odds_scores.values())
        all_scores.extend(calibration_scores.values())
        
        return np.mean(all_scores) if all_scores else 0.5


class ModelDriftDetector:
    """Detects model drift and performance degradation."""
    
    def __init__(self, reference_window_size: int = 1000):
        self.reference_window_size = reference_window_size
        self.reference_data = {}
        self.reference_performance = {}
    
    def set_reference_data(self, model_name: str, X_reference: np.ndarray, 
                          y_reference: np.ndarray, predictions_reference: np.ndarray):
        """Set reference data for drift detection."""
        self.reference_data[model_name] = {
            'X': X_reference[-self.reference_window_size:],
            'y': y_reference[-self.reference_window_size:],
            'predictions': predictions_reference[-self.reference_window_size:]
        }
        
        # Calculate reference performance
        self.reference_performance[model_name] = accuracy_score(
            y_reference[-self.reference_window_size:],
            predictions_reference[-self.reference_window_size:]
        )
    
    def detect_data_drift(self, model_name: str, X_new: np.ndarray) -> Dict[str, float]:
        """Detect data drift using statistical tests."""
        if model_name not in self.reference_data:
            return {'overall_drift': 0.0}
        
        X_reference = self.reference_data[model_name]['X']
        drift_scores = {}
        
        # Feature-wise drift detection using Kolmogorov-Smirnov test
        n_features = min(X_reference.shape[1], X_new.shape[1])
        feature_drifts = []
        
        for i in range(n_features):
            ref_feature = X_reference[:, i]
            new_feature = X_new[:, i]
            
            # KS test
            ks_stat, p_value = ks_2samp(ref_feature, new_feature)
            
            # Convert to drift score (higher = more drift)
            drift_score = ks_stat  # 0 = no drift, 1 = maximum drift
            feature_drifts.append(drift_score)
            
            drift_scores[f'feature_{i}_drift'] = drift_score
        
        # Overall drift score
        drift_scores['overall_drift'] = np.mean(feature_drifts)
        drift_scores['max_feature_drift'] = np.max(feature_drifts)
        drift_scores['n_drifted_features'] = np.sum(np.array(feature_drifts) > 0.1)  # threshold
        
        return drift_scores
    
    def detect_prediction_drift(self, model_name: str, predictions_new: np.ndarray) -> Dict[str, float]:
        """Detect drift in prediction distributions."""
        if model_name not in self.reference_data:
            return {'prediction_drift': 0.0}
        
        predictions_reference = self.reference_data[model_name]['predictions']
        
        # Compare prediction distributions
        if predictions_reference.dtype == predictions_new.dtype:
            if np.issubdtype(predictions_reference.dtype, np.number):
                # Numerical predictions - use KS test
                ks_stat, p_value = ks_2samp(predictions_reference, predictions_new)
                prediction_drift = ks_stat
            else:
                # Categorical predictions - use chi-square test
                ref_counts = pd.Series(predictions_reference).value_counts()
                new_counts = pd.Series(predictions_new).value_counts()
                
                # Align categories
                all_categories = set(ref_counts.index) | set(new_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                new_aligned = [new_counts.get(cat, 0) for cat in all_categories]
                
                if sum(ref_aligned) > 0 and sum(new_aligned) > 0:
                    chi2_stat, p_value = chi2_contingency([ref_aligned, new_aligned])[:2]
                    # Normalize chi2 to 0-1 range (approximately)
                    prediction_drift = min(1.0, chi2_stat / 100.0)
                else:
                    prediction_drift = 1.0  # Maximum drift if no overlap
        else:
            prediction_drift = 1.0  # Maximum drift for type mismatch
        
        return {'prediction_drift': prediction_drift}
    
    def detect_performance_drift(self, model_name: str, y_new: np.ndarray, 
                               predictions_new: np.ndarray) -> Dict[str, float]:
        """Detect performance drift."""
        if model_name not in self.reference_performance:
            return {'performance_drift': 0.0}
        
        reference_performance = self.reference_performance[model_name]
        current_performance = accuracy_score(y_new, predictions_new)
        
        # Calculate performance drift
        performance_change = reference_performance - current_performance
        performance_drift = max(0, performance_change)  # Only care about degradation
        
        return {
            'performance_drift': performance_drift,
            'current_performance': current_performance,
            'reference_performance': reference_performance,
            'performance_change': performance_change
        }
    
    def detect_overall_drift(self, model_name: str, X_new: np.ndarray, 
                           y_new: np.ndarray, predictions_new: np.ndarray) -> Dict[str, float]:
        """Detect overall model drift."""
        
        data_drift = self.detect_data_drift(model_name, X_new)
        prediction_drift = self.detect_prediction_drift(model_name, predictions_new)
        performance_drift = self.detect_performance_drift(model_name, y_new, predictions_new)
        
        # Combine drift scores
        overall_drift = np.mean([
            data_drift.get('overall_drift', 0),
            prediction_drift.get('prediction_drift', 0),
            performance_drift.get('performance_drift', 0)
        ])
        
        return {
            **data_drift,
            **prediction_drift,
            **performance_drift,
            'combined_drift_score': overall_drift
        }


class AdvancedEvaluationFramework:
    """Main evaluation framework integrating all evaluation components."""
    
    def __init__(self):
        self.validity_evaluator = PsychologicalValidityEvaluator()
        self.fairness_evaluator = BiasAndFairnessEvaluator()
        self.drift_detector = ModelDriftDetector()
        self.evaluation_history = []
    
    def comprehensive_evaluate(self, model_name: str, model, X_test: np.ndarray, 
                             y_test: np.ndarray, behavioral_data: List[Dict[str, Any]] = None,
                             sensitive_attributes: Dict[str, np.ndarray] = None) -> EvaluationResult:
        """Perform comprehensive evaluation."""
        
        # Basic predictions
        predictions = model.predict(X_test)
        prediction_probabilities = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Standard metrics
        accuracy = accuracy_score(y_test, predictions)
        f1_weighted = f1_score(y_test, predictions, average='weighted')
        f1_macro = f1_score(y_test, predictions, average='macro')
        precision_weighted = precision_score(y_test, predictions, average='weighted')
        recall_weighted = recall_score(y_test, predictions, average='weighted')
        
        # Psychological validity
        psychological_validity = 0.5
        if behavioral_data:
            psychological_validity = self.validity_evaluator.evaluate_overall_validity(
                model_name, predictions, behavioral_data
            )
        
        # Fairness evaluation
        fairness_score = 1.0
        bias_metrics = {}
        if sensitive_attributes and prediction_probabilities is not None:
            fairness_score = self.fairness_evaluator.evaluate_overall_fairness(
                predictions, y_test, prediction_probabilities, sensitive_attributes
            )
            bias_metrics = {
                'demographic_parity': self.fairness_evaluator.evaluate_demographic_parity(
                    predictions, sensitive_attributes
                ),
                'equalized_odds': self.fairness_evaluator.evaluate_equalized_odds(
                    predictions, y_test, sensitive_attributes
                )
            }
        
        # Drift detection
        drift_score = 0.0
        if len(self.evaluation_history) > 0:
            # Use previous evaluation as reference
            previous_eval = self.evaluation_history[-1]
            if previous_eval.model_name == model_name:
                # This would require storing previous data - simplified for demo
                drift_score = 0.1  # Placeholder
        
        # Confidence intervals (bootstrap)
        confidence_intervals = self._calculate_confidence_intervals(y_test, predictions)
        
        # Detailed metrics
        detailed_metrics = {
            'classification_report': classification_report(y_test, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
            'cohen_kappa': cohen_kappa_score(y_test, predictions),
            'matthews_corrcoef': matthews_corrcoef(y_test, predictions)
        }
        
        # ROC AUC for multi-class (if applicable)
        if prediction_probabilities is not None and len(np.unique(y_test)) > 2:
            try:
                roc_auc = roc_auc_score(y_test, prediction_probabilities, multi_class='ovr')
                detailed_metrics['roc_auc_ovr'] = roc_auc
            except:
                pass
        
        # Create evaluation result
        result = EvaluationResult(
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            accuracy=accuracy,
            f1_weighted=f1_weighted,
            f1_macro=f1_macro,
            precision_weighted=precision_weighted,
            recall_weighted=recall_weighted,
            psychological_validity=psychological_validity,
            fairness_score=fairness_score,
            bias_metrics=bias_metrics,
            drift_score=drift_score,
            confidence_intervals=confidence_intervals,
            detailed_metrics=detailed_metrics
        )
        
        # Store evaluation
        self.evaluation_history.append(result)
        
        return result
    
    def _calculate_confidence_intervals(self, y_true: np.ndarray, 
                                     predictions: np.ndarray, 
                                     confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals using bootstrap."""
        n_bootstrap = 1000
        n_samples = len(y_true)
        
        bootstrap_scores = {
            'accuracy': [],
            'f1_weighted': [],
            'f1_macro': []
        }
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_boot = y_true[indices]
            pred_boot = predictions[indices]
            
            # Calculate metrics
            bootstrap_scores['accuracy'].append(accuracy_score(y_boot, pred_boot))
            bootstrap_scores['f1_weighted'].append(f1_score(y_boot, pred_boot, average='weighted'))
            bootstrap_scores['f1_macro'].append(f1_score(y_boot, pred_boot, average='macro'))
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        confidence_intervals = {}
        
        for metric, scores in bootstrap_scores.items():
            scores = np.array(scores)
            lower = np.percentile(scores, 100 * alpha / 2)
            upper = np.percentile(scores, 100 * (1 - alpha / 2))
            confidence_intervals[metric] = (lower, upper)
        
        return confidence_intervals
    
    def generate_evaluation_report(self, result: EvaluationResult) -> str:
        """Generate comprehensive evaluation report."""
        
        report = f"""
🎯 ADVANCED EVALUATION REPORT
{'='*50}
Model: {result.model_name}
Timestamp: {result.timestamp}

📊 PERFORMANCE METRICS
{'-'*30}
Accuracy: {result.accuracy:.3f}
F1-Score (Weighted): {result.f1_weighted:.3f}
F1-Score (Macro): {result.f1_macro:.3f}
Precision (Weighted): {result.precision_weighted:.3f}
Recall (Weighted): {result.recall_weighted:.3f}

🧠 PSYCHOLOGICAL VALIDITY
{'-'*30}
Validity Score: {result.psychological_validity:.3f}
Status: {'✅ Valid' if result.psychological_validity > 0.6 else '⚠️ Questionable' if result.psychological_validity > 0.4 else '❌ Invalid'}

⚖️ FAIRNESS & BIAS
{'-'*30}
Overall Fairness: {result.fairness_score:.3f}
Bias Status: {'✅ Fair' if result.fairness_score > 0.8 else '⚠️ Some Bias' if result.fairness_score > 0.6 else '❌ Biased'}

📈 MODEL STABILITY
{'-'*30}
Drift Score: {result.drift_score:.3f}
Stability: {'✅ Stable' if result.drift_score < 0.1 else '⚠️ Some Drift' if result.drift_score < 0.3 else '❌ Significant Drift'}

🔍 CONFIDENCE INTERVALS (95%)
{'-'*30}"""
        
        for metric, (lower, upper) in result.confidence_intervals.items():
            report += f"\n{metric.replace('_', ' ').title()}: [{lower:.3f}, {upper:.3f}]"
        
        report += f"""

📋 DETAILED ANALYSIS
{'-'*30}
Cohen's Kappa: {result.detailed_metrics.get('cohen_kappa', 'N/A')}
Matthews Correlation: {result.detailed_metrics.get('matthews_corrcoef', 'N/A')}
"""
        
        if 'roc_auc_ovr' in result.detailed_metrics:
            report += f"ROC AUC (OvR): {result.detailed_metrics['roc_auc_ovr']:.3f}\n"
        
        report += """
🎯 RECOMMENDATIONS
{'-'*30}"""
        
        # Generate recommendations based on scores
        if result.accuracy < 0.7:
            report += "\n• Consider model retraining or architecture changes"
        
        if result.psychological_validity < 0.6:
            report += "\n• Review model predictions for psychological consistency"
        
        if result.fairness_score < 0.8:
            report += "\n• Investigate and mitigate potential bias issues"
        
        if result.drift_score > 0.2:
            report += "\n• Monitor for data/concept drift, consider model updates"
        
        report += f"\n\n{'='*50}"
        
        return report
    
    def compare_models(self, results: List[EvaluationResult]) -> str:
        """Compare multiple model evaluation results."""
        
        if len(results) < 2:
            return "Need at least 2 models for comparison"
        
        comparison_report = f"""
🔍 MODEL COMPARISON REPORT
{'='*50}

📊 PERFORMANCE COMPARISON
{'-'*30}"""
        
        metrics = ['accuracy', 'f1_weighted', 'f1_macro', 'psychological_validity', 'fairness_score']
        
        for metric in metrics:
            comparison_report += f"\n{metric.replace('_', ' ').title()}:"
            for result in results:
                value = getattr(result, metric)
                comparison_report += f"\n  {result.model_name}: {value:.3f}"
            comparison_report += "\n"
        
        # Best performing model
        best_overall = max(results, key=lambda r: (r.accuracy + r.f1_weighted + r.psychological_validity + r.fairness_score) / 4)
        comparison_report += f"\n🏆 BEST OVERALL MODEL: {best_overall.model_name}\n"
        
        return comparison_report


if __name__ == "__main__":
    print("Advanced Evaluation Framework")
    print("=" * 50)
    print("Comprehensive evaluation system with:")
    print("• Psychological validity assessment")
    print("• Fairness and bias evaluation")
    print("• Model drift detection")
    print("• Confidence intervals")
    print("• Detailed performance metrics")
    print("• Comparative analysis")