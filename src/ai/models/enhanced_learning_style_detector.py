"""
📚 Enhanced Learning Style Detector - Advanced Ensemble Architecture
===================================================================

PERFORMANCE TARGET: 21.3% → 65%+ accuracy

Advanced ensemble classifier combining:
- XGBoost for gradient boosted decision trees
- LightGBM for efficient gradient boosting
- Random Forest for robust ensemble learning
- Neural Networks for complex pattern recognition
- Support Vector Machines for high-dimensional classification
- Stacking meta-learner with cross-validation

Features:
✓ Multi-class classification with VARK+Multimodal styles
✓ Advanced feature engineering (35+ behavioral indicators)
✓ SMOTE variants for class imbalance handling
✓ Bayesian hyperparameter optimization
✓ Dynamic ensemble weighting
✓ Model interpretation and explainability
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import warnings
from dataclasses import dataclass
import joblib
from pathlib import Path

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import (cross_val_score, StratifiedKFold, GridSearchCV, 
                                   RandomizedSearchCV, cross_validate)
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, f1_score, precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# Advanced ML Libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not available. Using fallback ensemble.", ImportWarning)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not available. Using fallback ensemble.", ImportWarning)

try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    warnings.warn("imbalanced-learn not available. Using class weights for imbalance handling.", ImportWarning)

try:
    import optuna
    from optuna.integration import OptunaSearchCV
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    warnings.warn("Optuna not available. Using GridSearchCV for hyperparameter optimization.", ImportWarning)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not available. Limited model interpretation.", ImportWarning)

from ..core.base_model import BaseAIModel, PredictionResult, LearningContext, ModelStatus
from ..data.enhanced_feature_engineering import EnhancedLearningStyleFeatures

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class LearningStyleMetrics:
    """Metrics for learning style detection evaluation."""
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    macro_f1: float
    weighted_f1: float
    roc_auc: Optional[float]
    cross_val_scores: List[float]
    feature_importance: Dict[str, float]
    model_weights: Dict[str, float]
    confusion_matrix: np.ndarray

class LearningStyle:
    """Learning style constants based on VARK+ model."""
    VISUAL = "visual"           
    AUDITORY = "auditory"       
    READING = "reading"         
    KINESTHETIC = "kinesthetic" 
    MULTIMODAL = "multimodal"
    
    @classmethod
    def get_all_styles(cls) -> List[str]:
        return [cls.VISUAL, cls.AUDITORY, cls.READING, cls.KINESTHETIC, cls.MULTIMODAL]

class DynamicEnsembleWeighter:
    """
    Dynamic ensemble weighting based on model performance and confidence.
    Adjusts weights based on prediction confidence and historical accuracy.
    """
    
    def __init__(self, models: Dict[str, Any], initial_weights: Optional[Dict[str, float]] = None):
        self.models = models
        self.weights = initial_weights or {name: 1.0/len(models) for name in models.keys()}
        self.performance_history = {name: [] for name in models.keys()}
        self.confidence_history = {name: [] for name in models.keys()}
        
    def update_weights(self, predictions: Dict[str, np.ndarray], 
                      confidences: Dict[str, float], actual: Optional[np.ndarray] = None):
        """Update weights based on performance and confidence."""
        if actual is not None:
            # Update performance history
            for name, pred in predictions.items():
                accuracy = accuracy_score(actual, pred)
                self.performance_history[name].append(accuracy)
                
                # Keep only recent history (last 100 predictions)
                if len(self.performance_history[name]) > 100:
                    self.performance_history[name] = self.performance_history[name][-100:]
        
        # Update confidence history
        for name, conf in confidences.items():
            self.confidence_history[name].append(conf)
            if len(self.confidence_history[name]) > 100:
                self.confidence_history[name] = self.confidence_history[name][-100:]
        
        # Calculate new weights
        self._recalculate_weights()
    
    def _recalculate_weights(self):
        """Recalculate weights based on performance and confidence."""
        new_weights = {}
        
        for name in self.models.keys():
            # Performance component
            perf_score = np.mean(self.performance_history[name]) if self.performance_history[name] else 0.5
            
            # Confidence component
            conf_score = np.mean(self.confidence_history[name]) if self.confidence_history[name] else 0.5
            
            # Combined score with slight bias towards performance
            combined_score = 0.7 * perf_score + 0.3 * conf_score
            new_weights[name] = combined_score
        
        # Normalize weights
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            self.weights = {name: weight/total_weight for name, weight in new_weights.items()}
    
    def get_weighted_prediction(self, predictions: Dict[str, np.ndarray], 
                               probabilities: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Get weighted ensemble prediction."""
        weighted_probs = np.zeros_like(list(probabilities.values())[0])
        
        for name, probs in probabilities.items():
            weight = self.weights.get(name, 0)
            weighted_probs += weight * probs
        
        # Get final prediction
        final_prediction = np.argmax(weighted_probs, axis=1)
        
        return final_prediction, weighted_probs

class EnhancedLearningStyleDetector(BaseAIModel):
    """
    📚 Enhanced Learning Style Detector with Advanced Ensemble Architecture
    ======================================================================
    
    Multi-class classifier combining multiple algorithms for superior learning style detection:
    - XGBoost: Gradient boosting with regularization
    - LightGBM: Fast gradient boosting with categorical support
    - Random Forest: Robust ensemble decision trees
    - Neural Network: Deep pattern recognition
    - SVM: High-dimensional pattern separation
    - Naive Bayes: Probabilistic classification
    - KNN: Instance-based learning
    - Stacking Meta-learner: Optimal model combination
    
    Advanced Features:
    - Dynamic ensemble weighting
    - SMOTE variants for class imbalance
    - Bayesian hyperparameter optimization
    - Probability calibration
    - Model interpretation with SHAP
    - Multi-modal style detection
    """
    
    def __init__(self, version: str = "2.0.0"):
        super().__init__("enhanced_learning_style_detector", version)
        self.feature_extractor = EnhancedLearningStyleFeatures()
        self.scaler = RobustScaler()  # More robust to outliers
        self.label_encoder = LabelEncoder()
        
        # Ensemble components
        self.base_models = {}
        self.calibrated_models = {}
        self.stacking_model = None
        self.voting_model = None
        self.dynamic_ensemble = None
        self.best_model = None
        
        # Configuration
        self.use_dynamic_weighting = True
        self.use_probability_calibration = True
        self.use_smote = HAS_IMBLEARN
        self.use_optuna = HAS_OPTUNA
        self.n_folds = 5
        self.random_state = 42
        
        # Metrics and interpretation
        self.training_metrics = None
        self.feature_importance = {}
        self.model_interpretation = {}
        
        # Enhanced tracking
        self.style_history: List[Dict[str, Any]] = []
        
    def get_required_fields(self) -> List[str]:
        """Required input fields for enhanced learning style analysis."""
        return [
            "content_interactions",
            "time_spent_by_type",
            "performance_by_type",
            "content_preferences", 
            "engagement_metrics",
            "completion_rates",
            "replay_behaviors",
            "navigation_patterns",
            "timestamp",
            "learning_session_data",
            "content_format_preferences",
            "interaction_depth_metrics",
            "cognitive_style_indicators"
        ]
    
    def _initialize_base_models(self) -> Dict[str, Any]:
        """Initialize comprehensive set of base models."""
        models = {}
        
        # Random Forest - Robust baseline with feature importance
        models['random_forest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Neural Network - Complex pattern recognition
        models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=self.random_state
        )
        
        # Support Vector Machine - High-dimensional separation
        models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=self.random_state
        )
        
        # Naive Bayes - Probabilistic baseline
        models['naive_bayes'] = GaussianNB()
        
        # K-Nearest Neighbors - Instance-based learning
        models['knn'] = KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='minkowski'
        )
        
        # XGBoost - Advanced gradient boosting
        if HAS_XGBOOST:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
        
        # LightGBM - Fast gradient boosting
        if HAS_LIGHTGBM:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            )
        
        return models
    
    def _handle_class_imbalance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Handle class imbalance using advanced SMOTE variants."""
        if not HAS_IMBLEARN:
            logger.info("Using class weights for imbalance handling")
            return X, y
            
        try:
            # Check class distribution
            unique_classes, class_counts = np.unique(y, return_counts=True)
            logger.info(f"Original class distribution: {dict(zip(unique_classes, class_counts))}")
            
            # Use BorderlineSMOTE for better boundary handling
            smote = BorderlineSMOTE(
                sampling_strategy='auto',
                random_state=self.random_state,
                k_neighbors=min(5, len(unique_classes) - 1),
                m_neighbors=min(10, len(unique_classes) - 1)
            )
            
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Check new distribution
            unique_classes_new, class_counts_new = np.unique(y_resampled, return_counts=True)
            logger.info(f"Resampled class distribution: {dict(zip(unique_classes_new, class_counts_new))}")
            logger.info(f"BorderlineSMOTE applied: {X.shape[0]} → {X_resampled.shape[0]} samples")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.warning(f"BorderlineSMOTE failed: {e}. Trying regular SMOTE...")
            
            try:
                smote = SMOTE(
                    sampling_strategy='auto',
                    random_state=self.random_state,
                    k_neighbors=min(5, len(unique_classes) - 1)
                )
                X_resampled, y_resampled = smote.fit_resample(X, y)
                logger.info(f"Regular SMOTE applied: {X.shape[0]} → {X_resampled.shape[0]} samples")
                return X_resampled, y_resampled
            except Exception as e2:
                logger.warning(f"Regular SMOTE also failed: {e2}. Using original data.")
                return X, y
    
    def _optimize_hyperparameters(self, model, param_grid: Dict, X: np.ndarray, y: np.ndarray) -> Any:
        """Optimize hyperparameters using Optuna or RandomizedSearch."""
        if not param_grid:
            return model
            
        if not HAS_OPTUNA:
            # Use RandomizedSearchCV for efficiency
            random_search = RandomizedSearchCV(
                model, param_grid,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
                scoring='f1_macro',
                n_iter=30,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0
            )
            random_search.fit(X, y)
            return random_search.best_estimator_
        
        # Use Optuna for Bayesian optimization
        try:
            optuna_search = OptunaSearchCV(
                model, param_grid,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
                scoring='f1_macro',
                n_trials=50,
                random_state=self.random_state,
                verbose=0
            )
            optuna_search.fit(X, y)
            return optuna_search.best_estimator_
        except Exception as e:
            logger.warning(f"Optuna optimization failed: {e}. Using default parameters.")
            return model
    
    def _create_stacking_ensemble(self, base_models: Dict, X: np.ndarray, y: np.ndarray) -> StackingClassifier:
        """Create stacking ensemble with cross-validation."""
        estimators = [(name, model) for name, model in base_models.items()]
        
        # Meta-learner with regularization
        meta_learner = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000,
            multi_class='ovr'
        )
        
        # Create stacking classifier
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state),
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        return stacking_clf
    
    def _create_voting_ensemble(self, base_models: Dict) -> VotingClassifier:
        """Create voting ensemble with soft voting."""
        estimators = [(name, model) for name, model in base_models.items()]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        return voting_clf
    
    def _calibrate_probabilities(self, models: Dict, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calibrate model probabilities using Platt scaling."""
        calibrated_models = {}
        
        for name, model in models.items():
            try:
                # Use CalibratedClassifierCV for probability calibration
                calibrated_model = CalibratedClassifierCV(
                    model,
                    method='sigmoid',  # Platt scaling
                    cv=3
                )
                calibrated_model.fit(X, y)
                calibrated_models[name] = calibrated_model
                logger.info(f"Calibrated probabilities for {name}")
            except Exception as e:
                logger.warning(f"Calibration failed for {name}: {e}. Using original model.")
                calibrated_models[name] = model
        
        return calibrated_models
    
    def _evaluate_comprehensive(self, model, X: np.ndarray, y: np.ndarray) -> LearningStyleMetrics:
        """Comprehensive model evaluation with multiple metrics."""
        # Cross-validation with multiple metrics
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        scoring = ['accuracy', 'f1_macro', 'f1_weighted']
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        # Fit model for detailed metrics
        model.fit(X, y)
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average=None, zero_division=0)
        macro_f1 = f1_score(y, y_pred, average='macro')
        weighted_f1 = f1_score(y, y_pred, average='weighted')
        
        # Classification report for detailed metrics
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        
        # ROC AUC for multiclass
        try:
            roc_auc = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            roc_auc = None
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Feature importance
        feature_importance = {}
        try:
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(enumerate(model.feature_importances_))
            elif hasattr(model, 'estimators_'):
                # For ensemble models
                importances = []
                for estimator in model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances.append(estimator.feature_importances_)
                if importances:
                    avg_importance = np.mean(importances, axis=0)
                    feature_importance = dict(enumerate(avg_importance))
        except:
            pass
        
        # Model weights (for ensemble)
        model_weights = {}
        if hasattr(model, 'estimators_'):
            model_weights = {f"model_{i}": 1.0/len(model.estimators_) 
                           for i in range(len(model.estimators_))}
        
        # Create class-wise metrics dictionaries
        class_names = [str(i) for i in range(len(precision))]
        precision_dict = dict(zip(class_names, precision))
        recall_dict = dict(zip(class_names, recall))
        f1_dict = dict(zip(class_names, f1))
        
        return LearningStyleMetrics(
            accuracy=accuracy,
            precision=precision_dict,
            recall=recall_dict,
            f1_score=f1_dict,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            roc_auc=roc_auc,
            cross_val_scores=cv_results['test_accuracy'].tolist(),
            feature_importance=feature_importance,
            model_weights=model_weights,
            confusion_matrix=cm
        )
    
    def train(self, training_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Train the enhanced learning style detection ensemble."""
        self.status = ModelStatus.TRAINING
        
        try:
            logger.info("Starting enhanced learning style detector training...")
            
            # Prepare training data
            X = training_data.drop(['learning_style', 'user_id', 'session_id'], axis=1, errors='ignore')
            y = training_data['learning_style']
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Handle class imbalance
            X_balanced, y_balanced = self._handle_class_imbalance(X_scaled, y_encoded)
            
            # Initialize base models
            self.base_models = self._initialize_base_models()
            
            # Train and optimize individual models
            logger.info("Training and optimizing base models...")
            optimized_models = {}
            individual_scores = {}
            
            for name, model in self.base_models.items():
                logger.info(f"Training {name}...")
                
                # Get parameter grid for optimization
                param_grids = self._get_param_grids()
                param_grid = param_grids.get(name, {})
                
                # Optimize hyperparameters
                optimized_model = self._optimize_hyperparameters(model, param_grid, X_balanced, y_balanced)
                optimized_models[name] = optimized_model
                
                # Evaluate individual model
                metrics = self._evaluate_comprehensive(optimized_model, X_balanced, y_balanced)
                individual_scores[name] = metrics.accuracy
                logger.info(f"{name} accuracy: {metrics.accuracy:.4f} (F1-macro: {metrics.macro_f1:.4f})")
            
            # Probability calibration
            if self.use_probability_calibration:
                logger.info("Calibrating model probabilities...")
                self.calibrated_models = self._calibrate_probabilities(optimized_models, X_balanced, y_balanced)
            else:
                self.calibrated_models = optimized_models
            
            # Create ensemble models
            logger.info("Creating ensemble models...")
            
            # Stacking ensemble
            self.stacking_model = self._create_stacking_ensemble(self.calibrated_models, X_balanced, y_balanced)
            self.stacking_model.fit(X_balanced, y_balanced)
            stacking_metrics = self._evaluate_comprehensive(self.stacking_model, X_balanced, y_balanced)
            
            # Voting ensemble
            self.voting_model = self._create_voting_ensemble(self.calibrated_models)
            self.voting_model.fit(X_balanced, y_balanced)
            voting_metrics = self._evaluate_comprehensive(self.voting_model, X_balanced, y_balanced)
            
            # Dynamic ensemble
            if self.use_dynamic_weighting:
                logger.info("Creating dynamic weighted ensemble...")
                self.dynamic_ensemble = DynamicEnsembleWeighter(self.calibrated_models)
            
            # Select best model
            all_scores = {
                **individual_scores,
                'stacking': stacking_metrics.accuracy,
                'voting': voting_metrics.accuracy
            }
            
            best_model_name = max(all_scores.keys(), key=lambda k: all_scores[k])
            best_score = all_scores[best_model_name]
            
            if best_model_name == 'stacking':
                self.best_model = self.stacking_model
                self.training_metrics = stacking_metrics
            elif best_model_name == 'voting':
                self.best_model = self.voting_model
                self.training_metrics = voting_metrics
            else:
                self.best_model = self.calibrated_models[best_model_name]
                self.training_metrics = self._evaluate_comprehensive(self.best_model, X_balanced, y_balanced)
            
            logger.info(f"Best model: {best_model_name} (accuracy: {best_score:.4f})")
            
            # Model interpretation
            if HAS_SHAP:
                try:
                    self._generate_model_interpretation(X_balanced)
                except Exception as e:
                    logger.warning(f"SHAP interpretation failed: {e}")
            
            # Prepare metrics for return
            final_metrics = {
                'train_accuracy': self.training_metrics.accuracy,
                'macro_f1': self.training_metrics.macro_f1,
                'weighted_f1': self.training_metrics.weighted_f1,
                'cross_val_mean': np.mean(self.training_metrics.cross_val_scores),
                'cross_val_std': np.std(self.training_metrics.cross_val_scores),
                'feature_count': X.shape[1],
                'training_samples': len(X_balanced),
                'n_classes': len(np.unique(y_balanced))
            }
            
            if self.training_metrics.roc_auc:
                final_metrics['roc_auc'] = self.training_metrics.roc_auc
            
            # Store training history
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'metrics': final_metrics,
                'model_version': self.version,
                'best_model': best_model_name,
                'ensemble_used': best_model_name in ['stacking', 'voting']
            }
            self.training_history.append(training_record)
            
            self.status = ModelStatus.TRAINED
            logger.info("Enhanced learning style detector trained successfully!")
            logger.info(f"Final accuracy: {final_metrics['train_accuracy']:.4f}")
            logger.info(f"Macro F1: {final_metrics['macro_f1']:.4f}")
            logger.info(f"Cross-validation: {final_metrics['cross_val_mean']:.4f} ± {final_metrics['cross_val_std']:.4f}")
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self.status = ModelStatus.ERROR
            raise
    
    def _get_param_grids(self) -> Dict[str, Dict]:
        """Get parameter grids for hyperparameter optimization."""
        param_grids = {
            'random_forest': {
                'n_estimators': [200, 300, 400],
                'max_depth': [15, 20, 25, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'neural_network': {
                'hidden_layer_sizes': [(128, 64), (256, 128), (256, 128, 64)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            },
            'svm': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        }
        
        if HAS_XGBOOST:
            param_grids['xgboost'] = {
                'n_estimators': [200, 300, 400],
                'max_depth': [8, 10, 12],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        if HAS_LIGHTGBM:
            param_grids['lightgbm'] = {
                'n_estimators': [200, 300, 400],
                'max_depth': [8, 10, 12],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        return param_grids
    
    def _generate_model_interpretation(self, X: np.ndarray):
        """Generate model interpretation using SHAP."""
        try:
            if hasattr(self.best_model, 'predict_proba'):
                explainer = shap.Explainer(self.best_model.predict_proba, X[:100])  # Sample for speed
                shap_values = explainer(X[:50])  # Smaller sample for memory
                
                self.model_interpretation = {
                    'feature_importance': dict(zip(range(X.shape[1]), np.abs(shap_values.values).mean(0))),
                    'interpretation_available': True
                }
        except Exception as e:
            logger.warning(f"SHAP interpretation failed: {e}")
            self.model_interpretation = {'interpretation_available': False}
    
    def predict(self, input_data: Dict[str, Any], context: LearningContext) -> PredictionResult:
        """Predict learning style using enhanced ensemble model."""
        self.status = ModelStatus.PREDICTING
        
        try:
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data")
            
            if self.best_model is None:
                raise ValueError("Model not trained. Call train() first.")
            
            # Extract enhanced features
            features = self._extract_enhanced_features(input_data)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction with best model
            prediction_proba = self.best_model.predict_proba(features_scaled)[0]
            predicted_class_encoded = self.best_model.predict(features_scaled)[0]
            
            # Decode prediction
            predicted_style = self.label_encoder.inverse_transform([predicted_class_encoded])[0]
            confidence = max(prediction_proba)
            
            # Get class probabilities
            class_labels = self.label_encoder.inverse_transform(range(len(prediction_proba)))
            probabilities = dict(zip(class_labels, prediction_proba))
            
            # Dynamic ensemble prediction if available and better
            if self.dynamic_ensemble and len(self.calibrated_models) > 1:
                try:
                    # Get predictions from all models
                    model_predictions = {}
                    model_probabilities = {}
                    model_confidences = {}
                    
                    for name, model in self.calibrated_models.items():
                        pred = model.predict(features_scaled)
                        proba = model.predict_proba(features_scaled)
                        model_predictions[name] = pred
                        model_probabilities[name] = proba
                        model_confidences[name] = max(proba[0])
                    
                    # Get weighted ensemble prediction
                    ensemble_pred, ensemble_proba = self.dynamic_ensemble.get_weighted_prediction(
                        model_predictions, model_probabilities
                    )
                    
                    # Use ensemble if confidence is higher
                    ensemble_confidence = max(ensemble_proba[0])
                    if ensemble_confidence > confidence:
                        predicted_class_encoded = ensemble_pred[0]
                        predicted_style = self.label_encoder.inverse_transform([predicted_class_encoded])[0]
                        confidence = ensemble_confidence
                        probabilities = dict(zip(class_labels, ensemble_proba[0]))
                        
                except Exception as e:
                    logger.warning(f"Dynamic ensemble prediction failed: {e}")
            
            # Check for multimodal tendencies
            adjusted_style = self._check_multimodal_tendency(probabilities, predicted_style)
            
            # Create prediction result
            result = PredictionResult(
                value=adjusted_style,
                confidence=confidence,
                metadata={
                    'style_probabilities': probabilities,
                    'original_prediction': predicted_style,
                    'multimodal_detected': adjusted_style != predicted_style,
                    'feature_vector_size': features.shape[1],
                    'model_type': 'enhanced_ensemble',
                    'context': {
                        'user_id': context.user_id,
                        'session_id': context.session_id,
                        'content_id': context.content_id
                    },
                    'recommendations': self._generate_enhanced_recommendations(adjusted_style, probabilities, input_data),
                    'content_preferences': self._extract_detailed_preferences(input_data),
                    'learning_indicators': self._extract_learning_indicators(input_data)
                },
                timestamp=datetime.now().isoformat(),
                model_version=self.version
            )
            
            # Store in history
            self.style_history.append({
                'timestamp': result.timestamp,
                'learning_style': adjusted_style,
                'original_prediction': predicted_style,
                'confidence': confidence,
                'probabilities': probabilities,
                'user_id': context.user_id,
                'session_id': context.session_id
            })
            
            self.status = ModelStatus.TRAINED
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            self.status = ModelStatus.ERROR
            raise
    
    def _extract_enhanced_features(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Extract enhanced features using feature engineering pipeline."""
        if hasattr(self.feature_extractor, 'extract_comprehensive_features'):
            return self.feature_extractor.extract_comprehensive_features(input_data)
        
        # Fallback to basic feature extraction
        return self.preprocess_data(input_data)
    
    def _check_multimodal_tendency(self, probabilities: Dict[str, float], predicted_style: str) -> str:
        """Check for multimodal learning tendencies based on probability distribution."""
        # Count styles with significant probabilities (>15%)
        significant_styles = [style for style, prob in probabilities.items() if prob > 0.15]
        
        # If multiple styles are significant and no single dominant style (>60%)
        max_prob = max(probabilities.values())
        if len(significant_styles) >= 3 and max_prob < 0.6:
            return LearningStyle.MULTIMODAL
        
        # If two styles are very close in probability (within 10%)
        sorted_probs = sorted(probabilities.values(), reverse=True)
        if len(sorted_probs) >= 2 and (sorted_probs[0] - sorted_probs[1]) < 0.1:
            return LearningStyle.MULTIMODAL
        
        return predicted_style
    
    def _generate_enhanced_recommendations(self, learning_style: str, probabilities: Dict[str, float],
                                         input_data: Dict[str, Any]) -> List[str]:
        """Generate enhanced recommendations based on learning style and behavior analysis."""
        recommendations = []
        
        # Base recommendations by learning style
        style_recommendations = {
            LearningStyle.VISUAL: [
                "Use rich visual content: charts, diagrams, infographics",
                "Implement color-coded information and visual hierarchies",
                "Provide mind maps and flowcharts for complex concepts",
                "Use video content with visual demonstrations",
                "Include interactive visual simulations"
            ],
            LearningStyle.AUDITORY: [
                "Include audio explanations and narrated content",
                "Provide discussion forums and verbal feedback",
                "Use music and sound effects for engagement",
                "Offer podcast-style learning content",
                "Implement voice-based interactions and commands"
            ],
            LearningStyle.READING: [
                "Provide comprehensive text-based explanations",
                "Include detailed reading materials and articles",
                "Use well-structured bullet points and lists",
                "Offer downloadable PDFs and written summaries",
                "Implement note-taking and annotation features"
            ],
            LearningStyle.KINESTHETIC: [
                "Include hands-on interactive simulations",
                "Provide practical exercises and real-world applications",
                "Use drag-and-drop activities and manipulables",
                "Implement gamified learning experiences",
                "Offer physical or virtual lab experiments"
            ],
            LearningStyle.MULTIMODAL: [
                "Combine multiple content formats in single lessons",
                "Provide learner choice in content delivery methods",
                "Create rich multimedia learning experiences",
                "Allow easy switching between different formats",
                "Personalize content mix based on performance"
            ]
        }
        
        recommendations.extend(style_recommendations.get(learning_style, []))
        
        # Probability-based fine-tuning
        sorted_styles = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # If second-highest style is significant, include some recommendations
        if len(sorted_styles) >= 2 and sorted_styles[1][1] > 0.25:
            secondary_style = sorted_styles[1][0]
            secondary_recs = style_recommendations.get(secondary_style, [])
            if secondary_recs:
                recommendations.append(f"Secondary tendency ({secondary_style}): " + secondary_recs[0])
        
        # Confidence-based recommendations
        max_confidence = max(probabilities.values())
        if max_confidence < 0.5:
            recommendations.append("Low confidence prediction - gather more behavioral data")
        elif max_confidence > 0.8:
            recommendations.append("High confidence prediction - implement recommendations immediately")
        
        # Data-driven recommendations
        time_spent = input_data.get("time_spent_by_type", {})
        if time_spent:
            dominant_time_type = max(time_spent, key=time_spent.get)
            if dominant_time_type and time_spent[dominant_time_type] > 0.4:
                recommendations.append(f"High engagement with {dominant_time_type} content detected")
        
        performance_by_type = input_data.get("performance_by_type", {})
        if performance_by_type:
            best_performance_type = max(performance_by_type, key=lambda k: np.mean(performance_by_type[k]))
            recommendations.append(f"Best performance observed with {best_performance_type} content")
        
        return recommendations
    
    def _extract_detailed_preferences(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed content preferences from behavioral data."""
        preferences = {}
        
        # Time allocation preferences
        time_spent = data.get("time_spent_by_type", {})
        if time_spent:
            total_time = sum(time_spent.values())
            preferences['time_allocation'] = {
                content_type: time / total_time if total_time > 0 else 0
                for content_type, time in time_spent.items()
            }
        
        # Performance preferences
        performance = data.get("performance_by_type", {})
        if performance:
            preferences['performance_by_type'] = {
                content_type: np.mean(scores) if scores else 0
                for content_type, scores in performance.items()
            }
        
        # Engagement preferences
        engagement = data.get("engagement_metrics", {})
        if engagement:
            preferences['engagement_scores'] = engagement
        
        # Completion preferences
        completion_rates = data.get("completion_rates", {})
        if completion_rates:
            preferences['completion_rates'] = completion_rates
        
        # Replay behavior analysis
        replay_behaviors = data.get("replay_behaviors", {})
        if replay_behaviors:
            preferences['replay_preferences'] = replay_behaviors
        
        return preferences
    
    def _extract_learning_indicators(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract learning behavior indicators."""
        indicators = {}
        
        # Engagement consistency
        engagement = data.get("engagement_metrics", {})
        if engagement:
            engagement_values = list(engagement.values())
            indicators['engagement_consistency'] = 1 / (np.std(engagement_values) + 0.001) if engagement_values else 0
        
        # Content format diversity
        time_spent = data.get("time_spent_by_type", {})
        if time_spent:
            non_zero_types = sum(1 for time in time_spent.values() if time > 0)
            indicators['content_diversity'] = non_zero_types / len(time_spent) if time_spent else 0
        
        # Learning persistence
        completion_rates = data.get("completion_rates", {})
        if completion_rates:
            indicators['learning_persistence'] = np.mean(list(completion_rates.values()))
        
        # Interactive tendency
        interactions = data.get("content_interactions", {})
        if interactions:
            total_interactions = sum(interactions.values()) if isinstance(interactions, dict) else len(interactions)
            session_duration = data.get("session_duration", 1)
            indicators['interaction_rate'] = total_interactions / (session_duration / 60)  # per minute
        
        return indicators
    
    def preprocess_data(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """Fallback preprocessing for basic features."""
        features = []
        
        # Content interaction ratios
        content_interactions = raw_data.get("content_interactions", {})
        total_interactions = sum(content_interactions.values()) if content_interactions else 1
        
        for content_type in ['visual', 'auditory', 'text', 'interactive']:
            ratio = content_interactions.get(content_type, 0) / total_interactions
            features.append(ratio)
        
        # Time allocation ratios
        time_spent = raw_data.get("time_spent_by_type", {})
        total_time = sum(time_spent.values()) if time_spent else 1
        
        for content_type in ['visual', 'auditory', 'text', 'interactive']:
            ratio = time_spent.get(content_type, 0) / total_time
            features.append(ratio)
        
        # Performance scores
        performance = raw_data.get("performance_by_type", {})
        for content_type in ['visual', 'auditory', 'text', 'interactive']:
            scores = performance.get(content_type, [0.5])
            avg_performance = np.mean(scores) if scores else 0.5
            features.append(avg_performance)
        
        # Engagement scores
        engagement = raw_data.get("engagement_metrics", {})
        for content_type in ['visual', 'auditory', 'text', 'interactive']:
            engagement_score = engagement.get(content_type, 0.5)
            features.append(engagement_score)
        
        # Additional behavioral features
        completion_rates = raw_data.get("completion_rates", {})
        completion_variance = np.var(list(completion_rates.values())) if completion_rates else 0
        features.append(completion_variance)
        
        replay_behaviors = raw_data.get("replay_behaviors", {})
        replay_concentration = max(replay_behaviors.values()) / max(1, sum(replay_behaviors.values())) if replay_behaviors else 0
        features.append(replay_concentration)
        
        return np.array(features).reshape(1, -1)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = super().get_model_info()
        
        if self.training_metrics:
            info.update({
                'ensemble_accuracy': self.training_metrics.accuracy,
                'macro_f1': self.training_metrics.macro_f1,
                'weighted_f1': self.training_metrics.weighted_f1,
                'cross_val_scores': self.training_metrics.cross_val_scores,
                'feature_importance_available': bool(self.training_metrics.feature_importance),
                'model_interpretation_available': self.model_interpretation.get('interpretation_available', False),
                'roc_auc': self.training_metrics.roc_auc
            })
        
        info.update({
            'ensemble_components': list(self.base_models.keys()) if self.base_models else [],
            'dynamic_weighting_enabled': self.use_dynamic_weighting,
            'probability_calibration_enabled': self.use_probability_calibration,
            'smote_enabled': self.use_smote,
            'hyperparameter_optimization': 'optuna' if self.use_optuna else 'randomized_search',
            'supported_styles': LearningStyle.get_all_styles()
        })
        
        return info
    
    def save_model(self, filepath: str) -> bool:
        """Save enhanced model to file."""
        try:
            model_data = {
                'best_model': self.best_model,
                'base_models': self.base_models,
                'calibrated_models': self.calibrated_models,
                'stacking_model': self.stacking_model,
                'voting_model': self.voting_model,
                'dynamic_ensemble': self.dynamic_ensemble,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_extractor': self.feature_extractor,
                'training_metrics': self.training_metrics,
                'model_interpretation': self.model_interpretation,
                'version': self.version,
                'model_info': self.get_model_info()
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Enhanced learning style detector saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load enhanced model from file."""
        try:
            model_data = joblib.load(filepath)
            
            self.best_model = model_data['best_model']
            self.base_models = model_data.get('base_models', {})
            self.calibrated_models = model_data.get('calibrated_models', {})
            self.stacking_model = model_data.get('stacking_model')
            self.voting_model = model_data.get('voting_model')
            self.dynamic_ensemble = model_data.get('dynamic_ensemble')
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_extractor = model_data.get('feature_extractor', EnhancedLearningStyleFeatures())
            self.training_metrics = model_data.get('training_metrics')
            self.model_interpretation = model_data.get('model_interpretation', {})
            
            self.status = ModelStatus.TRAINED
            logger.info(f"Enhanced learning style detector loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False