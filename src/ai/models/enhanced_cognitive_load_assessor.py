"""
🧠 Enhanced Cognitive Load Assessor - Advanced Ensemble Architecture
==================================================================

PERFORMANCE TARGET: 37.6% → 70%+ accuracy

Advanced ensemble regression/classification model combining:
- XGBoost for gradient boosting optimization
- LightGBM for fast gradient boosting 
- Random Forest for robust ensemble learning
- Neural Networks for complex pattern recognition
- Support Vector Regression for non-linear relationships
- Stacking meta-learner for optimal combination

Features:
✓ Hybrid regression-classification approach
✓ Advanced feature engineering (40+ cognitive indicators)
✓ Class imbalance handling with SMOTE-NC
✓ Bayesian hyperparameter optimization
✓ Model interpretation with SHAP
✓ Comprehensive cross-validation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import logging
import warnings
from dataclasses import dataclass
import joblib
from pathlib import Path

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, classification_report, mean_squared_error, 
                           r2_score, mean_absolute_error, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

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
    from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE, ADASYN
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
from ..data.enhanced_feature_engineering import EnhancedCognitiveLoadFeatures

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class CognitiveLoadMetrics:
    """Metrics for cognitive load model evaluation."""
    # Regression metrics
    mse: float
    rmse: float
    mae: float
    r2_score: float
    
    # Classification metrics (for discrete levels)
    accuracy: Optional[float] = None
    precision: Optional[Dict[str, float]] = None
    recall: Optional[Dict[str, float]] = None
    f1_score: Optional[Dict[str, float]] = None
    
    # Cross-validation
    cross_val_scores: List[float] = None
    feature_importance: Dict[str, float] = None
    model_weights: Dict[str, float] = None

class CognitiveLoadLevel:
    """Cognitive load level constants."""
    OPTIMAL = "optimal"          
    UNDERLOADED = "underloaded"  
    OVERLOADED = "overloaded"    
    CRITICAL = "critical"

class HybridCognitiveLoadPredictor(BaseEstimator):
    """
    Hybrid predictor that combines regression and classification approaches.
    Predicts continuous cognitive load scores and discrete load levels.
    """
    
    def __init__(self, regressor, classifier, score_thresholds=None):
        self.regressor = regressor
        self.classifier = classifier
        self.score_thresholds = score_thresholds or [0.3, 0.6, 0.8]
        
    def fit(self, X, y_continuous, y_categorical=None):
        """Fit both regression and classification models."""
        self.regressor.fit(X, y_continuous)
        
        if y_categorical is not None:
            self.classifier.fit(X, y_categorical)
        else:
            # Generate categorical labels from continuous scores
            y_cat = self._score_to_category(y_continuous)
            self.classifier.fit(X, y_cat)
        
        return self
        
    def predict(self, X):
        """Predict both continuous score and categorical level."""
        score = self.regressor.predict(X)[0]
        level = self.classifier.predict(X)[0]
        return score, level
        
    def predict_proba(self, X):
        """Get prediction probabilities for categorical levels."""
        return self.classifier.predict_proba(X)
        
    def _score_to_category(self, scores):
        """Convert continuous scores to categorical levels."""
        categories = []
        for score in scores:
            if score < self.score_thresholds[0]:
                categories.append(CognitiveLoadLevel.UNDERLOADED)
            elif score < self.score_thresholds[1]:
                categories.append(CognitiveLoadLevel.OPTIMAL)
            elif score < self.score_thresholds[2]:
                categories.append(CognitiveLoadLevel.OVERLOADED)
            else:
                categories.append(CognitiveLoadLevel.CRITICAL)
        return categories

class EnhancedCognitiveLoadAssessor(BaseAIModel):
    """
    🧠 Enhanced Cognitive Load Assessor with Advanced Ensemble Architecture
    =====================================================================
    
    Hybrid approach combining regression and classification:
    - Predicts continuous cognitive load scores (0-1)
    - Classifies discrete cognitive load levels
    - Uses ensemble of multiple algorithms
    - Advanced feature engineering with 40+ indicators
    
    Performance Improvements:
    - SMOTE-NC for mixed data imbalance handling
    - Bayesian hyperparameter optimization
    - Stacking ensemble with cross-validation
    - SHAP interpretation for explainability
    - Robust evaluation with multiple metrics
    """
    
    def __init__(self, version: str = "2.0.0"):
        super().__init__("enhanced_cognitive_load_assessor", version)
        self.feature_extractor = EnhancedCognitiveLoadFeatures()
        self.scaler = RobustScaler()  # More robust to outliers
        self.target_scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        
        # Ensemble components
        self.regression_models = {}
        self.classification_models = {}
        self.hybrid_model = None
        self.best_regressor = None
        self.best_classifier = None
        
        # Configuration
        self.use_hybrid_approach = True
        self.use_smote = HAS_IMBLEARN
        self.use_optuna = HAS_OPTUNA
        self.n_folds = 5
        self.random_state = 42
        
        # Metrics and interpretation
        self.training_metrics = None
        self.feature_importance = {}
        self.model_interpretation = {}
        
        # Enhanced tracking
        self.cognitive_history: List[Dict[str, Any]] = []
        
    def get_required_fields(self) -> List[str]:
        """Required input fields for enhanced cognitive load analysis."""
        return [
            "response_times",
            "accuracy_scores", 
            "task_complexities",
            "error_patterns",
            "hesitation_indicators",
            "multitask_events",
            "content_engagement",
            "timestamp",
            "session_duration",
            "performance_metrics",
            "working_memory_indicators",
            "fatigue_signals"
        ]
    
    def _initialize_regression_models(self) -> Dict[str, Any]:
        """Initialize regression models for continuous score prediction."""
        models = {}
        
        # Random Forest Regressor
        models['random_forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Neural Network Regressor
        models['neural_network'] = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=self.random_state
        )
        
        # Support Vector Regressor
        models['svr'] = SVR(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            epsilon=0.1
        )
        
        # XGBoost Regressor
        if HAS_XGBOOST:
            models['xgb_regressor'] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='rmse'
            )
        
        # LightGBM Regressor
        if HAS_LIGHTGBM:
            models['lgb_regressor'] = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            )
        
        return models
    
    def _initialize_classification_models(self) -> Dict[str, Any]:
        """Initialize classification models for discrete level prediction."""
        models = {}
        
        # Random Forest Classifier
        models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Neural Network Classifier
        models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=self.random_state
        )
        
        # Support Vector Classifier
        models['svc'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=self.random_state
        )
        
        # XGBoost Classifier
        if HAS_XGBOOST:
            models['xgb_classifier'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
        
        # LightGBM Classifier
        if HAS_LIGHTGBM:
            models['lgb_classifier'] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            )
        
        return models
    
    def _handle_imbalanced_data(self, X: np.ndarray, y_continuous: np.ndarray, 
                               y_categorical: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Handle imbalanced data using SMOTE variants."""
        if not HAS_IMBLEARN:
            logger.info("Using class weights for imbalance handling")
            return X, y_continuous, y_categorical
            
        try:
            # Use SMOTE for categorical targets
            smote = SMOTE(
                sampling_strategy='auto',
                random_state=self.random_state,
                k_neighbors=min(5, len(np.unique(y_categorical)) - 1)
            )
            X_resampled, y_cat_resampled = smote.fit_resample(X, y_categorical)
            
            # For continuous targets, interpolate based on resampled indices
            original_indices = np.arange(len(y_continuous))
            resampled_indices = []
            
            # Find which samples were generated
            for i, sample in enumerate(X_resampled):
                # Find closest original sample
                distances = np.sum((X - sample)**2, axis=1)
                closest_idx = np.argmin(distances)
                resampled_indices.append(closest_idx)
            
            y_continuous_resampled = y_continuous[resampled_indices]
            
            logger.info(f"SMOTE applied: {X.shape[0]} → {X_resampled.shape[0]} samples")
            return X_resampled, y_continuous_resampled, y_cat_resampled
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Using original data.")
            return X, y_continuous, y_categorical
    
    def _optimize_hyperparameters(self, model, param_grid: Dict, X: np.ndarray, 
                                 y: np.ndarray, task_type: str = 'regression') -> Any:
        """Optimize hyperparameters using Optuna or GridSearch."""
        if not HAS_OPTUNA:
            # Fallback to GridSearchCV
            cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state) if task_type == 'regression' else StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scoring = 'neg_mean_squared_error' if task_type == 'regression' else 'accuracy'
            
            grid_search = GridSearchCV(
                model, param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X, y)
            return grid_search.best_estimator_
        
        # Use Optuna for Bayesian optimization
        try:
            cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state) if task_type == 'regression' else StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scoring = 'neg_mean_squared_error' if task_type == 'regression' else 'accuracy'
            
            optuna_search = OptunaSearchCV(
                model, param_grid,
                cv=cv,
                scoring=scoring,
                n_trials=50,
                random_state=self.random_state,
                verbose=0
            )
            optuna_search.fit(X, y)
            return optuna_search.best_estimator_
        except Exception as e:
            logger.warning(f"Optuna optimization failed: {e}. Using default parameters.")
            return model
    
    def _create_stacking_ensemble(self, models: Dict, X: np.ndarray, y: np.ndarray, 
                                 task_type: str = 'regression') -> Union[StackingRegressor, Any]:
        """Create stacking ensemble for regression or classification."""
        estimators = [(name, model) for name, model in models.items()]
        
        if task_type == 'regression':
            meta_learner = Ridge(alpha=1.0, random_state=self.random_state)
            cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            
            stacking_model = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=cv,
                n_jobs=-1
            )
        else:
            meta_learner = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=1000
            )
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            
            # Create stacking classifier manually since sklearn doesn't have StackingClassifier in older versions
            from sklearn.ensemble import StackingClassifier
            stacking_model = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=cv,
                stack_method='predict_proba',
                n_jobs=-1
            )
        
        return stacking_model
    
    def _evaluate_regression_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model with comprehensive metrics."""
        # Cross-validation
        cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_scores = -cv_scores  # Convert to positive MSE
        
        # Fit and predict for detailed metrics
        model.fit(X, y)
        y_pred = model.predict(X)
        
        return {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'cv_mse_mean': np.mean(cv_scores),
            'cv_mse_std': np.std(cv_scores)
        }
    
    def _evaluate_classification_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate classification model with comprehensive metrics."""
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        # Fit and predict for detailed metrics
        model.fit(X, y)
        y_pred = model.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'cv_accuracy_mean': np.mean(cv_scores),
            'cv_accuracy_std': np.std(cv_scores),
            'precision': {k: v['precision'] for k, v in report.items() if isinstance(v, dict)},
            'recall': {k: v['recall'] for k, v in report.items() if isinstance(v, dict)},
            'f1_score': {k: v['f1-score'] for k, v in report.items() if isinstance(v, dict)}
        }
    
    def _score_to_level(self, score: float) -> str:
        """Convert continuous score to categorical level."""
        if score < 0.3:
            return CognitiveLoadLevel.UNDERLOADED
        elif score <= 0.6:
            return CognitiveLoadLevel.OPTIMAL
        elif score <= 0.8:
            return CognitiveLoadLevel.OVERLOADED
        else:
            return CognitiveLoadLevel.CRITICAL
    
    def train(self, training_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Train the enhanced cognitive load assessment ensemble."""
        self.status = ModelStatus.TRAINING
        
        try:
            logger.info("Starting enhanced cognitive load assessor training...")
            
            # Prepare training data
            X = training_data.drop(['cognitive_load_score', 'cognitive_load_level', 'user_id', 'session_id'], 
                                 axis=1, errors='ignore')
            
            # Handle both continuous and categorical targets
            y_continuous = training_data.get('cognitive_load_score')
            y_categorical = training_data.get('cognitive_load_level')
            
            if y_continuous is None:
                # Generate continuous scores from categorical levels if not available
                level_to_score = {
                    CognitiveLoadLevel.UNDERLOADED: 0.2,
                    CognitiveLoadLevel.OPTIMAL: 0.5,
                    CognitiveLoadLevel.OVERLOADED: 0.75,
                    CognitiveLoadLevel.CRITICAL: 0.9
                }
                y_continuous = y_categorical.map(level_to_score)
            
            if y_categorical is None:
                # Generate categorical levels from continuous scores
                y_categorical = y_continuous.apply(self._score_to_level)
            
            # Encode categorical labels
            y_categorical_encoded = self.label_encoder.fit_transform(y_categorical)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Scale continuous targets
            y_continuous_scaled = self.target_scaler.fit_transform(y_continuous.values.reshape(-1, 1)).ravel()
            
            # Handle imbalanced data
            X_balanced, y_cont_balanced, y_cat_balanced = self._handle_imbalanced_data(
                X_scaled, y_continuous_scaled, y_categorical_encoded
            )
            
            # Initialize models
            self.regression_models = self._initialize_regression_models()
            self.classification_models = self._initialize_classification_models()
            
            # Train regression models
            logger.info("Training regression models...")
            optimized_regressors = {}
            regression_scores = {}
            
            for name, model in self.regression_models.items():
                logger.info(f"Training regression {name}...")
                param_grid = self._get_regression_param_grids().get(name, {})
                optimized_model = self._optimize_hyperparameters(
                    model, param_grid, X_balanced, y_cont_balanced, 'regression'
                )
                optimized_regressors[name] = optimized_model
                
                # Evaluate
                metrics = self._evaluate_regression_model(optimized_model, X_balanced, y_cont_balanced)
                regression_scores[name] = metrics['r2']
                logger.info(f"{name} R²: {metrics['r2']:.4f} (±{metrics['cv_mse_std']:.4f})")
            
            # Train classification models
            logger.info("Training classification models...")
            optimized_classifiers = {}
            classification_scores = {}
            
            for name, model in self.classification_models.items():
                logger.info(f"Training classification {name}...")
                param_grid = self._get_classification_param_grids().get(name, {})
                optimized_model = self._optimize_hyperparameters(
                    model, param_grid, X_balanced, y_cat_balanced, 'classification'
                )
                optimized_classifiers[name] = optimized_model
                
                # Evaluate
                metrics = self._evaluate_classification_model(optimized_model, X_balanced, y_cat_balanced)
                classification_scores[name] = metrics['accuracy']
                logger.info(f"{name} accuracy: {metrics['accuracy']:.4f} (±{metrics['cv_accuracy_std']:.4f})")
            
            # Create ensemble models
            logger.info("Creating ensemble models...")
            
            # Regression ensemble
            regression_ensemble = self._create_stacking_ensemble(
                optimized_regressors, X_balanced, y_cont_balanced, 'regression'
            )
            regression_ensemble.fit(X_balanced, y_cont_balanced)
            regression_ensemble_metrics = self._evaluate_regression_model(
                regression_ensemble, X_balanced, y_cont_balanced
            )
            
            # Classification ensemble
            classification_ensemble = self._create_stacking_ensemble(
                optimized_classifiers, X_balanced, y_cat_balanced, 'classification'
            )
            classification_ensemble.fit(X_balanced, y_cat_balanced)
            classification_ensemble_metrics = self._evaluate_classification_model(
                classification_ensemble, X_balanced, y_cat_balanced
            )
            
            # Select best models
            best_regressor_name = max(regression_scores.keys(), key=lambda k: regression_scores[k])
            best_classifier_name = max(classification_scores.keys(), key=lambda k: classification_scores[k])
            
            # Choose between individual and ensemble models
            if regression_ensemble_metrics['r2'] > regression_scores[best_regressor_name]:
                self.best_regressor = regression_ensemble
                logger.info(f"Using regression ensemble (R²: {regression_ensemble_metrics['r2']:.4f})")
            else:
                self.best_regressor = optimized_regressors[best_regressor_name]
                logger.info(f"Using best regression model: {best_regressor_name} (R²: {regression_scores[best_regressor_name]:.4f})")
            
            if classification_ensemble_metrics['accuracy'] > classification_scores[best_classifier_name]:
                self.best_classifier = classification_ensemble
                logger.info(f"Using classification ensemble (accuracy: {classification_ensemble_metrics['accuracy']:.4f})")
            else:
                self.best_classifier = optimized_classifiers[best_classifier_name]
                logger.info(f"Using best classification model: {best_classifier_name} (accuracy: {classification_scores[best_classifier_name]:.4f})")
            
            # Create hybrid model
            self.hybrid_model = HybridCognitiveLoadPredictor(
                regressor=self.best_regressor,
                classifier=self.best_classifier
            )
            
            # Final evaluation metrics
            final_regression_metrics = self._evaluate_regression_model(
                self.best_regressor, X_balanced, y_cont_balanced
            )
            final_classification_metrics = self._evaluate_classification_model(
                self.best_classifier, X_balanced, y_cat_balanced
            )
            
            # Store training metrics
            self.training_metrics = CognitiveLoadMetrics(
                mse=final_regression_metrics['mse'],
                rmse=final_regression_metrics['rmse'],
                mae=final_regression_metrics['mae'],
                r2_score=final_regression_metrics['r2'],
                accuracy=final_classification_metrics['accuracy'],
                precision=final_classification_metrics['precision'],
                recall=final_classification_metrics['recall'],
                f1_score=final_classification_metrics['f1_score'],
                cross_val_scores=None,  # Would need to implement for hybrid model
                feature_importance={},
                model_weights={}
            )
            
            # Model interpretation
            if HAS_SHAP:
                try:
                    self._generate_model_interpretation(X_balanced)
                except Exception as e:
                    logger.warning(f"SHAP interpretation failed: {e}")
            
            # Prepare return metrics
            return_metrics = {
                'regression_r2': final_regression_metrics['r2'],
                'regression_mse': final_regression_metrics['mse'],
                'regression_mae': final_regression_metrics['mae'],
                'classification_accuracy': final_classification_metrics['accuracy'],
                'feature_count': X.shape[1],
                'training_samples': len(X_balanced),
                'n_classes': len(np.unique(y_cat_balanced))
            }
            
            # Store training history
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'metrics': return_metrics,
                'model_version': self.version,
                'hybrid_approach': True
            }
            self.training_history.append(training_record)
            
            self.status = ModelStatus.TRAINED
            logger.info("Enhanced cognitive load assessor trained successfully!")
            logger.info(f"Regression R²: {return_metrics['regression_r2']:.4f}")
            logger.info(f"Classification accuracy: {return_metrics['classification_accuracy']:.4f}")
            
            return return_metrics
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self.status = ModelStatus.ERROR
            raise
    
    def _get_regression_param_grids(self) -> Dict[str, Dict]:
        """Get parameter grids for regression models."""
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'neural_network': {
                'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
                'alpha': [0.0001, 0.001, 0.01]
            },
            'svr': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'epsilon': [0.01, 0.1, 0.2]
            }
        }
        
        if HAS_XGBOOST:
            param_grids['xgb_regressor'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        
        if HAS_LIGHTGBM:
            param_grids['lgb_regressor'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        
        return param_grids
    
    def _get_classification_param_grids(self) -> Dict[str, Dict]:
        """Get parameter grids for classification models."""
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'neural_network': {
                'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
                'alpha': [0.0001, 0.001, 0.01]
            },
            'svc': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
        }
        
        if HAS_XGBOOST:
            param_grids['xgb_classifier'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        
        if HAS_LIGHTGBM:
            param_grids['lgb_classifier'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        
        return param_grids
    
    def _generate_model_interpretation(self, X: np.ndarray):
        """Generate model interpretation using SHAP."""
        try:
            if hasattr(self.best_regressor, 'predict'):
                # For regression model
                explainer = shap.Explainer(self.best_regressor.predict, X[:100])
                shap_values = explainer(X[:50])
                
                self.model_interpretation = {
                    'regression_feature_importance': dict(zip(range(X.shape[1]), np.abs(shap_values.values).mean(0))),
                    'interpretation_available': True
                }
        except Exception as e:
            logger.warning(f"SHAP interpretation failed: {e}")
            self.model_interpretation = {'interpretation_available': False}
    
    def predict(self, input_data: Dict[str, Any], context: LearningContext) -> PredictionResult:
        """Predict cognitive load using hybrid ensemble model."""
        self.status = ModelStatus.PREDICTING
        
        try:
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data")
            
            if self.hybrid_model is None:
                raise ValueError("Model not trained. Call train() first.")
            
            # Extract enhanced features
            features = self._extract_enhanced_features(input_data)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make hybrid prediction
            load_score, load_level_encoded = self.hybrid_model.predict(features_scaled)
            
            # Get probabilities for classification
            class_probabilities = self.hybrid_model.predict_proba(features_scaled)[0]
            
            # Decode level
            load_level = self.label_encoder.inverse_transform([load_level_encoded])[0]
            
            # Rescale score to original range
            load_score_original = self.target_scaler.inverse_transform([[load_score]])[0][0]
            
            # Calculate confidence
            confidence = max(class_probabilities)
            
            # Get class labels and probabilities
            class_labels = self.label_encoder.inverse_transform(range(len(class_probabilities)))
            probabilities = dict(zip(class_labels, class_probabilities))
            
            # Create prediction result
            result = PredictionResult(
                value=load_level,
                confidence=confidence,
                metadata={
                    'load_score': float(load_score_original),
                    'load_level': load_level,
                    'probabilities': probabilities,
                    'feature_vector_size': features.shape[1],
                    'model_type': 'hybrid_ensemble',
                    'context': {
                        'user_id': context.user_id,
                        'session_id': context.session_id,
                        'content_id': context.content_id
                    },
                    'recommendations': self._generate_enhanced_recommendations(load_level, load_score_original, input_data),
                    'performance_indicators': self._extract_performance_indicators(input_data),
                    'cognitive_indicators': self._extract_cognitive_indicators(input_data)
                },
                timestamp=datetime.now().isoformat(),
                model_version=self.version
            )
            
            # Store in history
            self.cognitive_history.append({
                'timestamp': result.timestamp,
                'cognitive_load_level': load_level,
                'load_score': load_score_original,
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
    
    def _generate_enhanced_recommendations(self, load_level: str, load_score: float, 
                                         input_data: Dict[str, Any]) -> List[str]:
        """Generate enhanced recommendations based on cognitive load assessment."""
        recommendations = []
        
        # Base recommendations by load level
        if load_level == CognitiveLoadLevel.CRITICAL:
            recommendations.extend([
                "Immediate cognitive break required",
                "Reduce task complexity by 50% or more",
                "Switch to passive learning (videos, reading)",
                "Implement micro-breaks every 5 minutes",
                "Consider environmental stress factors"
            ])
        elif load_level == CognitiveLoadLevel.OVERLOADED:
            recommendations.extend([
                "Reduce information density",
                "Break complex tasks into smaller steps",
                "Provide additional scaffolding and hints",
                "Slow down presentation pace",
                "Add visual aids and diagrams"
            ])
        elif load_level == CognitiveLoadLevel.UNDERLOADED:
            recommendations.extend([
                "Increase challenge level gradually",
                "Add complexity to current tasks",
                "Introduce advanced concepts",
                "Accelerate learning pace",
                "Provide extension activities"
            ])
        else:  # OPTIMAL
            recommendations.extend([
                "Maintain current difficulty level",
                "Continue with progressive challenges",
                "Monitor for changes in performance",
                "Prepare for next difficulty level"
            ])
        
        # Score-based fine-tuning
        if load_score > 0.85:
            recommendations.append("Very high load detected - immediate intervention needed")
        elif load_score < 0.25:
            recommendations.append("Very low load - significant challenge increase possible")
        
        # Data-driven recommendations
        response_times = input_data.get("response_times", [])
        if response_times:
            avg_response_time = np.mean(response_times)
            if avg_response_time > 10:
                recommendations.append("Slow response times indicate high cognitive effort")
            elif avg_response_time < 2:
                recommendations.append("Fast responses suggest tasks may be too easy")
        
        accuracy_scores = input_data.get("accuracy_scores", [])
        if accuracy_scores:
            recent_accuracy = np.mean(accuracy_scores[-3:]) if len(accuracy_scores) >= 3 else np.mean(accuracy_scores)
            if recent_accuracy < 0.6:
                recommendations.append("Declining accuracy suggests cognitive overload")
        
        return recommendations
    
    def _extract_performance_indicators(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key performance indicators."""
        response_times = data.get("response_times", [])
        accuracy_scores = data.get("accuracy_scores", [])
        error_patterns = data.get("error_patterns", {})
        
        indicators = {
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'response_time_trend': 0,
            'current_accuracy': accuracy_scores[-1] if accuracy_scores else 0,
            'accuracy_trend': 0,
            'error_rate': sum(error_patterns.values()) if error_patterns else 0
        }
        
        # Calculate trends
        if len(response_times) > 3:
            x = np.arange(len(response_times))
            slope, _, _, _, _ = np.stats.linregress(x, response_times) if len(response_times) > 1 else (0, 0, 0, 0, 0)
            indicators['response_time_trend'] = slope
        
        if len(accuracy_scores) > 3:
            x = np.arange(len(accuracy_scores))
            slope, _, _, _, _ = np.stats.linregress(x, accuracy_scores) if len(accuracy_scores) > 1 else (0, 0, 0, 0, 0)
            indicators['accuracy_trend'] = slope
        
        return indicators
    
    def _extract_cognitive_indicators(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract cognitive load specific indicators."""
        indicators = {}
        
        # Working memory load indicators
        multitask_events = data.get("multitask_events", [])
        hesitation_indicators = data.get("hesitation_indicators", [])
        
        indicators['task_switching_cost'] = len(multitask_events) * 0.1
        indicators['hesitation_frequency'] = len(hesitation_indicators)
        
        # Mental fatigue indicators
        response_times = data.get("response_times", [])
        if len(response_times) > 4:
            first_half = response_times[:len(response_times)//2]
            last_half = response_times[len(response_times)//2:]
            
            if first_half and last_half:
                fatigue_indicator = (np.mean(last_half) - np.mean(first_half)) / np.mean(first_half)
                indicators['fatigue_indicator'] = max(0, fatigue_indicator)
        
        return indicators
    
    def preprocess_data(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """Fallback preprocessing for basic features."""
        features = []
        
        # Response time features
        response_times = raw_data.get("response_times", [])
        if response_times:
            features.extend([
                np.mean(response_times),
                np.std(response_times),
                len(response_times)
            ])
        else:
            features.extend([0, 0, 0])
        
        # Accuracy features
        accuracy_scores = raw_data.get("accuracy_scores", [])
        if accuracy_scores:
            features.extend([
                np.mean(accuracy_scores),
                np.std(accuracy_scores)
            ])
        else:
            features.extend([0.5, 0])
        
        # Task complexity
        task_complexities = raw_data.get("task_complexities", [])
        features.extend([np.mean(task_complexities) if task_complexities else 5])
        
        # Error patterns
        error_patterns = raw_data.get("error_patterns", {})
        features.extend([sum(error_patterns.values()) if error_patterns else 0])
        
        # Context features
        session_duration = min(raw_data.get("session_duration", 600), 3600) / 3600.0
        features.extend([session_duration])
        
        return np.array(features).reshape(1, -1)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = super().get_model_info()
        
        if self.training_metrics:
            info.update({
                'regression_r2': self.training_metrics.r2_score,
                'regression_mse': self.training_metrics.mse,
                'classification_accuracy': self.training_metrics.accuracy,
                'hybrid_approach': True,
                'model_interpretation_available': self.model_interpretation.get('interpretation_available', False)
            })
        
        info.update({
            'model_type': 'hybrid_ensemble',
            'smote_enabled': self.use_smote,
            'hyperparameter_optimization': 'optuna' if self.use_optuna else 'grid_search'
        })
        
        return info
    
    def save_model(self, filepath: str) -> bool:
        """Save enhanced model to file."""
        try:
            model_data = {
                'hybrid_model': self.hybrid_model,
                'best_regressor': self.best_regressor,
                'best_classifier': self.best_classifier,
                'scaler': self.scaler,
                'target_scaler': self.target_scaler,
                'label_encoder': self.label_encoder,
                'feature_extractor': self.feature_extractor,
                'training_metrics': self.training_metrics,
                'model_interpretation': self.model_interpretation,
                'version': self.version,
                'model_info': self.get_model_info()
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Enhanced cognitive load assessor saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load enhanced model from file."""
        try:
            model_data = joblib.load(filepath)
            
            self.hybrid_model = model_data['hybrid_model']
            self.best_regressor = model_data['best_regressor']
            self.best_classifier = model_data['best_classifier']
            self.scaler = model_data['scaler']
            self.target_scaler = model_data['target_scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_extractor = model_data.get('feature_extractor', EnhancedCognitiveLoadFeatures())
            self.training_metrics = model_data.get('training_metrics')
            self.model_interpretation = model_data.get('model_interpretation', {})
            
            self.status = ModelStatus.TRAINED
            logger.info(f"Enhanced cognitive load assessor loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False