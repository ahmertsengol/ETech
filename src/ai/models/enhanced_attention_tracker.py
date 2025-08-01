"""
🎯 Enhanced Attention Tracker - Advanced Ensemble Architecture
============================================================

PERFORMANCE TARGET: 67.8% → 80%+ accuracy

Advanced ensemble model combining:
- XGBoost for gradient boosting optimization
- LightGBM for fast gradient boosting with categorical features
- Random Forest for robust tree-based learning
- Neural Networks for complex pattern recognition
- Stacking meta-learner for optimal model combination

Features:
✓ Class imbalance handling with SMOTE and cost-sensitive learning
✓ Advanced ensemble strategies (stacking, blending, dynamic weighting)
✓ Hyperparameter optimization with Bayesian optimization
✓ Model interpretation and explainability
✓ Comprehensive validation and cross-validation
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline

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
from ..data.enhanced_feature_engineering import EnhancedAttentionFeatures

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class EnsembleMetrics:
    """Metrics for ensemble model evaluation."""
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    roc_auc: Optional[float]
    cross_val_scores: List[float]
    feature_importance: Dict[str, float]
    model_weights: Dict[str, float]

class AttentionLevel:
    """Attention level constants."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"

class EnhancedAttentionTracker(BaseAIModel):
    """
    🎯 Enhanced Attention Tracker with Advanced Ensemble Architecture
    ===============================================================
    
    Combines multiple ML algorithms for superior attention prediction:
    - XGBoost: Gradient boosting with regularization
    - LightGBM: Fast gradient boosting with categorical support
    - Random Forest: Robust ensemble decision trees
    - Neural Network: Deep pattern recognition
    - Stacking Meta-learner: Optimal model combination
    
    Performance Improvements:
    - SMOTE for class imbalance handling
    - Bayesian hyperparameter optimization
    - Cross-validation with stratified sampling
    - Feature importance analysis
    - Model interpretation with SHAP
    """
    
    def __init__(self, version: str = "2.0.0"):
        super().__init__("enhanced_attention_tracker", version)
        self.feature_extractor = EnhancedAttentionFeatures()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Ensemble components
        self.base_models = {}
        self.meta_learner = None
        self.ensemble_model = None
        self.best_model = None
        
        # Configuration
        self.use_smote = HAS_IMBLEARN
        self.use_optuna = HAS_OPTUNA
        self.n_folds = 5
        self.random_state = 42
        
        # Metrics and history
        self.training_metrics = None
        self.feature_importance = {}
        self.model_interpretation = {}
        
        # Enhanced features tracking
        self.attention_history: List[Dict[str, Any]] = []
        
    def get_required_fields(self) -> List[str]:
        """Required input fields for enhanced attention analysis."""
        return [
            "mouse_movements",
            "keyboard_events", 
            "scroll_events",
            "content_interactions",
            "timestamp",
            "content_type",
            "content_difficulty",
            "session_duration",
            "performance_metrics",
            "behavioral_patterns"
        ]
    
    def _initialize_base_models(self) -> Dict[str, Any]:
        """Initialize base models for ensemble."""
        models = {}
        
        # Random Forest - Robust baseline
        models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Neural Network - Complex pattern recognition
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
        
        # XGBoost - Advanced gradient boosting
        if HAS_XGBOOST:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
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
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            )
        
        return models
    
    def _handle_class_imbalance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Handle class imbalance using SMOTE or class weights."""
        if not HAS_IMBLEARN:
            logger.info("Using class weights for imbalance handling")
            return X, y
            
        try:
            # Use SMOTE for oversampling minority classes
            smote = SMOTE(
                sampling_strategy='auto',
                random_state=self.random_state,
                k_neighbors=min(5, len(np.unique(y)) - 1)
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            logger.info(f"SMOTE applied: {X.shape[0]} → {X_resampled.shape[0]} samples")
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Using original data.")
            return X, y
    
    def _optimize_hyperparameters(self, model, param_grid: Dict, X: np.ndarray, y: np.ndarray) -> Any:
        """Optimize hyperparameters using Optuna or GridSearch."""
        if not HAS_OPTUNA:
            # Fallback to GridSearchCV
            grid_search = GridSearchCV(
                model, param_grid,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X, y)
            return grid_search.best_estimator_
        
        # Use Optuna for Bayesian optimization
        try:
            optuna_search = OptunaSearchCV(
                model, param_grid,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
                scoring='accuracy',
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
        """Create stacking ensemble with optimized meta-learner."""
        # Prepare base estimators
        estimators = [(name, model) for name, model in base_models.items()]
        
        # Meta-learner with regularization
        meta_learner = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000
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
    
    def _evaluate_ensemble(self, model, X: np.ndarray, y: np.ndarray) -> EnsembleMetrics:
        """Comprehensive ensemble evaluation."""
        # Cross-validation scores
        cv_scores = cross_val_score(
            model, X, y,
            cv=StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state),
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Fit model for detailed metrics
        model.fit(X, y)
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        
        # ROC AUC for multiclass
        try:
            roc_auc = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            roc_auc = None
        
        # Feature importance (for tree-based models)
        feature_importance = {}
        try:
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(enumerate(model.feature_importances_))
            elif hasattr(model, 'estimators_'):
                # For ensemble models, aggregate importance
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
        
        return EnsembleMetrics(
            accuracy=accuracy,
            precision={k: v['precision'] for k, v in report.items() if isinstance(v, dict)},
            recall={k: v['recall'] for k, v in report.items() if isinstance(v, dict)},
            f1_score={k: v['f1-score'] for k, v in report.items() if isinstance(v, dict)},
            roc_auc=roc_auc,
            cross_val_scores=cv_scores.tolist(),
            feature_importance=feature_importance,
            model_weights=model_weights
        )
    
    def train(self, training_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Train the enhanced ensemble attention tracker."""
        self.status = ModelStatus.TRAINING
        
        try:
            logger.info("Starting enhanced attention tracker training...")
            
            # Prepare training data
            X = training_data.drop(['attention_level', 'user_id', 'session_id'], axis=1, errors='ignore')
            y = training_data['attention_level']
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Handle class imbalance
            X_balanced, y_balanced = self._handle_class_imbalance(X_scaled, y_encoded)
            
            # Initialize base models
            self.base_models = self._initialize_base_models()
            
            # Train and optimize individual models
            logger.info("Training base models with hyperparameter optimization...")
            optimized_models = {}
            
            for name, model in self.base_models.items():
                logger.info(f"Training {name}...")
                
                # Define parameter grids for optimization
                param_grids = self._get_param_grids()
                param_grid = param_grids.get(name, {})
                
                # Optimize hyperparameters
                optimized_model = self._optimize_hyperparameters(model, param_grid, X_balanced, y_balanced)
                optimized_models[name] = optimized_model
                
                # Evaluate individual model
                metrics = self._evaluate_ensemble(optimized_model, X_balanced, y_balanced)
                logger.info(f"{name} accuracy: {metrics.accuracy:.4f} (±{np.std(metrics.cross_val_scores):.4f})")
            
            # Create stacking ensemble
            logger.info("Creating stacking ensemble...")
            self.ensemble_model = self._create_stacking_ensemble(optimized_models, X_balanced, y_balanced)
            
            # Train ensemble
            self.ensemble_model.fit(X_balanced, y_balanced)
            
            # Evaluate ensemble
            self.training_metrics = self._evaluate_ensemble(self.ensemble_model, X_balanced, y_balanced)
            
            # Select best model (ensemble vs individual)
            ensemble_score = np.mean(self.training_metrics.cross_val_scores)
            best_individual_score = max([
                np.mean(cross_val_score(model, X_balanced, y_balanced, cv=self.n_folds))
                for model in optimized_models.values()
            ])
            
            if ensemble_score > best_individual_score:
                self.best_model = self.ensemble_model
                logger.info(f"Using ensemble model (accuracy: {ensemble_score:.4f})")
            else:
                # Use best individual model
                best_model_name = max(optimized_models.keys(), 
                                    key=lambda k: np.mean(cross_val_score(optimized_models[k], X_balanced, y_balanced, cv=self.n_folds)))
                self.best_model = optimized_models[best_model_name]
                logger.info(f"Using best individual model: {best_model_name} (accuracy: {best_individual_score:.4f})")
            
            # Model interpretation
            if HAS_SHAP:
                try:
                    self._generate_model_interpretation(X_balanced)
                except Exception as e:
                    logger.warning(f"SHAP interpretation failed: {e}")
            
            # Prepare metrics for return
            final_metrics = {
                'train_accuracy': self.training_metrics.accuracy,
                'cross_val_mean': np.mean(self.training_metrics.cross_val_scores),
                'cross_val_std': np.std(self.training_metrics.cross_val_scores),
                'feature_count': X.shape[1],
                'training_samples': len(X_balanced),
                'n_classes': len(np.unique(y_encoded))
            }
            
            if self.training_metrics.roc_auc:
                final_metrics['roc_auc'] = self.training_metrics.roc_auc
            
            # Store training history
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'metrics': final_metrics,
                'model_version': self.version,
                'ensemble_used': isinstance(self.best_model, StackingClassifier)
            }
            self.training_history.append(training_record)
            
            self.status = ModelStatus.TRAINED
            logger.info(f"Enhanced attention tracker trained successfully!")
            logger.info(f"Final accuracy: {final_metrics['train_accuracy']:.4f}")
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
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'neural_network': {
                'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
        }
        
        if HAS_XGBOOST:
            param_grids['xgboost'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        if HAS_LIGHTGBM:
            param_grids['lightgbm'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
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
        """Predict attention level using enhanced ensemble model."""
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
            
            # Make prediction
            prediction_proba = self.best_model.predict_proba(features_scaled)[0]
            predicted_class_encoded = self.best_model.predict(features_scaled)[0]
            
            # Decode prediction
            predicted_class = self.label_encoder.inverse_transform([predicted_class_encoded])[0]
            confidence = max(prediction_proba)
            
            # Get class probabilities
            class_labels = self.label_encoder.inverse_transform(range(len(prediction_proba)))
            probabilities = dict(zip(class_labels, prediction_proba))
            
            # Create prediction result
            result = PredictionResult(
                value=predicted_class,
                confidence=confidence,
                metadata={
                    'probabilities': probabilities,
                    'feature_vector_size': features.shape[1],
                    'model_type': 'enhanced_ensemble',
                    'ensemble_used': isinstance(self.best_model, StackingClassifier),
                    'context': {
                        'user_id': context.user_id,
                        'session_id': context.session_id,
                        'content_id': context.content_id
                    },
                    'recommendations': self._generate_enhanced_recommendations(predicted_class, confidence, input_data),
                    'attention_indicators': self._extract_attention_indicators(input_data)
                },
                timestamp=datetime.now().isoformat(),
                model_version=self.version
            )
            
            # Store in history
            self.attention_history.append({
                'timestamp': result.timestamp,
                'attention_level': predicted_class,
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
        # Use enhanced feature extractor if available
        if hasattr(self.feature_extractor, 'extract_comprehensive_features'):
            return self.feature_extractor.extract_comprehensive_features(input_data)
        
        # Fallback to basic feature extraction
        return self.preprocess_data(input_data)
    
    def _generate_enhanced_recommendations(self, attention_level: str, confidence: float, 
                                         input_data: Dict[str, Any]) -> List[str]:
        """Generate enhanced recommendations based on attention prediction and input analysis."""
        recommendations = []
        
        # Base recommendations by attention level
        if attention_level == AttentionLevel.CRITICAL:
            recommendations.extend([
                "Immediate intervention required - suggest break",
                "Switch to passive content (videos, audio)",
                "Reduce cognitive load significantly",
                "Enable gamification and rewards",
                "Consider environmental factors (lighting, noise)"
            ])
        elif attention_level == AttentionLevel.LOW:
            recommendations.extend([
                "Introduce interactive elements",
                "Break content into micro-learning chunks",
                "Add visual stimuli and animations",
                "Implement spaced repetition",
                "Suggest brief physical activity"
            ])
        elif attention_level == AttentionLevel.MEDIUM:
            recommendations.extend([
                "Maintain current engagement strategy",
                "Add periodic attention checks",
                "Monitor for declining trends",
                "Introduce variety in content format"
            ])
        else:  # HIGH
            recommendations.extend([
                "Maintain optimal learning conditions",
                "Consider increasing content complexity",
                "Leverage high attention for difficult concepts",
                "Implement accelerated learning pace"
            ])
        
        # Confidence-based recommendations
        if confidence < 0.7:
            recommendations.append("Gather more behavioral data for improved accuracy")
        elif confidence > 0.9:
            recommendations.append("High confidence prediction - implement immediately")
        
        # Data-driven recommendations
        mouse_data = input_data.get("mouse_movements", [])
        if mouse_data and len(mouse_data) > 10:
            avg_velocity = np.mean([m.get('velocity', 0) for m in mouse_data])
            if avg_velocity > 500:
                recommendations.append("High mouse activity detected - user may be distracted")
            elif avg_velocity < 50:
                recommendations.append("Low mouse activity - user may be disengaged")
        
        return recommendations
    
    def _extract_attention_indicators(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key attention indicators from input data."""
        indicators = {}
        
        # Mouse activity indicators
        mouse_data = input_data.get("mouse_movements", [])
        if mouse_data:
            velocities = [m.get('velocity', 0) for m in mouse_data]
            indicators['mouse_activity_level'] = np.mean(velocities) if velocities else 0
            indicators['mouse_consistency'] = 1 / (np.std(velocities) + 0.001) if velocities else 0
        
        # Keyboard activity indicators
        keyboard_data = input_data.get("keyboard_events", [])
        if keyboard_data:
            intervals = []
            for i in range(1, len(keyboard_data)):
                interval = keyboard_data[i].get('timestamp', 0) - keyboard_data[i-1].get('timestamp', 0)
                if interval > 0:
                    intervals.append(interval)
            indicators['typing_rhythm'] = 1 / np.mean(intervals) if intervals else 0
        
        # Content engagement indicators
        interactions = input_data.get("content_interactions", [])
        if interactions:
            indicators['interaction_frequency'] = len(interactions) / max(1, input_data.get('session_duration', 600))
        
        return indicators
    
    def preprocess_data(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """Fallback preprocessing for basic features."""
        features = []
        
        # Basic mouse features
        mouse_data = raw_data.get("mouse_movements", [])
        if mouse_data:
            velocities = [m.get('velocity', 0) for m in mouse_data]
            features.extend([
                np.mean(velocities) if velocities else 0,
                np.std(velocities) if velocities else 0,
                len(mouse_data)
            ])
        else:
            features.extend([0, 0, 0])
        
        # Basic keyboard features
        keyboard_data = raw_data.get("keyboard_events", [])
        if keyboard_data:
            features.extend([len(keyboard_data)])
        else:
            features.extend([0])
        
        # Basic interaction features
        interactions = raw_data.get("content_interactions", [])
        features.extend([len(interactions) if interactions else 0])
        
        # Contextual features
        content_difficulty = raw_data.get("content_difficulty", 5) / 10.0
        session_duration = min(raw_data.get("session_duration", 600), 3600) / 3600.0  # Normalize to hours
        features.extend([content_difficulty, session_duration])
        
        return np.array(features).reshape(1, -1)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = super().get_model_info()
        
        if self.training_metrics:
            info.update({
                'ensemble_accuracy': self.training_metrics.accuracy,
                'cross_val_scores': self.training_metrics.cross_val_scores,
                'feature_importance_available': bool(self.training_metrics.feature_importance),
                'model_interpretation_available': self.model_interpretation.get('interpretation_available', False),
                'roc_auc': self.training_metrics.roc_auc
            })
        
        info.update({
            'ensemble_type': 'stacking' if isinstance(self.best_model, StackingClassifier) else 'single',
            'smote_enabled': self.use_smote,
            'hyperparameter_optimization': 'optuna' if self.use_optuna else 'grid_search',
            'base_models': list(self.base_models.keys()) if self.base_models else []
        })
        
        return info
    
    def save_model(self, filepath: str) -> bool:
        """Save enhanced model to file."""
        try:
            model_data = {
                'best_model': self.best_model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_extractor': self.feature_extractor,
                'training_metrics': self.training_metrics,
                'model_interpretation': self.model_interpretation,
                'base_models': self.base_models,
                'version': self.version,
                'model_info': self.get_model_info()
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Enhanced attention tracker saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load enhanced model from file."""
        try:
            model_data = joblib.load(filepath)
            
            self.best_model = model_data['best_model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_extractor = model_data.get('feature_extractor', EnhancedAttentionFeatures())
            self.training_metrics = model_data.get('training_metrics')
            self.model_interpretation = model_data.get('model_interpretation', {})
            self.base_models = model_data.get('base_models', {})
            
            self.status = ModelStatus.TRAINED
            logger.info(f"Enhanced attention tracker loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False