#!/usr/bin/env python3
"""
Advanced Ensemble Models for Learning Psychology Analysis
========================================================

High-performance ensemble models designed to significantly improve accuracy
for cognitive load assessment, attention tracking, and learning style detection.

Target improvements:
- Cognitive Load: 37.6% → 70%+ accuracy
- Learning Style: 21.3% → 65%+ accuracy
- Attention Tracking: 67.8% → 80%+ accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')


class AdvancedCognitiveLoadEnsemble(BaseEstimator, ClassifierMixin):
    """Advanced ensemble classifier for cognitive load assessment."""
    
    def __init__(self, use_smote: bool = True, random_state: int = 42):
        self.use_smote = use_smote
        self.random_state = random_state
        self.models = {}
        self.meta_learner = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def _create_base_models(self) -> Dict[str, Any]:
        """Create diverse base models for ensemble.""" 
        models = {}
        
        # XGBoost - excellent for tabular data
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='mlogloss',
            verbosity=0
        )
        
        # LightGBM - fast and accurate
        models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbosity=-1
        )
        
        # Random Forest with optimized parameters
        models['random_forest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Gradient Boosting
        models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state
        )
        
        # Neural Network
        models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=500,
            random_state=self.random_state
        )
        
        # SVM with probability estimates
        models['svm'] = SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=self.random_state
        )
        
        return models
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters for each base model."""
        optimized_models = {}
        
        # XGBoost optimization
        xgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        xgb_search = RandomizedSearchCV(
            xgb.XGBClassifier(random_state=self.random_state, eval_metric='mlogloss', verbosity=0),
            xgb_params,
            n_iter=20,
            cv=3,
            scoring='f1_weighted',
            random_state=self.random_state,
            n_jobs=-1
        )
        xgb_search.fit(X, y)
        optimized_models['xgboost'] = xgb_search.best_estimator_
        
        # LightGBM optimization
        lgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        lgb_search = RandomizedSearchCV(
            lgb.LGBMClassifier(random_state=self.random_state, verbosity=-1),
            lgb_params,
            n_iter=20,
            cv=3,
            scoring='f1_weighted',
            random_state=self.random_state,
            n_jobs=-1
        )
        lgb_search.fit(X, y)
        optimized_models['lightgbm'] = lgb_search.best_estimator_
        
        # Random Forest optimization
        rf_params = {
            'n_estimators': [200, 300, 400],
            'max_depth': [8, 12, 16],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=self.random_state, class_weight='balanced', n_jobs=-1),
            rf_params,
            n_iter=20,
            cv=3,
            scoring='f1_weighted',
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_search.fit(X, y)
        optimized_models['random_forest'] = rf_search.best_estimator_
        
        # Use default parameters for other models to save time
        optimized_models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, 
            subsample=0.8, random_state=self.random_state
        )
        
        optimized_models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
            alpha=0.01, learning_rate='adaptive', max_iter=500,
            random_state=self.random_state
        )
        
        return optimized_models
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
            optimize_hyperparams: bool = True) -> 'AdvancedCognitiveLoadEnsemble':
        """Fit the ensemble model."""
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle class imbalance with SMOTE
        if self.use_smote:
            smote = SMOTE(random_state=self.random_state)
            X_balanced, y_balanced = smote.fit_resample(X_scaled, y_encoded)
        else:
            X_balanced, y_balanced = X_scaled, y_encoded
        
        # Create and optimize base models
        if optimize_hyperparams:
            print("Optimizing hyperparameters...")
            self.models = self._optimize_hyperparameters(X_balanced, y_balanced)
        else:
            self.models = self._create_base_models()
        
        # Train base models
        print("Training base models...")
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_balanced, y_balanced)
        
        # Create meta-features for stacking
        print("Creating meta-features...")
        meta_features = self._get_meta_features(X_balanced, y_balanced)
        
        # Train meta-learner
        print("Training meta-learner...")
        self.meta_learner = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=self.random_state
        )
        self.meta_learner.fit(meta_features, y_balanced)
        
        self.is_fitted = True
        return self
    
    def _get_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate meta-features using cross-validation."""
        n_models = len(self.models)
        n_classes = len(np.unique(y))
        n_samples = X.shape[0]
        
        # Meta-features matrix: (n_samples, n_models * n_classes)
        meta_features = np.zeros((n_samples, n_models * n_classes))
        
        # Use cross-validation to generate meta-features
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]
            
            for model_idx, (name, model) in enumerate(self.models.items()):
                # Train model on fold
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train_fold, y_train_fold)
                
                # Get predictions for validation set
                pred_proba = model_copy.predict_proba(X_val_fold)
                
                # Store predictions in meta-features matrix
                start_col = model_idx * n_classes
                end_col = start_col + n_classes
                meta_features[val_idx, start_col:end_col] = pred_proba
        
        return meta_features
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from base models
        n_models = len(self.models)
        n_classes = len(self.label_encoder.classes_)
        n_samples = X.shape[0]
        
        meta_features = np.zeros((n_samples, n_models * n_classes))
        
        for model_idx, (name, model) in enumerate(self.models.items()):
            pred_proba = model.predict_proba(X_scaled)
            start_col = model_idx * n_classes
            end_col = start_col + n_classes
            meta_features[:, start_col:end_col] = pred_proba
        
        # Get final predictions from meta-learner
        final_proba = self.meta_learner.predict_proba(meta_features)
        return final_proba
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        proba = self.predict_proba(X)
        predictions_encoded = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from tree-based models."""
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = model.feature_importances_
        
        return importance_dict
    
    def evaluate_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance with comprehensive metrics."""
        predictions = self.predict(X_test)
        predictions_proba = self.predict_proba(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, predictions),
            'f1_weighted': f1_score(y_test, predictions, average='weighted'),
            'f1_macro': f1_score(y_test, predictions, average='macro'),
            'classification_report': classification_report(y_test, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist()
        }
        
        # Individual model performances
        results['individual_performances'] = {}
        X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in self.models.items():
            individual_pred = model.predict(X_test_scaled)
            individual_pred_decoded = self.label_encoder.inverse_transform(individual_pred)
            results['individual_performances'][name] = {
                'accuracy': accuracy_score(y_test, individual_pred_decoded),
                'f1_weighted': f1_score(y_test, individual_pred_decoded, average='weighted')
            }
        
        return results


class AdvancedLearningStyleEnsemble(BaseEstimator, ClassifierMixin):
    """Advanced ensemble classifier for learning style detection."""
    
    def __init__(self, use_smote: bool = True, random_state: int = 42):
        self.use_smote = use_smote
        self.random_state = random_state
        self.models = {}
        self.meta_learner = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def _create_base_models(self) -> Dict[str, Any]:
        """Create diverse base models for learning style detection."""
        models = {}
        
        # XGBoost optimized for multi-class
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=self.random_state,
            eval_metric='mlogloss',
            verbosity=0
        )
        
        # Random Forest for stability
        models['random_forest'] = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced_subsample',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Extra Trees for diversity
        from sklearn.ensemble import ExtraTreesClassifier
        models['extra_trees'] = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # LightGBM
        models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=self.random_state,
            verbosity=-1
        )
        
        # Neural Network with deeper architecture
        models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            batch_size=32,
            max_iter=800,
            random_state=self.random_state
        )
        
        return models
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'AdvancedLearningStyleEnsemble':
        """Fit the learning style ensemble."""
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle severe class imbalance with SMOTEENN
        if self.use_smote:
            smoteenn = SMOTEENN(random_state=self.random_state)
            X_balanced, y_balanced = smoteenn.fit_resample(X_scaled, y_encoded)
            print(f"Data balancing: {len(X_scaled)} → {len(X_balanced)} samples")
        else:
            X_balanced, y_balanced = X_scaled, y_encoded
        
        # Create base models
        self.models = self._create_base_models()
        
        # Train base models
        print("Training base models for learning style detection...")
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_balanced, y_balanced)
        
        # Create meta-features
        print("Creating meta-features...")
        meta_features = self._get_meta_features(X_balanced, y_balanced)
        
        # Train meta-learner with regularization
        print("Training meta-learner...")
        self.meta_learner = LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            C=0.1,  # Increased regularization
            random_state=self.random_state
        )
        self.meta_learner.fit(meta_features, y_balanced)
        
        self.is_fitted = True
        return self
    
    def _get_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate meta-features using stratified cross-validation."""
        n_models = len(self.models)
        n_classes = len(np.unique(y))
        n_samples = X.shape[0]
        
        meta_features = np.zeros((n_samples, n_models * n_classes))
        
        # Use stratified k-fold to handle class imbalance
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]
            
            for model_idx, (name, model) in enumerate(self.models.items()):
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train_fold, y_train_fold)
                
                pred_proba = model_copy.predict_proba(X_val_fold)
                
                start_col = model_idx * n_classes
                end_col = start_col + n_classes
                meta_features[val_idx, start_col:end_col] = pred_proba
        
        return meta_features
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        X_scaled = self.scaler.transform(X)
        
        n_models = len(self.models)
        n_classes = len(self.label_encoder.classes_)
        n_samples = X.shape[0]
        
        meta_features = np.zeros((n_samples, n_models * n_classes))
        
        for model_idx, (name, model) in enumerate(self.models.items()):
            pred_proba = model.predict_proba(X_scaled)
            start_col = model_idx * n_classes
            end_col = start_col + n_classes
            meta_features[:, start_col:end_col] = pred_proba
        
        final_proba = self.meta_learner.predict_proba(meta_features)
        return final_proba
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        proba = self.predict_proba(X)
        predictions_encoded = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def evaluate_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, predictions),
            'f1_weighted': f1_score(y_test, predictions, average='weighted'),
            'f1_macro': f1_score(y_test, predictions, average='macro'),
            'classification_report': classification_report(y_test, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist()
        }
        
        return results


class AdvancedAttentionEnsemble(BaseEstimator, ClassifierMixin):
    """Advanced ensemble classifier for attention tracking.""" 
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def _create_voting_ensemble(self) -> VotingClassifier:
        """Create a voting ensemble optimized for attention tracking."""
        
        # Base models with optimized parameters for attention
        xgb_model = xgb.XGBClassifier(
            n_estimators=250,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=self.random_state,
            eval_metric='mlogloss',
            verbosity=0
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=400,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=250,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=self.random_state,
            verbosity=-1
        )
        
        # Voting ensemble with soft voting
        voting_ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model),
                ('lgb', lgb_model)
            ],
            voting='soft'
        )
        
        return voting_ensemble
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'AdvancedAttentionEnsemble':
        """Fit the attention ensemble."""
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and train ensemble
        self.ensemble_model = self._create_voting_ensemble()
        print("Training attention tracking ensemble...")
        self.ensemble_model.fit(X_scaled, y_encoded)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        X_scaled = self.scaler.transform(X)
        return self.ensemble_model.predict_proba(X_scaled)
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        X_scaled = self.scaler.transform(X)
        predictions_encoded = self.ensemble_model.predict(X_scaled)
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def evaluate_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, predictions),
            'f1_weighted': f1_score(y_test, predictions, average='weighted'),
            'f1_macro': f1_score(y_test, predictions, average='macro'),
            'classification_report': classification_report(y_test, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist()
        }
        
        return results


def create_ensemble_for_model_type(model_type: str, **kwargs) -> BaseEstimator:
    """Factory function to create appropriate ensemble for model type."""
    
    if model_type == 'cognitive_load':
        return AdvancedCognitiveLoadEnsemble(**kwargs)
    elif model_type == 'learning_style':
        return AdvancedLearningStyleEnsemble(**kwargs)
    elif model_type == 'attention_tracker':
        return AdvancedAttentionEnsemble(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("Advanced Ensemble Models")
    print("=" * 50)
    print("Target Performance Improvements:")
    print("• Cognitive Load: 37.6% → 70%+ accuracy")
    print("• Learning Style: 21.3% → 65%+ accuracy") 
    print("• Attention Tracking: 67.8% → 80%+ accuracy")
    print("\nKey Features:")
    print("• Stacking with meta-learners")
    print("• SMOTE for class imbalance")
    print("• Hyperparameter optimization")
    print("• Multiple algorithm diversity")
    print("• Cross-validation for robust training")