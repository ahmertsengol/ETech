"""
AI Learning Psychology Models Training Script
Trains all AI models with generated data and saves them.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Import our data generator
from ai.data.training_data_generator import generate_all_training_data


class ModelTrainer:
    """Comprehensive model training for all AI components."""
    
    def __init__(self, data_path: str = "training_data/"):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.metrics = {}
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        
    def train_all_models(self):
        """Train all AI models."""
        print("🚀 Starting AI Model Training Pipeline...")
        print("=" * 60)
        
        # Generate training data if not exists
        if not os.path.exists(self.data_path):
            print("📊 Generating training data first...")
            generate_all_training_data(self.data_path)
        
        # Train each model
        self.train_attention_tracker()
        self.train_cognitive_load_assessor()
        self.train_learning_style_detector()
        
        # Save all models
        self.save_all_models()
        
        # Generate training report
        self.generate_training_report()
        
        print("\n🎉 All models trained successfully!")
        print("=" * 60)
    
    def train_attention_tracker(self):
        """Train the attention tracking model."""
        print("\n🎯 Training Attention Tracker Model...")
        
        # Load data
        data = pd.read_csv(f"{self.data_path}/attention_training_data.csv")
        
        # Prepare features
        feature_columns = [
            'avg_mouse_speed', 'mouse_movement_variance', 'click_frequency',
            'idle_periods', 'typing_speed', 'typing_consistency', 'backspace_frequency',
            'scroll_speed', 'scroll_direction_changes', 'content_interaction_depth',
            'tab_switches', 'focus_duration', 'session_time_of_day', 'session_duration'
        ]
        
        X = data[feature_columns]
        y = data['attention_level']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'cross_val_accuracy': cross_val_score(model, X_train, y_train, cv=5).mean()
        }
        
        # Feature importance
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        
        # Store everything
        self.models['attention_tracker'] = model
        self.scalers['attention_tracker'] = scaler
        self.encoders['attention_tracker'] = label_encoder
        self.metrics['attention_tracker'] = {
            'performance': metrics,
            'feature_importance': feature_importance,
            'classes': label_encoder.classes_,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"   ✅ Accuracy: {metrics['accuracy']:.3f}")
        print(f"   ✅ F1 Score: {metrics['f1']:.3f}")
        print(f"   ✅ Cross-validation: {metrics['cross_val_accuracy']:.3f}")
    
    def train_cognitive_load_assessor(self):
        """Train the cognitive load assessment model."""
        print("\n🧠 Training Cognitive Load Assessor Model...")
        
        # Load data
        data = pd.read_csv(f"{self.data_path}/cognitive_load_training_data.csv")
        
        # Prepare features
        feature_columns = [
            'avg_response_time', 'response_time_std', 'avg_accuracy',
            'accuracy_decline', 'task_complexity', 'error_count', 'critical_errors',
            'hesitation_count', 'multitask_events', 'fatigue_indicators', 'session_duration'
        ]
        
        X = data[feature_columns]
        y = data['cognitive_load_score']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model (Neural Network for regression)
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'cross_val_r2': cross_val_score(model, X_train, y_train, cv=5).mean()
        }
        
        # Store everything
        self.models['cognitive_load_assessor'] = model
        self.scalers['cognitive_load_assessor'] = scaler
        self.metrics['cognitive_load_assessor'] = {
            'performance': metrics,
            'feature_columns': feature_columns,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"   ✅ R² Score: {metrics['r2']:.3f}")
        print(f"   ✅ RMSE: {metrics['rmse']:.3f}")
        print(f"   ✅ Cross-validation R²: {metrics['cross_val_r2']:.3f}")
    
    def train_learning_style_detector(self):
        """Train the learning style detection model."""
        print("\n🎨 Training Learning Style Detector Model...")
        
        # Load data
        data = pd.read_csv(f"{self.data_path}/learning_style_training_data.csv")
        
        # Prepare features
        feature_columns = [
            'visual_interactions', 'auditory_interactions', 'text_interactions',
            'interactive_interactions', 'visual_time_ratio', 'auditory_time_ratio',
            'text_time_ratio', 'interactive_time_ratio', 'visual_performance',
            'auditory_performance', 'text_performance', 'interactive_performance',
            'visual_engagement', 'auditory_engagement', 'text_engagement',
            'interactive_engagement', 'completion_variance', 'replay_concentration'
        ]
        
        X = data[feature_columns]
        y = data['learning_style']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'cross_val_accuracy': cross_val_score(model, X_train, y_train, cv=5).mean()
        }
        
        # Feature importance
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        
        # Store everything
        self.models['learning_style_detector'] = model
        self.scalers['learning_style_detector'] = scaler
        self.encoders['learning_style_detector'] = label_encoder
        self.metrics['learning_style_detector'] = {
            'performance': metrics,
            'feature_importance': feature_importance,
            'classes': label_encoder.classes_,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"   ✅ Accuracy: {metrics['accuracy']:.3f}")
        print(f"   ✅ F1 Score: {metrics['f1']:.3f}")
        print(f"   ✅ Cross-validation: {metrics['cross_val_accuracy']:.3f}")
    
    def save_all_models(self):
        """Save all trained models and preprocessing objects."""
        print("\n💾 Saving all models and preprocessors...")
        
        # Save models
        for name, model in self.models.items():
            with open(f"models/{name}_model.pkl", 'wb') as f:
                pickle.dump(model, f)
            print(f"   ✅ Saved {name} model")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            with open(f"models/{name}_scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
            print(f"   ✅ Saved {name} scaler")
        
        # Save encoders
        for name, encoder in self.encoders.items():
            with open(f"models/{name}_encoder.pkl", 'wb') as f:
                pickle.dump(encoder, f)
            print(f"   ✅ Saved {name} encoder")
        
        # Save metrics
        with open("models/training_metrics.pkl", 'wb') as f:
            pickle.dump(self.metrics, f)
        print("   ✅ Saved training metrics")
    
    def generate_training_report(self):
        """Generate comprehensive training report."""
        print("\n📊 Generating Training Report...")
        
        report = []
        report.append("🎯 AI LEARNING PSYCHOLOGY MODELS TRAINING REPORT")
        report.append("=" * 60)
        report.append(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Models Trained: {len(self.models)}")
        report.append("")
        
        # Attention Tracker Report
        if 'attention_tracker' in self.metrics:
            metrics = self.metrics['attention_tracker']
            report.append("🎯 ATTENTION TRACKER MODEL")
            report.append("-" * 30)
            report.append(f"Model Type: Random Forest Classifier")
            report.append(f"Training Samples: {metrics['training_samples']:,}")
            report.append(f"Test Samples: {metrics['test_samples']:,}")
            report.append(f"Accuracy: {metrics['performance']['accuracy']:.3f}")
            report.append(f"Precision: {metrics['performance']['precision']:.3f}")
            report.append(f"Recall: {metrics['performance']['recall']:.3f}")
            report.append(f"F1 Score: {metrics['performance']['f1']:.3f}")
            report.append(f"Cross-validation: {metrics['performance']['cross_val_accuracy']:.3f}")
            report.append(f"Classes: {', '.join(metrics['classes'])}")
            
            # Top features
            top_features = sorted(metrics['feature_importance'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            report.append("Top Features:")
            for feature, importance in top_features:
                report.append(f"  • {feature}: {importance:.3f}")
            report.append("")
        
        # Cognitive Load Assessor Report
        if 'cognitive_load_assessor' in self.metrics:
            metrics = self.metrics['cognitive_load_assessor']
            report.append("🧠 COGNITIVE LOAD ASSESSOR MODEL")
            report.append("-" * 30)
            report.append(f"Model Type: Neural Network Regressor")
            report.append(f"Training Samples: {metrics['training_samples']:,}")
            report.append(f"Test Samples: {metrics['test_samples']:,}")
            report.append(f"R² Score: {metrics['performance']['r2']:.3f}")
            report.append(f"RMSE: {metrics['performance']['rmse']:.3f}")
            report.append(f"MAE: {metrics['performance']['mae']:.3f}")
            report.append(f"Cross-validation R²: {metrics['performance']['cross_val_r2']:.3f}")
            report.append("")
        
        # Learning Style Detector Report
        if 'learning_style_detector' in self.metrics:
            metrics = self.metrics['learning_style_detector']
            report.append("🎨 LEARNING STYLE DETECTOR MODEL")
            report.append("-" * 30)
            report.append(f"Model Type: Random Forest Classifier")
            report.append(f"Training Samples: {metrics['training_samples']:,}")
            report.append(f"Test Samples: {metrics['test_samples']:,}")
            report.append(f"Accuracy: {metrics['performance']['accuracy']:.3f}")
            report.append(f"Precision: {metrics['performance']['precision']:.3f}")
            report.append(f"Recall: {metrics['performance']['recall']:.3f}")
            report.append(f"F1 Score: {metrics['performance']['f1']:.3f}")
            report.append(f"Cross-validation: {metrics['performance']['cross_val_accuracy']:.3f}")
            report.append(f"Classes: {', '.join(metrics['classes'])}")
            
            # Top features
            top_features = sorted(metrics['feature_importance'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            report.append("Top Features:")
            for feature, importance in top_features:
                report.append(f"  • {feature}: {importance:.3f}")
            report.append("")
        
        # Overall Assessment
        report.append("🎯 OVERALL ASSESSMENT")
        report.append("-" * 30)
        report.append("✅ All models trained successfully")
        report.append("✅ Models saved to 'models/' directory")
        report.append("✅ Ready for production deployment")
        report.append("")
        
        # Model Performance Summary
        report.append("📈 PERFORMANCE SUMMARY")
        report.append("-" * 30)
        
        if 'attention_tracker' in self.metrics:
            acc = self.metrics['attention_tracker']['performance']['accuracy']
            report.append(f"Attention Tracker: {acc:.1%} accuracy")
        
        if 'cognitive_load_assessor' in self.metrics:
            r2 = self.metrics['cognitive_load_assessor']['performance']['r2']
            report.append(f"Cognitive Load Assessor: {r2:.1%} R² score")
        
        if 'learning_style_detector' in self.metrics:
            acc = self.metrics['learning_style_detector']['performance']['accuracy']
            report.append(f"Learning Style Detector: {acc:.1%} accuracy")
        
        report.append("")
        report.append("🚀 Models ready for integration into production system!")
        
        # Save report
        with open("models/training_report.txt", 'w') as f:
            f.write('\n'.join(report))
        
        # Print report
        print('\n'.join(report))
        print("   ✅ Training report saved to 'models/training_report.txt'")


def main():
    """Main training function."""
    print("🔥 AI Learning Psychology Models Training")
    print("=" * 50)
    
    # Create trainer and train all models
    trainer = ModelTrainer()
    trainer.train_all_models()
    
    print("\n🎊 Training Complete! Your AI models are ready!")
    print("💡 Next steps:")
    print("   1. Run 'python demo.py' to test the trained models")
    print("   2. Check 'models/' directory for saved models")
    print("   3. Review 'models/training_report.txt' for detailed metrics")


if __name__ == "__main__":
    main() 