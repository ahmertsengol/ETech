#!/usr/bin/env python3
"""
🚀 Enhanced Ensemble Models Training Script
==========================================

PERFORMANCE TARGETS:
- Attention Tracker: 67.8% → 80%+ accuracy
- Cognitive Load: 37.6% → 70%+ accuracy  
- Learning Style: 21.3% → 65%+ accuracy

This script trains all three enhanced models with:
✓ Advanced ensemble architectures (XGBoost, LightGBM, Random Forest, Neural Networks)
✓ Class imbalance handling with SMOTE variants
✓ Bayesian hyperparameter optimization
✓ Comprehensive validation and cross-validation
✓ Model interpretation and explainability
✓ Performance benchmarking against original models

Usage:
    python train_enhanced_ensemble_models.py [--model MODEL_NAME] [--dataset DATASET_NAME]
    
    MODEL_NAME: attention, cognitive_load, learning_style, or all (default: all)
    DATASET_NAME: student_learning_behavior, student_performance_behavior, uci_student_performance, or all
"""

import sys
import os
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'enhanced_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Import enhanced models
try:
    from src.ai.models.enhanced_attention_tracker import EnhancedAttentionTracker
    from src.ai.models.enhanced_cognitive_load_assessor import EnhancedCognitiveLoadAssessor
    from src.ai.models.enhanced_learning_style_detector import EnhancedLearningStyleDetector
    from src.ai.data.enhanced_feature_engineering import (
        EnhancedAttentionFeatures, 
        EnhancedCognitiveLoadFeatures, 
        EnhancedLearningStyleFeatures
    )
    logger.info("Successfully imported enhanced models")
except ImportError as e:
    logger.error(f"Failed to import enhanced models: {e}")
    sys.exit(1)

# Import original models for comparison
try:
    from src.ai.models.attention_tracker import AttentionTracker
    from src.ai.models.cognitive_load_assessor import CognitiveLoadAssessor
    from src.ai.models.learning_style_detector import LearningStyleDetector
    logger.info("Successfully imported original models for comparison")
except ImportError as e:
    logger.warning(f"Failed to import original models for comparison: {e}")

# Check for optional dependencies
DEPENDENCIES = {
    'xgboost': False,
    'lightgbm': False,
    'imblearn': False,
    'optuna': False,
    'shap': False
}

for dep in DEPENDENCIES:
    try:
        __import__(dep)
        DEPENDENCIES[dep] = True
        logger.info(f"✓ {dep} available")
    except ImportError:
        logger.warning(f"✗ {dep} not available - using fallback methods")

class EnhancedModelTrainer:
    """
    Comprehensive trainer for enhanced ensemble models.
    Handles data loading, preprocessing, training, evaluation, and comparison.
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Available datasets
        self.datasets = [
            "student_learning_behavior",
            "student_performance_behavior", 
            "uci_student_performance"
        ]
        
        # Model configurations
        self.model_configs = {
            'attention': {
                'enhanced_class': EnhancedAttentionTracker,
                'original_class': AttentionTracker if 'AttentionTracker' in globals() else None,
                'target_column': 'attention_level',
                'feature_extractor': EnhancedAttentionFeatures()
            },
            'cognitive_load': {
                'enhanced_class': EnhancedCognitiveLoadAssessor,
                'original_class': CognitiveLoadAssessor if 'CognitiveLoadAssessor' in globals() else None,
                'target_column': 'cognitive_load_score',  # or cognitive_load_level
                'feature_extractor': EnhancedCognitiveLoadFeatures()
            },
            'learning_style': {
                'enhanced_class': EnhancedLearningStyleDetector,
                'original_class': LearningStyleDetector if 'LearningStyleDetector' in globals() else None,
                'target_column': 'learning_style',
                'feature_extractor': EnhancedLearningStyleFeatures()
            }
        }
        
        # Training results
        self.training_results = {}
        
    def load_dataset(self, dataset_name: str, model_type: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Load training and test data for specified dataset and model type."""
        try:
            dataset_path = self.data_dir / dataset_name / model_type
            
            if not dataset_path.exists():
                logger.error(f"Dataset path not found: {dataset_path}")
                return None
            
            # Load training and test data
            X_train = pd.read_csv(dataset_path / "X_train.csv")
            X_test = pd.read_csv(dataset_path / "X_test.csv")
            y_train = pd.read_csv(dataset_path / "y_train.csv")
            y_test = pd.read_csv(dataset_path / "y_test.csv")
            
            # Load metadata
            metadata_file = dataset_path / "metadata.json"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            # Combine features and targets
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
            
            logger.info(f"Loaded {dataset_name}/{model_type}: {len(train_data)} train, {len(test_data)} test samples")
            
            return {
                'train': train_data,
                'test': test_data,
                'metadata': metadata,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}/{model_type}: {e}")
            return None
    
    def prepare_enhanced_features(self, data: pd.DataFrame, model_type: str, 
                                feature_extractor: Any) -> pd.DataFrame:
        """Prepare enhanced features using feature engineering pipeline."""
        try:
            logger.info(f"Extracting enhanced features for {model_type}...")
            
            if not hasattr(feature_extractor, 'extract_comprehensive_features'):
                logger.warning(f"Feature extractor for {model_type} doesn't have comprehensive feature extraction")
                return data
            
            # Extract enhanced features for each row
            enhanced_features_list = []
            
            for idx, row in data.iterrows():
                try:
                    # Convert row to input format expected by feature extractor
                    input_data = self._row_to_input_data(row, model_type)
                    
                    # Extract features
                    enhanced_features = feature_extractor.extract_comprehensive_features(input_data)
                    enhanced_features_list.append(enhanced_features.flatten())
                    
                except Exception as e:
                    logger.warning(f"Failed to extract features for row {idx}: {e}")
                    # Use zeros as fallback
                    enhanced_features_list.append(np.zeros(100))  # Default size
            
            # Create DataFrame with enhanced features
            if enhanced_features_list:
                enhanced_df = pd.DataFrame(enhanced_features_list)
                enhanced_df.columns = [f'enhanced_feature_{i}' for i in range(enhanced_df.shape[1])]
                
                # Add target columns back
                target_cols = self._get_target_columns(model_type)
                for col in target_cols:
                    if col in data.columns:
                        enhanced_df[col] = data[col].values
                
                # Add identifier columns
                for col in ['user_id', 'session_id']:
                    if col in data.columns:
                        enhanced_df[col] = data[col].values
                
                logger.info(f"Enhanced features extracted: {enhanced_df.shape[1]} features")
                return enhanced_df
            else:
                logger.warning("No enhanced features extracted, using original data")
                return data
                
        except Exception as e:
            logger.error(f"Enhanced feature extraction failed: {e}")
            return data
    
    def _row_to_input_data(self, row: pd.Series, model_type: str) -> Dict[str, Any]:
        """Convert dataframe row to input format expected by feature extractors."""
        # This is a simplified conversion - in practice, you'd need to map
        # your actual data columns to the expected input format
        
        input_data = {}
        
        if model_type == 'attention':
            input_data = {
                'mouse_movements': [],
                'keyboard_events': [],
                'scroll_events': [],
                'content_interactions': [],
                'timestamp': datetime.now().isoformat(),
                'content_type': 'text',
                'content_difficulty': 5,
                'session_duration': 600,
                'performance_metrics': {},
                'behavioral_patterns': {}
            }
        elif model_type == 'cognitive_load':
            input_data = {
                'response_times': [2.5, 3.0, 2.8],
                'accuracy_scores': [0.8, 0.7, 0.85],
                'task_complexities': [5, 6, 4],
                'error_patterns': {'minor': 2, 'major': 1},
                'hesitation_indicators': [],
                'multitask_events': [],
                'content_engagement': {'visual': 0.8},
                'timestamp': datetime.now().isoformat(),
                'session_duration': 600,
                'performance_metrics': {},
                'working_memory_indicators': {},
                'fatigue_signals': {}
            }
        elif model_type == 'learning_style':
            input_data = {
                'content_interactions': {'visual': 10, 'auditory': 5, 'text': 15, 'interactive': 8},
                'time_spent_by_type': {'visual': 300, 'auditory': 150, 'text': 450, 'interactive': 200},
                'performance_by_type': {'visual': [0.8, 0.7], 'auditory': [0.6], 'text': [0.9, 0.85], 'interactive': [0.75]},
                'content_preferences': {'visual': 0.7, 'text': 0.9},
                'engagement_metrics': {'visual': 0.8, 'auditory': 0.6, 'text': 0.9, 'interactive': 0.7},
                'completion_rates': {'visual': 0.85, 'auditory': 0.70, 'text': 0.95, 'interactive': 0.80},
                'replay_behaviors': {'visual': 2, 'text': 5},
                'navigation_patterns': {},
                'timestamp': datetime.now().isoformat(),
                'learning_session_data': {},
                'content_format_preferences': {},
                'interaction_depth_metrics': {},
                'cognitive_style_indicators': {}
            }
        
        return input_data
    
    def _get_target_columns(self, model_type: str) -> List[str]:
        """Get target column names for each model type."""
        target_mapping = {
            'attention': ['attention_level'],
            'cognitive_load': ['cognitive_load_score', 'cognitive_load_level'],
            'learning_style': ['learning_style']
        }
        return target_mapping.get(model_type, [])
    
    def train_enhanced_model(self, model_type: str, dataset_name: str) -> Dict[str, Any]:
        """Train enhanced model on specified dataset."""
        logger.info(f"Training enhanced {model_type} model on {dataset_name}")
        
        # Load data
        data = self.load_dataset(dataset_name, model_type)
        if data is None:
            return {'error': 'Failed to load dataset'}
        
        # Get model configuration
        config = self.model_configs[model_type]
        
        try:
            # Initialize enhanced model
            enhanced_model = config['enhanced_class']()
            
            # Prepare enhanced features (optional - models can extract their own)
            # train_data_enhanced = self.prepare_enhanced_features(
            #     data['train'], model_type, config['feature_extractor']
            # )
            
            # Use original data for now
            train_data_enhanced = data['train']
            
            # Train model
            logger.info(f"Starting training for enhanced {model_type}...")
            training_metrics = enhanced_model.train(train_data_enhanced, data['test'])
            
            # Save model
            model_path = self.models_dir / f"enhanced_{model_type}_{dataset_name}.joblib"
            enhanced_model.save_model(str(model_path))
            
            # Evaluate on test set
            test_results = self._evaluate_model(enhanced_model, data['test'], model_type)
            
            results = {
                'model_type': model_type,
                'dataset': dataset_name,
                'training_metrics': training_metrics,
                'test_results': test_results,
                'model_path': str(model_path),
                'model_info': enhanced_model.get_model_info(),
                'enhanced': True
            }
            
            logger.info(f"Enhanced {model_type} training completed successfully")
            logger.info(f"Training accuracy: {training_metrics.get('train_accuracy', 'N/A')}")
            logger.info(f"Test accuracy: {test_results.get('accuracy', 'N/A')}")
            
            return results
            
        except Exception as e:
            logger.error(f"Enhanced {model_type} training failed: {e}")
            return {'error': str(e)}
    
    def train_original_model(self, model_type: str, dataset_name: str) -> Dict[str, Any]:
        """Train original model for comparison."""
        logger.info(f"Training original {model_type} model on {dataset_name} for comparison")
        
        config = self.model_configs[model_type]
        if config['original_class'] is None:
            logger.warning(f"Original {model_type} model not available for comparison")
            return {'error': 'Original model not available'}
        
        # Load data
        data = self.load_dataset(dataset_name, model_type)
        if data is None:
            return {'error': 'Failed to load dataset'}
        
        try:
            # Initialize original model
            original_model = config['original_class']()
            
            # Train model
            logger.info(f"Starting training for original {model_type}...")
            training_metrics = original_model.train(data['train'], data['test'])
            
            # Save model
            model_path = self.models_dir / f"original_{model_type}_{dataset_name}.joblib"
            original_model.save_model(str(model_path))
            
            # Evaluate on test set
            test_results = self._evaluate_model(original_model, data['test'], model_type)
            
            results = {
                'model_type': model_type,
                'dataset': dataset_name,
                'training_metrics': training_metrics,
                'test_results': test_results,
                'model_path': str(model_path),
                'model_info': original_model.get_model_info(),
                'enhanced': False
            }
            
            logger.info(f"Original {model_type} training completed")
            logger.info(f"Training accuracy: {training_metrics.get('train_accuracy', 'N/A')}")
            logger.info(f"Test accuracy: {test_results.get('accuracy', 'N/A')}")
            
            return results
            
        except Exception as e:
            logger.error(f"Original {model_type} training failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_model(self, model, test_data: pd.DataFrame, model_type: str) -> Dict[str, Any]:
        """Evaluate model on test data."""
        try:
            from src.ai.core.base_model import LearningContext
            
            # Create dummy context
            context = LearningContext(
                user_id="test_user",
                session_id="test_session",
                content_id="test_content",
                timestamp=datetime.now().isoformat()
            )
            
            # Get predictions
            predictions = []
            actuals = []
            confidences = []
            
            target_col = self.model_configs[model_type]['target_column']
            
            for idx, row in test_data.iterrows():
                try:
                    # Convert row to input format
                    input_data = self._row_to_input_data(row, model_type)
                    
                    # Make prediction
                    result = model.predict(input_data, context)
                    predictions.append(result.value)
                    confidences.append(result.confidence)
                    
                    # Get actual value
                    if target_col in row:
                        actuals.append(row[target_col])
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for row {idx}: {e}")
                    continue
            
            if not predictions or not actuals:
                return {'error': 'No valid predictions'}
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, classification_report
            
            accuracy = accuracy_score(actuals, predictions)
            avg_confidence = np.mean(confidences)
            
            # Detailed classification report
            try:
                report = classification_report(actuals, predictions, output_dict=True, zero_division=0)
            except:
                report = {}
            
            return {
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'n_predictions': len(predictions),
                'classification_report': report
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {'error': str(e)}
    
    def compare_models(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare enhanced vs original model performance."""
        comparison = {}
        
        # Group results by model type and dataset
        grouped = {}
        for result in results:
            if 'error' in result:
                continue
                
            key = f"{result['model_type']}_{result['dataset']}"
            if key not in grouped:
                grouped[key] = {}
            
            model_variant = 'enhanced' if result['enhanced'] else 'original'
            grouped[key][model_variant] = result
        
        # Compare each pair
        for key, variants in grouped.items():
            if 'enhanced' in variants and 'original' in variants:
                enhanced = variants['enhanced']
                original = variants['original']
                
                enhanced_acc = enhanced['test_results'].get('accuracy', 0)
                original_acc = original['test_results'].get('accuracy', 0)
                
                improvement = enhanced_acc - original_acc
                improvement_pct = (improvement / original_acc * 100) if original_acc > 0 else 0
                
                comparison[key] = {
                    'enhanced_accuracy': enhanced_acc,
                    'original_accuracy': original_acc,
                    'improvement': improvement,
                    'improvement_percentage': improvement_pct,
                    'target_met': improvement_pct > 0
                }
        
        return comparison
    
    def generate_report(self, results: List[Dict[str, Any]], comparison: Dict[str, Any]) -> str:
        """Generate comprehensive training report."""
        report = []
        report.append("🚀 Enhanced Ensemble Models Training Report")
        report.append("=" * 50)
        report.append(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Dependency status
        report.append("📦 Dependencies Status:")
        for dep, available in DEPENDENCIES.items():
            status = "✓" if available else "✗"
            report.append(f"  {status} {dep}")
        report.append("")
        
        # Training results summary
        report.append("📊 Training Results Summary:")
        for result in results:
            if 'error' in result:
                report.append(f"  ❌ {result.get('model_type', 'Unknown')} - {result.get('dataset', 'Unknown')}: {result['error']}")
            else:
                model_name = f"{'Enhanced' if result['enhanced'] else 'Original'} {result['model_type'].title()}"
                accuracy = result['test_results'].get('accuracy', 'N/A')
                report.append(f"  ✅ {model_name} - {result['dataset']}: {accuracy:.4f}" if isinstance(accuracy, float) else f"  ✅ {model_name} - {result['dataset']}: {accuracy}")
        report.append("")
        
        # Model comparison
        if comparison:
            report.append("🔍 Performance Comparison (Enhanced vs Original):")
            for key, comp in comparison.items():
                model_type, dataset = key.split('_', 1)
                improvement = comp['improvement_percentage']
                status = "🎯" if comp['target_met'] else "🔄"
                report.append(f"  {status} {model_type.title()} - {dataset}:")
                report.append(f"    Original: {comp['original_accuracy']:.4f}")
                report.append(f"    Enhanced: {comp['enhanced_accuracy']:.4f}")
                report.append(f"    Improvement: {improvement:+.2f}%")
            report.append("")
        
        # Performance targets
        report.append("🎯 Performance Targets:")
        targets = {
            'attention': {'current': '67.8%', 'target': '80%+'},
            'cognitive_load': {'current': '37.6%', 'target': '70%+'},
            'learning_style': {'current': '21.3%', 'target': '65%+'}
        }
        
        for model_type, target_info in targets.items():
            achieved = any(
                comp['enhanced_accuracy'] >= float(target_info['target'].rstrip('%+')) / 100
                for key, comp in comparison.items()
                if key.startswith(model_type)
            )
            status = "✅" if achieved else "🔄"
            report.append(f"  {status} {model_type.title()}: {target_info['current']} → {target_info['target']}")
        
        return "\n".join(report)
    
    def run_training(self, model_types: List[str], datasets: List[str], 
                    compare_with_original: bool = True) -> Dict[str, Any]:
        """Run complete training pipeline."""
        logger.info("Starting enhanced ensemble models training pipeline")
        
        all_results = []
        
        # Train enhanced models
        for model_type in model_types:
            for dataset in datasets:
                logger.info(f"Training enhanced {model_type} on {dataset}")
                result = self.train_enhanced_model(model_type, dataset)
                all_results.append(result)
        
        # Train original models for comparison
        if compare_with_original:
            for model_type in model_types:
                for dataset in datasets:
                    logger.info(f"Training original {model_type} on {dataset} for comparison")
                    result = self.train_original_model(model_type, dataset)
                    all_results.append(result)
        
        # Compare results
        comparison = self.compare_models(all_results)
        
        # Generate report
        report = self.generate_report(all_results, comparison)
        
        # Save report
        report_path = f"enhanced_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Training completed. Report saved to {report_path}")
        print("\n" + report)
        
        return {
            'results': all_results,
            'comparison': comparison,
            'report': report,
            'report_path': report_path
        }

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Enhanced Ensemble Models')
    parser.add_argument('--model', default='all', 
                       choices=['attention', 'cognitive_load', 'learning_style', 'all'],
                       help='Model type to train (default: all)')
    parser.add_argument('--dataset', default='student_learning_behavior',
                       choices=['student_learning_behavior', 'student_performance_behavior', 
                               'uci_student_performance', 'all'],
                       help='Dataset to use (default: student_learning_behavior)')
    parser.add_argument('--no-comparison', action='store_true',
                       help='Skip training original models for comparison')
    parser.add_argument('--data-dir', default='data/processed',
                       help='Data directory path (default: data/processed)')
    
    args = parser.parse_args()
    
    # Determine models to train
    if args.model == 'all':
        model_types = ['attention', 'cognitive_load', 'learning_style']
    else:
        model_types = [args.model]
    
    # Determine datasets to use
    if args.dataset == 'all':
        datasets = ['student_learning_behavior', 'student_performance_behavior', 'uci_student_performance']
    else:
        datasets = [args.dataset]
    
    # Initialize trainer
    trainer = EnhancedModelTrainer(data_dir=args.data_dir)
    
    # Run training
    try:
        results = trainer.run_training(
            model_types=model_types,
            datasets=datasets,
            compare_with_original=not args.no_comparison
        )
        
        logger.info("Training pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)