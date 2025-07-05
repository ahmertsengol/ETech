#!/usr/bin/env python3
"""
🧠 AI Learning Psychology Analyzer - Hugging Face Inference
==========================================================

Easy-to-use inference wrapper for the AI Learning Psychology models hosted on Hugging Face Hub.
This module provides a simple interface to load and use the trained models for predictions.

Usage:
    from huggingface_inference import LearningPsychologyAnalyzer
    
    # Initialize analyzer
    analyzer = LearningPsychologyAnalyzer("your-username/ai-learning-psychology-analyzer")
    
    # Analyze student data
    results = analyzer.analyze_student({
        'attention_features': [0.8, 0.7, 0.9, 0.6, 0.8],
        'cognitive_features': [0.6, 0.4, 0.7, 0.5, 0.3],
        'style_features': [0.7, 0.6, 0.8, 0.9, 0.5]
    })
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files
    import huggingface_hub
    print(f"✅ Loaded dependencies successfully (huggingface_hub: {huggingface_hub.__version__})")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install huggingface-hub scikit-learn pandas numpy")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when a required model file is not found."""
    pass


class PredictionError(Exception):
    """Raised when model prediction fails."""
    pass


class LearningPsychologyAnalyzer:
    """
    Main interface for AI Learning Psychology analysis using Hugging Face hosted models.
    
    This class provides easy access to three specialized models:
    1. Attention Tracker - Monitors student attention levels
    2. Cognitive Load Assessor - Evaluates cognitive burden
    3. Learning Style Detector - Identifies learning preferences
    """
    
    def __init__(self, repo_id: str = "your-username/ai-learning-psychology-analyzer", 
                 cache_dir: Optional[str] = None):
        """
        Initialize the analyzer with models from Hugging Face Hub.
        
        Args:
            repo_id: Hugging Face repository ID (username/model-name)
            cache_dir: Custom cache directory for downloaded models
        """
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_metadata = {}
        
        # Model configurations
        self.model_configs = {
            'attention_tracker': {
                'features': ['academic_engagement', 'attendance_rate', 'study_consistency', 
                           'study_commitment', 'session_consistency'],
                'classes': ['CRITICAL', 'LOW', 'MEDIUM', 'HIGH'],
                'description': 'Monitors student attention levels during learning sessions'
            },
            'cognitive_load_assessor': {
                'features': ['task_difficulty', 'confusion_level', 'workload_intensity', 
                           'processing_time', 'performance_pressure'],
                'classes': ['OPTIMAL', 'MODERATE', 'OVERLOADED', 'CRITICAL'],
                'description': 'Evaluates cognitive burden and mental workload'
            },
            'learning_style_detector': {
                'features': ['study_intensity', 'interactive_preference', 'reflective_tendency', 
                           'positive_engagement', 'structured_study'],
                'classes': ['VISUAL', 'AUDITORY', 'READING', 'KINESTHETIC', 'MULTIMODAL'],
                'description': 'Identifies preferred learning styles and preferences'
            }
        }
        
        logger.info(f"Initializing LearningPsychologyAnalyzer with repo: {repo_id}")
        
        # Load models
        self.load_models()
    
    def list_available_files(self) -> List[str]:
        """List all available files in the repository."""
        try:
            files = list_repo_files(self.repo_id)
            return files
        except Exception as e:
            logger.error(f"Error listing repository files: {str(e)}")
            return []
    
    def download_file(self, filename: str) -> str:
        """Download a specific file from the repository."""
        try:
            file_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                cache_dir=self.cache_dir
            )
            logger.info(f"Downloaded {filename} to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error downloading {filename}: {str(e)}")
            raise ModelNotFoundError(f"Could not download {filename} from {self.repo_id}")
    
    def load_models(self) -> None:
        """Load all models, scalers, and encoders from Hugging Face Hub."""
        logger.info("Loading models from Hugging Face Hub...")
        
        # Check available files
        available_files = self.list_available_files()
        if not available_files:
            raise ModelNotFoundError(f"No files found in repository {self.repo_id}")
        
        logger.info(f"Found {len(available_files)} files in repository")
        
        # Load each model
        for model_name in self.model_configs.keys():
            try:
                self._load_model_components(model_name, available_files)
                logger.info(f"✅ Loaded {model_name} successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name}: {str(e)}")
                # Continue loading other models
        
        # Load training metrics if available
        try:
            if 'training_metrics.pkl' in available_files:
                metrics_path = self.download_file('training_metrics.pkl')
                with open(metrics_path, 'rb') as f:
                    self.model_metadata['training_metrics'] = pickle.load(f)
                logger.info("✅ Loaded training metrics")
        except Exception as e:
            logger.warning(f"Could not load training metrics: {str(e)}")
        
        # Summary
        loaded_models = len(self.models)
        total_models = len(self.model_configs)
        logger.info(f"Model loading complete: {loaded_models}/{total_models} models loaded")
        
        if loaded_models == 0:
            raise ModelNotFoundError("No models could be loaded successfully")
    
    def _load_model_components(self, model_name: str, available_files: List[str]) -> None:
        """Load model, scaler, and encoder for a specific model."""
        # Load model
        model_file = f"{model_name}_model.pkl"
        if model_file in available_files:
            model_path = self.download_file(model_file)
            with open(model_path, 'rb') as f:
                self.models[model_name] = pickle.load(f)
        else:
            raise ModelNotFoundError(f"Model file {model_file} not found")
        
        # Load scaler
        scaler_file = f"{model_name}_scaler.pkl"
        if scaler_file in available_files:
            scaler_path = self.download_file(scaler_file)
            with open(scaler_path, 'rb') as f:
                self.scalers[model_name] = pickle.load(f)
        else:
            logger.warning(f"Scaler file {scaler_file} not found")
            self.scalers[model_name] = None
        
        # Load encoder (if exists)
        encoder_file = f"{model_name}_encoder.pkl"
        if encoder_file in available_files:
            encoder_path = self.download_file(encoder_file)
            with open(encoder_path, 'rb') as f:
                self.encoders[model_name] = pickle.load(f)
        else:
            logger.info(f"Encoder file {encoder_file} not found (might not be needed)")
            self.encoders[model_name] = None
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about loaded models."""
        if model_name:
            if model_name not in self.model_configs:
                raise ValueError(f"Unknown model: {model_name}")
            
            info = {
                'name': model_name,
                'loaded': model_name in self.models,
                'config': self.model_configs[model_name],
                'has_scaler': self.scalers.get(model_name) is not None,
                'has_encoder': self.encoders.get(model_name) is not None
            }
            
            if self.model_metadata.get('training_metrics', {}).get(model_name):
                info['training_metrics'] = self.model_metadata['training_metrics'][model_name]
            
            return info
        else:
            return {
                name: self.get_model_info(name) for name in self.model_configs.keys()
            }
    
    def predict_attention(self, features: List[float]) -> Dict[str, Any]:
        """
        Predict attention level from educational engagement features.
        
        Args:
            features: List of 5 features [academic_engagement, attendance_rate, 
                     study_consistency, study_commitment, session_consistency]
                     
        Returns:
            Dictionary with prediction results
        """
        return self._predict_with_model('attention_tracker', features)
    
    def predict_cognitive_load(self, features: List[float]) -> Dict[str, Any]:
        """
        Predict cognitive load level from task performance features.
        
        Args:
            features: List of 5 features [task_difficulty, confusion_level, 
                     workload_intensity, processing_time, performance_pressure]
                     
        Returns:
            Dictionary with prediction results
        """
        return self._predict_with_model('cognitive_load_assessor', features)
    
    def predict_learning_style(self, features: List[float]) -> Dict[str, Any]:
        """
        Predict learning style from behavioral features.
        
        Args:
            features: List of 5 features [study_intensity, interactive_preference, 
                     reflective_tendency, positive_engagement, structured_study]
                     
        Returns:
            Dictionary with prediction results
        """
        return self._predict_with_model('learning_style_detector', features)
    
    def _predict_with_model(self, model_name: str, features: List[float]) -> Dict[str, Any]:
        """Internal method to make predictions with a specific model."""
        if model_name not in self.models:
            raise PredictionError(f"Model {model_name} not loaded")
        
        # Validate features
        expected_features = len(self.model_configs[model_name]['features'])
        if len(features) != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {len(features)}")
        
        try:
            # Convert to numpy array
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features if scaler available
            if self.scalers[model_name] is not None:
                features_scaled = self.scalers[model_name].transform(features_array)
            else:
                features_scaled = features_array
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(features_scaled)[0]
            
            # Get prediction probabilities if available
            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    prob_array = model.predict_proba(features_scaled)[0]
                    classes = self.model_configs[model_name]['classes']
                    probabilities = dict(zip(classes, prob_array))
                except:
                    pass
            
            # Decode prediction if encoder available
            if self.encoders[model_name] is not None:
                try:
                    decoded_prediction = self.encoders[model_name].inverse_transform([prediction])[0]
                except:
                    decoded_prediction = prediction
            else:
                decoded_prediction = prediction
            
            # Get confidence score
            confidence = max(probabilities.values()) if probabilities else 0.5
            
            return {
                'prediction': decoded_prediction,
                'confidence': confidence,
                'probabilities': probabilities,
                'raw_prediction': prediction,
                'model_name': model_name,
                'features_used': self.model_configs[model_name]['features']
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {model_name}: {str(e)}")
            raise PredictionError(f"Failed to predict with {model_name}: {str(e)}")
    
    def analyze_student(self, student_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Perform complete psychological analysis of a student.
        
        Args:
            student_data: Dictionary with feature lists for each model:
                - 'attention_features': List of 5 attention-related features
                - 'cognitive_features': List of 5 cognitive load features  
                - 'style_features': List of 5 learning style features
                
        Returns:
            Complete analysis results
        """
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'student_analysis': {}
        }
        
        # Attention analysis
        if 'attention_features' in student_data:
            try:
                attention_result = self.predict_attention(student_data['attention_features'])
                results['student_analysis']['attention'] = attention_result
            except Exception as e:
                logger.error(f"Attention analysis failed: {str(e)}")
                results['student_analysis']['attention'] = {'error': str(e)}
        
        # Cognitive load analysis
        if 'cognitive_features' in student_data:
            try:
                cognitive_result = self.predict_cognitive_load(student_data['cognitive_features'])
                results['student_analysis']['cognitive_load'] = cognitive_result
            except Exception as e:
                logger.error(f"Cognitive load analysis failed: {str(e)}")
                results['student_analysis']['cognitive_load'] = {'error': str(e)}
        
        # Learning style analysis
        if 'style_features' in student_data:
            try:
                style_result = self.predict_learning_style(student_data['style_features'])
                results['student_analysis']['learning_style'] = style_result
            except Exception as e:
                logger.error(f"Learning style analysis failed: {str(e)}")
                results['student_analysis']['learning_style'] = {'error': str(e)}
        
        # Generate summary
        results['summary'] = self._generate_analysis_summary(results['student_analysis'])
        
        return results
    
    def _generate_analysis_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the analysis results."""
        summary = {
            'models_used': len([k for k in analysis.keys() if 'error' not in analysis[k]]),
            'total_models': len(analysis),
            'overall_confidence': 0.0,
            'recommendations': []
        }
        
        # Calculate overall confidence
        confidences = []
        for model_result in analysis.values():
            if 'confidence' in model_result:
                confidences.append(model_result['confidence'])
        
        if confidences:
            summary['overall_confidence'] = np.mean(confidences)
        
        # Generate recommendations
        if 'attention' in analysis and 'prediction' in analysis['attention']:
            att_level = analysis['attention']['prediction']
            if att_level in ['CRITICAL', 'LOW']:
                summary['recommendations'].append("Consider attention-boosting activities")
        
        if 'cognitive_load' in analysis and 'prediction' in analysis['cognitive_load']:
            cog_level = analysis['cognitive_load']['prediction']
            if cog_level in ['OVERLOADED', 'CRITICAL']:
                summary['recommendations'].append("Reduce task complexity or provide more support")
        
        if 'learning_style' in analysis and 'prediction' in analysis['learning_style']:
            style = analysis['learning_style']['prediction']
            summary['recommendations'].append(f"Adapt content for {style.lower()} learning style")
        
        return summary
    
    def batch_analyze(self, student_data_batch: List[Dict[str, List[float]]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple students at once.
        
        Args:
            student_data_batch: List of student data dictionaries
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i, student_data in enumerate(student_data_batch):
            try:
                result = self.analyze_student(student_data)
                result['student_id'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing student {i}: {str(e)}")
                results.append({
                    'student_id': i,
                    'error': str(e),
                    'timestamp': pd.Timestamp.now().isoformat()
                })
        
        return results
    
    def get_feature_importance(self, model_name: str) -> Optional[Dict[str, float]]:
        """Get feature importance for a specific model."""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            features = self.model_configs[model_name]['features']
            return dict(zip(features, model.feature_importances_))
        
        return None
    
    def create_example_data(self) -> Dict[str, List[float]]:
        """Create example data for testing."""
        return {
            'attention_features': [0.8, 0.7, 0.9, 0.6, 0.8],  # High attention
            'cognitive_features': [0.6, 0.4, 0.7, 0.5, 0.3],  # Moderate load
            'style_features': [0.7, 0.6, 0.8, 0.9, 0.5]       # Mixed style
        }


def main():
    """Example usage of the LearningPsychologyAnalyzer."""
    print("🧠 AI Learning Psychology Analyzer - Demo")
    print("=" * 50)
    
    # Initialize analyzer (replace with your repo ID)
    repo_id = "your-username/ai-learning-psychology-analyzer"
    
    try:
        analyzer = LearningPsychologyAnalyzer(repo_id)
        
        # Get model info
        print("\n📊 Model Information:")
        model_info = analyzer.get_model_info()
        for name, info in model_info.items():
            status = "✅ Loaded" if info['loaded'] else "❌ Not loaded"
            print(f"  {name}: {status}")
        
        # Create example data
        example_data = analyzer.create_example_data()
        print(f"\n🎯 Example Data Created:")
        for key, values in example_data.items():
            print(f"  {key}: {values}")
        
        # Analyze student
        print(f"\n🔍 Running Analysis...")
        results = analyzer.analyze_student(example_data)
        
        # Display results
        print(f"\n📋 Analysis Results:")
        for model_name, result in results['student_analysis'].items():
            if 'error' not in result:
                print(f"  {model_name}:")
                print(f"    Prediction: {result['prediction']}")
                print(f"    Confidence: {result['confidence']:.2f}")
            else:
                print(f"  {model_name}: Error - {result['error']}")
        
        # Show summary
        summary = results['summary']
        print(f"\n📊 Summary:")
        print(f"  Models used: {summary['models_used']}/{summary['total_models']}")
        print(f"  Overall confidence: {summary['overall_confidence']:.2f}")
        print(f"  Recommendations: {', '.join(summary['recommendations'])}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure you have uploaded your models to Hugging Face Hub first!")


if __name__ == "__main__":
    main() 