"""
Cognitive Load Assessor - Analyzes user's mental workload during learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import logging

from ..core.base_model import BaseAIModel, PredictionResult, LearningContext, ModelStatus
from ..core.model_loader import load_model_components, is_model_available

logger = logging.getLogger(__name__)


class CognitiveLoadLevel:
    """Cognitive load level constants."""
    OPTIMAL = "optimal"          
    UNDERLOADED = "underloaded"  
    OVERLOADED = "overloaded"    
    CRITICAL = "critical"        


class CognitiveLoadAssessor(BaseAIModel):
    """
    Assesses user's cognitive load during learning sessions.
    
    Analyzes:
    - Task complexity vs performance  
    - Response time patterns
    - Error rates and types
    - Multi-tasking indicators
    """
    
    def __init__(self, version: str = "1.0.0"):
        super().__init__("cognitive_load_assessor", version)
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.cognitive_history: List[Dict[str, Any]] = []
        self.use_trained_model = False
        self.trained_components = None
        
        # Try to load trained model components
        self._load_trained_model()
    
    def _load_trained_model(self):
        """Load trained model components if available."""
        try:
            if is_model_available("cognitive_load_assessor"):
                # Load components manually since cognitive load assessor doesn't need encoder
                from ..core.model_loader import get_model_loader
                loader = get_model_loader()
                
                self.trained_components = {
                    'model': loader.load_model("cognitive_load_assessor"),
                    'scaler': loader.load_scaler("cognitive_load_assessor"),
                    'encoder': None  # Cognitive load assessor doesn't need encoder
                }
                
                if (self.trained_components['model'] is not None and 
                    self.trained_components['scaler'] is not None):
                    self.use_trained_model = True
                    logger.info("Trained cognitive load assessor model loaded successfully")
                else:
                    logger.warning("Some trained model components are missing")
            else:
                logger.info("No trained cognitive load assessor model found, using fallback mode")
        except Exception as e:
            logger.error(f"Failed to load trained model: {str(e)}")
            self.use_trained_model = False
        
    def get_required_fields(self) -> List[str]:
        """Required input fields for cognitive load analysis."""
        return [
            "response_times",
            "accuracy_scores", 
            "task_complexities",
            "error_patterns",
            "hesitation_indicators",
            "multitask_events",
            "content_engagement",
            "timestamp",
            "session_duration"
        ]
    
    def train(self, training_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Train the cognitive load prediction model."""
        self.status = ModelStatus.TRAINING
        
        try:
            # Prepare training data
            X = training_data.drop(['cognitive_load_score', 'user_id', 'session_id'], axis=1)
            y = training_data['cognitive_load_score']  # Continuous score 0-1
            
            # Scale features and target
            X_scaled = self.scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
            
            # Train Neural Network model
            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.01,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
            
            self.model.fit(X_scaled, y_scaled)
            
            # Calculate training metrics
            train_predictions = self.model.predict(X_scaled)
            train_predictions_original = self.target_scaler.inverse_transform(
                train_predictions.reshape(-1, 1)
            ).ravel()
            
            train_mse = mean_squared_error(y, train_predictions_original)
            train_r2 = r2_score(y, train_predictions_original)
            
            metrics = {
                'train_mse': train_mse,
                'train_r2': train_r2,
                'feature_count': X.shape[1],
                'training_samples': len(X)
            }
            
            # Store training history
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'model_version': self.version
            }
            self.training_history.append(training_record)
            
            self.status = ModelStatus.TRAINED
            logger.info(f"Cognitive load assessor trained successfully. R²: {train_r2:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self.status = ModelStatus.ERROR
            raise
    
    def predict(self, input_data: Dict[str, Any], context: LearningContext) -> PredictionResult:
        """Predict cognitive load based on performance data."""
        self.status = ModelStatus.PREDICTING
        
        try:
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data")
            
            # Use trained model if available
            if self.use_trained_model and self.trained_components:
                return self._predict_with_trained_model(input_data, context)
            
            # Fallback to rule-based prediction
            return self._predict_fallback(input_data, context)
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            self.status = ModelStatus.ERROR
            raise
    
    def _predict_with_trained_model(self, input_data: Dict[str, Any], context: LearningContext) -> PredictionResult:
        """Use trained model for prediction."""
        # Prepare features from input data
        features = self._prepare_features_for_trained_model(input_data)
        
        # Scale features
        features_scaled = self.trained_components['scaler'].transform(features)
        
        # Make prediction
        model = self.trained_components['model']
        load_score = model.predict(features_scaled)[0]
        
        # Convert to categorical level
        load_level = self._score_to_level(load_score)
        confidence = self._calculate_confidence(features, load_score)
        
        # Create prediction result
        result = PredictionResult(
            value=load_level,
            confidence=confidence,
            metadata={
                'load_score': float(load_score),
                'feature_vector_size': features.shape[1],
                'model_type': 'trained',
                'context': {
                    'user_id': context.user_id,
                    'session_id': context.session_id,
                    'content_id': context.content_id
                },
                'recommendations': self._generate_recommendations(load_level, load_score),
                'performance_indicators': self._extract_performance_indicators(input_data)
            },
            timestamp=datetime.now().isoformat(),
            model_version=self.version
        )
        
        # Store in history
        self.cognitive_history.append({
            'timestamp': result.timestamp,
            'cognitive_load_level': load_level,
            'load_score': load_score,
            'confidence': confidence,
            'user_id': context.user_id,
            'session_id': context.session_id
        })
        
        self.status = ModelStatus.TRAINED
        return result
    
    def _predict_fallback(self, input_data: Dict[str, Any], context: LearningContext) -> PredictionResult:
        """Fallback rule-based prediction when trained model is not available."""
        # Analyze cognitive load patterns
        load_score = self._analyze_cognitive_load_patterns(input_data)
        
        # Convert to categorical level
        load_level = self._score_to_level(load_score)
        confidence = 0.65  # Lower confidence for fallback
        
        # Create prediction result
        result = PredictionResult(
            value=load_level,
            confidence=confidence,
            metadata={
                'load_score': float(load_score),
                'model_type': 'fallback',
                'context': {
                    'user_id': context.user_id,
                    'session_id': context.session_id,
                    'content_id': context.content_id
                },
                'recommendations': self._generate_recommendations(load_level, load_score),
                'performance_indicators': self._extract_performance_indicators(input_data)
            },
            timestamp=datetime.now().isoformat(),
            model_version=self.version
        )
        
        # Store in history
        self.cognitive_history.append({
            'timestamp': result.timestamp,
            'cognitive_load_level': load_level,
            'load_score': load_score,
            'confidence': confidence,
            'user_id': context.user_id,
            'session_id': context.session_id
        })
        
        self.status = ModelStatus.TRAINED
        return result
    
    def preprocess_data(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """Extract cognitive load features from raw data."""
        features = []
        
        # Performance features
        performance_features = self._extract_performance_features(raw_data)
        features.extend(performance_features)
        
        # Temporal features  
        temporal_features = self._extract_temporal_features(raw_data)
        features.extend(temporal_features)
        
        # Error features
        error_features = self._extract_error_features(raw_data)
        features.extend(error_features)
        
        # Focus features
        focus_features = self._extract_focus_features(raw_data)
        features.extend(focus_features)
        
        return np.array(features).reshape(1, -1)
    
    def _extract_performance_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract performance-related features."""
        response_times = data.get("response_times", [])
        accuracy_scores = data.get("accuracy_scores", [])
        task_complexities = data.get("task_complexities", [])
        
        if not response_times:
            return [0.0] * 5
        
        avg_response_time = np.mean(response_times)
        response_time_std = np.std(response_times)
        avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.5
        complexity_performance_ratio = self._calculate_complexity_ratio(task_complexities, accuracy_scores)
        fatigue_score = self._calculate_fatigue_score(response_times)
        
        return [avg_response_time, response_time_std, avg_accuracy, complexity_performance_ratio, fatigue_score]
    
    def _extract_temporal_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract temporal pattern features."""
        response_times = data.get("response_times", [])
        
        if not response_times:
            return [0.0] * 2
        
        time_consistency = 1 / (np.std(response_times) + 0.001)
        learning_rate = self._calculate_learning_rate(response_times)
        
        return [time_consistency, learning_rate]
    
    def _extract_error_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract error pattern features."""
        error_patterns = data.get("error_patterns", {})
        hesitation_indicators = data.get("hesitation_indicators", [])
        
        total_errors = sum(error_patterns.values()) if error_patterns else 0
        error_diversity = len(error_patterns) if error_patterns else 0
        hesitation_frequency = len(hesitation_indicators)
        
        return [total_errors, error_diversity, hesitation_frequency]
    
    def _extract_focus_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract attention and focus features."""
        multitask_events = data.get("multitask_events", [])
        content_engagement = data.get("content_engagement", {})
        
        task_switches = len(multitask_events)
        engagement_variance = np.var(list(content_engagement.values())) if content_engagement else 0
        
        return [task_switches, engagement_variance]
    
    def _calculate_complexity_ratio(self, complexities: List[float], accuracies: List[float]) -> float:
        """Calculate performance vs complexity ratio."""
        if not complexities or not accuracies:
            return 0.5
        
        avg_complexity = np.mean(complexities)
        avg_accuracy = np.mean(accuracies)
        
        # Expected accuracy based on complexity
        expected_accuracy = 1.0 - (avg_complexity / 10.0)
        ratio = abs(avg_accuracy - expected_accuracy)
        
        return ratio
    
    def _calculate_learning_rate(self, response_times: List[float]) -> float:
        """Calculate learning rate from response time improvement."""
        if len(response_times) < 3:
            return 0.0
        
        x = np.arange(len(response_times))
        slope = np.polyfit(x, response_times, 1)[0]
        
        # Negative slope indicates improvement
        return max(0, -slope / np.mean(response_times))
    
    def _calculate_fatigue_score(self, response_times: List[float]) -> float:
        """Calculate fatigue based on response time degradation."""
        if len(response_times) < 4:
            return 0.0
        
        # Compare first and last halves
        first_half = response_times[:len(response_times)//2]
        last_half = response_times[len(response_times)//2:]
        
        avg_first = np.mean(first_half)
        avg_last = np.mean(last_half)
        
        # Fatigue = relative increase in response time
        fatigue = (avg_last - avg_first) / avg_first if avg_first > 0 else 0
        
        return max(0, fatigue)
    
    def _score_to_level(self, score: float) -> str:
        """Convert continuous score to categorical level."""
        if score < 0.3:
            return CognitiveLoadLevel.UNDERLOADED
        elif score <= 0.7:
            return CognitiveLoadLevel.OPTIMAL
        elif score <= 0.9:
            return CognitiveLoadLevel.OVERLOADED
        else:
            return CognitiveLoadLevel.CRITICAL
    
    def _calculate_confidence(self, features: np.ndarray, score: float) -> float:
        """Calculate prediction confidence."""
        non_zero_features = np.count_nonzero(features)
        feature_completeness = non_zero_features / features.shape[1]
        score_stability = 1 - abs(score - 0.5) * 2
        
        return (feature_completeness + score_stability) / 2
    
    def _generate_recommendations(self, load_level: str, score: float) -> List[str]:
        """Generate recommendations based on cognitive load level."""
        recommendations = []
        
        if load_level == CognitiveLoadLevel.CRITICAL:
            recommendations.extend([
                "Immediate break required",
                "Reduce task complexity significantly",
                "Switch to review activities"
            ])
        elif load_level == CognitiveLoadLevel.OVERLOADED:
            recommendations.extend([
                "Reduce content complexity",
                "Provide additional scaffolding",
                "Break tasks into smaller steps"
            ])
        elif load_level == CognitiveLoadLevel.UNDERLOADED:
            recommendations.extend([
                "Increase challenge level",
                "Add complexity",
                "Accelerate learning pace"
            ])
        else:  # OPTIMAL
            recommendations.extend([
                "Maintain current difficulty level",
                "Continue with progressive challenges"
            ])
        
        return recommendations
    
    def _extract_performance_indicators(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key performance indicators."""
        response_times = data.get("response_times", [])
        accuracy_scores = data.get("accuracy_scores", [])
        
        return {
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'current_accuracy': accuracy_scores[-1] if accuracy_scores else 0,
            'response_consistency': 1 / (np.std(response_times) + 0.001) if response_times else 0
        }
    
    def _prepare_features_for_trained_model(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Prepare features for trained model using expected feature format."""
        response_times = input_data.get("response_times", [])
        accuracy_scores = input_data.get("accuracy_scores", [])
        task_complexities = input_data.get("task_complexities", [])
        error_patterns = input_data.get("error_patterns", {})
        hesitation_indicators = input_data.get("hesitation_indicators", [])
        multitask_events = input_data.get("multitask_events", [])
        session_duration = input_data.get("session_duration", 600)
        
        # Calculate features in expected order
        avg_response_time = np.mean(response_times) if response_times else 0
        response_time_std = np.std(response_times) if response_times else 0
        avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.5
        
        # Calculate accuracy decline
        accuracy_decline = 0
        if len(accuracy_scores) > 2:
            x = np.arange(len(accuracy_scores))
            slope = np.polyfit(x, accuracy_scores, 1)[0]
            accuracy_decline = max(0, -slope)
        
        task_complexity = np.mean(task_complexities) if task_complexities else 0.5
        error_count = sum(error_patterns.values()) if error_patterns else 0
        critical_errors = error_patterns.get('critical', 0) if error_patterns else 0
        hesitation_count = len(hesitation_indicators)
        multitask_events_count = len(multitask_events)
        
        # Calculate fatigue indicators
        fatigue_indicators = 0
        if len(response_times) > 3:
            first_half = response_times[:len(response_times)//2]
            last_half = response_times[len(response_times)//2:]
            avg_first = np.mean(first_half)
            avg_last = np.mean(last_half)
            fatigue_indicators = (avg_last - avg_first) / avg_first if avg_first > 0 else 0
            fatigue_indicators = max(0, fatigue_indicators)
        
        # Assemble features in expected order
        features = [
            avg_response_time,
            response_time_std,
            avg_accuracy,
            accuracy_decline,
            task_complexity,
            error_count,
            critical_errors,
            hesitation_count,
            multitask_events_count,
            fatigue_indicators,
            session_duration
        ]
        
        return np.array(features).reshape(1, -1)
    
    def _analyze_cognitive_load_patterns(self, input_data: Dict[str, Any]) -> float:
        """Analyze cognitive load patterns for fallback prediction."""
        load_score = 0.5  # baseline
        
        # Response time factor
        response_times = input_data.get("response_times", [])
        if response_times:
            avg_response_time = np.mean(response_times)
            # Slower response times indicate higher load
            if avg_response_time > 5:
                load_score += 0.2
            elif avg_response_time > 3:
                load_score += 0.1
            elif avg_response_time < 1:
                load_score -= 0.1
        
        # Accuracy factor
        accuracy_scores = input_data.get("accuracy_scores", [])
        if accuracy_scores:
            avg_accuracy = np.mean(accuracy_scores)
            # Lower accuracy indicates higher load
            if avg_accuracy < 0.5:
                load_score += 0.3
            elif avg_accuracy < 0.7:
                load_score += 0.1
            elif avg_accuracy > 0.9:
                load_score -= 0.1
        
        # Error factor
        error_patterns = input_data.get("error_patterns", {})
        if error_patterns:
            total_errors = sum(error_patterns.values())
            if total_errors > 5:
                load_score += 0.2
            elif total_errors > 2:
                load_score += 0.1
        
        # Task complexity factor
        task_complexities = input_data.get("task_complexities", [])
        if task_complexities:
            avg_complexity = np.mean(task_complexities)
            # Adjust load based on complexity
            load_score += (avg_complexity / 10.0) * 0.3
        
        # Multitasking factor
        multitask_events = input_data.get("multitask_events", [])
        if multitask_events:
            if len(multitask_events) > 3:
                load_score += 0.15
            elif len(multitask_events) > 1:
                load_score += 0.05
        
        # Hesitation factor
        hesitation_indicators = input_data.get("hesitation_indicators", [])
        if hesitation_indicators:
            load_score += min(0.1, len(hesitation_indicators) * 0.02)
        
        return max(0.0, min(1.0, load_score)) 