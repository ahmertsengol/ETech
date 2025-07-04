"""
Learning Style Detector - Identifies user's learning preferences.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import logging

from ..core.base_model import BaseAIModel, PredictionResult, LearningContext, ModelStatus
from ..core.model_loader import load_model_components, is_model_available

logger = logging.getLogger(__name__)


class LearningStyle:
    """Learning style constants based on VARK model."""
    VISUAL = "visual"           
    AUDITORY = "auditory"       
    READING = "reading"         
    KINESTHETIC = "kinesthetic" 
    MULTIMODAL = "multimodal"   


class LearningStyleDetector(BaseAIModel):
    """
    Detects user's preferred learning style based on interaction patterns.
    
    Analyzes:
    - Content type preferences
    - Interaction patterns with different media
    - Performance correlation with content formats
    - Engagement metrics per content type
    """
    
    def __init__(self, version: str = "1.0.0"):
        super().__init__("learning_style_detector", version)
        self.scaler = StandardScaler()
        self.style_history: List[Dict[str, Any]] = []
        self.use_trained_model = False
        self.trained_components = None
        
        # Try to load trained model components
        self._load_trained_model()
    
    def _load_trained_model(self):
        """Load trained model components if available."""
        try:
            if is_model_available("learning_style_detector"):
                self.trained_components = load_model_components("learning_style_detector")
                if (self.trained_components['model'] is not None and 
                    self.trained_components['scaler'] is not None and
                    self.trained_components['encoder'] is not None):
                    self.use_trained_model = True
                    logger.info("Trained learning style detector model loaded successfully")
                else:
                    logger.warning("Some trained model components are missing")
            else:
                logger.info("No trained learning style detector model found, using fallback mode")
        except Exception as e:
            logger.error(f"Failed to load trained model: {str(e)}")
            self.use_trained_model = False
        
    def get_required_fields(self) -> List[str]:
        """Required input fields for learning style analysis."""
        return [
            "content_interactions",
            "time_spent_by_type",
            "performance_by_type",
            "content_preferences", 
            "engagement_metrics",
            "completion_rates",
            "replay_behaviors",
            "navigation_patterns",
            "timestamp"
        ]

    def train(self, training_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Train the learning style classification model."""
        self.status = ModelStatus.TRAINING
        
        try:
            # Prepare training data
            X = training_data.drop(['learning_style', 'user_id', 'session_id'], axis=1)
            y = training_data['learning_style']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Random Forest classifier
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=3,
                class_weight='balanced',
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            
            # Calculate training metrics
            train_predictions = self.model.predict(X_scaled)
            train_accuracy = accuracy_score(y, train_predictions)
            
            metrics = {
                'train_accuracy': train_accuracy,
                'feature_count': X.shape[1],
                'training_samples': len(X),
                'classes': list(self.model.classes_)
            }
            
            # Store training history
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'model_version': self.version
            }
            self.training_history.append(training_record)
            
            self.status = ModelStatus.TRAINED
            logger.info(f"Learning style detector trained successfully. Accuracy: {train_accuracy:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self.status = ModelStatus.ERROR
            raise
    
    def predict(self, input_data: Dict[str, Any], context: LearningContext) -> PredictionResult:
        """Predict user's learning style based on behavior patterns."""
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
        # Prepare features from input data (simplified approach)
        features = self._prepare_features_for_trained_model(input_data)
        
        # Scale features
        features_scaled = self.trained_components['scaler'].transform(features)
        
        # Make prediction
        model = self.trained_components['model']
        encoder = self.trained_components['encoder']
        
        prediction_proba = model.predict_proba(features_scaled)[0]
        predicted_style_encoded = model.predict(features_scaled)[0]
        predicted_style = encoder.inverse_transform([predicted_style_encoded])[0]
        confidence = max(prediction_proba)
        
        # Check for multimodal tendencies
        style_scores = dict(zip(encoder.classes_, prediction_proba))
        adjusted_style = self._check_multimodal(style_scores, predicted_style)
        
        # Create prediction result
        result = PredictionResult(
            value=adjusted_style,
            confidence=confidence,
            metadata={
                'style_probabilities': style_scores,
                'original_prediction': predicted_style,
                'feature_vector_size': features.shape[1],
                'model_type': 'trained',
                'context': {
                    'user_id': context.user_id,
                    'session_id': context.session_id,
                    'content_id': context.content_id
                },
                'recommendations': self._generate_style_recommendations(adjusted_style),
                'content_preferences': self._extract_content_preferences(input_data)
            },
            timestamp=datetime.now().isoformat(),
            model_version=self.version
        )
        
        # Store in history
        self.style_history.append({
            'timestamp': result.timestamp,
            'learning_style': adjusted_style,
            'confidence': confidence,
            'style_scores': style_scores,
            'user_id': context.user_id,
            'session_id': context.session_id
        })
        
        self.status = ModelStatus.TRAINED
        return result
    
    def _predict_fallback(self, input_data: Dict[str, Any], context: LearningContext) -> PredictionResult:
        """Fallback rule-based prediction when trained model is not available."""
        # Analyze learning style patterns
        style_scores = self._analyze_learning_style_patterns(input_data)
        
        # Determine predicted style
        predicted_style = max(style_scores, key=style_scores.get)
        confidence = style_scores[predicted_style]
        
        # Check for multimodal tendencies
        adjusted_style = self._check_multimodal(style_scores, predicted_style)
        
        # Create prediction result
        result = PredictionResult(
            value=adjusted_style,
            confidence=confidence,
            metadata={
                'style_probabilities': style_scores,
                'original_prediction': predicted_style,
                'model_type': 'fallback',
                'context': {
                    'user_id': context.user_id,
                    'session_id': context.session_id,
                    'content_id': context.content_id
                },
                'recommendations': self._generate_style_recommendations(adjusted_style),
                'content_preferences': self._extract_content_preferences(input_data)
            },
            timestamp=datetime.now().isoformat(),
            model_version=self.version
        )
        
        # Store in history
        self.style_history.append({
            'timestamp': result.timestamp,
            'learning_style': adjusted_style,
            'confidence': confidence,
            'style_scores': style_scores,
            'user_id': context.user_id,
            'session_id': context.session_id
        })
        
        self.status = ModelStatus.TRAINED
        return result
    
    def preprocess_data(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """Extract learning style features from behavioral data."""
        features = []
        
        # Content interaction patterns
        interaction_features = self._extract_interaction_features(raw_data)
        features.extend(interaction_features)
        
        # Time allocation patterns
        time_features = self._extract_time_features(raw_data)
        features.extend(time_features)
        
        # Performance correlation features
        performance_features = self._extract_performance_features(raw_data)
        features.extend(performance_features)
        
        # Engagement pattern features
        engagement_features = self._extract_engagement_features(raw_data)
        features.extend(engagement_features)
        
        return np.array(features).reshape(1, -1)
    
    def _extract_interaction_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract content interaction pattern features."""
        content_interactions = data.get("content_interactions", {})
        
        total_interactions = sum(content_interactions.values()) if content_interactions else 0
        
        if total_interactions == 0:
            return [0.0] * 4
        
        visual_ratio = content_interactions.get('visual', 0) / total_interactions
        auditory_ratio = content_interactions.get('auditory', 0) / total_interactions  
        text_ratio = content_interactions.get('text', 0) / total_interactions
        interactive_ratio = content_interactions.get('interactive', 0) / total_interactions
        
        return [visual_ratio, auditory_ratio, text_ratio, interactive_ratio]
    
    def _extract_time_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract time allocation pattern features."""
        time_spent = data.get("time_spent_by_type", {})
        
        total_time = sum(time_spent.values()) if time_spent else 0
        
        if total_time == 0:
            return [0.0] * 4
        
        visual_time_ratio = time_spent.get('visual', 0) / total_time
        auditory_time_ratio = time_spent.get('auditory', 0) / total_time
        text_time_ratio = time_spent.get('text', 0) / total_time
        interactive_time_ratio = time_spent.get('interactive', 0) / total_time
        
        return [visual_time_ratio, auditory_time_ratio, text_time_ratio, interactive_time_ratio]
    
    def _extract_performance_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract performance correlation features."""
        performance_by_type = data.get("performance_by_type", {})
        
        # Average performance by content type
        visual_performance = np.mean(performance_by_type.get('visual', [0.5]))
        auditory_performance = np.mean(performance_by_type.get('auditory', [0.5]))
        text_performance = np.mean(performance_by_type.get('text', [0.5]))
        interactive_performance = np.mean(performance_by_type.get('interactive', [0.5]))
        
        return [visual_performance, auditory_performance, text_performance, interactive_performance]
    
    def _extract_engagement_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract engagement pattern features."""
        engagement_metrics = data.get("engagement_metrics", {})
        completion_rates = data.get("completion_rates", {})
        replay_behaviors = data.get("replay_behaviors", {})
        
        # Engagement scores by type
        visual_engagement = engagement_metrics.get('visual', 0.5)
        auditory_engagement = engagement_metrics.get('auditory', 0.5)
        text_engagement = engagement_metrics.get('text', 0.5)
        interactive_engagement = engagement_metrics.get('interactive', 0.5)
        
        # Completion rate variance (indicates preference consistency)
        completion_variance = np.var(list(completion_rates.values())) if completion_rates else 0
        
        # Replay behavior (indicates strong preference)
        total_replays = sum(replay_behaviors.values()) if replay_behaviors else 0
        dominant_replay_type = max(replay_behaviors, key=replay_behaviors.get) if replay_behaviors else 'none'
        replay_concentration = max(replay_behaviors.values()) / max(1, total_replays)
        
        return [visual_engagement, auditory_engagement, text_engagement, 
                interactive_engagement, completion_variance, replay_concentration]
    
    def _check_multimodal(self, style_scores: Dict[str, float], predicted_style: str) -> str:
        """Check if user shows multimodal learning tendencies."""
        # Count styles with significant scores (>20%)
        significant_styles = [style for style, score in style_scores.items() if score > 0.2]
        
        # If multiple styles are significant and no single dominant style
        if len(significant_styles) >= 3 and max(style_scores.values()) < 0.6:
            return LearningStyle.MULTIMODAL
        
        return predicted_style
    
    def _generate_style_recommendations(self, learning_style: str) -> List[str]:
        """Generate recommendations based on learning style."""
        recommendations = []
        
        if learning_style == LearningStyle.VISUAL:
            recommendations.extend([
                "Use charts, diagrams, and infographics",
                "Provide visual summaries and mind maps",
                "Include color-coded information",
                "Use flowcharts for processes"
            ])
        elif learning_style == LearningStyle.AUDITORY:
            recommendations.extend([
                "Include audio explanations and narration",
                "Provide discussion forums and verbal feedback", 
                "Use music and sound effects for engagement",
                "Offer podcast-style content"
            ])
        elif learning_style == LearningStyle.READING:
            recommendations.extend([
                "Provide detailed text explanations",
                "Include reading lists and articles",
                "Use bullet points and structured text",
                "Offer note-taking capabilities"
            ])
        elif learning_style == LearningStyle.KINESTHETIC:
            recommendations.extend([
                "Include interactive simulations",
                "Provide hands-on exercises and labs",
                "Use drag-and-drop activities",
                "Offer real-world application examples"
            ])
        else:  # MULTIMODAL
            recommendations.extend([
                "Combine multiple content formats",
                "Offer choice in content delivery method",
                "Provide rich multimedia experiences",
                "Allow switching between formats"
            ])
        
        return recommendations
    
    def _extract_content_preferences(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract content type preferences from data."""
        time_spent = data.get("time_spent_by_type", {})
        performance = data.get("performance_by_type", {})
        engagement = data.get("engagement_metrics", {})
        
        preferences = {}
        
        for content_type in ['visual', 'auditory', 'text', 'interactive']:
            time_score = time_spent.get(content_type, 0)
            perf_score = np.mean(performance.get(content_type, [0]))
            eng_score = engagement.get(content_type, 0)
            
            # Weighted preference score
            preferences[content_type] = (time_score * 0.3 + perf_score * 0.4 + eng_score * 0.3)
        
        return preferences
    
    def _prepare_features_for_trained_model(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Prepare features for trained model using expected feature format."""
        # Simplified feature extraction for trained model
        content_interactions = input_data.get("content_interactions", {})
        time_spent = input_data.get("time_spent_by_type", {})
        performance = input_data.get("performance_by_type", {})
        engagement = input_data.get("engagement_metrics", {})
        completion_rates = input_data.get("completion_rates", {})
        replay_behaviors = input_data.get("replay_behaviors", {})
        
        # Calculate total interactions
        total_interactions = sum(content_interactions.values()) if content_interactions else 1
        
        # Calculate total time
        total_time = sum(time_spent.values()) if time_spent else 1
        
        # Feature vector in expected order
        features = [
            # Interaction ratios
            content_interactions.get('visual', 0),
            content_interactions.get('auditory', 0),
            content_interactions.get('text', 0),
            content_interactions.get('interactive', 0),
            
            # Time ratios
            time_spent.get('visual', 0) / total_time,
            time_spent.get('auditory', 0) / total_time,
            time_spent.get('text', 0) / total_time,
            time_spent.get('interactive', 0) / total_time,
            
            # Performance scores
            np.mean(performance.get('visual', [0.5])),
            np.mean(performance.get('auditory', [0.5])),
            np.mean(performance.get('text', [0.5])),
            np.mean(performance.get('interactive', [0.5])),
            
            # Engagement scores
            engagement.get('visual', 0.5),
            engagement.get('auditory', 0.5),
            engagement.get('text', 0.5),
            engagement.get('interactive', 0.5),
            
            # Completion variance
            np.var(list(completion_rates.values())) if completion_rates else 0,
            
            # Replay concentration
            max(replay_behaviors.values()) / max(1, sum(replay_behaviors.values())) if replay_behaviors else 0
        ]
        
        return np.array(features).reshape(1, -1)
    
    def _analyze_learning_style_patterns(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze learning style patterns for fallback prediction."""
        # Initialize scores
        style_scores = {
            LearningStyle.VISUAL: 0.2,
            LearningStyle.AUDITORY: 0.2,
            LearningStyle.READING: 0.2,
            LearningStyle.KINESTHETIC: 0.2,
            LearningStyle.MULTIMODAL: 0.2
        }
        
        # Time spent analysis
        time_spent = input_data.get("time_spent_by_type", {})
        if time_spent:
            total_time = sum(time_spent.values())
            if total_time > 0:
                for content_type, time in time_spent.items():
                    ratio = time / total_time
                    if content_type == 'visual':
                        style_scores[LearningStyle.VISUAL] += ratio * 0.3
                    elif content_type == 'auditory':
                        style_scores[LearningStyle.AUDITORY] += ratio * 0.3
                    elif content_type == 'text':
                        style_scores[LearningStyle.READING] += ratio * 0.3
                    elif content_type == 'interactive':
                        style_scores[LearningStyle.KINESTHETIC] += ratio * 0.3
        
        # Performance analysis
        performance = input_data.get("performance_by_type", {})
        if performance:
            for content_type, scores in performance.items():
                avg_performance = np.mean(scores) if scores else 0.5
                boost = (avg_performance - 0.5) * 0.4
                
                if content_type == 'visual':
                    style_scores[LearningStyle.VISUAL] += boost
                elif content_type == 'auditory':
                    style_scores[LearningStyle.AUDITORY] += boost
                elif content_type == 'text':
                    style_scores[LearningStyle.READING] += boost
                elif content_type == 'interactive':
                    style_scores[LearningStyle.KINESTHETIC] += boost
        
        # Engagement analysis
        engagement = input_data.get("engagement_metrics", {})
        if engagement:
            for content_type, eng_score in engagement.items():
                boost = (eng_score - 0.5) * 0.3
                
                if content_type == 'visual':
                    style_scores[LearningStyle.VISUAL] += boost
                elif content_type == 'auditory':
                    style_scores[LearningStyle.AUDITORY] += boost
                elif content_type == 'text':
                    style_scores[LearningStyle.READING] += boost
                elif content_type == 'interactive':
                    style_scores[LearningStyle.KINESTHETIC] += boost
        
        # Check for multimodal pattern
        high_scores = sum(1 for score in style_scores.values() if score > 0.4)
        if high_scores >= 3:
            style_scores[LearningStyle.MULTIMODAL] = max(style_scores.values()) + 0.1
        
        # Normalize scores
        total = sum(style_scores.values())
        if total > 0:
            style_scores = {k: v/total for k, v in style_scores.items()}
        
        return style_scores 