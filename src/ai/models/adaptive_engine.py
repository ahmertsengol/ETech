"""
Adaptive Learning Engine - Orchestrates all AI models to personalize learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging

from ..core.base_model import BaseAIModel, PredictionResult, LearningContext, ModelStatus
from .attention_tracker import AttentionTracker, AttentionLevel
from .cognitive_load_assessor import CognitiveLoadAssessor, CognitiveLoadLevel
from .learning_style_detector import LearningStyleDetector, LearningStyle

logger = logging.getLogger(__name__)


@dataclass
class AdaptationDecision:
    """Represents an adaptation decision."""
    decision_type: str
    parameters: Dict[str, Any]
    reason: str
    priority: int  # 1-5, 5 being highest
    timestamp: str


class AdaptationType:
    """Types of adaptations."""
    CONTENT_DIFFICULTY = "content_difficulty"
    CONTENT_FORMAT = "content_format"
    PACING = "pacing"
    BREAK_SUGGESTION = "break_suggestion"
    MOTIVATION = "motivation"


class AdaptiveEngine(BaseAIModel):
    """
    Central engine that combines attention, cognitive load, and learning style
    to make real-time adaptations to the learning experience.
    """
    
    def __init__(self, version: str = "1.0.0"):
        super().__init__("adaptive_engine", version)
        
        # Initialize sub-models
        self.attention_tracker = AttentionTracker()
        self.cognitive_load_assessor = CognitiveLoadAssessor()
        self.learning_style_detector = LearningStyleDetector()
        
        # Adaptation state
        self.adaptation_history: List[AdaptationDecision] = []
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Weights for combining model outputs
        self.adaptation_weights = {
            'attention': 0.35,
            'cognitive_load': 0.40,
            'learning_style': 0.25
        }
        
    def get_required_fields(self) -> List[str]:
        """Required input fields - combines all sub-model fields."""
        attention_fields = self.attention_tracker.get_required_fields()
        cognitive_fields = self.cognitive_load_assessor.get_required_fields()
        style_fields = self.learning_style_detector.get_required_fields()
        
        all_fields = list(set(attention_fields + cognitive_fields + style_fields))
        all_fields.extend([
            "current_content_metadata",
            "user_preferences",
            "session_goals"
        ])
        
        return all_fields
    
    def train(self, training_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Train all sub-models."""
        self.status = ModelStatus.TRAINING
        
        try:
            metrics = {}
            
            # Train attention tracker
            if 'attention_level' in training_data.columns:
                attention_data = training_data[training_data['attention_level'].notna()]
                if not attention_data.empty:
                    attention_metrics = self.attention_tracker.train(attention_data)
                    metrics['attention'] = attention_metrics
            
            # Train cognitive load assessor
            if 'cognitive_load_score' in training_data.columns:
                cognitive_data = training_data[training_data['cognitive_load_score'].notna()]
                if not cognitive_data.empty:
                    cognitive_metrics = self.cognitive_load_assessor.train(cognitive_data)
                    metrics['cognitive_load'] = cognitive_metrics
            
            # Train learning style detector
            if 'learning_style' in training_data.columns:
                style_data = training_data[training_data['learning_style'].notna()]
                if not style_data.empty:
                    style_metrics = self.learning_style_detector.train(style_data)
                    metrics['learning_style'] = style_metrics
            
            self.status = ModelStatus.TRAINED
            
            overall_metrics = {
                'models_trained': len(metrics),
                'total_training_samples': len(training_data),
                'sub_model_metrics': metrics
            }
            
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'metrics': overall_metrics,
                'model_version': self.version
            }
            self.training_history.append(training_record)
            
            logger.info(f"Adaptive engine trained with {len(metrics)} sub-models")
            return overall_metrics
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self.status = ModelStatus.ERROR
            raise
    
    def predict(self, input_data: Dict[str, Any], context: LearningContext) -> PredictionResult:
        """Generate comprehensive learning adaptations."""
        self.status = ModelStatus.PREDICTING
        
        try:
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data")
            
            # Get predictions from all sub-models
            sub_predictions = self._get_sub_model_predictions(input_data, context)
            
            # Analyze current learning state
            learning_state = self._analyze_learning_state(sub_predictions, input_data)
            
            # Generate adaptations
            adaptations = self._generate_adaptations(learning_state, input_data, context)
            
            # Create result
            result = PredictionResult(
                value=adaptations,
                confidence=self._calculate_overall_confidence(sub_predictions),
                metadata={
                    'sub_model_predictions': sub_predictions,
                    'learning_state': learning_state,
                    'adaptation_count': len(adaptations),
                    'context': {
                        'user_id': context.user_id,
                        'session_id': context.session_id,
                        'content_id': context.content_id
                    },
                    'effectiveness_score': learning_state.get('effectiveness_score', 0.5),
                    'intervention_urgency': learning_state.get('intervention_urgency', 1)
                },
                timestamp=datetime.now().isoformat(),
                model_version=self.version
            )
            
            # Update user profile
            self._update_user_profile(context.user_id, sub_predictions, adaptations)
            
            self.status = ModelStatus.TRAINED
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            self.status = ModelStatus.ERROR
            raise
    
    def preprocess_data(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess data - delegates to sub-models."""
        return np.array([]).reshape(1, -1)
    
    def _get_sub_model_predictions(self, input_data: Dict[str, Any], context: LearningContext) -> Dict[str, Any]:
        """Get predictions from all sub-models."""
        predictions = {}
        
        # Attention prediction
        try:
            if self.attention_tracker.status == ModelStatus.TRAINED:
                attention_result = self.attention_tracker.predict(input_data, context)
                predictions['attention'] = {
                    'level': attention_result.value,
                    'confidence': attention_result.confidence,
                    'recommendations': attention_result.metadata.get('recommendations', [])
                }
        except Exception as e:
            logger.warning(f"Attention prediction failed: {str(e)}")
            predictions['attention'] = {
                'level': AttentionLevel.MEDIUM, 
                'confidence': 0.5, 
                'recommendations': []
            }
        
        # Cognitive load prediction
        try:
            if self.cognitive_load_assessor.status == ModelStatus.TRAINED:
                cognitive_result = self.cognitive_load_assessor.predict(input_data, context)
                predictions['cognitive_load'] = {
                    'level': cognitive_result.value,
                    'score': cognitive_result.metadata.get('load_score', 0.5),
                    'confidence': cognitive_result.confidence,
                    'recommendations': cognitive_result.metadata.get('recommendations', [])
                }
        except Exception as e:
            logger.warning(f"Cognitive load prediction failed: {str(e)}")
            predictions['cognitive_load'] = {
                'level': CognitiveLoadLevel.OPTIMAL, 
                'score': 0.5, 
                'confidence': 0.5, 
                'recommendations': []
            }
        
        # Learning style prediction
        try:
            if self.learning_style_detector.status == ModelStatus.TRAINED:
                style_result = self.learning_style_detector.predict(input_data, context)
                predictions['learning_style'] = {
                    'style': style_result.value,
                    'confidence': style_result.confidence,
                    'probabilities': style_result.metadata.get('style_probabilities', {}),
                    'recommendations': style_result.metadata.get('recommendations', [])
                }
        except Exception as e:
            logger.warning(f"Learning style prediction failed: {str(e)}")
            predictions['learning_style'] = {
                'style': LearningStyle.MULTIMODAL, 
                'confidence': 0.5, 
                'probabilities': {}, 
                'recommendations': []
            }
        
        return predictions
    
    def _analyze_learning_state(self, sub_predictions: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall learning state."""
        attention_level = sub_predictions.get('attention', {}).get('level', AttentionLevel.MEDIUM)
        cognitive_level = sub_predictions.get('cognitive_load', {}).get('level', CognitiveLoadLevel.OPTIMAL)
        learning_style = sub_predictions.get('learning_style', {}).get('style', LearningStyle.MULTIMODAL)
        
        # Calculate effectiveness
        effectiveness_score = self._calculate_effectiveness_score(sub_predictions)
        
        # Determine urgency
        urgency = self._calculate_intervention_urgency(attention_level, cognitive_level)
        
        return {
            'attention_level': attention_level,
            'cognitive_load_level': cognitive_level,
            'learning_style': learning_style,
            'effectiveness_score': effectiveness_score,
            'intervention_urgency': urgency,
            'optimal_state': self._is_optimal_learning_state(attention_level, cognitive_level)
        }
    
    def _generate_adaptations(self, learning_state: Dict[str, Any], input_data: Dict[str, Any], context: LearningContext) -> List[AdaptationDecision]:
        """Generate specific adaptations based on learning state."""
        adaptations = []
        
        attention_level = learning_state['attention_level']
        cognitive_level = learning_state['cognitive_load_level']
        learning_style = learning_state['learning_style']
        
        # Critical attention interventions
        if attention_level == AttentionLevel.CRITICAL:
            adaptations.append(AdaptationDecision(
                decision_type=AdaptationType.BREAK_SUGGESTION,
                parameters={'duration': 10, 'type': 'mandatory'},
                reason="Critical attention level detected",
                priority=5,
                timestamp=datetime.now().isoformat()
            ))
        
        # Cognitive load adaptations
        if cognitive_level == CognitiveLoadLevel.OVERLOADED:
            adaptations.append(AdaptationDecision(
                decision_type=AdaptationType.CONTENT_DIFFICULTY,
                parameters={'adjustment': -0.3, 'add_scaffolding': True},
                reason="High cognitive load detected",
                priority=4,
                timestamp=datetime.now().isoformat()
            ))
        elif cognitive_level == CognitiveLoadLevel.UNDERLOADED:
            adaptations.append(AdaptationDecision(
                decision_type=AdaptationType.CONTENT_DIFFICULTY,
                parameters={'adjustment': 0.2, 'add_challenge': True},
                reason="Low cognitive load detected",
                priority=3,
                timestamp=datetime.now().isoformat()
            ))
        
        # Learning style adaptations
        style_adaptation = self._generate_style_adaptation(learning_style, input_data)
        if style_adaptation:
            adaptations.append(style_adaptation)
        
        # Motivation adaptations
        if learning_state['effectiveness_score'] < 0.4:
            adaptations.append(AdaptationDecision(
                decision_type=AdaptationType.MOTIVATION,
                parameters={'gamification': True, 'positive_reinforcement': True},
                reason="Low learning effectiveness",
                priority=2,
                timestamp=datetime.now().isoformat()
            ))
        
        # Sort by priority
        adaptations.sort(key=lambda x: x.priority, reverse=True)
        
        # Store in history
        self.adaptation_history.extend(adaptations)
        
        return adaptations
    
    def _calculate_effectiveness_score(self, sub_predictions: Dict[str, Any]) -> float:
        """Calculate overall learning effectiveness score."""
        attention_map = {
            AttentionLevel.HIGH: 1.0,
            AttentionLevel.MEDIUM: 0.7,
            AttentionLevel.LOW: 0.4,
            AttentionLevel.CRITICAL: 0.1
        }
        
        cognitive_map = {
            CognitiveLoadLevel.OPTIMAL: 1.0,
            CognitiveLoadLevel.UNDERLOADED: 0.6,
            CognitiveLoadLevel.OVERLOADED: 0.3,
            CognitiveLoadLevel.CRITICAL: 0.1
        }
        
        attention_score = attention_map.get(
            sub_predictions.get('attention', {}).get('level', AttentionLevel.MEDIUM), 0.7
        )
        
        cognitive_score = cognitive_map.get(
            sub_predictions.get('cognitive_load', {}).get('level', CognitiveLoadLevel.OPTIMAL), 1.0
        )
        
        style_confidence = sub_predictions.get('learning_style', {}).get('confidence', 0.5)
        
        # Weighted combination
        effectiveness = (
            attention_score * self.adaptation_weights['attention'] +
            cognitive_score * self.adaptation_weights['cognitive_load'] +
            style_confidence * self.adaptation_weights['learning_style']
        )
        
        return effectiveness
    
    def _calculate_intervention_urgency(self, attention_level: str, cognitive_level: str) -> int:
        """Calculate urgency of intervention (1-5 scale)."""
        if attention_level == AttentionLevel.CRITICAL or cognitive_level == CognitiveLoadLevel.CRITICAL:
            return 5
        elif attention_level == AttentionLevel.LOW or cognitive_level == CognitiveLoadLevel.OVERLOADED:
            return 4
        elif cognitive_level == CognitiveLoadLevel.UNDERLOADED:
            return 2
        else:
            return 1
    
    def _is_optimal_learning_state(self, attention_level: str, cognitive_level: str) -> bool:
        """Check if current state is optimal for learning."""
        return (attention_level in [AttentionLevel.HIGH, AttentionLevel.MEDIUM] and 
                cognitive_level == CognitiveLoadLevel.OPTIMAL)
    
    def _generate_style_adaptation(self, learning_style: str, input_data: Dict[str, Any]) -> Optional[AdaptationDecision]:
        """Generate learning style specific adaptation."""
        current_format = input_data.get('current_content_metadata', {}).get('format', 'text')
        
        optimal_formats = {
            LearningStyle.VISUAL: ['diagram', 'infographic', 'chart'],
            LearningStyle.AUDITORY: ['audio', 'narration', 'podcast'],
            LearningStyle.READING: ['text', 'article', 'document'],
            LearningStyle.KINESTHETIC: ['interactive', 'simulation', 'exercise'],
            LearningStyle.MULTIMODAL: ['multimedia', 'mixed']
        }
        
        preferred_formats = optimal_formats.get(learning_style, ['mixed'])
        
        if current_format not in preferred_formats:
            return AdaptationDecision(
                decision_type=AdaptationType.CONTENT_FORMAT,
                parameters={
                    'new_format': preferred_formats[0],
                    'learning_style': learning_style
                },
                reason=f"Content format mismatch for {learning_style} learner",
                priority=3,
                timestamp=datetime.now().isoformat()
            )
        
        return None
    
    def _calculate_overall_confidence(self, sub_predictions: Dict[str, Any]) -> float:
        """Calculate overall confidence from sub-models."""
        confidences = []
        
        for model, prediction in sub_predictions.items():
            if 'confidence' in prediction:
                confidences.append(prediction['confidence'])
        
        return np.mean(confidences) if confidences else 0.5
    
    def _update_user_profile(self, user_id: str, sub_predictions: Dict[str, Any], adaptations: List[AdaptationDecision]) -> None:
        """Update user profile with latest data."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'created_at': datetime.now().isoformat(),
                'learning_preferences': {},
                'adaptation_count': 0,
                'effectiveness_history': []
            }
        
        profile = self.user_profiles[user_id]
        
        # Update learning style preference
        learning_style = sub_predictions.get('learning_style', {})
        if learning_style.get('style'):
            profile['learning_preferences']['style'] = learning_style['style']
            profile['learning_preferences']['style_confidence'] = learning_style.get('confidence', 0.5)
        
        # Update adaptation count
        profile['adaptation_count'] += len(adaptations)
        
        # Update effectiveness history
        effectiveness = self._calculate_effectiveness_score(sub_predictions)
        profile['effectiveness_history'].append({
            'timestamp': datetime.now().isoformat(),
            'score': effectiveness
        })
        
        # Keep only last 20 scores
        profile['effectiveness_history'] = profile['effectiveness_history'][-20:] 