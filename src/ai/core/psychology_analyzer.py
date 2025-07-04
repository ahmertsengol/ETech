"""
Learning Psychology Analyzer - Main orchestrator for all AI models.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .base_model import LearningContext, model_registry
from ..models.adaptive_engine import AdaptiveEngine
from ..models.attention_tracker import AttentionTracker
from ..models.cognitive_load_assessor import CognitiveLoadAssessor
from ..models.learning_style_detector import LearningStyleDetector

logger = logging.getLogger(__name__)


class LearningPsychologyAnalyzer:
    """
    Main analyzer that coordinates all AI models to provide comprehensive
    learning psychology insights and real-time adaptations.
    """
    
    def __init__(self):
        # Initialize all models
        self.adaptive_engine = AdaptiveEngine()
        self.attention_tracker = AttentionTracker()
        self.cognitive_assessor = CognitiveLoadAssessor()
        self.style_detector = LearningStyleDetector()
        
        # Register models
        model_registry.register_model(self.adaptive_engine)
        model_registry.register_model(self.attention_tracker)
        model_registry.register_model(self.cognitive_assessor)
        model_registry.register_model(self.style_detector)
        
        logger.info("Learning Psychology Analyzer initialized with all models")
    
    def analyze_learning_session(
        self, 
        behavioral_data: Dict[str, Any], 
        user_id: str, 
        session_id: str,
        content_id: str
    ) -> Dict[str, Any]:
        """
        Analyze a learning session and provide comprehensive insights.
        
        Args:
            behavioral_data: All behavioral and performance data
            user_id: User identifier
            session_id: Session identifier
            content_id: Content identifier
            
        Returns:
            Comprehensive analysis with adaptations
        """
        try:
            # Create learning context
            context = LearningContext(
                user_id=user_id,
                session_id=session_id,
                content_id=content_id,
                timestamp=datetime.now().isoformat()
            )
            
            # Get comprehensive analysis from adaptive engine
            result = self.adaptive_engine.predict(behavioral_data, context)
            
            # Structure the response
            analysis = {
                'timestamp': result.timestamp,
                'user_id': user_id,
                'session_id': session_id,
                'content_id': content_id,
                
                # Main results
                'adaptations': result.value,
                'confidence': result.confidence,
                
                # Detailed insights
                'learning_state': result.metadata.get('learning_state', {}),
                'sub_model_predictions': result.metadata.get('sub_model_predictions', {}),
                'effectiveness_score': result.metadata.get('effectiveness_score', 0.5),
                'intervention_urgency': result.metadata.get('intervention_urgency', 1),
                
                # System info
                'model_version': result.model_version,
                'analysis_quality': self._assess_analysis_quality(result)
            }
            
            logger.info(f"Analysis completed for user {user_id}, session {session_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return self._get_fallback_analysis(user_id, session_id, content_id)
    
    def get_individual_insights(
        self, 
        behavioral_data: Dict[str, Any], 
        context: LearningContext,
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get insights from individual models.
        
        Args:
            behavioral_data: Behavioral and performance data
            context: Learning context
            models: List of model names to query (None for all)
            
        Returns:
            Individual model insights
        """
        insights = {}
        
        # Default to all models if none specified
        if models is None:
            models = ['attention', 'cognitive_load', 'learning_style']
        
        # Attention analysis
        if 'attention' in models:
            try:
                attention_result = self.attention_tracker.predict(behavioral_data, context)
                insights['attention'] = {
                    'level': attention_result.value,
                    'confidence': attention_result.confidence,
                    'recommendations': attention_result.metadata.get('recommendations', []),
                    'model_info': self.attention_tracker.get_model_info()
                }
            except Exception as e:
                logger.warning(f"Attention analysis failed: {str(e)}")
                insights['attention'] = {'error': str(e)}
        
        # Cognitive load analysis
        if 'cognitive_load' in models:
            try:
                cognitive_result = self.cognitive_assessor.predict(behavioral_data, context)
                insights['cognitive_load'] = {
                    'level': cognitive_result.value,
                    'score': cognitive_result.metadata.get('load_score', 0.5),
                    'confidence': cognitive_result.confidence,
                    'recommendations': cognitive_result.metadata.get('recommendations', []),
                    'performance_indicators': cognitive_result.metadata.get('performance_indicators', {}),
                    'model_info': self.cognitive_assessor.get_model_info()
                }
            except Exception as e:
                logger.warning(f"Cognitive load analysis failed: {str(e)}")
                insights['cognitive_load'] = {'error': str(e)}
        
        # Learning style analysis
        if 'learning_style' in models:
            try:
                style_result = self.style_detector.predict(behavioral_data, context)
                insights['learning_style'] = {
                    'style': style_result.value,
                    'confidence': style_result.confidence,
                    'probabilities': style_result.metadata.get('style_probabilities', {}),
                    'recommendations': style_result.metadata.get('recommendations', []),
                    'content_preferences': style_result.metadata.get('content_preferences', {}),
                    'model_info': self.style_detector.get_model_info()
                }
            except Exception as e:
                logger.warning(f"Learning style analysis failed: {str(e)}")
                insights['learning_style'] = {'error': str(e)}
        
        return insights
    
    def train_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train all models with provided data.
        
        Args:
            training_data: Dictionary with data for each model
            
        Returns:
            Training results for all models
        """
        training_results = {}
        
        try:
            # Train adaptive engine (which trains all sub-models)
            if 'adaptive_engine' in training_data:
                engine_results = self.adaptive_engine.train(training_data['adaptive_engine'])
                training_results['adaptive_engine'] = engine_results
            
            # Individual model training if separate data provided
            if 'attention' in training_data:
                attention_results = self.attention_tracker.train(training_data['attention'])
                training_results['attention'] = attention_results
            
            if 'cognitive_load' in training_data:
                cognitive_results = self.cognitive_assessor.train(training_data['cognitive_load'])
                training_results['cognitive_load'] = cognitive_results
            
            if 'learning_style' in training_data:
                style_results = self.style_detector.train(training_data['learning_style'])
                training_results['learning_style'] = style_results
            
            logger.info("All models trained successfully")
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all models and system health."""
        return {
            'models': model_registry.get_models_status(),
            'adaptive_engine': self.adaptive_engine.get_model_info(),
            'total_adaptations': len(self.adaptive_engine.adaptation_history),
            'user_profiles': len(self.adaptive_engine.user_profiles),
            'system_health': self._assess_system_health(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user's learning profile."""
        return self.adaptive_engine.user_profiles.get(user_id)
    
    def get_adaptation_history(self, user_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get adaptation history, optionally filtered by user."""
        history = self.adaptive_engine.adaptation_history[-limit:]
        
        if user_id:
            # Note: Would need to track user_id in adaptations for this to work
            # For now, return all history
            pass
        
        return [
            {
                'decision_type': adaptation.decision_type,
                'parameters': adaptation.parameters,
                'reason': adaptation.reason,
                'priority': adaptation.priority,
                'timestamp': adaptation.timestamp
            }
            for adaptation in history
        ]
    
    def _assess_analysis_quality(self, result) -> Dict[str, Any]:
        """Assess the quality of the analysis."""
        sub_predictions = result.metadata.get('sub_model_predictions', {})
        
        # Count available predictions
        available_models = len(sub_predictions)
        total_models = 3  # attention, cognitive_load, learning_style
        
        # Average confidence
        confidences = []
        for prediction in sub_predictions.values():
            if 'confidence' in prediction:
                confidences.append(prediction['confidence'])
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Quality score
        completeness_score = available_models / total_models
        confidence_score = avg_confidence
        quality_score = (completeness_score + confidence_score) / 2
        
        return {
            'quality_score': quality_score,
            'available_models': available_models,
            'total_models': total_models,
            'average_confidence': avg_confidence,
            'completeness': completeness_score
        }
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health."""
        models_status = model_registry.get_models_status()
        
        # Count trained models
        trained_models = sum(1 for status in models_status.values() 
                           if status.get('status') == 'trained')
        total_models = len(models_status)
        
        # System health score
        health_score = trained_models / total_models if total_models > 0 else 0
        
        return {
            'health_score': health_score,
            'trained_models': trained_models,
            'total_models': total_models,
            'status': 'healthy' if health_score >= 0.8 else 'degraded' if health_score >= 0.5 else 'critical'
        }
    
    def _get_fallback_analysis(self, user_id: str, session_id: str, content_id: str) -> Dict[str, Any]:
        """Provide fallback analysis when main analysis fails."""
        return {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'session_id': session_id,
            'content_id': content_id,
            'adaptations': [],
            'confidence': 0.1,
            'learning_state': {
                'attention_level': 'medium',
                'cognitive_load_level': 'optimal',
                'learning_style': 'multimodal',
                'effectiveness_score': 0.5,
                'intervention_urgency': 1
            },
            'error': 'Analysis failed, using fallback',
            'analysis_quality': {
                'quality_score': 0.1,
                'available_models': 0,
                'total_models': 3
            }
        } 