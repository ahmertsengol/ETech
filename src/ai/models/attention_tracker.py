"""
Attention Tracker AI Model - Analyzes user attention patterns during learning.
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


class AttentionLevel:
    """Attention level constants."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"


class AttentionTracker(BaseAIModel):
    """
    Tracks and predicts user attention levels using behavioral data.
    
    Features analyzed:
    - Mouse movement patterns
    - Keyboard interaction frequency  
    - Scroll behavior
    - Content interactions
    - Time-based patterns
    """
    
    def __init__(self, version: str = "1.0.0"):
        super().__init__("attention_tracker", version)
        self.scaler = StandardScaler()
        self.attention_history: List[Dict[str, Any]] = []
        self.use_trained_model = False
        self.trained_components = None
        
        # Try to load trained model components
        self._load_trained_model()
    
    def _load_trained_model(self):
        """Load trained model components if available."""
        try:
            if is_model_available("attention_tracker"):
                self.trained_components = load_model_components("attention_tracker")
                if (self.trained_components['model'] is not None and 
                    self.trained_components['scaler'] is not None and
                    self.trained_components['encoder'] is not None):
                    self.use_trained_model = True
                    logger.info("Trained attention tracker model loaded successfully")
                else:
                    logger.warning("Some trained model components are missing")
            else:
                logger.info("No trained attention tracker model found, using fallback mode")
        except Exception as e:
            logger.error(f"Failed to load trained model: {str(e)}")
            self.use_trained_model = False
        
    def get_required_fields(self) -> List[str]:
        """Required input fields for attention analysis."""
        return [
            "mouse_movements",
            "keyboard_events", 
            "scroll_events",
            "content_interactions",
            "timestamp",
            "content_type",
            "content_difficulty"
        ]
    
    def train(self, training_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Train the attention prediction model."""
        self.status = ModelStatus.TRAINING
        
        try:
            # Prepare training data
            X = training_data.drop(['attention_level', 'user_id', 'session_id'], axis=1)
            y = training_data['attention_level']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            
            # Calculate training metrics
            train_predictions = self.model.predict(X_scaled)
            train_accuracy = accuracy_score(y, train_predictions)
            
            metrics = {
                'train_accuracy': train_accuracy,
                'feature_count': X.shape[1],
                'training_samples': len(X)
            }
            
            # Validation metrics if provided
            if validation_data is not None:
                X_val = validation_data.drop(['attention_level', 'user_id', 'session_id'], axis=1)
                y_val = validation_data['attention_level']
                X_val_scaled = self.scaler.transform(X_val)
                
                val_predictions = self.model.predict(X_val_scaled)
                val_accuracy = accuracy_score(y_val, val_predictions)
                metrics['val_accuracy'] = val_accuracy
            
            # Store training history
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'model_version': self.version
            }
            self.training_history.append(training_record)
            
            self.status = ModelStatus.TRAINED
            logger.info(f"Attention tracker trained successfully. Accuracy: {train_accuracy:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self.status = ModelStatus.ERROR
            raise
    
    def predict(self, input_data: Dict[str, Any], context: LearningContext) -> PredictionResult:
        """Predict user attention level based on behavioral data."""
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
        encoder = self.trained_components['encoder']
        
        prediction_proba = model.predict_proba(features_scaled)[0]
        predicted_class_encoded = model.predict(features_scaled)[0]
        predicted_class = encoder.inverse_transform([predicted_class_encoded])[0]
        confidence = max(prediction_proba)
        
        # Create prediction result
        result = PredictionResult(
            value=predicted_class,
            confidence=confidence,
            metadata={
                'probabilities': dict(zip(encoder.classes_, prediction_proba)),
                'feature_vector_size': features.shape[1],
                'model_type': 'trained',
                'context': {
                    'user_id': context.user_id,
                    'session_id': context.session_id,
                    'content_id': context.content_id
                },
                'recommendations': self._generate_recommendations(predicted_class, confidence)
            },
            timestamp=datetime.now().isoformat(),
            model_version=self.version
        )
        
        # Store in history
        self.attention_history.append({
            'timestamp': result.timestamp,
            'attention_level': predicted_class,
            'confidence': confidence,
            'user_id': context.user_id,
            'session_id': context.session_id
        })
        
        self.status = ModelStatus.TRAINED
        return result
    
    def _predict_fallback(self, input_data: Dict[str, Any], context: LearningContext) -> PredictionResult:
        """Fallback rule-based prediction when trained model is not available."""
        # Analyze behavioral patterns
        attention_score = self._analyze_attention_patterns(input_data)
        
        # Determine attention level
        if attention_score > 0.7:
            predicted_class = AttentionLevel.HIGH
            confidence = 0.75
        elif attention_score > 0.4:
            predicted_class = AttentionLevel.MEDIUM
            confidence = 0.70
        elif attention_score > 0.2:
            predicted_class = AttentionLevel.LOW
            confidence = 0.65
        else:
            predicted_class = AttentionLevel.CRITICAL
            confidence = 0.60
        
        # Create prediction result
        result = PredictionResult(
            value=predicted_class,
            confidence=confidence,
            metadata={
                'attention_score': attention_score,
                'model_type': 'fallback',
                'context': {
                    'user_id': context.user_id,
                    'session_id': context.session_id,
                    'content_id': context.content_id
                },
                'recommendations': self._generate_recommendations(predicted_class, confidence)
            },
            timestamp=datetime.now().isoformat(),
            model_version=self.version
        )
        
        # Store in history
        self.attention_history.append({
            'timestamp': result.timestamp,
            'attention_level': predicted_class,
            'confidence': confidence,
            'user_id': context.user_id,
            'session_id': context.session_id
        })
        
        self.status = ModelStatus.TRAINED
        return result
    
    def preprocess_data(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """Extract behavioral features from raw data."""
        features = []
        
        # Mouse movement features
        mouse_data = raw_data.get("mouse_movements", [])
        mouse_features = self._extract_mouse_features(mouse_data)
        features.extend(mouse_features)
        
        # Keyboard interaction features
        keyboard_data = raw_data.get("keyboard_events", [])
        keyboard_features = self._extract_keyboard_features(keyboard_data)
        features.extend(keyboard_features)
        
        # Scroll behavior features
        scroll_data = raw_data.get("scroll_events", [])
        scroll_features = self._extract_scroll_features(scroll_data)
        features.extend(scroll_features)
        
        # Content interaction features
        interaction_data = raw_data.get("content_interactions", [])
        interaction_features = self._extract_interaction_features(interaction_data)
        features.extend(interaction_features)
        
        # Contextual features
        context_features = self._extract_contextual_features(raw_data)
        features.extend(context_features)
        
        return np.array(features).reshape(1, -1)
    
    def _extract_mouse_features(self, mouse_movements: List[Dict]) -> List[float]:
        """Extract mouse movement features."""
        if not mouse_movements:
            return [0.0] * 4
        
        velocities = []
        for i in range(1, len(mouse_movements)):
            prev = mouse_movements[i-1]
            curr = mouse_movements[i]
            
            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            dt = curr['timestamp'] - prev['timestamp']
            
            if dt > 0:
                velocity = np.sqrt(dx**2 + dy**2) / dt
                velocities.append(velocity)
        
        avg_velocity = np.mean(velocities) if velocities else 0
        velocity_variance = np.var(velocities) if velocities else 0
        movement_frequency = len(mouse_movements) / 30  # normalize to 30 seconds
        idle_time = self._calculate_idle_time(mouse_movements)
        
        return [avg_velocity, velocity_variance, movement_frequency, idle_time]
    
    def _extract_keyboard_features(self, keyboard_events: List[Dict]) -> List[float]:
        """Extract keyboard interaction features."""
        if not keyboard_events:
            return [0.0] * 3
        
        intervals = []
        for i in range(1, len(keyboard_events)):
            interval = keyboard_events[i]['timestamp'] - keyboard_events[i-1]['timestamp']
            intervals.append(interval)
        
        typing_speed = 1 / np.mean(intervals) if intervals else 0
        typing_consistency = 1 / (np.std(intervals) + 0.001) if intervals else 0
        key_diversity = len(set(event.get('key', '') for event in keyboard_events))
        
        return [typing_speed, typing_consistency, key_diversity]
    
    def _extract_scroll_features(self, scroll_events: List[Dict]) -> List[float]:
        """Extract scroll behavior features."""
        if not scroll_events:
            return [0.0] * 3
        
        scroll_speeds = []
        for i in range(1, len(scroll_events)):
            prev = scroll_events[i-1]
            curr = scroll_events[i]
            
            dy = abs(curr.get('scroll_y', 0) - prev.get('scroll_y', 0))
            dt = curr['timestamp'] - prev['timestamp']
            
            if dt > 0:
                speed = dy / dt
                scroll_speeds.append(speed)
        
        avg_scroll_speed = np.mean(scroll_speeds) if scroll_speeds else 0
        scroll_consistency = 1 / (np.std(scroll_speeds) + 0.001) if scroll_speeds else 0
        reading_pattern = self._analyze_reading_pattern(scroll_events)
        
        return [avg_scroll_speed, scroll_consistency, reading_pattern]
    
    def _extract_interaction_features(self, interactions: List[Dict]) -> List[float]:
        """Extract content interaction features.""" 
        if not interactions:
            return [0.0] * 3
        
        click_count = len([i for i in interactions if i.get('type') == 'click'])
        hover_duration = sum(i.get('duration', 0) for i in interactions if i.get('type') == 'hover')
        focus_changes = len([i for i in interactions if i.get('type') == 'focus'])
        
        return [click_count, hover_duration, focus_changes]
    
    def _extract_contextual_features(self, raw_data: Dict[str, Any]) -> List[float]:
        """Extract contextual features."""
        content_type_map = {'text': 1.0, 'video': 2.0, 'interactive': 3.0, 'quiz': 4.0}
        content_type = content_type_map.get(raw_data.get('content_type', 'text'), 1.0)
        content_difficulty = float(raw_data.get('content_difficulty', 5.0)) / 10.0
        time_factor = self._encode_time_of_day(raw_data.get('timestamp'))
        
        return [content_type, content_difficulty, time_factor]
    
    def _calculate_idle_time(self, movements: List[Dict]) -> float:
        """Calculate total idle time from movement data."""
        if len(movements) < 2:
            return 0.0
        
        idle_time = 0
        threshold = 5.0  # seconds
        
        for i in range(1, len(movements)):
            time_diff = movements[i]['timestamp'] - movements[i-1]['timestamp']
            if time_diff > threshold:
                idle_time += time_diff
        
        return idle_time
    
    def _analyze_reading_pattern(self, scroll_events: List[Dict]) -> float:
        """Analyze scroll pattern for reading quality score."""
        if len(scroll_events) < 3:
            return 0.5
        
        speeds = []
        pauses = 0
        
        for i in range(1, len(scroll_events)):
            prev = scroll_events[i-1]
            curr = scroll_events[i]
            
            dy = abs(curr.get('scroll_y', 0) - prev.get('scroll_y', 0))
            dt = curr['timestamp'] - prev['timestamp']
            
            if dt > 0:
                speed = dy / dt
                speeds.append(speed)
                
                if speed < 10:  # Pause threshold
                    pauses += 1
        
        if not speeds:
            return 0.5
        
        # Good reading: moderate speed with pauses
        avg_speed = np.mean(speeds)
        optimal = 50 <= avg_speed <= 200  # pixels per second
        pause_ratio = pauses / len(speeds)
        
        return (0.7 if optimal else 0.3) + (0.3 * pause_ratio)
    
    def _encode_time_of_day(self, timestamp: Optional[str]) -> float:
        """Encode time considering attention patterns."""
        if not timestamp:
            return 0.5
        
        try:
            dt = datetime.fromisoformat(timestamp)
            hour = dt.hour
            # Peak attention: 9-11 AM and 1-3 PM
            if 9 <= hour <= 11 or 13 <= hour <= 15:
                return 1.0
            elif 6 <= hour <= 8 or 16 <= hour <= 18:
                return 0.8
            elif 19 <= hour <= 22:
                return 0.6
            else:
                return 0.3
        except:
            return 0.5

    def _generate_recommendations(self, attention_level: str, confidence: float) -> List[str]:
        """Generate recommendations based on attention level."""
        recommendations = []
        
        if attention_level == AttentionLevel.CRITICAL:
            recommendations.extend([
                "Suggest immediate break",
                "Switch to lighter content",
                "Enable gamification elements",
                "Reduce content complexity"
            ])
        elif attention_level == AttentionLevel.LOW:
            recommendations.extend([
                "Introduce interactive elements",
                "Break content into smaller chunks",
                "Add visual stimuli",
                "Suggest brief exercise break"
            ])
        elif attention_level == AttentionLevel.MEDIUM:
            recommendations.extend([
                "Maintain current pace",
                "Add occasional interactive check-ins",
                "Monitor for decline"
            ])
        else:  # HIGH
            recommendations.extend([
                "Optimize current approach",
                "Consider increasing difficulty",
                "Maintain engagement momentum"
            ])
        
        if confidence < 0.7:
            recommendations.append("Gather more behavioral data for better accuracy")
        
        return recommendations
    
    def _prepare_features_for_trained_model(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Prepare features for trained model using expected feature format."""
        # Expected features for trained model (from training data)
        features = []
        
        # Mouse features
        mouse_data = input_data.get("mouse_movements", [])
        if mouse_data:
            velocities = []
            for i in range(1, len(mouse_data)):
                prev = mouse_data[i-1]
                curr = mouse_data[i]
                dx = curr.get('x', 0) - prev.get('x', 0)
                dy = curr.get('y', 0) - prev.get('y', 0)
                dt = curr.get('timestamp', 0) - prev.get('timestamp', 0)
                if dt > 0:
                    velocity = np.sqrt(dx**2 + dy**2) / dt
                    velocities.append(velocity)
            
            avg_mouse_speed = np.mean(velocities) if velocities else 0
            mouse_movement_variance = np.var(velocities) if velocities else 0
            click_frequency = len([m for m in mouse_data if m.get('type') == 'click']) / max(1, len(mouse_data))
            idle_periods = sum(1 for i in range(1, len(mouse_data)) 
                              if mouse_data[i].get('timestamp', 0) - mouse_data[i-1].get('timestamp', 0) > 5)
        else:
            avg_mouse_speed = 0
            mouse_movement_variance = 0
            click_frequency = 0
            idle_periods = 0
        
        # Keyboard features
        keyboard_data = input_data.get("keyboard_events", [])
        if keyboard_data:
            intervals = []
            for i in range(1, len(keyboard_data)):
                interval = keyboard_data[i].get('timestamp', 0) - keyboard_data[i-1].get('timestamp', 0)
                if interval > 0:
                    intervals.append(interval)
            
            typing_speed = 1 / np.mean(intervals) if intervals else 0
            typing_consistency = 1 / (np.std(intervals) + 0.001) if intervals else 0
            backspace_frequency = len([k for k in keyboard_data if k.get('key') == 'Backspace']) / max(1, len(keyboard_data))
        else:
            typing_speed = 0
            typing_consistency = 0
            backspace_frequency = 0
        
        # Scroll features
        scroll_data = input_data.get("scroll_events", [])
        if scroll_data:
            scroll_speeds = []
            direction_changes = 0
            for i in range(1, len(scroll_data)):
                prev = scroll_data[i-1]
                curr = scroll_data[i]
                dy = curr.get('scroll_y', 0) - prev.get('scroll_y', 0)
                dt = curr.get('timestamp', 0) - prev.get('timestamp', 0)
                if dt > 0:
                    speed = abs(dy) / dt
                    scroll_speeds.append(speed)
                    if i > 1:
                        prev_dy = scroll_data[i-1].get('scroll_y', 0) - scroll_data[i-2].get('scroll_y', 0)
                        if (dy > 0) != (prev_dy > 0):
                            direction_changes += 1
            
            scroll_speed = np.mean(scroll_speeds) if scroll_speeds else 0
            scroll_direction_changes = direction_changes
        else:
            scroll_speed = 0
            scroll_direction_changes = 0
        
        # Content interaction features
        interaction_data = input_data.get("content_interactions", [])
        if interaction_data:
            total_duration = sum(i.get('duration', 0) for i in interaction_data)
            content_interaction_depth = total_duration / max(1, len(interaction_data))
            tab_switches = len([i for i in interaction_data if i.get('type') == 'tab_switch'])
            focus_duration = sum(i.get('duration', 0) for i in interaction_data if i.get('type') == 'focus')
        else:
            content_interaction_depth = 0
            tab_switches = 0
            focus_duration = 0
        
        # Context features
        session_time_of_day = self._encode_time_of_day(input_data.get('timestamp'))
        session_duration = input_data.get('session_duration', 600)  # default 10 minutes
        
        # Assemble features in expected order
        features = [
            avg_mouse_speed,
            mouse_movement_variance,
            click_frequency,
            idle_periods,
            typing_speed,
            typing_consistency,
            backspace_frequency,
            scroll_speed,
            scroll_direction_changes,
            content_interaction_depth,
            tab_switches,
            focus_duration,
            session_time_of_day * 24,  # Convert to hour
            session_duration
        ]
        
        return np.array(features).reshape(1, -1)
    
    def _analyze_attention_patterns(self, input_data: Dict[str, Any]) -> float:
        """Analyze attention patterns for fallback prediction."""
        attention_score = 0.5  # baseline
        
        # Mouse activity factor
        mouse_data = input_data.get("mouse_movements", [])
        if mouse_data:
            velocities = []
            for i in range(1, len(mouse_data)):
                prev = mouse_data[i-1]
                curr = mouse_data[i]
                dx = curr.get('x', 0) - prev.get('x', 0)
                dy = curr.get('y', 0) - prev.get('y', 0)
                dt = curr.get('timestamp', 0) - prev.get('timestamp', 0)
                if dt > 0:
                    velocity = np.sqrt(dx**2 + dy**2) / dt
                    velocities.append(velocity)
            
            if velocities:
                avg_velocity = np.mean(velocities)
                # Optimal mouse speed suggests good attention
                if 100 <= avg_velocity <= 300:
                    attention_score += 0.15
                elif avg_velocity > 500:  # Too fast, might be distracted
                    attention_score -= 0.1
        
        # Keyboard activity factor
        keyboard_data = input_data.get("keyboard_events", [])
        if keyboard_data:
            intervals = []
            for i in range(1, len(keyboard_data)):
                interval = keyboard_data[i].get('timestamp', 0) - keyboard_data[i-1].get('timestamp', 0)
                if interval > 0:
                    intervals.append(interval)
            
            if intervals:
                typing_rhythm = 1 / np.mean(intervals)
                # Steady typing suggests attention
                if 0.5 <= typing_rhythm <= 2.0:
                    attention_score += 0.15
        
        # Scroll behavior factor
        scroll_data = input_data.get("scroll_events", [])
        if scroll_data:
            scroll_speeds = []
            for i in range(1, len(scroll_data)):
                prev = scroll_data[i-1]
                curr = scroll_data[i]
                dy = abs(curr.get('scroll_y', 0) - prev.get('scroll_y', 0))
                dt = curr.get('timestamp', 0) - prev.get('timestamp', 0)
                if dt > 0:
                    speed = dy / dt
                    scroll_speeds.append(speed)
            
            if scroll_speeds:
                avg_scroll_speed = np.mean(scroll_speeds)
                # Moderate scroll speed suggests reading
                if 50 <= avg_scroll_speed <= 200:
                    attention_score += 0.1
                elif avg_scroll_speed > 500:  # Too fast, might be skimming
                    attention_score -= 0.15
        
        # Content interaction factor
        interaction_data = input_data.get("content_interactions", [])
        if interaction_data:
            interaction_count = len(interaction_data)
            if interaction_count > 0:
                attention_score += min(0.15, interaction_count * 0.05)
        
        # Time of day factor
        timestamp = input_data.get('timestamp')
        if timestamp:
            time_factor = self._encode_time_of_day(timestamp)
            attention_score *= time_factor
        
        return max(0.0, min(1.0, attention_score)) 