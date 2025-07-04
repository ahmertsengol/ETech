"""
Base model class for Learning Psychology AI components.
Implements common functionality following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model training and prediction status."""
    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    PREDICTING = "predicting"
    ERROR = "error"


@dataclass
class PredictionResult:
    """Standardized prediction result structure."""
    value: Any
    confidence: float
    metadata: Dict[str, Any]
    timestamp: str
    model_version: str


@dataclass
class LearningContext:
    """Context information for learning session."""
    user_id: str
    session_id: str
    content_id: str
    timestamp: str
    additional_context: Optional[Dict[str, Any]] = None


class BaseAIModel(ABC):
    """
    Abstract base class for all AI models in the learning psychology system.
    Implements common functionality and enforces interface contracts.
    """
    
    def __init__(self, model_name: str, version: str = "1.0.0"):
        self.model_name = model_name
        self.version = version
        self.status = ModelStatus.UNTRAINED
        self.model = None
        self.training_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Return list of required input fields for this model."""
        pass
    
    @abstractmethod
    def train(self, training_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Train the model with provided data."""
        pass
    
    @abstractmethod
    def predict(self, input_data: Dict[str, Any], context: LearningContext) -> PredictionResult:
        """Make prediction based on input data and context."""
        pass
    
    @abstractmethod
    def preprocess_data(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess raw data into feature vector."""
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate that input data contains required fields."""
        required_fields = self.get_required_fields()
        missing_fields = [field for field in required_fields if field not in input_data]
        
        if missing_fields:
            logger.warning(f"Missing required fields for {self.model_name}: {missing_fields}")
            return False
        
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'model_name': self.model_name,
            'version': self.version,
            'status': self.status.value,
            'training_history_count': len(self.training_history),
            'last_training': self.training_history[-1]['timestamp'] if self.training_history else None
        }
    
    def save_model(self, filepath: str) -> bool:
        """Save model to file."""
        try:
            # Implementation would depend on model type
            logger.info(f"Model {self.model_name} saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model {self.model_name}: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load model from file."""
        try:
            # Implementation would depend on model type
            logger.info(f"Model {self.model_name} loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            return False


class ModelRegistry:
    """Registry to manage all AI models in the system."""
    
    def __init__(self):
        self.models: Dict[str, BaseAIModel] = {}
    
    def register_model(self, model: BaseAIModel) -> None:
        """Register a model with the registry."""
        self.models[model.model_name] = model
        logger.info(f"Model {model.model_name} registered")
    
    def get_model(self, model_name: str) -> Optional[BaseAIModel]:
        """Get a model by name."""
        return self.models.get(model_name)
    
    def get_models_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered models."""
        return {name: model.get_model_info() for name, model in self.models.items()}
    
    def get_trained_models(self) -> List[BaseAIModel]:
        """Get list of trained models."""
        return [model for model in self.models.values() if model.status == ModelStatus.TRAINED]


# Global model registry instance
model_registry = ModelRegistry() 