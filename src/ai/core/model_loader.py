"""
Model Loader Utilities
Loads trained models and preprocessing objects for inference.
"""

import os
import pickle
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelLoader:
    """Utility class for loading trained models and preprocessing objects."""
    
    def __init__(self, models_dir: str = "models/"):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.loaded_scalers = {}
        self.loaded_encoders = {}
        self.model_metadata = {}
        
    def load_model(self, model_name: str) -> Optional[Any]:
        """Load a trained model by name."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        model_path = os.path.join(self.models_dir, f"{model_name}_model.pkl")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self.loaded_models[model_name] = model
            logger.info(f"Successfully loaded model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None
    
    def load_scaler(self, scaler_name: str) -> Optional[Any]:
        """Load a preprocessing scaler by name."""
        if scaler_name in self.loaded_scalers:
            return self.loaded_scalers[scaler_name]
        
        scaler_path = os.path.join(self.models_dir, f"{scaler_name}_scaler.pkl")
        
        if not os.path.exists(scaler_path):
            logger.warning(f"Scaler file not found: {scaler_path}")
            return None
        
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            self.loaded_scalers[scaler_name] = scaler
            logger.info(f"Successfully loaded scaler: {scaler_name}")
            return scaler
            
        except Exception as e:
            logger.error(f"Error loading scaler {scaler_name}: {str(e)}")
            return None
    
    def load_encoder(self, encoder_name: str) -> Optional[Any]:
        """Load a label encoder by name."""
        if encoder_name in self.loaded_encoders:
            return self.loaded_encoders[encoder_name]
        
        encoder_path = os.path.join(self.models_dir, f"{encoder_name}_encoder.pkl")
        
        if not os.path.exists(encoder_path):
            logger.warning(f"Encoder file not found: {encoder_path}")
            return None
        
        try:
            with open(encoder_path, 'rb') as f:
                encoder = pickle.load(f)
            
            self.loaded_encoders[encoder_name] = encoder
            logger.info(f"Successfully loaded encoder: {encoder_name}")
            return encoder
            
        except Exception as e:
            logger.error(f"Error loading encoder {encoder_name}: {str(e)}")
            return None
    
    def load_training_metrics(self) -> Optional[Dict]:
        """Load training metrics."""
        metrics_path = os.path.join(self.models_dir, "training_metrics.pkl")
        
        if not os.path.exists(metrics_path):
            logger.warning(f"Training metrics file not found: {metrics_path}")
            return None
        
        try:
            with open(metrics_path, 'rb') as f:
                metrics = pickle.load(f)
            
            self.model_metadata['training_metrics'] = metrics
            logger.info("Successfully loaded training metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error loading training metrics: {str(e)}")
            return None
    
    def load_all_for_model(self, model_name: str) -> Dict[str, Any]:
        """Load model, scaler, and encoder for a specific model."""
        components = {
            'model': self.load_model(model_name),
            'scaler': self.load_scaler(model_name),
            'encoder': self.load_encoder(model_name)
        }
        
        return components
    
    def is_model_trained(self, model_name: str) -> bool:
        """Check if a model has been trained and saved."""
        model_path = os.path.join(self.models_dir, f"{model_name}_model.pkl")
        return os.path.exists(model_path)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a trained model."""
        if not self.is_model_trained(model_name):
            return {"trained": False, "error": "Model not found"}
        
        model_path = os.path.join(self.models_dir, f"{model_name}_model.pkl")
        
        info = {
            "trained": True,
            "model_path": model_path,
            "file_size": os.path.getsize(model_path),
            "last_modified": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
        }
        
        # Add training metrics if available
        if not self.model_metadata.get('training_metrics'):
            self.load_training_metrics()
        
        if self.model_metadata.get('training_metrics') and model_name in self.model_metadata['training_metrics']:
            info['training_metrics'] = self.model_metadata['training_metrics'][model_name]
        
        return info
    
    def get_all_models_status(self) -> Dict[str, Dict]:
        """Get status of all available models."""
        model_names = ['attention_tracker', 'cognitive_load_assessor', 'learning_style_detector']
        
        status = {}
        for model_name in model_names:
            status[model_name] = self.get_model_info(model_name)
        
        return status
    
    def clear_cache(self):
        """Clear all loaded models from memory."""
        self.loaded_models.clear()
        self.loaded_scalers.clear()
        self.loaded_encoders.clear()
        self.model_metadata.clear()
        logger.info("Model cache cleared")


# Global model loader instance
_global_loader = None


def get_model_loader() -> ModelLoader:
    """Get the global model loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = ModelLoader()
    return _global_loader


def load_model_components(model_name: str) -> Dict[str, Any]:
    """Convenience function to load all components for a model."""
    loader = get_model_loader()
    return loader.load_all_for_model(model_name)


def is_model_available(model_name: str) -> bool:
    """Check if a model is available for use."""
    loader = get_model_loader()
    return loader.is_model_trained(model_name) 