"""
🚀 Enhanced Model Registry - Advanced Model Management
====================================================

Manages both original and enhanced ensemble models with:
✓ Automatic model discovery and registration
✓ Performance-based model selection
✓ Fallback to original models when enhanced unavailable
✓ Model comparison and benchmarking
✓ Dynamic model switching based on performance
✓ Comprehensive model metadata tracking
"""

import logging
import warnings
from typing import Dict, Any, List, Optional, Union, Type
from datetime import datetime
from pathlib import Path
import json

from .base_model import BaseAIModel, ModelRegistry, ModelStatus

# Import available models with error handling
logger = logging.getLogger(__name__)

class ModelCapability:
    """Model capability constants."""
    ENSEMBLE = "ensemble"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    CLASS_IMBALANCE_HANDLING = "class_imbalance_handling"
    PROBABILITY_CALIBRATION = "probability_calibration"
    MODEL_INTERPRETATION = "model_interpretation"
    CROSS_VALIDATION = "cross_validation"
    DYNAMIC_WEIGHTING = "dynamic_weighting"

class ModelPerformanceTracker:
    """Track model performance over time."""
    
    def __init__(self):
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
    def record_performance(self, model_name: str, metrics: Dict[str, float], 
                          timestamp: Optional[str] = None):
        """Record performance metrics for a model."""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
            
        record = {
            'timestamp': timestamp or datetime.now().isoformat(),
            'metrics': metrics
        }
        
        self.performance_history[model_name].append(record)
        
        # Keep only last 100 records
        if len(self.performance_history[model_name]) > 100:
            self.performance_history[model_name] = self.performance_history[model_name][-100:]
    
    def get_latest_performance(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get latest performance record for a model."""
        if model_name in self.performance_history and self.performance_history[model_name]:
            return self.performance_history[model_name][-1]
        return None
    
    def get_average_performance(self, model_name: str, metric: str) -> Optional[float]:
        """Get average performance for a specific metric."""
        if model_name not in self.performance_history:
            return None
            
        values = [
            record['metrics'].get(metric, 0) 
            for record in self.performance_history[model_name] 
            if metric in record['metrics']
        ]
        
        return sum(values) / len(values) if values else None
    
    def compare_models(self, metric: str) -> Dict[str, float]:
        """Compare all models on a specific metric."""
        comparison = {}
        for model_name in self.performance_history:
            avg_performance = self.get_average_performance(model_name, metric)
            if avg_performance is not None:
                comparison[model_name] = avg_performance
        return comparison

class EnhancedModelRegistry(ModelRegistry):
    """
    🚀 Enhanced Model Registry with Advanced Management
    =================================================
    
    Extends the base model registry with:
    - Support for both original and enhanced models
    - Performance-based model selection
    - Automatic fallback mechanisms
    - Model capability tracking
    - Performance monitoring and comparison
    """
    
    def __init__(self):
        super().__init__()
        self.enhanced_models: Dict[str, BaseAIModel] = {}
        self.model_capabilities: Dict[str, List[str]] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.performance_tracker = ModelPerformanceTracker()
        self.auto_discovery_enabled = True
        
        # Performance thresholds for model selection
        self.performance_thresholds = {
            'attention_tracker': 0.75,  # 75% accuracy threshold
            'cognitive_load_assessor': 0.65,  # 65% accuracy threshold
            'learning_style_detector': 0.55   # 55% accuracy threshold
        }
        
        # Auto-discover and register available models
        if self.auto_discovery_enabled:
            self._discover_and_register_models()
    
    def _discover_and_register_models(self):
        """Automatically discover and register available models."""
        logger.info("Discovering and registering available models...")
        
        # Original models
        original_models = {
            'attention_tracker': self._try_import_original_attention,
            'cognitive_load_assessor': self._try_import_original_cognitive_load,
            'learning_style_detector': self._try_import_original_learning_style
        }
        
        # Enhanced models
        enhanced_models = {
            'enhanced_attention_tracker': self._try_import_enhanced_attention,
            'enhanced_cognitive_load_assessor': self._try_import_enhanced_cognitive_load,
            'enhanced_learning_style_detector': self._try_import_enhanced_learning_style
        }
        
        # Register original models
        for name, import_func in original_models.items():
            model_class = import_func()
            if model_class:
                try:
                    model_instance = model_class()
                    self.register_model(model_instance)
                    self._register_capabilities(name, ['basic'])
                    logger.info(f"Registered original model: {name}")
                except Exception as e:
                    logger.warning(f"Failed to register original model {name}: {e}")
        
        # Register enhanced models
        for name, import_func in enhanced_models.items():
            model_class = import_func()
            if model_class:
                try:
                    model_instance = model_class()
                    self.register_enhanced_model(model_instance)
                    
                    # Register enhanced capabilities
                    capabilities = [
                        ModelCapability.ENSEMBLE,
                        ModelCapability.HYPERPARAMETER_OPTIMIZATION,
                        ModelCapability.CLASS_IMBALANCE_HANDLING,
                        ModelCapability.CROSS_VALIDATION
                    ]
                    
                    # Check for additional capabilities
                    if hasattr(model_instance, 'use_probability_calibration'):
                        capabilities.append(ModelCapability.PROBABILITY_CALIBRATION)
                    if hasattr(model_instance, 'model_interpretation'):
                        capabilities.append(ModelCapability.MODEL_INTERPRETATION)
                    if hasattr(model_instance, 'dynamic_ensemble'):
                        capabilities.append(ModelCapability.DYNAMIC_WEIGHTING)
                    
                    self._register_capabilities(name, capabilities)
                    logger.info(f"Registered enhanced model: {name}")
                except Exception as e:
                    logger.warning(f"Failed to register enhanced model {name}: {e}")
    
    def _try_import_original_attention(self):
        """Try to import original attention tracker."""
        try:
            from ..models.attention_tracker import AttentionTracker
            return AttentionTracker
        except ImportError:
            return None
    
    def _try_import_original_cognitive_load(self):
        """Try to import original cognitive load assessor."""
        try:
            from ..models.cognitive_load_assessor import CognitiveLoadAssessor
            return CognitiveLoadAssessor
        except ImportError:
            return None
    
    def _try_import_original_learning_style(self):
        """Try to import original learning style detector."""
        try:
            from ..models.learning_style_detector import LearningStyleDetector
            return LearningStyleDetector
        except ImportError:
            return None
    
    def _try_import_enhanced_attention(self):
        """Try to import enhanced attention tracker."""
        try:
            from ..models.enhanced_attention_tracker import EnhancedAttentionTracker
            return EnhancedAttentionTracker
        except ImportError:
            return None
    
    def _try_import_enhanced_cognitive_load(self):
        """Try to import enhanced cognitive load assessor."""
        try:
            from ..models.enhanced_cognitive_load_assessor import EnhancedCognitiveLoadAssessor
            return EnhancedCognitiveLoadAssessor
        except ImportError:
            return None
    
    def _try_import_enhanced_learning_style(self):
        """Try to import enhanced learning style detector."""
        try:
            from ..models.enhanced_learning_style_detector import EnhancedLearningStyleDetector
            return EnhancedLearningStyleDetector
        except ImportError:
            return None
    
    def register_enhanced_model(self, model: BaseAIModel) -> None:
        """Register an enhanced model."""
        self.enhanced_models[model.model_name] = model
        self.models[model.model_name] = model  # Also register in base registry
        
        # Store metadata
        self.model_metadata[model.model_name] = {
            'type': 'enhanced',
            'version': model.version,
            'registered_at': datetime.now().isoformat(),
            'status': model.status.value
        }
        
        logger.info(f"Enhanced model {model.model_name} registered")
    
    def _register_capabilities(self, model_name: str, capabilities: List[str]) -> None:
        """Register model capabilities."""
        self.model_capabilities[model_name] = capabilities
    
    def get_best_model(self, model_type: str, capability_requirements: Optional[List[str]] = None) -> Optional[BaseAIModel]:
        """
        Get the best available model for a given type and capability requirements.
        
        Selection criteria:
        1. Enhanced models are preferred over original models
        2. Model must meet capability requirements
        3. Model must meet performance thresholds
        4. Model must be in TRAINED status
        """
        candidates = []
        
        # Find all models of the requested type
        for model_name, model in self.models.items():
            # Check if model matches type (exact match or starts with enhanced_)
            if (model_name == model_type or 
                (model_name.startswith('enhanced_') and model_name.endswith(model_type))):
                
                # Check capability requirements
                if capability_requirements:
                    model_caps = self.model_capabilities.get(model_name, [])
                    if not all(req in model_caps for req in capability_requirements):
                        continue
                
                # Check if model is trained
                if model.status != ModelStatus.TRAINED:
                    continue
                
                # Check performance threshold
                threshold = self.performance_thresholds.get(model_type, 0.5)
                avg_performance = self.performance_tracker.get_average_performance(model_name, 'accuracy')
                if avg_performance is not None and avg_performance < threshold:
                    continue
                
                candidates.append((model_name, model))
        
        if not candidates:
            logger.warning(f"No suitable model found for type: {model_type}")
            return None
        
        # Prefer enhanced models
        enhanced_candidates = [(name, model) for name, model in candidates if name.startswith('enhanced_')]
        if enhanced_candidates:
            # Select best performing enhanced model
            best_enhanced = self._select_best_performing(enhanced_candidates)
            if best_enhanced:
                logger.info(f"Selected enhanced model: {best_enhanced[0]}")
                return best_enhanced[1]
        
        # Fallback to original models
        original_candidates = [(name, model) for name, model in candidates if not name.startswith('enhanced_')]
        if original_candidates:
            best_original = self._select_best_performing(original_candidates)
            if best_original:
                logger.info(f"Selected original model: {best_original[0]} (enhanced not available)")
                return best_original[1]
        
        # Return first available as last resort
        logger.warning(f"Using first available model for {model_type}")
        return candidates[0][1]
    
    def _select_best_performing(self, candidates: List[tuple]) -> Optional[tuple]:
        """Select best performing model from candidates."""
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Compare by average accuracy
        best_candidate = None
        best_performance = 0
        
        for name, model in candidates:
            avg_performance = self.performance_tracker.get_average_performance(name, 'accuracy')
            if avg_performance is None:  # No performance data available
                avg_performance = 0.5  # Default neutral score
            
            if avg_performance > best_performance:
                best_performance = avg_performance
                best_candidate = (name, model)
        
        return best_candidate or candidates[0]
    
    def get_model_with_fallback(self, preferred_model: str, fallback_model: str) -> Optional[BaseAIModel]:
        """Get preferred model with fallback option."""
        model = self.get_model(preferred_model)
        if model and model.status == ModelStatus.TRAINED:
            return model
        
        logger.warning(f"Preferred model {preferred_model} not available, using fallback: {fallback_model}")
        return self.get_model(fallback_model)
    
    def compare_model_performance(self, metric: str = 'accuracy') -> Dict[str, Any]:
        """Compare performance of all registered models."""
        comparison = self.performance_tracker.compare_models(metric)
        
        # Group by model type
        grouped_comparison = {}
        for model_name, performance in comparison.items():
            # Extract base model type
            base_type = model_name.replace('enhanced_', '')
            if base_type not in grouped_comparison:
                grouped_comparison[base_type] = {}
            
            model_variant = 'enhanced' if model_name.startswith('enhanced_') else 'original'
            grouped_comparison[base_type][model_variant] = performance
        
        # Calculate improvements
        improvements = {}
        for base_type, variants in grouped_comparison.items():
            if 'enhanced' in variants and 'original' in variants:
                improvement = variants['enhanced'] - variants['original']
                improvement_pct = (improvement / variants['original'] * 100) if variants['original'] > 0 else 0
                improvements[base_type] = {
                    'absolute_improvement': improvement,
                    'percentage_improvement': improvement_pct,
                    'enhanced_performance': variants['enhanced'],
                    'original_performance': variants['original']
                }
        
        return {
            'raw_comparison': comparison,
            'grouped_comparison': grouped_comparison,
            'improvements': improvements
        }
    
    def get_enhanced_models_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all enhanced models."""
        status = {}
        
        for model_name, model in self.enhanced_models.items():
            capabilities = self.model_capabilities.get(model_name, [])
            metadata = self.model_metadata.get(model_name, {})
            latest_performance = self.performance_tracker.get_latest_performance(model_name)
            
            status[model_name] = {
                'status': model.status.value,
                'version': model.version,
                'capabilities': capabilities,
                'metadata': metadata,
                'latest_performance': latest_performance,
                'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
            }
        
        return status
    
    def record_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """Record performance metrics for a model."""
        self.performance_tracker.record_performance(model_name, metrics)
        logger.info(f"Recorded performance for {model_name}: {metrics}")
    
    def get_model_recommendations(self, task_type: str, performance_requirements: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Get model recommendations based on task type and performance requirements."""
        recommendations = {
            'primary_recommendation': None,
            'alternatives': [],
            'reasoning': [],
            'performance_comparison': {}
        }
        
        # Get all models for this task type
        task_models = [
            (name, model) for name, model in self.models.items()
            if task_type in name or name.endswith(task_type)
        ]
        
        if not task_models:
            recommendations['reasoning'].append(f"No models found for task type: {task_type}")
            return recommendations
        
        # Check performance requirements
        suitable_models = []
        for name, model in task_models:
            if performance_requirements:
                meets_requirements = True
                for metric, min_value in performance_requirements.items():
                    avg_performance = self.performance_tracker.get_average_performance(name, metric)
                    if avg_performance is None or avg_performance < min_value:
                        meets_requirements = False
                        break
                
                if meets_requirements:
                    suitable_models.append((name, model))
            else:
                suitable_models.append((name, model))
        
        if not suitable_models:
            recommendations['reasoning'].append("No models meet the performance requirements")
            recommendations['alternatives'] = [name for name, _ in task_models]
            return recommendations
        
        # Select primary recommendation (prefer enhanced)
        enhanced_models = [(name, model) for name, model in suitable_models if name.startswith('enhanced_')]
        original_models = [(name, model) for name, model in suitable_models if not name.startswith('enhanced_')]
        
        if enhanced_models:
            primary = self._select_best_performing(enhanced_models)
            if primary:
                recommendations['primary_recommendation'] = primary[0]
                recommendations['reasoning'].append(f"Enhanced model {primary[0]} provides superior performance")
        
        if not recommendations['primary_recommendation'] and original_models:
            primary = self._select_best_performing(original_models)
            if primary:
                recommendations['primary_recommendation'] = primary[0]
                recommendations['reasoning'].append(f"Original model {primary[0]} selected (enhanced not available)")
        
        # Add alternatives
        recommendations['alternatives'] = [
            name for name, _ in suitable_models 
            if name != recommendations['primary_recommendation']
        ]
        
        # Add performance comparison
        recommendations['performance_comparison'] = self.compare_model_performance()
        
        return recommendations
    
    def save_registry_state(self, filepath: str) -> bool:
        """Save registry state to file."""
        try:
            state = {
                'model_capabilities': self.model_capabilities,
                'model_metadata': self.model_metadata,
                'performance_history': self.performance_tracker.performance_history,
                'performance_thresholds': self.performance_thresholds,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Registry state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save registry state: {e}")
            return False
    
    def load_registry_state(self, filepath: str) -> bool:
        """Load registry state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.model_capabilities = state.get('model_capabilities', {})
            self.model_metadata = state.get('model_metadata', {})
            self.performance_tracker.performance_history = state.get('performance_history', {})
            self.performance_thresholds = state.get('performance_thresholds', self.performance_thresholds)
            
            logger.info(f"Registry state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load registry state: {e}")
            return False

# Global enhanced model registry instance
enhanced_model_registry = EnhancedModelRegistry()