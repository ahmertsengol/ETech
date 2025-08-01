"""
AI Models Package

Individual AI models for different aspects of learning psychology analysis.
Includes both original models and enhanced ensemble models.
"""

# Original models
from .attention_tracker import AttentionTracker, AttentionLevel
from .cognitive_load_assessor import CognitiveLoadAssessor, CognitiveLoadLevel
from .learning_style_detector import LearningStyleDetector, LearningStyle
from .adaptive_engine import AdaptiveEngine

# Enhanced ensemble models
try:
    from .enhanced_attention_tracker import EnhancedAttentionTracker
    ENHANCED_ATTENTION_AVAILABLE = True
except ImportError:
    ENHANCED_ATTENTION_AVAILABLE = False

try:
    from .enhanced_cognitive_load_assessor import EnhancedCognitiveLoadAssessor
    ENHANCED_COGNITIVE_LOAD_AVAILABLE = True
except ImportError:
    ENHANCED_COGNITIVE_LOAD_AVAILABLE = False

try:
    from .enhanced_learning_style_detector import EnhancedLearningStyleDetector
    ENHANCED_LEARNING_STYLE_AVAILABLE = True
except ImportError:
    ENHANCED_LEARNING_STYLE_AVAILABLE = False

# Base exports
__all__ = [
    "AttentionTracker",
    "AttentionLevel",
    "CognitiveLoadAssessor", 
    "CognitiveLoadLevel",
    "LearningStyleDetector",
    "LearningStyle",
    "AdaptiveEngine"
]

# Add enhanced models if available
if ENHANCED_ATTENTION_AVAILABLE:
    __all__.append("EnhancedAttentionTracker")

if ENHANCED_COGNITIVE_LOAD_AVAILABLE:
    __all__.append("EnhancedCognitiveLoadAssessor")

if ENHANCED_LEARNING_STYLE_AVAILABLE:
    __all__.append("EnhancedLearningStyleDetector")

# Enhanced models availability flags
__all__.extend([
    "ENHANCED_ATTENTION_AVAILABLE",
    "ENHANCED_COGNITIVE_LOAD_AVAILABLE", 
    "ENHANCED_LEARNING_STYLE_AVAILABLE"
]) 