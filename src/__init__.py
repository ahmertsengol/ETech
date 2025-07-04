"""
AI Learning Psychology Analyzer Package

A comprehensive AI system for analyzing learning behavior and providing
personalized educational adaptations.
"""

__version__ = "1.0.0"
__author__ = "AI Learning Psychology Team"
__email__ = "team@learningpsychology.ai"

# Main exports
from .ai.core.psychology_analyzer import LearningPsychologyAnalyzer
from .ai.models.attention_tracker import AttentionTracker, AttentionLevel
from .ai.models.cognitive_load_assessor import CognitiveLoadAssessor, CognitiveLoadLevel
from .ai.models.learning_style_detector import LearningStyleDetector, LearningStyle
from .ai.models.adaptive_engine import AdaptiveEngine

__all__ = [
    "LearningPsychologyAnalyzer",
    "AttentionTracker", 
    "AttentionLevel",
    "CognitiveLoadAssessor",
    "CognitiveLoadLevel", 
    "LearningStyleDetector",
    "LearningStyle",
    "AdaptiveEngine"
] 