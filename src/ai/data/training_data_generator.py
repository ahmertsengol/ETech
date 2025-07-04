"""
Training Data Generator for Learning Psychology AI Models
Generates realistic synthetic training data for all models.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json


class TrainingDataGenerator:
    """Generates realistic training data for all AI models."""
    
    def __init__(self, num_users: int = 1000, sessions_per_user: int = 5):
        self.num_users = num_users
        self.sessions_per_user = sessions_per_user
        self.user_profiles = self._generate_user_profiles()
        
    def _generate_user_profiles(self) -> Dict[str, Dict]:
        """Generate diverse user profiles."""
        profiles = {}
        
        learning_styles = ['visual', 'auditory', 'reading', 'kinesthetic', 'multimodal']
        attention_baselines = ['high', 'medium', 'low']
        cognitive_capacities = ['high', 'medium', 'low']
        
        for i in range(self.num_users):
            user_id = f"user_{i:04d}"
            profiles[user_id] = {
                'learning_style': random.choice(learning_styles),
                'attention_baseline': random.choice(attention_baselines),
                'cognitive_capacity': random.choice(cognitive_capacities),
                'experience_level': random.randint(1, 10),
                'motivation_level': random.uniform(0.3, 1.0),
                'fatigue_resistance': random.uniform(0.4, 1.0)
            }
        
        return profiles
    
    def generate_attention_training_data(self) -> pd.DataFrame:
        """Generate training data for attention tracker."""
        data = []
        
        for user_id, profile in self.user_profiles.items():
            for session in range(self.sessions_per_user):
                session_id = f"{user_id}_session_{session}"
                
                # Generate attention data based on user profile
                attention_level = self._determine_attention_level(profile, session)
                
                # Generate behavioral indicators
                mouse_data = self._generate_mouse_data(attention_level)
                keyboard_data = self._generate_keyboard_data(attention_level)
                scroll_data = self._generate_scroll_data(attention_level)
                interaction_data = self._generate_interaction_data(attention_level)
                
                data.append({
                    'user_id': user_id,
                    'session_id': session_id,
                    'attention_level': attention_level,
                    'avg_mouse_speed': mouse_data['avg_speed'],
                    'mouse_movement_variance': mouse_data['variance'],
                    'click_frequency': mouse_data['click_freq'],
                    'idle_periods': mouse_data['idle_periods'],
                    'typing_speed': keyboard_data['speed'],
                    'typing_consistency': keyboard_data['consistency'],
                    'backspace_frequency': keyboard_data['backspace_freq'],
                    'scroll_speed': scroll_data['speed'],
                    'scroll_direction_changes': scroll_data['direction_changes'],
                    'content_interaction_depth': interaction_data['depth'],
                    'tab_switches': interaction_data['tab_switches'],
                    'focus_duration': interaction_data['focus_duration'],
                    'session_time_of_day': random.randint(8, 22),
                    'session_duration': random.randint(300, 3600),
                    'timestamp': datetime.now().isoformat()
                })
        
        return pd.DataFrame(data)
    
    def generate_cognitive_load_training_data(self) -> pd.DataFrame:
        """Generate training data for cognitive load assessor."""
        data = []
        
        for user_id, profile in self.user_profiles.items():
            for session in range(self.sessions_per_user):
                session_id = f"{user_id}_session_{session}"
                
                # Generate cognitive load based on task complexity and user capacity
                task_complexity = random.uniform(0.2, 0.9)
                cognitive_load_score = self._calculate_cognitive_load(profile, task_complexity)
                
                # Generate performance indicators
                response_times = self._generate_response_times(cognitive_load_score, task_complexity)
                accuracy_scores = self._generate_accuracy_scores(cognitive_load_score, profile)
                error_patterns = self._generate_error_patterns(cognitive_load_score)
                
                data.append({
                    'user_id': user_id,
                    'session_id': session_id,
                    'cognitive_load_score': cognitive_load_score,
                    'avg_response_time': np.mean(response_times),
                    'response_time_std': np.std(response_times),
                    'avg_accuracy': np.mean(accuracy_scores),
                    'accuracy_decline': self._calculate_accuracy_decline(accuracy_scores),
                    'task_complexity': task_complexity,
                    'error_count': error_patterns['total'],
                    'critical_errors': error_patterns['critical'],
                    'hesitation_count': random.randint(0, 5),
                    'multitask_events': random.randint(0, 3),
                    'fatigue_indicators': min(1.0, session * 0.1 + random.uniform(0, 0.2)),
                    'session_duration': random.randint(600, 3600),
                    'timestamp': datetime.now().isoformat()
                })
        
        return pd.DataFrame(data)
    
    def generate_learning_style_training_data(self) -> pd.DataFrame:
        """Generate training data for learning style detector."""
        data = []
        
        content_types = ['visual', 'auditory', 'text', 'interactive']
        
        for user_id, profile in self.user_profiles.items():
            for session in range(self.sessions_per_user):
                session_id = f"{user_id}_session_{session}"
                
                learning_style = profile['learning_style']
                
                # Generate content interaction patterns
                interactions = self._generate_content_interactions(learning_style)
                time_spent = self._generate_time_allocation(learning_style)
                performance = self._generate_performance_by_type(learning_style)
                engagement = self._generate_engagement_metrics(learning_style)
                
                data.append({
                    'user_id': user_id,
                    'session_id': session_id,
                    'learning_style': learning_style,
                    'visual_interactions': interactions['visual'],
                    'auditory_interactions': interactions['auditory'],
                    'text_interactions': interactions['text'],
                    'interactive_interactions': interactions['interactive'],
                    'visual_time_ratio': time_spent['visual'],
                    'auditory_time_ratio': time_spent['auditory'],
                    'text_time_ratio': time_spent['text'],
                    'interactive_time_ratio': time_spent['interactive'],
                    'visual_performance': performance['visual'],
                    'auditory_performance': performance['auditory'],
                    'text_performance': performance['text'],
                    'interactive_performance': performance['interactive'],
                    'visual_engagement': engagement['visual'],
                    'auditory_engagement': engagement['auditory'],
                    'text_engagement': engagement['text'],
                    'interactive_engagement': engagement['interactive'],
                    'completion_variance': np.var(list(engagement.values())),
                    'replay_concentration': random.uniform(0.1, 0.9),
                    'timestamp': datetime.now().isoformat()
                })
        
        return pd.DataFrame(data)
    
    def _determine_attention_level(self, profile: Dict, session: int) -> str:
        """Determine attention level based on user profile and session."""
        baseline = profile['attention_baseline']
        fatigue_factor = min(1.0, session * 0.15)  # Fatigue increases with sessions
        motivation = profile['motivation_level']
        
        # Calculate attention score
        base_score = {'high': 0.8, 'medium': 0.6, 'low': 0.4}[baseline]
        adjusted_score = base_score * (1 - fatigue_factor) * motivation
        
        # Add some randomness
        adjusted_score += random.uniform(-0.2, 0.2)
        adjusted_score = max(0, min(1, adjusted_score))
        
        if adjusted_score > 0.7:
            return 'high'
        elif adjusted_score > 0.4:
            return 'medium'
        elif adjusted_score > 0.2:
            return 'low'
        else:
            return 'critical'
    
    def _generate_mouse_data(self, attention_level: str) -> Dict:
        """Generate mouse movement data based on attention level."""
        if attention_level == 'high':
            return {
                'avg_speed': random.uniform(200, 400),
                'variance': random.uniform(100, 200),
                'click_freq': random.uniform(0.5, 1.0),
                'idle_periods': random.randint(0, 2)
            }
        elif attention_level == 'medium':
            return {
                'avg_speed': random.uniform(150, 300),
                'variance': random.uniform(150, 250),
                'click_freq': random.uniform(0.3, 0.7),
                'idle_periods': random.randint(1, 3)
            }
        elif attention_level == 'low':
            return {
                'avg_speed': random.uniform(100, 200),
                'variance': random.uniform(200, 350),
                'click_freq': random.uniform(0.1, 0.4),
                'idle_periods': random.randint(2, 5)
            }
        else:  # critical
            return {
                'avg_speed': random.uniform(50, 150),
                'variance': random.uniform(300, 500),
                'click_freq': random.uniform(0.0, 0.2),
                'idle_periods': random.randint(4, 8)
            }
    
    def _generate_keyboard_data(self, attention_level: str) -> Dict:
        """Generate keyboard interaction data."""
        attention_multiplier = {'high': 1.0, 'medium': 0.8, 'low': 0.6, 'critical': 0.4}[attention_level]
        
        return {
            'speed': random.uniform(30, 80) * attention_multiplier,
            'consistency': random.uniform(0.6, 0.9) * attention_multiplier,
            'backspace_freq': random.uniform(0.05, 0.2) * (2 - attention_multiplier)
        }
    
    def _generate_scroll_data(self, attention_level: str) -> Dict:
        """Generate scroll behavior data."""
        if attention_level in ['high', 'medium']:
            return {
                'speed': random.uniform(100, 300),
                'direction_changes': random.randint(2, 8)
            }
        else:
            return {
                'speed': random.uniform(50, 150),
                'direction_changes': random.randint(8, 20)
            }
    
    def _generate_interaction_data(self, attention_level: str) -> Dict:
        """Generate content interaction data."""
        attention_score = {'high': 0.9, 'medium': 0.7, 'low': 0.4, 'critical': 0.2}[attention_level]
        
        return {
            'depth': attention_score + random.uniform(-0.1, 0.1),
            'tab_switches': random.randint(0, 5) if attention_level != 'high' else random.randint(0, 2),
            'focus_duration': random.uniform(30, 300) * attention_score
        }
    
    def _calculate_cognitive_load(self, profile: Dict, task_complexity: float) -> float:
        """Calculate cognitive load score."""
        capacity = {'high': 0.9, 'medium': 0.6, 'low': 0.4}[profile['cognitive_capacity']]
        experience_factor = profile['experience_level'] / 10.0
        
        # Cognitive load = task complexity / (capacity * experience)
        load = task_complexity / (capacity * (0.5 + experience_factor * 0.5))
        
        # Add some noise
        load += random.uniform(-0.1, 0.1)
        
        return max(0, min(1, load))
    
    def _generate_response_times(self, cognitive_load: float, task_complexity: float) -> List[float]:
        """Generate response times based on cognitive load."""
        base_time = 1.0 + task_complexity * 2.0  # 1-3 seconds base
        load_multiplier = 1 + cognitive_load * 2  # 1x to 3x multiplier
        
        times = []
        for _ in range(random.randint(5, 15)):
            time = base_time * load_multiplier * random.uniform(0.5, 2.0)
            times.append(max(0.5, time))
        
        return times
    
    def _generate_accuracy_scores(self, cognitive_load: float, profile: Dict) -> List[float]:
        """Generate accuracy scores."""
        base_accuracy = 0.8 + profile['experience_level'] * 0.02
        load_penalty = cognitive_load * 0.3  # Up to 30% penalty for high load
        
        scores = []
        for _ in range(random.randint(5, 15)):
            score = base_accuracy - load_penalty + random.uniform(-0.1, 0.1)
            scores.append(max(0.1, min(1.0, score)))
        
        return scores
    
    def _generate_error_patterns(self, cognitive_load: float) -> Dict:
        """Generate error patterns."""
        error_rate = cognitive_load * 5  # 0-5 errors based on load
        total_errors = int(error_rate + random.uniform(0, 2))
        critical_errors = int(total_errors * 0.2) if total_errors > 2 else 0
        
        return {
            'total': total_errors,
            'critical': critical_errors
        }
    
    def _calculate_accuracy_decline(self, accuracy_scores: List[float]) -> float:
        """Calculate accuracy decline over time."""
        if len(accuracy_scores) < 3:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(accuracy_scores))
        slope = np.polyfit(x, accuracy_scores, 1)[0]
        
        return max(0, -slope)  # Return positive value for decline
    
    def _generate_content_interactions(self, learning_style: str) -> Dict:
        """Generate content interactions based on learning style."""
        base_interactions = 20
        
        if learning_style == 'visual':
            return {
                'visual': base_interactions + random.randint(10, 20),
                'auditory': base_interactions + random.randint(-5, 5),
                'text': base_interactions + random.randint(-5, 5),
                'interactive': base_interactions + random.randint(0, 10)
            }
        elif learning_style == 'auditory':
            return {
                'visual': base_interactions + random.randint(-5, 5),
                'auditory': base_interactions + random.randint(10, 20),
                'text': base_interactions + random.randint(-5, 5),
                'interactive': base_interactions + random.randint(0, 10)
            }
        elif learning_style == 'reading':
            return {
                'visual': base_interactions + random.randint(-5, 5),
                'auditory': base_interactions + random.randint(-10, 0),
                'text': base_interactions + random.randint(15, 25),
                'interactive': base_interactions + random.randint(-5, 5)
            }
        elif learning_style == 'kinesthetic':
            return {
                'visual': base_interactions + random.randint(0, 10),
                'auditory': base_interactions + random.randint(-5, 5),
                'text': base_interactions + random.randint(-10, 0),
                'interactive': base_interactions + random.randint(15, 25)
            }
        else:  # multimodal
            return {
                'visual': base_interactions + random.randint(5, 15),
                'auditory': base_interactions + random.randint(5, 15),
                'text': base_interactions + random.randint(5, 15),
                'interactive': base_interactions + random.randint(5, 15)
            }
    
    def _generate_time_allocation(self, learning_style: str) -> Dict:
        """Generate time allocation ratios."""
        if learning_style == 'visual':
            visual_time = random.uniform(0.4, 0.6)
            remaining = 1 - visual_time
            return {
                'visual': visual_time,
                'auditory': remaining * random.uniform(0.1, 0.3),
                'text': remaining * random.uniform(0.2, 0.4),
                'interactive': remaining * random.uniform(0.3, 0.5)
            }
        elif learning_style == 'auditory':
            auditory_time = random.uniform(0.4, 0.6)
            remaining = 1 - auditory_time
            return {
                'visual': remaining * random.uniform(0.2, 0.4),
                'auditory': auditory_time,
                'text': remaining * random.uniform(0.1, 0.3),
                'interactive': remaining * random.uniform(0.3, 0.5)
            }
        elif learning_style == 'reading':
            text_time = random.uniform(0.5, 0.7)
            remaining = 1 - text_time
            return {
                'visual': remaining * random.uniform(0.2, 0.4),
                'auditory': remaining * random.uniform(0.1, 0.2),
                'text': text_time,
                'interactive': remaining * random.uniform(0.3, 0.5)
            }
        elif learning_style == 'kinesthetic':
            interactive_time = random.uniform(0.4, 0.6)
            remaining = 1 - interactive_time
            return {
                'visual': remaining * random.uniform(0.3, 0.5),
                'auditory': remaining * random.uniform(0.2, 0.4),
                'text': remaining * random.uniform(0.1, 0.3),
                'interactive': interactive_time
            }
        else:  # multimodal
            return {
                'visual': random.uniform(0.2, 0.3),
                'auditory': random.uniform(0.2, 0.3),
                'text': random.uniform(0.2, 0.3),
                'interactive': random.uniform(0.2, 0.3)
            }
    
    def _generate_performance_by_type(self, learning_style: str) -> Dict:
        """Generate performance scores by content type."""
        base_performance = 0.7
        style_bonus = 0.2
        
        performances = {
            'visual': base_performance,
            'auditory': base_performance,
            'text': base_performance,
            'interactive': base_performance
        }
        
        if learning_style == 'visual':
            performances['visual'] += style_bonus
        elif learning_style == 'auditory':
            performances['auditory'] += style_bonus
        elif learning_style == 'reading':
            performances['text'] += style_bonus
        elif learning_style == 'kinesthetic':
            performances['interactive'] += style_bonus
        else:  # multimodal
            for key in performances:
                performances[key] += style_bonus * 0.5
        
        # Add noise
        for key in performances:
            performances[key] += random.uniform(-0.1, 0.1)
            performances[key] = max(0.1, min(1.0, performances[key]))
        
        return performances
    
    def _generate_engagement_metrics(self, learning_style: str) -> Dict:
        """Generate engagement metrics by content type."""
        return self._generate_performance_by_type(learning_style)  # Similar pattern


def generate_all_training_data(save_path: str = "training_data/"):
    """Generate and save all training datasets."""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("🔧 Generating training data for all AI models...")
    
    generator = TrainingDataGenerator(num_users=500, sessions_per_user=10)
    
    # Generate attention training data
    print("   📊 Generating attention tracker data...")
    attention_data = generator.generate_attention_training_data()
    attention_data.to_csv(f"{save_path}/attention_training_data.csv", index=False)
    print(f"   ✅ Saved {len(attention_data)} attention samples")
    
    # Generate cognitive load training data
    print("   🧠 Generating cognitive load data...")
    cognitive_data = generator.generate_cognitive_load_training_data()
    cognitive_data.to_csv(f"{save_path}/cognitive_load_training_data.csv", index=False)
    print(f"   ✅ Saved {len(cognitive_data)} cognitive load samples")
    
    # Generate learning style training data
    print("   🎨 Generating learning style data...")
    style_data = generator.generate_learning_style_training_data()
    style_data.to_csv(f"{save_path}/learning_style_training_data.csv", index=False)
    print(f"   ✅ Saved {len(style_data)} learning style samples")
    
    print("🎉 Training data generation complete!")
    
    return {
        'attention': attention_data,
        'cognitive_load': cognitive_data,
        'learning_style': style_data
    }


if __name__ == "__main__":
    generate_all_training_data() 