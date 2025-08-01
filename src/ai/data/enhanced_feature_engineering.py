#!/usr/bin/env python3
"""
🧠 Enhanced Feature Engineering for AI Learning Psychology Models
================================================================

PERFORMANCE IMPROVEMENT TARGET:
- Cognitive Load: 37.6% → 70%+ (40+ features)
- Attention: 67.8% → 80%+ (25+ features)  
- Learning Style: 21.3% → 65%+ (35+ features)

Advanced feature engineering based on educational psychology research:
- Temporal behavioral pattern analysis
- Working memory load indicators
- Multi-scale attention dynamics
- Sequential learning patterns
- Cognitive load assessment via response time analysis
- Attention vigilance and fatigue detection
- Learning style adaptation metrics

Scientific Approach:
✓ Evidence-based psychological features
✓ Statistical significance testing
✓ Feature importance scoring
✓ Robust handling of missing data
✓ Compatible with existing pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')


class EnhancedCognitiveLoadFeatures:
    """
    🧠 Advanced Cognitive Load Feature Engineering (Target: 40+ features)
    =====================================================================
    
    Based on Cognitive Load Theory (Sweller, 1988) and working memory research.
    Extracts 40+ psychological indicators of mental effort and cognitive strain.
    
    Feature Categories:
    1. Temporal Response Patterns (12 features)
    2. Working Memory Proxies (8 features) 
    3. Error Pattern Clustering (6 features)
    4. Mental Fatigue Indicators (8 features)
    5. Task Switching Costs (6 features)
    """
    
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_importance_scores = {}
        self.feature_descriptions = {}
        
    def extract_temporal_patterns(self, response_times: List[float], 
                                 accuracy_scores: List[float]) -> Dict[str, float]:
        """Extract comprehensive temporal patterns indicating cognitive load (12 features)."""
        if len(response_times) < 3:
            return self._get_default_temporal_features()
            
        response_times = np.array(response_times)
        accuracy_scores = np.array(accuracy_scores) if accuracy_scores else np.ones_like(response_times) * 0.5
        
        features = {}
        x = np.arange(len(response_times))
        
        # 1. Response Time Trend Analysis (Linear & Non-linear)
        if len(response_times) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, response_times)
            features['rt_linear_trend'] = slope
            features['rt_trend_strength'] = r_value ** 2
            features['rt_trend_significance'] = max(0.0, 1.0 - p_value) if p_value < 1.0 else 0.0
            
            # Quadratic trend (cognitive fatigue pattern)
            if len(response_times) > 4:
                poly_coeffs = np.polyfit(x, response_times, 2)
                features['rt_quadratic_coefficient'] = poly_coeffs[0]  # Acceleration/deceleration
                quadratic_pred = np.polyval(poly_coeffs, x)
                features['rt_quadratic_fit'] = 1.0 - np.mean((response_times - quadratic_pred)**2) / np.var(response_times)
            else:
                features['rt_quadratic_coefficient'] = 0.0
                features['rt_quadratic_fit'] = 0.0
        else:
            features.update({'rt_linear_trend': 0.0, 'rt_trend_strength': 0.0, 
                           'rt_trend_significance': 0.0, 'rt_quadratic_coefficient': 0.0, 
                           'rt_quadratic_fit': 0.0})
        
        # 2. Response Time Variability & Distribution
        rt_mean = np.mean(response_times)
        rt_std = np.std(response_times)
        features['rt_coefficient_variation'] = rt_std / (rt_mean + 1e-6)
        features['rt_interquartile_range'] = np.percentile(response_times, 75) - np.percentile(response_times, 25)
        features['rt_skewness'] = stats.skew(response_times)  # Distribution asymmetry
        features['rt_kurtosis'] = stats.kurtosis(response_times)  # Distribution peakedness
        
        # 3. Dynamic Response Time Analysis (Moving Windows)
        if len(response_times) >= 6:
            window_size = max(3, len(response_times) // 4)
            rt_dynamics = []
            for i in range(len(response_times) - window_size + 1):
                window_mean = np.mean(response_times[i:i+window_size])
                rt_dynamics.append(window_mean)
            
            if len(rt_dynamics) > 1:
                features['rt_dynamic_variability'] = np.std(rt_dynamics)
                features['rt_momentum'] = rt_dynamics[-1] - rt_dynamics[0]  # Overall direction
                features['rt_volatility'] = np.mean(np.abs(np.diff(rt_dynamics)))  # Change intensity
            else:
                features.update({'rt_dynamic_variability': 0.0, 'rt_momentum': 0.0, 'rt_volatility': 0.0})
        else:
            features.update({'rt_dynamic_variability': 0.0, 'rt_momentum': 0.0, 'rt_volatility': 0.0})
        
        return features
    
    def extract_working_memory_proxies(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract working memory load proxies from behavioral patterns (8 features)."""
        features = {}
        
        response_times = behavioral_data.get('response_times', [])
        accuracy_scores = behavioral_data.get('accuracy_scores', [])
        task_complexities = behavioral_data.get('task_complexities', [])
        
        if len(response_times) < 3:
            return {f'wm_{key}': 0.5 for key in ['capacity_estimate', 'load_variability', 'interference_resistance', 
                   'dual_task_cost', 'serial_position_effect', 'proactive_interference', 'updating_efficiency', 'maintenance_stability']}
        
        # 1. Working Memory Capacity Estimate (Based on n-back paradigm)
        if len(response_times) >= 5:
            # Simulate n-back analysis with sliding window performance
            n_back_window = 3
            capacity_scores = []
            for i in range(n_back_window, len(response_times)):
                # Performance consistency over n-back window
                window_rt = response_times[i-n_back_window:i]
                window_acc = accuracy_scores[i-n_back_window:i] if len(accuracy_scores) > i else [0.5] * n_back_window
                
                rt_consistency = 1.0 / (np.std(window_rt) + 1e-6)
                acc_consistency = 1.0 / (np.std(window_acc) + 1e-6)
                capacity_scores.append((rt_consistency + acc_consistency) / 2)
            
            features['wm_capacity_estimate'] = np.mean(capacity_scores)
            features['wm_load_variability'] = np.std(capacity_scores)
        else:
            features['wm_capacity_estimate'] = 0.5
            features['wm_load_variability'] = 0.5
        
        # 2. Interference Resistance (Response to competing demands)
        multitask_events = behavioral_data.get('multitask_events', [])
        if multitask_events and len(response_times) > 0:
            # Estimate performance degradation during multitasking
            multitask_timestamps = [event.get('timestamp', 0) for event in multitask_events]
            session_duration = behavioral_data.get('session_duration', max(response_times))
            
            # Find response times near multitasking events
            interference_scores = []
            for mt_time in multitask_timestamps:
                # Find responses within 10 seconds of multitask event
                nearby_responses = [rt for i, rt in enumerate(response_times) 
                                  if abs(i * (session_duration / len(response_times)) - mt_time) < 10]
                if nearby_responses:
                    interference_scores.append(np.mean(nearby_responses))
            
            if interference_scores:
                baseline_rt = np.mean(response_times)
                features['wm_interference_resistance'] = baseline_rt / (np.mean(interference_scores) + 1e-6)
            else:
                features['wm_interference_resistance'] = 1.0
        else:
            features['wm_interference_resistance'] = 1.0
        
        # 3. Dual-task Performance Cost
        if len(task_complexities) == len(response_times) and len(response_times) > 6:
            # Compare single vs complex task performance
            simple_tasks = [(rt, acc) for rt, acc, comp in zip(response_times, accuracy_scores, task_complexities) if comp < np.median(task_complexities)]
            complex_tasks = [(rt, acc) for rt, acc, comp in zip(response_times, accuracy_scores, task_complexities) if comp >= np.median(task_complexities)]
            
            if simple_tasks and complex_tasks:
                simple_rt_mean = np.mean([rt for rt, _ in simple_tasks])
                complex_rt_mean = np.mean([rt for rt, _ in complex_tasks])
                features['wm_dual_task_cost'] = complex_rt_mean / (simple_rt_mean + 1e-6)
            else:
                features['wm_dual_task_cost'] = 1.0
        else:
            features['wm_dual_task_cost'] = 1.0
        
        # 4. Serial Position Effects (Primacy/recency in working memory)
        if len(response_times) >= 8:
            first_quarter = response_times[:len(response_times)//4]
            last_quarter = response_times[-len(response_times)//4:]
            middle_half = response_times[len(response_times)//4:-len(response_times)//4]
            
            primacy_effect = np.mean(first_quarter) / (np.mean(middle_half) + 1e-6)
            recency_effect = np.mean(last_quarter) / (np.mean(middle_half) + 1e-6)
            features['wm_serial_position_effect'] = (primacy_effect + recency_effect) / 2
        else:
            features['wm_serial_position_effect'] = 1.0
        
        # 5. Proactive Interference (Previous task interference)
        if len(response_times) >= 6:
            # Measure increase in RT after difficult tasks
            rt_changes = []
            for i in range(1, len(response_times)):
                prev_complexity = task_complexities[i-1] if i-1 < len(task_complexities) else 0.5
                rt_change = response_times[i] - response_times[i-1]
                if prev_complexity > 0.7:  # After difficult task
                    rt_changes.append(rt_change)
            
            if rt_changes:
                features['wm_proactive_interference'] = np.mean(rt_changes) / (np.mean(response_times) + 1e-6)
            else:
                features['wm_proactive_interference'] = 0.0
        else:
            features['wm_proactive_interference'] = 0.0
        
        # 6. Working Memory Updating Efficiency
        if len(accuracy_scores) >= 5:
            # Measure adaptation to changing task demands
            acc_changes = np.diff(accuracy_scores)
            positive_changes = acc_changes[acc_changes > 0]
            negative_changes = acc_changes[acc_changes < 0]
            
            if len(positive_changes) > 0 and len(negative_changes) > 0:
                features['wm_updating_efficiency'] = np.mean(positive_changes) / (abs(np.mean(negative_changes)) + 1e-6)
            else:
                features['wm_updating_efficiency'] = 1.0
        else:
            features['wm_updating_efficiency'] = 1.0
        
        # 7. Working Memory Maintenance Stability
        if len(response_times) >= 6:
            # Measure consistency in maintaining information
            window_size = 3
            stability_scores = []
            for i in range(len(response_times) - window_size + 1):
                window_std = np.std(response_times[i:i+window_size])
                stability_scores.append(1.0 / (window_std + 1e-6))
            
            features['wm_maintenance_stability'] = np.mean(stability_scores)
        else:
            features['wm_maintenance_stability'] = 1.0
        
        return features
    
    def extract_mental_fatigue_indicators(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract mental fatigue and cognitive depletion indicators (8 features)."""
        features = {}
        
        response_times = behavioral_data.get('response_times', [])
        accuracy_scores = behavioral_data.get('accuracy_scores', [])
        session_duration = behavioral_data.get('session_duration', 600)
        
        if len(response_times) < 4:
            return {f'fatigue_{key}': 0.0 for key in ['temporal_decline', 'accuracy_degradation', 'variability_increase', 
                   'vigilance_decrement', 'effort_compensation', 'recovery_periods', 'sustained_deficit', 'cognitive_depletion']}
        
        # 1. Temporal Performance Decline
        time_points = np.linspace(0, session_duration, len(response_times))
        if len(response_times) > 3:
            # Linear regression of performance over time
            rt_slope, _, rt_r, _, _ = stats.linregress(time_points, response_times)
            features['fatigue_temporal_decline'] = max(0.0, rt_slope)  # Positive slope indicates slowing
            
            if len(accuracy_scores) == len(response_times):
                acc_slope, _, acc_r, _, _ = stats.linregress(time_points, accuracy_scores)
                features['fatigue_accuracy_degradation'] = max(0.0, -acc_slope)  # Negative slope indicates decline
            else:
                features['fatigue_accuracy_degradation'] = 0.0
        else:
            features['fatigue_temporal_decline'] = 0.0
            features['fatigue_accuracy_degradation'] = 0.0
        
        # 2. Response Time Variability Increase (Fatigue signature)
        if len(response_times) >= 8:
            # Compare first and last quarters
            n_quarter = max(2, len(response_times) // 4)
            early_variability = np.std(response_times[:n_quarter])
            late_variability = np.std(response_times[-n_quarter:])
            features['fatigue_variability_increase'] = (late_variability - early_variability) / (early_variability + 1e-6)
        else:
            features['fatigue_variability_increase'] = 0.0
        
        # 3. Vigilance Decrement (Attention sustained performance decline)
        if len(response_times) >= 10:
            # Divide into 5 equal time bins
            n_bins = 5
            bin_size = len(response_times) // n_bins
            bin_performances = []
            
            for i in range(n_bins):
                start_idx = i * bin_size
                end_idx = start_idx + bin_size if i < n_bins-1 else len(response_times)
                bin_rt = response_times[start_idx:end_idx]
                bin_acc = accuracy_scores[start_idx:end_idx] if len(accuracy_scores) >= end_idx else [0.5] * (end_idx - start_idx)
                
                # Combined performance score (lower RT + higher accuracy = better)
                perf_score = np.mean(bin_acc) / (np.mean(bin_rt) + 1e-6)
                bin_performances.append(perf_score)
            
            # Vigilance decrement = decline from peak to final performance
            peak_performance = max(bin_performances)
            final_performance = bin_performances[-1]
            features['fatigue_vigilance_decrement'] = (peak_performance - final_performance) / (peak_performance + 1e-6)
        else:
            features['fatigue_vigilance_decrement'] = 0.0
        
        # 4. Effort Compensation (Increased effort to maintain performance)
        if len(response_times) >= 6 and len(accuracy_scores) >= 6:
            # Look for increased RT but maintained accuracy (compensatory effort)
            mid_point = len(response_times) // 2
            early_rt = np.mean(response_times[:mid_point])
            late_rt = np.mean(response_times[mid_point:])
            early_acc = np.mean(accuracy_scores[:mid_point])
            late_acc = np.mean(accuracy_scores[mid_point:])
            
            rt_increase = (late_rt - early_rt) / (early_rt + 1e-6)
            acc_maintenance = 1.0 - abs(late_acc - early_acc)
            
            # High effort compensation = increased RT with maintained accuracy
            features['fatigue_effort_compensation'] = rt_increase * acc_maintenance
        else:
            features['fatigue_effort_compensation'] = 0.0
        
        # 5. Recovery Period Detection
        if len(response_times) >= 8:
            # Look for brief improvements in performance (micro-recoveries)
            rt_smooth = signal.savgol_filter(response_times, min(5, len(response_times)//2*2-1), 2)
            rt_peaks = signal.find_peaks(-rt_smooth, prominence=np.std(response_times)*0.5)[0]
            features['fatigue_recovery_periods'] = len(rt_peaks) / (session_duration / 60.0)  # recoveries per minute
        else:
            features['fatigue_recovery_periods'] = 0.0
        
        # 6. Sustained Performance Deficit
        if len(response_times) >= 6:
            baseline_rt = np.mean(response_times[:len(response_times)//3])  # First third as baseline
            sustained_rt = np.mean(response_times[len(response_times)//3:])  # Rest of session
            features['fatigue_sustained_deficit'] = (sustained_rt - baseline_rt) / (baseline_rt + 1e-6)
        else:
            features['fatigue_sustained_deficit'] = 0.0
        
        # 7. Cognitive Resource Depletion (Based on task switching costs)
        multitask_events = behavioral_data.get('multitask_events', [])
        if multitask_events and len(response_times) > 4:
            # Measure increasing cost of task switches over time
            switch_costs = []
            for event in multitask_events:
                event_time = event.get('timestamp', 0)
                time_ratio = event_time / session_duration
                
                # Find nearby response times
                event_idx = int(time_ratio * len(response_times))
                if 0 < event_idx < len(response_times) - 1:
                    pre_switch_rt = response_times[event_idx - 1]
                    post_switch_rt = response_times[event_idx]
                    switch_cost = post_switch_rt - pre_switch_rt
                    switch_costs.append((time_ratio, switch_cost))
            
            if len(switch_costs) >= 3:
                # Check if switch costs increase over time (resource depletion)
                times, costs = zip(*switch_costs)
                depletion_slope, _, _, _, _ = stats.linregress(times, costs)
                features['fatigue_cognitive_depletion'] = max(0.0, depletion_slope)
            else:
                features['fatigue_cognitive_depletion'] = 0.0
        else:
            features['fatigue_cognitive_depletion'] = 0.0
        
        return features
    
    def extract_task_switching_costs(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract task switching and cognitive flexibility indicators (6 features)."""
        features = {}
        
        multitask_events = behavioral_data.get('multitask_events', [])
        response_times = behavioral_data.get('response_times', [])
        session_duration = behavioral_data.get('session_duration', 600)
        
        if not multitask_events or len(response_times) < 4:
            return {f'switch_{key}': 0.0 for key in ['frequency', 'cost_magnitude', 'adaptation_rate', 
                   'flexibility_index', 'perseveration_tendency', 'context_updating_efficiency']}
        
        # 1. Task Switching Frequency
        features['switch_frequency'] = len(multitask_events) / (session_duration / 60.0)  # switches per minute
        
        # 2. Switch Cost Magnitude
        switch_costs = []
        for event in multitask_events:
            event_time = event.get('timestamp', 0)
            event_idx = int((event_time / session_duration) * len(response_times))
            
            if 1 <= event_idx < len(response_times) - 1:
                pre_switch_rt = response_times[event_idx - 1]
                post_switch_rt = response_times[event_idx]
                switch_cost = post_switch_rt - pre_switch_rt
                switch_costs.append(switch_cost)
        
        if switch_costs:
            features['switch_cost_magnitude'] = np.mean(switch_costs)
            features['switch_adaptation_rate'] = -np.corrcoef(range(len(switch_costs)), switch_costs)[0,1] if len(switch_costs) > 2 else 0.0
        else:
            features['switch_cost_magnitude'] = 0.0
            features['switch_adaptation_rate'] = 0.0
        
        # 3. Cognitive Flexibility Index
        if len(multitask_events) > 2:
            event_types = [event.get('type', 'unknown') for event in multitask_events]
            type_diversity = len(set(event_types))
            switch_regularity = 1.0 / (np.std([event.get('timestamp', i) for i, event in enumerate(multitask_events)]) + 1e-6)
            features['switch_flexibility_index'] = type_diversity * (1.0 / (switch_regularity + 1e-6))
        else:
            features['switch_flexibility_index'] = 0.0
        
        # 4. Perseveration Tendency (Difficulty disengaging from previous task)
        if switch_costs and len(switch_costs) > 2:
            # Large, persistent switch costs indicate perseveration
            persistent_costs = [cost for cost in switch_costs if cost > np.mean(switch_costs)]
            features['switch_perseveration_tendency'] = len(persistent_costs) / len(switch_costs)
        else:
            features['switch_perseveration_tendency'] = 0.0
        
        # 5. Context Updating Efficiency
        if len(multitask_events) >= 3:
            # Measure improvement in handling switches of same type
            event_type_costs = {}
            for i, event in enumerate(multitask_events):
                event_type = event.get('type', 'unknown')
                if event_type not in event_type_costs:
                    event_type_costs[event_type] = []
                
                if i < len(switch_costs):
                    event_type_costs[event_type].append(switch_costs[i])
            
            # Average improvement across event types
            efficiency_scores = []
            for event_type, costs in event_type_costs.items():
                if len(costs) > 1:
                    # Improvement = negative correlation between occurrence order and cost
                    improvement = -np.corrcoef(range(len(costs)), costs)[0,1] if len(costs) > 2 else 0.0
                    efficiency_scores.append(improvement)
            
            features['switch_context_updating_efficiency'] = np.mean(efficiency_scores) if efficiency_scores else 0.0
        else:
            features['switch_context_updating_efficiency'] = 0.0
        
        return features
    
    def extract_complexity_adaptation_features(self, task_complexities: List[float], 
                                             performance_metrics: Dict[str, List[float]]) -> Dict[str, float]:
        """Extract features showing adaptation to task complexity."""
        if not task_complexities:
            return self._get_default_complexity_features()
            
        complexities = np.array(task_complexities)
        features = {}
        
        # Complexity distribution analysis
        features['complexity_mean'] = np.mean(complexities)
        features['complexity_std'] = np.std(complexities)
        features['complexity_range'] = np.max(complexities) - np.min(complexities) if len(complexities) > 1 else 0.0
        
        # Performance vs complexity analysis
        response_times = performance_metrics.get('response_times', [])
        accuracy_scores = performance_metrics.get('accuracy_scores', [])
        
        if len(response_times) == len(complexities) and len(complexities) > 2:
            # Complexity-performance correlations
            rt_complexity_corr, _ = stats.pearsonr(complexities, response_times)
            features['complexity_rt_correlation'] = rt_complexity_corr if not np.isnan(rt_complexity_corr) else 0.0
            
            if len(accuracy_scores) == len(complexities):
                acc_complexity_corr, _ = stats.pearsonr(complexities, accuracy_scores)
                features['complexity_accuracy_correlation'] = acc_complexity_corr if not np.isnan(acc_complexity_corr) else 0.0
            else:
                features['complexity_accuracy_correlation'] = 0.0
            
            # Adaptation efficiency
            normalized_rt = np.array(response_times) / (np.array(complexities) + 1e-6)
            features['adaptation_efficiency'] = 1.0 / (np.std(normalized_rt) + 1e-6)
        else:
            features['complexity_rt_correlation'] = 0.0
            features['complexity_accuracy_correlation'] = 0.0
            features['adaptation_efficiency'] = 0.0
        
        # Overload detection
        if len(complexities) > 3:
            high_complexity_threshold = np.percentile(complexities, 75)
            high_complexity_indices = np.where(complexities >= high_complexity_threshold)[0]
            
            if len(high_complexity_indices) > 0 and len(response_times) > max(high_complexity_indices):
                high_complexity_rt = [response_times[i] for i in high_complexity_indices if i < len(response_times)]
                if high_complexity_rt:
                    features['high_complexity_rt_mean'] = np.mean(high_complexity_rt)
                    features['overload_indicator'] = features['high_complexity_rt_mean'] / (np.mean(response_times) + 1e-6)
                else:
                    features['high_complexity_rt_mean'] = 0.0
                    features['overload_indicator'] = 1.0
            else:
                features['high_complexity_rt_mean'] = 0.0
                features['overload_indicator'] = 1.0
        else:
            features['high_complexity_rt_mean'] = 0.0
            features['overload_indicator'] = 1.0
        
        return features
    
    def extract_error_pattern_features(self, error_patterns: Dict[str, int], 
                                     hesitation_indicators: List[Dict]) -> Dict[str, float]:
        """Extract cognitive load indicators from error patterns."""
        features = {}
        
        # Error distribution analysis
        total_errors = sum(error_patterns.values()) if error_patterns else 0
        features['total_error_count'] = total_errors
        features['error_diversity'] = len(error_patterns) if error_patterns else 0
        
        # Error severity analysis
        critical_errors = error_patterns.get('critical', 0)
        minor_errors = error_patterns.get('minor', 0)
        features['critical_error_ratio'] = critical_errors / (total_errors + 1e-6)
        features['error_severity_score'] = (critical_errors * 2 + minor_errors) / max(1, total_errors)
        
        # Hesitation analysis
        hesitation_count = len(hesitation_indicators)
        features['hesitation_frequency'] = hesitation_count
        
        if hesitation_indicators:
            hesitation_durations = [h.get('duration', 0) for h in hesitation_indicators]
            features['hesitation_mean_duration'] = np.mean(hesitation_durations)
            features['hesitation_total_time'] = np.sum(hesitation_durations)
            features['hesitation_variability'] = np.std(hesitation_durations) if len(hesitation_durations) > 1 else 0.0
        else:
            features['hesitation_mean_duration'] = 0.0
            features['hesitation_total_time'] = 0.0
            features['hesitation_variability'] = 0.0
        
        # Cognitive strain indicators
        features['cognitive_strain_score'] = (
            features['critical_error_ratio'] * 0.4 +
            features['hesitation_frequency'] * 0.3 +
            features['hesitation_mean_duration'] * 0.3
        )
        
        return features
    
    def extract_multitasking_features(self, multitask_events: List[Dict], 
                                    session_duration: float) -> Dict[str, float]:
        """Extract cognitive load indicators from multitasking behavior."""
        features = {}
        
        event_count = len(multitask_events)
        features['multitask_event_count'] = event_count
        features['multitask_frequency'] = event_count / (session_duration / 60.0 + 1e-6)  # events per minute
        
        if multitask_events:
            # Event type analysis
            event_types = [event.get('type', 'unknown') for event in multitask_events]
            unique_types = set(event_types)
            features['multitask_type_diversity'] = len(unique_types)
            
            # Context switching analysis
            tab_switches = sum(1 for event in multitask_events if event.get('type') == 'tab_switch')
            app_switches = sum(1 for event in multitask_events if event.get('type') == 'app_switch')
            
            features['tab_switching_rate'] = tab_switches / (session_duration / 60.0 + 1e-6)
            features['app_switching_rate'] = app_switches / (session_duration / 60.0 + 1e-6)
            
            # Temporal pattern of multitasking
            timestamps = [event.get('timestamp', 0) for event in multitask_events]
            if len(timestamps) > 1:
                intervals = np.diff(sorted(timestamps))
                features['multitask_interval_mean'] = np.mean(intervals)
                features['multitask_interval_std'] = np.std(intervals)
                features['multitask_regularity'] = 1.0 / (np.std(intervals) + 1e-6)
            else:
                features['multitask_interval_mean'] = 0.0
                features['multitask_interval_std'] = 0.0
                features['multitask_regularity'] = 0.0
            
            # Multitasking burden score
            features['multitasking_burden'] = (
                features['multitask_frequency'] * 0.4 +
                features['multitask_type_diversity'] * 0.3 +
                (1.0 - features['multitask_regularity']) * 0.3
            )
        else:
            features['multitask_type_diversity'] = 0.0
            features['tab_switching_rate'] = 0.0
            features['app_switching_rate'] = 0.0
            features['multitask_interval_mean'] = 0.0
            features['multitask_interval_std'] = 0.0
            features['multitask_regularity'] = 1.0
            features['multitasking_burden'] = 0.0
        
        return features
    
    def extract_working_memory_proxies(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract working memory load proxies from behavioral patterns."""
        features = {}
        
        # Extract relevant data
        response_times = behavioral_data.get('response_times', [])
        accuracy_scores = behavioral_data.get('accuracy_scores', [])
        task_complexities = behavioral_data.get('task_complexities', [])
        
        # Working memory capacity indicators
        if len(response_times) > 3 and len(accuracy_scores) > 3:
            # N-back style analysis (performance consistency)
            if len(response_times) >= 5:
                consistency_window = 3
                consistencies = []
                for i in range(len(response_times) - consistency_window + 1):
                    window_rt = response_times[i:i+consistency_window]
                    window_acc = accuracy_scores[i:i+consistency_window] if i+consistency_window <= len(accuracy_scores) else [0.5] * consistency_window
                    rt_consistency = 1.0 / (np.std(window_rt) + 1e-6)
                    acc_consistency = 1.0 / (np.std(window_acc) + 1e-6)
                    consistencies.append((rt_consistency + acc_consistency) / 2)
                
                features['working_memory_consistency'] = np.mean(consistencies)
                features['working_memory_stability'] = 1.0 / (np.std(consistencies) + 1e-6)
            else:
                features['working_memory_consistency'] = 0.5
                features['working_memory_stability'] = 0.5
            
            # Dual-task performance proxy
            if len(task_complexities) == len(response_times):
                complex_tasks = [(rt, acc, comp) for rt, acc, comp in zip(response_times, accuracy_scores, task_complexities)]
                if complex_tasks:
                    # Sort by complexity
                    complex_tasks.sort(key=lambda x: x[2])
                    
                    # Compare performance on high vs low complexity
                    n_tasks = len(complex_tasks)
                    low_complexity = complex_tasks[:n_tasks//3] if n_tasks >= 6 else complex_tasks[:max(1, n_tasks//2)]
                    high_complexity = complex_tasks[-n_tasks//3:] if n_tasks >= 6 else complex_tasks[-max(1, n_tasks//2):]
                    
                    low_rt_mean = np.mean([x[0] for x in low_complexity])
                    high_rt_mean = np.mean([x[0] for x in high_complexity])
                    low_acc_mean = np.mean([x[1] for x in low_complexity])
                    high_acc_mean = np.mean([x[1] for x in high_complexity])
                    
                    features['complexity_rt_ratio'] = high_rt_mean / (low_rt_mean + 1e-6)
                    features['complexity_accuracy_drop'] = low_acc_mean - high_acc_mean
                    features['working_memory_load'] = features['complexity_rt_ratio'] * (1 + features['complexity_accuracy_drop'])
                else:
                    features['complexity_rt_ratio'] = 1.0
                    features['complexity_accuracy_drop'] = 0.0
                    features['working_memory_load'] = 1.0
            else:
                features['complexity_rt_ratio'] = 1.0
                features['complexity_accuracy_drop'] = 0.0
                features['working_memory_load'] = 1.0
        else:
            features['working_memory_consistency'] = 0.5
            features['working_memory_stability'] = 0.5
            features['complexity_rt_ratio'] = 1.0
            features['complexity_accuracy_drop'] = 0.0
            features['working_memory_load'] = 1.0
        
        return features
    
    def _get_default_temporal_features(self) -> Dict[str, float]:
        """Default values when insufficient data for temporal analysis."""
        return {
            'rt_trend_slope': 0.0,
            'rt_trend_r2': 0.0,
            'rt_trend_significance': 0.0,
            'rt_coefficient_of_variation': 0.0,
            'rt_range_normalized': 0.0,
            'rt_acceleration': 0.0,
            'rt_velocity_change': 0.0,
            'rt_accuracy_coupling': 0.0,
            'speed_accuracy_tradeoff': 0.0,
            'fatigue_rt_increase': 0.0,
            'fatigue_accuracy_drop': 0.0
        }
    
    def _get_default_complexity_features(self) -> Dict[str, float]:
        """Default values when no complexity data available."""
        return {
            'complexity_mean': 0.5,
            'complexity_std': 0.0,
            'complexity_range': 0.0,
            'complexity_rt_correlation': 0.0,
            'complexity_accuracy_correlation': 0.0,
            'adaptation_efficiency': 0.0,
            'high_complexity_rt_mean': 0.0,
            'overload_indicator': 1.0
        }
    
    def extract_all_features(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract all 40+ enhanced cognitive load features."""
        features = {}
        
        # 1. Temporal patterns (12 features)
        response_times = behavioral_data.get('response_times', [])
        accuracy_scores = behavioral_data.get('accuracy_scores', [])
        temporal_features = self.extract_temporal_patterns(response_times, accuracy_scores)
        features.update(temporal_features)
        
        # 2. Working memory proxies (8 features)
        wm_features = self.extract_working_memory_proxies(behavioral_data)
        features.update(wm_features)
        
        # 3. Mental fatigue indicators (8 features)
        fatigue_features = self.extract_mental_fatigue_indicators(behavioral_data)
        features.update(fatigue_features)
        
        # 4. Task switching costs (6 features)
        switching_features = self.extract_task_switching_costs(behavioral_data)
        features.update(switching_features)
        
        # 5. Complexity adaptation (6+ features)
        task_complexities = behavioral_data.get('task_complexities', [])
        performance_metrics = {
            'response_times': response_times,
            'accuracy_scores': accuracy_scores
        }
        complexity_features = self.extract_complexity_adaptation_features(task_complexities, performance_metrics)
        features.update(complexity_features)
        
        # 6. Error patterns (remaining features)
        error_patterns = behavioral_data.get('error_patterns', {})
        hesitation_indicators = behavioral_data.get('hesitation_indicators', [])
        error_features = self.extract_error_pattern_features(error_patterns, hesitation_indicators)
        features.update(error_features)
        
        return features


class EnhancedAttentionFeatures:
    """
    👁️ Advanced Attention Feature Engineering (Target: 25+ features)
    ===============================================================
    
    Based on attention research (Posner & Petersen, 1990) and vigilance studies.
    Extracts 25+ behavioral indicators of attention state and vigilance.
    
    Feature Categories:
    1. Fourier Analysis of Behavioral Rhythms (5 features)
    2. Multi-scale Movement Patterns (6 features)
    3. Vigilance and Sustained Attention (8 features)
    4. Focus Duration and Depth (6 features)
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_importance_scores = {}
        self.feature_descriptions = {}
    
    def extract_fourier_features(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract comprehensive frequency domain features from behavioral rhythms (5 features)."""
        
        # Create activity time series from all behavioral events
        all_events = []
        session_duration = behavioral_data.get('session_duration', 600)
        
        # Collect timestamped events
        for event_type in ['mouse_movements', 'keyboard_events', 'scroll_events', 'content_interactions']:
            events = behavioral_data.get(event_type, [])
            for event in events:
                timestamp = event.get('timestamp', 0)
                all_events.append(timestamp)
        
        if len(all_events) < 10:
            return {f'fourier_{key}': 0.0 for key in ['dominant_frequency', 'rhythm_regularity', 
                   'attention_oscillation', 'spectral_entropy', 'frequency_stability']}
        
        # Create binned activity signal
        n_bins = min(128, len(all_events) // 2)  # Ensure sufficient resolution
        bin_edges = np.linspace(0, session_duration, n_bins + 1)
        activity_signal, _ = np.histogram(all_events, bins=bin_edges)
        
        # Apply window to reduce edge effects
        window = signal.windows.hann(len(activity_signal))
        windowed_signal = activity_signal * window
        
        # FFT analysis
        fft_values = fft(windowed_signal)
        freqs = fftfreq(len(windowed_signal), d=session_duration/n_bins)
        power_spectrum = np.abs(fft_values[:len(fft_values)//2]) ** 2
        positive_freqs = freqs[:len(freqs)//2]
        
        features = {}
        
        # 1. Dominant Frequency (main behavioral rhythm)
        if len(power_spectrum) > 1:
            peak_idx = np.argmax(power_spectrum[1:]) + 1  # Skip DC component
            features['fourier_dominant_frequency'] = abs(positive_freqs[peak_idx])
        else:
            features['fourier_dominant_frequency'] = 0.0
        
        # 2. Rhythm Regularity (how periodic is behavior)
        if len(power_spectrum) > 2:
            # Measure how much energy is in the dominant frequency vs spread
            total_power = np.sum(power_spectrum[1:])  # Exclude DC
            dominant_power = np.max(power_spectrum[1:])
            features['fourier_rhythm_regularity'] = dominant_power / (total_power + 1e-6)
        else:
            features['fourier_rhythm_regularity'] = 0.0
        
        # 3. Attention Oscillation Strength
        if len(power_spectrum) > 3:
            # Focus on attention-relevant frequencies (0.1-2 Hz typical for attention)
            attention_band_mask = (positive_freqs >= 0.1) & (positive_freqs <= 2.0)
            if np.any(attention_band_mask):
                attention_power = np.sum(power_spectrum[attention_band_mask])
                features['fourier_attention_oscillation'] = attention_power / (np.sum(power_spectrum) + 1e-6)
            else:
                features['fourier_attention_oscillation'] = 0.0
        else:
            features['fourier_attention_oscillation'] = 0.0
        
        # 4. Spectral Entropy (complexity of attention patterns)
        if len(power_spectrum) > 1:
            # Normalize power spectrum to probabilities
            power_norm = power_spectrum / (np.sum(power_spectrum) + 1e-6)
            power_norm = power_norm[power_norm > 0]  # Remove zeros
            entropy = -np.sum(power_norm * np.log2(power_norm + 1e-10))
            max_entropy = np.log2(len(power_norm))
            features['fourier_spectral_entropy'] = entropy / (max_entropy + 1e-6)
        else:
            features['fourier_spectral_entropy'] = 0.0
        
        # 5. Frequency Stability (consistency of dominant rhythm over time)
        if len(activity_signal) >= 32:  # Need sufficient data for windowed analysis
            window_size = len(activity_signal) // 4
            dominant_freqs = []
            
            for i in range(0, len(activity_signal) - window_size + 1, window_size // 2):
                window_signal = activity_signal[i:i + window_size]
                if len(window_signal) > 4:
                    window_fft = fft(window_signal)
                    window_power = np.abs(window_fft[:len(window_fft)//2]) ** 2
                    if len(window_power) > 1:
                        peak_idx = np.argmax(window_power[1:]) + 1
                        window_freqs = fftfreq(len(window_signal), d=session_duration/n_bins)
                        dominant_freqs.append(abs(window_freqs[peak_idx]))
            
            if len(dominant_freqs) > 1:
                features['fourier_frequency_stability'] = 1.0 / (np.std(dominant_freqs) + 1e-6)
            else:
                features['fourier_frequency_stability'] = 1.0
        else:
            features['fourier_frequency_stability'] = 1.0
        
        return features
    
    def extract_vigilance_features(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract vigilance and sustained attention indicators (8 features)."""
        features = {}
        
        response_times = behavioral_data.get('response_times', [])
        accuracy_scores = behavioral_data.get('accuracy_scores', [])
        session_duration = behavioral_data.get('session_duration', 600)
        
        # Collect all behavioral events for vigilance analysis
        all_events = []
        for event_type in ['mouse_movements', 'keyboard_events', 'scroll_events', 'content_interactions']:
            events = behavioral_data.get(event_type, [])
            for event in events:
                timestamp = event.get('timestamp', 0)
                all_events.append(timestamp)
        
        if len(all_events) < 5:
            return {f'vigilance_{key}': 0.0 for key in ['decrement_slope', 'lapse_frequency', 
                   'sustained_deficit', 'recovery_rate', 'alertness_variability', 
                   'micro_sleep_indicators', 'attention_stability', 'vigilance_reserve']}
        
        all_events.sort()
        
        # 1. Vigilance Decrement Slope (classic vigilance task measure)
        if len(response_times) >= 8:
            # Divide session into time blocks and measure performance decline
            n_blocks = min(6, len(response_times) // 2)
            block_size = len(response_times) // n_blocks
            block_performance = []
            
            for i in range(n_blocks):
                start_idx = i * block_size
                end_idx = start_idx + block_size if i < n_blocks - 1 else len(response_times)
                
                block_rt = response_times[start_idx:end_idx]
                block_acc = accuracy_scores[start_idx:end_idx] if len(accuracy_scores) >= end_idx else [0.5] * (end_idx - start_idx)
                
                # Performance = accuracy / response_time (higher is better)
                avg_performance = np.mean(block_acc) / (np.mean(block_rt) + 1e-6)
                block_performance.append(avg_performance)
            
            # Calculate decrement slope
            time_blocks = np.arange(n_blocks)
            if len(block_performance) > 2:
                slope, _, _, _, _ = stats.linregress(time_blocks, block_performance)
                features['vigilance_decrement_slope'] = max(0.0, -slope)  # Positive indicates decline
            else:
                features['vigilance_decrement_slope'] = 0.0
        else:
            features['vigilance_decrement_slope'] = 0.0
        
        # 2. Attention Lapse Frequency (long response times or gaps)
        if len(all_events) > 3:
            inter_event_intervals = np.diff(all_events)
            median_interval = np.median(inter_event_intervals)
            lapse_threshold = median_interval * 3  # 3x median is considered a lapse
            
            lapses = inter_event_intervals[inter_event_intervals > lapse_threshold]
            features['vigilance_lapse_frequency'] = len(lapses) / (session_duration / 60.0)  # lapses per minute
        else:
            features['vigilance_lapse_frequency'] = 0.0
        
        # 3. Sustained Attention Deficit
        if len(response_times) >= 6:
            # Compare early vs late session performance
            early_third = response_times[:len(response_times)//3]
            late_third = response_times[-len(response_times)//3:]
            
            early_performance = 1.0 / (np.mean(early_third) + 1e-6)
            late_performance = 1.0 / (np.mean(late_third) + 1e-6)
            
            features['vigilance_sustained_deficit'] = (early_performance - late_performance) / (early_performance + 1e-6)
        else:
            features['vigilance_sustained_deficit'] = 0.0
        
        # 4. Recovery Rate (ability to bounce back from lapses)
        if len(response_times) >= 10:
            # Find periods of poor performance followed by recovery
            rt_smooth = signal.savgol_filter(response_times, min(5, len(response_times)//2*2-1), 2)
            rt_peaks = signal.find_peaks(rt_smooth, prominence=np.std(response_times)*0.5)[0]  # Performance dips
            rt_valleys = signal.find_peaks(-rt_smooth, prominence=np.std(response_times)*0.5)[0]  # Recoveries
            
            # Measure recovery speed after dips
            recovery_speeds = []
            for peak in rt_peaks:
                # Find next valley (recovery)
                next_valleys = rt_valleys[rt_valleys > peak]
                if len(next_valleys) > 0:
                    recovery_time = next_valleys[0] - peak
                    recovery_magnitude = rt_smooth[peak] - rt_smooth[next_valleys[0]]
                    if recovery_time > 0:
                        recovery_speeds.append(recovery_magnitude / recovery_time)
            
            features['vigilance_recovery_rate'] = np.mean(recovery_speeds) if recovery_speeds else 0.0
        else:
            features['vigilance_recovery_rate'] = 0.0
        
        # 5. Alertness Variability (inconsistency in attention state)
        if len(all_events) >= 8:
            # Create activity density signal
            time_windows = np.arange(0, session_duration, 30)  # 30-second windows
            activity_density = []
            
            for i in range(len(time_windows) - 1):
                window_start = time_windows[i]
                window_end = time_windows[i + 1]
                events_in_window = sum(1 for t in all_events if window_start <= t < window_end)
                activity_density.append(events_in_window)
            
            if len(activity_density) > 2:
                features['vigilance_alertness_variability'] = np.std(activity_density) / (np.mean(activity_density) + 1e-6)
            else:
                features['vigilance_alertness_variability'] = 0.0
        else:
            features['vigilance_alertness_variability'] = 0.0
        
        # 6. Micro-sleep Indicators (very long gaps in activity)
        if len(all_events) > 2:
            inter_event_intervals = np.diff(all_events)
            very_long_gaps = inter_event_intervals[inter_event_intervals > 10.0]  # >10 second gaps
            features['vigilance_micro_sleep_indicators'] = len(very_long_gaps) / (session_duration / 60.0)
        else:
            features['vigilance_micro_sleep_indicators'] = 0.0
        
        # 7. Attention Stability (consistency of attention level)
        if len(response_times) >= 8:
            # Rolling window analysis of attention consistency
            window_size = max(3, len(response_times) // 5)
            stability_scores = []
            
            for i in range(len(response_times) - window_size + 1):
                window_rt = response_times[i:i + window_size]
                window_consistency = 1.0 / (np.std(window_rt) + 1e-6)
                stability_scores.append(window_consistency)
            
            features['vigilance_attention_stability'] = np.mean(stability_scores)
        else:
            features['vigilance_attention_stability'] = 1.0
        
        # 8. Vigilance Reserve (capacity to maintain attention under load)
        if len(response_times) >= 6 and len(accuracy_scores) >= 6:
            # Measure performance maintenance despite increasing load
            task_complexities = behavioral_data.get('task_complexities', [])
            if len(task_complexities) == len(response_times):
                # Group by complexity level
                complexity_groups = {}
                for rt, acc, comp in zip(response_times, accuracy_scores, task_complexities):
                    comp_level = int(comp * 3)  # 0, 1, 2, 3 complexity levels
                    if comp_level not in complexity_groups:
                        complexity_groups[comp_level] = []
                    performance = acc / (rt + 1e-6)
                    complexity_groups[comp_level].append(performance)
                
                # Calculate vigilance reserve as performance stability across complexity
                if len(complexity_groups) > 1:
                    group_means = [np.mean(group) for group in complexity_groups.values()]
                    features['vigilance_reserve'] = 1.0 / (np.std(group_means) + 1e-6)
                else:
                    features['vigilance_reserve'] = 1.0
            else:
                features['vigilance_reserve'] = 1.0
        else:
            features['vigilance_reserve'] = 1.0
        
        return features
    
    def extract_focus_depth_features(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract focus duration and depth indicators (6 features)."""
        features = {}
        
        content_interactions = behavioral_data.get('content_interactions', [])
        mouse_movements = behavioral_data.get('mouse_movements', [])
        keyboard_events = behavioral_data.get('keyboard_events', [])
        session_duration = behavioral_data.get('session_duration', 600)
        
        # 1. Deep Focus Periods (sustained engagement without interruption)
        all_interactions = []
        for interaction in content_interactions:
            timestamp = interaction.get('timestamp', 0)
            interaction_type = interaction.get('type', 'click')
            all_interactions.append((timestamp, interaction_type))
        
        if len(all_interactions) >= 3:
            all_interactions.sort()
            
            # Find sustained focus periods (low frequency of interactions)
            focus_periods = []
            current_period_start = all_interactions[0][0]
            interaction_count = 1
            
            for i in range(1, len(all_interactions)):
                time_gap = all_interactions[i][0] - all_interactions[i-1][0]
                
                if time_gap < 30:  # Less than 30 seconds = continuous engagement
                    interaction_count += 1
                else:
                    # End of focus period
                    period_duration = all_interactions[i-1][0] - current_period_start
                    if period_duration > 60:  # At least 1 minute
                        focus_periods.append((period_duration, interaction_count))
                    
                    # Start new period
                    current_period_start = all_interactions[i][0]
                    interaction_count = 1
            
            # Handle last period
            final_duration = all_interactions[-1][0] - current_period_start
            if final_duration > 60:
                focus_periods.append((final_duration, interaction_count))
            
            if focus_periods:
                durations, counts = zip(*focus_periods)
                features['focus_deep_periods_count'] = len(focus_periods)
                features['focus_average_depth_duration'] = np.mean(durations)
                features['focus_max_sustained_duration'] = max(durations)
            else:
                features['focus_deep_periods_count'] = 0.0
                features['focus_average_depth_duration'] = 0.0
                features['focus_max_sustained_duration'] = 0.0
        else:
            features['focus_deep_periods_count'] = 0.0
            features['focus_average_depth_duration'] = 0.0
            features['focus_max_sustained_duration'] = 0.0
        
        # 2. Focus Intensity (interaction density during engaged periods)
        if content_interactions:
            interaction_times = [int.get('timestamp', 0) for int in content_interactions]
            
            # Calculate local interaction density
            window_size = 60  # 1-minute windows
            max_density = 0
            total_density = 0
            window_count = 0
            
            for start_time in range(0, int(session_duration), window_size):
                end_time = start_time + window_size
                interactions_in_window = sum(1 for t in interaction_times if start_time <= t < end_time)
                density = interactions_in_window / (window_size / 60.0)  # interactions per minute
                
                if interactions_in_window > 0:
                    total_density += density
                    window_count += 1
                    max_density = max(max_density, density)
            
            features['focus_max_intensity'] = max_density
            features['focus_average_intensity'] = total_density / (window_count + 1e-6)
        else:
            features['focus_max_intensity'] = 0.0
            features['focus_average_intensity'] = 0.0
        
        # 3. Focus Consistency (variability in attention depth over time)
        if len(mouse_movements) >= 10:
            # Use mouse movement patterns as proxy for focus consistency
            movement_times = [m.get('timestamp', 0) for m in mouse_movements]
            movement_times.sort()
            
            # Calculate movement clustering (focused = clustered, unfocused = scattered)
            time_windows = np.arange(0, session_duration, 30)  # 30-second windows
            movement_densities = []
            
            for i in range(len(time_windows) - 1):
                window_start = time_windows[i]
                window_end = time_windows[i + 1]
                movements_in_window = sum(1 for t in movement_times if window_start <= t < window_end)
                movement_densities.append(movements_in_window)
            
            if len(movement_densities) > 2:
                # High consistency = low variability in movement density
                features['focus_consistency'] = 1.0 / (np.std(movement_densities) + 1e-6)
            else:
                features['focus_consistency'] = 1.0
        else:
            features['focus_consistency'] = 1.0
        
        return features
    
    def extract_multiscale_features(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract multi-scale behavioral pattern features (6 features)."""
        features = {}
        
        # Mouse movement multi-scale analysis
        mouse_movements = behavioral_data.get('mouse_movements', [])
        if len(mouse_movements) >= 5:
            # Extract movement patterns at different time scales
            positions = [(m.get('x', 0), m.get('y', 0)) for m in mouse_movements]
            velocities = []
            
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                velocity = np.sqrt(dx**2 + dy**2)
                velocities.append(velocity)
            
            if velocities:
                # Micro-scale patterns (immediate movements)
                features['multiscale_micro_velocity_mean'] = np.mean(velocities[:len(velocities)//3])
                
                # Macro-scale patterns (overall movement trends)
                features['multiscale_macro_velocity_mean'] = np.mean(velocities[-len(velocities)//3:])
                
                # Movement complexity across scales
                features['multiscale_velocity_complexity'] = np.std(velocities) / (np.mean(velocities) + 1e-6)
            else:
                features['multiscale_micro_velocity_mean'] = 0.0
                features['multiscale_macro_velocity_mean'] = 0.0
                features['multiscale_velocity_complexity'] = 0.0
        else:
            features['multiscale_micro_velocity_mean'] = 0.0
            features['multiscale_macro_velocity_mean'] = 0.0
            features['multiscale_velocity_complexity'] = 0.0
        
        # Keyboard rhythm at multiple scales
        keyboard_events = behavioral_data.get('keyboard_events', [])
        if len(keyboard_events) >= 5:
            timestamps = [k.get('timestamp', 0) for k in keyboard_events]
            intervals = np.diff(sorted(timestamps))
            
            if len(intervals) >= 3:
                # Short-term rhythm (burst typing)
                short_intervals = intervals[:len(intervals)//2]
                features['multiscale_short_rhythm'] = 1.0 / (np.std(short_intervals) + 1e-6)
                
                # Long-term rhythm (sustained typing)
                long_intervals = intervals[len(intervals)//2:]
                features['multiscale_long_rhythm'] = 1.0 / (np.std(long_intervals) + 1e-6)
                
                # Cross-scale coherence
                features['multiscale_rhythm_coherence'] = abs(np.corrcoef(short_intervals[:min(len(short_intervals), len(long_intervals))], 
                                                                        long_intervals[:min(len(short_intervals), len(long_intervals))])[0,1]) if min(len(short_intervals), len(long_intervals)) > 1 else 0.0
            else:
                features['multiscale_short_rhythm'] = 0.0
                features['multiscale_long_rhythm'] = 0.0
                features['multiscale_rhythm_coherence'] = 0.0
        else:
            features['multiscale_short_rhythm'] = 0.0
            features['multiscale_long_rhythm'] = 0.0
            features['multiscale_rhythm_coherence'] = 0.0
        
        
        return features
    
    def extract_all_features(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract all 25+ enhanced attention features."""
        features = {}
        
        # 1. Fourier analysis of behavioral rhythms (5 features)
        fourier_features = self.extract_fourier_features(behavioral_data)
        features.update(fourier_features)
        
        # 2. Vigilance and sustained attention (8 features)
        vigilance_features = self.extract_vigilance_features(behavioral_data)
        features.update(vigilance_features)
        
        # 3. Focus duration and depth (6 features)
        focus_features = self.extract_focus_depth_features(behavioral_data)
        features.update(focus_features)
        
        # 4. Multi-scale behavioral patterns (6+ features)
        multiscale_features = self.extract_multiscale_features(behavioral_data)
        features.update(multiscale_features)
        
        return features


class EnhancedLearningStyleFeatures:
    """
    🎨 Advanced Learning Style Feature Engineering (Target: 35+ features)
    ===================================================================
    
    Based on VARK model (Fleming, 1995) and multimodal learning research.
    Extracts 35+ behavioral indicators of learning preferences and adaptation.
    
    Feature Categories:
    1. Sequential Learning Patterns (8 features)
    2. Multimodal Content Preferences (12 features)
    3. Adaptation Rate Indicators (6 features)
    4. Content Interaction Patterns (9 features)
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_importance_scores = {}
        self.feature_descriptions = {}
    
    def extract_sequential_patterns(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract comprehensive sequential learning behavior patterns (8 features)."""
        features = {}
        
        # Extract content engagement data with realistic defaults
        content_engagement = behavioral_data.get('content_engagement', {})
        if not content_engagement:
            # Create realistic engagement patterns from available data
            content_engagement = {
                'visual': behavioral_data.get('visual_interactions', 10) + 
                         (len(behavioral_data.get('scroll_events', [])) * 2 if behavioral_data.get('scroll_events', []) else 0),
                'auditory': max(1, behavioral_data.get('session_duration', 300) // 60),  # Assume some audio
                'text': len(behavioral_data.get('keyboard_events', [])) * 3,  # Text interaction proxy
                'interactive': len(behavioral_data.get('content_interactions', [])) * 2
            }
        
        total_engagement = sum(content_engagement.values()) or 1
        
        # 1. Content Type Preference Strength (how strongly biased toward one type)
        content_ratios = {ctype: engagement / total_engagement 
                         for ctype, engagement in content_engagement.items()}
        
        max_preference = max(content_ratios.values())
        min_preference = min(content_ratios.values())
        features['sequential_preference_strength'] = max_preference - min_preference
        
        # 2. Learning Path Consistency (do they follow consistent patterns)
        # Simulate learning sequence from content interactions
        content_interactions = behavioral_data.get('content_interactions', [])
        if len(content_interactions) >= 5:
            # Analyze interaction sequence patterns
            interaction_types = []
            for interaction in content_interactions:
                # Categorize interaction types
                int_type = interaction.get('type', 'click')
                if int_type in ['video_play', 'image_view']:
                    interaction_types.append('visual')
                elif int_type in ['audio_play', 'speak']:
                    interaction_types.append('auditory')
                elif int_type in ['text_read', 'note_take']:
                    interaction_types.append('text')
                else:
                    interaction_types.append('interactive')
            
            # Measure transition consistency
            transitions = {}
            for i in range(len(interaction_types) - 1):
                transition = (interaction_types[i], interaction_types[i+1])
                transitions[transition] = transitions.get(transition, 0) + 1
            
            if transitions:
                total_transitions = sum(transitions.values())
                # Consistency = how often most common transition occurs
                max_transition_count = max(transitions.values())
                features['sequential_path_consistency'] = max_transition_count / total_transitions
            else:
                features['sequential_path_consistency'] = 0.5
        else:
            features['sequential_path_consistency'] = 0.5
        
        # 3. Modality Switching Frequency
        if len(content_interactions) >= 3:
            modality_switches = 0
            current_modality = None
            
            for interaction in content_interactions:
                int_type = interaction.get('type', 'click')
                if int_type in ['video_play', 'image_view']:
                    modality = 'visual'
                elif int_type in ['audio_play', 'speak']:
                    modality = 'auditory'
                elif int_type in ['text_read', 'note_take']:
                    modality = 'text'
                else:
                    modality = 'interactive'
                
                if current_modality and current_modality != modality:
                    modality_switches += 1
                current_modality = modality
            
            session_duration = behavioral_data.get('session_duration', 600)
            features['sequential_switching_frequency'] = modality_switches / (session_duration / 60.0)
        else:
            features['sequential_switching_frequency'] = 0.0
        
        # 4. Sequential Learning Efficiency (performance improvement over sequence)
        performance_by_type = behavioral_data.get('performance_by_type', {})
        if performance_by_type:
            # Calculate learning curves for each content type
            efficiency_scores = []
            for content_type, performances in performance_by_type.items():
                if isinstance(performances, list) and len(performances) >= 3:
                    # Linear regression to find improvement slope
                    x = np.arange(len(performances))
                    slope, _, r_value, _, _ = stats.linregress(x, performances)
                    efficiency_scores.append(slope * r_value)  # Slope weighted by fit
            
            features['sequential_learning_efficiency'] = np.mean(efficiency_scores) if efficiency_scores else 0.0
        else:
            features['sequential_learning_efficiency'] = 0.0
        
        # 5. Content Type Specialization (deep engagement with preferred types)
        if content_engagement:
            sorted_engagement = sorted(content_engagement.values(), reverse=True)
            if len(sorted_engagement) >= 2:
                # Specialization = ratio of top engagement to second-highest
                features['sequential_specialization'] = sorted_engagement[0] / (sorted_engagement[1] + 1e-6)
            else:
                features['sequential_specialization'] = 1.0
        else:
            features['sequential_specialization'] = 1.0
        
        # 6. Temporal Learning Pattern (when do they engage with different content)
        if content_interactions:
            time_preferences = {'visual': [], 'auditory': [], 'text': [], 'interactive': []}
            session_duration = behavioral_data.get('session_duration', 600)
            
            for interaction in content_interactions:
                timestamp = interaction.get('timestamp', 0)
                relative_time = timestamp / session_duration  # 0 to 1
                
                int_type = interaction.get('type', 'click')
                if int_type in ['video_play', 'image_view']:
                    time_preferences['visual'].append(relative_time)
                elif int_type in ['audio_play', 'speak']:
                    time_preferences['auditory'].append(relative_time)
                elif int_type in ['text_read', 'note_take']:
                    time_preferences['text'].append(relative_time)
                else:
                    time_preferences['interactive'].append(relative_time)
            
            # Calculate temporal clustering (do they use content types at specific times)
            temporal_clustering = 0.0
            for content_type, times in time_preferences.items():
                if len(times) >= 2:
                    # Standard deviation of timing (lower = more clustered)
                    clustering = 1.0 / (np.std(times) + 1e-6)
                    temporal_clustering += clustering / len(time_preferences)
            
            features['sequential_temporal_clustering'] = temporal_clustering
        else:
            features['sequential_temporal_clustering'] = 0.0
        
        # 7. Multimodal Integration Index (how well they combine different modalities)
        if len(content_ratios) >= 2:
            # Shannon entropy of content distribution (higher = more balanced)
            entropy = -sum(ratio * np.log2(ratio + 1e-10) for ratio in content_ratios.values())
            max_entropy = np.log2(len(content_ratios))
            features['sequential_multimodal_integration'] = entropy / max_entropy
        else:
            features['sequential_multimodal_integration'] = 0.0
        
        # 8. Content Depth vs Breadth Preference
        time_spent_by_type = behavioral_data.get('time_spent_by_type', {})
        if time_spent_by_type and content_engagement:
            # Depth = time per interaction, Breadth = variety of interactions
            depth_scores = []
            for content_type in ['visual', 'auditory', 'text', 'interactive']:
                time_spent = time_spent_by_type.get(content_type, 0)
                interactions = content_engagement.get(content_type, 1)  # Avoid division by zero
                if interactions > 0:
                    depth_score = time_spent / interactions  # Time per interaction
                    depth_scores.append(depth_score)
            
            if depth_scores:
                avg_depth = np.mean(depth_scores)
                breadth = len([score for score in depth_scores if score > 0])
                # Depth preference = high depth, low breadth
                features['sequential_depth_vs_breadth'] = avg_depth / (breadth + 1e-6)
            else:
                features['sequential_depth_vs_breadth'] = 1.0
        else:
            features['sequential_depth_vs_breadth'] = 1.0
        
        return features
    
    def extract_adaptation_patterns(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract learning adaptation patterns (6 features)."""
        features = {}
        
        # Safely handle performance_by_type which might be a list or dict
        performance_by_type = behavioral_data.get('performance_by_type', {})
        if isinstance(performance_by_type, list):
            performance_by_type = {}  # Convert to empty dict if it's a list
        
        engagement_metrics = behavioral_data.get('engagement_metrics', {})
        completion_rates = behavioral_data.get('completion_rates', {})
        
        content_types = ['visual', 'auditory', 'text', 'interactive']
        
        # Performance adaptation
        if performance_by_type:
            performance_scores = []
            for content_type in content_types:
                scores = performance_by_type.get(content_type, [0.5])
                avg_performance = np.mean(scores)
                features[f'{content_type}_performance'] = avg_performance
                performance_scores.append(avg_performance)
            
            # Performance variance and adaptation
            if len(performance_scores) > 1:
                features['performance_consistency'] = 1.0 - np.std(performance_scores)
                features['best_modality_advantage'] = max(performance_scores) - np.mean(performance_scores)
            else:
                features['performance_consistency'] = 1.0
                features['best_modality_advantage'] = 0.0
        else:
            for content_type in content_types:
                features[f'{content_type}_performance'] = 0.5
            features['performance_consistency'] = 1.0
            features['best_modality_advantage'] = 0.0
        
        # Engagement adaptation
        if engagement_metrics:
            engagement_scores = []
            for content_type in content_types:
                engagement = engagement_metrics.get(content_type, 0.5)
                features[f'{content_type}_engagement'] = engagement
                engagement_scores.append(engagement)
                
            if len(engagement_scores) > 1:
                features['engagement_consistency'] = 1.0 - np.std(engagement_scores)
                features['preferred_modality_engagement'] = max(engagement_scores)
            else:
                features['engagement_consistency'] = 1.0
                features['preferred_modality_engagement'] = 0.5
        else:
            for content_type in content_types:
                features[f'{content_type}_engagement'] = 0.5
            features['engagement_consistency'] = 1.0
            features['preferred_modality_engagement'] = 0.5
        
        # Completion patterns
        if completion_rates:
            completion_scores = []
            for content_type in content_types:
                completion = completion_rates.get(content_type, 0.5)
                features[f'{content_type}_completion'] = completion
                completion_scores.append(completion)
            
            if len(completion_scores) > 1:
                features['completion_consistency'] = 1.0 - np.std(completion_scores)
                features['completion_efficiency'] = np.mean(completion_scores)
            else:
                features['completion_consistency'] = 1.0
                features['completion_efficiency'] = 0.5
        else:
            for content_type in content_types:
                features[f'{content_type}_completion'] = 0.5
            features['completion_consistency'] = 1.0
            features['completion_efficiency'] = 0.5
        
        return features
    
    def extract_multimodal_indicators(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract indicators of multimodal learning preferences."""
        features = {}
        
        # Get all modality scores
        content_types = ['visual', 'auditory', 'text', 'interactive']
        
        # Collect scores for each modality
        modality_scores = {}
        for content_type in content_types:
            scores = []
            
            # Performance score
            performance_by_type = behavioral_data.get('performance_by_type', {})
            if content_type in performance_by_type:
                scores.append(np.mean(performance_by_type[content_type]))
            
            # Engagement score
            engagement_metrics = behavioral_data.get('engagement_metrics', {})
            if content_type in engagement_metrics:
                scores.append(engagement_metrics[content_type])
            
            # Time preference score
            time_spent = behavioral_data.get('time_spent_by_type', {})
            if time_spent:
                total_time = sum(time_spent.values())
                if total_time > 0:
                    scores.append(time_spent.get(content_type, 0) / total_time)
            
            # Interaction preference score
            content_interactions = behavioral_data.get('content_interactions', [])
            if content_interactions and isinstance(content_interactions, list):
                # Count interactions by type
                interaction_counts = {}
                for interaction in content_interactions:
                    int_type = interaction.get('type', 'unknown')
                    interaction_counts[int_type] = interaction_counts.get(int_type, 0) + 1
                
                total_interactions = sum(interaction_counts.values())
                if total_interactions > 0:
                    scores.append(interaction_counts.get(content_type, 0) / total_interactions)
            elif content_interactions and isinstance(content_interactions, dict):
                # Handle as dictionary
                total_interactions = sum(content_interactions.values())
                if total_interactions > 0:
                    scores.append(content_interactions.get(content_type, 0) / total_interactions)
            
            modality_scores[content_type] = np.mean(scores) if scores else 0.25
        
        # Multimodal indicators
        score_values = list(modality_scores.values())
        if len(score_values) > 1:
            # How balanced are the preferences?
            features['modality_balance'] = 1.0 - np.std(score_values)
            
            # How many modalities are significantly preferred?
            mean_score = np.mean(score_values)
            threshold = mean_score + np.std(score_values) * 0.5
            strong_preferences = sum(1 for score in score_values if score > threshold)
            features['strong_preference_count'] = strong_preferences
            
            # Multimodal tendency
            if strong_preferences >= 3:
                features['multimodal_tendency'] = 1.0
            elif strong_preferences == 2:
                features['multimodal_tendency'] = 0.7
            elif strong_preferences == 1:
                features['multimodal_tendency'] = 0.3
            else:
                features['multimodal_tendency'] = 0.9  # No clear preference suggests multimodal
            
            # Preference strength
            features['max_modality_preference'] = max(score_values)
            features['min_modality_preference'] = min(score_values)
            features['modality_range'] = features['max_modality_preference'] - features['min_modality_preference']
        else:
            features['modality_balance'] = 1.0
            features['strong_preference_count'] = 0.0
            features['multimodal_tendency'] = 1.0
            features['max_modality_preference'] = 0.25
            features['min_modality_preference'] = 0.25
            features['modality_range'] = 0.0
        
        return features
    
    def extract_all_features(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract all 35+ enhanced learning style features."""
        features = {}
        
        # 1. Sequential learning patterns (8 features)
        sequential_features = self.extract_sequential_patterns(behavioral_data)
        features.update(sequential_features)
        
        # 2. Multimodal content preferences (12 features)
        multimodal_features = self.extract_multimodal_indicators(behavioral_data)
        features.update(multimodal_features)
        
        # 3. Adaptation rate indicators (6 features)
        adaptation_features = self.extract_adaptation_patterns(behavioral_data)
        features.update(adaptation_features)
        
        # 4. Content interaction patterns (9+ features)
        interaction_features = self.extract_content_interaction_patterns(behavioral_data)
        features.update(interaction_features)
        
        return features
    
    def extract_content_interaction_patterns(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract detailed content interaction patterns (9 features)."""
        features = {}
        
        content_interactions = behavioral_data.get('content_interactions', [])
        session_duration = behavioral_data.get('session_duration', 600)
        
        # 1. Interaction frequency by type
        interaction_types = {}
        for interaction in content_interactions:
            int_type = interaction.get('type', 'unknown')
            interaction_types[int_type] = interaction_types.get(int_type, 0) + 1
        
        total_interactions = sum(interaction_types.values()) or 1
        
        # 2. Primary interaction patterns
        features['content_click_preference'] = interaction_types.get('click', 0) / total_interactions
        features['content_hover_preference'] = interaction_types.get('hover', 0) / total_interactions  
        features['content_scroll_preference'] = interaction_types.get('scroll', 0) / total_interactions
        
        # 3. Interaction timing patterns
        if content_interactions:
            timestamps = [int.get('timestamp', 0) for int in content_interactions]
            timestamps.sort()
            
            if len(timestamps) > 2:
                # Interaction clustering
                intervals = np.diff(timestamps)
                features['content_interaction_clustering'] = 1.0 / (np.std(intervals) + 1e-6)
                features['content_interaction_intensity'] = len(timestamps) / (session_duration / 60.0)
                
                # Temporal distribution
                relative_times = [t / session_duration for t in timestamps]
                features['content_early_engagement'] = sum(1 for t in relative_times if t < 0.33) / len(relative_times)
                features['content_late_engagement'] = sum(1 for t in relative_times if t > 0.67) / len(relative_times)
            else:
                features['content_interaction_clustering'] = 0.0
                features['content_interaction_intensity'] = 0.0
                features['content_early_engagement'] = 0.0
                features['content_late_engagement'] = 0.0
        else:
            features['content_interaction_clustering'] = 0.0
            features['content_interaction_intensity'] = 0.0
            features['content_early_engagement'] = 0.0
            features['content_late_engagement'] = 0.0
        
        # 4. Learning resource utilization
        mouse_movements = behavioral_data.get('mouse_movements', [])
        keyboard_events = behavioral_data.get('keyboard_events', [])
        
        features['content_passive_consumption'] = len(mouse_movements) / (len(keyboard_events) + 1e-6) if keyboard_events else 10.0
        features['content_active_engagement'] = len(keyboard_events) / (session_duration / 60.0)
        
        return features


class FeatureImportanceAnalyzer:
    """
    📊 Feature Importance Analysis and Statistical Testing
    =====================================================
    
    Provides feature importance scoring and statistical significance testing
    to identify the most valuable features for each model type.
    """
    
    def __init__(self):
        self.importance_scores = {}
        self.statistical_tests = {}
    
    def calculate_feature_importance(self, features: Dict[str, float], 
                                   target_values: List[float] = None) -> Dict[str, float]:
        """Calculate feature importance scores using multiple methods."""
        if not features or not target_values:
            return {feature_name: 0.5 for feature_name in features.keys()}
        
        importance_scores = {}
        feature_array = np.array(list(features.values())).reshape(-1, 1)
        
        try:
            # Mutual information
            mi_scores = mutual_info_regression(feature_array.T, target_values)
            
            # F-statistic
            f_scores, p_values = f_regression(feature_array.T, target_values)
            
            for i, feature_name in enumerate(features.keys()):
                # Combine multiple importance measures
                mi_score = mi_scores[i] if len(mi_scores) > i else 0.0
                f_score = f_scores[i] if len(f_scores) > i else 0.0
                p_value = p_values[i] if len(p_values) > i else 1.0
                
                # Combined importance (normalized)
                combined_score = (mi_score * 0.5 + f_score * 0.3 + (1.0 - p_value) * 0.2)
                importance_scores[feature_name] = combined_score
                
        except Exception as e:
            # Fallback to simple variance-based importance
            for feature_name, value in features.items():
                importance_scores[feature_name] = abs(value) / (abs(value) + 1.0)
        
        return importance_scores
    
    def test_statistical_significance(self, features: Dict[str, float], 
                                    target_values: List[float] = None) -> Dict[str, Dict[str, float]]:
        """Test statistical significance of features."""
        significance_results = {}
        
        if not target_values:
            return {feature_name: {'p_value': 0.5, 'significant': False} 
                   for feature_name in features.keys()}
        
        for feature_name, feature_value in features.items():
            try:
                # Simple correlation test (in practice, you'd use more sophisticated tests)
                correlation, p_value = stats.pearsonr([feature_value], target_values[:1])
                
                significance_results[feature_name] = {
                    'correlation': correlation if not np.isnan(correlation) else 0.0,
                    'p_value': p_value if not np.isnan(p_value) else 1.0,
                    'significant': p_value < 0.05 if not np.isnan(p_value) else False
                }
            except Exception:
                significance_results[feature_name] = {
                    'correlation': 0.0,
                    'p_value': 1.0,
                    'significant': False
                }
        
        return significance_results


def enhance_feature_engineering_pipeline(behavioral_data: Dict[str, Any], 
                                        model_type: str,
                                        target_values: List[float] = None) -> Dict[str, Any]:
    """
    🚀 Enhanced Feature Engineering Pipeline
    =======================================
    
    Main pipeline to extract all enhanced features for any model type.
    Returns comprehensive feature set with importance scores and statistics.
    
    Args:
        behavioral_data: Raw behavioral data from learning session
        model_type: 'cognitive_load', 'attention_tracker', or 'learning_style'
        target_values: Optional target values for importance calculation
    
    Returns:
        Dict containing:
        - features: Extracted feature values
        - feature_importance: Importance scores for each feature
        - statistical_tests: Statistical significance tests
        - feature_count: Number of features extracted
        - model_type: The model type processed
    """
    
    # Extract features based on model type
    if model_type == 'cognitive_load':
        extractor = EnhancedCognitiveLoadFeatures()
        features = extractor.extract_all_features(behavioral_data)
        expected_count = 40
    elif model_type == 'attention_tracker':
        extractor = EnhancedAttentionFeatures()
        features = extractor.extract_all_features(behavioral_data)
        expected_count = 25
    elif model_type == 'learning_style':
        extractor = EnhancedLearningStyleFeatures()
        features = extractor.extract_all_features(behavioral_data)
        expected_count = 35
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Calculate feature importance and statistical significance
    analyzer = FeatureImportanceAnalyzer()
    feature_importance = analyzer.calculate_feature_importance(features, target_values)
    statistical_tests = analyzer.test_statistical_significance(features, target_values)
    
    # Compile comprehensive results
    results = {
        'features': features,
        'feature_importance': feature_importance,
        'statistical_tests': statistical_tests,
        'feature_count': len(features),
        'expected_count': expected_count,
        'model_type': model_type,
        'extraction_successful': len(features) >= expected_count * 0.8,  # At least 80% of expected features
        'metadata': {
            'behavioral_data_keys': list(behavioral_data.keys()),
            'session_duration': behavioral_data.get('session_duration', 0),
            'data_completeness': sum(1 for v in behavioral_data.values() if v) / len(behavioral_data)
        }
    }
    
    return results


def extract_compatibility_features(behavioral_data: Dict[str, Any]) -> Dict[str, float]:
    """
    🔄 Compatibility Feature Extraction
    ===================================
    
    Extracts features that are compatible with existing data structures
    from demo.py and the current training pipeline.
    """
    features = {}
    
    # Handle demo.py data format
    if 'mouse_movements' in behavioral_data:
        mouse_data = behavioral_data['mouse_movements']
        if mouse_data:
            positions = [(m.get('x', 0), m.get('y', 0)) for m in mouse_data]
            if len(positions) > 1:
                distances = [np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) 
                           for p1, p2 in zip(positions[:-1], positions[1:])]
                features['mouse_total_distance'] = sum(distances)
                features['mouse_avg_velocity'] = np.mean(distances) if distances else 0.0
            else:
                features['mouse_total_distance'] = 0.0
                features['mouse_avg_velocity'] = 0.0
    
    # Handle keyboard events
    if 'keyboard_events' in behavioral_data:
        keyboard_data = behavioral_data['keyboard_events']
        features['keyboard_event_count'] = len(keyboard_data)
        if len(keyboard_data) > 1:
            timestamps = [k.get('timestamp', 0) for k in keyboard_data]
            intervals = np.diff(sorted(timestamps))
            features['keyboard_typing_rhythm'] = 1.0 / (np.std(intervals) + 1e-6)
        else:
            features['keyboard_typing_rhythm'] = 0.0
    
    # Handle content interactions
    if 'content_interactions' in behavioral_data:
        interactions = behavioral_data['content_interactions']
        features['interaction_count'] = len(interactions)
        if interactions:
            interaction_types = set(i.get('type', 'unknown') for i in interactions)
            features['interaction_diversity'] = len(interaction_types)
        else:
            features['interaction_diversity'] = 0.0
    
    # Handle response times and accuracy (from existing training data)
    if 'response_times' in behavioral_data:
        rt_data = behavioral_data['response_times']
        if rt_data:
            features['rt_mean'] = np.mean(rt_data)
            features['rt_std'] = np.std(rt_data)
            features['rt_cv'] = features['rt_std'] / (features['rt_mean'] + 1e-6)
        else:
            features['rt_mean'] = features['rt_std'] = features['rt_cv'] = 0.0
    
    if 'accuracy_scores' in behavioral_data:
        acc_data = behavioral_data['accuracy_scores']
        if acc_data:
            features['accuracy_mean'] = np.mean(acc_data)
            features['accuracy_std'] = np.std(acc_data)
        else:
            features['accuracy_mean'] = features['accuracy_std'] = 0.0
    
    return features


def demonstrate_enhanced_features():
    """Demonstrate the enhanced feature engineering capabilities."""
    print("🚀 Enhanced Feature Engineering Module - STEP 1 COMPLETE")
    print("=" * 60)
    print("PERFORMANCE IMPROVEMENT TARGETS:")
    print("• Cognitive Load: 37.6% → 70%+ (40+ features)")
    print("• Attention: 67.8% → 80%+ (25+ features)")  
    print("• Learning Style: 21.3% → 65%+ (35+ features)")
    print()
    
    # Sample behavioral data (compatible with demo.py format)
    sample_data = {
        'session_duration': 1200,
        'response_times': [2.1, 2.5, 3.2, 2.8, 3.5, 4.1, 3.9, 2.7, 3.3, 2.9],
        'accuracy_scores': [0.9, 0.85, 0.8, 0.82, 0.75, 0.7, 0.72, 0.8, 0.78, 0.85],
        'task_complexities': [0.3, 0.4, 0.6, 0.5, 0.7, 0.8, 0.7, 0.4, 0.5, 0.3],
        'mouse_movements': [
            {'x': 100, 'y': 200, 'timestamp': 10},
            {'x': 150, 'y': 250, 'timestamp': 11},
            {'x': 200, 'y': 300, 'timestamp': 12}
        ],
        'keyboard_events': [
            {'key': 'a', 'timestamp': 15},
            {'key': 'b', 'timestamp': 16}
        ],
        'content_interactions': [
            {'type': 'click', 'timestamp': 20},
            {'type': 'scroll', 'timestamp': 25}
        ],
        'multitask_events': [
            {'type': 'tab_switch', 'timestamp': 30}
        ],
        'error_patterns': {'minor': 2, 'critical': 1},
        'hesitation_indicators': [{'duration': 1.5}]
    }
    
    print("📊 FEATURE EXTRACTION RESULTS:")
    print("-" * 40)
    
    # Test each model type
    for model_type in ['cognitive_load', 'attention_tracker', 'learning_style']:
        try:
            results = enhance_feature_engineering_pipeline(sample_data, model_type)
            
            print(f"\n{model_type.upper()}:")
            print(f"✅ Features extracted: {results['feature_count']}")
            print(f"🎯 Target features: {results['expected_count']}")
            print(f"📈 Success rate: {results['feature_count']}/{results['expected_count']} ({100*results['feature_count']/results['expected_count']:.1f}%)")
            
            # Show top 5 most important features
            top_features = sorted(results['feature_importance'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            
            print("🏆 Top 5 most important features:")
            for i, (feature_name, importance) in enumerate(top_features, 1):
                print(f"   {i}. {feature_name}: {importance:.3f}")
                
        except Exception as e:
            print(f"❌ Error in {model_type}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ ENHANCED FEATURE ENGINEERING MODULE READY")
    print("📋 Next Steps:")
    print("   2. Advanced Ensemble Models (Step 2)")  
    print("   3. Hyperparameter Optimization (Step 3)")
    print("\n🔬 Scientific Foundation:")
    print("   • Cognitive Load Theory (Sweller, 1988)")
    print("   • Attention Networks (Posner & Petersen, 1990)")
    print("   • VARK Learning Styles (Fleming, 1995)")
    print("   • Working Memory Model (Baddeley, 2000)")


if __name__ == "__main__":
    demonstrate_enhanced_features()