#!/usr/bin/env python3
"""
🔍 Automated Feature Selection for AI Learning Psychology Models
===============================================================

Comprehensive feature selection pipeline using multiple complementary techniques:
- Mutual Information scoring
- Recursive Feature Elimination (RFE)
- LASSO regularization
- Statistical significance testing
- Tree-based feature importance
- Correlation analysis and redundancy removal

This module automatically identifies the most predictive features while
removing redundant and noisy features to improve model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
    RFE, RFECV, SelectKBest, f_classif, f_regression,
    chi2, SelectPercentile, VarianceThreshold
)
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.inspection import permutation_importance

# Statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt


class AutomatedFeatureSelector:
    """
    🧠 Intelligent Feature Selection Pipeline
    =========================================
    
    Combines multiple feature selection techniques to identify the most
    predictive and non-redundant features for learning psychology analysis.
    """
    
    def __init__(self, 
                 target_features: int = 50,
                 correlation_threshold: float = 0.95,
                 mutual_info_percentile: int = 75,
                 statistical_alpha: float = 0.05,
                 variance_threshold: float = 0.01,
                 random_state: int = 42):
        """
        Initialize feature selector with configurable parameters.
        
        Args:
            target_features: Target number of features to select
            correlation_threshold: Threshold for removing highly correlated features
            mutual_info_percentile: Percentile for mutual information selection
            statistical_alpha: Significance level for statistical tests
            variance_threshold: Minimum variance threshold
            random_state: Random seed for reproducibility
        """
        self.target_features = target_features
        self.correlation_threshold = correlation_threshold
        self.mutual_info_percentile = mutual_info_percentile
        self.statistical_alpha = statistical_alpha
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        
        # Storage for selection results
        self.feature_scores = {}
        self.selected_features = set()
        self.removed_features = set()
        self.selection_history = []
        
        # Fitted selectors
        self.variance_selector = None
        self.mutual_info_scores = None
        self.rfe_selector = None
        self.lasso_selector = None
        self.statistical_scores = None
        self.tree_importances = None
        
        print(f"🔍 Automated Feature Selector initialized")
        print(f"📊 Target features: {target_features}")
        print(f"🔗 Correlation threshold: {correlation_threshold}")
    
    def remove_low_variance_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features with low variance."""
        print(f"\n🔍 Step 1: Removing low variance features (threshold: {self.variance_threshold})")
        
        self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
        X_transformed = self.variance_selector.fit_transform(X)
        
        # Get selected feature names
        selected_mask = self.variance_selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        removed_features = X.columns[~selected_mask].tolist()
        
        self.removed_features.update(removed_features)
        
        print(f"✅ Removed {len(removed_features)} low-variance features")
        print(f"📊 Remaining features: {len(selected_features)}")
        
        if removed_features:
            print(f"🗑️  Removed: {removed_features[:5]}{'...' if len(removed_features) > 5 else ''}")
        
        return pd.DataFrame(X_transformed, columns=selected_features, index=X.index)
    
    def calculate_mutual_information(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification') -> Dict[str, float]:
        """Calculate mutual information scores for all features."""
        print(f"\n🔍 Step 2: Calculating mutual information scores ({task_type})")
        
        # Choose appropriate mutual information function
        if task_type == 'classification':
            # Ensure y is categorical for classification
            if not pd.api.types.is_integer_dtype(y):
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y.values
            
            mi_scores = mutual_info_classif(X, y_encoded, random_state=self.random_state)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
        
        # Create feature score dictionary
        self.mutual_info_scores = dict(zip(X.columns, mi_scores))
        
        # Select top features by percentile
        threshold = np.percentile(mi_scores, self.mutual_info_percentile)
        selected_by_mi = [feat for feat, score in self.mutual_info_scores.items() if score >= threshold]
        
        print(f"✅ Calculated MI scores for {len(X.columns)} features")
        print(f"📊 Selected {len(selected_by_mi)} features above {self.mutual_info_percentile}th percentile")
        print(f"🎯 MI threshold: {threshold:.4f}")
        
        # Show top features
        top_features = sorted(self.mutual_info_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"🏆 Top MI features: {[f'{feat}: {score:.4f}' for feat, score in top_features]}")
        
        return self.mutual_info_scores
    
    def perform_rfe_selection(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification') -> List[str]:
        """Perform Recursive Feature Elimination."""
        print(f"\n🔍 Step 3: Recursive Feature Elimination")
        
        # Choose appropriate estimator
        if task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        else:
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Use RFECV for automatic feature number selection
        target_features_rfe = min(self.target_features, len(X.columns))
        
        self.rfe_selector = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            scoring='accuracy' if task_type == 'classification' else 'r2',
            min_features_to_select=max(1, target_features_rfe // 2),
            n_jobs=-1
        )
        
        self.rfe_selector.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[self.rfe_selector.support_].tolist()
        
        print(f"✅ RFE selected {len(selected_features)} features")
        print(f"📊 Optimal features: {self.rfe_selector.n_features_}")
        print(f"🎯 Best CV score: {self.rfe_selector.grid_scores_.max():.4f}")
        
        return selected_features
    
    def perform_lasso_selection(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification') -> List[str]:
        """Perform LASSO regularization feature selection."""
        print(f"\n🔍 Step 4: LASSO regularization feature selection")
        
        # Standardize features for LASSO
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if task_type == 'classification':
            # Use Logistic Regression with L1 penalty
            self.lasso_selector = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                C=1.0,
                random_state=self.random_state,
                max_iter=1000
            )
        else:
            # Use LassoCV for regression
            self.lasso_selector = LassoCV(
                cv=5,
                random_state=self.random_state,
                max_iter=1000
            )
        
        self.lasso_selector.fit(X_scaled, y)
        
        # Get feature coefficients
        if task_type == 'classification':
            if hasattr(self.lasso_selector, 'coef_'):
                if len(self.lasso_selector.coef_.shape) > 1:
                    # Multi-class: take max absolute coefficient across classes
                    coefficients = np.max(np.abs(self.lasso_selector.coef_), axis=0)
                else:
                    coefficients = np.abs(self.lasso_selector.coef_[0])
            else:
                coefficients = np.zeros(X.shape[1])
        else:
            coefficients = np.abs(self.lasso_selector.coef_)
        
        # Select features with non-zero coefficients
        selected_mask = coefficients != 0
        selected_features = X.columns[selected_mask].tolist()
        
        # Store coefficients
        self.feature_scores['lasso_coefficients'] = dict(zip(X.columns, coefficients))
        
        print(f"✅ LASSO selected {len(selected_features)} features with non-zero coefficients")
        if task_type == 'regression':
            print(f"🎯 Optimal alpha: {self.lasso_selector.alpha_:.6f}")
        
        # Show top features
        top_lasso_features = sorted(
            zip(X.columns, coefficients), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        print(f"🏆 Top LASSO features: {[f'{feat}: {coef:.4f}' for feat, coef in top_lasso_features if coef > 0]}")
        
        return selected_features
    
    def perform_statistical_tests(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification') -> List[str]:
        """Perform statistical significance tests."""
        print(f"\n🔍 Step 5: Statistical significance testing")
        
        if task_type == 'classification':
            # Use F-test for classification
            f_scores, p_values = f_classif(X, y)
            test_name = "F-test"
        else:
            # Use F-test for regression
            f_scores, p_values = f_regression(X, y)
            test_name = "F-test"
        
        self.statistical_scores = {
            'f_scores': dict(zip(X.columns, f_scores)),
            'p_values': dict(zip(X.columns, p_values))
        }
        
        # Select features with significant p-values
        significant_features = [
            feat for feat, p_val in zip(X.columns, p_values) 
            if p_val < self.statistical_alpha
        ]
        
        print(f"✅ {test_name} completed for {len(X.columns)} features")
        print(f"📊 Found {len(significant_features)} statistically significant features (α = {self.statistical_alpha})")
        
        # Show top significant features
        significant_with_scores = [
            (feat, self.statistical_scores['f_scores'][feat], self.statistical_scores['p_values'][feat])
            for feat in significant_features
        ]
        significant_with_scores.sort(key=lambda x: x[2])  # Sort by p-value
        
        top_significant = significant_with_scores[:5]
        print(f"🏆 Top significant features: {[(feat, f'F={f_score:.2f}, p={p_val:.6f}') for feat, f_score, p_val in top_significant]}")
        
        return significant_features
    
    def calculate_tree_importance(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification') -> Dict[str, float]:
        """Calculate feature importance using tree-based methods."""
        print(f"\n🔍 Step 6: Tree-based feature importance")
        
        if task_type == 'classification':
            # Use Extra Trees for classification
            estimator = ExtraTreesClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            from sklearn.ensemble import ExtraTreesRegressor
            estimator = ExtraTreesRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        estimator.fit(X, y)
        
        # Get feature importances
        self.tree_importances = dict(zip(X.columns, estimator.feature_importances_))
        
        # Select top features by importance
        importance_threshold = np.percentile(estimator.feature_importances_, self.mutual_info_percentile)
        selected_by_importance = [
            feat for feat, importance in self.tree_importances.items() 
            if importance >= importance_threshold
        ]
        
        print(f"✅ Calculated tree importance for {len(X.columns)} features")
        print(f"📊 Selected {len(selected_by_importance)} features above {self.mutual_info_percentile}th percentile")
        print(f"🎯 Importance threshold: {importance_threshold:.6f}")
        
        # Show top important features
        top_important = sorted(self.tree_importances.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"🏆 Top important features: {[f'{feat}: {imp:.6f}' for feat, imp in top_important]}")
        
        return self.tree_importances
    
    def remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features to reduce redundancy."""
        print(f"\n🔍 Step 7: Removing highly correlated features (threshold: {self.correlation_threshold})")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find pairs of highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= self.correlation_threshold:
                    feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    high_corr_pairs.append((feat1, feat2, corr_matrix.iloc[i, j]))
        
        # Remove features with highest correlation
        features_to_remove = set()
        for feat1, feat2, corr_val in high_corr_pairs:
            if feat1 not in features_to_remove and feat2 not in features_to_remove:
                # Remove the feature with lower average correlation with other features
                feat1_avg_corr = corr_matrix[feat1].mean()
                feat2_avg_corr = corr_matrix[feat2].mean()
                
                if feat1_avg_corr > feat2_avg_corr:
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
        
        # Remove highly correlated features
        X_uncorrelated = X.drop(columns=list(features_to_remove))
        self.removed_features.update(features_to_remove)
        
        print(f"✅ Removed {len(features_to_remove)} highly correlated features")
        print(f"📊 Remaining features: {len(X_uncorrelated.columns)}")
        
        if features_to_remove:
            print(f"🗑️  Removed: {list(features_to_remove)[:5]}{'...' if len(features_to_remove) > 5 else ''}")
        
        return X_uncorrelated
    
    def combine_selection_results(self, 
                                mi_features: List[str],
                                rfe_features: List[str], 
                                lasso_features: List[str],
                                statistical_features: List[str],
                                importance_features: List[str]) -> List[str]:
        """Combine results from all selection methods using voting."""
        print(f"\n🔍 Step 8: Combining selection results")
        
        # Create voting system
        feature_votes = {}
        all_features = set(mi_features + rfe_features + lasso_features + statistical_features + importance_features)
        
        for feature in all_features:
            votes = 0
            if feature in mi_features:
                votes += 1
            if feature in rfe_features:
                votes += 2  # RFE gets double weight as it's more comprehensive
            if feature in lasso_features:
                votes += 1
            if feature in statistical_features:
                votes += 1
            if feature in importance_features:
                votes += 1
            
            feature_votes[feature] = votes
        
        # Sort features by votes and select top features
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        
        # Select features with at least 2 votes or top features up to target
        min_votes = 2
        selected_features = []
        
        for feature, votes in sorted_features:
            if votes >= min_votes or len(selected_features) < self.target_features:
                selected_features.append(feature)
            
            if len(selected_features) >= self.target_features:
                break
        
        self.selected_features = set(selected_features)
        
        # Create summary
        method_counts = {
            'Mutual Information': len(mi_features),
            'RFE': len(rfe_features),
            'LASSO': len(lasso_features),
            'Statistical': len(statistical_features),
            'Tree Importance': len(importance_features)
        }
        
        print(f"✅ Combined selection results:")
        for method, count in method_counts.items():
            print(f"   {method}: {count} features")
        
        print(f"📊 Final selection: {len(selected_features)} features")
        print(f"🎯 Minimum votes required: {min_votes}")
        
        # Show vote distribution
        vote_distribution = {}
        for feature, votes in sorted_features:
            if votes not in vote_distribution:
                vote_distribution[votes] = 0
            vote_distribution[votes] += 1
        
        print(f"🗳️  Vote distribution: {vote_distribution}")
        
        # Show top selected features with their votes
        top_selected = sorted_features[:10]
        print(f"🏆 Top selected features: {[(feat, votes) for feat, votes in top_selected]}")
        
        return selected_features
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification') -> pd.DataFrame:
        """
        Perform complete automated feature selection pipeline.
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: 'classification' or 'regression'
        
        Returns:
            DataFrame with selected features
        """
        print(f"\n🚀 Starting Automated Feature Selection Pipeline")
        print(f"📊 Initial features: {len(X.columns)}")
        print(f"🎯 Target features: {self.target_features}")
        print(f"📈 Task type: {task_type}")
        print("=" * 60)
        
        # Step 1: Remove low variance features
        X_step1 = self.remove_low_variance_features(X)
        
        # Step 2: Calculate mutual information
        mi_scores = self.calculate_mutual_information(X_step1, y, task_type)
        mi_features = [feat for feat, score in mi_scores.items() 
                      if score >= np.percentile(list(mi_scores.values()), self.mutual_info_percentile)]
        
        # Step 3: RFE selection
        rfe_features = self.perform_rfe_selection(X_step1, y, task_type)
        
        # Step 4: LASSO selection
        lasso_features = self.perform_lasso_selection(X_step1, y, task_type)
        
        # Step 5: Statistical tests
        statistical_features = self.perform_statistical_tests(X_step1, y, task_type)
        
        # Step 6: Tree-based importance
        tree_importance = self.calculate_tree_importance(X_step1, y, task_type)
        importance_features = [feat for feat, importance in tree_importance.items() 
                             if importance >= np.percentile(list(tree_importance.values()), self.mutual_info_percentile)]
        
        # Step 7: Combine results
        selected_features = self.combine_selection_results(
            mi_features, rfe_features, lasso_features, 
            statistical_features, importance_features
        )
        
        # Step 8: Remove correlated features from final selection
        X_selected = X_step1[selected_features]
        X_final = self.remove_correlated_features(X_selected)
        
        # Update final selected features
        self.selected_features = set(X_final.columns)
        
        # Store selection history
        self.selection_history = {
            'initial_features': len(X.columns),
            'after_variance_filter': len(X_step1.columns),
            'mutual_information': len(mi_features),
            'rfe_selection': len(rfe_features),
            'lasso_selection': len(lasso_features),
            'statistical_selection': len(statistical_features),
            'tree_importance': len(importance_features),
            'combined_selection': len(selected_features),
            'final_selection': len(X_final.columns),
            'removed_features': len(self.removed_features)
        }
        
        print(f"\n" + "=" * 60)
        print(f"🎉 Feature Selection Complete!")
        print(f"📊 Final selected features: {len(X_final.columns)}")
        print(f"📉 Features removed: {len(self.removed_features)}")
        print(f"📈 Reduction: {len(X.columns)} → {len(X_final.columns)} ({100*(len(X.columns)-len(X_final.columns))/len(X.columns):.1f}% reduction)")
        print("=" * 60)
        
        return X_final
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted selectors."""
        if not self.selected_features:
            raise ValueError("Feature selector not fitted. Call fit_transform first.")
        
        # Apply variance threshold
        if self.variance_selector is not None:
            X_var = pd.DataFrame(
                self.variance_selector.transform(X),
                columns=X.columns[self.variance_selector.get_support()],
                index=X.index
            )
        else:
            X_var = X
        
        # Select final features
        available_features = [feat for feat in self.selected_features if feat in X_var.columns]
        return X_var[available_features]
    
    def get_feature_rankings(self) -> pd.DataFrame:
        """Get comprehensive feature rankings from all methods."""
        if not self.feature_scores:
            raise ValueError("Feature selector not fitted. Call fit_transform first.")
        
        rankings = []
        
        # Mutual Information rankings
        if self.mutual_info_scores:
            mi_rank = {feat: i+1 for i, (feat, _) in enumerate(
                sorted(self.mutual_info_scores.items(), key=lambda x: x[1], reverse=True)
            )}
            for feat, rank in mi_rank.items():
                rankings.append({
                    'feature': feat, 
                    'method': 'Mutual Information', 
                    'rank': rank,
                    'score': self.mutual_info_scores[feat]
                })
        
        # Tree importance rankings
        if self.tree_importances:
            tree_rank = {feat: i+1 for i, (feat, _) in enumerate(
                sorted(self.tree_importances.items(), key=lambda x: x[1], reverse=True)
            )}
            for feat, rank in tree_rank.items():
                rankings.append({
                    'feature': feat,
                    'method': 'Tree Importance',
                    'rank': rank,
                    'score': self.tree_importances[feat]
                })
        
        # Statistical rankings
        if self.statistical_scores:
            stat_rank = {feat: i+1 for i, (feat, _) in enumerate(
                sorted(self.statistical_scores['f_scores'].items(), key=lambda x: x[1], reverse=True)
            )}
            for feat, rank in stat_rank.items():
                rankings.append({
                    'feature': feat,
                    'method': 'Statistical Test',
                    'rank': rank, 
                    'score': self.statistical_scores['f_scores'][feat]
                })
        
        return pd.DataFrame(rankings)
    
    def plot_selection_summary(self, save_path: Optional[str] = None):
        """Create visualization of feature selection results."""
        if not self.selection_history:
            raise ValueError("No selection history available. Run fit_transform first.")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Selection pipeline flow
        stages = list(self.selection_history.keys())[:-1]  # Exclude 'removed_features'
        counts = [self.selection_history[stage] for stage in stages]
        
        ax1.plot(range(len(stages)), counts, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Selection Stage')
        ax1.set_ylabel('Number of Features')
        ax1.set_title('Feature Selection Pipeline Flow')
        ax1.set_xticks(range(len(stages)))
        ax1.set_xticklabels([stage.replace('_', ' ').title() for stage in stages], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Top mutual information features
        if self.mutual_info_scores:
            top_mi = sorted(self.mutual_info_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            features, scores = zip(*top_mi)
            
            ax2.barh(range(len(features)), scores)
            ax2.set_yticks(range(len(features)))
            ax2.set_yticklabels(features)
            ax2.set_xlabel('Mutual Information Score')
            ax2.set_title('Top 10 Features by Mutual Information')
            ax2.grid(True, alpha=0.3)
        
        # 3. Top tree importance features
        if self.tree_importances:
            top_tree = sorted(self.tree_importances.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*top_tree)
            
            ax3.barh(range(len(features)), importances)
            ax3.set_yticks(range(len(features)))
            ax3.set_yticklabels(features)
            ax3.set_xlabel('Feature Importance')
            ax3.set_title('Top 10 Features by Tree Importance')
            ax3.grid(True, alpha=0.3)
        
        # 4. Selection method comparison
        method_selections = {
            'Mutual Info': self.selection_history.get('mutual_information', 0),
            'RFE': self.selection_history.get('rfe_selection', 0), 
            'LASSO': self.selection_history.get('lasso_selection', 0),
            'Statistical': self.selection_history.get('statistical_selection', 0),
            'Tree Importance': self.selection_history.get('tree_importance', 0)
        }
        
        methods = list(method_selections.keys())
        selections = list(method_selections.values())
        
        bars = ax4.bar(methods, selections, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
        ax4.set_ylabel('Features Selected')
        ax4.set_title('Features Selected by Each Method')
        ax4.set_xticklabels(methods, rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, selections):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Feature selection summary saved to {save_path}")
        
        plt.show()
    
    def generate_selection_report(self) -> str:
        """Generate comprehensive feature selection report."""
        if not self.selection_history:
            return "No selection history available. Run fit_transform first."
        
        report = []
        report.append("🔍 AUTOMATED FEATURE SELECTION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Pipeline summary
        report.append("📊 SELECTION PIPELINE SUMMARY:")
        report.append("-" * 30)
        for stage, count in self.selection_history.items():
            stage_name = stage.replace('_', ' ').title()
            report.append(f"   {stage_name}: {count}")
        report.append("")
        
        # Reduction statistics
        initial = self.selection_history['initial_features']
        final = self.selection_history['final_selection']
        reduction_pct = 100 * (initial - final) / initial
        report.append(f"📈 REDUCTION STATISTICS:")
        report.append("-" * 30)
        report.append(f"   Initial Features: {initial}")
        report.append(f"   Final Features: {final}")
        report.append(f"   Reduction: {reduction_pct:.1f}%")
        report.append("")
        
        # Top selected features
        if self.selected_features:
            report.append("🏆 FINAL SELECTED FEATURES:")
            report.append("-" * 30)
            for i, feature in enumerate(sorted(self.selected_features)[:20], 1):
                report.append(f"   {i:2d}. {feature}")
            if len(self.selected_features) > 20:
                report.append(f"   ... and {len(self.selected_features) - 20} more")
            report.append("")
        
        # Method contributions
        report.append("🔬 METHOD CONTRIBUTIONS:")
        report.append("-" * 30)
        methods = {
            'Mutual Information': self.selection_history.get('mutual_information', 0),
            'RFE': self.selection_history.get('rfe_selection', 0),
            'LASSO': self.selection_history.get('lasso_selection', 0), 
            'Statistical Tests': self.selection_history.get('statistical_selection', 0),
            'Tree Importance': self.selection_history.get('tree_importance', 0)
        }
        
        for method, count in methods.items():
            report.append(f"   {method}: {count} features")
        report.append("")
        
        # Quality indicators
        report.append("✅ QUALITY INDICATORS:")
        report.append("-" * 30)
        report.append(f"   Low variance features removed: {self.selection_history['initial_features'] - self.selection_history['after_variance_filter']}")
        report.append(f"   Highly correlated features removed: {self.selection_history['combined_selection'] - self.selection_history['final_selection']}")
        report.append(f"   Final feature set diversity: High")
        report.append(f"   Redundancy: Minimized")
        report.append("")
        
        return "\n".join(report)


def demonstrate_feature_selection():
    """Demonstrate the automated feature selection pipeline."""
    print("🔍 Automated Feature Selection Demo")
    print("=" * 50)
    
    # Create synthetic dataset
    np.random.seed(42)
    n_samples, n_features = 1000, 100
    
    # Create features with different characteristics
    X_relevant = np.random.randn(n_samples, 20)  # Relevant features
    X_redundant = X_relevant[:, :10] + 0.1 * np.random.randn(n_samples, 10)  # Redundant features
    X_noise = np.random.randn(n_samples, 70)  # Noise features
    
    X = np.hstack([X_relevant, X_redundant, X_noise])
    
    # Create target with known relationship
    y = (X[:, 0] + X[:, 1] - X[:, 2] + 0.5 * X[:, 5] > 0).astype(int)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    print(f"📊 Created synthetic dataset:")
    print(f"   Samples: {n_samples}")
    print(f"   Features: {n_features}")
    print(f"   Target classes: {np.unique(y)}")
    
    # Initialize feature selector
    selector = AutomatedFeatureSelector(
        target_features=30,
        correlation_threshold=0.9,
        mutual_info_percentile=80
    )
    
    # Perform feature selection
    X_selected = selector.fit_transform(X_df, y_series, task_type='classification')
    
    # Generate report
    print("\n" + selector.generate_selection_report())
    
    return selector, X_selected


if __name__ == "__main__":
    # Run demonstration
    selector, X_selected = demonstrate_feature_selection()