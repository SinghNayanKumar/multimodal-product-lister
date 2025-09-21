import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
import json
import argparse

class StatisticalTester:
    def __init__(self):
        pass
    
    def bootstrap_confidence_interval(self, data, n_bootstrap=1000, confidence=0.95):
        """Calculate bootstrap confidence interval."""
        bootstrap_samples = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_samples.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (100 - alpha / 2)
        
        ci_lower = np.percentile(bootstrap_samples, lower_percentile)
        ci_upper = np.percentile(bootstrap_samples, upper_percentile)
        
        return ci_lower, ci_upper
    
    def compare_model_performance(self, model_results, metric_name):
        """Compare performance between models with statistical tests."""
        comparisons = []
        model_names = list(model_results.keys())
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model_a = model_names[i]
                model_b = model_names[j]
                
                # Get metric values (assuming they're stored as lists or arrays)
                values_a = model_results[model_a].get(metric_name, [])
                values_b = model_results[model_b].get(metric_name, [])
                
                if len(values_a) > 1 and len(values_b) > 1:
                    # Paired t-test (if same samples)
                    if len(values_a) == len(values_b):
                        t_stat, p_value = ttest_rel(values_a, values_b)
                        test_type = "paired_t_test"
                    else:
                        # Independent t-test
                        t_stat, p_value = stats.ttest_ind(values_a, values_b)
                        test_type = "independent_t_test"
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(values_a) - 1) * np.var(values_a, ddof=1) + 
                                         (len(values_b) - 1) * np.var(values_b, ddof=1)) / 
                                        (len(values_a) + len(values_b) - 2))
                    cohens_d = (np.mean(values_a) - np.mean(values_b)) / pooled_std
                    
                    # Bootstrap CIs
                    ci_a = self.bootstrap_confidence_interval(values_a)
                    ci_b = self.bootstrap_confidence_interval(values_b)
                    
                    comparison = {
                        'model_a': model_a,
                        'model_b': model_b,
                        'metric': metric_name,
                        'mean_a': np.mean(values_a),
                        'mean_b': np.mean(values_b),
                        'std_a': np.std(values_a),
                        'std_b': np.std(values_b),
                        'test_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'effect_size_cohens_d': cohens_d,
                        'ci_95_a': ci_a,
                        'ci_95_b': ci_b,
                        'test_type': test_type
                    }
                    
                    comparisons.append(comparison)
        
        return comparisons
    
    def generate_significance_report(self, comparisons, output_path):
        """Generate a statistical significance report."""
        report_lines = []
        report_lines.append("STATISTICAL SIGNIFICANCE ANALYSIS")
        report_lines.append("=" * 50)
        
        for comp in comparisons:
            report_lines.append(f"\nComparison: {comp['model_a']} vs {comp['model_b']}")
            report_lines.append(f"Metric: {comp['metric']}")
            report_lines.append(f"Mean {comp['model_a']}: {comp['mean_a']:.4f} (±{comp['std_a']:.4f})")
            report_lines.append(f"Mean {comp['model_b']}: {comp['mean_b']:.4f} (±{comp['std_b']:.4f})")
            report_lines.append(f"Test: {comp['test_type']}")
            report_lines.append(f"p-value: {comp['p_value']:.4f}")
            report_lines.append(f"Significant: {'Yes' if comp['significant'] else 'No'}")
            report_lines.append(f"Effect size (Cohen's d): {comp['effect_size_cohens_d']:.4f}")
            
            if comp['significant']:
                better_model = comp['model_a'] if comp['mean_a'] > comp['mean_b'] else comp['model_b']
                report_lines.append(f"→ {better_model} performs significantly better")
            
            report_lines.append("-" * 30)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return '\n'.join(report_lines)
