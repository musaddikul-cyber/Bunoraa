"""
ML Evaluate Command

Django management command to evaluate ML models.

Usage:
    python manage.py ml_evaluate                  # Evaluate all models
    python manage.py ml_evaluate --model ncf    # Evaluate specific model
    python manage.py ml_evaluate --report       # Generate detailed report
"""

import logging
import json
from datetime import datetime

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger("bunoraa.ml.commands")


class Command(BaseCommand):
    help = "Evaluate ML models"
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--model',
            type=str,
            default='all',
            choices=[
                'all', 'ncf', 'deepfm', 'two_tower', 'sequence',
                'embeddings', 'demand', 'fraud', 'churn', 'search', 'image'
            ],
            help='Model to evaluate (default: all)',
        )
        parser.add_argument(
            '--report',
            action='store_true',
            help='Generate detailed evaluation report',
        )
        parser.add_argument(
            '--output',
            type=str,
            default=None,
            help='Output file for report (JSON)',
        )
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.7,
            help='Performance threshold for pass/fail (default: 0.7)',
        )
    
    def handle(self, *args, **options):
        model_type = options['model']
        generate_report = options['report']
        output_file = options['output']
        threshold = options['threshold']
        
        self.stdout.write(f"Evaluating ML models: {model_type}")
        
        start_time = timezone.now()
        results = {}
        
        try:
            from ml.core.registry import ModelRegistry
            from ml.core.metrics import MetricsTracker
            
            registry = ModelRegistry()
            metrics_tracker = MetricsTracker()
            
            models_to_evaluate = self._get_models_to_evaluate(model_type)
            
            for model_name in models_to_evaluate:
                self.stdout.write(f"\nEvaluating {model_name}...")
                
                try:
                    model_results = self._evaluate_model(
                        model_name, registry, metrics_tracker
                    )
                    results[model_name] = model_results
                    
                    # Display results
                    self._display_results(model_name, model_results, threshold)
                    
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f"  Failed to evaluate {model_name}: {e}")
                    )
                    results[model_name] = {'error': str(e)}
            
            # Summary
            elapsed = (timezone.now() - start_time).total_seconds()
            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(f"Evaluation completed in {elapsed:.1f}s")
            
            # Pass/fail summary
            passed = sum(1 for r in results.values() if r.get('score', 0) >= threshold)
            total = len([r for r in results.values() if 'error' not in r])
            
            if total > 0:
                self.stdout.write(f"Models passing threshold ({threshold}): {passed}/{total}")
            
            # Generate report
            if generate_report or output_file:
                report = self._generate_report(results, threshold)
                
                if output_file:
                    with open(output_file, 'w') as f:
                        json.dump(report, f, indent=2, default=str)
                    self.stdout.write(f"\nReport saved to {output_file}")
                else:
                    self.stdout.write("\n" + json.dumps(report, indent=2, default=str))
            
            self.stdout.write(self.style.SUCCESS("\nEvaluation completed!"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Evaluation failed: {e}"))
            logger.exception("Evaluation error")
            raise CommandError(str(e))
    
    def _get_models_to_evaluate(self, model_type: str) -> list:
        """Get list of models to evaluate."""
        all_models = [
            'ncf', 'deepfm', 'two_tower', 'sequence',
            'embeddings', 'demand', 'fraud', 'churn', 'search', 'image'
        ]
        
        if model_type == 'all':
            return all_models
        return [model_type]
    
    def _evaluate_model(self, model_name: str, registry, metrics_tracker) -> dict:
        """Evaluate a single model."""
        results = {
            'model': model_name,
            'timestamp': timezone.now().isoformat(),
            'version': None,
            'metrics': {},
            'score': 0.0,
            'health': 'unknown',
        }
        
        # Get model from registry
        model_info = registry.get_model_info(model_name)
        
        if not model_info:
            results['health'] = 'not_found'
            return results
        
        results['version'] = model_info.get('version', 'unknown')
        
        # Get stored metrics
        stored_metrics = model_info.get('metrics', {})
        results['metrics'] = stored_metrics
        
        # Check for drift
        drift_detected = metrics_tracker.check_drift(model_name)
        results['drift_detected'] = drift_detected
        
        # Calculate overall score based on model type
        score = self._calculate_score(model_name, stored_metrics)
        results['score'] = score
        
        # Determine health
        if score >= 0.8:
            results['health'] = 'excellent'
        elif score >= 0.6:
            results['health'] = 'good'
        elif score >= 0.4:
            results['health'] = 'fair'
        else:
            results['health'] = 'poor'
        
        if drift_detected:
            results['health'] = 'drift_detected'
        
        return results
    
    def _calculate_score(self, model_name: str, metrics: dict) -> float:
        """Calculate overall score for a model based on its metrics."""
        if not metrics:
            return 0.0
        
        # Different metrics matter for different models
        metric_weights = {
            'ncf': {'ndcg': 0.4, 'hit_rate': 0.3, 'precision': 0.3},
            'deepfm': {'auc': 0.5, 'logloss': -0.3, 'precision': 0.2},
            'two_tower': {'recall': 0.4, 'ndcg': 0.3, 'hit_rate': 0.3},
            'sequence': {'hit_rate': 0.4, 'mrr': 0.3, 'ndcg': 0.3},
            'embeddings': {'similarity_score': 0.5, 'coverage': 0.5},
            'demand': {'mape': -0.5, 'rmse': -0.3, 'r2': 0.2},
            'fraud': {'precision': 0.3, 'recall': 0.4, 'f1': 0.3},
            'churn': {'auc': 0.4, 'precision': 0.3, 'recall': 0.3},
            'search': {'mrr': 0.4, 'ndcg': 0.3, 'precision': 0.3},
            'image': {'accuracy': 0.4, 'f1': 0.3, 'precision': 0.3},
        }
        
        weights = metric_weights.get(model_name, {})
        
        score = 0.0
        weight_sum = 0.0
        
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                # Negative weight means lower is better (invert)
                if weight < 0:
                    # For metrics like MAPE, lower is better
                    # Normalize to 0-1 (assuming max is 1.0)
                    value = max(0, 1 - value)
                    weight = abs(weight)
                
                score += value * weight
                weight_sum += weight
        
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def _display_results(self, model_name: str, results: dict, threshold: float):
        """Display evaluation results for a model."""
        health = results.get('health', 'unknown')
        score = results.get('score', 0)
        version = results.get('version', 'unknown')
        
        # Color code based on health
        if health in ('excellent', 'good'):
            status_style = self.style.SUCCESS
        elif health == 'fair':
            status_style = self.style.WARNING
        else:
            status_style = self.style.ERROR
        
        self.stdout.write(f"  Version: {version}")
        self.stdout.write(f"  Score: {score:.3f}")
        self.stdout.write(status_style(f"  Health: {health}"))
        
        if results.get('drift_detected'):
            self.stdout.write(self.style.WARNING("  ⚠ Drift detected"))
        
        # Show key metrics
        metrics = results.get('metrics', {})
        if metrics:
            self.stdout.write("  Metrics:")
            for key, value in list(metrics.items())[:5]:
                if isinstance(value, float):
                    self.stdout.write(f"    {key}: {value:.4f}")
                else:
                    self.stdout.write(f"    {key}: {value}")
        
        # Pass/fail indicator
        if score >= threshold:
            self.stdout.write(self.style.SUCCESS(f"  ✓ PASS (>= {threshold})"))
        else:
            self.stdout.write(self.style.ERROR(f"  ✗ FAIL (< {threshold})"))
    
    def _generate_report(self, results: dict, threshold: float) -> dict:
        """Generate detailed evaluation report."""
        report = {
            'timestamp': timezone.now().isoformat(),
            'threshold': threshold,
            'summary': {
                'total_models': len(results),
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'drift_detected': 0,
            },
            'models': results,
            'recommendations': [],
        }
        
        for model_name, model_results in results.items():
            if 'error' in model_results:
                report['summary']['errors'] += 1
            elif model_results.get('score', 0) >= threshold:
                report['summary']['passed'] += 1
            else:
                report['summary']['failed'] += 1
            
            if model_results.get('drift_detected'):
                report['summary']['drift_detected'] += 1
        
        # Generate recommendations
        for model_name, model_results in results.items():
            if 'error' in model_results:
                report['recommendations'].append({
                    'model': model_name,
                    'action': 'investigate',
                    'reason': f"Model evaluation failed: {model_results['error']}",
                })
            elif model_results.get('drift_detected'):
                report['recommendations'].append({
                    'model': model_name,
                    'action': 'retrain',
                    'reason': 'Data drift detected',
                })
            elif model_results.get('score', 0) < threshold:
                report['recommendations'].append({
                    'model': model_name,
                    'action': 'retrain',
                    'reason': f"Score {model_results.get('score', 0):.3f} below threshold {threshold}",
                })
        
        return report
