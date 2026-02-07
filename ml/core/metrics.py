"""
Metrics Tracking for ML Models

Provides comprehensive metrics collection and monitoring:
- Training metrics
- Inference metrics
- Model drift detection
- Performance monitoring
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

logger = logging.getLogger("bunoraa.ml.metrics")


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "tags": self.tags
        }


class MetricsTracker:
    """
    Track and monitor ML metrics.
    
    Features:
    - Log training and inference metrics
    - Compute aggregate statistics
    - Detect model drift
    - Export metrics to various backends
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_history: int = 10000
    ):
        self.storage_path = storage_path or Path(__file__).parent.parent / "metrics"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_history = max_history
        
        # In-memory metrics storage
        self._metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        
        # Baselines for drift detection
        self._baselines: Dict[str, Dict[str, float]] = {}
        
        # Alerts
        self._alert_thresholds: Dict[str, Dict[str, float]] = {}
    
    def log(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[str] = None
    ):
        """
        Log a metric value.
        
        Args:
            name: Metric name (e.g., "training/loss", "inference/latency_ms")
            value: Metric value
            tags: Optional tags (model_id, version, etc.)
            timestamp: Optional timestamp (defaults to now)
        """
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=timestamp or datetime.utcnow().isoformat(),
            tags=tags or {}
        )
        
        self._metrics[name].append(point)
        
        # Trim history
        if len(self._metrics[name]) > self.max_history:
            self._metrics[name] = self._metrics[name][-self.max_history:]
        
        # Check alerts
        self._check_alerts(name, value, tags)
    
    def log_batch(self, metrics: Dict[str, float], tags: Optional[Dict[str, str]] = None):
        """Log multiple metrics at once."""
        timestamp = datetime.utcnow().isoformat()
        for name, value in metrics.items():
            self.log(name, value, tags, timestamp)
    
    def get_metric(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[MetricPoint]:
        """
        Get metric history.
        
        Args:
            name: Metric name
            start_time: Filter by start time
            end_time: Filter by end time
            tags: Filter by tags
        
        Returns:
            List of MetricPoint objects
        """
        points = self._metrics.get(name, [])
        
        if start_time:
            start_str = start_time.isoformat()
            points = [p for p in points if p.timestamp >= start_str]
        
        if end_time:
            end_str = end_time.isoformat()
            points = [p for p in points if p.timestamp <= end_str]
        
        if tags:
            points = [
                p for p in points
                if all(p.tags.get(k) == v for k, v in tags.items())
            ]
        
        return points
    
    def get_stats(
        self,
        name: str,
        window_minutes: int = 60,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Get aggregate statistics for a metric.
        
        Args:
            name: Metric name
            window_minutes: Time window in minutes
            tags: Optional tag filter
        
        Returns:
            Dictionary with min, max, mean, std, count, p50, p95, p99
        """
        start_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        points = self.get_metric(name, start_time=start_time, tags=tags)
        
        if not points:
            return {
                "min": 0, "max": 0, "mean": 0, "std": 0,
                "count": 0, "p50": 0, "p95": 0, "p99": 0
            }
        
        values = sorted([p.value for p in points])
        
        return {
            "min": min(values),
            "max": max(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "count": len(values),
            "p50": values[len(values) // 2],
            "p95": values[int(len(values) * 0.95)] if len(values) > 20 else values[-1],
            "p99": values[int(len(values) * 0.99)] if len(values) > 100 else values[-1]
        }
    
    def set_baseline(
        self,
        model_id: str,
        metrics: Dict[str, float]
    ):
        """
        Set baseline metrics for drift detection.
        
        Args:
            model_id: Model identifier
            metrics: Baseline metric values
        """
        self._baselines[model_id] = metrics
        logger.info(f"Set baseline for {model_id}: {metrics}")
    
    def detect_drift(
        self,
        model_id: str,
        current_metrics: Dict[str, float],
        threshold_pct: float = 20.0
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect metric drift from baseline.
        
        Args:
            model_id: Model identifier
            current_metrics: Current metric values
            threshold_pct: Percentage change threshold for drift
        
        Returns:
            Dictionary with drift information for each metric
        """
        baseline = self._baselines.get(model_id, {})
        if not baseline:
            logger.warning(f"No baseline found for {model_id}")
            return {}
        
        drift_report = {}
        
        for metric_name, current_value in current_metrics.items():
            baseline_value = baseline.get(metric_name)
            if baseline_value is None or baseline_value == 0:
                continue
            
            pct_change = ((current_value - baseline_value) / baseline_value) * 100
            
            is_drift = abs(pct_change) > threshold_pct
            
            drift_report[metric_name] = {
                "baseline": baseline_value,
                "current": current_value,
                "pct_change": pct_change,
                "is_drift": is_drift,
                "severity": "high" if abs(pct_change) > threshold_pct * 2 else "medium" if is_drift else "low"
            }
            
            if is_drift:
                logger.warning(
                    f"Drift detected for {model_id}/{metric_name}: "
                    f"{baseline_value:.4f} -> {current_value:.4f} ({pct_change:+.1f}%)"
                )
        
        return drift_report
    
    def set_alert(
        self,
        metric_name: str,
        min_threshold: Optional[float] = None,
        max_threshold: Optional[float] = None
    ):
        """
        Set alert thresholds for a metric.
        
        Args:
            metric_name: Metric name
            min_threshold: Alert if value falls below this
            max_threshold: Alert if value exceeds this
        """
        self._alert_thresholds[metric_name] = {
            "min": min_threshold,
            "max": max_threshold
        }
    
    def _check_alerts(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Check if value triggers any alerts."""
        thresholds = self._alert_thresholds.get(name)
        if not thresholds:
            return
        
        if thresholds.get("min") is not None and value < thresholds["min"]:
            logger.error(f"ALERT: {name} = {value} is below minimum threshold {thresholds['min']}")
            # Could trigger notifications here
        
        if thresholds.get("max") is not None and value > thresholds["max"]:
            logger.error(f"ALERT: {name} = {value} exceeds maximum threshold {thresholds['max']}")
            # Could trigger notifications here
    
    def save_snapshot(self, snapshot_name: Optional[str] = None):
        """Save current metrics to disk."""
        snapshot_name = snapshot_name or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        snapshot_path = self.storage_path / f"{snapshot_name}.json"
        
        data = {
            name: [p.to_dict() for p in points]
            for name, points in self._metrics.items()
        }
        
        with open(snapshot_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved metrics snapshot to {snapshot_path}")
    
    def load_snapshot(self, snapshot_name: str):
        """Load metrics from a snapshot."""
        snapshot_path = self.storage_path / f"{snapshot_name}.json"
        
        with open(snapshot_path) as f:
            data = json.load(f)
        
        for name, points in data.items():
            self._metrics[name] = [
                MetricPoint(**p) for p in points
            ]
        
        logger.info(f"Loaded metrics from {snapshot_path}")
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for name, points in self._metrics.items():
            if not points:
                continue
            
            # Get latest value
            latest = points[-1]
            
            # Format metric name for Prometheus
            prom_name = name.replace("/", "_").replace(".", "_").replace("-", "_")
            
            # Format tags
            tags_str = ""
            if latest.tags:
                tag_pairs = [f'{k}="{v}"' for k, v in latest.tags.items()]
                tags_str = "{" + ",".join(tag_pairs) + "}"
            
            lines.append(f"{prom_name}{tags_str} {latest.value}")
        
        return "\n".join(lines)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics."""
        summary = {}
        for name in self._metrics:
            stats = self.get_stats(name)
            summary[name] = {
                "count": stats["count"],
                "mean": stats["mean"],
                "p95": stats["p95"],
            }
        return summary


# Singleton metrics tracker
_tracker: Optional[MetricsTracker] = None


def get_metrics_tracker() -> MetricsTracker:
    """Get or create the global metrics tracker."""
    global _tracker
    if _tracker is None:
        _tracker = MetricsTracker()
    return _tracker
