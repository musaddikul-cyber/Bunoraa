"""
Fraud Detection Service

Django service for fraud detection and risk assessment.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import json

try:
    from django.core.cache import cache
    from django.conf import settings
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    cache = None

logger = logging.getLogger("bunoraa.ml.services.fraud")


@dataclass
class RiskScore:
    """Risk score result."""
    score: float  # 0-1, higher = more risky
    level: str  # low, medium, high, critical
    factors: List[str]  # Contributing factors
    recommendation: str  # Action recommendation
    
    @property
    def is_blocked(self) -> bool:
        return self.level == "critical"
    
    @property
    def needs_review(self) -> bool:
        return self.level in ("high", "critical")


class FraudService:
    """
    Service for fraud detection and risk assessment.
    """
    
    def __init__(self):
        self._models = {}
        self._model_registry = None
        
        # Risk thresholds
        self.thresholds = {
            "low": 0.2,
            "medium": 0.5,
            "high": 0.7,
            "critical": 0.9,
        }
    
    def _get_registry(self):
        if self._model_registry is None:
            from ..core.registry import ModelRegistry
            self._model_registry = ModelRegistry()
        return self._model_registry
    
    def _get_model(self, model_name: str):
        if model_name not in self._models:
            registry = self._get_registry()
            self._models[model_name] = registry.get_latest(model_name)
        return self._models[model_name]
    
    def assess_order_risk(
        self,
        order_data: Dict[str, Any],
        user_data: Optional[Dict[str, Any]] = None
    ) -> RiskScore:
        """
        Assess fraud risk for an order.
        
        Args:
            order_data: Order information
            user_data: Optional user information
        
        Returns:
            Risk assessment
        """
        factors = []
        scores = []
        
        try:
            # Rule-based checks
            rule_score, rule_factors = self._apply_rules(order_data, user_data)
            scores.append(rule_score)
            factors.extend(rule_factors)
            
            # ML model prediction
            model = self._get_model("fraud_detector")
            if model:
                ml_score, ml_factors = self._ml_prediction(model, order_data, user_data)
                scores.append(ml_score * 0.6)  # Weight ML higher
                factors.extend(ml_factors)
            
            # Velocity checks
            velocity_score, velocity_factors = self._velocity_checks(order_data, user_data)
            scores.append(velocity_score)
            factors.extend(velocity_factors)
            
            # Device/IP reputation
            device_score, device_factors = self._device_reputation(order_data)
            scores.append(device_score)
            factors.extend(device_factors)
            
            # Combine scores
            final_score = sum(scores) / len(scores) if scores else 0.0
            final_score = min(max(final_score, 0.0), 1.0)
            
            # Determine risk level
            level = self._get_risk_level(final_score)
            recommendation = self._get_recommendation(level, factors)
            
            risk = RiskScore(
                score=final_score,
                level=level,
                factors=list(set(factors)),
                recommendation=recommendation
            )
            
            # Log assessment
            self._log_assessment(order_data, risk)
            
            return risk
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            # Return safe default
            return RiskScore(
                score=0.3,
                level="low",
                factors=["assessment_error"],
                recommendation="Manual review recommended due to assessment error"
            )
    
    def assess_user_risk(self, user_id: int) -> RiskScore:
        """
        Assess overall fraud risk for a user.
        
        Args:
            user_id: User ID
        
        Returns:
            User risk assessment
        """
        factors = []
        scores = []
        
        try:
            # Get user history
            user_history = self._get_user_history(user_id)
            
            # Account age check
            account_age_days = user_history.get("account_age_days", 0)
            if account_age_days < 1:
                scores.append(0.4)
                factors.append("new_account")
            elif account_age_days < 7:
                scores.append(0.2)
                factors.append("recent_account")
            else:
                scores.append(0.0)
            
            # Order history
            order_count = user_history.get("order_count", 0)
            if order_count == 0:
                scores.append(0.3)
                factors.append("no_order_history")
            elif order_count > 5:
                scores.append(-0.1)  # Reduce risk for established customers
            
            # Chargeback history
            chargebacks = user_history.get("chargebacks", 0)
            if chargebacks > 0:
                scores.append(0.8)
                factors.append(f"previous_chargebacks:{chargebacks}")
            
            # Fraud flags
            fraud_flags = user_history.get("fraud_flags", 0)
            if fraud_flags > 0:
                scores.append(0.7)
                factors.append("previous_fraud_flags")
            
            # Email reputation
            email_score = self._check_email_reputation(user_history.get("email"))
            if email_score > 0:
                scores.append(email_score)
                factors.append("suspicious_email")
            
            # Final score
            final_score = sum(scores) / max(len(scores), 1)
            final_score = min(max(final_score, 0.0), 1.0)
            
            level = self._get_risk_level(final_score)
            
            return RiskScore(
                score=final_score,
                level=level,
                factors=factors,
                recommendation=self._get_recommendation(level, factors)
            )
            
        except Exception as e:
            logger.error(f"User risk assessment failed: {e}")
            return RiskScore(
                score=0.3,
                level="medium",
                factors=["assessment_error"],
                recommendation="Manual review recommended"
            )
    
    def assess_payment_risk(
        self,
        payment_data: Dict[str, Any],
        order_data: Dict[str, Any]
    ) -> RiskScore:
        """
        Assess fraud risk for a payment.
        
        Args:
            payment_data: Payment information
            order_data: Order information
        
        Returns:
            Payment risk assessment
        """
        factors = []
        scores = []
        
        try:
            # Card checks
            if payment_data.get("payment_method") == "card":
                card_score, card_factors = self._check_card(payment_data)
                scores.append(card_score)
                factors.extend(card_factors)
            
            # Billing/shipping address mismatch
            if self._address_mismatch(order_data):
                scores.append(0.4)
                factors.append("address_mismatch")
            
            # High-risk payment method
            high_risk_methods = ["crypto", "wire_transfer", "prepaid_card"]
            if payment_data.get("payment_method") in high_risk_methods:
                scores.append(0.5)
                factors.append(f"high_risk_payment_method:{payment_data.get('payment_method')}")
            
            # Amount anomaly
            amount = order_data.get("total_amount", 0)
            if amount > 1000:
                scores.append(0.3)
                factors.append("high_value_order")
            if amount > 5000:
                scores.append(0.5)
                factors.append("very_high_value_order")
            
            # Final score
            final_score = sum(scores) / max(len(scores), 1)
            final_score = min(max(final_score, 0.0), 1.0)
            
            level = self._get_risk_level(final_score)
            
            return RiskScore(
                score=final_score,
                level=level,
                factors=factors,
                recommendation=self._get_recommendation(level, factors)
            )
            
        except Exception as e:
            logger.error(f"Payment risk assessment failed: {e}")
            return RiskScore(
                score=0.5,
                level="medium",
                factors=["assessment_error"],
                recommendation="Manual review recommended"
            )
    
    def detect_account_takeover(
        self,
        user_id: int,
        session_data: Dict[str, Any]
    ) -> Tuple[bool, float, List[str]]:
        """
        Detect potential account takeover.
        
        Args:
            user_id: User ID
            session_data: Current session information
        
        Returns:
            (is_suspicious, confidence, factors)
        """
        factors = []
        anomaly_score = 0.0
        
        try:
            # Get user's normal patterns
            normal_patterns = self._get_user_patterns(user_id)
            
            # Check IP location
            current_ip = session_data.get("ip_address")
            if current_ip:
                ip_anomaly = self._check_ip_anomaly(current_ip, normal_patterns.get("ips", []))
                if ip_anomaly > 0.5:
                    anomaly_score += 0.3
                    factors.append("unusual_ip_location")
            
            # Check device
            current_device = session_data.get("device_fingerprint")
            if current_device:
                device_anomaly = self._check_device_anomaly(
                    current_device,
                    normal_patterns.get("devices", [])
                )
                if device_anomaly > 0.5:
                    anomaly_score += 0.3
                    factors.append("new_device")
            
            # Check time of access
            current_hour = datetime.now().hour
            if normal_patterns.get("active_hours"):
                if current_hour not in normal_patterns["active_hours"]:
                    anomaly_score += 0.1
                    factors.append("unusual_access_time")
            
            # Check for password reset
            if session_data.get("after_password_reset"):
                anomaly_score += 0.2
                factors.append("recent_password_reset")
            
            # Check for sensitive actions
            if session_data.get("attempting_sensitive_action"):
                anomaly_score += 0.2
                factors.append("sensitive_action_attempt")
            
            is_suspicious = anomaly_score > 0.5
            
            return is_suspicious, anomaly_score, factors
            
        except Exception as e:
            logger.error(f"Account takeover detection failed: {e}")
            return False, 0.0, ["detection_error"]
    
    def flag_transaction(
        self,
        transaction_id: str,
        reason: str,
        flagged_by: str = "system"
    ):
        """Flag a transaction for review."""
        try:
            from apps.orders.models import Order
            
            order = Order.objects.get(id=transaction_id)
            order.is_flagged = True
            order.flag_reason = reason
            order.flagged_by = flagged_by
            order.flagged_at = datetime.now()
            order.save()
            
            logger.info(f"Flagged transaction {transaction_id}: {reason}")
            
        except Exception as e:
            logger.error(f"Failed to flag transaction: {e}")
    
    def get_fraud_dashboard_data(self) -> Dict[str, Any]:
        """Get fraud analytics for dashboard."""
        try:
            from apps.orders.models import Order
            from django.db.models import Count, Sum
            
            today = datetime.now().date()
            last_week = today - timedelta(days=7)
            
            # Flagged orders
            flagged_today = Order.objects.filter(
                is_flagged=True,
                created_at__date=today
            ).count()
            
            flagged_week = Order.objects.filter(
                is_flagged=True,
                created_at__date__gte=last_week
            ).count()
            
            # Blocked orders
            blocked_today = Order.objects.filter(
                status="blocked",
                created_at__date=today
            ).count()
            
            # Total amount blocked
            blocked_amount = Order.objects.filter(
                status="blocked",
                created_at__date__gte=last_week
            ).aggregate(total=Sum("total_amount"))["total"] or 0
            
            return {
                "flagged_today": flagged_today,
                "flagged_week": flagged_week,
                "blocked_today": blocked_today,
                "blocked_amount_week": float(blocked_amount),
            }
            
        except Exception as e:
            logger.error(f"Failed to get fraud dashboard data: {e}")
            return {}
    
    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------
    
    def _apply_rules(
        self,
        order_data: Dict,
        user_data: Optional[Dict]
    ) -> Tuple[float, List[str]]:
        """Apply rule-based fraud checks."""
        factors = []
        score = 0.0
        
        # High-value first order
        if user_data and user_data.get("order_count", 0) == 0:
            if order_data.get("total_amount", 0) > 500:
                score += 0.4
                factors.append("high_value_first_order")
        
        # Rush shipping on high-value
        if order_data.get("shipping_method") == "express":
            if order_data.get("total_amount", 0) > 300:
                score += 0.2
                factors.append("rush_shipping_high_value")
        
        # Multiple items of same expensive product
        items = order_data.get("items", [])
        for item in items:
            if item.get("quantity", 1) > 3 and item.get("price", 0) > 100:
                score += 0.3
                factors.append("bulk_expensive_items")
                break
        
        # Gift card / high-risk category
        for item in items:
            category = item.get("category", "").lower()
            if category in ["gift_cards", "electronics", "jewelry"]:
                score += 0.2
                factors.append(f"high_risk_category:{category}")
                break
        
        return min(score, 1.0), factors
    
    def _ml_prediction(
        self,
        model,
        order_data: Dict,
        user_data: Optional[Dict]
    ) -> Tuple[float, List[str]]:
        """Get ML model fraud prediction."""
        try:
            prediction = model.predict(
                order_data=order_data,
                user_data=user_data or {}
            )
            
            factors = []
            if prediction["is_fraud"]:
                factors.append("ml_fraud_detected")
            if prediction.get("is_bot"):
                factors.append("ml_bot_detected")
            
            return prediction["fraud_probability"], factors
            
        except Exception as e:
            logger.debug(f"ML prediction failed: {e}")
            return 0.0, []
    
    def _velocity_checks(
        self,
        order_data: Dict,
        user_data: Optional[Dict]
    ) -> Tuple[float, List[str]]:
        """Check for velocity anomalies."""
        factors = []
        score = 0.0
        
        try:
            from apps.orders.models import Order
            
            user_id = order_data.get("user_id") or (user_data or {}).get("id")
            if not user_id:
                return 0.0, []
            
            now = datetime.now()
            
            # Orders in last hour
            orders_hour = Order.objects.filter(
                user_id=user_id,
                created_at__gte=now - timedelta(hours=1)
            ).count()
            
            if orders_hour >= 3:
                score += 0.4
                factors.append(f"high_velocity:{orders_hour}_orders_per_hour")
            
            # Orders in last 24 hours
            orders_day = Order.objects.filter(
                user_id=user_id,
                created_at__gte=now - timedelta(hours=24)
            ).count()
            
            if orders_day >= 10:
                score += 0.5
                factors.append(f"very_high_velocity:{orders_day}_orders_per_day")
            
        except Exception as e:
            logger.debug(f"Velocity check failed: {e}")
        
        return min(score, 1.0), factors
    
    def _device_reputation(
        self,
        order_data: Dict
    ) -> Tuple[float, List[str]]:
        """Check device/IP reputation."""
        factors = []
        score = 0.0
        
        # Check for VPN/proxy
        ip = order_data.get("ip_address")
        if ip:
            if self._is_vpn_ip(ip):
                score += 0.3
                factors.append("vpn_or_proxy")
            
            if self._is_blacklisted_ip(ip):
                score += 0.7
                factors.append("blacklisted_ip")
        
        # Check device fingerprint
        device = order_data.get("device_fingerprint")
        if device:
            if self._is_suspicious_device(device):
                score += 0.4
                factors.append("suspicious_device")
        
        return min(score, 1.0), factors
    
    def _get_risk_level(self, score: float) -> str:
        """Convert score to risk level."""
        if score >= self.thresholds["critical"]:
            return "critical"
        elif score >= self.thresholds["high"]:
            return "high"
        elif score >= self.thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def _get_recommendation(self, level: str, factors: List[str]) -> str:
        """Get action recommendation based on risk level."""
        if level == "critical":
            return "Block transaction and investigate immediately"
        elif level == "high":
            return "Hold for manual review before processing"
        elif level == "medium":
            return "Process with enhanced monitoring"
        else:
            return "Process normally"
    
    def _log_assessment(self, order_data: Dict, risk: RiskScore):
        """Log fraud assessment for audit."""
        try:
            logger.info(
                f"Fraud assessment: order={order_data.get('id')} "
                f"score={risk.score:.2f} level={risk.level} "
                f"factors={risk.factors}"
            )
        except Exception:
            pass
    
    def _get_user_history(self, user_id: int) -> Dict[str, Any]:
        """Get user's history for risk assessment."""
        try:
            from apps.accounts.models import User
            from apps.orders.models import Order
            from django.db.models import Count
            
            user = User.objects.get(id=user_id)
            
            return {
                "account_age_days": (datetime.now() - user.date_joined).days,
                "order_count": Order.objects.filter(user_id=user_id).count(),
                "chargebacks": Order.objects.filter(
                    user_id=user_id,
                    status="chargeback"
                ).count(),
                "fraud_flags": Order.objects.filter(
                    user_id=user_id,
                    is_flagged=True
                ).count(),
                "email": user.email,
            }
        except Exception:
            return {}
    
    def _check_email_reputation(self, email: Optional[str]) -> float:
        """Check email reputation."""
        if not email:
            return 0.0
        
        # Simple checks
        suspicious_patterns = [
            "tempmail", "throwaway", "fakeinbox", "guerrilla",
            "mailinator", "10minute"
        ]
        
        email_lower = email.lower()
        for pattern in suspicious_patterns:
            if pattern in email_lower:
                return 0.6
        
        return 0.0
    
    def _check_card(self, payment_data: Dict) -> Tuple[float, List[str]]:
        """Check card for fraud indicators."""
        factors = []
        score = 0.0
        
        # Prepaid card
        if payment_data.get("card_type") == "prepaid":
            score += 0.3
            factors.append("prepaid_card")
        
        # International card
        if payment_data.get("card_country") != payment_data.get("billing_country"):
            score += 0.2
            factors.append("international_card")
        
        # Multiple cards on account
        if payment_data.get("cards_on_account", 0) > 5:
            score += 0.3
            factors.append("many_cards_on_account")
        
        return score, factors
    
    def _address_mismatch(self, order_data: Dict) -> bool:
        """Check if billing and shipping addresses mismatch."""
        billing = order_data.get("billing_address", {})
        shipping = order_data.get("shipping_address", {})
        
        if not billing or not shipping:
            return False
        
        # Check country mismatch
        if billing.get("country") != shipping.get("country"):
            return True
        
        # Check significant distance (simplified)
        if billing.get("postal_code") and shipping.get("postal_code"):
            if billing["postal_code"][:3] != shipping["postal_code"][:3]:
                return True
        
        return False
    
    def _get_user_patterns(self, user_id: int) -> Dict[str, Any]:
        """Get user's normal behavior patterns."""
        # This would ideally come from stored patterns
        return {
            "ips": [],
            "devices": [],
            "active_hours": list(range(8, 23)),
        }
    
    def _check_ip_anomaly(self, current_ip: str, known_ips: List[str]) -> float:
        """Check if IP is anomalous for user."""
        if not known_ips:
            return 0.3  # No history = slight anomaly
        
        if current_ip in known_ips:
            return 0.0
        
        # Could add geolocation comparison here
        return 0.5
    
    def _check_device_anomaly(self, current_device: str, known_devices: List[str]) -> float:
        """Check if device is anomalous for user."""
        if not known_devices:
            return 0.2
        
        if current_device in known_devices:
            return 0.0
        
        return 0.5
    
    def _is_vpn_ip(self, ip: str) -> bool:
        """Check if IP is a known VPN/proxy."""
        # Would integrate with IP reputation service
        return False
    
    def _is_blacklisted_ip(self, ip: str) -> bool:
        """Check if IP is blacklisted."""
        # Would integrate with blacklist service
        return False
    
    def _is_suspicious_device(self, fingerprint: str) -> bool:
        """Check if device fingerprint is suspicious."""
        # Would analyze device fingerprint
        return False
