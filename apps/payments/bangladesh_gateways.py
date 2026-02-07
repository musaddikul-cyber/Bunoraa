"""
Bangladesh Payment Gateway Services
Implements SSLCommerz, bKash, and Nagad integrations
"""
import hashlib
import json
import logging
import requests
from decimal import Decimal
from datetime import datetime
from typing import Dict, Optional, Any
from urllib.parse import urlencode

from django.conf import settings
from django.urls import reverse
from django.utils import timezone

logger = logging.getLogger('bunoraa.payments')


class SSLCommerzService:
    """
    SSLCommerz payment gateway integration for Bangladesh.
    Documentation: https://developer.sslcommerz.com/
    """
    
    SANDBOX_URL = 'https://sandbox.sslcommerz.com'
    LIVE_URL = 'https://securepay.sslcommerz.com'
    
    def __init__(self, gateway=None):
        """Initialize with optional gateway configuration."""
        self.gateway = gateway
        
        if gateway:
            self.store_id = gateway.ssl_store_id
            self.store_passwd = gateway.ssl_store_passwd
            self.is_sandbox = gateway.is_sandbox
        else:
            self.store_id = getattr(settings, 'SSLCOMMERZ_STORE_ID', '')
            self.store_passwd = getattr(settings, 'SSLCOMMERZ_STORE_PASSWD', '')
            self.is_sandbox = getattr(settings, 'SSLCOMMERZ_IS_SANDBOX', True)
        
        self.base_url = self.SANDBOX_URL if self.is_sandbox else self.LIVE_URL
    
    def init_transaction(
        self,
        order,
        success_url: str,
        fail_url: str,
        cancel_url: str,
        ipn_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Initialize a payment transaction with SSLCommerz.
        
        Args:
            order: Order model instance
            success_url: URL to redirect on successful payment
            fail_url: URL to redirect on failed payment
            cancel_url: URL to redirect on cancelled payment
            ipn_url: Instant Payment Notification URL
            
        Returns:
            Dict with status and redirect URL or error message
        """
        try:
            # Prepare transaction data
            tran_id = f"BUNORAA-{order.order_number}-{timezone.now().strftime('%Y%m%d%H%M%S')}"
            
            # Calculate total in BDT
            total_amount = float(order.total)
            
            post_data = {
                'store_id': self.store_id,
                'store_passwd': self.store_passwd,
                'total_amount': total_amount,
                'currency': 'BDT',
                'tran_id': tran_id,
                'success_url': success_url,
                'fail_url': fail_url,
                'cancel_url': cancel_url,
                'ipn_url': ipn_url or success_url,
                
                # Customer info
                'cus_name': order.shipping_full_name,
                'cus_email': order.email,
                'cus_phone': order.phone or '+8801700000000',
                'cus_add1': order.shipping_address_line_1,
                'cus_add2': order.shipping_address_line_2 or '',
                'cus_city': order.shipping_city,
                'cus_state': order.shipping_state or order.shipping_city,
                'cus_postcode': order.shipping_postal_code,
                'cus_country': 'Bangladesh',
                
                # Shipping info
                'ship_name': order.shipping_full_name,
                'shipping_method': 'Courier',
                'ship_add1': order.shipping_address_line_1,
                'ship_add2': order.shipping_address_line_2 or '',
                'ship_city': order.shipping_city,
                'ship_state': order.shipping_state or order.shipping_city,
                'ship_postcode': order.shipping_postal_code,
                'ship_country': 'Bangladesh',
                
                # Product info
                'product_name': f'Order {order.order_number}',
                'product_category': 'General',
                'product_profile': 'general',
                'num_of_item': order.item_count,
                
                # Additional params
                'value_a': str(order.id),
                'value_b': order.order_number,
            }
            
            # Make API request
            response = requests.post(
                f'{self.base_url}/gwprocess/v4/api.php',
                data=post_data,
                timeout=30
            )
            
            result = response.json()
            
            if result.get('status') == 'SUCCESS':
                # Store transaction ID in order
                order.stripe_payment_intent_id = tran_id  # Reusing field for SSLCommerz
                order.save(update_fields=['stripe_payment_intent_id'])
                
                return {
                    'status': 'success',
                    'redirect_url': result.get('GatewayPageURL'),
                    'session_key': result.get('sessionkey'),
                    'tran_id': tran_id,
                }
            else:
                logger.error(f"SSLCommerz init failed: {result}")
                return {
                    'status': 'error',
                    'message': result.get('failedreason', 'Payment initialization failed'),
                }
                
        except requests.RequestException as e:
            logger.error(f"SSLCommerz request error: {e}")
            return {
                'status': 'error',
                'message': 'Payment gateway connection error',
            }
        except Exception as e:
            logger.error(f"SSLCommerz error: {e}")
            return {
                'status': 'error',
                'message': 'An unexpected error occurred',
            }
    
    def verify_transaction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify IPN/callback payload from SSLCommerz.
        
        Args:
            payload: POST data from SSLCommerz callback
            
        Returns:
            Dict with verification result
        """
        try:
            tran_id = payload.get('tran_id')
            val_id = payload.get('val_id')
            status = payload.get('status')
            
            if status not in ['VALID', 'VALIDATED']:
                return {
                    'success': False,
                    'status': status,
                    'message': f'Transaction status: {status}',
                    'raw': payload,
                }
            
            # Validate transaction with SSLCommerz
            validation_url = f'{self.base_url}/validator/api/validationserverAPI.php'
            validation_params = {
                'val_id': val_id,
                'store_id': self.store_id,
                'store_passwd': self.store_passwd,
                'format': 'json',
            }
            
            response = requests.get(validation_url, params=validation_params, timeout=30)
            result = response.json()
            
            if result.get('status') == 'VALID' or result.get('status') == 'VALIDATED':
                return {
                    'success': True,
                    'tran_id': tran_id,
                    'val_id': val_id,
                    'amount': Decimal(result.get('amount', '0')),
                    'currency': result.get('currency', 'BDT'),
                    'card_type': result.get('card_type', ''),
                    'card_brand': result.get('card_brand', ''),
                    'bank_tran_id': result.get('bank_tran_id', ''),
                    'order_id': payload.get('value_a'),
                    'order_number': payload.get('value_b'),
                    'raw': result,
                }
            else:
                return {
                    'success': False,
                    'status': result.get('status'),
                    'message': result.get('error', 'Validation failed'),
                    'raw': result,
                }
                
        except Exception as e:
            logger.error(f"SSLCommerz verification error: {e}")
            return {
                'success': False,
                'message': str(e),
                'raw': payload,
            }
    
    def initiate_refund(
        self,
        bank_tran_id: str,
        refund_amount: Decimal,
        refund_remarks: str = ''
    ) -> Dict[str, Any]:
        """
        Initiate a refund through SSLCommerz.
        """
        try:
            refund_url = f'{self.base_url}/validator/api/merchantTransIDvalidationAPI.php'
            
            params = {
                'store_id': self.store_id,
                'store_passwd': self.store_passwd,
                'bank_tran_id': bank_tran_id,
                'refund_amount': float(refund_amount),
                'refund_remarks': refund_remarks,
                'format': 'json',
            }
            
            response = requests.get(refund_url, params=params, timeout=30)
            result = response.json()
            
            return {
                'success': result.get('APIConnect') == 'DONE',
                'refund_ref_id': result.get('refund_ref_id'),
                'raw': result,
            }
            
        except Exception as e:
            logger.error(f"SSLCommerz refund error: {e}")
            return {
                'success': False,
                'message': str(e),
            }


class BkashService:
    """
    bKash payment gateway integration.
    Documentation: https://developer.bka.sh/
    """
    
    SANDBOX_URL = 'https://tokenized.sandbox.bka.sh/v1.2.0-beta'
    LIVE_URL = 'https://tokenized.pay.bka.sh/v1.2.0-beta'
    
    def __init__(self, gateway=None):
        """Initialize with optional gateway configuration."""
        self.gateway = gateway
        
        if gateway:
            self.app_key = gateway.bkash_app_key
            self.app_secret = gateway.bkash_app_secret
            self.username = gateway.bkash_username
            self.password = gateway.bkash_password
            self.is_sandbox = gateway.is_sandbox
        else:
            self.app_key = getattr(settings, 'BKASH_APP_KEY', '')
            self.app_secret = getattr(settings, 'BKASH_APP_SECRET', '')
            self.username = getattr(settings, 'BKASH_USERNAME', '')
            self.password = getattr(settings, 'BKASH_PASSWORD', '')
            self.is_sandbox = getattr(settings, 'BKASH_IS_SANDBOX', True)
        
        self.base_url = self.SANDBOX_URL if self.is_sandbox else self.LIVE_URL
        self._token = None
        self._token_expiry = None
    
    def _get_headers(self, with_token: bool = True) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }
        
        if with_token and self._token:
            headers['Authorization'] = self._token
            headers['X-APP-Key'] = self.app_key
        
        return headers
    
    def get_token(self) -> Optional[str]:
        """
        Get or refresh authentication token.
        """
        try:
            # Check if existing token is still valid
            if self._token and self._token_expiry:
                if timezone.now() < self._token_expiry:
                    return self._token
            
            # Request new token
            url = f'{self.base_url}/tokenized/checkout/token/grant'
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'username': self.username,
                'password': self.password,
            }
            
            data = {
                'app_key': self.app_key,
                'app_secret': self.app_secret,
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            result = response.json()
            
            if result.get('statusCode') == '0000':
                self._token = result.get('id_token')
                # Token typically valid for 1 hour
                self._token_expiry = timezone.now() + timezone.timedelta(minutes=55)
                return self._token
            else:
                logger.error(f"bKash token error: {result}")
                return None
                
        except Exception as e:
            logger.error(f"bKash token error: {e}")
            return None
    
    def create_payment(
        self,
        order,
        callback_url: str,
        payer_reference: str = ''
    ) -> Dict[str, Any]:
        """
        Create a bKash payment request.
        
        Args:
            order: Order model instance
            callback_url: URL to redirect after payment
            payer_reference: Optional reference for the payer
            
        Returns:
            Dict with payment URL or error
        """
        try:
            token = self.get_token()
            if not token:
                return {
                    'status': 'error',
                    'message': 'Authentication failed',
                }
            
            invoice_number = f"BUNORAA-{order.order_number}"
            
            url = f'{self.base_url}/tokenized/checkout/create'
            
            data = {
                'mode': '0011',
                'payerReference': payer_reference or str(order.user.id if order.user else order.email),
                'callbackURL': callback_url,
                'amount': str(float(order.total)),
                'currency': 'BDT',
                'intent': 'sale',
                'merchantInvoiceNumber': invoice_number,
            }
            
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=data,
                timeout=30
            )
            
            result = response.json()
            
            if result.get('statusCode') == '0000':
                return {
                    'status': 'success',
                    'payment_id': result.get('paymentID'),
                    'bkash_url': result.get('bkashURL'),
                    'invoice_number': invoice_number,
                }
            else:
                return {
                    'status': 'error',
                    'message': result.get('statusMessage', 'Payment creation failed'),
                    'raw': result,
                }
                
        except Exception as e:
            logger.error(f"bKash payment error: {e}")
            return {
                'status': 'error',
                'message': str(e),
            }
    
    def execute_payment(self, payment_id: str) -> Dict[str, Any]:
        """
        Execute a bKash payment after user authorization.
        """
        try:
            token = self.get_token()
            if not token:
                return {
                    'success': False,
                    'message': 'Authentication failed',
                }
            
            url = f'{self.base_url}/tokenized/checkout/execute'
            
            data = {
                'paymentID': payment_id,
            }
            
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=data,
                timeout=30
            )
            
            result = response.json()
            
            if result.get('statusCode') == '0000':
                return {
                    'success': True,
                    'payment_id': result.get('paymentID'),
                    'trx_id': result.get('trxID'),
                    'amount': Decimal(result.get('amount', '0')),
                    'currency': result.get('currency', 'BDT'),
                    'payer_reference': result.get('payerReference'),
                    'merchant_invoice': result.get('merchantInvoiceNumber'),
                    'raw': result,
                }
            else:
                return {
                    'success': False,
                    'message': result.get('statusMessage', 'Payment execution failed'),
                    'raw': result,
                }
                
        except Exception as e:
            logger.error(f"bKash execute error: {e}")
            return {
                'success': False,
                'message': str(e),
            }
    
    def query_payment(self, payment_id: str) -> Dict[str, Any]:
        """
        Query payment status.
        """
        try:
            token = self.get_token()
            if not token:
                return {'success': False, 'message': 'Authentication failed'}
            
            url = f'{self.base_url}/tokenized/checkout/payment/status'
            
            data = {'paymentID': payment_id}
            
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=data,
                timeout=30
            )
            
            result = response.json()
            
            return {
                'success': result.get('statusCode') == '0000',
                'status': result.get('transactionStatus'),
                'raw': result,
            }
            
        except Exception as e:
            logger.error(f"bKash query error: {e}")
            return {'success': False, 'message': str(e)}
    
    def refund_payment(
        self,
        payment_id: str,
        trx_id: str,
        amount: Decimal,
        reason: str = ''
    ) -> Dict[str, Any]:
        """
        Initiate a refund for a bKash payment.
        """
        try:
            token = self.get_token()
            if not token:
                return {'success': False, 'message': 'Authentication failed'}
            
            url = f'{self.base_url}/tokenized/checkout/payment/refund'
            
            data = {
                'paymentID': payment_id,
                'trxID': trx_id,
                'amount': str(float(amount)),
                'reason': reason or 'Customer refund request',
                'sku': 'REFUND',
            }
            
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=data,
                timeout=30
            )
            
            result = response.json()
            
            return {
                'success': result.get('statusCode') == '0000',
                'refund_trx_id': result.get('refundTrxID'),
                'raw': result,
            }
            
        except Exception as e:
            logger.error(f"bKash refund error: {e}")
            return {'success': False, 'message': str(e)}


class NagadService:
    """
    Nagad payment gateway integration.
    Documentation: https://nagad.com.bd/developers/
    """
    
    SANDBOX_URL = 'http://sandbox.mynagad.com:10080/remote-payment-gateway-1.0'
    LIVE_URL = 'https://api.mynagad.com/api/dfs'
    
    def __init__(self, gateway=None):
        """Initialize with optional gateway configuration."""
        self.gateway = gateway
        
        if gateway:
            self.merchant_id = gateway.nagad_merchant_id
            self.public_key = gateway.nagad_public_key
            self.private_key = gateway.nagad_private_key
            self.is_sandbox = gateway.is_sandbox
        else:
            self.merchant_id = getattr(settings, 'NAGAD_MERCHANT_ID', '')
            self.public_key = getattr(settings, 'NAGAD_PUBLIC_KEY', '')
            self.private_key = getattr(settings, 'NAGAD_PRIVATE_KEY', '')
            self.is_sandbox = getattr(settings, 'NAGAD_IS_SANDBOX', True)
        
        self.base_url = self.SANDBOX_URL if self.is_sandbox else self.LIVE_URL
    
    def _generate_signature(self, data: str) -> str:
        """Generate signature for Nagad API."""
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.backends import default_backend
        import base64
        
        try:
            private_key = serialization.load_pem_private_key(
                self.private_key.encode(),
                password=None,
                backend=default_backend()
            )
            
            signature = private_key.sign(
                data.encode(),
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode()
        except Exception as e:
            logger.error(f"Nagad signature error: {e}")
            return ''
    
    def init_payment(
        self,
        order,
        callback_url: str
    ) -> Dict[str, Any]:
        """
        Initialize a Nagad payment.
        """
        try:
            order_id = f"BUNORAA-{order.order_number}"
            datetime_str = timezone.now().strftime('%Y%m%d%H%M%S')
            
            # Step 1: Initialize
            init_url = f'{self.base_url}/check-out/initialize/{self.merchant_id}/{order_id}'
            
            sensitive_data = {
                'merchantId': self.merchant_id,
                'datetime': datetime_str,
                'orderId': order_id,
                'challenge': hashlib.sha256(order_id.encode()).hexdigest()[:20],
            }
            
            # Encrypt sensitive data
            # Note: In production, use proper RSA encryption with Nagad's public key
            
            headers = {
                'Content-Type': 'application/json',
                'X-KM-Api-Version': 'v-0.2.0',
                'X-KM-IP-V4': '127.0.0.1',
                'X-KM-Client-Type': 'PC_WEB',
            }
            
            # This is a simplified implementation
            # Full implementation requires proper encryption/decryption
            
            return {
                'status': 'pending_implementation',
                'message': 'Nagad integration requires proper cryptographic implementation',
                'order_id': order_id,
            }
            
        except Exception as e:
            logger.error(f"Nagad init error: {e}")
            return {
                'status': 'error',
                'message': str(e),
            }


class PaymentGatewayFactory:
    """
    Factory for creating payment gateway service instances.
    """
    
    SERVICES = {
        'sslcommerz': SSLCommerzService,
        'bkash': BkashService,
        'nagad': NagadService,
    }
    
    @classmethod
    def get_service(cls, gateway_code: str, gateway=None):
        """
        Get payment service instance by gateway code.
        
        Args:
            gateway_code: Code of the payment gateway
            gateway: Optional PaymentGateway model instance
            
        Returns:
            Payment service instance
        """
        service_class = cls.SERVICES.get(gateway_code.lower())
        
        if not service_class:
            raise ValueError(f"Unknown payment gateway: {gateway_code}")
        
        return service_class(gateway=gateway)
