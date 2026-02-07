"""
Orders tests
"""
from decimal import Decimal
from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase, APIClient
from rest_framework import status

from apps.catalog.models import Product, Category
from apps.commerce.models import Cart, CartItem, CheckoutSession
from apps.commerce.services import CartService, CheckoutService
from .models import Order, OrderItem, OrderStatusHistory
from .services import OrderService


User = get_user_model()


class OrderModelTest(TestCase):
    """Test cases for Order model."""
    
    def setUp(self):
        self.user = User.objects.create_user(
            email='test@example.com',
            password='testpass123',
            first_name='Test',
            last_name='User'
        )
    
    def test_create_order(self):
        """Test creating order."""
        order = Order.objects.create(
            user=self.user,
            email='test@example.com',
            shipping_first_name='John',
            shipping_last_name='Doe',
            shipping_address_line_1='123 Main St',
            shipping_city='New York',
            shipping_postal_code='10001',
            shipping_country='United States',
            billing_first_name='John',
            billing_last_name='Doe',
            billing_address_line_1='123 Main St',
            billing_city='New York',
            billing_postal_code='10001',
            billing_country='United States',
            subtotal=Decimal('100.00'),
            total=Decimal('105.99'),
        )
        
        self.assertIsNotNone(order.order_number)
        self.assertTrue(order.order_number.startswith('ORD-'))
    
    def test_order_number_generation(self):
        """Test unique order number generation."""
        order1 = Order.objects.create(
            user=self.user,
            email='test@example.com',
            shipping_first_name='John',
            shipping_last_name='Doe',
            shipping_address_line_1='123 Main St',
            shipping_city='New York',
            shipping_postal_code='10001',
            shipping_country='United States',
            billing_first_name='John',
            billing_last_name='Doe',
            billing_address_line_1='123 Main St',
            billing_city='New York',
            billing_postal_code='10001',
            billing_country='United States',
            subtotal=Decimal('100.00'),
            total=Decimal('100.00'),
        )
        
        order2 = Order.objects.create(
            user=self.user,
            email='test@example.com',
            shipping_first_name='John',
            shipping_last_name='Doe',
            shipping_address_line_1='123 Main St',
            shipping_city='New York',
            shipping_postal_code='10001',
            shipping_country='United States',
            billing_first_name='John',
            billing_last_name='Doe',
            billing_address_line_1='123 Main St',
            billing_city='New York',
            billing_postal_code='10001',
            billing_country='United States',
            subtotal=Decimal('100.00'),
            total=Decimal('100.00'),
        )
        
        self.assertNotEqual(order1.order_number, order2.order_number)
    
    def test_order_status_history(self):
        """Test order status history creation."""
        order = Order.objects.create(
            user=self.user,
            email='test@example.com',
            shipping_first_name='John',
            shipping_last_name='Doe',
            shipping_address_line_1='123 Main St',
            shipping_city='New York',
            shipping_postal_code='10001',
            shipping_country='United States',
            billing_first_name='John',
            billing_last_name='Doe',
            billing_address_line_1='123 Main St',
            billing_city='New York',
            billing_postal_code='10001',
            billing_country='United States',
            subtotal=Decimal('100.00'),
            total=Decimal('100.00'),
        )
        
        # Should have initial history entry
        self.assertEqual(order.status_history.count(), 1)
        
        # Change status
        order.status = Order.STATUS_CONFIRMED
        order.save()
        
        # Should have new history entry
        self.assertEqual(order.status_history.count(), 2)
    
    def test_can_cancel(self):
        """Test can_cancel property."""
        order = Order.objects.create(
            user=self.user,
            email='test@example.com',
            shipping_first_name='John',
            shipping_last_name='Doe',
            shipping_address_line_1='123 Main St',
            shipping_city='New York',
            shipping_postal_code='10001',
            shipping_country='United States',
            billing_first_name='John',
            billing_last_name='Doe',
            billing_address_line_1='123 Main St',
            billing_city='New York',
            billing_postal_code='10001',
            billing_country='United States',
            subtotal=Decimal('100.00'),
            total=Decimal('100.00'),
            status=Order.STATUS_PENDING
        )
        
        self.assertTrue(order.can_cancel)
        
        order.status = Order.STATUS_SHIPPED
        order.save()
        
        self.assertFalse(order.can_cancel)


class OrderItemModelTest(TestCase):
    """Test cases for OrderItem model."""
    
    def setUp(self):
        self.user = User.objects.create_user(
            email='test@example.com',
            password='testpass123'
        )
        self.order = Order.objects.create(
            user=self.user,
            email='test@example.com',
            shipping_first_name='John',
            shipping_last_name='Doe',
            shipping_address_line_1='123 Main St',
            shipping_city='New York',
            shipping_postal_code='10001',
            shipping_country='United States',
            billing_first_name='John',
            billing_last_name='Doe',
            billing_address_line_1='123 Main St',
            billing_city='New York',
            billing_postal_code='10001',
            billing_country='United States',
            subtotal=Decimal('100.00'),
            total=Decimal('100.00'),
        )
    
    def test_create_order_item(self):
        """Test creating order item."""
        item = OrderItem.objects.create(
            order=self.order,
            product_name='Test Product',
            unit_price=Decimal('29.99'),
            quantity=2
        )
        
        self.assertEqual(item.product_name, 'Test Product')
        self.assertEqual(item.quantity, 2)
    
    def test_line_total(self):
        """Test line total calculation."""
        item = OrderItem.objects.create(
            order=self.order,
            product_name='Test Product',
            unit_price=Decimal('29.99'),
            quantity=3
        )
        
        self.assertEqual(item.line_total, Decimal('89.97'))

    def test_line_total_with_missing_unit_price(self):
        """line_total should handle missing unit_price on unsaved instances and return 0.00."""
        # Create an unsaved OrderItem (unit_price omitted -> None)
        item = OrderItem(order=self.order, product_name='Test Product', quantity=2)
        self.assertEqual(item.line_total, Decimal('0.00'))


class OrderServiceTest(TestCase):
    """Test cases for OrderService."""
    
    def setUp(self):
        self.user = User.objects.create_user(
            email='test@example.com',
            password='testpass123',
            first_name='Test',
            last_name='User'
        )
        self.category = Category.objects.create(name='Test', slug='test')
        self.product = Product.objects.create(
            name='Test Product',
            slug='test-product',
            price=Decimal('29.99'),
            stock_quantity=100,
            is_active=True
        )
        self.product.categories.add(self.category)
        
        # Create cart with item
        self.cart = CartService.get_or_create_cart(user=self.user)
        CartService.add_item(self.cart, self.product, quantity=2)
        
        # Create checkout session
        self.checkout_session = CheckoutService.get_or_create_session(
            cart=self.cart,
            user=self.user
        )
        CheckoutService.update_shipping_address(self.checkout_session, {
            'first_name': 'John',
            'last_name': 'Doe',
            'email': 'john@example.com',
            'phone': '555-1234',
            'address_line_1': '123 Main St',
            'city': 'New York',
            'state': 'NY',
            'postal_code': '10001',
            'country': 'United States',
            'billing_same_as_shipping': True
        })
    
    def test_create_order_from_checkout(self):
        """Test creating order from checkout."""
        order = OrderService.create_order_from_checkout(self.checkout_session)
        
        self.assertIsNotNone(order)
        self.assertEqual(order.user, self.user)
        self.assertEqual(order.items.count(), 1)
        self.assertEqual(order.status, Order.STATUS_CONFIRMED)
    
    def test_order_updates_stock(self):
        """Test that order creation updates stock."""
        initial_stock = self.product.stock_quantity
        
        OrderService.create_order_from_checkout(self.checkout_session)
        
        self.product.refresh_from_db()
        self.assertEqual(self.product.stock_quantity, initial_stock - 2)
    
    def test_get_user_orders(self):
        """Test getting user orders."""
        OrderService.create_order_from_checkout(self.checkout_session)
        
        # Create new cart and checkout for another order
        cart2 = Cart.objects.create(user=self.user)
        CartItem.objects.create(
            cart=cart2,
            product=self.product,
            quantity=1,
            price_at_add=self.product.price
        )
        session2 = CheckoutService.get_or_create_session(cart=cart2, user=self.user)
        CheckoutService.update_shipping_address(session2, {
            'first_name': 'John',
            'last_name': 'Doe',
            'email': 'john@example.com',
            'address_line_1': '123 Main St',
            'city': 'New York',
            'postal_code': '10001',
            'country': 'United States',
            'billing_same_as_shipping': True
        })
        OrderService.create_order_from_checkout(session2)
        
        orders = OrderService.get_user_orders(self.user)
        self.assertEqual(orders.count(), 2)
    
    def test_update_order_status(self):
        """Test updating order status."""
        order = OrderService.create_order_from_checkout(self.checkout_session)
        
        updated_order = OrderService.update_order_status(
            order,
            Order.STATUS_PROCESSING,
            notes='Processing started'
        )
        
        self.assertEqual(updated_order.status, Order.STATUS_PROCESSING)
    
    def test_cancel_order(self):
        """Test cancelling order."""
        order = OrderService.create_order_from_checkout(self.checkout_session)
        # Set to pending (cancellable)
        order.status = Order.STATUS_PENDING
        order.payment_status = 'pending'
        order.save()
        
        success, message = OrderService.cancel_order(order, reason='Changed mind')
        
        self.assertTrue(success)
        order.refresh_from_db()
        self.assertEqual(order.status, Order.STATUS_CANCELLED)


class OrderAPITest(APITestCase):
    """Test cases for Order API."""
    
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email='test@example.com',
            password='testpass123',
            first_name='Test',
            last_name='User'
        )
        
        # Create order
        self.order = Order.objects.create(
            user=self.user,
            email='test@example.com',
            shipping_first_name='John',
            shipping_last_name='Doe',
            shipping_address_line_1='123 Main St',
            shipping_city='New York',
            shipping_postal_code='10001',
            shipping_country='United States',
            billing_first_name='John',
            billing_last_name='Doe',
            billing_address_line_1='123 Main St',
            billing_city='New York',
            billing_postal_code='10001',
            billing_country='United States',
            subtotal=Decimal('100.00'),
            total=Decimal('105.99'),
        )
    
    def test_list_orders(self):
        """Test listing user orders."""
        self.client.force_authenticate(user=self.user)
        response = self.client.get('/api/v1/orders/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
    
    def test_get_order_detail(self):
        """Test getting order detail."""
        self.client.force_authenticate(user=self.user)
        response = self.client.get(f'/api/v1/orders/{self.order.id}/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
        self.assertEqual(response.data['data']['order_number'], self.order.order_number)
    
    def test_cancel_order(self):
        """Test cancelling order."""
        self.client.force_authenticate(user=self.user)
        
        # Set order to cancellable state
        self.order.status = Order.STATUS_PENDING
        self.order.payment_status = 'pending'
        self.order.save()
        
        response = self.client.post(
            f'/api/v1/orders/{self.order.id}/cancel/',
            {'reason': 'Changed my mind'}
        )
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
    
    def test_cannot_view_other_user_order(self):
        """Test cannot view another user's order."""
        other_user = User.objects.create_user(
            email='other@example.com',
            password='testpass123'
        )
        self.client.force_authenticate(user=other_user)
        
        response = self.client.get(f'/api/v1/orders/{self.order.id}/')
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
    
    def test_unauthenticated_access(self):
        """Test unauthenticated access is denied."""
        response = self.client.get('/api/v1/orders/')
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
