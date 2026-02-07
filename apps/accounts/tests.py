"""
Account tests
"""
from django.test import TestCase
from django.urls import reverse
import pyotp
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from .models import User, Address
from .services import UserService, AddressService, MfaService
from .behavior_models import UserSession


class UserModelTest(TestCase):
    """Tests for User model."""
    
    def test_create_user(self):
        """Test creating a regular user."""
        user = User.objects.create_user(
            email='test@example.com',
            password='testpass123',
            first_name='Test',
            last_name='User'
        )
        self.assertEqual(user.email, 'test@example.com')
        self.assertTrue(user.check_password('testpass123'))
        self.assertFalse(user.is_staff)
        self.assertFalse(user.is_superuser)
    
    def test_create_superuser(self):
        """Test creating a superuser."""
        admin = User.objects.create_superuser(
            email='admin@example.com',
            password='adminpass123'
        )
        self.assertTrue(admin.is_staff)
        self.assertTrue(admin.is_superuser)
    
    def test_user_string_representation(self):
        """Test user string representation."""
        user = User.objects.create_user(
            email='test@example.com',
            password='testpass123'
        )
        self.assertEqual(str(user), 'test@example.com')
    
    def test_get_full_name(self):
        """Test get_full_name method."""
        user = User.objects.create_user(
            email='test@example.com',
            password='testpass123',
            first_name='Test',
            last_name='User'
        )
        self.assertEqual(user.get_full_name(), 'Test User')
    
    def test_soft_delete(self):
        """Test soft delete functionality."""
        user = User.objects.create_user(
            email='test@example.com',
            password='testpass123'
        )
        user.soft_delete()
        self.assertTrue(user.is_deleted)
        self.assertFalse(user.is_active)


class AddressModelTest(TestCase):
    """Tests for Address model."""
    
    def setUp(self):
        self.user = User.objects.create_user(
            email='test@example.com',
            password='testpass123'
        )
    
    def test_create_address(self):
        """Test creating an address."""
        address = Address.objects.create(
            user=self.user,
            full_name='Test User',
            phone='+1234567890',
            address_line_1='123 Test St',
            city='Test City',
            state='Test State',
            postal_code='12345',
            country='Bangladesh'
        )
        self.assertEqual(address.user, self.user)
        self.assertEqual(address.city, 'Test City')
    
    def test_default_address(self):
        """Test default address functionality."""
        address1 = Address.objects.create(
            user=self.user,
            full_name='Test User',
            phone='+1234567890',
            address_line_1='123 Test St',
            city='Test City',
            state='Test State',
            postal_code='12345',
            is_default=True
        )
        address2 = Address.objects.create(
            user=self.user,
            full_name='Test User 2',
            phone='+1234567890',
            address_line_1='456 Test Ave',
            city='Test City 2',
            state='Test State 2',
            postal_code='67890',
            is_default=True
        )
        address1.refresh_from_db()
        self.assertFalse(address1.is_default)
        self.assertTrue(address2.is_default)


class UserAPITest(APITestCase):
    """API tests for user endpoints."""
    
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email='test@example.com',
            password='testpass123',
            first_name='Test',
            last_name='User'
        )
    
    def test_register_user(self):
        """Test user registration endpoint."""
        data = {
            'email': 'newuser@example.com',
            'password': 'newuserpass123!',
            'password_confirm': 'newuserpass123!',
            'first_name': 'New',
            'last_name': 'User'
        }
        response = self.client.post('/api/v1/accounts/register/', data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(response.data['success'])
    
    def test_get_profile(self):
        """Test getting user profile."""
        self.client.force_authenticate(user=self.user)
        response = self.client.get('/api/v1/accounts/profile/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['email'], 'test@example.com')
    
    def test_update_profile(self):
        """Test updating user profile."""
        self.client.force_authenticate(user=self.user)
        data = {'first_name': 'Updated'}
        response = self.client.patch('/api/v1/accounts/profile/', data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['first_name'], 'Updated')
    
    def test_change_password(self):
        """Test password change."""
        self.client.force_authenticate(user=self.user)
        data = {
            'current_password': 'testpass123',
            'new_password': 'newtestpass123!',
            'new_password_confirm': 'newtestpass123!'
        }
        response = self.client.post('/api/v1/accounts/password/change/', data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_mfa_totp_setup_and_verify(self):
        """Test TOTP setup and verification flow."""
        self.client.force_authenticate(user=self.user)
        setup = self.client.post('/api/v1/accounts/mfa/totp/setup/')
        self.assertEqual(setup.status_code, status.HTTP_200_OK)
        secret = setup.data['data']['secret']
        code = pyotp.TOTP(secret).now()
        verify = self.client.post('/api/v1/accounts/mfa/totp/verify/', {'code': code})
        self.assertEqual(verify.status_code, status.HTTP_200_OK)
        self.assertIn('backup_codes', verify.data['data'])

    def test_mfa_verify_exchange(self):
        """Test MFA token exchange for JWTs."""
        self.client.force_authenticate(user=self.user)
        setup = self.client.post('/api/v1/accounts/mfa/totp/setup/')
        secret = setup.data['data']['secret']
        enable_code = pyotp.TOTP(secret).now()
        self.client.post('/api/v1/accounts/mfa/totp/verify/', {'code': enable_code})
        mfa_token = MfaService.create_mfa_token(self.user)
        verify_code = pyotp.TOTP(secret).now()
        response = self.client.post('/api/v1/accounts/mfa/verify/', {
            'mfa_token': mfa_token,
            'method': 'totp',
            'code': verify_code
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data['data'])

    def test_preferences(self):
        """Test preferences endpoints."""
        self.client.force_authenticate(user=self.user)
        response = self.client.get('/api/v1/accounts/preferences/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        patch = self.client.patch('/api/v1/accounts/preferences/', {'theme': 'dark'})
        self.assertEqual(patch.status_code, status.HTTP_200_OK)

    def test_export_request(self):
        """Test data export request endpoint."""
        self.client.force_authenticate(user=self.user)
        response = self.client.post('/api/v1/accounts/export/', {})
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    def test_account_deletion_request(self):
        """Test account deletion request endpoint."""
        self.client.force_authenticate(user=self.user)
        response = self.client.post('/api/v1/accounts/delete/', {'reason': 'privacy'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_session_revoke(self):
        """Test revoking an auth session."""
        self.client.force_authenticate(user=self.user)
        session = UserSession.objects.create(
            user=self.user,
            session_key='testsession',
            session_type=UserSession.SESSION_TYPE_AUTH,
            access_jti='access-jti',
            refresh_jti='refresh-jti',
            is_active=True
        )
        response = self.client.post(f'/api/v1/accounts/sessions/{session.id}/revoke/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        session.refresh_from_db()
        self.assertFalse(session.is_active)


class AddressAPITest(APITestCase):
    """API tests for address endpoints."""
    
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email='test@example.com',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)
        self.address = Address.objects.create(
            user=self.user,
            full_name='Test User',
            phone='+1234567890',
            address_line_1='123 Test St',
            city='Test City',
            state='Test State',
            postal_code='12345',
            country='Bangladesh'
        )
    
    def test_list_addresses(self):
        """Test listing addresses."""
        response = self.client.get('/api/v1/accounts/addresses/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']), 1)
    
    def test_create_address(self):
        """Test creating an address."""
        data = {
            'full_name': 'New Address',
            'phone': '+1234567890',
            'address_line_1': '456 New St',
            'city': 'New City',
            'state': 'New State',
            'postal_code': '67890',
            'country': 'Bangladesh'
        }
        response = self.client.post('/api/v1/accounts/addresses/', data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
    
    def test_update_address(self):
        """Test updating an address."""
        data = {'city': 'Updated City'}
        response = self.client.patch(f'/api/v1/accounts/addresses/{self.address.id}/', data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['city'], 'Updated City')
    
    def test_delete_address(self):
        """Test deleting an address."""
        response = self.client.delete(f'/api/v1/accounts/addresses/{self.address.id}/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.address.refresh_from_db()
        self.assertTrue(self.address.is_deleted)
