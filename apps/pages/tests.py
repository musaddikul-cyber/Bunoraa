"""
Pages app tests
"""
import uuid
from django.test import TestCase, Client
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status

from .models import Page, FAQ, ContactMessage, SiteSettings, Subscriber
from .services import PageService, FAQService, ContactService, SubscriberService


class PageModelTest(TestCase):
    """Tests for Page model."""
    
    def setUp(self):
        self.page = Page.objects.create(
            title='About Us',
            slug='about-us',
            content='<p>About our company</p>',
            is_published=True,
            show_in_menu=True,
            menu_order=1
        )
    
    def test_page_creation(self):
        """Test page is created correctly."""
        self.assertEqual(self.page.title, 'About Us')
        self.assertEqual(self.page.slug, 'about-us')
        self.assertTrue(self.page.is_published)
    
    def test_page_str(self):
        """Test page string representation."""
        self.assertEqual(str(self.page), 'About Us')
    
    def test_page_uuid_pk(self):
        """Test page uses UUID as primary key."""
        self.assertIsInstance(self.page.id, uuid.UUID)


class FAQModelTest(TestCase):
    """Tests for FAQ model."""
    
    def setUp(self):
        self.faq = FAQ.objects.create(
            question='What is your return policy?',
            answer='30-day return policy',
            category='Returns',
            is_active=True,
            sort_order=1
        )
    
    def test_faq_creation(self):
        """Test FAQ is created correctly."""
        self.assertEqual(self.faq.question, 'What is your return policy?')
        self.assertEqual(self.faq.category, 'Returns')
    
    def test_faq_str(self):
        """Test FAQ string representation."""
        self.assertEqual(str(self.faq), 'What is your return policy?')


class SiteSettingsModelTest(TestCase):
    """Tests for SiteSettings model."""
    
    def test_singleton_pattern(self):
        """Test site settings singleton pattern."""
        settings1 = SiteSettings.get_settings()
        settings2 = SiteSettings.get_settings()
        
        self.assertEqual(settings1.pk, settings2.pk)
        self.assertEqual(settings1.pk, 1)
    
    def test_settings_creation(self):
        """Test settings are created with defaults."""
        settings = SiteSettings.get_settings()
        self.assertIsNotNone(settings.site_name)


class ContactMessageModelTest(TestCase):
    """Tests for ContactMessage model."""
    
    def setUp(self):
        self.message = ContactMessage.objects.create(
            name='John Doe',
            email='john@example.com',
            subject='Inquiry',
            message='I have a question...'
        )
    
    def test_message_creation(self):
        """Test contact message is created correctly."""
        self.assertEqual(self.message.name, 'John Doe')
        self.assertEqual(self.message.email, 'john@example.com')
        self.assertFalse(self.message.is_read)
        self.assertFalse(self.message.is_replied)


class SubscriberModelTest(TestCase):
    """Tests for Subscriber model."""
    
    def setUp(self):
        self.subscriber = Subscriber.objects.create(
            email='subscriber@example.com',
            name='Jane Doe',
            source='website'
        )
    
    def test_subscriber_creation(self):
        """Test subscriber is created correctly."""
        self.assertEqual(self.subscriber.email, 'subscriber@example.com')
        self.assertTrue(self.subscriber.is_active)


class PageServiceTest(TestCase):
    """Tests for PageService."""
    
    def setUp(self):
        self.page1 = Page.objects.create(
            title='Page 1',
            slug='page-1',
            is_published=True,
            show_in_menu=True,
            menu_order=1
        )
        self.page2 = Page.objects.create(
            title='Page 2',
            slug='page-2',
            is_published=True,
            show_in_footer=True
        )
        self.unpublished = Page.objects.create(
            title='Unpublished',
            slug='unpublished',
            is_published=False
        )
    
    def test_get_menu_pages(self):
        """Test getting menu pages."""
        pages = PageService.get_menu_pages()
        self.assertEqual(pages.count(), 1)
        self.assertEqual(pages.first().slug, 'page-1')
    
    def test_get_footer_pages(self):
        """Test getting footer pages."""
        pages = PageService.get_footer_pages()
        self.assertEqual(pages.count(), 1)
        self.assertEqual(pages.first().slug, 'page-2')
    
    def test_get_page_by_slug(self):
        """Test getting page by slug."""
        page = PageService.get_page_by_slug('page-1')
        self.assertEqual(page.title, 'Page 1')
        
        # Unpublished should not be found
        page = PageService.get_page_by_slug('unpublished')
        self.assertIsNone(page)


class FAQServiceTest(TestCase):
    """Tests for FAQService."""
    
    def setUp(self):
        self.faq1 = FAQ.objects.create(
            question='Q1',
            answer='A1',
            category='Shipping',
            is_active=True,
            sort_order=1
        )
        self.faq2 = FAQ.objects.create(
            question='Q2',
            answer='A2',
            category='Returns',
            is_active=True,
            sort_order=2
        )
    
    def test_get_active_faqs(self):
        """Test getting active FAQs."""
        faqs = FAQService.get_active_faqs()
        self.assertEqual(faqs.count(), 2)
    
    def test_get_faqs_by_category(self):
        """Test getting FAQs by category."""
        faqs = FAQService.get_faqs_by_category('Shipping')
        self.assertEqual(faqs.count(), 1)
        self.assertEqual(faqs.first().question, 'Q1')
    
    def test_get_faq_categories(self):
        """Test getting FAQ categories."""
        categories = list(FAQService.get_faq_categories())
        self.assertIn('Shipping', categories)
        self.assertIn('Returns', categories)


class SubscriberServiceTest(TestCase):
    """Tests for SubscriberService."""
    
    def test_subscribe(self):
        """Test subscribing to newsletter."""
        result = SubscriberService.subscribe('new@example.com', 'New User')
        self.assertTrue(result['success'])
        
        subscriber = Subscriber.objects.get(email='new@example.com')
        self.assertEqual(subscriber.name, 'New User')
    
    def test_subscribe_duplicate(self):
        """Test subscribing with existing email."""
        SubscriberService.subscribe('dupe@example.com')
        result = SubscriberService.subscribe('dupe@example.com')
        self.assertFalse(result['success'])
    
    def test_unsubscribe(self):
        """Test unsubscribing."""
        SubscriberService.subscribe('unsub@example.com')
        result = SubscriberService.unsubscribe('unsub@example.com')
        self.assertTrue(result['success'])
        
        subscriber = Subscriber.objects.get(email='unsub@example.com')
        self.assertFalse(subscriber.is_active)
    
    def test_resubscribe(self):
        """Test re-subscribing after unsubscribe."""
        SubscriberService.subscribe('resub@example.com')
        SubscriberService.unsubscribe('resub@example.com')
        result = SubscriberService.subscribe('resub@example.com')
        self.assertTrue(result['success'])
        
        subscriber = Subscriber.objects.get(email='resub@example.com')
        self.assertTrue(subscriber.is_active)


class PagesAPITest(TestCase):
    """Tests for Pages API."""
    
    def setUp(self):
        self.client = APIClient()
        self.page = Page.objects.create(
            title='Test Page',
            slug='test-page',
            content='<p>Test content</p>',
            is_published=True,
            show_in_menu=True
        )
    
    def test_list_pages(self):
        """Test listing pages."""
        response = self.client.get('/api/v1/pages/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
    
    def test_get_page_detail(self):
        """Test getting page detail."""
        response = self.client.get(f'/api/v1/pages/{self.page.slug}/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['title'], 'Test Page')
    
    def test_get_menu_pages(self):
        """Test getting menu pages."""
        response = self.client.get('/api/v1/pages/menu/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])


class FAQAPITest(TestCase):
    """Tests for FAQ API."""
    
    def setUp(self):
        self.client = APIClient()
        self.faq = FAQ.objects.create(
            question='Test Question',
            answer='Test Answer',
            category='General',
            is_active=True
        )
    
    def test_list_faqs(self):
        """Test listing FAQs."""
        response = self.client.get('/api/v1/faqs/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
    
    def test_get_grouped_faqs(self):
        """Test getting grouped FAQs."""
        response = self.client.get('/api/v1/faqs/grouped/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])


class ContactAPITest(TestCase):
    """Tests for Contact API."""
    
    def setUp(self):
        self.client = APIClient()
    
    def test_submit_contact_message(self):
        """Test submitting contact message."""
        data = {
            'name': 'Test User',
            'email': 'test@example.com',
            'subject': 'Test Subject',
            'message': 'Test message content'
        }
        response = self.client.post('/api/v1/contact/', data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(response.data['success'])
        
        # Verify message was created
        message = ContactMessage.objects.filter(email='test@example.com').first()
        self.assertIsNotNone(message)


class SubscriberAPITest(TestCase):
    """Tests for Subscriber API."""
    
    def setUp(self):
        self.client = APIClient()
    
    def test_subscribe(self):
        """Test newsletter subscription."""
        data = {'email': 'newsubscriber@example.com', 'name': 'New Sub'}
        response = self.client.post('/api/v1/subscribers/', data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(response.data['success'])
    
    def test_unsubscribe(self):
        """Test newsletter unsubscribe."""
        # First subscribe
        Subscriber.objects.create(email='unsub@example.com', is_active=True)
        
        # Then unsubscribe
        data = {'email': 'unsub@example.com'}
        response = self.client.post('/api/v1/subscribers/unsubscribe/', data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])


class SiteSettingsAPITest(TestCase):
    """Tests for Site Settings API."""
    
    def setUp(self):
        self.client = APIClient()
    
    def test_get_settings(self):
        """Test getting site settings."""
        response = self.client.get('/api/v1/settings/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
