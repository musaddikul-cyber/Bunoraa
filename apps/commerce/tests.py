from decimal import Decimal

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APITestCase

from apps.commerce.models import CheckoutSession, SavedForLater
from apps.catalog.models import Product
from apps.commerce.models import Cart, CartItem, SessionWishlist, SessionWishlistItem, Wishlist
from apps.commerce.services import SessionMergeService
from apps.promotions.models import Coupon


User = get_user_model()
TEST_MIDDLEWARE = [
    middleware
    for middleware in settings.MIDDLEWARE
    if middleware != "debug_toolbar.middleware.DebugToolbarMiddleware"
]


def create_product(name: str = "Session Merge Product") -> Product:
    return Product.objects.create(
        name=name,
        price=Decimal("199.00"),
        stock_quantity=25,
        is_active=True,
    )


def response_data(response):
    payload = response.json()
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def response_items(response):
    data = response_data(response)
    if isinstance(data, dict):
        if isinstance(data.get("results"), list):
            return data["results"]
        if isinstance(data.get("items"), list):
            return data["items"]
    if isinstance(data, list):
        return data
    return []


class SessionMergeServiceTests(TestCase):
    def test_merge_guest_cart_and_wishlist_to_user_and_clear_session_records(self):
        user = User.objects.create_user(email="merge@test.com", password="pass12345")
        product = create_product()
        session_key = "guest-session-key-merge"

        guest_cart = Cart.objects.create(session_key=session_key)
        CartItem.objects.create(
            cart=guest_cart,
            product=product,
            quantity=2,
            price_at_add=product.current_price,
        )

        session_wishlist = SessionWishlist.objects.create(session_key=session_key)
        SessionWishlistItem.objects.create(
            wishlist=session_wishlist,
            product=product,
            desired_quantity=3,
            priority=3,
        )

        result = SessionMergeService.merge_guest_state_to_user(user=user, session_key=session_key)

        self.assertGreaterEqual(result["merged_cart_items"], 1)
        self.assertGreaterEqual(result["merged_wishlist_items"], 1)

        user_cart = Cart.objects.filter(user=user).first()
        self.assertIsNotNone(user_cart)
        self.assertEqual(user_cart.items.count(), 1)
        self.assertEqual(user_cart.items.first().quantity, 2)

        user_wishlist = Wishlist.objects.filter(user=user).first()
        self.assertIsNotNone(user_wishlist)
        self.assertEqual(user_wishlist.items.count(), 1)

        self.assertFalse(Cart.objects.filter(session_key=session_key, user__isnull=True).exists())
        self.assertFalse(SessionWishlist.objects.filter(session_key=session_key).exists())

    def test_merge_moves_active_guest_checkout_session_to_user_cart(self):
        user = User.objects.create_user(email="checkoutmerge@test.com", password="pass12345")
        product = create_product("Checkout Merge Product")
        session_key = "guest-checkout-session-key"

        guest_cart = Cart.objects.create(session_key=session_key)
        CartItem.objects.create(
            cart=guest_cart,
            product=product,
            quantity=1,
            price_at_add=product.current_price,
        )
        guest_checkout = CheckoutSession.objects.create(
            cart=guest_cart,
            session_key=session_key,
            current_step=CheckoutSession.STEP_SHIPPING,
            shipping_email="guest@example.com",
        )

        result = SessionMergeService.merge_guest_state_to_user(user=user, session_key=session_key)

        user_cart = Cart.objects.filter(user=user).first()
        self.assertIsNotNone(user_cart)
        migrated = CheckoutSession.objects.filter(id=guest_checkout.id).first()
        self.assertIsNotNone(migrated)
        self.assertEqual(migrated.user_id, user.id)
        self.assertEqual(migrated.cart_id, user_cart.id)
        self.assertIsNone(migrated.session_key)
        self.assertGreaterEqual(result["migrated_checkout_sessions"], 1)

    def test_merge_saved_for_later_deduplicates(self):
        user = User.objects.create_user(email="savedlater@test.com", password="pass12345")
        product = create_product("Saved For Later Product")
        session_key = "guest-saved-later-key"

        SavedForLater.objects.create(
            user=user,
            product=product,
            quantity=2,
            price_at_save=Decimal("210.00"),
        )
        SavedForLater.objects.create(
            session_key=session_key,
            product=product,
            quantity=3,
            price_at_save=Decimal("190.00"),
            notify_on_price_drop=True,
            notify_on_restock=True,
        )

        result = SessionMergeService.merge_guest_state_to_user(user=user, session_key=session_key)

        merged_item = SavedForLater.objects.filter(user=user, product=product).first()
        self.assertIsNotNone(merged_item)
        self.assertEqual(merged_item.quantity, 5)
        self.assertEqual(merged_item.price_at_save, Decimal("190.00"))
        self.assertEqual(result["merged_saved_for_later_items"], 1)
        self.assertFalse(
            SavedForLater.objects.filter(session_key=session_key, user__isnull=True).exists()
        )

    def test_merge_adopts_valid_guest_coupon_when_user_cart_has_none(self):
        user = User.objects.create_user(email="couponmerge@test.com", password="pass12345")
        product = create_product("Coupon Merge Product")
        session_key = "guest-coupon-session"

        coupon = Coupon.objects.create(
            code="GUEST10",
            discount_type=Coupon.DISCOUNT_FIXED,
            discount_value=Decimal("10.00"),
            is_active=True,
        )

        guest_cart = Cart.objects.create(session_key=session_key, coupon=coupon)
        CartItem.objects.create(
            cart=guest_cart,
            product=product,
            quantity=1,
            price_at_add=product.current_price,
        )

        result = SessionMergeService.merge_guest_state_to_user(user=user, session_key=session_key)

        user_cart = Cart.objects.filter(user=user).first()
        self.assertIsNotNone(user_cart)
        self.assertEqual(user_cart.coupon_id, coupon.id)
        self.assertEqual(result["adopted_guest_coupon"], 1)


@override_settings(DEBUG=False, SECURE_SSL_REDIRECT=False, MIDDLEWARE=TEST_MIDDLEWARE)
class GuestWishlistApiTests(APITestCase):
    def setUp(self):
        self.product = create_product("Guest Wishlist Product")

    def test_guest_can_add_list_and_move_wishlist_item_to_cart(self):
        add_response = self.client.post(
            "/api/v1/commerce/wishlist/",
            {"product_id": str(self.product.id)},
            format="json",
        )
        self.assertIn(add_response.status_code, (200, 201))
        add_payload = response_data(add_response)
        item_id = add_payload["item"]["id"] if isinstance(add_payload, dict) else add_response.json()["item"]["id"]

        list_response = self.client.get("/api/v1/commerce/wishlist/")
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(len(response_items(list_response)), 1)

        move_response = self.client.post(
            f"/api/v1/commerce/wishlist/move-to-cart/{item_id}/",
            {},
            format="json",
        )
        self.assertEqual(move_response.status_code, 200)

        cart_response = self.client.get("/api/v1/commerce/cart/")
        self.assertEqual(cart_response.status_code, 200)
        cart_data = response_data(cart_response)
        self.assertEqual(cart_data["item_count"], 1)

        wishlist_after_move = self.client.get("/api/v1/commerce/wishlist/")
        self.assertEqual(wishlist_after_move.status_code, 200)
        self.assertEqual(len(response_items(wishlist_after_move)), 0)

    def test_login_merges_guest_cart_and_wishlist_and_clears_session_rows(self):
        guest_cart_response = self.client.post(
            "/api/v1/commerce/cart/add/",
            {"product_id": str(self.product.id), "quantity": 1},
            format="json",
        )
        self.assertEqual(guest_cart_response.status_code, 201)

        guest_wishlist_response = self.client.post(
            "/api/v1/commerce/wishlist/",
            {"product_id": str(self.product.id)},
            format="json",
        )
        self.assertIn(guest_wishlist_response.status_code, (200, 201))

        session_key = self.client.session.session_key
        self.assertTrue(session_key)
        self.assertTrue(Cart.objects.filter(session_key=session_key, user__isnull=True).exists())
        self.assertTrue(SessionWishlist.objects.filter(session_key=session_key).exists())

        user = User.objects.create_user(email="guestmerge@login.com", password="pass12345")

        login_response = self.client.post(
            "/api/v1/auth/token/",
            {"email": user.email, "password": "pass12345"},
            format="json",
        )
        self.assertEqual(login_response.status_code, 200)
        access = response_data(login_response)["access"]
        self.assertTrue(access)
        rotated_session_key = self.client.session.session_key
        self.assertTrue(rotated_session_key)
        self.assertNotEqual(rotated_session_key, session_key)

        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {access}")

        cart_response = self.client.get("/api/v1/commerce/cart/")
        self.assertEqual(cart_response.status_code, 200)
        cart_data = response_data(cart_response)
        self.assertEqual(cart_data["item_count"], 1)

        wishlist_response = self.client.get("/api/v1/commerce/wishlist/")
        self.assertEqual(wishlist_response.status_code, 200)
        self.assertEqual(len(response_items(wishlist_response)), 1)

        self.assertFalse(Cart.objects.filter(session_key=session_key, user__isnull=True).exists())
        self.assertFalse(SessionWishlist.objects.filter(session_key=session_key).exists())
