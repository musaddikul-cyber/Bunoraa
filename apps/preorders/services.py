"""
Pre-orders services - Business logic for custom pre-order management
"""
import logging
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from django.db import transaction
from django.utils import timezone
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.conf import settings
from django.db.models import Q, Sum, Count

from .models import (
    PreOrderCategory, PreOrderOption, PreOrderOptionChoice,
    PreOrder, PreOrderItem, PreOrderOptionValue, PreOrderDesign,
    PreOrderReference, PreOrderStatusHistory, PreOrderPayment,
    PreOrderMessage, PreOrderRevision, PreOrderQuote, PreOrderTemplate
)

logger = logging.getLogger(__name__)


class PreOrderService:
    """Service for pre-order operations."""
    
    @staticmethod
    @transaction.atomic
    def create_preorder(
        user=None,
        category: PreOrderCategory = None,
        data: Dict = None,
        options_data: Dict = None,
        files: List = None,
        references: List = None
    ) -> PreOrder:
        """
        Create a new pre-order.
        
        Args:
            user: Authenticated user (optional for guest)
            category: PreOrderCategory instance
            data: Dictionary with pre-order data
            options_data: Dictionary with option values
            files: List of design file uploads
            references: List of reference image uploads
            
        Returns:
            PreOrder instance
        """
        # Calculate deposit required
        deposit_required = (category.base_price * category.deposit_percentage) / 100
        
        # Create pre-order
        preorder = PreOrder.objects.create(
            user=user,
            category=category,
            email=data.get('email', user.email if user else ''),
            phone=data.get('phone', ''),
            full_name=data.get('full_name', user.get_full_name() if user else ''),
            title=data.get('title', ''),
            description=data.get('description', ''),
            quantity=data.get('quantity', 1),
            special_instructions=data.get('special_instructions', ''),
            is_gift=data.get('is_gift', False),
            gift_wrap=data.get('gift_wrap', False),
            gift_message=data.get('gift_message', ''),
            is_rush_order=data.get('is_rush_order', False),
            requested_delivery_date=data.get('requested_delivery_date'),
            customer_notes=data.get('customer_notes', ''),
            deposit_required=deposit_required,
            estimated_price=category.base_price * data.get('quantity', 1),
            currency=data.get('currency', 'BDT'),
            # Shipping address
            shipping_first_name=data.get('shipping_first_name', ''),
            shipping_last_name=data.get('shipping_last_name', ''),
            shipping_address_line_1=data.get('shipping_address_line_1', ''),
            shipping_address_line_2=data.get('shipping_address_line_2', ''),
            shipping_city=data.get('shipping_city', ''),
            shipping_state=data.get('shipping_state', ''),
            shipping_postal_code=data.get('shipping_postal_code', ''),
            shipping_country=data.get('shipping_country', 'Bangladesh'),
            # Source tracking
            source=data.get('source', ''),
            utm_source=data.get('utm_source', ''),
            utm_medium=data.get('utm_medium', ''),
            utm_campaign=data.get('utm_campaign', ''),
            status=PreOrder.STATUS_DRAFT,
        )
        
        # Calculate rush order fee
        if preorder.is_rush_order and category.allow_rush_order:
            preorder.rush_order_fee = (preorder.estimated_price * category.rush_order_fee_percentage) / 100
            preorder.save()
        
        # Save option values
        if options_data:
            PreOrderService._save_option_values(preorder, options_data)
        
        # Save design files
        if files:
            for file in files:
                PreOrderDesign.objects.create(
                    preorder=preorder,
                    file=file,
                    original_filename=file.name,
                    design_type=PreOrderDesign.DESIGN_CUSTOMER,
                    uploaded_by=user
                )
        
        # Save reference images
        if references:
            for ref in references:
                PreOrderReference.objects.create(
                    preorder=preorder,
                    file=ref,
                    original_filename=ref.name
                )
        
        # Create status history
        PreOrderStatusHistory.objects.create(
            preorder=preorder,
            from_status='',
            to_status=PreOrder.STATUS_DRAFT,
            is_system=True,
            notes='Pre-order created'
        )
        
        logger.info(f"Created pre-order {preorder.preorder_number}")
        return preorder
    
    @staticmethod
    def _save_option_values(preorder: PreOrder, options_data: Dict):
        """Save option values for a pre-order."""
        total_price_modifier = Decimal('0.00')
        
        for option_id, value in options_data.items():
            try:
                option = PreOrderOption.objects.get(id=option_id)
                option_value = PreOrderOptionValue.objects.create(
                    preorder=preorder,
                    option=option
                )
                
                price_modifier = option.price_modifier
                
                if option.option_type in [PreOrderOption.OPTION_TEXT, PreOrderOption.OPTION_TEXTAREA, PreOrderOption.OPTION_COLOR]:
                    option_value.text_value = value
                elif option.option_type == PreOrderOption.OPTION_NUMBER:
                    option_value.number_value = Decimal(value) if value else None
                elif option.option_type == PreOrderOption.OPTION_CHECKBOX:
                    option_value.boolean_value = bool(value)
                elif option.option_type == PreOrderOption.OPTION_DATE:
                    option_value.date_value = value
                elif option.option_type == PreOrderOption.OPTION_SELECT:
                    choice = PreOrderOptionChoice.objects.get(id=value)
                    option_value.choice_value = choice
                    price_modifier += choice.price_modifier
                elif option.option_type == PreOrderOption.OPTION_MULTISELECT:
                    choices = PreOrderOptionChoice.objects.filter(id__in=value)
                    option_value.save()
                    option_value.multi_choice_values.set(choices)
                    price_modifier += sum(c.price_modifier for c in choices)
                
                option_value.price_modifier_applied = price_modifier
                option_value.save()
                total_price_modifier += price_modifier
                
            except (PreOrderOption.DoesNotExist, PreOrderOptionChoice.DoesNotExist) as e:
                logger.warning(f"Invalid option or choice: {e}")
                continue
        
        # Update estimated price with modifiers
        if total_price_modifier > 0:
            preorder.estimated_price += total_price_modifier * preorder.quantity
            preorder.save()
    
    @staticmethod
    @transaction.atomic
    def submit_preorder(preorder: PreOrder, user=None) -> PreOrder:
        """
        Submit a pre-order for review.
        
        Args:
            preorder: PreOrder instance
            user: User submitting (optional)
            
        Returns:
            Updated PreOrder instance
        """
        if preorder.status != PreOrder.STATUS_DRAFT:
            raise ValueError("Can only submit draft pre-orders")
        
        old_status = preorder.status
        preorder.status = PreOrder.STATUS_SUBMITTED
        preorder.submitted_at = timezone.now()
        preorder.save()
        
        # Create status history
        PreOrderStatusHistory.objects.create(
            preorder=preorder,
            from_status=old_status,
            to_status=preorder.status,
            changed_by=user,
            notes='Pre-order submitted for review'
        )
        
        # Send notification emails
        PreOrderService.send_submission_notification(preorder)
        
        logger.info(f"Submitted pre-order {preorder.preorder_number}")
        return preorder
    
    @staticmethod
    @transaction.atomic
    def update_status(
        preorder: PreOrder,
        new_status: str,
        user=None,
        notes: str = ''
    ) -> PreOrder:
        """
        Update pre-order status with proper tracking.
        
        Args:
            preorder: PreOrder instance
            new_status: New status string
            user: User making the change
            notes: Optional notes
            
        Returns:
            Updated PreOrder instance
        """
        old_status = preorder.status
        preorder.status = new_status
        
        # Update relevant timestamps
        timestamp_map = {
            PreOrder.STATUS_SUBMITTED: 'submitted_at',
            PreOrder.STATUS_QUOTED: 'quoted_at',
            PreOrder.STATUS_QUOTE_ACCEPTED: 'approved_at',
            PreOrder.STATUS_IN_PRODUCTION: 'production_started_at',
            PreOrder.STATUS_COMPLETED: 'completed_at',
            PreOrder.STATUS_SHIPPED: 'shipped_at',
            PreOrder.STATUS_DELIVERED: 'delivered_at',
            PreOrder.STATUS_CANCELLED: 'cancelled_at',
        }
        
        if new_status in timestamp_map:
            setattr(preorder, timestamp_map[new_status], timezone.now())
        
        if new_status == PreOrder.STATUS_IN_PRODUCTION:
            preorder.production_start_date = timezone.now().date()
        elif new_status == PreOrder.STATUS_COMPLETED:
            preorder.actual_completion_date = timezone.now().date()
        
        preorder.save()
        
        # Create status history
        PreOrderStatusHistory.objects.create(
            preorder=preorder,
            from_status=old_status,
            to_status=new_status,
            changed_by=user,
            notes=notes
        )
        
        # Send status update notification
        PreOrderService.send_status_update_notification(preorder, old_status)
        
        logger.info(f"Updated pre-order {preorder.preorder_number} status: {old_status} → {new_status}")
        return preorder
    
    @staticmethod
    @transaction.atomic
    def create_quote(
        preorder: PreOrder,
        quote_data: Dict,
        user=None
    ) -> PreOrderQuote:
        """
        Create a quote for a pre-order.
        
        Args:
            preorder: PreOrder instance
            quote_data: Dictionary with quote details
            user: User creating the quote
            
        Returns:
            PreOrderQuote instance
        """
        # Mark previous quotes as superseded
        PreOrderQuote.objects.filter(
            preorder=preorder,
            status__in=[PreOrderQuote.STATUS_PENDING, PreOrderQuote.STATUS_SENT]
        ).update(status=PreOrderQuote.STATUS_SUPERSEDED)
        
        # Calculate total
        base_price = Decimal(quote_data.get('base_price', '0'))
        customization_cost = Decimal(quote_data.get('customization_cost', '0'))
        rush_fee = Decimal(quote_data.get('rush_fee', '0'))
        discount = Decimal(quote_data.get('discount', '0'))
        shipping = Decimal(quote_data.get('shipping', '0'))
        tax = Decimal(quote_data.get('tax', '0'))
        total = base_price + customization_cost + rush_fee - discount + shipping + tax
        
        # Create quote
        quote = PreOrderQuote.objects.create(
            preorder=preorder,
            base_price=base_price,
            customization_cost=customization_cost,
            rush_fee=rush_fee,
            discount=discount,
            shipping=shipping,
            tax=tax,
            total=total,
            valid_from=timezone.now(),
            valid_until=quote_data.get('valid_until', timezone.now() + timezone.timedelta(days=7)),
            estimated_production_days=quote_data.get('estimated_production_days', 14),
            estimated_delivery_date=quote_data.get('estimated_delivery_date'),
            terms=quote_data.get('terms', ''),
            notes=quote_data.get('notes', ''),
            created_by=user
        )
        
        # Update pre-order
        preorder.estimated_price = total
        preorder.quote_valid_until = quote.valid_until
        preorder.quote_notes = quote.notes
        preorder.status = PreOrder.STATUS_QUOTED
        preorder.quoted_at = timezone.now()
        
        # Calculate deposit
        preorder.deposit_required = (total * preorder.category.deposit_percentage) / 100
        preorder.save()
        
        # Status history
        PreOrderStatusHistory.objects.create(
            preorder=preorder,
            from_status=PreOrder.STATUS_UNDER_REVIEW,
            to_status=PreOrder.STATUS_QUOTED,
            changed_by=user,
            notes=f'Quote {quote.quote_number} created'
        )
        
        logger.info(f"Created quote {quote.quote_number} for pre-order {preorder.preorder_number}")
        return quote
    
    @staticmethod
    @transaction.atomic
    def accept_quote(preorder: PreOrder, quote: PreOrderQuote, user=None) -> PreOrder:
        """Accept a quote and move to deposit pending."""
        quote.status = PreOrderQuote.STATUS_ACCEPTED
        quote.responded_at = timezone.now()
        quote.save()
        
        preorder.final_price = quote.total
        preorder.total_amount = quote.total
        preorder.tax_amount = quote.tax
        preorder.shipping_cost = quote.shipping
        preorder.discount_amount = quote.discount
        preorder.estimated_completion_date = quote.estimated_delivery_date
        
        # Recalculate remaining amount
        preorder.amount_remaining = preorder.total_amount - preorder.amount_paid
        
        PreOrderService.update_status(
            preorder,
            PreOrder.STATUS_DEPOSIT_PENDING,
            user,
            f'Quote {quote.quote_number} accepted'
        )
        
        return preorder
    
    @staticmethod
    @transaction.atomic
    def reject_quote(preorder: PreOrder, quote: PreOrderQuote, user=None, reason: str = '') -> PreOrder:
        """Reject a quote."""
        quote.status = PreOrderQuote.STATUS_REJECTED
        quote.responded_at = timezone.now()
        quote.customer_response_notes = reason
        quote.save()
        
        PreOrderService.update_status(
            preorder,
            PreOrder.STATUS_QUOTE_REJECTED,
            user,
            f'Quote {quote.quote_number} rejected: {reason}'
        )
        
        return preorder
    
    @staticmethod
    @transaction.atomic
    def process_payment(
        preorder: PreOrder,
        amount: Decimal,
        payment_method: str,
        payment_type: str = PreOrderPayment.PAYMENT_DEPOSIT,
        transaction_id: str = '',
        gateway_response: Dict = None,
        user=None
    ) -> PreOrderPayment:
        """
        Process a payment for a pre-order.
        
        Args:
            preorder: PreOrder instance
            amount: Payment amount
            payment_method: Payment method
            payment_type: Type of payment (deposit, partial, final)
            transaction_id: External transaction ID
            gateway_response: Raw gateway response
            user: User recording the payment
            
        Returns:
            PreOrderPayment instance
        """
        payment = PreOrderPayment.objects.create(
            preorder=preorder,
            payment_type=payment_type,
            amount=amount,
            currency=preorder.currency,
            payment_method=payment_method,
            transaction_id=transaction_id,
            gateway_response=gateway_response or {},
            status=PreOrderPayment.STATUS_COMPLETED,
            paid_at=timezone.now(),
            recorded_by=user
        )
        
        # Update preorder payment tracking
        preorder.amount_paid += amount
        
        if payment_type == PreOrderPayment.PAYMENT_DEPOSIT:
            preorder.deposit_paid += amount
            if preorder.deposit_paid >= preorder.deposit_required:
                PreOrderService.update_status(
                    preorder,
                    PreOrder.STATUS_DEPOSIT_PAID,
                    user,
                    f'Deposit payment received: {preorder.currency} {amount}'
                )
        
        preorder.amount_remaining = preorder.total_amount - preorder.amount_paid if preorder.total_amount else Decimal('0')
        preorder.save()
        
        # Send payment confirmation
        PreOrderService.send_payment_confirmation(preorder, payment)
        
        logger.info(f"Processed payment of {amount} for pre-order {preorder.preorder_number}")
        return payment
    
    @staticmethod
    @transaction.atomic
    def request_revision(
        preorder: PreOrder,
        description: str,
        user=None
    ) -> PreOrderRevision:
        """Request a revision for a pre-order."""
        # Check revision limit
        if preorder.revision_count >= preorder.max_revisions:
            raise ValueError(f"Maximum revisions ({preorder.max_revisions}) reached")
        
        revision_number = preorder.revision_count + 1
        
        # Calculate additional cost if beyond free revisions
        additional_cost = Decimal('0.00')
        if revision_number > 2:  # First 2 revisions are free
            additional_cost = Decimal('100.00')  # Configurable
        
        revision = PreOrderRevision.objects.create(
            preorder=preorder,
            revision_number=revision_number,
            description=description,
            additional_cost=additional_cost,
            requested_by=user
        )
        
        preorder.revision_count = revision_number
        PreOrderService.update_status(
            preorder,
            PreOrder.STATUS_REVISION_REQUESTED,
            user,
            f'Revision #{revision_number} requested'
        )
        
        return revision
    
    @staticmethod
    @transaction.atomic
    def send_message(
        preorder: PreOrder,
        message: str,
        subject: str = '',
        sender=None,
        is_from_customer: bool = True,
        attachment=None
    ) -> PreOrderMessage:
        """Send a message on a pre-order thread."""
        msg = PreOrderMessage.objects.create(
            preorder=preorder,
            sender=sender,
            is_from_customer=is_from_customer,
            subject=subject,
            message=message,
            attachment=attachment
        )
        
        # Send email notification
        if is_from_customer:
            PreOrderService.send_admin_message_notification(preorder, msg)
        else:
            PreOrderService.send_customer_message_notification(preorder, msg)
        
        return msg
    
    @staticmethod
    def get_preorder_summary(preorder: PreOrder) -> Dict:
        """Get a summary of a pre-order."""
        return {
            'preorder_number': preorder.preorder_number,
            'title': preorder.title,
            'status': preorder.status,
            'status_display': preorder.get_status_display(),
            'category': preorder.category.name,
            'quantity': preorder.quantity,
            'estimated_price': str(preorder.estimated_price or 0),
            'final_price': str(preorder.final_price or 0),
            'total_amount': str(preorder.total_amount or 0),
            'deposit_required': str(preorder.deposit_required),
            'deposit_paid': str(preorder.deposit_paid),
            'amount_paid': str(preorder.amount_paid),
            'amount_remaining': str(preorder.amount_remaining),
            'is_fully_paid': preorder.is_fully_paid,
            'deposit_is_paid': preorder.deposit_is_paid,
            'is_rush_order': preorder.is_rush_order,
            'requested_delivery_date': str(preorder.requested_delivery_date) if preorder.requested_delivery_date else None,
            'estimated_completion_date': str(preorder.estimated_completion_date) if preorder.estimated_completion_date else None,
            'days_until_deadline': preorder.days_until_deadline,
            'created_at': preorder.created_at.isoformat(),
            'items_count': preorder.items.count(),
            'designs_count': preorder.designs.count(),
            'messages_count': preorder.messages.count(),
            'unread_messages': preorder.messages.filter(is_read=False, is_from_customer=False).count(),
        }
    
    @staticmethod
    def get_user_preorders(user, status: str = None) -> List[PreOrder]:
        """Get all pre-orders for a user."""
        queryset = PreOrder.objects.filter(user=user, is_deleted=False)
        if status:
            queryset = queryset.filter(status=status)
        return queryset.select_related('category').order_by('-created_at')
    
    @staticmethod
    def get_preorder_statistics(user=None) -> Dict:
        """Get pre-order statistics."""
        queryset = PreOrder.objects.filter(is_deleted=False)
        if user:
            queryset = queryset.filter(user=user)
        
        return {
            'total': queryset.count(),
            'draft': queryset.filter(status=PreOrder.STATUS_DRAFT).count(),
            'pending': queryset.filter(status__in=[
                PreOrder.STATUS_SUBMITTED,
                PreOrder.STATUS_UNDER_REVIEW,
                PreOrder.STATUS_QUOTED
            ]).count(),
            'in_production': queryset.filter(status=PreOrder.STATUS_IN_PRODUCTION).count(),
            'completed': queryset.filter(status=PreOrder.STATUS_COMPLETED).count(),
            'delivered': queryset.filter(status=PreOrder.STATUS_DELIVERED).count(),
            'total_value': queryset.aggregate(Sum('total_amount'))['total_amount__sum'] or 0,
            'total_paid': queryset.aggregate(Sum('amount_paid'))['amount_paid__sum'] or 0,
        }
    
    # Email notification methods
    @staticmethod
    def send_submission_notification(preorder: PreOrder):
        """Send notification when a pre-order is submitted."""
        try:
            # Email to customer
            subject = f"Pre-Order #{preorder.preorder_number} Submitted Successfully"
            html_message = render_to_string('emails/preorder/submitted_customer.html', {
                'preorder': preorder,
                'site_name': getattr(settings, 'SITE_NAME', 'Bunoraa')
            })
            
            send_mail(
                subject=subject,
                message='',
                html_message=html_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[preorder.email],
                fail_silently=True
            )
            
            # Email to admin
            admin_subject = f"New Pre-Order Submitted: {preorder.preorder_number}"
            admin_html = render_to_string('emails/preorder/submitted_admin.html', {
                'preorder': preorder
            })
            
            admin_emails = getattr(settings, 'PREORDER_ADMIN_EMAILS', [settings.DEFAULT_FROM_EMAIL])
            send_mail(
                subject=admin_subject,
                message='',
                html_message=admin_html,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=admin_emails,
                fail_silently=True
            )
            
        except Exception as e:
            logger.error(f"Failed to send submission notification: {e}")
    
    @staticmethod
    def send_status_update_notification(preorder: PreOrder, old_status: str):
        """Send notification when status changes."""
        try:
            subject = f"Pre-Order #{preorder.preorder_number} Status Update"
            html_message = render_to_string('emails/preorder/status_update.html', {
                'preorder': preorder,
                'old_status': dict(PreOrder.STATUS_CHOICES).get(old_status, old_status),
                'new_status': preorder.get_status_display(),
                'site_name': getattr(settings, 'SITE_NAME', 'Bunoraa')
            })
            
            send_mail(
                subject=subject,
                message='',
                html_message=html_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[preorder.email],
                fail_silently=True
            )
            
            # Update status history notification flag
            PreOrderStatusHistory.objects.filter(
                preorder=preorder,
                to_status=preorder.status,
                notification_sent=False
            ).update(
                notification_sent=True,
                notification_sent_at=timezone.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to send status update notification: {e}")
    
    @staticmethod
    def send_payment_confirmation(preorder: PreOrder, payment: PreOrderPayment):
        """Send payment confirmation email."""
        try:
            subject = f"Payment Confirmed - Pre-Order #{preorder.preorder_number}"
            html_message = render_to_string('emails/preorder/payment_confirmation.html', {
                'preorder': preorder,
                'payment': payment,
                'site_name': getattr(settings, 'SITE_NAME', 'Bunoraa')
            })
            
            send_mail(
                subject=subject,
                message='',
                html_message=html_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[preorder.email],
                fail_silently=True
            )
            
            payment.receipt_sent = True
            payment.receipt_sent_at = timezone.now()
            payment.save()
            
        except Exception as e:
            logger.error(f"Failed to send payment confirmation: {e}")
    
    @staticmethod
    def send_admin_message_notification(preorder: PreOrder, message: PreOrderMessage):
        """Notify admin of new customer message."""
        try:
            subject = f"New Message on Pre-Order #{preorder.preorder_number}"
            html_message = render_to_string('emails/preorder/new_message_admin.html', {
                'preorder': preorder,
                'message': message
            })
            
            admin_emails = getattr(settings, 'PREORDER_ADMIN_EMAILS', [settings.DEFAULT_FROM_EMAIL])
            send_mail(
                subject=subject,
                message='',
                html_message=html_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=admin_emails,
                fail_silently=True
            )
            
        except Exception as e:
            logger.error(f"Failed to send admin message notification: {e}")
    
    @staticmethod
    def send_customer_message_notification(preorder: PreOrder, message: PreOrderMessage):
        """Notify customer of new admin message."""
        try:
            subject = f"New Message on Your Pre-Order #{preorder.preorder_number}"
            html_message = render_to_string('emails/preorder/new_message_customer.html', {
                'preorder': preorder,
                'message': message,
                'site_name': getattr(settings, 'SITE_NAME', 'Bunoraa')
            })
            
            send_mail(
                subject=subject,
                message='',
                html_message=html_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[preorder.email],
                fail_silently=True
            )
            
        except Exception as e:
            logger.error(f"Failed to send customer message notification: {e}")


class PreOrderCategoryService:
    """Service for pre-order category operations."""
    
    @staticmethod
    def get_active_categories():
        """Get all active categories with prefetched options."""
        return PreOrderCategory.objects.filter(
            is_active=True
        ).prefetch_related(
            'options__choices'
        ).order_by('order', 'name')
    
    @staticmethod
    def get_category_with_options(category_id) -> PreOrderCategory:
        """Get a category with all its options."""
        return PreOrderCategory.objects.prefetch_related(
            'options__choices',
            'templates'
        ).get(id=category_id, is_active=True)


class PreOrderTemplateService:
    """Service for pre-order template operations."""
    
    @staticmethod
    def get_featured_templates():
        """Get featured templates."""
        return PreOrderTemplate.objects.filter(
            is_active=True,
            is_featured=True
        ).select_related('category').order_by('order')[:8]
    
    @staticmethod
    def get_templates_by_category(category_id):
        """Get templates for a specific category."""
        return PreOrderTemplate.objects.filter(
            category_id=category_id,
            is_active=True
        ).order_by('order', 'name')
    
    @staticmethod
    def use_template(template: PreOrderTemplate) -> Dict:
        """Increment template use count and return default values."""
        template.use_count += 1
        template.save()
        
        return {
            'category_id': str(template.category_id),
            'default_quantity': template.default_quantity,
            'base_price': str(template.base_price),
            'estimated_days': template.estimated_days,
            'default_options': template.default_options
        }


class QuoteService:
    """Service for quote operations."""
    
    @staticmethod
    def get_pending_quotes():
        """Get all pending quotes that need attention."""
        return PreOrderQuote.objects.filter(
            status__in=[PreOrderQuote.STATUS_PENDING, PreOrderQuote.STATUS_SENT],
            valid_until__gte=timezone.now()
        ).select_related('preorder', 'created_by').order_by('valid_until')
    
    @staticmethod
    def check_expired_quotes():
        """Mark expired quotes."""
        expired = PreOrderQuote.objects.filter(
            status__in=[PreOrderQuote.STATUS_PENDING, PreOrderQuote.STATUS_SENT],
            valid_until__lt=timezone.now()
        )
        count = expired.update(status=PreOrderQuote.STATUS_EXPIRED)
        logger.info(f"Marked {count} quotes as expired")
        return count
