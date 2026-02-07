"""
Pre-orders views - Views for custom pre-order management
"""
import json
import logging
from decimal import Decimal
from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.views.generic import ListView, DetailView, CreateView, UpdateView, TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.http import JsonResponse, HttpResponseBadRequest
from django.urls import reverse, reverse_lazy
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django.db.models import Q
from django.core.paginator import Paginator

from .models import (
    PreOrderCategory, PreOrderOption, PreOrderOptionChoice,
    PreOrder, PreOrderItem, PreOrderDesign, PreOrderReference,
    PreOrderMessage, PreOrderRevision, PreOrderQuote, PreOrderTemplate
)
from .forms import (
    PreOrderCreateForm, PreOrderStepOneForm, PreOrderStepTwoForm,
    PreOrderStepThreeForm, PreOrderStepFourForm, PreOrderMessageForm,
    PreOrderRevisionForm, PreOrderDesignUploadForm, QuoteResponseForm,
    PreOrderSearchForm
)
from .services import PreOrderService, PreOrderCategoryService, PreOrderTemplateService

logger = logging.getLogger(__name__)


class PreOrderLandingView(TemplateView):
    """Landing page for pre-orders with categories and templates."""
    template_name = 'preorders/landing.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = PreOrderCategoryService.get_active_categories()
        context['featured_templates'] = PreOrderTemplateService.get_featured_templates()
        context['page_title'] = _('Custom Pre-Orders')
        return context


class PreOrderCategoryListView(ListView):
    """List all pre-order categories."""
    model = PreOrderCategory
    template_name = 'preorders/category_list.html'
    context_object_name = 'categories'
    
    def get_queryset(self):
        return PreOrderCategoryService.get_active_categories()


class PreOrderCategoryDetailView(DetailView):
    """Detail view for a pre-order category with its options."""
    model = PreOrderCategory
    template_name = 'preorders/category_detail.html'
    context_object_name = 'category'
    slug_field = 'slug'
    slug_url_kwarg = 'slug'
    
    def get_queryset(self):
        return PreOrderCategory.objects.filter(is_active=True).prefetch_related(
            'options__choices',
            'templates'
        )
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['templates'] = self.object.templates.filter(is_active=True).order_by('order')
        return context


class PreOrderCreateWizardView(View):
    """Multi-step wizard for creating a pre-order."""
    
    STEP_TEMPLATES = {
        1: 'preorders/wizard/step1_category.html',
        2: 'preorders/wizard/step2_options.html',
        3: 'preorders/wizard/step3_files.html',
        4: 'preorders/wizard/step4_details.html',
        5: 'preorders/wizard/step5_review.html',
    }
    
    def get_session_data(self, request):
        """Get pre-order data from session."""
        return request.session.get('preorder_wizard', {})
    
    def save_session_data(self, request, data):
        """Save pre-order data to session."""
        session_data = self.get_session_data(request)
        session_data.update(data)
        request.session['preorder_wizard'] = session_data
        request.session.modified = True
    
    def clear_session_data(self, request):
        """Clear pre-order wizard data from session."""
        if 'preorder_wizard' in request.session:
            del request.session['preorder_wizard']
    
    def get(self, request, step=1):
        step = int(step)
        session_data = self.get_session_data(request)
        
        context = {
            'step': step,
            'total_steps': 5,
            'session_data': session_data,
            'categories': PreOrderCategoryService.get_active_categories(),
        }
        
        if step == 1:
            context['form'] = PreOrderStepOneForm(initial=session_data)
        
        elif step == 2:
            category_id = session_data.get('category')
            if not category_id:
                return redirect('preorders:wizard', step=1)
            
            try:
                category = PreOrderCategory.objects.get(id=category_id)
                context['category'] = category
                context['form'] = PreOrderStepTwoForm(category=category, initial=session_data.get('options', {}))
            except PreOrderCategory.DoesNotExist:
                return redirect('preorders:wizard', step=1)
        
        elif step == 3:
            context['form'] = PreOrderStepThreeForm(initial=session_data)
            # Pass existing uploaded files to template as JSON
            existing_files = request.session.get('preorder_files', {})
            context['existing_files'] = existing_files
            context['existing_files_json'] = {
                'design_files': json.dumps(existing_files.get('design_files', [])),
                'reference_images': json.dumps(existing_files.get('reference_images', [])),
            }
        
        elif step == 4:
            initial = session_data.copy()
            if request.user.is_authenticated:
                initial.setdefault('full_name', request.user.get_full_name())
                initial.setdefault('email', request.user.email)
                initial.setdefault('phone', request.user.phone)
            context['form'] = PreOrderStepFourForm(initial=initial)
            
            # Get category for rush order info
            category_id = session_data.get('category')
            if category_id:
                try:
                    context['category'] = PreOrderCategory.objects.get(id=category_id)
                except PreOrderCategory.DoesNotExist:
                    pass
        
        elif step == 5:
            # Review step - show all collected data
            category_id = session_data.get('category')
            if category_id:
                try:
                    category = PreOrderCategory.objects.get(id=category_id)
                    context['category'] = category
                    
                    # Calculate price breakdown
                    quantity = session_data.get('quantity', 1)
                    base_subtotal = category.base_price * quantity
                    context['base_subtotal'] = base_subtotal
                    
                    # Calculate options total (if any)
                    options_total = Decimal('0')
                    options_data = session_data.get('options', {})
                    for option_id, value in options_data.items():
                        try:
                            option = PreOrderOption.objects.get(id=option_id)
                            options_total += option.price_modifier * quantity
                        except PreOrderOption.DoesNotExist:
                            pass
                    context['options_total'] = options_total
                    
                    # Rush order fee
                    rush_fee = Decimal('0')
                    if session_data.get('is_rush_order') and category.allow_rush_order:
                        rush_fee = (base_subtotal + options_total) * category.rush_order_fee_percentage / 100
                    context['rush_fee'] = rush_fee
                    
                    # Estimated total
                    estimated_total = base_subtotal + options_total + rush_fee
                    context['estimated_total'] = estimated_total
                    context['estimated_price'] = estimated_total
                    
                    # Deposit amount
                    deposit_amount = estimated_total * category.deposit_percentage / 100
                    context['deposit_amount'] = deposit_amount
                    
                    # Get file info from session
                    files_data = request.session.get('preorder_files', {})
                    context['design_files'] = files_data.get('design_files', [])
                    context['reference_files'] = files_data.get('reference_images', [])
                    
                except PreOrderCategory.DoesNotExist:
                    return redirect('preorders:wizard', step=1)
        
        template = self.STEP_TEMPLATES.get(step, self.STEP_TEMPLATES[1])
        return render(request, template, context)
    
    def post(self, request, step=1):
        step = int(step)
        session_data = self.get_session_data(request)
        
        if step == 1:
            form = PreOrderStepOneForm(request.POST)
            if form.is_valid():
                self.save_session_data(request, {
                    'category': str(form.cleaned_data['category'].id),
                    'title': form.cleaned_data['title'],
                    'description': form.cleaned_data['description'],
                    'quantity': form.cleaned_data['quantity'],
                })
                return redirect('preorders:wizard', step=2)
            
            context = {
                'form': form,
                'step': step,
                'total_steps': 5,
                'categories': PreOrderCategoryService.get_active_categories(),
            }
            return render(request, self.STEP_TEMPLATES[1], context)
        
        elif step == 2:
            category_id = session_data.get('category')
            try:
                category = PreOrderCategory.objects.get(id=category_id)
            except PreOrderCategory.DoesNotExist:
                return redirect('preorders:wizard', step=1)
            
            form = PreOrderStepTwoForm(request.POST, request.FILES, category=category)
            if form.is_valid():
                # Store options data
                options_data = {}
                for field_name, value in form.cleaned_data.items():
                    if field_name.startswith('option_'):
                        option_id = field_name.replace('option_', '')
                        if hasattr(value, 'id'):  # File field
                            # Handle file uploads separately
                            pass
                        elif isinstance(value, list):
                            options_data[option_id] = value
                        else:
                            options_data[option_id] = str(value) if value else ''
                
                self.save_session_data(request, {'options': options_data})
                return redirect('preorders:wizard', step=3)
            
            context = {
                'form': form,
                'step': step,
                'total_steps': 5,
                'category': category,
            }
            return render(request, self.STEP_TEMPLATES[2], context)
        
        elif step == 3:
            form = PreOrderStepThreeForm(request.POST, request.FILES)
            if form.is_valid():
                self.save_session_data(request, {
                    'special_instructions': form.cleaned_data.get('special_instructions', ''),
                })
                
                # Get existing files from session
                existing_files = request.session.get('preorder_files', {})
                existing_design_files = existing_files.get('design_files', [])
                existing_reference_files = existing_files.get('reference_images', [])
                existing_session_folder = existing_files.get('session_folder')
                
                # Check if new files were uploaded
                new_design_files = request.FILES.getlist('design_files')
                new_reference_files = request.FILES.getlist('reference_images')
                
                # Only process files if new ones are uploaded
                if new_design_files or new_reference_files:
                    import os
                    import uuid
                    from django.conf import settings
                    
                    from django.core.files.storage import default_storage
                    from django.core.files.base import ContentFile

                    # Use existing session folder or create new one (stored in remote storage)
                    session_folder = existing_session_folder or str(uuid.uuid4())
                    storage_base = f'temp_preorders/{session_folder}'

                    # Handle removed files from form
                    removed_design_ids = request.POST.getlist('remove_design_files', [])
                    removed_ref_ids = request.POST.getlist('remove_reference_files', [])

                    # Filter out removed files from existing files
                    design_files_info = [f for i, f in enumerate(existing_design_files) 
                                        if str(i) not in removed_design_ids]
                    reference_files_info = [f for i, f in enumerate(existing_reference_files) 
                                           if str(i) not in removed_ref_ids]

                    # Save new design files directly to configured storage
                    for f in new_design_files:
                        filename = f'design_{uuid.uuid4().hex}_{f.name}'
                        storage_path = f"{storage_base}/{filename}"
                        # Use default_storage.save which will upload to S3/R2 in production
                        saved_path = default_storage.save(storage_path, ContentFile(f.read()))
                        design_files_info.append({
                            'name': f.name,
                            'path': saved_path,
                            'size': f.size,
                            'content_type': f.content_type,
                        })

                    # Save new reference images directly to configured storage
                    for f in new_reference_files:
                        filename = f'ref_{uuid.uuid4().hex}_{f.name}'
                        storage_path = f"{storage_base}/{filename}"
                        saved_path = default_storage.save(storage_path, ContentFile(f.read()))
                        reference_files_info.append({
                            'name': f.name,
                            'path': saved_path,
                            'size': f.size,
                            'content_type': f.content_type,
                        })
                    
                    # Store file info in session
                    request.session['preorder_files'] = {
                        'session_folder': session_folder,
                        'design_files': design_files_info,
                        'reference_images': reference_files_info,
                    }
                else:
                    # Handle file removal even without new uploads
                    removed_design_ids = request.POST.getlist('remove_design_files', [])
                    removed_ref_ids = request.POST.getlist('remove_reference_files', [])
                    
                    if removed_design_ids or removed_ref_ids:
                        design_files_info = [f for i, f in enumerate(existing_design_files) 
                                            if str(i) not in removed_design_ids]
                        reference_files_info = [f for i, f in enumerate(existing_reference_files) 
                                               if str(i) not in removed_ref_ids]
                        
                        request.session['preorder_files'] = {
                            'session_folder': existing_session_folder,
                            'design_files': design_files_info,
                            'reference_images': reference_files_info,
                        }
                
                return redirect('preorders:wizard', step=4)
            
            context = {
                'form': form,
                'step': step,
                'total_steps': 5,
                'existing_files': request.session.get('preorder_files', {}),
            }
            return render(request, self.STEP_TEMPLATES[3], context)
        
        elif step == 4:
            form = PreOrderStepFourForm(request.POST)
            if form.is_valid():
                # Convert date objects to ISO format strings for JSON serialization
                cleaned_data = {}
                for key, value in form.cleaned_data.items():
                    if hasattr(value, 'isoformat'):  # date or datetime object
                        cleaned_data[key] = value.isoformat() if value else None
                    else:
                        cleaned_data[key] = value
                self.save_session_data(request, cleaned_data)
                return redirect('preorders:wizard', step=5)
            
            context = {
                'form': form,
                'step': step,
                'total_steps': 5,
                'session_data': session_data,
            }
            return render(request, self.STEP_TEMPLATES[4], context)
        
        elif step == 5:
            # Final submission
            return self.create_preorder(request, session_data)
        
        return redirect('preorders:wizard', step=1)
    
    def calculate_estimated_price(self, session_data, category):
        """Calculate estimated price based on session data."""
        quantity = session_data.get('quantity', 1)
        base_price = category.base_price * quantity
        
        # Add option modifiers
        options_data = session_data.get('options', {})
        for option_id, value in options_data.items():
            try:
                option = PreOrderOption.objects.get(id=option_id)
                base_price += option.price_modifier * quantity
                
                if option.option_type == PreOrderOption.OPTION_SELECT and value:
                    choice = PreOrderOptionChoice.objects.get(id=value)
                    base_price += choice.price_modifier * quantity
                elif option.option_type == PreOrderOption.OPTION_MULTISELECT and value:
                    choices = PreOrderOptionChoice.objects.filter(id__in=value)
                    for choice in choices:
                        base_price += choice.price_modifier * quantity
            except (PreOrderOption.DoesNotExist, PreOrderOptionChoice.DoesNotExist):
                continue
        
        # Add rush order fee if applicable
        if session_data.get('is_rush_order') and category.allow_rush_order:
            base_price += (base_price * category.rush_order_fee_percentage) / 100
        
        return base_price
    
    def create_preorder(self, request, session_data):
        """Create the pre-order from session data."""
        import os
        import shutil
        from datetime import date
        from django.core.files import File
        from django.conf import settings
        
        try:
            category = PreOrderCategory.objects.get(id=session_data.get('category'))
            
            # Parse requested_delivery_date from ISO string if present
            requested_delivery_date = session_data.get('requested_delivery_date')
            if requested_delivery_date and isinstance(requested_delivery_date, str):
                try:
                    requested_delivery_date = date.fromisoformat(requested_delivery_date)
                except ValueError:
                    requested_delivery_date = None
            
            # Prepare data dict
            data = {
                'title': session_data.get('title', ''),
                'description': session_data.get('description', ''),
                'quantity': session_data.get('quantity', 1),
                'special_instructions': session_data.get('special_instructions', ''),
                'is_gift': session_data.get('is_gift', False),
                'gift_wrap': session_data.get('gift_wrap', False),
                'gift_message': session_data.get('gift_message', ''),
                'is_rush_order': session_data.get('is_rush_order', False),
                'requested_delivery_date': requested_delivery_date,
                'full_name': session_data.get('full_name', ''),
                'email': session_data.get('email', ''),
                'phone': session_data.get('phone', ''),
                'shipping_first_name': session_data.get('shipping_first_name', ''),
                'shipping_last_name': session_data.get('shipping_last_name', ''),
                'shipping_address_line_1': session_data.get('shipping_address_line_1', ''),
                'shipping_address_line_2': session_data.get('shipping_address_line_2', ''),
                'shipping_city': session_data.get('shipping_city', ''),
                'shipping_state': session_data.get('shipping_state', ''),
                'shipping_postal_code': session_data.get('shipping_postal_code', ''),
                'shipping_country': session_data.get('shipping_country', 'Bangladesh'),
            }
            
            # Create pre-order
            user = request.user if request.user.is_authenticated else None
            preorder = PreOrderService.create_preorder(
                user=user,
                category=category,
                data=data,
                options_data=session_data.get('options', {}),
            )
            
            # Handle uploaded files from session
            files_data = request.session.get('preorder_files', {})
            session_folder = files_data.get('session_folder')
            
            from django.core.files.storage import default_storage

            # Process design files (read from storage)
            for file_info in files_data.get('design_files', []):
                storage_path = file_info.get('path')
                if storage_path and default_storage.exists(storage_path):
                    try:
                        with default_storage.open(storage_path, 'rb') as f:
                            PreOrderDesign.objects.create(
                                preorder=preorder,
                                title=file_info.get('name', 'Design'),
                                file=File(f, name=file_info.get('name')),
                            )
                    except Exception as e:
                        logger.warning(f"Failed to save design file: {e}")

            # Process reference images (read from storage)
            for file_info in files_data.get('reference_images', []):
                storage_path = file_info.get('path')
                if storage_path and default_storage.exists(storage_path):
                    try:
                        with default_storage.open(storage_path, 'rb') as f:
                            PreOrderReference.objects.create(
                                preorder=preorder,
                                title=file_info.get('name', 'Reference'),
                                image=File(f, name=file_info.get('name')),
                            )
                    except Exception as e:
                        logger.warning(f"Failed to save reference image: {e}")

            # Clean up temp files from storage
            try:
                for file_info in files_data.get('design_files', []):
                    storage_path = file_info.get('path')
                    if storage_path and default_storage.exists(storage_path):
                        try:
                            default_storage.delete(storage_path)
                        except Exception:
                            logger.warning(f"Failed to delete temporary stored file: {storage_path}")

                for file_info in files_data.get('reference_images', []):
                    storage_path = file_info.get('path')
                    if storage_path and default_storage.exists(storage_path):
                        try:
                            default_storage.delete(storage_path)
                        except Exception:
                            logger.warning(f"Failed to delete temporary stored file: {storage_path}")
            except Exception:
                # Ignore storage cleanup errors
                pass
            
            # Submit the pre-order
            preorder = PreOrderService.submit_preorder(preorder, user)
            
            # Clear session
            self.clear_session_data(request)
            if 'preorder_files' in request.session:
                del request.session['preorder_files']
            
            messages.success(request, _('Your pre-order has been submitted successfully!'))
            return redirect('preorders:success', preorder_number=preorder.preorder_number)
            
        except Exception as e:
            logger.error(f"Error creating pre-order: {e}")
            messages.error(request, _('An error occurred while creating your pre-order. Please try again.'))
            return redirect('preorders:wizard', step=1)


class PreOrderSuccessView(TemplateView):
    """Success page after pre-order submission."""
    template_name = 'preorders/success.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        preorder_number = self.kwargs.get('preorder_number')
        context['preorder'] = get_object_or_404(PreOrder, preorder_number=preorder_number)
        return context


class MyPreOrdersView(LoginRequiredMixin, ListView):
    """List user's pre-orders."""
    model = PreOrder
    template_name = 'preorders/my_preorders.html'
    context_object_name = 'preorders'
    paginate_by = 10
    
    def get_queryset(self):
        queryset = PreOrder.objects.filter(
            user=self.request.user,
            is_deleted=False
        ).select_related('category').order_by('-created_at')
        
        # Apply filters
        status = self.request.GET.get('status')
        if status:
            queryset = queryset.filter(status=status)
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['status_choices'] = PreOrder.STATUS_CHOICES
        context['current_status'] = self.request.GET.get('status', '')
        context['statistics'] = PreOrderService.get_preorder_statistics(self.request.user)
        return context


class PreOrderDetailView(DetailView):
    """Detail view for a pre-order."""
    model = PreOrder
    template_name = 'preorders/detail.html'
    context_object_name = 'preorder'
    slug_field = 'preorder_number'
    slug_url_kwarg = 'preorder_number'
    
    def get_queryset(self):
        queryset = PreOrder.objects.filter(is_deleted=False).select_related(
            'category', 'user', 'assigned_to'
        ).prefetch_related(
            'items', 'designs', 'references', 'option_values__option',
            'payments', 'messages', 'revisions', 'quotes', 'status_history'
        )
        
        # Non-authenticated users can view by email verification
        if not self.request.user.is_authenticated:
            email = self.request.session.get('preorder_email')
            if email:
                queryset = queryset.filter(email=email)
            else:
                queryset = queryset.none()
        else:
            # Staff can see all, users see their own
            if not self.request.user.is_staff:
                queryset = queryset.filter(user=self.request.user)
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['message_form'] = PreOrderMessageForm()
        context['revision_form'] = PreOrderRevisionForm()
        context['design_upload_form'] = PreOrderDesignUploadForm()
        
        # Get pending quote if exists
        context['pending_quote'] = self.object.quotes.filter(
            status__in=[PreOrderQuote.STATUS_PENDING, PreOrderQuote.STATUS_SENT],
            valid_until__gte=timezone.now()
        ).first()
        
        context['quote_response_form'] = QuoteResponseForm()
        
        return context


class PreOrderTrackingView(View):
    """Track pre-order by number and email."""
    template_name = 'preorders/tracking.html'
    
    def get(self, request):
        return render(request, self.template_name, {})
    
    def post(self, request):
        preorder_number = request.POST.get('preorder_number', '').strip()
        email = request.POST.get('email', '').strip()
        
        if not preorder_number or not email:
            messages.error(request, _('Please provide both order number and email.'))
            return render(request, self.template_name, {})
        
        try:
            preorder = PreOrder.objects.get(
                preorder_number=preorder_number,
                email__iexact=email,
                is_deleted=False
            )
            # Store email in session for access
            request.session['preorder_email'] = email
            return redirect('preorders:detail', preorder_number=preorder.preorder_number)
        except PreOrder.DoesNotExist:
            messages.error(request, _('Pre-order not found. Please check your order number and email.'))
            return render(request, self.template_name, {
                'preorder_number': preorder_number,
                'email': email
            })


class PreOrderMessageView(LoginRequiredMixin, View):
    """Handle sending messages on a pre-order."""
    
    def post(self, request, preorder_number):
        preorder = get_object_or_404(
            PreOrder,
            preorder_number=preorder_number,
            user=request.user,
            is_deleted=False
        )
        
        form = PreOrderMessageForm(request.POST, request.FILES)
        if form.is_valid():
            PreOrderService.send_message(
                preorder=preorder,
                message=form.cleaned_data['message'],
                subject=form.cleaned_data.get('subject', ''),
                sender=request.user,
                is_from_customer=True,
                attachment=form.cleaned_data.get('attachment')
            )
            messages.success(request, _('Your message has been sent.'))
        else:
            messages.error(request, _('Please check your message and try again.'))
        
        return redirect('preorders:detail', preorder_number=preorder_number)


class PreOrderRevisionRequestView(LoginRequiredMixin, View):
    """Handle revision requests."""
    
    def post(self, request, preorder_number):
        preorder = get_object_or_404(
            PreOrder,
            preorder_number=preorder_number,
            user=request.user,
            is_deleted=False
        )
        
        # Check if revision can be requested
        if preorder.status not in [PreOrder.STATUS_AWAITING_APPROVAL]:
            messages.error(request, _('Revisions can only be requested when awaiting approval.'))
            return redirect('preorders:detail', preorder_number=preorder_number)
        
        form = PreOrderRevisionForm(request.POST)
        if form.is_valid():
            try:
                PreOrderService.request_revision(
                    preorder=preorder,
                    description=form.cleaned_data['description'],
                    user=request.user
                )
                messages.success(request, _('Your revision request has been submitted.'))
            except ValueError as e:
                messages.error(request, str(e))
        else:
            messages.error(request, _('Please provide a description for your revision request.'))
        
        return redirect('preorders:detail', preorder_number=preorder_number)


class PreOrderDesignUploadView(LoginRequiredMixin, View):
    """Handle design file uploads."""
    
    def post(self, request, preorder_number):
        preorder = get_object_or_404(
            PreOrder,
            preorder_number=preorder_number,
            user=request.user,
            is_deleted=False
        )
        
        form = PreOrderDesignUploadForm(request.POST, request.FILES)
        if form.is_valid():
            PreOrderDesign.objects.create(
                preorder=preorder,
                file=form.cleaned_data['file'],
                original_filename=form.cleaned_data['file'].name,
                design_type=PreOrderDesign.DESIGN_CUSTOMER,
                notes=form.cleaned_data.get('notes', ''),
                uploaded_by=request.user
            )
            messages.success(request, _('Your design file has been uploaded.'))
        else:
            messages.error(request, _('Please select a valid file to upload.'))
        
        return redirect('preorders:detail', preorder_number=preorder_number)


class QuoteResponseView(LoginRequiredMixin, View):
    """Handle quote acceptance/rejection."""
    
    def post(self, request, preorder_number, quote_id):
        preorder = get_object_or_404(
            PreOrder,
            preorder_number=preorder_number,
            user=request.user,
            is_deleted=False
        )
        
        quote = get_object_or_404(PreOrderQuote, id=quote_id, preorder=preorder)
        
        if quote.is_expired:
            messages.error(request, _('This quote has expired.'))
            return redirect('preorders:detail', preorder_number=preorder_number)
        
        form = QuoteResponseForm(request.POST)
        if form.is_valid():
            action = form.cleaned_data['action']
            notes = form.cleaned_data.get('response_notes', '')
            
            if action == 'accept':
                PreOrderService.accept_quote(preorder, quote, request.user)
                messages.success(request, _('Quote accepted! Please proceed with the deposit payment.'))
            else:
                PreOrderService.reject_quote(preorder, quote, request.user, notes)
                messages.info(request, _('Quote rejected. Our team will contact you for alternatives.'))
        
        return redirect('preorders:detail', preorder_number=preorder_number)


class PreOrderApproveView(LoginRequiredMixin, View):
    """Handle customer approval of completed work."""
    
    def post(self, request, preorder_number):
        preorder = get_object_or_404(
            PreOrder,
            preorder_number=preorder_number,
            user=request.user,
            status=PreOrder.STATUS_AWAITING_APPROVAL,
            is_deleted=False
        )
        
        action = request.POST.get('action')
        
        if action == 'approve':
            if preorder.is_fully_paid:
                PreOrderService.update_status(
                    preorder,
                    PreOrder.STATUS_READY_TO_SHIP,
                    request.user,
                    'Customer approved the work'
                )
                messages.success(request, _('Thank you for approving! Your order will be shipped soon.'))
            else:
                PreOrderService.update_status(
                    preorder,
                    PreOrder.STATUS_FINAL_PAYMENT_PENDING,
                    request.user,
                    'Customer approved, pending final payment'
                )
                messages.success(request, _('Thank you for approving! Please complete the final payment.'))
        
        return redirect('preorders:detail', preorder_number=preorder_number)


# API Views for AJAX operations
class PreOrderCategoryOptionsAPIView(View):
    """Get options for a category (AJAX)."""
    
    def get(self, request, category_id):
        try:
            category = PreOrderCategory.objects.prefetch_related(
                'options__choices'
            ).get(id=category_id, is_active=True)
            
            options = []
            for option in category.options.filter(is_active=True).order_by('order'):
                opt_data = {
                    'id': str(option.id),
                    'name': option.name,
                    'type': option.option_type,
                    'required': option.is_required,
                    'price_modifier': str(option.price_modifier),
                    'placeholder': option.placeholder,
                    'help_text': option.help_text,
                    'choices': []
                }
                
                if option.option_type in [PreOrderOption.OPTION_SELECT, PreOrderOption.OPTION_MULTISELECT]:
                    for choice in option.choices.filter(is_active=True).order_by('order'):
                        opt_data['choices'].append({
                            'id': str(choice.id),
                            'value': choice.value,
                            'display_name': choice.display_name,
                            'price_modifier': str(choice.price_modifier),
                            'color_code': choice.color_code,
                            'image_url': choice.image.url if choice.image else None
                        })
                
                options.append(opt_data)
            
            return JsonResponse({
                'success': True,
                'category': {
                    'id': str(category.id),
                    'name': category.name,
                    'base_price': str(category.base_price),
                    'deposit_percentage': category.deposit_percentage,
                    'min_production_days': category.min_production_days,
                    'max_production_days': category.max_production_days,
                    'requires_design': category.requires_design,
                    'allow_rush_order': category.allow_rush_order,
                    'rush_order_fee_percentage': category.rush_order_fee_percentage,
                    'min_quantity': category.min_quantity,
                    'max_quantity': category.max_quantity,
                },
                'options': options
            })
            
        except PreOrderCategory.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Category not found'}, status=404)


class PreOrderPriceCalculatorAPIView(View):
    """Calculate estimated price (AJAX)."""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            category_id = data.get('category_id')
            quantity = int(data.get('quantity', 1))
            options = data.get('options', {})
            is_rush = data.get('is_rush_order', False)
            
            category = PreOrderCategory.objects.get(id=category_id, is_active=True)
            
            base_price = category.base_price * quantity
            options_price = Decimal('0')
            
            for option_id, value in options.items():
                try:
                    option = PreOrderOption.objects.get(id=option_id)
                    options_price += option.price_modifier * quantity
                    
                    if option.option_type == PreOrderOption.OPTION_SELECT and value:
                        choice = PreOrderOptionChoice.objects.get(id=value)
                        options_price += choice.price_modifier * quantity
                    elif option.option_type == PreOrderOption.OPTION_MULTISELECT and value:
                        choices = PreOrderOptionChoice.objects.filter(id__in=value)
                        for choice in choices:
                            options_price += choice.price_modifier * quantity
                except:
                    continue
            
            subtotal = base_price + options_price
            rush_fee = Decimal('0')
            
            if is_rush and category.allow_rush_order:
                rush_fee = (subtotal * category.rush_order_fee_percentage) / 100
            
            total = subtotal + rush_fee
            deposit = (total * category.deposit_percentage) / 100
            
            return JsonResponse({
                'success': True,
                'pricing': {
                    'base_price': str(base_price),
                    'options_price': str(options_price),
                    'rush_fee': str(rush_fee),
                    'subtotal': str(subtotal),
                    'total': str(total),
                    'deposit_required': str(deposit),
                    'deposit_percentage': category.deposit_percentage,
                    'currency': 'BDT'
                }
            })
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=400)


class PreOrderStatusAPIView(LoginRequiredMixin, View):
    """Get pre-order status (AJAX)."""
    
    def get(self, request, preorder_number):
        try:
            preorder = PreOrder.objects.get(
                preorder_number=preorder_number,
                user=request.user,
                is_deleted=False
            )
            
            return JsonResponse({
                'success': True,
                'status': preorder.status,
                'status_display': preorder.get_status_display(),
                'summary': PreOrderService.get_preorder_summary(preorder)
            })
            
        except PreOrder.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Pre-order not found'}, status=404)


class PreOrderTemplateUseView(View):
    """Use a template to start pre-order (AJAX)."""
    
    def get(self, request, template_id):
        try:
            template = PreOrderTemplate.objects.select_related('category').get(
                id=template_id,
                is_active=True
            )
            
            template_data = PreOrderTemplateService.use_template(template)
            
            return JsonResponse({
                'success': True,
                'template': {
                    'id': str(template.id),
                    'name': template.name,
                    'description': template.description,
                    'image_url': template.image.url if template.image else None,
                    **template_data
                }
            })
            
        except PreOrderTemplate.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Template not found'}, status=404)


class MarkMessagesReadView(LoginRequiredMixin, View):
    """Mark messages as read (AJAX)."""
    
    def post(self, request, preorder_number):
        try:
            preorder = PreOrder.objects.get(
                preorder_number=preorder_number,
                user=request.user,
                is_deleted=False
            )
            
            # Mark admin messages as read
            updated = PreOrderMessage.objects.filter(
                preorder=preorder,
                is_from_customer=False,
                is_read=False
            ).update(is_read=True, read_at=timezone.now())
            
            return JsonResponse({'success': True, 'marked_read': updated})
            
        except PreOrder.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Pre-order not found'}, status=404)
