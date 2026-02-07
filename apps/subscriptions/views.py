"""
Subscriptions views - Views for subscription management
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, DetailView, CreateView, TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.http import JsonResponse
from django.urls import reverse_lazy
from django.utils import timezone
from django.views import View

from .models import Plan, Subscription


class SubscriptionLandingView(TemplateView):
    """Landing page showing all subscription plans."""
    template_name = 'subscriptions/landing.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['plans'] = Plan.objects.filter(active=True).order_by('price_amount')
        context['monthly_plans'] = Plan.objects.filter(active=True, interval=Plan.INTERVAL_MONTH).order_by('price_amount')
        context['yearly_plans'] = Plan.objects.filter(active=True, interval=Plan.INTERVAL_YEAR).order_by('price_amount')
        
        # Check if user has active subscription
        if self.request.user.is_authenticated:
            context['active_subscription'] = Subscription.objects.filter(
                user=self.request.user,
                status=Subscription.STATUS_ACTIVE
            ).first()
        
        return context


class PlanDetailView(DetailView):
    """Detail view for a subscription plan."""
    model = Plan
    template_name = 'subscriptions/plan_detail.html'
    context_object_name = 'plan'
    
    def get_queryset(self):
        return Plan.objects.filter(active=True)


class MySubscriptionsView(LoginRequiredMixin, ListView):
    """List user's subscriptions."""
    model = Subscription
    template_name = 'subscriptions/my_subscriptions.html'
    context_object_name = 'subscriptions'
    
    def get_queryset(self):
        return Subscription.objects.filter(
            user=self.request.user,
            is_deleted=False
        ).select_related('plan').order_by('-created_at')


class SubscriptionDetailView(LoginRequiredMixin, DetailView):
    """Detail view for a user's subscription."""
    model = Subscription
    template_name = 'subscriptions/subscription_detail.html'
    context_object_name = 'subscription'
    
    def get_queryset(self):
        return Subscription.objects.filter(user=self.request.user, is_deleted=False)


class SubscribeView(LoginRequiredMixin, View):
    """Subscribe to a plan."""
    
    def post(self, request, plan_id):
        plan = get_object_or_404(Plan, id=plan_id, active=True)
        
        # Check if user already has active subscription
        existing = Subscription.objects.filter(
            user=request.user,
            status=Subscription.STATUS_ACTIVE
        ).first()
        
        if existing:
            messages.warning(request, 'You already have an active subscription. Please cancel it first or upgrade.')
            return redirect('subscriptions:my-subscriptions')
        
        # Create subscription (in production, integrate with Stripe here)
        subscription = Subscription.objects.create(
            user=request.user,
            plan=plan,
            status=Subscription.STATUS_ACTIVE if plan.trial_period_days == 0 else Subscription.STATUS_TRIALING,
            current_period_start=timezone.now(),
            current_period_end=timezone.now() + timezone.timedelta(days=30 if plan.interval == Plan.INTERVAL_MONTH else 365),
            trial_ends=timezone.now() + timezone.timedelta(days=plan.trial_period_days) if plan.trial_period_days > 0 else None,
        )
        
        messages.success(request, f'Successfully subscribed to {plan.name}!')
        return redirect('subscriptions:subscription-detail', pk=subscription.id)


class CancelSubscriptionView(LoginRequiredMixin, View):
    """Cancel a subscription."""
    
    def post(self, request, pk):
        subscription = get_object_or_404(
            Subscription,
            id=pk,
            user=request.user,
            is_deleted=False
        )
        
        if subscription.status == Subscription.STATUS_CANCELED:
            messages.warning(request, 'This subscription is already canceled.')
            return redirect('subscriptions:subscription-detail', pk=pk)
        
        subscription.mark_canceled()
        messages.success(request, 'Your subscription has been canceled. It will remain active until the end of your billing period.')
        return redirect('subscriptions:subscription-detail', pk=pk)


class ReactivateSubscriptionView(LoginRequiredMixin, View):
    """Reactivate a canceled subscription."""
    
    def post(self, request, pk):
        subscription = get_object_or_404(
            Subscription,
            id=pk,
            user=request.user,
            is_deleted=False
        )
        
        if subscription.status != Subscription.STATUS_CANCELED:
            messages.warning(request, 'This subscription is not canceled.')
            return redirect('subscriptions:subscription-detail', pk=pk)
        
        # Check if still within current period
        if subscription.current_period_end and subscription.current_period_end > timezone.now():
            subscription.status = Subscription.STATUS_ACTIVE
            subscription.canceled_at = None
            subscription.save(update_fields=['status', 'canceled_at'])
            messages.success(request, 'Your subscription has been reactivated!')
        else:
            messages.error(request, 'Cannot reactivate - please create a new subscription.')
        
        return redirect('subscriptions:subscription-detail', pk=pk)


class ChangePlanView(LoginRequiredMixin, View):
    """Change subscription plan (upgrade/downgrade)."""
    
    def get(self, request, pk):
        subscription = get_object_or_404(
            Subscription,
            id=pk,
            user=request.user,
            status=Subscription.STATUS_ACTIVE
        )
        
        available_plans = Plan.objects.filter(active=True).exclude(id=subscription.plan_id)
        
        return render(request, 'subscriptions/change_plan.html', {
            'subscription': subscription,
            'available_plans': available_plans,
        })
    
    def post(self, request, pk):
        subscription = get_object_or_404(
            Subscription,
            id=pk,
            user=request.user,
            status=Subscription.STATUS_ACTIVE
        )
        
        new_plan_id = request.POST.get('plan_id')
        new_plan = get_object_or_404(Plan, id=new_plan_id, active=True)
        
        # Update subscription (in production, handle proration with Stripe)
        subscription.plan = new_plan
        subscription.save(update_fields=['plan'])
        
        messages.success(request, f'Successfully changed to {new_plan.name} plan!')
        return redirect('subscriptions:subscription-detail', pk=pk)


# API Views
class SubscriptionStatusAPIView(LoginRequiredMixin, View):
    """API endpoint to check subscription status."""
    
    def get(self, request):
        subscription = Subscription.objects.filter(
            user=request.user,
            status=Subscription.STATUS_ACTIVE
        ).select_related('plan').first()
        
        if subscription:
            return JsonResponse({
                'has_subscription': True,
                'plan_name': subscription.plan.name,
                'status': subscription.status,
                'current_period_end': subscription.current_period_end.isoformat() if subscription.current_period_end else None,
            })
        
        return JsonResponse({'has_subscription': False})
