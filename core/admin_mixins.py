"""
Bunoraa Admin Mixins and Utilities
Provides reusable functionality for admin classes.
"""
import csv
import json
from datetime import datetime
from io import StringIO

from django.contrib import admin, messages
from django.contrib.admin import SimpleListFilter
from django.http import HttpResponse
from django.utils import timezone
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _


# =============================================================================
# EXPORT MIXINS
# =============================================================================

class ExportCSVMixin:
    """Mixin to add CSV export action to admin."""
    
    def get_export_fields(self):
        """Override to specify which fields to export."""
        if hasattr(self, 'export_fields'):
            return self.export_fields
        # Default to list_display fields
        return [f for f in self.list_display if f != 'action_checkbox']
    
    def get_export_filename(self):
        """Generate export filename."""
        model_name = self.model._meta.model_name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f'{model_name}_export_{timestamp}.csv'
    
    def export_as_csv(self, request, queryset):
        """Export selected items as CSV."""
        meta = self.model._meta
        field_names = self.get_export_fields()
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{self.get_export_filename()}"'
        
        writer = csv.writer(response)
        
        # Write header
        headers = []
        for field_name in field_names:
            if hasattr(self, field_name) and hasattr(getattr(self, field_name), 'short_description'):
                headers.append(getattr(self, field_name).short_description)
            elif hasattr(meta.model, field_name):
                try:
                    field = meta.get_field(field_name)
                    headers.append(field.verbose_name)
                except:
                    headers.append(field_name.replace('_', ' ').title())
            else:
                headers.append(field_name.replace('_', ' ').title())
        writer.writerow(headers)
        
        # Write data
        for obj in queryset:
            row = []
            for field_name in field_names:
                if hasattr(self, field_name) and callable(getattr(self, field_name)):
                    value = getattr(self, field_name)(obj)
                elif hasattr(obj, field_name):
                    value = getattr(obj, field_name)
                    if callable(value):
                        value = value()
                else:
                    value = ''
                
                # Handle special types
                if hasattr(value, 'isoformat'):
                    value = value.isoformat()
                elif hasattr(value, '__iter__') and not isinstance(value, str):
                    value = ', '.join(str(v) for v in value)
                
                # Strip HTML tags if present
                if isinstance(value, str) and '<' in value:
                    import re
                    value = re.sub('<[^<]+?>', '', value)
                
                row.append(value)
            writer.writerow(row)
        
        self.message_user(request, f'Exported {queryset.count()} records to CSV.', messages.SUCCESS)
        return response
    
    export_as_csv.short_description = _("Export selected as CSV")


class ExportJSONMixin:
    """Mixin to add JSON export action to admin."""
    
    def export_as_json(self, request, queryset):
        """Export selected items as JSON."""
        from django.core.serializers import serialize
        
        model_name = self.model._meta.model_name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{model_name}_export_{timestamp}.json'
        
        response = HttpResponse(content_type='application/json')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        data = serialize('json', queryset, indent=2)
        response.write(data)
        
        self.message_user(request, f'Exported {queryset.count()} records to JSON.', messages.SUCCESS)
        return response
    
    export_as_json.short_description = _("Export selected as JSON")


# =============================================================================
# AUDIT LOGGING MIXIN
# =============================================================================

class AuditLogMixin:
    """Mixin to add audit logging to admin actions."""
    
    def save_model(self, request, obj, form, change):
        """Log model save actions."""
        super().save_model(request, obj, form, change)
        
        action = 'updated' if change else 'created'
        self._log_admin_action(request, obj, action, form.changed_data if change else None)
    
    def delete_model(self, request, obj):
        """Log model delete actions."""
        self._log_admin_action(request, obj, 'deleted')
        super().delete_model(request, obj)
    
    def delete_queryset(self, request, queryset):
        """Log bulk delete actions."""
        for obj in queryset:
            self._log_admin_action(request, obj, 'deleted')
        super().delete_queryset(request, queryset)
    
    def _log_admin_action(self, request, obj, action, changed_fields=None):
        """Create an audit log entry."""
        from django.contrib.admin.models import LogEntry, ADDITION, CHANGE, DELETION
        from django.contrib.contenttypes.models import ContentType
        
        action_flag = {
            'created': ADDITION,
            'updated': CHANGE,
            'deleted': DELETION
        }.get(action, CHANGE)
        
        message = action.capitalize()
        if changed_fields:
            message += f': {", ".join(changed_fields)}'
        
        LogEntry.objects.log_action(
            user_id=request.user.pk,
            content_type_id=ContentType.objects.get_for_model(obj).pk,
            object_id=obj.pk,
            object_repr=str(obj)[:200],
            action_flag=action_flag,
            change_message=message
        )


# =============================================================================
# STATUS BADGE MIXIN
# =============================================================================

class StatusBadgeMixin:
    """Mixin to add colored status badges."""
    
    STATUS_COLORS = {
        # Order statuses
        'pending': ('warning', '#fef3c7', '#92400e'),
        'confirmed': ('info', '#dbeafe', '#1e40af'),
        'processing': ('primary', '#e0e7ff', '#3730a3'),
        'shipped': ('info', '#cffafe', '#0e7490'),
        'delivered': ('success', '#dcfce7', '#166534'),
        'cancelled': ('danger', '#fee2e2', '#991b1b'),
        'refunded': ('secondary', '#f3e8ff', '#7c3aed'),
        
        # Payment statuses
        'paid': ('success', '#dcfce7', '#166534'),
        'unpaid': ('warning', '#fef3c7', '#92400e'),
        'failed': ('danger', '#fee2e2', '#991b1b'),
        'partial': ('info', '#dbeafe', '#1e40af'),
        
        # General
        'active': ('success', '#dcfce7', '#166534'),
        'inactive': ('secondary', '#f1f5f9', '#64748b'),
        'draft': ('warning', '#fef3c7', '#92400e'),
        'published': ('success', '#dcfce7', '#166534'),
        'archived': ('secondary', '#f1f5f9', '#64748b'),
        
        # Chat/Conversation statuses
        'open': ('info', '#dbeafe', '#1e40af'),
        'waiting': ('warning', '#fef3c7', '#92400e'),
        'resolved': ('success', '#dcfce7', '#166534'),
        'closed': ('secondary', '#f1f5f9', '#64748b'),
        
        # Boolean
        True: ('success', '#dcfce7', '#166534'),
        False: ('danger', '#fee2e2', '#991b1b'),
    }
    
    def get_status_badge(self, status):
        """Generate HTML for a status badge."""
        if status is None:
            status = 'inactive'
        
        # Get colors
        colors = self.STATUS_COLORS.get(status.lower() if isinstance(status, str) else status, 
                                        ('secondary', '#f1f5f9', '#64748b'))
        _, bg_color, text_color = colors
        
        label = str(status).replace('_', ' ').title() if isinstance(status, str) else ('Yes' if status else 'No')
        
        return format_html(
            '<span style="display: inline-flex; align-items: center; padding: 0.25rem 0.75rem; '
            'border-radius: 9999px; font-size: 0.75rem; font-weight: 600; '
            'background-color: {}; color: {};">{}</span>',
            bg_color, text_color, label
        )


# =============================================================================
# SOFT DELETE MIXIN
# =============================================================================

class SoftDeleteMixin:
    """Mixin for handling soft-deleted models."""
    
    class IsDeletedFilter(SimpleListFilter):
        title = _('Deleted')
        parameter_name = 'is_deleted'
        
        def lookups(self, request, model_admin):
            return (
                ('no', _('Active only')),
                ('yes', _('Deleted only')),
                ('all', _('All')),
            )
        
        def queryset(self, request, queryset):
            if self.value() == 'yes':
                return queryset.filter(is_deleted=True)
            elif self.value() == 'no' or self.value() is None:
                return queryset.filter(is_deleted=False)
            return queryset
    
    def get_queryset(self, request):
        """Show only non-deleted items by default."""
        qs = super().get_queryset(request)
        if hasattr(qs.model, 'is_deleted'):
            # Check if filter is applied
            if 'is_deleted' not in request.GET:
                qs = qs.filter(is_deleted=False)
        return qs
    
    def soft_delete_selected(self, request, queryset):
        """Soft delete selected items."""
        updated = queryset.update(is_deleted=True, deleted_at=timezone.now())
        self.message_user(request, f'{updated} items soft deleted.', messages.SUCCESS)
    soft_delete_selected.short_description = _("Soft delete selected")
    
    def restore_selected(self, request, queryset):
        """Restore soft-deleted items."""
        updated = queryset.update(is_deleted=False, deleted_at=None)
        self.message_user(request, f'{updated} items restored.', messages.SUCCESS)
    restore_selected.short_description = _("Restore selected")


# =============================================================================
# ENHANCED LIST FILTERS
# =============================================================================

class DateRangeFilter(SimpleListFilter):
    """Filter by date range."""
    title = _('Date Range')
    parameter_name = 'date_range'
    date_field = 'created_at'
    
    def lookups(self, request, model_admin):
        return (
            ('today', _('Today')),
            ('yesterday', _('Yesterday')),
            ('week', _('This week')),
            ('month', _('This month')),
            ('quarter', _('This quarter')),
            ('year', _('This year')),
        )
    
    def queryset(self, request, queryset):
        from datetime import timedelta
        now = timezone.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        filters = {
            'today': {'gte': today},
            'yesterday': {'gte': today - timedelta(days=1), 'lt': today},
            'week': {'gte': today - timedelta(days=7)},
            'month': {'gte': today - timedelta(days=30)},
            'quarter': {'gte': today - timedelta(days=90)},
            'year': {'gte': today - timedelta(days=365)},
        }
        
        if self.value() in filters:
            filter_args = {}
            for op, value in filters[self.value()].items():
                filter_args[f'{self.date_field}__{op}'] = value
            return queryset.filter(**filter_args)
        
        return queryset


class PriceRangeFilter(SimpleListFilter):
    """Filter by price range."""
    title = _('Price Range')
    parameter_name = 'price_range'
    price_field = 'price'
    
    def lookups(self, request, model_admin):
        return (
            ('0-500', _('Under ৳500')),
            ('500-1000', _('৳500 - ৳1,000')),
            ('1000-5000', _('৳1,000 - ৳5,000')),
            ('5000-10000', _('৳5,000 - ৳10,000')),
            ('10000+', _('Over ৳10,000')),
        )
    
    def queryset(self, request, queryset):
        if self.value() == '0-500':
            return queryset.filter(**{f'{self.price_field}__lt': 500})
        elif self.value() == '500-1000':
            return queryset.filter(**{f'{self.price_field}__gte': 500, f'{self.price_field}__lt': 1000})
        elif self.value() == '1000-5000':
            return queryset.filter(**{f'{self.price_field}__gte': 1000, f'{self.price_field}__lt': 5000})
        elif self.value() == '5000-10000':
            return queryset.filter(**{f'{self.price_field}__gte': 5000, f'{self.price_field}__lt': 10000})
        elif self.value() == '10000+':
            return queryset.filter(**{f'{self.price_field}__gte': 10000})
        return queryset


class StockFilter(SimpleListFilter):
    """Filter by stock status."""
    title = _('Stock Status')
    parameter_name = 'stock_status'
    
    def lookups(self, request, model_admin):
        return (
            ('in_stock', _('In Stock')),
            ('low_stock', _('Low Stock')),
            ('out_of_stock', _('Out of Stock')),
        )
    
    def queryset(self, request, queryset):
        from django.db.models import F
        
        if self.value() == 'in_stock':
            return queryset.filter(stock_quantity__gt=F('low_stock_threshold'))
        elif self.value() == 'low_stock':
            return queryset.filter(
                stock_quantity__lte=F('low_stock_threshold'),
                stock_quantity__gt=0
            )
        elif self.value() == 'out_of_stock':
            return queryset.filter(stock_quantity=0)
        return queryset


# =============================================================================
# BULK ACTION MIXINS
# =============================================================================

class BulkActivateMixin:
    """Mixin to add bulk activate/deactivate actions."""
    
    def activate_selected(self, request, queryset):
        """Activate selected items."""
        field_name = 'is_active' if hasattr(queryset.model, 'is_active') else 'is_visible'
        updated = queryset.update(**{field_name: True})
        self.message_user(request, f'{updated} items activated.', messages.SUCCESS)
    activate_selected.short_description = _("Activate selected")
    
    def deactivate_selected(self, request, queryset):
        """Deactivate selected items."""
        field_name = 'is_active' if hasattr(queryset.model, 'is_active') else 'is_visible'
        updated = queryset.update(**{field_name: False})
        self.message_user(request, f'{updated} items deactivated.', messages.SUCCESS)
    deactivate_selected.short_description = _("Deactivate selected")


class BulkFeaturedMixin:
    """Mixin to add bulk featured/unfeatured actions."""
    
    def mark_featured(self, request, queryset):
        """Mark selected items as featured."""
        updated = queryset.update(is_featured=True)
        self.message_user(request, f'{updated} items marked as featured.', messages.SUCCESS)
    mark_featured.short_description = _("Mark as featured")
    
    def unmark_featured(self, request, queryset):
        """Remove featured status from selected items."""
        updated = queryset.update(is_featured=False)
        self.message_user(request, f'{updated} items unmarked as featured.', messages.SUCCESS)
    unmark_featured.short_description = _("Remove featured status")


# =============================================================================
# ENHANCED MODEL ADMIN BASE
# =============================================================================

class EnhancedModelAdmin(ExportCSVMixin, ExportJSONMixin, AuditLogMixin, StatusBadgeMixin, admin.ModelAdmin):
    """
    Enhanced base ModelAdmin with common functionality:
    - CSV/JSON export
    - Audit logging
    - Status badges
    - Date hierarchy
    """
    
    # Default settings
    show_full_result_count = True
    list_per_page = 25
    list_max_show_all = 500
    save_on_top = True
    
    # Default actions
    actions = ['export_as_csv', 'export_as_json']
    
    def get_actions(self, request):
        """Get actions and add export actions."""
        actions = super().get_actions(request)
        
        # Add export actions if not already present
        if hasattr(self, 'export_as_csv') and 'export_as_csv' not in actions:
            actions['export_as_csv'] = (
                self.export_as_csv,
                'export_as_csv',
                self.export_as_csv.short_description
            )
        if hasattr(self, 'export_as_json') and 'export_as_json' not in actions:
            actions['export_as_json'] = (
                self.export_as_json,
                'export_as_json',
                self.export_as_json.short_description
            )
        
        return actions


class EnhancedStackedInline(admin.StackedInline):
    """Enhanced stacked inline with common settings."""
    extra = 0
    show_change_link = True


class EnhancedTabularInline(admin.TabularInline):
    """Enhanced tabular inline with common settings."""
    extra = 0
    show_change_link = True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_change_url(obj):
    """Get the admin change URL for an object."""
    from django.urls import reverse
    from django.contrib.contenttypes.models import ContentType
    
    content_type = ContentType.objects.get_for_model(obj)
    return reverse(
        f'admin:{content_type.app_label}_{content_type.model}_change',
        args=[obj.pk]
    )


def format_currency(amount, currency='BDT'):
    """Format amount as currency."""
    if amount is None:
        return '-'
    
    symbols = {
        'BDT': '৳',
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'INR': '₹',
    }
    symbol = symbols.get(currency, currency)
    
    try:
        return f'{symbol}{float(amount):,.2f}'
    except (ValueError, TypeError):
        return f'{symbol}{amount}'


def format_number(number):
    """Format number with thousands separator."""
    if number is None:
        return '-'
    try:
        return f'{int(number):,}'
    except (ValueError, TypeError):
        return str(number)


def truncate_text(text, max_length=50):
    """Truncate text to max length."""
    if not text:
        return '-'
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + '...'
