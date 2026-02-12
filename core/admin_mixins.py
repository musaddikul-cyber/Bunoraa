"""
Bunoraa Admin Mixins and Utilities
Provides reusable functionality for admin classes.
"""
import csv
import json
import zipfile
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path

from django.contrib import admin, messages
from django.contrib.admin import SimpleListFilter
from django.http import HttpResponse
from django.core.management import call_command
from django.core.serializers.json import DjangoJSONEncoder
from django.conf import settings
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
# AUTO LIST FILTERS & SELECT RELATED
# =============================================================================


class AutoListFilterMixin:
    """Auto-attach common list filters and select_related for admin performance."""

    auto_boolean_filters = ("is_active", "is_deleted", "is_verified")
    auto_status_filter = "status"
    auto_date_fields = ("created_at", "updated_at")

    def _has_field(self, field_name: str) -> bool:
        try:
            self.model._meta.get_field(field_name)
            return True
        except Exception:
            return False

    def _is_boolean_field(self, field_name: str) -> bool:
        try:
            field = self.model._meta.get_field(field_name)
        except Exception:
            return False
        from django.db import models as dj_models

        return isinstance(field, (dj_models.BooleanField, getattr(dj_models, "NullBooleanField", dj_models.BooleanField)))

    def get_list_filter(self, request):
        filters = list(super().get_list_filter(request))  # type: ignore[misc]

        existing = set()
        for item in filters:
            if isinstance(item, str):
                existing.add(item)
            elif hasattr(item, "parameter_name"):
                existing.add(getattr(item, "parameter_name"))

        # Date range filter
        has_date_filter = any(
            isinstance(item, type) and issubclass(item, DateRangeFilter) for item in filters
        )
        if not has_date_filter:
            date_field = None
            for field_name in self.auto_date_fields:
                if self._has_field(field_name):
                    date_field = field_name
                    break
            if date_field:
                class _AutoDateRangeFilter(DateRangeFilter):
                    pass

                _AutoDateRangeFilter.date_field = date_field  # type: ignore[attr-defined]
                filters.append(_AutoDateRangeFilter)

        # Boolean filters
        for field_name in self.auto_boolean_filters:
            if field_name not in existing and self._is_boolean_field(field_name):
                filters.append(field_name)

        # Status filter (only if field exists)
        if self.auto_status_filter not in existing and self._has_field(self.auto_status_filter):
            filters.append(self.auto_status_filter)

        return filters

    def get_list_select_related(self, request):
        base = super().get_list_select_related(request)  # type: ignore[misc]
        if base is True:
            return True
        select_related = set(base or [])

        for field_name in getattr(self, "list_display", []):
            if not isinstance(field_name, str):
                continue
            try:
                field = self.model._meta.get_field(field_name)
            except Exception:
                continue
            if field.is_relation and (field.many_to_one or field.one_to_one):
                select_related.add(field_name)

        return tuple(select_related)


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

class EnhancedModelAdmin(
    AutoListFilterMixin, ExportCSVMixin, ExportJSONMixin, AuditLogMixin, StatusBadgeMixin, admin.ModelAdmin
):
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
        """Get actions and add export + seed actions."""
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

        # Add seed actions if this model has seed specs
        if self._seed_has_specs():
            for action_name in (
                "seed_import_sync",
                "seed_import_no_sync",
                "seed_sync_and_save",
                "seed_export_json",
                "seed_export_csv",
            ):
                if hasattr(self, action_name) and action_name not in actions:
                    action = getattr(self, action_name)
                    actions[action_name] = (action, action_name, action.short_description)
        
        return actions

    # -------------------------------------------------------------------------
    # Seed actions (auto-enabled for models with JSON seed specs)
    # -------------------------------------------------------------------------

    seed_specs: list[str] | None = None

    def _seed_has_specs(self) -> bool:
        return len(self._seed_get_specs()) > 0

    def _seed_get_specs(self):
        from core.seed.registry import autodiscover_seeders, get_seed_specs
        from core.seed.base import JSONSeedSpec

        autodiscover_seeders()
        specs = get_seed_specs()

        if self.seed_specs:
            matched = []
            for name in self.seed_specs:
                spec = specs.get(name)
                if spec:
                    matched.append(spec)
            return matched

        # Auto-detect JSON seed specs for this model
        matched = []
        for spec in specs.values():
            if isinstance(spec, JSONSeedSpec) and getattr(spec, "model", None) is self.model and spec.kind == "prod":
                matched.append(spec)
        return matched

    def _seed_get_export_fields(self, spec):
        fields: list[str] = []
        key_fields = list(spec.key_fields or [])
        update_fields = list(spec.update_fields or [])

        # Avoid duplicating key fields that are represented by base field names
        for field in key_fields:
            if "__" in field:
                base = field.split("__", 1)[0]
                if base in update_fields or base in getattr(spec, "fk_fields", {}) or base in getattr(spec, "m2m_fields", {}):
                    continue
            if field not in fields:
                fields.append(field)

        for field in update_fields:
            if field not in fields:
                fields.append(field)

        for field in getattr(spec, "m2m_fields", {}).keys():
            if field not in fields:
                fields.append(field)

        return fields

    def _seed_resolve_attr(self, obj, path: str):
        value = obj
        for part in path.split("__"):
            value = getattr(value, part, None)
            if value is None:
                break
        return value

    def _seed_export_records(self, spec):
        from core.seed.base import JSONSeedSpec

        if not isinstance(spec, JSONSeedSpec):
            raise ValueError("Export only supported for JSON seed specs.")

        fields = self._seed_get_export_fields(spec)
        fk_fields = getattr(spec, "fk_fields", {}) or {}
        m2m_fields = getattr(spec, "m2m_fields", {}) or {}

        qs = spec.model.objects.all()
        # Stable ordering if possible
        order_fields = [f for f in (spec.key_fields or []) if "__" not in f]
        if order_fields:
            qs = qs.order_by(*order_fields)
        else:
            qs = qs.order_by("pk")

        records = []
        for obj in qs:
            record = {}
            for field in fields:
                if field in m2m_fields:
                    model, lookup_field = m2m_fields[field]
                    items = getattr(obj, field).all()
                    record[field] = list(items.values_list(lookup_field, flat=True))
                    continue
                if field in fk_fields:
                    _, lookup_field = fk_fields[field]
                    rel = getattr(obj, field, None)
                    record[field] = getattr(rel, lookup_field, None) if rel else None
                    continue
                if "__" in field:
                    value = self._seed_resolve_attr(obj, field)
                else:
                    value = getattr(obj, field, None)
                if hasattr(value, "pk"):
                    value = value.pk
                record[field] = value
            records.append(record)
        return records

    def _seed_write_data_path(self, spec, records):
        data_path = Path(spec.data_path)
        if not data_path.is_absolute():
            data_path = Path(settings.BASE_DIR) / data_path
        data_path.parent.mkdir(parents=True, exist_ok=True)

        # Preserve existing JSON structure when possible
        payload = None
        if data_path.exists():
            try:
                with data_path.open("r", encoding="utf-8-sig") as fh:
                    payload = json.load(fh)
            except Exception:
                payload = None

        if hasattr(spec, "section_key"):
            if not isinstance(payload, dict):
                payload = {}
            payload[getattr(spec, "section_key")] = records
        elif isinstance(payload, dict):
            if "items" in payload:
                payload["items"] = records
            elif "data" in payload:
                payload["data"] = records
            else:
                payload = records
        else:
            payload = records

        with data_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, cls=DjangoJSONEncoder)
        return data_path

    def _seed_build_download(self, specs, fmt: str):
        if len(specs) == 1:
            spec = specs[0]
            records = self._seed_export_records(spec)
            if fmt == "json":
                data = json.dumps(records, indent=2, ensure_ascii=False, cls=DjangoJSONEncoder).encode("utf-8")
                content_type = "application/json"
                filename = f"{spec.name.replace('.', '_')}.json"
            else:
                output = StringIO()
                fields = self._seed_get_export_fields(spec)
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                for row in records:
                    flat = {}
                    for key, value in row.items():
                        if isinstance(value, list):
                            flat[key] = ",".join(str(v) for v in value)
                        elif isinstance(value, dict):
                            flat[key] = json.dumps(value, ensure_ascii=False)
                        else:
                            flat[key] = value
                    writer.writerow(flat)
                data = output.getvalue().encode("utf-8")
                content_type = "text/csv"
                filename = f"{spec.name.replace('.', '_')}.csv"
            return data, content_type, filename

        # Multiple specs -> zip
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for spec in specs:
                records = self._seed_export_records(spec)
                if fmt == "json":
                    content = json.dumps(records, indent=2, ensure_ascii=False, cls=DjangoJSONEncoder).encode("utf-8")
                    name = f"{spec.name.replace('.', '_')}.json"
                else:
                    output = StringIO()
                    fields = self._seed_get_export_fields(spec)
                    writer = csv.DictWriter(output, fieldnames=fields)
                    writer.writeheader()
                    for row in records:
                        flat = {}
                    for key, value in row.items():
                        if isinstance(value, list):
                            flat[key] = ",".join(str(v) for v in value)
                        elif isinstance(value, dict):
                            flat[key] = json.dumps(value, ensure_ascii=False)
                        else:
                            flat[key] = value
                        writer.writerow(flat)
                    content = output.getvalue().encode("utf-8")
                    name = f"{spec.name.replace('.', '_')}.csv"
                zf.writestr(name, content)
        buffer.seek(0)
        return buffer.read(), "application/zip", f"seed_export_{fmt}.zip"

    def _seed_run(self, request, *, no_prune: bool):
        specs = self._seed_get_specs()
        if not specs:
            self.message_user(request, "No seed specs found for this model.", level=messages.WARNING)
            return False
        names = ",".join([spec.name for spec in specs])
        call_command("seed", only=names, no_prune=no_prune, confirm_prune=True)
        return True

    def seed_import_sync(self, request, queryset):
        """Import seed data (sync + prune)."""
        try:
            if self._seed_run(request, no_prune=False):
                self.message_user(request, "Imported seed data with sync (prune).", level=messages.SUCCESS)
        except Exception as exc:
            self.message_user(request, f"Error importing seed data: {exc}", level=messages.ERROR)
    seed_import_sync.short_description = "Import seed data (sync + prune)"

    def seed_import_no_sync(self, request, queryset):
        """Import seed data without pruning existing records."""
        try:
            if self._seed_run(request, no_prune=True):
                self.message_user(request, "Imported seed data without pruning.", level=messages.SUCCESS)
        except Exception as exc:
            self.message_user(request, f"Error importing seed data: {exc}", level=messages.ERROR)
    seed_import_no_sync.short_description = "Import seed data (no prune)"

    def seed_sync_and_save(self, request, queryset):
        """Sync seed data and export back to the data file."""
        try:
            specs = self._seed_get_specs()
            if not specs:
                self.message_user(request, "No seed specs found for this model.", level=messages.WARNING)
                return
            self._seed_run(request, no_prune=False)
            exported_paths = []
            for spec in specs:
                records = self._seed_export_records(spec)
                path = self._seed_write_data_path(spec, records)
                exported_paths.append(str(path))
            self.message_user(
                request,
                f"Synced seed data and saved export: {', '.join(exported_paths)}",
                level=messages.SUCCESS,
            )
        except Exception as exc:
            self.message_user(request, f"Error syncing/exporting seed data: {exc}", level=messages.ERROR)
    seed_sync_and_save.short_description = "Sync and save seed data"

    def seed_export_json(self, request, queryset):
        """Export seed data as JSON (download)."""
        try:
            specs = self._seed_get_specs()
            if not specs:
                self.message_user(request, "No seed specs found for this model.", level=messages.WARNING)
                return
            data, content_type, filename = self._seed_build_download(specs, "json")
            response = HttpResponse(data, content_type=content_type)
            response["Content-Disposition"] = f'attachment; filename="{filename}"'
            return response
        except Exception as exc:
            self.message_user(request, f"Error exporting seed data: {exc}", level=messages.ERROR)
    seed_export_json.short_description = "Export seed data as JSON"

    def seed_export_csv(self, request, queryset):
        """Export seed data as CSV (download)."""
        try:
            specs = self._seed_get_specs()
            if not specs:
                self.message_user(request, "No seed specs found for this model.", level=messages.WARNING)
                return
            data, content_type, filename = self._seed_build_download(specs, "csv")
            response = HttpResponse(data, content_type=content_type)
            response["Content-Disposition"] = f'attachment; filename="{filename}"'
            return response
        except Exception as exc:
            self.message_user(request, f"Error exporting seed data: {exc}", level=messages.ERROR)
    seed_export_csv.short_description = "Export seed data as CSV"


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
