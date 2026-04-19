from django import forms
from django.conf import settings
from django.contrib import admin
from django.contrib.admin.widgets import RelatedFieldWidgetWrapper, FilteredSelectMultiple
from django.db import models
from django.db.models import Q
from django.forms.utils import flatatt
from django.utils.html import conditional_escape
from django.utils.safestring import mark_safe
from apps.i18n.models import Currency as I18nCurrency
from .models import (
    ASPECT_RATIO_DEFAULT_CODE,
    AspectRatioChoice,
    Category,
    Product,
    get_active_aspect_ratio_choices,
)


# =============================================================================
# ENHANCED CATEGORY TREE WIDGET
# =============================================================================

class PrimaryCategoryTreeWidget(forms.Select):
    """
    Enhanced tree widget for primary category selection.
    Shows GitHub-style expandable tree with search and filtering.
    """
    
    template_name = 'admin/catalog/widgets/primary_category_tree.html'
    
    def __init__(self, attrs=None, choices=()):
        super().__init__(attrs)
        self.choices = list(choices)
    
    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        
        # Get all categories for tree building
        categories = Category.objects.all_with_deleted().filter(is_deleted=False).order_by('path')
        
        categories_list = [
            {
                'id': str(cat.id),
                'name': cat.name,
                'parent_id': str(cat.parent_id) if cat.parent_id else None,
                'depth': cat.depth,
                'sort_order': getattr(cat, 'sort_order', 0),
                'has_children': bool(cat.get_children_count() if hasattr(cat, 'get_children_count') else True),
            }
            for cat in categories
        ]
        
        # Add reference to categories list for is_descendant filter
        for cat in categories_list:
            cat['_categories'] = categories_list
        
        context['widget']['categories'] = categories_list
        context['widget']['selected_value'] = str(value) if value else None
        
        return context


class CategoriesFilteredWidget(forms.SelectMultiple):
    """
    Multi-select widget for categories that filters based on primary category.
    Shows tree structure with checkboxes.
    """
    
    template_name = 'admin/catalog/widgets/categories_filtered.html'
    
    def __init__(self, attrs=None, choices=()):
        super().__init__(attrs)
        self.choices = list(choices)
    
    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        
        # Get all categories for tree building
        categories = Category.objects.all_with_deleted().filter(is_deleted=False).order_by('path')
        
        categories_list = [
            {
                'id': str(cat.id),
                'name': cat.name,
                'parent_id': str(cat.parent_id) if cat.parent_id else None,
                'depth': cat.depth,
                'sort_order': getattr(cat, 'sort_order', 0),
            }
            for cat in categories
        ]
        
        # Add reference to categories list for is_descendant filter
        for cat in categories_list:
            cat['_categories'] = categories_list
        
        context['widget']['categories'] = categories_list
        context['widget']['selected_values'] = [str(v) for v in value] if value else []
        
        return context


def _dynamic_aspect_choices(current_code: str | None = None):
    choices = get_active_aspect_ratio_choices(include_code=current_code)
    if not choices:
        return [(ASPECT_RATIO_DEFAULT_CODE, ASPECT_RATIO_DEFAULT_CODE)]
    return choices


_ASPECT_RATIO_REL = models.ForeignKey(
    AspectRatioChoice,
    to_field="code",
    on_delete=models.PROTECT,
    related_name="+",
).remote_field


class CategoryDropdownWidget(forms.Select):
    """
    Modern category dropdown with drill-down navigation.
    Features: Search, breadcrumbs, hierarchical browsing, dark mode support.
    """
    template_name = 'admin/catalog/widgets/category_dropdown.html'

    class Media:
        css = {
            'all': ('css/admin/category_dropdown.css',)
        }

    def __init__(self, attrs=None, choices=()):
        super().__init__(attrs)
        self.choices = list(choices)

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)

        # Normalize value - handle various invalid inputs
        if isinstance(value, list):
            value = value[0] if value else None
        # Handle string representations of lists
        if isinstance(value, str):
            if value.startswith('[') and value.endswith(']'):
                # Try to parse as list
                try:
                    import ast
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, list) and parsed:
                        value = parsed[0]
                    else:
                        value = None
                except (ValueError, SyntaxError):
                    value = None
            # Handle empty or invalid strings
            if value in ('', "''", '""', 'None', 'null', '[]'):
                value = None

        # Get all categories for the tree
        try:
            categories = Category.objects.all_with_deleted().filter(is_deleted=False).order_by('path')
        except Exception:
            categories = Category.objects.filter(is_deleted=False).order_by('path')

        # Build hierarchical data structure for JavaScript
        categories_list = []
        root_categories = []

        for cat in categories:
            try:
                has_children = categories.filter(parent_id=cat.id).exists()
                cat_data = {
                    'id': str(cat.id),
                    'name': cat.name,
                    'parent_id': str(cat.parent_id) if cat.parent_id else None,
                    'depth': getattr(cat, 'depth', 0),
                    'has_children': has_children,
                    'path': getattr(cat, 'path', '') or str(cat.id),
                }
                categories_list.append(cat_data)

                if not cat.parent_id:
                    root_categories.append(cat_data)
            except Exception:
                # Skip problematic categories
                continue

        import json

        # Ensure context has widget dict
        if 'widget' not in context:
            context['widget'] = {}

        context['widget']['categories_json'] = json.dumps(categories_list)
        context['widget']['root_categories'] = root_categories
        context['widget']['selected_value'] = str(value) if value else None

        # Get selected category name
        if value:
            selected_cat = categories.filter(id=value).first()
            if selected_cat:
                context['widget']['selected_name'] = selected_cat.name

        return context


class ProductAdminForm(forms.ModelForm):
    currency = forms.ModelChoiceField(
        queryset=I18nCurrency.objects.none(),
        to_field_name='code',
        empty_label=None,
        label='Currency',
    )
    aspect_ratio = forms.ModelChoiceField(
        queryset=AspectRatioChoice.objects.none(),
        to_field_name="code",
        required=False,
        label='Aspect ratio',
    )

    class Meta:
        model = Product
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        request = kwargs.pop("request", None)
        super().__init__(*args, **kwargs)

        # Get categories for queryset
        categories_qs = Category.objects.all_with_deleted().filter(is_deleted=False).order_by('path')

        # Configure Facebook-style primary category dropdown
        if "primary_category" in self.fields:
            self.fields["primary_category"].queryset = categories_qs
            self.fields["primary_category"].required = True
            # Use modern dropdown widget
            self.fields["primary_category"].widget = CategoryDropdownWidget()
        
        # Configure enhanced categories widget
        if "categories" in self.fields:
            self.fields['categories'].queryset = categories_qs
            self.fields['categories'].required = False
            # Use custom widget with tree structure  
            self.fields['categories'].widget = CategoriesFilteredWidget()

        # Configure tags widget
        if "tags" in self.fields:
            from .models import Tag
            self.fields['tags'].queryset = Tag.objects.all()
            self.fields['tags'].required = False
            # Use select2-style widget or simple select multiple
            self.fields['tags'].widget.attrs.update({
                'class': 'select2-tags',
                'style': 'min-width: 300px;',
                'data-placeholder': 'Select tags...'
            })

        currencies = I18nCurrency.objects.order_by('sort_order', 'code')
        self.fields['currency'].queryset = currencies
        current_aspect = (
            self.initial.get('aspect_ratio')
            or getattr(self.instance, 'aspect_ratio', '')
            or ASPECT_RATIO_DEFAULT_CODE
        )
        aspect_queryset = AspectRatioChoice.objects.filter(
            Q(is_active=True) | Q(code=current_aspect)
        ).order_by("sort_order", "label", "code")
        self.fields['aspect_ratio'].queryset = aspect_queryset
        selected_aspect = aspect_queryset.filter(code=current_aspect).first()
        if selected_aspect:
            self.fields['aspect_ratio'].initial = selected_aspect

        aspect_admin = admin.site._registry.get(AspectRatioChoice)
        can_add_related = bool(aspect_admin)
        can_change_related = bool(aspect_admin)
        can_delete_related = bool(aspect_admin)
        can_view_related = bool(aspect_admin)
        if request is not None and aspect_admin is not None:
            can_add_related = aspect_admin.has_add_permission(request)
            can_change_related = aspect_admin.has_change_permission(request)
            can_delete_related = aspect_admin.has_delete_permission(request)
            can_view_related = aspect_admin.has_view_permission(request)
        self.fields['aspect_ratio'].widget = RelatedFieldWidgetWrapper(
            self.fields['aspect_ratio'].widget,
            _ASPECT_RATIO_REL,
            admin.site,
            can_add_related=can_add_related,
            can_change_related=can_change_related,
            can_delete_related=can_delete_related,
            can_view_related=can_view_related,
        )

        current_code = (self.initial.get('currency') or getattr(self.instance, 'currency_id', '') or '').upper()
        if current_code:
            selected_currency = currencies.filter(code=current_code).first()
            if selected_currency:
                self.fields['currency'].initial = selected_currency

    def clean_aspect_ratio(self):
        selected = self.cleaned_data.get("aspect_ratio")
        if not selected:
            return ""
        return selected.code

    def clean(self):
        cleaned_data = super().clean()
        primary_category = cleaned_data.get('primary_category')
        categories = cleaned_data.get('categories')

        if not primary_category or categories is None:
            return cleaned_data

        ancestor_ids = list(
            primary_category.get_ancestors(include_self=True).values_list('id', flat=True)
        )
        if not ancestor_ids:
            return cleaned_data

        selected_ids = set(categories.values_list('id', flat=True))
        missing_ids = [category_id for category_id in ancestor_ids if category_id not in selected_ids]
        if missing_ids:
            cleaned_data['categories'] = categories | Category.objects.filter(
                id__in=missing_ids,
                is_deleted=False,
            )
        return cleaned_data


class CategoryAdminForm(forms.ModelForm):
    aspect_ratio = forms.ChoiceField(
        choices=(),
        required=False,
        label='Aspect ratio',
    )

    class Meta:
        model = Category
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        current_aspect = (
            self.initial.get('aspect_ratio')
            or getattr(self.instance, 'aspect_ratio', '')
            or ASPECT_RATIO_DEFAULT_CODE
        )
        self.fields['aspect_ratio'].choices = _dynamic_aspect_choices(current_aspect)
        self.fields['aspect_ratio'].initial = current_aspect
