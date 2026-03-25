from django import forms
from django.conf import settings
from django.contrib import admin
from django.contrib.admin.widgets import RelatedFieldWidgetWrapper
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

class CategoryTreeWidget(forms.Widget):
    def render(self, name, value, attrs=None, renderer=None):
        """
        Renders the widget as HTML, bypassing the template system.
        """
        # Get all categories to build the tree
        categories = Category.objects.all_with_deleted().filter(is_deleted=False).order_by('path')
        
        # Build the source list for the JS to consume
        source_lis = ""
        for category in categories:
            parent_id = category.parent_id if category.parent_id else ''
            source_lis += (
                f'<li data-id="{category.id}" data-parent-id="{parent_id}" '
                f'data-depth="{category.depth}">{conditional_escape(category.name)}</li>'
            )

        # The main HTML structure for the widget
        attrs_str = flatatt(self.build_attrs(attrs))
        html = f"""
        <div class="category-tree-widget-container">
            <input type="hidden" name="{name}" id="id_{name}" value="{value if value is not None else ''}"{attrs_str}>
            <div class="category-tree-wrapper">
                <ul class="category-tree-source" hidden>
                    {source_lis}
                </ul>
                <div class="category-tree-display"></div>
            </div>
        </div>
        """
        return mark_safe(html)

def _should_use_category_tree_widget() -> bool:
    enabled = getattr(settings, "ADMIN_CATEGORY_TREE_WIDGET_ENABLED", False)
    if not enabled:
        return False

    max_count = int(getattr(settings, "ADMIN_CATEGORY_TREE_WIDGET_MAX", 0) or 0)
    if max_count <= 0:
        return True

    try:
        count = Category.objects.all_with_deleted().filter(is_deleted=False).count()
    except Exception:
        return False

    return count <= max_count


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
        categories_qs = Category.objects.all_with_deleted().filter(is_deleted=False).order_by('path')
        if "categories" in self.fields:
            self.fields['categories'].queryset = categories_qs
        if "primary_category" in self.fields:
            self.fields["primary_category"].queryset = categories_qs
            if _should_use_category_tree_widget():
                self.fields["primary_category"].widget = CategoryTreeWidget()
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
