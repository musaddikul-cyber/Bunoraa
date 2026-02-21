from django import forms
from django.forms.utils import flatatt
from django.utils.html import conditional_escape
from django.utils.safestring import mark_safe
from apps.i18n.models import Currency as I18nCurrency
from .models import (
    ASPECT_RATIO_DEFAULT_CODE,
    Category,
    Product,
    get_active_aspect_ratio_choices,
)


def _dynamic_aspect_choices(current_code: str | None = None):
    choices = get_active_aspect_ratio_choices(include_code=current_code)
    if not choices:
        return [(ASPECT_RATIO_DEFAULT_CODE, ASPECT_RATIO_DEFAULT_CODE)]
    return choices

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

class ProductAdminForm(forms.ModelForm):
    primary_category = forms.ModelChoiceField(
        queryset=Category.objects.all_with_deleted().filter(is_deleted=False).order_by('path'),
        required=False,
        widget=CategoryTreeWidget,
        label='Primary Category',
    )
    currency = forms.ModelChoiceField(
        queryset=I18nCurrency.objects.none(),
        to_field_name='code',
        empty_label=None,
        label='Currency',
    )
    aspect_ratio = forms.ChoiceField(
        choices=(),
        required=False,
        label='Aspect ratio',
    )

    class Meta:
        model = Product
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['categories'].queryset = Category.objects.all_with_deleted().filter(is_deleted=False).order_by('path')
        currencies = I18nCurrency.objects.order_by('sort_order', 'code')
        self.fields['currency'].queryset = currencies
        current_aspect = (
            self.initial.get('aspect_ratio')
            or getattr(self.instance, 'aspect_ratio', '')
            or ASPECT_RATIO_DEFAULT_CODE
        )
        self.fields['aspect_ratio'].choices = _dynamic_aspect_choices(current_aspect)
        self.fields['aspect_ratio'].initial = current_aspect

        current_code = (self.initial.get('currency') or getattr(self.instance, 'currency', '') or '').upper()
        if current_code:
            selected_currency = currencies.filter(code=current_code).first()
            if selected_currency:
                self.fields['currency'].initial = selected_currency

    def save(self, commit=True):
        instance = super().save(commit=False)
        currency = self.cleaned_data.get('currency')
        if currency:
            instance.currency = currency.code
        if commit:
            instance.save()
            self.save_m2m()
        return instance


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
