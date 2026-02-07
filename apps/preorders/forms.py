"""
Pre-orders forms - Forms for custom pre-order management
"""
from decimal import Decimal
from django import forms
from django.utils.translation import gettext_lazy as _
from django.core.validators import FileExtensionValidator
from django.core.exceptions import ValidationError

from .models import (
    PreOrderCategory, PreOrderOption, PreOrderOptionChoice,
    PreOrder, PreOrderDesign, PreOrderReference, PreOrderMessage,
    PreOrderRevision, PreOrderQuote
)


class PreOrderCreateForm(forms.ModelForm):
    """Form for creating a new pre-order."""
    
    class Meta:
        model = PreOrder
        fields = [
            'category', 'title', 'description', 'quantity',
            'special_instructions', 'is_gift', 'gift_wrap', 'gift_message',
            'is_rush_order', 'requested_delivery_date', 'customer_notes',
            'full_name', 'email', 'phone',
            'shipping_first_name', 'shipping_last_name',
            'shipping_address_line_1', 'shipping_address_line_2',
            'shipping_city', 'shipping_state', 'shipping_postal_code', 'shipping_country'
        ]
        widgets = {
            'category': forms.Select(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white'
            }),
            'title': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                'placeholder': _('Brief title for your custom order')
            }),
            'description': forms.Textarea(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                'rows': 5,
                'placeholder': _('Describe your custom order in detail...')
            }),
            'quantity': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                'min': 1
            }),
            'special_instructions': forms.Textarea(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                'rows': 3,
                'placeholder': _('Any special requirements or instructions...')
            }),
            'gift_message': forms.Textarea(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                'rows': 2,
                'placeholder': _('Message to include with the gift...')
            }),
            'requested_delivery_date': forms.DateInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                'type': 'date'
            }),
            'customer_notes': forms.Textarea(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                'rows': 2,
                'placeholder': _('Any additional notes...')
            }),
            'full_name': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                'placeholder': _('Your full name')
            }),
            'email': forms.EmailInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                'placeholder': _('your@email.com')
            }),
            'phone': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                'placeholder': _('Your phone number')
            }),
            'shipping_first_name': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white'
            }),
            'shipping_last_name': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white'
            }),
            'shipping_address_line_1': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                'placeholder': _('Street address')
            }),
            'shipping_address_line_2': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                'placeholder': _('Apartment, suite, etc.')
            }),
            'shipping_city': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white'
            }),
            'shipping_state': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white'
            }),
            'shipping_postal_code': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white'
            }),
            'shipping_country': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white'
            }),
            'is_gift': forms.CheckboxInput(attrs={
                'class': 'w-5 h-5 text-primary-600 rounded focus:ring-primary-500'
            }),
            'gift_wrap': forms.CheckboxInput(attrs={
                'class': 'w-5 h-5 text-primary-600 rounded focus:ring-primary-500'
            }),
            'is_rush_order': forms.CheckboxInput(attrs={
                'class': 'w-5 h-5 text-primary-600 rounded focus:ring-primary-500'
            }),
        }
    
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        # Only show active categories
        self.fields['category'].queryset = PreOrderCategory.objects.filter(is_active=True)
        
        # Pre-fill user data if authenticated
        if self.user and self.user.is_authenticated:
            self.fields['full_name'].initial = self.user.get_full_name()
            self.fields['email'].initial = self.user.email
            self.fields['phone'].initial = self.user.phone
    
    def clean_quantity(self):
        quantity = self.cleaned_data.get('quantity')
        category = self.cleaned_data.get('category')
        
        if category and quantity:
            if quantity < category.min_quantity:
                raise ValidationError(
                    _('Minimum quantity for this category is %(min)s'),
                    params={'min': category.min_quantity}
                )
            if quantity > category.max_quantity:
                raise ValidationError(
                    _('Maximum quantity for this category is %(max)s'),
                    params={'max': category.max_quantity}
                )
        
        return quantity
    
    def clean_is_rush_order(self):
        is_rush = self.cleaned_data.get('is_rush_order')
        category = self.cleaned_data.get('category')
        
        if is_rush and category and not category.allow_rush_order:
            raise ValidationError(_('Rush orders are not available for this category'))
        
        return is_rush


class PreOrderStepOneForm(forms.Form):
    """Step 1: Category selection and basic details."""
    
    category = forms.ModelChoiceField(
        queryset=PreOrderCategory.objects.filter(is_active=True),
        widget=forms.RadioSelect(attrs={
            'class': 'sr-only peer'
        }),
        empty_label=None
    )
    
    title = forms.CharField(
        max_length=300,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'placeholder': _('e.g., Custom Wedding Ring Set')
        })
    )
    
    description = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'rows': 5,
            'placeholder': _('Describe your custom order in detail. Include size, color, material preferences, etc.')
        })
    )
    
    quantity = forms.IntegerField(
        min_value=1,
        initial=1,
        widget=forms.NumberInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'min': 1
        })
    )


class PreOrderStepTwoForm(forms.Form):
    """Step 2: Customization options (dynamic based on category)."""
    
    def __init__(self, *args, category=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.category = category
        
        if category:
            # Dynamically add fields based on category options
            for option in category.options.filter(is_active=True).order_by('order'):
                field_name = f'option_{option.id}'
                
                if option.option_type == PreOrderOption.OPTION_TEXT:
                    self.fields[field_name] = forms.CharField(
                        label=option.name,
                        required=option.is_required,
                        max_length=option.max_length,
                        help_text=option.help_text,
                        widget=forms.TextInput(attrs={
                            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                            'placeholder': option.placeholder
                        })
                    )
                
                elif option.option_type == PreOrderOption.OPTION_TEXTAREA:
                    self.fields[field_name] = forms.CharField(
                        label=option.name,
                        required=option.is_required,
                        help_text=option.help_text,
                        widget=forms.Textarea(attrs={
                            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                            'rows': 3,
                            'placeholder': option.placeholder
                        })
                    )
                
                elif option.option_type == PreOrderOption.OPTION_NUMBER:
                    self.fields[field_name] = forms.DecimalField(
                        label=option.name,
                        required=option.is_required,
                        help_text=option.help_text,
                        widget=forms.NumberInput(attrs={
                            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                            'placeholder': option.placeholder,
                            'step': 'any'
                        })
                    )
                
                elif option.option_type == PreOrderOption.OPTION_SELECT:
                    choices = [(str(c.id), f"{c.display_name} (+{c.price_modifier})" if c.price_modifier > 0 else c.display_name) 
                              for c in option.choices.filter(is_active=True).order_by('order')]
                    self.fields[field_name] = forms.ChoiceField(
                        label=option.name,
                        required=option.is_required,
                        choices=[('', _('Select an option...'))] + choices,
                        help_text=option.help_text,
                        widget=forms.Select(attrs={
                            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white'
                        })
                    )
                
                elif option.option_type == PreOrderOption.OPTION_MULTISELECT:
                    choices = [(str(c.id), f"{c.display_name} (+{c.price_modifier})" if c.price_modifier > 0 else c.display_name)
                              for c in option.choices.filter(is_active=True).order_by('order')]
                    self.fields[field_name] = forms.MultipleChoiceField(
                        label=option.name,
                        required=option.is_required,
                        choices=choices,
                        help_text=option.help_text,
                        widget=forms.CheckboxSelectMultiple(attrs={
                            'class': 'space-y-2'
                        })
                    )
                
                elif option.option_type == PreOrderOption.OPTION_CHECKBOX:
                    self.fields[field_name] = forms.BooleanField(
                        label=option.name,
                        required=False,
                        help_text=option.help_text,
                        widget=forms.CheckboxInput(attrs={
                            'class': 'w-5 h-5 text-primary-600 rounded focus:ring-primary-500'
                        })
                    )
                
                elif option.option_type == PreOrderOption.OPTION_COLOR:
                    self.fields[field_name] = forms.CharField(
                        label=option.name,
                        required=option.is_required,
                        help_text=option.help_text,
                        widget=forms.TextInput(attrs={
                            'class': 'w-16 h-12 p-1 border border-gray-300 dark:border-gray-600 rounded-lg cursor-pointer',
                            'type': 'color'
                        })
                    )
                
                elif option.option_type == PreOrderOption.OPTION_DATE:
                    self.fields[field_name] = forms.DateField(
                        label=option.name,
                        required=option.is_required,
                        help_text=option.help_text,
                        widget=forms.DateInput(attrs={
                            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                            'type': 'date'
                        })
                    )
                
                elif option.option_type == PreOrderOption.OPTION_FILE:
                    self.fields[field_name] = forms.FileField(
                        label=option.name,
                        required=option.is_required,
                        help_text=option.help_text,
                        widget=forms.ClearableFileInput(attrs={
                            'class': 'block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100'
                        })
                    )


class PreOrderStepThreeForm(forms.Form):
    """Step 3: File uploads (designs and references)."""
    
    design_files = forms.FileField(
        required=False,
        widget=forms.FileInput(
            attrs={
                'class': 'hidden',
                'accept': '.pdf,.png,.jpg,.jpeg,.ai,.psd,.svg,.eps,.cdr,.zip,.rar',
                'id': 'design_files'
            }
        ),
        help_text=_('Upload your design files (PDF, PNG, JPG, AI, PSD, SVG, EPS, CDR, ZIP, RAR)')
    )
    
    reference_images = forms.FileField(
        required=False,
        widget=forms.FileInput(
            attrs={
                'class': 'hidden',
                'accept': 'image/*',
                'id': 'reference_images'
            }
        ),
        help_text=_('Upload reference images for inspiration')
    )
    
    special_instructions = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'rows': 4,
            'placeholder': _('Any special requirements or instructions for your order...')
        })
    )


class PreOrderStepFourForm(forms.Form):
    """Step 4: Contact and shipping information."""
    
    full_name = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'placeholder': _('Your full name')
        })
    )
    
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'placeholder': _('your@email.com')
        })
    )
    
    phone = forms.CharField(
        max_length=20,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'placeholder': _('+880 1XXX-XXXXXX')
        })
    )
    
    shipping_first_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'placeholder': _('First name')
        })
    )
    
    shipping_last_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'placeholder': _('Last name')
        })
    )
    
    shipping_address_line_1 = forms.CharField(
        max_length=255,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'placeholder': _('Street address')
        })
    )
    
    shipping_address_line_2 = forms.CharField(
        max_length=255,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'placeholder': _('Apartment, suite, etc. (optional)')
        })
    )
    
    shipping_city = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'placeholder': _('City')
        })
    )
    
    shipping_state = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'placeholder': _('State/Division')
        })
    )
    
    shipping_postal_code = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'placeholder': _('Postal code')
        })
    )
    
    shipping_country = forms.CharField(
        max_length=100,
        initial='Bangladesh',
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'placeholder': _('Country')
        })
    )
    
    is_rush_order = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'w-5 h-5 text-primary-600 rounded focus:ring-primary-500'
        })
    )
    
    requested_delivery_date = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'type': 'date'
        })
    )
    
    is_gift = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'w-5 h-5 text-primary-600 rounded focus:ring-primary-500'
        })
    )
    
    gift_wrap = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'w-5 h-5 text-primary-600 rounded focus:ring-primary-500'
        })
    )
    
    gift_message = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'rows': 2,
            'placeholder': _('Gift message (optional)')
        })
    )
    
    terms_accepted = forms.BooleanField(
        required=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'w-5 h-5 text-primary-600 rounded focus:ring-primary-500'
        })
    )


class PreOrderMessageForm(forms.ModelForm):
    """Form for sending messages on a pre-order."""
    
    class Meta:
        model = PreOrderMessage
        fields = ['subject', 'message', 'attachment']
        widgets = {
            'subject': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                'placeholder': _('Subject (optional)')
            }),
            'message': forms.Textarea(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                'rows': 4,
                'placeholder': _('Type your message...')
            }),
            'attachment': forms.ClearableFileInput(attrs={
                'class': 'block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100'
            }),
        }


class PreOrderRevisionForm(forms.ModelForm):
    """Form for requesting a revision."""
    
    class Meta:
        model = PreOrderRevision
        fields = ['description']
        widgets = {
            'description': forms.Textarea(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
                'rows': 5,
                'placeholder': _('Describe the changes you would like...')
            }),
        }


class PreOrderDesignUploadForm(forms.Form):
    """Form for uploading additional design files."""
    
    file = forms.FileField(
        validators=[FileExtensionValidator(
            allowed_extensions=['pdf', 'png', 'jpg', 'jpeg', 'ai', 'psd', 
                               'svg', 'eps', 'cdr', 'zip', 'rar']
        )],
        widget=forms.ClearableFileInput(attrs={
            'class': 'hidden',
            'accept': '.pdf,.png,.jpg,.jpeg,.ai,.psd,.svg,.eps,.cdr,.zip,.rar',
            'id': 'design_upload'
        })
    )
    
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'rows': 2,
            'placeholder': _('Notes about this file...')
        })
    )


class QuoteResponseForm(forms.Form):
    """Form for responding to a quote."""
    
    action = forms.ChoiceField(
        choices=[
            ('accept', _('Accept Quote')),
            ('reject', _('Reject Quote')),
        ],
        widget=forms.RadioSelect(attrs={
            'class': 'sr-only peer'
        })
    )
    
    response_notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'rows': 3,
            'placeholder': _('Add any notes or concerns (optional)...')
        })
    )


class AdminQuoteForm(forms.ModelForm):
    """Admin form for creating/editing quotes."""
    
    class Meta:
        model = PreOrderQuote
        fields = [
            'base_price', 'customization_cost', 'rush_fee',
            'discount', 'shipping', 'tax',
            'valid_until', 'estimated_production_days', 'estimated_delivery_date',
            'terms', 'notes'
        ]
        widgets = {
            'base_price': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg',
                'step': '0.01'
            }),
            'customization_cost': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg',
                'step': '0.01'
            }),
            'rush_fee': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg',
                'step': '0.01'
            }),
            'discount': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg',
                'step': '0.01'
            }),
            'shipping': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg',
                'step': '0.01'
            }),
            'tax': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg',
                'step': '0.01'
            }),
            'valid_until': forms.DateTimeInput(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg',
                'type': 'datetime-local'
            }),
            'estimated_production_days': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg'
            }),
            'estimated_delivery_date': forms.DateInput(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg',
                'type': 'date'
            }),
            'terms': forms.Textarea(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg',
                'rows': 4
            }),
            'notes': forms.Textarea(attrs={
                'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg',
                'rows': 3
            }),
        }


class PreOrderSearchForm(forms.Form):
    """Form for searching pre-orders."""
    
    query = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'placeholder': _('Search by order number, title, or email...')
        })
    )
    
    status = forms.ChoiceField(
        required=False,
        choices=[('', _('All Statuses'))] + list(PreOrder.STATUS_CHOICES),
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white'
        })
    )
    
    category = forms.ModelChoiceField(
        required=False,
        queryset=PreOrderCategory.objects.filter(is_active=True),
        empty_label=_('All Categories'),
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white'
        })
    )
    
    date_from = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'type': 'date'
        })
    )
    
    date_to = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 dark:bg-gray-700 dark:text-white',
            'type': 'date'
        })
    )
