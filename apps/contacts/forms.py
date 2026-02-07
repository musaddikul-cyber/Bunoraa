from django import forms
from .models import CustomizationRequest

class CustomizationRequestForm(forms.ModelForm):
    class Meta:
        model = CustomizationRequest
        fields = ['name', 'email', 'phone', 'message']
        widgets = {
            'message': forms.Textarea(attrs={'rows': 4}),
        }
