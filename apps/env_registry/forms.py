from django import forms

from .models import EnvValue


class EnvValueForm(forms.ModelForm):
    value = forms.CharField(required=False)

    class Meta:
        model = EnvValue
        fields = ["variable", "environment", "value"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        instance = self.instance
        variable = getattr(instance, "variable", None)
        if variable and variable.is_secret:
            self.fields["value"].widget = forms.PasswordInput(render_value=False)
            self.fields["value"].help_text = "Leave blank to keep the existing secret."
        else:
            if instance and instance.pk:
                self.fields["value"].initial = instance.get_value() or ""

    def save(self, commit=True):
        instance = super().save(commit=False)
        value = self.cleaned_data.get("value", "")
        variable = instance.variable

        if variable.is_secret:
            if value:
                instance.set_value(value, is_secret=True)
        else:
            instance.set_value(value, is_secret=False)

        if commit:
            instance.save()
        return instance
