from django import template
from django.utils.html import strip_tags
from django.utils.safestring import mark_safe

register = template.Library()


def _clean_and_truncate(text, limit):
    if text is None:
        return ''
    text = strip_tags(text).strip()
    if len(text) <= limit:
        return text
    # Truncate without chopping words
    truncated = text[:limit].rsplit(' ', 1)[0]
    if not truncated:
        truncated = text[:limit]
    return truncated.rstrip() + '…'


@register.filter(name='seo_title')
def seo_title(value):
    """Ensure title is safe and <=60 chars."""
    return mark_safe(_clean_and_truncate(value, 60))


@register.filter(name='seo_desc')
def seo_desc(value):
    """Ensure meta description is safe and <=155 chars."""
    return mark_safe(_clean_and_truncate(value, 155))
