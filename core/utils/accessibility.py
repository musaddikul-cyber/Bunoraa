"""
Accessibility utilities and helpers for WCAG compliance.
"""
from django import template
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from typing import Dict, Optional


def generate_skip_links() -> str:
    """
    Generate skip navigation links for keyboard users.
    Place at the very beginning of the page.
    """
    return mark_safe('''
        <nav class="skip-links" aria-label="Skip to content">
            <a href="#main-content" class="skip-link">Skip to main content</a>
            <a href="#main-navigation" class="skip-link">Skip to navigation</a>
            <a href="#search" class="skip-link">Skip to search</a>
            <a href="#footer" class="skip-link">Skip to footer</a>
        </nav>
        <style>
            .skip-links {
                position: absolute;
                top: 0;
                left: 0;
                z-index: 10000;
            }
            .skip-link {
                position: absolute;
                top: -100%;
                left: 0;
                background: #000;
                color: #fff;
                padding: 1rem 1.5rem;
                text-decoration: none;
                font-weight: 600;
                z-index: 10001;
                transition: top 0.3s;
            }
            .skip-link:focus {
                top: 0;
                outline: 3px solid #FFD700;
                outline-offset: 2px;
            }
        </style>
    ''')


def generate_aria_live_region(region_id: str = 'announcements', politeness: str = 'polite') -> str:
    """
    Generate an ARIA live region for dynamic announcements.
    
    Args:
        region_id: ID for the live region element
        politeness: 'polite', 'assertive', or 'off'
    """
    return mark_safe(f'''
        <div 
            id="{region_id}" 
            aria-live="{politeness}" 
            aria-atomic="true" 
            class="visually-hidden"
            role="status"
        ></div>
    ''')


def make_image_accessible(src: str, alt: str, decorative: bool = False, 
                          loading: str = 'lazy', **attrs) -> str:
    """
    Generate an accessible image tag.
    
    Args:
        src: Image source URL
        alt: Alt text (empty string for decorative images)
        decorative: If True, mark image as decorative (empty alt, role=presentation)
        loading: 'lazy' or 'eager'
        **attrs: Additional HTML attributes
    """
    extra_attrs = ' '.join(f'{k}="{v}"' for k, v in attrs.items())
    
    if decorative:
        return format_html(
            '<img src="{}" alt="" role="presentation" loading="{}" {}>',
            src, loading, mark_safe(extra_attrs)
        )
    else:
        return format_html(
            '<img src="{}" alt="{}" loading="{}" {}>',
            src, alt, loading, mark_safe(extra_attrs)
        )


def make_button_accessible(text: str, button_type: str = 'button', 
                           aria_label: Optional[str] = None,
                           disabled: bool = False, **attrs) -> str:
    """
    Generate an accessible button element.
    """
    aria_attr = f'aria-label="{aria_label}"' if aria_label else ''
    disabled_attr = 'disabled aria-disabled="true"' if disabled else ''
    extra_attrs = ' '.join(f'{k}="{v}"' for k, v in attrs.items())
    
    return format_html(
        '<button type="{}" {} {} {}>{}</button>',
        button_type, 
        mark_safe(aria_attr),
        mark_safe(disabled_attr),
        mark_safe(extra_attrs),
        text
    )


def make_icon_accessible(icon_class: str, label: str, 
                        decorative: bool = False) -> str:
    """
    Generate an accessible icon element.
    
    Args:
        icon_class: CSS class(es) for the icon
        label: Accessible label for the icon
        decorative: If True, hide from assistive technology
    """
    if decorative:
        return format_html(
            '<span class="{}" aria-hidden="true"></span>',
            icon_class
        )
    else:
        return format_html(
            '<span class="{}" role="img" aria-label="{}"></span>',
            icon_class, label
        )


def make_link_accessible(href: str, text: str, 
                        external: bool = False,
                        aria_label: Optional[str] = None,
                        **attrs) -> str:
    """
    Generate an accessible link element.
    
    Args:
        href: Link destination
        text: Link text
        external: If True, opens in new tab with proper attributes
        aria_label: Optional screen reader label
    """
    extra_attrs = ' '.join(f'{k}="{v}"' for k, v in attrs.items())
    aria_attr = f'aria-label="{aria_label}"' if aria_label else ''
    
    if external:
        return format_html(
            '<a href="{}" {} {} target="_blank" rel="noopener noreferrer">'
            '{}'
            '<span class="visually-hidden"> (opens in new tab)</span>'
            '</a>',
            href, mark_safe(aria_attr), mark_safe(extra_attrs), text
        )
    else:
        return format_html(
            '<a href="{}" {} {}>{}</a>',
            href, mark_safe(aria_attr), mark_safe(extra_attrs), text
        )


def generate_breadcrumb_nav(items: list, current_page: str) -> str:
    """
    Generate an accessible breadcrumb navigation.
    
    Args:
        items: List of tuples (label, url) or just label for current page
        current_page: Label for the current page
    """
    breadcrumb_items = []
    
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            label, url = item
            breadcrumb_items.append(
                f'<li class="breadcrumb-item">'
                f'<a href="{url}">{label}</a>'
                f'</li>'
            )
        else:
            breadcrumb_items.append(
                f'<li class="breadcrumb-item">{item}</li>'
            )
    
    # Add current page
    breadcrumb_items.append(
        f'<li class="breadcrumb-item active" aria-current="page">{current_page}</li>'
    )
    
    return mark_safe(f'''
        <nav aria-label="Breadcrumb" class="breadcrumb-nav">
            <ol class="breadcrumb">
                {''.join(breadcrumb_items)}
            </ol>
        </nav>
    ''')


def generate_form_field(field_id: str, label: str, field_type: str = 'text',
                       required: bool = False, error_msg: Optional[str] = None,
                       help_text: Optional[str] = None, **attrs) -> str:
    """
    Generate an accessible form field with proper labeling.
    """
    required_attr = 'required aria-required="true"' if required else ''
    error_id = f'{field_id}-error'
    help_id = f'{field_id}-help'
    
    aria_describedby = []
    if error_msg:
        aria_describedby.append(error_id)
    if help_text:
        aria_describedby.append(help_id)
    
    describedby_attr = f'aria-describedby="{" ".join(aria_describedby)}"' if aria_describedby else ''
    invalid_attr = 'aria-invalid="true"' if error_msg else ''
    extra_attrs = ' '.join(f'{k}="{v}"' for k, v in attrs.items())
    
    error_html = f'<span id="{error_id}" class="error-message" role="alert">{error_msg}</span>' if error_msg else ''
    help_html = f'<span id="{help_id}" class="help-text">{help_text}</span>' if help_text else ''
    required_indicator = '<span class="required-indicator" aria-hidden="true">*</span>' if required else ''
    
    return mark_safe(f'''
        <div class="form-field {'has-error' if error_msg else ''}">
            <label for="{field_id}" class="form-label">
                {label} {required_indicator}
            </label>
            <input 
                type="{field_type}" 
                id="{field_id}" 
                name="{field_id}"
                {required_attr}
                {describedby_attr}
                {invalid_attr}
                {extra_attrs}
                class="form-input"
            >
            {error_html}
            {help_html}
        </div>
    ''')


def generate_loading_indicator(text: str = 'Loading...') -> str:
    """Generate an accessible loading indicator."""
    return mark_safe(f'''
        <div class="loading-indicator" role="status" aria-live="polite">
            <span class="loading-spinner" aria-hidden="true"></span>
            <span class="loading-text">{text}</span>
        </div>
    ''')


def generate_modal_structure(modal_id: str, title: str, content: str) -> str:
    """
    Generate an accessible modal dialog structure.
    """
    return mark_safe(f'''
        <div 
            id="{modal_id}" 
            class="modal" 
            role="dialog" 
            aria-modal="true"
            aria-labelledby="{modal_id}-title"
            aria-describedby="{modal_id}-content"
            hidden
        >
            <div class="modal-overlay" data-modal-close></div>
            <div class="modal-container" role="document">
                <header class="modal-header">
                    <h2 id="{modal_id}-title" class="modal-title">{title}</h2>
                    <button 
                        type="button" 
                        class="modal-close" 
                        data-modal-close
                        aria-label="Close modal"
                    >
                        <span aria-hidden="true">&times;</span>
                    </button>
                </header>
                <div id="{modal_id}-content" class="modal-body">
                    {content}
                </div>
                <footer class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-modal-close>
                        Cancel
                    </button>
                    <button type="button" class="btn btn-primary">
                        Confirm
                    </button>
                </footer>
            </div>
        </div>
    ''')


# CSS for accessibility utilities
ACCESSIBILITY_CSS = '''
/* Visually hidden but accessible to screen readers */
.visually-hidden,
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}

/* Focus styles for keyboard navigation */
:focus-visible {
    outline: 3px solid var(--focus-color, #0066cc);
    outline-offset: 2px;
}

/* Reduced motion for users who prefer it */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}

/* High contrast mode support */
@media (prefers-contrast: high) {
    .btn,
    .form-input,
    .card {
        border: 2px solid currentColor;
    }
    
    a {
        text-decoration: underline;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    :root {
        --text-color: #f0f0f0;
        --bg-color: #1a1a1a;
        --focus-color: #4da6ff;
    }
}

/* Required field indicator */
.required-indicator {
    color: #dc3545;
    margin-left: 0.25rem;
}

/* Error messages */
.error-message {
    color: #dc3545;
    font-size: 0.875rem;
    margin-top: 0.25rem;
    display: block;
}

.has-error .form-input {
    border-color: #dc3545;
    box-shadow: 0 0 0 2px rgba(220, 53, 69, 0.25);
}

/* Loading indicator */
.loading-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.loading-spinner {
    width: 1.5rem;
    height: 1.5rem;
    border: 2px solid #e0e0e0;
    border-top-color: #0066cc;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Modal styles */
.modal {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
}

.modal[hidden] {
    display: none;
}

.modal-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
}

.modal-container {
    position: relative;
    background: #fff;
    border-radius: 8px;
    max-width: 500px;
    width: 90%;
    max-height: 90vh;
    overflow-y: auto;
}
'''
