"""
Template tags for catalog app.
"""
from django import template

register = template.Library()


@register.filter
def get_children(category, categories):
    """Get direct children of a category from a list of categories."""
    # Handle both dict and object access
    cat_id = category.get('id') if isinstance(category, dict) else getattr(category, 'id', None)
    return [
        c for c in categories
        if (c.get('parent_id') if isinstance(c, dict) else getattr(c, 'parent_id', None)) == cat_id
    ]


@register.filter
def is_descendant(selected_id, parent_category):
    """
    Check if selected_id is a descendant of parent_category.
    Usage: selected|is_descendant:category
    """
    if not selected_id:
        return False

    # Get all categories from the parent_category dict (auto-injected by widget)
    # dict has been prepared with _categories key containing the full list
    categories = parent_category.get('_categories') if isinstance(parent_category, dict) else None
    if not categories:
        return False  # Can't determine without full tree

    # Helper to get id/parent_id from dict or object
    def get_id(obj):
        return obj.get('id') if isinstance(obj, dict) else getattr(obj, 'id', None)

    def get_parent_id(obj):
        return obj.get('parent_id') if isinstance(obj, dict) else getattr(obj, 'parent_id', None)

    # Build parent lookup
    parent_map = {get_id(c): get_parent_id(c) for c in categories}

    # Walk up from selected_id
    current_id = int(selected_id) if selected_id else None
    parent_id = get_id(parent_category) if parent_category else None
    parent_id = int(parent_id) if parent_id else None

    visited = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        current_parent = parent_map.get(current_id)
        if current_parent == parent_id:
            return True
        current_id = current_parent

    return False
