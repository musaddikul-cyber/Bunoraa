from django import template

register = template.Library()


def _normalize(value):
    if value is None:
        return ""
    return str(value).strip()


@register.simple_tag
def format_address_line(
    address_line_1=None,
    address_line_2=None,
    city=None,
    postal_code=None,
    state=None,
    country=None,
):
    """
    Format address as:
    Address line 2, Address line 1, City-Postal Code, State/Division, Country
    """
    line1 = _normalize(address_line_1)
    line2 = _normalize(address_line_2)
    city_val = _normalize(city)
    postal_val = _normalize(postal_code)
    state_val = _normalize(state)
    country_val = _normalize(country)

    city_postal = ""
    if city_val and postal_val:
        city_postal = f"{city_val}-{postal_val}"
    else:
        city_postal = city_val or postal_val

    parts = []
    if line2:
        parts.append(line2)
    if line1:
        parts.append(line1)
    if city_postal:
        parts.append(city_postal)
    if state_val:
        parts.append(state_val)
    if country_val:
        parts.append(country_val)

    return ", ".join(parts)
