from django.test import RequestFactory, override_settings
from django.http import HttpResponse
from core.middleware.early_hints import EarlyHintsMiddleware


def test_early_hints_without_assets_does_not_add_link_header():
    rf = RequestFactory()
    req = rf.get('/')
    resp = HttpResponse('<html>ok</html>', content_type='text/html')
    mw = EarlyHintsMiddleware(lambda request: resp)
    resp2 = mw.process_response(req, resp)
    assert 'Link' not in resp2


@override_settings(MAIN_CSS='/static/build/main.css', MAIN_JS='/static/build/main.js')
def test_early_hints_adds_link_header_for_configured_assets():
    rf = RequestFactory()
    req = rf.get('/')
    resp = HttpResponse('<html>ok</html>', content_type='text/html')
    mw = EarlyHintsMiddleware(lambda request: resp)
    resp2 = mw.process_response(req, resp)
    assert 'Link' in resp2
    link = resp2['Link']
    assert 'rel=preload' in link
    assert '/static/build/main.css' in link
    assert '/static/build/main.js' in link
