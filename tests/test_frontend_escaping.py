"""Frontend escaping guards, checked against the shipped HTML.

The UI renders third-party text: video titles, channel names, and RSS `<link>`
values. It is a single-file app with no build step and no JS test runner, so
these tests read `frontend/index.html` and assert the two properties that stop
a hostile feed from running JavaScript in the page.

That matters more here than in a typical app, because the API has no
authentication: script running in this page can call `POST /api/clear` (drops
the whole index) or read the corpus, same-origin, with no further steps.

A real case this pins: a podcast feed whose `<link>` is
`x" onmouseover="fetch('/api/clear',{method:'POST'})` — escaping only
`& < >` leaves the quote intact, the attribute closes, and the handler is live
as soon as the user's pointer crosses the discovery results.
"""
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
HTML = FRONTEND.read_text(encoding="utf-8")


def _py_esc(s: str) -> str:
    """Mirror of the JS esc(), used to render the assertions below."""
    return (
        str(s)
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;").replace("'", "&#39;")
    )


def _py_safe_url(u: str) -> str:
    return u.strip() if re.match(r"^https?://", u.strip(), re.IGNORECASE) else "#"


class _Attrs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


# --- the shipped source must implement both defences ------------------------

def test_esc_escapes_quotes_not_just_angle_brackets():
    """esc() output is interpolated into attributes, so quotes must be escaped."""
    esc_src = re.search(r"function esc\(s\)\s*\{.*?\n\}", HTML, re.S)
    assert esc_src, "esc() not found in frontend/index.html"
    body = esc_src.group(0)
    assert "&quot;" in body, 'esc() must escape double quotes (attribute breakout)'
    assert "&#39;" in body, "esc() must escape single quotes"


def test_every_href_interpolation_is_scheme_checked():
    """`javascript:` needs no quotes to fire, so escaping alone is not enough."""
    hrefs = re.findall(r'href="\$\{([^}]*)\}"', HTML)
    assert hrefs, "expected at least one interpolated href"
    for expr in hrefs:
        assert "safeUrl(" in expr, (
            f'href interpolation {expr!r} does not pass through safeUrl(); '
            "a hostile RSS <link> could use a javascript: URL"
        )


# --- the behaviour those defences produce -----------------------------------

def test_hostile_feed_link_cannot_inject_an_event_handler():
    hostile = "x\" onmouseover=\"fetch('/api/clear',{method:'POST'})\" data-x=\""
    rendered = f'<a href="{_py_esc(_py_safe_url(hostile))}">title</a>'
    p = _Attrs()
    p.feed(rendered)
    _, attrs = p.tags[0]
    assert not [k for k in attrs if k.startswith("on")], f"event handler injected: {attrs}"
    assert set(attrs) == {"href"}, f"unexpected attributes materialised: {attrs}"


def test_javascript_scheme_is_neutralised():
    for hostile in (
        "javascript:fetch('/api/clear',{method:'POST'})",
        "JaVaScRiPt:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "vbscript:msgbox(1)",
    ):
        assert _py_safe_url(hostile) == "#", f"{hostile!r} survived safeUrl()"


def test_ordinary_links_still_work():
    for benign in ("https://www.youtube.com/watch?v=abc123", "http://pod.example/ep1"):
        assert _py_safe_url(benign) == benign
