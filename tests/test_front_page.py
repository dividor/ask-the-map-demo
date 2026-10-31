"""The GitHub Pages front door (docs/index.html): shows the waking-up message at once and sends the visitor on
to the demo as soon as its server answers."""

from pathlib import Path

from askmap.config import PROJECT_ROOT

APP = "https://askmap-demo.delightfulmoss-7e73fb67.eastus2.azurecontainerapps.io/"


def test_front_page_message_and_redirect():
    page = (PROJECT_ROOT / "docs" / "index.html").read_text()
    assert "Get a cup of tea and a biscuit, it's waking up. Then you can search away!" in page
    assert f'const APP = "{APP}"' in page
    assert "window.location.replace(APP)" in page
    assert 'mode: "no-cors"' in page  # the reply is never read, so the demo needs no cross-origin header
    assert f'href="{APP}"' in page  # a direct link if the script cannot run
    assert "fd5c1ca7e496" in page  # the blog post


def test_front_page_is_a_complete_document():
    page = (PROJECT_ROOT / "docs" / "index.html").read_text()
    assert page.lstrip().lower().startswith("<!doctype html>")
    assert '<meta name="viewport"' in page and "<title>Ask the map</title>" in page
    assert Path(PROJECT_ROOT / "docs" / "index.html").stat().st_size < 20_000
