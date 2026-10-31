import json

import numpy as np
import pytest

from askmap.demo.site import STATIC_BODY, static_page, write_site
from askmap.score import pooled_counts


def test_static_page_switches_mode():
    assert STATIC_BODY in static_page('<html><body data-mode="live"><p>x</p></body></html>')
    with pytest.raises(ValueError):
        static_page("<html><body></body></html>")


def test_write_site(tmp_path):
    areas = {"a": {"label": "Area A", "tiles": []}}
    answers = [
        {"question": "corn", "area": "a", "result": {"facts": {"shown_pixels": 3}}},
        {"question": "wetlands", "area": "a", "result": {"facts": {"shown_pixels": 0}}},
    ]
    files = write_site(tmp_path, '<body data-mode="live"></body>', areas, answers)
    assert {f.name for f in files} == {"q01.json", "q02.json", "meta.json", "index.html"}
    meta = json.loads((tmp_path / "data" / "meta.json").read_text())
    assert meta["answers"][1] == {"question": "wetlands", "area": "a", "file": "q02.json"}
    assert json.loads((tmp_path / "data" / "q01.json").read_text())["facts"]["shown_pixels"] == 3
    assert STATIC_BODY in (tmp_path / "index.html").read_text()
    with pytest.raises(KeyError):
        write_site(tmp_path, '<body data-mode="live"></body>', areas, [dict(answers[0], area="b")])
    with pytest.raises(ValueError):
        write_site(tmp_path, '<body data-mode="live"></body>', areas, [])


def test_pooled_counts():
    pred = {"t": np.array([True, True, False, False])}
    truth = {"t": np.array([True, False, True, False])}
    valid = {"t": np.array([True, True, True, False])}
    c = pooled_counts(pred, truth, valid)
    assert c["both"] == 1 and c["predicted"] == 2 and c["true"] == 2
    assert c["precision"] == 0.5 and c["recall"] == 0.5 and c["f1"] == 0.5
    empty = pooled_counts({"t": np.zeros(3, bool)}, {"t": np.zeros(3, bool)}, {"t": np.ones(3, bool)})
    assert empty["precision"] is None and empty["recall"] is None and empty["f1"] == 1.0


def test_waking_up_modal_shows_only_the_message():
    """While the server wakes up, the modal shows the blogger's message and nothing else: the stage line is
    hidden in the markup and the script no longer writes a loading stage or a timer into it."""
    from askmap.demo.app import PAGE

    page = PAGE.read_text()
    assert "Get a cup of tea and a biscuit, it's waking up. Then you can search away!" in page
    assert '<p id="warm-stage" class="small" style="margin:8px 0 0" hidden></p>' in page
    assert "s so far" not in page
    assert "Loading failed: " in page  # a failed load is still reported


def test_page_links_to_the_blog_post():
    from askmap.demo.app import PAGE

    assert (
        'href="https://medium.com/@astrobagel/ask-the-map-finding-places-using-satellite-map-embeddings-and-nasa-power-climate-data-fd5c1ca7e496"'
        in PAGE.read_text()
    )


def test_page_has_a_phone_layout():
    """On narrow screens the map sits right under the question (the side panel's sections are placed around it),
    the legend folds behind a Key button at the top right, method names are shortened, and a search scrolls the
    question and map into view."""
    from askmap.demo.app import PAGE

    page = PAGE.read_text()
    assert '"title" "ask" "map" "intro" "more"' in page
    assert "aside { display: contents; }" in page
    for section in ("side-title", "side-intro", "side-ask", "side-more"):
        assert f'class="side {section}"' in page
    assert page.count('class="live-only"') == 2  # hidden together on the static copy of the page
    assert 'legend.setPosition(NARROW.matches ? "topright" : "bottomleft")' in page
    assert '<button class="legend-toggle" type="button" aria-expanded="false">Key</button>' in page
    assert page.count("data-short=") == 3
    assert 'window.scrollTo({ top: 0, behavior: "smooth" })' in page
    assert 'if (NARROW.matches) document.getElementById("q").after(box);' in page  # examples under the question
    assert "100vh" not in page  # dvh units only, so the map fits the visible part of a phone screen


def test_title_carries_the_page_icon():
    """The title shows the same globe-and-red-place icon as the browser tab."""
    from askmap.demo.app import PAGE

    page = PAGE.read_text()
    head, body = page.split("<body", 1)
    assert "<h1 class=\"brand\"><svg viewBox=\"0 0 32 32\" aria-hidden=\"true\">" in body
    # the drawing matches the favicon's: blue globe, white lines, red place
    for part in ("fill=\"#1f6fd0\"", "rx=\"6.5\" ry=\"14\"", "cx=\"21.5\" cy=\"11.5\" r=\"3.4\" fill=\"#ff1717\""):
        assert part in body
    assert "%231f6fd0" in head and "%23ff1717" in head
