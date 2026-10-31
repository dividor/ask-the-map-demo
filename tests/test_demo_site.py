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
