import json

import httpx
import openai
import pytest

from askmap.catalogue import catalogue_text
from askmap.config import PROJECT_ROOT, load_config
from askmap.llm import call, parse_plan, system_prompt, translate

CFG = load_config(PROJECT_ROOT / "configs/feasibility.yaml")
LLM = dict(CFG["llm"], retry_wait_s=0)


class Raw:
    def __init__(self, content):
        self.headers = {"x-inference-provider": "fake"}
        self.content = content

    def parse(self):
        msg = openai.types.chat.ChatCompletionMessage(role="assistant", content=self.content)
        choice = openai.types.chat.chat_completion.Choice(index=0, message=msg, finish_reason="stop")
        return openai.types.chat.ChatCompletion(
            id="x", choices=[choice], created=0, model="fake/model", object="chat.completion"
        )


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.chat = self
        self.completions = self
        self.with_raw_response = self

    def create(self, **kw):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return Raw(r)


def test_parse_plan_strict():
    assert parse_plan('  {"a": 1} ') == {"a": 1}
    assert parse_plan('```json\n{"a": 1}\n```') == {"a": 1}
    for bad in ("", None, "[1, 2]", "Here is the plan: {}", '{"a": 1} trailing'):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            parse_plan(bad)


def test_translate_records_saves_and_reuses(tmp_path):
    c = FakeClient(['{"area": "x"}'])
    rec = translate(c, "m", "sys", "q?", tmp_path / "a.json", LLM)
    assert rec["plan"] == {"area": "x"} and rec["error"] is None and rec["provider"] == "fake"
    assert rec["model_returned"] == "fake/model" and rec["request"]["messages"][1]["content"] == "q?"
    again = translate(c, "m", "sys", "q?", tmp_path / "a.json", LLM)
    assert again == json.loads((tmp_path / "a.json").read_text()) and c.calls == 1
    assert (tmp_path / "a.json").read_text().endswith("}\n")  # a text file, so the hooks leave it alone
    bad = translate(FakeClient(["not json"]), "m", "sys", "q", tmp_path / "b.json", LLM)
    assert bad["plan"] is None and bad["error"].startswith("malformed plan")


def test_call_retries_only_transport_errors():
    req = httpx.Request("POST", "https://x")
    c = FakeClient([openai.APIConnectionError(request=req), '{"a": 1}'])
    assert call(c, "m", "s", "q", 10, 0, 2, 0)["transport_attempts"] == 2
    c = FakeClient([openai.APIConnectionError(request=req)] * 2)
    with pytest.raises(openai.APIConnectionError):
        call(c, "m", "s", "q", 10, 0, 1, 0)


def test_system_prompt_and_catalogue():
    text = catalogue_text(CFG)
    assert "`minnesota`" in text and "`solar_farm`" in text and "`LOW_WIND_SUN`" in text
    assert "`town`" in text and "`developed_open`" in text and "`developed`" in text
    older = catalogue_text(CFG, ["corn", "developed", "other"])
    assert "`developed`" in older and "`town`" not in older
    assert "unsupported" in system_prompt((PROJECT_ROOT / "configs" / "language.md").read_text(), text)
    assert "# Query language" in system_prompt("L", text)


def test_same_plan_ignores_order_and_number_spelling():
    from askmap.llm import same_plan

    a = {
        "area": "x",
        "land": {"classes": ["corn", "soybeans"], "min_patch_ha": None},
        "near": [{"classes": ["wetland"], "distance_m": 100, "negate": False}],
        "climate": [],
        "output": "map",
    }
    b = {
        "output": "map",
        "climate": [],
        "area": "x",
        "near": [{"negate": False, "distance_m": 100.0, "classes": ["wetland"]}],
        "land": {"min_patch_ha": None, "classes": ["soybeans", "corn"]},
    }
    assert same_plan(a, b)
    b["near"][0]["distance_m"] = 30
    assert not same_plan(a, b)


def test_hf_token_sources(tmp_path, monkeypatch):
    from askmap.llm import hf_token

    monkeypatch.delenv("HF_TOKEN", raising=False)
    f = tmp_path / "token"
    f.write_text("abc\n")
    assert hf_token(f) == "abc"
    monkeypatch.setenv("HF_TOKEN", "xyz")
    assert hf_token(f) == "xyz"
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(RuntimeError):
        hf_token(tmp_path / "missing")


def test_system_prompt_tells_the_model_when_now_is():
    from datetime import date

    text = system_prompt("L", "C", today=date(2026, 10, 29), data_years=(2001, 2025))
    assert "Today is 2026-10-29" in text and '"last year" is 2025' in text and '"This year" is 2026' in text
    assert "covers 2001 to 2025 inclusive" in text
    assert "Today is " + date.today().isoformat() in system_prompt("L", "C")
