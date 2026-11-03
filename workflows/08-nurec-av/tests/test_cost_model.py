"""Mcity cost slide: line items, totals, ratios, and the plug-in model."""

import json

import cost_model as cm
import pytest


def test_slide_arithmetic_checks_out():
    chk = cm.verify_slide()
    assert chk["items_sum_usd"] == pytest.approx(2308.40)
    assert chk["items_sum_matches_total_within_usd_5"]
    assert chk["model_reproduces_items"]
    assert chk["cheaper_x_matches_65"] and chk["cheaper_x_from_items"] == pytest.approx(64.98, abs=0.01)
    assert chk["faster_x_matches_90"] and chk["faster_x_30_day_months"] == 90.0
    assert chk["faster_x_calendar_months"] == pytest.approx(91.3, abs=0.05)


def test_default_model_reproduces_slide_items():
    items = cm.NuRecCost().items()
    assert list(items.values()) == pytest.approx([1300.0, 30.0, 18.40, 960.0])
    res = cm.compare(cm.NuRecCost(), 150_000, 180, 2)
    assert (res["cheaper_x"], res["faster_x"]) == (65.0, 90.0)
    assert res["traditional_usd_per_acre"] == 5000.0


def test_custom_rates():
    c = cm.NuRecCost(gpu_hours=16, gpu_rate_usd_h=2.5, storage_months=12, staff_hours=24, staff_rate_usd_h=85)
    assert c.total() == pytest.approx(1300 + 40 + 220.8 + 2040)
    assert cm.compare(c, 150_000, 180, 2)["cheaper_x"] == pytest.approx(41.7, abs=0.05)


def test_cli_json(capsys):
    assert cm.main(["--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["nurec_total_usd"] == pytest.approx(2308.4) and out["slide_check"]["cheaper_x_matches_65"]
