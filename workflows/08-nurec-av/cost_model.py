#!/usr/bin/env python3
"""Mcity digital-twin cost: hand-built vs NuRec, reproduced from the slide, with your own rates plugged in.

Source: the Mcity (University of Michigan) slide "Traditional vs NuRec - Cost Comparison" shown in the SIGGRAPH 2026
talk "AV Simulation with NuRec, Cosmos-Dreams, & 3D Gaussian Splatting" (assets/screens/mcity-cost-comparison.jpg):

    $150,000 -> ~$2,310 | ~65x cheaper          6 months -> 2 days | ~90x faster

    Track time (scan)   $1,300
    Compute (8 hrs)     ~$30
    S3 storage (800 GB) $18.40/mo
    Salary              $960
    TOTAL               ~ $2,310

The talk (caption) adds: the hand-built twin covered a 30-acre track and took "about 6 months ... 150k or so" including
contract, salary, and track time; the NuRec twin took about half a day of data collection plus a day and a half of
conversion and reconstruction ("if you have more GPUs you can cut this time in half").

What the slide does not give, and this model therefore treats as inputs: the GPU type and hourly rate behind "~$30",
the person-hours behind "$960", and how many months of storage are counted. Defaults reproduce the slide:
$3.75/GPU-hour x 8 h, $0.023/GB-month x 800 GB x 1 month (both implied by the slide's numbers), 16 h x $60/h salary
(an assumption that reproduces $960; the slide shows only the total).

Usage::

    python cost_model.py                                   # the slide, checked
    python cost_model.py --gpu-rate 2.5 --gpu-hours 16 --staff-rate 85 --staff-hours 24 --storage-months 12
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

# Numbers printed on the slide.
SLIDE = {
    "traditional_usd": 150_000.0,
    "traditional_days": 6 * 30.0,  # "6 months"; 30-day months make the slide's ~90x exact
    "nurec_total_usd": 2_310.0,  # "~$2,310"
    "nurec_days": 2.0,
    "cheaper_x": 65.0,
    "faster_x": 90.0,
    "items": {"Track time (scan)": 1_300.0, "Compute (8 hrs)": 30.0, "S3 storage (800 GB)": 18.40, "Salary": 960.0},
}
TRACK_ACRES = 30.0  # talk (caption): "our 30-acre twin"


@dataclass(frozen=True)
class NuRecCost:
    track_time_usd: float = 1_300.0
    gpu_hours: float = 8.0
    gpu_rate_usd_h: float = 30.0 / 8.0  # implied by "Compute (8 hrs) ~$30"
    storage_gb: float = 800.0
    storage_usd_gb_month: float = 18.40 / 800.0  # implied by "$18.40/mo" for 800 GB
    storage_months: float = 1.0
    staff_hours: float = 16.0  # assumption: two 8-hour days
    staff_rate_usd_h: float = 60.0  # assumption: 16 h x $60 = the slide's $960

    def items(self) -> dict[str, float]:
        return {
            "Track time (scan)": self.track_time_usd,
            f"Compute ({self.gpu_hours:g} hrs)": self.gpu_hours * self.gpu_rate_usd_h,
            f"S3 storage ({self.storage_gb:g} GB x {self.storage_months:g} mo)": self.storage_gb
            * self.storage_usd_gb_month
            * self.storage_months,
            "Salary": self.staff_hours * self.staff_rate_usd_h,
        }

    def total(self) -> float:
        return sum(self.items().values())


def compare(nurec: NuRecCost, traditional_usd: float, traditional_days: float, nurec_days: float) -> dict:
    total = nurec.total()
    return {
        "inputs": asdict(nurec),
        "line_items_usd": {k: round(v, 2) for k, v in nurec.items().items()},
        "nurec_total_usd": round(total, 2),
        "traditional_usd": traditional_usd,
        "cheaper_x": round(traditional_usd / total, 1),
        "nurec_days": nurec_days,
        "traditional_days": traditional_days,
        "faster_x": round(traditional_days / nurec_days, 1),
        "nurec_usd_per_acre": round(total / TRACK_ACRES, 1),
        "traditional_usd_per_acre": round(traditional_usd / TRACK_ACRES, 1),
    }


def verify_slide() -> dict:
    """Check the slide's arithmetic: items sum to ~$2,310, ratios round to ~65x and ~90x."""
    items_sum = sum(SLIDE["items"].values())
    default = NuRecCost()
    checks = {
        "items_sum_usd": round(items_sum, 2),
        "items_sum_matches_total_within_usd_5": abs(items_sum - SLIDE["nurec_total_usd"]) <= 5.0,
        "model_reproduces_items": all(
            abs(a - b) < 1e-9 for a, b in zip(default.items().values(), SLIDE["items"].values(), strict=True)
        ),
        "cheaper_x_from_items": round(SLIDE["traditional_usd"] / items_sum, 2),
        "cheaper_x_matches_65": round(SLIDE["traditional_usd"] / items_sum) == SLIDE["cheaper_x"],
        "faster_x_30_day_months": SLIDE["traditional_days"] / SLIDE["nurec_days"],
        "faster_x_calendar_months": round(6 * 365.25 / 12 / SLIDE["nurec_days"], 1),
        "faster_x_matches_90": round(SLIDE["traditional_days"] / SLIDE["nurec_days"]) == SLIDE["faster_x"],
        "ann_arbor_extrapolation_note": "The talk's ~$90M linear extrapolation to Ann Arbor (caption) implies "
        f"{90e6 / (SLIDE['traditional_usd'] / TRACK_ACRES):,.0f} acres at the hand-built rate of "
        f"${SLIDE['traditional_usd'] / TRACK_ACRES:,.0f}/acre.",
    }
    return checks


def table(result: dict) -> str:
    lines = ["| Item | Cost (USD) |", "|---|---:|"]
    lines += [f"| {k} | {v:,.2f} |" for k, v in result["line_items_usd"].items()]
    lines.append(f"| **Total** | **{result['nurec_total_usd']:,.2f}** |")
    lines.append("")
    lines.append(
        f"Hand-built ${result['traditional_usd']:,.0f} / {result['traditional_days']:g} days vs NuRec "
        f"${result['nurec_total_usd']:,.0f} / {result['nurec_days']:g} days: "
        f"{result['cheaper_x']}x cheaper, {result['faster_x']}x faster."
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    d = NuRecCost()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track-time", type=float, default=d.track_time_usd)
    ap.add_argument("--gpu-hours", type=float, default=d.gpu_hours)
    ap.add_argument("--gpu-rate", type=float, default=d.gpu_rate_usd_h, help="USD per GPU-hour")
    ap.add_argument("--storage-gb", type=float, default=d.storage_gb)
    ap.add_argument("--storage-rate", type=float, default=d.storage_usd_gb_month, help="USD per GB-month")
    ap.add_argument("--storage-months", type=float, default=d.storage_months)
    ap.add_argument("--staff-hours", type=float, default=d.staff_hours)
    ap.add_argument("--staff-rate", type=float, default=d.staff_rate_usd_h)
    ap.add_argument("--traditional", type=float, default=SLIDE["traditional_usd"])
    ap.add_argument("--traditional-days", type=float, default=SLIDE["traditional_days"])
    ap.add_argument("--nurec-days", type=float, default=SLIDE["nurec_days"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    cost = NuRecCost(
        a.track_time,
        a.gpu_hours,
        a.gpu_rate,
        a.storage_gb,
        a.storage_rate,
        a.storage_months,
        a.staff_hours,
        a.staff_rate,
    )
    res = compare(cost, a.traditional, a.traditional_days, a.nurec_days)
    res["slide_check"] = verify_slide()
    print(json.dumps(res, indent=2) if a.json else table(res) + "\n\nslide check: " + json.dumps(res["slide_check"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
