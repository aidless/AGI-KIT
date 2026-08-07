from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from eval_round20_sft import normalized_equal


def test_normalized_equal_handles_equivalent_decimal_renderings():
    assert normalized_equal("560.0", "560")
    assert normalized_equal("-2.50", "-2.5")
    assert not normalized_equal("560.1", "560")
    assert not normalized_equal("not-a-number", "560")
