"""다음 분기 연구 후보의 성장률 단위와 선택 기준을 검증한다."""

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_seoul_next_quarter_tail as tail


def test_gentle_transform_preserves_extremes_and_round_trips():
    targets = np.array([-1.0, -0.01, 0.0, 0.01, 100.0])
    transformed = tail.gentle_transform(targets)
    assert np.arcsinh(targets[-1]) < transformed[-1] < targets[-1]
    assert tail.gentle_inverse(transformed) == pytest.approx(targets)
    tail_transformed = tail.tail_transform(targets)
    assert tail_transformed[:-1] == pytest.approx(targets[:-1])
    assert tail_transformed[-1] < targets[-1]
    assert tail.tail_inverse(tail_transformed) == pytest.approx(targets)


def test_selection_rejects_lower_mean_with_worse_p90_or_extreme_count():
    selection = pd.DataFrame(
        {
            "target": np.zeros(40),
            tail.RAW: [0.2] * 39 + [1.05],
            tail.ASINH: [0.0] * 35 + [0.35] * 5,
            tail.BLEND: [0.1] * 38 + [1.1] * 2,
            tail.GENTLE: [0.16] * 40,
            tail.TAIL_ONLY: [0.3] * 40,
            **{name: [0.3] * 40 for name in tail.LOW_SALES_CUTOFFS},
        }
    )
    assert tail.choose_candidate(selection) == tail.GENTLE
    selection[tail.GENTLE] = 0.3
    assert tail.choose_candidate(selection) == tail.RAW


def test_district_reference_uses_exact_mapping_and_refuses_wrong_date(tmp_path):
    mapping = tmp_path / "mapping.csv"
    mapping.write_text(
        "reference_date,sgis_code,h10_code,seoul_code\n2025-06-30,11111111,1111234500,11119999\n",
        encoding="utf-8",
    )
    areas = tmp_path / "areas.csv"
    areas.write_text("reference_date,sgis_code,district_name\n2025-06-30,11111111,종로구\n")
    assert tail.read_districts(mapping, areas) == {"11119999": "종로구"}
    areas.write_text("reference_date,sgis_code,district_name\n2026-06-30,11111111,종로구\n")
    with pytest.raises(ValueError, match="2025-06-30"):
        tail.read_districts(mapping, areas)
