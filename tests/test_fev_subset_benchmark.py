from pathlib import Path

import pytest

from fev_subset_benchmark import DEFAULT_TASKS, metric_comparison, select_tasks, sha256


def test_selected_tasks_are_multivariate() -> None:
    tasks = select_tasks(DEFAULT_TASKS)

    assert [task.task_name for task in tasks] == list(DEFAULT_TASKS)
    assert all(len(task.target_columns) > 1 for task in tasks)


def test_unknown_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown FEV tasks"):
        select_tasks(("not-a-real-task",))


def test_metric_comparison() -> None:
    local = {"SQL": 1.1, "MASE": 2.2, "WAPE": 3.3, "WQL": 4.4}
    official = {"SQL": 1.0, "MASE": 2.0, "WAPE": 3.0, "WQL": 4.0}

    result = metric_comparison(local, official)

    assert result["SQL"]["absolute_delta"] == pytest.approx(0.1)
    assert result["SQL"]["relative_delta_pct"] == pytest.approx(10.0)


def test_sha256(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("abc", encoding="utf-8")

    assert sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
