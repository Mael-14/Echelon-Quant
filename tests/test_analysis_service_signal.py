"""analysis-service: the artifact contract and the fail-closed behaviour.

The two things worth pinning are that an artifact carries the assumptions it was
fitted under, and that a mismatch stops the service rather than producing
plausible wrong numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
from forex_agent import artifact as artifact_module


class FakeScaler:
    def transform(self, rows):
        return np.asarray(rows, dtype=float)


class FakeModel:
    """Returns a fixed probability, so tests assert on plumbing not on fitting."""

    def __init__(self, p: float = 0.8) -> None:
        self.p = p
        self.seen: list[list[float]] = []

    def predict_proba(self, rows):
        self.seen.extend(np.asarray(rows, dtype=float).tolist())
        return np.asarray([[1.0 - self.p, self.p] for _ in rows])


def _save(tmp_path, **overrides):
    payload = {
        "model": FakeModel(),
        "scaler": FakeScaler(),
        "feature_names": ["h4_trend", "h4_stop_prox", "pullback"],
        "exec_frame": "1m",
        "macro_frames": ["4h"],
        "spread_atr": 0.3186,
        "encoding": "v2",
        "hold": 24,
    }
    payload.update(overrides)
    path = tmp_path / "model.joblib"
    return artifact_module.save(path, **payload)


# ------------------------------------------------------------------- artifact


def test_an_artifact_round_trips_with_its_assumptions(tmp_path):
    loaded = artifact_module.load(_save(tmp_path))

    assert loaded.exec_frame == "1m"
    assert loaded.macro_frames == ("4h",)
    assert loaded.spread_atr == pytest.approx(0.3186)
    assert loaded.feature_names == ("h4_trend", "h4_stop_prox", "pullback")
    assert loaded.trained_at  # stamped at save time


def test_the_summary_names_the_cascade(tmp_path):
    assert artifact_module.load(_save(tmp_path)).summary == "4h -> 1m (v2, hold 24)"


def test_a_missing_artifact_says_so(tmp_path):
    with pytest.raises(artifact_module.ArtifactError, match="no model artifact"):
        artifact_module.load(tmp_path / "absent.joblib")


def test_a_wrong_version_is_refused_rather_than_read(tmp_path):
    """An old artifact must fail here, not somewhere deeper and stranger."""
    import joblib

    path = tmp_path / "old.joblib"
    joblib.dump({"version": 0, "model": None}, path)

    with pytest.raises(artifact_module.ArtifactError, match="version"):
        artifact_module.load(path)


def test_an_incomplete_artifact_names_what_is_missing(tmp_path):
    import joblib

    path = tmp_path / "partial.joblib"
    joblib.dump({"version": artifact_module.ARTIFACT_VERSION, "model": object()}, path)

    with pytest.raises(artifact_module.ArtifactError, match="scaler"):
        artifact_module.load(path)


# -------------------------------------------------------------- fail closed


def test_serving_the_trained_cascade_is_allowed(tmp_path):
    artifact_module.load(_save(tmp_path)).check_serves(exec_frame="1m", macro_frames=["4h"])


def test_a_different_execution_frame_is_refused(tmp_path):
    """A model trained on M1 triggers has learned M1 dynamics.

    Served at H4 it answers a different question with the same numbers, and
    nothing downstream can detect that.
    """
    loaded = artifact_module.load(_save(tmp_path))

    with pytest.raises(artifact_module.ArtifactError, match="trigger on 1m"):
        loaded.check_serves(exec_frame="4h", macro_frames=["4h"])


def test_different_macro_frames_are_refused(tmp_path):
    loaded = artifact_module.load(_save(tmp_path))

    with pytest.raises(artifact_module.ArtifactError, match="macro frames"):
        loaded.check_serves(exec_frame="1m", macro_frames=["1d", "4h"])


def test_the_engine_refuses_a_mismatch_at_construction(tmp_path):
    """Caught when the service starts, not at the first signal hours later."""
    import importlib
    import sys
    from pathlib import Path

    service = Path(__file__).resolve().parents[1] / "backend" / "services" / "analysis-service"
    sys.path.insert(0, str(service))
    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]
    engine_module = importlib.import_module("app.engine")

    with pytest.raises(artifact_module.ArtifactError):
        engine_module.AnalysisEngine(
            artifact=artifact_module.load(_save(tmp_path)),
            exec_frame="4h",
            macro_frames=["1d"],
        )


# -------------------------------------------------------------- probability


def test_prediction_uses_the_fitted_scaler(tmp_path):
    """Re-fitting on live data would standardise against a moving distribution."""
    loaded = artifact_module.load(_save(tmp_path))

    p = loaded.predict_proba([0.5, 0.25, 0.1])

    assert p == pytest.approx(0.8)
    assert loaded.model.seen[-1] == [0.5, 0.25, 0.1]


def test_the_wrong_column_count_raises_rather_than_scoring(tmp_path):
    """Silently scoring a short row is how a cascade mismatch stays invisible."""
    loaded = artifact_module.load(_save(tmp_path))

    with pytest.raises(artifact_module.ArtifactError, match="expected 3 features"):
        loaded.predict_proba([0.5, 0.25])
