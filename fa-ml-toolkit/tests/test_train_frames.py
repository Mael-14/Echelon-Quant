"""The cascade is a flag now, so the column contract has to be pinned.

``scripts/`` is not an installed package -- these are entry points, not library
code -- so the module under test is loaded by path.
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from forex_agent.models import Timeframe


def _load(name: str):
    """Import a script by path, under a namespaced key in ``sys.modules``.

    Registering it is not optional: ``@dataclass(slots=True)`` resolves its
    annotations through ``sys.modules[cls.__module__]``, and an unregistered
    module makes that lookup return ``None``. The prefix keeps these entries
    from colliding with any real package of the same name.
    """
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    key = f"fa_ml_toolkit_scripts.{name}"
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


trainer = _load("train_scalp_model")
nulltest = _load("null_test_conditional")
backtest = _load("backtest_scalp")


def test_v2_names_reproduce_the_frozen_tuple():
    """The generalisation must be a no-op for the cascade that was hardcoded.

    A model is persisted with its feature names; if this ordering drifts, a
    saved artifact is silently served columns in the wrong order.
    """
    assert trainer.feature_names_v2((Timeframe.D1, Timeframe.H4)) == trainer.FEATURES_V2


def test_v1_names_reproduce_the_frozen_tuple():
    assert trainer.feature_names_v1((Timeframe.D1, Timeframe.H4)) == trainer.FEATURES


def test_h4_over_m1_is_the_single_macro_cascade():
    """H4 analysis into an M1 trigger: seven columns, not twelve."""
    names = trainer.feature_names_v2((Timeframe.H4,))
    assert names == (
        "h4_trend",
        "h4_stop_prox",
        "h4_tgt_prox",
        "h4_stop_dist",
        "h4_tgt_dist",
        "pullback",
        "momentum",
    )


@pytest.mark.parametrize("count", [1, 2, 3])
def test_column_count_follows_the_frame_count(count):
    frames = (Timeframe.W1, Timeframe.D1, Timeframe.H4)[:count]
    # Five columns per macro frame (trend, two proximities, two distances),
    # plus pullback and momentum from the coarsest frame.
    assert len(trainer.feature_names_v2(frames)) == 5 * count + 2
    assert len(trainer.feature_names_v1(frames)) == 3 * count + 2


def test_encode_row_v2_matches_its_own_names():
    """Row width and name width cannot disagree, whatever the cascade."""

    class _Fake:
        macro_trend_score = 0.5
        distance_to_support = 1.0
        distance_to_resistance = 2.0
        retracement_factor = 0.25
        micro_wave_momentum = 0.75

    for frames in ((Timeframe.H4,), (Timeframe.D1, Timeframe.H4)):
        macro = [_Fake() for _ in frames]
        for is_long in (True, False):
            row = trainer.encode_row_v2(macro, is_long)
            assert len(row) == len(trainer.feature_names_v2(frames))


def test_short_mirrors_the_long_row():
    """Direction-relative encoding: a short must not reuse a long's geometry."""

    class _Fake:
        macro_trend_score = 0.5
        distance_to_support = 1.0
        distance_to_resistance = 2.0
        retracement_factor = 0.25
        micro_wave_momentum = 0.75

    names = trainer.feature_names_v2((Timeframe.H4,))
    long_row = trainer.encode_row_v2([_Fake()], True)
    short_row = trainer.encode_row_v2([_Fake()], False)

    trend = names.index("h4_trend")
    assert long_row[trend] == pytest.approx(-short_row[trend])
    # Support is the stop for a long and the target for a short.
    assert long_row[names.index("h4_stop_dist")] == short_row[names.index("h4_tgt_dist")]


def test_m1_drag_is_the_largest_on_the_curve():
    """Cost in R falls as 1/sqrt(time); M1 must not look cheap by accident."""
    drag = trainer.SPREAD_DRAG_R
    ordered = sorted(drag, key=lambda f: f.seconds)
    values = [drag[f] for f in ordered]
    assert values == sorted(values, reverse=True)
    assert drag[Timeframe.M1] > drag[Timeframe.M5] > drag[Timeframe.H4]


def test_default_spread_is_the_drag_against_a_1_5_atr_stop():
    expected = trainer.SPREAD_DRAG_R[Timeframe.M1] * trainer.STOP_ATR_MULTIPLE
    assert trainer.default_spread_atr(Timeframe.M1) == pytest.approx(expected)
    # W1 has no entry in the table, and a silent 0.0 cost would be a free lunch.
    assert trainer.default_spread_atr(Timeframe.W1) == 0.0


@pytest.mark.parametrize("module", [trainer, nulltest, backtest], ids=["train", "null", "backtest"])
def test_every_script_agrees_on_the_cost_curve(module):
    """Three scripts score the same trades; a disagreement here is silent drift."""
    assert module.SPREAD_DRAG_R == trainer.SPREAD_DRAG_R


def test_exec_window_covers_a_utc_session():
    """`momentum` is distance from session VWAP, which re-anchors if truncated.

    A UTC day is 1440 M1 bars and 288 M5 bars, so the historical 400-bar default
    already covered a session at M5 and coarser -- those must not move -- while
    M1 needs more than three times it.
    """
    assert trainer.default_exec_window(Timeframe.M1) == 1440
    assert nulltest.micro_window_for(Timeframe.M1) == 1440
    for frame in (Timeframe.M5, Timeframe.M15, Timeframe.H4, Timeframe.D1):
        assert trainer.default_exec_window(frame) == trainer.EXEC_WINDOW
        assert nulltest.micro_window_for(frame) == nulltest.MICRO_WINDOW


def _separable_with_noise_columns():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 6))
    x[:, 3:] = rng.normal(size=(400, 3)) * 0.01     # three near-constant passengers
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    return x, y


def test_the_classifier_is_actually_lasso():
    """The bug this replaces: it reported "Lasso" while fitting ridge.

    L1 drives a useless column's weight to exactly zero; L2 only shrinks it
    towards zero and reaches it essentially never. Asserting on exact zeros is
    what distinguishes the two, and it is what the ZEROED verdict depends on.
    """
    x, y = _separable_with_noise_columns()
    fitted = trainer.lasso_logistic(0.1).fit(x, y)
    assert (fitted.coef_[0] == 0.0).any()


def test_lasso_spelling_matches_the_installed_sklearn():
    """Neither spelling may warn: on 1.10 the deprecated one stops working."""
    x, y = _separable_with_noise_columns()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trainer.lasso_logistic(0.1).fit(x, y)
