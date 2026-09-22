"""Saving and loading a trained model together with its assumptions.

The trainer previously persisted nothing. The fitted ``LogisticRegression`` and
``StandardScaler`` were local variables in ``train_and_report``, garbage
collected on return, and the entire output was stdout text -- so there was no
model to serve and no way to make one.

**Why the metadata travels with the weights.** A model is only meaningful under
the assumptions it was fitted under, and every one of them is invisible at
serving time:

* ``feature_names`` fixes column order. Serving the same columns in a different
  order produces confident, plausible, wrong probabilities and raises nothing.
* ``exec_frame`` and ``macro_frames`` fix the cascade. A model trained on M1
  triggers has learned M1 dynamics; served on H4 it is answering a different
  question with the same numbers.
* ``spread_atr`` fixes the cost the gate was calibrated against. At M1 the drag
  is roughly 0.212R against H4's 0.017R, so a model carried between frames has
  its accept/reject boundary moved by more than ten points of win rate.

:func:`load` therefore returns the metadata beside the estimator, and
:meth:`ModelArtifact.check_serves` gives a caller one call to refuse a
mismatch. Failing closed on that is the whole point -- see
``expectancy.evaluate_edge``, which catches its own bad inputs and returns a
rejecting verdict rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib

#: Bumped when the payload's shape changes incompatibly, so an old artifact is
#: refused with a clear message instead of failing somewhere deeper.
ARTIFACT_VERSION = 1


class ArtifactError(RuntimeError):
    """An artifact cannot be loaded, or cannot serve the requested cascade."""


@dataclass(frozen=True)
class ModelArtifact:
    model: Any
    scaler: Any
    feature_names: tuple[str, ...]
    exec_frame: str
    macro_frames: tuple[str, ...]
    spread_atr: float
    encoding: str
    hold: int
    trained_at: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    version: int = ARTIFACT_VERSION

    @property
    def summary(self) -> str:
        cascade = " -> ".join(self.macro_frames)
        return f"{cascade} -> {self.exec_frame} ({self.encoding}, hold {self.hold})"

    def check_serves(self, *, exec_frame: str, macro_frames: Sequence[str]) -> None:
        """Raise unless this artifact was trained on exactly this cascade.

        Exact rather than compatible: there is no such thing as a near-enough
        frame here. Either the model saw these dynamics in training or it did
        not, and the failure mode of guessing is silent.
        """
        if self.exec_frame != exec_frame:
            raise ArtifactError(
                f"artifact was trained to trigger on {self.exec_frame}, "
                f"but this service is configured for {exec_frame}"
            )
        if tuple(self.macro_frames) != tuple(macro_frames):
            raise ArtifactError(
                f"artifact was trained on macro frames "
                f"{', '.join(self.macro_frames)}, but this service is configured "
                f"for {', '.join(macro_frames)}"
            )

    def predict_proba(self, row: Sequence[float]) -> float:
        """Win probability for one encoded row, standardised the same way.

        Reusing the fitted scaler rather than re-fitting is the point: a scaler
        fitted on live data would standardise against a different distribution
        every time it was called.
        """
        if len(row) != len(self.feature_names):
            raise ArtifactError(
                f"expected {len(self.feature_names)} features "
                f"({', '.join(self.feature_names)}), got {len(row)}"
            )
        scaled = self.scaler.transform([list(row)])
        return float(self.model.predict_proba(scaled)[0][1])


def save(
    path: str | Path,
    *,
    model: Any,
    scaler: Any,
    feature_names: Sequence[str],
    exec_frame: str,
    macro_frames: Sequence[str],
    spread_atr: float,
    encoding: str,
    hold: int,
    metrics: dict[str, float] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": ARTIFACT_VERSION,
        "model": model,
        "scaler": scaler,
        "feature_names": tuple(feature_names),
        "exec_frame": exec_frame,
        "macro_frames": tuple(macro_frames),
        "spread_atr": float(spread_atr),
        "encoding": encoding,
        "hold": int(hold),
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "metrics": dict(metrics or {}),
    }
    joblib.dump(payload, destination)
    return destination


def load(path: str | Path) -> ModelArtifact:
    destination = Path(path)
    if not destination.exists():
        raise ArtifactError(f"no model artifact at {destination}")
    try:
        payload = joblib.load(destination)
    except Exception as exc:
        raise ArtifactError(f"could not read the artifact at {destination}") from exc

    if not isinstance(payload, dict):
        raise ArtifactError(f"{destination} does not hold a model artifact")

    version = payload.get("version")
    if version != ARTIFACT_VERSION:
        raise ArtifactError(
            f"{destination} is artifact version {version!r}, this code reads "
            f"version {ARTIFACT_VERSION}; retrain it"
        )

    missing = [
        key
        for key in ("model", "scaler", "feature_names", "exec_frame", "macro_frames", "spread_atr")
        if key not in payload
    ]
    if missing:
        raise ArtifactError(f"{destination} is missing {', '.join(missing)}")

    return ModelArtifact(
        model=payload["model"],
        scaler=payload["scaler"],
        feature_names=tuple(payload["feature_names"]),
        exec_frame=str(payload["exec_frame"]),
        macro_frames=tuple(payload["macro_frames"]),
        spread_atr=float(payload["spread_atr"]),
        encoding=str(payload.get("encoding", "v2")),
        hold=int(payload.get("hold", 0)),
        trained_at=str(payload.get("trained_at", "")),
        metrics=dict(payload.get("metrics") or {}),
        version=version,
    )
