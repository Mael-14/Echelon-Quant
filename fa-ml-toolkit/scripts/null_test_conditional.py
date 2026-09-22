"""Is there anything for a model to find, once the setup has already fired?

ADR-004 measured the *unconditional* null: sample every Nth bar in both
directions and the touch probability matches ``1/(1+b)`` to within 0.005. That
settles unconditional direction prediction. It does **not** settle this, and
the difference is the whole reason this script exists:

    A martingale can contain sub-populations with drift, provided they cancel.

So the question a meta-labelling model actually depends on is narrower. Not
"can you predict the next move" -- ADR-004 says no -- but "among the bars where
the setup is present, does the touch probability still sit on the martingale
line". If it does, XGBoost has nothing to rank and the model should not be
built. If it does not, the deviation is the effect size a model would be
trying to capture.

The conditioning here is the five encoded features in ``strategy/features.py``
and nothing else. No technique book, no ICT, no prior strategy: the point is
to test *this* setup, on its own terms.

**What the verdict has to clear.** Landing above the martingale line is not
enough. Cost is a fixed subtraction, so the bar is the breakeven probability
``(1 + cost) / (1 + b)``, and the gap between those two lines is set entirely
by the execution frame. At H4 it is 0.017/(1+b), under a point, and a test that
cannot resolve a gap that narrow cannot return a meaningful negative -- so the
report prints the minimum detectable effect beside every measurement, and that
column is the one to read first.

At M1 the problem inverts. The gap is 0.212/(1+b), over ten points, so the test
has no trouble resolving it -- but a setup now has to beat the martingale by
ten points to pay for itself, and ADR-004 found deviations of -0.010 to +0.003.
The cost defaults per ``--micro`` for exactly this reason: measuring an M1 run
against H4's hurdle would turn a hopeless cell into a promising one.

Usage:

    python scripts/null_test_conditional.py --history data/deriv_history.json --macro 4h --micro 1m

"""
from __future__ import annotations

import argparse
import bisect
import json
from datetime import datetime, timedelta, timezone
from math import sqrt
from pathlib import Path

from forex_agent.models import Bar, Timeframe
from forex_agent.strategy.barriers import martingale_probability, resolve
from forex_agent.strategy.expectancy import breakeven_probability
from forex_agent.strategy.features import FEATURE_COLUMNS, encode_features

#: Trailing macro bars handed to the encoder. The EMA200 seed decays by
#: ``(1 - 2/201)**600 ~= 0.003``, so 600 reproduces the full-history baseline
#: to three decimals while keeping each call bounded.
MACRO_WINDOW = 600

#: Trailing micro bars. This must cover a whole UTC session at the execution
#: frame or the VWAP anchor is silently truncated: a day is 288 M5 bars, and a
#: 200-bar window would cut the morning off every afternoon reading.
#:
#: Which is why it is a floor rather than the value: a day is 1440 M1 bars, so
#: a fixed 400 would cut off exactly what this comment warns about as soon as
#: the execution frame drops below M5. :func:`micro_window_for` applies it.
MICRO_WINDOW = 400


def micro_window_for(frame: Timeframe) -> int:
    """Micro bars to keep: one whole UTC session, never fewer than 400."""
    return max(MICRO_WINDOW, 86_400 // frame.seconds)


#: Deviations are scanned across many cells, so the single-test 1.96 would
#: manufacture hits. Roughly Bonferroni for ~50 cells at 0.05.
SIGNIFICANCE_Z = 3.3

#: Spread drag in R per execution frame, from ADR-004. This sets the breakeven
#: line every deviation below is measured against, so it cannot be a constant:
#: pinned at H4's 0.017 an M1 run understates its own hurdle by over 12x, which
#: is the difference between "nothing here" and "worth building a model".
#: M1 is extrapolated from M5 by the 1/sqrt(time) law, not measured.
SPREAD_DRAG_R = {
    Timeframe.M1: 0.095 * sqrt(300 / 60),
    Timeframe.M5: 0.095,
    Timeframe.M15: 0.045,
    Timeframe.H4: 0.017,
    Timeframe.D1: 0.007,
}


# --------------------------------------------------------------------- data


def load_bars(symbol: str, candles: list[dict], timeframe: Timeframe) -> list[Bar]:
    span = timedelta(seconds=timeframe.seconds)
    bars: list[Bar] = []
    for candle in candles:
        start = datetime.fromtimestamp(float(candle["epoch"]), tz=timezone.utc)
        try:
            bars.append(
                Bar(
                    symbol, timeframe, start, start + span,
                    float(candle["open"]), float(candle["high"]),
                    float(candle["low"]), float(candle["close"]),
                    int(candle.get("volume", 0) or 0),
                )
            )
        except ValueError:
            continue
    bars.sort(key=lambda bar: bar.start)
    return bars


# ---------------------------------------------------------------- statistics


class Cell:
    """One population being tested against the martingale line."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.wins = 0
        self.losses = 0
        self.timeouts = 0

    def add(self, outcome: bool | None) -> None:
        if outcome is None:
            self.timeouts += 1
        elif outcome:
            self.wins += 1
        else:
            self.losses += 1

    @property
    def resolved(self) -> int:
        return self.wins + self.losses

    @property
    def p(self) -> float:
        return self.wins / self.resolved if self.resolved else 0.0

    def report(self, reward: float, cost: float, overlap: float) -> dict:
        """Measurement, the two lines it sits between, and whether we could tell.

        ``overlap`` discounts the sample count for the fact that consecutive
        trades share bars: a naive standard error on overlapping windows is
        optimistic, and this test exists precisely to avoid fooling itself.
        """
        n = self.resolved
        p0 = martingale_probability(reward)
        target = breakeven_probability(reward, cost)
        n_eff = max(1.0, n / overlap)
        se = sqrt(p0 * (1.0 - p0) / n_eff) if n_eff else 0.0
        return {
            "label": self.label,
            "n": n,
            "n_eff": n_eff,
            "timeouts": self.timeouts,
            "p": self.p,
            "martingale": p0,
            "breakeven": target,
            "deviation": self.p - p0,
            "z": (self.p - p0) / se if se else 0.0,
            # What a deviation would have to be before this many samples could
            # call it real, and what it would have to be to pay for the spread.
            "mde": SIGNIFICANCE_Z * se,
            "needed": target - p0,
        }


def print_table(rows: list[dict], title: str) -> None:
    print()
    print(title)
    print("-" * 108)
    print(
        f"{'cell':<34}{'n':>7}{'n_eff':>8}{'p':>8}{'null':>8}"
        f"{'dev':>9}{'z':>7}{'MDE':>8}{'needed':>8}{'verdict':>12}"
    )
    print("-" * 108)
    for row in rows:
        if row["n"] < 100:
            verdict = "too few"
        elif row["mde"] > row["needed"] * 2:
            verdict = "underpowered"
        elif abs(row["z"]) >= SIGNIFICANCE_Z:
            verdict = "DEVIATES"
        else:
            verdict = "on the null"
        print(
            f"{row['label']:<34}{row['n']:>7}{row['n_eff']:>8.0f}{row['p']:>8.4f}"
            f"{row['martingale']:>8.4f}{row['deviation']:>+9.4f}{row['z']:>+7.2f}"
            f"{row['mde']:>8.4f}{row['needed']:>8.4f}{verdict:>12}"
        )


# --------------------------------------------------------------------- setup


def setup_side(row: dict[str, float]) -> bool | None:
    """The setup, stated in the encoded features alone.

    Long when the macro tide is up, price has pulled back below the swing
    midpoint, and structure is within an ATR beneath. Short is the mirror.
    ``None`` means no setup is present.

    These thresholds are fixed deliberately. Sweeping them until something
    looks significant is the multiple-testing trap this whole script is built
    to avoid, and the sweep belongs to the model, not to the null.
    """
    if (
        row["macro_trend_score"] >= 0.5
        and row["retracement_factor"] <= 0.5
        and row["distance_to_support"] <= 1.0
    ):
        return True
    if (
        row["macro_trend_score"] <= -0.5
        and row["retracement_factor"] >= 0.5
        and row["distance_to_resistance"] <= 1.0
    ):
        return False
    return None


# ----------------------------------------------------------------- collection


def collect(history: dict, macro: Timeframe, micro: Timeframe, args) -> list[dict]:
    """Encode every sampled bar and resolve its barriers in both directions."""
    samples: list[dict] = []
    for symbol, frames in sorted(history.items()):
        if args.symbols and symbol not in args.symbols:
            continue
        micro_window = micro_window_for(micro)
        macro_bars = load_bars(symbol, frames.get(macro.value) or [], macro)
        micro_bars = load_bars(symbol, frames.get(micro.value) or [], micro)
        if len(macro_bars) < MACRO_WINDOW or len(micro_bars) < micro_window:
            print(f"  {symbol}: skipped, {len(macro_bars)}/{MACRO_WINDOW} macro "
                  f"and {len(micro_bars)}/{micro_window} micro bars")
            continue

        macro_ends = [bar.end for bar in macro_bars]
        encoded = skipped = 0
        start = max(micro_window, args.stride)
        for index in range(start, len(micro_bars) - 1, args.stride):
            now = micro_bars[index].end
            # Only macro bars that have already closed. The macro bar
            # *containing* this one is still forming, and its high and low
            # would be tomorrow's information.
            cut = bisect.bisect_right(macro_ends, now)
            if cut < MACRO_WINDOW:
                continue
            features = encode_features(
                macro_bars[max(0, cut - MACRO_WINDOW) : cut],
                micro_bars[max(0, index + 1 - micro_window) : index + 1],
            )
            if features is None:
                skipped += 1
                continue
            row = dict(zip(FEATURE_COLUMNS, features.as_row()))
            atr = _atr_at(micro_bars, index, args.atr_length, micro_window)
            if atr <= 0:
                skipped += 1
                continue
            stop = atr * args.stop_atr
            outcomes = {
                reward: {
                    side: resolve(
                        micro_bars, index, is_long=side,
                        stop_distance=stop, reward_r=reward, max_hold=args.max_hold,
                    )
                    for side in (True, False)
                }
                for reward in args.reward
            }
            samples.append({"symbol": symbol, "features": row, "outcomes": outcomes})
            encoded += 1
        print(f"  {symbol}: {encoded} encoded, {skipped} unreadable")
    return samples


def _atr_at(bars: list[Bar], index: int, length: int, window: int) -> float:
    from forex_agent.strategy.indicators import true_range_atr

    return true_range_atr(bars[max(0, index + 1 - window) : index + 1], length)


# -------------------------------------------------------------------- reports


def quintile_edges(values: list[float], buckets: int) -> list[float]:
    ordered = sorted(values)
    return [ordered[int(len(ordered) * i / buckets)] for i in range(1, buckets)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("data/deriv_history.json"))
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="restrict to these symbols; default is every symbol in the file")
    parser.add_argument("--macro", default="4h", help="structural frame (default 4h)")
    parser.add_argument("--micro", default="1m", help="execution frame (default 1m)")
    parser.add_argument("--stride", type=int, default=12, help="micro bars between samples")
    parser.add_argument("--reward", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--stop-atr", type=float, default=1.5)
    parser.add_argument("--atr-length", type=int, default=14)
    parser.add_argument("--max-hold", type=int, default=120)
    parser.add_argument(
        "--cost", type=float, default=None,
        help="spread drag in R at the execution frame; defaults from the "
             "ADR-004 table (0.017 at H4, 0.095 at M5; 1m is extrapolated)",
    )
    parser.add_argument("--buckets", type=int, default=5)
    args = parser.parse_args()

    macro, micro = Timeframe(args.macro), Timeframe(args.micro)
    if macro.seconds <= micro.seconds:
        raise SystemExit(
            f"--macro ({macro.value}) must be coarser than --micro ({micro.value})"
        )
    if args.cost is None:
        if micro not in SPREAD_DRAG_R:
            raise SystemExit(
                f"no spread drag on record for {micro.value}; pass --cost explicitly"
            )
        args.cost = SPREAD_DRAG_R[micro]
    history = json.loads(args.history.read_text())

    measured = " (EXTRAPOLATED)" if micro is Timeframe.M1 else ""
    print(f"Conditional null test: {macro.value} structure / {micro.value} execution")
    print(f"stop {args.stop_atr}xATR({args.atr_length}), max hold {args.max_hold} bars, "
          f"stride {args.stride}, cost {args.cost:.4f}R{measured}")
    print()
    samples = collect(history, macro, micro, args)
    if not samples:
        print("\nNo samples encoded; nothing to test.")
        return

    # Consecutive trades share bars, so the naive n overstates the evidence.
    overlap = max(1.0, args.max_hold / args.stride)
    print(f"\n{len(samples)} samples; overlap factor {overlap:.1f}x "
          f"(hold {args.max_hold} / stride {args.stride})")

    for reward in args.reward:
        print()
        print("=" * 108)
        print(f"REWARD b = {reward}   martingale {martingale_probability(reward):.4f}"
              f"   breakeven {breakeven_probability(reward, args.cost):.4f}")
        print("=" * 108)

        # A. unconditional, to re-measure ADR-004's finding at this frame.
        rows = []
        for side, name in ((True, "unconditional long"), (False, "unconditional short")):
            cell = Cell(name)
            for sample in samples:
                cell.add(sample["outcomes"][reward][side])
            rows.append(cell.report(reward, args.cost, overlap))
        print_table(rows, "A. Unconditional -- does ADR-004 still hold at this frame?")

        # B. one feature at a time, to see whether any single column carries
        #    information before asking a model to find it in combination.
        for column in FEATURE_COLUMNS:
            values = [s["features"][column] for s in samples]
            edges = quintile_edges(values, args.buckets)
            cells = [Cell(f"{column[:20]} q{i + 1} long") for i in range(args.buckets)]
            for sample in samples:
                slot = bisect.bisect_right(edges, sample["features"][column])
                cells[min(slot, args.buckets - 1)].add(sample["outcomes"][reward][True])
            print_table(
                [cell.report(reward, args.cost, overlap) for cell in cells],
                f"B. {column} in quintiles (long side)",
            )

        # C. the conjunction -- the setup as such.
        taken = Cell("setup, its own direction")
        against = Cell("setup, opposite direction")
        for sample in samples:
            side = setup_side(sample["features"])
            if side is None:
                continue
            taken.add(sample["outcomes"][reward][side])
            against.add(sample["outcomes"][reward][not side])
        print_table(
            [taken.report(reward, args.cost, overlap), against.report(reward, args.cost, overlap)],
            "C. The setup conjunction -- the go/no-go for a model",
        )
        fired = taken.resolved + taken.timeouts
        print(f"\n   setup fired on {fired} of {len(samples)} samples "
              f"({100.0 * fired / len(samples):.1f}%)")

    print()
    print("Read the MDE column before the z column. A cell whose minimum")
    print("detectable effect is larger than the 'needed' edge cannot return a")
    print("meaningful negative -- it is silent, not clean.")


if __name__ == "__main__":
    main()
