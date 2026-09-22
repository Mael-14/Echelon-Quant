"""Which feature *combinations* pay, and can XGBoost find them by itself?

Two questions, one script.

**First**, an exhaustive conjunction search. Momentum and trend are not two
separate half-signals that add up; a pullback is only a buying opportunity if
the tide is with it and there is structure beneath it. A linear model cannot
represent that -- it can weight features but never multiply them -- so the
search enumerates every one-, two- and three-way conjunction of quantile
predicates over the encoded features and scores each by realised R.

Selection happens on the training split and is reported on the held-out split.
Roughly 36,000 conjunctions get tested, so the best one *on the training split
alone* is guaranteed to look good whether or not it means anything. The test
column is the honest number, and the gap between the two is the size of the
lie that the search would have told.

**Second**, XGBoost on the same rows. Every path through a decision tree is a
conjunction, so a boosted ensemble is the mechanical version of the search
above -- it finds interactions instead of being told them. If the search finds
a combination the model cannot, the model is underfit; if the model beats
every hand-built conjunction, the interaction is finer than three terms.

Everything is scored in realised R after cost, never accuracy: a filter that
lifts the win rate while raising the cost hurdle by more has improved nothing.

Usage:

    python scripts/search_combinations.py --cache <dataset.npz> --hold 24
"""
from __future__ import annotations

import argparse
from itertools import combinations
from math import sqrt
from pathlib import Path

import numpy as np

from forex_agent.models import Timeframe
from forex_agent.strategy.expectancy import breakeven_probability

#: Quantile cut points, computed on the training split only.
QUANTILES = (15, 30, 70, 85)


def load(cache: Path, hold: int) -> dict:
    blob = np.load(cache, allow_pickle=False)
    key = f"y_{hold}"
    if key not in blob.files:
        raise SystemExit(f"{cache} has no hold {hold}; holds present: "
                         f"{[f[2:] for f in blob.files if f.startswith('y_')]}")
    return {
        "X": blob["X"],
        "y": blob[key],
        "reward": blob["reward"],
        "cost": blob["cost"],
        "stamp": blob["stamp"],
        "names": [str(n) for n in blob["names"]],
    }


def realised_r(data: dict) -> np.ndarray:
    """R per sample: a win pays the reward less cost, a loss pays 1 plus cost."""
    win = data["reward"] - data["cost"]
    loss = -(1.0 + data["cost"])
    return np.where(data["y"] == 1, win, loss)


def build_predicates(X: np.ndarray, names: list[str], train: np.ndarray):
    """Boolean masks over every row, with thresholds taken from train only."""
    masks, labels, owners = [], [], []
    for index, name in enumerate(names):
        column = X[:, index]
        q15, q30, q70, q85 = np.percentile(column[train], QUANTILES)
        for mask, label in (
            (column <= q15, f"{name}<=q15"),
            (column <= q30, f"{name}<=q30"),
            (column >= q70, f"{name}>=q70"),
            (column >= q85, f"{name}>=q85"),
            ((column > q30) & (column < q70), f"{name}~mid"),
        ):
            masks.append(mask)
            labels.append(label)
            owners.append(index)
    return np.asarray(masks), labels, owners


def search(masks, labels, owners, r_values, train, test, args):
    """Enumerate conjunctions, score on train, carry the survivors to test."""
    n_pred = len(labels)
    train_mask = np.zeros(len(r_values), dtype=bool)
    train_mask[train] = True
    test_mask = np.zeros(len(r_values), dtype=bool)
    test_mask[test] = True

    def score(selector: np.ndarray, where: np.ndarray) -> tuple[int, float]:
        both = selector & where
        count = int(both.sum())
        if count == 0:
            return 0, 0.0
        return count, float(r_values[both].mean())

    found: list[tuple[float, int, str, np.ndarray]] = []
    pairs_kept: list[tuple[np.ndarray, str, tuple[int, ...]]] = []

    for i in range(n_pred):
        count, mean = score(masks[i], train_mask)
        if count >= args.min_support:
            found.append((mean, count, labels[i], masks[i]))

    for i, j in combinations(range(n_pred), 2):
        if owners[i] == owners[j]:
            continue
        joint = masks[i] & masks[j]
        count, mean = score(joint, train_mask)
        if count < args.min_support:
            continue
        found.append((mean, count, f"{labels[i]} & {labels[j]}", joint))
        pairs_kept.append((joint, f"{labels[i]} & {labels[j]}", (i, j)))

    for joint, label, (i, j) in pairs_kept:
        for k in range(j + 1, n_pred):
            if owners[k] in (owners[i], owners[j]):
                continue
            triple = joint & masks[k]
            count, mean = score(triple, train_mask)
            if count < args.min_support:
                continue
            found.append((mean, count, f"{label} & {labels[k]}", triple))

    print(f"  {len(found)} conjunctions met the {args.min_support}-sample floor")
    found.sort(key=lambda row: -row[0])
    return [
        {
            "label": label,
            "train_n": count,
            "train_r": mean,
            "test_n": score(selector, test_mask)[0],
            "test_r": score(selector, test_mask)[1],
        }
        for mean, count, label, selector in found[: args.top]
    ]


def gate_rows(p_win, data, index, thresholds) -> list[dict]:
    reward, cost = data["reward"][index], data["cost"][index]
    r_values = realised_r(data)[index]
    need = np.array([breakeven_probability(r, c) for r, c in zip(reward, cost)])
    out = []
    for name, taken in thresholds(p_win, need):
        count = int(taken.sum())
        out.append(
            {
                "label": name,
                "n": count,
                "win": float((data["y"][index][taken] == 1).mean()) if count else 0.0,
                "total_r": float(r_values[taken].sum()) if count else 0.0,
                "mean_r": float(r_values[taken].mean()) if count else 0.0,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--hold", type=int, default=24,
                        help="time stop, in --exec-frame bars")
    parser.add_argument("--exec-frame", type=Timeframe, default=Timeframe.M1,
                        choices=list(Timeframe), metavar="{1m,5m,15m,4h,1d,1w}",
                        help="frame the cached dataset was built on; sets the "
                             "purge horizon (default 1m)")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--min-support", type=int, default=250)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    data = load(args.cache, args.hold)
    r_values = realised_r(data)
    stamps = data["stamp"]
    cut = stamps[int(len(stamps) * args.train_fraction)]
    train = np.where(stamps + args.hold * args.exec_frame.seconds < cut)[0]
    test = np.where(stamps >= cut)[0]

    print(f"hold {args.hold} bars | {len(data['y'])} rows "
          f"| train {len(train)} test {len(test)}")
    print(f"base rate {data['y'].mean():.4f} | mean R taking everything "
          f"{r_values[test].mean():+.4f}\n")

    print("CONJUNCTION SEARCH")
    masks, labels, owners = build_predicates(data["X"], data["names"], train)
    print(f"  {len(labels)} predicates over {len(data['names'])} features")
    results = search(masks, labels, owners, r_values, train, test, args)

    print()
    print(f"  {'conjunction':<56}{'trainN':>8}{'trainR':>9}{'testN':>7}{'testR':>9}")
    print("  " + "-" * 89)
    for row in results:
        print(f"  {row['label']:<56}{row['train_n']:>8}{row['train_r']:>+9.4f}"
              f"{row['test_n']:>7}{row['test_r']:>+9.4f}")

    survivors = [r for r in results if r["test_r"] > 0 and r["test_n"] >= 50]
    print(f"\n  {len(survivors)} of {len(results)} stayed positive out of sample")

    print("\nXGBOOST ON THE SAME ROWS")
    from xgboost import XGBClassifier

    x_train, x_test = data["X"][train], data["X"][test]
    y_train, y_test = data["y"][train], data["y"][test]
    inner = int(len(train) * 0.85)
    model = XGBClassifier(
        max_depth=4, n_estimators=400, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, min_child_weight=20,
        eval_metric="logloss", early_stopping_rounds=30, tree_method="hist",
    )
    model.fit(
        x_train[:inner], y_train[:inner],
        eval_set=[(x_train[inner:], y_train[inner:])], verbose=False,
    )
    p_test = model.predict_proba(x_test)[:, 1]
    print(f"  trees used {model.best_iteration + 1}")
    print("  gain importance:")
    for name, gain in sorted(
        zip(data["names"], model.feature_importances_), key=lambda item: -item[1]
    )[:6]:
        print(f"    {name:<16}{gain:>8.4f}")

    def thresholds(p, need):
        yield "cost gate (p > p*)", p > need
        for pct in (90, 95, 99):
            yield f"top {100 - pct}% by p_win", p >= np.percentile(p, pct)

    print(f"\n  {'selection':<28}{'trades':>8}{'win rate':>10}{'total R':>11}{'R/trade':>10}")
    print("  " + "-" * 67)
    everything = {"label": "take every setup", "n": len(test),
                  "win": float((y_test == 1).mean()),
                  "total_r": float(r_values[test].sum()),
                  "mean_r": float(r_values[test].mean())}
    for row in [everything] + gate_rows(p_test, data, test, thresholds):
        print(f"  {row['label']:<28}{row['n']:>8}{row['win']:>10.4f}"
              f"{row['total_r']:>11.1f}{row['mean_r']:>10.4f}")

    if survivors:
        best = survivors[0]
        print(f"\n  best surviving conjunction: {best['label']}")
        print(f"    test n {best['test_n']}  R/trade {best['test_r']:+.4f}")
        se = r_values[test].std() / sqrt(max(1, best["test_n"]))
        print(f"    standard error {se:.4f} -> t = {best['test_r'] / se:+.2f}")


if __name__ == "__main__":
    main()
