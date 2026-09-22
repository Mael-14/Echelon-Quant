"""Download Deriv candle history for training and gate replay.

Deriv caps a single ``ticks_history`` response (roughly 1000 candles for the
forex symbols), so history is walked backwards page by page using the ``end``
parameter until the requested depth is reached or the broker stops returning
new candles.

Each timeframe is requested at its own granularity rather than downsampled
from M5.  Downsampling is what previously reduced a forex pair to ten 4H rows:
1000 five-minute candles only span about three days, which is ten 4H buckets.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
import websockets  # noqa: E402

from forex_agent.config import settings  # noqa: E402

PUBLIC_WS = "wss://api.derivws.com/trading/v1/options/ws/public"
DEFAULT_SYMBOLS = ("frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "R_75", "1HZ75V")
GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "4h": 14_400, "1d": 86_400}
PAGE = 1000


async def _socket_url() -> str:
    """Prefer the authenticated socket when credentials are configured."""
    token, app_id, account = settings.deriv_token, settings.deriv_app_id, settings.deriv_account_id
    if not token or "YOUR_" in token or not app_id or not account:
        return PUBLIC_WS
    url = f"https://api.derivws.com/trading/v1/options/accounts/{account}/otp"
    headers = {"Authorization": f"Bearer {token}", "Deriv-App-ID": app_id}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(url, headers=headers)
        response.raise_for_status()
        return str(response.json()["data"]["url"])


async def _request(ws, payload: dict) -> dict:
    await ws.send(json.dumps(payload))
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=40))
        if message.get("error"):
            raise RuntimeError(message["error"].get("message", message["error"]))
        if message.get("msg_type") == "candles":
            return message


async def fetch_frame(ws, symbol: str, frame: str, target: int) -> list[dict]:
    """Walk ``symbol`` history backwards until ``target`` candles are collected.

    Termination is decided by *backward progress*, not by how many rows were
    added.  Deriv honours ``end`` for the intraday granularities but ignores it
    for some symbols on the daily frame, answering every request with the same
    rolling window shifted by a second or two.  Each such page contributes a
    couple of unseen epochs, so a "did the set grow" test never trips and the
    loop walks hundreds of pages without reaching further back.  Requiring each
    page to start strictly earlier than the last stops immediately instead.
    """
    granularity = GRANULARITY[frame]
    collected: dict[int, dict] = {}
    end: object = "latest"
    oldest_seen: int | None = None
    while len(collected) < target:
        try:
            response = await _request(ws, {
                "ticks_history": symbol, "count": PAGE, "end": end,
                "style": "candles", "granularity": granularity, "req_id": 1,
            })
        except (RuntimeError, asyncio.TimeoutError):
            break
        candles = response.get("candles") or []
        if not candles:
            break
        for candle in candles:
            try:
                epoch = int(float(candle["epoch"]))
                collected[epoch] = {
                    "epoch": epoch,
                    "open": float(candle["open"]), "high": float(candle["high"]),
                    "low": float(candle["low"]), "close": float(candle["close"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
        oldest = int(float(candles[0]["epoch"]))
        if oldest_seen is not None and oldest >= oldest_seen:
            break  # The broker stopped honouring `end`; no deeper history.
        oldest_seen = oldest
        end = oldest - 1
    return [collected[key] for key in sorted(collected)][-target:]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/deriv_training.csv"))
    parser.add_argument("--raw", type=Path, default=None, help="also write full OHLC JSON")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--frames", nargs="+", default=list(GRANULARITY))
    parser.add_argument("--count", type=int, default=20_000, help="target candles per frame")
    args = parser.parse_args()

    url = await _socket_url()
    print(f"connected via {'authenticated' if '/ws/demo' in url else 'public'} socket", flush=True)

    history: dict[str, dict[str, list[dict]]] = {}
    rows: list[tuple[str, str, float]] = []
    async with websockets.connect(url, open_timeout=25, max_size=None) as ws:
        for symbol in args.symbols:
            history[symbol] = {}
            summary = []
            for frame in args.frames:
                target = args.count if frame == "5m" else max(500, args.count // 4)
                candles = await fetch_frame(ws, symbol, frame, target)
                history[symbol][frame] = candles
                print(f"    {symbol} {frame}: {len(candles)}", flush=True)
                rows.extend((symbol, frame, c["close"]) for c in candles)
                if candles:
                    span = (candles[-1]["epoch"] - candles[0]["epoch"]) / 86_400
                    oldest = datetime.fromtimestamp(candles[0]["epoch"], tz=timezone.utc)
                    summary.append(f"{frame}={len(candles)} ({span:.0f}d, from {oldest:%Y-%m-%d})")
                else:
                    summary.append(f"{frame}=0")
            print(f"  {symbol:<12} " + "  ".join(summary), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("symbol", "timeframe", "close"))
        writer.writerows(rows)
    print(f"saved {args.output} ({len(rows)} rows)")

    if args.raw:
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        args.raw.write_text(json.dumps(history), encoding="utf-8")
        print(f"saved {args.raw}")


if __name__ == "__main__":
    asyncio.run(main())
