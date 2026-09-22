"""Prove real Deriv demo credentials work, before anything is built on them.

This is the first thing to run after putting DERIV_APP_ID and DERIV_TOKEN in
``.env``. It answers, in order, the four questions that every later phase
assumes and none of them check:

1. Does the app id open a v3 socket at all?
2. Does the token authorise, and **is the account actually a demo one**? A
   real-money token authorises just as happily, and the loginid is the only
   thing that tells them apart -- demo accounts start with ``VRTC``.
3. Do candles come back on the frame the strategy runs on?
4. Are multiplier contracts offered on this symbol? They are not available on
   every instrument in every jurisdiction, and finding that out here is much
   cheaper than finding it out from a rejected order in Phase 5.

Nothing is bought and no order is placed: every call here is a read.

Usage:

    python scripts/deriv_smoke.py --symbol frxEURUSD --frame 1m --count 5
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.shared.config import get_settings  # noqa: E402
from backend.shared.deriv_v3_client import DerivV3Client, DerivV3Error  # noqa: E402

#: Wire value -> Deriv granularity in seconds. Deriv serves each of these
#: natively, so nothing here is resampled.
GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14_400, "1d": 86_400}


async def run(args) -> int:
    settings = get_settings()

    if settings.deriv_app_id is None:
        print("DERIV_APP_ID is not set. Register an app at https://api.deriv.com")
        return 2
    if not settings.deriv_token:
        print("DERIV_TOKEN is not set. app.deriv.com -> Settings -> Security -> API token")
        return 2

    print(f"app_id {settings.deriv_app_id} | trade mode {settings.deriv_trade_mode}")
    client = DerivV3Client(app_id=settings.deriv_app_id, ws_url=settings.deriv_v3_ws_url)

    try:
        async with client:
            print(f"1. connected to {client.url}")

            account = await client.authorize(settings.deriv_token)
            loginid = str(account.get("loginid", ""))
            currency = account.get("currency")
            is_demo = loginid.upper().startswith("VRTC")
            print(f"2. authorised as {loginid} ({currency}) -- "
                  f"{'DEMO' if is_demo else 'REAL MONEY'}")
            if not is_demo:
                print("   This is a real-money account. Stop here and issue a token with")
                print("   the demo (VRTC...) account selected before going further.")
                return 1
            if settings.deriv_trade_mode != "demo":
                print(f"   DERIV_TRADE_MODE is {settings.deriv_trade_mode!r}, not 'demo'.")

            balance = await client.balance()
            print(f"   balance {balance.get('balance')} {balance.get('currency')}")

            granularity = GRANULARITY[args.frame]
            candles = await client.ticks_history(
                args.symbol, granularity=granularity, count=args.count
            )
            print(f"3. {len(candles)} {args.frame} candles for {args.symbol}")
            for candle in candles[-args.count:]:
                print(f"   {candle['epoch']}  O {candle['open']}  H {candle['high']}"
                      f"  L {candle['low']}  C {candle['close']}")
            if not candles:
                print("   No candles came back. Check the symbol is spelled as Deriv spells it.")
                return 1

            offered = await client.contracts_for(args.symbol, currency=currency)
            types = sorted({
                str(entry.get("contract_type"))
                for entry in (offered.get("available") or [])
            })
            multipliers = [t for t in types if t.startswith("MULT")]
            print(f"4. contract types on {args.symbol}: {len(types)}")
            if multipliers:
                print(f"   multipliers available: {', '.join(multipliers)}")
            else:
                print("   NO multiplier contracts on this symbol. Phase 5 cannot trade it;")
                print("   pick a fallback symbol before building the execution path.")
                return 1

        print("\nAll four checks passed.")
        return 0

    except DerivV3Error as exc:
        print(f"\nFailed: {exc}")
        if exc.code == "InvalidToken":
            print("The token was rejected. Re-issue it with read + trade scopes.")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="frxEURUSD")
    parser.add_argument("--frame", default="1m", choices=sorted(GRANULARITY))
    parser.add_argument("--count", type=int, default=5)
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
