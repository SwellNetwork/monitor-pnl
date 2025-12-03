#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


DERIVE_BASE_URL = os.environ.get("DERIVE_BASE_URL", "https://api.lyra.finance").rstrip("/")


class DeriveAPIError(RuntimeError):
    pass


def _post(path: str, payload: Dict[str, Any], timeout: float = 15.0) -> Dict[str, Any]:
    """
    Derive REST RPC endpoints are POST endpoints like:
      /public/get_instruments
      /public/get_ticker
    Responses include { id, result, ... } or { error, ... }.

    Docs:
      - public/get_instruments :contentReference[oaicite:3]{index=3}
      - public/get_ticker :contentReference[oaicite:4]{index=4}
    """
    url = f"{DERIVE_BASE_URL}{path}"
    r = requests.post(url, json=payload, timeout=timeout)
    try:
        data = r.json()
    except Exception as e:
        raise DeriveAPIError(f"Non-JSON response from {url}: HTTP {r.status_code} {r.text[:300]}") from e

    if "error" in data and data["error"]:
        raise DeriveAPIError(f"Derive error from {path}: {data['error']}")
    if "result" not in data:
        raise DeriveAPIError(f"Missing 'result' in response from {path}: keys={list(data.keys())}")

    return data


def get_instruments(currency: str, instrument_type: str, expired: bool = False) -> List[Dict[str, Any]]:
    payload = {"currency": currency, "instrument_type": instrument_type, "expired": expired}
    data = _post("/public/get_instruments", payload)
    return data["result"]


def get_ticker(instrument_name: str) -> Dict[str, Any]:
    payload = {"instrument_name": instrument_name}
    data = _post("/public/get_ticker", payload)
    return data["result"]


@dataclass(frozen=True)
class OptionInstrument:
    instrument_name: str
    expiry_sec: int
    strike: float
    option_type: str  # "C" or "P"


def _now_sec() -> int:
    return int(time.time())


def _pick_perp_name(perps: List[Dict[str, Any]]) -> str:
    # Heuristic: pick the first instrument containing "PERP" (common naming),
    # else fallback to the first perp instrument.
    names = [p.get("instrument_name", "") for p in perps]
    for n in names:
        if "PERP" in n.upper():
            return n
    if names and names[0]:
        return names[0]
    raise DeriveAPIError("No perp instrument_name found for this currency.")


def _parse_options(raw_opts: List[Dict[str, Any]]) -> List[OptionInstrument]:
    out: List[OptionInstrument] = []
    for it in raw_opts:
        od = it.get("option_details") or {}
        name = it.get("instrument_name")
        if not name or not od:
            continue
        try:
            expiry = int(od["expiry"])
            strike = float(od["strike"])
            opt_type = str(od["option_type"])
        except Exception:
            continue
        out.append(OptionInstrument(name, expiry, strike, opt_type))
    return out


def _closest_expiry(options: List[OptionInstrument], target_expiry_sec: int) -> int:
    expiries = sorted({o.expiry_sec for o in options})
    if not expiries:
        raise DeriveAPIError("No option expiries available.")
    return min(expiries, key=lambda e: abs(e - target_expiry_sec))


def _closest_strike(options: List[OptionInstrument], expiry_sec: int, spot: float) -> float:
    strikes = sorted({o.strike for o in options if o.expiry_sec == expiry_sec})
    if not strikes:
        raise DeriveAPIError("No strikes found for chosen expiry.")
    return min(strikes, key=lambda k: abs(k - spot))


def _pick_atm_option_name(
    currency: str,
    target_days: int = 30,
    prefer_call: bool = True,
) -> Tuple[str, float, int, float]:
    """
    Returns:
      (instrument_name, spot_price, expiry_sec, strike)

    Strategy:
      1) Get perps, pick a perp instrument, read index_price as spot proxy.
      2) Get options, choose expiry closest to now + target_days.
      3) Choose strike closest to spot.
      4) Pick call or put at that strike.
    """
    perps = get_instruments(currency=currency, instrument_type="perp", expired=False)
    perp_name = _pick_perp_name(perps)
    perp_ticker = get_ticker(perp_name)
    spot = float(perp_ticker["index_price"])

    raw_opts = get_instruments(currency=currency, instrument_type="option", expired=False)
    opts = _parse_options(raw_opts)

    target_expiry = _now_sec() + target_days * 24 * 3600
    expiry = _closest_expiry(opts, target_expiry)
    strike = _closest_strike(opts, expiry, spot)

    wanted_type = "C" if prefer_call else "P"
    # If the preferred type doesn't exist at this strike, take the other side.
    candidates = [o for o in opts if o.expiry_sec == expiry and o.strike == strike and o.option_type == wanted_type]
    if not candidates:
        other_type = "P" if wanted_type == "C" else "C"
        candidates = [o for o in opts if o.expiry_sec == expiry and o.strike == strike and o.option_type == other_type]
    if not candidates:
        raise DeriveAPIError("Could not find ATM option instrument at chosen expiry/strike.")

    return candidates[0].instrument_name, spot, expiry, strike


def fetch_current_mark_iv(currency: str = "HYPE", target_days: int = 30) -> Dict[str, Any]:
    """
    "Mark IV" here is Derive's option_pricing.iv from public/get_ticker. :contentReference[oaicite:5]{index=5}
    """
    inst, spot, expiry, strike = _pick_atm_option_name(currency=currency, target_days=target_days, prefer_call=True)
    t = get_ticker(inst)

    op = t.get("option_pricing") or {}
    if not op:
        raise DeriveAPIError(f"No option_pricing in ticker for {inst} (is it an option instrument?)")

    # Derive timestamps are in milliseconds; convert to seconds for Python datetime.
    raw_ts = float(t["timestamp"])
    ts_sec = raw_ts / 1000.0

    out = {
        "ts_exchange_sec": int(ts_sec),
        "ts_iso": datetime.fromtimestamp(ts_sec, tz=timezone.utc).isoformat(),
        "currency": currency,
        "instrument_name": inst,
        "expiry_sec": expiry,
        "strike": strike,
        "spot_index_price": spot,
        "mark_price": float(t["mark_price"]),
        "iv": float(op["iv"]),
        "bid_iv": float(op["bid_iv"]),
        "ask_iv": float(op["ask_iv"]),
        "forward_price": float(op["forward_price"]),
    }
    return out


def append_csv_row(path: str, row: Dict[str, Any]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def export_last_n_days(in_path: str, out_path: str, days: int = 30) -> int:
    cutoff = datetime.now(tz=timezone.utc).timestamp() - days * 24 * 3600
    kept = 0
    with open(in_path, "r", newline="") as fin, open(out_path, "w", newline="") as fout:
        r = csv.DictReader(fin)
        w = csv.DictWriter(fout, fieldnames=r.fieldnames or [])
        w.writeheader()
        for row in r:
            try:
                ts = float(row.get("ts_exchange_sec", "0"))
            except Exception:
                continue
            if ts >= cutoff:
                w.writerow(row)
                kept += 1
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("snapshot", help="Fetch one current mark IV datapoint (1M-ish ATM option)")
    s1.add_argument("--currency", default="HYPE")
    s1.add_argument("--target-days", type=int, default=30)
    s1.add_argument("--out", default="hype_mark_iv.csv", help="CSV to append to")

    s2 = sub.add_parser("collect", help="Continuously collect mark IV into a CSV (build your own 1-month history)")
    s2.add_argument("--currency", default="HYPE")
    s2.add_argument("--target-days", type=int, default=30)
    s2.add_argument("--interval-sec", type=int, default=60)
    s2.add_argument("--out", default="hype_mark_iv.csv")

    s3 = sub.add_parser("export-last-month", help="Export last N days from a collected CSV")
    s3.add_argument("--in", dest="inp", default="hype_mark_iv.csv")
    s3.add_argument("--out", default="hype_mark_iv_last30d.csv")
    s3.add_argument("--days", type=int, default=30)

    args = ap.parse_args()

    if args.cmd == "snapshot":
        row = fetch_current_mark_iv(currency=args.currency, target_days=args.target_days)
        append_csv_row(args.out, row)
        print(row)

    elif args.cmd == "collect":
        print(f"Collecting {args.currency} mark IV every {args.interval_sec}s -> {args.out}")
        while True:
            try:
                row = fetch_current_mark_iv(currency=args.currency, target_days=args.target_days)
                append_csv_row(args.out, row)
                print(f"{row['ts_iso']} iv={row['iv']:.4f} inst={row['instrument_name']}")
            except KeyboardInterrupt:
                print("\nStopped.")
                return
            except Exception as e:
                # keep going; transient network or instrument issues happen
                print(f"[warn] {e}")
            time.sleep(args.interval_sec)

    elif args.cmd == "export-last-month":
        n = export_last_n_days(args.inp, args.out, days=args.days)
        print(f"Wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
