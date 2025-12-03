#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any, Dict, List, Optional

import requests

# Docs: /public/get_book_summary_by_currency params: currency (required), kind (optional) :contentReference[oaicite:2]{index=2}


def now_ms() -> int:
    return int(time.time() * 1000)


def request_json(session: requests.Session, url: str, params: Dict[str, Any], retries: int = 8, timeout: float = 20.0) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(min(10.0, 0.5 * (2 ** attempt)))
                continue
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(payload["error"])
            return payload
        except Exception as e:
            last_err = e
            time.sleep(min(10.0, 0.5 * (2 ** attempt)))
    raise RuntimeError(f"Request failed after retries: {last_err}")


def fetch_book_summary(session: requests.Session, base_url: str, currency: str, kind: Optional[str]) -> List[Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/public/get_book_summary_by_currency"
    params: Dict[str, Any] = {"currency": currency}
    if kind:
        params["kind"] = kind  # future/option/spot/future_combo/option_combo :contentReference[oaicite:3]{index=3}

    payload = request_json(session, url, params=params)
    result = payload.get("result", []) or []
    ts = now_ms()

    rows: List[Dict[str, Any]] = []
    for obj in result:
        if not isinstance(obj, dict):
            continue
        row = {
            "fetch_ts_ms": ts,
            "currency": currency,
            "kind_filter": kind or "ALL",
            **obj,  # keep ALL returned fields (ask_price, bid_price, mark_iv, open_interest, volume_usd, ...) :contentReference[oaicite:4]{index=4}
        }
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    cols = sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            cleaned = {}
            for k, v in r.items():
                if isinstance(v, (dict, list)):
                    cleaned[k] = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                else:
                    cleaned[k] = v
            w.writerow(cleaned)


def collect_loop(args: argparse.Namespace) -> None:
    base_url = "https://test.deribit.com/api/v2" if args.env == "test" else "https://www.deribit.com/api/v2"
    session = requests.Session()

    all_rows: List[Dict[str, Any]] = []
    end_at = time.time() + args.minutes * 60.0

    while time.time() < end_at:
        t0 = time.time()
        rows: List[Dict[str, Any]] = []

        for ccy in args.currencies:
            if args.mode == "all":
                # one call per currency: maximum in a single response (all kinds) :contentReference[oaicite:5]{index=5}
                rows.extend(fetch_book_summary(session, base_url, ccy, kind=None))
            else:
                # split kinds: sometimes easier to manage and ensures you get every kind
                for k in args.kinds:
                    rows.extend(fetch_book_summary(session, base_url, ccy, kind=k))

        all_rows.extend(rows)
        write_csv(all_rows, args.out)
        print(f"[collect] +{len(rows)} rows, total={len(all_rows)} -> {args.out}")

        sleep_s = max(0.0, args.interval - (time.time() - t0))
        time.sleep(sleep_s)


def main() -> None:
    p = argparse.ArgumentParser(description="Export Deribit /public/get_book_summary_by_currency to CSV (BTC/ETH max).")
    p.add_argument("--env", choices=["test", "prod"], default="test")
    p.add_argument("--currencies", nargs="+", default=["BTC", "ETH"])
    p.add_argument("--out", default="deribit_book_summary_btc_eth.csv")

    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("snapshot", help="One-time snapshot to CSV.")
    ps.add_argument("--mode", choices=["all", "split"], default="all",
                    help="all: call without kind (max). split: call each kind separately.")
    ps.add_argument("--kinds", nargs="+", default=["option", "future", "spot", "future_combo", "option_combo"])
    ps.set_defaults(func="snapshot")

    pc = sub.add_parser("collect", help="Collect snapshots over time (build your own history).")
    pc.add_argument("--mode", choices=["all", "split"], default="all")
    pc.add_argument("--kinds", nargs="+", default=["option", "future", "spot", "future_combo", "option_combo"])
    pc.add_argument("--interval", type=float, default=60.0)
    pc.add_argument("--minutes", type=float, default=60.0)
    pc.set_defaults(func="collect")

    args = p.parse_args()

    base_url = "https://test.deribit.com/api/v2" if args.env == "test" else "https://www.deribit.com/api/v2"
    session = requests.Session()

    if args.func == "snapshot":
        rows: List[Dict[str, Any]] = []
        for ccy in args.currencies:
            if args.mode == "all":
                rows.extend(fetch_book_summary(session, base_url, ccy, kind=None))
            else:
                for k in args.kinds:
                    rows.extend(fetch_book_summary(session, base_url, ccy, kind=k))
        write_csv(rows, args.out)
        print(f"Wrote {len(rows)} rows -> {args.out}")
    else:
        collect_loop(args)


if __name__ == "__main__":
    main()
