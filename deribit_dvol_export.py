#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import time
from typing import Dict, List, Optional, Tuple

import requests


def now_ms() -> int:
    return int(time.time() * 1000)


def ms_to_iso(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).isoformat()


def fetch_page(
    session: requests.Session,
    base_url: str,
    currency: str,
    start_ts: int,
    end_ts: int,
    resolution: str,
    timeout: float = 20.0,
    retries: int = 8,
) -> Tuple[List[List[float]], Optional[int]]:
    """
    Returns (data, continuation)
    data rows: [timestamp_ms, open_iv, high_iv, low_iv, close_iv]
    continuation: int or None
    Endpoint: /public/get_volatility_index_data :contentReference[oaicite:2]{index=2}
    """
    url = f"{base_url.rstrip('/')}/public/get_volatility_index_data"
    params = {
        "currency": currency,
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "resolution": resolution,
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                # simple backoff (Deribit may rate-limit)
                time.sleep(min(10.0, 0.5 * (2 ** attempt)))
                continue
            r.raise_for_status()
            payload = r.json()
            if "error" in payload:
                raise RuntimeError(payload["error"])
            result = payload.get("result", {}) or {}
            data = result.get("data", []) or []
            cont = result.get("continuation", None)
            return data, cont
        except Exception as e:
            last_err = e
            time.sleep(min(10.0, 0.5 * (2 ** attempt)))

    raise RuntimeError(f"Failed after retries: {last_err}")


def fetch_all(
    base_url: str,
    currencies: List[str],
    resolution: str,
    start_ts: int,
    end_ts: int,
) -> Dict[str, List[Dict]]:
    """
    Keep paging backward using `continuation` as new end_timestamp until None. :contentReference[oaicite:3]{index=3}
    Returns dict mapping currency to list of rows.
    """
    session = requests.Session()
    out_rows_by_currency: Dict[str, List[Dict]] = {ccy: [] for ccy in currencies}
    seen = set()  # (currency, timestamp_ms)

    for ccy in currencies:
        cur_end = end_ts
        while True:
            data, cont = fetch_page(
                session=session,
                base_url=base_url,
                currency=ccy,
                start_ts=start_ts,
                end_ts=cur_end,
                resolution=resolution,
            )

            for row in data:
                # row: [ts, o, h, l, c]
                ts = int(row[0])
                key = (ccy, ts)
                if key in seen:
                    continue
                seen.add(key)
                out_rows_by_currency[ccy].append(
                    {
                        "currency": ccy,
                        "resolution": resolution,
                        "timestamp_ms": ts,
                        "timestamp_iso_utc": ms_to_iso(ts),
                        "open_iv": row[1],
                        "high_iv": row[2],
                        "low_iv": row[3],
                        "close_iv": row[4],
                    }
                )

            if cont is None:
                break

            # docs: continuation -> use as end_timestamp next request :contentReference[oaicite:4]{index=4}
            # safety break if API returns non-decreasing continuation:
            if cont >= cur_end:
                break

            # stop if we've moved earlier than start bound
            if cont <= start_ts:
                break

            cur_end = cont

        # sort by timestamp_ms for each currency
        out_rows_by_currency[ccy].sort(key=lambda x: x["timestamp_ms"])

    return out_rows_by_currency


def write_csv(rows: List[Dict], out_path: str) -> None:
    cols = ["currency", "resolution", "timestamp_ms", "timestamp_iso_utc", "open_iv", "high_iv", "low_iv", "close_iv"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser(description="Download Deribit DVOL candles (BTC/ETH) via public/get_volatility_index_data")
    p.add_argument("--env", choices=["test", "prod"], default="test",
                   help="test -> https://test.deribit.com/api/v2, prod -> https://www.deribit.com/api/v2")
    p.add_argument("--resolution", default="1D", choices=["1", "60", "3600", "43200", "1D"],
                   help="Supported resolutions per docs. :contentReference[oaicite:5]{index=5}")
    p.add_argument("--start", type=int, default=0, help="start_timestamp (ms). Default 0 = earliest possible.")
    p.add_argument("--end", type=int, default=0, help="end_timestamp (ms). Default 0 = now.")
    p.add_argument("--out", default="deribit_dvol", help="Output CSV base path (will generate _btc.csv and _eth.csv).")
    args = p.parse_args()

    base_url = "https://test.deribit.com/api/v2" if args.env == "test" else "https://www.deribit.com/api/v2"
    end_ts = args.end if args.end > 0 else now_ms()

    rows_by_currency = fetch_all(
        base_url=base_url,
        currencies=["BTC", "ETH"],
        resolution=args.resolution,
        start_ts=args.start,
        end_ts=end_ts,
    )
    
    # Write separate CSV files for each currency
    for ccy, rows in rows_by_currency.items():
        # Generate output filename: base_path_btc.csv or base_path_eth.csv
        base_path = args.out.rstrip(".csv")
        out_path = f"{base_path}_{ccy.lower()}.csv"
        write_csv(rows, out_path)
        print(f"Wrote {len(rows)} {ccy} rows -> {out_path}")


if __name__ == "__main__":
    main()
