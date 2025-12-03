#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import aiohttp

# Deribit JSON-RPC over HTTP: /api/v2/public/get_order_book, /public/get_instruments, etc. :contentReference[oaicite:2]{index=2}
DEFAULT_BASE_URL = "https://www.deribit.com/api/v2"


def now_ms() -> int:
    return int(time.time() * 1000)


def flatten_json(obj: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dicts. Lists are kept as JSON strings (so we preserve *all* fields).
    """
    out: Dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, dict):
                out.update(flatten_json(v, key, sep=sep))
            elif isinstance(v, list):
                # preserve full depth content (bids/asks) as JSON string
                out[key] = json.dumps(v, separators=(",", ":"), ensure_ascii=False)
            else:
                out[key] = v
    elif isinstance(obj, list):
        out[parent_key or "value"] = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    else:
        out[parent_key or "value"] = obj

    return out


def parse_option_instrument_name(name: str) -> Dict[str, Any]:
    """
    Deribit option naming: BTC-25MAR23-42000-C  (currency-expiry-strike-type)
    """
    parts = name.split("-")
    out: Dict[str, Any] = {"instrument_name": name}
    if len(parts) >= 4:
        out["base_currency"] = parts[0]
        out["expiry_code"] = parts[1]
        out["strike"] = safe_float(parts[2])
        out["option_type"] = parts[3]
    return out


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def ms_to_datetime(ts_ms: Any) -> Optional[str]:
    """
    Convert milliseconds timestamp to ISO format datetime string.
    Returns None if conversion fails or input is None/empty.
    """
    try:
        if ts_ms is None or ts_ms == "":
            return None
        ts_ms_int = int(float(ts_ms))
        dt = datetime.fromtimestamp(ts_ms_int / 1000.0, tz=timezone.utc)
        return dt.isoformat()
    except (ValueError, TypeError, OSError, OverflowError):
        return None


def convert_timestamps_to_datetime(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert all timestamp_ms fields to datetime strings.
    Adds new columns with _datetime suffix alongside original timestamp columns.
    """
    converted_rows: List[Dict[str, Any]] = []
    
    for row in rows:
        new_row = row.copy()
        
        # Find all timestamp fields (ending with _ms or containing 'timestamp')
        for key, value in row.items():
            if value is None or value == "":
                continue
                
            # Check if this is a timestamp field
            is_timestamp = (
                key.endswith("_ms") or 
                "timestamp" in key.lower()
            )
            
            if is_timestamp:
                # Try to convert to datetime
                dt_str = ms_to_datetime(value)
                if dt_str is not None:
                    # Add datetime column with _datetime suffix
                    datetime_key = key.replace("_ms", "_datetime") if key.endswith("_ms") else f"{key}_datetime"
                    new_row[datetime_key] = dt_str
        
        converted_rows.append(new_row)
    
    return converted_rows


@dataclass
class DeribitClient:
    base_url: str
    session: aiohttp.ClientSession
    timeout_s: float = 12.0
    max_retries: int = 6

    async def get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.get(url, params=params, timeout=self.timeout_s) as resp:
                    # Handle rate limiting
                    if resp.status == 429:
                        retry_after = resp.headers.get("Retry-After")
                        sleep_s = float(retry_after) if retry_after else min(8.0, 0.5 * (2 ** attempt))
                        sleep_s = sleep_s + random.random() * 0.25
                        await asyncio.sleep(sleep_s)
                        continue

                    resp.raise_for_status()
                    data = await resp.json()

                    # Deribit returns {"jsonrpc":"2.0","id":...,"result":...} on success
                    if isinstance(data, dict) and "error" in data:
                        raise RuntimeError(f"Deribit error: {data.get('error')}")
                    return data
            except Exception as e:
                last_err = e
                # backoff with jitter
                sleep_s = min(8.0, 0.4 * (2 ** attempt)) + random.random() * 0.2
                await asyncio.sleep(sleep_s)

        raise RuntimeError(f"GET {path} failed after retries: {last_err}")

    async def get_instruments(self, currency: str, expired: bool) -> List[Dict[str, Any]]:
        # docs: /public/get_instruments params currency, kind, expired :contentReference[oaicite:3]{index=3}
        payload = await self.get(
            "/public/get_instruments",
            {"currency": currency, "kind": "option", "expired": str(expired).lower()},
        )
        return payload.get("result", [])

    async def get_order_book(self, instrument_name: str, depth: int) -> Dict[str, Any]:
        # docs: /public/get_order_book params instrument_name, depth :contentReference[oaicite:4]{index=4}
        return await self.get("/public/get_order_book", {"instrument_name": instrument_name, "depth": depth})


async def fetch_snapshot_rows(
    client: DeribitClient,
    currencies: Sequence[str],
    depth: int,
    expired: bool,
    max_instruments_per_ccy: Optional[int],
    concurrency: int,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    rows: List[Dict[str, Any]] = []

    async def worker(inst: Dict[str, Any]) -> None:
        name = inst.get("instrument_name")
        if not name:
            return
        async with sem:
            ts_fetch_ms = now_ms()
            try:
                ob = await client.get_order_book(name, depth=depth)
                # Flatten the full response (id, jsonrpc, and all result fields)
                flat: Dict[str, Any] = {"fetch_ts_ms": ts_fetch_ms}

                # Attach instrument metadata from get_instruments (useful + stable)
                flat.update(flatten_json({"instrument": inst}, parent_key="meta"))

                # Attach quick parsed columns too (handy in CSV)
                flat.update(flatten_json(parse_option_instrument_name(name), parent_key="parsed"))

                flat.update(flatten_json(ob, parent_key="rpc"))
                rows.append(flat)
            except Exception as e:
                rows.append(
                    {
                        "fetch_ts_ms": ts_fetch_ms,
                        "parsed.instrument_name": name,
                        "error": str(e),
                    }
                )

    for ccy in currencies:
        instruments = await client.get_instruments(ccy, expired=expired)
        # Filter to active tradeable instruments if present
        # (get_instruments has is_active field) :contentReference[oaicite:5]{index=5}
        instruments = [i for i in instruments if i.get("kind") == "option"]
        if not expired:
            instruments = [i for i in instruments if i.get("is_active") is True]

        if max_instruments_per_ccy is not None:
            instruments = instruments[: max_instruments_per_ccy]

        tasks = [asyncio.create_task(worker(inst)) for inst in instruments]
        await asyncio.gather(*tasks)

    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    # Convert all timestamp_ms fields to datetime
    rows = convert_timestamps_to_datetime(rows)
    
    # union columns
    cols: List[str] = sorted({k for r in rows for k in r.keys()})

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            # ensure all non-primitive values are strings
            cleaned = {}
            for k, v in r.items():
                if isinstance(v, (dict, list)):
                    cleaned[k] = json.dumps(v, separators=(",", ":"), ensure_ascii=False)
                else:
                    cleaned[k] = v
            w.writerow(cleaned)


def write_csv_by_currency(rows: List[Dict[str, Any]], base_out_path: str) -> None:
    """
    Write separate CSV files for each currency (BTC and ETH).
    Groups rows by parsed.base_currency or meta.instrument.base_currency.
    """
    # Group rows by currency
    rows_by_currency: Dict[str, List[Dict[str, Any]]] = {}
    
    for row in rows:
        # Try to find currency from parsed or meta fields
        currency = None
        if "parsed.base_currency" in row:
            currency = row["parsed.base_currency"]
        elif "meta.instrument.base_currency" in row:
            currency = row["meta.instrument.base_currency"]
        
        # Fallback: try to infer from instrument_name if present
        if not currency:
            inst_name = row.get("parsed.instrument_name") or row.get("meta.instrument.instrument_name", "")
            if inst_name.startswith("BTC"):
                currency = "BTC"
            elif inst_name.startswith("ETH"):
                currency = "ETH"
        
        # Default to "UNKNOWN" if we can't determine currency
        if not currency:
            currency = "UNKNOWN"
        
        if currency not in rows_by_currency:
            rows_by_currency[currency] = []
        rows_by_currency[currency].append(row)
    
    # Write separate CSV for each currency
    base_path = base_out_path.rstrip(".csv")
    for currency, currency_rows in rows_by_currency.items():
        out_path = f"{base_path}_{currency.lower()}.csv"
        write_csv(currency_rows, out_path)
        print(f"Wrote {len(currency_rows)} {currency} rows -> {out_path}")


async def cmd_snapshot(args: argparse.Namespace) -> None:
    async with aiohttp.ClientSession(headers={"Accept": "application/json"}) as session:
        client = DeribitClient(base_url=args.base_url.rstrip("/"), session=session)
        rows = await fetch_snapshot_rows(
            client=client,
            currencies=args.currencies,
            depth=args.depth,
            expired=args.expired,
            max_instruments_per_ccy=args.max_instruments,
            concurrency=args.concurrency,
        )
        write_csv_by_currency(rows, args.out)
    print(f"Wrote snapshot CSVs: {args.out} (total rows={len(rows)})")


async def cmd_collect(args: argparse.Namespace) -> None:
    """
    Collect snapshots every --interval seconds for --minutes minutes, appending to separate CSVs per currency.
    This is how you build "historical order book" since the API itself doesn't provide past snapshots. :contentReference[oaicite:6]{index=6}
    """
    end_at = time.time() + args.minutes * 60.0
    all_rows: List[Dict[str, Any]] = []

    async with aiohttp.ClientSession(headers={"Accept": "application/json"}) as session:
        client = DeribitClient(base_url=args.base_url.rstrip("/"), session=session)

        while time.time() < end_at:
            t0 = time.time()
            rows = await fetch_snapshot_rows(
                client=client,
                currencies=args.currencies,
                depth=args.depth,
                expired=args.expired,
                max_instruments_per_ccy=args.max_instruments,
                concurrency=args.concurrency,
            )
            all_rows.extend(rows)

            # write incrementally (so you don't lose progress) - separate files per currency
            write_csv_by_currency(all_rows, args.out)
            print(f"[collect] appended rows={len(rows)} total={len(all_rows)} -> {args.out}")

            # sleep until next tick
            dt = time.time() - t0
            sleep_s = max(0.0, args.interval - dt)
            await asyncio.sleep(sleep_s)

    print(f"Done. Wrote collected CSVs: {args.out} (total rows={len(all_rows)})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export Deribit BTC/ETH options order books (public/get_order_book) to CSV."
    )
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Default: https://www.deribit.com/api/v2")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--currencies", nargs="+", default=["BTC", "ETH"], help="e.g. BTC ETH")
    common.add_argument("--depth", type=int, default=10000, help="Order book depth (max 10000).")
    common.add_argument("--expired", action="store_true", help="Use expired=true in get_instruments (recently expired).")
    common.add_argument(
        "--max-instruments",
        type=int,
        default=None,
        help="Limit instruments per currency (useful to avoid huge dumps).",
    )
    common.add_argument("--concurrency", type=int, default=5, help="Concurrent get_order_book requests.")
    common.add_argument("--out", required=True, help="Output CSV base path (will generate _btc.csv and _eth.csv).")

    ps = sub.add_parser("snapshot", parents=[common], help="One-time snapshot for all instruments.")
    ps.set_defaults(func=cmd_snapshot)

    pc = sub.add_parser("collect", parents=[common], help="Collect snapshots over time (build your own history).")
    pc.add_argument("--interval", type=float, default=60.0, help="Seconds between snapshots.")
    pc.add_argument("--minutes", type=float, default=30.0, help="Total duration in minutes.")
    pc.set_defaults(func=cmd_collect)

    return p


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)


if __name__ == "__main__":
    main()
