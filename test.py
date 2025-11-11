# Clean PnL Parser (exit-price based)
# Author: CatKhanh, 2025-11-05

import re
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

# --- REGEX DEFINITIONS ---
entry_re = re.compile(
    r"EntryPlan\(trade_id='(?P<trade_id>[a-f0-9\-]+)'.*?timestamp=(?P<ts>datetime\.datetime\([^\)]*\))"
    r".*?receiver='(?P<recv>[xy])', "
    r"x_position=Position\(exchange=<Exchange\.(?P<x_ex>\w+).*?is_long=(?P<x_long>True|False), entry_price=(?P<x_entry>[\d\.eE\-]+).*?funding_rate=(?P<x_fr>[\d\.eE\-]+)\)"
    r".*?y_position=Position\(exchange=<Exchange\.(?P<y_ex>\w+).*?is_long=(?P<y_long>True|False), entry_price=(?P<y_entry>[\d\.eE\-]+).*?funding_rate=(?P<y_fr>[\d\.eE\-]+)\)",
    re.DOTALL
)
op_re = re.compile(
    r"Operation\(exchange=<Exchange\.(?P<ex>\w+).*?metadata=\{.*?'is_long': (?P<is_long>True|False).*?'sz': (?P<sz>[\d\.eE\-]+), 'limit_px': (?P<px>[\d\.eE\-]+).*?\}",
    re.DOTALL
)
exit_op_re = re.compile(
    r"Operation\(exchange=<Exchange\.(?P<ex>\w+).*?metadata=\{.*?'limit_px': (?P<px>[\d\.eE\-]+).*?'cloid': '(?P<trade_id>[a-f0-9\-]+)_(?P<which>x_exit|y_exit)'.*?\}",
    re.DOTALL
)
state_re = re.compile(
    r"engine state after cycle: EngineState\(.*?hours_open=(?P<hours>[\d\.eE\-]+), "
    r"last_entry_funding_x=(?P<last_x_fr>[\d\.eE\-]+), last_entry_funding_y=(?P<last_y_fr>[\d\.eE\-]+), "
    r"last_receiver='(?P<recv>[xy])', current_trade_id='(?P<trade_id>[a-f0-9\-]+)'",
    re.DOTALL
)

def py_datetime_str_to_dt(s: str) -> datetime:
    nums = re.findall(r"datetime\.datetime\((\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)", s)
    if nums:
        y, M, d, h, m, sec, usec = map(int, nums[0])
        return datetime(y, M, d, h, m, sec, usec, tzinfo=timezone.utc)
    nums2 = re.findall(r"datetime\.datetime\((\d+), (\d+), (\d+), (\d+), (\d+), (\d+)", s)
    if nums2:
        y, M, d, h, m, sec = map(int, nums2[0])
        return datetime(y, M, d, h, m, sec, tzinfo=timezone.utc)
    return None

def process_single_log_file(log_file: Path, output_folder: Path):
    """Process a single log file and save CSV to output_folder."""
    try:
        # Read log file
        raw = log_file.read_text(encoding="utf-8", errors="ignore")
        
        # --- ENTRY PARSING ---
        entries = []
        for em in entry_re.finditer(raw):
            trade_id = em.group("trade_id")
            ts = py_datetime_str_to_dt(em.group("ts"))
            recv = em.group("recv")
            x_ex, y_ex = em.group("x_ex").lower(), em.group("y_ex").lower()
            x_long, y_long = em.group("x_long") == "True", em.group("y_long") == "True"
            x_entry, y_entry = float(em.group("x_entry")), float(em.group("y_entry"))
            x_fr, y_fr = float(em.group("x_fr")), float(em.group("y_fr")) / 100.0

            sizes = {}
            for m in op_re.finditer(raw, em.end(), em.end() + 3000):
                sizes[m.group("ex").lower()] = float(m.group("sz"))

            entries.append({
                "trade_id": trade_id, "entry_time": ts, "receiver": recv,
                "x": {"ex": x_ex, "is_long": x_long, "entry_price": x_entry, "funding_rate_hr": x_fr, "size": sizes.get(x_ex)},
                "y": {"ex": y_ex, "is_long": y_long, "entry_price": y_entry, "funding_rate_hr": y_fr, "size": sizes.get(y_ex)},
                "entry_log_pos": em.end()
            })
        
        # --- MAIN LOOP ---
        orders = []
        for i, ent in enumerate(entries):
            trade_id = ent["trade_id"]

            # Get last known funding + duration
            states = [m for m in state_re.finditer(raw, ent["entry_log_pos"]) if m.group("trade_id") == trade_id]
            if states:
                st = states[-1]
                hours_open = float(st.group("hours"))
                last_x_fr, last_y_fr = float(st.group("last_x_fr")), float(st.group("last_y_fr")) / 100.0
            else:
                hours_open = 0.0
                last_x_fr, last_y_fr = ent["x"]["funding_rate_hr"], ent["y"]["funding_rate_hr"]

            # --- Extract Exit Prices ---
            x_exit_px, y_exit_px = None, None
            for xm in exit_op_re.finditer(raw, ent["entry_log_pos"]):
                if xm.group("trade_id") == trade_id:
                    px = float(xm.group("px"))
                    if xm.group("which") == "x_exit": x_exit_px = px
                    elif xm.group("which") == "y_exit": y_exit_px = px

            # Skip incomplete exits
            if x_exit_px is None or y_exit_px is None:
                continue

            # --- PnL & Funding Direction ---
            def price_pnl(is_long, entry, exit, size):
                if not size: return np.nan
                return (exit - entry) * size if is_long else (entry - exit) * size

            def leg_funding_pnl(is_long, entry_price, size, fr, hours):
                """Funding PnL for a single leg: longs pay when fr>0, shorts pay when fr<0."""
                if not size:
                    return 0.0
                notional = entry_price * size
                # Longs pay when fr>0 -> negative PnL; receive when fr<0 -> positive PnL
                # Shorts receive when fr>0 -> positive PnL; pay when fr<0 -> negative PnL
                sign = -1.0 if is_long else 1.0
                return sign * notional * fr * hours

            x_price_pnl = price_pnl(ent["x"]["is_long"], ent["x"]["entry_price"], x_exit_px, ent["x"]["size"])
            y_price_pnl = price_pnl(ent["y"]["is_long"], ent["y"]["entry_price"], y_exit_px, ent["y"]["size"])

            # Per-leg funding PnL
            x_funding_pnl = leg_funding_pnl(ent["x"]["is_long"], ent["x"]["entry_price"], ent["x"]["size"], last_x_fr, hours_open)
            y_funding_pnl = leg_funding_pnl(ent["y"]["is_long"], ent["y"]["entry_price"], ent["y"]["size"], last_y_fr, hours_open)

            # Sum funding PnL across legs
            total_funding_pnl = x_funding_pnl + y_funding_pnl

            # Per-position funding flow (relative to our position)
            def flow_for_position(is_long, fr):
                if fr == 0:
                    return "flat"
                # Long pays when fr>0; short pays when fr<0
                pays = (is_long and fr > 0) or ((not is_long) and fr < 0)
                return "you_pay" if pays else "you_receive"

            total_pnl = (x_price_pnl or 0) + (y_price_pnl or 0) + total_funding_pnl

            orders.append({
                "trade_id": trade_id,
                "entry_time": ent["entry_time"],
                "x_entry_price": ent["x"]["entry_price"],
                "y_entry_price": ent["y"]["entry_price"],
                "x_exit_price": x_exit_px,
                "y_exit_price": y_exit_px,
                "hours_open": hours_open,
                "x_is_long": ent["x"]["is_long"],
                "y_is_long": ent["y"]["is_long"],
                "x_price_pnl_usd": x_price_pnl,
                "y_price_pnl_usd": y_price_pnl,
                "x_funding_pnl_usd": x_funding_pnl,
                "y_funding_pnl_usd": y_funding_pnl,
                
                # Funding direction per leg
                "x_funding_direction": ("long_pays_short" if last_x_fr > 0 else ("short_pays_long" if last_x_fr < 0 else "flat")),
                "y_funding_direction": ("long_pays_short" if last_y_fr > 0 else ("short_pays_long" if last_y_fr < 0 else "flat")),
                "x_funding_flow_for_position": flow_for_position(ent["x"]["is_long"], last_x_fr),
                "y_funding_flow_for_position": flow_for_position(ent["y"]["is_long"], last_y_fr),

                # Totals
                "funding_pnl_usd": total_funding_pnl,
                "total_pnl_usd": total_pnl,
                "funding_direction": "x_pays_y" if last_x_fr > last_y_fr else "y_pays_x"
            })

        # --- SAVE OUTPUT ---
        if orders:
            orders_df = pd.DataFrame(orders).sort_values("entry_time")
            # Create CSV filename from log filename (replace .log with .csv)
            csv_filename = log_file.stem + ".csv"
            csv_path = output_folder / csv_filename
            orders_df.to_csv(csv_path, index=False)
            return len(orders), None
        else:
            return 0, "No complete orders found"
    except Exception as e:
        return 0, str(e)

def process_logs(input_folder: Path, output_folder: Path):
    """Process all log files in input_folder and save individual CSVs to output_folder."""
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all .log files in input folder
    log_files = sorted(input_folder.glob("*.log"))
    if not log_files:
        print(f"No .log files found in {input_folder}")
        return
    
    print(f"Processing {len(log_files)} log files from {input_folder.name}...")
    
    # Process each log file individually
    total_orders = 0
    results = []
    for log_file in tqdm(log_files, desc=f"Processing {input_folder.name}"):
        orders_count, error = process_single_log_file(log_file, output_folder)
        total_orders += orders_count
        results.append((log_file.name, orders_count, error))
    
    # Print summary
    print(f"\nSummary for {input_folder.name}:")
    for log_name, count, error in results:
        if error:
            print(f"  ⚠ {log_name}: {error}")
        else:
            csv_name = Path(log_name).stem + ".csv"
            print(f"  ✓ {log_name} -> {csv_name} ({count} orders)")
    print(f"Total: {total_orders} orders processed across {len(log_files)} files\n")

def main():
    base_path = Path(__file__).parent
    
    # Process logs folder -> pnl_logs folder
    logs_folder = base_path / "logs"
    pnl_logs_folder = base_path / "pnl_logs"
    if logs_folder.exists():
        print("\n" + "="*60)
        print("Processing logs folder -> pnl_logs folder")
        print("="*60)
        process_logs(logs_folder, pnl_logs_folder)
    else:
        print(f"Warning: {logs_folder} does not exist")
    
    # Process old_logs folder -> pnl_old_logs folder
    old_logs_folder = base_path / "old_logs"
    pnl_old_logs_folder = base_path / "pnl_old_logs"
    if old_logs_folder.exists():
        print("\n" + "="*60)
        print("Processing old_logs folder -> pnl_old_logs folder")
        print("="*60)
        process_logs(old_logs_folder, pnl_old_logs_folder)
    else:
        print(f"Warning: {old_logs_folder} does not exist")

if __name__ == "__main__":
    main()