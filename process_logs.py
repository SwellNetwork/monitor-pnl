# Clean PnL Parser (exit-price based)
# Author: CatKhanh, 2025-11-05

import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
import math
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
# Regex to extract funding rate updates from processed market data (has both rates together)
market_data_re = re.compile(
    r"processed market data.*?trade_id='(?P<trade_id>[a-f0-9\-]+)'.*?"
    r"market_data=PairedMarketData\(timestamp=(?P<ts>datetime\.datetime\([^\)]*\)).*?"
    r"x_market_data=MarketData.*?funding_rate=(?P<x_fr>[\d\.eE\-]+).*?"
    r"y_market_data=MarketData.*?funding_rate=(?P<y_fr>[\d\.eE\-]+)",
    re.DOTALL
)


def price_pnl(is_long: bool, entry: float, exit: float, size: float) -> float:
    if not size or exit is None or entry is None:
        return 0.0
    return (exit - entry) * size if is_long else (entry - exit) * size


def leg_funding_pnl(is_long: bool, entry_price: float, size: float, fr: float, hours: float) -> float:
    """Funding PnL for a single leg: longs pay when fr>0, shorts pay when fr<0."""
    if not size or fr is None or hours <= 0:
        return 0.0
    notional = entry_price * size
    sign = -1.0 if is_long else 1.0
    return sign * notional * fr * hours


def compute_hourly_funding(entry_time, entry_rates, funding_rate_updates, final_hours, entry_info):
    """Compute funding PnL by summing hourly contributions. Calculate funding for each hour separately and sum."""
    if final_hours is None or final_hours <= 0:
        return 0.0, entry_rates["x"], entry_rates["y"]

    # Build timeline of funding rate changes, sorted by hours
    rate_changes = []
    for update in funding_rate_updates:
        if update["hours"] >= 0 and update["hours"] <= final_hours:
            rate_changes.append({
                "hours": update["hours"],
                "x_fr": update["x_fr"],
                "y_fr": update["y_fr"]
            })
    rate_changes.sort(key=lambda x: x["hours"])
    
    # Function to get the funding rate active at a given time point
    def get_rates_at(hours_elapsed: float):
        """Get the funding rates that were active at a given time point."""
        x_rate = entry_rates["x"]
        y_rate = entry_rates["y"]
        # Find the latest rate change before or at this time point
        for change in rate_changes:
            if change["hours"] <= hours_elapsed + 1e-9:
                x_rate = change["x_fr"]
                y_rate = change["y_fr"]
            else:
                break
        return x_rate, y_rate
    
    total_funding = 0.0
    
    # Calculate funding for each full hour
    num_full_hours = int(math.floor(final_hours))
    
    # Process each hour separately
    for hour in range(num_full_hours):
        hour_start = float(hour)
        # Get the funding rate active at the start of this hour
        x_rate, y_rate = get_rates_at(hour_start)
        
        # Calculate funding for this hour (1.0 hour)
        hour_funding = leg_funding_pnl(
            entry_info["x"]["is_long"],
            entry_info["x"]["entry_price"],
            entry_info["x"]["size"],
            x_rate,
            1.0,  # 1 hour
        )
        hour_funding += leg_funding_pnl(
            entry_info["y"]["is_long"],
            entry_info["y"]["entry_price"],
            entry_info["y"]["size"],
            y_rate,
            1.0,  # 1 hour
        )
        total_funding += hour_funding
    
    # Handle remaining fraction of an hour
    remaining_fraction = final_hours - num_full_hours
    if remaining_fraction > 0:
        # Get the funding rate active at the start of the fractional hour
        x_rate, y_rate = get_rates_at(float(num_full_hours))
        
        # Calculate funding for the fractional hour
        fraction_funding = leg_funding_pnl(
            entry_info["x"]["is_long"],
            entry_info["x"]["entry_price"],
            entry_info["x"]["size"],
            x_rate,
            remaining_fraction,
        )
        fraction_funding += leg_funding_pnl(
            entry_info["y"]["is_long"],
            entry_info["y"]["entry_price"],
            entry_info["y"]["size"],
            y_rate,
            remaining_fraction,
        )
        total_funding += fraction_funding
    
    # Get final rates (rates active at the end)
    final_x_rate, final_y_rate = get_rates_at(final_hours)
    
    return total_funding, final_x_rate, final_y_rate

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
        
        # Track latest observed prices per exchange to support simulated exits
        latest_price_by_exchange = {}
        for op_match in op_re.finditer(raw):
            latest_price_by_exchange[op_match.group("ex").lower()] = float(op_match.group("px"))

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
        
        if not entries:
            return 0, "No entries found"

        entries.sort(key=lambda ent: ent["entry_time"] or datetime.min.replace(tzinfo=timezone.utc))

        processed_trade_ids = set()
        actual_orders = []
        simulated_candidate = None
        last_exit_time = None

        # --- MAIN LOOP ---
        for ent in entries:
            if ent["trade_id"] in processed_trade_ids:
                continue
            processed_trade_ids.add(ent["trade_id"])

            trade_id = ent["trade_id"]

            # Get last known funding + duration
            state_snapshots = []
            for st_match in state_re.finditer(raw, ent["entry_log_pos"]):
                if st_match.group("trade_id") != trade_id:
                    continue
                state_snapshots.append({
                    "hours": float(st_match.group("hours")),
                    "last_x_fr": float(st_match.group("last_x_fr")),
                    "last_y_fr": float(st_match.group("last_y_fr")) / 100.0,
                })

            state_snapshots.sort(key=lambda s: s["hours"])
            hours_open = state_snapshots[-1]["hours"] if state_snapshots else 0.0
            
            # Extract funding rate updates from processed market data after entry
            funding_rate_updates = []
            entry_time_dt = ent["entry_time"]
            if entry_time_dt:
                # Find all market data updates for this trade after entry
                search_start = ent["entry_log_pos"]
                rate_updates_by_hour = {}  # Map of hours -> (x_fr, y_fr)
                
                # Extract funding rates from processed market data
                for md_match in market_data_re.finditer(raw, search_start):
                    if md_match.group("trade_id") != trade_id:
                        continue
                    ts_str = md_match.group("ts")
                    ts_dt = py_datetime_str_to_dt(ts_str)
                    if ts_dt and ts_dt >= entry_time_dt:
                        hours_elapsed = (ts_dt - entry_time_dt).total_seconds() / 3600.0
                        if hours_elapsed <= hours_open:
                            x_fr = float(md_match.group("x_fr"))
                            y_fr = float(md_match.group("y_fr")) / 100.0  # Convert percentage
                            # Use latest rate for each hour (overwrite if multiple updates in same hour)
                            hour_key = math.floor(hours_elapsed)
                            if hour_key not in rate_updates_by_hour or hours_elapsed > rate_updates_by_hour[hour_key][2]:
                                rate_updates_by_hour[hour_key] = (x_fr, y_fr, hours_elapsed)
                
                # Convert to list sorted by hours
                for hour_key in sorted(rate_updates_by_hour.keys()):
                    x_fr, y_fr, hours_elapsed = rate_updates_by_hour[hour_key]
                    funding_rate_updates.append({
                        "hours": hours_elapsed,
                        "x_fr": x_fr,
                        "y_fr": y_fr
                    })

            # --- Extract Exit Prices ---
            x_exit_px, y_exit_px = None, None
            for xm in exit_op_re.finditer(raw, ent["entry_log_pos"]):
                if xm.group("trade_id") == trade_id:
                    px = float(xm.group("px"))
                    if xm.group("which") == "x_exit": x_exit_px = px
                    elif xm.group("which") == "y_exit": y_exit_px = px

            # Skip incomplete exits
            if x_exit_px is not None and y_exit_px is not None:
                # Compute funding PnL via hourly aggregation
                entry_rates = {
                    "x": ent["x"]["funding_rate_hr"],
                    "y": ent["y"]["funding_rate_hr"],
                }
                funding_pnl, final_x_rate, final_y_rate = compute_hourly_funding(
                    ent["entry_time"], entry_rates, funding_rate_updates, hours_open, ent
                )

                x_price_pnl = price_pnl(ent["x"]["is_long"], ent["x"]["entry_price"], x_exit_px, ent["x"]["size"])
                y_price_pnl = price_pnl(ent["y"]["is_long"], ent["y"]["entry_price"], y_exit_px, ent["y"]["size"])

                total_pnl = x_price_pnl + y_price_pnl + funding_pnl

                def flow_for_position(is_long, fr):
                    if fr == 0:
                        return "flat"
                    pays = (is_long and fr > 0) or ((not is_long) and fr < 0)
                    return "you_pay" if pays else "you_receive"

                exit_time = None
                if ent["entry_time"] and hours_open:
                    exit_time = ent["entry_time"] + timedelta(hours=hours_open)

                actual_orders.append({
                    "order_type": "actual",
                    "trade_id": trade_id,
                    "entry_time": ent["entry_time"],
                    "exit_time": exit_time,
                    "x_entry_price": ent["x"]["entry_price"],
                    "y_entry_price": ent["y"]["entry_price"],
                    "x_exit_price": x_exit_px,
                    "y_exit_price": y_exit_px,
                    "hours_open": hours_open,
                    "x_is_long": ent["x"]["is_long"],
                    "y_is_long": ent["y"]["is_long"],
                    "x_price_pnl_usd": x_price_pnl,
                    "y_price_pnl_usd": y_price_pnl,
                    "funding_pnl_usd": funding_pnl,
                    "total_pnl_usd": total_pnl,
                    "x_funding_rate_final": final_x_rate,
                    "y_funding_rate_final": final_y_rate,
                    "x_funding_flow_for_position": flow_for_position(ent["x"]["is_long"], final_x_rate),
                    "y_funding_flow_for_position": flow_for_position(ent["y"]["is_long"], final_y_rate),
                })

                last_exit_time = exit_time or ent["entry_time"]
                simulated_candidate = None  # reset simulated candidate after an actual exit
                continue

            # Potential simulated order if no exit found
            candidate_entry_time = ent["entry_time"]
            if simulated_candidate is None:
                if last_exit_time is None:
                    simulated_candidate = {"entry": ent, "states": state_snapshots}
                else:
                    if candidate_entry_time and last_exit_time:
                        if candidate_entry_time >= last_exit_time:
                            simulated_candidate = {"entry": ent, "states": state_snapshots}
                    elif candidate_entry_time and last_exit_time is None:
                        simulated_candidate = {"entry": ent, "states": state_snapshots}

        simulated_orders = []
        if simulated_candidate:
            entry = simulated_candidate["entry"]
            states = simulated_candidate["states"]
            hours_open = states[-1]["hours"] if states else 0.0

            x_exit_px = latest_price_by_exchange.get(entry["x"]["ex"], entry["x"]["entry_price"])
            y_exit_px = latest_price_by_exchange.get(entry["y"]["ex"], entry["y"]["entry_price"])

            # Extract funding rate updates for simulated order
            sim_funding_rate_updates = []
            entry_time_dt = entry["entry_time"]
            if entry_time_dt:
                search_start = entry["entry_log_pos"]
                rate_updates_by_hour = {}
                
                # Extract funding rates from processed market data
                for md_match in market_data_re.finditer(raw, search_start):
                    if md_match.group("trade_id") != entry["trade_id"]:
                        continue
                    ts_str = md_match.group("ts")
                    ts_dt = py_datetime_str_to_dt(ts_str)
                    if ts_dt and ts_dt >= entry_time_dt:
                        hours_elapsed = (ts_dt - entry_time_dt).total_seconds() / 3600.0
                        if hours_elapsed <= hours_open:
                            x_fr = float(md_match.group("x_fr"))
                            y_fr = float(md_match.group("y_fr")) / 100.0  # Convert percentage
                            hour_key = math.floor(hours_elapsed)
                            if hour_key not in rate_updates_by_hour or hours_elapsed > rate_updates_by_hour[hour_key][2]:
                                rate_updates_by_hour[hour_key] = (x_fr, y_fr, hours_elapsed)
                
                for hour_key in sorted(rate_updates_by_hour.keys()):
                    x_fr, y_fr, hours_elapsed = rate_updates_by_hour[hour_key]
                    sim_funding_rate_updates.append({
                        "hours": hours_elapsed,
                        "x_fr": x_fr,
                        "y_fr": y_fr
                    })

            entry_rates = {
                "x": entry["x"]["funding_rate_hr"],
                "y": entry["y"]["funding_rate_hr"],
            }
            funding_pnl, final_x_rate, final_y_rate = compute_hourly_funding(
                entry["entry_time"], entry_rates, sim_funding_rate_updates, hours_open, entry
            )

            x_price_pnl = price_pnl(entry["x"]["is_long"], entry["x"]["entry_price"], x_exit_px, entry["x"]["size"])
            y_price_pnl = price_pnl(entry["y"]["is_long"], entry["y"]["entry_price"], y_exit_px, entry["y"]["size"])
            total_pnl = x_price_pnl + y_price_pnl + funding_pnl

            def flow_for_position(is_long, fr):
                if fr == 0:
                    return "flat"
                pays = (is_long and fr > 0) or ((not is_long) and fr < 0)
                return "you_pay" if pays else "you_receive"

            exit_time = None
            if entry["entry_time"] and hours_open:
                exit_time = entry["entry_time"] + timedelta(hours=hours_open)

            simulated_orders.append({
                "order_type": "simulated",
                "trade_id": entry["trade_id"],
                "entry_time": entry["entry_time"],
                "exit_time": exit_time,
                "x_entry_price": entry["x"]["entry_price"],
                "y_entry_price": entry["y"]["entry_price"],
                "x_exit_price": x_exit_px,
                "y_exit_price": y_exit_px,
                "hours_open": hours_open,
                "x_is_long": entry["x"]["is_long"],
                "y_is_long": entry["y"]["is_long"],
                "x_price_pnl_usd": x_price_pnl,
                "y_price_pnl_usd": y_price_pnl,
                "funding_pnl_usd": funding_pnl,
                "total_pnl_usd": total_pnl,
                "x_funding_rate_final": final_x_rate,
                "y_funding_rate_final": final_y_rate,
                "x_funding_flow_for_position": flow_for_position(entry["x"]["is_long"], final_x_rate),
                "y_funding_flow_for_position": flow_for_position(entry["y"]["is_long"], final_y_rate),
            })

        combined_orders = actual_orders + simulated_orders

        # --- SAVE OUTPUT ---
        if combined_orders:
            orders_df = pd.DataFrame(combined_orders).sort_values("entry_time")
            # Create CSV filename from log filename (replace .log with .csv)
            csv_filename = log_file.stem + ".csv"
            csv_path = output_folder / csv_filename
            orders_df.to_csv(csv_path, index=False)
            return len(combined_orders), None, {"actual": len(actual_orders), "simulated": len(simulated_orders)}
        else:
            return 0, "No complete orders found", {"actual": 0, "simulated": 0}
    except Exception as e:
        return 0, str(e), {"actual": 0, "simulated": 0}

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
    total_actual_orders = 0
    total_simulated_orders = 0
    results = []
    for log_file in tqdm(log_files, desc=f"Processing {input_folder.name}"):
        orders_count, error, stats = process_single_log_file(log_file, output_folder)
        total_orders += orders_count
        total_actual_orders += stats.get("actual", 0)
        total_simulated_orders += stats.get("simulated", 0)
        results.append((log_file.name, orders_count, error, stats))
    
    # Print summary
    print(f"\nSummary for {input_folder.name}:")
    for log_name, count, error, stats in results:
        if error:
            print(f"  ⚠ {log_name}: {error}")
        else:
            csv_name = Path(log_name).stem + ".csv"
            actual = stats.get("actual", 0)
            simulated = stats.get("simulated", 0)
            print(f"  ✓ {log_name} -> {csv_name} ({count} orders | actual={actual}, simulated={simulated})")
    print(
        f"Total: {total_orders} orders processed across {len(log_files)} files "
        f"(actual={total_actual_orders}, simulated={total_simulated_orders})\n"
    )

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

if __name__ == "__main__":
    main()