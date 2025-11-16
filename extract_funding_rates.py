# Extract Funding Rates from Log Files
# Author: CatKhanh, 2025-11-11
# Generates CSV files with timestamp, hyperliquid funding rate, lighter funding rate

import re
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from tqdm import tqdm

# Regex to extract funding rate updates from processed market data
market_data_re = re.compile(
    r"processed market data.*?"
    r"market_data=PairedMarketData\(timestamp=(?P<ts>datetime\.datetime\([^\)]*\)).*?"
    r"x_market_data=MarketData.*?funding_rate=(?P<x_fr>[\d\.eE\-]+).*?"
    r"y_market_data=MarketData.*?funding_rate=(?P<y_fr>[\d\.eE\-]+)",
    re.DOTALL
)

def py_datetime_str_to_dt(s: str) -> datetime:
    """Convert Python datetime string representation to datetime object."""
    nums = re.findall(r"datetime\.datetime\((\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)", s)
    if nums:
        y, M, d, h, m, sec, usec = map(int, nums[0])
        return datetime(y, M, d, h, m, sec, usec, tzinfo=timezone.utc)
    nums2 = re.findall(r"datetime\.datetime\((\d+), (\d+), (\d+), (\d+), (\d+), (\d+)", s)
    if nums2:
        y, M, d, h, m, sec = map(int, nums2[0])
        return datetime(y, M, d, h, m, sec, tzinfo=timezone.utc)
    return None

def extract_funding_rates_from_log(log_file: Path):
    """Extract funding rates from a single log file."""
    try:
        # Read log file
        raw = log_file.read_text(encoding="utf-8", errors="ignore")
        
        # Extract all funding rate updates
        funding_rate_data = []
        seen_timestamps = set()  # To avoid duplicates
        
        for md_match in market_data_re.finditer(raw):
            ts_str = md_match.group("ts")
            ts_dt = py_datetime_str_to_dt(ts_str)
            
            if ts_dt is None:
                continue
            
            # Avoid duplicates (same timestamp)
            ts_key = ts_dt.isoformat()
            if ts_key in seen_timestamps:
                continue
            seen_timestamps.add(ts_key)
            
            x_fr = float(md_match.group("x_fr"))
            y_fr = float(md_match.group("y_fr")) / 100.0  # Convert percentage to decimal
            
            funding_rate_data.append({
                "timestamp": ts_dt,
                "hyperliquid_funding_rate": x_fr,
                "lighter_funding_rate": y_fr
            })
        
        # Sort by timestamp
        funding_rate_data.sort(key=lambda x: x["timestamp"])
        
        return funding_rate_data, None
    except Exception as e:
        return [], str(e)

def process_logs(input_folder: Path, output_folder: Path):
    """Process all log files in input_folder and save funding rate CSVs to output_folder."""
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all .log files in input folder
    log_files = sorted(input_folder.glob("*.log"))
    if not log_files:
        print(f"No .log files found in {input_folder}")
        return
    
    print(f"Processing {len(log_files)} log files from {input_folder.name}...")
    
    # Process each log file
    results = []
    for log_file in tqdm(log_files, desc=f"Processing {input_folder.name}"):
        funding_data, error = extract_funding_rates_from_log(log_file)
        
        if error:
            results.append((log_file.name, 0, error))
            continue
        
        if not funding_data:
            results.append((log_file.name, 0, "No funding rate data found"))
            continue
        
        # Create DataFrame
        df = pd.DataFrame(funding_data)
        
        # Create CSV filename from log filename (replace .log with _funding_rates.csv)
        csv_filename = log_file.stem + "_funding_rates.csv"
        csv_path = output_folder / csv_filename
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        results.append((log_file.name, len(funding_data), None))
    
    # Print summary
    print(f"\nSummary for {input_folder.name}:")
    total_records = 0
    for log_name, count, error in results:
        if error:
            print(f"  ⚠ {log_name}: {error}")
        else:
            csv_name = Path(log_name).stem + "_funding_rates.csv"
            print(f"  ✓ {log_name} -> {csv_name} ({count} funding rate records)")
            total_records += count
    print(f"Total: {total_records} funding rate records extracted across {len(log_files)} files\n")

def main():
    base_path = Path(__file__).parent
    
    # Process logs folder -> funding_rates folder
    logs_folder = base_path / "logs_new"
    funding_rates_folder = base_path / "new_funding_rates"
    
    if logs_folder.exists():
        print("\n" + "="*60)
        print("Extracting Funding Rates from logs folder")
        print("="*60)
        process_logs(logs_folder, funding_rates_folder)
    else:
        print(f"Warning: {logs_folder} does not exist")

if __name__ == "__main__":
    main()

