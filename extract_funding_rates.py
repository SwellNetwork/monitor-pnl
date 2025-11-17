# Extract Funding Rates from Decision CSV Files
# Author: CatKhanh, 2025-11-11
# Generates CSV files with timestamp, hyperliquid funding rate, lighter funding rate

from pathlib import Path
import pandas as pd
from tqdm import tqdm

def extract_funding_rates_from_csv(csv_file: Path):
    """Extract funding rates from a single decision CSV file."""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file, parse_dates=["timestamp"])
        
        # Extract funding rate data
        funding_rate_data = []
        seen_timestamps = set()  # To avoid duplicates
        
        for _, row in df.iterrows():
            timestamp = row["timestamp"]
            
            # Avoid duplicates (same timestamp)
            ts_key = pd.Timestamp(timestamp).isoformat()
            if ts_key in seen_timestamps:
                continue
            seen_timestamps.add(ts_key)
            
            # Get funding rates based on exchange names
            x_exchange = str(row["x_exchange"]).lower()
            y_exchange = str(row["y_exchange"]).lower()
            x_fr = float(row["x_funding_rate"])
            y_fr = float(row["y_funding_rate"])
            
            # Determine which is hyperliquid and which is lighter
            hyperliquid_fr = None
            lighter_fr = None
            
            if x_exchange == "hyperliquid":
                hyperliquid_fr = x_fr
            elif y_exchange == "hyperliquid":
                hyperliquid_fr = y_fr
            
            if x_exchange == "lighter":
                lighter_fr = x_fr
            elif y_exchange == "lighter":
                lighter_fr = y_fr
            
            # Only add if we found both rates
            if hyperliquid_fr is not None and lighter_fr is not None:
                funding_rate_data.append({
                    "timestamp": timestamp,
                    "hyperliquid_funding_rate": hyperliquid_fr,
                    "lighter_funding_rate": lighter_fr
                })
        
        # Sort by timestamp
        funding_rate_data.sort(key=lambda x: x["timestamp"])
        
        return funding_rate_data, None
    except Exception as e:
        return [], str(e)

def process_csvs(input_folder: Path, output_folder: Path):
    """Process all decision CSV files in input_folder and save funding rate CSVs to output_folder."""
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all decision CSV files in input folder
    csv_files = sorted(input_folder.glob("*_decisions.csv"))
    if not csv_files:
        print(f"No *_decisions.csv files found in {input_folder}")
        return
    
    print(f"Processing {len(csv_files)} CSV files from {input_folder.name}...")
    
    # Process each CSV file
    results = []
    for csv_file in tqdm(csv_files, desc=f"Processing {input_folder.name}"):
        funding_data, error = extract_funding_rates_from_csv(csv_file)
        
        if error:
            results.append((csv_file.name, 0, error))
            continue
        
        if not funding_data:
            results.append((csv_file.name, 0, "No funding rate data found"))
            continue
        
        # Create DataFrame
        df = pd.DataFrame(funding_data)
        
        # Create CSV filename from decision CSV filename
        # e.g., logs_v3_BTC_2025-11-09_decisions.csv -> BTC_2025-11-09_funding_rates.csv
        # e.g., old_logs_BTC_2025-11-09_decisions.csv -> BTC_2025-11-09_funding_rates.csv
        # Extract symbol and date from filename
        stem = csv_file.stem.replace("_decisions", "")
        # Remove logs_v3_ prefix if present
        if stem.startswith("logs_v3_"):
            stem = stem.replace("logs_v3_", "")
        # Remove old_logs_ prefix if present
        if stem.startswith("old_logs_"):
            stem = stem.replace("old_logs_", "")
        csv_filename = stem + "_funding_rates.csv"
        csv_path = output_folder / csv_filename
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        results.append((csv_file.name, len(funding_data), None))
    
    # Print summary
    print(f"\nSummary for {input_folder.name}:")
    total_records = 0
    for csv_name, count, error in results:
        if error:
            print(f"  ⚠ {csv_name}: {error}")
        else:
            # Extract output filename
            stem = Path(csv_name).stem.replace("_decisions", "")
            if stem.startswith("logs_v3_"):
                stem = stem.replace("logs_v3_", "")
            if stem.startswith("old_logs_"):
                stem = stem.replace("old_logs_", "")
            output_name = stem + "_funding_rates.csv"
            print(f"  ✓ {csv_name} -> {output_name} ({count} funding rate records)")
            total_records += count
    print(f"Total: {total_records} funding rate records extracted across {len(csv_files)} files\n")

def main():
    base_path = Path(__file__).parent
    
    # Process exit_only folder -> funding_rates folder
    input_folder = base_path / "decisions_output" / "exit_only"
    funding_rates_folder = base_path / "funding_rates"
    
    if input_folder.exists():
        print("\n" + "="*60)
        print("Extracting Funding Rates from decision CSV files")
        print("="*60)
        process_csvs(input_folder, funding_rates_folder)
    else:
        print(f"Warning: {input_folder} does not exist")

if __name__ == "__main__":
    main()
