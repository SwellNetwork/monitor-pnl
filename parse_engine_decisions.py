#!/usr/bin/env python3
"""
Parse arbitrager log files and extract EngineDecision entries into a pandas DataFrame.

For each EngineDecision, extracts:
- Timestamp
- Funding rates for x and y legs
- Prices for x and y legs
- hours_open (time position has been open)
- x exchange name
- y exchange name
- Receiver (x or y)
- Which side is long and which side is short
"""
# python parse_engine_decisions.py --batch logs --output decisions_output_v2
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import sys


def parse_datetime_str(dt_str: str) -> Optional[datetime]:
    """Parse datetime string from log format: datetime.datetime(2025, 11, 8, 16, 30, 38, 351411, tzinfo=datetime.timezone.utc)"""
    try:
        # Extract the datetime components
        match = re.search(r'datetime\.datetime\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)', dt_str)
        if match:
            year, month, day, hour, minute, second, microsecond = map(int, match.groups())
            return datetime(year, month, day, hour, minute, second, microsecond)
    except Exception:
        pass
    return None


def extract_exchange_name(exchange_str: str) -> str:
    """Extract exchange name from <Exchange.Name: 'name'> format"""
    match = re.search(r"<Exchange\.(\w+):\s*'(\w+)'", exchange_str)
    if match:
        return match.group(2).lower()  # Return the lowercase name
    return ""


def extract_nested_parens(text: str, start_pos: int) -> Optional[str]:
    """Extract a nested parentheses expression starting at start_pos."""
    if start_pos >= len(text) or text[start_pos] != '(':
        return None
    
    depth = 0
    start = start_pos
    for i in range(start_pos, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def parse_engine_decision_line(line: str) -> Optional[Dict]:
    """
    Parse a single line containing an EngineDecision.
    Returns a dictionary with extracted fields, or None if parsing fails.
    """
    # First, check if this line contains an EngineDecision
    if "decision=EngineDecision" not in line or "processed market data" not in line:
        return None
    
    # Extract basic market data fields
    timestamp_match = re.search(r"timestamp=(datetime\.datetime\([^\)]+\))", line)
    if not timestamp_match:
        return None
    
    timestamp = parse_datetime_str(timestamp_match.group(1))
    
    # Extract x_market_data fields
    x_exchange_match = re.search(r"x_market_data=MarketData\(exchange=(<Exchange\.\w+:[^>]+>)", line)
    x_price_match = re.search(r"x_market_data=MarketData\([^)]*price=([\d\.eE\-]+)", line)
    x_fr_match = re.search(r"x_market_data=MarketData\([^)]*funding_rate=([\d\.eE\-]+)", line)
    
    # Extract y_market_data fields
    y_exchange_match = re.search(r"y_market_data=MarketData\(exchange=(<Exchange\.\w+:[^>]+>)", line)
    y_price_match = re.search(r"y_market_data=MarketData\([^)]*price=([\d\.eE\-]+)", line)
    y_fr_match = re.search(r"y_market_data=MarketData\([^)]*funding_rate=([\d\.eE\-]+)", line)
    
    # Extract action
    action_match = re.search(r"action=<EngineAction\.(\w+):\s*'(\w+)'", line)
    
    if not all([timestamp_match, x_exchange_match, x_price_match, x_fr_match, 
                y_exchange_match, y_price_match, y_fr_match, action_match]):
        return None
    
    x_exchange = extract_exchange_name(x_exchange_match.group(1))
    y_exchange = extract_exchange_name(y_exchange_match.group(1))
    x_price = float(x_price_match.group(1))
    y_price = float(y_price_match.group(1))
    x_funding_rate = float(x_fr_match.group(1))
    y_funding_rate = float(y_fr_match.group(1))
    action = action_match.group(2)  # 'enter', 'exit', or 'hold'
    
    # Initialize result dictionary
    result = {
        "timestamp": timestamp,
        "action": action,
        "x_exchange": x_exchange,
        "y_exchange": y_exchange,
        "x_price": x_price,
        "y_price": y_price,
        "x_funding_rate": x_funding_rate,
        "y_funding_rate": y_funding_rate,
        "hours_open": None,
        "receiver": None,
        "x_is_long": None,
        "y_is_long": None,
        "exit_reason": None,
        "reason": None,
    }
    
    # Extract exit_reason from ExitPlan's exit_info (for EXIT actions)
    if action == "exit":
        # Look for exit_reason in ExitPlan's exit_info
        # First try to find it in exit_info=ExitInfo(...)
        exit_info_start = line.find("exit_info=ExitInfo")
        if exit_info_start != -1:
            # Look for exit_reason after exit_info=ExitInfo
            exit_info_section = line[exit_info_start:exit_info_start+1000]  # Look at next 1000 chars
            exit_reason_match = re.search(r"exit_reason='([^']+)'", exit_info_section)
            if exit_reason_match:
                result["exit_reason"] = exit_reason_match.group(1)
        
        # Fallback: look for exit_reason in operations metadata
        if result["exit_reason"] is None:
            exit_reason_op_match = re.search(r"'exit_reason':\s*'([^']+)'", line)
            if exit_reason_op_match:
                result["exit_reason"] = exit_reason_op_match.group(1)
    
    # Extract reason from EngineDecision (usually None for HOLD/ENTER)
    reason_match = re.search(r"reason=(None|'[^']+'|\"[^\"]+\")", line)
    if reason_match:
        reason_str = reason_match.group(1)
        if reason_str != "None":
            # Remove quotes
            result["reason"] = reason_str.strip("'\"")
    
    # Extract plan information
    plan_match = re.search(r"plan=(EntryPlan|ExitPlan|None)", line)
    if plan_match and plan_match.group(1) != "None":
        plan_type = plan_match.group(1)
        plan_start = plan_match.end() - len(plan_match.group(1))
        
        # Extract receiver from plan
        receiver_match = re.search(r"receiver='([xy])'", line)
        if receiver_match:
            result["receiver"] = receiver_match.group(1)
        
        # Extract position info - search for is_long after x_position=Position or y_position=Position
        # We'll search for the pattern: x_position=Position(...is_long=True/False...)
        # Since Position has nested parentheses, we need to find is_long that appears after x_position=Position
        x_pos_start = line.find("x_position=Position")
        y_pos_start = line.find("y_position=Position")
        
        if x_pos_start != -1:
            # Find is_long after x_position=Position
            x_section = line[x_pos_start:x_pos_start+500]  # Look at next 500 chars
            x_long_match = re.search(r"is_long=(True|False)", x_section)
            if x_long_match:
                result["x_is_long"] = x_long_match.group(1) == "True"
        
        if y_pos_start != -1:
            # Find is_long after y_position=Position
            y_section = line[y_pos_start:y_pos_start+500]  # Look at next 500 chars
            y_long_match = re.search(r"is_long=(True|False)", y_section)
            if y_long_match:
                result["y_is_long"] = y_long_match.group(1) == "True"
    
    return result


def parse_engine_state_line(line: str) -> Optional[Dict]:
    """
    Parse an "engine state after cycle" line to extract hours_open and position info.
    Returns a dictionary with extracted fields, or None if parsing fails.
    """
    if "engine state after cycle" not in line:
        return None
    
    # Extract hours_open
    hours_match = re.search(r"hours_open=([\d\.eE\-]+)", line)
    if not hours_match:
        return None
    
    # Extract receiver
    receiver_match = re.search(r"last_receiver='([xy])'", line)
    if not receiver_match:
        return None
    
    # Extract x_position is_long - find x_position=Position and then find is_long after it
    x_pos_start = line.find("x_position=Position")
    y_pos_start = line.find("y_position=Position")
    
    x_is_long = None
    y_is_long = None
    
    if x_pos_start != -1:
        x_section = line[x_pos_start:x_pos_start+300]
        x_long_match = re.search(r"is_long=(True|False)", x_section)
        if x_long_match:
            x_is_long = x_long_match.group(1) == "True"
    
    if y_pos_start != -1:
        y_section = line[y_pos_start:y_pos_start+300]
        y_long_match = re.search(r"is_long=(True|False)", y_section)
        if y_long_match:
            y_is_long = y_long_match.group(1) == "True"
    
    if x_is_long is None or y_is_long is None:
        return None
    
    return {
        "hours_open": float(hours_match.group(1)),
        "receiver": receiver_match.group(1),
        "x_is_long": x_is_long,
        "y_is_long": y_is_long,
    }


def parse_log_file(log_file: Path) -> pd.DataFrame:
    """
    Parse a log file and extract all EngineDecision entries.
    Returns a pandas DataFrame with all extracted decisions.
    """
    decisions = []
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line contains an EngineDecision
        decision_data = parse_engine_decision_line(line)
        
        if decision_data:
            # For HOLD/EXIT decisions or when plan info is missing, get info from the next "engine state after cycle" line
            if decision_data["action"] in ["hold", "exit"] or \
               decision_data["hours_open"] is None or \
               decision_data["receiver"] is None or \
               decision_data["x_is_long"] is None or \
               decision_data["y_is_long"] is None:
                # Look ahead for the "engine state after cycle" line (usually within next 10 lines)
                for j in range(i + 1, min(i + 11, len(lines))):
                    state_data = parse_engine_state_line(lines[j])
                    if state_data:
                        # Fill in missing fields from state
                        if decision_data["hours_open"] is None:
                            decision_data["hours_open"] = state_data["hours_open"]
                        if decision_data["receiver"] is None:
                            decision_data["receiver"] = state_data["receiver"]
                        if decision_data["x_is_long"] is None:
                            decision_data["x_is_long"] = state_data["x_is_long"]
                        if decision_data["y_is_long"] is None:
                            decision_data["y_is_long"] = state_data["y_is_long"]
                        break
            
            decisions.append(decision_data)
        
        i += 1
    
    # Create DataFrame
    if decisions:
        df = pd.DataFrame(decisions)
        # Reorder columns for better readability
        column_order = [
            "timestamp", "action", "hours_open",
            "x_exchange", "y_exchange",
            "x_price", "y_price",
            "x_funding_rate", "y_funding_rate",
            "receiver", "x_is_long", "y_is_long",
            "exit_reason", "reason"
        ]
        # Only include columns that exist
        df = df[[col for col in column_order if col in df.columns]]
        return df
    else:
        return pd.DataFrame()


def process_folder(folder_path: Path, output_folder: Path):
    """Process all log files in a folder."""
    log_files = sorted(folder_path.glob("*.log"))
    
    if not log_files:
        print(f"No log files found in {folder_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing folder: {folder_path}")
    print(f"Found {len(log_files)} log files")
    print(f"{'='*60}\n")
    
    results = []
    
    # Use folder name as prefix to avoid conflicts
    folder_prefix = folder_path.name
    
    for log_file in log_files:
        print(f"Processing: {log_file.name}...", end=" ", flush=True)
        
        try:
            df = parse_log_file(log_file)
            
            if df.empty:
                print("No decisions found")
                results.append({
                    "file": log_file.name,
                    "status": "no_decisions",
                    "count": 0
                })
                continue
            
            # Create output filename with folder prefix to avoid conflicts
            output_file = output_folder / f"{folder_prefix}_{log_file.stem}_decisions.csv"
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            
            action_counts = df["action"].value_counts().to_dict()
            print(f"✓ {len(df)} decisions ({action_counts})")
            
            results.append({
                "file": log_file.name,
                "status": "success",
                "count": len(df),
                "output": str(output_file),
                "actions": action_counts
            })
            
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "file": log_file.name,
                "status": "error",
                "error": str(e)
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    no_decisions = [r for r in results if r["status"] == "no_decisions"]
    
    print(f"Successfully processed: {len(successful)} files")
    print(f"Failed: {len(failed)} files")
    print(f"No decisions found: {len(no_decisions)} files")
    
    if successful:
        total_decisions = sum(r["count"] for r in successful)
        print(f"\nTotal decisions extracted: {total_decisions}")
    
    return results


def main():
    """Main function to parse log files."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python parse_engine_decisions.py <log_file> [output_csv]")
        print("  Batch mode:  python parse_engine_decisions.py --batch <folder1> [folder2] ...")
        print("\nExample:")
        print("  python parse_engine_decisions.py logs_new/BTC_2025-11-09.log decisions.csv")
        print("  python parse_engine_decisions.py --batch logs_new logs")
        sys.exit(1)
    
    # Check if batch mode
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Error: Please specify at least one folder for batch processing")
            sys.exit(1)
        
        # Determine output folder (can be specified with --output flag)
        output_folder = Path("decisions_output")
        folder_args = sys.argv[2:]
        
        # Check if --output flag is used
        if "--output" in folder_args:
            output_idx = folder_args.index("--output")
            if output_idx + 1 < len(folder_args):
                output_folder = Path(folder_args[output_idx + 1])
                folder_args = [f for i, f in enumerate(folder_args) if i not in [output_idx, output_idx + 1]]
            else:
                print("Error: --output flag requires a folder path")
                sys.exit(1)
        
        # Create output folder
        output_folder.mkdir(exist_ok=True, parents=True)
        print(f"Output folder: {output_folder}")
        
        # Process each folder
        for folder_arg in folder_args:
            folder_path = Path(folder_arg)
            if not folder_path.exists():
                print(f"Warning: Folder not found: {folder_path}, skipping...")
                continue
            process_folder(folder_path, output_folder)
        
        return
    
    # Single file mode
    log_file_path = Path(sys.argv[1])
    if not log_file_path.exists():
        print(f"Error: Log file not found: {log_file_path}")
        sys.exit(1)
    
    print(f"Parsing log file: {log_file_path}")
    df = parse_log_file(log_file_path)
    
    if df.empty:
        print("No EngineDecision entries found in the log file.")
        sys.exit(0)
    
    print(f"\nFound {len(df)} EngineDecision entries")
    print(f"\nAction distribution:")
    print(df["action"].value_counts())
    
    # Determine output file
    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        # Default: same name as log file but with .csv extension
        output_file = log_file_path.parent / f"{log_file_path.stem}_decisions.csv"
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")
    
    # Display first few rows
    print("\nFirst few rows:")
    print(df.head(10).to_string())
    
    return df


if __name__ == "__main__":
    main()

