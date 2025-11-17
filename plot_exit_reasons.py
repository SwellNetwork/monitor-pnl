#!/usr/bin/env python3
"""
Create bar charts showing exit reason distribution for each CSV file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import os

def plot_exit_reasons_for_file(csv_file: Path, output_dir: Path):
    """Create a bar chart of exit reasons for a single CSV file."""
    df = pd.read_csv(csv_file)
    
    # Filter to only EXIT actions
    exit_df = df[df['action'] == 'exit']
    
    if len(exit_df) == 0:
        print(f"No exit decisions found in {csv_file.name}")
        return None
    
    # Count exit reasons
    exit_reasons = exit_df['exit_reason'].value_counts()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(exit_reasons.index, exit_reasons.values, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Exit Reason', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Exit Reasons Distribution\n{csv_file.stem}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / f"{csv_file.stem}_exit_reasons.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created chart: {output_file.name} ({len(exit_df)} exit decisions)")
    return output_file

def plot_all_exit_reasons(csv_pattern: str, output_dir: str = "exit_reason_plots"):
    """Create bar charts for all CSV files matching the pattern."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all CSV files
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        print(f"No CSV files found matching pattern: {csv_pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    print(f"Output directory: {output_path}\n")
    
    created_charts = []
    
    for csv_file in csv_files:
        csv_path = Path(csv_file)
        chart_path = plot_exit_reasons_for_file(csv_path, output_path)
        if chart_path:
            created_charts.append(chart_path)
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")
    print(f"Total CSV files processed: {len(csv_files)}")
    print(f"Charts created: {len(created_charts)}")
    print(f"Output directory: {output_path}")
    
    return created_charts

def create_summary_chart(csv_pattern: str, output_file: str = "exit_reason_summary.png"):
    """Create a combined summary chart showing exit reasons across all files."""
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        print(f"No CSV files found matching pattern: {csv_pattern}")
        return
    
    # Collect all exit reasons
    all_reasons = {}
    file_names = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        exit_df = df[df['action'] == 'exit']
        
        if len(exit_df) > 0:
            file_name = Path(csv_file).stem.replace('_decisions', '')
            file_names.append(file_name)
            reasons = exit_df['exit_reason'].value_counts().to_dict()
            all_reasons[file_name] = reasons
    
    if not all_reasons:
        print("No exit decisions found in any files")
        return
    
    # Get all unique exit reasons
    unique_reasons = set()
    for reasons_dict in all_reasons.values():
        unique_reasons.update(reasons_dict.keys())
    unique_reasons = sorted(unique_reasons)
    
    # Create a DataFrame for easier plotting
    summary_data = []
    for file_name in file_names:
        row = {'file': file_name}
        for reason in unique_reasons:
            row[reason] = all_reasons.get(file_name, {}).get(reason, 0)
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index('file')
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(max(12, len(file_names) * 0.8), 8))
    
    x = range(len(file_names))
    width = 0.8 / len(unique_reasons)
    
    colors = plt.cm.Set3(range(len(unique_reasons)))
    
    for i, reason in enumerate(unique_reasons):
        offset = (i - len(unique_reasons) / 2) * width + width / 2
        values = summary_df[reason].values
        bars = ax.bar([xi + offset for xi in x], values, width, 
                      label=reason, color=colors[i], edgecolor='black', alpha=0.7)
        
        # Add value labels
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., val,
                       f'{int(val)}' if val >= 1 else '',
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('File', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Exit Reasons Distribution Across All Files', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(file_names, rotation=45, ha='right')
    ax.legend(title='Exit Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Created summary chart: {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python plot_exit_reasons.py <csv_pattern> [output_dir]")
        print("\nExample:")
        print("  python plot_exit_reasons.py 'decisions_output/old_logs_*.csv'")
        print("  python plot_exit_reasons.py 'decisions_output/old_logs_*.csv' exit_plots")
        sys.exit(1)
    
    csv_pattern = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "exit_reason_plots"
    
    # Create individual charts
    plot_all_exit_reasons(csv_pattern, output_dir)
    
    # Create summary chart
    summary_file = Path(output_dir) / "exit_reason_summary.png"
    create_summary_chart(csv_pattern, str(summary_file))

