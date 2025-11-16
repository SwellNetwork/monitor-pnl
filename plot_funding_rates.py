"""Plot Funding Rates over Time and compute mean rates."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_funding_rates(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    return df


def compute_mean_rates(df: pd.DataFrame) -> dict:
    mean_hyper = df["hyperliquid_funding_rate"].mean() * 100  # convert to %
    mean_lighter = df["lighter_funding_rate"].mean() * 100
    return {
        "hyperliquid_mean_pct": mean_hyper,
        "lighter_mean_pct": mean_lighter,
    }


def plot_funding_rates(df: pd.DataFrame, title: str, output_path: Path) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(df["timestamp"], df["hyperliquid_funding_rate"] * 100, label="Hyperliquid", linewidth=1.2)
    plt.plot(df["timestamp"], df["lighter_funding_rate"] * 100, label="Lighter", linewidth=1.2)
    plt.title(title)
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("Funding Rate (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def process_file(csv_path: Path, output_dir: Path) -> dict:
    df = load_funding_rates(csv_path)
    means = compute_mean_rates(df)

    # Plot
    plot_title = f"Funding Rates for {csv_path.stem}"
    plot_path = output_dir / f"{csv_path.stem}_funding_rates.png"
    plot_funding_rates(df, plot_title, plot_path)

    return {
        "file": csv_path.name,
        "hyperliquid_mean_pct": means["hyperliquid_mean_pct"],
        "lighter_mean_pct": means["lighter_mean_pct"],
        "plot": plot_path.name,
    }


def main():
    parser = argparse.ArgumentParser(description="Plot funding rates and compute mean values")
    parser.add_argument(
        "--input",
        type=str,
        default="new_funding_rates",
        help="Directory containing funding rate CSV files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="funding_rate_plots",
        help="Directory to save plots.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    summaries = []
    for csv_path in csv_files:
        summary = process_file(csv_path, output_dir)
        summaries.append(summary)
        print(
            f"Processed {csv_path.name}: Hyperliquid mean={summary['hyperliquid_mean_pct']:.6f}% | "
            f"Lighter mean={summary['lighter_mean_pct']:.6f}% -> plot {summary['plot']}"
        )

    # Save summary CSV
    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "funding_rate_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
