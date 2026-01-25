#!/usr/bin/env python3
"""Convert CSV time series data to GluonTS parquet format.

This script loads a CSV file with time series data, applies preprocessing
(outlier removal, interpolation), and outputs it in GluonTS-compatible parquet format.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from gluonts.dataset.arrow import ParquetWriter
from gluonts.dataset.pandas import PandasDataset
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_and_preprocess_csv(
    csv_path: str,
    target_column: str | None = None,
    date_column: str = "date",
    freq: str | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Load CSV and apply preprocessing: outlier removal, interpolation, cleaning.

    Args:
        csv_path: Path to input CSV file.
        target_column: Name of target column. If None, uses last numeric column.
        date_column: Name of date/timestamp column. Defaults to "date".
        freq: Pandas frequency string. If None, auto-detects from data.

    Returns:
        Tuple of (processed_dataframe, target_column_name, frequency_string)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path, sep=",")

    # Parse date column with flexible format handling
    if date_column not in df.columns:
        # Try common alternatives
        for col in ["Data", "Date", "datetime", "timestamp"]:
            if col in df.columns:
                date_column = col
                logger.info(f"Using '{date_column}' as date column")
                break
        else:
            raise ValueError(f"Date column '{date_column}' not found in CSV. Available columns: {df.columns.tolist()}")

    try:
        df[date_column] = pd.to_datetime(df[date_column])
    except Exception:
        try:
            df[date_column] = pd.to_datetime(df[date_column], dayfirst=True)
        except Exception as e:
            raise ValueError(f"Failed to parse date column '{date_column}': {e}")

    df.set_index(date_column, inplace=True)

    # Infer or set frequency
    if freq is None:
        freq = pd.infer_freq(df.index)
        if freq is None:
            # Calculate median time delta
            time_deltas = df.index.to_series().diff().dropna()
            median_delta = time_deltas.median()
            
            # Map common deltas to frequency strings
            if median_delta <= pd.Timedelta(hours=1):
                freq = "h"
            elif median_delta <= pd.Timedelta(hours=2):
                freq = "2h"
            elif median_delta <= pd.Timedelta(days=1):
                freq = "D"
            elif median_delta <= pd.Timedelta(weeks=1):
                freq = "W"
            else:
                freq = "ME"  # Default to month-end
            logger.warning(f"Could not infer frequency, using median delta-based guess: {freq}")
        logger.info(f"Inferred frequency: {freq}")
    else:
        logger.info(f"Using specified frequency: {freq}")

    # Drop iron-related columns if present (domain-specific cleanup)
    iron_columns = ["Fe C flot (%)", "% Iron Concentrate", "P_CONFLTTO_QQ_GLOBAL_FE"]
    for col in iron_columns:
        if col in df.columns:
            logger.info(f"Dropping column: {col}")
            df.drop(columns=[col], inplace=True)

    # Convert object columns (e.g., "1,23" -> 1.23)
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = df[col].str.replace(",", ".").astype(float)
                logger.info(f"Converted column '{col}' from comma decimal to float")
            except Exception:
                # Not a numeric column, drop it
                logger.warning(f"Dropping non-numeric column: {col}")
                df.drop(columns=[col], inplace=True)

    # Remove outliers using z-score (keep rows where all values have |z-score| < 3)
    logger.info("Removing outliers (z-score > 3)")
    original_rows = len(df)
    z_scores = np.abs(stats.zscore(df, nan_policy="omit"))
    df = df[(z_scores < 3).all(axis=1)]
    removed_rows = original_rows - len(df)
    if removed_rows > 0:
        logger.info(f"Removed {removed_rows} outlier rows ({removed_rows/original_rows*100:.1f}%)")

    # Interpolate missing values
    logger.info("Interpolating missing values")
    df.interpolate(method="linear", inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)  # Back-fill any remaining NaNs at the start
    
    # Drop rows with NaNs (if any remain) and resample to ensure uniform spacing
    if df.isnull().any().any():
        logger.warning(f"Dropping {df.isnull().any(axis=1).sum()} rows with remaining NaNs")
        df.dropna(inplace=True)
    
    # Ensure uniform spacing by resampling to the specified frequency
    # This fills any gaps created by outlier removal with interpolated values
    logger.info(f"Resampling to ensure uniform frequency: {freq}")
    df = df.resample(freq).mean().interpolate(method="linear").ffill().bfill()

    if df.empty:
        raise ValueError("DataFrame is empty after preprocessing")

    # Determine target column
    if target_column is None or target_column not in df.columns:
        if target_column is not None:
            logger.warning(f"Target column '{target_column}' not found, using last column")
        target_column = df.columns[-1]
        logger.info(f"Using target column: {target_column}")

    return df, target_column, freq


def convert_to_gluonts_parquet(
    df: pd.DataFrame,
    target_column: str,
    output_path: str,
    item_id: str,
    freq: str,
) -> None:
    """Convert processed DataFrame to GluonTS parquet format.

    Args:
        df: Preprocessed DataFrame with datetime index.
        target_column: Name of column containing target values.
        output_path: Path for output parquet file.
        item_id: Identifier for this time series.
        freq: Pandas frequency string (e.g., "h", "D", "ME").
    """
    logger.info(f"Converting to GluonTS long format (item_id: {item_id}, freq: {freq})")

    # Create long-format DataFrame
    long_df = pd.DataFrame({
        "timestamp": df.index,
        "target": df[target_column].values,
        "item_id": item_id,
    })

    logger.info(f"Time series length: {len(long_df)} periods")
    logger.info(f"Date range: {long_df['timestamp'].min()} to {long_df['timestamp'].max()}")
    logger.info(f"Target range: {long_df['target'].min():.4f} to {long_df['target'].max():.4f}")

    # Convert to GluonTS dataset
    gluonts_ds = PandasDataset.from_long_dataframe(
        long_df,
        item_id="item_id",
        timestamp="timestamp",
        target="target",
        freq=freq,
    )

    # Write to parquet
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing GluonTS parquet to {output_path}")
    ParquetWriter(metadata={"freq": freq}).write_to_file(gluonts_ds, output_file)
    logger.info("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV time series to GluonTS parquet format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output parquet file path",
    )
    parser.add_argument(
        "--target",
        "-t",
        default=None,
        help="Target column name (if not specified, uses last numeric column)",
    )
    parser.add_argument(
        "--date-column",
        "-d",
        default="date",
        help="Date/timestamp column name",
    )
    parser.add_argument(
        "--item-id",
        default=None,
        help="Item ID for the time series (if not specified, uses input filename stem)",
    )
    parser.add_argument(
        "--freq",
        "-f",
        default=None,
        help="Pandas frequency string (e.g., 'h', '2h', 'D', 'W', 'ME'). Auto-detected if not specified.",
    )

    args = parser.parse_args()

    # Default item_id to input filename
    if args.item_id is None:
        args.item_id = Path(args.input).stem
        logger.info(f"Using item_id from filename: {args.item_id}")

    # Load and preprocess
    df, target_column, freq = load_and_preprocess_csv(
        csv_path=args.input,
        target_column=args.target,
        date_column=args.date_column,
        freq=args.freq,
    )

    # Convert to GluonTS parquet
    convert_to_gluonts_parquet(
        df=df,
        target_column=target_column,
        output_path=args.output,
        item_id=args.item_id,
        freq=freq,
    )


if __name__ == "__main__":
    main()
