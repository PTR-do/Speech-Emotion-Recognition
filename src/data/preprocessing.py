"""
Preprocessing utilities.
This module contains the functions:
- dataset_info: inspect the audformat database
- values_normalization: normalize emotion values using min-max normalization
- table_format: convert MultiIndex annotation tables into flat tables
- save_processed_table: save processed tables
"""

from pathlib import Path
import numpy as np
import pandas as pd
import audformat

EMOTION_COLUMNS = [
    "happiness",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
]


def dataset_info(db: audformat.Database) -> None:
    """
    Parameters
    db : audformat.Database
        Loaded audformat database.
    """

    print("=" * 80)
    print("DATASET INFORMATION")
    print(f"\nName: {db.name}")
    if db.description:
        print(f"\nDescription:\n{db.description}")
    print("\nTables:")
    for table_name in db.tables.keys():
        print(f"  - {table_name}")
    print(f"\nNumber of tables: {len(db.tables)}")
    print("=" * 80)


def values_normalization(
    df: pd.DataFrame,
    emotion_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Parameters
    df : pd.DataFrame
        Input emotion table.
    emotion_columns : list[str] | None
        Columns to normalize.
    ------
    Returns
    pd.DataFrame
    """

    normalized_df = df.copy()
    # Select columns to normalize
    if emotion_columns is None:
        columns = normalized_df.select_dtypes(include=np.number).columns.tolist()
    else:
        columns = emotion_columns
    if len(columns) == 0:
        raise ValueError("No numeric columns found for normalization.")
    # Compute global minimum and maximum
    global_min = normalized_df[columns].min().min()
    global_max = normalized_df[columns].max().max()
    if global_min == global_max:
        raise ValueError(
            "Cannot apply min-max normalization: "
            "minimum and maximum values are identical."
        )
    # Apply min-max normalization
    normalized_df[columns] = (normalized_df[columns] - global_min) / (
        global_max - global_min
    )
    print("Normalization completed.")
    print(f"Global min: {global_min}")
    print(f"Global max: {global_max}")
    return normalized_df


def table_format(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parameters
    df : pd.DataFrame
        Input DataFrame with a MultiIndex containing:
        file, start, end.
    -------
    Returns
    pd.DataFrame
        Flat DataFrame with:
        file
        start
        end
        duration
        <emotion columns>
    """

    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError("The input DataFrame must have a MultiIndex.")
    # Verify that the MultiIndex contains the expected levels
    missing_levels = [
        level
        for level in [
            "file",
            "start",
            "end",
        ]
        if level not in df.index.names
    ]
    if missing_levels:
        raise ValueError(f"Missing required MultiIndex levels: " f"{missing_levels}")
    # Explicitly extract the MultiIndex levels into new columns
    formatted_df = df.copy()
    formatted_df.insert(
        0,
        "file",
        formatted_df.index.get_level_values("file"),
    )
    formatted_df.insert(
        1,
        "start",
        formatted_df.index.get_level_values("start"),
    )
    formatted_df.insert(
        2,
        "end",
        formatted_df.index.get_level_values("end"),
    )
    # Remove the original MultiIndex and assign a progressive index
    formatted_df = formatted_df.reset_index(drop=True)
    # Convert timestamps to pandas Timedelta
    formatted_df["start"] = pd.to_timedelta(formatted_df["start"])
    formatted_df["end"] = pd.to_timedelta(formatted_df["end"])
    # Convert Timedelta to seconds
    formatted_df["start"] = formatted_df["start"].dt.total_seconds()
    formatted_df["end"] = formatted_df["end"].dt.total_seconds()
    # Handle negative start timestamps
    negative_start_mask = formatted_df["start"] < 0
    negative_start_count = negative_start_mask.sum()
    if negative_start_count > 0:
        print(f"Warning: {negative_start_count} negative " f"start timestamp(s) found.")
        print("Negative start timestamps will be set to 0 seconds.")
        formatted_df.loc[negative_start_mask, "start"] = 0.0
    # Segment duration
    formatted_df["duration"] = formatted_df["end"] - formatted_df["start"]
    # Reset index
    formatted_df = formatted_df.reset_index(drop=True)
    return formatted_df


def save_processed_table(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Parameters
    df : pd.DataFrame
        Processed DataFrame to save.
    output_path : str | Path
        Destination path.
    """
    output_path = Path(output_path)
    # Create parent directories if necessary
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif suffix in {".pkl", ".pickle"}:
        df.to_pickle(output_path)
    else:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            "Use .parquet, .csv, or .pkl/.pickle."
        )
    print(f"Processed table saved to: {output_path}")
