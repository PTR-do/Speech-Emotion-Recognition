"""
Extract all audio segments belonging to a specific dataset split in a new dataframe.
"""

import pandas as pd


def get_split_dataframe(
    processed_df: pd.DataFrame,
    split_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parameters
    processed_df : pd.DataFrame
        Preprocessed and formatted emotion dataframe.
    split_table : pd.DataFrame
        Table containing the audio files belonging to a dataset split.
    -------
    Returns
    pd.DataFrame
        Dataframe containing only the segments whose audio file belongs
        to the specified split.
    """

    if "file" not in processed_df.columns:
        raise ValueError("processed_df must contain a 'file' column.")
    # Extract audio file identifiers from the split table.
    split_files = split_table.index
    split_files = pd.Index(split_files).unique()
    # Keep every segment belonging to one of the selected audio files.
    split_df = processed_df[processed_df["file"].isin(split_files)].copy()
    split_df.reset_index(drop=True, inplace=True)
    return split_df
