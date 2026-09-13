"""
Waveform extraction function.
This module contains the function required to:
- extract audio waveforms from a formatted dataframe
- process audio segments using multiple CPU processes
- preserve the original dataframe order
- associate each waveform with its emotion target
- save the resulting dataset
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
import torch
import h5py
from src.audio.audio_processing import prepare_audio

EMOTION_COLUMNS = [
    "happiness",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
]


# Extract one waveform and its corresponding emotion target.
# This function is executed by a worker process.
def _extract_waveform(task):
    index, audio_path, start, end, target = task
    try:
        waveform = prepare_audio(
            audio_path=audio_path,
            start=start,
            end=end,
        )
        waveform = waveform.detach().cpu().numpy()
        return index, waveform, target

    except Exception as exc:
        raise RuntimeError(
            f"Failed processing segment: "
            f"index={index}, "
            f"file={audio_path}, "
            f"start={start}, "
            f"end={end}"
        ) from exc


# Extract all audio waveforms from a dataframe using multiple CPU processes.
def waveform_extraction(
    dataframe,
    audio_root,
    output_path=None,
    num_workers=4,
):
    """
    Parameters
    dataframe : pandas.DataFrame
    audio_root : str or pathlib.Path
        Root directory containing the audio files.
    output_path : str or pathlib.Path, optional
        Destination .h5 file.
    num_workers : int, default=4
        Number of worker processes used for waveform extraction.
    -------
    Returns
        Path to the output .h5 file when output_path is specified.
        Otherwise, dictionary containing:
        waveforms : list[torch.Tensor]
            Extracted CPU waveforms in the same order as the
            input dataframe.
        targets : torch.Tensor
            Emotion targets with shape [N, 6].
    """

    if num_workers < 1:
        raise ValueError(f"'num_workers' must be >= 1, got {num_workers}")
    dataframe = dataframe.reset_index(drop=True)
    audio_root = Path(audio_root)
    # Validate dataframe
    required_columns = [
        "file",
        "start",
        "end",
        *EMOTION_COLUMNS,
    ]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError("Missing required dataframe columns: " f"{missing_columns}")

    # Build worker tasks.
    tasks = []
    for index, row in dataframe.iterrows():
        target = [float(row[column]) for column in EMOTION_COLUMNS]
        tasks.append(
            (
                index,
                audio_root / row["file"],
                float(row["start"]),
                float(row["end"]),
                target,
            )
        )

    # If no output path is specified, fall back to the in-memory
    # representation required by the function return value.
    if output_path is None:
        results = [None] * len(tasks)
        with ProcessPoolExecutor(
            max_workers=num_workers,
        ) as executor:
            pending = set()
            task_iterator = iter(tasks)
            # Keep only a small number of tasks in flight to avoid excessive memory consumption.
            for _ in range(min(num_workers * 2, len(tasks))):
                try:
                    task = next(task_iterator)
                except StopIteration:
                    break
                pending.add(executor.submit(_extract_waveform, task))

            while pending:
                completed, pending = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )

                for future in completed:
                    index, waveform, target = future.result()
                    results[index] = (
                        torch.from_numpy(waveform),
                        torch.tensor(
                            target,
                            dtype=torch.float32,
                        ),
                    )

                    try:
                        task = next(task_iterator)
                    except StopIteration:
                        continue

                    pending.add(
                        executor.submit(
                            _extract_waveform,
                            task,
                        )
                    )

        # Assemble dataset in the original dataframe order.
        waveforms = [result[0] for result in results]
        targets = torch.stack([result[1] for result in results])
        dataset = {
            "waveforms": waveforms,
            "targets": targets,
        }
        return dataset

    # Disk-backed extraction
    output_path = Path(output_path)

    # The output directory is created before starting the workers.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    pending_results = {}
    next_index = 0
    total_samples = 0
    offsets = np.zeros(
        len(tasks) + 1,
        dtype=np.int64,
    )

    try:
        with h5py.File(
            temporary_path,
            "w",
        ) as h5_file:
            waveform_dataset = h5_file.create_dataset(
                "waveforms",
                shape=(0,),
                maxshape=(None,),
                dtype="float32",
                chunks=True,
            )
            target_dataset = h5_file.create_dataset(
                "targets",
                shape=(len(tasks), len(EMOTION_COLUMNS)),
                dtype="float32",
            )
            offset_dataset = h5_file.create_dataset(
                "offsets",
                shape=(len(tasks) + 1,),
                dtype="int64",
            )

            with ProcessPoolExecutor(
                max_workers=num_workers,
            ) as executor:
                pending = set()
                task_iterator = iter(tasks)
                # Keep only a small number of tasks in flight to avoid excessive memory consumption.
                for _ in range(min(num_workers * 2, len(tasks))):
                    try:
                        task = next(task_iterator)
                    except StopIteration:
                        break

                    pending.add(
                        executor.submit(
                            _extract_waveform,
                            task,
                        )
                    )

                while pending:
                    completed, pending = wait(
                        pending,
                        return_when=FIRST_COMPLETED,
                    )

                    for future in completed:
                        index, waveform, target = future.result()
                        pending_results[index] = (
                            waveform,
                            target,
                        )

                        try:
                            task = next(task_iterator)
                        except StopIteration:
                            pass
                        else:
                            pending.add(
                                executor.submit(
                                    _extract_waveform,
                                    task,
                                )
                            )

                    # Write only consecutive indices.
                    while next_index in pending_results:
                        waveform, target = pending_results.pop(next_index)
                        waveform = np.asarray(
                            waveform,
                            dtype=np.float32,
                        )
                        waveform_length = waveform.shape[0]
                        new_total_samples = total_samples + waveform_length
                        waveform_dataset.resize((new_total_samples,))
                        waveform_dataset[total_samples:new_total_samples] = waveform
                        target_dataset[next_index] = np.asarray(
                            target,
                            dtype=np.float32,
                        )
                        offsets[next_index] = total_samples
                        total_samples = new_total_samples
                        next_index += 1

            # Final offset points immediately after the final waveform.
            offsets[next_index] = total_samples
            offset_dataset[:] = offsets
            h5_file.attrs["num_samples"] = total_samples
            h5_file.attrs["num_segments"] = len(tasks)
            h5_file.attrs["sample_rate"] = 16000

        # Replace the previous dataset only after the complete
        # extraction has finished successfully.
        temporary_path.replace(
            output_path,
        )

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    return output_path
