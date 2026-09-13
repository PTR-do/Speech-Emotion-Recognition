import numpy as np
import torch
import matplotlib.pyplot as plt

EMOTION_NAMES = [
    "happiness",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
]


def plot_training_history(history_path):
    history = torch.load(
        history_path,
        map_location="cpu",
        weights_only=False,
    )

    history_train = history["history_train"]
    history_val = history["history_val"]
    history_train_emotions = history["history_train_emotions"]
    history_val_emotions = history["history_val_emotions"]
    train_emotions = np.array(history_train_emotions)
    val_emotions = np.array(history_val_emotions)
    epochs = range(1, len(history_train) + 1)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 13),
    )
    axes[0].plot(
        epochs,
        history_train,
        label="Train",
    )
    axes[0].plot(
        epochs,
        history_val,
        label="Validation",
    )
    axes[0].set_title("Total Smooth L1 Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    for emotion_index, emotion_name in enumerate(EMOTION_NAMES):
        axes[1].plot(
            epochs,
            train_emotions[:, emotion_index],
            label=emotion_name,
        )
    axes[1].set_title("Train Smooth L1 Loss by Emotion")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)
    for emotion_index, emotion_name in enumerate(EMOTION_NAMES):
        axes[2].plot(
            epochs,
            val_emotions[:, emotion_index],
            label=emotion_name,
        )
    axes[2].set_title("Validation Smooth L1 Loss by Emotion")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()
    plt.show()
