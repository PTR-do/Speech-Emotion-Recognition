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


def plot_test_results(history_path):
    history = torch.load(
        history_path,
        map_location="cpu",
        weights_only=False,
    )

    history_test = history["test_loss"]
    history_test_emotions = history["test_loss_emotions"]
    test_emotions = np.array(history_test_emotions)
    fig, ax = plt.subplots(
        figsize=(8, 5),
    )
    ax.bar(
        EMOTION_NAMES,
        test_emotions,
        color=[
            "green",
            "blue",
            "red",
            "purple",
            "orange",
            "brown",
        ],
    )
    ax.set_title(f"Test Smooth L1 Loss by Emotion — " f"Total Loss: {history_test:.4f}")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Loss")
    ax.grid(
        axis="y",
        alpha=0.3,
    )
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
