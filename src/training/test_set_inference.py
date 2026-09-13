import torch
from tqdm.auto import tqdm

EMOTION_NAMES = [
    "happiness",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
]


def test_step(
    model,
    device,
    loss_fn,
    test_dataloader,
):
    model.eval()
    total_loss = 0.0
    emotion_loss = torch.zeros(
        len(EMOTION_NAMES),
        dtype=torch.float32,
    )
    progress_bar = tqdm(
        test_dataloader,
        desc="Test",
        unit="batch",
        leave=True,
    )
    with torch.no_grad():
        for batch in progress_bar:
            waveform = batch["waveform"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            pred = model(
                waveform,
                attention_mask,
            )
            loss_matrix = loss_fn(
                pred,
                target,
            )
            loss_emotion = loss_matrix.mean(dim=0)
            loss = loss_emotion.sum()
            total_loss += loss.item()
            emotion_loss += loss_emotion.cpu()
            progress_bar.set_postfix(loss=f"{total_loss / len(test_dataloader):.4f}")
    num_batches = len(test_dataloader)
    total_loss /= num_batches
    emotion_loss /= num_batches
    return total_loss, emotion_loss


def test_model(
    model,
    device,
    loss_fn,
    test_dataloader,
    history_path,
):
    test_loss, test_loss_emotions = test_step(
        model,
        device,
        loss_fn,
        test_dataloader,
    )
    history = {
        "test_loss": test_loss,
        "test_loss_emotions": test_loss_emotions.numpy(),
    }
    torch.save(history, history_path)
    print(f"Test Loss: {test_loss:.4f}")
    for emotion_name, emotion_loss in zip(
        EMOTION_NAMES,
        test_loss_emotions,
    ):
        print(f"{emotion_name}: {emotion_loss.item():.4f}")
    return test_loss, test_loss_emotions
