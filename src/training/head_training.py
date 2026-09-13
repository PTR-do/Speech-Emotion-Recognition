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


def train_step_head(
    model,
    device,
    loss_fn,
    optimizer,
    train_dataloader,
):
    model.train()
    total_loss = 0.0
    emotion_loss = torch.zeros(
        len(EMOTION_NAMES),
        dtype=torch.float32,
    )
    progress_bar = tqdm(
        train_dataloader,
        desc="Training",
        unit="batch",
    )
    for batch in progress_bar:
        features, target = batch
        features = features.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        pred = model(features)
        loss_matrix = loss_fn(pred, target)
        loss_per_emotion = loss_matrix.mean(dim=0)
        loss = loss_per_emotion.sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        emotion_loss += loss_per_emotion.detach().cpu()
        progress_bar.set_postfix(
            loss=f"{total_loss / len(train_dataloader):.4f}",
        )
    num_batches = len(train_dataloader)
    total_loss /= num_batches
    emotion_loss /= num_batches
    return total_loss, emotion_loss


def test_step_head(
    model,
    device,
    loss_fn,
    val_dataloader,
):
    model.eval()
    total_loss = 0.0
    emotion_loss = torch.zeros(
        len(EMOTION_NAMES),
        dtype=torch.float32,
    )
    progress_bar = tqdm(
        val_dataloader,
        desc="Validation",
        unit="batch",
    )
    with torch.no_grad():
        for batch in progress_bar:
            features, target = batch
            features = features.to(device)
            target = target.to(device)
            pred = model(features)
            loss_matrix = loss_fn(pred, target)
            loss_per_emotion = loss_matrix.mean(dim=0)
            loss = loss_per_emotion.sum()
            total_loss += loss.item()
            emotion_loss += loss_per_emotion.cpu()
            progress_bar.set_postfix(
                loss=f"{total_loss / len(val_dataloader):.4f}",
            )
    num_batches = len(val_dataloader)
    total_loss /= num_batches
    emotion_loss /= num_batches
    return total_loss, emotion_loss


def train_model_head(
    model,
    device,
    epochs,
    loss_fn,
    optimizer,
    scheduler,
    train_dataloader,
    val_dataloader,
    model_path,
    history_path,
    early_stopping=None,
):
    history_train = []
    history_val = []
    history_train_emotions = []
    history_val_emotions = []
    best_val_loss = torch.inf
    for epoch in range(1, epochs + 1):
        train_loss, train_loss_emotions = train_step_head(
            model,
            device,
            loss_fn,
            optimizer,
            train_dataloader,
        )
        val_loss, val_loss_emotions = test_step_head(
            model,
            device,
            loss_fn,
            val_dataloader,
        )
        history_train.append(train_loss)
        history_val.append(val_loss)
        history_train_emotions.append(train_loss_emotions.numpy())
        history_val_emotions.append(val_loss_emotions.numpy())
        torch.save(
            {
                "history_train": history_train,
                "history_val": history_val,
                "history_train_emotions": history_train_emotions,
                "history_val_emotions": history_val_emotions,
            },
            history_path,
        )
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
        if early_stopping is not None:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break
        if scheduler is not None:
            scheduler.step()
    return history_train, history_val, history_train_emotions, history_val_emotions
