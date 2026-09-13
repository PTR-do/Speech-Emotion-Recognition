import torch


def load_checkpoint(model, checkpoint_path, device):
    print("\nLoading checkpoint:")
    print(f"  {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(
        state_dict,
        strict=True,
    )
    print("Checkpoint loaded successfully.")
    return model


def load_head_checkpoint(
    regressor,
    checkpoint_path,
    device,
):
    print("\nLoading regressor head checkpoint:")
    print(f"  {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    regressor.load_state_dict(
        state_dict,
        strict=True,
    )
    print("Regressor head checkpoint loaded successfully.")
    return regressor


def load_head_ap_checkpoint(
    attention_pooling,
    regressor,
    checkpoint_path,
    device,
):
    print("\nLoading attention pooling + regressor checkpoint:")
    print(f"  {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    # Extract AttentionPooling parameters
    pooling_state_dict = {
        key.removeprefix("pooling."): value
        for key, value in state_dict.items()
        if key.startswith("pooling.")
    }
    # Extract EmotionRegressor parameters
    regressor_state_dict = {
        key.removeprefix("regressor."): value
        for key, value in state_dict.items()
        if key.startswith("regressor.")
    }
    # Verify that both components are present
    if not pooling_state_dict:
        raise KeyError("No AttentionPooling parameters found in checkpoint.")
    if not regressor_state_dict:
        raise KeyError("No EmotionRegressor parameters found in checkpoint.")
    # Load AttentionPooling
    attention_pooling.load_state_dict(
        pooling_state_dict,
        strict=True,
    )
    # Load EmotionRegressor
    regressor.load_state_dict(
        regressor_state_dict,
        strict=True,
    )
    print("AttentionPooling + EmotionRegressor " "checkpoint loaded successfully.")
    print(f"  AttentionPooling parameters: {len(pooling_state_dict)} tensors")
    print(f"  EmotionRegressor parameters: {len(regressor_state_dict)} tensors")
    return attention_pooling, regressor
