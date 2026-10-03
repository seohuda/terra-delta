from .unet_r18 import TerraDeltaUNetR18


def build_model(config=None):
    config = config or {}
    if "model" in config:
        config = config["model"]
    if config.get("encoder", "resnet18") != "resnet18":
        raise ValueError("Only baseline-compatible resnet18 is supported")
    if config.get("in_channels", 6) != 6 or config.get("classes", 3) != 3:
        raise ValueError("TerraDelta requires 6 input channels and 3 output classes")
    # Explicit opt-in only; inference and submission always use None.
    return TerraDeltaUNetR18(encoder_weights=config.get("encoder_weights"))
