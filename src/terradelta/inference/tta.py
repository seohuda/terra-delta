"""Invert spatial transforms before averaging probabilities."""
import torch


TRANSFORMS = {"identity": (), "horizontal": (-1,), "vertical": (-2,)}


@torch.inference_mode()
def predict_probabilities(model, image, transforms=("identity",)):
    if not transforms or len(set(transforms)) != len(transforms):
        raise ValueError("TTA must contain unique transforms")
    results = []
    for name in transforms:
        if name not in TRANSFORMS:
            raise ValueError(f"Unsupported TTA transform: {name}")
        dims = TRANSFORMS[name]
        view = image.flip(dims) if dims else image
        probabilities = torch.softmax(model(view).float(), dim=1)
        results.append(probabilities.flip(dims) if dims else probabilities)
    return torch.stack(results).mean(0)
