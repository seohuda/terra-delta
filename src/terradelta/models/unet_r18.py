"""Official SMP UNet; no wrapper prefix is added to checkpoint keys."""
import segmentation_models_pytorch as smp


class TerraDeltaUNetR18(smp.Unet):
    def __init__(self, encoder_weights=None):
        super().__init__(encoder_name="resnet18", encoder_weights=encoder_weights, in_channels=6, classes=3)
