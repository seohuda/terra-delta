import os
from pathlib import Path

import pytest
import torch

# CPU-only validation, bounded threads; no optimizer steps in the test suite.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(2)


@pytest.fixture(scope="session")
def official_checkpoint():
    path = Path(__file__).resolve().parents[1] / "baseline/original/03_illegal structure submission/assets/model/unet_r18_cd.pt"
    if not path.is_file():
        pytest.skip("Official checkpoint is local-only; retrieve per baseline/README.md")
    return path
