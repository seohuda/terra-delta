"""Prove loss of labels, spatial leakage and output overwrite cannot be silent."""
import csv

import numpy as np
from PIL import Image
import pytest
import torch

from terradelta.data.dataset import ChangeDataset
from terradelta.data.transforms import PairedTransform
from terradelta.training.splits import prepare_datasets
from terradelta.training.losses import build_loss
from terradelta.training.trainer import validate_batch
from terradelta.inference.predictor import prepare_pair, Predictor
from terradelta.utils.io import training_run_lock


def test_partial_review_mask_and_auxiliary_labels(tmp_path):
    pre = np.full((32,32,3),127,np.uint8)
    building = np.zeros((32,32),np.uint8)
    building[5:15,5:15] = 255
    valid = np.zeros((32,32),np.uint8)
    valid[2:25,2:25] = 255
    for name,image in (("pre",pre),("post",pre),("building",building),("valid",valid)):
        Image.fromarray(image).save(tmp_path / f"{name}.png")
    manifest = tmp_path / "data.csv"
    manifest.write_text("id,pre,post,new_building,tree_removal,review_mask\n1,pre.png,post.png,building.png,absent,valid.png\n")
    transform = PairedTransform(seed=3,photometric_probability=0,affine_probability=1)
    sample = ChangeDataset(manifest,transform=transform,return_auxiliary=True)[0]
    assert sample["auxiliary_mask"].shape == (2,32,32)
    assert (~sample["valid_mask"]).any() and sample["valid_mask"].any()
    assert torch.all(sample["mask"][~sample["valid_mask"]] == -100)
    assert torch.equal(sample["auxiliary_mask"][0],sample["mask"]==1)
    validate_batch({"image":sample["image"][None],"mask":sample["mask"][None]})
    logits = torch.zeros(1,3,32,32)
    loss = build_loss("ce")(logits,sample["mask"][None])
    altered = logits.clone()
    altered[0,:,~sample["valid_mask"]] = torch.tensor([100.,-100.,50.])[:,None]
    assert build_loss("ce")(altered,sample["mask"][None]) == loss


def test_training_split_rejects_adjacent_regions(tmp_path):
    # Four regions: touching A/B must stay together even with a random strategy.
    rows = [{"id":str(i),"region_id":str(i),"source":"synthetic","license_status":"commercial_ok",
             "bounds":f"[{i*10},0,{i*10+10},10]","crs":"EPSG:32610"} for i in range(2)]
    paths = [tmp_path / "train.csv",tmp_path / "val.csv"]
    for row,path in zip(rows,paths):
        with path.open("w",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    for strategy in ("region","random"):
        with pytest.raises(ValueError,match="adjacent"):
            prepare_datasets({"data":{"train_manifest":str(paths[0]),"val_manifest":str(paths[1]),
                                      "split":{"strategy":strategy}}},dataset_factory=lambda *a,**k:None)


def test_predictor_registration_is_shared_by_validation_and_directory():
    import cv2
    rng = np.random.default_rng(2)
    pre = rng.integers(0,256,(64,64,3),dtype=np.uint8)
    post = cv2.warpAffine(pre,np.float32([[1,0,1],[0,1,1]]),(64,64),borderMode=cv2.BORDER_REFLECT_101)
    class Capture(torch.nn.Module):
        def forward(self,x):
            self.received = x.clone()
            return torch.zeros(x.shape[0],3,*x.shape[2:])
    model = Capture()
    predictor = Predictor(model,{"inference":{"alignment":"phase","tta":["identity"]}})
    predictor.predict_batch(prepare_pair(pre,post)[None])
    expected = prepare_pair(pre,post,"phase")
    torch.testing.assert_close(model.received[0],expected)


def test_output_lock_and_no_clobber(tmp_path):
    with training_run_lock(tmp_path):
        with pytest.raises(RuntimeError,match="in use"):
            with training_run_lock(tmp_path):
                pass
    (tmp_path / "last.pt").write_bytes(b"protected checkpoint fixture")
    with pytest.raises(FileExistsError):
        with training_run_lock(tmp_path):
            pass
    with training_run_lock(tmp_path,resume=True):
        assert (tmp_path / "last.pt").read_bytes() == b"protected checkpoint fixture"


def test_rgb_alpha_coverage_never_becomes_nir(tmp_path):
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio.transform import from_origin
    from terradelta.data.pairing import iter_pair_tiles
    paths = [tmp_path / "pre.tif",tmp_path / "post.tif"]
    for path in paths:
        data = np.full((4,32,32),127,np.uint8)
        data[3] = 255
        data[3,:,:4] = 0
        with rasterio.open(path,"w",driver="GTiff",height=32,width=32,count=4,dtype="uint8",
                           crs="EPSG:32610",transform=from_origin(500000,4300000,1,1)) as ds:
            ds.write(data)
            ds.colorinterp = (ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
    tiles = list(iter_pair_tiles(*paths,tile_size=32,min_valid_fraction=.8,max_tiles=1))
    assert len(tiles)==1
    assert tiles[0]["pre"].shape==(32,32,3)
    assert not tiles[0]["valid"][:,:4].any()
    with pytest.raises(ValueError,match="tagged as alpha"):
        list(iter_pair_tiles(*paths,tile_size=32,bands=(1,2,3,4),max_tiles=1))
