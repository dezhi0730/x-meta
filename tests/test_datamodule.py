# tests/test_datamodule.py
"""
pytest -q tests/test_datamodule.py
依赖：
  * config 文件里 data.train_meta / data.val_meta / dna_dir / tree_dir 等路径已指向实际数据
  * 至少保证 train_meta.csv 有 ≥1 个样本、tree_dataset 能匹配
"""
import pytest, torch
from omegaconf import OmegaConf
from gutclip.data import GutDataModule


@pytest.fixture(scope="session")
def cfg():
    cfg = OmegaConf.load("gutclip/configs/default.yaml")
    # 如需在 CI 用小样本快速跑，可在此覆盖 batch_size / num_workers
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0
    return cfg


def test_dataloaders(cfg):
    dm = GutDataModule(cfg)
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()

    # --- train batch ---
    batch = next(iter(train_loader))
    assert batch is not None, "No valid samples in the batch"
    assert batch.dna.shape[0] > 0, "Batch size should be greater than 0"
    assert hasattr(batch, "x") and hasattr(batch, "edge_index")

    # --- val batch ---
    batch_v = next(iter(val_loader))
    assert batch_v is not None, "No valid samples in the validation batch"
    assert batch_v.dna.shape[0] > 0, "Validation batch size should be greater than 0"

    # --- device / dtype sanity ---
    assert batch.dna.dtype == torch.float32
    assert batch.x.dtype   == torch.float32