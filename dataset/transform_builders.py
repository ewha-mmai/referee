import torchvision.transforms as T
from .transforms import *
from utils.utils import instantiate_from_config
from omegaconf import DictConfig

def build_transform_sequence(seq_cfg):
    modules = []
    for entry in seq_cfg:
        module = instantiate_from_config(entry)
        modules.append(module)
    return T.Compose(modules)

def get_train_transforms(cfg):
    if hasattr(cfg, "transform_sequence_train"):
        seq = cfg.transform_sequence_train
    elif hasattr(cfg.data, "train_transform"):
        seq = cfg.data.train_transform
    else:
        raise KeyError("No train transform sequence found in config.")
    return build_transform_sequence(seq)

def get_val_transforms(cfg):
    if hasattr(cfg, "transform_sequence_test"):
        seq = cfg.transform_sequence_test
    elif hasattr(cfg.data, "val_transform"):
        seq = cfg.data.val_transform
    else:
        raise KeyError("No val transform sequence found in config.")
    return build_transform_sequence(seq)
