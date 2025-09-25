from torch.utils.data import Dataset
import torch
import json
import os
from typing import Dict, Any
from .dataset_utils import get_video_and_audio

import pandas as pd
import random

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF_CSV = PROJECT_ROOT / "data" / "all_real_with_split.csv"


class TrainDataset(Dataset):
    def __init__(self, json_path: str, transform_fn=None, ref_csv_path=DEFAULT_REF_CSV):
        with open(json_path, "r") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict) and "data" in loaded:
                self.samples = loaded["data"]
            else:
                self.samples = loaded

        self.transforms = transform_fn
        ref_csv_path = Path(ref_csv_path)
        df = pd.read_csv(ref_csv_path)
        self.reference_map = df.groupby(['source_id', 'split'])['new_path'].apply(list).to_dict()

    def __len__(self):
        return len(self.samples)

    def _load_pair(self, tgt_path: str, ref_path: str):
        tgt_video, tgt_audio, tgt_meta = get_video_and_audio(tgt_path, get_meta=True)
        ref_video, ref_audio, ref_meta = get_video_and_audio(ref_path, get_meta=True)

        if self.transforms is not None:
            tgt_loaded = {'video': tgt_video, 'audio': tgt_audio, 'meta': tgt_meta}
            transformed_tgt = self.transforms(tgt_loaded)
            tgt_video = transformed_tgt['video']
            tgt_audio = transformed_tgt['audio']
            
            ref_loaded = {'video': ref_video, 'audio': ref_audio, 'meta': ref_meta}
            transformed_ref = self.transforms(ref_loaded)
            ref_video = transformed_ref['video']
            ref_audio = transformed_ref['audio']

        return (tgt_video, tgt_audio), (ref_video, ref_audio)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        tgt_path = item["file_path"]
        tgt_id = item["id"]
        tgt_split = item["split"]

        key = (tgt_id, tgt_split)
        possible_references = self.reference_map.get(key)

        if not possible_references:
            raise ValueError(f"No reference video found for id: '{tgt_id}' and split: '{tgt_split}'")

        ref_path = random.choice(possible_references)

        (tgt_v, tgt_a), (ref_v, ref_a) = self._load_pair(tgt_path, ref_path)

        fake_label = int(item.get("fake_label", 0))
        same_identity = 1 - fake_label

        return {
            "target_video": tgt_v,
            "target_audio": tgt_a,
            "reference_video": ref_v,
            "reference_audio": ref_a,
            "fake_label": torch.tensor(fake_label, dtype=torch.long),
            "id_label": torch.tensor(same_identity, dtype=torch.long),
        }


class TestDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict) and "data" in loaded:
                self.samples = loaded["data"]
            else:
                self.samples = loaded

    def __len__(self):
        return len(self.samples)

    def _load_pair(self, item: Dict[str, Any]):
        tgt_path = item["target_file"]
        ref_path = item["reference_file"]

        tgt_video, tgt_audio, tgt_meta = get_video_and_audio(tgt_path, get_meta=True)
        ref_video, ref_audio, ref_meta = get_video_and_audio(ref_path, get_meta=True)

        return (tgt_video, tgt_audio), (ref_video, ref_audio)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        (tgt_v, tgt_a), (ref_v, ref_a) = self._load_pair(item)

        fake_label = int(item.get("fake_label", 0))
        same_identity = int(item.get("same_identity", 0))

        return {
            "target_video": tgt_v,
            "target_audio": tgt_a,
            "reference_video": ref_v,
            "reference_audio": ref_a,
            "fake_label": torch.tensor(fake_label, dtype=torch.long),
            "id_label": torch.tensor(same_identity, dtype=torch.long),
        }


