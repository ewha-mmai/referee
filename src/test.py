import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, classification_report, average_precision_score, roc_auc_score
from torchvision import transforms as T
from tqdm import tqdm

from dataset.transforms import *
from utils.utils import instantiate_from_config
    
def test(model, test_loader, args, cfg):
    """
    Evaluates the model on the test set for both Real/Fake (RF) and Identity (ID) tasks.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = nn.DataParallel(model) if not isinstance(model, nn.DataParallel) else model
    model.eval()

    # Windowing parameters
    CLIP_SECONDS = 2.88
    FPS = 25
    AUDIO_SR = 16000
    STRIDE_SECONDS = CLIP_SECONDS * 0.95

    REF_CLIP_SECONDS = 3.0
    ref_video_clip_size = int(REF_CLIP_SECONDS * FPS)
    ref_audio_clip_size = int(REF_CLIP_SECONDS * AUDIO_SR)

    video_window_size = int(CLIP_SECONDS * FPS)
    video_stride = int(STRIDE_SECONDS * FPS)
    audio_window_size = int(CLIP_SECONDS * AUDIO_SR)
    audio_stride = int(STRIDE_SECONDS * AUDIO_SR)

    transform_configs = [tc for tc in cfg.transform_sequence_test]
    transform_list = []
    for tc in transform_configs:
        module = instantiate_from_config(tc)
        transform_list.append(module)

    single_clip_transformer = T.Compose(transform_list)

    # Lists for RF task metrics
    A_loss_rf, all_rf_preds, all_rf_labels, all_rf_probs = [], [], [], []

    # Lists for ID task metrics
    A_loss_id, all_id_preds, all_id_labels, all_id_probs = [], [], [], []


    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing Progress")):
            target_v_full = batch['target_video']
            target_a_full = batch['target_audio']
            ref_v_full = batch['reference_video']
            ref_a_full = batch['reference_audio']
            labels_rf = batch['fake_label'].to(device, non_blocking=True)
            labels_id = batch['id_label'].to(device, non_blocking=True)

            B, T_v_ref, C, H, W = ref_v_full.shape
            T_a_ref = ref_a_full.shape[1]
            
            # Padding for short target sequences
            if target_v_full.shape[1] < video_window_size:
                pad_size = video_window_size - target_v_full.shape[1]
                target_v_full = target_v_full.permute(0, 2, 3, 4, 1)
                target_v_full = F.pad(target_v_full, (0, pad_size), "constant", 0)
                target_v_full = target_v_full.permute(0, 4, 1, 2, 3)
            if target_a_full.shape[1] < audio_window_size:
                pad_size = audio_window_size - target_a_full.shape[1]
                target_a_full = F.pad(target_a_full, (0, pad_size), "constant", 0)
        
            # Window unfolding (for target)
            target_v_windows = target_v_full.unfold(1, video_window_size, video_stride)
            target_a_windows = target_a_full.unfold(1, audio_window_size, audio_stride)

            # Convert windows into batch form (for target)
            target_v_batch = target_v_windows.permute(0, 1, 5, 2, 3, 4).squeeze(0)
            target_a_batch = target_a_windows.squeeze(0)

            final_num_windows = min(target_v_batch.shape[0], target_a_batch.shape[0])
            target_v_batch = target_v_batch[:final_num_windows]
            target_a_batch = target_a_batch[:final_num_windows]
            

            # Video Reference Clipping/Padding
            if T_v_ref > ref_video_clip_size:
                # Select a random start frame for the 3-seconds clip
                start_v_idx = random.randint(0, T_v_ref - ref_video_clip_size)
                ref_v_3sec = ref_v_full[:, start_v_idx : start_v_idx + ref_video_clip_size, :, :, :]
            else:
                # If shorter than 3 seconds, pad/duplicate last frame
                ref_v_3sec = ref_v_full
                if T_v_ref < ref_video_clip_size:
                    pad_size = ref_video_clip_size - T_v_ref
                    # Move time dimension to end for F.pad, then move back
                    ref_v_3sec = ref_v_3sec.permute(0, 2, 3, 4, 1)
                    ref_v_3sec = F.pad(ref_v_3sec, (0, pad_size), "replicate")
                    ref_v_3sec = ref_v_3sec.permute(0, 4, 1, 2, 3)

            # Remove batch dimension for concatenation later (shape: (T_v_3sec, C, H, W))
            ref_v_3sec_single = ref_v_3sec.squeeze(0)

            # Audio Reference Clipping/Padding
            if T_a_ref > ref_audio_clip_size:
                # Select a random start sample for the 3-second clip
                start_a_idx = random.randint(0, T_a_ref - ref_audio_clip_size)
                ref_a_3sec = ref_a_full[:, start_a_idx : start_a_idx + ref_audio_clip_size]
            else:
                # If shorter than 3 seconds, pad/duplicate last sample
                ref_a_3sec = ref_a_full
                if T_a_ref < ref_audio_clip_size:
                    pad_size = ref_audio_clip_size - T_a_ref
                    ref_a_3sec = F.pad(ref_a_3sec, (0, pad_size), "replicate")

            # Remove batch dimension (shape: (T_a_3sec,))
            ref_a_3sec_single = ref_a_3sec.squeeze(0)

            # duplicate the single 3-sec reference clip to match the number of target windows
            ref_v_batch = ref_v_3sec_single.unsqueeze(0).repeat(final_num_windows, 1, 1, 1, 1)
            ref_a_batch = ref_a_3sec_single.unsqueeze(0).repeat(final_num_windows, 1)

            # Apply transforms
            transformed_target_v, transformed_target_a = [], []
            transformed_ref_v, transformed_ref_a = [], []
            for k in range(final_num_windows):
                # Target
                sample = {'video': target_v_batch[k], 'audio': target_a_batch[k], 'meta': {'video': {}, 'audio': {}}}
                transformed = single_clip_transformer(sample)
                transformed_target_v.append(transformed['video'])
                transformed_target_a.append(transformed['audio'])

                # Reference
                sample = {'video': ref_v_batch[k], 'audio': ref_a_batch[k], 'meta': {'video': {}, 'audio': {}}}
                transformed = single_clip_transformer(sample)
                transformed_ref_v.append(transformed['video'])
                transformed_ref_a.append(transformed['audio'])
            
            final_target_v = torch.stack(transformed_target_v).to(device)
            final_target_a = torch.stack(transformed_target_a).to(device)
            final_ref_v = torch.stack(transformed_ref_v).to(device)
            final_ref_a = torch.stack(transformed_ref_a).to(device)

            with autocast():
                logits_rf_batch, logits_id_batch = model(final_target_v, final_target_a, final_ref_v, final_ref_a)

            # Aggregate predictions across windows
            avg_rf_probs = torch.softmax(logits_rf_batch, dim=1).mean(dim=0)
            avg_id_probs = torch.softmax(logits_id_batch, dim=1).mean(dim=0)

            final_rf_pred = avg_rf_probs.argmax()
            final_id_pred = avg_id_probs.argmax()

            # Store metrics
            all_rf_probs.append(avg_rf_probs.unsqueeze(0).cpu())
            all_rf_preds.append(final_rf_pred.unsqueeze(0).cpu())
            all_rf_labels.append(labels_rf.cpu())

            all_id_probs.append(avg_id_probs.unsqueeze(0).cpu())
            all_id_preds.append(final_id_pred.unsqueeze(0).cpu())
            all_id_labels.append(labels_id.cpu())
            
            # Monitor loss
            A_loss_rf.append(F.cross_entropy(avg_rf_probs.unsqueeze(0), labels_rf).cpu().detach())
            A_loss_id.append(F.cross_entropy(avg_id_probs.unsqueeze(0), labels_id).cpu().detach())

    # --- Compute Metrics ---
    all_rf_preds_np = torch.cat(all_rf_preds).numpy()
    all_rf_labels_np = torch.cat(all_rf_labels).numpy()
    all_rf_probs_np = torch.cat(all_rf_probs).numpy()

    loss_rf = np.mean(A_loss_rf)
    acc_rf = accuracy_score(all_rf_labels_np, all_rf_preds_np)
    try:
        auc_rf = roc_auc_score(all_rf_labels_np, all_rf_probs_np[:, 1])
        ap_rf = average_precision_score(all_rf_labels_np, all_rf_probs_np[:, 1])
    except ValueError:
        auc_rf, ap_rf = -1, -1

    all_id_preds_np = torch.cat(all_id_preds).numpy()
    all_id_labels_np = torch.cat(all_id_labels).numpy()
    all_id_probs_np = torch.cat(all_id_probs).numpy()

    loss_id = np.mean(A_loss_id)
    acc_id = accuracy_score(all_id_labels_np, all_id_preds_np)
    try:
        auc_id = roc_auc_score(all_id_labels_np, all_id_probs_np[:, 1])
        ap_id = average_precision_score(all_id_labels_np, all_id_probs_np[:, 1])
    except ValueError:
        auc_id, ap_id = -1, -1

    # --- Print Results ---
    print("\n[Test Results]")
    print(f"RF Task -> Loss: {loss_rf:.4f} | Acc: {acc_rf:.4f} | AUC: {auc_rf:.4f} | AP: {ap_rf:.4f}")
    print(f"ID Task -> Loss: {loss_id:.4f} | Acc: {acc_id:.4f} | AUC: {auc_id:.4f} | AP: {ap_id:.4f}")
    
    print("\n[Classification Report - Real/Fake]")
    print(classification_report(all_rf_labels_np, all_rf_preds_np, target_names=['Real', 'Fake']))
    
    print("\n[Classification Report - Identity]")
    print(classification_report(all_id_labels_np, all_id_preds_np, target_names=['Different_ID', 'Same_ID']))

if __name__ == "__main__":
    import argparse
    from model.referee import Referee
    from omegaconf import OmegaConf
    from dataset.dataloader import TestDataset
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_path', type=str, default=None, required=True)
    parser.add_argument('--test_json', type=str, required=True)
    parser.add_argument('--config', type=str, default="configs/pair_sync.yaml", help="Path to config yaml")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    model = Referee(cfg, ckpt_path=args.model_path)

    test_dataset = TestDataset(args.test_json)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    print("\nRunning evaluation...")
    test(model, test_loader, args, cfg)