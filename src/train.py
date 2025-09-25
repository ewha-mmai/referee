import os
import time
import datetime
import pickle
import argparse
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0.*")

from src.lr_scheduler import build_scheduler
from dataset.dataloader import TrainDataset
from dataset.transform_builders import get_train_transforms, get_val_transforms
from model.referee import Referee

class AverageMeter:
    def __init__(self): 
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0
    def update(self, v, n=1):
        v = float(v)
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / max(1, self.count)

def train(model, train_loader, val_loader, args, cfg):
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    loss_meter_rf, acc_meter_rf = AverageMeter(), AverageMeter()
    loss_meter_id, acc_meter_id = AverageMeter(), AverageMeter()

    exp_dir = args.exp_dir
    progress = []

    best_epoch_loss = 0
    best_loss = np.inf
    best_epoch_auc = 0
    best_auc = 0.0

    progress_pkl_path = os.path.join(exp_dir, "progress.pkl")
    if os.path.exists(progress_pkl_path):
        print(f"Loading previous training progress from '{progress_pkl_path}'.")
        with open(progress_pkl_path, "rb") as f:
            progress = pickle.load(f)

    if progress:
        last_record = progress[-1]
        # format : [epoch, global_step, best_epoch_loss, best_loss, best_epoch_auc, best_auc, time] 
        best_epoch_loss = last_record[2]
        best_loss = last_record[3]
        best_epoch_auc = last_record[4]
        best_auc = last_record[5]
        print(f"Restored Best (Loss): epoch={best_epoch_loss}, loss={best_loss:.6f}")
        print(f"Restored Best (AUC) : epoch={best_epoch_auc}, auc={best_auc:.6f}")
    else:
        print("No previous progress found. Starting fresh.")

    global_step, epoch = args.start_step, args.start_epoch
    start_time = time.time()
    
    def _save_progress():
        progress.append([epoch, global_step, best_epoch_loss, best_loss, best_epoch_auc, best_auc, time.time() - start_time])
        with open(progress_pkl_path, "wb") as f:
            pickle.dump(progress, f)

    # Optimizer / Scheduler / Scaler
    trainables = [p for p in model.parameters() if p.requires_grad]
    print('Total params    : {:.3f}M'.format(sum(p.numel() for p in model.parameters())/1e6))
    print('Trainable params: {:.3f}M'.format(sum(p.numel() for p in trainables)/1e6))
    optimizer = torch.optim.Adam(trainables, cfg.training.lr_initial, weight_decay=5e-7, betas=(0.95, 0.999))

    n_iter_per_epoch = len(train_loader)
    scheduler = build_scheduler(cfg.training, optimizer, n_iter_per_epoch)
    scaler = GradScaler()

    print('Scheduler:', scheduler.__class__.__name__)
    print("Start training...") 

    try:
        for epoch in range(args.start_epoch + 1, args.n_epochs + 1):
            model.train()

            train_probs_rf, train_labels_rf = [], []
            train_probs_id, train_labels_id = [], []
            
            print('---------------')
            print(datetime.datetime.now())
            print(f"Epoch={epoch}, Current Step={global_step}")
            
            for i, batch in enumerate(train_loader):
                target_v = batch['target_video'].to(device, non_blocking=True)
                target_a = batch['target_audio'].to(device, non_blocking=True)
                ref_v = batch['reference_video'].to(device, non_blocking=True)
                ref_a = batch['reference_audio'].to(device, non_blocking=True)
                labels_rf = batch['fake_label'].to(device, non_blocking=True)
                labels_id = batch['id_label'].to(device, non_blocking=True) 
                B = target_v.size(0)

                with autocast(enabled=True):
                    loss_rf, logits_rf, loss_id, logits_id = model(target_v, target_a, ref_v, ref_a, labels_rf, labels_id)
                    total_loss = (loss_rf + loss_id).mean()

                optimizer.zero_grad()    
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                if scheduler:
                    scheduler.step_update(global_step)

                probs_rf = torch.softmax(logits_rf, dim=1)
                train_probs_rf.append(probs_rf.detach().cpu())
                train_labels_rf.append(labels_rf.detach().cpu())
                loss_meter_rf.update(loss_rf.mean().item(), B)
                acc_meter_rf.update(accuracy_score(labels_rf.cpu().numpy(), probs_rf.argmax(dim=1).cpu().numpy()), B)

                probs_id = torch.softmax(logits_id, dim=1)
                train_probs_id.append(probs_id.detach().cpu())
                train_labels_id.append(labels_id.detach().cpu())
                loss_meter_id.update(loss_id.mean().item(), B)
                acc_meter_id.update(accuracy_score(labels_id.cpu().numpy(), probs_id.argmax(dim=1).cpu().numpy()), B)

                if global_step % args.n_print_steps == 0 and global_step != 0:
                    print(f'E: [{epoch}][{i+1}/{len(train_loader)}] | Loss RF {loss_meter_rf.val:.4f}/ID {loss_meter_id.val:.4f} | Acc RF {acc_meter_rf.val:.3f}/ID {acc_meter_id.val:.3f}')
                
                global_step += 1

            # --- End of epoch: compute train AUC/AP
            print(f"\n--- Epoch {epoch}: Evaluation Start ---")
            try:
                auc_rf = roc_auc_score(torch.cat(train_labels_rf).numpy(), torch.cat(train_probs_rf)[:, 1].numpy())
                ap_rf = average_precision_score(torch.cat(train_labels_rf).numpy(), torch.cat(train_probs_rf)[:, 1].numpy())
            except ValueError: auc_rf, ap_rf = -1, -1
            
            try:
                auc_id = roc_auc_score(torch.cat(train_labels_id).numpy(), torch.cat(train_probs_id)[:, 1].numpy())
                ap_id = average_precision_score(torch.cat(train_labels_id).numpy(), torch.cat(train_probs_id)[:, 1].numpy())
            except ValueError: auc_id, ap_id = -1, -1

            # --- Validation
            val_metrics = validate(model, val_loader, device)

            print(f"Epoch {epoch} Summary:")
            print(f"  Train -> Loss RF: {loss_meter_rf.avg:.4f}, Acc RF: {acc_meter_rf.avg:.4f}, AUC RF: {auc_rf:.4f}")
            print(f"  Val   -> Loss RF: {val_metrics['rf']['loss']:.4f}, Acc RF: {val_metrics['rf']['acc']:.4f}, AUC RF: {val_metrics['rf']['auc']:.4f}")

            # Save checkpoints
            if args.save_model and epoch % 1 == 0:
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save({'epoch': epoch, 'global_step': global_step, 'model_state_dict': model_state}, 
                           os.path.join(exp_dir, "model", f"model.{epoch}.pth"))

            save_best_model = False
            if val_metrics['rf']['loss'] < best_loss:
                best_loss = val_metrics['rf']['loss']
                best_epoch_loss = epoch
                print(f"New Best Loss: {best_loss:.4f} at epoch {epoch}.")
                save_best_model = True

            if val_metrics['rf']['auc'] > best_auc:
                best_auc = val_metrics['rf']['auc']
                best_epoch_auc = epoch
                print(f"New Best AUC: {best_auc:.4f} at epoch {epoch}.")
                save_best_model = True

            if save_best_model:
                print(f"Saving best_model.pth due to new best performance.")
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save({'epoch': epoch, 'global_step': global_step, 'model_state_dict': model_state},
                        os.path.join(exp_dir, "model", "best_model.pth"))
            
            _save_progress()
            print(f"--- Epoch {epoch}: Evaluation End ---\n")

    except (KeyboardInterrupt, Exception) as e:
        print(f"\nTraining stopped by {type(e).__name__}. Saving current state...")
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        filename = "ckpt_interrupt.pth" if isinstance(e, KeyboardInterrupt) else "ckpt_error.pth"
        torch.save({
            'model_state_dict': model_state,
        }, os.path.join(exp_dir, "model", filename))
        _save_progress()
        if not isinstance(e, KeyboardInterrupt): raise e


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    all_losses_rf, all_probs_rf, all_labels_rf = [], [], []
    all_losses_id, all_probs_id, all_labels_id = [], [], []
    all_modify_types = []

    for i, batch in enumerate(val_loader):
        target_v = batch['target_video'].to(device, non_blocking=True)
        target_a = batch['target_audio'].to(device, non_blocking=True)
        ref_v = batch['reference_video'].to(device, non_blocking=True)
        ref_a = batch['reference_audio'].to(device, non_blocking=True)
        labels_rf = batch['fake_label'].to(device, non_blocking=True)
        labels_id = batch['id_label'].to(device, non_blocking=True)
        

        with autocast():
            loss_rf, logits_rf, loss_id, logits_id = model(target_v, target_a, ref_v, ref_a, labels_rf, labels_id)

        all_losses_rf.append(loss_rf.mean().item())
        all_probs_rf.append(torch.softmax(logits_rf, dim=1).cpu())
        all_labels_rf.append(labels_rf.cpu())

        all_losses_id.append(loss_id.mean().item())
        all_probs_id.append(torch.softmax(logits_id, dim=1).cpu())
        all_labels_id.append(labels_id.cpu())


    all_probs_rf = torch.cat(all_probs_rf)
    all_labels_rf = torch.cat(all_labels_rf)
    all_preds_rf = all_probs_rf.argmax(dim=1)
    
    loss_rf = np.mean(all_losses_rf)
    acc_rf = accuracy_score(all_labels_rf.numpy(), all_preds_rf.numpy())

    try:
        auc_rf = roc_auc_score(all_labels_rf.numpy(), all_probs_rf[:, 1].numpy())
        ap_rf = average_precision_score(all_labels_rf.numpy(), all_probs_rf[:, 1].numpy())
    except ValueError:
        auc_rf, ap_rf = -1, -1

    all_probs_id = torch.cat(all_probs_id)
    all_labels_id = torch.cat(all_labels_id)
    all_preds_id = all_probs_id.argmax(dim=1)

    loss_id = np.mean(all_losses_id)
    acc_id = accuracy_score(all_labels_id.numpy(), all_preds_id.numpy())

    try:
        auc_id = roc_auc_score(all_labels_id.numpy(), all_probs_id[:, 1].numpy())
        ap_id = average_precision_score(all_labels_id.numpy(), all_probs_id[:, 1].numpy())
    except ValueError:
        auc_id, ap_id = -1, -1


    final_metrics = {
        'rf': {'loss': loss_rf, 'acc': acc_rf, 'auc': auc_rf, 'ap': ap_rf},
        'id': {'loss': loss_id, 'acc': acc_id, 'auc': auc_id, 'ap': ap_id},
    }
    return final_metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--exp_dir', type=str, default='./exp')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--config', type=str, default='configs/pair_sync_262.yaml', help='Path to config yaml')
    parser.add_argument('--train_json', type=str, default='data/train_set.json')
    parser.add_argument('--val_json', type=str, default='data/val_set.json')

    parser.add_argument('--start_epoch', type=int, default=0, help='last epoch of pre-trained model')
    parser.add_argument('--start_step', type=int, default=0, help='last global step of pre-trained model')
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--n_print_steps', type=int, default=50)
    parser.add_argument('--save_model', action='store_true')

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    os.makedirs(args.exp_dir + "/model", exist_ok=True)

    model = Referee(cfg, ckpt_path=args.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    train_transforms = get_train_transforms(cfg)
    val_transforms = get_val_transforms(cfg)

    train_dataset = TrainDataset(args.train_json, train_transforms)
    val_dataset = TrainDataset(args.val_json, val_transforms)

    print("Start making a weighted sampler for 'fake_label'...")

    label_counts = Counter(sample['fake_label'] for sample in train_dataset.samples)
    total_samples = len(train_dataset)
    class_weights = {
        label: total_samples / count 
        for label, count in label_counts.items()
    }

    sample_weights = [
        class_weights[sample['fake_label']] 
        for sample in train_dataset.samples
    ]

    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=total_samples, 
        replacement=True
    )
    print("Making weighted sampler is Done!")

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, sampler=sampler, shuffle=False, num_workers=args.num_workers, pin_memory=False, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, persistent_workers=False)

    train(model, train_loader, val_loader, args, cfg) 
