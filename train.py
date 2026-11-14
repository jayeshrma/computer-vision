import os
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn
from dataset import get_mumbai_dataloaders, IGNORE_INDEX
from models import SegmentationUNet, count_parameters

def set_deterministic_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def compute_confusion_matrix(preds, targets, num_classes=6, ignore_index=IGNORE_INDEX):
    valid = (targets != ignore_index)
    preds = preds[valid]
    targets = targets[valid]
    k = (targets >= 0) & (targets < num_classes)
    inds = num_classes * targets[k].to(torch.int64) + preds[k].to(torch.int64)
    return torch.bincount(inds, minlength=num_classes**2).reshape(num_classes, num_classes).cpu().numpy()

def miou_from_confmat(confmat):
    intersection = np.diag(confmat)
    union = confmat.sum(axis=1) + confmat.sum(axis=0) - intersection
    iou = intersection / np.maximum(union, 1e-8)
    valid_classes = confmat.sum(axis=1) > 0
    return float(np.mean(iou[valid_classes]) * 100.0), (iou * 100.0)

def train_single_experiment(model_tag, use_dilated, log_writer, epochs=60, patience=15, lr=1e-3, seed=42):
    set_deterministic_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_mumbai_dataloaders(batch_size=8, seed=seed)
    model = SegmentationUNet(num_classes=6, base_ch=32, use_dilated_bottleneck=use_dilated).to(device)
    params = count_parameters(model)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_miou = 0.0
    best_epoch = 0
    no_improve_epochs = 0
    ckpt_path = os.path.join("mumbai_seg_exp", "checkpoints", f"{model_tag}_best.pth")

    print(f"\n=======================================================")
    print(f"Training {model_tag.upper()} | Params: {params:,} | Seed: {seed}")
    print(f"=======================================================")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation Evaluation
        model.eval()
        val_loss = 0.0
        confmat = np.zeros((6, 6), dtype=np.int64)
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(logits, dim=1)
                confmat += compute_confusion_matrix(preds, masks)

        val_loss /= len(val_loader.dataset)
        val_miou, _ = miou_from_confmat(confmat)
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        log_writer.writerow([model_tag, epoch, f"{cur_lr:.6f}", f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_miou:.2f}", f"{elapsed:.2f}"])

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
            marker = " *BEST*"
        else:
            no_improve_epochs += 1
            marker = ""

        if epoch == 1 or epoch % 5 == 0 or marker or no_improve_epochs >= patience:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.2f}% | Time: {elapsed:.1f}s{marker}")

        if no_improve_epochs >= patience:
            print(f"--> Early stopping triggered at epoch {epoch} (patience={patience}). Best Val mIoU: {best_val_miou:.2f}% (Epoch {best_epoch})")
            break

    print(f"Finished {model_tag}: Best Val mIoU = {best_val_miou:.2f}% at Epoch {best_epoch}")

if __name__ == "__main__":
    log_path = "mumbai_seg_exp/logs/training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epoch", "lr", "train_loss", "val_loss", "val_miou", "epoch_time_s"])
        train_single_experiment("baseline_unet", False, writer, epochs=60, patience=15)
        train_single_experiment("dilated_unet",  True,  writer, epochs=60, patience=15)
