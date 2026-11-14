import os
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset import get_mumbai_dataloaders, CLASS_NAMES, CLASS_COLORS, IGNORE_INDEX, MEAN, STD
from models import SegmentationUNet, count_parameters
from train import compute_confusion_matrix, miou_from_confmat

@torch.no_grad()
def measure_latency_ms(model, device, input_shape=(1, 3, 256, 256), warmup=50, runs=200):
    model.eval()
    dummy = torch.randn(*input_shape, device=device)
    for _ in range(warmup):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms / runs

def decode_segmap(mask_2d):
    h, w = mask_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in range(6):
        rgb[mask_2d == cid] = CLASS_COLORS[cid]
    rgb[mask_2d == IGNORE_INDEX] = [0, 0, 0]
    return rgb

def unnormalize_img(tensor_img):
    img = tensor_img.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1)

@torch.no_grad()
def evaluate_model(model, test_loader, device):
    model.eval()
    confmat = np.zeros((6, 6), dtype=np.int64)
    sample_records = []

    for imgs, masks, names in test_loader:
        imgs_dev, masks_dev = imgs.to(device), masks.to(device)
        logits = model(imgs_dev)
        preds = torch.argmax(logits, dim=1)
        confmat += compute_confusion_matrix(preds, masks_dev, num_classes=6)

        for i in range(imgs.size(0)):
            c_i = compute_confusion_matrix(preds[i:i+1], masks_dev[i:i+1], num_classes=6)
            miou_i, _ = miou_from_confmat(c_i)
            sample_records.append({
                "name": names[i],
                "img": imgs[i].cpu(),
                "gt": masks[i].cpu().numpy(),
                "pred": preds[i].cpu().numpy(),
                "miou": miou_i
            })

    miou, per_class_iou = miou_from_confmat(confmat)
    return miou, per_class_iou, sample_records, confmat

def save_visual_grid(base_samples, mod_samples, indices, out_path, title):
    n = len(indices)
    fig, axes = plt.subplots(n, 5, figsize=(16, 3.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = ["Source Satellite Tile", "Reference GT Mask", "Baseline U-Net (d=1)", "Modified (Dilated d=1,2,4,8)", "Modified Error Map (Red=Err)"]
    for row_idx, s_idx in enumerate(indices):
        b_rec = base_samples[s_idx]
        m_rec = mod_samples[s_idx]

        img_rgb = unnormalize_img(b_rec["img"])
        gt_rgb  = decode_segmap(b_rec["gt"])
        b_rgb   = decode_segmap(b_rec["pred"])
        m_rgb   = decode_segmap(m_rec["pred"])

        valid = (b_rec["gt"] != IGNORE_INDEX)
        err_map = np.zeros_like(gt_rgb)
        err_map[valid & (m_rec["pred"] == b_rec["gt"])] = [35, 175, 35]   # Green = Correct
        err_map[valid & (m_rec["pred"] != b_rec["gt"])] = [230, 40, 40]   # Red = Misclassified

        panels = [img_rgb, gt_rgb, b_rgb, m_rgb, err_map]
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10.5, fontweight="bold")
        axes[row_idx, 0].set_ylabel(f"{b_rec['name']}\nBase: {b_rec['miou']:.1f}% | Mod: {m_rec['miou']:.1f}%", fontsize=9)

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_mumbai_dataloaders(batch_size=8, seed=42)

    base_model = SegmentationUNet(num_classes=6, base_ch=32, use_dilated_bottleneck=False).to(device)
    mod_model  = SegmentationUNet(num_classes=6, base_ch=32, use_dilated_bottleneck=True).to(device)

    base_model.load_state_dict(torch.load("mumbai_seg_exp/checkpoints/baseline_unet_best.pth", map_location=device))
    mod_model.load_state_dict(torch.load("mumbai_seg_exp/checkpoints/dilated_unet_best.pth", map_location=device))

    base_miou, base_ious, base_samples, base_cm = evaluate_model(base_model, test_loader, device)
    mod_miou,  mod_ious,  mod_samples,  mod_cm  = evaluate_model(mod_model,  test_loader, device)

    base_lat = measure_latency_ms(base_model, device)
    mod_lat  = measure_latency_ms(mod_model,  device)

    base_params = count_parameters(base_model)
    mod_params  = count_parameters(mod_model)

    print("\n================ FINAL UNSEEN TEST SET EVALUATION (17 TILES) ================")
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("Latency Conditions: Batch Size = 1, Input = 1x3x256x256, FP32, 50 Warmup + 200 Synchronized CUDA Passes")
    print("-" * 86)
    print(f"{'Class ID & Name':<30} | {'Baseline IoU (%)':<18} | {'Dilated IoU (%)':<18} | {'Delta (%)':<10}")
    print("-" * 86)
    for idx, cname in enumerate(CLASS_NAMES):
        delta = mod_ious[idx] - base_ious[idx]
        label_str = f"{idx}: {cname}"
        print(f"{label_str:<30} | {base_ious[idx]:>16.2f}% | {mod_ious[idx]:>16.2f}% | {delta:>+8.2f}%")
    print("-" * 86)
    print(f"{'Mean IoU (mIoU)':<30} | {base_miou:>16.2f}% | {mod_miou:>16.2f}% | {mod_miou - base_miou:>+8.2f}%")
    print(f"{'Trainable Parameters':<30} | {base_params:>17,d} | {mod_params:>17,d} | {mod_params - base_params:>+9,d}")
    print(f"{'Inference Latency (ms/img)':<30} | {base_lat:>15.2f} ms | {mod_lat:>15.2f} ms | {mod_lat - base_lat:>+7.2f} ms")
    print("=" * 86)

    # Sort test tiles by improvement (Successes) and lowest absolute mIoU (Failures)
    diffs = [mod_samples[i]["miou"] - base_samples[i]["miou"] for i in range(len(base_samples))]
    top_gains = np.argsort(diffs)[-3:][::-1]
    worst_failures = np.argsort([m["miou"] for m in mod_samples])[:3]

    save_visual_grid(base_samples, mod_samples, top_gains,
                     "mumbai_seg_exp/results/qualitative_comparison.png",
                     "Qualitative Results: Source Images, Reference Masks & Multi-Scale Dilated Improvements")
    save_visual_grid(base_samples, mod_samples, worst_failures,
                     "mumbai_seg_exp/results/failure_cases.png",
                     "Failure Analysis: Boundary Bleeding & Shadow/Roof Ambiguity on Hard Test Tiles")

    summary = {
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "baseline": {"miou": base_miou, "per_class_iou": base_ious.tolist(), "params": base_params, "latency_ms": base_lat},
        "dilated":  {"miou": mod_miou,  "per_class_iou": mod_ious.tolist(),  "params": mod_params,  "latency_ms": mod_lat}
    }
    with open("mumbai_seg_exp/results/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
