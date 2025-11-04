#!/usr/bin/python3

"""
VFast-SCNN + GRU (temporal head) — single file

Catatan:
- GRU dipasang pada vektor global (hasil Global AvgPool -> 1x1) untuk smoothing temporal antar frame.
- Single image inference tetap jalan (h=None). Untuk video/stream, simpan-kembalikan hidden state (h).
- ONNX export mendukung input hidden state (opsional), sehingga enak untuk deployment real-time.

Asal-usul:
- Berbasis skrip user (reduksi channel + penggantian pyramid pooling dgn average pooling).
- Ditambahkan GRU head yang ringan (bukan ConvGRU2D) agar tetap cepat di CPU-only.

"""

import os
import cv2
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


def gray_world_wb(img_rgb: np.ndarray) -> np.ndarray:
    # img_rgb: uint8 (H,W,3) RGB
    img = img_rgb.astype(np.float32)
    mean_b = img[:, :, 2].mean()  # R
    mean_g = img[:, :, 1].mean()  # G
    mean_r = img[:, :, 0].mean()  # B  (ingat: kamu pakai cv2.cvtColor BGR->RGB sebelumnya)
    mean_gray = (mean_r + mean_g + mean_b) / 3.0 + 1e-6
    gain_r = mean_gray / (mean_r + 1e-6)
    gain_g = mean_gray / (mean_g + 1e-6)
    gain_b = mean_gray / (mean_b + 1e-6)
    img[:, :, 0] *= gain_r
    img[:, :, 1] *= gain_g
    img[:, :, 2] *= gain_b
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def clahe_rgb(img_rgb: np.ndarray, clip=2.0, tile=8) -> np.ndarray:
    # CLAHE di channel L (LAB), lalu balik ke RGB
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return out

def adaptive_gamma(img_rgb: np.ndarray, target_mean=0.5, eps=1e-6):
    """
    Set gamma supaya mean luminance mendekati target_mean (0..1).
    """
    # luminance dari Y in YCrCb
    ycc = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y = ycc[:, :, 0].astype(np.float32) / 255.0
    m = y.mean()
    # cari gamma: y_out = y^{gamma} -> mean mendekati target
    # heuristik sederhana:
    gamma = np.log(target_mean + eps) / np.log(m + eps)
    gamma = float(np.clip(gamma, 0.5, 2.0))  # batasi agar stabil
    img = np.clip((img_rgb.astype(np.float32) / 255.0) ** gamma, 0, 1)
    return (img * 255).astype(np.uint8), gamma

def print_grad_norm(module, name="last"):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item()
    print(f"[grad] {name} L2 = {total:.6f}")

# ========== Konfigurasi dasar ==========
pth_name = "lanjut_bismillah5.pth"
onnx_name = "lanjut_bismillah5_gru.onnx"
check_image = "/home/wildan/proyek/robotika/omoda/go-go-ml/datasets/v2/v2_frame_03430.png"
dataset_dir = "./datasets/gabungan"

image_width = 640
image_height = 360

# training & export switches
is_training = True
is_export_onnx = False

# channel config (dipertahankan dari versi user)
layer1 = 32
layer2 = 48
layer3 = 64
layer4 = 96
layer5 = 128   # juga dipakai sbg dim GRU (input_size==hidden_size)
layer6 = 64

epoch = 60
batch_size = 4
learning_rate = 1e-4
pos_weight_val = 20.0
threshold_val = 0.70008

# ===========================================================
# Dataset
# ===========================================================
class DatasetLoader(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Gambar tidak ditemukan: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = gray_world_wb(image)
        image = clahe_rgb(image, clip=2.0, tile=8)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask tidak ditemukan: {mask_path}")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']               # Tensor (C,H,W), float, normalized
            mask = augmented['mask']                 # Tensor (H,W) float (0..1)

        # pastikan biner (float) — kalau ToTensorV2 sudah float, ini aman
        if isinstance(mask, torch.Tensor):
            mask = (mask > 0.5).float()
        else:
            mask = torch.from_numpy((mask > 127).astype(np.float32))

        return image, mask


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        """
        logits: (B,1,H,W) atau (B,H,W)
        targets: (B,1,H,W) atau (B,H,W) berisi {0,1}
        """
        p = torch.sigmoid(logits)

        # Samakan dimensi menjadi (B,1,H,W)
        if p.dim() == 3:
            p = p.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # Flatten per-sample
        B = p.size(0)
        p = p.reshape(B, -1)
        t = targets.reshape(B, -1)

        inter = (p * t).sum(dim=1)
        denom = (p + t).sum(dim=1) + self.eps
        dice = (2.0 * inter) / denom
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.6, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.w = bce_weight

    def forward(self, logits, targets):
        # Samakan dimensi dulu untuk BCE: (B,1,H,W)
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        return self.w * self.bce(logits, targets) + (1 - self.w) * self.dice(logits, targets)


def compute_mean_std(dataset):
    # opsional saja, tak dipakai default karena kita sudah Normalize pakai ImageNet stats
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_images = 0

    for images, _ in tqdm(loader, desc="Computing mean/std"):
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = torch.from_numpy(images).permute(2, 0, 1).unsqueeze(0)
            elif images.ndim == 4:
                images = torch.from_numpy(images).permute(0, 3, 1, 2)
            images = images.float() / 255.0
        elif isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            elif images.ndim == 4 and images.shape[1] != 3:
                images = images.permute(0, 3, 1, 2)
            images = images.float()
            # diasumsikan sudah 0..1 jika dari ToTensorV2; jika tidak, uncomment:
            # images = images / 255.0

        b = images.size(0)
        n_images += b
        mean += images.mean(dim=[0, 2, 3]) * b
        std  += images.std(dim=[0, 2, 3])  * b

    mean /= n_images
    std /= n_images
    return mean.tolist(), std.tolist()


# def get_transforms(img_size=(image_height, image_width)):
#     return A.Compose([
#         # A.HorizontalFlip(p=0.5),
#         A.RandomBrightnessContrast(p=0.35),
#         # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
#         A.GaussianBlur(p=0.2),
#         A.Resize(*img_size),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ])

# def get_transforms(img_size=(image_height, image_width), is_train=True):
#     photometric_strong = A.OneOf([
#         A.RandomGamma(gamma_limit=(40, 120), p=1.0),               # exposure
#         A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=1.0),
#         A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
#         A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
#         A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
#     ], p=0.9)

#     outdoor_effects = A.OneOf([
#         A.RandomShadow(p=1.0),     # bayangan keras
#         A.RandomSunFlare(src_radius=80, flare_roi=(0,0,1,0.5), angle_lower=0.5, num_flare_circles_lower=2, p=1.0),
#         A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.15, p=1.0),
#     ], p=0.87)

#     base = [
#         A.Resize(*img_size),
#         photometric_strong if is_train else A.NoOp(),
#         outdoor_effects if is_train else A.NoOp(),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ]
#     return A.Compose(base)

def get_transforms(img_hw=(360,640), is_train=True, use_outdoor=False, use_sensor=False, micro_geo=True):
    H,W = img_hw
    if not is_train:
        return A.Compose([
            A.Resize(H, W),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])

    photometric = A.OneOf([
        A.RandomGamma(gamma_limit=(40,120), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.25, p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=1.0),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
    ], p=0.7)

    outdoor = A.OneOf([
        A.RandomShadow(p=1.0),
        A.RandomSunFlare(p=1.0),
        A.RandomFog(p=1.0),  # aktifkan nanti kalau model sudah stabil
    ], p=0.7) if use_outdoor else A.NoOp()

    comp = A.ImageCompression(quality_lower=60, quality_upper=95, p=1.0) if hasattr(A,"ImageCompression") else A.NoOp()
    sensor = A.SomeOf([
        A.ISONoise(color_shift=(0.01,0.05), intensity=(0.05,0.2), p=1.0),
        A.MotionBlur(blur_limit=3, p=1.0),
        comp,
        A.RandomToneCurve(scale=0.1, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
    ], n=2, p=0.7) if use_sensor else A.NoOp()

    geo = A.Affine(scale=(0.98,1.02), translate_percent=(0.0,0.02), rotate=0, shear=0, p=0.35) if micro_geo else A.NoOp()

    return A.Compose([
        A.Resize(H, W),
        photometric,
        outdoor,
        sensor,
        geo,
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])


# ===========================================================
# VFast-SCNN building blocks
# ===========================================================
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# ===========================================================
# VFast-SCNN + GRU (temporal head)
# ===========================================================
class VFastSCNN_GRU(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # ====== Learning to Downsample ======
        self.downsample = nn.Sequential(
            ConvBNReLU(3, layer1, 3, 2, 1),
            ConvBNReLU(layer1, layer2, 3, 2, 1),
            ConvBNReLU(layer2, layer3, 3, 2, 1),
        )

        # ====== Global Feature Extractor (depthwise 9x9 -> pw 1x1 -> GAP -> 1x1) ======
        self.gfe_dw_pw = nn.Sequential(
            nn.Conv2d(layer3, layer3, 3, padding=1, groups=layer3, bias=False),
            nn.BatchNorm2d(layer3),
            nn.ReLU(inplace=True),
            nn.Conv2d(layer3, layer4, 1, bias=False),
            nn.BatchNorm2d(layer4),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gfe_proj = nn.Conv2d(layer4, layer5, 1)  # -> (B, layer5, 1, 1)

        # ====== GRU temporal head ======
        # Input = layer5, Hidden = layer5 (ringan & konsisten)
        self.gru = nn.GRU(input_size=layer5, hidden_size=layer5, num_layers=1, batch_first=True)

        # ====== Classifier ======
        self.classifier = nn.Sequential(
            ConvBNReLU(layer3 + layer5, layer6, 3, 1, 1),
            nn.Dropout(0.1),
            nn.Conv2d(layer6, num_classes, 1),
        )

    @torch.jit.unused
    def reset_state(self):
        # Helper kalau mau simpan state internal (tidak dipakai default).
        self._h = None

    def forward(self, x, h: torch.Tensor = None, return_state: bool = False):
        """
        x: (B, 3, H, W)
        h: optional hidden state untuk GRU, shape (1, B, layer5)
        return_state: kalau True, kembalikan (out, h_new)
        """
        size = x.size()[2:]

        # 1) Downsample
        x_down = self.downsample(x)  # (B, layer3, H', W')

        # 2) Global feature -> vector (B, layer5)
        x_g = self.gfe_dw_pw(x_down)       # (B, layer4, H', W')
        x_g = self.gap(x_g)                # (B, layer4, 1, 1)
        x_g = self.gfe_proj(x_g)           # (B, layer5, 1, 1)

        # 3) GRU pada vektor global
        B, C, _, _ = x_g.shape  # C==layer5
        seq = x_g.view(B, 1, C)  # (B, T=1, C)
        out_seq, h_new = self.gru(seq, h)  # out_seq: (B,1,C), h_new: (1,B,C)

        # 4) Naikkan lagi ke spatial size H'×W', lalu fuse
        x_global_up = out_seq.view(B, C, 1, 1)
        x_global_up = F.interpolate(x_global_up, size=x_down.size()[2:], mode='bilinear', align_corners=True)

        x_cat = torch.cat([x_down, x_global_up], dim=1)

        # 5) Classifier + upsample ke resolusi asli
        out = self.classifier(x_cat)
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)

        if return_state:
            return out, h_new
        return out

# ===========================================================
# Utils
# ===========================================================
def binarize_from_logits(logits: torch.Tensor, thr: float = threshold_val):
    prob = torch.sigmoid(logits)
    mask = (prob.squeeze().cpu().numpy() > thr).astype(np.uint8)
    return mask, prob


def overlay_mask_on_image(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    # rgb: (H,W,3) uint8, mask: (H,W) 0/1
    overlay = np.zeros_like(rgb)
    overlay[:, :, 1] = 255  # green
    blended = rgb.copy()
    mask_b = mask.astype(bool)
    blended[mask_b] = cv2.addWeighted(rgb[mask_b], 1 - alpha, overlay[mask_b], alpha, 0)
    return blended

# ===========================================================
# Main
# ===========================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_training:
        train_dataset = DatasetLoader(
            image_dir=os.path.join(dataset_dir, "images"),
            mask_dir=os.path.join(dataset_dir, "masks"),
            # transform=get_transforms((image_height, image_width))
            transform=get_transforms((image_height, image_width), use_outdoor=True, use_sensor=True, micro_geo=False, is_train=True)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        pos_rate = []
        for _ in range(200):
            i = np.random.randint(0, len(train_dataset))
            _, m = train_dataset[i]
            pos_rate.append(m.float().mean().item())
        print("mean road ratio:", np.mean(pos_rate))

        model = VFastSCNN_GRU(num_classes=1).to(device)
        # pos_weight = torch.tensor([pos_weight_val], device=device)
        # # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # criterion = BCEDiceLoss(bce_w=0.6)

        pos_weight = torch.tensor([pos_weight_val], device=device)
        criterion = BCEDiceLoss(bce_weight=0.6, pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()

        prev_loss = float('inf')
        for ep in range(epoch):
            running = 0.0
            for images, masks in tqdm(train_loader, desc=f"Epoch {ep+1}/{epoch}"):
                images = images.to(device)
                masks = masks.to(device).float()  # (B,H,W)

                # forward (tanpa state, karena training single-frame)
                logits = model(images)            # (B,1,H,W)
                # logits = logits.squeeze(1)        # (B,H,W)

                loss = criterion(logits, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item()

            if hasattr(model, 'classifier'):
                print_grad_norm(model.classifier[-1], name="classifier.last_conv")
            print(f"Epoch {ep+1} Loss: {running / max(1, len(train_loader)):.4f}")

            # Simpan model jika loss menurun
            if running / max(1, len(train_loader)) < prev_loss:
                prev_loss = running / max(1, len(train_loader))
                print(f"Loss menurun, menyimpan model...")
                torch.save(model.state_dict(), pth_name)

        last_pth_name = "last_" + pth_name
        torch.save(model.state_dict(), last_pth_name)
        print(f"Model disimpan ke: {last_pth_name}")

    else:
        # ===== Inference Single Image + (opsional) Export ONNX =====
        import matplotlib.pyplot as plt

        # load model
        model = VFastSCNN_GRU(num_classes=1).to(device)
        if os.path.exists(pth_name):
            sd = torch.load(pth_name, map_location=device)
            model.load_state_dict(sd)
            print(f"State dict loaded: {pth_name}")
        model.eval()

        # Export ONNX (dengan dukungan hidden state)
        if is_export_onnx:
            dummy_x = torch.randn(1, 3, image_height, image_width, device=device)
            dummy_h = torch.zeros(1, 1, layer5, device=device)  # (num_layers=1, B=1, C=layer5)
            # Kita ekspor versi "return_state=True" agar keluar hN
            torch.onnx.export(
                model,
                (dummy_x, dummy_h, True),
                onnx_name,
                input_names=["input", "h0", "return_state"],
                output_names=["logits", "hN"],
                opset_version=11,
                dynamic_axes={
                    "input": {"0": "batch"},
                    "h0": {"1": "batch"},
                    "logits": {"0": "batch"},
                    "hN": {"1": "batch"},
                }
            )
            print(f"ONNX diekspor: {onnx_name}")

        # === Load & preprocess image ===
        test_img_path = check_image
        img_bgr = cv2.imread(test_img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Gambar tidak ditemukan: {test_img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        original_image = img_rgb.copy()

        transform = get_transforms((image_height, image_width))
        dummy_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        augmented = transform(image=img_rgb, mask=dummy_mask)
        input_tensor = augmented["image"].unsqueeze(0).to(device)  # (1,3,H,W)

        # === Inference single frame (tanpa state) ===
        with torch.no_grad():
            t0 = time.time()
            logits = model(input_tensor)              # (1,1,H,W)
            print("Logits stats:", logits.min().item(), logits.max().item())
            mask_small, prob = binarize_from_logits(logits, thr=threshold_val)
            dt = (time.time() - t0) * 1000
            print(f"Inference time: {dt:.1f} ms")

        # Resize mask ke ukuran asli
        mask_resized = cv2.resize(mask_small, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        blended = overlay_mask_on_image(original_image, mask_resized, alpha=0.5)

        # === Visualisasi ===
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.title("Original"); plt.imshow(original_image); plt.axis("off")
        plt.subplot(1, 3, 2); plt.title("Predicted Mask"); plt.imshow(mask_resized, cmap='gray'); plt.axis("off")
        plt.subplot(1, 3, 3); plt.title("Overlay"); plt.imshow(blended); plt.axis("off")
        plt.tight_layout()
        plt.show()

        # ====== (Opsional) Contoh streaming dengan state ======
        # Contoh pseudo:
        # h = None
        # for frame_rgb in stream_frames():
        #     aug = transform(image=frame_rgb, mask=np.zeros(frame_rgb.shape[:2], np.uint8))
        #     x = aug["image"].unsqueeze(0).to(device)
        #     with torch.no_grad():
        #         logits, h = model(x, h=h, return_state=True)
        #         mask, _ = binarize_from_logits(logits, thr=threshold_val)
        #     # ... render mask ...
        # # jika ganti video/scene: h = None
