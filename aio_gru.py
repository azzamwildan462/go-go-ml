#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VFast-SCNN + GRU (temporal) — Video Inference
Fitur:
- Preproc robust lighting: Gray-World White Balance, CLAHE(L), Adaptive Gamma
- Mini-TTA gamma (avg prob)
- Post-proc: ambil kontur terbesar (opsional: yang menyentuh bawah) & isi bolong
- GRU stateful antar frame
- Output video side-by-side: original | mask | overlay

Contoh:
python3 infer_video_vfast_gru_robust.py \
  --weights ayo.pth --video /path/in.mp4 --out out.mp4 --show \
  --thr 0.45 --touch_bottom --min_area 1500 --ksize 7 \
  --use_wb --use_clahe --use_adaptive_gamma --tta_gamma 0.9,1.1

  python3 aio_gru.py --weights woilah_beneran_serius_ayo.pth --video vids/1_rgb_output.mp4 --thr 0.0025 --out new_boss.mp4 --show --reset_on_scene_cut --use_wb --use_clahe --use_adaptive_gamma
  python3 aio_gru.py --weights yo_yoks.pth --video vids/2_rgb_output.mp4 --thr 0.72 --out ik.mp4 --show --use_wb --use_clahe --use_adaptive_gamma

"""

import os
import cv2
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --------------------
# Channel config (sesuai set terakhirmu)
# --------------------
LAYER1 = 32
LAYER2 = 48
LAYER3 = 64
LAYER4 = 96
LAYER5 = 128  # juga dim GRU
LAYER6 = 64

DEFAULT_W = 640
DEFAULT_H = 360

def _line_kernel(size: int, angle_deg: int) -> np.ndarray:
    """
    Bikin kernel garis tipis berukuran `size` dengan orientasi `angle_deg`.
    - size ganjil (3,5,7, ...)
    """
    size = max(3, int(size) | 1)  # paksa ganjil
    k = np.zeros((size, size), np.uint8)
    c = size // 2
    # titik start & end di tepi kernel sesuai sudut
    ang = np.deg2rad(angle_deg)
    r = c
    x0 = int(round(c - r * np.cos(ang)))
    y0 = int(round(c - r * np.sin(ang)))
    x1 = int(round(c + r * np.cos(ang)))
    y1 = int(round(c + r * np.sin(ang)))
    cv2.line(k, (x0, y0), (x1, y1), 1, 1)  # tebal 1 px
    return k

def cut_thin_connections(mask_bin: np.ndarray,
                         max_bridge_px: int = 3,
                         iters: int = 1,
                         orientations=(0, 45, 90, 135)) -> np.ndarray:
    """
    Putus jembatan sempit antar area putih (1/255) dengan morphological opening
    multi-orientasi. Bridge dengan lebar <= max_bridge_px akan terpotong.

    mask_bin: HxW, dtype uint8, nilai {0,1} atau {0,255}
    """
    # pastikan 0/255
    if mask_bin.max() <= 1:
        m = (mask_bin * 255).astype(np.uint8)
    else:
        m = mask_bin.astype(np.uint8)

    out = m.copy()
    for _ in range(max(1, iters)):
        for ang in orientations:
            k = _line_kernel(size=max_bridge_px, angle_deg=ang)
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)

    # balikin ke {0,1}
    return (out > 127).astype(np.uint8)

def cut_thin_necks_disk(mask_bin: np.ndarray, radius_px: int = 2, iters: int = 1) -> np.ndarray:
    """
    Alternatif yang lebih agresif (isotropik): opening dengan kernel disk.
    Memutus "leher" sempit (neck) dengan diameter <= 2*radius_px + 1.
    """
    if mask_bin.max() <= 1:
        m = (mask_bin * 255).astype(np.uint8)
    else:
        m = mask_bin.astype(np.uint8)

    ksz = 2 * radius_px + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    out = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=max(1, iters))
    return (out > 127).astype(np.uint8)


# =========================
# Util: Lighting Robustness
# =========================
def gray_world_wb(img_rgb: np.ndarray) -> np.ndarray:
    # img_rgb uint8 RGB
    img = img_rgb.astype(np.float32)
    mean_r = img[:, :, 0].mean()
    mean_g = img[:, :, 1].mean()
    mean_b = img[:, :, 2].mean()
    mean_gray = (mean_r + mean_g + mean_b) / 3.0 + 1e-6
    gains = np.array([mean_gray/(mean_r+1e-6),
                      mean_gray/(mean_g+1e-6),
                      mean_gray/(mean_b+1e-6)], dtype=np.float32)
    img *= gains.reshape(1,1,3)
    return np.clip(img, 0, 255).astype(np.uint8)

def clahe_rgb(img_rgb: np.ndarray, clip=2.0, tile=8) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

def adaptive_gamma(img_rgb: np.ndarray, target_mean=0.5, lo=0.5, hi=2.0):
    # Set gamma agar mean luminance ~ target_mean
    ycc = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y = ycc[:, :, 0].astype(np.float32) / 255.0
    m = float(y.mean() + 1e-6)
    gamma = np.log(target_mean)/(np.log(m))
    gamma = float(np.clip(gamma, lo, hi))
    out = np.clip((img_rgb.astype(np.float32)/255.0) ** gamma, 0, 1)
    return (out*255).astype(np.uint8), gamma

# =========================
# Albumentations (resize+norm)
# =========================
def get_preproc(img_h, img_w):
    return A.Compose([
        A.Resize(img_h, img_w),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ===============
# Model blocks
# ===============
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class VFastSCNN_GRU(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.downsample = nn.Sequential(
            ConvBNReLU(3, LAYER1, 3, 2, 1),
            ConvBNReLU(LAYER1, LAYER2, 3, 2, 1),
            ConvBNReLU(LAYER2, LAYER3, 3, 2, 1),
        )
        self.gfe_dw_pw = nn.Sequential(
            nn.Conv2d(LAYER3, LAYER3, 3, padding=1, groups=LAYER3, bias=False),
            nn.BatchNorm2d(LAYER3),
            nn.ReLU(inplace=True),
            nn.Conv2d(LAYER3, LAYER4, 1, bias=False),
            nn.BatchNorm2d(LAYER4),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gfe_proj = nn.Conv2d(LAYER4, LAYER5, 1)  # (B,L5,1,1)
        self.gru = nn.GRU(input_size=LAYER5, hidden_size=LAYER5, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            ConvBNReLU(LAYER3 + LAYER5, LAYER6, 3, 1, 1),
            nn.Dropout(0.1),
            nn.Conv2d(LAYER6, num_classes, 1),
        )
    def forward(self, x, h=None, return_state=False):
        size = x.size()[2:]
        x_down = self.downsample(x)
        x_g = self.gfe_dw_pw(x_down)
        x_g = self.gap(x_g)
        x_g = self.gfe_proj(x_g)
        B, C, _, _ = x_g.shape
        seq = x_g.view(B, 1, C)
        out_seq, h_new = self.gru(seq, h)
        x_global_up = out_seq.view(B, C, 1, 1)
        x_global_up = F.interpolate(x_global_up, size=x_down.size()[2:], mode='bilinear', align_corners=True)
        x_cat = torch.cat([x_down, x_global_up], dim=1)
        out = self.classifier(x_cat)
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        if return_state: return out, h_new
        return out

# =========================
# Post-processing helpers
# =========================
def overlay_mask(bgr, mask, alpha=0.5):
    # bgr HxWx3 uint8, mask HxW {0,1}
    mb = mask.astype(bool)
    overlay = np.zeros_like(bgr)
    overlay[:, :, 1] = 255
    mixed = cv2.addWeighted(bgr, 1 - alpha, overlay, alpha, 0)
    out = bgr.copy()
    if mb.any(): out[mb] = mixed[mb]
    return out

def side_by_side(bgr, mask_bin, overlay_bgr):
    m = (mask_bin * 255).astype(np.uint8)
    m3 = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    return np.concatenate([bgr, m3, overlay_bgr], axis=1)

def largest_contour_filled(bin_mask, min_area=500, touch_bottom=True, ksize=5):
    """
    Pilih kontur terbesar (opsional: yang menyentuh bawah), isi penuh (holes tertutup).
    bin_mask: HxW uint8 {0,1/255} -> return {0,1}
    """
    m = bin_mask.astype(np.uint8)
    if m.max() == 1: m *= 255
    H, W = m.shape[:2]
    if ksize and ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (m > 0).astype(np.uint8)

    def touches_bottom(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        return (y + h) >= (H - 2)

    cands = contours
    if touch_bottom:
        tb = [c for c in contours if touches_bottom(c)]
        if tb: cands = tb

    best, best_area = None, -1
    for c in cands:
        area = cv2.contourArea(c)
        if area >= min_area and area > best_area:
            best, best_area = c, area
    if best is None:
        for c in cands:
            area = cv2.contourArea(c)
            if area > best_area:
                best, best_area = c, area
        if best is None:
            return (m > 0).astype(np.uint8)

    filled = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(filled, [best], -1, 255, thickness=cv2.FILLED)
    return (filled > 0).astype(np.uint8)

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path .pth (state_dict)")
    ap.add_argument("--video", type=str, required=True, help="Path video atau index webcam (mis. 0)")
    ap.add_argument("--out", type=str, default="out.mp4", help="Output video")
    ap.add_argument("--w", type=int, default=DEFAULT_W, help="Width inference (default 640)")
    ap.add_argument("--h", type=int, default=DEFAULT_H, help="Height inference (default 360)")
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold sigmoid -> biner")
    ap.add_argument("--cuda", action="store_true", help="Pakai CUDA jika ada")
    ap.add_argument("--show", action="store_true", help="Preview window")
    ap.add_argument("--reset_on_scene_cut", action="store_true", help="Reset GRU kalau scene cut")
    # Lighting toggles
    ap.add_argument("--use_wb", action="store_true", help="Enable Gray-World WB")
    ap.add_argument("--use_clahe", action="store_true", help="Enable CLAHE(L)")
    ap.add_argument("--use_adaptive_gamma", action="store_true", help="Enable Adaptive Gamma")
    # TTA
    ap.add_argument("--tta_gamma", type=str, default="", help="Comma list gamma (e.g. 0.9,1.1)")
    # Post-proc
    ap.add_argument("--touch_bottom", action="store_true", help="Pilih kontur yang menyentuh bawah")
    ap.add_argument("--min_area", type=int, default=500, help="Min area kontur")
    ap.add_argument("--ksize", type=int, default=5, help="Kernel morph close (0=skip)")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    # Model
    model = VFastSCNN_GRU(num_classes=1).to(device)
    sd = torch.load(args.weights, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"Loaded weights: {args.weights}")

    # Video input (file/webcam)
    try:
        cam_index = int(args.video)
        cap = cv2.VideoCapture(cam_index)
    except ValueError:
        cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal buka video: {args.video}")

    in_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    if math.isnan(fps) or fps <= 0: fps = 25.0
    print(f"Input: {in_w}x{in_h} @ {fps:.2f} FPS")

    out_w, out_h = in_w*3, in_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Gagal membuat writer: {args.out}")

    preproc = get_preproc(args.h, args.w)

    # TTA gamma list
    tta_list = []
    if args.tta_gamma.strip():
        try:
            tta_list = [float(x) for x in args.tta_gamma.split(",") if x.strip()]
        except Exception:
            print("Peringatan: --tta_gamma tidak valid, diabaikan.")

    h_state = None
    prev_gray = None
    frame_idx, t_infer_total = 0, 0.0

    print("Mulai inferensi...")
    while True:
        ret, frame_bgr = cap.read()
        if not ret: break

        # Scene-cut reset (opsional)
        if args.reset_on_scene_cut:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                hist1 = cv2.calcHist([prev_gray],[0],None,[64],[0,256])
                hist2 = cv2.calcHist([gray],[0],None,[64],[0,256])
                cv2.normalize(hist1, hist1); cv2.normalize(hist2, hist2)
                diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
                if diff > 0.35: h_state = None
            prev_gray = gray

        # BGR->RGB + robust lighting preproc
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if args.use_wb: frame_rgb = gray_world_wb(frame_rgb)
        if args.use_clahe: frame_rgb = clahe_rgb(frame_rgb, clip=2.0, tile=8)
        if args.use_adaptive_gamma:
            frame_rgb, _ = adaptive_gamma(frame_rgb, target_mean=0.5)

        # Inference (dengan TTA gamma jika ada)
        with torch.no_grad():
            t0 = time.time()
            if tta_list:
                probs = []
                for g in tta_list:
                    g_img = np.clip((frame_rgb.astype(np.float32)/255.0) ** g, 0, 1)
                    g_img = (g_img*255).astype(np.uint8)
                    aug = preproc(image=g_img)
                    x = aug["image"].unsqueeze(0).to(device)
                    logits, h_state = model(x, h=h_state, return_state=True)
                    probs.append(torch.sigmoid(logits))
                prob = torch.mean(torch.stack(probs, dim=0), dim=0)  # (1,1,H,W)
            else:
                aug = preproc(image=frame_rgb)
                x = aug["image"].unsqueeze(0).to(device)
                logits, h_state = model(x, h=h_state, return_state=True)
                prob = torch.sigmoid(logits)
            t_infer = (time.time() - t0) * 1000.0
        t_infer_total += t_infer

        # Threshold & resize ke ukuran asli
        mask_small = (prob.squeeze().cpu().numpy() > args.thr).astype(np.uint8)
        H0, W0 = frame_bgr.shape[:2]
        mask = cv2.resize(mask_small, (W0, H0), interpolation=cv2.INTER_NEAREST)

        # Cut koneksi kecil antar contour 
        mask = cut_thin_connections(mask, max_bridge_px=7, iters=1)

        # 2) (opsional) tambah pemotongan isotropik kalau masih ada leher kecil
        mask = cut_thin_necks_disk(mask, radius_px=max(1, 7//2), iters=1)

        # not 
        mask_bottom_rectangle = np.zeros_like(mask)
        mask_bottom_rectangle[H0-100:H0, :] = 1
        mask_bottom_rectangle = cv2.bitwise_not(mask_bottom_rectangle)

        # and 
        mask = cv2.bitwise_and(mask, mask_bottom_rectangle)

        # Post-proc: kontur terbesar + isi bolong
        mask = largest_contour_filled(
            mask, min_area=args.min_area,
            touch_bottom=args.touch_bottom, ksize=args.ksize
        )


        overlay = overlay_mask(frame_bgr, mask, alpha=0.5)
        # canvas = side_by_side(frame_bgr, mask_bottom_rectangle, overlay)
        writer.write(overlay)

        if args.show:
            cv2.imshow("VFast-SCNN-GRU (orig | mask | overlay)  [q=quit, r=reset]", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'): h_state = None

        frame_idx += 1

    cap.release(); writer.release()
    if args.show: cv2.destroyAllWindows()
    if frame_idx > 0:
        print(f"Selesai. Disimpan: {args.out}")
        print(f"Avg inferensi: {t_infer_total/frame_idx:.2f} ms/frame | thr={args.thr} | TTA={tta_list}")
    else:
        print("Tidak ada frame terproses.")

if __name__ == "__main__":
    main()
