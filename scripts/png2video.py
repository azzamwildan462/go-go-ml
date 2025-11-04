import cv2
import os
import glob
import re

# === Settings ===
# frames_folder = "/home/wildan/Desktop/map_pra_final/images/rgb"                 # folder containing your PNGs
frames_folder = "/home/wildan/Desktop/lagi_map_baru/camera/rgb"                 # folder containing your PNGs
output_video  = "uji_coba_lambat.mp4"
fps = 10                                   # set your desired FPS
 
def extract_id(path: str) -> int:
    """
    Return a numeric ID from the filename.
    - If the whole stem is digits (e.g., '12.png'), use that.
    - Otherwise, use the LAST number found (e.g., 'foo_12_bar_003.png' -> 3).
    - Returns -1 if no number is found (those frames will be skipped).
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.isdigit():
        return int(stem)
    nums = re.findall(r"\d+", stem)
    return int(nums[-1]) if nums else -1

# === Collect and sort frames by numeric ID ===
all_pngs = glob.glob(os.path.join(frames_folder, "*.png"))
pairs = [(extract_id(p), p) for p in all_pngs]
pairs = [pr for pr in pairs if pr[0] != -1]          # drop files without an ID
pairs.sort(key=lambda x: x[0])                       # sort by numeric ID
frame_files = [p for _, p in pairs]

if not frame_files:
    raise ValueError(f"No usable PNG files with numeric IDs found in '{frames_folder}'")

# === Read the first frame to get dimensions ===
first = cv2.imread(frame_files[0])
if first is None:
    raise RuntimeError(f"Failed to read first frame: {frame_files[0]}")
h, w = first.shape[:2]

# === Create the video writer ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")            # codec for .mp4
out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

# === Write frames (resizing any mismatched images to match the first) ===
written = 0
for path in frame_files:
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️  Skipping unreadable image: {path}")
        continue
    if img.shape[1] != w or img.shape[0] != h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    out.write(img)
    written += 1

out.release()
print(f"✅ Video saved as '{output_video}' ({written} frames, {fps} fps) sorted by numeric ID.")
