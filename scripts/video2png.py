import cv2
import os

# === Settings ===
video_path = "../vids/8_rgb_output.mp4"  # <-- replace with your .mp4 file path
namespace = "v8"
output_folder = os.path.join("..","datasets", namespace)
divider = 10  # <-- adjust if you want to skip frames (e.g., every 2nd frame, set to 2)

# === Create output directory ===
os.makedirs(output_folder, exist_ok=True)

# === Open the video ===
cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_frames = 0
success, frame = cap.read()

while success:
    frame_filename = os.path.join(output_folder, f"{namespace}_frame_{frame_count:05d}.png")

    if frame_count % divider == 0:
        cv2.imwrite(frame_filename, frame)  # Save frame as PNG
        saved_frames += 1

    frame_count += 1
    success, frame = cap.read()

cap.release()
print(f"✅ Done. {saved_frames} frames saved to '{output_folder}'")
