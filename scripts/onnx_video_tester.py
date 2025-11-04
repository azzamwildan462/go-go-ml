import os
import cv2
import time
import numpy as np
import onnxruntime as ort

# === CONFIG ===
# onnx_model_path = "bismillah1.onnx"  # Your ONNX model # bagus tanpa au
onnx_model_path = "bismillah2_augpt2.onnx"  # Your ONNX model
video_path = "uji_coba_lambat.mp4"              # Input video path
output_video_path = "hasil_uji_coba_aug1.mp4"  # Output video file
# video_path = "uji_coba_beda.mp4"              # Input video path
# output_video_path = "hasil_uji_coba6_beda_1.mp4"  # Output video file
image_height, image_width = 360, 640        # Resize input to model shape
r_check = 360  # Radius for circle check
circle_area = np.pi * (r_check ** 2) * 0.5 # Karena setengah lingkaran

max_speed = 2.2 # meter per second
min_speed = 0.55
buffer_speed = 0.0
cf_new_speed_gain_down = 0.055 # 0.32
cf_new_speed_gain_up = 0.255 # 0.32

# === PREPROCESSING FUNCTION ===
def preprocess_image_fast(image, size=(image_height, image_width)):
    image_resized = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    image_float = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_normalized = (image_float - mean) / std
    image_chw = np.transpose(image_normalized, (2, 0, 1))
    input_tensor = np.expand_dims(image_chw, axis=0)
    return input_tensor

# === POSTPROCESS FUNCTION ===
def postprocess_output(outputs, threshold=0.80):
    logits = outputs[0]
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    binary_mask = (probabilities > threshold).astype(np.uint8) * 255
    return np.squeeze(binary_mask)

# === MAIN PIPELINE ===
def main():
    global buffer_speed

    # Load ONNX model
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])  # or ['CUDAExecutionProvider']
    input_name = session.get_inputs()[0].name
    print("Loaded model:", onnx_model_path)
    print("Processing video:", video_path)

    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_inference_time = 0
    total_frames = 0

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mencatatkan waktu untuk inferensi
        time_start = time.time()

        original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess_image_fast(original_rgb)

        # ONNX inference
        outputs = session.run(None, {input_name: input_tensor})
        elapsed = time.time() - time_start

        # Postprocess
        pred_mask = postprocess_output(outputs)
        pred_mask_resized = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        pred_mask_resized_orig = pred_mask_resized.copy()

        # Morphological filtering
        # kernel = np.ones((7, 7), np.uint8)
        # pred_mask_resized = cv2.erode(pred_mask_resized, kernel, iterations=1)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        # pred_mask_resized = cv2.morphologyEx(pred_mask_resized, cv2.MORPH_CLOSE, kernel)
        # pred_mask_resized = cv2.morphologyEx(pred_mask_resized, cv2.MORPH_OPEN, kernel)

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (75, 75))  # or (3,3)
        # opened = cv2.morphologyEx(pred_mask_resized, cv2.MORPH_OPEN, kernel)
        # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=4)
        # areas = stats[1:, cv2.CC_STAT_AREA]  # exclude background
        # max_label = 1 + np.argmax(areas)
        # pred_mask_resized = np.uint8(labels == max_label) * 255

        # # Keep only the largest blob
        # contours, _ = cv2.findContours(pred_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if contours:
        #     largest_contour = max(contours, key=cv2.contourArea)
        #     pred_mask_resized = np.zeros_like(pred_mask_resized)
        #     cv2.drawContours(pred_mask_resized, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Way circle around the center of the mask
        circle_to_check = np.zeros_like(pred_mask_resized)
        cv2.circle(circle_to_check, (pred_mask_resized.shape[1] // 2, pred_mask_resized.shape[0]), r_check, 255, thickness=-1)
        cv2.bitwise_and(pred_mask_resized, circle_to_check, dst=pred_mask_resized)

        # Find confidence level by the area of the mask and the mask circle
        mask_area = np.sum(pred_mask_resized > 0)
        confidence = mask_area / circle_area

        # Calculate desired speed
        speed_now = max_speed * (confidence ** 2)
        if speed_now < min_speed:
            speed_now = 0.0

        if buffer_speed > speed_now:
            buffer_speed = buffer_speed * (1 - cf_new_speed_gain_down) + speed_now * cf_new_speed_gain_down
        else:
            buffer_speed = buffer_speed * (1 - cf_new_speed_gain_up) + speed_now * cf_new_speed_gain_up

        # Overlay the raw pred_mask 
        overlay_color = np.zeros_like(original_rgb)
        overlay_color[:, :, 0] = 255
        alpha = 0.4
        mask_u8 = (pred_mask_resized_orig > 0).astype(np.uint8) * 255
        tinted  = cv2.addWeighted(original_rgb, 1 - alpha, overlay_color, alpha, 0)
        blended = original_rgb.copy()
        cv2.copyTo(tinted, mask_u8, blended)

        original_rgb = blended.copy()
        
        # Overlay mask on original frame
        overlay_color = np.zeros_like(original_rgb)
        overlay_color[:, :, 1] = 255  # Green overlay
        alpha = 0.5
        mask_u8 = (pred_mask_resized > 0).astype(np.uint8) * 255
        tinted  = cv2.addWeighted(original_rgb, 1 - alpha, overlay_color, alpha, 0)
        blended = original_rgb.copy()
        cv2.copyTo(tinted, mask_u8, blended)



        # Convert back to BGR for saving
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        # Draw confidence level on the frame
        cv2.putText(blended_bgr, f"Confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
        cv2.putText(blended_bgr, f"Speed_Vis: {buffer_speed:.2f} m/s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)

        out_video.write(blended_bgr)

        total_inference_time += elapsed
        total_frames += 1

        # print(f"[Frame {frame_idx:04d}] Inference time: {elapsed*1000:.1f} ms")
        frame_idx += 1

    # Calculate average inference time
    avg_inference_time = total_inference_time / total_frames if total_frames > 0 else 0
    print(f"Average inference time: {avg_inference_time*1000:.1f} ms/frame")

    cap.release()
    out_video.release()
    print(f"✅ Saved result to {output_video_path}")

if __name__ == "__main__":
    main()
