import os
import glob
import subprocess
import shutil

# Paths
json_dir = "../datasets/v3"     # <- your folder with JSONs
output_img_dir = "../datasets/used_v3/images"
output_mask_dir = "../datasets/used_v3/masks"

# Create output folders
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Find all json files
json_files = glob.glob(os.path.join(json_dir, "*.json"))

for json_file in json_files:
    print(f"Processing: {json_file}")
    
    # Convert using labelme_json_to_dataset
    subprocess.run(["labelme_json_to_dataset", json_file])
    
    # Get filename base
    base = os.path.splitext(os.path.basename(json_file))[0]
    temp_dir = os.path.join(json_dir, base + "_json")

    # Move the original image
    img_path = os.path.join(temp_dir, "img.png")
    if os.path.exists(img_path):
        shutil.copy(img_path, os.path.join(output_img_dir, base + ".png"))

    # Move the mask
    label_path = os.path.join(temp_dir, "label.png")
    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(output_mask_dir, base + ".png"))

    # Clean up temp conversion folder
    shutil.rmtree(temp_dir)

print("✅ Done! Dataset ready in {output_img_dir} and {output_mask_dir}")
