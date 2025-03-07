import os
import numpy as np
from tqdm import tqdm

# Paths
npz_folder = "npz_data"  # Folder containing existing .npz files
output_folder = "data/images"  # Folder to save extracted .npy files
os.makedirs(output_folder, exist_ok=True)

# Process each .npz file
for npz_file in tqdm(os.listdir(npz_folder), desc="Processing Families"):
    if not npz_file.endswith(".npz"):
        continue

    npz_path = os.path.join(npz_folder, npz_file)
    family = npz_file.split(".")[0]  # Extract family name

    # Create folder for the family
    family_folder = os.path.join(output_folder, family)
    os.makedirs(family_folder, exist_ok=True)

    with np.load(npz_path, allow_pickle=True) as data:  # Ensures automatic file closure
        images = data["images"]  # Extract stored images

        # Save each image as an individual .npy file
        for idx, image in tqdm(enumerate(images), desc=f"Extracting {family}", leave=False):
            np.save(os.path.join(family_folder, f"{idx}.npy"), image)
            
    # Delete old .npz file after extraction
    os.remove(npz_path)
    print(f"✅ Extracted {len(images)} images from {npz_file} and deleted the original .npz")