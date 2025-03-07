import os
import numpy as np
import shutil
from tqdm import tqdm

# ✅ Define paths
data_folder = "data/images"
output_folder = "npz_data"
os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

# ✅ Process each family folder
for family in tqdm(os.listdir(data_folder), desc="Compressing Families"):
    family_folder = os.path.join(data_folder, family)
    if not os.path.isdir(family_folder):  # Skip if not a directory
        continue

    npy_files = sorted(os.listdir(family_folder))  # Get all .npy files
    images = []

    # ✅ Load .npy files into memory using `with`
    for npy_file in tqdm(npy_files, desc=f"Processing {family}", leave=False):
        npy_path = os.path.join(family_folder, npy_file)
        with open(npy_path, "rb") as f:  # ✅ Ensure safe file handling
            image = np.load(f)
            images.append(image)

    # ✅ Save as compressed .npz
    npz_path = os.path.join(output_folder, f"{family}.npz")
    with open(npz_path, "wb") as f:  # ✅ Ensure safe file writing
        np.savez_compressed(f, images=np.array(images, dtype=object))  # Keep dtype=object for variable sizes

    # ✅ Delete family folder after compression
    shutil.rmtree(family_folder)
    print(f"✅ Compressed {family} and deleted folder.")