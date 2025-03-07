import os
import zlib
import numpy as np
import cv2
from tqdm import tqdm

# Define malware families
FAMILIES = ["adware", "flooder", "ransomware", "dropper", "spyware", "packed",
            "crypto_miner", "file_infector", "installer", "worm", "downloader"]

# Define file size to image width mapping
WIDTH_MAPPING = [
    (10 * 1024, 32), (30 * 1024, 64), (60 * 1024, 128), (100 * 1024, 256),
    (200 * 1024, 384), (500 * 1024, 512), (1000 * 1024, 768), (float('inf'), 1024)
]

def get_image_width(file_size):
    """Determine image width based on file size."""
    for size, width in WIDTH_MAPPING:
        if file_size < size:
            return width
    return 1024

def binary_to_grayscale(binary_path):
    """Convert a zlib-compressed malware binary into a 2D grayscale NumPy array."""
    with open(binary_path, "rb") as f:
        compressed_data = f.read()
        try:
            byte_data = zlib.decompress(compressed_data)  # Decompress
        except zlib.error:
            print(f"❌ Failed to decompress {binary_path}")
            return None

    file_size = len(byte_data)
    width = get_image_width(file_size)
    height = int(np.ceil(file_size / width))

    # Convert bytes to uint8 NumPy array
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)

    # Pad with zeros if needed
    padded_data = np.pad(byte_array, (0, height * width - file_size), mode='constant', constant_values=0)

    # Reshape into a 2D array (grayscale image)
    return padded_data.reshape((height, width))

def process_binaries_by_family(input_folder, output_folder, png_folder):
    """Process binaries family-by-family, save .npz files, and save one PNG per family."""
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(png_folder, exist_ok=True)

    for family in FAMILIES:
        print(f"\n🔹 Processing family: {family}")

        family_images = []
        family_filenames = []
        png_saved = False  # Track if we saved a PNG for this family
        
        # Process one family at a time
        progress_bar = tqdm(os.listdir(input_folder), desc=f"Processing {family}")
        for filename in progress_bar:
            if not filename.startswith(family + "_"):  # Check if file belongs to this family
                continue

            binary_path = os.path.join(input_folder, filename)
            if not os.path.isfile(binary_path):
                continue  # Skip non-files

            image = binary_to_grayscale(binary_path)
            if image is not None:
                family_images.append(image)
                family_filenames.append(filename)

                # Save the first image from this family as a PNG
                if not png_saved:
                    png_path = os.path.join(png_folder, f"{family}.png")
                    cv2.imwrite(png_path, image)
                    print(f"🖼️ Saved sample PNG for {family}: {png_path}")
                    png_saved = True

        progress_bar.close()
        # Save family images to .npz (if there are valid images)
        if family_images:
            npz_path = os.path.join(output_folder, f"{family}.npz")
            np.savez_compressed(npz_path, images=np.array(family_images, dtype=object), filenames=family_filenames)
            print(f"✅ Saved {len(family_images)} images for {family} in {npz_path}")

        # **Clear memory before moving to the next family**
        del family_images, family_filenames

# Example Usage
input_folder = "data/binaries"  # Folder containing zlib-compressed binaries
output_folder = "npz_data"      # Output folder for per-family .npz files
png_folder = "sample_images"    # Folder to store one PNG per family
process_binaries_by_family(input_folder, output_folder, png_folder)