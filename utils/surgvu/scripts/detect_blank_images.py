import os
import argparse
import subprocess
from pathlib import Path
import logging
from PIL import Image, UnidentifiedImageError
import numpy as np
from tqdm import tqdm 
from functools import partial

logging.basicConfig(
    filename="detect_blank_images.log",
    filemode="w",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()

def is_blank_or_corrupt(image_path, threshold=10):
    try:
        with Image.open(image_path) as img:
            img = img.convert("L")  # convert to grayscale
            img_array = np.array(img)
            if img_array.mean() < threshold:
                return True, "black"
    except (UnidentifiedImageError, OSError):
        return True, "corrupt"
    return False, None

def worker(path):
    is_invalid, reason = is_blank_or_corrupt(path)
    if is_invalid:
        logger.info(f"Found invalid image: {path} (reason: {reason})")
        return str(path)
    return None

def main():
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory to scan for .jpg files")
    parser.add_argument("--output_file", type=str, default="invalid_images.txt", help="File to save paths of invalid images")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    jpg_paths = list(input_dir.rglob("*.jpg"))
    total = len(jpg_paths)
    print(f"Found {total} .jpg files.")
    print("Scanning for invalid images...")
    logger.info("Scanning for invalid images...")

    invalid_paths = []
    with multiprocessing.Pool(processes=min(32, multiprocessing.cpu_count())) as pool:
        for result in tqdm(pool.imap_unordered(worker, jpg_paths, chunksize=100), total=total, desc="Processing images"):
            if result is not None:
                invalid_paths.append(result)

    with open(output_file, "w") as f:
        for path in invalid_paths:
            f.write(path + "\n")

    logger.info(f"Found {len(invalid_paths)} invalid images. Paths written to {output_file}.")
if __name__ == "__main__":
    main()
