import argparse
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(
    filename="delete_invalid_images.log",
    filemode="w",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="invalid_images.txt", help="Path to file listing invalid image paths")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be deleted without removing files")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Input file {input_file} does not exist.")
        return

    with open(input_file, "r") as f:
        paths = [line.strip() for line in f if line.strip()]

    deleted_count = 0
    for path_str in tqdm(paths):
        path = Path(path_str)
        if path.exists():
            if args.dry_run:
                logger.info(f"[DRY RUN] Would delete: {path}")
            else:
                try:
                    path.unlink()
                    logger.info(f"Deleted: {path}")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {path}: {e}")
        else:
            logger.warning(f"File not found: {path}")

    print(f"Processed {len(paths)} files. Deleted: {deleted_count}")
    logger.info(f"Total deleted: {deleted_count}")

if __name__ == "__main__":
    main()
