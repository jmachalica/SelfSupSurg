import os
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(
    filename="processing.log",
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()

def extract_frames(video_path, output_dir, fps_out):
    output_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        logger.error(f"Cannot write to output directory: {output_dir}")
        return
    log_path = output_dir / "ffmpeg_output.log" 

    cmd = [
        "ffmpeg",
        "-threads", "1",
        "-i", str(video_path),
        "-vf", f"fps={fps_out}",
        "-q:v", "2",
        "-nostdin",
        str(output_dir / "frame_%05d.jpg")
    ]
    try:
        with open(log_path, "w") as log_file:
            subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            # subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:  
        logger.error(f"Error while processing {video_path.name}: {e}")

def main():
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to folder containing .mp4 videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted frames")
    parser.add_argument("--fps_out", type=int, default=1, help="Target FPS for sampling (default: 1)")
    parser.add_argument("--max_files", type=int, default=-1, help="Max number of videos to process (default: unlimited)")
    args = parser.parse_args()

    max_workers = min(6, os.cpu_count() or 1)
    logger.info(f"Using {max_workers} worker threads.")

    video_paths = list(Path(args.input_dir).rglob("*.mp4"))
    if args.max_files > 0:
        video_paths = video_paths[:args.max_files]

    logger.info(f"Found {len(video_paths)} video(s) to process.")

    def process_video(video):
        out_path = Path(args.output_dir) / video.stem
        logger.info(f"Processing: {video.name}")
        start_time = time.time()
        extract_frames(video, out_path, args.fps_out)
        elapsed_time = time.time() - start_time
        logger.info(f"Finished {video.name} in {elapsed_time:.2f} seconds.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Adjust number of threads as needed
        executor.map(process_video, video_paths)

if __name__ == "__main__":
    main()
