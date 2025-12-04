import cv2
import os
import glob
from tqdm import tqdm
import argparse

def extract_frames(video_path, output_dir, sample_rate=1):
    """
    Extracts frames from a video file.
    
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save extracted frames.
        sample_rate (int): Extract every Nth frame. Default is 1 (all frames).
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    saved_count = 0
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    while success:
        if count % sample_rate == 0:
            frame_name = f"{video_name}_frame{count:05d}.png"
            cv2.imwrite(os.path.join(video_output_dir, frame_name), image)
            saved_count += 1
        success, image = vidcap.read()
        count += 1
        
    vidcap.release()
    return saved_count

def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .mp4 videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted frames")
    parser.add_argument("--sample_rate", type=int, default=10, help="Extract every Nth frame (default: 10)")
    
    args = parser.parse_args()
    
    video_files = glob.glob(os.path.join(args.input_dir, "*.mp4"))
    print(f"Found {len(video_files)} videos in {args.input_dir}")
    
    if not video_files:
        print("No .mp4 files found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    total_frames = 0
    for video in tqdm(video_files, desc="Processing Videos"):
        frames_extracted = extract_frames(video, args.output_dir, args.sample_rate)
        total_frames += frames_extracted
        
    print(f"Extraction complete. Total {total_frames} frames saved to {args.output_dir}")

if __name__ == "__main__":
    main()
