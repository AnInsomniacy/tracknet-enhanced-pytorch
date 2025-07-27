"""
Badminton Dataset Preprocessor

Processes raw badminton video datasets for machine learning training.
Extracts video frames, resizes to 512×288 resolution, and generates Gaussian
heatmaps for shuttlecock position detection in a streamlined pipeline.

Usage Examples:
    python preprocess_to_hdf5.py --source dataset --output dataset_preprocessed.h5
    python video_to_heatmap.py --source /path/to/data --output /path/to/output.h5 --sigma 5
    python video_to_heatmap.py --source dataset --sigma 4 --force

Dependencies:
    pip install opencv-python pandas numpy scipy tqdm h5py
"""

import argparse
import os
import sys
import gc
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import h5py
from scipy.stats import multivariate_normal
from tqdm import tqdm

IGNORED_FILES = {'.DS_Store', 'Thumbs.db', '.gitignore', '.gitkeep'}
IGNORED_DIRS = {'.git', '__pycache__', '.vscode', '.idea', 'node_modules'}

TARGET_WIDTH = 512
TARGET_HEIGHT = 288


def is_valid_path(name):
    if name.startswith('.') and name not in {'.', '..'}:
        return False
    return name not in IGNORED_FILES and name not in IGNORED_DIRS


def generate_heatmap(center_x, center_y, width=TARGET_WIDTH, height=TARGET_HEIGHT, sigma=3):
    x_coords = np.arange(0, width)
    y_coords = np.arange(0, height)
    mesh_x, mesh_y = np.meshgrid(x_coords, y_coords)
    coordinates = np.dstack((mesh_x, mesh_y))

    gaussian_mean = [center_x, center_y]
    covariance_matrix = [[sigma ** 2, 0], [0, sigma ** 2]]

    distribution = multivariate_normal(gaussian_mean, covariance_matrix)
    heatmap = distribution.pdf(coordinates)

    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap_uint8 = (heatmap_normalized * 255).astype(np.uint8)

    return heatmap_uint8


def resize_with_aspect_ratio(image, target_w=TARGET_WIDTH, target_h=TARGET_HEIGHT):
    original_h, original_w = image.shape[:2]

    scale_width = target_w / original_w
    scale_height = target_h / original_h
    scale_factor = min(scale_width, scale_height)

    new_width = int(original_w * scale_factor)
    new_height = int(original_h * scale_factor)

    resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    offset_x = (target_w - new_width) // 2
    offset_y = (target_h - new_height) // 2
    canvas[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized_img

    return canvas, scale_factor, offset_x, offset_y


def transform_annotation_coords(x, y, scale, offset_x, offset_y):
    transformed_x = x * scale + offset_x
    transformed_y = y * scale + offset_y
    return transformed_x, transformed_y


def validate_dataset_structure(source_path):
    if not os.path.exists(source_path):
        return False, f"Source path does not exist: {source_path}"

    entries = [item for item in os.listdir(source_path) if is_valid_path(item)]
    match_dirs = [
        item for item in entries
        if item.startswith("match") and os.path.isdir(os.path.join(source_path, item))
    ]

    if not match_dirs:
        return False, "No match directories found"

    valid_matches = 0
    video_count = 0
    annotation_count = 0

    for match_dir in match_dirs:
        match_path = os.path.join(source_path, match_dir)
        annotations_path = os.path.join(match_path, "csv")
        videos_path = os.path.join(match_path, "video")

        if os.path.exists(annotations_path) and os.path.exists(videos_path):
            valid_matches += 1

            csv_files = [f for f in os.listdir(annotations_path)
                         if f.endswith('_ball.csv') and is_valid_path(f)]
            annotation_count += len(csv_files)

            mp4_files = [f for f in os.listdir(videos_path)
                         if f.endswith('.mp4') and is_valid_path(f)]
            video_count += len(mp4_files)

    if valid_matches == 0:
        return False, "No valid match directories found (must contain both csv and video subdirectories)"

    summary = f"Found {valid_matches} match directories, {video_count} videos, {annotation_count} annotation files"
    return True, summary


def estimate_total_frames(source_path):
    frame_total = 0
    match_dirs = [
        item for item in os.listdir(source_path) if is_valid_path(item)
                                                    and item.startswith("match") and os.path.isdir(
            os.path.join(source_path, item))
    ]

    for match_dir in match_dirs:
        match_path = os.path.join(source_path, match_dir)
        videos_path = os.path.join(match_path, "video")

        if os.path.exists(videos_path):
            mp4_files = [f for f in os.listdir(videos_path)
                         if f.endswith('.mp4') and is_valid_path(f)]

            for mp4_file in mp4_files:
                video_path = os.path.join(videos_path, mp4_file)
                video_capture = cv2.VideoCapture(video_path)
                if video_capture.isOpened():
                    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_total += total_frames
                video_capture.release()

    return frame_total


def process_video_sequence(video_path, annotation_path, h5_file, match_name, sequence_name, sigma_value, progress_bar):
    if not os.path.exists(annotation_path):
        tqdm.write(f"Warning: Annotation file not found {annotation_path}")
        return 0

    try:
        annotations_df = pd.read_csv(annotation_path)
    except Exception as e:
        tqdm.write(f"Error: Cannot read annotation file {annotation_path}: {e}")
        return 0

    video_stream = cv2.VideoCapture(video_path)
    if not video_stream.isOpened():
        tqdm.write(f"Error: Cannot open video {video_path}")
        return 0

    frames_processed = 0
    current_frame = 0

    annotation_lookup = {}
    for _, row in annotations_df.iterrows():
        annotation_lookup[row['Frame']] = row

    input_group = h5_file.require_group(f"{match_name}/inputs/{sequence_name}")
    heatmap_group = h5_file.require_group(f"{match_name}/heatmaps/{sequence_name}")

    try:
        while True:
            frame_available, frame_data = video_stream.read()
            if not frame_available:
                break

            progress_bar.update(1)

            if current_frame in annotation_lookup:
                annotation_row = annotation_lookup[current_frame]

                processed_frame, scale_factor, x_offset, y_offset = resize_with_aspect_ratio(frame_data)

                if annotation_row['Visibility'] == 1:
                    original_x = annotation_row['X']
                    original_y = annotation_row['Y']

                    if pd.isna(original_x) or pd.isna(original_y):
                        heatmap = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
                    else:
                        transformed_x, transformed_y = transform_annotation_coords(
                            original_x, original_y, scale_factor, x_offset, y_offset
                        )

                        transformed_x = max(0, min(TARGET_WIDTH - 1, transformed_x))
                        transformed_y = max(0, min(TARGET_HEIGHT - 1, transformed_y))

                        heatmap = generate_heatmap(transformed_x, transformed_y, sigma=sigma_value)
                else:
                    heatmap = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)

                input_group.create_dataset(str(current_frame), data=processed_frame, compression='gzip')
                heatmap_group.create_dataset(str(current_frame), data=heatmap, compression='gzip')

                frames_processed += 1

                del processed_frame, heatmap

            current_frame += 1

            if current_frame % 100 == 0:
                gc.collect()

    finally:
        video_stream.release()

    return frames_processed


def process_match_directory(match_path, h5_file, sigma_value, progress_bar):
    match_name = os.path.basename(match_path)

    videos_dir = os.path.join(match_path, "video")
    annotations_dir = os.path.join(match_path, "csv")

    if not os.path.exists(videos_dir) or not os.path.exists(annotations_dir):
        tqdm.write(f"Warning: Missing video or csv directory in {match_name}")
        return

    mp4_files = [f for f in os.listdir(videos_dir)
                 if f.endswith('.mp4') and is_valid_path(f)]

    sequences_processed = 0
    total_frames_processed = 0

    for mp4_file in mp4_files:
        video_path = os.path.join(videos_dir, mp4_file)
        sequence_name = Path(mp4_file).stem

        annotation_filename = f"{sequence_name}_ball.csv"
        annotation_path = os.path.join(annotations_dir, annotation_filename)

        if os.path.exists(annotation_path):
            frames_count = process_video_sequence(
                video_path, annotation_path, h5_file, match_name, sequence_name, sigma_value, progress_bar
            )

            if frames_count > 0:
                sequences_processed += 1
                total_frames_processed += frames_count
        else:
            tqdm.write(f"Warning: Annotation file not found for {mp4_file}")

    tqdm.write(f"Completed {match_name}: {sequences_processed} sequences, {total_frames_processed} frames")


def preprocess_dataset(source_path, output_path, sigma_value=3.0, force_overwrite=False):
    structure_valid, validation_message = validate_dataset_structure(source_path)
    if not structure_valid:
        print(f"❌ {validation_message}")
        return False

    print(f"✅ {validation_message}")

    if os.path.exists(output_path):
        if force_overwrite:
            print(f"🗑️ Removing existing file: {output_path}")
            os.remove(output_path)
        else:
            user_input = input(f"⚠️ Output file exists: {output_path}\n   Delete and rebuild? (y/n): ")
            if user_input.lower() != 'y':
                print("❌ Operation cancelled")
                return False
            os.remove(output_path)

    entries = [item for item in os.listdir(source_path) if is_valid_path(item)]
    match_dirs = [
        item for item in entries
        if item.startswith("match") and os.path.isdir(os.path.join(source_path, item))
    ]

    valid_match_dirs = []
    for match_dir_name in match_dirs:
        match_dir_path = os.path.join(source_path, match_dir_name)
        has_csv = os.path.exists(os.path.join(match_dir_path, "csv"))
        has_video = os.path.exists(os.path.join(match_dir_path, "video"))

        if has_csv and has_video:
            valid_match_dirs.append(match_dir_name)
        else:
            print(f"⚠️ Skipping {match_dir_name}: missing csv or video directory")

    if not valid_match_dirs:
        print("❌ No valid match directories found")
        return False

    print(f"🚀 Processing {len(valid_match_dirs)} match directories...")

    print("📊 Estimating workload...")
    total_frame_count = estimate_total_frames(source_path)
    print(f"📊 Total frames to process: {total_frame_count}")

    with h5py.File(output_path, 'w') as h5_file:
        with tqdm(total=total_frame_count, desc="Processing frames", unit="frame") as progress_bar:
            for match_dir_name in valid_match_dirs:
                match_dir_path = os.path.join(source_path, match_dir_name)
                process_match_directory(match_dir_path, h5_file, sigma_value, progress_bar)

                gc.collect()

    print(f"\n🎉 Preprocessing completed!")
    print(f"   Source: {source_path}")
    print(f"   Output: {output_path}")
    print(f"   Heatmap sigma: {sigma_value}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Badminton Dataset Preprocessor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input Structure:
    dataset/
    ├── match1/
    │   ├── csv/
    │   │   └── rally1_ball.csv
    │   └── video/
    │       └── rally1.mp4
    └── match2/...

Output Structure:
    dataset_preprocessed.h5
    ├── match1/
    │   ├── inputs/rally1/[0,1,2,3...] (512×288×3 arrays)
    │   └── heatmaps/rally1/[0,1,2,3...] (288×512 arrays)
    └── match2/...

Output Data Format:
- Input frames: uint8 arrays (512×288×3) - RGB images, values 0-255
- Heatmap frames: uint8 arrays (288×512) - Grayscale heatmaps, values 0-255
- Compression: gzip for space efficiency
- Storage: HDF5 hierarchical structure

Annotation Format (rally1_ball.csv):
    Frame,Visibility,X,Y
    0,1,637.0,346.0
    1,1,639.0,346.0
    2,0,640.0,345.0
        """
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Source dataset directory path"
    )

    parser.add_argument(
        "--output",
        help="Output HDF5 file path (default: source + '_preprocessed.h5')"
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Gaussian heatmap standard deviation (default: 3.0)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing output file"
    )

    args = parser.parse_args()

    print("🏸 Badminton Dataset Preprocessor")
    print("=" * 50)

    try:
        print(f"📦 OpenCV {cv2.__version__}")
        print(f"📦 NumPy {np.__version__}")
        print(f"📦 Pandas {pd.__version__}")
        print(f"📦 HDF5 {h5py.__version__}")
    except ImportError as error:
        print(f"❌ Missing dependency: {error}")
        print("Install with: pip install opencv-python pandas numpy scipy tqdm h5py")
        sys.exit(1)

    if not args.output:
        dataset_name = os.path.basename(args.source.rstrip('/'))
        parent_directory = os.path.dirname(args.source) or '.'
        args.output = os.path.join(parent_directory, f"{dataset_name}_preprocessed.h5")

    print(f"📂 Source: {args.source}")
    print(f"📂 Output: {args.output}")
    print(f"🎯 Heatmap sigma: {args.sigma}")

    success = preprocess_dataset(args.source, args.output, args.sigma, args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
