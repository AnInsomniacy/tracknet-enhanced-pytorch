"""
HDF5 Dataset Training Player

Interactive image sequence player with heatmap overlay for badminton training datasets.
Reads from HDF5 preprocessed files, allows navigation between matches and sequences,
displays heatmaps transparently overlaid on original images with adjustable playback speed.

Usage Examples:
    python hdf5_visualizer.py --source dataset_train.h5
    python hdf5_visualizer.py --source dataset_train.h5 --fps 15 --alpha 0.4
    python hdf5_visualizer.py --source /path/to/data.h5 --fps 30 --alpha 0.3 --match 2

Parameters:
--source: HDF5 dataset file path (required)
--fps: Base playback frame rate (default: 30)
--alpha: Heatmap transparency 0.0-1.0 (default: 0.3)
--match: Starting match index (default: 0)

Dependencies:
    pip install opencv-python numpy h5py
"""

import argparse
import h5py
import numpy as np
import cv2
from pathlib import Path


class HDF5SequencePlayer:
    def __init__(self, h5_file_path: str, alpha: float = 0.3, base_fps: int = 30, start_match: int = 0):
        self.h5_file_path = h5_file_path
        self.alpha = alpha
        self.base_fps = base_fps
        self.current_match_index = start_match
        self.current_sequence_index = 0
        self.matches = []
        self.sequences = []
        self.show_original_only = False
        self.speed_multiplier = 1.0
        self.speed_options = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        self.speed_index = 2

    def scan_h5_structure(self):
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            self.matches = sorted([name for name in h5_file.keys() if name.startswith('match')])

        if not self.matches:
            print(f"❌ No match folders found in {self.h5_file_path}")
            return False

        if self.current_match_index >= len(self.matches):
            self.current_match_index = 0

        return True

    def scan_current_match_sequences(self):
        if not self.matches:
            return False

        match_name = self.matches[self.current_match_index]

        with h5py.File(self.h5_file_path, 'r') as h5_file:
            match_group = h5_file[match_name]
            if 'inputs' not in match_group or 'heatmaps' not in match_group:
                return False

            inputs_group = match_group['inputs']
            heatmaps_group = match_group['heatmaps']

            input_sequences = set(inputs_group.keys())
            heatmap_sequences = set(heatmaps_group.keys())
            common_sequences = input_sequences.intersection(heatmap_sequences)

            self.sequences = sorted(list(common_sequences))

        if self.current_sequence_index >= len(self.sequences):
            self.current_sequence_index = 0

        return len(self.sequences) > 0

    def load_sequence_data(self, match_name: str, sequence_name: str):
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            input_group = h5_file[f"{match_name}/inputs/{sequence_name}"]
            heatmap_group = h5_file[f"{match_name}/heatmaps/{sequence_name}"]

            input_keys = sorted([int(k) for k in input_group.keys()])
            heatmap_keys = sorted([int(k) for k in heatmap_group.keys()])

            min_length = min(len(input_keys), len(heatmap_keys))

            input_images = []
            heatmap_images = []

            for i in range(min_length):
                input_key = str(input_keys[i])
                heatmap_key = str(heatmap_keys[i])

                input_data = input_group[input_key][:]
                heatmap_data = heatmap_group[heatmap_key][:]

                input_images.append(input_data)
                heatmap_images.append(heatmap_data)

        return input_images, heatmap_images

    def apply_colormap_to_heatmap(self, heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        if len(heatmap.shape) == 3:
            heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        else:
            heatmap_gray = heatmap

        heatmap_norm = cv2.normalize(heatmap_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(heatmap_norm, colormap)
        return heatmap_colored

    def overlay_images(self, input_img: np.ndarray, heatmap_img: np.ndarray) -> np.ndarray:
        if self.show_original_only:
            return input_img

        heatmap_colored = self.apply_colormap_to_heatmap(heatmap_img)
        overlayed = cv2.addWeighted(input_img, 1 - self.alpha, heatmap_colored, self.alpha, 0)
        return overlayed

    def get_current_fps(self):
        return int(self.base_fps / self.speed_multiplier)

    def change_speed(self, direction: int):
        if direction > 0 and self.speed_index < len(self.speed_options) - 1:
            self.speed_index += 1
        elif direction < 0 and self.speed_index > 0:
            self.speed_index -= 1

        self.speed_multiplier = self.speed_options[self.speed_index]
        print(f"🏃 Speed: {self.speed_multiplier}x")

    def change_match(self, direction: int):
        if direction > 0 and self.current_match_index < len(self.matches) - 1:
            self.current_match_index += 1
            self.current_sequence_index = 0
            return True
        elif direction < 0 and self.current_match_index > 0:
            self.current_match_index -= 1
            self.current_sequence_index = 0
            return True
        return False

    def play_sequence(self, match_name: str, sequence_name: str):
        print(f"🏸 Loading {match_name}/{sequence_name}")

        input_images, heatmap_images = self.load_sequence_data(match_name, sequence_name)

        if not input_images or not heatmap_images:
            print(f"❌ Cannot load sequence {match_name}/{sequence_name}")
            return False

        frame_count = len(input_images)
        print(f"🎬 Playing {frame_count} frames")

        window_name = f"HDF5 Player - {match_name}/{sequence_name} ({self.current_match_index + 1}/{len(self.matches)}) ({self.current_sequence_index + 1}/{len(self.sequences)})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_index = 0
        paused = False

        while frame_index < frame_count:
            if not paused:
                combined_frame = self.overlay_images(input_images[frame_index], heatmap_images[frame_index])

                alpha_text = "Original Only" if self.show_original_only else f"Alpha: {self.alpha:.2f}"
                info_text = f"Frame: {frame_index + 1}/{frame_count} | {alpha_text} | Speed: {self.speed_multiplier}x | Match: {match_name} | Seq: {sequence_name}"

                text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(combined_frame, (5, 5), (text_size[0] + 15, 30), (0, 0, 0), -1)
                cv2.putText(combined_frame, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow(window_name, combined_frame)
                frame_index += 1

            key = cv2.waitKey(max(1, int(1000 / self.get_current_fps()))) & 0xFF

            if key == ord('q') or key == 27:
                cv2.destroyWindow(window_name)
                return False
            elif key == ord(' '):
                paused = not paused
                print("⏸️  Paused" if paused else "▶️  Resumed")
            elif key == ord('n') or key == ord('.'):
                cv2.destroyWindow(window_name)
                return True
            elif key == ord('p') or key == ord(','):
                cv2.destroyWindow(window_name)
                return "previous"
            elif key == ord('m'):
                if self.change_match(1):
                    cv2.destroyWindow(window_name)
                    return "match_changed"
            elif key == ord('k'):
                if self.change_match(-1):
                    cv2.destroyWindow(window_name)
                    return "match_changed"
            elif key == ord('r'):
                frame_index = 0
                paused = False
            elif key == ord('o'):
                self.show_original_only = not self.show_original_only
                print(f"🖼️  Display mode: {'Original only' if self.show_original_only else 'Heatmap overlay'}")
            elif key == ord('s'):
                save_path = f"frame_{match_name}_{sequence_name}_{frame_index}.jpg"
                cv2.imwrite(save_path, combined_frame)
                print(f"💾 Saved frame to: {save_path}")
            elif key == ord('f'):
                frame_index = min(frame_index + 10, frame_count - 1)
            elif key == ord('b'):
                frame_index = max(frame_index - 10, 0)
            elif key == ord('+') or key == ord('='):
                self.alpha = min(1.0, self.alpha + 0.05)
                print(f"🔆 Alpha: {self.alpha:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.alpha = max(0.0, self.alpha - 0.05)
                print(f"🔅 Alpha: {self.alpha:.2f}")
            elif key == ord('1'):
                self.change_speed(1)
            elif key == ord('2'):
                self.change_speed(-1)

        print(f"✅ Sequence {sequence_name} completed")
        cv2.destroyWindow(window_name)
        return True

    def run(self):
        if not Path(self.h5_file_path).exists():
            print(f"❌ File does not exist: {self.h5_file_path}")
            return

        print(f"🔍 Scanning HDF5 file: {self.h5_file_path}")

        if not self.scan_h5_structure():
            return

        print(f"🏸 Found {len(self.matches)} matches:")
        for i, match in enumerate(self.matches):
            print(f"  {i + 1}. {match}")

        print(f"\n🎯 Initial alpha: {self.alpha:.2f}")
        print(f"🎯 Initial speed: {self.speed_multiplier}x")
        print("\n🎮 Controls:")
        print("  Space: Pause/Resume")
        print("  n or .: Next sequence")
        print("  p or ,: Previous sequence")
        print("  m: Next match")
        print("  k: Previous match")
        print("  r: Restart current sequence")
        print("  o: Toggle display mode (overlay/original only)")
        print("  + or =: Increase heatmap transparency")
        print("  - or _: Decrease heatmap transparency")
        print("  1: Increase playback speed")
        print("  2: Decrease playback speed")
        print("  s: Save current frame")
        print("  f: Fast forward 10 frames")
        print("  b: Fast backward 10 frames")
        print("  q or ESC: Exit")
        print()

        while self.current_match_index < len(self.matches):
            if not self.scan_current_match_sequences():
                print(f"❌ No valid sequences in {self.matches[self.current_match_index]}")
                if not self.change_match(1):
                    break
                continue

            while self.current_sequence_index < len(self.sequences):
                match_name = self.matches[self.current_match_index]
                sequence_name = self.sequences[self.current_sequence_index]
                result = self.play_sequence(match_name, sequence_name)

                if result is False:
                    cv2.destroyAllWindows()
                    print("🏸 Player exited")
                    return
                elif result == "previous":
                    self.current_sequence_index = max(0, self.current_sequence_index - 1)
                elif result == "match_changed":
                    break
                else:
                    self.current_sequence_index += 1

            if self.current_sequence_index >= len(self.sequences):
                if not self.change_match(1):
                    break

        cv2.destroyAllWindows()
        print("🏸 Player completed all sequences")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive HDF5 Dataset Training Player with Heatmap Overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Required HDF5 Structure:
    dataset.h5
    ├── match1/
    │   ├── inputs/
    │   │   ├── sequence1/[0,1,2...] (512×288×3 arrays)
    │   │   └── sequence2/[0,1,2...] (512×288×3 arrays)
    │   └── heatmaps/
    │       ├── sequence1/[0,1,2...] (288×512 arrays)
    │       └── sequence2/[0,1,2...] (288×512 arrays)
    └── match2/...
        """
    )

    parser.add_argument("--source", required=True, help="HDF5 dataset file path")
    parser.add_argument("--fps", type=int, default=30, help="Base playback frame rate (default: 30)")
    parser.add_argument("--alpha", type=float, default=0.3, help="Heatmap transparency (0.0-1.0, default: 0.3)")
    parser.add_argument("--match", type=int, default=0, help="Starting match index (default: 0)")

    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        print("❌ Alpha value must be between 0.0 and 1.0")
        return

    print("🏸 HDF5 Dataset Training Player")
    print("=" * 50)

    player = HDF5SequencePlayer(args.source, args.alpha, args.fps, args.match)
    player.run()


if __name__ == "__main__":
    main()
