"""
HDF5 Frame Heatmap Dataset for PyTorch

Reads frame images and corresponding heatmaps from HDF5 file for TrackNet training.

Usage Examples:
    dataset = HDF5FrameHeatmapDataset('dataset_train.h5')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    dataloader = DataLoader(HDF5FrameHeatmapDataset('/path/to/data.h5'), batch_size=8, shuffle=True, num_workers=2)

Parameters:
h5_file_path: Path to HDF5 dataset file (required)

Dataset Structure:
dataset_preprocessed.h5
├── match1/
│   ├── inputs/frame1/[0,1,2,3...] (512×288×3 arrays)
│   └── heatmaps/frame1/[0,1,2,3...] (288×512 arrays)
└── match2/...

Input Data Format:
- Input frames: uint8 arrays (512×288×3) - RGB images, values 0-255
- Heatmap frames: uint8 arrays (288×512) - Grayscale heatmaps, values 0-255
- Storage: HDF5 with gzip compression

Output Format:
- inputs: (9, 288, 512) - 3 RGB images concatenated, normalized to [0,1]
- heatmaps: (3, 288, 512) - 3 grayscale heatmaps concatenated, normalized to [0,1]
"""

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class HDF5FrameHeatmapDataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file_path = h5_file_path
        self.data_items = self._scan_dataset()

    def _scan_dataset(self):
        items = []

        with h5py.File(self.h5_file_path, 'r') as h5_file:
            match_names = [name for name in h5_file.keys() if name.startswith('match')]
            print(f"Scanning {len(match_names)} match folders...")

            for match_name in sorted(match_names):
                items.extend(self._process_match(h5_file, match_name))

        print(f"Found {len(items)} valid samples")
        return items

    def _process_match(self, h5_file, match_name):
        match_group = h5_file[match_name]

        if 'inputs' not in match_group or 'heatmaps' not in match_group:
            return []

        items = []
        inputs_group = match_group['inputs']
        heatmaps_group = match_group['heatmaps']

        common_frames = self._get_common_frames(inputs_group, heatmaps_group)

        for frame_name in sorted(common_frames):
            items.extend(
                self._process_frame(match_name, frame_name, inputs_group[frame_name], heatmaps_group[frame_name]))

        return items

    def _get_common_frames(self, inputs_group, heatmaps_group):
        input_frames = set(inputs_group.keys())
        heatmap_frames = set(heatmaps_group.keys())
        return input_frames.intersection(heatmap_frames)

    def _process_frame(self, match_name, frame_name, input_frame_group, heatmap_frame_group):
        input_keys = sorted([int(k) for k in input_frame_group.keys()])
        heatmap_keys = sorted([int(k) for k in heatmap_frame_group.keys()])

        if len(input_keys) != len(heatmap_keys) or len(input_keys) < 3:
            return []

        return [
            {
                'match': match_name,
                'frame': frame_name,
                'input_keys': input_keys[i:i + 3],
                'heatmap_keys': heatmap_keys[i:i + 3],
                'idx': i
            }
            for i in range(len(input_keys) - 2)
        ]

    def _load_image(self, match_name, frame_name, key, is_heatmap=False):
        try:
            with h5py.File(self.h5_file_path, 'r') as h5_file:
                if is_heatmap:
                    data = h5_file[f"{match_name}/heatmaps/{frame_name}/{key}"][:]
                    tensor = torch.from_numpy(data.astype(np.float32) / 255.0).unsqueeze(0)
                else:
                    data = h5_file[f"{match_name}/inputs/{frame_name}/{key}"][:]
                    tensor = torch.from_numpy(data.astype(np.float32) / 255.0).permute(2, 0, 1)
                return tensor
        except Exception as e:
            print(f"Failed to load data: {match_name}/{frame_name}/{key}")
            channels = 1 if is_heatmap else 3
            return torch.zeros(channels, 288, 512)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]

        inputs = torch.cat([
            self._load_image(item['match'], item['frame'], str(key), False)
            for key in item['input_keys']
        ], dim=0)

        heatmaps = torch.cat([
            self._load_image(item['match'], item['frame'], str(key), True)
            for key in item['heatmap_keys']
        ], dim=0)

        return inputs, heatmaps

    def get_info(self, idx):
        return self.data_items[idx]


if __name__ == "__main__":
    h5_file_path = "../dataset/Test_preprocessed.h5"

    dataset = HDF5FrameHeatmapDataset(h5_file_path)
    print(f"Dataset size: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=2
    )

    print("\nTesting data loading:")
    for batch_idx, (inputs, heatmaps) in enumerate(dataloader):
        print(f"Batch {batch_idx}: inputs{inputs.shape}, heatmaps{heatmaps.shape}")
        print(f"  Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"  Heatmap range: [{heatmaps.min():.3f}, {heatmaps.max():.3f}]")

        if batch_idx == 0:
            info = dataset.get_info(0)
            print(f"  Sample info: {info['match']}/{info['frame']}, start index {info['idx']}")
        break
