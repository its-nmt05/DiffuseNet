import os
from torch.utils.data import Dataset
from PIL import Image
from vae.utils import extract_frames, assign_frames, load_cached_latents
import subprocess
import csv
import ast
import numpy as np


class VideoDataset(Dataset):
    def __init__(
        self,
        video_path,
        dataset_dir,
        hf_cache_dir,
        use_latents=True,
        split='train',
        transform=None,
        interval=1.0,
        resolution=None,
        test_ratio=0.2,
        seed=None,
        start=0,
        end=None
    ):
        """
        video_path:     path to your .mp4/.avi/etc
        dataset_dir:    parent dir to store the dataset
        hf_cache_dir:   Cache dir to store HF models locally
        use_latents:    use precomputed latents instead
        split:          "train" or "test"
        transform:      any torchvision transforms
        interval:       seconds between frames
        resolution:     int or None (square resize)
        test_ratio:     fraction of frames to hold out
        seed:           random seed for split
        """
        self.video_path = video_path
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform
        self.latent_maps = None
        self.use_latents = use_latents

        os.makedirs(self.dataset_dir, exist_ok=True)

        frames_npz_file = os.path.join(self.dataset_dir, "frames.npz")
        latent_save_path = os.path.join(self.dataset_dir, "vae_latents.npz")

        # lazy load frames from npz
        self.frames_data = np.load(frames_npz_file, mmap_mode='r')['frames']

        # check if dataset_dir contains frames_npz
        if not os.path.exists(frames_npz_file):
            extract_frames(
                input_video=video_path,
                interval=interval,
                output_path=frames_npz_file,
                resolution=resolution,
                start=start,
                end=end
            )

        # check for index files
        train_idx = os.path.join(self.dataset_dir, 'train_indices.txt')
        test_idx = os.path.join(self.dataset_dir, 'test_indices.txt')

        if not (os.path.isfile(train_idx) and os.path.isfile(test_idx)):
            assign_frames(
                frames_npz_file,
                test_ratio=test_ratio,
                seed=seed,
                train_out=train_idx,
                test_out=test_idx,
            )

        index_file = train_idx if split == 'train' else test_idx
        with open(index_file, 'r') as f:
            content = f.read().strip()
            indices = ast.literal_eval(content)

        # check for captions file
        captions = {}
        captions_file = os.path.join(self.dataset_dir, 'captions.csv')
        
        # captions don't exit, generate them
        if not os.path.isfile(captions_file):
            script_path = os.path.join(os.path.dirname(__file__), "extract_captions.py")
            subprocess.run([
                "python", script_path,
                "--hf_cache_dir", hf_cache_dir,
                "--npz_file", frames_npz_file,
                "--batch_size", "128",
            ])  

        with open(captions_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                captions[int(row['idx'])] = row['caption']

        # create combined (img, caption) list
        self.samples = [(idx, captions[idx]) for idx in indices]

        # load cached latents
        if self.use_latents:
            self.latent_maps = load_cached_latents(latent_save_path)
        else:
            self.latent_maps = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_idx, caption = self.samples[idx]

        # serve latents directly
        if self.use_latents:
            return self.latent_maps[frame_idx], caption
        else:
            image = Image.fromarray(self.frames_data[frame_idx])
            if self.transform:
                image = self.transform(image)
            return image, caption