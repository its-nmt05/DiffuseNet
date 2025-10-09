import os
from torch.utils.data import Dataset
from PIL import Image
from vae.utils import extract_frames, assign_frames, load_cached_latents
import subprocess
import csv


class VideoDataset(Dataset):
    def __init__(
        self,
        video_path,
        frames_dir,
        hf_cache_dir,
        vae_latent_save_dir,
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
        video_path:          path to your .mp4/.avi/etc
        frames_dir:          where to dump/read extracted frames
        hf_cache_dir:        Cache dir to store HF models locally
        use_latents:         use precomputed latents instead
        vae_latent_save_dir: dir to save precomputed latents
        split:               "train" or "test"
        transform:           any torchvision transforms
        interval:            seconds between frames
        resolution:          int or None (square resize)
        test_ratio:          fraction of frames to hold out
        seed:                random seed for split
        """
        self.video_path = video_path
        self.frames_dir = frames_dir
        self.split = split
        self.transform = transform
        self.latent_maps = None
        self.use_latents = use_latents

        os.makedirs(frames_dir, exist_ok=True)

        # check if frames_dir is empty
        has_images = any(
            f for f in os.listdir(frames_dir)
            if os.path.isfile(os.path.join(frames_dir, f))
        )
        if not has_images:
            extract_frames(
                input_video=video_path,
                output_dir=frames_dir,
                interval=interval,
                resolution=resolution,
                start=start,
                end=end
            )

        # check for index files
        parent = os.path.abspath(os.path.join(frames_dir, os.pardir))
        train_idx = os.path.join(parent, 'train_indices.txt')
        test_idx = os.path.join(parent, 'test_indices.txt')

        if not (os.path.isfile(train_idx) and os.path.isfile(test_idx)):
            assign_frames(
                frames_dir,
                test_ratio=test_ratio,
                seed=seed,
                train_out=train_idx,
                test_out=test_idx,
            )

        index_file = train_idx if split == 'train' else test_idx
        with open(index_file, 'r') as f:
            indices = [line.strip() for line in f if line.strip()]

        # check for captions file
        captions = {}
        captions_file = os.path.join(parent, 'captions.csv')
        
        # captions don't exit, generate them
        if not os.path.isfile(captions_file):
            script_path = os.path.join(os.path.dirname(__file__), "extract_captions.py")
            subprocess.run([
                "python", script_path,
                "--hf_cache_dir", hf_cache_dir,
                "--frames_dir", frames_dir,
                "--batch_size", "256",
            ])  

        with open(captions_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                captions[row['image']] = row['caption']

        # create combined (img, caption) list
        self.samples = [(frame, captions[frame]) for frame in indices]

        # load cached latents
        if self.use_latents:
            latents = load_cached_latents(vae_latent_save_dir)
            frames = set(frame for frame, _ in self.samples)
            self.latent_maps = {k: v for k, v in latents.items() if k in frames}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]

        # serve latents directly
        if self.use_latents:
            return self.latent_maps[img_name], caption
        else:
            img_path = os.path.join(self.frames_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, caption