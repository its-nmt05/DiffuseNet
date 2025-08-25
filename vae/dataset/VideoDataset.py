import os
from torch.utils.data import Dataset
from PIL import Image
from vae.utils import extract_frames, assign_frames


class VideoDataset(Dataset):
    def __init__(
        self,
        video_path,
        frames_dir,
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
        video_path:   path to your .mp4/.avi/etc
        frames_dir:   where to dump/read extracted frames
        split:        "train" or "test"
        transform:    any torchvision transforms
        interval:     seconds between frames
        resolution:   int or None (square resize)
        test_ratio:   fraction of frames to hold out
        seed:         random seed for split
        """
        self.video_path = video_path
        self.frames_dir = frames_dir
        self.split = split
        self.transform = transform

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
            self.samples = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        img_path = os.path.join(self.frames_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, idx

