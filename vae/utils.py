import os
import glob
import random
import pickle

from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision
import cv2


def get_norm(norm_type='bn', norm_channels=32, num_groups=None):
    if norm_type == 'bn':
        return nn.BatchNorm2d(norm_channels) 
    elif norm_type == 'gn':
        if num_groups is None: 
            raise ValueError("num_groups must be specified for GroupNorm")
        return nn.GroupNorm(num_groups=num_groups, num_channels=norm_channels)
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")
    

def get_activation(activation_type='relu', **kwargs):
    activation_type = activation_type.lower()
    if activation_type == 'silu':
        return nn.SiLU(**kwargs)
    elif activation_type == 'gelu':
        return nn.GELU(**kwargs)    
    elif activation_type == 'elu':
        return nn.ELU(**kwargs)
    elif activation_type == 'leakyrelu':
        return nn.LeakyReLU(kwargs.get('negative_slope', 0.2))
    else: 
        raise ValueError(f"Activation type '{activation_type}' not in list of supported activations: ['silu', 'gelu', 'elu', 'leakyrelu']")


# extract frames from video with specified parameters
def extract_frames(input_video, output_dir, interval, resolution=None, start=0, end=None):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    if end and end < start:
        raise ValueError("End time must be greater than start time")

    interval_frames = int(round(interval * fps))
    start_frame = int(round(start * fps))
    end_frame = int(round(end * fps)) if end is not None else None
    
    frame_idx = 0
    saved_idx = 0
    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # skip until start_frame
        if frame_idx < start_frame:
            frame_idx += 1
            continue

        # stop after end_frame
        if end_frame is not None and frame_idx > end_frame:
            break

        # save frames at the given interval
        if (frame_idx - start_frame) % interval_frames == 0:
            if resolution:
                frame = cv2.resize(frame, (resolution, resolution),
                                   interpolation=cv2.INTER_AREA)
            filename = f"frame_{saved_idx:05d}.jpg"
            path = os.path.join(output_dir, filename)
            cv2.imwrite(path, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved_idx} frames to {output_dir}")


# assign frames to train and test splits
def assign_frames(frames_dir, test_ratio=0.2, seed=None, train_out="train_indices.txt", test_out="test_indices.txt"):
    all_frames = sorted(
        f for f in os.listdir(frames_dir)
        if os.path.isfile(os.path.join(frames_dir, f))
    )
    total = len(all_frames)
    if total == 0:
        print("No frames found in", frames_dir)
        return

    # optionally seed 
    if seed is not None:
        random.seed(seed)
        
    indices = list(range(total))
    random.shuffle(indices)
    test_count = int(total * test_ratio)
    test_indices = sorted(indices[:test_count])
    train_indices = sorted(indices[test_count:])

    with open(train_out, "w") as f:
        for idx in train_indices:
            f.write(f"{all_frames[idx]}\n")

    with open(test_out, "w") as f:
        for idx in test_indices:
            f.write(f"{all_frames[idx]}\n")

    print(f"Total frames: {total}")
    print(f"Train: {len(train_indices)} -> {train_out}")
    print(f"Test:  {len(test_indices)} -> {test_out}")


def get_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_kl_loss(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()) # (B, C, H, W)
    kl = kl.flatten(1).sum(dim=1) # [B, C, H, W] -> [B, C*H*W] -> [B]
    return kl.mean()
    

def sample_images(model, dataloader, device, num_samples=8):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(dataloader))
        images = images.to(device)
        recon, _, _ = model(images)
        recon = recon[:num_samples]
        images = images[:num_samples]

        # convert to grid and save
        grid = make_grid(torch.cat([images, recon]), nrow=num_samples)
        img = torchvision.transforms.ToPILImage()(grid)
        return img


def cache_latents(vae, frames_dir, latent_save_dir, vae_scale, batch_size=128, device='cuda'):
    os.makedirs(latent_save_dir, exist_ok=True)
    latent_save_path = os.path.join(latent_save_dir, 'vae_latents.pkl')
    latent_maps = {}

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # load all img paths
    all_frames = sorted(glob.glob(os.path.join(frames_dir, '*')))

    vae.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(all_frames), batch_size), desc='Caching latents: '):
            batched_paths = all_frames[i:i+batch_size]
            batched_frames = [transform(Image.open(p).convert('RGB')) for p in batched_paths]
            batched_frames = torch.stack(batched_frames).to(device)

            latents, _, _ = vae.encode(batched_frames)
            latents = latents / vae_scale

            for idx, path in enumerate(batched_paths):
                fname = os.path.basename(path) 
                latent_maps[fname] = [latents[idx].cpu()]

        # save all the latents
        with open(latent_save_path, 'wb') as f:
            pickle.dump(latent_maps, f)

        print(f"Cached latents for {len(all_frames)} images to {latent_save_path}")


def load_cached_latents(latent_save_dir):
    latent_maps = {}
    for fname in glob.glob(os.path.join(latent_save_dir, '*.pkl')):
        s = pickle.load(open(fname, 'rb'))
        for k, v in s.items():
            latent_maps[k] = v[0]   # unwrap from the list
        return latent_maps  
