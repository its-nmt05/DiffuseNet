import os
import glob
import random
import numpy as np
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


# extract frames from video to npz file
def extract_frames(input_video, output_path, interval, resolution=None, start=0, end=None):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {input_video}")

    if end and end < start:
        raise ValueError("End time must be greater than start time")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = int(round(interval * fps))
    start_frame = int(round(start * fps))
    end_frame = int(round(end * fps)) if end is not None else total_frames
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)   # set to start frame
    frame_idx = 0
    
    while frame_idx < end_frame:
        ret = cap.grab()
        if not ret:
            frame_idx += 1
            continue

        # read frame at required intervals
        if (frame_idx - start_frame) % interval_frames == 0:
            ret, frame = cap.retrieve()
            if not ret:
                frame_idx += 1
                continue
                
            if resolution:
                frame = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        frame_idx += 1

    cap.release()
    np.savez_compressed(output_path, frames=np.stack(frames))
    print(f"Saved {len(frames)} frames to {output_path}")


# assign frames to train and test splits
def assign_frames(npz_file, test_ratio=0.2, seed=None, train_out="train_indices.txt", test_out="test_indices.txt"):
    data = np.load(npz_file)
    frames = data['frames']
    total = len(frames)

    if total == 0:
        print("No frames found in", npz_file)
        return

    # optionally seed 
    if seed is not None:
        random.seed(seed)
        
    indices = list(range(total))
    random.shuffle(indices)
    test_count = int(total * test_ratio)
    test_indices = sorted(indices[:test_count])
    train_indices = sorted(indices[test_count:])

    with open(train_out, 'w') as f:
        f.write(str(train_indices))

    with open(test_out, 'w') as f:
        f.write(str(test_indices))

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


def cache_latents(vae, dataset_dir, vae_scale, batch_size=128, device='cuda'):
    frames_npz_file = os.path.join(dataset_dir, 'frames.npz')
    latent_save_path = os.path.join(dataset_dir, 'vae_latents.npz')

    # load all the frames from npz file
    frames = np.load(frames_npz_file, mmap_mode='r')['frames']
    all_latents = []

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    vae.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(frames), batch_size), desc='Caching latents: '):
            batched_frames = frames[i:i+batch_size]
            batched_frames = [transform(Image.fromarray(frame)) for frame in batched_frames]
            batched_frames = torch.stack(batched_frames).to(device)

            latents, _, _ = vae.encode(batched_frames)
            latents = latents / vae_scale
            all_latents.append(latents.cpu())

        all_latents = np.concatenate(all_latents, axis=0)
        np.savez_compressed(latent_save_path, latents=all_latents)
        print(f"Cached latents for {len(all_latents)} images to {latent_save_path}")


def load_cached_latents(latent_save_path):
    data = np.load(latent_save_path, mmap_mode='r')
    latents = data['latents']
    # construct latents dict
    latents_maps = {idx: latent for idx, latent in enumerate(latents)}
    return latents_maps
