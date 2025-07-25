import torch
import torch.utils.data
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class DataClass:

    def __init__(self, batch_size, img_size, num_img):
        self.BATCH_SIZE = batch_size
        self.IMG_SIZE = img_size
        self.NUM_IMG = num_img

    def load_transformed_dataset(self, dataset_path='./dataset'):
        data_transform = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),  # convert the images to tensor. Scales to [0, 1]
            transforms.Lambda(lambda t: (t * 2) - 1)  # Scaled b/w [-1, 1]
        ])
        # dataset = torchvision.datasets.ImageFolder(
        #     root=dataset_path, transform=data_transform)
        dataset = torchvision.datasets.MNIST(root=dataset_path, download=True, transform=data_transform)
        indices = torch.randperm(len(dataset))[:self.NUM_IMG]
        subset = Subset(dataset, indices)  # take only a subset of the dataset
        return DataLoader(subset, batch_size=self.BATCH_SIZE, shuffle=True)

    def show_tensor_image(self, img_tensor, save=False, output_dir=None, nrow=5):
        grid = make_grid(img_tensor, nrow=nrow, normalize=True, value_range=(-1, 1))
        grid = grid.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
        grid = (grid * 255).astype(np.uint8)  # Scale to [0, 255]
        image = Image.fromarray(grid)

        if save and output_dir is not None:
            image.save(output_dir, format='png')

        plt.imshow(image)
        plt.axis("off")
        plt.tight_layout(pad=0.5)