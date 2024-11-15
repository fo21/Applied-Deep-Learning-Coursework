#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset import MIT
from dataset import crop_to_region

torch.backends.cudnn.benchmark = True

train_dataset_path = './train_data.pth.tar'
val_dataset_path = './val_data.pth.tar'
test_dataset_path = './test_data.pth.tar'

train_dataset = MIT(dataset_path=train_dataset_path)
test_dataset = MIT(dataset_path=test_dataset_path)
val_dataset = MIT(dataset_path=val_dataset_path)

print("hi")

val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=10,
        num_workers=2,
        pin_memory=True,
    )


#print(train_loader)



#print(test_loader.dataset[1]['file'])


test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=128,
        num_workers=1,
        pin_memory=True,
    )


#processing the training dataset

# Define the Gaussian blur transform
transform = transforms.Compose([
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # kernel_size is the size of the Gaussian kernel
    transforms.ToTensor()  # Convert image to tensor if it's not already
])

print(test_loader.dataset)

for i, (batch, labels) in enumerate(test_loader):
    print(i)
    #print(f"spatial coords {test_loader.dataset[i]["spatial_coords"][0]}")
    for img in batch:
        print(f"img: {img.size()}")

                
        
        
"""
for i in range(0, len(test_loader)):
    gt_path = f"ALLFIXATIONMAPS/{test_loader.dataset[i]['file'][:-5]}_fixMap.jpg"
    image = Image.open(gt_path) # Load an image
    original_width, original_height = image.size  # Store the original dimensions


    # Calculate the threshold value (20% of the max intensity)
    #threshold_value = 0.2 * blurred_image.max()

    # Apply thresholding
    #thresholded_image = (blurred_image > threshold_value).float()  # Set pixels to 1 if > threshold, else 0

    spatial_coords = test_loader.dataset[i]["spatial_coords"]
    for index in range(0,len(spatial_coords)):
        center_x, center_y = test_loader.dataset[index][0],test_loader.dataset[index][1]   # Example center point (x, y)
        crop_size = 50  # Size of the crop region (50x50)

        # Calculate crop box (left, upper, right, lower)
        left = max(center_x - crop_size // 2, 0)
        upper = max(center_y - crop_size // 2, 0)
        right = min(center_x + crop_size // 2, original_width)
        lower = min(center_y + crop_size // 2, original_height)

        # Crop the image
        cropped_image = image.crop((left, upper, right, lower))

        # Resize the cropped image back to the original size
        resize_transform = transforms.Resize((original_height, original_width))
        resized_image = resize_transform(cropped_image)
"""






        


