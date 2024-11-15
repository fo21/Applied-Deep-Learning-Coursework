#!/usr/bin/env python3
import os
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

from torch.utils.data import Dataset
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

import argparse
from pathlib import Path

from dataset import MIT

import numpy as np
from skimage.transform import resize

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a MRCNN on MIT dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)y
#default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
train_dataset_path = './train_data.pth.tar'
val_dataset_path = './val_data.pth.tar'
test_dataset_path = './test_data.pth.tar'
#parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-2, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=128,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)


class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    transform = transforms.ToTensor()
    #args.dataset_root.mkdir(parents=True, exist_ok=True)
    train_dataset = MIT(dataset_path=train_dataset_path)
    test_dataset = MIT(dataset_path=test_dataset_path)
    val_dataset = MIT(dataset_path=val_dataset_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )
    

    model = MrCNN(input_channels=3, output_classes=1)

    ## TASK 8: Redefine the criterion to be softmax cross entropy
    criterion = nn.CrossEntropyLoss()

    ## TASK 11: Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list  # This should be your list of dictionaries

    def __len__(self):
        # Returns the number of samples in the dataset
        return len(self.data_list)

    def __getitem__(self, idx):
        # Fetch the dictionary at index `idx`
        sample = self.data_list[idx]
        
        # Ensure each field in the dictionary is a tensor for easier batching
        sample = {key: torch.tensor(value) if not isinstance(value, torch.Tensor) else value 
                  for key, value in sample.items()}
        
        return sample

class MrCNN(nn.Module):
    def __init__(self, input_channels=3, output_classes=1):
        super(MrCNN, self).__init__()

        # Define one set of convolutional layers to be shared across all three streams
        self.shared_conv1 = nn.Conv2d(input_channels, 96, kernel_size=7, stride=1)
        self.shared_conv2 = nn.Conv2d(96, 160, kernel_size=3, stride=1)
        self.shared_conv3 = nn.Conv2d(160, 288, kernel_size=3, stride=1)
        
        # Define the pooling and FC layers for each stream
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(288 * 3 * 3, 512)  # Output size from convolutional layers is 3x3 after pooling

        # Fusion layer after concatenating the three streams
        self.fusion_fc = nn.Linear(512 * 3, 512)

        # Output layer for binary classification
        self.output_layer = nn.Linear(512, output_classes)

    def forward_stream(self, x):
        # Pass through shared convolutional layers with pooling
        x = F.relu(self.shared_conv1(x))
        x = self.pool(x)
        x = F.relu(self.shared_conv2(x))
        x = self.pool(x)
        x = F.relu(self.shared_conv3(x))
        x = self.pool(x)
        
        # Flatten and pass through FC layer for each stream
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        # Split input into three crops for each stream
        crop1, crop2, crop3 = x[:, 0], x[:, 1], x[:, 2]  # Assuming x has shape [batch_size, 3, 3, 42, 42]
        
        # Pass each crop through the shared convolutional layers independently
        stream1_out = self.forward_stream(crop1)
        stream2_out = self.forward_stream(crop2)
        stream3_out = self.forward_stream(crop3)

        # Concatenate outputs from the three streams
        combined = torch.cat((stream1_out, stream2_out, stream3_out), dim=1)

        # Pass through fusion and output layers
        fusion_out = F.relu(self.fusion_fc(combined))
        output = torch.sigmoid(self.output_layer(fusion_out)).squeeze(1)  # Use sigmoid for binary classification

        return output

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for batch,label in self.train_loader:
                batch = batch.to(self.device)       
                label = label.to(self.device).float() 
                data_load_end_time = time.time()


                ## TASK 1: Compute the forward pass of the model, print the output shape
                ##         and quit the program
                #output =
                #output = self.model.forward(batch)
                #print(output.shape)
                #import sys; sys.exit(1)
                ## TASK 7: Rename `output` to `logits`, remove the output shape printing
                ##         and get rid of the `import sys; sys.exit(1)`
                logits = self.model.forward(batch)
                #print(f"logits: {logits}")
                ## TASK 9: Compute the loss using self.criterion and
                ##         store it in a variable called `loss`
                loss = self.criterion(logits, label)
                #print(f"Epoch [{epoch + 1}/{epochs}], Batch Loss: {loss.item()}")
                ## TASK 10: Compute the backward pass
                loss.backward()
                ## TASK 12: Step the optimizer and then zero out the gradient buffers.
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                with torch.no_grad():
                    preds = torch.sigmoid(logits).round()
                    #print(f"label: {label}, pred: {preds}")
                    accuracy = compute_accuracy(label, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()
        
        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            print(f"length of val loader: {len(self.val_loader)}")
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device).float() #need to compute labels

                # 1. Generate saliency map for each image in the batch
                saliency_maps = []
                for img in batch:
                    saliency_map = calculate_saliency_map(img.cpu().numpy(), self.model)
                    saliency_maps.append(saliency_map)
                saliency_maps = np.array(saliency_maps)  # Shape: (batch_size, H, W)

                # 2. Generate binary labels for each pixel based on saliency map
                # For simplicity, we threshold the saliency map at 0.5 to generate binary labels
                # Adjust this threshold as per your requirement.
                saliency_labels = (saliency_maps > 0.5).astype(float)
                
                logits = self.model(batch)
                #print(f"Predictions inside validation: {logits.size()}")
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = torch.sigmoid(logits).round()

                # Ensure preds and labels are iterable
                if preds.dim() == 0:  # If preds is a scalar, make it a list
                    preds = preds.unsqueeze(0)

                # Extend results with the batch predictions and labels
                results["preds"].extend(preds.view(-1).cpu().tolist())  # Flatten preds to 1D list
                results["labels"].extend(labels.view(-1).cpu().numpy().tolist())  # Flatten labels to 1D list
                

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")

def compute_accuracy(labels, preds):
    # Ensure labels and preds are tensors
    labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
    preds = torch.tensor(preds) if not isinstance(preds, torch.Tensor) else preds

    # Check if labels and preds are in the correct shape (should be 1D)
    if labels.dim() == 1 and preds.dim() == 1:  # Ensure labels and preds are 1D
        total = labels.size(0)  # Get the batch size
        correct = (labels == preds).sum().item()  # Count the number of correct predictions
        accuracy = correct / total  # Calculate accuracy
    else:
        # Handle unexpected dimensions (e.g., multi-dimensional labels)
        raise ValueError("Labels and predictions must be 1D tensors.")

    return accuracy

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


def calculate_saliency_map(image, model, grid_size=50, patch_size=[(150, 150), (250, 250), (400, 400)]):
    """
    Calculate the saliency map for a given image using the trained Mr-CNN model.
    
    Args:
        image: Input image (H, W, C) in numpy format.
        model: Trained Mr-CNN model.
        grid_size: Number of locations sampled along each axis (default 50x50 grid).
        patch_size: List of patch sizes to be resized to for processing.
        
    Returns:
        saliency_map: Final saliency map resized to the original image size.
    """
    # 1. Prepare the down-sampled grid of locations
    height, width = image.shape[:2]
    x_coords = np.linspace(0, width - 1, grid_size, dtype=int)
    y_coords = np.linspace(0, height - 1, grid_size, dtype=int)
    sampled_locations = [(x, y) for y in y_coords for x in x_coords]
    
    # 2. Extract patches for each sampled location at 3 scales
    saliency_values = np.zeros((grid_size, grid_size))  # Initialize the saliency map
    
    for i, (x, y) in enumerate(sampled_locations):
        patches = []
        for size in patch_size:
            half_h, half_w = size[0] // 2, size[1] // 2
            patch = image[max(0, y - half_h):min(height, y + half_h),
                          max(0, x - half_w):min(width, x + half_w)]
            patch = torch.tensor(patch).float()  # Convert to tensor
            
            # Resize patch using PyTorch's interpolate
            patch_tensor = patch.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, C, H, W)
            resized_patch = F.interpolate(patch_tensor, size=size, mode='bilinear', align_corners=False)
            patches.append(resized_patch.squeeze(0).permute(1, 2, 0))  # Convert back to (H, W, C)
        
        # Convert patches to tensor and pass through the model
        patches_tensor = torch.stack(patches).permute(0, 3, 1, 2).float().to(device)  # (N, C, H, W)
        with torch.no_grad():
            saliency_score = model(patches_tensor).cpu().numpy().mean()
        
        # Store the saliency value for this location
        saliency_values[i // grid_size, i % grid_size] = saliency_score
    
    # 3. Rescale to the original image size using PyTorch's interpolate
    saliency_map_tensor = torch.tensor(saliency_values).unsqueeze(0).unsqueeze(0).float().to(device)  # (1, 1, H, W)
    saliency_map_resized = F.interpolate(saliency_map_tensor, size=(height, width), mode='bilinear', align_corners=False)
    saliency_map = saliency_map_resized.squeeze().cpu().numpy()  # Convert back to numpy array

    return saliency_map


if __name__ == "__main__":
    main(parser.parse_args())