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
import torchvision.io as io
from torch.optim.lr_scheduler import StepLR

import argparse
from pathlib import Path

from dataset import MIT, crop_to_region
from metrics import calculate_auc

import numpy as np

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a MRCNN on MIT dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
#default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
train_dataset_path = './train_data.pth.tar'
val_dataset_path = './val_data.pth.tar'
test_dataset_path = './test_data.pth.tar'
#parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=2e-3, type=float, help="Learning rate") #learning rate set to 0.002
parser.add_argument(
    "--batch-size",
    default=256, #specified in the paper to choose 256 bath size for MIT
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
    default=200, #originally set to 2
    type=int,
    help="How frequently to test the model on the validation set in number of steps", #originally measure number of epochs
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
    print("I am using cuda")
    print(f"Should print True if CUDA is available{torch.cuda.is_available()}")  # Should print True if CUDA is available
    print(f"Number of GPUS: {torch.cuda.device_count()}")  #  Should print the number of GPUs available
    print(f"print name of first device: {torch.cuda.get_device_name(0)}")  # Print the name of the first GPU
else:
    DEVICE = torch.device("cpu")
    print("Device detected is cpu")

def main(args):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.ToTensor(),
    ])
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
        batch_size=2500,  #originally: args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )
    

    model = MrCNN(input_channels=3, output_classes=1)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum = 0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    print("begin training")
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, summary_writer, DEVICE, scheduler, transform
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()

class MrCNN(nn.Module):
    def __init__(self, input_channels=3, output_classes=1):
        super(MrCNN, self).__init__()
        self.dropout = nn.Dropout(p = 0.5)

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
        x = self.dropout(x)

        # Flatten and pass through FC layer for each stream
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

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
        fusion_out = self.dropout(fusion_out)

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
        scheduler: torch.optim.lr_scheduler.StepLR,
        transform: transforms.Compose
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
        self.scheduler = scheduler
        self.transform = transform

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
                logits = self.model.forward(batch)
                loss = self.criterion(logits, label)
                #print(f"Epoch [{epoch + 1}/{epochs}], Batch Loss: {loss.item()}")

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
                
                #data augumentation
                batch = self.transform(batch).to(self.device)
                label = label.to(self.device).float()
                data_load_end_time = time.time()
                logits = self.model.forward(batch)
                loss = self.criterion(logits, label)
                #print(f"Epoch [{epoch + 1}/{epochs}], Batch Loss: {loss.item()}")

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
            if ((self.step + 1) % val_frequency) == 0: #originally if ((epoch + 1) % val_frequency) == 0 but need to check val every 200 steps
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

            self.scheduler.step() # used for learning rate decay 

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
        preds = {}
        targets = {}

        self.model.eval()

        with torch.no_grad():  # No gradient computation needed for validation
          for idx, (batch, _) in enumerate(self.val_loader):
              print(f"Validation {idx}/100 complete")

              batch = batch.to(self.device)
              logits = self.model(batch)  # Output shape: [2500]

              # Get original image height and width
              height, width = self.val_loader.dataset.dataset[idx]['X'].shape[1], self.val_loader.dataset.dataset[idx]['X'].shape[2]

              # Reshape logits to down-sampled saliency map
              down_sampled_saliency_map = logits.reshape(50, 50)

              # Interpolate to match original image size
              saliency_map = F.interpolate(
                  down_sampled_saliency_map.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions: [1, 1, 50, 50]
                  size=(height, width),  # Target height and width
                  mode="bilinear",  # Interpolation method
                  align_corners=False
              ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions: [h, w]

              # File name extraction
              filename = self.val_loader.dataset.dataset[idx]['file'][:-5]

              # Extract ground truth fixation map
              gt_path = f"ALLFIXATIONMAPS/{filename}_fixMap.jpg"
              gt_fixation_map = io.read_image(gt_path)  # Returns a tensor with shape (channels, height, width)
              
              #Move saliency map and fixation map to cpu
              saliency_map = saliency_map.cpu().numpy()
              gt_fixation_map = gt_fixation_map.cpu().numpy()

              # Store predictions and targets in dictionaries
              preds[filename] = saliency_map
              targets[filename] = gt_fixation_map

        AUC = calculate_auc(preds, targets)

        self.summary_writer.add_scalars(
                "AUC",
                {"test": AUC},
                self.step
        )

        print(f"validation AUC: {AUC}")

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
    tb_log_dir_prefix = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_momentum=0.9_stepLR_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

if __name__ == "__main__":
    main(parser.parse_args())
