#!/usr/bin/env python3
import os
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

import argparse
from pathlib import Path

from dataset import MIT

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a MRCNN on MIT dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

train_dataset_path = './train_data.pth.tar'
val_dataset_path = './val_data.pth.tar'
test_dataset_path = './test_data.pth.tar'

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

    # Load datasets
    train_dataset = MIT(dataset_path=train_dataset_path).dataset
    test_dataset = MIT(dataset_path=test_dataset_path).dataset
    val_dataset = MIT(dataset_path=val_dataset_path).dataset

    # DataLoader for training, testing, and validation datasets
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

    # Initialize the model (using multiresolution approach)
    model = MrCNN(input_channels=3, output_classes=1)

    # Criterion: Binary Cross Entropy Loss
    criterion = nn.BCEWithLogitsLoss()  # TASK 8: Using BCE loss for binary classification

    # Optimizer: SGD with momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0002)  # Task 11: Adjusted momentum and weight decay

    # TensorBoard summary writer
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(str(log_dir), flush_secs=5)

    # Trainer setup
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, summary_writer, DEVICE
    )

    # Start training process
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

        # Define one set of convolutional layers for each resolution stream
        self.shared_conv1 = nn.Conv2d(input_channels, 96, kernel_size=7, stride=1)
        self.shared_conv2 = nn.Conv2d(96, 160, kernel_size=3, stride=1)
        self.shared_conv3 = nn.Conv2d(160, 288, kernel_size=3, stride=1)

        # Pooling and fully connected layers for each stream
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(288 * 3 * 3, 512)  # Output size is 3x3 after pooling (adjust for 3 stream fusion)

        # Fusion layer after concatenation of all three streams
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
       
        # Flatten and pass through the FC layer for each stream
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        # Split input into three crops for each stream (resolution-specific inputs)
        crop1, crop2, crop3 = x[:, 0], x[:, 1], x[:, 2]  # Assuming x has shape [batch_size, 3, 3, 42, 42]

        # Pass each crop through the shared convolutional layers independently
        stream1_out = self.forward_stream(crop1)
        stream2_out = self.forward_stream(crop2)
        stream3_out = self.forward_stream(crop3)

        # Concatenate outputs from the three streams
        combined = torch.cat((stream1_out, stream2_out, stream3_out), dim=1)

        # Pass through fusion and output layers
        fusion_out = F.relu(self.fusion_fc(combined))
        output = torch.sigmoid(self.output_layer(fusion_out)).squeeze(1)  # TASK 8: Sigmoid for binary output

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
            for i, batch in enumerate(self.train_loader):  # Adjusted for batch structure with 'X' and 'y'
                images = batch['X'].to(self.device)  # batch['X'] holds the input images
                labels = batch['y'].to(self.device).float()  # batch['y'] holds the ground truth labels
               
                data_load_end_time = time.time()

                # Forward pass and loss computation
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                # Backward pass and optimizer step
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Accuracy calculation
                with torch.no_grad():
                    preds = torch.sigmoid(logits).round()  # TASK 10: Sigmoid and rounding for binary output
                    accuracy = compute_accuracy(labels, preds)

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
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: {data_load_time:.3f}, "
                f"step time: {step_time:.3f}"
            )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("train/loss", loss, self.step)
        self.summary_writer.add_scalar("train/accuracy", accuracy, self.step)

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            val_accuracy = 0
            total_samples = 0
            for batch in self.val_loader:
                images = batch['X'].to(self.device)  # Adjusted for batch structure with 'X' and 'y'
                labels = batch['y'].to(self.device)  # batch['y'] holds the ground truth labels

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.sigmoid(logits).round()  # TASK 10: Sigmoid and rounding for binary output
                val_accuracy += compute_accuracy(labels, preds)

                total_samples += len(labels)

            val_loss /= total_samples
            val_accuracy /= total_samples

        print(f"Validation Loss: {val_loss:.5f}, Validation Accuracy: {val_accuracy * 100:.2f}%")
        self.summary_writer.add_scalar("val/loss", val_loss, self.step)
        self.summary_writer.add_scalar("val/accuracy", val_accuracy, self.step)


def compute_accuracy(y_true, y_pred):
    correct = (y_true == y_pred).sum().item()
    accuracy = correct / len(y_true)
    return accuracy


def get_summary_writer_log_dir(args):
    log_dir = args.log_dir / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)