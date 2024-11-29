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
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
train_dataset_path = './train_data.pth.tar'
val_dataset_path = './val_data.pth.tar'
test_dataset_path = './test_data.pth.tar'
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=2e-3, type=float, help="Learning rate") 
parser.add_argument(
    "--batch-size",
    default=256,
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
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.ToTensor(),
    ])
    args.dataset_root.mkdir(parents=True, exist_ok=True)
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
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, summary_writer, DEVICE, scheduler
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

        self.shared_conv1 = nn.Conv2d(input_channels, 96, kernel_size=7, stride=1)
        self.shared_conv2 = nn.Conv2d(96, 160, kernel_size=3, stride=1)
        self.shared_conv3 = nn.Conv2d(160, 288, kernel_size=3, stride=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(288 * 3 * 3, 512)  

        self.fusion_fc = nn.Linear(512 * 3, 512)

        self.output_layer = nn.Linear(512, output_classes)

    def forward_stream(self, x):
        x = F.relu(self.shared_conv1(x))
        x = self.pool(x)
        x = F.relu(self.shared_conv2(x))
        x = self.pool(x)
        x = F.relu(self.shared_conv3(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        return x

    def forward(self, x):
      
        crop1, crop2, crop3 = x[:, 0], x[:, 1], x[:, 2] 

        stream1_out = self.forward_stream(crop1)
        stream2_out = self.forward_stream(crop2)
        stream3_out = self.forward_stream(crop3)

        combined = torch.cat((stream1_out, stream2_out, stream3_out), dim=1)

        fusion_out = F.relu(self.fusion_fc(combined))
        fusion_out = self.dropout(fusion_out)

        output = torch.sigmoid(self.output_layer(fusion_out)).squeeze(1) 

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
            for batch,labels in self.train_loader:
                #print(f"labels in training shape: {labels.shape}")
                batch = batch.to(self.device)
                labels = labels.to(self.device).float()
                data_load_end_time = time.time()
                logits = self.model.forward(batch) #logits has a value between 0 and 1 
                loss = self.criterion(logits, labels)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits > 0.9
                    preds = preds.int()
                    accuracy = compute_accuracy(labels, preds)
                    tp,tn,fp,fn = compute_statistics(labels, preds)
                    precision = compute_precision(tp,fp)
                    sensitivity = compute_sensitivity(tp,fn)


                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, sensitivity, precision, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, sensitivity, precision,loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()
                
            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0: 
                print("call validate()")
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

            self.scheduler.step() # used for learning rate decay 

    def print_metrics(self, epoch, accuracy,sensitivity,precision, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"batch sensitivity: {sensitivity * 100:2.2f}, "
                f"batch precision: {precision * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, sensitivity, precision, loss, data_load_time, step_time):
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
        self.summary_writer.add_scalars(
                "sensitivity",
                {"train": sensitivity},
                self.step
        )
        self.summary_writer.add_scalars(
                "precision",
                {"train": precision},
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

        total_sensitivity = 0
        total_accuracy = 0
        total_precision = 0
        total_loss = 0
        #print("entering val loop")
        with torch.no_grad():  
            for idx, (batch, _) in enumerate(self.val_loader):
                #print(f"in validation: {idx}")
                batch = batch.to(self.device) # batch is a tensor of shape [2500 42 42 3 3] 
                
                logits = self.model(batch)  # logits has shape [2500] and holds predictions for each 2500 [42 42 3 3] tensors; (1 image -> 2500 predictions)

                #get original image height and width
                height, width = self.val_loader.dataset.dataset[idx]['X'].shape[1], self.val_loader.dataset.dataset[idx]['X'].shape[2] 

                down_sampled_saliency_map = logits.reshape(50, 50) #reshape logits to 50x50 

                saliency_map = F.interpolate(
                    down_sampled_saliency_map.unsqueeze(0).unsqueeze(0), 
                    size=(height, width), 
                    mode="bilinear", 
                    align_corners=False
                ).squeeze(0).squeeze(0)  #using interpolation to bring down sampled saliency map to original img size
                
                filename = self.val_loader.dataset.dataset[idx]['file'][:-5] #prepare filename to extract ground truth fixation

                gt_path = f"ALLFIXATIONMAPS/{filename}_fixMap.jpg"
                gt_fixation_map = io.read_image(gt_path) 
              #-----------------------
                sm_flatten = saliency_map.flatten().to(self.device)
                sm_flatten = sm_flatten > 0.9
                sm_flatten = sm_flatten.int()
                gt_flatten = gt_fixation_map.squeeze(0).flatten().to(self.device)
                (tp, tn, fp, fn) = compute_statistics(gt_flatten, sm_flatten)
                accuracy = compute_accuracy(gt_flatten,sm_flatten)
                precision = compute_precision(tp, fp)
                sensitivity = compute_sensitivity(tp,fn)
                total_sensitivity += sensitivity
                total_accuracy += accuracy
                total_precision += precision
                #print(f"shape gt_flat: {gt_flatten.shape} and {sm_flatten.shape}")
                #gt_flat_long = gt_flatten.long()
                #loss = self.criterion(sm_flatten_raw, gt_flatten.float())
                #total_loss += loss.item()
                #print(f"(per batch) saliency map[{idx}]: accuracy {accuracy}, precision: {precision}, sensitivity: {sensitivity}")
               #-----------------------
                saliency_map = saliency_map.cpu().numpy()
                gt_fixation_map = gt_fixation_map.cpu().numpy()

                preds[filename] = saliency_map
                targets[filename] = gt_fixation_map

        n = len(self.val_loader)

        average_accuracy = total_accuracy / n
        self.summary_writer.add_scalars(
                "accuracy",
                {"validation": average_accuracy},
                self.step
        )
        average_loss = total_loss / n
        self.summary_writer.add_scalars(
                "loss",
                {"validation": average_loss},
                self.step
        )
        average_sensitivity = total_sensitivity / n
        self.summary_writer.add_scalars(
                "sensitivity",
                {"validation": average_sensitivity},
                self.step
        )
        average_precision = total_precision / n
        self.summary_writer.add_scalars(
                "precision",
                {"validation": average_precision},
                self.step
        )

        AUC = calculate_auc(preds, targets)
        self.summary_writer.add_scalars(
                "AUC",
                {"validation": AUC},
                self.step
        )
        print(f"validation AUC: {AUC}")
        

def compute_accuracy(labels, preds):
    #print("computing accuracy")
    total = labels.size(0) 
    correct = (labels == preds).sum().item()  
    accuracy = correct / total 
    
    return accuracy

def compute_precision(tp, fp):
    #print("compute precision")
    if tp + fp == 0:  # Avoid division by zero
        return 0.0  # Or you can return a default value like None, depending on your needs
    precision = tp / (tp + fp)
    return precision

def compute_sensitivity(tp, fn):
    #print("compute sensitivity")
    if tp + fn == 0:  # Avoid division by zero
        return 0.0  # Or return None or any other default value
    sensitivity = tp / (tp + fn)
    return sensitivity

def compute_statistics(labels, preds):
    true_positives = ((labels == 1) & (preds == 1)).sum().item()
    false_negatives = ((labels == 1) & (preds == 0)).sum().item()
    true_negatives = ((labels == 0) & (preds == 0)).sum().item()
    false_positives = ((labels == 0) & (preds == 1)).sum().item()
        
    return (true_positives, true_negatives, false_positives, false_negatives)


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
