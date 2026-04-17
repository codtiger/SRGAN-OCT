# Copyright 2026. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import random
import time
from typing import Any, List

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import model
from imgproc import image_resize, image_to_tensor, tensor_to_image
from utils import build_iqa_model, make_directory, save_checkpoint, AverageMeter, ProgressMeter, Summary, \
    load_resume_state_dict, load_pretrained_state_dict


class OCTImageDataset(Dataset):
    """A medical OCT dataset that returns 128x128 GT images and 32x32 LR images."""

    def __init__(
            self,
            images_root: str,
            gt_image_size: int = 128,
            upscale_factor: int = 4,
    ) -> None:
        super(OCTImageDataset, self).__init__()

        images_root = resolve_path(images_root)
        if not os.path.isdir(images_root):
            raise FileNotFoundError(f"OCT image root does not exist: {images_root}")

        self.gt_image_size = gt_image_size
        self.upscale_factor = upscale_factor
        self.image_paths = self._collect_image_paths(images_root)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in OCT image root: {images_root}")

    @staticmethod
    def _collect_image_paths(images_root: str) -> List[str]:
        image_paths = []
        for root, _, files in os.walk(images_root):
            if os.path.basename(root).startswith("__MACOSX"):
                continue
            for file_name in sorted(files):
                lower_name = file_name.lower()
                if lower_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_paths.append(os.path.join(root, file_name))
        return image_paths

    @staticmethod
    def _prepare_image(image: np.ndarray, target_size: int) -> np.ndarray:
        height, width = image.shape[:2]
        if height == target_size and width == target_size:
            return image

        if height < target_size or width < target_size:
            return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

        top = (height - target_size) // 2
        left = (width - target_size) // 2
        image = image[top:top + target_size, left:left + target_size]
        if image.shape[0] != target_size or image.shape[1] != target_size:
            image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        return image

    def __getitem__(self, batch_index: int) -> dict:
        image_path = self.image_paths[batch_index]
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        image = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2RGB)
        image = self._prepare_image(image, self.gt_image_size)
        gt_tensor = image_to_tensor(image, False, False)
        lr_tensor = image_resize(gt_tensor, 1.0 / self.upscale_factor)

        return {
            "gt": gt_tensor,
            "lr": lr_tensor,
            "image_name": image_path,
        }

    def __len__(self) -> int:
        return len(self.image_paths)


def resolve_path(path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        return path

    guess = os.path.normpath(path)
    if os.path.isdir(guess):
        return guess

    guess_space = guess + " "
    if os.path.isdir(guess_space):
        return guess_space

    stripped = guess.rstrip()
    if os.path.isdir(stripped):
        return stripped

    if os.path.isdir(os.path.dirname(guess)):
        for entry in os.listdir(os.path.dirname(guess)):
            if entry.startswith(os.path.basename(stripped)):
                candidate = os.path.join(os.path.dirname(guess), entry)
                if os.path.isdir(candidate):
                    return candidate

    raise FileNotFoundError(f"Could not resolve path: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCT super-resolution training")
    parser.add_argument("--dataset_root",
                        type=str,
                        default="~/.cache/kagglehub/datasets/paultimothymooney/kermany2018/versions/2/oct2017/OCT2017 /",
                        help="Base OCT dataset root containing train/val/test subfolders.")
    parser.add_argument("--train_root",
                        type=str,
                        default=None,
                        help="Optional explicit train image root.")
    parser.add_argument("--val_root",
                        type=str,
                        default=None,
                        help="Optional explicit validation image root.")
    parser.add_argument("--exp_name",
                        type=str,
                        default="oct_srresnet",
                        help="Experiment name for checkpoints and logs.")
    parser.add_argument("--epochs",
                        type=int,
                        default=100,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Training batch size.")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--image_size",
                        type=int,
                        default=128,
                        help="Target ground truth image size.")
    parser.add_argument("--scale",
                        type=int,
                        default=4,
                        choices=[2, 4, 8],
                        help="Upscale factor between LR and GT images.")
    parser.add_argument("--num_workers",
                        type=int,
                        default=8,
                        help="Number of data loader workers.")
    parser.add_argument("--device",
                        type=int,
                        default=0,
                        help="CUDA device id.")
    parser.add_argument("--seed",
                        type=int,
                        default=123,
                        help="Random seed.")
    parser.add_argument("--compile_model",
                        type=bool,
                        default=True,
                        help="Whether to use torch.compile for faster training.")
    parser.add_argument("--save_dir",
                        type=str,
                        default=".",
                        help="Directory for saving samples and checkpoints.")
    parser.add_argument("--resume",
                        type=str,
                        default=None,
                        help="Path to resume checkpoint.")
    parser.add_argument("--pretrained",
                        type=str,
                        default=None,
                        help="Path to pretrained generator checkpoint.")
    return parser.parse_args()


def build_dataset(
        train_root: str,
        val_root: str,
        gt_image_size: int,
        scale: int,
        batch_size: int,
        num_workers: int,
        device: torch.device,
):
    train_dataset = OCTImageDataset(train_root, gt_image_size, scale)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    train_prefetcher = CUDAPrefetcher(train_dataloader, device)

    val_prefetcher = None
    if val_root is not None and os.path.isdir(val_root):
        val_dataset = OCTImageDataset(val_root, gt_image_size, scale)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    drop_last=False,
                                    persistent_workers=True)
        val_prefetcher = CUDAPrefetcher(val_dataloader, device)

    return train_prefetcher, val_prefetcher


class CUDAPrefetcher:
    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device
        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)


def build_model(device: torch.device, scale: int, compile_model: bool = False):
    g_model = model.SRResNet(in_channels=3,
                             out_channels=3,
                             channels=64,
                             num_rcb=16,
                             upscale=scale)
    g_model = g_model.to(device)
    
    # Compile model for faster training if requested
    if compile_model:
        g_model = torch.compile(g_model)
    
    return g_model


def define_loss(device: torch.device):
    pixel_criterion = nn.MSELoss().to(device)
    return pixel_criterion


def define_optimizer(g_model: nn.Module, lr: float):
    return optim.Adam(g_model.parameters(), lr, (0.9, 0.999), eps=1e-8, weight_decay=0)


def define_scheduler(optimizer: optim.Optimizer, epochs: int):
    milestones = [max(1, epochs // 2), max(1, epochs * 3 // 4)]
    return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)


def train(
        g_model: nn.Module,
        ema_g_model: Any,
        train_prefetcher: CUDAPrefetcher,
        pixel_criterion: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device,
        config: Any,
) -> None:
    batches = len(train_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":6.6f", Summary.NONE)
    progress = ProgressMeter(batches,
                             [batch_time, data_time, losses],
                             prefix=f"Epoch: [{epoch + 1}] ")

    g_model.train()

    batch_index = 0
    train_prefetcher.reset()
    end = time.time()
    batch_data = train_prefetcher.next()

    while batch_data is not None:
        gt = batch_data["gt"].to(device, non_blocking=True)
        lr = batch_data["lr"].to(device, non_blocking=True)

        data_time.update(time.time() - end)

        g_model.zero_grad(set_to_none=True)
        with amp.autocast():
            sr = g_model(lr)
            loss = pixel_criterion(sr, gt)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema_g_model is not None:
            ema_g_model.update_parameters(g_model)

        losses.update(loss.item(), lr.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % config["PRINT_FREQ"] == 0:
            writer.add_scalar("Train/Loss", loss.item(), epoch * batches + batch_index)
            progress.display(batch_index)

        batch_data = train_prefetcher.next()
        batch_index += 1


def validate(
        g_model: nn.Module,
        val_prefetcher: CUDAPrefetcher,
        psnr_model: Any,
        ssim_model: Any,
        device: torch.device,
) -> tuple[float, float]:
    batches = len(val_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    psnres = AverageMeter("PSNR", ":4.2f", Summary.AVERAGE)
    ssimes = AverageMeter("SSIM", ":4.4f", Summary.AVERAGE)
    progress = ProgressMeter(batches,
                             [batch_time, psnres, ssimes],
                             prefix="Validate: ")

    g_model.eval()
    val_prefetcher.reset()
    batch_data = val_prefetcher.next()
    end = time.time()
    batch_index = 0

    with torch.no_grad():
        while batch_data is not None:
            gt = batch_data["gt"].to(device, non_blocking=True)
            lr = batch_data["lr"].to(device, non_blocking=True)
            sr = g_model(lr)

            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)

            psnres.update(psnr.mean().item(), sr.size(0))
            ssimes.update(ssim.mean().item(), sr.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_index % max(1, batches // 10) == 0:
                progress.display(batch_index)

            batch_data = val_prefetcher.next()
            batch_index += 1

    progress.display_summary()
    return psnres.avg, ssimes.avg


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    dataset_root = resolve_path(args.dataset_root)
    train_root = resolve_path(args.train_root) if args.train_root else os.path.join(dataset_root, "train")
    val_root = resolve_path(args.val_root) if args.val_root else os.path.join(dataset_root, "val")
    if not os.path.isdir(val_root):
        val_root = None

    device = torch.device("cuda", args.device)

    config = {
        "PRINT_FREQ": 10,
    }

    train_prefetcher, val_prefetcher = build_dataset(
        train_root,
        val_root,
        args.image_size,
        args.scale,
        args.batch_size,
        args.num_workers,
        device,
    )

    g_model = build_model(device, args.scale, args.compile_model)
    pixel_criterion = define_loss(device)
    optimizer = define_optimizer(g_model, args.lr)
    scheduler = define_scheduler(optimizer, args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    samples_dir = os.path.join(args.save_dir, "samples", args.exp_name)
    results_dir = os.path.join(args.save_dir, "results", args.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    writer = SummaryWriter(os.path.join(args.save_dir, "samples", "logs", args.exp_name))

    if args.pretrained is not None:
        g_model = load_pretrained_state_dict(g_model,
                                             args.compile_model,
                                             args.pretrained)
        print(f"Loaded pretrained model from {args.pretrained}")

    start_epoch = 0
    best_psnr = 0.0
    best_ssim = 0.0

    if args.resume is not None:
        g_model, _, start_epoch, best_psnr, best_ssim, optimizer = load_resume_state_dict(
            g_model,
            None,
            optimizer,
            scheduler,
            args.compile_model,
            args.resume,
        )
        print(f"Resumed training from {args.resume} at epoch {start_epoch}")

    psnr_model = None
    ssim_model = None
    if val_prefetcher is not None:
        psnr_model, ssim_model = build_iqa_model(args.scale, False, device)

    for epoch in range(start_epoch, args.epochs):
        train(g_model,
              None,
              train_prefetcher,
              pixel_criterion,
              optimizer,
              epoch,
              scaler,
              writer,
              device,
              config)
        scheduler.step()

        psnr = 0.0
        ssim = 0.0
        if val_prefetcher is not None:
            psnr, ssim = validate(g_model, val_prefetcher, psnr_model, ssim_model, device)
            writer.add_scalar("Validate/PSNR", psnr, epoch + 1)
            writer.add_scalar("Validate/SSIM", ssim, epoch + 1)

        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == args.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)

        save_checkpoint({
            "epoch": epoch + 1,
            "psnr": psnr,
            "ssim": ssim,
            "state_dict": g_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
            f"epoch_{epoch + 1}.pth.tar",
            samples_dir,
            results_dir,
            "g_best.pth.tar",
            "g_last.pth.tar",
            is_best,
            is_last)


if __name__ == "__main__":
    main()
