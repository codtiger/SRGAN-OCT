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
from typing import Any, Optional
from functools import partial

import numpy as np
import yaml

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model
from imgproc import tensor_to_image
from utils import build_iqa_model, make_directory, save_checkpoint, AverageMeter, ProgressMeter, Summary, \
    load_resume_state_dict, load_pretrained_state_dict, ema_avg_fn, resolve_path
from oct_dataset import OCTImageDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCT super-resolution SRResNet training")
    parser.add_argument("--config_path",
                        type=str,
                        default="./configs/train/OCT_SRGAN_x4.yaml",
                        help="Path to train config file.")
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


def build_model(device: torch.device, config: dict):
    # Generator
    g_config = config["MODEL"]["G"]
    g_model = model.SRResNet(in_channels=g_config["IN_CHANNELS"],
                             out_channels=g_config["OUT_CHANNELS"],
                             channels=g_config["CHANNELS"],
                             num_rcb=g_config["NUM_RCB"],
                             upscale=config["SCALE"])
    g_model = g_model.to(device)

    if config["MODEL"]["EMA"]["ENABLE"]:
        ema_decay = config["MODEL"]["EMA"]["DECAY"]
        ema_g_model = AveragedModel(g_model, device=device,
                            avg_fn=partial(ema_avg_fn, decay=ema_decay))
    else:
        ema_g_model = None

    if g_config["COMPILED"]:
        print("compiling generator with torch.compile for faster training...")
        g_model = torch.compile(g_model)
    if config["MODEL"]["EMA"]["COMPILED"] and ema_g_model is not None:
        ema_g_model = torch.compile(ema_g_model)

    return g_model, ema_g_model


def define_loss(device: torch.device):
    pixel_criterion = nn.MSELoss().to(device)
    return pixel_criterion


def define_optimizer(g_model: nn.Module, config: dict):
    g_lr = config["TRAIN"]["OPTIM"]["LR"]
    betas = tuple(config["TRAIN"]["OPTIM"]["BETAS"])
    eps = config["TRAIN"]["OPTIM"]["EPS"]
    weight_decay = config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"]
    g_optimizer = optim.Adam(g_model.parameters(), g_lr, betas, eps, weight_decay)
    return g_optimizer


def train(
        g_model: nn.Module,
        ema_g_model: Optional[nn.Module],
        train_prefetcher: CUDAPrefetcher,
        pixel_criterion: nn.Module,
        g_optimizer: optim.Optimizer,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device,
        config: Any,
) -> None:
    batches = len(train_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    g_losses = AverageMeter("G Loss", ":6.6f", Summary.NONE)
    progress = ProgressMeter(batches,
                             [batch_time, data_time, g_losses],
                             prefix=f"Epoch: [{epoch + 1}] ")

    g_model.train()

    pixel_weight = torch.tensor(config["PIXEL_LOSS_WEIGHT"]).to(device)

    batch_index = 0
    train_prefetcher.reset()
    end = time.time()
    batch_data = train_prefetcher.next()

    while batch_data is not None:
        gt = batch_data["gt"].to(device, non_blocking=True)
        lr = batch_data["lr"].to(device, non_blocking=True)

        data_time.update(time.time() - end)

        # Train Generator with pixel loss only
        g_model.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            sr = g_model(lr)
            pixel_loss = torch.sum(torch.mul(pixel_weight, pixel_criterion(sr, gt)))

        scaler.scale(pixel_loss).backward()
        scaler.step(g_optimizer)
        scaler.update()

        if ema_g_model is not None:
            ema_g_model.update_parameters(g_model)

        g_losses.update(pixel_loss.item(), gt.shape[0])
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % config["PRINT_FREQ"] == 0:
            iters = epoch * batches + batch_index
            writer.add_scalar("Train/Loss", pixel_loss.item(), iters)
            progress.display(batch_index)

        batch_data = train_prefetcher.next()
        batch_index += 1


def validate(
        g_model: nn.Module,
        val_prefetcher: CUDAPrefetcher,
        psnr_model: Any,
        ssim_model: Any,
        device: torch.device,
        writer: SummaryWriter,
        epoch: int,
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
    sample_count = 0

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

            # Save sample images to TensorBoard
            if sample_count < 4:
                for i in range(min(sr.size(0), 4 - sample_count)):
                    gt_img = tensor_to_image(gt[i:i+1], False, False)
                    sr_img = tensor_to_image(sr[i:i+1], False, False)
                    lr_img = tensor_to_image(lr[i:i+1], False, False)

                    # Convert to tensor for TensorBoard (B, C, H, W)
                    gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    sr_tensor = torch.from_numpy(sr_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                    writer.add_images(f"Samples/GT_{sample_count}", gt_tensor, epoch)
                    writer.add_images(f"Samples/SR_{sample_count}", sr_tensor, epoch)
                    writer.add_images(f"Samples/LR_{sample_count}", lr_tensor, epoch)
                    sample_count += 1

            if batch_index % max(1, batches // 10) == 0:
                progress.display(batch_index)

            batch_data = val_prefetcher.next()
            batch_index += 1

    progress.display_summary()
    return psnres.avg, ssimes.avg


def main() -> None:
    args = parse_args()
    
    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)
    
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed_all(config["SEED"])
    cudnn.benchmark = True

    dataset_root = resolve_path(config["TRAIN"]["DATASET"]["TRAIN_GT_IMAGES_DIR"])
    train_root = dataset_root
    val_root = resolve_path(config["TEST"]["DATASET"]["PAIRED_TEST_GT_IMAGES_DIR"]) if config["TEST"]["DATASET"]["PAIRED_TEST_GT_IMAGES_DIR"] else None
    if val_root and not os.path.isdir(val_root):
        val_root = None

    device = torch.device("cuda", config["DEVICE_ID"])

    config_dict = {
        "PRINT_FREQ": config["TRAIN"]["PRINT_FREQ"],
        "PIXEL_LOSS_WEIGHT": config["TRAIN"]["LOSSES"]["PIXEL_LOSS"]["WEIGHT"][0],
    }

    train_prefetcher, val_prefetcher = build_dataset(
        train_root,
        val_root,
        config["TRAIN"]["DATASET"]["GT_IMAGE_SIZE"],
        config["SCALE"],
        config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
        config["TRAIN"]["HYP"]["NUM_WORKERS"],
        device,
    )

    g_model, ema_g_model = build_model(device, config)
    pixel_criterion = define_loss(device)
    g_optimizer = define_optimizer(g_model, config)
    scaler = amp.GradScaler()

    samples_dir = os.path.join("samples", config["EXP_NAME"])
    results_dir = os.path.join("results", config["EXP_NAME"])
    make_directory(samples_dir)
    make_directory(results_dir)

    writer = SummaryWriter(os.path.join(config["LOG_DIR"], "samples", "logs", config["EXP_NAME"]))

    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_G_MODEL"]:
        g_model = load_pretrained_state_dict(g_model,
                                             config["MODEL"]["G"]["COMPILED"],
                                             config["TRAIN"]["CHECKPOINT"]["PRETRAINED_G_MODEL"])
        print(f"Loaded pretrained generator from {config['TRAIN']['CHECKPOINT']['PRETRAINED_G_MODEL']}")

    start_epoch = 0
    best_psnr = 0.0
    best_ssim = 0.0

    if config["TRAIN"]["CHECKPOINT"]["RESUMED_G_MODEL"]:
        g_model, ema_g_model, start_epoch, best_psnr, best_ssim, g_optimizer = load_resume_state_dict(
            g_model,
            ema_g_model,
            g_optimizer,
            None,
            config["MODEL"]["G"]["COMPILED"],
            config["TRAIN"]["CHECKPOINT"]["RESUMED_G_MODEL"],
        )
        print(f"Resumed generator from {config['TRAIN']['CHECKPOINT']['RESUMED_G_MODEL']} at epoch {start_epoch}")

    psnr_model = None
    ssim_model = None
    if val_prefetcher is not None:
        psnr_model, ssim_model = build_iqa_model(config["SCALE"], False, device)

    for epoch in range(start_epoch, config["TRAIN"]["HYP"]["EPOCHS"]):
        train(g_model,
              ema_g_model,
              train_prefetcher,
              pixel_criterion,
              g_optimizer,
              epoch,
              scaler,
              writer,
              device,
              config_dict)

        psnr = 0.0
        ssim = 0.0
        if val_prefetcher is not None:
            psnr, ssim = validate(g_model, val_prefetcher, psnr_model, ssim_model, device, writer, epoch)
            writer.add_scalar("Validate/PSNR", psnr, epoch + 1)
            writer.add_scalar("Validate/SSIM", ssim, epoch + 1)

        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == config["TRAIN"]["HYP"]["EPOCHS"]
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)

        save_checkpoint({
            "epoch": epoch + 1,
            "psnr": psnr,
            "ssim": ssim,
            "state_dict": g_model.state_dict(),
            "ema_state_dict": ema_g_model.state_dict() if ema_g_model is not None else None,
            "optimizer": g_optimizer.state_dict(),
            # "scheduler": g_scheduler.state_dict(),
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
