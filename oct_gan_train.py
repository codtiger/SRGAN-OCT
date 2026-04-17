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
from typing import Any, List, Optional
from functools import partial

import cv2
import numpy as np
import yaml

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import model
from imgproc import image_resize, image_to_tensor, tensor_to_image
from utils import build_iqa_model, make_directory, save_checkpoint, AverageMeter, ProgressMeter, Summary, \
    load_resume_state_dict, load_pretrained_state_dict, ema_avg_fn, resolve_path
from oct_dataset import OCTImageDataset



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCT super-resolution GAN training")
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

    # Discriminator
    d_config = config["MODEL"]["D"]
    d_model = model.DiscriminatorForVGG(in_channels=d_config["IN_CHANNELS"], 
                                        out_channels=d_config["OUT_CHANNELS"], 
                                        channels=d_config["CHANNELS"])
    d_model = d_model.to(device)

    if config["MODEL"]["EMA"]["ENABLE"]:
        ema_decay = config["MODEL"]["EMA"]["DECAY"]
        ema_g_model = AveragedModel(g_model, device=device,
                            avg_fn=partial(ema_avg_fn, decay=ema_decay))
    else:
        ema_g_model = None

    if g_config["COMPILED"]:
        print ("compiling generator with torch.compile for faster training...")
        g_model = torch.compile(g_model)
    if config["MODEL"]["EMA"]["COMPILED"] and ema_g_model is not None:
        ema_g_model = torch.compile(ema_g_model) 
    if d_config["COMPILED"]:
        print ("compiling discriminator with torch.compile for faster training...")
        d_model = torch.compile(d_model)
    

    return g_model, ema_g_model, d_model


def define_loss(device: torch.device):
    pixel_criterion = nn.MSELoss().to(device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)
    return pixel_criterion, adversarial_criterion


def compute_gradient_penalty(d_model, real_samples, fake_samples, device):
    """Compute gradient penalty for WGAN-GP"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = d_model(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def define_optimizer(g_model: nn.Module, d_model: nn.Module, config: dict):
    g_lr = config["TRAIN"]["OPTIM"]["LR"]
    d_lr = config["TRAIN"]["OPTIM"].get("D_LR", g_lr)  # Use different LR for discriminator if specified
    betas = tuple(config["TRAIN"]["OPTIM"]["BETAS"])
    eps = config["TRAIN"]["OPTIM"]["EPS"]
    weight_decay = config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"]
    g_optimizer = optim.Adam(g_model.parameters(), g_lr, betas, eps, weight_decay)
    d_optimizer = optim.Adam(d_model.parameters(), d_lr, betas, eps, weight_decay)
    return g_optimizer, d_optimizer



def define_scheduler(g_optimizer: optim.Optimizer, d_optimizer: optim.Optimizer, config: dict):
    milestones = config["TRAIN"]["LR_SCHEDULER"]["MILESTONES"]
    gamma = config["TRAIN"]["LR_SCHEDULER"]["GAMMA"]
    g_scheduler = lr_scheduler.MultiStepLR(g_optimizer, milestones=milestones, gamma=gamma)
    d_scheduler = lr_scheduler.MultiStepLR(d_optimizer, milestones=milestones, gamma=gamma)
    return g_scheduler, d_scheduler


def train(
        g_model: nn.Module,
        ema_g_model: Optional[nn.Module],
        d_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        pixel_criterion: nn.Module,
        content_criterion: model.ContentLoss,
        adversarial_criterion: nn.Module,
        g_optimizer: optim.Optimizer,
        d_optimizer: optim.Optimizer,
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
    d_losses = AverageMeter("D Loss", ":6.6f", Summary.NONE)
    progress = ProgressMeter(batches,
                             [batch_time, data_time, g_losses, d_losses],
                             prefix=f"Epoch: [{epoch + 1}] ")

    g_model.train()
    d_model.train()

    pixel_weight = torch.tensor(config["PIXEL_LOSS_WEIGHT"]).to(device)
    feature_weight = torch.tensor(config["CONTENT_LOSS_WEIGHT"]).to(device)
    adversarial_weight = torch.tensor(config["ADVERSARIAL_LOSS_WEIGHT"]).to(device)

    batch_index = 0
    train_prefetcher.reset()
    end = time.time()
    batch_data = train_prefetcher.next()

    batch_size = batch_data["gt"].shape[0]
    real_label = torch.full([batch_size, 1], 0.9, dtype=torch.float, device=device)  # Label smoothing
    fake_label = torch.full([batch_size, 1], 0.1, dtype=torch.float, device=device)  # Label smoothing

    while batch_data is not None:
        gt = batch_data["gt"].to(device, non_blocking=True)
        lr = batch_data["lr"].to(device, non_blocking=True)

        data_time.update(time.time() - end)

        # TODO: apply data augmentation here if needed
        # Train Generator
        for d_param in d_model.parameters():
            d_param.requires_grad = False

        g_model.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            sr = g_model(lr)
            pixel_loss = torch.sum(torch.mul(pixel_weight, pixel_criterion(sr, gt)))
            # Cast to float32 for content loss to avoid type mismatch
            feature_loss = torch.sum(torch.mul(feature_weight, content_criterion(sr.float(), gt.float())))
            # Resize to 96x96 for discriminator
            sr_resized = torch.nn.functional.interpolate(sr, size=(96, 96), mode='bicubic', align_corners=False)
            adversarial_loss = torch.sum(torch.mul(adversarial_weight,
                                           adversarial_criterion(d_model(sr_resized), real_label)))
            pixel_loss = torch.sum(torch.mul(pixel_weight, pixel_loss))
            feature_loss = torch.sum(torch.mul(feature_weight, feature_loss))
            g_loss = pixel_loss + feature_loss + adversarial_loss

        scaler.scale(g_loss).backward()
        scaler.step(g_optimizer)
        scaler.update()

        # Train Discriminator
        for d_param in d_model.parameters():
            d_param.requires_grad = True

        d_model.zero_grad(set_to_none=True)
        with amp.autocast():
            # Resize to 96x96 for discriminator
            gt_resized = torch.nn.functional.interpolate(gt, size=(96, 96), mode='bicubic', align_corners=False)
            gt_output = d_model(gt_resized)
            d_loss_gt = adversarial_criterion(gt_output, real_label)

        scaler.scale(d_loss_gt).backward()

        with amp.autocast():
            sr_output = d_model(sr_resized.detach().clone())
            d_loss_sr = adversarial_criterion(sr_output, fake_label)

        scaler.scale(d_loss_sr).backward()
        
        # Add gradient penalty if configured
        gp_weight = config["ADVERSARIAL_LOSS_GP_WEIGHT"]
        if gp_weight > 0:
            gradient_penalty = compute_gradient_penalty(d_model, gt_resized, sr_resized.detach().clone(), device)
            gp_loss = gp_weight * gradient_penalty
            scaler.scale(gp_loss).backward()
            d_loss = d_loss_gt + d_loss_sr + gp_loss
        else:
            d_loss = d_loss_gt + d_loss_sr
            
        scaler.step(d_optimizer)
        scaler.update()

        if ema_g_model is not None:
            ema_g_model.update_parameters(g_model)

        d_losses.update(d_loss.item(), batch_size)
        g_losses.update(g_loss.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % config["PRINT_FREQ"] == 0:
            iters = epoch * batches + batch_index
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/D(GT)_Loss", d_loss_gt.item(), iters)
            writer.add_scalar("Train/D(SR)_Loss", d_loss_sr.item(), iters)
            if gp_weight > 0:
                writer.add_scalar("Train/GP_Loss", gp_loss.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Feature_Loss", feature_loss.item(), iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/D(GT)_Probability", torch.sigmoid_(torch.mean(gt_output.detach())).item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", torch.sigmoid_(torch.mean(sr_output.detach())).item(), iters)
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
        "CONTENT_LOSS_WEIGHT": config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["WEIGHT"][0],
        "ADVERSARIAL_LOSS_WEIGHT": config["TRAIN"]["LOSSES"]["ADVERSARIAL_LOSS"]["WEIGHT"][0],
        "ADVERSARIAL_LOSS_GP_WEIGHT": config["TRAIN"]["LOSSES"]["ADVERSARIAL_LOSS"]["GP_WEIGHT"],
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

    g_model, ema_g_model, d_model = build_model(device, config)
    pixel_criterion, adversarial_criterion = define_loss(device)
    content_criterion = model.ContentLoss(
        config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["NET_CFG_NAME"],
        config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["BATCH_NORM"],
        config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["NUM_CLASSES"],
        config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["MODEL_WEIGHTS_PATH"],
        config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["FEATURE_NODES"],
        config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["FEATURE_NORMALIZE_MEAN"],
        config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["FEATURE_NORMALIZE_STD"],
    ).to(device).float()
    g_optimizer, d_optimizer = define_optimizer(g_model, d_model, config)
    g_scheduler, d_scheduler = define_scheduler(g_optimizer, d_optimizer, config)
    scaler = amp.GradScaler()

    samples_dir = os.path.join("results", "samples", config["EXP_NAME"])
    results_dir = os.path.join("results", "results", config["EXP_NAME"])
    make_directory(samples_dir)
    make_directory(results_dir)

    writer = SummaryWriter(os.path.join(config["LOG_DIR"], "samples", "logs", config["EXP_NAME"]))

    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_G_MODEL"]:
        g_model = load_pretrained_state_dict(g_model,
                                             config["MODEL"]["G"]["COMPILED"],
                                             config["TRAIN"]["CHECKPOINT"]["PRETRAINED_G_MODEL"])
        print(f"Loaded pretrained generator from {config['TRAIN']['CHECKPOINT']['PRETRAINED_G_MODEL']}")
    elif config["TRAIN"]["CHECKPOINT"]["SRRESNET_WEIGHTS"]:
        # Load SRResNet weights as starting point instead of ImageNet SRGAN
        g_model = load_pretrained_state_dict(g_model,
                                             config["MODEL"]["G"]["COMPILED"],
                                             config["TRAIN"]["CHECKPOINT"]["SRRESNET_WEIGHTS"])
        print(f"Loaded SRResNet weights as starting point from {config['TRAIN']['CHECKPOINT']['SRRESNET_WEIGHTS']}")

    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_D_MODEL"]:
        d_model = load_pretrained_state_dict(d_model,
                                             config["MODEL"]["D"]["COMPILED"],
                                             config["TRAIN"]["CHECKPOINT"]["PRETRAINED_D_MODEL"])
        print(f"Loaded pretrained discriminator from {config['TRAIN']['CHECKPOINT']['PRETRAINED_D_MODEL']}")

    start_epoch = 0
    best_psnr = 0.0
    best_ssim = 0.0

    if config["TRAIN"]["CHECKPOINT"]["RESUMED_G_MODEL"]:
        g_model, _, start_epoch, best_psnr, best_ssim, g_optimizer = load_resume_state_dict(
            g_model,
            None,
            g_optimizer,
            g_scheduler,
            config["MODEL"]["G"]["COMPILED"],
            config["TRAIN"]["CHECKPOINT"]["RESUMED_G_MODEL"],
        )
        print(f"Resumed generator from {config['TRAIN']['CHECKPOINT']['RESUMED_G_MODEL']} at epoch {start_epoch}")

    if config["TRAIN"]["CHECKPOINT"]["RESUMED_D_MODEL"]:
        checkpoint = torch.load(config["TRAIN"]["CHECKPOINT"]["RESUMED_D_MODEL"], map_location="cpu")
        d_model.load_state_dict(checkpoint["state_dict"])
        d_optimizer.load_state_dict(checkpoint["optimizer"])
        d_scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Resumed discriminator from {config['TRAIN']['CHECKPOINT']['RESUMED_D_MODEL']}")

    psnr_model = None
    ssim_model = None
    if val_prefetcher is not None:
        psnr_model, ssim_model = build_iqa_model(config["SCALE"], False, device)

    for epoch in range(start_epoch, config["TRAIN"]["HYP"]["EPOCHS"]):
        train(g_model,
              ema_g_model,
              d_model,
              train_prefetcher,
              pixel_criterion,
              content_criterion,
              adversarial_criterion,
              g_optimizer,
              d_optimizer,
              epoch,
              scaler,
              writer,
              device,
              config_dict)

        g_scheduler.step()
        d_scheduler.step()

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
            "optimizer": g_optimizer.state_dict(),
            "scheduler": g_scheduler.state_dict(),
        },
            f"epoch_{epoch + 1}.pth.tar",
            samples_dir,
            results_dir,
            "g_best.pth.tar",
            "g_last.pth.tar",
            is_best,
            is_last)

        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": d_model.state_dict(),
            "optimizer": d_optimizer.state_dict(),
            "scheduler": d_scheduler.state_dict(),
        },
            f"epoch_{epoch + 1}.pth.tar",
            samples_dir,
            results_dir,
            "d_best.pth.tar",
            "d_last.pth.tar",
            is_best,
            is_last)


if __name__ == "__main__":
    main()
