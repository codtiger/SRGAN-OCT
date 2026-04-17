from accelerate.test_utils.scripts.test_distributed_data_loop import test_data_loader
from imgproc import image_resize, image_to_tensor, tensor_to_image
from utils import make_directory, ProgressMeter, Summary, \
    load_resume_state_dict, load_pretrained_state_dict

import argparse
import os
import random
from pathlib import Path
import time
from typing import Any, List, Optional
from functools import partial

import cv2
import numpy as np
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from imgproc import image_resize, image_to_tensor, tensor_to_image
from oct_dataset import OCTImageDataset
import model
from utils import load_pretrained_state_dict

def main():
    parser = argparse.ArgumentParser(description='SRGAN Super Resolution')
    parser.add_argument('--config', default='config_imagegen.yaml', type=str, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(config['DEVICE'] if torch.cuda.is_available() else 'cpu')

    g_model = model.__dict__[config["MODEL"]["G"]["NAME"]](in_channels=config["MODEL"]["G"]["IN_CHANNELS"],
                                                           out_channels=config["MODEL"]["G"]["OUT_CHANNELS"],
                                                           channels=config["MODEL"]["G"]["CHANNELS"],
                                                           num_rcb=config["MODEL"]["G"]["NUM_RCB"])
    g_model = g_model.to(device=device)
    if config["MODEL"]["G"]["COMPILED"]:
        g_model = torch.compile(g_model)
    g_model = load_pretrained_state_dict(g_model, config["MODEL"]["G"]["COMPILED"],
                                          config['MODEL']['PATH'])
    g_model = g_model.to(device=device)
    g_model.eval()
    train_path = Path(config['DATASET']['PATH']) / "train"
    test_path = Path(config['DATASET']['PATH']) / "test"

    train_oct = OCTImageDataset(str(train_path), transform=None)
    train_data = DataLoader(train_oct, batch_size=1, shuffle=False)
    test_oct = OCTImageDataset(str(test_path), transform=None)
    test_data = DataLoader(test_oct, batch_size=1, shuffle=False)

    output_dir = Path(config['OUTPUT_DIR'])
    output_train_dir = output_dir / "train"
    output_test_dir = output_dir / "test"
    output_train_dir.mkdir(parents=True, exist_ok=True)
    output_test_dir.mkdir(parents=True, exist_ok=True)

    for class_name in config["CLASSES"]:
        (output_train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (output_test_dir / class_name).mkdir(parents=True, exist_ok=True)

    # Generate high-resolution train images
    with torch.no_grad():
        for batch in tqdm(train_data):
            lr_image = batch['lr'].to(device)
            sr_tensor = g_model(lr_image)

            sr_image = tensor_to_image(sr_tensor.squeeze(0), False, False)
            # sr_image = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)

            class_name = batch['image_name'][0].split('/')[-2]
            image_name = batch['image_name'][0].split('/')[-1]
            image_name = str(Path(image_name).stem + ".tif")

            output_path = output_train_dir / f"{class_name}/sr_{image_name}"
            cv2.imwrite(str(output_path), sr_image[..., ::-1])
            # print(f"Generated high-resolution image saved to {output_path}")

    # Generate high-resolution test images
    with torch.no_grad():
        for batch in tqdm(test_data):
            lr_image = batch['lr'].to(device)
            # Generate super-resolution image
            sr_tensor = g_model(lr_image)

            sr_image = tensor_to_image(sr_tensor.squeeze(0), False, False)
            # sr_image = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)

            class_name = batch['image_name'][0].split('/')[-2]
            image_name = batch['image_name'][0].split('/')[-1]
            image_name = str(Path(image_name).stem + ".tif")
            output_path = output_test_dir / f"{class_name}/sr_{image_name}"
            cv2.imwrite(str(output_path), cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR))
            # print(f"Generated high-resolution image saved to {output_path}")

if __name__ == "__main__":
    main()