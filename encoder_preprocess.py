import os
from glob import glob
from typing import Tuple, Optional
from copy import deepcopy
from collections import OrderedDict
import logging
import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
import torch.distributed as dist

from datasets import DatasetFromCSV, get_transforms_video
from vae import VideoAutoEncoderKL
from t5 import T5Encoder
from clip import ClipEncoder
from config import Config


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def to_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        dtype_mapping = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype not in dtype_mapping:
            raise ValueError
        dtype = dtype_mapping[dtype]
        return dtype
    else:
        raise ValueError


def main():
    # create configs
    cfg = Config()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.manual_seed(cfg.seed)
    torch.cuda.set_device(device)
    dtype = to_torch_dtype(cfg.dtype)
    torch.set_default_dtype(dtype)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(os.path.join(cfg.root, cfg.preprocessed_dir),
                    exist_ok=True)

    # prepare dataset
    dataset = DatasetFromCSV(cfg.data_path,
                             num_frames=cfg.num_frames,
                             frame_interval=cfg.frame_interval,
                             transform=get_transforms_video(),
                             root=cfg.root)

    dataloader = DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        sampler=SequentialSampler(dataset),
        batch_size=cfg.preprocess_batch_size,
        drop_last=False,
    )

    print(f"Dataset contains {len(dataset):,} videos ({cfg.data_path})")
    total_batch_size = cfg.preprocess_batch_size * dist.get_world_size(
    ) // cfg.sp_size
    print(f"Total batch size: {total_batch_size}")

    # video VAE
    vae = VideoAutoEncoderKL(cfg.vae_pretrained,
                             cfg.vae_scaling_factor,
                             dtype=dtype)

    # text encoder
    if "t5" in cfg.textenc_pretrained:
        text_encoder_cls = T5Encoder
    elif ("stable-diffusion" in cfg.textenc_pretrained
          or "sd" in cfg.textenc_pretrained):
        text_encoder_cls = ClipEncoder
    text_encoder = text_encoder_cls(from_pretrained=cfg.textenc_pretrained,
                                    model_max_length=cfg.model_max_length,
                                    dtype=dtype)

    # 4.3. move to device
    vae = vae.to(device)
    vae.eval()

    num_steps_per_epoch = len(dataloader)

    # =======================================================
    # 6. encoder loop
    # =======================================================
    start_epoch = start_step = log_step = sampler_start_idx = 0

    dataloader_iter = iter(dataloader)
    epoch = 0
    with tqdm(
            range(start_step, num_steps_per_epoch),
            desc=f"Epoch {epoch}",
            # disable=not coordinator.is_master(),
            total=num_steps_per_epoch,
            initial=start_step,
    ) as pbar:

        for step in pbar:
            global_step = epoch * num_steps_per_epoch + step

            # step
            batch = next(dataloader_iter)
            x = batch["video"].to(device, dtype)  # [B, C, T, H, W]
            y = batch[cfg.text_key]
            video_ids = batch["video_id"]

            # video and text encoding
            with torch.no_grad():
                x = vae.encode(x)
                model_args = text_encoder.encode(y)

                # if encode only, we save results to file
                for idx in range(len(video_ids)):
                    vid = video_ids[idx]
                    save_dir = os.path.join(cfg.root, cfg.preprocessed_dir)
                    save_fpath = os.path.join(save_dir, vid + ".pt")
                    if not os.path.exists(
                            save_fpath) or cfg.override_preprocessed:
                        saved_data = {
                            "x": x[idx].cpu(),
                            "y": model_args["y"][idx].cpu(),
                            "mask": model_args["mask"][idx].cpu(),
                            "video_id": vid,
                        }
                        torch.save(saved_data, save_fpath)

    print("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
